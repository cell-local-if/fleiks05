"""Tests for the full-store import endpoint.

The endpoint is::

    POST /v1/admin/store/import

It restores a fresh empty instance from the version-1 recovery document
— the body is a JSON object with exactly the nine sections ``version``,
``operations``, ``checkpoints``, ``policies``, ``transactions``,
``acks``, ``repairExecutions``, ``policyEvents``, and ``compensations``.
The first import into a fresh store commits and answers 201
``{"status":"created","version":1}``; an identical resubmission is the
idempotent 200 ``{"status":"ok","version":1}`` and appends nothing; any
other document against a store that already holds state is 409
``{"error":"store_conflict"}``. A malformed or inconsistent document is
400 ``{"error":"invalid_request"}`` with memory and the data file
untouched. With ``--data-file`` the imported state is persisted
atomically before the success is observed; a persistence failure is 500
``{"error":"internal_error"}`` with memory and file unchanged.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

IMPORT_PATH = "/v1/admin/store/import"
EXPORT_PATH = "/v1/admin/store/export"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
RW_TOKEN = "read-write-token"
SINGLE_TOKEN = "the-one-token"

POLICY = {
    READ_TOKEN: frozenset({"read"}),
    WRITE_TOKEN: frozenset({"write"}),
    ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
    RW_TOKEN: frozenset({"read", "write"}),
}

EMPTY_DOCUMENT = {
    "version": 1,
    "operations": [],
    "checkpoints": {},
    "policies": [],
    "transactions": [],
    "acks": [],
    "repairExecutions": [],
    "policyEvents": [],
    "compensations": [],
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def populated_document() -> dict:
    """A valid version-1 recovery document with every section filled."""
    return {
        "version": 1,
        "operations": [
            {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})},
            {"replicaId": "r2", "operation": operation("o2", "k", "v2", {"r2": 1})},
            {"replicaId": "r1", "operation": operation("t1", "k2", "new", {"r1": 2})},
        ],
        "checkpoints": {"peer-a": 3},
        "policies": [
            {"replicaId": "r1", "operationId": "o1", "policy": "lowest_identity"}
        ],
        "transactions": [
            {
                "transactionId": "tx-1",
                "operations": [
                    {
                        "key": "k2",
                        "replicaId": "r1",
                        "operationId": "t1",
                        "value": "new",
                        "clock": {"r1": 2},
                        "candidates": [],
                    }
                ],
            }
        ],
        "acks": [
            {
                "peerId": "peer-a",
                "ackId": "ack-1",
                "cursor": 3,
                "operations": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r2", "operationId": "o2"},
                    {"replicaId": "r1", "operationId": "t1"},
                ],
            }
        ],
        "repairExecutions": [
            {
                "peerId": "peer-a",
                "ackId": "ack-2",
                "expectedCheckpoint": 0,
                "expectedReceipts": "b" * 64,
                "suggestions": [
                    {
                        "action": "correct_cursor",
                        "ackId": "ack-2",
                        "location": {"start": 0, "end": 0},
                        "target": {"start": 3, "end": 3},
                    }
                ],
                "results": [
                    {"action": "correct_cursor", "boundary": {"start": 3, "end": 3}}
                ],
                "cursor": 3,
            }
        ],
        "policyEvents": [{"sequence": 1, "digest": "a" * 64, "tokens": 2}],
        "compensations": [
            {
                "compensationId": "c-1",
                "transactionId": "tx-1",
                "expectedPlanDigest": "c" * 64,
                "operations": [
                    {
                        "key": "k",
                        "replicaId": "r1",
                        "operationId": "o1",
                        "value": "v1",
                        "clock": {"r1": 1},
                    }
                ],
                "status": "committed",
            }
        ],
    }


class ServerTestCase(unittest.TestCase):
    """Base harness: a fresh running server per test plus a raw HTTP helper."""

    server_kwargs: dict = {}

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_file = os.path.join(self.tmp.name, "state.json")
        kwargs = dict(self.server_kwargs)
        if kwargs.pop("with_data_file", False):
            kwargs["data_file"] = self.data_file
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, **kwargs
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = None,
        raw_body: bytes | None = None,
    ) -> tuple[int, dict, dict, bytes]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw_body is not None:
            conn.request(method, path, body=raw_body, headers=headers)
        elif body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers, raw

    def import_document(
        self, document: object, token: str | None = None
    ) -> tuple[int, dict, dict, bytes]:
        return self.request("POST", IMPORT_PATH, body=document, token=token)

    def export(self, token: str | None = None) -> tuple[int, dict, dict, bytes]:
        return self.request("GET", EXPORT_PATH, token=token)

    def post_raw_over_limit(self, token: str | None = None) -> tuple[int, dict]:
        """POST an over-limit body, tolerating the server's early close.

        The server answers 413 on the declared length alone and closes
        the connection before the client finishes writing the body, so
        the write may race the close; the response is what matters.
        """
        body = b" " * (1_048_576 + 1)
        lines = [
            b"POST /v1/admin/store/import HTTP/1.1",
            b"Host: 127.0.0.1",
            b"Content-Type: application/json",
            b"Connection: close",
            f"Content-Length: {len(body)}".encode(),
        ]
        if token is not None:
            lines.append(b"Authorization: Bearer " + token.encode())
        head = b"\r\n".join(lines) + b"\r\n\r\n"
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            try:
                sock.sendall(head + body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            chunks = []
            while True:
                try:
                    data = sock.recv(4096)
                except ConnectionResetError:
                    break
                if not data:
                    break
                chunks.append(data)
        raw = b"".join(chunks)
        status = int(raw.split(b" ", 2)[1])
        payload = json.loads(raw.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
        return status, payload


class EmptyDocumentImportTests(ServerTestCase):
    """The all-empty document imports into a fresh store, then replays."""

    def test_first_import_is_created(self) -> None:
        status, payload, _, _ = self.import_document(EMPTY_DOCUMENT)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})

    def test_identical_resubmission_is_ok_and_appends_nothing(self) -> None:
        self.assertEqual(self.import_document(EMPTY_DOCUMENT)[0], 201)
        before = self.export()[3]
        status, payload, _, _ = self.import_document(EMPTY_DOCUMENT)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        # No operation, cursor, or audit was appended.
        self.assertEqual(self.export()[3], before)
        status, payload, _, _ = self.import_document(EMPTY_DOCUMENT)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        self.assertEqual(self.export()[3], before)

    def test_export_after_import_is_the_empty_document(self) -> None:
        self.assertEqual(self.import_document(EMPTY_DOCUMENT)[0], 201)
        status, payload, _, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_DOCUMENT)


class PopulatedDocumentImportTests(ServerTestCase):
    """A populated document restores every section and replays idempotently."""

    def test_import_commit_export_and_replay(self) -> None:
        document = populated_document()
        status, payload, _, _ = self.import_document(document)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})
        # The existing endpoints work on the imported state: the export
        # returns the same content.
        status, exported, _, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(exported, document)
        # An identical resubmission is the idempotent replay.
        status, payload, _, _ = self.import_document(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        self.assertEqual(self.export()[1], document)

    def test_imported_state_serves_the_business_endpoints(self) -> None:
        self.assertEqual(self.import_document(populated_document())[0], 201)
        status, payload, _, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        values = {candidate["value"] for candidate in payload["candidates"]}
        self.assertEqual(values, {"v1", "v2"})
        # The imported checkpoint is the peer's committed cursor.
        status, payload, _, _ = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload["cursor"], 3)

    def test_different_document_against_committed_state_conflicts(self) -> None:
        self.assertEqual(self.import_document(populated_document())[0], 201)
        before = self.export()[3]
        other = populated_document()
        other["operations"][0]["operation"]["value"] = "changed"
        for different in (EMPTY_DOCUMENT, other):
            with self.subTest(different=different["operations"][:1]):
                status, payload, _, _ = self.import_document(different)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "store_conflict"})
        self.assertEqual(self.export()[3], before)

    def test_imported_local_bindings_stay_out_of_the_sync_stream(self) -> None:
        self.assertEqual(self.import_document(populated_document())[0], 201)
        # The sync export carries only the accepted operations — never the
        # transaction, receipt, repair, or compensation bindings.
        status, page, _, _ = self.request("GET", "/v1/sync/operations?limit=100")
        self.assertEqual(status, 200)
        identities = [
            (record["replicaId"], record["operation"]["operationId"])
            for record in page["operations"]
        ]
        self.assertEqual(identities, [("r1", "o1"), ("r2", "o2"), ("r1", "t1")])


class ConflictAgainstOrdinaryStateTests(ServerTestCase):
    """A store populated through the ordinary endpoints rejects imports."""

    def test_different_document_is_409_and_matching_document_is_200(self) -> None:
        status, _, _, _ = self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        before = self.export()[3]
        status, payload, _, _ = self.import_document(populated_document())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "store_conflict"})
        self.assertEqual(self.export()[3], before)
        # The document identical to the current export is the idempotent
        # replay even without a prior import.
        status, payload, _, _ = self.import_document(self.export()[1])
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        self.assertEqual(self.export()[3], before)


class ConcurrentImportTests(ServerTestCase):
    """Concurrent first imports commit exactly once."""

    def test_only_one_first_import_commits(self) -> None:
        document = populated_document()
        statuses: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def submit() -> None:
            barrier.wait()
            status, _, _, _ = self.import_document(document)
            with lock:
                statuses.append(status)

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(sorted(statuses), [200] * 7 + [201])
        self.assertEqual(self.export()[1], document)

    def test_concurrent_different_documents_conflict(self) -> None:
        winner = populated_document()
        loser = populated_document()
        loser["operations"][0]["operation"]["value"] = "changed"
        statuses: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def submit(document: dict) -> None:
            barrier.wait()
            status, _, _, _ = self.import_document(document)
            with lock:
                statuses.append(status)

        threads = [
            threading.Thread(target=submit, args=(winner if index % 2 == 0 else loser,))
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        # Exactly one first import committed; every other submission was
        # either the identical replay or a conflict.
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(len(statuses), 8)
        self.assertTrue(set(statuses) <= {200, 201, 409})
        self.assertIn(409, statuses)


class InvalidDocumentTests(ServerTestCase):
    """Malformed or inconsistent documents are 400 and change nothing."""

    def assert_invalid(self, body: object = None, raw_body: bytes | None = None) -> None:
        before = self.export()[3]
        status, payload, _, _ = self.request(
            "POST", IMPORT_PATH, body=body, raw_body=raw_body
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.export()[3], before)

    def test_non_utf8_body_is_400(self) -> None:
        self.assert_invalid(raw_body=b"\xff\xfe{" )

    def test_non_json_body_is_400(self) -> None:
        self.assert_invalid(raw_body=b"{")

    def test_non_object_root_is_400(self) -> None:
        self.assert_invalid(body=[1, 2, 3])
        self.assert_invalid(body="document")
        self.assert_invalid(body=None)

    def test_missing_section_is_400(self) -> None:
        for section in EMPTY_DOCUMENT:
            with self.subTest(section=section):
                document = {k: v for k, v in EMPTY_DOCUMENT.items() if k != section}
                self.assert_invalid(body=document)

    def test_extra_root_field_is_400(self) -> None:
        self.assert_invalid(body={**EMPTY_DOCUMENT, "extra": []})

    def test_wrong_version_is_400(self) -> None:
        for version in (2, "1", True, None, 1.0):
            with self.subTest(version=version):
                self.assert_invalid(body={**EMPTY_DOCUMENT, "version": version})

    def test_operations_must_be_a_list(self) -> None:
        self.assert_invalid(body={**EMPTY_DOCUMENT, "operations": {}})

    def test_duplicate_operation_identity_is_400(self) -> None:
        record = {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
        self.assert_invalid(
            body={**EMPTY_DOCUMENT, "operations": [record, dict(record)]}
        )

    def test_operation_with_an_unexpected_shape_is_400(self) -> None:
        self.assert_invalid(
            body={
                **EMPTY_DOCUMENT,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": {
                            "operationId": "o1",
                            "key": "k",
                            "value": "v",
                            "clock": {"r1": 1},
                            "extra": 1,
                        },
                    }
                ],
            }
        )

    def test_checkpoint_past_the_log_is_400(self) -> None:
        self.assert_invalid(body={**EMPTY_DOCUMENT, "checkpoints": {"peer-a": 1}})

    def test_negative_checkpoint_is_400(self) -> None:
        self.assert_invalid(body={**EMPTY_DOCUMENT, "checkpoints": {"peer-a": -1}})

    def test_policy_naming_no_accepted_operation_is_400(self) -> None:
        self.assert_invalid(
            body={
                **EMPTY_DOCUMENT,
                "policies": [
                    {"replicaId": "r1", "operationId": "o1", "policy": "lowest_identity"}
                ],
            }
        )

    def test_transaction_naming_no_accepted_operation_is_400(self) -> None:
        self.assert_invalid(
            body={
                **EMPTY_DOCUMENT,
                "transactions": [
                    {
                        "transactionId": "tx-1",
                        "operations": [
                            {
                                "key": "k",
                                "replicaId": "r1",
                                "operationId": "o1",
                                "value": "v",
                                "clock": {"r1": 1},
                                "candidates": [],
                            }
                        ],
                    }
                ],
            }
        )

    def test_ack_for_an_unregistered_peer_is_400(self) -> None:
        self.assert_invalid(
            body={
                **EMPTY_DOCUMENT,
                "acks": [
                    {"peerId": "peer-a", "ackId": "ack-1", "cursor": 0, "operations": []}
                ],
            }
        )

    def test_ack_segment_mismatching_the_log_is_400(self) -> None:
        document = populated_document()
        document["acks"][0]["operations"] = [
            {"replicaId": "r2", "operationId": "o2"},
            {"replicaId": "r1", "operationId": "o1"},
            {"replicaId": "r1", "operationId": "t1"},
        ]
        self.assert_invalid(body=document)

    def test_compensation_naming_no_committed_transaction_is_400(self) -> None:
        document = populated_document()
        document["compensations"][0]["transactionId"] = "tx-unknown"
        self.assert_invalid(body=document)

    def test_policy_event_sequence_gap_is_400(self) -> None:
        self.assert_invalid(
            body={
                **EMPTY_DOCUMENT,
                "policyEvents": [{"sequence": 2, "digest": "a" * 64, "tokens": 1}],
            }
        )

    def test_repair_past_the_checkpoint_is_400(self) -> None:
        document = populated_document()
        document["checkpoints"] = {"peer-a": 2}
        self.assert_invalid(body=document)


class ImportClockWidthTests(ServerTestCase):
    """The configured vector-clock width bound applies to imported clocks."""

    def setUp(self) -> None:
        self._saved_limit = server_module._MAX_CLOCK_COMPONENTS
        server_module._MAX_CLOCK_COMPONENTS = 1
        self.addCleanup(self._restore_limit)
        super().setUp()

    def _restore_limit(self) -> None:
        server_module._MAX_CLOCK_COMPONENTS = self._saved_limit

    def test_over_wide_clock_is_400(self) -> None:
        document = populated_document()
        document["operations"][0]["operation"]["clock"] = {"r1": 1, "r2": 1}
        before = self.export()[3]
        status, payload, _, _ = self.import_document(document)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.export()[3], before)


class ImportRouteTests(ServerTestCase):
    """The endpoint is published exactly at /v1/admin/store/import."""

    def assert_not_found(self, method: str, path: str) -> None:
        status, payload, _, _ = self.request(method, path, body=EMPTY_DOCUMENT)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_trailing_slash_and_wrong_shapes_are_404(self) -> None:
        self.assert_not_found("POST", "/v1/admin/store/import/")
        self.assert_not_found("POST", "/v1/admin/store/import/extra")
        self.assert_not_found("POST", "/v1/admin/store")
        self.assert_not_found("POST", "/v1/admin/store/export")
        self.assert_not_found("POST", "/v1/admin")

    def test_get_on_the_import_path_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", IMPORT_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejections_change_no_state(self) -> None:
        before = self.export()[3]
        self.assert_not_found("POST", "/v1/admin/store/import/")
        self.assertEqual(self.export()[3], before)


class ImportRequestLimitTests(ServerTestCase):
    """The shared POST Content-Length contract applies to the import."""

    def test_over_limit_declared_length_is_413(self) -> None:
        before = self.export()[3]
        status, payload = self.post_raw_over_limit()
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})
        self.assertEqual(self.export()[3], before)

    def test_missing_content_length_is_400(self) -> None:
        before = self.export()[3]
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(
                b"POST /v1/admin/store/import HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        raw = b"".join(chunks)
        self.assertIn(b" 400 ", raw.split(b"\r\n", 1)[0])
        self.assertIn(b"invalid_request", raw)
        self.assertEqual(self.export()[3], before)


class ImportSingleTokenAuthTests(ServerTestCase):
    """The legacy single token keeps full access; bad credentials are 401."""

    server_kwargs = {"auth_token": SINGLE_TOKEN}

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.import_document(
                    EMPTY_DOCUMENT, token=token
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_the_configured_token_imports(self) -> None:
        status, payload, _, _ = self.import_document(EMPTY_DOCUMENT, token=SINGLE_TOKEN)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})


class ImportScopePolicyAuthTests(ServerTestCase):
    """In scope-policy mode only the admin scope may import."""

    server_kwargs = {"auth_scopes": dict(POLICY)}

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.import_document(
                    EMPTY_DOCUMENT, token=token
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_read_and_write_tokens_are_403_without_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN, RW_TOKEN):
            with self.subTest(token=token):
                status, payload, headers, _ = self.import_document(
                    EMPTY_DOCUMENT, token=token
                )
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)
                self.assertNotIn("Www-Authenticate", headers)

    def test_forbidden_requests_never_change_or_observe_state(self) -> None:
        status, _, _, _ = self.import_document(populated_document(), token=READ_TOKEN)
        self.assertEqual(status, 403)
        status, payload, _, _ = self.export(token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_DOCUMENT)

    def test_admin_token_imports(self) -> None:
        status, payload, _, _ = self.import_document(
            populated_document(), token=ADMIN_TOKEN
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})
        status, exported, _, _ = self.export(token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(exported, populated_document())

    def test_length_check_precedes_authentication(self) -> None:
        # The shared POST contract answers the 413 before any credential
        # check: an anonymous over-limit request is 413, not 401.
        status, payload = self.post_raw_over_limit()
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})


class ImportPersistenceTests(ServerTestCase):
    """With --data-file the import commits atomically before success."""

    server_kwargs = {"with_data_file": True}

    def test_import_persists_the_document_and_recovers_identically(self) -> None:
        document = populated_document()
        self.assertEqual(self.import_document(document)[0], 201)
        with open(self.data_file, "rb") as handle:
            persisted = json.loads(handle.read().decode("utf-8"))
        self.assertEqual(persisted, document)
        # A fresh store recovers the imported state from the file, and
        # its export is the imported document again.
        recovered = StateStore(data_file=self.data_file)
        try:
            self.assertEqual(recovered.get_store_export(), document)
        finally:
            del recovered
        # No temporary files are left behind.
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name != os.path.basename(self.data_file)
        ]
        self.assertEqual(leftovers, [])

    def test_failed_persist_is_500_and_changes_nothing(self) -> None:
        with open(self.data_file, "rb") as handle:
            file_before = handle.read()
        memory_before = self.export()[3]
        with mock.patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("boom")
        ):
            status, payload, _, _ = self.import_document(populated_document())
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "internal_error"})
        # Memory and the data file are unchanged, and a retry succeeds.
        self.assertEqual(self.export()[3], memory_before)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), file_before)
        self.assertEqual(self.import_document(populated_document())[0], 201)
        self.assertEqual(self.export()[1], populated_document())


class ImportWithoutDataFileTests(ServerTestCase):
    """Without --data-file the import only touches memory."""

    def test_import_creates_no_files(self) -> None:
        self.assertEqual(self.import_document(populated_document())[0], 201)
        self.assertEqual(os.listdir(self.tmp.name), [])
        self.assertEqual(self.export()[1], populated_document())


if __name__ == "__main__":
    unittest.main()
