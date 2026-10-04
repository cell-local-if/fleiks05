"""Tests for the full-store import endpoint.

The endpoint is::

    POST /v1/admin/store/import

It restores a brand-new empty instance from the version-1 recovery
document — exactly the document ``GET /v1/admin/store/export`` returns
and ``--data-file`` persists. The body must be a JSON object carrying
exactly the nine fields ``version`` (the integer 1), ``operations``,
``checkpoints``, ``policies``, ``transactions``, ``acks``,
``repairExecutions``, ``policyEvents``, and ``compensations``, validated
with exactly the recovery format and cross-section constraints of
startup recovery.

A virgin store commits the document and answers HTTP 201
``{"status":"created","version":1}``; the same document submitted again
is an idempotent HTTP 200 ``{"status":"ok","version":1}`` that appends
no operation, cursor, or audit record; a store whose committed state
differs from the document answers HTTP 409 ``{"error":"store_conflict"}``.
Any malformed or constraint-violating body is HTTP 400
``{"error":"invalid_request"}`` with memory and the data file untouched.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
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


def rich_document() -> dict:
    """A valid version-1 recovery document exercising every section."""
    return {
        "version": 1,
        "operations": [
            {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})},
            {"replicaId": "r2", "operation": operation("o2", "k", "v2", {"r2": 1})},
            {"replicaId": "r3", "operation": operation("o3", "k2", "back", {"r3": 1})},
        ],
        "checkpoints": {"peer-a": 2},
        "policies": [
            {"replicaId": "r1", "operationId": "o1", "policy": "lowest_identity"}
        ],
        "transactions": [
            {
                "transactionId": "tx-1",
                "operations": [
                    {
                        "key": "k",
                        "replicaId": "r1",
                        "operationId": "o1",
                        "value": "v1",
                        "clock": {"r1": 1},
                        "candidates": [],
                    }
                ],
            }
        ],
        "acks": [
            {
                "peerId": "peer-a",
                "ackId": "ack-1",
                "cursor": 2,
                "operations": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r2", "operationId": "o2"},
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
                        "target": {"start": 2, "end": 2},
                    }
                ],
                "results": [
                    {"action": "correct_cursor", "boundary": {"start": 2, "end": 2}}
                ],
                "cursor": 2,
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
                        "key": "k2",
                        "replicaId": "r3",
                        "operationId": "o3",
                        "value": "back",
                        "clock": {"r3": 1},
                    }
                ],
                "status": "committed",
            }
        ],
    }


class ImportTestCase(unittest.TestCase):
    """Base harness: a fresh running server per test plus HTTP helpers."""

    def start_server(self, **kwargs) -> SemanticStateServer:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        return server

    def request(
        self,
        server: SemanticStateServer,
        method: str,
        path: str,
        body: object = None,
        token: str | None = None,
    ) -> tuple[int, dict, dict, bytes]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers, raw

    def post_raw(
        self,
        server: SemanticStateServer,
        path: str,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> tuple[int, dict]:
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def import_doc(
        self, server: SemanticStateServer, document: dict, token: str | None = None
    ) -> tuple[int, dict]:
        status, payload, _, _ = self.request(
            server, "POST", IMPORT_PATH, body=document, token=token
        )
        return status, payload

    def export(self, server: SemanticStateServer, token: str | None = None):
        return self.request(server, "GET", EXPORT_PATH, token=token)


class StoreImportCommitTests(ImportTestCase):
    """201 on a virgin store, idempotent 200 on replay, 409 on divergence."""

    def test_first_import_commits_and_answers_201(self) -> None:
        server = self.start_server()
        document = rich_document()
        status, payload = self.import_doc(server, document)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})

    def test_export_after_import_returns_the_same_document(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        status, payload, _, _ = self.export(server)
        self.assertEqual(status, 200)
        self.assertEqual(payload, document)
        self.assertEqual(set(payload.keys()), set(EMPTY_DOCUMENT.keys()))

    def test_existing_endpoints_see_the_imported_state(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        # k holds the two concurrent imported writes as a conflict.
        status, state, _, _ = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual({c["value"] for c in state["candidates"]}, {"v1", "v2"})
        # The imported checkpoint is the peer's registered cursor.
        status, checkpoint, _, _ = self.request(
            server, "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        self.assertEqual(status, 200)
        self.assertEqual(checkpoint["cursor"], 2)
        # The imported operations joined the accepted log.
        status, metrics, _, _ = self.request(server, "GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["acceptedOperations"], 3)

    def test_imported_operations_flow_through_sync_but_bindings_stay_local(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        status, page, _, _ = self.request(server, "GET", "/v1/sync/operations?limit=100")
        self.assertEqual(status, 200)
        identities = [
            (record["replicaId"], record["operation"]["operationId"])
            for record in page["operations"]
        ]
        self.assertEqual(identities, [("r1", "o1"), ("r2", "o2"), ("r3", "o3")])
        # The sync stream carries operation records only: no policy,
        # transaction, receipt, repair, policy-event, or compensation
        # binding ever leaks into it.
        for record in page["operations"]:
            self.assertEqual(set(record.keys()), {"replicaId", "operation"})

    def test_identical_reimport_is_an_idempotent_200(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        before = self.export(server)[3]
        status, payload = self.import_doc(server, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        # Nothing is appended or advanced: the export is byte-identical.
        self.assertEqual(self.export(server)[3], before)
        status, metrics, _, _ = self.request(server, "GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 3)

    def test_empty_document_on_a_virgin_store_is_already_satisfied(self) -> None:
        server = self.start_server()
        # The empty store's nine sections are all empty, so the empty
        # document is exactly the current export: nothing to create.
        status, payload = self.import_doc(server, EMPTY_DOCUMENT)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        self.assertEqual(self.export(server)[1], EMPTY_DOCUMENT)

    def test_different_document_on_a_committed_store_is_409(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        before = self.export(server)[3]
        divergent = rich_document()
        divergent["checkpoints"] = {"peer-a": 3}
        status, payload = self.import_doc(server, divergent)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "store_conflict"})
        self.assertEqual(self.export(server)[3], before)

    def test_import_after_live_writes_conflicts_unless_identical(self) -> None:
        server = self.start_server()
        status, _, _, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        # A different valid document cannot displace the committed state.
        status, payload = self.import_doc(server, rich_document())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "store_conflict"})
        # The exact current export, however, is an idempotent replay.
        current = self.export(server)[1]
        status, payload = self.import_doc(server, current)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "version": 1})
        self.assertEqual(self.export(server)[1], current)


class StoreImportPersistenceTests(ImportTestCase):
    """With --data-file the import is durable before it is visible."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_file = os.path.join(self.tmp.name, "state.json")

    def test_committed_import_is_persisted_atomically(self) -> None:
        server = self.start_server(data_file=self.data_file)
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        with open(self.data_file, "rb") as handle:
            persisted = json.loads(handle.read().decode("utf-8"))
        self.assertEqual(persisted, document)
        # The commit leaves no temporary files behind.
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name != os.path.basename(self.data_file)
        ]
        self.assertEqual(leftovers, [])
        # A fresh store recovers the imported state from the file.
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_store_export(), document)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server(data_file=self.data_file)
        with open(self.data_file, "rb") as handle:
            before = handle.read()
        document = rich_document()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.import_doc(server, document)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
        # Memory and the data file are exactly the pre-failure state ...
        self.assertEqual(self.export(server)[1], EMPTY_DOCUMENT)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), before)
        # ... and the import is retryable afterwards.
        self.assertEqual(self.import_doc(server, document)[0], 201)
        self.assertEqual(self.export(server)[1], document)

    def test_import_without_data_file_updates_memory_only(self) -> None:
        server = self.start_server()
        document = rich_document()
        self.assertEqual(self.import_doc(server, document)[0], 201)
        self.assertEqual(self.export(server)[1], document)
        self.assertFalse(os.path.exists(self.data_file))


class StoreImportValidationTests(ImportTestCase):
    """Malformed bodies and constraint violations are 400 and commit nothing."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_file = os.path.join(self.tmp.name, "state.json")
        self.server = self.start_server(data_file=self.data_file)
        with open(self.data_file, "rb") as handle:
            self.file_before = handle.read()

    def tearDown(self) -> None:
        # No rejected request touched memory or the data file.
        status, payload, _, _ = self.export(self.server)
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_DOCUMENT)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), self.file_before)

    def assert_invalid_raw(self, body: bytes) -> None:
        status, payload = self.post_raw(
            self.server,
            IMPORT_PATH,
            [("Content-Length", str(len(body)))],
            body,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def assert_invalid(self, document: dict) -> None:
        status, payload = self.import_doc(self.server, document)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_non_utf8_and_malformed_bodies_are_400(self) -> None:
        self.assert_invalid_raw(b"\xff\xfe{}")
        self.assert_invalid_raw(b"{")
        self.assert_invalid_raw(b"")
        self.assert_invalid_raw(b"[1,2]")
        self.assert_invalid_raw(b'"text"')
        self.assert_invalid_raw(b"1")

    def test_missing_and_extra_root_fields_are_400(self) -> None:
        for field in EMPTY_DOCUMENT:
            with self.subTest(missing=field):
                document = rich_document()
                del document[field]
                self.assert_invalid(document)
        with self.subTest(extra="unknown"):
            document = rich_document()
            document["unknown"] = []
            self.assert_invalid(document)

    def test_wrong_versions_are_400(self) -> None:
        for version in (0, 2, "1", 1.0, True, None):
            with self.subTest(version=version):
                document = rich_document()
                document["version"] = version
                self.assert_invalid(document)

    def test_duplicate_operation_identity_is_400(self) -> None:
        document = rich_document()
        document["operations"].append(dict(document["operations"][0]))
        self.assert_invalid(document)

    def test_illegal_operation_fields_are_400(self) -> None:
        document = rich_document()
        document["operations"][0] = {"replicaId": "r1"}
        self.assert_invalid(document)
        document = rich_document()
        document["operations"][0]["operation"]["clock"] = {"r9": 1}  # no replica id
        self.assert_invalid(document)
        document = rich_document()
        document["operations"][0]["operation"]["value"] = ""
        self.assert_invalid(document)

    def test_out_of_log_cursors_are_400(self) -> None:
        document = rich_document()
        document["checkpoints"] = {"peer-a": 4}  # past the 3-record log
        self.assert_invalid(document)
        document = rich_document()
        document["checkpoints"] = {"peer-a": -1}
        self.assert_invalid(document)
        document = rich_document()
        document["acks"][0]["cursor"] = 4
        self.assert_invalid(document)
        document = rich_document()
        document["repairExecutions"][0]["cursor"] = 4
        self.assert_invalid(document)

    def test_inconsistent_references_are_400(self) -> None:
        # A policy binding naming no accepted operation.
        document = rich_document()
        document["policies"] = [
            {"replicaId": "r1", "operationId": "zz", "policy": "lowest_identity"}
        ]
        self.assert_invalid(document)
        # An unknown policy string.
        document = rich_document()
        document["policies"][0]["policy"] = "nope"
        self.assert_invalid(document)
        # A transaction entry naming no accepted operation.
        document = rich_document()
        document["transactions"][0]["operations"][0]["operationId"] = "zz"
        self.assert_invalid(document)
        # A receipt for a peer with no registered checkpoint.
        document = rich_document()
        document["acks"][0]["peerId"] = "peer-b"
        self.assert_invalid(document)
        # A receipt whose covered segment does not match the log.
        document = rich_document()
        document["acks"][0]["operations"][0]["operationId"] = "zz"
        self.assert_invalid(document)
        # A repair for a peer with no registered checkpoint.
        document = rich_document()
        document["repairExecutions"][0]["peerId"] = "peer-b"
        self.assert_invalid(document)
        # A compensation naming no committed transaction.
        document = rich_document()
        document["compensations"][0]["transactionId"] = "tx-zz"
        self.assert_invalid(document)
        # A compensation entry naming no accepted operation.
        document = rich_document()
        document["compensations"][0]["operations"][0]["operationId"] = "zz"
        self.assert_invalid(document)
        # A compensation status other than committed.
        document = rich_document()
        document["compensations"][0]["status"] = "planned"
        self.assert_invalid(document)
        # A policy-event history with a sequence gap.
        document = rich_document()
        document["policyEvents"] = [{"sequence": 2, "digest": "a" * 64, "tokens": 2}]
        self.assert_invalid(document)

    def test_wrong_section_types_are_400(self) -> None:
        for field, bad in (
            ("operations", {}),
            ("checkpoints", []),
            ("policies", {}),
            ("transactions", {}),
            ("acks", {}),
            ("repairExecutions", {}),
            ("policyEvents", {}),
            ("compensations", {}),
        ):
            with self.subTest(field=field):
                document = rich_document()
                document[field] = bad
                self.assert_invalid(document)


class StoreImportClockWidthTests(ImportTestCase):
    """An over-wide vector clock in the document is 400 under the bound."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved = server_module._MAX_CLOCK_COMPONENTS
        server_module._MAX_CLOCK_COMPONENTS = 2

    @classmethod
    def tearDownClass(cls) -> None:
        server_module._MAX_CLOCK_COMPONENTS = cls._saved

    def test_over_wide_operation_clock_is_400(self) -> None:
        server = self.start_server()
        document = rich_document()
        document["operations"][0]["operation"]["clock"] = {"r1": 1, "r2": 1, "r3": 1}
        status, payload = self.import_doc(server, document)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.export(server)[1], EMPTY_DOCUMENT)

    def test_document_within_the_bound_imports(self) -> None:
        server = self.start_server()
        document = rich_document()
        status, _ = self.import_doc(server, document)
        self.assertEqual(status, 201)


class StoreImportRouteTests(ImportTestCase):
    """Path-shape mismatches are 404; Content-Length keeps its priority."""

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        server = self.start_server()
        document = rich_document()
        for path in (
            "/v1/admin/store",
            "/v1/admin/store/import/extra",
            "/v1/admin/store/import/",
            "/v1/admin",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request(server, "POST", path, body=document)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
        self.assertEqual(self.export(server)[1], EMPTY_DOCUMENT)

    def test_get_on_the_import_path_is_404(self) -> None:
        server = self.start_server()
        status, payload, _, _ = self.request(server, "GET", IMPORT_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_and_malformed_content_length_are_400(self) -> None:
        server = self.start_server()
        status, payload = self.post_raw(server, IMPORT_PATH, [], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.post_raw(
            server, IMPORT_PATH, [("Content-Length", "abc")], b"{}"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_content_length_is_413(self) -> None:
        server = self.start_server()
        status, payload = self.post_raw(
            server,
            IMPORT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"junk",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_length_contract_ranks_above_authentication(self) -> None:
        # A 400/413 is answered before authentication, even with no
        # credential presented at all.
        server = self.start_server(auth_scopes=dict(POLICY))
        status, payload = self.post_raw(server, IMPORT_PATH, [], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.post_raw(
            server,
            IMPORT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"junk",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})


class StoreImportSingleTokenAuthTests(ImportTestCase):
    """The legacy single token imports; bad credentials are 401."""

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        server = self.start_server(auth_token=SINGLE_TOKEN)
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.request(
                    server, "POST", IMPORT_PATH, body=rich_document(), token=token
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        # The rejections committed nothing.
        status, payload, _, _ = self.export(server, token=SINGLE_TOKEN)
        self.assertEqual(payload, EMPTY_DOCUMENT)

    def test_the_configured_token_imports(self) -> None:
        server = self.start_server(auth_token=SINGLE_TOKEN)
        status, payload = self.import_doc(server, rich_document(), token=SINGLE_TOKEN)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})


class StoreImportScopePolicyAuthTests(ImportTestCase):
    """In scope-policy mode only the admin scope imports."""

    def start_policy_server(self) -> SemanticStateServer:
        return self.start_server(auth_scopes=dict(POLICY))

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        server = self.start_policy_server()
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.request(
                    server, "POST", IMPORT_PATH, body=rich_document(), token=token
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_read_and_write_tokens_are_403_without_challenge(self) -> None:
        server = self.start_policy_server()
        for token in (READ_TOKEN, WRITE_TOKEN, RW_TOKEN):
            with self.subTest(token=token):
                status, payload, headers, _ = self.request(
                    server, "POST", IMPORT_PATH, body=rich_document(), token=token
                )
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)
        # The rejections committed nothing: a later admin import of the
        # same document is still the first import.
        status, payload = self.import_doc(server, rich_document(), token=ADMIN_TOKEN)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "version": 1})

    def test_admin_token_imports(self) -> None:
        server = self.start_policy_server()
        status, payload = self.import_doc(server, rich_document(), token=ADMIN_TOKEN)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.export(server, token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload, rich_document())

    def test_rejected_requests_never_read_the_body(self) -> None:
        # Declare a valid length but never send the body: the 401/403
        # must arrive without the server waiting to read it.
        server = self.start_policy_server()
        for token, expected in ((None, 401), (READ_TOKEN, 403)):
            with self.subTest(token=token):
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_address[1], timeout=5
                )
                conn.putrequest("POST", IMPORT_PATH)
                conn.putheader("Content-Length", "64")
                if token is not None:
                    conn.putheader("Authorization", f"Bearer {token}")
                conn.endheaders()
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                self.assertEqual(response.status, expected)


class StoreImportConcurrencyTests(ImportTestCase):
    """Concurrent first imports commit exactly once."""

    def run_concurrent(self, documents: list[dict]) -> list[int]:
        server = self.start_server()
        barrier = threading.Barrier(len(documents))
        statuses: list[int] = [0] * len(documents)

        def submit(index: int, document: dict) -> None:
            barrier.wait()
            statuses[index], _ = self.import_doc(server, document)

        threads = [
            threading.Thread(target=submit, args=(index, document))
            for index, document in enumerate(documents)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        return statuses

    def test_identical_documents_commit_once(self) -> None:
        statuses = self.run_concurrent([rich_document() for _ in range(8)])
        self.assertEqual(sorted(statuses), [200] * 7 + [201])

    def test_different_documents_have_one_winner(self) -> None:
        documents = []
        for index in range(8):
            document = rich_document()
            document["operations"].append(
                {
                    "replicaId": "r9",
                    "operation": operation(f"win-{index}", "k9", "x", {"r9": 1}),
                }
            )
            documents.append(document)
        statuses = self.run_concurrent(documents)
        self.assertEqual(sorted(statuses), [201] + [409] * 7)


if __name__ == "__main__":
    unittest.main()
