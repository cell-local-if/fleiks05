"""Tests for the read-only full-store export.

``GET /v1/admin/store/export`` exports the complete committed store as
one version-1 document — the same document the persistence layer writes
to the data file — from a single atomic snapshot. The tests cover the
success contract (all eight sections plus ``version``, compact UTF-8
JSON with one trailing newline, empty sections as empty arrays/objects),
byte-equality with the data file and across a restart, the request
precedence chain (404 path shape, 401 authentication with a Bearer
challenge, 403 without one, 400 query validation), and the strictly
read-only guarantees (no file reads or writes, no temporary files, no
cursor or idempotency changes).
"""

import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
)

EXPORT_PATH = "/v1/admin/store/export"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
LEGACY_TOKEN = "legacy-token"

SCOPE_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}

EMPTY_EXPORT = {
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


def canonical(document: dict) -> bytes:
    """The exact byte shape of the export body (and the data file)."""
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


# A complete version-1 document exercising all eight sections at once,
# valid under the startup recovery constraints.
FULL_DOCUMENT = {
    "version": 1,
    "operations": [
        {
            "replicaId": "r1",
            "operation": operation("op-1", "color", "blue", {"r1": 1}),
        },
        {
            "replicaId": "r2",
            "operation": operation("op-2", "color", "green", {"r2": 1}),
        },
        {
            "replicaId": "r3",
            "operation": operation("op-3", "color", "blue", {"r1": 1, "r2": 1, "r3": 1}),
        },
        {
            "replicaId": "r1",
            "operation": operation("op-4", "size", "s", {"r1": 2}),
        },
        {
            "replicaId": "r1",
            "operation": operation("op-5", "size", "m", {"r1": 3}),
        },
    ],
    "checkpoints": {"peer-a": 2},
    "policies": [
        {"replicaId": "r3", "operationId": "op-3", "policy": "lowest_identity"},
    ],
    "transactions": [
        {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "size",
                    "replicaId": "r1",
                    "operationId": "op-4",
                    "value": "s",
                    "clock": {"r1": 2},
                    "candidates": [],
                },
            ],
        },
    ],
    "acks": [
        {
            "peerId": "peer-a",
            "ackId": "ack-1",
            "cursor": 2,
            "operations": [
                {"replicaId": "r1", "operationId": "op-1"},
                {"replicaId": "r2", "operationId": "op-2"},
            ],
        },
    ],
    "repairExecutions": [
        {
            "peerId": "peer-a",
            "ackId": "ack-2",
            "expectedCheckpoint": 1,
            "expectedReceipts": "a" * 64,
            "suggestions": [
                {
                    "action": "resend",
                    "ackId": "ack-2",
                    "location": {"start": 1, "end": 2},
                    "target": {"start": 1, "end": 2},
                },
            ],
            "results": [
                {"action": "resend", "boundary": {"start": 1, "end": 2}},
            ],
            "cursor": 2,
        },
    ],
    "policyEvents": [{"sequence": 1, "digest": "b" * 64, "tokens": 3}],
    "compensations": [
        {
            "compensationId": "comp-1",
            "transactionId": "tx-1",
            "expectedPlanDigest": "c" * 64,
            "operations": [
                {
                    "key": "size",
                    "replicaId": "r1",
                    "operationId": "op-5",
                    "value": "m",
                    "clock": {"r1": 3},
                },
            ],
            "status": "committed",
        },
    ],
}


class StoreExportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-store-export-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = os.path.join(self.tmpdir, "state.json")

    def populate(self, store: StateStore) -> None:
        """Commit one of every section through the live store API."""
        self.assertEqual(
            store.apply_operation("r1", operation("op-1", "color", "blue", {"r1": 1})), 201
        )
        self.assertEqual(
            store.apply_operation("r2", operation("op-2", "color", "green", {"r2": 1})), 201
        )
        status, _, _ = store.apply_auto_resolution(
            "color",
            {
                "replicaId": "r3",
                "operationId": "op-3",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "lowest_identity",
            },
        )
        self.assertEqual(status, 201)
        status, _, _, _, _ = store.apply_transaction(
            "tx-1",
            [
                {
                    "key": "size",
                    "replicaId": "r1",
                    "operationId": "op-4",
                    "value": "s",
                    "clock": {"r1": 2},
                    "candidates": [],
                },
            ],
        )
        self.assertEqual(status, 201)
        self.assertEqual(store.save_checkpoint("peer-a", 0), (200, None))
        status, _ = store.acknowledge_operations(
            "peer-a",
            "ack-1",
            2,
            [
                {"replicaId": "r1", "operationId": "op-1"},
                {"replicaId": "r2", "operationId": "op-2"},
            ],
        )
        self.assertEqual(status, 201)
        store.record_policy_reload("b" * 64, 3)

    def test_empty_memory_store_exports_all_sections_empty(self) -> None:
        store = StateStore()
        document = store.get_store_export()
        self.assertEqual(document, EMPTY_EXPORT)
        # Exactly the nine contracted root fields, nothing internal.
        self.assertEqual(
            set(document),
            {
                "version",
                "operations",
                "checkpoints",
                "policies",
                "transactions",
                "acks",
                "repairExecutions",
                "policyEvents",
                "compensations",
            },
        )

    def test_export_matches_committed_state_and_data_file_bytes(self) -> None:
        store = StateStore(data_file=self.data_file)
        self.populate(store)
        document = store.get_store_export()
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            [record["operation"]["operationId"] for record in document["operations"]],
            ["op-1", "op-2", "op-3", "op-4"],
        )
        self.assertEqual(document["checkpoints"], {"peer-a": 2})
        self.assertEqual(
            document["policies"],
            [{"replicaId": "r3", "operationId": "op-3", "policy": "lowest_identity"}],
        )
        self.assertEqual([record["transactionId"] for record in document["transactions"]], ["tx-1"])
        self.assertEqual([record["ackId"] for record in document["acks"]], ["ack-1"])
        self.assertEqual(document["repairExecutions"], [])
        self.assertEqual(
            document["policyEvents"], [{"sequence": 1, "digest": "b" * 64, "tokens": 3}]
        )
        self.assertEqual(document["compensations"], [])
        # The export is byte-identical to the committed data file.
        with open(self.data_file, "rb") as handle:
            self.assertEqual(canonical(document), handle.read())

    def test_export_is_identical_across_a_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        self.populate(store)
        before = canonical(store.get_store_export())
        recovered = StateStore(data_file=self.data_file)
        after = canonical(recovered.get_store_export())
        self.assertEqual(before, after)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(after, handle.read())

    def test_recovered_full_document_exports_all_sections(self) -> None:
        with open(self.data_file, "wb") as handle:
            handle.write(canonical(FULL_DOCUMENT))
        store = StateStore(data_file=self.data_file)
        document = store.get_store_export()
        # Every section round-trips field for field, in the stored order.
        self.assertEqual(document["operations"], FULL_DOCUMENT["operations"])
        self.assertEqual(document["checkpoints"], FULL_DOCUMENT["checkpoints"])
        self.assertEqual(document["policies"], FULL_DOCUMENT["policies"])
        self.assertEqual(document["transactions"], FULL_DOCUMENT["transactions"])
        self.assertEqual(document["acks"], FULL_DOCUMENT["acks"])
        self.assertEqual(document["repairExecutions"], FULL_DOCUMENT["repairExecutions"])
        self.assertEqual(document["policyEvents"], FULL_DOCUMENT["policyEvents"])
        self.assertEqual(document["compensations"], FULL_DOCUMENT["compensations"])
        self.assertEqual(canonical(document), canonical(FULL_DOCUMENT))

    def test_export_is_a_detached_snapshot(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        document = store.get_store_export()
        # Mutating the returned document must not leak into the store.
        document["operations"][0]["operation"]["value"] = "tampered"
        document["checkpoints"]["peer-x"] = 99
        document["transactions"].append({"transactionId": "ghost"})
        fresh = store.get_store_export()
        self.assertEqual(fresh["operations"][0]["operation"]["value"], "v")
        self.assertEqual(fresh["checkpoints"], {})
        self.assertEqual(fresh["transactions"], [])

    def test_export_changes_neither_memory_nor_the_data_file(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        with open(self.data_file, "rb") as handle:
            before = handle.read()
        store.get_store_export()
        store.get_store_export()
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), before)
        # No temporary files were created.
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["state.json"])
        # Idempotency judgments are unchanged: replay 200, conflict 409.
        self.assertEqual(
            store.apply_operation("r1", operation("op-1", "k", "v", {"r1": 1})), 200
        )
        self.assertEqual(
            store.apply_operation("r1", operation("op-1", "k", "other", {"r1": 1})), 409
        )
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), before)


class StoreExportHttpTests(unittest.TestCase):
    """The endpoint over HTTP, in anonymous mode with a data file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-store-export-http-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=cls.data_path
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        if os.path.lexists(self.data_path):
            os.unlink(self.data_path)
        self.server.store = StateStore(data_file=self.data_path)

    def request(self, method: str, path: str, body: object = None, headers: dict = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        all_headers = {"Content-Type": "application/json"}
        all_headers.update(headers or {})
        if body is None:
            conn.request(method, path, headers=all_headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=all_headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def export(self, query: str = ""):
        return self.request("GET", EXPORT_PATH + query)

    def post_operation(self, replica: str, body: dict) -> int:
        status, _, _, _ = self.request("POST", f"/v1/replicas/{replica}/operations", body)
        return status

    # -- success contract --

    def test_empty_store_export_is_compact_json_with_a_single_newline(self) -> None:
        status, payload, raw, headers = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_EXPORT)
        self.assertEqual(raw, canonical(EMPTY_EXPORT) + b"\n")
        self.assertTrue(raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")

    def test_export_reflects_commits_and_matches_the_data_file(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "color", "blue", {"r1": 1})), 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("op-2", "color", "green", {"r2": 1})), 201
        )
        status, payload, raw, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(
            [record["operation"]["operationId"] for record in payload["operations"]],
            ["op-1", "op-2"],
        )
        with open(self.data_path, "rb") as handle:
            self.assertEqual(raw, handle.read() + b"\n")

    def test_export_is_identical_across_a_server_restart(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "color", "blue", {"r1": 1})), 201
        )
        _, _, before, _ = self.export()
        self.server.store = StateStore(data_file=self.data_path)
        status, _, after, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

    def test_local_bindings_are_exported_but_never_enter_the_sync_stream(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1})), 201
        )
        status, _, _, _ = self.request(
            "POST", "/v1/transactions/apply",
            {
                "transactionId": "tx-1",
                "operations": [
                    {
                        "key": "other",
                        "replicaId": "r1",
                        "operationId": "op-2",
                        "value": "w",
                        "clock": {"r1": 2},
                        "candidates": [],
                    },
                ],
            },
        )
        self.assertEqual(status, 201)
        status, payload, _, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(
            [record["transactionId"] for record in payload["transactions"]], ["tx-1"]
        )
        # The incremental sync stream carries only the operations.
        status, sync_payload, _, _ = self.request("GET", "/v1/sync/operations?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            [record["operation"]["operationId"] for record in sync_payload["operations"]],
            ["op-1", "op-2"],
        )
        self.assertNotIn("transactions", sync_payload)

    # -- route shape precedes the query check --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            "/v1/admin/store/export/",
            "/v1/admin/store/export/extra",
            "/v1/admin/store",
            "/v1/admin/store/exportx",
            "/v1/admin",
            "/admin/store/export",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request("GET", path + "?bogus=1")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    # -- query validation --

    def test_any_query_parameter_is_400(self) -> None:
        for query in ("?x=1", "?x=", "?x", "?=1", "?x=1&x=2", "?after=0", "?x=1&y=2"):
            with self.subTest(query=query):
                status, payload, raw, _ = self.export(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    # -- the export is strictly read-only --

    def test_export_and_rejected_requests_change_nothing(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1})), 201
        )
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        self.export()
        self.export("?x=1")
        self.request("GET", EXPORT_PATH + "/")
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        # Idempotency is untouched: identical replay 200, conflict 409.
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1})), 200
        )
        self.assertEqual(
            self.post_operation("r1", operation("op-1", "k", "other", {"r1": 1})), 409
        )
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class StoreExportAuthTests(unittest.TestCase):
    """Authentication and scope enforcement of the export endpoint."""

    def serve(self, **kwargs):
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server.server_address[1]

    def raw_request(self, port: int, authorization_headers: list, path: str = EXPORT_PATH):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("GET", path)
        for value in authorization_headers:
            conn.putheader("Authorization", value)
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read()
        headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, headers

    def test_single_token_mode(self) -> None:
        port = self.serve(auth_token=LEGACY_TOKEN)
        # Missing, mismatched, malformed, and duplicated headers are 401
        # with the Bearer challenge.
        for headers in (
            [],
            [f"Bearer {LEGACY_TOKEN}-wrong"],
            [f"Token {LEGACY_TOKEN}"],
            [f"Bearer {LEGACY_TOKEN}", f"Bearer {LEGACY_TOKEN}"],
        ):
            with self.subTest(headers=headers):
                status, payload, response_headers = self.raw_request(port, headers)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(response_headers["WWW-Authenticate"], "Bearer")
        # The legacy token keeps its unrestricted access.
        status, payload, _ = self.raw_request(port, [f"Bearer {LEGACY_TOKEN}"])
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_EXPORT)

    def test_scope_policy_mode_requires_admin(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="sestate-store-export-auth-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        policy_path = os.path.join(tmpdir, "scopes.json")
        with open(policy_path, "wb") as handle:
            handle.write(json.dumps(SCOPE_POLICY).encode("utf-8"))
        port = self.serve(
            auth_scopes=dict(load_scope_policy(policy_path)),
            scope_policy_file=policy_path,
        )
        # A missing or bad credential is 401 with the challenge.
        status, payload, headers = self.raw_request(port, [])
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.raw_request(port, ["Bearer nobody"])
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Read-only and write-only tokens are 403 without a challenge.
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                status, payload, headers = self.raw_request(port, [f"Bearer {token}"])
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)
        # The scope decision precedes the query check: a bad query for a
        # read-only token is still 403, never downgraded to 400.
        status, _, _ = self.raw_request(
            port, [f"Bearer {READ_TOKEN}"], path=EXPORT_PATH + "?x=1"
        )
        self.assertEqual(status, 403)
        # An admin token reaches the snapshot.
        status, payload, _ = self.raw_request(port, [f"Bearer {ADMIN_TOKEN}"])
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_EXPORT)

    def test_health_stays_anonymous(self) -> None:
        port = self.serve(auth_token=LEGACY_TOKEN)
        status, payload, _ = self.raw_request(port, [], path="/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


if __name__ == "__main__":
    unittest.main()
