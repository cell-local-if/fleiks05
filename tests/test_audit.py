"""HTTP, concurrency, persistence-failure, and recovery tests for the audit log.

The per-key audit endpoint is::

    GET /v1/audit/keys/{key}/operations?after=N&limit=N

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


class HttpServerTestCase(unittest.TestCase):
    """Spin up one in-memory server per class; reset the store per test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        elif isinstance(body, (bytes, str)):
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_audit(self, key: str, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations{query}")

    def drain_audit(self, key: str) -> list[dict]:
        """Page through the key's whole stream using the public cursor protocol."""
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.get_audit(key, f"?after={after}&limit=2")
            assert status == 200
            page = payload["operations"]
            seen.extend(page)
            self.assertEqual(payload["nextCursor"], after + len(page))
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        return seen


class AuditFilterTests(HttpServerTestCase):
    def test_empty_history_is_an_empty_page(self) -> None:
        status, payload = self.get_audit("missing")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_only_the_path_key_is_returned(self) -> None:
        self.post_operation("r1", operation("a", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("b", "size", "xl", {"r2": 1}))
        self.post_operation("r1", operation("c", "color", "green", {"r1": 2}))

        status, payload = self.get_audit("color")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["operations"],
            [
                record("r1", operation("a", "color", "blue", {"r1": 1})),
                record("r1", operation("c", "color", "green", {"r1": 2})),
            ],
        )
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        # Records carry the sync shape and nothing else.
        for entry in payload["operations"]:
            self.assertEqual(set(entry), {"replicaId", "operation"})
            self.assertEqual(set(entry["operation"]), {"operationId", "key", "value", "clock"})

        # The other key's stream is independent and complete.
        status, payload = self.get_audit("size")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["b"])

    def test_stale_writes_are_audited_in_accept_order(self) -> None:
        self.post_operation("r1", operation("new", "k", "new", {"r1": 2}))
        self.post_operation("r1", operation("stale", "k", "stale", {"r1": 1}))
        ops = self.drain_audit("k")
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["new", "stale"])

    def test_accepted_resolution_is_audited(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        resolution = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = self.request("POST", "/v1/states/k/resolve", resolution)
        self.assertEqual(status, 201)
        ops = self.drain_audit("k")
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1")],
        )
        # The resolution record is an ordinary operation on the audited key.
        resolution_record = ops[-1]["operation"]
        self.assertEqual(resolution_record["key"], "k")
        self.assertEqual(resolution_record["value"], "merged")
        self.assertEqual(resolution_record["clock"], {"r1": 1, "r2": 1, "r3": 1})

    def test_rejected_requests_leave_no_audit_records(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        # Conflicting replay of a known identity.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "tampered", {"r1": 1}))[0], 409
        )
        # Resolution on a key that is not in conflict.
        self.assertEqual(
            self.request(
                "POST",
                "/v1/states/k/resolve",
                {
                    "replicaId": "r2",
                    "operationId": "fix",
                    "value": "x",
                    "clock": {"r1": 1, "r2": 1},
                    "candidates": [{"replicaId": "r1", "operationId": "o1"}],
                },
            )[0],
            409,
        )
        ops = self.drain_audit("k")
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o1"])

    def test_key_with_escaped_characters(self) -> None:
        self.post_operation("r1", operation("o1", "a b/c", "v", {"r1": 1}))
        status, payload = self.get_audit("a%20b%2Fc")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["key"] for e in payload["operations"]], ["a b/c"])
        self.assertEqual(self.get_audit("a%20b")[1]["operations"], [])

    def test_audit_and_sync_share_one_commit_order(self) -> None:
        self.post_operation("r1", operation("a", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("b", "other", "v2", {"r2": 1}))
        self.post_operation("r1", operation("c", "k", "v3", {"r1": 2}))
        audit = self.drain_audit("k")
        sync = self.request("GET", "/v1/sync/operations")[1]["operations"]
        # The audit stream is exactly the sync stream filtered on the key.
        self.assertEqual(
            audit,
            [entry for entry in sync if entry["operation"]["key"] == "k"],
        )


class AuditPaginationTests(HttpServerTestCase):
    def seed(self) -> None:
        for i in range(5):
            self.post_operation("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i + 1}))

    def test_pagination_walks_the_key_stream(self) -> None:
        self.seed()
        status, first = self.get_audit("k", "?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in first["operations"]], ["o0", "o1"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)

        status, middle = self.get_audit("k", f"?after={first['nextCursor']}&limit=2")
        self.assertEqual([e["operation"]["operationId"] for e in middle["operations"]], ["o2", "o3"])
        self.assertEqual(middle["nextCursor"], 4)
        self.assertIs(middle["hasMore"], True)

        status, last = self.get_audit("k", f"?after={middle['nextCursor']}&limit=2")
        self.assertEqual([e["operation"]["operationId"] for e in last["operations"]], ["o4"])
        self.assertEqual(last["nextCursor"], 5)
        self.assertIs(last["hasMore"], False)

        # after == stream length is a valid empty tail.
        status, tail = self.get_audit("k", "?after=5")
        self.assertEqual(status, 200)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)

    def test_default_limit_is_100(self) -> None:
        self.seed()
        status, payload = self.get_audit("k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["nextCursor"], 5)
        self.assertIs(payload["hasMore"], False)

    def test_after_counts_only_this_key_records(self) -> None:
        self.post_operation("r1", operation("a", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("b", "other", "v2", {"r2": 1}))
        self.post_operation("r1", operation("c", "k", "v3", {"r1": 2}))
        # Skipping one record of the key's stream lands on "c", even though
        # "c" sits at position 2 of the shared commit log.
        status, payload = self.get_audit("k", "?after=1")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["c"])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)

    def test_invalid_after_and_limit_are_400(self) -> None:
        self.post_operation("r1", operation("o", "k", "v", {"r1": 1}))
        for query in (
            "?after=-1",
            "?after=x",
            "?after=",
            "?after=1.5",
            "?after=%201",
            "?after=%EF%BC%91",  # full-width digit 1
            "?limit=-1",
            "?limit=0",
            "?limit=101",
            "?limit=x",
            "?limit=",
            "?limit=%EF%BC%91",
            "?after=1&bogus=2",
            "?after=1&after=2",
            "?limit=1&limit=2",
        ):
            status, payload = self.get_audit("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_after_past_end_of_key_stream_is_400(self) -> None:
        self.post_operation("r1", operation("o", "k", "v", {"r1": 1}))
        status, payload = self.get_audit("k", "?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The bound is per key: the shared log is longer than this stream.
        self.post_operation("r2", operation("x", "other", "v", {"r2": 1}))
        self.assertEqual(self.get_audit("k", "?after=2")[0], 400)
        self.assertEqual(self.get_audit("k", "?after=1")[0], 200)

    def test_unknown_audit_route_is_404(self) -> None:
        for path in (
            "/v1/audit/keys/k",
            "/v1/audit/keys/k/operations/extra",
            "/v1/audit",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)


class AuditConcurrencyTests(HttpServerTestCase):
    def test_import_batches_are_contiguous_in_the_audit_stream(self) -> None:
        thread_count = 10
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                batch = {
                    "operations": [
                        record(f"s{index}a", operation(f"sync-{index}a", "k", "a", {f"s{index}a": 1})),
                        record(f"s{index}b", operation(f"sync-{index}b", "k", "b", {f"s{index}b": 1})),
                    ]
                }
                status, payload = self.request("POST", "/v1/sync/operations", batch)
                assert status == 201 and payload["accepted"] == 2
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])

        ops = self.drain_audit("k")
        identities = [(e["replicaId"], e["operation"]["operationId"]) for e in ops]
        self.assertEqual(len(identities), 2 * thread_count)
        self.assertEqual(len(set(identities)), len(identities))
        # Each batch's two records land adjacently and in request order.
        positions = {identity: i for i, identity in enumerate(identities)}
        for i in range(thread_count):
            self.assertEqual(
                positions[(f"s{i}a", f"sync-{i}a")] + 1, positions[(f"s{i}b", f"sync-{i}b")]
            )


class PersistentAuditTestCase(unittest.TestCase):
    """Audit against a data-file-backed server with real HTTP."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def drain(self, server: SemanticStateServer, key: str) -> list[dict]:
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.request(
                server, "GET", f"/v1/audit/keys/{key}/operations?after={after}&limit=2"
            )
            assert status == 200
            seen.extend(payload["operations"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return seen

    def test_restart_preserves_order_resume_stale_and_repairs(self) -> None:
        server = self.start_server()
        self.request(server, "POST", "/v1/replicas/r1/operations", operation("a", "k", "v1", {"r1": 1}))
        self.request(server, "POST", "/v1/replicas/r2/operations", operation("b", "k", "v2", {"r2": 1}))
        # Stale relative to r1's clock: recorded, adds no candidate.
        self.request(server, "POST", "/v1/replicas/r1/operations", operation("stale", "k", "old", {"r1": 0}))
        self.request(server, "POST", "/v1/replicas/r9/operations", operation("noise", "other", "x", {"r9": 1}))
        resolution = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "a"},
                {"replicaId": "r2", "operationId": "b"},
            ],
        }
        self.assertEqual(self.request(server, "POST", "/v1/states/k/resolve", resolution)[0], 201)
        expected = [("r1", "a"), ("r2", "b"), ("r1", "stale"), ("r3", "fix-1")]
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in self.drain(server, "k")],
            expected,
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The audit order, including the stale write and the repair, is
        # identical to a process that never restarted.
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in self.drain(server, "k")],
            expected,
        )
        # Resume from a mid-stream cursor.
        status, page = self.request(server, "GET", "/v1/audit/keys/k/operations?after=2&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            expected[2:],
        )
        self.assertEqual(page["nextCursor"], 4)
        self.assertIs(page["hasMore"], False)
        # The other key's stream survived recovery untouched.
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in self.drain(server, "other")],
            [("r9", "noise")],
        )

    def test_persistence_failure_leaves_no_audit_records(self) -> None:
        server = self.start_server()
        self.request(server, "POST", "/v1/replicas/r0/operations", operation("o0", "k", "v0", {"r0": 1}))
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server, "POST", "/v1/replicas/r1/operations", operation("o1", "k", "v1", {"r1": 1})
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # Neither the audit stream nor the file picked up the failed write.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(
            [e["operation"]["operationId"] for e in self.drain(server, "k")], ["o0"]
        )
        # The failed operation commits cleanly once persistence works again.
        status, _ = self.request(
            server, "POST", "/v1/replicas/r1/operations", operation("o1", "k", "v1", {"r1": 1})
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            [e["operation"]["operationId"] for e in self.drain(server, "k")], ["o0", "o1"]
        )

    def test_conflicting_import_batch_leaves_no_audit_records(self) -> None:
        server = self.start_server()
        self.request(server, "POST", "/v1/replicas/r1/operations", operation("o1", "k", "v1", {"r1": 1}))
        before = self.data_file.read_bytes()
        batch = {
            "operations": [
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
                record("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        }
        status, payload = self.request(server, "POST", "/v1/sync/operations", batch)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(
            [e["operation"]["operationId"] for e in self.drain(server, "k")], ["o1"]
        )
        # Recovery agrees: the rolled-back batch left nothing behind.
        reloaded = StateStore(data_file=str(self.data_file))
        page, next_cursor, has_more = reloaded.get_audit_operations("k", 0, 100)
        self.assertEqual([e["operation"]["operationId"] for e in page], ["o1"])
        self.assertEqual((next_cursor, has_more), (1, False))
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)


if __name__ == "__main__":
    unittest.main()
