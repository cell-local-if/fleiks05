"""HTTP, concurrency, persistence-failure, and recovery tests for key audit.

The key-audit endpoint is::

    GET /v1/audit/keys/{key}/operations?after=N&limit=N

It returns the shared accepted-operation log (the same records and commit
order as sync export and the data file) filtered to one key. Everything here
goes through the real HTTP entry point (``SemanticStateServer`` + a request
thread); only the Python standard library is used.
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


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


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

    def post_sync(self, body: object) -> tuple[int, object]:
        return self.request("POST", "/v1/sync/operations", body)

    def post_resolve(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve", body)

    def get_audit(self, key: str, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations{query}")

    def drain_audit(self, key: str, limit: int = 3) -> list[dict]:
        """Page through one key's audit stream using the public cursor."""
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.get_audit(key, f"?after={after}&limit={limit}")
            assert status == 200
            page = payload["operations"]
            seen.extend(page)
            self.assertEqual(payload["nextCursor"], after + len(page))
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        return seen


class AuditContentTests(HttpServerTestCase):
    def test_missing_key_history_is_empty_page(self) -> None:
        status, payload = self.get_audit("absent")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_filters_to_one_key_in_global_commit_order(self) -> None:
        timeline = [
            ("r1", operation("k-a", "k", "a", {"r1": 1})),
            ("r2", operation("x-a", "x", "xa", {"r2": 1})),
            ("r1", operation("k-b", "k", "b", {"r1": 2})),
            ("r3", operation("y-a", "y", "ya", {"r3": 1})),
            ("r2", operation("x-b", "x", "xb", {"r2": 2})),
            ("r1", operation("k-c", "k", "c", {"r1": 3})),
        ]
        for replica, op in timeline:
            self.assertEqual(self.post_operation(replica, op)[0], 201)

        status, payload = self.get_audit("k")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["operations"],
            [
                record("r1", operation("k-a", "k", "a", {"r1": 1})),
                record("r1", operation("k-b", "k", "b", {"r1": 2})),
                record("r1", operation("k-c", "k", "c", {"r1": 3})),
            ],
        )
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)
        # Each record keeps the synced {"replicaId","operation"} shape.
        for entry in payload["operations"]:
            self.assertEqual(set(entry), {"replicaId", "operation"})
            self.assertEqual(set(entry["operation"]), {"operationId", "key", "value", "clock"})

        # Other keys are independent streams.
        _, x_page = self.get_audit("x")
        self.assertEqual([e["operation"]["operationId"] for e in x_page["operations"]], ["x-a", "x-b"])
        _, y_page = self.get_audit("y")
        self.assertEqual([e["operation"]["operationId"] for e in y_page["operations"]], ["y-a"])

    def test_stale_writes_appear_in_accept_order(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("new", "k", "new", {"r1": 2}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r1", operation("stale", "k", "stale", {"r1": 1}))[0], 201
        )
        ops = self.drain_audit("k")
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["new", "stale"])

    def test_resolution_fix_appears_between_later_commits(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        self.assertEqual(self.post_resolve("k", body)[0], 201)
        self.post_operation("r4", operation("after", "k", "v4", {"r4": 1}))
        ops = self.drain_audit("k")
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1"), ("r4", "after")],
        )
        self.assertEqual(
            ops[2],
            record("r3", operation("fix-1", "k", "merged", {"r1": 1, "r2": 1, "r3": 1})),
        )

    def test_replays_conflicts_and_rejected_requests_leave_no_record(self) -> None:
        op = operation("o1", "k", "v1", {"r1": 1})
        self.assertEqual(self.post_operation("r1", op)[0], 201)
        # Identical replay: 200, no new audit record.
        self.assertEqual(self.post_operation("r1", op)[0], 200)
        # Same identity, different content: 409, no audit record.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "tampered", {"r1": 1}))[0],
            409,
        )
        # Malformed request: 400, no audit record.
        self.assertEqual(self.request("POST", "/v1/replicas/r1/operations", b"{not json")[0], 400)
        # Resolution rejected as 409: no audit record.
        self.assertEqual(
            self.post_resolve(
                "missing",
                resolution(
                    "r3",
                    "fix-x",
                    "missing",
                    "v",
                    {"r1": 1, "r3": 1},
                    [candidate("r1", "o1")],
                ),
            )[0],
            409,
        )
        ops = self.drain_audit("k")
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o1"])
        # The rejected resolution never landed on its target key either.
        _, missing = self.get_audit("missing")
        self.assertEqual(missing["operations"], [])

    def test_resolution_replay_appears_once(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        self.assertEqual(self.post_resolve("k", body)[0], 201)
        self.assertEqual(self.post_resolve("k", body)[0], 200)
        ops = self.drain_audit("k")
        self.assertEqual(
            [e["operation"]["operationId"] for e in ops], ["o1", "o2", "fix-1"]
        )

    def test_imported_records_including_stale_and_fixes_are_audited(self) -> None:
        body = {
            "operations": [
                record("r1", operation("i1", "k", "v1", {"r1": 1})),
                record("r2", operation("i2", "x", "v2", {"r2": 1})),
                # Stale relative to i1 but for the audited key.
                record("r1", operation("i3", "k", "old", {"r1": 0})),
            ]
        }
        self.assertEqual(self.post_sync(body)[0], 201)
        ops = self.drain_audit("k")
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r1", "i1"), ("r1", "i3")],
        )

    def test_url_encoded_key_is_supported(self) -> None:
        # "/" and " " in a key, percent-encoded in the path.
        key = "names/color wheel"
        self.post_operation("r1", operation("o1", key, "blue", {"r1": 1}))
        status, payload = self.get_audit("names%2Fcolor%20wheel")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["o1"])


class AuditPaginationTests(HttpServerTestCase):
    def test_per_key_cursor_ignores_other_keys(self) -> None:
        timeline = [
            ("r1", operation("k1", "k", "a", {"r1": 1})),
            ("r2", operation("x1", "x", "xa", {"r2": 1})),
            ("r1", operation("k2", "k", "b", {"r1": 2})),
            ("r2", operation("x2", "x", "xb", {"r2": 2})),
            ("r1", operation("k3", "k", "c", {"r1": 3})),
        ]
        for replica, op in timeline:
            self.post_operation(replica, op)

        status, first = self.get_audit("k", "?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in first["operations"]], ["k1", "k2"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)

        status, last = self.get_audit("k", "?after=2&limit=2")
        self.assertEqual([e["operation"]["operationId"] for e in last["operations"]], ["k3"])
        self.assertEqual(last["nextCursor"], 3)
        self.assertIs(last["hasMore"], False)

        # after == the key's record count is a valid empty tail even though
        # the global log is longer.
        status, tail = self.get_audit("k", "?after=3&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)

    def test_default_limit_is_100(self) -> None:
        for i in range(105):
            self.post_operation("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i + 1}))
        status, page = self.get_audit("k")
        self.assertEqual(status, 200)
        self.assertEqual(len(page["operations"]), 100)
        self.assertEqual(page["nextCursor"], 100)
        self.assertIs(page["hasMore"], True)
        status, rest = self.get_audit("k", "?after=100")
        self.assertEqual([e["operation"]["operationId"] for e in rest["operations"]], ["o100", "o101", "o102", "o103", "o104"])
        self.assertEqual(rest["nextCursor"], 105)
        self.assertIs(rest["hasMore"], False)

    def test_full_walk_with_limit_one_keeps_commit_order(self) -> None:
        for replica, op in [
            ("r1", operation("a", "k", "1", {"r1": 1})),
            ("r9", operation("z", "z", "z", {"r9": 1})),
            ("r2", operation("b", "k", "2", {"r2": 1})),
            ("r9", operation("y", "z", "y", {"r9": 2})),
            ("r3", operation("c", "k", "3", {"r3": 1})),
        ]:
            self.post_operation(replica, op)
        ops = self.drain_audit("k", limit=1)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r1", "a"), ("r2", "b"), ("r3", "c")],
        )


class AuditValidationTests(HttpServerTestCase):
    def test_invalid_after_and_limit_are_400(self) -> None:
        self.post_operation("r1", operation("o", "k", "v", {"r1": 1}))
        # Arabic-Indic digits U+0661 U+0662 ("12"): numerals, but not ASCII
        # decimal, so they must be rejected.
        non_ascii_twelve = "%D9%A1%D9%A2"
        for query in (
            "?after=-1",
            "?after=x",
            "?after=",
            "?after=1.5",
            "?after=%201",
            "?after=+1",
            f"?after={non_ascii_twelve}",
            "?limit=-1",
            "?limit=0",
            "?limit=101",
            "?limit=x",
            "?limit=",
            "?limit=1.0",
            "?after=1&bogus=2",
            "?after=1&after=2",
            "?limit=1&limit=2",
            "?bogus=1",
        ):
            status, payload = self.get_audit("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_after_past_key_record_count_is_400_but_other_keys_count_not(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("x1", "x", "v2", {"r2": 1}))
        self.post_operation("r3", operation("x2", "x", "v3", {"r3": 1}))
        # The key has one record even though the global log has three.
        status, payload = self.get_audit("k", "?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # A key with no history rejects every positive after.
        status, payload = self.get_audit("absent", "?after=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # after == the key's count is still fine.
        status, _ = self.get_audit("x", "?after=2")
        self.assertEqual(status, 200)

    def test_unknown_audit_route_shapes_are_404(self) -> None:
        for path in (
            "/v1/audit/keys/k/operations/extra",
            "/v1/audit/keys/k",
            "/v1/audit/keys",
            "/v1/audit",
            "/v1/audit/k/operations",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)


class AuditImportBatchTests(HttpServerTestCase):
    def test_batch_records_for_one_key_are_contiguous(self) -> None:
        self.post_operation("r0", operation("before", "k", "v0", {"r0": 1}))
        # One batch mixes the audited key with another key; the key's records
        # must land adjacently in the per-key stream (imports are indivisible).
        batch = {
            "operations": [
                record("r1", operation("i1", "k", "v1", {"r1": 1})),
                record("r2", operation("ix", "x", "vx", {"r2": 1})),
                record("r1", operation("i2", "k", "v2", {"r1": 2})),
            ]
        }
        self.assertEqual(self.post_sync(batch)[0], 201)
        self.post_operation("r3", operation("after", "k", "v3", {"r3": 1}))
        ops = self.drain_audit("k", limit=1)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r0", "before"), ("r1", "i1"), ("r1", "i2"), ("r3", "after")],
        )

    def test_conflicting_batch_leaves_no_audit_records(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = {
            "operations": [
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
                record("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        }
        self.assertEqual(self.post_sync(body)[0], 409)
        ops = self.drain_audit("k")
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o1"])


class AuditConcurrencyTests(HttpServerTestCase):
    def test_concurrent_commits_and_audit_reads_stay_consistent(self) -> None:
        thread_count = 10
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                status, _ = self.post_operation(
                    f"l{index}", operation(f"local-{index}", "k", "l", {f"l{index}": 1})
                )
                assert status == 201
                batch = {
                    "operations": [
                        record(f"s{index}a", operation(f"sync-{index}a", "k", "a", {f"s{index}a": 1})),
                        record(f"s{index}b", operation(f"sync-{index}b", "x", "b", {f"s{index}b": 1})),
                        record(f"s{index}c", operation(f"sync-{index}c", "k", "c", {f"s{index}c": 1})),
                    ]
                }
                status, payload = self.post_sync(batch)
                assert status == 201 and payload["accepted"] == 3
            except BaseException as exc:  # reported below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(50):
                    after = 0
                    while True:
                        status, payload = self.get_audit("k", f"?after={after}&limit=4")
                        assert status == 200
                        page = payload["operations"]
                        # Snapshot invariants: cursor math always agrees, a
                        # hasMore page is exactly full, and every record is
                        # for the audited key.
                        assert payload["nextCursor"] == after + len(page)
                        if payload["hasMore"]:
                            assert len(page) == 4
                        assert all(e["operation"]["key"] == "k" for e in page)
                        after = payload["nextCursor"]
                        if not payload["hasMore"]:
                            break
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        threads.append(threading.Thread(target=reader))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertEqual(errors, [])

        ops = self.drain_audit("k")
        identities = [(e["replicaId"], e["operation"]["operationId"]) for e in ops]
        # 10 local writes + 2 key-k records per batch, each identity once.
        self.assertEqual(len(identities), 3 * thread_count)
        self.assertEqual(len(set(identities)), len(identities))
        # Within the per-key stream each batch's two k records are adjacent
        # (the middle x record is filtered out) and in request order.
        positions = {identity: i for i, identity in enumerate(identities)}
        for i in range(thread_count):
            self.assertEqual(
                positions[(f"s{i}a", f"sync-{i}a")] + 1,
                positions[(f"s{i}c", f"sync-{i}c")],
            )


class PersistentAuditTestCase(unittest.TestCase):
    """Key audit against a data-file-backed server with real HTTP."""

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

    def drain(self, server: SemanticStateServer, key: str, limit: int = 2) -> list[dict]:
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.request(
                server, "GET", f"/v1/audit/keys/{key}/operations?after={after}&limit={limit}"
            )
            assert status == 200
            seen.extend(payload["operations"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return seen

    def seed_mixed_history(self, server: SemanticStateServer) -> list[tuple[str, str]]:
        """Local writes, a resolution, a stale write, other keys, and a batch."""
        expected: list[tuple[str, str]] = []
        for replica, op in (
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
        ):
            status, _ = self.request(server, "POST", f"/v1/replicas/{replica}/operations", op)
            assert status == 201
            expected.append((replica, op["operationId"]))
        body = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        status, _ = self.request(server, "POST", "/v1/states/k/resolve", body)
        assert status == 201
        expected.append(("r3", "fix-1"))
        # Stale write for k and unrelated writes for other keys.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("stale", "k", "old", {"r1": 0}),
        )
        assert status == 201
        expected.append(("r1", "stale"))
        for replica, op in (
            ("r7", operation("x1", "x", "x1", {"r7": 1})),
            ("r8", operation("y1", "y", "y1", {"r8": 1})),
        ):
            status, _ = self.request(server, "POST", f"/v1/replicas/{replica}/operations", op)
            assert status == 201
        batch = {
            "operations": [
                record("r4", operation("b1", "k", "b1", {"r4": 1})),
                record("r9", operation("bx", "x", "bx", {"r9": 1})),
                record("r4", operation("b2", "k", "b2", {"r4": 2})),
            ]
        }
        status, payload = self.request(server, "POST", "/v1/sync/operations", batch)
        assert status == 201 and payload["accepted"] == 3
        expected.extend([("r4", "b1"), ("r4", "b2")])
        return expected

    def test_persistence_failure_and_conflict_batch_leave_no_audit_records(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r0/operations",
            operation("o0", "k", "v0", {"r0": 1}),
        )
        self.assertEqual(status, 201)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations",
                operation("o1", "k", "v1", {"r1": 1}),
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # A conflicting batch under the failure also leaves no trace.
            status, payload = self.request(
                server,
                "POST",
                "/v1/sync/operations",
                {
                    "operations": [
                        record("r2", operation("o2", "k", "v2", {"r2": 1})),
                        record("r0", operation("o0", "k", "tampered", {"r0": 1})),
                    ]
                },
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "operation_conflict"})

        self.assertEqual(self.data_file.read_bytes(), before)
        ops = self.drain(server, "k")
        self.assertEqual([(e["replicaId"], e["operation"]["operationId"]) for e in ops], [("r0", "o0")])
        # The failed operation commits cleanly once persistence works again.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        ops = self.drain(server, "k")
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r0", "o0"), ("r1", "o1")],
        )

    def test_restart_preserves_audit_order_resume_stale_and_fixes(self) -> None:
        server = self.start_server()
        expected = self.seed_mixed_history(server)

        # Capture paged streams (multiple page sizes) before the restart.
        def pages(srv: SemanticStateServer, key: str, limit: int) -> list[dict]:
            out: list[dict] = []
            after = 0
            while True:
                status, payload = self.request(
                    srv, "GET", f"/v1/audit/keys/{key}/operations?after={after}&limit={limit}"
                )
                assert status == 200
                out.append(payload)
                after = payload["nextCursor"]
                if not payload["hasMore"]:
                    return out

        before_limit1 = pages(server, "k", 1)
        before_limit3 = pages(server, "k", 3)
        # Mid-stream resume point for post-restart comparison.
        status, resume_before = self.request(
            server, "GET", "/v1/audit/keys/k/operations?after=2&limit=2"
        )
        self.assertEqual(status, 200)
        # Cursor past the key's count was invalid before restart.
        status, _ = self.request(server, "GET", "/v1/audit/keys/k/operations?after=7&limit=2")
        self.assertEqual(status, 400)

        server.shutdown()
        server.server_close()
        server = self.start_server()

        after_limit1 = pages(server, "k", 1)
        after_limit3 = pages(server, "k", 3)

        def summarize(stream_pages: list[dict]) -> list[tuple[list[tuple[str, str]], int, bool]]:
            return [
                (
                    [(e["replicaId"], e["operation"]["operationId"]) for e in p["operations"]],
                    p["nextCursor"],
                    p["hasMore"],
                )
                for p in stream_pages
            ]

        # Full order, page boundaries, cursors, and hasMore survive restart.
        self.assertEqual(summarize(after_limit1), summarize(before_limit1))
        self.assertEqual(summarize(after_limit3), summarize(before_limit3))
        drained = self.drain(server, "k", limit=1)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in drained],
            expected,
        )
        # Mid-stream pagination resumes identically.
        status, resume_after = self.request(
            server, "GET", "/v1/audit/keys/k/operations?after=2&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(resume_after, resume_before)
        # The stale operation and the fix are present in the right slots.
        ids = [e["operation"]["operationId"] for e in drained]
        self.assertEqual(ids[2], "fix-1")
        self.assertEqual(ids[3], "stale")
        # Other keys stay isolated after restart.
        x_ops = self.drain(server, "x")
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in x_ops],
            [("r7", "x1"), ("r9", "bx")],
        )
        # Cursor bounds are unchanged by restart.
        status, payload = self.request(
            server, "GET", "/v1/audit/keys/k/operations?after=6&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        self.assertEqual(payload["nextCursor"], 6)
        status, payload = self.request(
            server, "GET", "/v1/audit/keys/k/operations?after=7&limit=2"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The recovered file is exactly the shared log, format unchanged.
        self.assertEqual(
            [(r, o["operationId"]) for r, o in load_data_file(str(self.data_file))],
            [
                ("r1", "o1"),
                ("r2", "o2"),
                ("r3", "fix-1"),
                ("r1", "stale"),
                ("r7", "x1"),
                ("r8", "y1"),
                ("r4", "b1"),
                ("r9", "bx"),
                ("r4", "b2"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
