"""HTTP, concurrency, persistence-failure, and recovery tests for sync.

The incremental sync endpoints are::

    GET  /v1/sync/operations?after=N&limit=N
    POST /v1/sync/operations

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def post_sync(self, body: object) -> tuple[int, object]:
        return self.request("POST", "/v1/sync/operations", body)

    def drain_sync(self) -> list[dict]:
        """Page through the whole log using the public cursor protocol."""
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.get_sync(f"?after={after}&limit=3")
            assert status == 200
            page = payload["operations"]
            seen.extend(page)
            self.assertEqual(payload["nextCursor"], after + len(page))
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        return seen


class SyncExportTests(HttpServerTestCase):
    def seed(self, *items: tuple[str, dict]) -> None:
        for replica, op in items:
            self.post_operation(replica, op)

    def test_empty_log(self) -> None:
        status, payload = self.get_sync()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_default_limit_is_100_and_shape_is_records(self) -> None:
        for i in range(3):
            self.post_operation("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i + 1}))
        status, payload = self.get_sync()
        self.assertEqual(status, 200)
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(
            payload["operations"],
            [
                record("r1", operation("o0", "k", "v0", {"r1": 1})),
                record("r1", operation("o1", "k", "v1", {"r1": 2})),
                record("r1", operation("o2", "k", "v2", {"r1": 3})),
            ],
        )
        # Each record carries only replicaId and operation.
        for entry in payload["operations"]:
            self.assertEqual(set(entry), {"replicaId", "operation"})
            self.assertEqual(set(entry["operation"]), {"operationId", "key", "value", "clock"})

    def test_pagination_walks_commit_order(self) -> None:
        timeline = [
            ("r1", operation("a", "k", "1", {"r1": 1})),
            ("r2", operation("b", "k", "2", {"r2": 1})),
            ("r1", operation("c", "k", "3", {"r1": 2, "r2": 1})),
            ("r2", operation("d", "k", "4", {"r2": 2})),
            ("r1", operation("e", "k", "5", {"r1": 3})),
        ]
        self.seed(*timeline)

        status, first = self.get_sync("?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        self.assertEqual([e["operation"]["operationId"] for e in first["operations"]], ["a", "b"])

        status, middle = self.get_sync(f"?after={first['nextCursor']}&limit=2")
        self.assertEqual(middle["nextCursor"], 4)
        self.assertIs(middle["hasMore"], True)
        self.assertEqual([e["operation"]["operationId"] for e in middle["operations"]], ["c", "d"])

        status, last = self.get_sync(f"?after={middle['nextCursor']}&limit=2")
        self.assertEqual(last["nextCursor"], 5)
        self.assertIs(last["hasMore"], False)
        self.assertEqual([e["operation"]["operationId"] for e in last["operations"]], ["e"])

        # Resuming at a cursor returns exactly the committed suffix.
        status, resume = self.get_sync("?after=3&limit=100")
        self.assertEqual([e["operation"]["operationId"] for e in resume["operations"]], ["d", "e"])
        self.assertEqual(resume["nextCursor"], 5)

        # after == length is a valid empty tail.
        status, tail = self.get_sync("?after=5")
        self.assertEqual(status, 200)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)

    def test_stale_writes_are_exported_in_accept_order(self) -> None:
        self.post_operation("r1", operation("new", "k", "new", {"r1": 2}))
        self.post_operation("r1", operation("stale", "k", "stale", {"r1": 1}))
        ops = self.drain_sync()
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["new", "stale"])

    def test_invalid_after_and_limit_are_400(self) -> None:
        self.post_operation("r1", operation("o", "k", "v", {"r1": 1}))
        for query in (
            "?after=-1",
            "?after=x",
            "?after=",
            "?after=1.5",
            "?after=%201",
            "?limit=-1",
            "?limit=0",
            "?limit=101",
            "?limit=x",
            "?limit=",
            "?after=1&bogus=2",
            "?after=1&after=2",
            "?limit=1&limit=2",
        ):
            status, payload = self.get_sync(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_after_past_end_is_400(self) -> None:
        self.post_operation("r1", operation("o", "k", "v", {"r1": 1}))
        status, payload = self.get_sync("?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_unknown_sync_route_is_404(self) -> None:
        status, payload = self.request("GET", "/v1/sync/operations/extra")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class SyncImportTests(HttpServerTestCase):
    def test_import_new_operations_is_201(self) -> None:
        body = {
            "operations": [
                record("r1", operation("o1", "k", "v1", {"r1": 1})),
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
            ]
        }
        status, payload = self.post_sync(body)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 2, "replayed": 0})
        # Unknown identities were accepted with normal write semantics.
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)

    def test_mixed_batch_counts_accepted_and_replayed(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = {
            "operations": [
                record("r1", operation("o1", "k", "v1", {"r1": 1})),
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
            ]
        }
        status, payload = self.post_sync(body)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 1, "replayed": 1})

    def test_all_replay_is_200_ok(self) -> None:
        op = operation("o1", "k", "v1", {"r1": 1})
        self.post_operation("r1", op)
        status, payload = self.post_sync({"operations": [record("r1", op), record("r1", op)]})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "accepted": 0, "replayed": 2})
        # Replays append nothing.
        ops = self.drain_sync()
        self.assertEqual(len(ops), 1)

    def test_stale_import_is_accepted_but_adds_no_candidate(self) -> None:
        self.post_operation("r1", operation("new", "k", "new", {"r1": 2}))
        body = {"operations": [record("r1", operation("old", "k", "stale", {"r1": 1}))]}
        status, payload = self.post_sync(body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "new")
        # The stale write still travels in the log.
        ops = self.drain_sync()
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["new", "old"])

    def test_within_batch_duplicate_identity_is_replayed_or_conflicts(self) -> None:
        op = operation("o1", "k", "v1", {"r1": 1})
        # Same identity twice with identical content: one accept, one replay.
        status, payload = self.post_sync({"operations": [record("r1", op), record("r1", dict(op))]})
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 1, "replayed": 1})
        # A brand-new identity appearing twice with different content: the
        # second occurrence conflicts against the first staged in this batch.
        status, payload = self.post_sync(
            {
                "operations": [
                    record("r2", operation("o9", "k", "v9", {"r2": 1})),
                    record("r9", operation("zz", "k", "zz", {"r9": 1})),
                    record("r2", operation("o9", "k", "tampered", {"r2": 1})),
                ]
            }
        )
        self.assertEqual(status, 409)
        ops = self.drain_sync()
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o1"])

    def test_conflict_is_409_and_batch_is_unchanged(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = {
            "operations": [
                # A brand-new record placed before the conflict must not commit.
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
                record("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        }
        status, payload = self.post_sync(body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # The whole batch is absent: only the pre-existing operation remains.
        ops = self.drain_sync()
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o1"])
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v1")
        # The rolled-back identity still accepts its genuine content later.
        status, _ = self.post_sync(
            {"operations": [record("r2", operation("o2", "k", "v2", {"r2": 1}))]}
        )
        self.assertEqual(status, 201)

    def test_invalid_bodies_are_400(self) -> None:
        valid_op = operation("o1", "k", "v", {"r1": 1})
        bad_bodies = [
            b"{not json",
            [],
            {},
            {"extra": 1},
            {"operations": []},
            {"operations": [record("r1", valid_op)], "extra": 1},
            {"operations": "nope"},
            {"operations": [record("r1", valid_op)] * 101},
            {"operations": [{"operation": valid_op}]},
            {"operations": [{"replicaId": "r1"}]},
            {"operations": [{"replicaId": "r1", "operation": valid_op, "x": 1}]},
            {"operations": [{"replicaId": "", "operation": valid_op}]},
            {"operations": [{"replicaId": 42, "operation": valid_op}]},
            {"operations": [{"replicaId": "r1", "operation": {}}]},
            {"operations": [record("r2", valid_op)]},  # clock must contain replica
            {"operations": [record("r1", operation("o1", "", "v", {"r1": 1}))]},
            {"operations": [record("r1", operation("o1", "k", "v", {"r1": -1}))]},
        ]
        for body in bad_bodies:
            status, payload = self.post_sync(body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        # Nothing invalid was imported.
        status, empty = self.get_sync()
        self.assertEqual(empty["operations"], [])

    def test_exactly_100_records_are_accepted(self) -> None:
        body = {
            "operations": [
                record(f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}))
                for i in range(100)
            ]
        }
        status, payload = self.post_sync(body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 100)

    def test_imported_batch_shares_commit_order(self) -> None:
        self.post_operation("r0", operation("local", "k", "v0", {"r0": 1}))
        batch = [
            record("r1", operation("i1", "k", "v1", {"r1": 1})),
            record("r2", operation("i2", "k", "v2", {"r2": 1})),
        ]
        self.assertEqual(self.post_sync({"operations": batch})[0], 201)
        self.post_operation("r3", operation("after", "k", "v3", {"r3": 1}))
        ops = self.drain_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops],
            [("r0", "local"), ("r1", "i1"), ("r2", "i2"), ("r3", "after")],
        )


class SyncConcurrencyTests(HttpServerTestCase):
    def test_local_writes_and_imports_share_one_commit_order(self) -> None:
        thread_count = 10
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                status, _ = self.post_operation(
                    f"l{index}", operation(f"local-{index}", "k", f"l{index}", {f"l{index}": 1})
                )
                assert status == 201
                batch = {
                    "operations": [
                        record(f"s{index}a", operation(f"sync-{index}a", "k", "a", {f"s{index}a": 1})),
                        record(f"s{index}b", operation(f"sync-{index}b", "k", "b", {f"s{index}b": 1})),
                    ]
                }
                status, payload = self.post_sync(batch)
                assert status == 201 and payload["accepted"] == 2
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])

        ops = self.drain_sync()
        # Every committed operation appears exactly once in a single order.
        identities = [(e["replicaId"], e["operation"]["operationId"]) for e in ops]
        self.assertEqual(len(identities), 3 * thread_count)
        self.assertEqual(len(set(identities)), len(identities))
        # Each batch's two records land adjacently and in request order.
        positions = {identity: i for i, identity in enumerate(identities)}
        for i in range(thread_count):
            self.assertEqual(positions[(f"s{i}a", f"sync-{i}a")] + 1, positions[(f"s{i}b", f"sync-{i}b")])


class PersistentSyncTestCase(unittest.TestCase):
    """Sync against a data-file-backed server with real HTTP."""

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

    def drain(self, server: SemanticStateServer) -> list[dict]:
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.request(server, "GET", f"/v1/sync/operations?after={after}&limit=2")
            assert status == 200
            seen.extend(payload["operations"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return seen

    def test_batch_is_durable_as_one_unit_before_success(self) -> None:
        server = self.start_server()
        batch = {
            "operations": [
                record("r1", operation("o1", "k", "v1", {"r1": 1})),
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
            ]
        }
        status, payload = self.request(server, "POST", "/v1/sync/operations", batch)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        # The file already contains the whole batch (one committed document).
        records = load_data_file(str(self.data_file))
        self.assertEqual([r[0] for r in records], ["r1", "r2"])
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        # Seed one durable operation so replay-under-failure is observable.
        seed = record("r0", operation("o0", "k", "v0", {"r0": 1}))
        self.assertEqual(self.request(server, "POST", "/v1/sync/operations", {"operations": [seed]})[0], 201)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            batch = {
                "operations": [
                    record("r1", operation("o1", "k", "v1", {"r1": 1})),
                    record("r2", operation("o2", "k", "v2", {"r2": 1})),
                ]
            }
            status, payload = self.request(server, "POST", "/v1/sync/operations", batch)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # A pure replay needs no durable write and still succeeds.
            status, payload = self.request(
                server, "POST", "/v1/sync/operations", {"operations": [seed]}
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"status": "ok", "accepted": 0, "replayed": 1})

        # Memory, identity index, and file are unchanged.
        self.assertEqual(self.data_file.read_bytes(), before)
        ops = self.drain(server)
        self.assertEqual([e["operation"]["operationId"] for e in ops], ["o0"])
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "v0")
        reloaded = StateStore(data_file=str(self.data_file))
        status, _ = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)
        # The failed batch commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", "/v1/sync/operations", batch)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_restart_preserves_order_resume_replay_and_conflict(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("local", "k", "v1", {"r1": 1}),
        )
        batch = {
            "operations": [
                record("r2", operation("sync-new", "k", "v2", {"r2": 1})),
                # Stale relative to r1's clock: recorded, adds no candidate.
                record("r1", operation("sync-stale", "k", "old", {"r1": 0})),
            ]
        }
        self.assertEqual(self.request(server, "POST", "/v1/sync/operations", batch)[0], 201)
        expected_order = [("r1", "local"), ("r2", "sync-new"), ("r1", "sync-stale")]

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Full export order is unchanged across the restart.
        ops = self.drain(server)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in ops], expected_order
        )
        # Resume from a mid-log cursor.
        status, page = self.request(server, "GET", "/v1/sync/operations?after=1&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            expected_order[1:],
        )
        self.assertEqual(page["nextCursor"], 3)
        self.assertIs(page["hasMore"], False)
        # Replays after restart are 200 and append nothing.
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/operations",
            {
                "operations": [
                    record("r2", operation("sync-new", "k", "v2", {"r2": 1})),
                    record("r1", operation("sync-stale", "k", "old", {"r1": 0})),
                ]
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "accepted": 0, "replayed": 2})
        # A tampered known identity conflicts after restart, changing nothing.
        tampered = {
            "operations": [
                record("r9", operation("zz", "k", "zz", {"r9": 1})),
                record("r2", operation("sync-new", "k", "tampered", {"r2": 1})),
            ]
        }
        status, payload = self.request(server, "POST", "/v1/sync/operations", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in self.drain(server)],
            expected_order,
        )
        # Stale-write candidate semantics survived recovery too.
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class CommandLineSyncTests(unittest.TestCase):
    """The real ``python -m`` entry point serves sync end to end."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"
        self.port = self._free_port()

    @staticmethod
    def _free_port() -> int:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        return port

    def spawn(self) -> subprocess.Popen:
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "semantic_state_engine.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--data-file",
                str(self.data_file),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def wait_for_health(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                conn.request("GET", "/health")
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status == 200:
                    return
            except OSError:
                time.sleep(0.05)
        self.fail("service did not become healthy")

    def stop(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    def test_cli_sync_import_export_and_restart(self) -> None:
        proc = self.spawn()
        try:
            self.wait_for_health()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                "/v1/sync/operations",
                body=json.dumps(
                    {
                        "operations": [
                            record("r1", operation("o1", "color", "blue", {"r1": 1})),
                            record("r2", operation("o2", "color", "green", {"r2": 1})),
                        ]
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.request("GET", "/v1/sync/operations?after=0&limit=1")
            response = conn.getresponse()
            page = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(page["nextCursor"], 1)
            self.assertIs(page["hasMore"], True)
            self.assertEqual(page["operations"][0]["replicaId"], "r1")
            conn.close()
        finally:
            self.stop(proc)

        proc = self.spawn()
        try:
            self.wait_for_health()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", "/v1/sync/operations")
            response = conn.getresponse()
            page = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(
                [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
                [("r1", "o1"), ("r2", "o2")],
            )
            self.assertIs(page["hasMore"], False)
            conn.close()
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
