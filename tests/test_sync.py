"""Tests for incremental replica sync: GET/POST /v1/sync/operations."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def sync_item(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


class SyncHTTPTestCase(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
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

    def post_local(self, replica: str, op: dict) -> int:
        status, _ = self.request("POST", f"/v1/replicas/{replica}/operations", op)
        return status

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def post_sync(self, items: object) -> tuple[int, object]:
        return self.request("POST", "/v1/sync/operations", {"operations": items})

    # --- export ---------------------------------------------------------

    def test_empty_export(self) -> None:
        status, payload = self.get_sync()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_export_follows_local_commit_order_including_stale_writes(self) -> None:
        timeline = [
            ("r1", operation("op-1", "k", "new", {"r1": 2})),
            ("r2", operation("op-2", "k", "v2", {"r2": 1})),
            # A dominated (stale) write is still exported in accept order.
            ("r1", operation("op-3", "k", "stale", {"r1": 1})),
            ("r9", operation("op-4", "k", "v9", {"r9": 3})),
        ]
        for replica, op in timeline:
            self.assertEqual(self.post_local(replica, op), 201)
        status, payload = self.get_sync()
        self.assertEqual(status, 200)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["nextCursor"], 4)
        self.assertEqual(
            payload["operations"],
            [sync_item(replica, op) for replica, op in timeline],
        )

    def test_export_records_have_exactly_two_fields(self) -> None:
        self.post_local("r1", operation("op-1", "k", "v", {"r1": 1}))
        _, payload = self.get_sync()
        (record,) = payload["operations"]
        self.assertEqual(set(record.keys()), {"replicaId", "operation"})
        self.assertEqual(
            set(record["operation"].keys()),
            {"operationId", "key", "value", "clock"},
        )

    def test_pagination_walks_the_whole_log(self) -> None:
        for index in range(5):
            self.post_local(f"r{index}", operation(f"op-{index}", "k", f"v{index}", {f"r{index}": 1}))

        collected: list = []
        cursor = 0
        pages = 0
        while True:
            status, payload = self.get_sync(f"?after={cursor}&limit=2")
            self.assertEqual(status, 200)
            page = payload["operations"]
            self.assertEqual(payload["nextCursor"], cursor + len(page))
            if pages < 2:
                self.assertEqual(len(page), 2)
                self.assertTrue(payload["hasMore"])
            collected.extend(page)
            cursor = payload["nextCursor"]
            pages += 1
            if not payload["hasMore"]:
                break
        self.assertEqual(pages, 3)
        self.assertEqual(cursor, 5)
        self.assertEqual([item["operation"]["operationId"] for item in collected],
                         [f"op-{i}" for i in range(5)])

    def test_after_at_end_returns_empty_final_page(self) -> None:
        self.post_local("r1", operation("op-1", "k", "v", {"r1": 1}))
        status, payload = self.get_sync("?after=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])

    def test_default_limit_is_100(self) -> None:
        for index in range(105):
            self.post_local("r1", operation(f"op-{index}", "k", "v", {"r1": index + 1}))
        status, payload = self.get_sync()
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 100)
        self.assertEqual(payload["nextCursor"], 100)
        self.assertTrue(payload["hasMore"])
        status, payload = self.get_sync("?after=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 5)
        self.assertFalse(payload["hasMore"])

    def test_export_query_errors_are_400(self) -> None:
        for query in [
            "?after=-1",
            "?after=abc",
            "?after=1.5",
            "?after=",
            "?limit=-1",
            "?limit=0",
            "?limit=101",
            "?limit=abc",
            "?foo=1",
            "?after=0&after=1",
            "?after=1&limit=1&extra=x",
        ]:
            status, payload = self.get_sync(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_after_past_end_is_400(self) -> None:
        self.post_local("r1", operation("op-1", "k", "v", {"r1": 1}))
        status, payload = self.get_sync("?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_limit_boundaries_are_accepted(self) -> None:
        for index in range(3):
            self.post_local("r1", operation(f"op-{index}", "k", "v", {"r1": index + 1}))
        status, payload_a = self.get_sync("?limit=1")
        status_b, payload_b = self.get_sync("?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(status_b, 200)
        self.assertEqual(len(payload_a["operations"]), 1)
        self.assertEqual(len(payload_b["operations"]), 3)

    # --- import ---------------------------------------------------------

    def test_import_new_batch_is_201_with_counts(self) -> None:
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
        ]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 2, "replayed": 0})
        _, exported = self.get_sync("?limit=100")
        self.assertEqual(exported["operations"], items)

    def test_unknown_replica_is_accepted(self) -> None:
        items = [sync_item("brand-new-replica", operation("o", "k", "v", {"brand-new-replica": 1}))]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)

    def test_stale_import_is_accepted_and_exported(self) -> None:
        self.post_local("r1", operation("op-1", "k", "new", {"r1": 2}))
        items = [sync_item("r1", operation("op-2", "k", "stale", {"r1": 1}))]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "new")
        _, exported = self.get_sync()
        self.assertEqual(
            [op["operation"]["operationId"] for op in exported["operations"]],
            ["op-1", "op-2"],
        )

    def test_replayed_batch_is_200(self) -> None:
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
        ]
        self.assertEqual(self.post_sync(items)[0], 201)
        status, payload = self.post_sync(items)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "accepted": 0, "replayed": 2})

    def test_mixed_batch_is_201_with_split_counts(self) -> None:
        self.post_local("r1", operation("op-1", "k", "v1", {"r1": 1}))
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
        ]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 1, "replayed": 1})

    def test_duplicate_identity_within_batch_is_replayed_once(self) -> None:
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
            # Same identity as an earlier item in this same batch, identical.
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
        ]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 2, "replayed": 1})
        _, exported = self.get_sync()
        self.assertEqual(
            [r["operation"]["operationId"] for r in exported["operations"]],
            ["op-1", "op-2"],
        )

    def test_duplicate_identity_within_batch_with_different_content_is_409(self) -> None:
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r1", operation("op-1", "k", "tampered", {"r1": 1})),
        ]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, exported = self.get_sync()
        self.assertEqual(exported["operations"], [])

    def test_conflict_is_409_and_whole_batch_is_atomic(self) -> None:
        self.post_local("r1", operation("op-1", "k", "original", {"r1": 1}))
        items = [
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
            sync_item("r3", operation("op-3", "k", "v3", {"r3": 1})),
            # Conflict on a previously accepted identity, late in the batch.
            sync_item("r1", operation("op-1", "k", "tampered", {"r1": 1})),
            sync_item("r4", operation("op-4", "k", "v4", {"r4": 1})),
        ]
        status, payload = self.post_sync(items)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # None of the otherwise-new items committed.
        _, exported = self.get_sync()
        self.assertEqual(
            [op["operation"]["operationId"] for op in exported["operations"]], ["op-1"]
        )
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "original")
        # The identity index is untouched: op-2/op-3 still accept afterwards,
        # and the original op-1 still replays.
        self.assertEqual(
            self.post_sync([sync_item("r2", operation("op-2", "k", "v2", {"r2": 1}))])[0],
            201,
        )
        self.assertEqual(
            self.post_sync([sync_item("r1", operation("op-1", "k", "original", {"r1": 1}))])[0],
            200,
        )

    def test_batch_size_boundaries(self) -> None:
        hundred = [
            sync_item(f"r{i}", operation(f"op-{i}", "k", f"v{i}", {f"r{i}": 1}))
            for i in range(100)
        ]
        self.assertEqual(self.post_sync(hundred)[0], 201)
        self.assertEqual(self.post_sync([])[0], 400)
        too_many = [
            sync_item("r1", operation(f"extra-{i}", "k", "v", {"r1": 1000 + i}))
            for i in range(101)
        ]
        status, payload = self.post_sync(too_many)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_import_body_errors_are_400(self) -> None:
        bad_bodies = [
            b"{not json",
            [],
            {},
            {"operations": []},
            {"operations": {}},
            {"operations": "nope"},
            {"operations": [], "extra": 1},
            [{"replicaId": "r1", "operation": operation("o", "k", "v", {"r1": 1})}],
        ]
        for body in bad_bodies:
            status, payload = self.request("POST", "/v1/sync/operations", body)
            self.assertEqual(status, 400, body)
            self.assertEqual(payload, {"error": "invalid_request"}, body)

    def test_import_item_errors_are_400(self) -> None:
        valid = operation("op-1", "k", "v", {"r1": 1})
        bad_items = [
            [{"replicaId": "r1"}],
            [{"operation": valid}],
            [{"replicaId": "", "operation": valid}],
            [{"replicaId": 42, "operation": valid}],
            [{"replicaId": "r1", "operation": valid, "extra": 1}],
            "not-an-item",
            [sync_item("r1", operation("", "k", "v", {"r1": 1}))],
            [sync_item("r1", operation("o", "k", "v", {}))],
            [sync_item("r1", operation("o", "k", "v", {"r1": -1}))],
            # Clock must contain the item's own replica id.
            [sync_item("r1", operation("o", "k", "v", {"r2": 1}))],
            [sync_item("r1", ["not", "an", "operation"])],
        ]
        for items in bad_items:
            status, payload = self.post_sync(items)
            self.assertEqual(status, 400, items)
            self.assertEqual(payload, {"error": "invalid_request"}, items)

    def test_existing_routes_are_unchanged(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})
        for method, path in [("GET", "/v1/sync/nope"), ("POST", "/v1/sync")]:
            status, payload = self.request(method, path)
            self.assertEqual(status, 404, (method, path))

    # --- concurrency ----------------------------------------------------

    def test_concurrent_commits_never_expose_half_batches(self) -> None:
        batch_count, batch_size = 12, 5
        stop = threading.Event()
        violations: list[str] = []
        read_errors: list[BaseException] = []
        write_errors: list[tuple[str, object]] = []
        seen_pages: list[int] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    _, payload = self.get_sync("?after=0&limit=100")
                    seen_pages.append(len(payload["operations"]))
                    grouped: dict[str, int] = {}
                    for record in payload["operations"]:
                        op_id = record["operation"]["operationId"]
                        if op_id.startswith("b"):
                            batch = op_id.split("-")[0]
                            grouped[batch] = grouped.get(batch, 0) + 1
                    for batch, count in grouped.items():
                        if count not in (0, batch_size):
                            violations.append(f"{batch} partially visible: {count}")
            except BaseException as exc:  # a dropped connection is a failure
                read_errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for thread in readers:
            thread.start()

        def importer(batch_index: int) -> None:
            replica = f"rb{batch_index}"
            items = [
                sync_item(
                    replica,
                    operation(f"b{batch_index}-{j}", "k", f"v{batch_index}-{j}", {replica: j + 1}),
                )
                for j in range(batch_size)
            ]
            status, payload = self.post_sync(items)
            if status != 201 or payload["accepted"] != batch_size:
                write_errors.append((f"batch-{batch_index}", (status, payload)))

        def local_writer(index: int) -> None:
            status = self.post_local(
                f"rl{index}", operation(f"local-{index}", "k", f"vl{index}", {f"rl{index}": 1})
            )
            if status != 201:
                write_errors.append((f"local-{index}", status))

        importers = [threading.Thread(target=importer, args=(i,)) for i in range(batch_count)]
        writers = [threading.Thread(target=local_writer, args=(i,)) for i in range(8)]
        for thread in importers + writers:
            thread.start()
        for thread in importers + writers:
            thread.join(timeout=15)
        stop.set()
        for thread in readers:
            thread.join(timeout=5)

        self.assertEqual(read_errors, [])
        self.assertEqual(write_errors, [])
        self.assertEqual(violations, [])
        self.assertTrue(seen_pages)
        _, payload = self.get_sync("?limit=100")
        self.assertEqual(
            len(payload["operations"]), batch_count * batch_size + 8
        )
        # Every batch survived as a contiguous, complete unit.
        op_ids = [r["operation"]["operationId"] for r in payload["operations"]]
        for batch_index in range(batch_count):
            expected = [f"b{batch_index}-{j}" for j in range(batch_size)]
            start = op_ids.index(expected[0])
            self.assertEqual(op_ids[start : start + batch_size], expected)
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class PersistentSyncTestCase(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        if body is None:
            conn.request(method, path)
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

    def sync_get(self, server: SemanticStateServer, query: str = ""):
        return self.request(server, "GET", f"/v1/sync/operations{query}")

    def sync_post(self, server: SemanticStateServer, items: list):
        return self.request(server, "POST", "/v1/sync/operations", {"operations": items})

    def test_batch_persists_as_one_unit_and_survives_restart(self) -> None:
        server = self.start_server()
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/replicas/r0/operations",
                operation("local-0", "k", "v0", {"r0": 1}),
            )[0],
            201,
        )
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r0": 1, "r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
            sync_item("r1", operation("op-3", "k", "stale", {"r1": 0, "r0": 1})),
        ]
        status, payload = self.sync_post(server, items)
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 3, "replayed": 0})

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, page = self.sync_get(server)
        self.assertEqual(status, 200)
        self.assertEqual(
            page["operations"],
            [sync_item("r0", operation("local-0", "k", "v0", {"r0": 1})), *items],
        )
        # Resume (续传) from a cursor taken before restart.
        status, rest = self.sync_get(server, "?after=3&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual([r["operation"]["operationId"] for r in rest["operations"]], ["op-3"])
        self.assertEqual(rest["nextCursor"], 4)
        self.assertFalse(rest["hasMore"])
        # Replay and conflict are unchanged after restart.
        status, payload = self.sync_post(server, items)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "accepted": 0, "replayed": 3})
        tampered = [dict(items[0])]
        tampered[0] = sync_item("r1", operation("op-1", "k", "tampered", {"r0": 1, "r1": 1}))
        status, payload = self.sync_post(server, tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(len(self.sync_get(server)[1]["operations"]), 4)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        port = server.server_address[1]
        before = self.data_file.read_bytes()
        items = [
            sync_item("r1", operation("op-1", "k", "v1", {"r1": 1})),
            sync_item("r2", operation("op-2", "k", "v2", {"r2": 1})),
        ]
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST",
                "/v1/sync/operations",
                body=json.dumps({"operations": items}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 500)
            self.assertEqual(
                json.loads(response.read().decode()), {"error": "internal_error"}
            )
            conn.close()

        # Memory state, identity index and the file are all unchanged.
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 404)
        status, exported = self.sync_get(server)
        self.assertEqual(exported["operations"], [])
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name.startswith(".sestate-")]
        self.assertEqual(leftovers, [])
        # Retrying after recovery accepts the full batch (nothing was indexed).
        status, payload = self.sync_post(server, items)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        reloaded = StateStore(data_file=str(self.data_file))
        status, _ = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.OK)

    def test_concurrent_local_and_sync_commits_share_one_log(self) -> None:
        store = StateStore(data_file=str(self.data_file))
        errors: list[BaseException] = []

        def sync_worker(index: int) -> None:
            try:
                replica = f"rs{index}"
                # Same key and replica: later clocks dominate earlier ones, so
                # only the last item survives as a candidate -- but all three
                # must remain in the accepted log, contiguous in import order.
                items = [
                    (
                        replica,
                        operation(f"s{index}-{j}", "k", f"{index}-{j}", {replica: j + 1}),
                    )
                    for j in range(3)
                ]
                accepted, replayed = store.import_operations(items)
                assert (accepted, replayed) == (3, 0)
            except BaseException as exc:  # reported below
                errors.append(exc)

        def local_worker(index: int) -> None:
            try:
                status = store.apply_operation(
                    f"rl{index}",
                    operation(f"l{index}", "k", f"l{index}", {f"rl{index}": 1}),
                )
                assert status is HTTPStatus.CREATED
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=sync_worker, args=(i,)) for i in range(8)]
        threads += [threading.Thread(target=local_worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])

        # The durable log contains exactly every committed record; each batch
        # is contiguous in its own import order, and recovery reproduces state.
        records = self.sync_get_records(store)
        self.assertEqual(len(records), 8 * 3 + 8)
        for index in range(8):
            op_ids = [
                r["operation"]["operationId"]
                for r in records
                if r["replicaId"] == f"rs{index}"
            ]
            self.assertEqual(op_ids, [f"s{index}-{j}" for j in range(3)])
        reloaded = StateStore(data_file=str(self.data_file))
        status, state = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "conflict")
        # One surviving candidate per replica (its highest tick for sync
        # replicas), all mutually concurrent: 8 sync + 8 local replicas.
        self.assertEqual(len(state["candidates"]), 16)
        self.assertEqual(
            self.sync_get_records(reloaded),
            records,
        )

    @staticmethod
    def sync_get_records(store: StateStore) -> list:
        records: list = []
        cursor = 0
        while True:
            page, cursor, has_more = store.get_operations_page(cursor, 10)
            records.extend(page)
            if not has_more:
                return records


if __name__ == "__main__":
    unittest.main()
