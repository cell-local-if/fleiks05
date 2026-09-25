"""Tests for sender-progress pickup of accepted operations.

The pickup endpoint is::

    GET /v1/sync/peers/{peerId}/operations?after=N&limit=N

It returns the accepted operations a consuming replica has not yet
consumed: the tail of the shared accepted-operation log beginning right
after the peer's registered checkpoint cursor, in global commit order.
The endpoint is strictly read-only: it never advances or writes the
checkpoint and never touches the candidates, audit streams, metrics,
summaries, or the data file.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread), the ``StateStore``
directly, or the real ``python -m`` entry point; only the Python
standard library is used.
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

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file_full,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN = "pickup-token_42"
AUTH_HEADER = f"Bearer {TOKEN}"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


class PeerPickupStoreTests(unittest.TestCase):
    """Store-level pickup semantics, in memory."""

    def test_unregistered_peer_is_not_found(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_peer_operations("nobody", 0, 100)
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_pickup_is_the_log_tail_after_the_checkpoint(self) -> None:
        store = StateStore()
        ops = [
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
            ("r3", operation("o3", "k", "v3", {"r3": 1})),
            ("r1", operation("o4", "k", "v4", {"r1": 2, "r2": 1, "r3": 1})),
        ]
        for replica, op in ops:
            self.assertIs(store.apply_operation(replica, op), HTTPStatus.CREATED)
        store.save_checkpoint("peer-a", 1)
        status, payload = store.get_peer_operations("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r2", "o2"), ("r3", "o3"), ("r1", "o4")],
        )
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)
        # The items are exactly the sync-export tail at the checkpoint.
        tail, _, _ = store.get_sync_operations(1, 100)
        self.assertEqual(payload["operations"], tail)

    def test_pickup_includes_records_from_other_replicas(self) -> None:
        # The peer only anchors the progress cursor: every accepted
        # record past it is returned with its own replica identity.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.save_checkpoint("peer-a", 0)
        store.apply_operation("r2", operation("o2", "k", "v2", {"r1": 1, "r2": 1}))
        store.apply_operation("r9", operation("o3", "k", "v3", {"r1": 1, "r2": 1, "r9": 1}))
        status, payload = store.get_peer_operations("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r1", "o1"), ("r2", "o2"), ("r9", "o3")],
        )

    def test_stale_writes_imports_and_repairs_are_visible(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        # An ordinary write and a stale write that adds no candidate.
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("stale", "k", "old", {"r1": 0}))
        # A sync-imported record.
        store.import_operations(
            [("r2", operation("o2", "k2", "v2", {"r2": 1}))]
        )
        status, payload = store.get_peer_operations("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r1", "o1"), ("r1", "stale"), ("r2", "o2")],
        )

    def test_manual_and_automatic_repairs_are_visible(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        store.apply_operation("r1", operation("o1", "k", "red", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "blue", {"r2": 1}))
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
        status, error = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        # A fresh conflict and an automatic repair.
        store.apply_operation("r1", operation("o3", "k2", "red", {"r1": 2}))
        store.apply_operation("r2", operation("o4", "k2", "blue", {"r2": 2}))
        status, chosen, error = store.apply_auto_resolution(
            "k2",
            {
                "replicaId": "r3",
                "operationId": "auto-1",
                "clock": {"r1": 2, "r2": 2, "r3": 2},
                "policy": "lowest_identity",
            },
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNotNone(chosen)
        self.assertIsNone(error)
        _, payload = store.get_peer_operations("peer-a", 0, 100)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1"),
             ("r1", "o3"), ("r2", "o4"), ("r3", "auto-1")],
        )

    def test_replays_rejections_and_uncommitted_requests_never_appear(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        op = operation("o1", "k", "v1", {"r1": 1})
        store.apply_operation("r1", op)
        # Identical replay adds no record.
        self.assertIs(store.apply_operation("r1", op), HTTPStatus.OK)
        # A conflicting identity changes nothing.
        conflict = operation("o1", "k", "other", {"r1": 2})
        self.assertIs(store.apply_operation("r1", conflict), HTTPStatus.CONFLICT)
        # A rejected import batch (409) adds nothing.
        status, _, _ = store.import_operations(
            [
                ("r2", operation("o9", "k", "z", {"r2": 1})),
                ("r1", operation("o1", "k", "tampered", {"r1": 3})),
            ]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        _, payload = store.get_peer_operations("peer-a", 0, 100)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r1", "o1")],
        )

    def test_paging_and_empty_tail(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        for i in range(5):
            store.apply_operation(
                f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            )
        # Pages use an after/limit cursor relative to the checkpoint.
        status, first = store.get_peer_operations("peer-a", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([e["operation"]["operationId"] for e in first["operations"]], ["o0", "o1"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        status, second = store.get_peer_operations("peer-a", first["nextCursor"], 2)
        self.assertEqual([e["operation"]["operationId"] for e in second["operations"]], ["o2", "o3"])
        self.assertEqual(second["nextCursor"], 4)
        self.assertIs(second["hasMore"], True)
        status, third = store.get_peer_operations("peer-a", second["nextCursor"], 2)
        self.assertEqual([e["operation"]["operationId"] for e in third["operations"]], ["o4"])
        self.assertEqual(third["nextCursor"], 5)
        self.assertIs(third["hasMore"], False)
        # after equal to the unconsumed count is a valid empty tail.
        status, tail = store.get_peer_operations("peer-a", 5, 2)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)
        # after past the count is rejected.
        with self.assertRaises(ValueError):
            store.get_peer_operations("peer-a", 6, 2)

    def test_cursor_is_relative_to_the_checkpoint(self) -> None:
        store = StateStore()
        for i in range(4):
            store.apply_operation(
                f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            )
        store.save_checkpoint("peer-a", 2)
        # after=1 skips one record *past the checkpoint*, not the first
        # record of the log.
        _, payload = store.get_peer_operations("peer-a", 1, 100)
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["operations"]],
            ["o3"],
        )
        self.assertEqual(payload["nextCursor"], 2)

    def test_pickup_is_read_only(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        metrics_before = store.get_metrics()
        checkpoint_before = store.get_checkpoint("peer-a")
        sync_before, _, _ = store.get_sync_operations(0, 100)
        # A pickup must never attempt a durable write, even when one is
        # configured (store here is in memory; the patch documents intent).
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("pickup must not persist"),
        ):
            for after in (0, 1):
                status, _ = store.get_peer_operations("peer-a", after, 1)
                self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("peer-a"), checkpoint_before)
        self.assertEqual(store.get_metrics(), metrics_before)
        sync_after, _, _ = store.get_sync_operations(0, 100)
        self.assertEqual(sync_after, sync_before)

    def test_peers_are_independent(self) -> None:
        store = StateStore()
        for i in range(3):
            store.apply_operation(
                f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            )
        store.save_checkpoint("peer-a", 1)
        store.save_checkpoint("peer-b", 3)
        _, a = store.get_peer_operations("peer-a", 0, 100)
        _, b = store.get_peer_operations("peer-b", 0, 100)
        self.assertEqual([e["operation"]["operationId"] for e in a["operations"]], ["o1", "o2"])
        self.assertEqual(b["operations"], [])
        self.assertEqual(b["nextCursor"], 0)
        self.assertIs(b["hasMore"], False)


class PersistentPeerPickupStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def test_recovery_reproduces_pickup_pages(self) -> None:
        store = self.make_store()
        for i in range(4):
            store.apply_operation(
                f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            )
        store.save_checkpoint("peer-a", 1)
        del store

        reloaded = self.make_store()
        _, payload = reloaded.get_peer_operations("peer-a", 0, 1)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["o1"])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], True)
        _, payload = reloaded.get_peer_operations("peer-a", 1, 1)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["o2"])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], True)
        _, payload = reloaded.get_peer_operations("peer-a", 2, 10)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["o3"])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)

    def test_recovered_cursor_past_log_length_rejects_startup(self) -> None:
        self.data_file.write_text(
            '{"version":1,"operations":[],'
            '"checkpoints":{"peer-a":1}}',
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()

    def test_pickup_creates_no_temp_file(self) -> None:
        store = self.make_store()
        store.save_checkpoint("peer-a", 0)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("pickup must not persist"),
        ):
            self.assertIs(store.get_peer_operations("peer-a", 0, 100)[0], HTTPStatus.OK)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        _, checkpoints = load_data_file_full(str(self.data_file))[:2]
        self.assertEqual(checkpoints, {"peer-a": 0})


class HttpPeerPickupTests(unittest.TestCase):
    """HTTP contract against an in-memory server."""

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

    def request_raw(
        self, method: str, path: str, body: object = None, headers: dict | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        merged = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=merged)
        else:
            merged.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers=merged,
            )
        response = conn.getresponse()
        raw = response.read()
        sent_headers = {k.lower(): v for k, v in response.getheaders()}
        conn.close()
        return response.status, sent_headers, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        status, _, raw = self.request_raw(method, path, body)
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_operation(self, replica: str, op: dict) -> None:
        status, _ = self.request("POST", f"/v1/replicas/{replica}/operations", op)
        self.assertEqual(status, 201)

    def register(self, peer: str, cursor: int) -> None:
        status, payload = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        self.assertEqual(status, 200, payload)

    def pickup(self, peer: str, query: str = "?after=0&limit=100") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/operations{query}")

    def seed(self, *pairs: tuple[str, str]) -> None:
        for replica, op_id in pairs:
            self.post_operation(replica, operation(op_id, "k", op_id, {replica: 1}))

    def test_unregistered_peer_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.pickup("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_success_body_is_compact_json_with_one_newline(self) -> None:
        self.register("peer-a", 0)
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertEqual(
            raw,
            b'{"hasMore":false,"nextCursor":1,'
            b'"operations":[{"operation":{"clock":{"r1":1},"key":"k",'
            b'"operationId":"o1","value":"v"},"replicaId":"r1"}]}\n',
        )
        # Exactly the three contracted fields.
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(set(payload), {"operations", "nextCursor", "hasMore"})

    def test_empty_unconsumed_tail(self) -> None:
        self.register("peer-a", 0)
        status, payload = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_pickup_follows_checkpoint_and_pages(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"), ("r3", "o3"), ("r1", "o4"))
        self.register("peer-a", 1)
        status, first = self.pickup("peer-a", "?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in first["operations"]],
            [("r2", "o2"), ("r3", "o3")],
        )
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        status, second = self.pickup("peer-a", f"?after={first['nextCursor']}&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in second["operations"]],
            [("r1", "o4")],
        )
        self.assertEqual(second["nextCursor"], 3)
        self.assertIs(second["hasMore"], False)
        # Cursor resume lands exactly on the empty tail.
        status, tail = self.pickup("peer-a", "?after=3&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)

    def test_records_keep_committed_replica_identities(self) -> None:
        self.register("peer-a", 0)
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k2", "v2", {"r2": 1}))
        status, payload = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r1", "o1"), ("r2", "o2")],
        )

    def test_pickup_does_not_advance_or_write_checkpoint(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"))
        self.register("peer-a", 1)
        # Missing-parameter queries are rejected before any state read.
        for query in ("", "?after=0", "?limit=1"):
            status, _ = self.pickup("peer-a", query)
            self.assertEqual(status, 400, query)
        status, _ = self.pickup("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        # Repeated pickups return the same first page.
        first_a = self.pickup("peer-a", "?after=0&limit=1")[1]
        first_b = self.pickup("peer-a", "?after=0&limit=1")[1]
        self.assertEqual(first_a, first_b)

    def test_stale_write_is_picked_up(self) -> None:
        self.register("peer-a", 0)
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r1", operation("old", "k", "stale", {"r1": 0}))
        status, payload = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["operations"]],
            ["o1", "old"],
        )

    def test_sync_imports_are_picked_up(self) -> None:
        self.register("peer-a", 0)
        body = {
            "operations": [
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
                record("r3", operation("o3", "k", "v3", {"r3": 1})),
            ]
        }
        status, _ = self.request("POST", "/v1/sync/operations", body)
        self.assertEqual(status, 201)
        status, payload = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["operations"]],
            [("r2", "o2"), ("r3", "o3")],
        )

    def test_invalid_queries_are_400_and_change_nothing(self) -> None:
        self.seed(("r1", "o1"))
        self.register("peer-a", 0)
        bad_queries = [
            # Missing parameters (the bare request included).
            "",
            "?after=0",
            "?limit=10",
            # Blank, malformed, negative, or non-ASCII values.
            "?after=&limit=10",
            "?after&limit=10",
            "?after=0&limit=",
            "?after=0&limit",
            "?after=-1&limit=10",
            "?after=1.0&limit=10",
            "?after=0x1&limit=10",
            "?after=%2B1&limit=10",
            "?after=1%20&limit=10",
            "?after=%D9%A1&limit=10",  # non-ASCII Arabic-Indic digit one
            # Out-of-range limit.
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=-1",
            "?after=0&limit=one",
            # Repeated names and unknown parameters.
            "?after=1&after=0&limit=10",
            "?after=0&limit=1&limit=2",
            "?foo=1&after=0&limit=10",
            "?after=0&limit=10&foo=1",
            "?=1&after=0&limit=10",
            # after past the unconsumed count (1 record past cursor 0).
            "?after=2&limit=10",
        ]
        for query in bad_queries:
            status, payload = self.pickup("peer-a", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # The checkpoint and the log are untouched.
        self.assertEqual(
            self.request("GET", "/v1/sync/peers/peer-a/checkpoint")[1],
            {"peerId": "peer-a", "cursor": 0},
        )
        status, payload = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in payload["operations"]], ["o1"])

    def test_after_equal_to_count_is_an_empty_page(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"))
        self.register("peer-a", 0)
        status, payload = self.pickup("peer-a", "?after=2&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 2, "hasMore": False})

    def test_route_shapes_are_404_before_query_checks(self) -> None:
        self.register("peer-a", 0)
        not_found_paths = [
            "/v1/sync/peers//operations?after=0",
            "/v1/sync/peers//operations?after=x",
            "/v1/sync/peers/peer-a/operations/extra",
            "/v1/sync/peers/peer-a/operations/extra?after=0",
            "/v1/sync/peers/peer-a/operations/",
            "/v1/sync/peers",
            "/v1/sync/peers/peer-a",
            "/v1/sync/peers/peer-a/not-operations",
            "/v1/sync/operations/peer-a",
            "/v1/sync/peer/peer-a/operations",
            "/v2/sync/peers/peer-a/operations",
        ]
        for path in not_found_paths:
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_operations_route_does_not_shadow_checkpoint_route(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"))
        self.register("peer-a", 2)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 2})

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(("r1", "o1"))
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer%20one/operations?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["operations"]], ["o1"]
        )

    def test_advancing_checkpoint_moves_the_pickup_anchor(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"), ("r3", "o3"))
        self.register("peer-a", 1)
        status, first = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in first["operations"]], ["o2", "o3"])
        # The pickup never moves the anchor itself; only a checkpoint
        # advance does, after which after is again relative to the new
        # checkpoint and the previously consumed records are gone.
        self.register("peer-a", 3)
        status, advanced = self.pickup("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(advanced["operations"], [])
        self.assertEqual((advanced["nextCursor"], advanced["hasMore"]), (0, False))
        self.seed(("r4", "o4"))
        status, grown = self.pickup("peer-a")
        self.assertEqual([e["operation"]["operationId"] for e in grown["operations"]], ["o4"])

    def test_unknown_peer_with_well_formed_query_is_404(self) -> None:
        # Query validation runs first: a missing/malformed parameter is a
        # 400 even for an unregistered peer; a fully well-formed query
        # reaches the existence check and stays a not-found (the after
        # bound cannot be evaluated without a registered anchor).
        for query in ("", "?after=0", "?limit=1", "?after=999"):
            status, payload = self.pickup("nope", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        for query in ("?after=0&limit=1", "?after=999&limit=10"):
            status, payload = self.pickup("nope", query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_empty_peer_beats_query_validation(self) -> None:
        # Path shape takes precedence: an empty peer segment with a
        # malformed query is still 404, not 400.
        for query in ("?limit=0", "?after=x", "?unknown=1", "?after=-1"):
            status, payload = self.request("GET", f"/v1/sync/peers//operations{query}")
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_head_and_other_methods_are_404_or_405(self) -> None:
        # POST to the operations route is not a published write route.
        status, _ = self.request("POST", "/v1/sync/peers/peer-a/operations", {"cursor": 0})
        self.assertEqual(status, 404)


class AuthenticatedPeerPickupTests(unittest.TestCase):
    """Bearer authentication on the pickup route; health stays anonymous."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token=TOKEN
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def raw_request(self, path: str, auth_headers: list[tuple[str, str]] | None) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", path)
        for name, value in auth_headers or []:
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read()
        headers = {k.lower(): v for k, v in response.getheaders()}
        conn.close()
        return response.status, headers, raw

    def test_health_stays_anonymous(self) -> None:
        status, _, raw = self.raw_request("/health", None)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {"service": "semantic-state-engine", "status": "ok"})

    def test_missing_header_is_401(self) -> None:
        status, headers, raw = self.raw_request("/v1/sync/peers/peer-a/operations", None)
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_wrong_and_malformed_tokens_are_401(self) -> None:
        for value in ("Bearer wrong", "Bearer", "Basic " + TOKEN, "Token " + TOKEN,
                      f"Bearer  {TOKEN}", f"bearer {TOKEN}"):
            status, headers, _ = self.raw_request(
                "/v1/sync/peers/peer-a/operations", [("Authorization", value)]
            )
            self.assertEqual(status, 401, value)
            self.assertEqual(headers.get("www-authenticate"), "Bearer", value)

    def test_duplicate_headers_are_401_even_when_both_match(self) -> None:
        status, _, _ = self.raw_request(
            "/v1/sync/peers/peer-a/operations",
            [("Authorization", AUTH_HEADER), ("Authorization", AUTH_HEADER)],
        )
        self.assertEqual(status, 401)

    def test_authenticated_pickup_works(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/sync/peers/peer-a/checkpoint",
            body=json.dumps({"cursor": 0}),
            headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        response.read()
        conn.request(
            "GET",
            "/v1/sync/peers/peer-a/operations?after=0&limit=100",
            headers={"Authorization": AUTH_HEADER},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            json.loads(response.read().decode("utf-8")),
            {"operations": [], "nextCursor": 0, "hasMore": False},
        )
        conn.close()


class PersistentPeerPickupHttpTests(unittest.TestCase):
    """Pickup over real HTTP with a data file: recovery and no temp files."""

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
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")) if raw else None

    def test_pickup_survives_restart_with_same_pages(self) -> None:
        server = self.start_server()
        for i in range(4):
            self.request(
                server,
                "POST",
                f"/v1/replicas/r{i}/operations",
                operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}),
            )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 2})
        status, before = self.request(
            server, "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in before["operations"]], ["o2"])
        self.assertEqual((before["nextCursor"], before["hasMore"]), (1, True))
        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Resume with the returned cursor after restart.
        status, after = self.request(
            server, "GET", "/v1/sync/peers/peer-a/operations?after=1&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([e["operation"]["operationId"] for e in after["operations"]], ["o3"])
        self.assertEqual((after["nextCursor"], after["hasMore"]), (2, False))
        # Full tail matches.
        status, full = self.request(
            server, "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [e["operation"]["operationId"] for e in full["operations"]], ["o2", "o3"]
        )
        # Error boundary survives recovery as well.
        status, payload = self.request(
            server, "GET", "/v1/sync/peers/peer-a/operations?after=3&limit=10"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_queries_create_no_temp_files(self) -> None:
        server = self.start_server()
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 0})
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before = self.data_file.read_bytes()
        for query in ("?after=0&limit=1", "?after=1&limit=1", "?after=0&limit=100"):
            status, _ = self.request(
                server, "GET", f"/v1/sync/peers/peer-a/operations{query}"
            )
            self.assertEqual(status, 200)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


class CommandLinePeerPickupTests(unittest.TestCase):
    """The real ``python -m`` entry point: a stale cursor refuses startup."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()

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

    def test_pickup_served_end_to_end(self) -> None:
        proc = self.spawn()
        try:
            self.wait_for_health()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                "/v1/sync/operations",
                body=json.dumps(
                    {"operations": [record("r1", operation("o1", "k", "v", {"r1": 1}))]}
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.request(
                "POST",
                "/v1/sync/peers/peer-a/checkpoint",
                body=json.dumps({"cursor": 0}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            conn.request("GET", "/v1/sync/peers/peer-a/operations?after=0&limit=1")
            response = conn.getresponse()
            raw = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(raw, b'{"hasMore":false,"nextCursor":1,"operations":['
                                 b'{"operation":{"clock":{"r1":1},"key":"k",'
                                 b'"operationId":"o1","value":"v"},"replicaId":"r1"}]}\n')
            conn.close()
        finally:
            self.stop(proc)

    def test_recovered_cursor_past_log_length_exits_2(self) -> None:
        # One durable operation, then tamper the checkpoint past the log.
        proc = self.spawn()
        try:
            self.wait_for_health()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                "/v1/sync/operations",
                body=json.dumps(
                    {"operations": [record("r1", operation("o1", "k", "v", {"r1": 1}))]}
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.close()
        finally:
            self.stop(proc)

        document = json.loads(self.data_file.read_text(encoding="utf-8"))
        document["checkpoints"] = {"peer-a": 5}
        self.data_file.write_text(json.dumps(document), encoding="utf-8")

        proc = self.spawn()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        finally:
            self.stop(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"startup failed", stderr)
        self.assertEqual(stdout, b"")


if __name__ == "__main__":
    unittest.main()
