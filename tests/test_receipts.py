"""Tests for the per-peer consumption-receipts query.

The receipts endpoint is::

    GET /v1/sync/peers/{peerId}/receipts?after=N&limit=N

It returns the consumption receipts a sending peer has committed through
``POST /v1/sync/peers/{peerId}/acknowledge``, in commit (creation) order,
together with an integrity summary (``algorithm``/``digest``/
``receiptsCount``) that covers the peer's whole committed receipt set.
The endpoint is strictly read-only: it never advances or writes the
checkpoint, records no receipt, and never touches the candidates, audit
streams, metrics, summaries, or the data file.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread), the ``StateStore``
directly, or the real ``python -m`` entry point; only the Python
standard library is used.
"""

from __future__ import annotations

import hashlib
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
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file_acks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN = "receipts-token_42"
AUTH_HEADER = f"Bearer {TOKEN}"

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PeerReceiptsStoreTests(unittest.TestCase):
    """Store-level receipts semantics, in memory."""

    def seed_operations(self, store: StateStore, count: int) -> None:
        for i in range(count):
            replica = f"r{i}"
            self.assertIs(
                store.apply_operation(
                    replica, operation(f"o{i}", "k", f"v{i}", {replica: 1})
                ),
                HTTPStatus.CREATED,
            )

    def test_unregistered_peer_is_not_found(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_peer_receipts("nobody", 0, 100)
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts_hashes_the_empty_array(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        status, payload = store.get_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["receiptsCount"], 0)

    def test_receipts_keep_commit_order_and_confirmation_cursor(self) -> None:
        store = StateStore()
        self.seed_operations(store, 3)
        store.save_checkpoint("peer-a", 0)
        status, error = store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, _ = store.acknowledge_operations(
            "peer-a", "ack-2", 3, [identity("r2", "o2")]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, payload = store.get_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["receipts"],
            [
                {
                    "peerId": "peer-a",
                    "ackId": "ack-1",
                    "cursor": 2,
                    "operations": [identity("r0", "o0"), identity("r1", "o1")],
                },
                {
                    "peerId": "peer-a",
                    "ackId": "ack-2",
                    "cursor": 3,
                    "operations": [identity("r2", "o2")],
                },
            ],
        )
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["receiptsCount"], 2)
        # The digest covers the whole committed set in creation order,
        # with the fixed field order peerId, ackId, cursor, operations.
        expected = digest_of(
            '[{"peerId":"peer-a","ackId":"ack-1","cursor":2,"operations":['
            '{"replicaId":"r0","operationId":"o0"},'
            '{"replicaId":"r1","operationId":"o1"}]},'
            '{"peerId":"peer-a","ackId":"ack-2","cursor":3,"operations":['
            '{"replicaId":"r2","operationId":"o2"}]}]'
        )
        self.assertEqual(payload["digest"], expected)

    def test_empty_segment_receipt_is_listed(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        status, _ = store.acknowledge_operations("peer-a", "ack-0", 0, [])
        self.assertIs(status, HTTPStatus.CREATED)
        status, payload = store.get_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["receipts"],
            [
                {
                    "peerId": "peer-a",
                    "ackId": "ack-0",
                    "cursor": 0,
                    "operations": [],
                }
            ],
        )
        self.assertEqual(payload["receiptsCount"], 1)
        expected = digest_of(
            '[{"peerId":"peer-a","ackId":"ack-0","cursor":0,"operations":[]}]'
        )
        self.assertEqual(payload["digest"], expected)

    def test_replayed_acknowledge_adds_no_receipt(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.save_checkpoint("peer-a", 0)
        operations = [identity("r0", "o0")]
        store.acknowledge_operations("peer-a", "ack-1", 1, operations)
        status, _ = store.acknowledge_operations("peer-a", "ack-1", 1, operations)
        self.assertIs(status, HTTPStatus.OK)
        _, payload = store.get_peer_receipts("peer-a", 0, 100)
        self.assertEqual(payload["receiptsCount"], 1)
        self.assertEqual(len(payload["receipts"]), 1)

    def test_paging_and_empty_tail(self) -> None:
        store = StateStore()
        self.seed_operations(store, 5)
        store.save_checkpoint("peer-a", 0)
        for i in range(5):
            status, _ = store.acknowledge_operations(
                "peer-a", f"ack-{i}", i + 1, [identity(f"r{i}", f"o{i}")]
            )
            self.assertIs(status, HTTPStatus.CREATED)
        status, first = store.get_peer_receipts("peer-a", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([r["ackId"] for r in first["receipts"]], ["ack-0", "ack-1"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        # The summary covers the whole set, not the page.
        self.assertEqual(first["receiptsCount"], 5)
        status, second = store.get_peer_receipts("peer-a", first["nextCursor"], 2)
        self.assertEqual([r["ackId"] for r in second["receipts"]], ["ack-2", "ack-3"])
        self.assertEqual(second["nextCursor"], 4)
        self.assertIs(second["hasMore"], True)
        status, third = store.get_peer_receipts("peer-a", second["nextCursor"], 2)
        self.assertEqual([r["ackId"] for r in third["receipts"]], ["ack-4"])
        self.assertEqual(third["nextCursor"], 5)
        self.assertIs(third["hasMore"], False)
        # The digest is identical on every page: it never follows paging.
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(second["digest"], third["digest"])
        # after equal to the receipt count is a valid empty tail.
        status, tail = store.get_peer_receipts("peer-a", 5, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["receiptsCount"], 5)
        self.assertEqual(tail["digest"], first["digest"])
        # after past the count is rejected.
        with self.assertRaises(ValueError):
            store.get_peer_receipts("peer-a", 6, 2)

    def test_peers_are_independent(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        store.save_checkpoint("peer-b", 0)
        store.acknowledge_operations("peer-a", "ack-1", 1, [identity("r0", "o0")])
        store.acknowledge_operations(
            "peer-b", "ack-9", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        _, a = store.get_peer_receipts("peer-a", 0, 100)
        _, b = store.get_peer_receipts("peer-b", 0, 100)
        self.assertEqual([r["ackId"] for r in a["receipts"]], ["ack-1"])
        self.assertEqual(a["receiptsCount"], 1)
        self.assertEqual([r["ackId"] for r in b["receipts"]], ["ack-9"])
        self.assertEqual(b["receiptsCount"], 1)
        self.assertNotEqual(a["digest"], b["digest"])
        # Each digest names its own peer in the covered receipts.
        self.assertEqual(
            a["digest"],
            digest_of(
                '[{"peerId":"peer-a","ackId":"ack-1","cursor":1,"operations":['
                '{"replicaId":"r0","operationId":"o0"}]}]'
            ),
        )

    def test_digest_escapes_only_quotes_backslashes_and_controls(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        peer = 'peer"\\\nü'  # quote, backslash, U+000A, and a literal non-ASCII point
        store.save_checkpoint(peer, 0)
        store.acknowledge_operations(peer, 'ack"\t', 1, [identity("r0", "o0")])
        _, payload = store.get_peer_receipts(peer, 0, 100)
        # " and \ are escaped, U+000A/U+0009 become lowercase \u00xx, and
        # every other code point (ü) is written literally as UTF-8.
        expected = digest_of(
            '[{"peerId":"peer\\"\\\\\\u000aü","ackId":"ack\\"\\u0009",'
            '"cursor":1,"operations":[{"replicaId":"r0","operationId":"o0"}]}]'
        )
        self.assertEqual(payload["digest"], expected)

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations("peer-a", "ack-1", 1, [identity("r0", "o0")])
        metrics_before = store.get_metrics()
        checkpoint_before = store.get_checkpoint("peer-a")
        sync_before, _, _ = store.get_sync_operations(0, 100)
        # A receipts query must never attempt a durable write, even when
        # one is configured (store here is in memory; the patch documents
        # intent).
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("receipts query must not persist"),
        ):
            for after in (0, 1):
                status, _ = store.get_peer_receipts("peer-a", after, 1)
                self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("peer-a"), checkpoint_before)
        self.assertEqual(store.get_metrics(), metrics_before)
        sync_after, _, _ = store.get_sync_operations(0, 100)
        self.assertEqual(sync_after, sync_before)


class PersistentPeerReceiptsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def commit_receipts(self, store: StateStore) -> str:
        for i in range(3):
            replica = f"r{i}"
            store.apply_operation(
                replica, operation(f"o{i}", "k", f"v{i}", {replica: 1})
            )
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        store.acknowledge_operations("peer-a", "ack-2", 3, [identity("r2", "o2")])
        _, payload = store.get_peer_receipts("peer-a", 0, 100)
        return payload["digest"]

    def test_recovery_reproduces_order_pages_and_digest(self) -> None:
        store = self.make_store()
        digest_before = self.commit_receipts(store)
        del store

        reloaded = self.make_store()
        status, payload = reloaded.get_peer_receipts("peer-a", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([r["ackId"] for r in payload["receipts"]], ["ack-1"])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], True)
        self.assertEqual(payload["receiptsCount"], 2)
        self.assertEqual(payload["digest"], digest_before)
        status, payload = reloaded.get_peer_receipts("peer-a", 1, 10)
        self.assertEqual([r["ackId"] for r in payload["receipts"]], ["ack-2"])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["digest"], digest_before)
        # The full page preserves commit order and confirmation cursors.
        _, full = reloaded.get_peer_receipts("peer-a", 0, 100)
        self.assertEqual(
            [(r["ackId"], r["cursor"]) for r in full["receipts"]],
            [("ack-1", 2), ("ack-2", 3)],
        )

    def test_file_without_acks_recovers_with_an_empty_set(self) -> None:
        self.data_file.write_text(
            '{"version":1,"operations":[],' '"checkpoints":{"peer-a":0}}',
            encoding="utf-8",
        )
        store = self.make_store()
        status, payload = store.get_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(load_data_file_acks(str(self.data_file)), {})

    def test_query_creates_no_temp_file(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("receipts query must not persist"),
        ):
            self.assertIs(store.get_peer_receipts("peer-a", 0, 100)[0], HTTPStatus.OK)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


class HttpPeerReceiptsTests(unittest.TestCase):
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

    def acknowledge(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, payload = self.request(
            "POST",
            f"/v1/sync/peers/{peer}/acknowledge",
            {"ackId": ack_id, "cursor": cursor, "operations": operations},
        )
        self.assertEqual(status, 201, payload)

    def receipts(self, peer: str, query: str = "?after=0&limit=100") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/receipts{query}")

    def seed(self, *pairs: tuple[str, str]) -> None:
        for replica, op_id in pairs:
            self.post_operation(replica, operation(op_id, "k", op_id, {replica: 1}))

    def test_unregistered_peer_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.receipts("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_success_body_is_compact_json_with_one_newline(self) -> None:
        self.seed(("r1", "o1"))
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r1", "o1")])
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        # The declared length matches the exact compact body.
        self.assertEqual(int(headers["content-length"]), len(raw))
        digest = digest_of(
            '[{"peerId":"peer-a","ackId":"ack-1","cursor":1,"operations":['
            '{"replicaId":"r1","operationId":"o1"}]}]'
        )
        self.assertEqual(
            raw,
            (
                '{"receipts":[{"peerId":"peer-a","ackId":"ack-1","cursor":1,'
                '"operations":[{"replicaId":"r1","operationId":"o1"}]}],'
                '"nextCursor":1,"hasMore":false,"algorithm":"sha256",'
                '"digest":"%s","receiptsCount":1}\n' % digest
            ).encode("utf-8"),
        )
        # Exactly the six contracted fields, in the contracted order.
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            list(payload),
            ["receipts", "nextCursor", "hasMore", "algorithm", "digest", "receiptsCount"],
        )
        self.assertEqual(
            list(payload["receipts"][0]),
            ["peerId", "ackId", "cursor", "operations"],
        )
        self.assertEqual(
            list(payload["receipts"][0]["operations"][0]),
            ["replicaId", "operationId"],
        )

    def test_empty_receipt_set(self) -> None:
        self.register("peer-a", 0)
        status, payload = self.receipts("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "receipts": [],
                "nextCursor": 0,
                "hasMore": False,
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "receiptsCount": 0,
            },
        )

    def test_receipts_follow_commit_order_and_page(self) -> None:
        self.seed(("r1", "o1"), ("r2", "o2"), ("r3", "o3"))
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 2, [identity("r1", "o1"), identity("r2", "o2")])
        self.acknowledge("peer-a", "ack-2", 3, [identity("r3", "o3")])
        status, first = self.receipts("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(
            first["receipts"],
            [
                {
                    "peerId": "peer-a",
                    "ackId": "ack-1",
                    "cursor": 2,
                    "operations": [identity("r1", "o1"), identity("r2", "o2")],
                }
            ],
        )
        self.assertEqual(first["nextCursor"], 1)
        self.assertIs(first["hasMore"], True)
        self.assertEqual(first["receiptsCount"], 2)
        status, second = self.receipts("peer-a", f"?after={first['nextCursor']}&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in second["receipts"]], ["ack-2"])
        self.assertEqual(second["nextCursor"], 2)
        self.assertIs(second["hasMore"], False)
        self.assertEqual(second["digest"], first["digest"])
        # Cursor resume lands exactly on the empty tail.
        status, tail = self.receipts("peer-a", "?after=2&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 2)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["receiptsCount"], 2)
        self.assertEqual(tail["digest"], first["digest"])

    def test_query_does_not_change_checkpoint_or_receipts(self) -> None:
        self.seed(("r1", "o1"))
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r1", "o1")])
        # Missing-parameter queries are rejected before any state read.
        for query in ("", "?after=0", "?limit=1"):
            status, _ = self.receipts("peer-a", query)
            self.assertEqual(status, 400, query)
        status, first = self.receipts("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        # Repeated queries return the same page and summary.
        status, again = self.receipts("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(again, first)

    def test_invalid_queries_are_400_and_change_nothing(self) -> None:
        self.seed(("r1", "o1"))
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r1", "o1")])
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
            "?after=0&after=0&limit=10",
            "?after=0&limit=1&limit=2",
            "?foo=1&after=0&limit=10",
            "?after=0&limit=10&foo=1",
            "?=1&after=0&limit=10",
            # after past the receipt count (one committed receipt).
            "?after=2&limit=10",
        ]
        for query in bad_queries:
            status, payload = self.receipts("peer-a", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # The receipts and the summary are untouched.
        status, payload = self.receipts("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in payload["receipts"]], ["ack-1"])
        self.assertEqual(payload["receiptsCount"], 1)

    def test_route_shapes_are_404_before_query_checks(self) -> None:
        self.register("peer-a", 0)
        not_found_paths = [
            "/v1/sync/peers//receipts?after=0&limit=1",
            "/v1/sync/peers//receipts?after=x",
            "/v1/sync/peers/peer-a/receipts/extra",
            "/v1/sync/peers/peer-a/receipts/extra?after=0&limit=1",
            "/v1/sync/peers/peer-a/receipts/",
            "/v1/sync/peers",
            "/v1/sync/peers/peer-a",
            "/v1/sync/peers/peer-a/not-receipts",
            "/v1/sync/receipts/peer-a",
            "/v1/sync/peer/peer-a/receipts",
            "/v2/sync/peers/peer-a/receipts",
        ]
        for path in not_found_paths:
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_receipts_route_does_not_shadow_sibling_routes(self) -> None:
        self.seed(("r1", "o1"))
        self.register("peer-a", 1)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(("r1", "o1"))
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        status, _ = self.request(
            "POST",
            "/v1/sync/peers/peer%20one/acknowledge",
            {"ackId": "ack-1", "cursor": 1, "operations": [identity("r1", "o1")]},
        )
        self.assertEqual(status, 201)
        status, payload = self.receipts("peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receipts"][0]["peerId"], "peer one")
        self.assertEqual(payload["receiptsCount"], 1)

    def test_unknown_peer_with_well_formed_query_is_404(self) -> None:
        # Query validation runs first: a missing/malformed parameter is a
        # 400 even for an unregistered peer; a fully well-formed query
        # reaches the existence check and stays a not-found.
        for query in ("", "?after=0", "?limit=1", "?after=999"):
            status, payload = self.receipts("nope", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        for query in ("?after=0&limit=1", "?after=999&limit=10"):
            status, payload = self.receipts("nope", query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_empty_peer_beats_query_validation(self) -> None:
        # Path shape takes precedence: an empty peer segment with a
        # malformed query is still 404, not 400.
        for query in ("?limit=0", "?after=x", "?unknown=1", "?after=-1"):
            status, payload = self.request("GET", f"/v1/sync/peers//receipts{query}")
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_post_to_receipts_route_is_404(self) -> None:
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/receipts", {"cursor": 0}
        )
        self.assertEqual(status, 404)


class AuthenticatedPeerReceiptsTests(unittest.TestCase):
    """Bearer authentication on the receipts route; health stays anonymous."""

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
        status, headers, raw = self.raw_request("/v1/sync/peers/peer-a/receipts", None)
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_wrong_and_malformed_tokens_are_401(self) -> None:
        for value in ("Bearer wrong", "Bearer", "Basic " + TOKEN, "Token " + TOKEN,
                      f"Bearer  {TOKEN}", f"bearer {TOKEN}"):
            status, headers, _ = self.raw_request(
                "/v1/sync/peers/peer-a/receipts", [("Authorization", value)]
            )
            self.assertEqual(status, 401, value)
            self.assertEqual(headers.get("www-authenticate"), "Bearer", value)

    def test_duplicate_headers_are_401_even_when_both_match(self) -> None:
        status, _, _ = self.raw_request(
            "/v1/sync/peers/peer-a/receipts",
            [("Authorization", AUTH_HEADER), ("Authorization", AUTH_HEADER)],
        )
        self.assertEqual(status, 401)

    def test_authenticated_query_works(self) -> None:
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
            "/v1/sync/peers/peer-a/receipts?after=0&limit=100",
            headers={"Authorization": AUTH_HEADER},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            json.loads(response.read().decode("utf-8")),
            {
                "receipts": [],
                "nextCursor": 0,
                "hasMore": False,
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "receiptsCount": 0,
            },
        )
        conn.close()


class PersistentPeerReceiptsHttpTests(unittest.TestCase):
    """Receipts over real HTTP with a data file: recovery and no temp files."""

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

    def commit_receipts(self, server: SemanticStateServer) -> None:
        for i in range(3):
            replica = f"r{i}"
            self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(f"o{i}", "k", f"v{i}", {replica: 1}),
            )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 0})
        self.request(
            server,
            "POST",
            "/v1/sync/peers/peer-a/acknowledge",
            {
                "ackId": "ack-1",
                "cursor": 2,
                "operations": [identity("r0", "o0"), identity("r1", "o1")],
            },
        )
        self.request(
            server,
            "POST",
            "/v1/sync/peers/peer-a/acknowledge",
            {"ackId": "ack-2", "cursor": 3, "operations": [identity("r2", "o2")]},
        )

    def test_receipts_survive_restart_with_same_pages_and_digest(self) -> None:
        server = self.start_server()
        self.commit_receipts(server)
        status, before = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in before["receipts"]], ["ack-1"])
        self.assertEqual((before["nextCursor"], before["hasMore"]), (1, True))
        self.assertEqual(before["receiptsCount"], 2)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Resume with the returned cursor after restart.
        status, after = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts?after=1&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in after["receipts"]], ["ack-2"])
        self.assertEqual((after["nextCursor"], after["hasMore"]), (2, False))
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["receiptsCount"], 2)
        # The full page preserves commit order after recovery.
        status, full = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["ackId"] for r in full["receipts"]], ["ack-1", "ack-2"]
        )
        self.assertEqual(full["digest"], before["digest"])
        # Error boundary survives recovery as well.
        status, payload = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts?after=3&limit=10"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_queries_create_no_temp_files(self) -> None:
        server = self.start_server()
        self.commit_receipts(server)
        before = self.data_file.read_bytes()
        for query in ("?after=0&limit=1", "?after=1&limit=1", "?after=0&limit=100"):
            status, _ = self.request(
                server, "GET", f"/v1/sync/peers/peer-a/receipts{query}"
            )
            self.assertEqual(status, 200)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


class CommandLinePeerReceiptsTests(unittest.TestCase):
    """The real ``python -m`` entry point serves the receipts route."""

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

    def test_receipts_served_end_to_end(self) -> None:
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
            conn.request(
                "POST",
                "/v1/sync/peers/peer-a/acknowledge",
                body=json.dumps(
                    {
                        "ackId": "ack-1",
                        "cursor": 1,
                        "operations": [identity("r1", "o1")],
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.request("GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=1")
            response = conn.getresponse()
            raw = response.read()
            self.assertEqual(response.status, 200)
            payload = json.loads(raw.decode("utf-8"))
            self.assertEqual(
                payload["receipts"],
                [
                    {
                        "peerId": "peer-a",
                        "ackId": "ack-1",
                        "cursor": 1,
                        "operations": [identity("r1", "o1")],
                    }
                ],
            )
            self.assertEqual(payload["receiptsCount"], 1)
            self.assertEqual(
                payload["digest"],
                digest_of(
                    '[{"peerId":"peer-a","ackId":"ack-1","cursor":1,'
                    '"operations":[{"replicaId":"r1","operationId":"o1"}]}]'
                ),
            )
            conn.close()
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
