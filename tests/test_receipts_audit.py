"""Tests for the per-peer confirmation-chain audit query.

The receipts audit endpoint is::

    GET /v1/sync/peers/{peerId}/receipts/audit?after=N&limit=N

It pages the same consumption receipts as the plain receipts route
(creation order, required ``after``/``limit`` paging) and additionally
reports the integrity of the peer's whole confirmation chain: whether
the receipts seamlessly cover the shared accepted log from the earliest
receipt's derived start through the last confirmation cursor, with
separate anomaly lists for gaps, overlaps, identity mismatches, and
cursor regressions. The digest, receipt count, and audit conclusion
always cover the complete receipt history, never the current page. The
endpoint is strictly read-only.

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
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN = "receipts-audit-token_42"
AUTH_HEADER = f"Bearer {TOKEN}"

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def empty_anomaly_lists() -> dict:
    return {
        "gaps": [],
        "overlaps": [],
        "identityMismatches": [],
        "cursorRegressions": [],
    }


def audit_ok(start: int, end: int) -> dict:
    audit = {"status": "ok", "coverage": {"start": start, "end": end}}
    audit.update(empty_anomaly_lists())
    return audit


class PeerReceiptsAuditStoreTests(unittest.TestCase):
    """Store-level chain-audit semantics, in memory."""

    def seed_operations(self, store: StateStore, count: int, key: str = "k") -> None:
        for i in range(count):
            replica = f"r{i}"
            self.assertIs(
                store.apply_operation(
                    replica, operation(f"o{i}", key, f"v{i}", {replica: 1})
                ),
                HTTPStatus.CREATED,
            )

    def ack(self, store: StateStore, peer: str, ack_id: str, cursor: int, ids: list) -> None:
        status, error = store.acknowledge_operations(peer, ack_id, cursor, ids)
        self.assertIs(status, HTTPStatus.CREATED, (status, error))

    def test_unregistered_peer_is_not_found(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        status, payload = store.get_peer_receipts_audit("nobody", 0, 100)
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unregistered_peer_lookup_changes_nothing(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.get_peer_receipts_audit("nobody", 0, 100)
        # The peer stays unregistered for every peer route.
        status, payload = store.get_checkpoint("nobody")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts_is_an_ok_empty_coverage(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        status, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["audit"], audit_ok(0, 0))

    def test_seamless_chain_with_empty_segment_is_ok_end_to_end(self) -> None:
        store = StateStore()
        self.seed_operations(store, 3)
        store.save_checkpoint("peer-a", 0)
        self.ack(store, "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        # An empty confirmation segment at the same cursor stays legal.
        self.ack(store, "peer-a", "ack-2", 2, [])
        self.ack(store, "peer-a", "ack-3", 3, [identity("r2", "o2")])
        status, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receiptsCount"], 3)
        self.assertEqual(payload["audit"], audit_ok(0, 3))

    def test_first_start_is_derived_from_count_and_cursor(self) -> None:
        # A checkpoint registered past 0: the first receipt confirms
        # accepted[2:4], so coverage starts at 2.
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 2)
        self.ack(store, "peer-a", "ack-1", 4, [identity("r2", "o2"), identity("r3", "o3")])
        status, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["audit"], audit_ok(2, 4))

    def test_gap_between_receipts_is_reported(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 4)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        # [2,3): accepted record 1 is confirmed by no receipt.
        store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        status, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        audit = payload["audit"]
        self.assertEqual(audit["status"], "broken")
        self.assertEqual(audit["coverage"], {"start": 0, "end": 3})
        self.assertEqual(
            audit["gaps"],
            [{"receiptIndex": 1, "ackId": "ack-2", "from": 1, "to": 2}],
        )
        self.assertEqual(audit["overlaps"], [])
        self.assertEqual(audit["identityMismatches"], [])
        self.assertEqual(audit["cursorRegressions"], [])

    def test_overlap_between_receipts_is_reported(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 4)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        # [1,3): record 1 is confirmed twice.
        store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r1", "o1"), identity("r2", "o2")],
        }
        _, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        audit = payload["audit"]
        self.assertEqual(audit["status"], "broken")
        self.assertEqual(
            audit["overlaps"],
            [{"receiptIndex": 1, "ackId": "ack-2", "from": 2, "to": 1}],
        )
        self.assertEqual(audit["gaps"], [])

    def test_identity_mismatch_keeps_position_and_both_identities(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 2)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r9", "bogus")],
        }
        _, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        audit = payload["audit"]
        self.assertEqual(audit["status"], "broken")
        self.assertEqual(
            audit["identityMismatches"],
            [
                {
                    "receiptIndex": 0,
                    "ackId": "ack-1",
                    "position": 1,
                    "expected": {"replicaId": "r1", "operationId": "o1"},
                    "observed": {"replicaId": "r9", "operationId": "bogus"},
                }
            ],
        )
        self.assertEqual(audit["gaps"], [])
        self.assertEqual(audit["overlaps"], [])
        self.assertEqual(audit["cursorRegressions"], [])

    def test_position_outside_the_log_is_a_mismatch_with_null_expected(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.save_checkpoint("peer-a", 1)
        # A receipt claiming a position the current log does not reach.
        store._acks[("peer-a", "ack-x")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r9", "ghost")],
        }
        _, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        mismatches = payload["audit"]["identityMismatches"]
        self.assertEqual(
            mismatches,
            [
                {
                    "receiptIndex": 0,
                    "ackId": "ack-x",
                    "position": 1,
                    "expected": None,
                    "observed": {"replicaId": "r9", "operationId": "ghost"},
                }
            ],
        )
        self.assertEqual(payload["audit"]["status"], "broken")

    def test_cursor_regression_is_reported(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 4)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        store._acks[("peer-a", "ack-2")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        _, payload = store.get_peer_receipts_audit("peer-a", 0, 100)
        audit = payload["audit"]
        self.assertEqual(audit["status"], "broken")
        self.assertEqual(
            audit["cursorRegressions"],
            [{"receiptIndex": 1, "ackId": "ack-2", "from": 2, "to": 1}],
        )

    def test_paging_trims_only_receipts_but_audit_and_summary_cover_history(self) -> None:
        store = StateStore()
        self.seed_operations(store, 3)
        store.save_checkpoint("peer-a", 0)
        self.ack(store, "peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.ack(store, "peer-a", "ack-2", 2, [identity("r1", "o1")])
        self.ack(store, "peer-a", "ack-3", 3, [identity("r2", "o2")])
        _, first = store.get_peer_receipts_audit("peer-a", 0, 1)
        _, middle = store.get_peer_receipts_audit("peer-a", 1, 1)
        _, tail = store.get_peer_receipts_audit("peer-a", 3, 1)
        self.assertEqual([r["ackId"] for r in first["receipts"]], ["ack-1"])
        self.assertEqual(first["nextCursor"], 1)
        self.assertIs(first["hasMore"], True)
        self.assertEqual([r["ackId"] for r in middle["receipts"]], ["ack-2"])
        self.assertEqual(middle["nextCursor"], 2)
        # The empty tail is a stable empty page over the same history.
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)
        for page in (first, middle, tail):
            self.assertEqual(page["receiptsCount"], 3)
            self.assertEqual(page["audit"], audit_ok(0, 3))
        self.assertEqual(first["digest"], middle["digest"])
        self.assertEqual(first["digest"], tail["digest"])
        # The digest is exactly the receipts-route digest of the full set.
        _, plain = store.get_peer_receipts("peer-a", 0, 100)
        self.assertEqual(first["digest"], plain["digest"])

    def test_after_past_count_is_value_error(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        with self.assertRaises(ValueError):
            store.get_peer_receipts_audit("peer-a", 1, 100)

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.save_checkpoint("peer-a", 0)
        self.ack(store, "peer-a", "ack-1", 1, [identity("r0", "o0")])
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("audit query must not persist"),
        ):
            status, payload = store.get_peer_receipts_audit("peer-a", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        # Neither the checkpoint nor the receipts move.
        status, checkpoint = store.get_checkpoint("peer-a")
        self.assertEqual(checkpoint, {"peerId": "peer-a", "cursor": 1})
        self.assertEqual(payload["receiptsCount"], 1)


class HttpPeerReceiptsAuditTests(unittest.TestCase):
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

    def audit(self, peer: str, query: str = "?after=0&limit=100") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/receipts/audit{query}")

    def seed(self, count: int, key: str = "k") -> None:
        for i in range(count):
            self.post_operation(f"r{i}", operation(f"o{i}", key, f"v{i}", {f"r{i}": 1}))

    def test_empty_set_success_body_is_compact_with_one_newline(self) -> None:
        self.register("peer-a", 0)
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(int(headers["content-length"]), len(raw))
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            list(payload),
            [
                "receipts",
                "nextCursor",
                "hasMore",
                "algorithm",
                "digest",
                "receiptsCount",
                "audit",
            ],
        )
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(
            list(payload["audit"]),
            [
                "status",
                "coverage",
                "gaps",
                "overlaps",
                "identityMismatches",
                "cursorRegressions",
            ],
        )
        self.assertEqual(payload["audit"], audit_ok(0, 0))

    def test_healthy_chain_reports_ok_and_full_coverage(self) -> None:
        self.seed(3)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        self.acknowledge("peer-a", "ack-2", 2, [])  # legal empty segment
        self.acknowledge("peer-a", "ack-3", 3, [identity("r2", "o2")])
        status, payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receiptsCount"], 3)
        self.assertEqual(payload["audit"], audit_ok(0, 3))

    def test_paging_is_stable_and_summary_is_page_independent(self) -> None:
        self.seed(2)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.acknowledge("peer-a", "ack-2", 2, [identity("r1", "o1")])
        pages = []
        cursor = 0
        while True:
            status, page = self.audit("peer-a", f"?after={cursor}&limit=1")
            self.assertEqual(status, 200)
            self.assertEqual(page["receiptsCount"], 2)
            self.assertEqual(page["audit"], audit_ok(0, 2))
            pages.append(page)
            if not page["hasMore"]:
                break
            cursor = page["nextCursor"]
        self.assertEqual([r["ackId"] for r in pages[0]["receipts"]], ["ack-1"])
        self.assertEqual([r["ackId"] for r in pages[1]["receipts"]], ["ack-2"])
        self.assertEqual(pages[0]["digest"], pages[1]["digest"])
        # Resuming at the count yields a stable empty page, same summary.
        status, tail = self.audit("peer-a", "?after=2&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 2)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["audit"], audit_ok(0, 2))
        self.assertEqual(tail["digest"], pages[0]["digest"])

    def test_query_is_read_only_and_repeatable(self) -> None:
        self.seed(1)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        status, first = self.audit("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        status, again = self.audit("peer-a", "?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(again, first)
        # The checkpoint is untouched.
        status, checkpoint = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(checkpoint, {"peerId": "peer-a", "cursor": 1})

    def test_unregistered_peer_with_well_formed_query_is_404(self) -> None:
        self.seed(1)
        for query in ("?after=0&limit=1", "?after=0&limit=100"):
            status, payload = self.audit("nope", query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)
        # Malformed queries are rejected at parameter validation, for
        # unknown peers too.
        for query in ("", "?after=0", "?limit=1", "?after=x&limit=1"):
            status, payload = self.audit("nope", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_invalid_queries_are_400_and_change_nothing(self) -> None:
        self.seed(1)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        bad_queries = [
            "",
            "?after=0",
            "?limit=10",
            "?after=&limit=10",
            "?after&limit=10",
            "?after=0&limit=",
            "?after=0&limit",
            "?after=-1&limit=10",
            "?after=1.0&limit=10",
            "?after=0x1&limit=10",
            "?after=%2B1&limit=10",
            "?after=1%20&limit=10",
            "?after=%D9%A1&limit=10",
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=-1",
            "?after=0&limit=one",
            "?after=0&after=0&limit=10",
            "?after=0&limit=1&limit=2",
            "?foo=1&after=0&limit=10",
            "?after=0&limit=10&foo=1",
            "?=1&after=0&limit=10",
            "?after=2&limit=10",  # past the single committed receipt
        ]
        for query in bad_queries:
            status, payload = self.audit("peer-a", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        status, payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receiptsCount"], 1)
        self.assertEqual(payload["audit"], audit_ok(0, 1))

    def test_route_shapes_are_404_before_query_checks(self) -> None:
        self.register("peer-a", 0)
        not_found_paths = [
            "/v1/sync/peers//receipts/audit?after=0&limit=1",
            "/v1/sync/peers//receipts/audit?after=x",
            "/v1/sync/peers/peer-a/receipts/audit/extra",
            "/v1/sync/peers/peer-a/receipts/audit/extra?after=0&limit=1",
            "/v1/sync/peers/peer-a/receipts/audit/",
            "/v1/sync/peers/peer-a/audit",
            "/v1/sync/peers/peer-a/receipts/not-audit",
            "/v1/sync/peers/peer-a/receipts/audit2",
            "/v1/sync/peers/peer-a/acknowledge/audit",
            "/v1/sync/peers/peer-a",
            "/v1/sync/peers",
            "/v2/sync/peers/peer-a/receipts/audit",
        ]
        for path in not_found_paths:
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_empty_peer_beats_query_validation(self) -> None:
        for query in ("?limit=0", "?after=x", "?unknown=1", "?after=-1&limit=1"):
            status, payload = self.request(
                "GET", f"/v1/sync/peers//receipts/audit{query}"
            )
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_post_to_audit_route_is_404(self) -> None:
        status, _, _ = self.request_raw(
            "POST", "/v1/sync/peers/peer-a/receipts/audit", {}
        )
        self.assertEqual(status, 404)

    def test_audit_route_does_not_shadow_sibling_routes(self) -> None:
        self.seed(1)
        self.register("peer-a", 1)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["receipts"], [])

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(1)
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        status, _ = self.request(
            "POST",
            "/v1/sync/peers/peer%20one/acknowledge",
            {"ackId": "ack-1", "cursor": 1, "operations": [identity("r0", "o0")]},
        )
        self.assertEqual(status, 201)
        status, payload = self.audit("peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receipts"][0]["peerId"], "peer one")
        self.assertEqual(payload["audit"], audit_ok(0, 1))

    def test_concurrent_commits_always_observe_a_complete_chain(self) -> None:
        self.seed(40)
        self.register("peer-a", 0)
        observations: list = []
        stop = threading.Event()

        def commit() -> None:
            for i in range(40):
                self.acknowledge(
                    "peer-a", f"ack-{i}", i + 1, [identity(f"r{i}", f"o{i}")]
                )

        def read_audit() -> None:
            while not stop.is_set():
                status, payload = self.audit("peer-a", "?after=0&limit=1")
                if status != 200:
                    observations.append(("status", status))
                    return
                audit = payload["audit"]
                # Acknowledge commits only ever extend a seamless chain, so
                # every observed snapshot must be internally consistent;
                # the page-independent audit covers all committed receipts.
                if audit["status"] != "ok":
                    observations.append(("broken", audit))
                    return
                if audit["coverage"] != {"start": 0, "end": payload["receiptsCount"]}:
                    observations.append(("coverage", audit, payload["receiptsCount"]))
                    return
                if payload["receiptsCount"] != audit["coverage"]["end"]:
                    observations.append(("count", payload))
                    return

        reader = threading.Thread(target=read_audit)
        reader.start()
        commit()
        stop.set()
        reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        self.assertEqual(observations, [])
        status, payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["audit"], audit_ok(0, 40))


class AuthenticatedPeerReceiptsAuditTests(unittest.TestCase):
    """Bearer authentication on the receipts audit route; health anonymous."""

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
        status, headers, raw = self.raw_request(
            "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1", None
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_wrong_and_malformed_tokens_are_401(self) -> None:
        for value in ("Bearer wrong", "Bearer", "Basic " + TOKEN, "Token " + TOKEN,
                      f"Bearer  {TOKEN}", f"bearer {TOKEN}"):
            status, headers, _ = self.raw_request(
                "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1",
                [("Authorization", value)],
            )
            self.assertEqual(status, 401, value)
            self.assertEqual(headers.get("www-authenticate"), "Bearer", value)

    def test_duplicate_headers_are_401_even_when_both_match(self) -> None:
        status, _, _ = self.raw_request(
            "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1",
            [("Authorization", AUTH_HEADER), ("Authorization", AUTH_HEADER)],
        )
        self.assertEqual(status, 401)

    def test_401_precedes_400_and_404(self) -> None:
        for path in (
            "/v1/sync/peers/peer-a/receipts/audit?after=zzz",
            "/v1/sync/peers/nope/receipts/audit?after=0&limit=1",
            "/v1/sync/peers//receipts/audit?after=0&limit=1",
            "/v1/sync/peers/peer-a/receipts/audit/extra",
        ):
            status, _, _ = self.raw_request(path, None)
            self.assertEqual(status, 401, path)

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
            "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100",
            headers={"Authorization": AUTH_HEADER},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["audit"], audit_ok(0, 0))
        conn.close()


class PersistentPeerReceiptsAuditHttpTests(unittest.TestCase):
    """Chain audit over real HTTP with a data file: recovery and read-only files."""

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

    def commit_chain(self, server: SemanticStateServer) -> None:
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

    def test_audit_survives_restart_identically(self) -> None:
        server = self.start_server()
        self.commit_chain(server)
        status, before = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in before["receipts"]], ["ack-1"])
        self.assertEqual(before["audit"], audit_ok(0, 3))
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=1&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in after["receipts"]], ["ack-2"])
        self.assertEqual((after["nextCursor"], after["hasMore"]), (2, False))
        self.assertEqual(after["audit"], before["audit"])
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["receiptsCount"], 2)

    def test_queries_leave_the_data_file_untouched(self) -> None:
        server = self.start_server()
        self.commit_chain(server)
        before = self.data_file.read_bytes()
        for query in ("?after=0&limit=1", "?after=1&limit=1", "?after=2&limit=1"):
            status, _ = self.request(
                server, "GET", f"/v1/sync/peers/peer-a/receipts/audit{query}"
            )
            self.assertEqual(status, 200)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_file_without_acks_recovers_with_an_ok_empty_audit(self) -> None:
        self.data_file.write_text(
            '{"version":1,"operations":[],"checkpoints":{"peer-a":0}}',
            encoding="utf-8",
        )
        server = self.start_server()
        status, payload = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["audit"], audit_ok(0, 0))


class CommandLinePeerReceiptsAuditTests(unittest.TestCase):
    """The real ``python -m`` entry point serves the receipts audit route."""

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

    def test_audit_served_end_to_end(self) -> None:
        proc = self.spawn()
        try:
            self.wait_for_health()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                "/v1/sync/operations",
                body=json.dumps(
                    {"operations": [
                        {"replicaId": "r1",
                         "operation": operation("o1", "k", "v", {"r1": 1})},
                    ]}
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
            conn.request("GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1")
            response = conn.getresponse()
            raw = response.read()
            self.assertEqual(response.status, 200)
            self.assertTrue(raw.endswith(b"\n"))
            payload = json.loads(raw.decode("utf-8"))
            self.assertEqual(payload["receiptsCount"], 1)
            self.assertEqual(
                [r["ackId"] for r in payload["receipts"]], ["ack-1"]
            )
            self.assertEqual(payload["audit"], audit_ok(0, 1))
            conn.close()
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
