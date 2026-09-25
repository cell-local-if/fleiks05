"""Tests for the sender-side receipt-chain audit endpoint.

The audit endpoint is::

    GET /v1/sync/peers/{peerId}/receipts/audit?after=N&limit=N

It pages a registered peer's committed consumption receipts using the
exact same receipt encoding and paging rules as
``GET /v1/sync/peers/{peerId}/receipts``, and additionally audits the
whole confirmation chain the receipts form against the shared accepted
log: the response carries the whole-history digest summary, the audited
coverage interval (from the first receipt's derived start to the last
confirmation cursor), an ``ok``/``broken`` conclusion, and an anomaly
list reporting gaps, overlaps, identity mismatches, and cursor
regressions with their positions. Paging only trims the exported
receipts; the conclusion, coverage, anomalies, and digest always come
from the complete history. The endpoint is strictly read-only.

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

AUDIT_FIELDS = [
    "receipts",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "receiptsCount",
    "coverage",
    "conclusion",
    "anomalies",
]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gap(index: int, start_from: int, to: int) -> dict:
    return {"kind": "gap", "index": index, "from": start_from, "to": to}


def overlap(index: int, start_from: int, to: int) -> dict:
    return {"kind": "overlap", "index": index, "from": start_from, "to": to}


def mismatch(index: int, position: int) -> dict:
    return {"kind": "identityMismatch", "index": index, "position": position}


def regression(index: int, cursor: int) -> dict:
    return {"kind": "cursorRegression", "index": index, "cursor": cursor}


class ReceiptsAuditStoreTests(unittest.TestCase):
    """Store-level audit semantics, in memory."""

    def seed_operations(self, store: StateStore, count: int) -> None:
        for i in range(count):
            replica = f"r{i}"
            self.assertIs(
                store.apply_operation(
                    replica, operation(f"o{i}", "k", f"v{i}", {replica: 1})
                ),
                HTTPStatus.CREATED,
            )

    def seed_mixed_log(self, store: StateStore, count: int) -> None:
        """Commit ``count`` records interleaved with other-key records."""
        for i in range(count):
            replica = f"r{i}"
            self.assertIs(
                store.apply_operation(
                    replica, operation(f"o{i}", "k", f"v{i}", {replica: 1})
                ),
                HTTPStatus.CREATED,
            )
            self.assertIs(
                store.apply_operation(
                    f"x{i}", operation(f"x{i}", "other", "x", {f"x{i}": 1})
                ),
                HTTPStatus.CREATED,
            )

    def test_unregistered_peer_is_not_found(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.audit_peer_receipts("nobody", 0, 100)
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_empty_receipt_set_is_a_complete_empty_chain(self) -> None:
        store = StateStore()
        store.save_checkpoint("peer-a", 0)
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 0})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])
        self.assertEqual(list(payload), AUDIT_FIELDS)
        self.assertEqual(list(payload["coverage"]), ["start", "end"])

    def test_contiguous_chain_is_ok_with_derived_coverage(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        # An empty confirmation segment in the middle stays legal.
        store.acknowledge_operations("peer-a", "ack-empty", 2, [])
        store.acknowledge_operations(
            "peer-a", "ack-2", 4, [identity("r2", "o2"), identity("r3", "o3")]
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([r["ackId"] for r in payload["receipts"]],
                         ["ack-1", "ack-empty", "ack-2"])
        self.assertEqual(payload["coverage"], {"start": 0, "end": 4})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])
        self.assertEqual(payload["receiptsCount"], 3)

    def test_chain_may_start_at_a_later_registered_checkpoint(self) -> None:
        store = StateStore()
        self.seed_operations(store, 5)
        # The peer registers once the first two records already exist; its
        # earliest receipt starts there, and the audit covers exactly that
        # interval rather than presuming start 0.
        store.save_checkpoint("peer-a", 2)
        self.assertIs(
            store.acknowledge_operations(
                "peer-a", "ack-1", 4, [identity("r2", "o2"), identity("r3", "o3")]
            )[0],
            HTTPStatus.CREATED,
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 2, "end": 4})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])

    def test_empty_first_and_last_segments_are_legal(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations("peer-a", "ack-0a", 0, [])
        store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        store.acknowledge_operations("peer-a", "ack-0b", 2, [])
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 2})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])
        self.assertEqual(
            [r["ackId"] for r in payload["receipts"]],
            ["ack-0a", "ack-1", "ack-0b"],
        )

    def test_identities_are_matched_at_global_log_positions(self) -> None:
        store = StateStore()
        # Every confirmed record is interleaved with an other-key record;
        # the acknowledged identities must still match the absolute log
        # position each cursor implies.
        self.seed_mixed_log(store, 3)
        store.save_checkpoint("peer-a", 0)
        self.assertIs(
            store.acknowledge_operations(
                "peer-a",
                "ack-1",
                6,
                [
                    identity("r0", "o0"),
                    identity("x0", "x0"),
                    identity("r1", "o1"),
                    identity("x1", "x1"),
                    identity("r2", "o2"),
                    identity("x2", "x2"),
                ],
            )[0],
            HTTPStatus.CREATED,
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 6})
        self.assertEqual(payload["conclusion"], "ok")

    def inject(self, store: StateStore, peer: str, items: list[tuple]) -> None:
        """Install receipts directly, bypassing acknowledge validation.

        This is how the audit sees chains the write API can never commit
        (gaps, overlaps, wrong identities, regressions): such snapshots can
        only arise from an out-of-band/corrupt state, which the read-only
        audit must still report rather than hide.
        """
        store._acks = {
            (peer, ack_id): {"cursor": cursor, "operations": operations}
            for ack_id, cursor, operations in items
        }

    def test_gap_between_receipts_is_reported(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]),
                # Covers only log position 3: position 2 is never confirmed.
                ("ack-2", 4, [identity("r3", "o3")]),
            ],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 4})
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["anomalies"], [gap(1, 2, 3)])

    def test_overlap_between_receipts_is_reported(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 3, [identity("r0", "o0"), identity("r1", "o1"),
                              identity("r2", "o2")]),
                # Starts back at position 2, already confirmed by ack-1.
                ("ack-2", 4, [identity("r2", "o2"), identity("r3", "o3")]),
            ],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["anomalies"], [overlap(1, 2, 3)])

    def test_identity_mismatch_keeps_position_and_audit_order(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 2, [identity("r0", "o0"), identity("r9", "wrong")]),
                ("ack-2", 4, [identity("r2", "o2"), identity("r8", "nope")]),
            ],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(
            payload["anomalies"], [mismatch(0, 1), mismatch(1, 3)]
        )
        self.assertEqual(list(payload["anomalies"][0]), ["kind", "index", "position"])

    def test_cursor_regression_is_reported_once(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]),
                ("ack-2", 4, []),  # advances via an empty segment
                ("ack-3", 3, [identity("r2", "o2")]),  # cursor moves back
            ],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 3})
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(
            payload["anomalies"],
            [gap(1, 2, 4), regression(2, 3)],
        )
        self.assertEqual(list(payload["anomalies"][1]), ["kind", "index", "cursor"])

    def test_each_anomaly_kind_can_appear_together(self) -> None:
        store = StateStore()
        self.seed_operations(store, 6)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 2, [identity("r0", "o0"), identity("r9", "bad")]),  # mismatch
                ("ack-2", 4, [identity("r3", "o3")]),                          # gap at 2
                ("ack-3", 4, [identity("r3", "o3")]),                          # overlap back to 3
                ("ack-4", 3, [identity("r2", "o2")]),                          # regression
            ],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["conclusion"], "broken")
        kinds = [entry["kind"] for entry in payload["anomalies"]]
        self.assertEqual(
            kinds,
            ["identityMismatch", "gap", "overlap", "cursorRegression"],
        )
        indices = [entry["index"] for entry in payload["anomalies"]]
        self.assertEqual(indices, sorted(indices))

    def test_positions_outside_the_log_are_identity_mismatches(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [("ack-1", 4, [identity("r0", "o0"), identity("r1", "o1"),
                           identity("r2", "o2"), identity("r3", "o3")])],
        )
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 4})
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["anomalies"], [mismatch(0, 2), mismatch(0, 3)])

    def test_empty_first_receipt_past_the_log_end_is_a_gap(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        self.inject(store, "peer-a", [("ack-1", 4, [])])
        status, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["coverage"], {"start": 4, "end": 4})
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["anomalies"], [gap(0, 2, 4)])

    def test_paging_only_trims_receipts_not_the_audit(self) -> None:
        store = StateStore()
        self.seed_operations(store, 5)
        store.save_checkpoint("peer-a", 0)
        for i in range(5):
            self.assertIs(
                store.acknowledge_operations(
                    "peer-a", f"ack-{i}", i + 1, [identity(f"r{i}", f"o{i}")]
                )[0],
                HTTPStatus.CREATED,
            )
        seen: list[dict] = []
        after = 0
        pages: list[dict] = []
        while True:
            status, page = store.audit_peer_receipts("peer-a", after, 2)
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
            seen.extend(page["receipts"])
            # The whole-history conclusions are invariant across pages.
            self.assertEqual(page["coverage"], {"start": 0, "end": 5})
            self.assertEqual(page["conclusion"], "ok")
            self.assertEqual(page["anomalies"], [])
            self.assertEqual(page["receiptsCount"], 5)
            after = page["nextCursor"]
            if not page["hasMore"]:
                break
        self.assertEqual(len(seen), 5)
        self.assertEqual([r["ackId"] for r in seen], [f"ack-{i}" for i in range(5)])
        digests = {page["digest"] for page in pages}
        self.assertEqual(len(digests), 1)
        # The digest is exactly the receipts-endpoint digest.
        _, receipts_report = store.get_peer_receipts("peer-a", 0, 100)
        self.assertEqual(digests.pop(), receipts_report["digest"])
        # An empty tail still carries the full audit.
        status, tail = store.audit_peer_receipts("peer-a", 5, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["coverage"], {"start": 0, "end": 5})
        self.assertEqual(tail["conclusion"], "ok")
        self.assertEqual(tail["anomalies"], [])
        # after past the receipt count is rejected.
        with self.assertRaises(ValueError):
            store.audit_peer_receipts("peer-a", 6, 2)

    def test_broken_chain_audit_is_stable_across_pages(self) -> None:
        store = StateStore()
        self.seed_operations(store, 4)
        store.save_checkpoint("peer-a", 0)
        self.inject(
            store,
            "peer-a",
            [
                ("ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]),
                ("ack-2", 4, [identity("r3", "o3")]),
            ],
        )
        first = store.audit_peer_receipts("peer-a", 0, 1)[1]
        second = store.audit_peer_receipts("peer-a", 1, 1)[1]
        self.assertEqual(first["anomalies"], second["anomalies"])
        self.assertEqual(first["anomalies"], [gap(1, 2, 3)])
        self.assertEqual(first["conclusion"], "broken")
        self.assertEqual(second["conclusion"], "broken")
        self.assertEqual(first["coverage"], second["coverage"])
        self.assertEqual(first["digest"], second["digest"])

    def test_peers_are_audited_independently(self) -> None:
        store = StateStore()
        self.seed_operations(store, 3)
        store.save_checkpoint("peer-a", 0)
        store.save_checkpoint("peer-b", 0)
        store.acknowledge_operations("peer-a", "ack-1", 1, [identity("r0", "o0")])
        store.acknowledge_operations(
            "peer-b", "ack-9", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        _, a = store.audit_peer_receipts("peer-a", 0, 100)
        _, b = store.audit_peer_receipts("peer-b", 0, 100)
        self.assertEqual(a["coverage"], {"start": 0, "end": 1})
        self.assertEqual(b["coverage"], {"start": 0, "end": 2})
        self.assertEqual((a["conclusion"], b["conclusion"]), ("ok", "ok"))
        self.assertNotEqual(a["digest"], b["digest"])

    def test_replayed_acknowledge_adds_no_receipt_and_moves_nothing(self) -> None:
        store = StateStore()
        self.seed_operations(store, 1)
        store.save_checkpoint("peer-a", 0)
        operations = [identity("r0", "o0")]
        store.acknowledge_operations("peer-a", "ack-1", 1, operations)
        self.assertIs(
            store.acknowledge_operations("peer-a", "ack-1", 1, operations)[0],
            HTTPStatus.OK,
        )
        _, payload = store.audit_peer_receipts("peer-a", 0, 100)
        self.assertEqual(payload["receiptsCount"], 1)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 1})
        self.assertEqual(payload["conclusion"], "ok")

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        self.seed_operations(store, 2)
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        metrics_before = store.get_metrics()
        checkpoint_before = store.get_checkpoint("peer-a")
        sync_before, _, _ = store.get_sync_operations(0, 100)
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("audit query must not persist"),
        ):
            for after in (0, 1):
                status, payload = store.audit_peer_receipts("peer-a", after, 1)
                self.assertIs(status, HTTPStatus.OK)
                self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(store.get_checkpoint("peer-a"), checkpoint_before)
        self.assertEqual(store.get_metrics(), metrics_before)
        sync_after, _, _ = store.get_sync_operations(0, 100)
        self.assertEqual(sync_after, sync_before)
        # The receipts themselves are untouched.
        _, receipts = store.get_peer_receipts("peer-a", 0, 100)
        self.assertEqual(receipts["receiptsCount"], 1)


class PersistentReceiptsAuditStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def make_store(self) -> StateStore:
        return StateStore(data_file=self.data_file)

    def commit_receipts(self, store: StateStore) -> None:
        for i in range(4):
            replica = f"r{i}"
            store.apply_operation(
                replica, operation(f"o{i}", "k", f"v{i}", {replica: 1})
            )
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        store.acknowledge_operations(
            "peer-a", "ack-2", 4, [identity("r2", "o2"), identity("r3", "o3")]
        )

    def test_recovery_reproduces_the_full_audit(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        before = store.audit_peer_receipts("peer-a", 0, 100)[1]
        del store

        reloaded = self.make_store()
        after = reloaded.audit_peer_receipts("peer-a", 0, 1)[1]
        self.assertEqual([r["ackId"] for r in after["receipts"]], ["ack-1"])
        self.assertEqual(after["nextCursor"], 1)
        self.assertIs(after["hasMore"], True)
        self.assertEqual(after["coverage"], {"start": 0, "end": 4})
        self.assertEqual(after["conclusion"], "ok")
        self.assertEqual(after["anomalies"], [])
        self.assertEqual(after["receiptsCount"], 2)
        self.assertEqual(after["digest"], before["digest"])
        full = reloaded.audit_peer_receipts("peer-a", 0, 100)[1]
        self.assertEqual(full["coverage"], before["coverage"])
        self.assertEqual(full["conclusion"], before["conclusion"])
        self.assertEqual(full["anomalies"], before["anomalies"])
        self.assertEqual(full["digest"], before["digest"])

    def test_empty_peer_recovers_to_empty_audit(self) -> None:
        store = self.make_store()
        store.save_checkpoint("peer-a", 0)
        del store
        reloaded = self.make_store()
        payload = reloaded.audit_peer_receipts("peer-a", 0, 100)[1]
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["coverage"], {"start": 0, "end": 0})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])
        self.assertEqual(payload["digest"], EMPTY_DIGEST)

    def test_query_creates_no_temp_file(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        before = Path(self.data_file).read_bytes()
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=AssertionError("audit query must not persist"),
        ):
            for after, limit in ((0, 1), (1, 1), (0, 100), (2, 10)):
                self.assertIs(
                    store.audit_peer_receipts("peer-a", after, limit)[0],
                    HTTPStatus.OK,
                )
        self.assertEqual(Path(self.data_file).read_bytes(), before)
        leftovers = [
            p.name
            for p in Path(self.data_file).parent.iterdir()
            if p.name != Path(self.data_file).name
        ]
        self.assertEqual(leftovers, [])


class HttpReceiptsAuditTests(unittest.TestCase):
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

    def seed(self, count: int) -> None:
        for i in range(count):
            self.post_operation(f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}))

    def test_unregistered_peer_is_404_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.audit("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # Nothing got registered by the failed lookup.
        status, checkpoint = self.request("GET", "/v1/sync/peers/nope/checkpoint")
        self.assertEqual(status, 404)
        self.assertEqual(checkpoint, {"error": "not_found"})
        status, again = self.audit("nope")
        self.assertEqual(status, 404)
        self.assertEqual(again, payload)

    def test_empty_receipt_set_body_is_compact_ordered_json(self) -> None:
        self.register("peer-a", 0)
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertEqual(int(headers["content-length"]), len(raw))
        self.assertEqual(
            raw,
            (
                '{"receipts":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
                f'"digest":"{EMPTY_DIGEST}","receiptsCount":0,'
                '"coverage":{"start":0,"end":0},"conclusion":"ok","anomalies":[]}\n'
            ).encode("utf-8"),
        )
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(payload), AUDIT_FIELDS)
        self.assertEqual(list(payload["coverage"]), ["start", "end"])

    def test_success_body_field_order_and_framing(self) -> None:
        self.seed(1)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertEqual(int(headers["content-length"]), len(raw))
        digest = digest_of(
            '[{"peerId":"peer-a","ackId":"ack-1","cursor":1,'
            '"operations":[{"replicaId":"r0","operationId":"o0"}]}]'
        )
        self.assertEqual(
            raw,
            (
                '{"receipts":[{"peerId":"peer-a","ackId":"ack-1","cursor":1,'
                '"operations":[{"replicaId":"r0","operationId":"o0"}]}],'
                '"nextCursor":1,"hasMore":false,"algorithm":"sha256",'
                f'"digest":"{digest}","receiptsCount":1,'
                '"coverage":{"start":0,"end":1},"conclusion":"ok","anomalies":[]}\n'
            ).encode("utf-8"),
        )
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(payload), AUDIT_FIELDS)
        self.assertEqual(
            list(payload["receipts"][0]), ["peerId", "ackId", "cursor", "operations"]
        )
        self.assertEqual(
            list(payload["receipts"][0]["operations"][0]), ["replicaId", "operationId"]
        )
        # Counts and cursors are JSON integers, never strings.
        self.assertIs(type(payload["nextCursor"]), int)
        self.assertIs(type(payload["receiptsCount"]), int)
        self.assertIs(type(payload["coverage"]["start"]), int)
        self.assertIs(type(payload["coverage"]["end"]), int)

    def test_audit_agrees_with_receipts_endpoint(self) -> None:
        self.seed(3)
        self.register("peer-a", 0)
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        self.acknowledge("peer-a", "ack-2", 3, [identity("r2", "o2")])
        status, audit_payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        status, receipts_payload = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(audit_payload["receipts"], receipts_payload["receipts"])
        self.assertEqual(audit_payload["digest"], receipts_payload["digest"])
        self.assertEqual(audit_payload["receiptsCount"], receipts_payload["receiptsCount"])
        self.assertEqual(audit_payload["coverage"], {"start": 0, "end": 3})
        self.assertEqual(audit_payload["conclusion"], "ok")

    def test_paging_walk_is_stable_and_resumes(self) -> None:
        self.seed(5)
        self.register("peer-a", 0)
        for i in range(5):
            self.acknowledge("peer-a", f"ack-{i}", i + 1, [identity(f"r{i}", f"o{i}")])
        seen: list[dict] = []
        after = 0
        conclusions: set[str] = set()
        coverages: list[dict] = []
        digest = None
        while True:
            status, payload = self.audit("peer-a", f"?after={after}&limit=2")
            self.assertEqual(status, 200)
            seen.extend(payload["receipts"])
            conclusions.add(payload["conclusion"])
            coverages.append(payload["coverage"])
            if digest is None:
                digest = payload["digest"]
            else:
                self.assertEqual(payload["digest"], digest)
            self.assertEqual(payload["anomalies"], [])
            self.assertEqual(payload["receiptsCount"], 5)
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual([r["ackId"] for r in seen], [f"ack-{i}" for i in range(5)])
        self.assertEqual(conclusions, {"ok"})
        self.assertEqual(coverages, [{"start": 0, "end": 5}] * 3)
        self.assertEqual(after, 5)
        # Resume straight onto the empty tail.
        status, tail = self.audit("peer-a", "?after=5&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["nextCursor"], 5)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["coverage"], {"start": 0, "end": 5})
        self.assertEqual(tail["conclusion"], "ok")

    def test_empty_confirmation_segment_keeps_the_chain_ok(self) -> None:
        self.seed(2)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-0", 0, [])
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        self.acknowledge("peer-a", "ack-2", 2, [])
        status, payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 2})
        self.assertEqual(payload["conclusion"], "ok")
        self.assertEqual(payload["anomalies"], [])
        self.assertEqual(payload["receiptsCount"], 3)

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
            # after past the receipt count (one committed receipt).
            "?after=2&limit=10",
        ]
        for query in bad_queries:
            status, payload = self.audit("peer-a", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        status, payload = self.audit("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receiptsCount"], 1)
        self.assertEqual(payload["conclusion"], "ok")

    def test_route_shapes_are_404_before_query_checks(self) -> None:
        self.register("peer-a", 0)
        not_found_paths = [
            "/v1/sync/peers//receipts/audit?after=0&limit=1",
            "/v1/sync/peers//receipts/audit?after=x",
            "/v1/sync/peers//receipts/audit?limit=0",
            "/v1/sync/peers/peer-a/receipts/audit/extra",
            "/v1/sync/peers/peer-a/receipts/audit/extra?after=0&limit=1",
            "/v1/sync/peers/peer-a/receipts/audit/",
            "/v1/sync/peers/peer-a/receipts/audit/?after=0&limit=1",
            "/v1/sync/peers/peer-a",
            "/v1/sync/peers",
            "/v1/sync/peers/peer-a/not-audit",
            "/v1/sync/audit/peer-a",
            "/v1/sync/peers/peer-a/receipts/not-audit",
            "/v2/sync/peers/peer-a/receipts/audit",
        ]
        for path in not_found_paths:
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_audit_route_does_not_shadow_sibling_routes(self) -> None:
        self.seed(1)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in payload["receipts"]], ["ack-1"])
        status, payload = self.request(
            "GET", "/v1/sync/peers/peer-a/operations?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])

    def test_unknown_peer_with_well_formed_query_is_404(self) -> None:
        for query in ("", "?after=0", "?limit=1", "?after=999"):
            status, payload = self.audit("nope", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        for query in ("?after=0&limit=1", "?after=999&limit=10"):
            status, payload = self.audit("nope", query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(1)
        self.request("POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0})
        self.request(
            "POST",
            "/v1/sync/peers/peer%20one/acknowledge",
            {"ackId": "ack-1", "cursor": 1, "operations": [identity("r0", "o0")]},
        )
        status, payload = self.audit("peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receipts"][0]["peerId"], "peer one")
        self.assertEqual(payload["coverage"], {"start": 0, "end": 1})
        self.assertEqual(payload["conclusion"], "ok")

    def test_post_to_audit_route_is_404(self) -> None:
        status, payload = self.request(
            "POST", "/v1/sync/peers/peer-a/receipts/audit", {"cursor": 0}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only(self) -> None:
        self.seed(2)
        self.register("peer-a", 0)
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        status, metrics_before = self.request("GET", "/v1/metrics")
        status, sync_before = self.request("GET", "/v1/sync/operations")
        status, checkpoint_before = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        status, receipts_before = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        for query in ("?after=0&limit=1", "?after=1&limit=1",
                      "?after=0&limit=100"):
            status, _ = self.audit("peer-a", query)
            self.assertEqual(status, 200)
        # Invalid queries and unknown peers must not mutate anything either.
        for path in (
            "/v1/sync/peers/peer-a/receipts/audit?after=99&limit=1",
            "/v1/sync/peers/nope/receipts/audit?after=0&limit=1",
        ):
            status, _ = self.request("GET", path)
            self.assertIn(status, (400, 404))
        status, metrics_after = self.request("GET", "/v1/metrics")
        status, sync_after = self.request("GET", "/v1/sync/operations")
        status, checkpoint_after = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        status, receipts_after = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(sync_after, sync_before)
        self.assertEqual(checkpoint_after, checkpoint_before)
        self.assertEqual(receipts_after, receipts_before)

    def test_concurrent_commits_and_audit_reads_stay_consistent(self) -> None:
        errors: list[BaseException] = []
        peer_count = 6
        log_count = peer_count * 3

        def worker(index: int) -> None:
            try:
                peer = f"peer-{index}"
                self.register(peer, 0)
                # Each peer independently consumes the whole shared log one
                # record at a time, learning the next identity from the
                # pickup stream anchored at its own checkpoint.
                while True:
                    status, checkpoint = self.request(
                        "GET", f"/v1/sync/peers/{peer}/checkpoint"
                    )
                    assert status == 200
                    cursor = checkpoint["cursor"]
                    if cursor >= log_count:
                        return
                    status, page = self.request(
                        "GET", f"/v1/sync/peers/{peer}/operations?after=0&limit=1"
                    )
                    assert status == 200 and page["operations"]
                    entry = page["operations"][0]
                    next_identity = {
                        "replicaId": entry["replicaId"],
                        "operationId": entry["operation"]["operationId"],
                    }
                    status, payload = self.request(
                        "POST",
                        f"/v1/sync/peers/{peer}/acknowledge",
                        {
                            "ackId": f"ack-{peer}-{cursor}",
                            "cursor": cursor + 1,
                            "operations": [next_identity],
                        },
                    )
                    assert status == 201, (status, payload)
            except BaseException as exc:  # reported below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(120):
                    for index in range(peer_count):
                        peer = f"peer-{index}"
                        after = 0
                        while True:
                            status, payload = self.request(
                                "GET",
                                f"/v1/sync/peers/{peer}/receipts/audit"
                                f"?after={after}&limit=2",
                            )
                            if status == 404:
                                # The worker for this peer may not have
                                # registered its checkpoint yet; retry later.
                                break
                            assert status == 200
                            page = payload["receipts"]
                            assert payload["nextCursor"] == after + len(page)
                            # Snapshot invariants: the digest, coverage,
                            # conclusion, and anomalies describe one commit.
                            # Every receipt confirmed one record from
                            # cursor 0, so the end cursor is the receipt
                            # count and the chain is always intact.
                            assert payload["coverage"]["start"] == 0
                            assert payload["coverage"]["end"] == payload["receiptsCount"]
                            assert payload["conclusion"] == "ok"
                            assert payload["anomalies"] == []
                            assert isinstance(payload["digest"], str)
                            after = payload["nextCursor"]
                            if not payload["hasMore"]:
                                break
            except BaseException as exc:  # reported below
                errors.append(exc)

        # Commit the shared log first (18 records, all peers start at 0).
        for index in range(peer_count):
            for suffix in ("a", "b", "c"):
                replica = f"r{index}{suffix}"
                self.post_operation(
                    replica,
                    operation(f"o{index}{suffix}", "k", suffix, {replica: 1}),
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(peer_count)]
        threads.append(threading.Thread(target=reader))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])

        for index in range(peer_count):
            status, payload = self.audit(f"peer-{index}")
            self.assertEqual(status, 200)
            self.assertEqual(payload["receiptsCount"], log_count)
            self.assertEqual(payload["coverage"], {"start": 0, "end": log_count})
            self.assertEqual(payload["conclusion"], "ok")


class AuthenticatedReceiptsAuditTests(unittest.TestCase):
    """Bearer authentication on the audit route; health stays anonymous."""

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

    def raw_request(
        self, path: str, auth_headers: list[tuple[str, str]] | None
    ) -> tuple[int, dict, bytes]:
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
        conn.close()
        self.assertEqual(
            payload,
            {
                "receipts": [],
                "nextCursor": 0,
                "hasMore": False,
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "receiptsCount": 0,
                "coverage": {"start": 0, "end": 0},
                "conclusion": "ok",
                "anomalies": [],
            },
        )


class PersistentReceiptsAuditHttpTests(unittest.TestCase):
    """Audit over real HTTP with a data file: recovery and no temp files."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"

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
        for i in range(4):
            self.request(
                server,
                "POST",
                f"/v1/replicas/r{i}/operations",
                operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}),
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
            {
                "ackId": "ack-2",
                "cursor": 4,
                "operations": [identity("r2", "o2"), identity("r3", "o3")],
            },
        )

    def test_audit_survives_restart(self) -> None:
        server = self.start_server()
        self.commit_receipts(server)
        status, before = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in before["receipts"]], ["ack-1"])
        self.assertEqual((before["nextCursor"], before["hasMore"]), (1, True))
        self.assertEqual(before["coverage"], {"start": 0, "end": 4})
        self.assertEqual(before["conclusion"], "ok")
        self.assertEqual(before["anomalies"], [])
        self.assertEqual(before["receiptsCount"], 2)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=1&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in after["receipts"]], ["ack-2"])
        self.assertEqual((after["nextCursor"], after["hasMore"]), (2, False))
        self.assertEqual(after["coverage"], before["coverage"])
        self.assertEqual(after["conclusion"], before["conclusion"])
        self.assertEqual(after["anomalies"], before["anomalies"])
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["receiptsCount"], 2)
        status, full = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in full["receipts"]], ["ack-1", "ack-2"])
        self.assertEqual(full["conclusion"], "ok")
        # The error boundary survives recovery as well.
        status, payload = self.request(
            server, "GET", "/v1/sync/peers/peer-a/receipts/audit?after=3&limit=10"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_queries_create_no_temp_files(self) -> None:
        server = self.start_server()
        self.commit_receipts(server)
        before = self.data_file.read_bytes()
        for query in ("?after=0&limit=1", "?after=1&limit=1", "?after=0&limit=100"):
            status, _ = self.request(
                server, "GET", f"/v1/sync/peers/peer-a/receipts/audit{query}"
            )
            self.assertEqual(status, 200)
        self.assertEqual(self.data_file.read_bytes(), before)
        leftovers = [
            p.name for p in self.data_file.parent.iterdir() if p.name != self.data_file.name
        ]
        self.assertEqual(leftovers, [])


class CommandLineReceiptsAuditTests(unittest.TestCase):
    """The real ``python -m`` entry point serves the audit route."""

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
            conn.request(
                "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=1"
            )
            response = conn.getresponse()
            raw = response.read()
            self.assertEqual(response.status, 200)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(raw.count(b"\n"), 1)
            payload = json.loads(raw.decode("utf-8"))
            conn.close()
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
            self.assertEqual(payload["coverage"], {"start": 0, "end": 1})
            self.assertEqual(payload["conclusion"], "ok")
            self.assertEqual(payload["anomalies"], [])
            self.assertEqual(
                payload["digest"],
                digest_of(
                    '[{"peerId":"peer-a","ackId":"ack-1","cursor":1,'
                    '"operations":[{"replicaId":"r1","operationId":"o1"}]}]'
                ),
            )
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
