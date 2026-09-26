"""Tests for the read-only replication repair-suggestions entry point::

    GET /v1/replication/repairs?after=N&limit=N

The endpoint audits every registered sender-side peer's whole
confirmation chain against the shared accepted log and returns one page
of read-only repair suggestions — one per reported gap, overlap,
identity mismatch, or cursor regression — each locating the affected log
interval or position and naming a suggested action (``resend``,
``deduplicate``, or ``correct_identity``) and the target boundary,
without executing anything. Paging trims only the suggestion list: the
summary, the anomaly counts, the overall coverage, and the conclusion
are always computed from every committed receipt and the whole accepted
log on one committed snapshot.

The tests cover the query parser (required, non-repeated, ASCII-decimal
``after``/``limit``; ``limit`` between 1 and 100), the store's
suggestion derivation, ordering, paging, coverage, conclusion, and
recovery semantics, the HTTP request precedence chain (401
authentication with a Bearer challenge, 403 in scope mode without one,
404 path shape, 400 query validation including an ``after`` past the
suggestion count), the compact ordered response body with its single
trailing newline, and the read-only guarantee.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot and recovery
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import http.client
from http import HTTPStatus
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_replication_repairs_query,
)

PLAN_FIELDS = [
    "suggestions",
    "nextCursor",
    "hasMore",
    "summary",
    "anomalies",
    "coverage",
    "conclusion",
]
SUMMARY_FIELDS = ["peers", "receipts", "suggestions"]
ANOMALY_FIELDS = ["gaps", "overlaps", "identityMismatches", "cursorRegressions"]
INTERVAL_FIELDS = ["peer", "ackId", "kind", "action", "location", "length", "target"]
MISMATCH_FIELDS = [
    "peer",
    "ackId",
    "kind",
    "action",
    "location",
    "expected",
    "observed",
    "target",
]

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def zero_anomalies() -> dict:
    return {"gaps": 0, "overlaps": 0, "identityMismatches": 0, "cursorRegressions": 0}


class ParseReplicationRepairsQueryTests(unittest.TestCase):
    def test_accepts_required_after_and_limit(self) -> None:
        self.assertEqual(parse_replication_repairs_query("after=0&limit=1"), (0, 1))
        self.assertEqual(parse_replication_repairs_query("limit=100&after=42"), (42, 100))
        self.assertEqual(parse_replication_repairs_query("after=007&limit=09"), (7, 9))

    def test_rejects_missing_repeated_unknown_and_blank(self) -> None:
        bad = [
            "",
            "after=0",
            "limit=1",
            "after=0&limit=1&after=2",
            "after=0&limit=1&limit=2",
            "after=0&limit=1&x=1",
            "x=1&after=0&limit=1",
            "after=&limit=1",
            "after=0&limit=",
            "after&limit=1",
            "after=0&limit=1&=",
            "=1&after=0&limit=1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_repairs_query(query))

    def test_rejects_non_ascii_decimal_signed_and_out_of_range(self) -> None:
        bad = [
            "after=-1&limit=1",
            "after=+1&limit=1",
            "after=1.0&limit=1",
            "after= 1&limit=1",
            "after=1 &limit=1",
            "after=١&limit=1",  # Arabic-Indic digit
            "after=１&limit=1",  # fullwidth digit
            "after=0&limit=0",
            "after=0&limit=101",
            "after=0&limit=-1",
            "after=0&limit=1.5",
            "after=0&limit=١",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_repairs_query(query))


class ReplicationRepairsStoreTests(unittest.TestCase):
    """Suggestion derivation, ordering, paging, and recovery semantics."""

    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int) -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            self.assertIn(status, (200, 201))

    def ack(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, error = self.store.acknowledge_operations(
            peer, ack_id, cursor, operations
        )
        self.assertEqual(status, 201, error)

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        # Direct seeding lets the tests construct chains the acknowledge
        # endpoint itself would reject (gaps, overlaps, wrong identities).
        self.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def plan(self, after: int = 0, limit: int = 100) -> dict:
        status, payload = self.store.get_replication_repairs(after, limit)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), PLAN_FIELDS)
        return payload

    def test_empty_registry_reports_an_empty_plan(self) -> None:
        payload = self.plan()
        self.assertEqual(
            payload,
            {
                "suggestions": [],
                "nextCursor": 0,
                "hasMore": False,
                "summary": {"peers": 0, "receipts": 0, "suggestions": 0},
                "anomalies": zero_anomalies(),
                "coverage": {"start": 0, "end": 0},
                "conclusion": "ok",
            },
        )
        self.assertEqual(list(payload["summary"]), SUMMARY_FIELDS)
        self.assertEqual(list(payload["anomalies"]), ANOMALY_FIELDS)

    def test_seamless_chains_report_no_suggestions_and_ok(self) -> None:
        self.seed_operations(4)
        self.store.save_checkpoint("peer-c", 0)
        self.store.save_checkpoint("peer-a", 0)
        self.store.save_checkpoint("peer-b", 0)
        self.ack("peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        self.ack("peer-c", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        self.ack("peer-c", "ack-2", 4, [identity("r2", "o2"), identity("r3", "o3")])
        # peer-b is registered but has no receipts at all.
        payload = self.plan()
        self.assertEqual(payload["suggestions"], [])
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(
            payload["summary"], {"peers": 3, "receipts": 3, "suggestions": 0}
        )
        self.assertEqual(payload["anomalies"], zero_anomalies())
        # The combined coverage spans the earliest start through the
        # latest end; the receipt-less peer contributes nothing.
        self.assertEqual(payload["coverage"], {"start": 0, "end": 4})
        self.assertEqual(payload["conclusion"], "ok")

    def test_after_equal_to_the_count_is_a_valid_empty_page(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 2)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r9", "bogus"), identity("r1", "o1")],
        )
        payload = self.plan()
        self.assertEqual(payload["summary"]["suggestions"], 1)
        tail = self.plan(after=1, limit=100)
        self.assertEqual(tail["suggestions"], [])
        self.assertEqual(tail["nextCursor"], 1)
        self.assertIs(tail["hasMore"], False)
        # Summary and conclusion are independent of the empty page.
        self.assertEqual(tail["summary"]["suggestions"], 1)
        self.assertEqual(tail["conclusion"], "broken")

    def test_after_past_the_count_is_rejected(self) -> None:
        empty = StateStore()
        status, _ = empty.get_replication_repairs(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        with self.assertRaises(ValueError):
            empty.get_replication_repairs(1, 100)
        self.store.save_checkpoint("peer-a", 0)
        with self.assertRaises(ValueError):
            self.store.get_replication_repairs(1, 100)

    def test_gap_suggests_resend_with_boundaries_and_length(self) -> None:
        self.seed_operations(4)
        self.store.save_checkpoint("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        # [1,2): accepted record 1 is confirmed by no receipt.
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        payload = self.plan()
        self.assertEqual(len(payload["suggestions"]), 1)
        suggestion = payload["suggestions"][0]
        self.assertEqual(list(suggestion), INTERVAL_FIELDS)
        self.assertEqual(
            suggestion,
            {
                "peer": "peer-a",
                "ackId": "ack-2",
                "kind": "gap",
                "action": "resend",
                "location": {"start": 1, "end": 2},
                "length": 1,
                "target": {"start": 1, "end": 2},
            },
        )
        self.assertEqual(payload["anomalies"]["gaps"], 1)
        self.assertEqual(payload["coverage"], {"start": 0, "end": 3})
        self.assertEqual(payload["conclusion"], "broken")

    def test_larger_gap_reports_its_full_length(self) -> None:
        self.seed_operations(6)
        self.store.save_checkpoint("peer-a", 6)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        # [1,5): four accepted records unconfirmed.
        self.seed_receipt("peer-a", "ack-2", 6, [identity("r5", "o5")])
        suggestion = self.plan()["suggestions"][0]
        self.assertEqual(suggestion["kind"], "gap")
        self.assertEqual(suggestion["location"], {"start": 1, "end": 5})
        self.assertEqual(suggestion["target"], {"start": 1, "end": 5})
        self.assertEqual(suggestion["length"], 4)

    def test_overlap_suggests_deduplicate_with_boundaries_and_length(self) -> None:
        self.seed_operations(4)
        self.store.save_checkpoint("peer-a", 4)
        self.seed_receipt(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        # [1,3): record 1 is confirmed twice.
        self.seed_receipt(
            "peer-a", "ack-2", 3, [identity("r1", "o1"), identity("r2", "o2")]
        )
        suggestion = self.plan()["suggestions"][0]
        self.assertEqual(list(suggestion), INTERVAL_FIELDS)
        self.assertEqual(suggestion["kind"], "overlap")
        self.assertEqual(suggestion["action"], "deduplicate")
        self.assertEqual(suggestion["location"], {"start": 1, "end": 2})
        self.assertEqual(suggestion["target"], {"start": 1, "end": 2})
        self.assertEqual(suggestion["length"], 1)

    def test_identity_mismatch_reports_expected_and_observed(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 2)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        suggestion = self.plan()["suggestions"][0]
        self.assertEqual(list(suggestion), MISMATCH_FIELDS)
        self.assertEqual(
            suggestion,
            {
                "peer": "peer-a",
                "ackId": "ack-1",
                "kind": "identityMismatch",
                "action": "correct_identity",
                "location": {"position": 1},
                "expected": {"replicaId": "r1", "operationId": "o1"},
                "observed": {"replicaId": "r9", "operationId": "bogus"},
                "target": {"position": 1},
            },
        )
        self.assertEqual(self.plan()["anomalies"]["identityMismatches"], 1)

    def test_mismatch_outside_the_log_reports_null_expected(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 2)
        # Confirms positions 2 and 3, but the log ends at position 1.
        self.seed_receipt(
            "peer-a",
            "ack-1",
            4,
            [identity("r2", "o2"), identity("r3", "o3")],
        )
        suggestions = self.plan()["suggestions"]
        self.assertEqual([item["location"] for item in suggestions], [
            {"position": 2},
            {"position": 3},
        ])
        for item in suggestions:
            self.assertIsNone(item["expected"])
            self.assertEqual(item["action"], "correct_identity")
            self.assertEqual(item["target"], item["location"])

    def test_cursor_regression_suggests_resend_to_the_prior_boundary(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 3)
        self.seed_receipt(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        # An empty segment at cursor 1 both overlaps [1,2) and regresses
        # the cursor 2 -> 1; each anomaly yields its own suggestion.
        self.seed_receipt("peer-a", "ack-2", 1, [])
        payload = self.plan()
        kinds = {(item["kind"], item["action"]) for item in payload["suggestions"]}
        self.assertEqual(
            kinds,
            {("overlap", "deduplicate"), ("cursorRegression", "resend")},
        )
        for suggestion in payload["suggestions"]:
            self.assertEqual(suggestion["ackId"], "ack-2")
            self.assertEqual(suggestion["location"], {"start": 1, "end": 2})
            self.assertEqual(suggestion["target"], {"start": 1, "end": 2})
            self.assertEqual(suggestion["length"], 1)
        self.assertEqual(payload["anomalies"]["cursorRegressions"], 1)
        self.assertEqual(payload["anomalies"]["overlaps"], 1)

    def test_suggestions_order_by_peer_then_chain_creation_order(self) -> None:
        self.seed_operations(6)
        self.store.save_checkpoint("peer-c", 6)
        self.store.save_checkpoint("peer-a", 6)
        # peer-a: gap on its second receipt, then a mismatch on its third.
        self.seed_receipt("peer-a", "ack-a1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-a2", 3, [identity("r2", "o2")])
        self.seed_receipt(
            "peer-a",
            "ack-a3",
            5,
            [identity("r3", "o3"), identity("r9", "bogus")],
        )
        # peer-c: a gap on its first chained receipt.
        self.seed_receipt("peer-c", "ack-c1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-c", "ack-c2", 3, [identity("r2", "o2")])
        payload = self.plan()
        ordered = [
            (item["peer"], item["ackId"], item["kind"])
            for item in payload["suggestions"]
        ]
        # peer-a precedes peer-c; within peer-a the boundary anomaly on
        # ack-a2 precedes the identity mismatch on the later ack-a3.
        self.assertEqual(
            ordered,
            [
                ("peer-a", "ack-a2", "gap"),
                ("peer-a", "ack-a3", "identityMismatch"),
                ("peer-c", "ack-c2", "gap"),
            ],
        )
        self.assertEqual(payload["summary"]["suggestions"], 3)

    def test_boundary_anomaly_precedes_a_mismatch_on_the_same_receipt(self) -> None:
        self.seed_operations(5)
        self.store.save_checkpoint("peer-a", 5)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        # Starts past the previous end (a gap) and names a wrong identity
        # at the position it does confirm.
        self.seed_receipt("peer-a", "ack-2", 4, [identity("r9", "bogus")])
        payload = self.plan()
        kinds = [item["kind"] for item in payload["suggestions"]]
        self.assertEqual(kinds, ["gap", "identityMismatch"])
        self.assertEqual(
            payload["suggestions"][0]["location"], {"start": 1, "end": 3}
        )
        self.assertEqual(
            payload["suggestions"][1]["location"], {"position": 3}
        )

    def test_paging_trims_only_suggestions_not_the_summary(self) -> None:
        self.seed_operations(6)
        self.store.save_checkpoint("peer-a", 6)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        # Gap [3,4) plus a wrong identity at the position it confirms.
        self.seed_receipt("peer-a", "ack-3", 5, [identity("r9", "bogus")])
        first = self.plan(after=0, limit=1)
        self.assertEqual(len(first["suggestions"]), 1)
        self.assertEqual(first["suggestions"][0]["kind"], "gap")
        self.assertEqual(first["nextCursor"], 1)
        self.assertIs(first["hasMore"], True)
        # The summary, anomaly counts, coverage, and conclusion always
        # describe the complete chains — two gaps and one mismatch.
        self.assertEqual(first["summary"], {"peers": 1, "receipts": 3, "suggestions": 3})
        self.assertEqual(
            first["anomalies"],
            {"gaps": 2, "overlaps": 0, "identityMismatches": 1, "cursorRegressions": 0},
        )
        self.assertEqual(first["coverage"], {"start": 0, "end": 5})
        self.assertEqual(first["conclusion"], "broken")
        second = self.plan(after=1, limit=1)
        self.assertEqual(second["suggestions"][0]["kind"], "gap")
        self.assertEqual(second["nextCursor"], 2)
        self.assertIs(second["hasMore"], True)
        third = self.plan(after=2, limit=1)
        self.assertEqual(third["suggestions"][0]["kind"], "identityMismatch")
        self.assertEqual(third["nextCursor"], 3)
        self.assertIs(third["hasMore"], False)
        for page in (first, second, third):
            self.assertEqual(page["summary"], first["summary"])
            self.assertEqual(page["anomalies"], first["anomalies"])
            self.assertEqual(page["coverage"], first["coverage"])
            self.assertEqual(page["conclusion"], "broken")

    def test_query_is_repeatable_and_read_only(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 3)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            3,
            [identity("r0", "o0"), identity("r9", "bogus"), identity("r2", "o2")],
        )
        before = self.plan()
        metrics_before = self.store.get_metrics()
        checkpoint_before = self.store.get_checkpoint("peer-a")
        receipts_before = self.store.get_peer_receipts("peer-a", 0, 100)
        for _ in range(3):
            self.assertEqual(self.plan(), before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_checkpoint("peer-a"), checkpoint_before)
        self.assertEqual(
            self.store.get_peer_receipts("peer-a", 0, 100), receipts_before
        )

    def test_recovery_reproduces_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            for index in range(4):
                store.apply_operation(
                    f"r{index}",
                    operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
                )
            store.save_checkpoint("peer-a", 4)
            store.save_checkpoint("peer-b", 0)
            store._acks[("peer-a", "ack-1")] = {
                "cursor": 1,
                "operations": [identity("r0", "o0")],
            }
            store._acks[("peer-a", "ack-2")] = {
                "cursor": 3,
                "operations": [identity("r2", "o2")],
            }
            status, error = store.acknowledge_operations(
                "peer-b",
                "ack-1",
                2,
                [identity("r0", "o0"), identity("r1", "o1")],
            )
            self.assertEqual(status, 201, error)
            before = store.get_replication_repairs(0, 100)
            recovered = StateStore(data_file=data_file)
            after = recovered.get_replication_repairs(0, 100)
            self.assertEqual(after, before)
            self.assertEqual(after[1]["conclusion"], "broken")
            self.assertEqual(after[1]["summary"]["suggestions"], 1)


class ReplicationRepairsHttpTests(unittest.TestCase):
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

    def raw_request(
        self, method: str, path: str, body: object = None, headers: dict | None = None
    ) -> tuple[int, object, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=request_headers)
        elif isinstance(body, (bytes, str)):
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers=request_headers,
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def repairs_get(self, query: str) -> tuple[int, object, bytes, dict]:
        return self.raw_request("GET", f"/v1/replication/repairs{query}")

    def seed(self, count: int = 4) -> None:
        for index in range(count):
            status, _ = self.request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int = 0) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status in (200, 201), status

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.server.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def test_empty_plan_is_compact_ordered_json_with_one_newline(self) -> None:
        status, payload, raw, headers = self.repairs_get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), PLAN_FIELDS)
        self.assertEqual(
            payload,
            {
                "suggestions": [],
                "nextCursor": 0,
                "hasMore": False,
                "summary": {"peers": 0, "receipts": 0, "suggestions": 0},
                "anomalies": zero_anomalies(),
                "coverage": {"start": 0, "end": 0},
                "conclusion": "ok",
            },
        )
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(
            raw[:-1], json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_broken_chain_plan_body_keeps_field_order_and_integer_numbers(self) -> None:
        self.seed(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        status, payload, raw, headers = self.repairs_get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), PLAN_FIELDS)
        self.assertEqual(list(payload["summary"]), SUMMARY_FIELDS)
        self.assertEqual(list(payload["anomalies"]), ANOMALY_FIELDS)
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["summary"]["suggestions"], 1)
        suggestion = payload["suggestions"][0]
        self.assertEqual(list(suggestion), INTERVAL_FIELDS)
        self.assertEqual(suggestion["kind"], "gap")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(
            raw[:-1], json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertIsInstance(payload["nextCursor"], int)
        self.assertNotIsInstance(payload["nextCursor"], bool)
        for field in SUMMARY_FIELDS:
            self.assertIsInstance(payload["summary"][field], int)
            self.assertNotIsInstance(payload["summary"][field], bool)
        for field in ANOMALY_FIELDS:
            self.assertIsInstance(payload["anomalies"][field], int)
            self.assertNotIsInstance(payload["anomalies"][field], bool)
        self.assertIsInstance(suggestion["length"], int)
        self.assertNotIsInstance(suggestion["length"], bool)
        for field in ("start", "end"):
            self.assertIsInstance(suggestion["location"][field], int)
            self.assertIsInstance(suggestion["target"][field], int)

    def test_bad_queries_are_400(self) -> None:
        self.register("peer-a")
        bad_queries = [
            "",
            "?",
            "?after=0",
            "?limit=1",
            "?after=0&limit=1&after=2",
            "?after=0&limit=1&limit=2",
            "?after=0&limit=1&x=1",
            "?x=1",
            "?after=&limit=1",
            "?after=0&limit=",
            "?after=-1&limit=1",
            "?after=+1&limit=1",
            "?after=1.0&limit=1",
            "?after=%201&limit=1",
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=-5",
            "?after=0&limit=%D9%A1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.repairs_get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_the_suggestion_count_is_400(self) -> None:
        status, _, _, _ = self.repairs_get("?after=1&limit=1")
        self.assertEqual(status, 400)
        # Equal to the count is a stable empty page, even when it is 0.
        status, payload, _, _ = self.repairs_get("?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["suggestions"], [])

    def test_route_shape_mismatches_are_404(self) -> None:
        self.register("peer-a")
        bad_paths = [
            "/v1/replication",
            "/v1/replication/repairs/",
            "/v1/replication/repairs/extra",
            "/v1/replication/repair",
            "/v1/replication/unknown",
            "/v1/repairs",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=0&limit=1"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs/extra?after=%ZZ&limit=x"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_repairs_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/repairs", {"cursor": 0}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only(self) -> None:
        self.seed(2)
        self.register("peer-a")
        status, _ = self.request(
            "POST",
            "/v1/sync/peers/peer-a/acknowledge",
            {
                "ackId": "ack-1",
                "cursor": 1,
                "operations": [identity("r0", "o0")],
            },
        )
        self.assertEqual(status, 201)
        _, metrics_before = self.request("GET", "/v1/metrics")
        _, checkpoint_before = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        _, receipts_before = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        _, first, _, _ = self.repairs_get("?after=0&limit=100")
        self.repairs_get("?after=0&limit=1")
        self.repairs_get("?after=5&limit=100")
        self.repairs_get("?after=%ZZ")
        _, second, _, _ = self.repairs_get("?after=0&limit=100")
        self.assertEqual(second, first)
        _, metrics_after = self.request("GET", "/v1/metrics")
        _, checkpoint_after = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        _, receipts_after = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(checkpoint_after, checkpoint_before)
        self.assertEqual(receipts_after, receipts_before)


class ReplicationRepairsAuthTests(unittest.TestCase):
    """The repairs endpoint authenticates like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-repairs-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret-token"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]

        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(SCOPE_POLICY, handle)
        cls.scope_server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
        )
        cls.scope_thread = threading.Thread(
            target=cls.scope_server.serve_forever, daemon=True
        )
        cls.scope_thread.start()
        cls.scope_port = cls.scope_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.single_server.shutdown()
        cls.single_server.server_close()
        cls.scope_server.shutdown()
        cls.scope_server.server_close()
        cls.single_thread.join(timeout=5)
        cls.scope_thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def get(
        self, port: int, path: str, headers: list[tuple[str, str]] | None = None
    ) -> tuple[int, object, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers=dict(headers or []))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, response_headers

    def test_single_token_missing_bad_or_wrong_is_401(self) -> None:
        path = "/v1/replication/repairs?after=0&limit=1"
        status, payload, headers = self.get(self.single_port, path)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, path, [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, path, [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_single_token_valid_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.single_port,
            "/v1/replication/repairs?after=0&limit=1",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["suggestions"], [])
        self.assertEqual(payload["conclusion"], "ok")

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            "/v1/replication/repairs?after=0&limit=1",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, headers = self.get(
            self.scope_port,
            "/v1/replication/repairs?after=%ZZ",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            "/v1/replication/repairs?after=0&limit=1",
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["peers"], 0)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
