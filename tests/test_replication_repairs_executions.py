"""Tests for the read-only replication-repair execution audit::

    GET /v1/replication/repairs/executions?after=N&limit=N

The endpoint pages the committed history of conditional repair
executions — one record per successful
``POST /v1/replication/repairs/apply``; an identical replay answers from
the committed binding and appends no record. The page lists executions in
ascending ``peerId`` order and, within a peer, in creation (first-commit)
order, while ``digest`` and ``executionsCount`` always cover the whole
history in pure creation order. Every response also carries an
independent ``verification`` conclusion over the complete history
(duplicate bindings, fixed action order, suggestion/result boundaries,
checkpoint advancement, and stored-record legality).

The tests cover the query parser, the independent integrity scan, the
store's ordering/paging/snapshot/recovery semantics, the HTTP request
precedence chain (404 path shape, 401 authentication with a Bearer
challenge, 403 in scope mode without one, 400 query validation including
an ``after`` past the execution count), the compact ordered single-newline
response body with its explicit Content-Length, restart consistency, and
the strict read-only guarantee. Only the Python standard library is used.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _repair_executions_digest_input,
    _repair_executions_verification_locked,
    _repair_receipts_digest,
    load_scope_policy,
    parse_replication_repair_executions_query,
)

EXECUTIONS_PATH = "/v1/replication/repairs/executions"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

RESPONSE_FIELDS = [
    "executions",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "executionsCount",
    "verification",
]
EXECUTION_FIELDS = [
    "peerId",
    "ackId",
    "expectedCheckpoint",
    "expectedReceipts",
    "suggestions",
    "results",
    "cursor",
]
VERIFICATION_FIELDS = [
    "status",
    "duplicateBindings",
    "outOfOrderActions",
    "boundaryViolations",
    "checkpointViolations",
    "recordViolations",
]
OK_VERIFICATION = {name: [] for name in VERIFICATION_FIELDS if name != "status"}
OK_VERIFICATION["status"] = "ok"
OK_VERIFICATION = {name: OK_VERIFICATION[name] for name in VERIFICATION_FIELDS}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def interval(action: str, ack_id: str, start: int, end: int) -> dict:
    return {
        "action": action,
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def gap_record(
    peer_id: str = "peer-a",
    ack_id: str = "exec-1",
    *,
    expected_checkpoint: int = 4,
    cursor: int = 4,
    digest: str = "a" * 64,
) -> dict:
    """One well-formed stored execution carrying a single resend action."""
    suggestion = interval("resend", "ack-2", 1, 2)
    return {
        "peerId": peer_id,
        "ackId": ack_id,
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": digest,
        "suggestions": [suggestion],
        "results": [{"action": "resend", "boundary": {"start": 1, "end": 2}}],
        "cursor": cursor,
    }


class ParseRepairExecutionsQueryTests(unittest.TestCase):
    def test_accepts_required_after_and_limit(self) -> None:
        self.assertEqual(
            parse_replication_repair_executions_query("after=0&limit=1"), (0, 1)
        )
        self.assertEqual(
            parse_replication_repair_executions_query("limit=100&after=42"), (42, 100)
        )
        self.assertEqual(
            parse_replication_repair_executions_query("after=007&limit=09"), (7, 9)
        )

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
                self.assertIsNone(parse_replication_repair_executions_query(query))

    def test_rejects_signed_decimal_non_ascii_and_out_of_range_limit(self) -> None:
        bad = [
            "after=-1&limit=1",
            "after=0&limit=-1",
            "after=+0&limit=1",
            "after=0&limit=+1",
            "after=1.0&limit=1",
            "after=0&limit=1.0",
            "after=%200&limit=1",
            "after=0&limit=1%20",
            "after=%C2%B2&limit=1",
            "after=0&limit=%D9%A1",
            "after=0&limit=0",
            "after=0&limit=101",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_repair_executions_query(query))


class RepairExecutionsVerificationTests(unittest.TestCase):
    def test_empty_history_is_intact(self) -> None:
        self.assertEqual(
            _repair_executions_verification_locked([], {}, 0), OK_VERIFICATION
        )

    def test_well_formed_history_is_ok(self) -> None:
        entries = [gap_record(), gap_record("peer-b", "exec-b")]
        verdict = _repair_executions_verification_locked(
            entries, {"peer-a": 4, "peer-b": 4}, 4
        )
        self.assertEqual(verdict, OK_VERIFICATION)

    def test_duplicate_binding_marks_only_the_later_occurrence(self) -> None:
        entries = [gap_record(), gap_record("peer-a", "exec-1")]
        verdict = _repair_executions_verification_locked(entries, {"peer-a": 4}, 4)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["duplicateBindings"],
            [{"executionIndex": 1, "peerId": "peer-a", "ackId": "exec-1"}],
        )

    def test_out_of_order_actions_are_marked(self) -> None:
        record = gap_record()
        # correct_cursor (rank 3) before resend (rank 0) violates the
        # fixed processing order.
        record["suggestions"] = [
            interval("correct_cursor", "ack-9", 2, 2),
            interval("resend", "ack-2", 1, 2),
        ]
        record["results"] = [
            {"action": "correct_cursor", "boundary": {"start": 2, "end": 2}},
            {"action": "resend", "boundary": {"start": 1, "end": 2}},
        ]
        verdict = _repair_executions_verification_locked(
            [record], {"peer-a": 4}, 4
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["outOfOrderActions"],
            [{"executionIndex": 0, "peerId": "peer-a", "ackId": "exec-1"}],
        )

    def test_same_action_repeated_stays_in_order(self) -> None:
        record = gap_record()
        record["suggestions"] = [
            interval("resend", "ack-2", 1, 2),
            interval("resend", "ack-3", 2, 3),
        ]
        record["results"] = [
            {"action": "resend", "boundary": {"start": 1, "end": 2}},
            {"action": "resend", "boundary": {"start": 2, "end": 3}},
        ]
        verdict = _repair_executions_verification_locked(
            [record], {"peer-a": 4}, 4
        )
        self.assertEqual(verdict["status"], "ok")

    def test_result_boundary_disagreeing_with_target_is_marked(self) -> None:
        record = gap_record()
        record["results"][0]["boundary"] = {"start": 1, "end": 3}
        verdict = _repair_executions_verification_locked(
            [record], {"peer-a": 4}, 4
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["boundaryViolations"],
            [
                {
                    "executionIndex": 0,
                    "peerId": "peer-a",
                    "ackId": "exec-1",
                    "suggestionIndex": 0,
                    "expected": {"start": 1, "end": 2},
                    "observed": {"start": 1, "end": 3},
                }
            ],
        )

    def test_checkpoint_advancement_violations_are_marked(self) -> None:
        # The restored cursor must never precede the anchor checkpoint.
        record = gap_record(expected_checkpoint=4, cursor=2)
        verdict = _repair_executions_verification_locked([record], {"peer-a": 4}, 6)
        self.assertEqual(verdict["status"], "broken")
        marker = verdict["checkpointViolations"][0]
        self.assertEqual(marker["executionIndex"], 0)
        self.assertEqual(marker["peerId"], "peer-a")
        self.assertEqual(marker["ackId"], "exec-1")
        self.assertEqual(
            marker["expected"],
            {"checkpoint": 4, "registered": 4, "logLength": 6},
        )
        self.assertEqual(marker["observed"], {"cursor": 2})
        # A registered checkpoint that has not reached the restored
        # cursor is broken as well.
        record = gap_record(cursor=4)
        verdict = _repair_executions_verification_locked([record], {"peer-a": 3}, 6)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["checkpointViolations"][0]["expected"]["registered"], 3
        )
        # An unregistered peer is broken.
        verdict = _repair_executions_verification_locked([gap_record()], {}, 6)
        self.assertIsNone(
            verdict["checkpointViolations"][0]["expected"]["registered"]
        )
        # A cursor past the recovered log length is broken.
        record = gap_record(cursor=7)
        verdict = _repair_executions_verification_locked([record], {"peer-a": 7}, 6)
        self.assertEqual(verdict["status"], "broken")

    def test_record_violations_are_marked(self) -> None:
        def tamper(mutate) -> dict:
            record = gap_record()
            mutate(record)
            return record

        cases = [
            tamper(lambda r: r.update(peerId="")),
            tamper(lambda r: r.update(ackId="")),
            tamper(lambda r: r.update(expectedCheckpoint=-1)),
            tamper(lambda r: r.update(expectedCheckpoint=True)),
            tamper(lambda r: r.update(cursor="4")),
            tamper(lambda r: r.update(expectedReceipts="zz")),
            tamper(lambda r: r.update(suggestions=[])),
            tamper(lambda r: r.update(suggestions=[object()])),
            tamper(lambda r: r.update(results=[])),
            tamper(lambda r: r["results"][0].update(action="deduplicate")),
            tamper(
                lambda r: r["suggestions"].append(
                    interval("resend", "ack-3", 1, 2)
                )
            ),
            tamper(
                lambda r: r["suggestions"][0].update(
                    location={"start": 2, "end": 1}
                )
            ),
        ]
        for record in cases:
            with self.subTest(record=record):
                verdict = _repair_executions_verification_locked(
                    [record], {"peer-a": 4}, 4
                )
                self.assertEqual(verdict["status"], "broken")
                self.assertEqual(
                    verdict["recordViolations"],
                    [
                        {
                            "executionIndex": 0,
                            "peerId": record.get("peerId"),
                            "ackId": record.get("ackId"),
                        }
                    ],
                )

    def test_non_object_record_is_marked_with_null_binding(self) -> None:
        verdict = _repair_executions_verification_locked(["nope"], {}, 0)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["recordViolations"],
            [{"executionIndex": 0, "peerId": None, "ackId": None}],
        )

    def test_independent_anomaly_lists_combine(self) -> None:
        first = gap_record()
        second = gap_record(cursor=2)  # checkpoint advancement violation
        second["peerId"] = "peer-a"
        second["ackId"] = "exec-1"  # also a duplicate binding
        verdict = _repair_executions_verification_locked(
            [first, second], {"peer-a": 4}, 6
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["duplicateBindings"][0]["executionIndex"], 1)
        self.assertEqual(verdict["checkpointViolations"][0]["executionIndex"], 1)


class RepairExecutionsStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int, key: str = "k") -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", key, f"v{index}", {f"r{index}": 1}),
            )
            self.assertIs(status, HTTPStatus.CREATED)

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.store.save_checkpoint(peer, cursor)
        self.assertIs(status, HTTPStatus.OK)

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def advice(self, peer: str) -> list[dict]:
        status, payload = self.store.get_replication_repairs(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        return [item for item in payload["suggestions"] if item["peer"] == peer]

    def receipts_digest(self, peer: str) -> str:
        committed = [
            (ack_id, receipt)
            for (receipt_peer, ack_id), receipt in self.store._acks.items()
            if receipt_peer == peer
        ]
        return _repair_receipts_digest(peer, committed)

    @staticmethod
    def suggestion(item: dict) -> dict:
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        return suggestion

    def gap_repair(self, peer: str, ack_id: str, *, checkpoint: int | None = None):
        """Commit one resend-gap repair for a peer with the standard seed."""
        item = self.advice(peer)[0]
        if checkpoint is None:
            checkpoint = self.store.get_checkpoint(peer)[1]["cursor"]
        return self.store.apply_replication_repairs(
            peer,
            ack_id,
            checkpoint,
            self.receipts_digest(peer),
            [self.suggestion(item)],
        )

    def seed_gap(self, peer: str, *, first: str = "ack-1", second: str = "ack-2") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, first, 1, [identity("r0", "o0")])
        self.seed_receipt(peer, second, 3, [identity("r2", "o2")])


class RepairExecutionsStoreTests(RepairExecutionsStoreFixture):
    def audit(self, after: int = 0, limit: int = 100):
        return self.store.get_replication_repair_executions(after, limit)

    def test_empty_history_is_empty_page_hash_empty_array_and_ok(self) -> None:
        status, payload = self.audit()
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(len(payload["digest"]), 64)
        self.assertEqual(payload["digest"], payload["digest"].lower())
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)

    def test_new_execution_appends_and_replay_appends_nothing(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        status, applied, error = self.gap_repair("peer-a", "exec-1")
        self.assertIs(status, HTTPStatus.CREATED, error)
        _, first = self.audit()
        self.assertEqual(first["executionsCount"], 1)
        item = self.advice("peer-a")[0]
        # An identical replay answers 200 from the committed binding.
        status, replay, error = self.store.apply_replication_repairs(
            "peer-a",
            "exec-1",
            4,
            self.receipts_digest("peer-a"),
            [self.suggestion(item)],
        )
        self.assertIs(status, HTTPStatus.OK, error)
        self.assertEqual(replay["status"], "ok")
        _, second = self.audit()
        self.assertEqual(second["executionsCount"], 1)
        self.assertEqual(second["digest"], first["digest"])
        self.assertEqual(second["verification"], first["verification"])
        # A different execution id appends a new record in creation order.
        status, _, error = self.gap_repair("peer-a", "exec-2")
        self.assertIs(status, HTTPStatus.CREATED, error)
        _, third = self.audit()
        self.assertEqual(third["executionsCount"], 2)
        self.assertEqual(
            [item["ackId"] for item in third["executions"]], ["exec-1", "exec-2"]
        )

    def test_execution_item_keeps_binding_anchors_suggestions_and_results(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        status, applied, error = self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        self.assertIs(status, HTTPStatus.CREATED, error)
        _, payload = self.audit()
        execution = payload["executions"][0]
        self.assertEqual(list(execution), EXECUTION_FIELDS)
        self.assertEqual(execution["peerId"], "peer-a")
        self.assertEqual(execution["ackId"], "exec-1")
        self.assertEqual(execution["expectedCheckpoint"], 4)
        self.assertEqual(execution["expectedReceipts"], self.receipts_digest("peer-a"))
        self.assertEqual(execution["suggestions"], [suggestion])
        self.assertEqual(
            execution["results"],
            [{"action": "resend", "boundary": {"start": 1, "end": 2}}],
        )
        self.assertEqual(execution["results"], applied["results"])
        self.assertEqual(execution["cursor"], 4)

    def test_identity_correction_execution_round_trips_its_evidence(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        item = self.advice("peer-a")[0]
        self.assertEqual(item["action"], "correct_identity")
        suggestion = self.suggestion(item)
        status, _, error = self.store.apply_replication_repairs(
            "peer-a", "exec-1", 2, self.receipts_digest("peer-a"), [suggestion]
        )
        self.assertIs(status, HTTPStatus.CREATED, error)
        _, payload = self.audit()
        execution = payload["executions"][0]
        self.assertEqual(execution["suggestions"], [suggestion])
        self.assertEqual(
            execution["results"],
            [{"action": "correct_identity", "boundary": {"position": 1}}],
        )
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_page_orders_by_peer_then_creation_digest_covers_creation(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.gap_repair("peer-a", "exec-a1")  # creation 0
        self.gap_repair("peer-b", "exec-b1")  # creation 1
        self.gap_repair("peer-a", "exec-a2")  # creation 2
        _, payload = self.audit()
        self.assertEqual(
            [(e["peerId"], e["ackId"]) for e in payload["executions"]],
            [("peer-a", "exec-a1"), ("peer-a", "exec-a2"), ("peer-b", "exec-b1")],
        )
        self.assertEqual(payload["executionsCount"], 3)
        history = [
            {
                "peerId": peer_id,
                "ackId": ack_id,
                **binding,
            }
            for (peer_id, ack_id), binding in self.store._repairs.items()
        ]
        self.assertEqual(
            [e["ackId"] for e in history], ["exec-a1", "exec-b1", "exec-a2"]
        )
        self.assertEqual(
            payload["digest"],
            hashlib.sha256(_repair_executions_digest_input(history)).hexdigest(),
        )

    def test_paging_trims_only_the_page(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.gap_repair("peer-a", "exec-a1")
        self.gap_repair("peer-b", "exec-b1")
        self.gap_repair("peer-a", "exec-a2")
        _, full = self.audit()
        pages = []
        for after in range(4):
            status, page = self.audit(after, 1)
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
        self.assertEqual(
            [[e["ackId"] for e in p["executions"]] for p in pages],
            [["exec-a1"], ["exec-a2"], ["exec-b1"], []],
        )
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])
        for page in pages:
            self.assertEqual(page["algorithm"], "sha256")
            self.assertEqual(page["digest"], full["digest"])
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"], full["verification"])

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        status, first = self.audit(1, 10)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(first["executions"], [])
        self.assertEqual(first["nextCursor"], 1)
        self.assertFalse(first["hasMore"])
        self.assertEqual(first["executionsCount"], 1)
        self.assertEqual(first["verification"]["status"], "ok")
        # The empty tail is stable across repeated reads.
        status, second = self.audit(1, 10)
        self.assertEqual(second, first)
        # Zero executions: after=0 is already the stable empty tail.
        fresh = StateStore()
        status, empty = fresh.get_replication_repair_executions(0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(empty["executions"], [])
        self.assertEqual(empty["executionsCount"], 0)

    def test_after_past_count_raises_value_error(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        with self.assertRaises(ValueError):
            self.audit(2, 1)

    def test_query_is_strictly_read_only(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        repairs_before = copy.deepcopy(dict(self.store._repairs))
        checkpoints_before = dict(self.store._checkpoints)
        acks_before = copy.deepcopy(dict(self.store._acks))
        metrics_before = self.store.get_metrics()
        advice_before = self.store.get_replication_repairs(0, 100)[1]
        for query in ((0, 100), (0, 1), (1, 1), (0, 100)):
            self.audit(*query)
        self.assertEqual(self.store._repairs, repairs_before)
        self.assertEqual(self.store._checkpoints, checkpoints_before)
        self.assertEqual(self.store._acks, acks_before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(
            self.store.get_replication_repairs(0, 100)[1], advice_before
        )

    def test_damaged_checkpoint_is_reported_broken_while_summary_covers_all(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        # Roll the registered checkpoint behind the execution's restored
        # cursor without touching the history: the verification must flag
        # the committed execution from a later snapshot.
        self.store._checkpoints["peer-a"] = 0
        _, payload = self.audit()
        self.assertEqual(payload["executionsCount"], 1)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["checkpointViolations"],
            [
                {
                    "executionIndex": 0,
                    "peerId": "peer-a",
                    "ackId": "exec-1",
                    "expected": {
                        "checkpoint": 4,
                        "registered": 0,
                        "logLength": 4,
                    },
                    "observed": {"cursor": 4},
                }
            ],
        )

    def test_damaged_result_boundary_is_reported_broken(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        self.store._repairs[("peer-a", "exec-1")]["results"][0]["boundary"] = {
            "start": 1,
            "end": 3,
        }
        _, payload = self.audit()
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["boundaryViolations"][0]["suggestionIndex"], 0
        )
        self.assertEqual(
            verdict["boundaryViolations"][0]["observed"], {"start": 1, "end": 3}
        )

    def test_damaged_record_shape_is_reported_broken(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        self.store._repairs[("peer-a", "exec-1")]["expectedReceipts"] = "broken"
        _, payload = self.audit()
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(
            payload["verification"]["recordViolations"],
            [{"executionIndex": 0, "peerId": "peer-a", "ackId": "exec-1"}],
        )

    def test_out_of_order_binding_is_reported_broken(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        binding = self.store._repairs[("peer-a", "exec-1")]
        binding["suggestions"] = [
            interval("correct_cursor", "ack-9", 2, 2),
            interval("resend", "ack-2", 1, 2),
        ]
        binding["results"] = [
            {"action": "correct_cursor", "boundary": {"start": 2, "end": 2}},
            {"action": "resend", "boundary": {"start": 1, "end": 2}},
        ]
        _, payload = self.audit()
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(
            payload["verification"]["outOfOrderActions"],
            [{"executionIndex": 0, "peerId": "peer-a", "ackId": "exec-1"}],
        )


class RepairExecutionsRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        store = StateStore(data_file=self.data_file)
        for index in range(4):
            store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
        helper = RepairExecutionsStoreFixture()
        helper.store = store
        helper.seed_gap("peer-a")
        helper.seed_gap("peer-b", first="ack-3", second="ack-4")
        helper.gap_repair("peer-a", "exec-a1")
        helper.gap_repair("peer-b", "exec-b1")
        helper.gap_repair("peer-a", "exec-a2")
        item = helper.suggestion(helper.advice("peer-a")[0])
        digest = helper.receipts_digest("peer-a")
        _, before = store.get_replication_repair_executions(0, 2)
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_replication_repair_executions(0, 2)
        self.assertEqual(after["executions"], before["executions"])
        self.assertEqual(after["nextCursor"], before["nextCursor"])
        self.assertEqual(after["hasMore"], before["hasMore"])
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["executionsCount"], before["executionsCount"])
        self.assertEqual(after["verification"], before["verification"])
        # The recovered binding still replays as ok and appends nothing.
        status, replay, error = recovered.apply_replication_repairs(
            "peer-a", "exec-a1", 4, digest, [item]
        )
        self.assertIs(status, HTTPStatus.OK, error)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(
            recovered.get_replication_repair_executions(0, 100)[1][
                "executionsCount"
            ],
            3,
        )

    def test_old_file_without_section_recovers_empty(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=self.data_file)
        status, payload = store.get_replication_repair_executions(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], OK_VERIFICATION)


class RepairExecutionsHttpFixture(unittest.TestCase):
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
        self.server.store = StateStore()

    def raw_request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=body if isinstance(body, (bytes, str)) else json.dumps(body),
                headers=request_headers,
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def get(self, query: str):
        return self.raw_request("GET", EXECUTIONS_PATH + query)

    def seed(self, count: int = 4) -> None:
        for index in range(count):
            status, _, _, _ = self.raw_request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int) -> None:
        status, _, _, _ = self.raw_request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status == 200

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.server.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def apply(self, peer: str, ack_id: str) -> None:
        status, advice, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=0&limit=100"
        )
        assert status == 200
        item = next(entry for entry in advice["suggestions"] if entry["peer"] == peer)
        status, receipts, _, _ = self.raw_request(
            "GET", f"/v1/sync/peers/{peer}/receipts?after=0&limit=100"
        )
        assert status == 200
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        status, payload, _, _ = self.raw_request(
            "POST",
            "/v1/replication/repairs/apply",
            {
                "peerId": peer,
                "ackId": ack_id,
                "expectedCheckpoint": self.server.store.get_checkpoint(peer)[1][
                    "cursor"
                ],
                "expectedReceipts": receipts["digest"],
                "suggestions": [suggestion],
            },
        )
        self.assertEqual(status, 201, payload)

    def seed_gap(self, peer: str, *, first: str = "ack-1", second: str = "ack-2") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, first, 1, [identity("r0", "o0")])
        self.seed_receipt(peer, second, 3, [identity("r2", "o2")])


class RepairExecutionsHttpTests(RepairExecutionsHttpFixture):
    def test_empty_history_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            raw,
            b'{"executions":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + payload["digest"].encode("ascii")
            + b'","executionsCount":0,"verification":{"status":"ok",'
            b'"duplicateBindings":[],"outOfOrderActions":[],'
            b'"boundaryViolations":[],"checkpointViolations":[],'
            b'"recordViolations":[]}}\n',
        )

    def test_populated_history_pages_and_replays_without_appending(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.apply("peer-a", "exec-a1")
        self.apply("peer-b", "exec-b1")
        self.apply("peer-a", "exec-a2")
        status, payload, raw, headers = self.get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            [(e["peerId"], e["ackId"]) for e in payload["executions"]],
            [("peer-a", "exec-a1"), ("peer-a", "exec-a2"), ("peer-b", "exec-b1")],
        )
        self.assertEqual(payload["nextCursor"], 3)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 3)
        self.assertEqual(payload["verification"]["status"], "ok")
        execution = payload["executions"][0]
        self.assertEqual(list(execution), EXECUTION_FIELDS)
        self.assertEqual(execution["expectedCheckpoint"], 4)
        self.assertEqual(
            execution["results"],
            [{"action": "resend", "boundary": {"start": 1, "end": 2}}],
        )
        # Page through with limit=1: digest, count, and verification are
        # page-independent.
        pages = []
        for after in range(4):
            status, page, _, _ = self.get(f"?after={after}&limit=1")
            self.assertEqual(status, 200)
            pages.append(page)
        self.assertEqual(
            [[e["ackId"] for e in p["executions"]] for p in pages],
            [["exec-a1"], ["exec-a2"], ["exec-b1"], []],
        )
        for page in pages:
            self.assertEqual(page["digest"], payload["digest"])
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"]["status"], "ok")
        # Replaying an existing execution answers 200 and appends nothing.
        replay_status, replay_body, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=0&limit=100"
        )
        self.assertEqual(replay_status, 200)
        replay_item = next(
            entry for entry in replay_body["suggestions"] if entry["peer"] == "peer-a"
        )
        replay_suggestion = {
            "action": replay_item["action"],
            "ackId": replay_item["ackId"],
            "location": replay_item["location"],
            "target": replay_item["target"],
        }
        _, replay_receipts, _, _ = self.raw_request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        status, replay, _, _ = self.raw_request(
            "POST",
            "/v1/replication/repairs/apply",
            {
                "peerId": "peer-a",
                "ackId": "exec-a1",
                "expectedCheckpoint": self.server.store.get_checkpoint("peer-a")[1][
                    "cursor"
                ],
                "expectedReceipts": replay_receipts["digest"],
                "suggestions": [replay_suggestion],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(replay["status"], "ok")
        status, replayed, _, _ = self.get("?after=3&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(replayed["executions"], [])
        self.assertEqual(replayed["executionsCount"], 3)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        status, payload, raw, _ = self.get("?after=1&limit=10")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 1)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_bad_queries_are_400_and_change_nothing(self) -> None:
        self.seed(1)
        self.register("peer-a", 1)
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
            "?after&limit=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, bad_payload, raw, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(bad_payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))
        # No rejected query created an execution.
        status, payload, _, _ = self.get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(self.server.store._repairs, {})

    def test_after_past_the_execution_count_is_400(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        status, payload, raw, _ = self.get("?after=2&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        bad_paths = [
            EXECUTIONS_PATH + "/",
            EXECUTIONS_PATH + "/extra",
            "/v1/replication/repairs/exec",
            "/v1/replication/repair/executions",
            "/v1/replication",
            "/v1/repairs/executions",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=%ZZ&limit=x"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_published_advice_route_keeps_its_own_query_contract(self) -> None:
        # ``/v1/replication/repairs`` is a different published route, so a
        # malformed query there keeps its own 400 rather than becoming a
        # 404 shape mismatch.
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=%ZZ&limit=x"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertNotIn("executions", payload)

    def test_post_to_executions_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", EXECUTIONS_PATH, {"cursor": 0}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_plan_and_apply_routes_stays_404(self) -> None:
        for path in (
            "/v1/replication/repairs/plan",
            "/v1/replication/repairs/apply",
        ):
            status, payload, _, _ = self.raw_request(
                "GET", f"{path}?after=0&limit=1"
            )
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only_over_http(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        status, first, _, _ = self.get("?after=0&limit=100")
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        checkpoint_before = self.server.store.get_checkpoint("peer-a")[1]
        for query in (
            "?after=0&limit=100",
            "?after=0&limit=1",
            "?after=1&limit=1",
            "?after=0&limit=x",
        ):
            self.get(query)
        status, second, _, _ = self.get("?after=0&limit=100")
        self.assertEqual(second, first)
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(
            self.server.store.get_checkpoint("peer-a")[1], checkpoint_before
        )


class RepairExecutionsPersistenceHttpTests(RepairExecutionsHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_query_persists_nothing(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in (
            "?after=0&limit=100",
            "?after=1&limit=1",
            "?after=0&limit=1&x=1",
            "?after=99&limit=1",
        ):
            self.get(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class RepairExecutionsAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-auth-")
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

    def get(self, port, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers=dict(headers or []))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, response_headers

    QUERY = "?after=0&limit=1"
    PATH = EXECUTIONS_PATH + QUERY

    def test_single_token_missing_bad_or_wrong_is_401(self) -> None:
        status, payload, headers = self.get(self.single_port, self.PATH)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, self.PATH, [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, self.PATH, [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_single_token_valid_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.single_port,
            self.PATH,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            self.PATH,
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, headers = self.get(
            self.scope_port,
            EXECUTIONS_PATH + "?after=%ZZ",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            self.PATH,
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)

    def test_unknown_shape_still_authenticates_before_the_404(self) -> None:
        # Like every other route, an unknown path authenticates first on
        # an auth-enabled server; a missing credential is 401 rather than
        # a shape 404.
        status, payload, headers = self.get(
            self.single_port,
            EXECUTIONS_PATH + "/extra" + self.QUERY,
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # A valid credential then sees the shape mismatch.
        status, payload, _ = self.get(
            self.single_port,
            EXECUTIONS_PATH + "/extra" + self.QUERY,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # The shape decision still takes priority over the query check.
        status, payload, _ = self.get(
            self.single_port,
            EXECUTIONS_PATH + "/extra?after=%ZZ&limit=x",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
