"""Tests for the read-only cross-stream integrity endpoint::

    GET /v1/integrity/verify

The endpoint cross-checks the two existing independent audit streams —
the atomic-transaction ledger and the conditional replication-repair
execution history — against the shared accepted-operation log from one
committed snapshot. It takes no request parameters and returns a compact
UTF-8 JSON object terminated by a single newline with exactly four fields
in order: ``status`` (``"ok"``/``"broken"``), ``transactions``,
``repairExecutions``, and ``cross``. Each history group reports
``algorithm`` (always ``"sha256"``), the full-history ``digest`` in that
stream's existing audit encoding, the full ``count``, and its existing
internal ``anomalies`` judgement (duplicate bindings, malformed records,
identity content mismatch, and in-batch duplicates for transactions;
duplicate bindings, action ordering, result boundaries, checkpoint
violations, and malformed records for repairs). ``cross`` reports
``status`` and only the between-stream inconsistencies against the shared
log — transaction identities the ledger binds but the shared log never
placed (``missingTransactionOperations``) and repair cursors the shared
log length or the peer's registered checkpoint does not confirm
(``unconfirmedRepairCursors``) — never repeating a purely intra-group
anomaly.

The tests cover the empty-histories stable report, populated digests and
counts, the per-group anomaly reuse, the two cross-stream anomaly classes
with their 0-based positions, business ids, locations and
expected/observed content, the HTTP precedence chain (404 path shape
before the 400 query check, 401 authentication with a Bearer challenge,
403 in scope mode without read/admin), the compact ordered
single-newline body with explicit Content-Length, restart consistency
under ``--data-file``, and the strict read-only guarantee. Only the
Python standard library is used.
"""

from __future__ import annotations

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
    _cross_stream_verification_locked,
    _repair_executions_digest_input,
    _repair_executions_verification_locked,
    _transactions_digest_input,
    _transactions_verification_locked,
    load_scope_policy,
)

VERIFY_PATH = "/v1/integrity/verify"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()

ROOT_FIELDS = ["status", "transactions", "repairExecutions", "cross"]
GROUP_FIELDS = ["algorithm", "digest", "count", "anomalies"]
TRANSACTION_ANOMALIES = [
    "duplicateTransactionIds",
    "recordViolations",
    "identityMismatches",
    "batchViolations",
]
REPAIR_ANOMALIES = [
    "duplicateBindings",
    "outOfOrderActions",
    "boundaryViolations",
    "checkpointViolations",
    "recordViolations",
]
CROSS_FIELDS = ["status", "anomalies"]
CROSS_ANOMALIES = [
    "missingTransactionOperations",
    "unconfirmedRepairCursors",
]


def tx_entry(
    key: str,
    replica: str = "r1",
    operation_id: str = "t1",
    value: str = "v",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {replica: 1},
        "candidates": list(candidates) if candidates is not None else [],
    }


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation}


def interval_suggestion(action: str, ack_id: str, start: int, end: int) -> dict:
    return {
        "action": action,
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def interval_result(action: str, start: int, end: int) -> dict:
    return {"action": action, "boundary": {"start": start, "end": end}}


def repair_record(
    peer_id: str = "peer-a",
    ack_id: str = "exec-1",
    *,
    expected_checkpoint: int = 0,
    cursor: int = 0,
    suggestions: list | None = None,
    results: list | None = None,
    digest: str | None = None,
) -> dict:
    if suggestions is None:
        suggestions = [interval_suggestion("resend", "ack-1", 0, 0)]
    if results is None:
        results = [interval_result("resend", 0, 0)]
    return {
        "peerId": peer_id,
        "ackId": ack_id,
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": digest if digest is not None else EMPTY_DIGEST,
        "suggestions": suggestions,
        "results": results,
        "cursor": cursor,
    }


class CrossStreamVerificationTests(unittest.TestCase):
    def test_empty_histories_verify_ok(self) -> None:
        verdict = _cross_stream_verification_locked([], {}, [], [], {"recordViolations": []}, {}, 0)
        self.assertEqual(
            verdict,
            {
                "status": "ok",
                "missingTransactionOperations": [],
                "unconfirmedRepairCursors": [],
            },
        )
        self.assertEqual(
            list(verdict),
            ["status", "missingTransactionOperations", "unconfirmedRepairCursors"],
        )

    def test_consistent_transaction_identity_in_log_is_not_cross_anomaly(self) -> None:
        history = [("tx-1", [tx_entry("k", "r1", "t1")])]
        accepted = [
            (
                "r1",
                operation("t1", "k", "v", {"r1": 1}),
            )
        ]
        operations_index = {("r1", "t1"): accepted[0][1]}
        verdict = _cross_stream_verification_locked(
            history, operations_index, accepted, [], {"recordViolations": []}, {}, 1
        )
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(verdict["missingTransactionOperations"], [])

    def test_identity_in_archive_but_missing_from_log_is_cross_anomaly(self) -> None:
        history = [("tx-1", [tx_entry("k", "r1", "t1")])]
        operations_index = {("r1", "t1"): operation("t1", "k", "v", {"r1": 1})}
        # The archive carries the identity but the shared log is empty.
        verdict = _cross_stream_verification_locked(
            history, operations_index, [], [], {"recordViolations": []}, {}, 0
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["missingTransactionOperations"],
            [
                {
                    "transactionIndex": 0,
                    "transactionId": "tx-1",
                    "operationIndex": 0,
                    "replicaId": "r1",
                    "operationId": "t1",
                }
            ],
        )

    def test_identity_in_neither_log_nor_archive_is_not_repeated_here(self) -> None:
        # The group scan already reports this as an identityMismatch
        # (expected null); the cross scan must not duplicate it.
        history = [("tx-1", [tx_entry("k", "r9", "t9")])]
        verdict = _cross_stream_verification_locked(
            history, {}, [], [], {"recordViolations": []}, {}, 0
        )
        self.assertEqual(verdict["missingTransactionOperations"], [])
        self.assertEqual(verdict["unconfirmedRepairCursors"], [])
        self.assertEqual(verdict["status"], "ok")

    def test_malformed_transaction_record_is_not_scanned_for_cross(self) -> None:
        history = [("tx-1", "not-a-list")]
        verdict = _cross_stream_verification_locked(
            history, {}, [], [], {"recordViolations": []}, {}, 0
        )
        self.assertEqual(verdict["missingTransactionOperations"], [])

    def test_confirmed_repair_cursor_is_not_an_anomaly(self) -> None:
        record = repair_record(cursor=2)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], {"recordViolations": []}, {"peer-a": 2}, 3
        )
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(verdict["unconfirmedRepairCursors"], [])

    def test_repair_cursor_past_log_length_is_cross_anomaly(self) -> None:
        record = repair_record(cursor=4)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], {"recordViolations": []}, {"peer-a": 4}, 3
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["unconfirmedRepairCursors"],
            [
                {
                    "executionIndex": 0,
                    "peerId": "peer-a",
                    "ackId": "exec-1",
                    "expected": {"registered": 4, "logLength": 3},
                    "observed": {"cursor": 4},
                }
            ],
        )

    def test_repair_cursor_ahead_of_registered_checkpoint_is_cross_anomaly(self) -> None:
        record = repair_record(cursor=3)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], {"recordViolations": []}, {"peer-a": 1}, 5
        )
        self.assertEqual(
            verdict["unconfirmedRepairCursors"][0]["expected"],
            {"registered": 1, "logLength": 5},
        )

    def test_unknown_peer_checkpoint_is_cross_anomaly_with_null_registered(self) -> None:
        record = repair_record(cursor=1)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], {"recordViolations": []}, {}, 5
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertIsNone(
            verdict["unconfirmedRepairCursors"][0]["expected"]["registered"]
        )

    def test_repair_record_violation_is_not_also_a_cross_anomaly(self) -> None:
        record = repair_record(cursor=9)
        record["expectedReceipts"] = "broken"
        group = _repair_executions_verification_locked([record], {}, 3)
        self.assertEqual(group["recordViolations"][0]["executionIndex"], 0)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], group, {}, 3
        )
        # The structurally damaged record is the group's recordViolation
        # only; the cross scan skips it.
        self.assertEqual(verdict["unconfirmedRepairCursors"], [])

    def test_cursor_below_anchor_but_confirmed_is_not_a_cross_anomaly(self) -> None:
        # cursor < expectedCheckpoint is the group's checkpointViolation
        # condition only; with the log and registered checkpoint reaching
        # the cursor, the cross stream has nothing to add.
        record = repair_record(expected_checkpoint=4, cursor=2)
        verdict = _cross_stream_verification_locked(
            [], {}, [], [record], {"recordViolations": []}, {"peer-a": 4}, 5
        )
        self.assertEqual(verdict["unconfirmedRepairCursors"], [])
        self.assertEqual(verdict["status"], "ok")


class IntegrityVerifyStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def apply_tx(self, *entries: dict, transaction_id: str) -> None:
        status, _, _, _, error = self.store.apply_transaction(
            transaction_id, list(entries)
        )
        self.assertEqual(status, HTTPStatus.CREATED, error)

    def seed_write(self, replica: str, operation_id: str, key: str, value: str) -> None:
        status = self.store.apply_operation(
            replica, operation(operation_id, key, value, {replica: 1})
        )
        self.assertIs(status, HTTPStatus.CREATED)

    def verify(self):
        return self.store.get_integrity_verify()


class IntegrityVerifyStoreTests(IntegrityVerifyStoreFixture):
    def test_empty_store_reports_empty_groups_and_ok(self) -> None:
        status, payload = self.verify()
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(payload["status"], "ok")
        transactions = payload["transactions"]
        repairs = payload["repairExecutions"]
        cross = payload["cross"]
        self.assertEqual(list(transactions), GROUP_FIELDS)
        self.assertEqual(list(repairs), GROUP_FIELDS)
        self.assertEqual(transactions["algorithm"], "sha256")
        self.assertEqual(repairs["algorithm"], "sha256")
        self.assertEqual(transactions["digest"], EMPTY_DIGEST)
        self.assertEqual(repairs["digest"], EMPTY_DIGEST)
        self.assertEqual(transactions["count"], 0)
        self.assertEqual(repairs["count"], 0)
        self.assertEqual(list(transactions["anomalies"]), TRANSACTION_ANOMALIES)
        self.assertEqual(list(repairs["anomalies"]), REPAIR_ANOMALIES)
        self.assertEqual(list(cross), CROSS_FIELDS)
        self.assertEqual(cross["status"], "ok")
        self.assertEqual(list(cross["anomalies"]), CROSS_ANOMALIES)
        for name in TRANSACTION_ANOMALIES:
            self.assertEqual(transactions["anomalies"][name], [])
        for name in REPAIR_ANOMALIES:
            self.assertEqual(repairs["anomalies"][name], [])
        for name in CROSS_ANOMALIES:
            self.assertEqual(cross["anomalies"][name], [])

    def test_populated_transaction_digest_count_and_audit_match_existing(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.apply_tx(
            tx_entry("k2", "r2", "t2", clock={"r2": 1}), transaction_id="tx-2"
        )
        _, payload = self.verify()
        history = list(self.store._transactions.items())
        self.assertEqual(payload["transactions"]["count"], 2)
        self.assertEqual(
            payload["transactions"]["digest"],
            hashlib.sha256(_transactions_digest_input(history)).hexdigest(),
        )
        self.assertEqual(payload["transactions"]["anomalies"]["recordViolations"], [])
        # Identical to the plain transaction audit's internal scan.
        self.assertEqual(
            payload["transactions"]["anomalies"],
            {
                key: value
                for key, value in _transactions_verification_locked(
                    history, dict(self.store._operations)
                ).items()
                if key != "status"
            },
        )
        self.assertEqual(payload["status"], "ok")

    def test_transaction_identity_mismatch_stays_with_the_group(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.store._transactions["tx-1"] = [tx_entry("k1", "r1", "t1", value="tampered")]
        _, payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        marker = payload["transactions"]["anomalies"]["identityMismatches"][0]
        self.assertEqual(marker["expected"]["value"], "v")
        self.assertEqual(marker["observed"]["value"], "tampered")
        # The pure group anomaly is not repeated in the cross report.
        self.assertEqual(payload["cross"]["anomalies"]["missingTransactionOperations"], [])
        self.assertEqual(payload["cross"]["status"], "ok")

    def test_transaction_duplicate_and_batch_anomalies_stay_with_the_group(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.store._transactions["tx-2"] = [
            tx_entry("k", "r9", "a"),
            tx_entry("k", "r9", "b"),
        ]
        _, payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        self.assertEqual(
            payload["transactions"]["anomalies"]["batchViolations"][0]["operationId"],
            "b",
        )
        self.assertEqual(payload["cross"]["anomalies"]["missingTransactionOperations"], [])

    def test_committed_repair_history_matches_existing_audit_and_is_ok(self) -> None:
        for index in range(3):
            self.seed_write(f"r{index}", f"o{index}", "k", f"v{index}")
        status, _ = self.store.save_checkpoint("peer-a", 3)
        self.assertIs(status, HTTPStatus.OK)
        record = repair_record(
            "peer-a", "exec-1", expected_checkpoint=3, cursor=3,
            suggestions=[interval_suggestion("resend", "ack-1", 0, 0)],
            results=[interval_result("resend", 0, 0)],
        )
        self.store._repairs[("peer-a", "exec-1")] = {
            key: value for key, value in record.items()
        }
        _, payload = self.verify()
        repairs = payload["repairExecutions"]
        self.assertEqual(repairs["count"], 1)
        history = [
            {
                "peerId": "peer-a",
                "ackId": "exec-1",
                "expectedCheckpoint": 3,
                "expectedReceipts": EMPTY_DIGEST,
                "suggestions": [interval_suggestion("resend", "ack-1", 0, 0)],
                "results": [interval_result("resend", 0, 0)],
                "cursor": 3,
            }
        ]
        self.assertEqual(
            repairs["digest"],
            hashlib.sha256(_repair_executions_digest_input(history)).hexdigest(),
        )
        self.assertEqual(repairs["anomalies"]["checkpointViolations"], [])
        self.assertEqual(payload["cross"]["status"], "ok")
        self.assertEqual(payload["status"], "ok")

    def test_repair_group_anomalies_are_reused_and_not_cross_repeated(self) -> None:
        # A cursor below the anchor: a group checkpointViolation only. The
        # shared log reaches the cursor and the registered checkpoint is at
        # or beyond it, so the cross stream has nothing to add.
        for index in range(4):
            self.seed_write(f"r{index}", f"o{index}", "k", f"v{index}")
        record = repair_record(expected_checkpoint=4, cursor=2)
        self.store._repairs[("peer-a", "exec-1")] = dict(record)
        self.store._checkpoints["peer-a"] = 4
        _, payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        marker = payload["repairExecutions"]["anomalies"]["checkpointViolations"][0]
        self.assertEqual(marker["observed"], {"cursor": 2})
        self.assertEqual(marker["expected"]["checkpoint"], 4)
        self.assertEqual(payload["cross"]["anomalies"]["unconfirmedRepairCursors"], [])
        self.assertEqual(payload["cross"]["status"], "ok")

    def test_cross_missing_transaction_operation_against_the_shared_log(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.apply_tx(tx_entry("k2", "r2", "t2", clock={"r2": 1}), transaction_id="tx-2")
        # Drop the second transaction's operation from the shared log while
        # leaving the accepted-operation archive intact.
        self.store._accepted[:] = [
            item for item in self.store._accepted if item[1]["operationId"] != "t2"
        ]
        _, payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        missing = payload["cross"]["anomalies"]["missingTransactionOperations"]
        self.assertEqual(
            missing,
            [
                {
                    "transactionIndex": 1,
                    "transactionId": "tx-2",
                    "operationIndex": 0,
                    "replicaId": "r2",
                    "operationId": "t2",
                }
            ],
        )
        # Identity content still matches the archive, so no group anomaly.
        self.assertEqual(
            payload["transactions"]["anomalies"]["identityMismatches"], []
        )

    def test_cross_unconfirmed_repair_cursor_against_log_and_checkpoint(self) -> None:
        self.seed_write("r1", "o1", "k", "v1")
        record = repair_record("peer-a", "exec-1", cursor=3)
        self.store._repairs[("peer-a", "exec-1")] = dict(record)
        # No registered checkpoint and a log of length 1: cursor 3 is
        # unconfirmed on both counts.
        _, payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        unconfirmed = payload["cross"]["anomalies"]["unconfirmedRepairCursors"]
        self.assertEqual(
            unconfirmed,
            [
                {
                    "executionIndex": 0,
                    "peerId": "peer-a",
                    "ackId": "exec-1",
                    "expected": {"registered": None, "logLength": 1},
                    "observed": {"cursor": 3},
                }
            ],
        )
        # The same condition is also the group's checkpointViolation; both
        # reports legitimately observe it under their own definition.
        group_marker = payload["repairExecutions"]["anomalies"][
            "checkpointViolations"
        ][0]
        self.assertEqual(group_marker["executionIndex"], 0)

    def test_top_status_is_broken_when_any_group_is_broken(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.store._transactions["tx-bad"] = "not-a-list"
        _, payload = self.verify()
        self.assertEqual(payload["transactions"]["anomalies"]["recordViolations"][0]["transactionId"], "tx-bad")
        self.assertEqual(payload["repairExecutions"]["count"], 0)
        self.assertEqual(payload["cross"]["status"], "ok")
        self.assertEqual(payload["status"], "broken")

    def test_repeated_queries_are_stable(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        _, first = self.verify()
        for _ in range(3):
            _, again = self.verify()
            self.assertEqual(again, first)


class IntegrityVerifyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_restart_gives_identical_report(self) -> None:
        store = StateStore(data_file=self.data_file)
        status = store.apply_operation(
            "r1", operation("o1", "k", "v1", {"r1": 1})
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, _, _, _, error = store.apply_transaction(
            "tx-1", [tx_entry("k2", "r2", "t2", clock={"r2": 1})]
        )
        self.assertEqual(status, HTTPStatus.CREATED, error)
        _, before = store.get_integrity_verify()
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_integrity_verify()
        self.assertEqual(after, before)
        self.assertEqual(after["status"], "ok")
        self.assertEqual(after["transactions"]["count"], 1)
        self.assertEqual(after["repairExecutions"]["count"], 0)

    def test_old_file_without_sections_verifies_empty_intact(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=self.data_file)
        _, payload = store.get_integrity_verify()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["transactions"]["count"], 0)
        self.assertEqual(payload["transactions"]["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["repairExecutions"]["count"], 0)
        self.assertEqual(payload["repairExecutions"]["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["cross"]["status"], "ok")


class IntegrityVerifyHttpFixture(unittest.TestCase):
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

    def get(self, path: str = VERIFY_PATH):
        return self.raw_request("GET", path)


class IntegrityVerifyHttpTests(IntegrityVerifyHttpFixture):
    EMPTY_BODY = (
        b'{"status":"ok","transactions":{"algorithm":"sha256","digest":"'
        + EMPTY_DIGEST.encode("ascii")
        + b'","count":0,"anomalies":{"duplicateTransactionIds":[],'
        b'"recordViolations":[],"identityMismatches":[],"batchViolations":[]'
        b'}},"repairExecutions":{"algorithm":"sha256","digest":"'
        + EMPTY_DIGEST.encode("ascii")
        + b'","count":0,"anomalies":{"duplicateBindings":[],'
        b'"outOfOrderActions":[],"boundaryViolations":[],'
        b'"checkpointViolations":[],"recordViolations":[]}},'
        b'"cross":{"status":"ok","anomalies":{"missingTransactionOperations":[],'
        b'"unconfirmedRepairCursors":[]}}}\n'
    )

    def test_empty_store_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(list(payload["transactions"]), GROUP_FIELDS)
        self.assertEqual(list(payload["repairExecutions"]), GROUP_FIELDS)
        self.assertEqual(list(payload["cross"]), CROSS_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(raw, self.EMPTY_BODY)

    def test_repeated_gets_are_identical(self) -> None:
        _, _, first, _ = self.get()
        _, _, second, _ = self.get()
        self.assertEqual(first, second)

    def test_any_parameter_is_400_invalid_request(self) -> None:
        bad_queries = [
            VERIFY_PATH + "?x=1",
            VERIFY_PATH + "?x=",
            VERIFY_PATH + "?x",
            VERIFY_PATH + "?=1",
            VERIFY_PATH + "?x=1&y=2",
            VERIFY_PATH + "?x=1&x=2",
            VERIFY_PATH + "?after=0",
            VERIFY_PATH + "?status=ok",
        ]
        for path in bad_queries:
            with self.subTest(path=path):
                status, payload, raw, _ = self.get(path)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_and_method_mismatches_are_404(self) -> None:
        bad_paths = [
            ("GET", VERIFY_PATH + "/"),
            ("GET", VERIFY_PATH + "/extra"),
            ("GET", "/v1/integrity"),
            ("GET", "/v1/integrity/verif"),
            ("GET", "/v1/integrities/verify"),
            ("POST", VERIFY_PATH),
            ("PUT", VERIFY_PATH),
            ("DELETE", VERIFY_PATH),
            ("PATCH", VERIFY_PATH),
            ("OPTIONS", VERIFY_PATH),
        ]
        for method, path in bad_paths:
            with self.subTest(method=method, path=path):
                status, payload, _, _ = self.raw_request(method, path + "?x=1")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_post_on_verify_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request("POST", VERIFY_PATH, {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_path_shape_404_precedes_query_check(self) -> None:
        status, payload, _, _ = self.get(VERIFY_PATH + "/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_health_stays_anonymous_and_unaffected(self) -> None:
        status, payload, _, _ = self.raw_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_query_is_strictly_read_only(self) -> None:
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        for _ in range(5):
            status, _, _, _ = self.get()
            self.assertEqual(status, 200)
        for bad in (VERIFY_PATH + "?x=1", VERIFY_PATH + "/"):
            self.get(bad)
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(self.server.store._accepted, [])
        self.assertEqual(self.server.store._transactions, {})
        self.assertEqual(self.server.store._repairs, {})

    def test_populated_state_reports_counts_and_ok(self) -> None:
        # A committed transaction through HTTP.
        body = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "shape",
                    "replicaId": "r2",
                    "operationId": "op-2",
                    "value": "round",
                    "clock": {"r2": 1},
                    "candidates": [],
                }
            ],
        }
        status, _, _, _ = self.raw_request(
            "POST", "/v1/transactions/apply", body
        )
        self.assertEqual(status, 201)
        status, payload, raw, _ = self.get()
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["transactions"]["count"], 1)
        self.assertEqual(payload["repairExecutions"]["count"], 0)
        self.assertEqual(payload["cross"]["status"], "ok")
        self.assertEqual(len(payload["transactions"]["digest"]), 64)


class IntegrityVerifyPersistenceHttpTests(IntegrityVerifyHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-verify-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_query_persists_nothing(self) -> None:
        body = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "shape",
                    "replicaId": "r2",
                    "operationId": "op-2",
                    "value": "round",
                    "clock": {"r2": 1},
                    "candidates": [],
                }
            ],
        }
        status, _, _, _ = self.raw_request("POST", "/v1/transactions/apply", body)
        self.assertEqual(status, 201)
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for _ in range(4):
            self.get()
        self.get(VERIFY_PATH + "?x=1")
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class IntegrityVerifyAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-verify-auth-")
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

    def test_single_token_missing_bad_or_wrong_is_401_with_challenge(self) -> None:
        status, payload, headers = self.get(self.single_port, VERIFY_PATH)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, VERIFY_PATH, [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, VERIFY_PATH, [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_single_token_valid_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            VERIFY_PATH,
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, headers = self.get(
            self.scope_port,
            VERIFY_PATH + "?x=1",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            VERIFY_PATH,
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_unknown_shape_still_authenticates_before_the_404(self) -> None:
        status, payload, headers = self.get(
            self.single_port, VERIFY_PATH + "/extra?x=1"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra?x=1",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_health_stays_anonymous_under_auth(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
