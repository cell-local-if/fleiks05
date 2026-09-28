"""Tests for the read-only cross-stream integrity verification entry::

    GET /v1/integrity/verify

From one committed snapshot the endpoint cross-checks the two existing
audit streams against the shared accepted-operation log and the
registered checkpoints. It takes no request parameters and reports:

- ``status``: ``"ok"`` only when both per-stream groups and the cross
  group are all intact, otherwise ``"broken"``;
- ``transactions``: ``algorithm`` (``"sha256"``), the full-history
  ``digest``, the ``count``, and the transaction group's four internal
  ``anomalies`` lists (duplicate bindings, damaged records, identity
  content mismatches, in-batch repeats);
- ``repairExecutions``: the same three summary fields and the repair
  group's five internal ``anomalies`` lists (duplicate bindings, action
  ordering, result boundaries, checkpoint violations, structural
  damage);
- ``cross``: its own ``status`` plus only cross-stream anomalies —
  transaction identities that disagree with the shared log
  (``identityMismatches``) and repair restore cursors that run below the
  log origin, past its length, or without peer-checkpoint confirmation
  (``checkpointViolations``). Purely group-internal anomalies are never
  repeated in the cross group.

The tests cover the store-level snapshot/recovery semantics and
independent scans, the HTTP precedence chain (404 path shape before the
400 query check, 401 authentication with a Bearer challenge, 403 in
scope mode without read/admin), the compact field-ordered
single-newline body with an explicit Content-Length, empty-history
stability, restart consistency, and the strict read-only guarantee.
Only the Python standard library is used.
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
    _integrity_cross_transaction_mismatches_locked,
    _repair_executions_digest_input,
    _transactions_digest_input,
    load_scope_policy,
)

VERIFY_PATH = "/v1/integrity/verify"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
DIGEST_64_A = "a" * 64


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


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


def resend_suggestion(start: int = 1, end: int = 2, ack_id: str = "ack-2") -> dict:
    return {
        "action": "resend",
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def repair_binding(
    *,
    cursor: int = 2,
    expected_checkpoint: int = 2,
    expected_receipts: str = DIGEST_64_A,
    suggestions=None,
    results=None,
) -> dict:
    if suggestions is None:
        suggestions = [resend_suggestion()]
    if results is None:
        results = [{"action": "resend", "boundary": {"start": 1, "end": 2}}]
    return {
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": expected_receipts,
        "suggestions": suggestions,
        "results": results,
        "cursor": cursor,
    }


class IntegrityStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def apply_tx(self, *entries: dict, transaction_id: str = "tx-1") -> None:
        # A batch whose operations were already accepted replies 200 but
        # still commits the (new) transaction binding; either success code
        # leaves the binding in place for the report.
        status, _, _, _, error = self.store.apply_transaction(
            transaction_id, list(entries)
        )
        self.assertIn(status, (HTTPStatus.OK, HTTPStatus.CREATED), error)

    def put_repair(self, peer: str, ack_id: str, binding: dict) -> None:
        self.store._repairs[(peer, ack_id)] = binding

    def verify(self):
        status, payload = self.store.get_integrity_verify()
        self.assertIs(status, HTTPStatus.OK)
        return payload


class IntegrityEmptyTests(IntegrityStoreFixture):
    def test_empty_histories_are_intact_with_empty_array_digests(self) -> None:
        payload = self.verify()
        self.assertEqual(list(payload), ["status", "transactions", "repairExecutions", "cross"])
        self.assertEqual(payload["status"], "ok")

        tx_group = payload["transactions"]
        self.assertEqual(list(tx_group), ["algorithm", "digest", "count", "anomalies"])
        self.assertEqual(tx_group["algorithm"], "sha256")
        self.assertEqual(tx_group["digest"], EMPTY_DIGEST)
        self.assertEqual(tx_group["count"], 0)
        self.assertEqual(
            list(tx_group["anomalies"]),
            [
                "duplicateTransactionIds",
                "recordViolations",
                "identityMismatches",
                "batchViolations",
            ],
        )
        self.assertTrue(all(v == [] for v in tx_group["anomalies"].values()))

        repair_group = payload["repairExecutions"]
        self.assertEqual(list(repair_group), ["algorithm", "digest", "count", "anomalies"])
        self.assertEqual(repair_group["algorithm"], "sha256")
        self.assertEqual(repair_group["digest"], EMPTY_DIGEST)
        self.assertEqual(repair_group["count"], 0)
        self.assertEqual(
            list(repair_group["anomalies"]),
            [
                "duplicateBindings",
                "outOfOrderActions",
                "boundaryViolations",
                "checkpointViolations",
                "recordViolations",
            ],
        )
        self.assertTrue(all(v == [] for v in repair_group["anomalies"].values()))

        cross = payload["cross"]
        self.assertEqual(list(cross), ["status", "anomalies"])
        self.assertEqual(cross["status"], "ok")
        self.assertEqual(list(cross["anomalies"]), ["identityMismatches", "checkpointViolations"])
        self.assertEqual(cross["anomalies"]["identityMismatches"], [])
        self.assertEqual(cross["anomalies"]["checkpointViolations"], [])


class IntegrityPopulatedTests(IntegrityStoreFixture):
    def test_committed_transaction_and_confirmed_repair_are_all_ok(self) -> None:
        # Two accepted operations, then one transaction over the first.
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k2", "w", {"r2": 1}))
        self.apply_tx(
            tx_entry("k1", "r1", "o1", "v", {"r1": 1}, []),
            transaction_id="tx-1",
        )
        # A repair restoring a confirmed cursor within the log.
        status, _ = self.store.save_checkpoint("peer-a", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.put_repair("peer-a", "exec-1", repair_binding())

        payload = self.verify()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["transactions"]["count"], 1)
        self.assertEqual(payload["repairExecutions"]["count"], 1)
        self.assertTrue(all(v == [] for v in payload["transactions"]["anomalies"].values()))
        self.assertTrue(all(v == [] for v in payload["repairExecutions"]["anomalies"].values()))
        self.assertEqual(payload["cross"]["status"], "ok")
        self.assertEqual(payload["cross"]["anomalies"]["identityMismatches"], [])
        self.assertEqual(payload["cross"]["anomalies"]["checkpointViolations"], [])

    def test_digests_follow_each_stream_audit_encoding(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.apply_tx(tx_entry("k1", "r1", "o1"), transaction_id="tx-1")
        self.put_repair("peer-a", "exec-1", repair_binding())

        payload = self.verify()
        tx_history = list(self.store._transactions.items())
        repair_history = [
            {
                "peerId": peer,
                "ackId": ack,
                **binding,
            }
            for (peer, ack), binding in self.store._repairs.items()
        ]
        self.assertEqual(
            payload["transactions"]["digest"],
            hashlib.sha256(_transactions_digest_input(tx_history)).hexdigest(),
        )
        self.assertEqual(
            payload["repairExecutions"]["digest"],
            hashlib.sha256(_repair_executions_digest_input(repair_history)).hexdigest(),
        )

    def test_transaction_identity_content_mismatch_is_group_and_cross(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.apply_tx(tx_entry("k1", "r1", "o1"), transaction_id="tx-1")
        # Tamper with the stored binding content only; the log stays intact.
        self.store._transactions["tx-1"] = [tx_entry("k1", "r1", "o1", value="tampered")]

        payload = self.verify()
        self.assertEqual(payload["status"], "broken")
        group_marker = payload["transactions"]["anomalies"]["identityMismatches"][0]
        self.assertEqual(group_marker["transactionIndex"], 0)
        self.assertEqual(group_marker["transactionId"], "tx-1")
        self.assertEqual(group_marker["expected"]["value"], "v")
        self.assertEqual(group_marker["observed"]["value"], "tampered")

        cross_markers = payload["cross"]["anomalies"]["identityMismatches"]
        self.assertEqual(len(cross_markers), 1)
        marker = cross_markers[0]
        self.assertEqual(marker["transactionIndex"], 0)
        self.assertEqual(marker["transactionId"], "tx-1")
        self.assertEqual(marker["operationIndex"], 0)
        self.assertEqual(marker["replicaId"], "r1")
        self.assertEqual(marker["operationId"], "o1")
        self.assertEqual(marker["position"], 0)
        self.assertEqual(marker["expected"]["value"], "v")
        self.assertEqual(marker["observed"]["value"], "tampered")

    def test_transaction_identity_missing_from_log_is_cross_mismatch_null(self) -> None:
        # A well-formed binding whose operation never entered the log.
        self.store._transactions["tx-ghost"] = [tx_entry("k9", "r9", "g9")]
        payload = self.verify()
        marker = payload["cross"]["anomalies"]["identityMismatches"][0]
        self.assertEqual(marker["transactionId"], "tx-ghost")
        self.assertIsNone(marker["position"])
        self.assertIsNone(marker["expected"])
        self.assertEqual(marker["observed"]["operationId"], "g9")
        self.assertEqual(payload["cross"]["status"], "broken")
        self.assertEqual(payload["status"], "broken")

    def test_in_batch_repeat_is_group_only_when_content_matches_log(self) -> None:
        # One accepted operation bound twice inside a single transaction:
        # the repeated occurrence is an in-group batchViolation, but every
        # well-formed entry agrees with the shared log, so cross is clean.
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store._transactions["tx-batch"] = [
            tx_entry("k1", "r1", "o1"),
            tx_entry("k1", "r1", "o1"),
        ]
        payload = self.verify()
        batch = payload["transactions"]["anomalies"]["batchViolations"]
        self.assertEqual(
            batch,
            [
                {
                    "transactionIndex": 0,
                    "transactionId": "tx-batch",
                    "operationIndex": 1,
                    "replicaId": "r1",
                    "operationId": "o1",
                }
            ],
        )
        self.assertEqual(payload["transactions"]["anomalies"]["identityMismatches"], [])
        self.assertEqual(payload["cross"]["anomalies"]["identityMismatches"], [])
        # A batch anomaly still breaks the overall status via the group.
        self.assertEqual(payload["status"], "broken")
        self.assertEqual(payload["cross"]["status"], "ok")

    def test_damaged_transaction_record_is_group_only_and_skipped_cross(self) -> None:
        self.store._transactions["tx-bad"] = "not-a-list"
        payload = self.verify()
        self.assertEqual(
            payload["transactions"]["anomalies"]["recordViolations"],
            [{"transactionIndex": 0, "transactionId": "tx-bad"}],
        )
        self.assertEqual(payload["cross"]["anomalies"]["identityMismatches"], [])

    def test_repair_cursor_past_log_is_group_and_cross(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        status, _ = self.store.save_checkpoint("peer-a", 1)
        self.assertIs(status, HTTPStatus.OK)
        # Cursor 5 runs past the one-record log even though it equals the
        # anchor and the registered checkpoint.
        self.put_repair(
            "peer-a",
            "exec-1",
            repair_binding(cursor=5, expected_checkpoint=5),
        )
        self.store._checkpoints["peer-a"] = 5
        payload = self.verify()
        self.assertTrue(payload["repairExecutions"]["anomalies"]["checkpointViolations"])
        cross = payload["cross"]["anomalies"]["checkpointViolations"]
        self.assertEqual(len(cross), 1)
        marker = cross[0]
        self.assertEqual(marker["executionIndex"], 0)
        self.assertEqual(marker["peerId"], "peer-a")
        self.assertEqual(marker["ackId"], "exec-1")
        self.assertEqual(marker["expected"]["logLength"], 1)
        self.assertEqual(marker["expected"]["registered"], 5)
        self.assertEqual(marker["observed"], {"cursor": 5})

    def test_unconfirmed_peer_checkpoint_is_cross_violation(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k2", "w", {"r2": 1}))
        # No registered checkpoint for the peer.
        self.put_repair("peer-x", "exec-1", repair_binding(cursor=2))
        payload = self.verify()
        cross = payload["cross"]["anomalies"]["checkpointViolations"]
        self.assertEqual(len(cross), 1)
        self.assertIsNone(cross[0]["expected"]["registered"])
        self.assertEqual(cross[0]["observed"]["cursor"], 2)

    def test_registered_checkpoint_behind_cursor_is_cross_violation(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k2", "w", {"r2": 2}))
        status, _ = self.store.save_checkpoint("peer-a", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.put_repair("peer-a", "exec-1", repair_binding(cursor=2, expected_checkpoint=2))
        payload = self.verify()
        cross = payload["cross"]["anomalies"]["checkpointViolations"]
        self.assertEqual(cross[0]["expected"]["registered"], 1)

    def test_anchor_regression_is_reported_in_group_and_cross(self) -> None:
        # cursor 2 is within the log and confirmed by the registered
        # checkpoint 2, but it precedes the repair-local expectedCheckpoint
        # 3: the existing execution audit judges this a checkpoint
        # violation, and that shared-log judgement is reused by the cross
        # stream verbatim.
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k2", "w", {"r2": 2}))
        status, _ = self.store.save_checkpoint("peer-a", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.put_repair(
            "peer-a",
            "exec-1",
            repair_binding(cursor=2, expected_checkpoint=3),
        )
        payload = self.verify()
        group = payload["repairExecutions"]["anomalies"]["checkpointViolations"]
        cross = payload["cross"]["anomalies"]["checkpointViolations"]
        self.assertEqual(len(group), 1)
        self.assertEqual(cross, group)
        self.assertEqual(group[0]["expected"]["checkpoint"], 3)
        self.assertEqual(group[0]["observed"], {"cursor": 2})
        self.assertEqual(payload["cross"]["status"], "broken")
        self.assertEqual(payload["status"], "broken")

    def test_negative_cursor_is_structural_damage_not_a_cross_violation(self) -> None:
        # A negative cursor cannot be a position in the shared log; the
        # execution audit judges it a malformed record (recordViolation)
        # before the checkpoint check, so the cross stream — which reuses
        # only the audit's checkpointViolations — reports nothing for it.
        self.put_repair(
            "peer-a",
            "exec-1",
            repair_binding(cursor=-1, expected_checkpoint=-1),
        )
        payload = self.verify()
        self.assertTrue(payload["repairExecutions"]["anomalies"]["recordViolations"])
        self.assertEqual(
            payload["repairExecutions"]["anomalies"]["checkpointViolations"], []
        )
        self.assertEqual(payload["cross"]["anomalies"]["checkpointViolations"], [])
        self.assertEqual(payload["cross"]["status"], "ok")

    def test_out_of_order_actions_and_boundary_damage_are_group_only(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k2", "w", {"r2": 2}))
        status, _ = self.store.save_checkpoint("peer-a", 2)
        self.assertIs(status, HTTPStatus.OK)
        # correct_cursor before resend violates the fixed action order.
        suggestions = [
            {
                "action": "correct_cursor",
                "ackId": "ack-9",
                "location": {"start": 2, "end": 2},
                "target": {"start": 2, "end": 2},
            },
            resend_suggestion(),
        ]
        results = [
            {"action": "correct_cursor", "boundary": {"start": 2, "end": 2}},
            {"action": "resend", "boundary": {"start": 1, "end": 2}},
        ]
        self.put_repair(
            "peer-a",
            "exec-order",
            repair_binding(suggestions=suggestions, results=results),
        )
        # A second execution whose restored boundary differs from target.
        self.put_repair(
            "peer-a",
            "exec-boundary",
            repair_binding(
                suggestions=[resend_suggestion()],
                results=[{"action": "resend", "boundary": {"start": 1, "end": 3}}],
            ),
        )
        payload = self.verify()
        self.assertTrue(payload["repairExecutions"]["anomalies"]["outOfOrderActions"])
        self.assertTrue(payload["repairExecutions"]["anomalies"]["boundaryViolations"])
        self.assertEqual(payload["cross"]["anomalies"]["checkpointViolations"], [])

    def test_structurally_damaged_repair_record_is_group_only(self) -> None:
        self.put_repair(
            "peer-a",
            "exec-1",
            repair_binding(expected_receipts="not-hex"),
        )
        payload = self.verify()
        self.assertTrue(payload["repairExecutions"]["anomalies"]["recordViolations"])
        # The malformed record is skipped rather than re-reported cross.
        self.assertEqual(payload["cross"]["anomalies"]["checkpointViolations"], [])


class IntegrityCrossHelperTests(unittest.TestCase):
    def test_transaction_helper_empty_inputs(self) -> None:
        self.assertEqual(_integrity_cross_transaction_mismatches_locked([], []), [])

    def test_transaction_helper_flags_missing_identity(self) -> None:
        accepted = [
            ("r1", operation("o1", "k1", "v", {"r1": 1})),
        ]
        history = [("tx-1", [tx_entry("k9", "r9", "g9")])]
        markers = _integrity_cross_transaction_mismatches_locked(history, accepted)
        self.assertEqual(len(markers), 1)
        self.assertIsNone(markers[0]["position"])
        self.assertIsNone(markers[0]["expected"])

    def test_transaction_helper_skips_damaged_records(self) -> None:
        accepted = [("r1", operation("o1", "k1", "v", {"r1": 1}))]
        history = [("tx-bad", "not-a-list")]
        self.assertEqual(
            _integrity_cross_transaction_mismatches_locked(history, accepted), []
        )


class IntegrityRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_restart_reproduces_the_same_report(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        status, _, _, _, error = store.apply_transaction(
            "tx-1", [tx_entry("k1", "r1", "o1")]
        )
        # The operation already exists, so the batch is a pure replay (200)
        # that still commits the transaction binding.
        self.assertEqual(status, HTTPStatus.OK, error)
        status, _ = store.save_checkpoint("peer-a", 1)
        self.assertIs(status, HTTPStatus.OK)
        # Inject a well-formed confirmed repair binding directly so the
        # report contains a repair execution without the full receipt setup.
        store._repairs[("peer-a", "exec-1")] = repair_binding(
            cursor=1, expected_checkpoint=1
        )
        store._persist_locked()

        status, before = StateStore(data_file=self.data_file).get_integrity_verify()
        self.assertIs(status, HTTPStatus.OK)
        status, after = StateStore(data_file=self.data_file).get_integrity_verify()
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(after, before)
        self.assertEqual(after["transactions"]["count"], 1)
        self.assertEqual(after["repairExecutions"]["count"], 1)
        self.assertEqual(after["status"], "ok")

    def test_old_file_recovers_two_empty_intact_groups(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        status, payload = StateStore(data_file=self.data_file).get_integrity_verify()
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["transactions"]["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["transactions"]["count"], 0)
        self.assertEqual(payload["repairExecutions"]["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["repairExecutions"]["count"], 0)
        self.assertEqual(payload["cross"]["status"], "ok")


class IntegrityHttpFixture(unittest.TestCase):
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
            conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def get(self, query: str = ""):
        return self.raw_request("GET", VERIFY_PATH + query)


class IntegrityHttpTests(IntegrityHttpFixture):
    def test_empty_report_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(list(payload), ["status", "transactions", "repairExecutions", "cross"])
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            raw,
            b'{"status":"ok","transactions":{"algorithm":"sha256","digest":"'
            + EMPTY_DIGEST.encode("ascii")
            + b'","count":0,"anomalies":{"duplicateTransactionIds":[],'
            b'"recordViolations":[],"identityMismatches":[],"batchViolations":[]}},'
            b'"repairExecutions":{"algorithm":"sha256","digest":"'
            + EMPTY_DIGEST.encode("ascii")
            + b'","count":0,"anomalies":{"duplicateBindings":[],'
            b'"outOfOrderActions":[],"boundaryViolations":[],'
            b'"checkpointViolations":[],"recordViolations":[]}},'
            b'"cross":{"status":"ok","anomalies":{"identityMismatches":[],'
            b'"checkpointViolations":[]}}}\n',
        )

    def test_any_query_parameter_is_400(self) -> None:
        bad_queries = [
            "?x=1",
            "?x=",
            "?x",
            "?=1",
            "?status=ok",
            "?x=1&y=2",
            "?x=1&x=2",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_empty_query_string_is_accepted(self) -> None:
        # A bare "?" carries no parameters exactly like no query string.
        status, _, _, _ = self.get("?")
        self.assertEqual(status, 200)

    def test_route_shape_mismatches_are_404_before_query_check(self) -> None:
        bad_paths = [
            VERIFY_PATH + "/",
            VERIFY_PATH + "/extra",
            "/v1/integrity",
            "/v1/integrity/verif",
            "/v1/integrity/verify/more",
            "/v1/integrit/verify",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request("GET", path + "?x=1")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_post_on_the_path_is_404(self) -> None:
        # The service implements GET and POST verbs; a POST on this
        # GET-only route is an unknown route and answers 404.
        status, payload, _, _ = self.raw_request("POST", VERIFY_PATH, {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_read_only_and_snapshot_stable(self) -> None:
        self.server.store.apply_operation(
            "r1", operation("o1", "k1", "v", {"r1": 1})
        )
        status, first, _, _ = self.get()
        self.assertEqual(status, 200)
        for query in ("", "?x=1", "?x="):
            self.get(query)
        status, second, _, _ = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(second, first)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        self.get()
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)

    def test_identity_mismatch_is_reported_over_http(self) -> None:
        store = self.server.store
        store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        store.apply_transaction("tx-1", [tx_entry("k1", "r1", "o1")])
        store._transactions["tx-1"] = [tx_entry("k1", "r1", "o1", value="z")]
        status, payload, _, _ = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "broken")
        self.assertEqual(payload["cross"]["status"], "broken")
        self.assertEqual(
            payload["cross"]["anomalies"]["identityMismatches"][0]["position"], 0
        )


class IntegrityPersistenceHttpTests(IntegrityHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=self.data_path)

    def test_query_persists_nothing(self) -> None:
        self.server.store.apply_operation(
            "r1", operation("o1", "k1", "v", {"r1": 1})
        )
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in ("", "?x=1", "?x", "?=1"):
            self.get(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class IntegrityAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-auth-")
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
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra?x=1",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
