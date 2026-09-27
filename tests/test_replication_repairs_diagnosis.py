"""Tests for the read-only replication-repair lifecycle diagnosis::

    POST /v1/replication/repairs/diagnosis

The endpoint takes the exact conditional-execution request body shared by
the preflight (``POST /v1/replication/repairs/plan``) and the committing
execution (``POST /v1/replication/repairs/apply``) and links the four
lifecycle stages — advice, preflight, execution, audit — into one
evidence report read entirely from one committed snapshot.

The success body is compact ordered UTF-8 JSON terminated by a single
newline with an explicit Content-Length; its top-level fields are
``status`` (always ``"diagnosed"``), ``execution``, ``links``,
``summary``, and ``conclusion``. ``execution`` is ``null`` until the
``(peerId, ackId)`` execution commits; once committed it carries the
commit position, expected anchors, restored cursor, and binding-match
flag. Each link (one per requested suggestion, in request order) carries
``action``, ``ackId``, ``location``, ``target``, ``boundary``,
``evidence``, and ``status`` — ``current``, ``superseded`` (the slot
exists but the log location moved), or ``unavailable`` (the same
``(action, ackId)`` group no longer holds the slot ordinal). The
conclusion is ``ready``/``unexecuted`` before commit and ``current``/
``superseded``/``broken`` after it; the summary counts the suggestions,
the three link statuses, the executed flag, and the historical anomaly
total. The tests cover the store semantics, the HTTP precedence chain
(404 path/shape, 400 query/body/length, 413 oversize before the body is
read, 401 with a Bearer challenge, 403 without one), the compact body,
restart consistency, and the strict read-only guarantee. Only the
Python standard library is used.
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

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _repair_receipts_digest,
    load_scope_policy,
)

DIAGNOSIS_PATH = "/v1/replication/repairs/diagnosis"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}


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


class DiagnosisStoreFixture(unittest.TestCase):
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

    def seed_gap(self, peer: str, *, first: str = "ack-1", second: str = "ack-2") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, first, 1, [identity("r0", "o0")])
        self.seed_receipt(peer, second, 3, [identity("r2", "o2")])

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

    def diagnose(self, peer: str, ack_id: str, suggestions, *, checkpoint=4, digest=None):
        return self.store.diagnose_replication_repairs(
            peer,
            ack_id,
            checkpoint,
            digest if digest is not None else self.receipts_digest(peer),
            suggestions,
        )


class DiagnosisStoreTests(DiagnosisStoreFixture):
    def test_top_level_field_order_and_fixed_status(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        payload = self.diagnose("peer-a", "exec-1", [self.suggestion(item)])
        self.assertEqual(
            list(payload),
            ["status", "execution", "links", "summary", "conclusion"],
        )
        self.assertEqual(payload["status"], "diagnosed")

    def test_uncommitted_execution_is_null_ready_when_all_slots_current(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        payload = self.diagnose("peer-a", "exec-1", [self.suggestion(item)])
        self.assertIsNone(payload["execution"])
        self.assertEqual(payload["conclusion"], "ready")
        link = payload["links"][0]
        self.assertEqual(
            list(link),
            ["action", "ackId", "location", "target", "boundary", "evidence", "status"],
        )
        self.assertEqual(link["action"], "resend")
        self.assertEqual(link["ackId"], "ack-2")
        self.assertEqual(link["location"], {"start": 1, "end": 2})
        self.assertEqual(link["target"], {"start": 1, "end": 2})
        self.assertEqual(link["boundary"], {"start": 1, "end": 2})
        self.assertIsNone(link["evidence"])
        self.assertEqual(link["status"], "current")
        self.assertEqual(
            payload["summary"],
            {
                "suggestions": 1,
                "current": 1,
                "superseded": 0,
                "unavailable": 0,
                "executed": 0,
                "anomalies": 0,
            },
        )

    def test_unregistered_peer_is_unexecuted_with_unavailable_links(self) -> None:
        self.seed_operations(1)
        suggestion = interval("resend", "ack-2", 1, 2)
        payload = self.diagnose(
            "peer-a", "exec-1", [suggestion], checkpoint=0, digest=hashlib.sha256(b"[]").hexdigest()
        )
        self.assertIsNone(payload["execution"])
        self.assertEqual(payload["conclusion"], "unexecuted")
        self.assertEqual(payload["links"][0]["status"], "unavailable")
        self.assertIsNone(payload["links"][0]["boundary"])
        self.assertEqual(payload["summary"]["unavailable"], 1)
        self.assertEqual(payload["summary"]["executed"], 0)

    def test_moved_location_before_commit_is_superseded_and_unexecuted(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        moved = {**self.suggestion(item), "target": {"start": 1, "end": 3}}
        payload = self.diagnose("peer-a", "exec-1", [moved])
        self.assertIsNone(payload["execution"])
        self.assertEqual(payload["links"][0]["status"], "superseded")
        # The observed boundary is the live advice target, which differs.
        self.assertEqual(payload["links"][0]["boundary"], {"start": 1, "end": 2})
        self.assertEqual(payload["conclusion"], "unexecuted")
        self.assertEqual(payload["summary"]["superseded"], 1)

    def test_missing_slot_before_commit_is_unavailable_and_unexecuted(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        ghost = interval("resend", "ack-9", 5, 6)
        payload = self.diagnose("peer-a", "exec-1", [ghost])
        self.assertEqual(payload["links"][0]["status"], "unavailable")
        self.assertEqual(payload["conclusion"], "unexecuted")

    def test_committed_execution_reports_position_anchors_cursor_and_match(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        status, _, error = self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        self.assertIs(status, HTTPStatus.CREATED, error)
        payload = self.diagnose("peer-a", "exec-1", [suggestion])
        execution = payload["execution"]
        self.assertEqual(execution["position"], 0)
        self.assertEqual(execution["expectedCheckpoint"], 4)
        self.assertEqual(execution["expectedReceipts"], self.receipts_digest("peer-a"))
        self.assertEqual(execution["cursor"], 4)
        self.assertTrue(execution["bindingMatch"])
        link = payload["links"][0]
        self.assertEqual(link["status"], "current")
        self.assertEqual(link["boundary"], {"start": 1, "end": 2})
        self.assertEqual(link["evidence"], {"execution": 0, "result": 0})
        self.assertEqual(payload["conclusion"], "current")
        self.assertEqual(payload["summary"]["executed"], 1)

    def test_committed_execution_position_is_creation_index(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        item_a = self.advice("peer-a")[0]
        item_b = self.advice("peer-b")[0]
        sug_a = self.suggestion(item_a)
        sug_b = self.suggestion(item_b)
        self.store.apply_replication_repairs(
            "peer-a", "exec-a1", 4, self.receipts_digest("peer-a"), [sug_a]
        )
        self.store.apply_replication_repairs(
            "peer-b", "exec-b1", 4, self.receipts_digest("peer-b"), [sug_b]
        )
        self.store.apply_replication_repairs(
            "peer-a", "exec-a2", 4, self.receipts_digest("peer-a"), [sug_a]
        )
        positions = {
            ("peer-a", "exec-a1"): 0,
            ("peer-b", "exec-b1"): 1,
            ("peer-a", "exec-a2"): 2,
        }
        for (peer, ack_id), expected in positions.items():
            payload = self.diagnose(peer, ack_id, [sug_a if peer == "peer-a" else sug_b])
            self.assertEqual(payload["execution"]["position"], expected)
            self.assertEqual(
                payload["links"][0]["evidence"]["execution"], expected
            )

    def test_committed_with_a_moved_target_is_superseded(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        moved = {**suggestion, "target": {"start": 1, "end": 3}}
        payload = self.diagnose("peer-a", "exec-1", [moved])
        self.assertFalse(payload["execution"]["bindingMatch"])
        self.assertEqual(payload["links"][0]["status"], "superseded")
        self.assertEqual(payload["links"][0]["boundary"], {"start": 1, "end": 2})
        self.assertEqual(payload["conclusion"], "superseded")

    def test_committed_with_missing_group_slot_is_superseded(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        ghost = interval("resend", "ack-9", 5, 6)
        payload = self.diagnose("peer-a", "exec-1", [suggestion, ghost])
        self.assertEqual([link["status"] for link in payload["links"]], ["current", "unavailable"])
        self.assertEqual(payload["conclusion"], "superseded")
        self.assertIsNone(payload["links"][1]["evidence"])

    def test_historical_anomaly_makes_the_conclusion_broken(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        # Roll the registered checkpoint behind the restored cursor without
        # touching the execution record: the audit flags a cursor violation.
        self.store._checkpoints["peer-a"] = 0
        payload = self.diagnose("peer-a", "exec-1", [suggestion])
        self.assertEqual(payload["conclusion"], "broken")
        self.assertEqual(payload["summary"]["anomalies"], 1)
        # The request's own link is still aligned against the committed
        # result; the breakage comes from the independent history scan.
        self.assertEqual(payload["links"][0]["status"], "current")

    def test_structural_history_damage_is_broken(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        self.store._repairs[("peer-a", "exec-1")]["expectedReceipts"] = "zz"
        payload = self.diagnose("peer-a", "exec-1", [suggestion])
        self.assertEqual(payload["conclusion"], "broken")
        self.assertGreaterEqual(payload["summary"]["anomalies"], 1)

    def test_a_tampered_result_record_is_broken_without_raising(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        # Corrupt the stored result shape: the history scan flags a record
        # violation and the diagnosis must still return a report (broken),
        # never a 500, with the damaged slot unavailable.
        self.store._repairs[("peer-a", "exec-1")]["results"] = [
            {"action": "deduplicate", "boundary": {"start": 1, "end": 2}}
        ]
        payload = self.diagnose("peer-a", "exec-1", [suggestion])
        self.assertEqual(payload["conclusion"], "broken")
        self.assertGreaterEqual(payload["summary"]["anomalies"], 1)
        self.assertEqual(payload["links"][0]["status"], "unavailable")

    def test_ordinal_aligns_repeated_same_slot_suggestions(self) -> None:
        # Two suggestions for the same (action, ackId) slot consume the
        # group's entries at ordinal 0 and 1; a request with two such
        # entries where the group holds one marks the second unavailable.
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        payload = self.diagnose("peer-a", "exec-1", [suggestion, suggestion])
        self.assertEqual(
            [link["status"] for link in payload["links"]],
            ["current", "unavailable"],
        )

    def test_diagnosis_is_strictly_read_only(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        self.store.apply_replication_repairs(
            "peer-a", "exec-1", 4, self.receipts_digest("peer-a"), [suggestion]
        )
        repairs_before = copy.deepcopy(dict(self.store._repairs))
        checkpoints_before = dict(self.store._checkpoints)
        acks_before = copy.deepcopy(dict(self.store._acks))
        metrics_before = self.store.get_metrics()
        advice_before = self.store.get_replication_repairs(0, 100)[1]
        for _ in range(3):
            self.diagnose("peer-a", "exec-1", [suggestion])
            self.diagnose("peer-a", "exec-new", [suggestion])
            self.diagnose("peer-a", "exec-1", [{**suggestion, "target": {"start": 1, "end": 3}}])
        self.assertEqual(self.store._repairs, repairs_before)
        self.assertEqual(self.store._checkpoints, checkpoints_before)
        self.assertEqual(self.store._acks, acks_before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_replication_repairs(0, 100)[1], advice_before)

    def test_repeated_diagnosis_is_deterministic(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        item = self.advice("peer-a")[0]
        suggestion = self.suggestion(item)
        first = self.diagnose("peer-a", "exec-1", [suggestion])
        second = self.diagnose("peer-a", "exec-1", [suggestion])
        self.assertEqual(first, second)


class DiagnosisRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-diagnosis-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_path = os.path.join(self.tmpdir, "state.json")

    def _fresh_store(self) -> StateStore:
        return StateStore(data_file=self.data_path)

    def test_diagnosis_is_identical_after_restart(self) -> None:
        store = self._fresh_store()
        for index in range(4):
            store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
        store.save_checkpoint("peer-a", 4)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        status, advice = store.get_replication_repairs(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        item = advice["suggestions"][0]
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        committed = [
            (ack_id, receipt)
            for (peer, ack_id), receipt in store._acks.items()
            if peer == "peer-a"
        ]
        digest = _repair_receipts_digest("peer-a", committed)
        store.apply_replication_repairs("peer-a", "exec-1", 4, digest, [suggestion])
        before = store.diagnose_replication_repairs(
            "peer-a", "exec-1", 4, digest, [suggestion]
        )
        reopened = self._fresh_store()
        after = reopened.diagnose_replication_repairs(
            "peer-a", "exec-1", 4, digest, [suggestion]
        )
        self.assertEqual(after, before)
        self.assertEqual(after["conclusion"], "current")
        self.assertEqual(after["execution"]["position"], 0)


class DiagnosisHttpFixture(unittest.TestCase):
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

    def raw_request(self, method, path, body=None, headers=None, *, raw_body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        payload = raw_body if raw_body is not None else (
            json.dumps(body) if body is not None else None
        )
        if payload is None:
            conn.request(method, path, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=payload, headers=request_headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        parsed = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, parsed, raw, response_headers

    def post(self, body, query: str = "", headers=None):
        return self.raw_request(
            "POST", DIAGNOSIS_PATH + query, body=body, headers=headers
        )

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

    def seed_gap(self, peer: str = "peer-a") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt(peer, "ack-2", 3, [identity("r2", "o2")])

    def request_body(self, peer: str = "peer-a", ack_id: str = "exec-1", **overrides) -> dict:
        status, receipts, _, _ = self.raw_request(
            "GET", f"/v1/sync/peers/{peer}/receipts?after=0&limit=100"
        )
        assert status == 200
        status, advice, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=0&limit=100"
        )
        assert status == 200
        item = next(entry for entry in advice["suggestions"] if entry["peer"] == peer)
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        body = {
            "peerId": peer,
            "ackId": ack_id,
            "expectedCheckpoint": self.server.store.get_checkpoint(peer)[1]["cursor"],
            "expectedReceipts": receipts["digest"],
            "suggestions": [suggestion],
        }
        body.update(overrides)
        return body


class DiagnosisHttpTests(DiagnosisHttpFixture):
    def test_uncommitted_success_is_compact_ordered_single_newline(self) -> None:
        body = {
            "peerId": "peer-a",
            "ackId": "exec-1",
            "expectedCheckpoint": 0,
            "expectedReceipts": hashlib.sha256(b"[]").hexdigest(),
            "suggestions": [interval("resend", "ack-2", 1, 2)],
        }
        status, payload, raw, headers = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            ["status", "execution", "links", "summary", "conclusion"],
        )
        self.assertEqual(payload["status"], "diagnosed")
        self.assertIsNone(payload["execution"])
        self.assertEqual(payload["conclusion"], "unexecuted")
        self.assertEqual(
            list(payload["links"][0]),
            ["action", "ackId", "location", "target", "boundary", "evidence", "status"],
        )
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_ready_then_current_across_the_apply(self) -> None:
        self.seed(4)
        self.seed_gap()
        body = self.request_body()
        status, before, _, _ = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(before["conclusion"], "ready")
        self.assertIsNone(before["execution"])
        status, applied, _, _ = self.raw_request(
            "POST", "/v1/replication/repairs/apply", body=body
        )
        self.assertEqual(status, 201, applied)
        status, after, _, _ = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(after["conclusion"], "current")
        self.assertEqual(after["execution"]["position"], 0)
        self.assertTrue(after["execution"]["bindingMatch"])
        self.assertEqual(after["links"][0]["evidence"], {"execution": 0, "result": 0})

    def test_non_post_and_wrong_shapes_are_404(self) -> None:
        body = {"x": 1}
        # The service implements GET and POST only; the "non-POST answers
        # 404" contract is exercised with GET exactly like the plan/apply
        # routes' tests. Wrong shapes are 404 under POST.
        for method, path in (
            ("GET", DIAGNOSIS_PATH),
            ("POST", DIAGNOSIS_PATH + "/"),
            ("POST", DIAGNOSIS_PATH + "/extra"),
            ("POST", "/v1/replication/repairs/diagnosi"),
            ("POST", "/v1/replication/repair/diagnosis"),
            ("POST", "/v1/replication/diagnosis"),
        ):
            with self.subTest(method=method, path=path):
                status, payload, _, _ = self.raw_request(method, path, body=body)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_illegal_query_is_400_before_body_validation(self) -> None:
        for query in ("?x=1", "?after=0", "?x=1&y=2"):
            with self.subTest(query=query):
                status, payload, raw, _ = self.post({"not": "valid"}, query=query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_malformed_bodies_are_400(self) -> None:
        good = {
            "peerId": "peer-a",
            "ackId": "exec-1",
            "expectedCheckpoint": 0,
            "expectedReceipts": hashlib.sha256(b"[]").hexdigest(),
            "suggestions": [interval("resend", "ack-2", 1, 2)],
        }
        bad_bodies = [
            {},
            [],
            {"nope": 1},
            {**good, "extra": 1},
            {k: v for k, v in good.items() if k != "peerId"},
            {**good, "peerId": ""},
            {**good, "expectedCheckpoint": -1},
            {**good, "expectedCheckpoint": True},
            {**good, "expectedReceipts": "zz"},
            {**good, "suggestions": []},
            {**good, "suggestions": [{"action": "nope", "ackId": "a"}]},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload, _, _ = self.post(body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    @staticmethod
    def _raw_status_and_body(sock) -> tuple[bytes, bytes]:
        data = b""
        while b"\r\n\r\n" not in data:
            data += sock.recv(4096)
        head, _, body = data.partition(b"\r\n\r\n")
        # Read the declared remainder so the error body is fully received.
        declared = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                declared = int(line.split(b":", 1)[1].strip())
        while len(body) < declared:
            chunk = sock.recv(4096)
            if not chunk:
                break
            body += chunk
        return head.split(b"\r\n", 1)[0], body

    def test_missing_content_length_is_400_without_reading_body(self) -> None:
        import socket

        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(
            b"POST " + DIAGNOSIS_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n\r\n"
        )
        status_line, body = self._raw_status_and_body(sock)
        sock.close()
        self.assertIn(b"400", status_line)
        self.assertEqual(body, b'{"error":"invalid_request"}')

    def test_oversize_declaration_is_413_before_body_read(self) -> None:
        import socket

        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(
            b"POST " + DIAGNOSIS_PATH.encode()
            + b" HTTP/1.1\r\nHost: x\r\nContent-Length: 2000000\r\n\r\n"
        )
        status_line, body = self._raw_status_and_body(sock)
        sock.close()
        self.assertIn(b"413", status_line)
        self.assertEqual(body, b'{"error":"payload_too_large"}')

    def test_query_is_strictly_read_only_over_http(self) -> None:
        self.seed(4)
        self.seed_gap()
        body = self.request_body()
        status, first, _, _ = self.post(body)
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        checkpoint_before = self.server.store.get_checkpoint("peer-a")[1]
        repairs_before = copy.deepcopy(dict(self.server.store._repairs))
        for _ in range(3):
            self.post(body)
        status, second, _, _ = self.post(body)
        self.assertEqual(second, first)
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(self.server.store.get_checkpoint("peer-a")[1], checkpoint_before)
        self.assertEqual(self.server.store._repairs, repairs_before)


class DiagnosisPersistenceHttpTests(DiagnosisHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-diagnosis-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_diagnosis_persists_nothing(self) -> None:
        self.seed(4)
        self.seed_gap()
        body = self.request_body()
        self.raw_request("POST", "/v1/replication/repairs/apply", body=body)
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for _ in range(3):
            self.post(body)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class DiagnosisAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-diagnosis-auth-")
        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(SCOPE_POLICY, handle)
        cls.auth_scopes = load_scope_policy(cls.policy_path)
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=cls.auth_scopes,
            scope_policy_file=cls.policy_path,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, True)

    def setUp(self) -> None:
        self.server.store = StateStore()

    def request(self, headers=None, *, body: object = b"{}"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            DIAGNOSIS_PATH,
            body=body if isinstance(body, bytes) else json.dumps(body),
            headers=headers or {},
        )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def test_missing_header_is_401_with_challenge_and_unread_body(self) -> None:
        status, raw, headers = self.request()
        self.assertEqual(status, 401)
        self.assertIn(b'"unauthorized"', raw)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_malformed_and_mismatched_tokens_are_401(self) -> None:
        for value in ("Bearer", "Bearer  ", "Token reader-token", f"Bearer other"):
            with self.subTest(value=value):
                status, _, headers = self.request({"Authorization": value})
                self.assertEqual(status, 401)
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_write_only_scope_is_403_without_challenge(self) -> None:
        status, raw, headers = self.request(
            {"Authorization": f"Bearer {WRITE_TOKEN}"}
        )
        self.assertEqual(status, 403)
        self.assertIn(b'"forbidden"', raw)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_read_scope_is_allowed(self) -> None:
        status, _, headers = self.request(
            {"Authorization": f"Bearer {READ_TOKEN}"}
        )
        # The empty body fails structurally, proving the read scope passed
        # the gate; it must be a 400, never the write-only 403.
        self.assertEqual(status, 400)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_health_stays_anonymous(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(raw)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
