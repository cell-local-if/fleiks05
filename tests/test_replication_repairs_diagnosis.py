"""Tests for the read-only replication-repair lifecycle diagnosis::

    POST /v1/replication/repairs/diagnosis

The body is the same conditional-execution request the preflight and the
committing apply share — ``peerId``, an ``ackId`` naming the execution,
the ``expectedCheckpoint`` the peer currently holds, the
``expectedReceipts`` digest, and the ordered ``suggestions`` — and
``(peerId, ackId)`` locks the one execution under diagnosis. The endpoint
chains the suggestion, preflight, execution, and audit stages into
evidence but writes nothing.

The response carries ``status`` (always ``"diagnosed"``), ``execution``
(``null`` before the execution commits; otherwise the creation-order
position, the expected anchor, the restored cursor, and whether the
presented lock matches the binding), one ``link`` per suggestion with an
action/receipt/location/target, the current boundary and evidence, and a
``current``/``superseded``/``unavailable`` status, a ``summary`` with the
suggestion count, the three link-status counts, the executed count, and
the whole-history anomaly count, and a ``conclusion``: ``ready`` vs
``unexecuted`` before commit (all links current vs not) and ``current``
vs ``superseded`` vs ``broken`` after commit (all current; any moved or
missing link; or an integrity anomaly implicating this execution).

The tests cover the response shape and compact single-newline encoding,
the uncommitted/committed conclusion matrix, the three link statuses, the
anchor and binding match, the five-class history anomaly count, the HTTP
precedence chain (404 path/non-POST, 400 query/body, 400/413 declared
length before authentication and without reading the body, 401 with a
Bearer challenge, 403 in scope mode without one), health staying
anonymous, the strict read-only and snapshot guarantees, and restart
consistency with ``--data-file``. Only the Python standard library is
used.
"""

from __future__ import annotations

import copy
import http.client
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
    _repair_receipts_digest,
    load_scope_policy,
)

DIAGNOSIS_PATH = "/v1/replication/repairs/diagnosis"
PLAN_PATH = "/v1/replication/repairs/plan"
APPLY_PATH = "/v1/replication/repairs/apply"
REPAIRS_PATH = "/v1/replication/repairs?after=0&limit=100"

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


def identity_fix(
    ack_id: str,
    position: int,
    expected: dict | None,
    observed: dict,
) -> dict:
    return {
        "action": "correct_identity",
        "ackId": ack_id,
        "location": {"position": position},
        "target": {"position": position},
        "expected": expected,
        "observed": observed,
    }


def suggestion_from_advice(item: dict) -> dict:
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
        self.server.store = type(self.server.store)()

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

    def request(self, method, path, body=None, headers=None):
        status, payload, _, _ = self.raw_request(method, path, body, headers)
        return status, payload

    def seed(self, count: int = 4) -> None:
        for index in range(count):
            status, _ = self.request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status == 200

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.server.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def advice(self, peer: str = "peer-a") -> list[dict]:
        status, payload = self.request("GET", REPAIRS_PATH)
        assert status == 200
        return [item for item in payload["suggestions"] if item["peer"] == peer]

    def receipt_digest(self, peer: str = "peer-a") -> str:
        status, payload = self.request(
            "GET", f"/v1/sync/peers/{peer}/receipts?after=0&limit=100"
        )
        assert status == 200
        return payload["digest"]

    def body(
        self,
        suggestions: list[dict],
        *,
        peer: str = "peer-a",
        ack_id: str = "exec-1",
        checkpoint: int | None = None,
        digest: str | None = None,
    ) -> dict:
        cursor = checkpoint
        if cursor is None:
            status, payload = self.request(
                "GET", f"/v1/sync/peers/{peer}/checkpoint"
            )
            assert status == 200
            cursor = payload["cursor"]
        return {
            "peerId": peer,
            "ackId": ack_id,
            "expectedCheckpoint": cursor,
            "expectedReceipts": digest or self.receipt_digest(peer),
            "suggestions": suggestions,
        }

    def gap_scenario(self) -> dict:
        """A peer whose receipts leave one gap, one resend suggestion."""
        self.seed(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        return self.body([suggestion_from_advice(self.advice()[0])])


class DiagnosisShapeTests(DiagnosisHttpFixture):
    def test_uncommitted_response_shape_is_compact_ordered_with_newline(self) -> None:
        document = self.gap_scenario()
        status, payload, raw, headers = self.raw_request(
            "POST", DIAGNOSIS_PATH, document
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            ["status", "execution", "links", "summary", "conclusion"],
        )
        self.assertEqual(payload["status"], "diagnosed")
        self.assertIsNone(payload["execution"])
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_link_carries_the_seven_fields_in_order(self) -> None:
        document = self.gap_scenario()
        status, payload, _, _ = self.raw_request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
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
        self.assertEqual(
            link["evidence"],
            {
                "location": {"start": 1, "end": 2},
                "target": {"start": 1, "end": 2},
            },
        )
        self.assertEqual(link["status"], "current")

    def test_summary_field_order_and_counts_when_uncommitted(self) -> None:
        document = self.gap_scenario()
        status, payload, _, _ = self.raw_request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload["summary"]),
            ["suggestions", "current", "superseded", "unavailable", "executed", "anomalies"],
        )
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

    def test_identity_link_evidence_carries_expected_and_observed(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        document = self.body([suggestion_from_advice(self.advice()[0])])
        status, payload, _, _ = self.raw_request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        link = payload["links"][0]
        self.assertEqual(link["action"], "correct_identity")
        self.assertEqual(list(link["evidence"]), ["location", "target", "expected", "observed"])
        self.assertEqual(link["evidence"]["location"], {"position": 1})
        self.assertEqual(link["evidence"]["target"], {"position": 1})
        self.assertEqual(link["evidence"]["expected"], identity("r1", "o1"))
        self.assertEqual(link["evidence"]["observed"], identity("r9", "bogus"))


class DiagnosisConclusionTests(DiagnosisHttpFixture):
    def test_uncommitted_all_current_is_ready(self) -> None:
        document = self.gap_scenario()
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "ready")
        self.assertIsNone(payload["execution"])

    def test_uncommitted_with_unavailable_slot_is_unexecuted(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        # A suggestion naming a receipt group that does not currently exist:
        # the peer's advice is empty, so the slot is missing.
        document = self.body([interval("resend", "ack-missing", 0, 1)])
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "unexecuted")
        self.assertIsNone(payload["execution"])
        self.assertEqual(payload["links"][0]["status"], "unavailable")
        self.assertIsNone(payload["links"][0]["boundary"])
        self.assertIsNone(payload["links"][0]["evidence"])
        self.assertEqual(payload["summary"]["current"], 0)
        self.assertEqual(payload["summary"]["unavailable"], 1)

    def test_committed_all_current_is_current(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "current")
        execution = payload["execution"]
        self.assertEqual(
            list(execution),
            [
                "position",
                "expectedCheckpoint",
                "expectedReceipts",
                "cursor",
                "bindingMatches",
            ],
        )
        self.assertEqual(execution["position"], 0)
        self.assertEqual(execution["expectedCheckpoint"], 4)
        self.assertEqual(execution["expectedReceipts"], document["expectedReceipts"])
        self.assertEqual(execution["cursor"], 4)
        self.assertTrue(execution["bindingMatches"])
        self.assertEqual(payload["summary"]["executed"], 1)

    def test_committed_slot_gone_is_superseded_with_unavailable_links(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        # Fill the gap so the resend advice disappears: the committed link's
        # slot is now missing from the same action/receipt group.
        self.seed_receipt(
            "peer-a", "ack-2", 3, [identity("r1", "o1"), identity("r2", "o2")]
        )
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "superseded")
        self.assertEqual(payload["links"][0]["status"], "unavailable")
        self.assertIsNone(payload["links"][0]["boundary"])
        self.assertIsNone(payload["links"][0]["evidence"])
        self.assertEqual(payload["summary"]["current"], 0)
        self.assertEqual(payload["summary"]["unavailable"], 1)
        # The execution remains committed; the anchor/cursor are reported.
        self.assertIsNotNone(payload["execution"])
        self.assertEqual(payload["execution"]["cursor"], 4)

    def test_committed_identity_moved_is_superseded_with_new_evidence(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        document = self.body([suggestion_from_advice(self.advice()[0])])
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        # The same receipt's mismatch moves from position 1 to position 0:
        # the (action, ackId, ordinal) slot still exists but its location
        # and the log identity it holds have moved.
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r9", "bogus"), identity("r1", "o1")],
        )
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "superseded")
        link = payload["links"][0]
        self.assertEqual(link["status"], "superseded")
        self.assertEqual(link["location"], {"position": 1})
        self.assertEqual(link["boundary"], {"position": 0})
        self.assertEqual(link["evidence"]["location"], {"position": 0})
        self.assertEqual(link["evidence"]["expected"], identity("r0", "o0"))
        self.assertEqual(link["evidence"]["observed"], identity("r9", "bogus"))
        self.assertEqual(payload["summary"]["superseded"], 1)

    def test_binding_mismatch_does_not_alone_change_conclusion(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        changed = dict(document)
        changed["expectedCheckpoint"] = 3
        status, payload = self.request("POST", DIAGNOSIS_PATH, changed)
        self.assertEqual(status, 200)
        # The presented lock no longer matches, but the committed
        # suggestion is still current against the present advice, so the
        # conclusion is current; bindingMatches simply reports False.
        self.assertFalse(payload["execution"]["bindingMatches"])
        self.assertEqual(payload["conclusion"], "current")

    def test_broken_when_history_anomaly_implicates_this_execution(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        # Damage this execution's stored record: its restored boundary no
        # longer matches its suggestion target, a boundary violation.
        binding = self.server.store._repairs[("peer-a", "exec-1")]
        binding["results"] = [
            {"action": "resend", "boundary": {"start": 9, "end": 9}}
        ]
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "broken")
        self.assertGreaterEqual(payload["summary"]["anomalies"], 1)

    def test_another_executions_anomaly_does_not_implicate_this_one(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        # Add a second, structurally damaged binding under another ackId.
        good = self.server.store._repairs[("peer-a", "exec-1")]
        bad = copy.deepcopy(good)
        bad["results"] = [
            {"action": "resend", "boundary": {"start": 9, "end": 9}}
        ]
        self.server.store._repairs[("peer-a", "exec-bad")] = bad
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        # The history reports the anomaly, but exec-1 itself is intact and
        # its link is still current, so its conclusion is not broken.
        self.assertGreaterEqual(payload["summary"]["anomalies"], 1)
        self.assertEqual(payload["conclusion"], "current")
        # The damaged sibling diagnoses as broken at the next position.
        sibling = dict(document)
        sibling["ackId"] = "exec-bad"
        status, payload = self.request("POST", DIAGNOSIS_PATH, sibling)
        self.assertEqual(payload["execution"]["position"], 1)
        self.assertEqual(payload["conclusion"], "broken")


class DiagnosisMultiLinkTests(DiagnosisHttpFixture):
    def test_links_follow_suggestion_order_with_per_slot_counts(self) -> None:
        self.seed(5)
        self.register("peer-a", 5)
        # Three seamless-with-gaps receipts: ack-1 anchors [0,1), then
        # ack-2 confirms [2,3) (gap [1,2)) and ack-3 confirms [4,5)
        # (gap [3,4)), producing two resend suggestions for two receipts.
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        self.seed_receipt("peer-a", "ack-3", 5, [identity("r4", "o4")])
        advice = self.advice()
        self.assertEqual(len(advice), 2)
        suggestions = [suggestion_from_advice(item) for item in advice]
        document = self.body(suggestions)
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["conclusion"], "ready")
        self.assertEqual(
            [link["ackId"] for link in payload["links"]], ["ack-2", "ack-3"]
        )
        self.assertEqual(
            [link["status"] for link in payload["links"]], ["current", "current"]
        )
        self.assertEqual(payload["summary"]["suggestions"], 2)
        self.assertEqual(payload["summary"]["current"], 2)
        # Apply and diagnose: the committed link order and executed count
        # follow the suggestion batch.
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        status, payload = self.request("POST", DIAGNOSIS_PATH, document)
        self.assertEqual(payload["conclusion"], "current")
        self.assertEqual(payload["summary"]["executed"], 2)
        self.assertEqual(payload["execution"]["cursor"], 5)


class DiagnosisRequestContractTests(DiagnosisHttpFixture):
    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/repairs",
            "/v1/replication/repairs/",
            "/v1/replication/repairs/diagnosis/",
            "/v1/replication/repairs/diagnosis/extra",
            "/v1/replication/repairs/other",
        ):
            status, payload, _, _ = self.raw_request("POST", f"{path}?x=%ZZ")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_diagnosis_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request("GET", DIAGNOSIS_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_parameter_is_400_before_body_check(self) -> None:
        # A declared-but-unwritten body: the query check precedes the body
        # read, so the client does not push bytes the server refuses.
        status, payload, raw, _ = self.raw_request(
            "POST", f"{DIAGNOSIS_PATH}?x=1", headers={"Content-Length": "0"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_invalid_bodies_are_400(self) -> None:
        self.register("peer-a", 0)
        for document in (
            {},
            {"peerId": "peer-a"},
            b"{not json",
            [],
            None,
            {"a": 1},
            {"peerId": "peer-a", "ackId": "e", "expectedCheckpoint": 0,
             "expectedReceipts": "z" * 64, "suggestions": []},
        ):
            status, payload, _, _ = self.raw_request("POST", DIAGNOSIS_PATH, document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_conflicting_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.putheader("Content-Length", "0")
        conn.putheader("Content-Length", "1")
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_over_limit_declaration_is_413_without_reading(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.putheader("Content-Length", str(1_048_577))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read()), {"error": "payload_too_large"})
        conn.close()

    def test_exactly_limit_declaration_reaches_body_validation(self) -> None:
        # A declaration exactly at the limit is read; {} padded to the
        # limit is malformed JSON, so it is a 400 (never a 413).
        padded = b"{ " + b" " * (1_048_576 - 2)
        self.assertEqual(len(padded), 1_048_576)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            DIAGNOSIS_PATH,
            body=padded,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()


class DiagnosisReadOnlyTests(DiagnosisHttpFixture):
    def test_diagnosis_changes_no_state(self) -> None:
        document = self.gap_scenario()
        before = {
            "repairs": dict(self.server.store._repairs),
            "checkpoints": dict(self.server.store._checkpoints),
            "acks": dict(self.server.store._acks),
            "accepted": list(self.server.store._accepted),
            "candidates": copy.deepcopy(self.server.store._candidates),
        }
        for _ in range(3):
            status, _ = self.request("POST", DIAGNOSIS_PATH, document)
            self.assertEqual(status, 200)
        self.assertEqual(self.server.store._repairs, before["repairs"])
        self.assertEqual(self.server.store._checkpoints, before["checkpoints"])
        self.assertEqual(self.server.store._acks, before["acks"])
        self.assertEqual(self.server.store._accepted, before["accepted"])
        self.assertEqual(self.server.store._candidates, before["candidates"])
        # No execution was committed by the diagnosis.
        self.assertNotIn(("peer-a", "exec-1"), self.server.store._repairs)
        # The advice and metrics are unchanged by the diagnosis.
        status, repairs = self.request("GET", REPAIRS_PATH)
        self.assertEqual(status, 200)
        self.assertEqual(repairs["summary"]["suggestions"], 1)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 4)

    def test_diagnosing_after_commit_does_not_replay_or_move_checkpoint(self) -> None:
        document = self.gap_scenario()
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        checkpoint_before = self.server.store._checkpoints["peer-a"]
        history_before = list(self.server.store._repairs)
        for _ in range(2):
            status, payload = self.request("POST", DIAGNOSIS_PATH, document)
            self.assertEqual(status, 200)
            self.assertEqual(payload["conclusion"], "current")
        self.assertEqual(self.server.store._checkpoints["peer-a"], checkpoint_before)
        self.assertEqual(list(self.server.store._repairs), history_before)


class DiagnosisAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-diagnosis-auth-")
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

    def post(self, port, path, headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request("POST", path, headers=request_headers)
        else:
            request_headers["Content-Type"] = "application/json"
            conn.request("POST", path, body=body, headers=request_headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw), response_headers

    def post_duplicate_auth(self, port: str) -> tuple[int, dict, dict]:
        # http.client's request() collapses headers to a mapping, so a
        # duplicated Authorization is sent with the low-level putheader API.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.putheader("Authorization", "Bearer s3cret-token")
        conn.putheader("Authorization", "Bearer s3cret-token")
        conn.putheader("Content-Length", "2")
        conn.endheaders()
        conn.send(b"{}")
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw), response_headers

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            conn.close()

    def test_single_token_required_with_challenge(self) -> None:
        status, payload, headers = self.post(self.single_port, DIAGNOSIS_PATH)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # A duplicated Authorization header is 401 with a challenge.
        status, _, headers = self.post_duplicate_auth(self.single_port)
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, _ = self.post(
            self.single_port,
            DIAGNOSIS_PATH,
            {"Authorization": "Bearer wrong-token"},
            b"{}",
        )
        self.assertEqual(status, 401)
        # The valid token reaches body validation ({} -> 400, never 401).
        status, _, _ = self.post(
            self.single_port,
            DIAGNOSIS_PATH,
            {"Authorization": "Bearer s3cret-token"},
            b"{}",
        )
        self.assertEqual(status, 400)

    def test_read_scope_suffices_write_scope_forbidden_without_challenge(self) -> None:
        # A write-only token cannot reach the read-only diagnosis.
        status, payload, headers = self.post(
            self.scope_port,
            DIAGNOSIS_PATH,
            {"Authorization": f"Bearer {WRITE_TOKEN}"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        # A read-only token reaches it (body {} -> 400).
        status, _, _ = self.post(
            self.scope_port,
            DIAGNOSIS_PATH,
            {"Authorization": f"Bearer {READ_TOKEN}"},
            b"{}",
        )
        self.assertEqual(status, 400)

    def test_length_rejection_precedes_authentication(self) -> None:
        # A missing Content-Length is 400 even with no credential; the
        # low-level API omits the header that request() would add for a
        # body-less POST.
        conn = http.client.HTTPConnection("127.0.0.1", self.scope_port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        # An over-limit declaration is 413 before the 401 check, even with
        # no credential and without the body being read.
        conn = http.client.HTTPConnection("127.0.0.1", self.scope_port, timeout=5)
        conn.putrequest("POST", DIAGNOSIS_PATH)
        conn.putheader("Content-Length", str(1_048_577))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read()), {"error": "payload_too_large"})
        conn.close()


class DiagnosisPersistenceTests(unittest.TestCase):
    def build_state(self, data_file: str) -> tuple[StateStore, dict]:
        store = StateStore(data_file=data_file)
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
        status, repairs = store.get_replication_repairs(0, 100)
        assert status.value == 200
        suggestion = suggestion_from_advice(repairs["suggestions"][0])
        committed = [
            (ack_id, receipt)
            for (peer, ack_id), receipt in store._acks.items()
            if peer == "peer-a"
        ]
        digest = _repair_receipts_digest("peer-a", committed)
        document = {
            "peerId": "peer-a",
            "ackId": "exec-1",
            "expectedCheckpoint": 4,
            "expectedReceipts": digest,
            "suggestions": [suggestion],
        }
        status, _, error = store.apply_replication_repairs(
            "peer-a", "exec-1", 4, digest, [suggestion]
        )
        assert status.value == 201, error
        return store, document

    def test_diagnosis_is_identical_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store, document = self.build_state(data_file)
            before = store.diagnose_replication_repair(
                document["peerId"],
                document["ackId"],
                document["expectedCheckpoint"],
                document["expectedReceipts"],
                document["suggestions"],
            )
            recovered = StateStore(data_file=data_file)
            after = recovered.diagnose_replication_repair(
                document["peerId"],
                document["ackId"],
                document["expectedCheckpoint"],
                document["expectedReceipts"],
                document["suggestions"],
            )
            self.assertEqual(after, before)
            self.assertEqual(after["conclusion"], "current")
            self.assertEqual(after["execution"]["position"], 0)
            self.assertEqual(after["summary"]["executed"], 1)
            self.assertEqual(after["summary"]["anomalies"], 0)

    def test_uncommitted_diagnosis_identical_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store, document = self.build_state(data_file)
            # A different, never-committed ackId diagnoses as unexecuted;
            # the result is rebuilt identically after recovery.
            document = dict(document)
            document["ackId"] = "never-applied"
            before = store.diagnose_replication_repair(
                document["peerId"],
                document["ackId"],
                document["expectedCheckpoint"],
                document["expectedReceipts"],
                document["suggestions"],
            )
            recovered = StateStore(data_file=data_file)
            after = recovered.diagnose_replication_repair(
                document["peerId"],
                document["ackId"],
                document["expectedCheckpoint"],
                document["expectedReceipts"],
                document["suggestions"],
            )
            self.assertEqual(after, before)
            self.assertIsNone(after["execution"])

    def test_diagnosis_creates_no_data_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store, document = self.build_state(data_file)
            with open(data_file, "rb") as handle:
                before_bytes = handle.read()
            store.diagnose_replication_repair(
                document["peerId"],
                document["ackId"],
                document["expectedCheckpoint"],
                document["expectedReceipts"],
                document["suggestions"],
            )
            with open(data_file, "rb") as handle:
                after_bytes = handle.read()
            self.assertEqual(after_bytes, before_bytes)


if __name__ == "__main__":
    unittest.main()
