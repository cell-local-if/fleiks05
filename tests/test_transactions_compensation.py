"""HTTP, idempotency, conflict, and persistence tests for compensation.

The two endpoints are::

    GET  /v1/transactions/{transactionId}/compensation
    POST /v1/transactions/{transactionId}/compensate

The GET is strictly read-only and builds a reversible/blocked plan from
the pre-transaction log snapshot and the current snapshot; the POST
copies the plan's compensation operations field by field and commits
them as one atomic transaction. Everything here goes through the real
HTTP entry point (``SemanticStateServer`` + a request thread); only the
Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    load_data_file,
    load_data_file_compensations,
    load_data_file_transactions,
    parse_compensate_payload,
)

COMPENSATION_PATH = "/v1/transactions/tx-1/compensation"
COMPENSATE_PATH = "/v1/transactions/tx-1/compensate"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def tx_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "t1",
    value: str = "n1",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {"r1": 1, "r3": 1},
        "candidates": list(candidates) if candidates is not None else [],
    }


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def tx_document(*entries: dict, transaction_id: str = "tx-1") -> dict:
    return {"transactionId": transaction_id, "operations": list(entries)}


def compensate_document(compensation_id: str, digest: str, operations: list) -> dict:
    return {
        "compensationId": compensation_id,
        "expectedPlanDigest": digest,
        "operations": operations,
    }


class ParseCompensatePayloadTests(unittest.TestCase):
    def entry(self) -> dict:
        return {
            "key": "k1",
            "replicaId": "r3",
            "operationId": "comp-1",
            "value": "v1",
            "clock": {"r1": 1, "r3": 2},
        }

    def test_valid_body_is_normalized(self) -> None:
        doc = compensate_document("c-1", "a" * 64, [self.entry()])
        compensation_id, digest, entries = parse_compensate_payload(json.dumps(doc))
        self.assertEqual(compensation_id, "c-1")
        self.assertEqual(digest, "a" * 64)
        self.assertEqual(entries, [self.entry()])

    def test_rejects_malformed_and_wrong_root_shapes(self) -> None:
        good = compensate_document("c-1", "a" * 64, [self.entry()])
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"compensationId": "c-1"},
            {"expectedPlanDigest": "a" * 64, "operations": good["operations"]},
            dict(good, extra=1),
            {"compensationId": "", "expectedPlanDigest": "a" * 64, "operations": good["operations"]},
            {"compensationId": 7, "expectedPlanDigest": "a" * 64, "operations": good["operations"]},
            {"compensationId": "c-1", "expectedPlanDigest": "abc", "operations": good["operations"]},
            {"compensationId": "c-1", "expectedPlanDigest": "A" * 64, "operations": good["operations"]},
            {"compensationId": "c-1", "expectedPlanDigest": "a" * 63, "operations": good["operations"]},
            {"compensationId": "c-1", "expectedPlanDigest": "a" * 64, "operations": {}},
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_compensate_payload(body)

    def test_rejects_bad_entry_shapes(self) -> None:
        good = self.entry()
        bad_entries = [
            [],
            "x",
            {},
            *[
                {key: value for key, value in good.items() if key != field}
                for field in ("key", "replicaId", "operationId", "value", "clock")
            ],
            dict(good, candidates=[]),
            dict(good, extra=1),
            dict(good, key=""),
            dict(good, replicaId=""),
            dict(good, operationId=""),
            dict(good, value=""),
            dict(good, value=3),
            dict(good, clock={}),
            dict(good, clock={"r9": 1}),
            dict(good, clock={"r3": -1}),
        ]
        for entries in bad_entries:
            with self.assertRaises(ValueError, msg=repr(entries)):
                parse_compensate_payload(compensate_document("c-1", "a" * 64, entries))

    def test_rejects_empty_and_oversized_batches(self) -> None:
        with self.assertRaises(ValueError):
            parse_compensate_payload(compensate_document("c-1", "a" * 64, []))
        entries = [
            {
                "key": f"k{i:03d}",
                "replicaId": "r3",
                "operationId": f"c{i:03d}",
                "value": "v",
                "clock": {"r3": i + 1},
            }
            for i in range(101)
        ]
        with self.assertRaises(ValueError):
            parse_compensate_payload(compensate_document("c-1", "a" * 64, entries))

    def test_rejects_duplicate_keys_and_identities(self) -> None:
        base = self.entry()
        with self.assertRaises(ValueError):
            parse_compensate_payload(
                compensate_document(
                    "c-1",
                    "a" * 64,
                    [dict(base, key="k1", operationId="a"), dict(base, key="k1", operationId="b")],
                )
            )
        with self.assertRaises(ValueError):
            parse_compensate_payload(
                compensate_document(
                    "c-1",
                    "a" * 64,
                    [dict(base, key="k1", operationId="same"), dict(base, key="k2", operationId="same")],
                )
            )


class HttpServerTestCase(unittest.TestCase):
    """Spin up one in-memory server per class; reset the store per test."""

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

    def request_bytes(
        self, method: str, path: str, body: bytes | None = None, headers: dict | None = None
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(method, path, body=body, headers=merged)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        if body is None:
            status, raw = self.request_bytes(method, path)
        elif isinstance(body, (bytes, str)):
            data = body.encode("utf-8") if isinstance(body, str) else body
            status, raw = self.request_bytes(method, path, data)
        else:
            status, raw = self.request_bytes(method, path, json.dumps(body).encode("utf-8"))
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_apply(self, doc: dict, transaction_id: str = "tx-1") -> tuple[int, object]:
        return self.request("POST", "/v1/transactions/apply", doc)

    def seed(self, key: str = "k1", replica: str = "r1", op_id: str = "o1", value: str = "v1") -> None:
        status, _ = self.post_operation(replica, operation(op_id, key, value, {replica: 1}))
        self.assertEqual(status, 201)

    def commit_simple_transaction(
        self,
        key: str = "k1",
        value: str = "n1",
        transaction_id: str = "tx-1",
        tx_replica: str = "r3",
        tx_op: str = "t1",
    ) -> None:
        doc = tx_document(
            tx_entry(
                key,
                tx_replica,
                tx_op,
                value,
                {"r1": 1, tx_replica: 1},
                [candidate("r1", "o1")],
            ),
            transaction_id=transaction_id,
        )
        status, payload = self.post_apply(doc, transaction_id)
        self.assertEqual(status, 201, payload)

    def get_plan(self, transaction_id: str = "tx-1") -> tuple[int, object]:
        return self.request("GET", f"/v1/transactions/{transaction_id}/compensation")

    def compensate(self, doc: dict, transaction_id: str = "tx-1") -> tuple[int, object]:
        return self.request("POST", f"/v1/transactions/{transaction_id}/compensate", doc)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")


class CompensationPlanTests(HttpServerTestCase):
    def test_plan_for_reversible_transaction(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "reversible")
        self.assertEqual(plan["transactionId"], "tx-1")
        self.assertEqual(plan["algorithm"], "sha256")
        self.assertEqual(plan["descendants"], [])
        self.assertEqual(len(plan["expectedPlanDigest"]), 64)
        key_item = plan["keys"][0]
        self.assertEqual(key_item["key"], "k1")
        self.assertEqual(
            key_item["beforeCandidates"],
            [
                {
                    "value": "v1",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o1",
                }
            ],
        )
        self.assertEqual(
            key_item["currentCandidates"],
            [
                {
                    "value": "n1",
                    "clock": {"r1": 1, "r3": 1},
                    "replicaId": "r3",
                    "operationId": "t1",
                }
            ],
        )
        planned = key_item["operation"]
        self.assertEqual(planned["key"], "k1")
        self.assertEqual(planned["replicaId"], "r3")
        self.assertEqual(planned["value"], "v1")
        self.assertEqual(planned["clock"], {"r1": 1, "r3": 2})

    def test_plan_for_fresh_key_has_no_before_candidate_but_is_blocked(self) -> None:
        # A transaction over a key with no pre-transaction candidate:
        # there is no unique value to restore, so the plan is blocked
        # and offers no operation.
        doc = tx_document(tx_entry("fresh", "r3", "t1", "n", {"r3": 1}, []))
        self.assertEqual(self.post_apply(doc)[0], 201)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        key_item = plan["keys"][0]
        self.assertEqual(key_item["beforeCandidates"], [])
        self.assertEqual(key_item["currentCandidates"][0]["operationId"], "t1")
        self.assertIsNone(key_item["operation"])

    def test_plan_is_read_only(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        status, _ = self.get_plan()
        self.assertEqual(status, 200)
        status, _ = self.get_plan()
        self.assertEqual(status, 200)
        status, page = self.request("GET", "/v1/sync/operations")
        # Only the seed and the transaction operation are in the log.
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in page["operations"]],
            [("r1", "o1"), ("r3", "t1")],
        )

    def test_plan_is_blocked_by_a_causal_descendant(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        # A later write whose clock dominates the transaction clock.
        status, _ = self.post_operation(
            "r4", operation("l1", "k1", "later", {"r1": 1, "r3": 1, "r4": 1})
        )
        self.assertEqual(status, 201)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(
            plan["descendants"],
            [
                {
                    "key": "k1",
                    "replicaId": "r4",
                    "operationId": "l1",
                    "clock": {"r1": 1, "r3": 1, "r4": 1},
                }
            ],
        )

    def test_plan_is_blocked_when_multiple_before_candidates(self) -> None:
        # The transaction resolved a two-way conflict: the key held more
        # than one candidate just before the transaction, so there is no
        # single before value to restore.
        self.seed()
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))[0], 201
        )
        doc = tx_document(
            tx_entry(
                "k1",
                "r3",
                "t1",
                "n1",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            )
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        status, plan = self.get_plan()
        self.assertEqual(plan["conclusion"], "blocked")
        key_item = plan["keys"][0]
        self.assertEqual(len(key_item["beforeCandidates"]), 2)
        self.assertIsNone(key_item["operation"])
        self.assertEqual(plan["descendants"], [])

    def test_concurrent_write_that_adds_a_candidate_blocks_the_plan(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        # A concurrent (non-dominating) write adds a second candidate to
        # the same key: the transaction operation is no longer the sole
        # current candidate.
        status, _ = self.post_operation(
            "r5", operation("c1", "k1", "branch", {"r5": 1})
        )
        self.assertEqual(status, 201)
        status, plan = self.get_plan()
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(len(plan["keys"][0]["currentCandidates"]), 2)
        self.assertEqual(plan["descendants"], [])

    def test_plan_for_unknown_transaction_is_404(self) -> None:
        status, payload = self.get_plan("missing")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_plan_query_parameter_is_400(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        status, payload = self.request("GET", f"{COMPENSATION_PATH}?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_plan_extra_path_and_trailing_slash_are_404(self) -> None:
        for path in (f"{COMPENSATION_PATH}/extra", f"{COMPENSATION_PATH}/"):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_plan_digest_is_stable_across_reads(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        _, first = self.get_plan()
        _, second = self.get_plan()
        self.assertEqual(first["expectedPlanDigest"], second["expectedPlanDigest"])


class CompensationCommitTests(HttpServerTestCase):
    def plan_doc(self, compensation_id: str = "c-1") -> tuple[dict, str]:
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        operations = [item["operation"] for item in plan["keys"]]
        return (
            compensate_document(compensation_id, plan["expectedPlanDigest"], operations),
            plan["expectedPlanDigest"],
        )

    def test_compensation_restores_before_value_and_reports_201(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        status, payload = self.compensate(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["compensationId"], "c-1")
        self.assertEqual(payload["transactionId"], "tx-1")
        self.assertEqual(
            payload["operations"],
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "compensation:tx-1:r3:t1",
                    "value": "v1",
                }
            ],
        )
        self.assertEqual(
            payload["finalState"],
            [
                {
                    "key": "k1",
                    "value": "v1",
                    "clock": {"r1": 1, "r3": 2},
                    "replicaId": "r3",
                    "operationId": "compensation:tx-1:r3:t1",
                }
            ],
        )
        _, state = self.get_state("k1")
        self.assertEqual((state["status"], state["value"]), ("resolved", "v1"))

    def test_compensation_operations_enter_log_sync_and_audit(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        self.assertEqual(self.compensate(doc)[0], 201)
        _, page = self.request("GET", "/v1/sync/operations")
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in page["operations"]],
            [("r1", "o1"), ("r3", "t1"), ("r3", "compensation:tx-1:r3:t1")],
        )
        _, audit = self.request("GET", "/v1/audit/keys/k1/operations")
        self.assertEqual(
            [r["operation"]["operationId"] for r in audit["operations"]],
            ["o1", "t1", "compensation:tx-1:r3:t1"],
        )

    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        self.assertEqual(self.compensate(doc)[0], 201)
        status, payload = self.compensate(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["compensationId"], "c-1")
        _, page = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(page["operations"]), 3)

    def test_same_compensation_id_different_content_is_409(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, digest = self.plan_doc()
        self.assertEqual(self.compensate(doc)[0], 201)
        # Same id, tampered operation content.
        tampered = compensate_document(
            "c-1", digest, [dict(doc["operations"][0], value="other")]
        )
        status, payload = self.compensate(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Same id, different expected digest is also a conflict.
        other_digest = compensate_document("c-1", "b" * 64, doc["operations"])
        status, payload = self.compensate(other_digest)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_stale_plan_digest_is_409_compensation_conflict(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        # The key moves on before the commit: a descendant appears and
        # the freshly recomputed digest no longer matches.
        self.assertEqual(
            self.post_operation(
                "r4", operation("l1", "k1", "later", {"r1": 1, "r3": 1, "r4": 1})
            )[0],
            201,
        )
        status, payload = self.compensate(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})

    def test_blocked_plan_commit_is_409_compensation_conflict(self) -> None:
        # Even with the current digest, a blocked conclusion rejects.
        self.seed()
        self.commit_simple_transaction()
        self.assertEqual(
            self.post_operation(
                "r4", operation("l1", "k1", "later", {"r1": 1, "r3": 1, "r4": 1})
            )[0],
            201,
        )
        _, plan = self.get_plan()
        self.assertEqual(plan["conclusion"], "blocked")
        # A blocked plan offers no operation for every key; hand-craft a
        # plausible body bound to the current digest to prove the
        # conclusion gate, not the body gate, rejects it.
        operations = []
        for item in plan["keys"]:
            operations.append(
                {
                    "key": item["key"],
                    "replicaId": "r3",
                    "operationId": "compensation:tx-1:r3:t1",
                    "value": "v1",
                    "clock": {"r1": 1, "r3": 2},
                }
            )
        doc = compensate_document("c-9", plan["expectedPlanDigest"], operations)
        status, payload = self.compensate(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})

    def test_unknown_transaction_commit_is_404(self) -> None:
        entry = {
            "key": "k1",
            "replicaId": "r3",
            "operationId": "c-1",
            "value": "v1",
            "clock": {"r3": 2},
        }
        doc = compensate_document("c-1", "a" * 64, [entry])
        status, payload = self.compensate(doc, transaction_id="missing")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_tampered_operations_against_valid_plan_are_400(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        for mutated in (
            dict(doc["operations"][0], value="other"),
            dict(doc["operations"][0], operationId="other"),
            dict(doc["operations"][0], replicaId="r9"),
            dict(doc["operations"][0], clock={"r1": 1, "r3": 9}),
            dict(doc["operations"][0], key="other"),
        ):
            bad = compensate_document("c-1", doc["expectedPlanDigest"], [mutated])
            status, payload = self.compensate(bad)
            self.assertEqual(status, 400, mutated)
            self.assertEqual(payload, {"error": "invalid_request"}, mutated)

    def test_malformed_bodies_are_400(self) -> None:
        bad_bodies = [
            b"{not json",
            b"[]",
            b"{}",
            json.dumps({"compensationId": "c-1"}).encode(),
            json.dumps(compensate_document("c-1", "a" * 64, [])).encode(),
            json.dumps(compensate_document("", "a" * 64, [])).encode(),
        ]
        for body in bad_bodies:
            status, payload = self.request("POST", COMPENSATE_PATH, body)
            self.assertEqual(status, 400, body)
            self.assertEqual(payload, {"error": "invalid_request"}, body)

    def test_query_parameter_is_400(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        status, payload = self.request("POST", f"{COMPENSATE_PATH}?x=1", doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_extra_path_and_trailing_slash_are_404(self) -> None:
        self.seed()
        self.commit_simple_transaction()
        doc, _ = self.plan_doc()
        for path in (f"{COMPENSATE_PATH}/extra", f"{COMPENSATE_PATH}/"):
            status, payload = self.request("POST", path, doc)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_compensate_route_and_post_on_compensation_route_are_404(self) -> None:
        status, payload = self.request("GET", COMPENSATE_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload = self.request("POST", COMPENSATION_PATH, {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class CompensationMultiKeyTests(HttpServerTestCase):
    def test_multi_key_transaction_restores_every_key_atomically(self) -> None:
        self.seed("k1", "r1", "o1", "v1")
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k2", "v2", {"r2": 1}))[0], 201
        )
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "reversible")
        self.assertEqual([item["key"] for item in plan["keys"]], ["k1", "k2"])
        operations = [item["operation"] for item in plan["keys"]]
        commit_doc = compensate_document("c-multi", plan["expectedPlanDigest"], operations)
        status, payload = self.compensate(commit_doc)
        self.assertEqual(status, 201)
        self.assertEqual(len(payload["operations"]), 2)
        _, state1 = self.get_state("k1")
        _, state2 = self.get_state("k2")
        self.assertEqual(state1["value"], "v1")
        self.assertEqual(state2["value"], "v2")
        self.assertEqual([f["key"] for f in payload["finalState"]], ["k1", "k2"])

    def test_descendant_on_any_key_blocks_the_whole_compensation(self) -> None:
        self.seed("k1", "r1", "o1", "v1")
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k2", "v2", {"r2": 1}))[0], 201
        )
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        # Move only k2 on.
        self.assertEqual(
            self.post_operation(
                "r4", operation("l1", "k2", "later", {"r2": 1, "r3": 1, "r4": 1})
            )[0],
            201,
        )
        status, plan = self.get_plan()
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(len(plan["descendants"]), 1)
        operations = [item["operation"] for item in plan["keys"]]
        commit_doc = compensate_document("c-multi", plan["expectedPlanDigest"], operations)
        status, payload = self.compensate(commit_doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})


class CompensationRequestLimitTests(unittest.TestCase):
    """The compensate route keeps the shared Content-Length contract."""

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

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", COMPENSATE_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_over_limit_declaration_is_413(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", COMPENSATE_PATH)
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read()), {"error": "payload_too_large"})
        conn.close()


class PersistentCompensationTestCase(unittest.TestCase):
    """Compensation plans, commits, and replays against a data file."""

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
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            conn.request(
                method, path, body=data, headers={"Content-Type": "application/json"}
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_and_commit(self, server: SemanticStateServer) -> dict:
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations",
                operation("o1", "k1", "v1", {"r1": 1}),
            )[0],
            201,
        )
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.request(server, "POST", "/v1/transactions/apply", doc)[0], 201)
        status, plan = self.request(server, "GET", "/v1/transactions/tx-1/compensation")
        self.assertEqual(status, 200)
        return plan

    def test_compensation_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        plan = self.seed_and_commit(server)
        digest = plan["expectedPlanDigest"]
        operations = [item["operation"] for item in plan["keys"]]
        doc = compensate_document("c-1", digest, operations)
        status, payload = self.request(server, "POST", "/v1/transactions/tx-1/compensate", doc)
        self.assertEqual(status, 201)

        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records],
            [("r1", "o1"), ("r3", "t1"), ("r3", "compensation:tx-1:r3:t1")],
        )
        bindings = load_data_file_compensations(str(self.data_file))
        self.assertEqual(list(bindings), ["c-1"])
        binding = bindings["c-1"]
        self.assertEqual(binding["transactionId"], "tx-1")
        self.assertEqual(binding["expectedPlanDigest"], digest)
        self.assertEqual(binding["status"], "committed")
        self.assertEqual(binding["operations"], operations)
        # The original transaction binding is untouched.
        self.assertEqual(list(load_data_file_transactions(str(self.data_file))), ["tx-1"])

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The identical compensation replays 200 after restart.
        status, payload = self.request(server, "POST", "/v1/transactions/tx-1/compensate", doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        # Same id with different content still conflicts after restart.
        changed = compensate_document(
            "c-1", digest, [dict(operations[0], value="other")]
        )
        status, payload = self.request(server, "POST", "/v1/transactions/tx-1/compensate", changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # The restored value survived the restart.
        status, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "v1")

    def test_plan_digest_is_stable_across_restart_before_compensation(self) -> None:
        server = self.start_server()
        plan = self.seed_and_commit(server)
        digest = plan["expectedPlanDigest"]
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, restarted = self.request(server, "GET", "/v1/transactions/tx-1/compensation")
        self.assertEqual(status, 200)
        self.assertEqual(restarted["expectedPlanDigest"], digest)
        self.assertEqual(restarted["conclusion"], "reversible")
        operations = [item["operation"] for item in restarted["keys"]]
        status, payload = self.request(
            server,
            "POST",
            "/v1/transactions/tx-1/compensate",
            compensate_document("c-1", digest, operations),
        )
        self.assertEqual(status, 201)

    def test_old_data_file_without_compensations_section_recovers(self) -> None:
        # A version:1 file written before compensations existed has no
        # compensations section and still recovers.
        old_document = {
            "version": 1,
            "operations": [
                {
                    "replicaId": "r1",
                    "operation": {
                        "operationId": "o1",
                        "key": "k1",
                        "value": "v1",
                        "clock": {"r1": 1},
                    },
                }
            ],
        }
        self.data_file.write_text(json.dumps(old_document))
        server = self.start_server()
        status, payload = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v1")
        # The next durable commit upgrades the file but leaves the
        # recovered state and the (still empty) compensation history
        # intact.
        status, plan = self.request(server, "GET", "/v1/transactions/tx-1/compensation")
        self.assertEqual(status, 404)
        self.assertEqual(
            load_data_file_compensations(str(self.data_file)),
            {},
        )


if __name__ == "__main__":
    unittest.main()
