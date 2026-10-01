"""HTTP, plan, and persistence tests for verifiable transaction compensation.

The endpoints are::

    GET  /v1/transactions/{transactionId}/compensation
    POST /v1/transactions/{transactionId}/compensate

The first is a strictly read-only plan: it reports, per key of the
committed transaction, the pre-transaction candidate, the current
candidate, and the compensation operations that would restore the
pre-transaction values, plus the plan digest and any blocking causal
successors. The second commits the plan's operations as one atomic
transaction under a caller-chosen compensation id. Everything here goes
through the real HTTP entry point (``SemanticStateServer`` + a request
thread); only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    load_data_file_compensations,
    load_data_file_transactions,
    parse_compensation_apply,
)

APPLY_PATH = "/v1/transactions/apply"


def plan_path(transaction_id: str) -> str:
    return f"/v1/transactions/{transaction_id}/compensation"


def compensate_path(transaction_id: str) -> str:
    return f"/v1/transactions/{transaction_id}/compensate"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def tx_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "tx-op-1",
    value: str = "v",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {"r3": 1},
        "candidates": list(candidates) if candidates is not None else [],
    }


def tx_document(*entries: dict, transaction_id: str = "tx-1") -> dict:
    return {"transactionId": transaction_id, "operations": list(entries)}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def comp_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "t1:compensation",
    value: str = "v1",
    clock: dict | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {"r3": 2},
    }


def comp_document(
    *entries: dict,
    compensation_id: str = "c1",
    digest: str = "a" * 64,
) -> dict:
    return {
        "compensationId": compensation_id,
        "expectedPlanDigest": digest,
        "operations": list(entries),
    }


class ParseCompensationApplyTests(unittest.TestCase):
    def test_valid_compensation_is_normalized_in_order(self) -> None:
        compensation_id, digest, entries = parse_compensation_apply(
            json.dumps(
                comp_document(
                    comp_entry("k1", "r3", "t1:compensation", "v1", {"r1": 1, "r3": 2}),
                    comp_entry("k2", "r9", "t2:compensation", "v2", {"r9": 3}),
                    compensation_id="c-9",
                    digest="b" * 64,
                )
            )
        )
        self.assertEqual(compensation_id, "c-9")
        self.assertEqual(digest, "b" * 64)
        self.assertEqual(
            entries,
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "t1:compensation",
                    "value": "v1",
                    "clock": {"r1": 1, "r3": 2},
                },
                {
                    "key": "k2",
                    "replicaId": "r9",
                    "operationId": "t2:compensation",
                    "value": "v2",
                    "clock": {"r9": 3},
                },
            ],
        )

    def test_rejects_malformed_and_wrong_root_shapes(self) -> None:
        valid = comp_document(comp_entry("k"))
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"compensationId": "c1"},
            {"expectedPlanDigest": "a" * 64, "operations": valid["operations"]},
            {"compensationId": "c1", "operations": valid["operations"]},
            {"compensationId": "c1", "expectedPlanDigest": "a" * 64},
            dict(valid, extra=1),
            {"compensationId": "", "expectedPlanDigest": "a" * 64, "operations": valid["operations"]},
            {"compensationId": 7, "expectedPlanDigest": "a" * 64, "operations": valid["operations"]},
            {"compensationId": None, "expectedPlanDigest": "a" * 64, "operations": valid["operations"]},
            {"compensationId": "c1", "expectedPlanDigest": "a" * 64, "operations": {}},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse_compensation_apply(body)

    def test_rejects_malformed_digests(self) -> None:
        for digest in [
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            7,
            None,
            ["a" * 64],
        ]:
            with self.subTest(digest=digest):
                with self.assertRaises(ValueError):
                    parse_compensation_apply(
                        comp_document(comp_entry("k"), digest=digest)
                    )

    def test_rejects_bad_entry_fields_identities_and_clocks(self) -> None:
        good = comp_entry("k")
        bad_entries = [
            [],
            [dict(good, extra=1)],
            [{k: v for k, v in good.items() if k != "clock"}],
            [comp_entry("")],
            [comp_entry("k", replica="")],
            [comp_entry("k", operation_id="")],
            [comp_entry("k", value="")],
            [comp_entry("k", value=7)],
            [comp_entry("k", clock={})],
            [comp_entry("k", clock={"r9": 1})],
            [comp_entry("k", clock={"r3": -1})],
            [comp_entry("k", clock={"r3": True})],
            [comp_entry("k"), comp_entry("k")],
            [comp_entry("k1"), comp_entry("k2")],
            [comp_entry("k1", clock={"r3": 1}), comp_entry("k2", clock={"r3": 1})],
        ]
        for entries in bad_entries:
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    parse_compensation_apply(comp_document(*entries))

    def test_entry_count_bounds(self) -> None:
        with self.assertRaises(ValueError):
            parse_compensation_apply(comp_document())
        entries = [
            comp_entry(f"k{i:03d}", operation_id=f"c{i:03d}") for i in range(100)
        ]
        _, _, parsed = parse_compensation_apply(comp_document(*entries))
        self.assertEqual(len(parsed), 100)
        entries.append(comp_entry("k100", operation_id="c100"))
        with self.assertRaises(ValueError):
            parse_compensation_apply(comp_document(*entries))


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

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_plan(self, transaction_id: str = "tx-1") -> tuple[int, object]:
        return self.request("GET", plan_path(transaction_id))

    def post_compensate(self, body: object, transaction_id: str = "tx-1") -> tuple[int, object]:
        return self.request("POST", compensate_path(transaction_id), body)

    def seed_key(
        self, key: str, replica: str = "r1", operation_id: str | None = None, value: str = "v1"
    ) -> str:
        """One write on ``key``; returns the operation id used."""
        op_id = operation_id or f"o-{key}"
        status, _ = self.post_operation(replica, operation(op_id, key, value, {replica: 1}))
        self.assertEqual(status, 201)
        return op_id

    def commit_transaction(self, doc: dict | None = None) -> dict:
        """Seed k1 (r1/o1 -> v1) and commit tx-1 overwriting it with n1."""
        if doc is not None:
            status, payload = self.request("POST", APPLY_PATH, doc)
            self.assertEqual(status, 201)
            return payload
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        status, payload = self.request("POST", APPLY_PATH, doc)
        self.assertEqual(status, 201)
        return payload

    def compensate_body(self, plan: dict, compensation_id: str = "c1") -> dict:
        return {
            "compensationId": compensation_id,
            "expectedPlanDigest": plan["expectedPlanDigest"],
            "operations": plan["operations"],
        }


class CompensationPlanTests(HttpServerTestCase):
    def test_reversible_plan_reports_candidates_operations_and_digest(self) -> None:
        self.commit_transaction()
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["transactionId"], "tx-1")
        self.assertEqual(plan["conclusion"], "reversible")
        self.assertEqual(plan["algorithm"], "sha256")
        self.assertEqual(len(plan["expectedPlanDigest"]), 64)
        self.assertEqual(plan["descendants"], [])
        self.assertEqual(
            plan["keys"],
            [
                {
                    "key": "k1",
                    "beforeCandidates": [
                        {
                            "replicaId": "r1",
                            "operationId": "o1",
                            "value": "v1",
                            "clock": {"r1": 1},
                        }
                    ],
                    "currentCandidates": [
                        {
                            "replicaId": "r3",
                            "operationId": "t1",
                            "value": "n1",
                            "clock": {"r1": 1, "r3": 1},
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            plan["operations"],
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "t1:compensation",
                    "value": "v1",
                    "clock": {"r1": 1, "r3": 2},
                }
            ],
        )
        # The compensation clock strictly dominates the transaction clock.
        clock = plan["operations"][0]["clock"]
        self.assertGreaterEqual(clock["r1"], 1)
        self.assertGreater(clock["r3"], 1)

    def test_plan_is_deterministic_across_repeated_reads(self) -> None:
        self.commit_transaction()
        _, first = self.get_plan()
        _, second = self.get_plan()
        self.assertEqual(first, second)

    def test_transaction_own_operations_are_not_causal_successors(self) -> None:
        # Two entries from the same replica whose clocks are causally
        # ordered (t2's clock dominates t1's): the transaction's own
        # operations never block its compensation.
        self.seed_key("k1", "r1", "o1", "v1")
        self.seed_key("k2", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 2}, [candidate("r2", "o2")]),
        )
        self.commit_transaction(doc)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "reversible")
        self.assertEqual(plan["descendants"], [])
        status, payload = self.post_compensate(self.compensate_body(plan))
        self.assertEqual(status, 201)
        _, state1 = self.get_state("k1")
        self.assertEqual(state1["value"], "v1")
        _, state2 = self.get_state("k2")
        self.assertEqual(state2["value"], "v2")

    def test_unknown_transaction_is_404(self) -> None:
        status, payload = self.get_plan("no-such-tx")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_parameter_is_400(self) -> None:
        self.commit_transaction()
        status, payload = self.request("GET", plan_path("tx-1") + "?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_plan_is_blocked_by_a_causal_successor(self) -> None:
        self.commit_transaction()
        # A later write on k1 whose clock dominates the transaction's clock.
        status, _ = self.post_operation(
            "r4", operation("o-later", "k1", "later", {"r1": 1, "r3": 1, "r4": 1})
        )
        self.assertEqual(status, 201)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(plan["operations"], [])
        self.assertEqual(
            plan["descendants"],
            [
                {
                    "key": "k1",
                    "replicaId": "r4",
                    "operationId": "o-later",
                    "clock": {"r1": 1, "r3": 1, "r4": 1},
                }
            ],
        )

    def test_plan_is_blocked_by_a_concurrent_candidate(self) -> None:
        self.commit_transaction()
        # A concurrent write (no clock domination either way) leaves the
        # transaction operation as one of two current candidates.
        status, _ = self.post_operation(
            "r5", operation("o-conc", "k1", "other", {"r5": 1})
        )
        self.assertEqual(status, 201)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(plan["operations"], [])
        self.assertEqual(
            [(d["replicaId"], d["operationId"]) for d in plan["descendants"]],
            [("r5", "o-conc")],
        )

    def test_plan_is_blocked_when_transaction_created_the_key(self) -> None:
        self.commit_transaction(
            tx_document(tx_entry("fresh", "r3", "t1", "n1", {"r3": 1}, []))
        )
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(plan["keys"][0]["beforeCandidates"], [])

    def test_plan_is_blocked_when_key_had_conflicting_predecessors(self) -> None:
        self.seed_key("k1", "r1", "o1", "v1")
        self.post_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
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
        self.commit_transaction(doc)
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "blocked")
        self.assertEqual(len(plan["keys"][0]["beforeCandidates"]), 2)

    def test_plan_does_not_touch_state_or_logs(self) -> None:
        self.commit_transaction()
        _, before_sync = self.request("GET", "/v1/sync/operations")
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_audit = self.request("GET", "/v1/audit/keys/k1/operations")
        status, plan = self.get_plan()
        self.assertEqual(status, 200)
        _, after_sync = self.request("GET", "/v1/sync/operations")
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_audit = self.request("GET", "/v1/audit/keys/k1/operations")
        self.assertEqual(before_sync, after_sync)
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_audit, after_audit)
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "n1")
        # A blocked plan is just as read-only.
        self.post_operation("r4", operation("o-l", "k1", "x", {"r1": 1, "r3": 1, "r4": 1}))
        _, before_sync = self.request("GET", "/v1/sync/operations")
        status, plan = self.get_plan()
        self.assertEqual((status, plan["conclusion"]), (200, "blocked"))
        _, after_sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_sync, after_sync)


class CompensationCommitTests(HttpServerTestCase):
    def test_compensation_restores_pre_transaction_values(self) -> None:
        self.seed_key("k1", "r1", "o1", "v1")
        self.seed_key("k2", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
        )
        self.commit_transaction(doc)
        _, plan = self.get_plan()
        self.assertEqual(plan["conclusion"], "reversible")
        status, payload = self.post_compensate(self.compensate_body(plan))
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "compensationId": "c1",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r3", "operationId": "t1:compensation", "value": "v1"},
                    {"key": "k2", "replicaId": "r3", "operationId": "t2:compensation", "value": "v2"},
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        _, state1 = self.get_state("k1")
        self.assertEqual((state1["status"], state1["value"]), ("resolved", "v1"))
        _, state2 = self.get_state("k2")
        self.assertEqual((state2["status"], state2["value"]), ("resolved", "v2"))

    def test_compensation_operations_enter_sync_audit_metrics_and_ledger(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        status, _ = self.post_compensate(self.compensate_body(plan))
        self.assertEqual(status, 201)

        _, page = self.request("GET", "/v1/sync/operations")
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in page["operations"]],
            [("r1", "o1"), ("r3", "t1"), ("r3", "t1:compensation")],
        )
        _, audit = self.request("GET", "/v1/audit/keys/k1/operations")
        self.assertEqual(
            [r["operation"]["operationId"] for r in audit["operations"]],
            ["o1", "t1", "t1:compensation"],
        )
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 3)
        # The compensation commits as one atomic transaction under its id.
        _, ledger = self.request(
            "GET",
            "/v1/transactions/verify?after=0&limit=100&expectedCount=2&expectedDigest="
            + "0" * 64,
        )
        self.assertEqual(
            [t["transactionId"] for t in ledger["transactions"]], ["tx-1", "c1"]
        )
        compensation = ledger["transactions"][1]
        self.assertEqual(
            [op["key"] for op in compensation["operations"]], ["k1"]
        )
        self.assertEqual(
            compensation["operations"][0]["candidates"], [candidate("r3", "t1")]
        )

    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        body = self.compensate_body(plan)
        status, _ = self.post_compensate(body)
        self.assertEqual(status, 201)
        _, before_sync = self.request("GET", "/v1/sync/operations")

        status, payload = self.post_compensate(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        self.assertEqual(payload["compensationId"], "c1")
        self.assertEqual(payload["transactionId"], "tx-1")
        self.assertEqual(
            payload["operations"],
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "t1:compensation",
                    "value": "v1",
                }
            ],
        )
        _, after_sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_sync, after_sync)

    def test_same_id_different_content_is_operation_conflict(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        body = self.compensate_body(plan)
        self.assertEqual(self.post_compensate(body)[0], 201)

        changed_digest = self.compensate_body(plan)
        changed_digest["expectedPlanDigest"] = "0" * 64
        changed_operations = self.compensate_body(plan)
        changed_operations["operations"] = [dict(plan["operations"][0], value="other")]
        for changed in (changed_digest, changed_operations):
            with self.subTest(changed=changed):
                status, payload = self.post_compensate(changed)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_same_id_bound_to_another_transaction_conflicts(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        self.assertEqual(self.post_compensate(self.compensate_body(plan))[0], 201)
        # The same compensation id under a different transaction id is
        # different content: operation conflict, not a new compensation.
        self.seed_key("k2", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
            transaction_id="tx-2",
        )
        self.assertEqual(self.request("POST", APPLY_PATH, doc)[0], 201)
        _, plan2 = self.get_plan("tx-2")
        status, payload = self.post_compensate(
            self.compensate_body(plan2), transaction_id="tx-2"
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_stale_plan_digest_is_compensation_conflict(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        # A causal successor commits after the plan was generated.
        self.post_operation("r4", operation("o-l", "k1", "x", {"r1": 1, "r3": 1, "r4": 1}))
        status, payload = self.post_compensate(self.compensate_body(plan))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})

    def test_wrong_digest_is_compensation_conflict(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        body = self.compensate_body(plan)
        body["expectedPlanDigest"] = "0" * 64
        status, payload = self.post_compensate(body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})

    def test_second_compensation_id_is_blocked_after_first_commits(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        self.assertEqual(self.post_compensate(self.compensate_body(plan))[0], 201)
        # The transaction operation is no longer the sole current
        # candidate, so a different compensation id cannot commit again.
        status, payload = self.post_compensate(
            self.compensate_body(plan, compensation_id="c2")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "compensation_conflict"})

    def test_operations_not_copying_the_plan_are_400(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        for operations in (
            [dict(plan["operations"][0], value="wrong")],
            [dict(plan["operations"][0], clock={"r1": 1, "r3": 3})],
            [dict(plan["operations"][0], operationId="renamed")],
            [],
        ):
            with self.subTest(operations=operations):
                body = self.compensate_body(plan)
                body["operations"] = operations
                status, payload = self.post_compensate(body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_unknown_transaction_is_404(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        status, payload = self.post_compensate(
            self.compensate_body(plan), transaction_id="no-such-tx"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_bodies_are_400(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        valid = self.compensate_body(plan)
        bad_bodies = [
            b"{not json",
            b"[]",
            json.dumps({}).encode(),
            json.dumps(dict(valid, extra=1)).encode(),
            json.dumps({k: v for k, v in valid.items() if k != "operations"}).encode(),
            json.dumps(dict(valid, expectedPlanDigest="xyz")).encode(),
            json.dumps(dict(valid, compensationId="")).encode(),
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload = self.request(
                    "POST", compensate_path("tx-1"), body
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_parameter_is_400(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        status, payload = self.request(
            "POST",
            compensate_path("tx-1") + "?x=1",
            self.compensate_body(plan),
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_mismatches_are_404(self) -> None:
        self.commit_transaction()
        for path in (
            "/v1/transactions/tx-1/compensate/extra",
            "/v1/transactions/tx-1/compensate/",
            "/v1/transactions//compensate",
        ):
            with self.subTest(path=path):
                status, payload = self.request("POST", path, {})
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
        for path in (
            "/v1/transactions/tx-1/compensation/extra",
            "/v1/transactions/tx-1/compensation/",
        ):
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
        # The verbs are not interchangeable.
        status, _ = self.request("POST", plan_path("tx-1"), {})
        self.assertEqual(status, 404)
        status, _ = self.request("GET", compensate_path("tx-1"))
        self.assertEqual(status, 404)

    def test_response_is_compact_json_ending_with_a_newline(self) -> None:
        self.commit_transaction()
        _, plan = self.get_plan()
        body = json.dumps(self.compensate_body(plan)).encode("utf-8")
        status, raw = self.request_bytes("POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw[:-1])


class CompensationRequestLimitTests(unittest.TestCase):
    """The compensate endpoint shares the POST body-size contract."""

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

    def post_raw(self, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", compensate_path("tx-1"))
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")) if raw else None

    def test_declared_length_over_the_cap_is_413_before_any_read(self) -> None:
        status, payload = self.post_raw(
            [("Content-Length", str(MAX_BODY_BYTES + 1))]
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_missing_and_malformed_lengths_are_400(self) -> None:
        for headers in (
            [],
            [("Content-Length", "abc")],
            [("Content-Length", "-1")],
            [("Content-Length", "1"), ("Content-Length", "2")],
        ):
            with self.subTest(headers=headers):
                status, payload = self.post_raw(headers)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})


class PersistentCompensationTestCase(unittest.TestCase):
    """Transaction compensation against a data-file-backed server."""

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
                method,
                path,
                body=data,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_and_commit(self, server: SemanticStateServer) -> None:
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        status, _ = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 201)

    def test_compensation_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_and_commit(server)
        _, plan = self.request(server, "GET", plan_path("tx-1"))
        body = {
            "compensationId": "c1",
            "expectedPlanDigest": plan["expectedPlanDigest"],
            "operations": plan["operations"],
        }
        status, payload = self.request(server, "POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 201)

        # The operations, the transaction binding, and the compensation
        # binding are all durable in the same commit.
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records],
            [("r1", "o1"), ("r3", "t1"), ("r3", "t1:compensation")],
        )
        self.assertEqual(
            list(load_data_file_transactions(str(self.data_file))), ["tx-1", "c1"]
        )
        compensations = load_data_file_compensations(str(self.data_file))
        self.assertEqual(list(compensations), ["c1"])
        self.assertEqual(compensations["c1"]["transactionId"], "tx-1")
        self.assertEqual(
            compensations["c1"]["expectedPlanDigest"], plan["expectedPlanDigest"]
        )
        self.assertEqual(compensations["c1"]["operations"], plan["operations"])

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The restored value survives the restart.
        _, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "v1")
        # The identical compensation replays as 200 after restart and
        # appends nothing; different content under the same id conflicts.
        status, payload = self.request(server, "POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        changed = dict(body, expectedPlanDigest="0" * 64)
        status, payload = self.request(server, "POST", compensate_path("tx-1"), changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # The plan after restart is identical to the plan before it
        # (blocked by the committed compensation, same digest).
        _, plan_before = self.request(server, "GET", plan_path("tx-1"))
        server.shutdown()
        server.server_close()
        server = self.start_server()
        _, plan_after = self.request(server, "GET", plan_path("tx-1"))
        self.assertEqual(plan_before, plan_after)
        self.assertEqual(plan_after["conclusion"], "blocked")
        # The transaction ledger verifies identically after the restart.
        _, ledger = self.request(
            server,
            "GET",
            "/v1/transactions/verify?after=0&limit=100&expectedCount=2&expectedDigest="
            + "0" * 64,
        )
        self.assertEqual(
            [t["transactionId"] for t in ledger["transactions"]], ["tx-1", "c1"]
        )
        _, ledger = self.request(
            server,
            "GET",
            "/v1/transactions/verify?after=0&limit=100&expectedCount=2&expectedDigest="
            + ledger["digest"],
        )
        self.assertEqual(ledger["verification"]["status"], "ok")

    def test_plan_is_stable_across_restart_before_compensating(self) -> None:
        server = self.start_server()
        self.seed_and_commit(server)
        _, plan_before = self.request(server, "GET", plan_path("tx-1"))
        server.shutdown()
        server.server_close()
        server = self.start_server()
        _, plan_after = self.request(server, "GET", plan_path("tx-1"))
        self.assertEqual(plan_before, plan_after)
        self.assertEqual(plan_after["conclusion"], "reversible")
        # And the restarted server commits the plan generated before it.
        body = {
            "compensationId": "c1",
            "expectedPlanDigest": plan_before["expectedPlanDigest"],
            "operations": plan_before["operations"],
        }
        status, _ = self.request(server, "POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 201)

    def test_data_file_without_compensations_section_recovers(self) -> None:
        server = self.start_server()
        self.seed_and_commit(server)
        server.shutdown()
        server.server_close()
        # A file written before compensations existed simply has no
        # section; it recovers with no compensation bindings.
        document = json.loads(self.data_file.read_text())
        del document["compensations"]
        self.data_file.write_text(json.dumps(document))
        server = self.start_server()
        self.assertEqual(load_data_file_compensations(str(self.data_file)), {})
        _, plan = self.request(server, "GET", plan_path("tx-1"))
        self.assertEqual(plan["conclusion"], "reversible")
        body = {
            "compensationId": "c1",
            "expectedPlanDigest": plan["expectedPlanDigest"],
            "operations": plan["operations"],
        }
        status, _ = self.request(server, "POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 201)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.seed_and_commit(server)
        _, plan = self.request(server, "GET", plan_path("tx-1"))
        body = {
            "compensationId": "c1",
            "expectedPlanDigest": plan["expectedPlanDigest"],
            "operations": plan["operations"],
        }
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(server, "POST", compensate_path("tx-1"), body)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File, memory, identity index, and bindings are exactly as before.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(load_data_file_compensations(str(self.data_file)), {})
        self.assertEqual(list(load_data_file_transactions(str(self.data_file))), ["tx-1"])
        _, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "n1")
        _, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual(len(page["operations"]), 2)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # The same compensation commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", compensate_path("tx-1"), body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(list(load_data_file_compensations(str(self.data_file))), ["c1"])

    def test_corrupt_compensations_section_fails_startup(self) -> None:
        server = self.start_server()
        self.seed_and_commit(server)
        _, plan = self.request(server, "GET", plan_path("tx-1"))
        body = {
            "compensationId": "c1",
            "expectedPlanDigest": plan["expectedPlanDigest"],
            "operations": plan["operations"],
        }
        self.assertEqual(
            self.request(server, "POST", compensate_path("tx-1"), body)[0], 201
        )
        server.shutdown()
        server.server_close()

        document = json.loads(self.data_file.read_text())
        # A binding naming an identity that was never accepted is corrupt.
        document["compensations"][0]["operations"][0]["operationId"] = "ghost"
        self.data_file.write_text(json.dumps(document))
        with self.assertRaises(server_module.PersistenceError):
            self.start_server()


if __name__ == "__main__":
    unittest.main()
