"""HTTP, scope, persistence, and concurrency tests for the transaction plan.

The endpoint is::

    POST /v1/transactions/plan

It accepts the very same ``{"transactionId","operations"}`` document as
``POST /v1/transactions/apply`` and reproduces apply's staged judgment
inside one snapshot, but it is strictly read-only: it only reports the
accepted/replayed split a subsequent apply would observe and creates no
operation, binding, candidate, audit entry, metric, or durable change.
Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
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
)

APPLY_PATH = "/v1/transactions/apply"
PLAN_PATH = "/v1/transactions/plan"


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

    def post_apply(self, body: object, path: str = APPLY_PATH) -> tuple[int, object]:
        return self.request("POST", path, body)

    def post_plan(self, body: object, path: str = PLAN_PATH) -> tuple[int, object]:
        return self.request("POST", path, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_key(
        self, key: str, replica: str = "r1", operation_id: str | None = None, value: str = "v1"
    ) -> str:
        """One write on ``key``; returns the operation id used."""
        op_id = operation_id or f"o-{key}"
        status, _ = self.post_operation(replica, operation(op_id, key, value, {replica: 1}))
        self.assertEqual(status, 201)
        return op_id


class TransactionPlanHappyPathTests(HttpServerTestCase):
    def test_plan_reports_accepted_split_in_request_order(self) -> None:
        self.seed_key("k1", "r1", "o1")
        self.seed_key("k2", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "planned",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r3", "operationId": "t1", "value": "n1"},
                    {"key": "k2", "replicaId": "r3", "operationId": "t2", "value": "n2"},
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )

    def test_success_has_exactly_the_five_top_level_and_four_entry_fields(self) -> None:
        body = json.dumps(tx_document(tx_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)
        key_orders: list[list[str]] = []
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=lambda pairs: _record(pairs, key_orders)
        )
        self.assertEqual(
            set(key_orders[-1]),
            {"status", "transactionId", "operations", "accepted", "replayed"},
        )
        self.assertEqual(len(key_orders[-1]), 5)
        self.assertEqual(
            set(key_orders[0]),
            {"key", "replicaId", "operationId", "value"},
        )
        self.assertEqual(len(key_orders[0]), 4)
        self.assertEqual(payload["status"], "planned")

    def test_response_is_compact_json_ending_with_one_newline(self) -> None:
        body = json.dumps(tx_document(tx_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b" ", raw[:-1])

    def test_empty_expected_set_plans_on_a_fresh_key(self) -> None:
        status, payload = self.post_plan(tx_document(tx_entry("fresh", "r3", "t1", "n")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual((payload["accepted"], payload["replayed"]), (1, 0))

    def test_one_hundred_entries_are_all_planned(self) -> None:
        entries = [
            tx_entry(f"k{i:03d}", "r3", f"t{i:03d}", f"n{i:03d}", {"r3": 1}, [])
            for i in range(100)
        ]
        status, payload = self.post_plan(tx_document(*entries))
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (100, 0))
        self.assertEqual([r["key"] for r in payload["operations"]], [f"k{i:03d}" for i in range(100)])


def _record(pairs: list, orders: list[list[str]]) -> dict:
    orders.append([key for key, _ in pairs])
    return dict(pairs)


class TransactionPlanReadOnlyTests(HttpServerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
        )

    def seed(self) -> None:
        self.seed_key("k1", "r1", "o1")

    def test_plan_changes_no_state_log_or_metrics(self) -> None:
        self.seed()
        _, before_sync = self.get_sync()
        _, before_metrics = self.get_metrics()
        status, payload = self.post_plan(self.doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 2)
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "v1")
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)
        _, after_sync = self.get_sync()
        self.assertEqual(after_sync, before_sync)
        _, after_metrics = self.get_metrics()
        self.assertEqual(after_metrics, before_metrics)
        # The previewed operation is not addressable in the archive.
        status, _ = self.request("GET", "/v1/replicas/r3/operations/t1")
        self.assertEqual(status, 404)

    def test_plan_does_not_bind_the_transaction_id(self) -> None:
        self.seed()
        for _ in range(3):
            status, payload = self.post_plan(self.doc)
            self.assertEqual(status, 200)
            self.assertEqual((payload["accepted"], payload["replayed"]), (2, 0))
        # A different body under the same id still plans (no conflict) and a
        # subsequent apply is a fresh 201 with the same split.
        changed = tx_document(tx_entry("k2", "r9", "t9", "n9", {"r9": 1}, []))
        status, payload = self.post_plan(changed)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 1)
        status, payload = self.post_apply(self.doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            (payload["accepted"], payload["replayed"]),
            (2, 0),
        )

    def test_repeated_plans_against_unchanged_state_are_identical(self) -> None:
        self.seed()
        status, first = self.post_plan(self.doc)
        self.assertEqual(status, 200)
        for _ in range(3):
            status, again = self.post_plan(self.doc)
            self.assertEqual(status, 200)
            self.assertEqual(again, first)


class TransactionPlanMatchesApplyTests(HttpServerTestCase):
    def test_plan_split_matches_the_following_apply(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual((planned["accepted"], planned["replayed"]), (2, 0))
        status, committed = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            (committed["accepted"], committed["replayed"]),
            (planned["accepted"], planned["replayed"]),
        )
        self.assertEqual(committed["operations"], planned["operations"])

    def test_mixed_replay_split_matches_the_following_apply(self) -> None:
        self.seed_key("k1", "r1", "o1")
        first = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            transaction_id="tx-a",
        )
        self.assertEqual(self.post_apply(first)[0], 201)
        second = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
            transaction_id="tx-b",
        )
        status, planned = self.post_plan(second)
        self.assertEqual(status, 200)
        self.assertEqual((planned["accepted"], planned["replayed"]), (1, 1))
        self.assertEqual([r["operationId"] for r in planned["operations"]], ["t1", "t2"])
        status, committed = self.post_apply(second)
        self.assertEqual(status, 201)
        self.assertEqual(
            (committed["accepted"], committed["replayed"]),
            (1, 1),
        )
        self.assertEqual(committed["operations"], planned["operations"])

    def test_planned_conflict_matches_the_following_apply(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, []),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(planned, {"error": "transaction_conflict"})
        status, committed = self.post_apply(doc)
        self.assertEqual(status, 409)
        self.assertEqual(committed, {"error": "transaction_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_planned_clock_failure_matches_the_following_apply(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, [candidate("r1", "o1")]),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 400)
        self.assertEqual(planned, {"error": "invalid_request"})
        status, committed = self.post_apply(doc)
        self.assertEqual(status, 400)
        self.assertEqual(committed, {"error": "invalid_request"})


class TransactionPlanReplayTests(HttpServerTestCase):
    def test_committed_transaction_plans_as_a_whole_replay(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        # The key moves on after the commit; the replay plan still answers
        # from the binding without re-checking the current candidates.
        self.assertEqual(
            self.post_operation(
                "r4", operation("o4", "k1", "later", {"r1": 1, "r3": 1, "r4": 1})
            )[0],
            201,
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "planned",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r3", "operationId": "t1", "value": "n1"}
                ],
                "accepted": 0,
                "replayed": 1,
            },
        )

    def test_bound_transaction_id_with_different_entries_is_409(self) -> None:
        self.assertEqual(self.post_apply(tx_document(tx_entry("k1", "r3", "t1")))[0], 201)
        variants = [
            tx_document(tx_entry("k1", "r3", "t1", "other")),
            tx_document(tx_entry("k2", "r3", "t1")),
            tx_document(tx_entry("k1", "r3", "t1"), tx_entry("k2", "r3", "t2")),
        ]
        for doc in variants:
            with self.subTest(doc=doc):
                status, payload = self.post_plan(doc)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_unbound_transaction_id_with_replayed_identity_content_is_replayed(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r1", "o1", "v1", {"r1": 1}, [candidate("r1", "o1")]),
            transaction_id="tx-other",
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        self.assertEqual(payload["operations"][0]["value"], "v1")

    def test_known_identity_with_different_content_is_operation_conflict(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r1", "o1", "n", {"r1": 1}, []),
            transaction_id="tx-new",
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Nothing was bound or committed.
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)
        status, payload = self.post_plan(tx_document(tx_entry("k3", "r9", "t9", "v", {"r9": 1})))
        self.assertEqual(status, 200)


class TransactionPlanConflictTests(HttpServerTestCase):
    def test_expected_set_mismatch_is_409_with_no_partial_plan(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, []),
        )
        status, raw = self.request_bytes("POST", PLAN_PATH, json.dumps(doc).encode())
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw), {"error": "transaction_conflict"})
        self.assertTrue(raw.endswith(b"\n"))
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)

    def test_nonempty_expectation_on_candidateless_key_is_409(self) -> None:
        status, payload = self.post_plan(
            tx_document(
                tx_entry("missing", "r3", "t1", "n", {"r3": 1}, [candidate("r1", "o1")])
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})

    def test_unknown_expected_identity_is_409(self) -> None:
        self.seed_key("k1", "r1", "o1")
        status, payload = self.post_plan(
            tx_document(
                tx_entry(
                    "k1", "r3", "t1", "n", {"r1": 1, "r3": 1},
                    [candidate("r1", "o1"), candidate("r9", "o9")],
                )
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})


class TransactionPlanValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        bad_bodies = [
            b"{not json",
            b"[]",
            b"{}",
            json.dumps({"operations": []}).encode(),
            json.dumps(tx_document()).encode(),
            json.dumps({"transactionId": "", "operations": [tx_entry("k")]}).encode(),
            json.dumps(tx_document(tx_entry("k", candidates="x"))).encode(),
            json.dumps(
                tx_document(tx_entry("k"), tx_entry("k", "r4", "t2", "v", {"r4": 1}))
            ).encode(),
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload = self.post_plan(body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_oversized_batch_is_400(self) -> None:
        doc = tx_document(
            *[tx_entry(f"k{i}", "r3", f"t{i}", "v", {"r3": 1}) for i in range(101)]
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_missing_own_replica_is_400(self) -> None:
        status, payload = self.post_plan(
            tx_document(tx_entry("k", "r3", "t1", "v", {"r9": 1}, []))
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_not_dominating_expected_candidates_is_400(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # No state changed and no binding was created: the corrected plan
        # plans cleanly and an apply still commits.
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "v1")
        fixed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, payload = self.post_plan(fixed)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 2)

    def test_query_parameter_is_400_before_body_validation(self) -> None:
        status, payload = self.post_plan(tx_document(tx_entry("k")), path=f"{PLAN_PATH}?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, page = self.get_sync()
        self.assertEqual(page["operations"], [])

    def test_missing_extra_segment_and_trailing_slash_are_404(self) -> None:
        for path in (
            "/v1/transactions",
            "/v1/transactions/",
            f"{PLAN_PATH}/extra",
            f"{PLAN_PATH}/",
            "/v1/transactions/plans",
            "/v1/transaction/plan",
        ):
            status, payload = self.post_plan(tx_document(tx_entry("k")), path=path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", PLAN_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class TransactionPlanRequestLimitTests(unittest.TestCase):
    """The plan route keeps the shared Content-Length/auth priority."""

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
        conn.putrequest("POST", PLAN_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_over_limit_declaration_is_413(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_declared_length_exactly_at_the_limit_is_planned(self) -> None:
        template = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "",
                    "replicaId": "r3",
                    "operationId": "t1",
                    "value": "v",
                    "clock": {"r3": 1},
                    "candidates": [],
                }
            ],
        }
        base = json.dumps(template, separators=(",", ":")).encode("utf-8")
        key_length = MAX_BODY_BYTES - len(base)
        self.assertGreater(key_length, 0)
        key = "k" + "x" * (key_length - 1)
        doc = json.dumps(tx_document(tx_entry(key, "r3", "t1")), separators=(",", ":")).encode(
            "utf-8"
        )
        self.assertEqual(len(doc), MAX_BODY_BYTES)
        status, payload = self.post_raw([("Content-Length", str(len(doc)))], doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["operations"][0]["key"], key)


class TransactionPlanAuthTests(unittest.TestCase):
    def test_unauthorized_plan_is_401_without_reading_the_body(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            PLAN_PATH,
            body=json.dumps(tx_document(tx_entry("k"))),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_authorized_plan_succeeds_with_legacy_token(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            PLAN_PATH,
            body=json.dumps(tx_document(tx_entry("k"))),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token",
            },
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["status"], "planned")
        conn.close()


class TransactionPlanScopePolicyTests(unittest.TestCase):
    """In scope mode the preflight needs read (or admin), not write."""

    READ_TOKEN = "reader-token"
    WRITE_TOKEN = "writer-token"
    ADMIN_TOKEN = "admin-token"
    POLICY = {
        READ_TOKEN: frozenset({"read"}),
        WRITE_TOKEN: frozenset({"write"}),
        ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(cls.POLICY)
        )
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

    def call(self, token: str | None, body: object = None, path: str = PLAN_PATH):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = None if body is None else json.dumps(body)
        conn.request("POST", path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw), response_headers

    def test_read_and_admin_scopes_plan(self) -> None:
        doc = tx_document(tx_entry("k"))
        for token in (self.READ_TOKEN, self.ADMIN_TOKEN):
            with self.subTest(token=token):
                status, payload, _ = self.call(token, doc)
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "planned")

    def test_write_scope_is_forbidden_without_challenge(self) -> None:
        status, payload, headers = self.call(self.WRITE_TOKEN, tx_document(tx_entry("k")))
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(headers.get("WWW-Authenticate"))

    def test_missing_or_bad_token_is_401_with_challenge(self) -> None:
        for token in (None, "unknown-token"):
            with self.subTest(token=token):
                status, payload, headers = self.call(token, tx_document(tx_entry("k")))
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_scope_failure_precedes_query_body_and_route_state(self) -> None:
        headers = [
            ("Content-Length", "2"),
            ("Authorization", f"Bearer {self.WRITE_TOKEN}"),
        ]
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", PLAN_PATH + "?x=1")
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(b"{}")
        response = conn.getresponse()
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(response.read()), {"error": "forbidden"})
        self.assertIsNone(response.getheader("WWW-Authenticate"))
        conn.close()


class TransactionPlanConcurrencyTests(HttpServerTestCase):
    def test_concurrent_plans_all_reflect_one_complete_snapshot(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        body = json.dumps(doc).encode("utf-8")
        payloads: list[dict] = []
        lock = threading.Lock()
        start = threading.Event()

        def planner() -> None:
            start.wait(timeout=5)
            status, payload = self.request("POST", PLAN_PATH, body)
            self.assertEqual(status, 200)
            with lock:
                payloads.append(payload)

        def committer() -> None:
            start.wait(timeout=5)
            status, _ = self.post_apply(doc)
            self.assertIn(status, (200, 201))

        threads = [threading.Thread(target=planner) for _ in range(8)]
        commit_thread = threading.Thread(target=committer)
        for thread in threads + [commit_thread]:
            thread.start()
        start.set()
        for thread in threads + [commit_thread]:
            thread.join(timeout=10)
        # Every plan is entirely the pre-commit snapshot (accepted 1) or
        # entirely the post-commit snapshot (replayed 1) — never a mix, and
        # every per-operation value is consistent with its split.
        for payload in payloads:
            self.assertEqual(payload["status"], "planned")
            self.assertEqual(payload["accepted"] + payload["replayed"], 1)
            self.assertEqual(len(payload["operations"]), 1)
            if payload["replayed"]:
                self.assertEqual(payload["operations"][0]["value"], "n1")
        splits = {(p["accepted"], p["replayed"]) for p in payloads}
        self.assertTrue(splits <= {(1, 0), (0, 1)})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)


class PersistentTransactionPlanTestCase(unittest.TestCase):
    """The transaction preflight against a --data-file-backed server."""

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

    def seed_key(self, server: SemanticStateServer, key: str, replica: str, op_id: str) -> None:
        status, _ = self.request(
            server,
            "POST",
            f"/v1/replicas/{replica}/operations",
            operation(op_id, key, f"v-{key}", {replica: 1}),
        )
        self.assertEqual(status, 201)

    def test_plan_never_touches_the_data_file_or_creates_temps(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
        )
        # Settle, then snapshot bytes and mtime.
        before = self.data_file.read_bytes()
        before_mtime = self.data_file.stat().st_mtime_ns
        for _ in range(3):
            status, payload = self.request(server, "POST", PLAN_PATH, doc)
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "planned")
            self.assertEqual((payload["accepted"], payload["replayed"]), (2, 0))
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(self.data_file.stat().st_mtime_ns, before_mtime)
        self.assertEqual(list(self.tmp.iterdir()), [self.data_file])
        # Only the seed write is durable.
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)

    def test_rejected_plans_never_touch_the_data_file(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        before = self.data_file.read_bytes()
        conflict = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [])
        )
        bad_clock = tx_document(
            tx_entry("k1", "r3", "t2", "n2", {"r3": 1}, [candidate("r1", "o1")])
        )
        for doc, expected_status, error in (
            (conflict, 409, "transaction_conflict"),
            (bad_clock, 400, "invalid_request"),
            (b"{not json", 400, "invalid_request"),
        ):
            with self.subTest(error=error):
                status, payload = self.request(server, "POST", PLAN_PATH, doc)
                self.assertEqual(status, expected_status)
                self.assertEqual(payload, {"error": error})
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(list(self.tmp.iterdir()), [self.data_file])

    def test_same_history_gives_same_plan_across_restart(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        committed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        status, _ = self.request(server, "POST", APPLY_PATH, committed)
        self.assertEqual(status, 201)
        # A mixed preview: t1 replays from the binding, t2 would be created.
        preview = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
            transaction_id="tx-preview",
        )
        status, before_restart = self.request(server, "POST", PLAN_PATH, preview)
        self.assertEqual(status, 200)
        self.assertEqual(
            (before_restart["accepted"], before_restart["replayed"]), (1, 1)
        )
        durable_records = len(load_data_file(str(self.data_file)))

        server.shutdown()
        server.server_close()
        server = self.start_server()
        status, after_restart = self.request(server, "POST", PLAN_PATH, preview)
        self.assertEqual(status, 200)
        self.assertEqual(after_restart, before_restart)
        # The previews on either side of the restart appended nothing.
        self.assertEqual(len(load_data_file(str(self.data_file))), durable_records)

    def test_whole_replay_plan_does_not_write_after_restart(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        self.assertEqual(self.request(server, "POST", APPLY_PATH, doc)[0], 201)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        before = self.data_file.read_bytes()
        status, payload = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(list(self.tmp.iterdir()), [self.data_file])


if __name__ == "__main__":
    unittest.main()
