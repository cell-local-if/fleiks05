"""HTTP and persistence tests for the read-only transaction preview.

The endpoint is::

    POST /v1/transactions/plan

It accepts exactly the ``POST /v1/transactions/apply`` body and replays the
committing transaction's staged judgement against one committed snapshot,
reporting what a commit *would* do without creating any operation, binding,
candidate, audit entry, metric, or data-file change. Everything here goes
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

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    load_data_file,
    load_data_file_transactions,
)

PLAN_PATH = "/v1/transactions/plan"
APPLY_PATH = "/v1/transactions/apply"


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

    def post_plan(self, body: object, path: str = PLAN_PATH) -> tuple[int, object]:
        return self.request("POST", path, body)

    def post_apply(self, body: object, path: str = APPLY_PATH) -> tuple[int, object]:
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


class PlanHappyPathTests(HttpServerTestCase):
    def test_plan_reports_the_commit_partition_in_request_order(self) -> None:
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

    def test_plan_counts_match_the_following_apply(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        status, applied = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            (planned["accepted"], planned["replayed"]),
            (applied["accepted"], applied["replayed"]),
        )
        self.assertEqual(planned["operations"], applied["operations"])

    def test_empty_expected_set_matches_a_key_without_candidates(self) -> None:
        status, payload = self.post_plan(
            tx_document(tx_entry("fresh", "r3", "t1", "n", {"r3": 1}, []))
        )
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (1, 0))

    def test_mixed_new_and_known_entries_split_the_counts(self) -> None:
        self.seed_key("k1", "r1", "o1")
        committed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.post_apply(committed)[0], 201)
        # A new transaction id mixing the committed operation (a replay)
        # with one new operation.
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
            transaction_id="tx-2",
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (1, 1))
        self.assertEqual(
            [op["operationId"] for op in payload["operations"]], ["t1", "t2"]
        )

    def test_committed_transaction_plans_as_a_full_replay(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        # Move the key on so the replay provably ignores the current state.
        self.seed_key("k1", "r2", "o2", "v2")
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        self.assertEqual(payload["operations"][0]["value"], "n1")

    def test_response_is_compact_json_ending_with_a_newline(self) -> None:
        body = json.dumps(tx_document(tx_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw[:-1])

    def test_repeated_plan_against_unchanged_state_is_identical(self) -> None:
        doc = tx_document(tx_entry("k", "r3", "t1", "n", {"r3": 1}, []))
        first = self.post_plan(doc)
        second = self.post_plan(doc)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 200)


class PlanNoStateChangeTests(HttpServerTestCase):
    def test_plan_creates_no_operation_binding_or_metric(self) -> None:
        self.seed_key("k1", "r1", "o1")
        _, before_sync = self.get_sync()
        _, before_metrics = self.get_metrics()
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, _ = self.post_plan(doc)
        self.assertEqual(status, 200)
        _, after_sync = self.get_sync()
        _, after_metrics = self.get_metrics()
        self.assertEqual(before_sync, after_sync)
        self.assertEqual(before_metrics, after_metrics)
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "v1")
        # The planned identity is not bound: the archive has no record.
        status, _ = self.request("GET", "/v1/replicas/r3/operations/t1")
        self.assertEqual(status, 404)

    def test_plan_does_not_bind_the_transaction_id(self) -> None:
        doc = tx_document(tx_entry("k", "r3", "t1", "n", {"r3": 1}, []))
        status, _ = self.post_plan(doc)
        self.assertEqual(status, 200)
        # Committing the previewed transaction is still a fresh create.
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)

    def test_failed_plan_emits_no_partial_plan_and_changes_nothing(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),  # would be accepted
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, []),  # wrong expectation
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)
        # The failed plan bound nothing: a corrected transaction commits.
        fixed = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        status, _ = self.post_apply(fixed)
        self.assertEqual(status, 201)


class PlanValidationTests(HttpServerTestCase):
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
        entries = [
            tx_entry(f"k{i:03d}", "r3", f"t{i:03d}", "v", {"r3": 1})
            for i in range(101)
        ]
        status, payload = self.post_plan(tx_document(*entries))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_parameter_is_400(self) -> None:
        for query in ("?x=1", "?after=0"):
            status, payload = self.post_plan(
                tx_document(tx_entry("k")), path=f"{PLAN_PATH}{query}"
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        for path in (
            "/v1/transactions",
            f"{PLAN_PATH}/extra",
            f"{PLAN_PATH}/",
        ):
            status, payload = self.post_plan(tx_document(tx_entry("k")), path=path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", PLAN_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_clock_missing_own_replica_is_400(self) -> None:
        status, payload = self.post_plan(
            tx_document(tx_entry("k", "r3", "t1", "v", {"r9": 1}, []))
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_not_dominating_expected_candidates_is_400(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            # The clock does not dominate {"r1": 1}.
            tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, [candidate("r1", "o1")]),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class PlanConflictTests(HttpServerTestCase):
    def test_expected_set_mismatch_is_409(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, []),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})

    def test_nonempty_expectation_on_candidateless_key_is_409(self) -> None:
        doc = tx_document(
            tx_entry("missing", "r3", "t1", "n", {"r3": 1}, [candidate("r1", "o1")])
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})

    def test_known_identity_with_different_content_is_409(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r1", "o1", "other", {"r1": 1}, [candidate("r1", "o1")]),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_bound_transaction_id_with_different_entries_is_409(self) -> None:
        doc = tx_document(tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, []))
        self.assertEqual(self.post_apply(doc)[0], 201)
        changed = tx_document(tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, []),
                              tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []))
        status, payload = self.post_plan(changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_expectation_checked_against_staged_state_of_earlier_entries(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 2)


class PlanConcurrencyTests(HttpServerTestCase):
    def test_concurrent_plans_all_observe_a_complete_snapshot(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        plan_body = json.dumps(doc).encode("utf-8")
        apply_body = json.dumps(doc).encode("utf-8")
        plans: list[tuple[int, bytes]] = []
        lock = threading.Lock()

        def plan_worker() -> None:
            status, raw = self.request_bytes("POST", PLAN_PATH, plan_body)
            with lock:
                plans.append((status, raw))

        def apply_worker() -> None:
            self.request_bytes("POST", APPLY_PATH, apply_body)

        threads = [threading.Thread(target=plan_worker) for _ in range(8)]
        threads.append(threading.Thread(target=apply_worker))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(len(plans), 8)
        # Every plan observed one complete snapshot: either entirely before
        # the commit (accepted) or entirely after it (replayed) — never a
        # mix, and the state changed exactly once.
        bodies = {raw for _, raw in plans}
        self.assertTrue(all(status == 200 for status, _ in plans))
        self.assertLessEqual(len(bodies), 2)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)


class PlanRequestLimitTests(unittest.TestCase):
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

    def test_malformed_content_length_is_400(self) -> None:
        for value in ("", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            status, payload = self.post_raw([("Content-Length", value)], b"{}")
            self.assertEqual(status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_over_limit_declaration_is_413(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_declared_length_exactly_at_the_limit_is_processed(self) -> None:
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
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["operations"][0]["key"], key)


class PlanAuthTests(unittest.TestCase):
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

    def test_authorized_plan_succeeds(self) -> None:
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
        response.read()
        conn.close()


class PlanScopePolicyTests(unittest.TestCase):
    """In scope-policy mode the read-only plan is gated by the read scope."""

    POLICY = {
        "reader-token": frozenset({"read"}),
        "writer-token": frozenset({"write"}),
        "admin-token": frozenset({"read", "write", "admin"}),
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

    def post_plan(self, token: str | None) -> tuple[int, dict, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            PLAN_PATH,
            body=json.dumps(tx_document(tx_entry("k"))),
            headers=headers,
        )
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def test_read_and_admin_tokens_may_plan(self) -> None:
        for token in ("reader-token", "admin-token"):
            with self.subTest(token=token):
                status, payload, _ = self.post_plan(token)
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "planned")

    def test_write_only_token_is_403_without_challenge(self) -> None:
        status, payload, headers = self.post_plan("writer-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_missing_token_is_401_with_challenge(self) -> None:
        status, payload, headers = self.post_plan(None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")


class PersistentPlanTestCase(unittest.TestCase):
    """The read-only plan against a data-file-backed server."""

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

    def test_plan_never_writes_the_data_file(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        before = self.data_file.read_bytes()
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, payload = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 2)
        # The file keeps its exact bytes and no temporary file appeared.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(load_data_file_transactions(str(self.data_file)), {})
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_same_history_plans_identically_across_a_restart(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        committed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.request(server, "POST", APPLY_PATH, committed)[0], 201)
        fresh = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
            transaction_id="tx-2",
        )
        before_restart = self.request(server, "POST", PLAN_PATH, fresh)
        self.assertEqual(before_restart[0], 200)
        replay_before = self.request(server, "POST", PLAN_PATH, committed)

        server.shutdown()
        server.server_close()

        server = self.start_server()
        after_restart = self.request(server, "POST", PLAN_PATH, fresh)
        replay_after = self.request(server, "POST", PLAN_PATH, committed)
        self.assertEqual(before_restart, after_restart)
        self.assertEqual(replay_before, replay_after)
        self.assertEqual(
            (after_restart[1]["accepted"], after_restart[1]["replayed"]), (1, 1)
        )
        self.assertEqual(
            (replay_after[1]["accepted"], replay_after[1]["replayed"]), (0, 1)
        )
        # The plan itself persisted nothing: only the committed transaction
        # is in the file.
        self.assertEqual(list(load_data_file_transactions(str(self.data_file))), ["tx-1"])


if __name__ == "__main__":
    unittest.main()
