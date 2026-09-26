"""HTTP, scope, and persistence tests for the read-only batch plan preview.

The endpoint is::

    POST /v1/resolve/auto/plan

It accepts the very same ``{"resolutions":[...]}`` document as the committing
batch endpoint (1-100 entries, distinct keys and identities), previews what
the equivalent ``POST /v1/resolve/auto/batch`` would choose, and commits
nothing: the top-level status is always ``"planned"``, ``accepted`` counts
entries the batch would newly create and ``replayed`` counts same-binding
identities answered from their committed operations. Everything here goes
through the real HTTP entry point (``SemanticStateServer`` + a request
thread); only the Python standard library is used.
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

PLAN_PATH = "/v1/resolve/auto/plan"
BATCH_PATH = "/v1/resolve/auto/batch"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def batch_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "fix-1",
    clock: dict | None = None,
    policy: str = "lowest_identity",
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock if clock is not None else {"r1": 1, "r2": 1, "r3": 1},
        "policy": policy,
    }


def batch_document(*entries: dict) -> dict:
    return {"resolutions": list(entries)}


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

    def post_plan(self, body: object) -> tuple[int, object]:
        return self.request("POST", PLAN_PATH, body)

    def post_batch(self, body: object) -> tuple[int, object]:
        return self.request("POST", BATCH_PATH, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_auto(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve/auto", body)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_conflict(self, key: str = "k", low: str = "v1", high: str = "v2") -> None:
        """Two concurrent writes with distinct global identities on ``key``."""
        salt = getattr(self, "_seed_salt", 0)
        self._seed_salt = salt + 1
        id1, id2 = f"o1-{key}-{salt}", f"o2-{key}-{salt}"
        self.assertEqual(
            self.post_operation("r1", operation(id1, key, low, {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation(id2, key, high, {"r2": 1}))[0], 201
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class PlanHappyPathTests(HttpServerTestCase):
    def test_plan_reports_planned_selections_and_counts_without_committing(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2", low="aaa", high="zzz")
        _, metrics_before = self.get_metrics()
        doc = batch_document(
            batch_entry("k1", "r3", "f1"),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "planned",
                "resolutions": [
                    {
                        "key": "k1",
                        "replicaId": "r3",
                        "operationId": "f1",
                        "value": "v1",
                        "policy": "lowest_identity",
                    },
                    {
                        "key": "k2",
                        "replicaId": "r3",
                        "operationId": "f2",
                        "value": "zzz",
                        "policy": "highest_identity",
                    },
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        self.assertIsInstance(payload["accepted"], int)
        self.assertIsInstance(payload["replayed"], int)
        # Nothing changed: both keys are still in conflict and no record was
        # appended.
        for key in ("k1", "k2"):
            _, state = self.get_state(key)
            self.assertEqual(state["status"], "conflict", key)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)
        _, metrics_after = self.get_metrics()
        self.assertEqual(metrics_after, metrics_before)

    def test_plan_then_commit_pick_the_same_values(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2", low="aaa", high="zzz")
        doc = batch_document(
            batch_entry("k1", "r3", "f1"),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        _, planned = self.post_plan(doc)
        status, committed = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])
        self.assertEqual(committed["accepted"], planned["accepted"])
        self.assertEqual(committed["replayed"], planned["replayed"])

    def test_response_is_compact_json_ending_with_a_newline(self) -> None:
        self.seed_conflict("k")
        body = json.dumps(batch_document(batch_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(json.loads(raw.decode("utf-8"))["status"], "planned")
        # Error responses for the route carry the same terminator.
        status, raw = self.request_bytes("POST", PLAN_PATH, b"{oops")
        self.assertEqual(status, 400)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"error": "invalid_request"})
        status, raw = self.request_bytes(
            "POST", PLAN_PATH, json.dumps(batch_document(batch_entry("absent"))).encode()
        )
        self.assertEqual(status, 409)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"error": "resolution_conflict"})

    def test_single_entry_plan_is_supported(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_plan(batch_document(batch_entry("k")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)

    def test_one_hundred_entries_are_previewed_in_order(self) -> None:
        for i in range(100):
            key = f"k{i:03d}"
            pa, pb = f"p{i:03d}a", f"p{i:03d}b"
            self.assertEqual(
                self.post_operation(pa, operation("o1", key, "lo", {pa: 1}))[0], 201
            )
            self.assertEqual(
                self.post_operation(pb, operation("o2", key, "hi", {pb: 1}))[0], 201
            )
        entries = [
            batch_entry(
                f"k{i:03d}",
                f"r{i:03d}",
                f"f{i:03d}",
                {f"p{i:03d}a": 1, f"p{i:03d}b": 1, f"r{i:03d}": 1},
                "highest_identity" if i % 2 else "lowest_identity",
            )
            for i in range(100)
        ]
        status, payload = self.post_plan(batch_document(*entries))
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 100)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(len(payload["resolutions"]), 100)
        self.assertEqual(
            [r["key"] for r in payload["resolutions"]],
            [f"k{i:03d}" for i in range(100)],
        )
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]],
            ["hi" if i % 2 else "lo" for i in range(100)],
        )
        # The 200 previewed entries are all still uncommitted.
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 200)
        self.assertEqual(metrics["conflictKeys"], 100)

    def test_results_follow_request_order(self) -> None:
        self.seed_conflict("k1", low="a", high="b")
        self.seed_conflict("k2", low="c", high="d")
        doc = batch_document(
            batch_entry("k2", "r9", "zzz-fix", {"r1": 1, "r2": 1, "r9": 1}),
            batch_entry("k1", "r3", "aaa-fix", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual([r["key"] for r in payload["resolutions"]], ["k2", "k1"])


class PlanReplayTests(HttpServerTestCase):
    def test_mixed_new_and_replayed_entries(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        # Commit the k2 resolution first; the plan then replays it alongside
        # a preview of the new k1 repair.
        self.assertEqual(
            self.post_auto(
                "k2",
                {
                    "replicaId": "r3",
                    "operationId": "f2",
                    "clock": {"r1": 1, "r2": 1, "r3": 2},
                    "policy": "lowest_identity",
                },
            )[0],
            201,
        )
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        _, page_before = self.get_sync()
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["v1", "v1"]
        )
        # The plan appended nothing, not even for its one "accepted" entry.
        _, page_after = self.get_sync()
        self.assertEqual(page_after, page_before)
        _, state1 = self.get_state("k1")
        self.assertEqual(state1["status"], "conflict")

    def test_all_replay_plan_stays_planned_with_zero_accepted(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        self.assertEqual(self.post_batch(doc)[0], 201)
        _, page_before = self.get_sync()
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        _, page_after = self.get_sync()
        self.assertEqual(page_after, page_before)

    def test_replay_after_key_moved_on_reports_original_value(self) -> None:
        self.seed_conflict("k")
        doc = batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        self.assertEqual(self.post_batch(doc)[0], 201)
        # A later concurrent write reopens the conflict on k.
        self.post_operation("r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2}))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        # The plan's replay is answered from the committed operation and
        # reports the originally chosen value.
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(payload["resolutions"][0]["value"], "v1")


class PlanConflictTests(HttpServerTestCase):
    def assert_still_conflicted(self, key: str, candidates: int = 2) -> None:
        _, state = self.get_state(key)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), candidates)

    def test_missing_key_entry_is_409_and_changes_nothing(self) -> None:
        self.seed_conflict("k1")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("absent", "r3", "f2", {"r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_still_conflicted("k1")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_unconflicted_key_entry_is_409(self) -> None:
        self.post_operation("r1", operation("o9", "k2", "solo", {"r1": 9}))
        status, payload = self.post_plan(
            batch_document(batch_entry("k2", "r3", "f2", {"r1": 10, "r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_plan(
            batch_document(batch_entry("k", "r3", "f", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_clock_not_dominating_is_409(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            # Concurrent with k2's r2 candidate: a legal clock that does not
            # dominate the live candidates.
            batch_entry("k2", "r3", "f2", {"r1": 1, "r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_still_conflicted("k1")
        self.assert_still_conflicted("k2")

    def test_operation_conflict_on_rebound_identity(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        self.assertEqual(
            self.post_auto(
                "k1",
                {
                    "replicaId": "r3",
                    "operationId": "f1",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "policy": "lowest_identity",
                },
            )[0],
            201,
        )
        doc = batch_document(
            batch_entry("k2", "r3", "f9", {"r1": 1, "r2": 1, "r3": 9}),
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assert_still_conflicted("k2")

    def test_identity_bound_to_plain_write_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r3", operation("f1", "other", "v", {"r3": 1}))
        status, payload = self.post_plan(
            batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_different_policy_under_known_identity_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        good = {
            "replicaId": "r3",
            "operationId": "f1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        self.assertEqual(self.post_auto("k", good)[0], 201)
        status, payload = self.post_plan(
            batch_document(batch_entry("k", "r3", "f1", good["clock"], "highest_identity"))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_conflict_kind_follows_first_failing_entry(self) -> None:
        self.seed_conflict("k1")
        self.post_operation("r3", operation("bound", "x", "v", {"r3": 1}))
        doc = batch_document(
            batch_entry("missing-a", "r3", "fa", {"r3": 2}),
            batch_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        doc = batch_document(
            batch_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
            batch_entry("missing-a", "r3", "fa", {"r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assert_still_conflicted("k1")


class PlanValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict("k")
        valid = batch_document(batch_entry("k"))
        bad_bodies = [
            b"{oops",
            [],
            {},
            {"operations": [batch_entry("k")]},
            dict(valid, extra=1),
            batch_document(),
            batch_document(
                *[batch_entry(f"k{i}", "r3", f"f{i}", {"r3": 1}) for i in range(101)]
            ),
            batch_document(dict(batch_entry("k"), extra=1)),
            batch_document({k: v for k, v in batch_entry("k").items() if k != "key"}),
            batch_document(dict(batch_entry("k"), key="")),
            batch_document(dict(batch_entry("k"), replicaId="")),
            batch_document(dict(batch_entry("k"), operationId="")),
            batch_document(dict(batch_entry("k"), clock={"r2": 2})),
            batch_document(dict(batch_entry("k"), clock={"r3": -1})),
            batch_document(dict(batch_entry("k"), policy="middle")),
        ]
        for body in bad_bodies:
            status, payload = self.post_plan(body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_float_clocks_are_400(self) -> None:
        self.seed_conflict("k")
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                doc = batch_document(batch_entry("k", clock={"r1": 1, "r2": 1, "r3": tick}))
                status, payload = self.post_plan(doc)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for literal in ("NaN", "Infinity", "-Infinity", "-0.0"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f1",'
                    b'"clock":{"r1":1,"r2":1,"r3":' + literal.encode("ascii")
                    + b'},"policy":"lowest_identity"}]}'
                )
                status, payload = self.post_plan(raw)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_duplicate_keys_and_identities_are_400(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        duplicate_key = batch_document(
            batch_entry("k1", "r3", "f1"),
            batch_entry("k1", "r4", "f2", {"r1": 1, "r2": 1, "r4": 1}),
        )
        status, payload = self.post_plan(duplicate_key)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        duplicate_identity = batch_document(
            batch_entry("k1", "r3", "same", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "same", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_plan(duplicate_identity)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class PlanRoutingTests(HttpServerTestCase):
    def test_bad_path_shapes_are_404_before_body_checks(self) -> None:
        for path in (
            "/v1/resolve/auto/plan/extra",
            "/v1/resolve/auto/plan/extra/two",
            "/v1/resolve/auto/",
            "/v1/resolve/auto",
            "/v1/resolve",
            "/v1/resolve/auto/plan/",
            "/v1/unknown/route",
        ):
            # Even a malformed body on a wrong shape is a 404: the route
            # decision precedes every body check.
            status, payload = self.request("POST", path, b"{not json")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_plan_route_is_unknown(self) -> None:
        status, payload = self.request("GET", PLAN_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_parameters_are_400_before_the_body(self) -> None:
        self.seed_conflict("k")
        for query in ("?x=1", "?x=", "?x", "?=1", "?x=1&x=2", "?x=1&y=2"):
            status, payload = self.request(
                "POST", PLAN_PATH + query, b"{not json"
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A valid body cannot rescue a route that carries a query parameter.
        status, payload = self.post_plan_batch_with_query("?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def post_plan_batch_with_query(self, query: str) -> tuple[int, object]:
        return self.request(
            "POST",
            PLAN_PATH + query,
            batch_document(batch_entry("k")),
        )


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

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})


class PlanAuthTests(unittest.TestCase):
    def start_server(self, **kwargs) -> tuple[SemanticStateServer, int]:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server, server.server_address[1]

    def call(
        self, port: int, headers: dict | None = None, body: object = batch_document()
    ) -> tuple[int, dict, str | None]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        conn.request("POST", PLAN_PATH, body=json.dumps(body), headers=merged)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, challenge

    def test_health_stays_anonymous_under_single_token_mode(self) -> None:
        _server, port = self.start_server(auth_token="secret-token")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        conn.close()

    def test_missing_duplicated_malformed_or_mismatched_bearer_is_401(self) -> None:
        _server, port = self.start_server(auth_token="secret-token")
        # Missing header.
        status, payload, challenge = self.call(port, headers={})
        self.assertEqual((status, payload), (401, {"error": "unauthorized"}))
        self.assertEqual(challenge, "Bearer")
        # Malformed header.
        status, payload, challenge = self.call(port, headers={"Authorization": "Basic abc"})
        self.assertEqual((status, payload), (401, {"error": "unauthorized"}))
        self.assertEqual(challenge, "Bearer")
        # Malformed bearer (no space).
        status, payload, challenge = self.call(
            port, headers={"Authorization": "Bearersecret-token"}
        )
        self.assertEqual((status, payload), (401, {"error": "unauthorized"}))
        # Token mismatch.
        status, payload, challenge = self.call(
            port, headers={"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual((status, payload), (401, {"error": "unauthorized"}))
        self.assertEqual(challenge, "Bearer")
        # Duplicated headers are rejected: send them raw with a valid
        # declared length so the request reaches the authentication check
        # (a missing Content-Length would be a 400 before authentication).
        body = json.dumps(batch_document()).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", PLAN_PATH)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Authorization", "Bearer secret-token")
        conn.putheader("Authorization", "Bearer secret-token")
        conn.endheaders(body)
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(response.read()), {"error": "unauthorized"})
        conn.close()

    def test_authorized_single_token_previews(self) -> None:
        server, port = self.start_server(auth_token="secret-token")
        # Seed a conflict directly on the store so the preview has something
        # to plan against.
        self.assertEqual(
            server.store.apply_operation(
                "r1",
                {"operationId": "o1", "key": "k", "value": "v1", "clock": {"r1": 1}},
            ),
            HTTPStatus.CREATED,
        )
        self.assertEqual(
            server.store.apply_operation(
                "r2",
                {"operationId": "o2", "key": "k", "value": "v2", "clock": {"r2": 1}},
            ),
            HTTPStatus.CREATED,
        )
        status, payload, _challenge = self.call(
            port,
            headers={"Authorization": "Bearer secret-token"},
            body=batch_document(batch_entry("k")),
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["resolutions"][0]["value"], "v1")

    def test_scope_mode_gate(self) -> None:
        policy = {
            "reader-token": ["read"],
            "writer-token": ["write"],
            "admin-token": ["read", "write", "admin"],
        }
        server, port = self.start_server(auth_scopes=dict(policy))
        self.assertEqual(
            server.store.apply_operation(
                "r1",
                {"operationId": "o1", "key": "k", "value": "v1", "clock": {"r1": 1}},
            ),
            HTTPStatus.CREATED,
        )
        self.assertEqual(
            server.store.apply_operation(
                "r2",
                {"operationId": "o2", "key": "k", "value": "v2", "clock": {"r2": 1}},
            ),
            HTTPStatus.CREATED,
        )
        doc = batch_document(batch_entry("k"))
        # A write-only token cannot preview: the plan needs read (or admin),
        # and the 403 carries no challenge.
        status, payload, challenge = self.call(
            port, headers={"Authorization": "Bearer writer-token"}, body=doc
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        # The store was not touched by the rejected preview.
        self.assertEqual(len(server.store._accepted), 2)
        # A read token previews successfully.
        status, payload, challenge = self.call(
            port, headers={"Authorization": "Bearer reader-token"}, body=doc
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["resolutions"][0]["value"], "v1")
        self.assertIsNone(challenge)
        # The admin scope covers the read-only preview too.
        status, payload, _challenge = self.call(
            port, headers={"Authorization": "Bearer admin-token"}, body=doc
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")


class PersistentPlanTestCase(unittest.TestCase):
    """The plan writes nothing and previews identically across restarts."""

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
        return response.status, payload, raw

    def seed_conflicts(self, server: SemanticStateServer, *keys: str) -> None:
        for key in keys:
            for replica, op_id in (("r1", f"o1-{key}"), ("r2", f"o2-{key}")):
                status, _payload, _raw = self.request(
                    server,
                    "POST",
                    f"/v1/replicas/{replica}/operations",
                    operation(op_id, key, f"v-{key}-{replica}", {replica: 1}),
                )
                self.assertEqual(status, 201)

    def test_plan_writes_no_file_and_creates_no_temp_file(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        before = self.data_file.read_bytes()
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, payload, _raw = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 2)
        # The data file is byte-for-byte the file the seeds produced ...
        self.assertEqual(self.data_file.read_bytes(), before)
        # ... no temporary file was left behind ...
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # ... and the log still holds only the four seeds.
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)

    def test_same_state_previews_identically_before_and_after_restart(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        _status, before, before_raw = self.request(server, "POST", PLAN_PATH, doc)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        _status, after, after_raw = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(after, before)
        self.assertEqual(after_raw, before_raw)
        self.assertEqual(
            [r["value"] for r in after["resolutions"]], ["v-k1-r1", "v-k2-r2"]
        )
        # Nothing was committed by either preview.
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)

    def test_replay_plan_is_stable_across_restart(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
        )
        status, _payload, _raw = self.request(server, "POST", BATCH_PATH, doc)
        self.assertEqual(status, 201)
        _status, before, _raw = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(before["accepted"], 0)
        self.assertEqual(before["replayed"], 1)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        _status, after, _raw = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(after, before)
        self.assertEqual(after["resolutions"][0]["value"], "v-k1-r1")


class PlanConcurrencyTests(HttpServerTestCase):
    def test_plans_observe_a_complete_snapshot_and_never_mutate(self) -> None:
        # Three conflicted keys; three batches each repair one key. Planning
        # threads run concurrently with the committing threads: every plan
        # must be internally consistent (each key either still conflicted —
        # its entry "accepted" — or already repaired by the matching
        # identity — "replayed"), and plans alone must never append a record.
        keys = ("k1", "k2", "k3")
        for index, key in enumerate(keys):
            self.seed_conflict(key, low=f"lo{index}", high=f"hi{index}")

        docs = [
            batch_document(
                batch_entry(key, "r3", f"f-{key}", {"r1": 1, "r2": 1, "r3": tick})
            )
            for tick, key in enumerate(keys, start=1)
        ]
        outcomes: list[object] = []
        lock = threading.Lock()

        def committer(doc: dict) -> None:
            status, payload = self.post_batch(json.loads(json.dumps(doc)))
            with lock:
                outcomes.append(("commit", status, payload))

        def planner(doc: dict) -> None:
            status, payload = self.post_plan(json.loads(json.dumps(doc)))
            with lock:
                outcomes.append(("plan", status, payload))

        threads = []
        for doc in docs:
            threads.append(threading.Thread(target=committer, args=(doc,)))
            threads.append(threading.Thread(target=planner, args=(doc,)))
            threads.append(threading.Thread(target=planner, args=(doc,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        for kind, status, payload in outcomes:
            if kind == "commit":
                self.assertIn(status, (200, 201))
            else:
                # A plan either sees the conflict still open (200 planned,
                # its single entry accepted) or the matching repair already
                # committed (200 planned, the entry replayed). It never sees
                # the half-state of an in-flight commit and never fails.
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "planned")
                self.assertEqual(payload["accepted"] + payload["replayed"], 1)
        # Exactly the three commits landed — plans appended nothing.
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 9)


if __name__ == "__main__":
    unittest.main()
