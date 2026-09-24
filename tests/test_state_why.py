"""Tests for the read-only causal conflict explanation endpoint.

Covers ``GET /v1/states/{key}/why``: route-shape 404s (which take
precedence over query checks), the 400 no-query-parameters contract,
404 for unseen or candidate-less keys, resolved/conflict payloads with
pairwise clock relations and the two identity-policy recommendations,
compact JSON terminated by one newline, strict read-only behavior,
bearer authentication, and byte-identical explanations across a
``--data-file`` restart.
"""

from __future__ import annotations

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
    clock_dominates,
)

TOKEN = "why-secret-token"
AUTH_HEADER = f"Bearer {TOKEN}"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class WhyServerTests(unittest.TestCase):
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

    def raw_request(self, method: str, path: str, body: object = None) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        status = response.status
        conn.close()
        return status, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, raw = self.raw_request(method, path, body)
        return status, json.loads(raw.decode("utf-8"))

    def post_operation(
        self, replica: str, operation_id: str, key: str, value: str, clock: dict
    ) -> tuple[int, dict]:
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            operation(operation_id, key, value, clock),
        )

    def why(self, key: str) -> tuple[int, dict]:
        return self.request("GET", f"/v1/states/{key}/why")

    def why_raw(self, path: str) -> tuple[int, bytes]:
        return self.raw_request("GET", path)

    # -- route shape: 404 takes precedence over the query check --

    def test_unknown_route_is_404(self) -> None:
        status, payload = self.request("GET", "/v1/why")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_key_segment_is_404_even_with_query(self) -> None:
        status, payload = self.request("GET", "/v1/states//why?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_extra_segment_is_404_even_with_query(self) -> None:
        status, payload = self.request("GET", "/v1/states/k/why/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_trailing_slash_is_404(self) -> None:
        status, payload = self.request("GET", "/v1/states/k/why/")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_states_root_route_is_not_the_why_route(self) -> None:
        # A literal key named "why" stays on the three-segment state route.
        self.post_operation("r1", "op-1", "why", "v", {"r1": 1})
        status, payload = self.request("GET", "/v1/states/why")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertNotIn("relations", payload)

    def test_non_get_method_on_why_path_is_404(self) -> None:
        status, payload = self.request("POST", "/v1/states/k/why", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # -- query parameters are rejected after the route shape --

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", "op-1", "k", "v", {"r1": 1})
        for query in ("?x=1", "?x=", "?x", "?=1", "?x=1&y=2", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload = self.request("GET", f"/v1/states/k/why{query}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_query_is_accepted(self) -> None:
        self.post_operation("r1", "op-1", "k", "v", {"r1": 1})
        status, payload = self.request("GET", "/v1/states/k/why?")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")

    def test_query_rejected_without_reading_or_changing_state(self) -> None:
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        _, before = self.request("GET", "/v1/metrics")
        status, payload = self.request("GET", "/v1/states/k/why?bogus=1")
        self.assertEqual((status, payload), (400, {"error": "invalid_request"}))
        _, after = self.request("GET", "/v1/metrics")
        self.assertEqual(after, before)
        # The rejected request explains nothing and leaves the conflict.
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")

    # -- 404 semantics --

    def test_key_never_seen_is_404(self) -> None:
        status, payload = self.why("absent")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_empty_candidate_set_is_404_at_the_store_boundary(self) -> None:
        # Over HTTP a current key always holds at least one candidate, but
        # the store contract is explicit: an empty candidate set — however
        # it arose — is the same 404 as a key that never appeared, even
        # though the key name itself has been seen.
        store = StateStore()
        store._candidates["ghost"] = []
        status, payload = store.get_state_why("ghost")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})
        self.assertEqual(store.get_state_why("never")[0], HTTPStatus.NOT_FOUND)

    def test_relation_labels_at_the_store_boundary(self) -> None:
        # The three relation labels for an arbitrary ordered pair.
        store = StateStore()
        candidate = lambda rep, oid, value, clock: {  # noqa: E731
            "replicaId": rep,
            "operationId": oid,
            "value": value,
            "clock": clock,
        }
        a = candidate("r1", "a", "x", {"r1": 2})
        b = candidate("r2", "b", "y", {"r1": 1})
        c = candidate("r3", "c", "z", {"r3": 1})
        # a dominates b (same component, strictly greater); both are
        # concurrent with c (disjoint components).
        self.assertTrue(clock_dominates(a["clock"], b["clock"]))
        relations = {
            (rel["from"]["operationId"], rel["to"]["operationId"]): rel["relation"]
            for rel in store._candidate_relations_locked([a, b, c])
        }
        self.assertEqual(relations[("a", "b")], "dominates")
        self.assertEqual(relations[("a", "c")], "concurrent")
        self.assertEqual(relations[("b", "c")], "concurrent")
        # A candidate later in query order that dominates the earlier one
        # is reported as "overwritten".
        d = candidate("r4", "d", "q", {"r3": 2})
        relations_dc = {
            (rel["from"]["operationId"], rel["to"]["operationId"]): rel["relation"]
            for rel in store._candidate_relations_locked([c, d])
        }
        self.assertTrue(clock_dominates(d["clock"], c["clock"]))
        self.assertEqual(relations_dc[("c", "d")], "overwritten")

    def test_recommendation_labels_at_the_store_boundary(self) -> None:
        store = StateStore()
        candidate = lambda rep, oid, value: {  # noqa: E731
            "replicaId": rep,
            "operationId": oid,
            "value": value,
            "clock": {rep: 1},
        }
        ordered = [candidate("r1", "op-z", "zzz"), candidate("r2", "op-a", "aaa")]
        self.assertEqual(
            store._identity_recommendation_locked("lowest_identity", ordered),
            {"policy": "lowest_identity", "replicaId": "r1", "operationId": "op-z", "value": "zzz"},
        )
        self.assertEqual(
            store._identity_recommendation_locked("highest_identity", ordered),
            {"policy": "highest_identity", "replicaId": "r2", "operationId": "op-a", "value": "aaa"},
        )

    # -- resolved: single source --

    def test_single_candidate_is_resolved_with_unique_source(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, payload = self.why("color")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "color")
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [
                {
                    "replicaId": "r1",
                    "operationId": "op-1",
                    "value": "blue",
                    "clock": {"r1": 1},
                }
            ],
        )
        self.assertEqual(payload["relations"], [])
        self.assertEqual(
            payload["recommendations"],
            [
                {
                    "policy": "lowest_identity",
                    "replicaId": "r1",
                    "operationId": "op-1",
                    "value": "blue",
                },
                {
                    "policy": "highest_identity",
                    "replicaId": "r1",
                    "operationId": "op-1",
                    "value": "blue",
                },
            ],
        )

    def test_concurrent_candidates_agreeing_in_value_are_resolved(self) -> None:
        # Concurrent clocks (neither dominates) but the same value: the
        # state route reports resolved; why keeps both candidates as the
        # value's sources and explains their concurrency.
        self.post_operation("r2", "op-2", "k", "same", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "same", {"r1": 1})
        status, payload = self.why("k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual([c["operationId"] for c in payload["candidates"]], ["op-1", "op-2"])
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": {"replicaId": "r1", "operationId": "op-1"},
                    "to": {"replicaId": "r2", "operationId": "op-2"},
                    "relation": "concurrent",
                }
            ],
        )
        for recommendation in payload["recommendations"]:
            self.assertEqual(recommendation["value"], "same")

    def test_dominated_candidate_is_not_a_source(self) -> None:
        # op-2 dominates op-1, so the stale op-1 is not a current candidate
        # and the explanation reports exactly one source.
        self.post_operation("r1", "op-1", "k", "old", {"r1": 1})
        self.post_operation("r1", "op-2", "k", "new", {"r1": 2})
        status, payload = self.why("k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual([c["operationId"] for c in payload["candidates"]], ["op-2"])
        self.assertEqual(payload["relations"], [])

    # -- conflict: pairwise concurrency explanation --

    def test_two_concurrent_conflicting_candidates(self) -> None:
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        status, payload = self.why("k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                {
                    "replicaId": "r1",
                    "operationId": "op-1",
                    "value": "v1",
                    "clock": {"r1": 1},
                },
                {
                    "replicaId": "r2",
                    "operationId": "op-2",
                    "value": "v2",
                    "clock": {"r2": 1},
                },
            ],
        )
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": {"replicaId": "r1", "operationId": "op-1"},
                    "to": {"replicaId": "r2", "operationId": "op-2"},
                    "relation": "concurrent",
                }
            ],
        )
        self.assertEqual(
            payload["recommendations"],
            [
                {
                    "policy": "lowest_identity",
                    "replicaId": "r1",
                    "operationId": "op-1",
                    "value": "v1",
                },
                {
                    "policy": "highest_identity",
                    "replicaId": "r2",
                    "operationId": "op-2",
                    "value": "v2",
                },
            ],
        )

    def test_three_way_conflict_explains_every_pair(self) -> None:
        self.post_operation("r3", "c", "k", "v3", {"r3": 1})
        self.post_operation("r1", "a", "k", "v1", {"r1": 1})
        self.post_operation("r2", "b", "k", "v2", {"r2": 1})
        status, payload = self.why("k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        # Candidates follow the existing query order: identity ascending.
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["candidates"]],
            [("r1", "a"), ("r2", "b"), ("r3", "c")],
        )
        # Every unordered pair appears once, in query order, and every pair
        # is concurrent: that is precisely why none dominates another.
        self.assertEqual(
            [
                (
                    (rel["from"]["replicaId"], rel["from"]["operationId"]),
                    (rel["to"]["replicaId"], rel["to"]["operationId"]),
                    rel["relation"],
                )
                for rel in payload["relations"]
            ],
            [
                (("r1", "a"), ("r2", "b"), "concurrent"),
                (("r1", "a"), ("r3", "c"), "concurrent"),
                (("r2", "b"), ("r3", "c"), "concurrent"),
            ],
        )
        # The clock semantics the explanation relies on.
        for left, right in (
            ({"r1": 1}, {"r2": 1}),
            ({"r1": 1}, {"r3": 1}),
            ({"r2": 1}, {"r3": 1}),
        ):
            self.assertFalse(clock_dominates(left, right))
            self.assertFalse(clock_dominates(right, left))

    def test_relations_cover_exactly_each_unordered_pair_once(self) -> None:
        replicas = ("r1", "r2", "r3", "r4")
        for replica in replicas:
            self.post_operation(
                replica, f"op-{replica}", "k", f"value-{replica}", {replica: 1}
            )
        _, payload = self.why("k")
        pairs = {
            (
                (rel["from"]["replicaId"], rel["from"]["operationId"]),
                (rel["to"]["replicaId"], rel["to"]["operationId"]),
            )
            for rel in payload["relations"]
        }
        expected = {
            (("r1", "op-r1"), ("r2", "op-r2")),
            (("r1", "op-r1"), ("r3", "op-r3")),
            (("r1", "op-r1"), ("r4", "op-r4")),
            (("r2", "op-r2"), ("r3", "op-r3")),
            (("r2", "op-r2"), ("r4", "op-r4")),
            (("r3", "op-r3"), ("r4", "op-r4")),
        }
        self.assertEqual(pairs, expected)
        self.assertTrue(all(rel["relation"] == "concurrent" for rel in payload["relations"]))

    def test_recommendations_match_automatic_resolution_policies(self) -> None:
        # Same replica, operation ids deliberately not value-sorted so the
        # policy choice is by identity, not by value.
        self.post_operation("r1", "op-z", "k", "zzz", {"r1": 1})
        self.post_operation("r2", "op-a", "k", "aaa", {"r2": 1})
        _, explanation = self.why("k")
        lowest = next(r for r in explanation["recommendations"] if r["policy"] == "lowest_identity")
        highest = next(
            r for r in explanation["recommendations"] if r["policy"] == "highest_identity"
        )
        self.assertEqual((lowest["replicaId"], lowest["operationId"], lowest["value"]),
                         ("r1", "op-z", "zzz"))
        self.assertEqual((highest["replicaId"], highest["operationId"], highest["value"]),
                         ("r2", "op-a", "aaa"))
        # The automatic repair endpoints select the same candidate values.
        for policy, expected_value in (
            ("lowest_identity", "zzz"),
            ("highest_identity", "aaa"),
        ):
            self.server.store = type(self.server.store)()
            self.post_operation("r1", "op-z", "k", "zzz", {"r1": 1})
            self.post_operation("r2", "op-a", "k", "aaa", {"r2": 1})
            status, repaired = self.request(
                "POST",
                "/v1/states/k/resolve/auto",
                {
                    "replicaId": "r9",
                    "operationId": f"fix-{policy}",
                    "clock": {"r1": 1, "r2": 1, "r9": 1},
                    "policy": policy,
                },
            )
            self.assertEqual(status, 201)
            self.assertEqual(repaired["value"], expected_value)

    # -- read-only behavior --

    def test_explanation_creates_no_operation_or_repair(self) -> None:
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        _, metrics_before = self.request("GET", "/v1/metrics")
        for _ in range(3):
            status, _ = self.why("k")
            self.assertEqual(status, 200)
        _, metrics_after = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["status"], "conflict")
        _, sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(sync["nextCursor"], metrics_before["acceptedOperations"])

    def test_response_has_exactly_the_five_information_categories(self) -> None:
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        _, payload = self.why("k")
        self.assertEqual(
            set(payload), {"key", "status", "candidates", "relations", "recommendations"}
        )
        for candidate in payload["candidates"]:
            self.assertEqual(set(candidate), {"replicaId", "operationId", "value", "clock"})
        for rel in payload["relations"]:
            self.assertEqual(set(rel), {"from", "to", "relation"})
            self.assertEqual(set(rel["from"]), {"replicaId", "operationId"})
            self.assertEqual(set(rel["to"]), {"replicaId", "operationId"})
        for recommendation in payload["recommendations"]:
            self.assertEqual(
                set(recommendation), {"policy", "replicaId", "operationId", "value"}
            )

    # -- encoding: compact JSON, one newline, integers only --

    def test_success_body_is_compact_json_ending_in_one_newline(self) -> None:
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        status, raw = self.why_raw("/v1/states/k/why")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b" ", raw)
        self.assertEqual(json.loads(raw.decode("utf-8")), json.loads(raw.strip()))
        # Re-serializing compactly reproduces the body apart from its line end.
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(raw, json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n")

    def test_clock_components_are_json_integers(self) -> None:
        self.post_operation("r1", "op-9", "k", "v", {"r1": 10, "r2": 0})
        status, raw_bytes = self.why_raw("/v1/states/k/why")
        self.assertEqual(status, 200)
        text = raw_bytes.decode("utf-8")
        # No decimal point (no floats), and no non-finite tokens.
        self.assertNotIn(".", text)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("-0", text)
        payload = json.loads(text)
        clock = payload["candidates"][0]["clock"]
        self.assertEqual(clock, {"r1": 10, "r2": 0})
        for tick in clock.values():
            self.assertIsInstance(tick, int)
            self.assertNotIsInstance(tick, bool)

    def test_percent_encoded_key_is_decoded(self) -> None:
        self.post_operation("r1", "op-1", "a/b", "v", {"r1": 1})
        status, payload = self.request("GET", "/v1/states/a%2Fb/why")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "a/b")


class WhyAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, auth_token=TOKEN)
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

    def request(
        self, method: str, path: str, auth: str | None = AUTH_HEADER
    ) -> tuple[int, dict, dict]:
        headers = {} if auth is None else {"Authorization": auth}
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers_out = dict(response.getheaders())
        conn.close()
        return response.status, payload, headers_out

    def test_health_stays_anonymous(self) -> None:
        status, _, _ = self.request("GET", "/health", auth=None)
        self.assertEqual(status, 200)

    def test_missing_authorization_is_401(self) -> None:
        status, payload, headers = self.request("GET", "/v1/states/k/why", auth=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_duplicate_authorization_is_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/v1/states/k/why")
        conn.putheader("Authorization", AUTH_HEADER)
        conn.putheader("Authorization", AUTH_HEADER)
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_mismatched_authorization_is_401(self) -> None:
        status, payload, headers = self.request("GET", "/v1/states/k/why", auth="Bearer nope")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_failed_auth_returns_401_before_route_and_query_checks(self) -> None:
        # A malformed route and a present query parameter must not turn the
        # 401 into a 404/400: authentication precedes route matching.
        for path in ("/v1/states//why?x=1", "/v1/states/k/why/extra?x=1"):
            status, payload, _ = self.request("GET", path, auth=None)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})

    def test_failed_auth_leaves_state_unchanged(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r1/operations",
            body=json.dumps(operation("op-1", "k", "v", {"r1": 1})),
            headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER},
        )
        self.assertEqual(conn.getresponse().status, 201)
        conn.close()

        status, metrics_before, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _, _ = self.request("GET", "/v1/states/k/why", auth=None)
            self.assertEqual(status, 401)
        status, metrics_after, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics_after, metrics_before)

    def test_authorized_explanation_succeeds(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r1/operations",
            body=json.dumps(operation("op-1", "k", "v", {"r1": 1})),
            headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER},
        )
        conn.getresponse().read()
        conn.close()
        status, payload, _ = self.request("GET", "/v1/states/k/why")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")


class WhyPersistenceRestartTests(unittest.TestCase):
    """The same candidate state explains identically across a restart."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-why-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_path = os.path.join(self.tmpdir, "state.json")

    def _start(self) -> tuple[SemanticStateServer, threading.Thread, int]:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_path
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, server.server_address[1]

    def _get(self, port: int, path: str) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read()
        status = response.status
        conn.close()
        return status, raw

    def _post(self, port: int, replica: str, document: dict) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            body=json.dumps(document),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status = response.status
        response.read()
        conn.close()
        return status

    def test_explanation_is_byte_identical_after_restart(self) -> None:
        server, thread, port = self._start()
        try:
            self.assertEqual(
                self._post(port, "r2", operation("op-2", "k", "v2", {"r2": 1})), 201
            )
            self.assertEqual(
                self._post(port, "r1", operation("op-1", "k", "v1", {"r1": 1})), 201
            )
            self.assertEqual(
                self._post(port, "r3", operation("op-3", "k", "v3", {"r3": 1})), 201
            )
            status, before = self._get(port, "/v1/states/k/why")
            self.assertEqual(status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        restarted, restarted_thread, restarted_port = self._start()
        try:
            status, after = self._get(restarted_port, "/v1/states/k/why")
            self.assertEqual(status, 200)
            self.assertEqual(after, before)
        finally:
            restarted.shutdown()
            restarted.server_close()
            restarted_thread.join(timeout=5)

    def test_repaired_key_explains_the_surviving_source_after_restart(self) -> None:
        server, thread, port = self._start()
        try:
            self._post(port, "r1", operation("op-1", "k", "v1", {"r1": 1}))
            self._post(port, "r2", operation("op-2", "k", "v2", {"r2": 1}))
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST",
                "/v1/states/k/resolve/auto",
                body=json.dumps(
                    {
                        "replicaId": "r3",
                        "operationId": "fix-1",
                        "clock": {"r1": 1, "r2": 1, "r3": 1},
                        "policy": "lowest_identity",
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.close()
            status, before = self._get(port, "/v1/states/k/why")
            self.assertEqual(status, 200)
            before_payload = json.loads(before)
            self.assertEqual(before_payload["status"], "resolved")
            self.assertEqual([c["operationId"] for c in before_payload["candidates"]], ["fix-1"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        restarted, restarted_thread, restarted_port = self._start()
        try:
            status, after = self._get(restarted_port, "/v1/states/k/why")
            self.assertEqual(status, 200)
            after_payload = json.loads(after)
            self.assertEqual(after_payload, before_payload)
            self.assertEqual(after_payload["recommendations"][0]["value"], "v1")
        finally:
            restarted.shutdown()
            restarted.server_close()
            restarted_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
