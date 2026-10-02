"""HTTP, sync, and persistence tests for the ``causal_dominant`` policy.

The policy extends the three automatic-resolution entries —

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

— with a causality-first deterministic selection: the resolution value is
taken from the unique current candidate whose vector clock strictly
dominates every other current candidate's clock. When no such unique
strict dominator exists the entries answer HTTP 409
``{"error":"causal_ambiguity"}`` (the batch and the preview fail as a
whole, with no partial results).

The live candidate frontier only ever holds pairwise concurrent candidates
(a dominating write clears the candidates it dominates), so a strict
dominator among conflicting candidates cannot be produced by writes
alone; the success-path tests seed the candidate set directly, exactly
the state the frontier would hold if dominated versions were retained.
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
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    parse_auto_resolve_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def auto_request(
    replica: str, operation_id: str, clock: dict, policy: str = "causal_dominant"
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


def candidate(replica: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
    }


class ParseCausalDominantPayloadTests(unittest.TestCase):
    def test_causal_dominant_policy_is_accepted(self) -> None:
        payload = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "fix-1", {"r1": 1, "r3": 1}))
        )
        self.assertEqual(payload["policy"], "causal_dominant")

    def test_unknown_policy_is_rejected(self) -> None:
        for policy in ("", "causal-dominant", "CAUSAL_DOMINANT", "causal"):
            body = auto_request("r3", "fix-1", {"r1": 1, "r3": 1}, policy)
            with self.assertRaises(ValueError, msg=policy):
                parse_auto_resolve_payload(json.dumps(body))


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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
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
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_auto(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve/auto", body)

    def post_batch(self, entries: list) -> tuple[int, object]:
        return self.request("POST", "/v1/resolve/auto/batch", {"resolutions": entries})

    def post_plan(self, entries: list) -> tuple[int, object]:
        return self.request("POST", "/v1/resolve/auto/plan", {"resolutions": entries})

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self) -> tuple[int, object]:
        return self.request("GET", "/v1/sync/operations")

    def seed_conflict(self, key: str = "k") -> None:
        """Two concurrent writes with different values on ``key``."""
        self.assertEqual(
            self.post_operation("r1", operation("o1", key, "v1", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", key, "v2", {"r2": 1}))[0], 201
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")

    def seed_dominance(self, key: str = "k") -> None:
        """Directly seed a conflict whose candidates hold a unique strict
        dominator: ``new``'s clock strictly dominates ``old``'s.

        The live frontier never holds such a pair on its own (a dominating
        write clears the dominated candidate), so the state is staged
        directly — the selection rule itself is what is under test.
        """
        store = self.server.store
        with store._lock:
            store._candidates[key] = [
                candidate("r1", "o1", "old", {"r1": 1}),
                candidate("r2", "o2", "new", {"r1": 1, "r2": 1}),
            ]

    def good_auto_request(
        self, operation_id: str = "fix-1", policy: str = "causal_dominant"
    ) -> dict:
        return auto_request("r3", operation_id, {"r1": 1, "r2": 1, "r3": 1}, policy)


class CausalDominantHappyPathTests(HttpServerTestCase):
    def test_unique_strict_dominator_value_is_chosen(self) -> None:
        self.seed_dominance()
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "key": "k",
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "new",
                "policy": "causal_dominant",
            },
        )
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "new",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_dominator_is_chosen_regardless_of_identity_order(self) -> None:
        # The dominated candidate carries the lexicographically extreme
        # identity; causality, not identity ordering, decides.
        store = self.server.store
        with store._lock:
            store._candidates["k"] = [
                candidate("r1", "o1", "dominated", {"r1": 1}),
                candidate("r2", "o2", "dominant", {"r1": 1, "r2": 1}),
            ]
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "dominant")

    def test_success_is_an_ordinary_operation_in_the_shared_order(self) -> None:
        self.seed_dominance()
        status, _ = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        status, sync_payload = self.get_sync()
        self.assertEqual(status, 200)
        exported = [
            record["operation"] for record in sync_payload["operations"]
        ]
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["operationId"], "fix-1")
        self.assertEqual(exported[0]["value"], "new")
        status, audit = self.request("GET", "/v1/audit/keys/k/operations")
        self.assertEqual(status, 200)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)


class CausalDominantAmbiguityTests(HttpServerTestCase):
    def test_concurrent_candidates_are_ambiguous(self) -> None:
        self.seed_conflict()
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_equal_clocks_with_different_values_are_ambiguous(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r1": 1, "r2": 0}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_three_way_concurrent_candidates_are_ambiguous(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r4", operation("o4", "k", "v4", {"r4": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r4": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_ambiguity_appends_nothing_to_the_shared_log(self) -> None:
        self.seed_conflict()
        _, before = self.get_sync()
        status, _ = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        _, after = self.get_sync()
        self.assertEqual(before, after)


class CausalDominantValidationTests(HttpServerTestCase):
    def test_unknown_policy_is_400(self) -> None:
        self.seed_conflict()
        for policy in ("", "causal-dominant", "CAUSAL_DOMINANT"):
            status, payload = self.post_auto(
                "k", self.good_auto_request(policy=policy)
            )
            self.assertEqual(status, 400, policy)
            self.assertEqual(payload, {"error": "invalid_request"}, policy)

    def test_illegal_clock_is_400(self) -> None:
        self.seed_conflict()
        bad_clocks = [
            {"r1": 1, "r2": 1},  # missing the resolving replica
            {"r1": 1, "r2": 1, "r3": -1},  # negative component
            {"r1": 1, "r2": 1, "r3": 1.0},  # non-integer component
            {},  # empty clock
        ]
        for clock in bad_clocks:
            status, payload = self.post_auto(
                "k", auto_request("r3", "fix-1", clock)
            )
            self.assertEqual(status, 400, repr(clock))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(clock))

    def test_legal_clock_not_dominating_candidates_is_409(self) -> None:
        # A unique dominator exists, so the policy selects a value and only
        # the request-clock precondition fails: a legal clock that does not
        # dominate every current candidate is a resolution conflict.
        self.seed_dominance()
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_identity_policies_keep_400_for_non_dominating_clock(self) -> None:
        # The existing policies are unchanged: on the single-key entry a
        # legal but non-dominating clock stays an invalid request.
        self.seed_conflict()
        for policy in ("lowest_identity", "highest_identity"):
            status, payload = self.post_auto(
                "k",
                auto_request("r3", f"fix-{policy}", {"r1": 1, "r3": 1}, policy),
            )
            self.assertEqual(status, 400, policy)
            self.assertEqual(payload, {"error": "invalid_request"}, policy)


class CausalDominantConflictTests(HttpServerTestCase):
    def test_missing_key_is_409_resolution_conflict(self) -> None:
        status, payload = self.post_auto(
            "absent", auto_request("r3", "fix-1", {"r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_unconflicted_key_is_409_resolution_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "only", {"r1": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409_resolution_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_identity_without_policy_binding_is_409_operation_conflict(self) -> None:
        self.seed_conflict()
        # (r1, o1) committed as a plain write carries no policy binding.
        status, payload = self.post_auto(
            "k", auto_request("r1", "o1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class CausalDominantReplayTests(HttpServerTestCase):
    def test_same_binding_replay_is_200_with_first_value(self) -> None:
        self.seed_dominance()
        status, first = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        _, before = self.get_sync()
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], first["value"])
        self.assertEqual(payload["policy"], "causal_dominant")
        _, after = self.get_sync()
        self.assertEqual(before, after)

    def test_same_identity_different_clock_is_409_operation_conflict(self) -> None:
        self.seed_dominance()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 2})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_same_identity_different_policy_is_409_operation_conflict(self) -> None:
        self.seed_dominance()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        for policy in ("lowest_identity", "highest_identity"):
            status, payload = self.post_auto(
                "k", self.good_auto_request(policy=policy)
            )
            self.assertEqual(status, 409, policy)
            self.assertEqual(payload, {"error": "operation_conflict"}, policy)

    def test_same_identity_on_another_key_is_409_operation_conflict(self) -> None:
        self.seed_dominance()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        self.seed_dominance("other")
        status, payload = self.post_auto("other", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class CausalDominantBatchTests(HttpServerTestCase):
    def test_batch_commits_dominator_entries_in_request_order(self) -> None:
        self.seed_dominance("k1")
        store = self.server.store
        with store._lock:
            store._candidates["k2"] = [
                candidate("r1", "o1", "p", {"r1": 1}),
                candidate("r2", "o2", "q", {"r1": 1, "r2": 1}),
            ]
        entries = [
            dict(auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}), key="k1"),
            dict(auto_request("r4", "fix-2", {"r1": 1, "r2": 1, "r4": 1}), key="k2"),
        ]
        status, payload = self.post_batch(entries)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            payload["resolutions"],
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "fix-1",
                    "value": "new",
                    "policy": "causal_dominant",
                },
                {
                    "key": "k2",
                    "replicaId": "r4",
                    "operationId": "fix-2",
                    "value": "q",
                    "policy": "causal_dominant",
                },
            ],
        )
        # A full replay is 200 with every entry replayed.
        status, payload = self.post_batch(entries)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["new", "q"]
        )

    def test_batch_ambiguity_fails_the_whole_batch(self) -> None:
        self.seed_dominance("k1")
        self.seed_conflict("k2")  # concurrent: no unique dominator
        entries = [
            dict(auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}), key="k1"),
            dict(auto_request("r4", "fix-2", {"r1": 1, "r2": 1, "r4": 1}), key="k2"),
        ]
        status, payload = self.post_batch(entries)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})
        # No partial results: even the resolvable first entry is uncommitted.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, state = self.get_state("k2")
        self.assertEqual(state["status"], "conflict")

    def test_batch_mixed_policies_keep_per_policy_selection(self) -> None:
        self.seed_conflict("k1")
        self.post_operation("r1", operation("o1b", "k2", "w1", {"r1": 2}))
        self.post_operation("r2", operation("o2b", "k2", "w2", {"r2": 2}))
        entries = [
            dict(
                auto_request(
                    "r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_identity"
                ),
                key="k1",
            ),
            dict(
                auto_request(
                    "r4", "fix-2", {"r1": 2, "r2": 2, "r4": 1}, "highest_identity"
                ),
                key="k2",
            ),
        ]
        status, payload = self.post_batch(entries)
        self.assertEqual(status, 201)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["v1", "w2"]
        )


class CausalDominantPlanTests(HttpServerTestCase):
    def test_plan_previews_the_dominator_value_without_committing(self) -> None:
        self.seed_dominance()
        entries = [dict(self.good_auto_request(), key="k")]
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(payload["resolutions"][0]["value"], "new")
        self.assertEqual(payload["resolutions"][0]["policy"], "causal_dominant")
        # Nothing changed and no identity was bound: the same identity can
        # still commit for real.
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "new")

    def test_plan_ambiguity_is_409_with_no_results(self) -> None:
        self.seed_conflict()
        entries = [dict(self.good_auto_request(), key="k")]
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_plan_legal_non_dominating_clock_is_409_resolution_conflict(self) -> None:
        self.seed_dominance()
        entries = [dict(auto_request("r3", "fix-1", {"r1": 1, "r3": 1}), key="k")]
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_plan_illegal_clock_is_400(self) -> None:
        self.seed_conflict()
        entries = [dict(auto_request("r3", "fix-1", {"r1": 1, "r2": 1}), key="k")]
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class CausalDominantPersistenceTests(unittest.TestCase):
    """The policy binding rides the data file exactly like the operation."""

    def test_restart_recovers_result_and_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            server = SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler, data_file=data_file
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            def post(path: str, body: dict) -> tuple[int, object]:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST",
                    path,
                    body=json.dumps(body),
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                return response.status, payload

            def get(path: str) -> tuple[int, object]:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", path)
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                return response.status, payload

            try:
                with server.store._lock:
                    server.store._candidates["k"] = [
                        candidate("r1", "o1", "old", {"r1": 1}),
                        candidate("r2", "o2", "new", {"r1": 1, "r2": 1}),
                    ]
                body = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
                status, payload = post("/v1/states/k/resolve/auto", body)
                self.assertEqual(status, 201)
                self.assertEqual(payload["value"], "new")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            server = SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler, data_file=data_file
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                status, state = get("/v1/states/k")
                self.assertEqual(status, 200)
                self.assertEqual(state["status"], "resolved")
                self.assertEqual(state["value"], "new")
                # The binding survived: an identical replay is answered from
                # the committed operation, and a different policy under the
                # same identity is an operation conflict.
                status, payload = post("/v1/states/k/resolve/auto", body)
                self.assertEqual(status, 200)
                self.assertEqual(payload["value"], "new")
                status, payload = post(
                    "/v1/states/k/resolve/auto",
                    auto_request(
                        "r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_identity"
                    ),
                )
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
