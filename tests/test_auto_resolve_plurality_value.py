"""Tests for the ``plurality_value`` automatic-resolution policy.

The three automatic-resolution entry points::

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

share one policy vocabulary. ``plurality_value`` groups the current
candidates by value, selects the value carried by the most candidates,
and breaks ties by the smallest value in Unicode code-point order. The
selection depends only on the multiset of candidate values — never on
candidate order, request order, or which identities carry a value. The
other four policies are untouched. Everything here goes through the real
HTTP entry point (``SemanticStateServer`` + a request thread); only the
Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    AUTO_RESOLVE_POLICIES,
    RequestHandler,
    SemanticStateServer,
    load_data_file,
    load_data_file_full,
    parse_auto_resolve_batch,
    parse_auto_resolve_payload,
)

BATCH_PATH = "/v1/resolve/auto/batch"
PLAN_PATH = "/v1/resolve/auto/plan"
POLICY = "plurality_value"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def auto_request(replica: str, operation_id: str, clock: dict, policy: str) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


def batch_entry(
    key: str,
    replica: str,
    operation_id: str,
    clock: dict,
    policy: str,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


class PluralityPolicyParseTests(unittest.TestCase):
    def test_constant_lists_plurality_value_before_causal_history(self) -> None:
        self.assertEqual(
            AUTO_RESOLVE_POLICIES,
            (
                "lowest_identity",
                "highest_identity",
                "lowest_value",
                "highest_value",
                "plurality_value",
                "largest_causal_history",
            ),
        )

    def test_plurality_value_parses_single_and_batch(self) -> None:
        single = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "f1", {"r3": 1}, POLICY))
        )
        self.assertEqual(single["policy"], POLICY)
        entries = parse_auto_resolve_batch(
            json.dumps(
                {"resolutions": [batch_entry("k", "r3", "f1", {"r3": 1}, POLICY)]}
            )
        )
        self.assertEqual(entries[0]["policy"], POLICY)

    def test_near_miss_policy_names_are_rejected(self) -> None:
        for policy in ("plurality", "plurality-value", "PLURALITY_VALUE",
                       "plurality_values", "majority_value"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_payload(
                        auto_request("r3", "f", {"r3": 1}, policy)
                    )
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(
                        {"resolutions": [batch_entry("k", "r3", "f", {"r3": 1}, policy)]}
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        elif isinstance(body, (bytes, str)):
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
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

    def post_batch(self, body: object) -> tuple[int, object]:
        return self.request("POST", BATCH_PATH, body)

    def post_plan(self, body: object) -> tuple[int, object]:
        return self.request("POST", PLAN_PATH, body)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def seed_values(self, key: str, values: list[str], prefix: str = "r") -> dict:
        """Concurrent writes carrying ``values`` on ``key``.

        Replicas and operation ids are namespaced by ``prefix`` so several
        keys can be seeded in one test without identity collisions.
        Returns the clock that dominates every seeded candidate (plus one
        extra component for the resolving replica ``r3``).
        """
        clock: dict[str, int] = {}
        for index, value in enumerate(values, start=1):
            replica = f"{prefix}{index}"
            self.assertEqual(
                self.post_operation(
                    replica,
                    operation(f"{prefix}o{index}", key, value, {replica: 1}),
                )[0],
                201,
            )
            clock[replica] = 1
        clock["r3"] = 1
        return clock

    def fix(
        self,
        policy: str = POLICY,
        operation_id: str = "fix-1",
        replica: str = "r3",
        clock: dict | None = None,
    ) -> dict:
        return auto_request(
            replica, operation_id, clock or {"r1": 1, "r2": 1, "r3": 1}, policy
        )


class SingleKeyPluralitySelectionTests(HttpServerTestCase):
    def test_majority_value_wins_over_smallest_string(self) -> None:
        # "aaa" is the smallest string but carries only one vote; the
        # plurality policy must not degenerate into lowest_value.
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "mmm")
        self.assertEqual(payload["policy"], POLICY)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "mmm")

    def test_majority_value_wins_over_largest_string(self) -> None:
        # Symmetric check against highest_value: the largest string loses
        # when another value carries more candidates.
        clock = self.seed_values("k", ["bbb", "zzz", "bbb"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "bbb")

    def test_tie_broken_by_smallest_unicode_code_point(self) -> None:
        # Two values with two votes each: the smaller string wins.
        clock = self.seed_values("k", ["zzz", "aaa", "zzz", "aaa"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "aaa")

    def test_all_distinct_values_resolve_to_smallest(self) -> None:
        # Every value has exactly one vote, so the tie rule decides.
        clock = self.seed_values("k", ["中", "z", "é"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "z")

    def test_tie_break_uses_code_point_order(self) -> None:
        # U+00E9 ("é") sorts below U+4E2D ("中"); both carry two votes.
        clock = self.seed_values("k", ["中", "é", "中", "é"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "é")

    def test_selection_is_independent_of_candidate_order(self) -> None:
        # The same value multiset committed in two different orders (under
        # different identities) selects the same value.
        for order in (["mmm", "aaa", "mmm"], ["aaa", "mmm", "mmm"]):
            with self.subTest(order=order):
                self.server.store = type(self.server.store)()
                clock = self.seed_values("k", order)
                status, payload = self.post_auto("k", self.fix(clock=clock))
                self.assertEqual(status, 201)
                self.assertEqual(payload["value"], "mmm")

    def test_selection_is_independent_of_identity_order(self) -> None:
        # The lexicographically smallest and largest identities both carry
        # the minority value; identity order decides nothing here.
        clock = self.seed_values("k", ["zzz", "bbb", "bbb"])
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "bbb")

    def test_chosen_value_becomes_the_only_version(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, state = self.get_state("k")
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "mmm",
                "clock": clock,
                "status": "resolved",
            },
        )


class SingleKeyPluralityConflictTests(HttpServerTestCase):
    def test_clock_not_dominating_is_400(self) -> None:
        self.seed_values("k", ["v1", "v2"])
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-x", {"r1": 1, "r3": 1}, POLICY)
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_key_is_409(self) -> None:
        status, payload = self.post_auto(
            "absent", auto_request("r3", "fix-1", {"r3": 1}, POLICY)
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409(self) -> None:
        # A unanimous value set is not a conflict even though one value
        # would win every vote.
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_auto("k", self.fix())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_rejects_a_new_plurality_identity(self) -> None:
        clock = self.seed_values("k", ["v1", "v2"])
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        status, payload = self.post_auto(
            "k", self.fix(operation_id="fix-2", clock=dict(clock, **{"r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class SingleKeyPluralityIdentityTests(HttpServerTestCase):
    def test_replay_is_200_and_reports_original_value(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        body = self.fix(clock=clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "key": "k",
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "mmm",
                "policy": POLICY,
            },
        )

    def test_different_policy_under_same_identity_is_409(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        for policy in ("lowest_identity", "highest_identity",
                       "lowest_value", "highest_value"):
            with self.subTest(policy=policy):
                status, payload = self.post_auto("k", self.fix(policy, clock=clock))
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_plurality_replay_after_key_moved_on_reports_original_value(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        body = self.fix(clock=clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on with a fresh concurrent conflict.
        self.post_operation(
            "r2", operation("o9", "k", "v9", {"r1": 1, "r2": 2, "r3": 0})
        )
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "mmm")

    def test_plain_write_identity_never_matches_plurality_request(self) -> None:
        clock = self.seed_values("k", ["v1", "v2"])
        status, _ = self.post_operation(
            "r3", operation("fix-1", "k", "v1", clock)
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class SingleKeyPluralityIntegrationTests(HttpServerTestCase):
    def test_plurality_resolution_is_exported_with_chosen_value(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][3],
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "mmm",
                    "clock": clock,
                },
            },
        )

    def test_imported_plurality_resolution_carries_no_binding(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, page = self.get_sync()

        self.server.store = type(self.server.store)()
        status, _ = self.request(
            "POST", "/v1/sync/operations", {"operations": page["operations"]}
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class BatchPluralityPolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plurality_entries_select_by_vote_in_request_order(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        clock2 = self.seed_values("k2", ["zzz", "aaa", "zzz", "aaa"], prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, POLICY),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "mmm", POLICY), ("k2", "aaa", POLICY)],
        )

    def test_batch_mixes_plurality_with_other_policies(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        clock2 = self.seed_values("k2", ["aaa", "zzz"], prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, "highest_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "mmm", POLICY), ("k2", "zzz", "highest_value")],
        )

    def test_one_conflict_rejects_whole_batch_unchanged(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        # k2 has no candidates at all.
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", {"r3": 2}, POLICY),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["ro1", "ro2", "ro3"],
        )
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_legal_clock_not_dominating_rejects_whole_batch_409(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        self.seed_values("k2", ["v1", "v2"], prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            # Structurally legal but dominates only one of k2's candidates.
            batch_entry("k2", "r3", "f2", {"s1": 1, "r3": 1}, POLICY),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 5)

    def test_batch_mixes_accepted_and_replayed_plurality_entries(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        first = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(first)[0], 201)
        clock2 = self.seed_values("k2", ["zzz", "aaa", "zzz", "aaa"], prefix="s")
        again = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, POLICY),
        )
        status, payload = self.post_batch(again)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(
            [(r["key"], r["value"]) for r in payload["resolutions"]],
            [("k1", "mmm"), ("k2", "aaa")],
        )

    def test_plurality_replay_with_other_policy_is_operation_conflict(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        doc = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(doc)[0], 201)
        tampered = self.document(batch_entry("k1", "r3", "f1", clock1, "lowest_value"))
        status, payload = self.post_batch(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanPluralityPolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plan_previews_plurality_selection_without_writing(self) -> None:
        clock1 = self.seed_values("k1", ["mmm", "aaa", "mmm"])
        clock2 = self.seed_values("k2", ["zzz", "aaa", "zzz", "aaa"], prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, POLICY),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "mmm", POLICY), ("k2", "aaa", POLICY)],
        )
        # Nothing was written: keys still conflict and no log record exists.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 7)

    def test_plan_then_commit_agrees_and_commit_is_still_fresh(self) -> None:
        clock = self.seed_values("k", ["mmm", "aaa", "mmm"])
        doc = self.document(batch_entry("k", "r3", "f1", clock, POLICY))
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(planned["resolutions"][0]["value"], "mmm")
        status, committed = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])
        # After committing, the same request previews as a replay.
        status, afterwards = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(afterwards["accepted"], 0)
        self.assertEqual(afterwards["replayed"], 1)
        self.assertEqual(afterwards["resolutions"][0]["value"], "mmm")

    def test_plan_conflict_for_missing_key_is_409(self) -> None:
        doc = self.document(batch_entry("absent", "r3", "f1", {"r3": 1}, POLICY))
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class PersistentPluralityPolicyTestCase(unittest.TestCase):
    """``plurality_value`` against a data-file-backed server with real HTTP."""

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
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed(self, server: SemanticStateServer) -> dict:
        clock: dict[str, int] = {}
        for replica, op_id, value in (
            ("r1", "o1", "mmm"),
            ("r2", "o2", "aaa"),
            ("r9", "o3", "mmm"),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", value, {replica: 1}),
            )
            self.assertEqual(status, 201)
            clock[replica] = 1
        clock["r3"] = 1
        return clock

    def fix(self, clock: dict, policy: str = POLICY) -> dict:
        return auto_request("r3", "fix-1", clock, policy)

    def test_plurality_binding_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        clock = self.seed(server)
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock)
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "mmm")
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r3", "fix-1"): POLICY})

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Same binding replays 200 with the original value.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock)
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "mmm")
        self.assertEqual(payload["policy"], POLICY)
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock, "lowest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "mmm")

    def test_plurality_policy_accepted_in_stored_policies_section(self) -> None:
        # Hand-write a data file whose binding uses the plurality policy
        # name; recovery validates it against the same shared constant.
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "a", {"r1": 1})},
                    ],
                    "policies": [
                        {"replicaId": "r1", "operationId": "o1", "policy": POLICY},
                    ],
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r1", "o1"): POLICY})
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
