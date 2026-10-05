"""Tests for the ``largest_causal_history`` automatic-resolution policy.

The three automatic-resolution entry points::

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

share one policy vocabulary. ``largest_causal_history`` selects the
current candidate whose clock strictly dominates the most distinct
operations in the shared accepted log — operations on every key count,
the candidate's own identity is excluded, equal clocks are never
ancestors, and a re-imported duplicate counts once — breaking a tie by
the smallest ``(replicaId, operationId)`` in Unicode code-point order.
The selection depends only on the candidate snapshot and the accepted
log, never on traversal order, batch position, or pagination, so the
same state always yields the same candidate, before and after a
``--data-file`` restart. The other five policies are untouched.
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
    AUTO_RESOLVE_POLICIES,
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    load_data_file,
    load_data_file_full,
    parse_auto_resolve_batch,
    parse_auto_resolve_payload,
)

BATCH_PATH = "/v1/resolve/auto/batch"
PLAN_PATH = "/v1/resolve/auto/plan"
POLICY = "largest_causal_history"


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


class CausalHistoryPolicyParseTests(unittest.TestCase):
    def test_constant_lists_largest_causal_history_last(self) -> None:
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

    def test_largest_causal_history_parses_single_and_batch(self) -> None:
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
        for policy in ("largest_causal", "largest-causal-history",
                       "LARGEST_CAUSAL_HISTORY", "causal_history",
                       "largest_causal_historys", "largest_causal_history ",
                       " largest_causal_history"):
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

    def seed_rich_vs_poor(self, key: str, prefix: str = "r") -> dict:
        """Seed ``key`` with a history-rich and a history-poor candidate.

        The ``{prefix}9`` replica first commits three operations on
        *other* keys, then carries that causal past onto ``key`` with
        clock ``{rich: 4}``; the ``{prefix}1`` replica arrives with no
        past at all (clock ``{poor: 1}``). The rich candidate therefore
        has causal history 3 (all of it cross-key) and the poor one 0.
        The rich identity and the rich value are both lexicographically
        larger, so the identity policies, ``lowest_value``, and
        ``plurality_value`` would all choose the poor candidate. Returns
        the clock that dominates both candidates (plus one component for
        the resolving replica ``r3``).
        """
        rich = f"{prefix}9"
        poor = f"{prefix}1"
        for index, side in enumerate(("hx", "hy", "hz"), start=1):
            self.assertEqual(
                self.post_operation(
                    rich,
                    operation(f"{prefix}a{index}", f"{key}-{side}", f"x{index}", {rich: index}),
                )[0],
                201,
            )
        self.assertEqual(
            self.post_operation(
                rich, operation(f"{prefix}c1", key, "rich", {rich: 4})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                poor, operation(f"{prefix}c2", key, "poor", {poor: 1})
            )[0],
            201,
        )
        return {rich: 4, poor: 1, "r3": 1}

    def fix(
        self,
        policy: str = POLICY,
        operation_id: str = "fix-1",
        replica: str = "r3",
        clock: dict | None = None,
    ) -> dict:
        return auto_request(replica, operation_id, clock or {"r3": 1}, policy)


class SingleKeyCausalHistorySelectionTests(HttpServerTestCase):
    def test_richest_history_wins_over_smallest_identity_and_value(self) -> None:
        # "poor" is the smaller string, carries the smaller identity, and
        # ties the plurality vote 1-1 — every other deterministic policy
        # would select it — but its causal history is empty.
        clock = self.seed_rich_vs_poor("k")
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")
        self.assertEqual(payload["policy"], POLICY)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "rich")

    def test_cross_key_operations_count_towards_history(self) -> None:
        # The rich candidate's entire causal past lives on other keys;
        # it still counts. Without cross-key operations both candidates
        # would tie at 0 and the smaller (poor) identity would win.
        clock = self.seed_rich_vs_poor("k")
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")

    def test_equal_clocks_are_not_ancestors(self) -> None:
        # opZ's clock is exactly equal to candidate B's clock. If equal
        # clocks counted as ancestors, B's history would be 2 against
        # A's 1 and B would win; strictly dominated ancestors only, so
        # both histories are 1 and the smaller identity (r1, oA) wins.
        self.assertEqual(
            self.post_operation("r1", operation("oX", "j", "x", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation(
                "r1", operation("oZ", "j2", "z", {"r1": 1, "r2": 2})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                "r1", operation("oA", "k", "winner-a", {"r1": 2, "r2": 1})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                "r2", operation("oB", "k", "winner-b", {"r1": 1, "r2": 2})
            )[0],
            201,
        )
        clock = {"r1": 2, "r2": 2, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "winner-a")

    def test_duplicate_import_does_not_inflate_history(self) -> None:
        # Candidate A has history 1 (opX), candidate B history 2. A
        # replayed import of opX alone must not count it twice: if it
        # did, A would tie B at 2 and win the identity tie-break.
        self.assertEqual(
            self.post_operation("r1", operation("opX", "j", "x", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r1", operation("oA", "k", "a-val", {"r1": 2}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("opY1", "j2", "y1", {"r2": 1}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("opY2", "j3", "y2", {"r2": 2}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("oB", "k", "b-val", {"r2": 3}))[0],
            201,
        )
        # Re-import opX by itself: a replay that appends no log record.
        status, _ = self.request(
            "POST",
            "/v1/sync/operations",
            {
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("opX", "j", "x", {"r1": 1}),
                    }
                ]
            },
        )
        self.assertEqual(status, 200)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 5)
        clock = {"r1": 2, "r2": 3, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "b-val")

    def test_tie_breaks_by_smallest_identity(self) -> None:
        # Both candidates have empty causal histories; the smaller
        # (replicaId, operationId) supplies the value even though its
        # string is the larger one.
        self.assertEqual(
            self.post_operation("r9", operation("o1", "k", "aaa", {"r9": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r1", operation("o2", "k", "mmm", {"r1": 1}))[0], 201
        )
        clock = {"r1": 1, "r9": 1, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "mmm")

    def test_tie_break_compares_operation_id_after_replica(self) -> None:
        # Two candidates from the same replica with concurrent (equal)
        # clocks: both histories are 0, so the operation id decides.
        self.assertEqual(
            self.post_operation("r1", operation("o9", "k", "v9", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))[0], 201
        )
        clock = {"r1": 1, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")

    def test_selection_is_independent_of_commit_order(self) -> None:
        # The same final state (same accepted operation set, same
        # candidates) reached through two different commit orders selects
        # the same candidate.
        orders = (
            [("r9", "ra1", "k-hx", "x1", {"r9": 1}),
             ("r9", "ra2", "k-hy", "x2", {"r9": 2}),
             ("r9", "ra3", "k-hz", "x3", {"r9": 3}),
             ("r9", "rc1", "k", "rich", {"r9": 4}),
             ("r1", "rc2", "k", "poor", {"r1": 1})],
            [("r1", "rc2", "k", "poor", {"r1": 1}),
             ("r9", "rc1", "k", "rich", {"r9": 4}),
             ("r9", "ra1", "k-hx", "x1", {"r9": 1}),
             ("r9", "ra2", "k-hy", "x2", {"r9": 2}),
             ("r9", "ra3", "k-hz", "x3", {"r9": 3})],
        )
        for order in orders:
            with self.subTest(order=order):
                self.server.store = type(self.server.store)()
                for replica, op_id, key, value, clock in order:
                    self.assertEqual(
                        self.post_operation(
                            replica, operation(op_id, key, value, clock)
                        )[0],
                        201,
                    )
                fix_clock = {"r1": 1, "r9": 4, "r3": 1}
                status, payload = self.post_auto("k", self.fix(clock=fix_clock))
                self.assertEqual(status, 201)
                self.assertEqual(payload["value"], "rich")

    def test_chosen_value_becomes_the_only_version(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, state = self.get_state("k")
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "rich",
                "clock": clock,
                "status": "resolved",
            },
        )


class SingleKeyCausalHistoryConflictTests(HttpServerTestCase):
    def test_clock_not_dominating_is_400(self) -> None:
        self.seed_rich_vs_poor("k")
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-x", {"r9": 4, "r3": 1}, POLICY)
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
        # A unanimous value set is not a conflict even when the
        # candidates carry different causal histories.
        self.post_operation("r1", operation("o1", "j", "x", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k", "same", {"r1": 2}))
        self.post_operation("r2", operation("o3", "k", "same", {"r2": 1}))
        status, payload = self.post_auto(
            "k", self.fix(clock={"r1": 2, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_rejects_a_new_causal_history_identity(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        status, payload = self.post_auto(
            "k", self.fix(operation_id="fix-2", clock=dict(clock, **{"r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class SingleKeyCausalHistoryIdentityTests(HttpServerTestCase):
    def test_replay_is_200_and_reports_original_value(self) -> None:
        clock = self.seed_rich_vs_poor("k")
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
                "value": "rich",
                "policy": POLICY,
            },
        )

    def test_different_policy_under_same_identity_is_409(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        for policy in ("lowest_identity", "highest_identity", "lowest_value",
                       "highest_value", "plurality_value"):
            with self.subTest(policy=policy):
                status, payload = self.post_auto("k", self.fix(policy, clock=clock))
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_replay_after_key_moved_on_reports_original_value(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        body = self.fix(clock=clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on with a fresh concurrent conflict.
        self.post_operation(
            "r2", operation("o9", "k", "v9", {"r1": 1, "r2": 2, "r9": 4, "r3": 0})
        )
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "rich")

    def test_plain_write_identity_never_matches_causal_history_request(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        status, _ = self.post_operation(
            "r3", operation("fix-1", "k", "rich", clock)
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class SingleKeyCausalHistoryIntegrationTests(HttpServerTestCase):
    def test_causal_history_resolution_is_exported_with_chosen_value(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][5],
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "rich",
                    "clock": clock,
                },
            },
        )

    def test_imported_causal_history_resolution_carries_no_binding(self) -> None:
        clock = self.seed_rich_vs_poor("k")
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


class BatchCausalHistoryPolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_causal_history_entries_select_by_history_in_request_order(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        clock2 = self.seed_rich_vs_poor("k2", prefix="s")
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
            [("k1", "rich", POLICY), ("k2", "rich", POLICY)],
        )

    def test_same_state_at_any_batch_position_selects_the_same_candidate(self) -> None:
        # k1 and k2 hold structurally identical conflicts (distinct
        # identities, same shape); both entry orders must select the
        # same value for each key, and the preview must agree.
        clock1 = self.seed_rich_vs_poor("k1")
        clock2 = self.seed_rich_vs_poor("k2", prefix="s")
        forward = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, POLICY),
        )
        reverse = self.document(
            batch_entry("k2", "r3", "f2", clock2, POLICY),
            batch_entry("k1", "r3", "f1", clock1, POLICY),
        )
        status, planned_forward = self.post_plan(forward)
        self.assertEqual(status, 200)
        status, planned_reverse = self.post_plan(reverse)
        self.assertEqual(status, 200)
        by_key_forward = {r["key"]: r["value"] for r in planned_forward["resolutions"]}
        by_key_reverse = {r["key"]: r["value"] for r in planned_reverse["resolutions"]}
        self.assertEqual(by_key_forward, {"k1": "rich", "k2": "rich"})
        self.assertEqual(by_key_forward, by_key_reverse)
        # The commit agrees with the preview.
        status, committed = self.post_batch(forward)
        self.assertEqual(status, 201)
        self.assertEqual(
            {r["key"]: r["value"] for r in committed["resolutions"]}, by_key_forward
        )

    def test_batch_mixes_causal_history_with_other_policies(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        self.post_operation("s1", operation("so1", "k2", "aaa", {"s1": 1}))
        self.post_operation("s2", operation("so2", "k2", "zzz", {"s2": 1}))
        clock2 = {"s1": 1, "s2": 1, "r3": 1}
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, "highest_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "rich", POLICY), ("k2", "zzz", "highest_value")],
        )

    def test_one_conflict_rejects_whole_batch_unchanged(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
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
            ["ra1", "ra2", "ra3", "rc1", "rc2"],
        )
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_legal_clock_not_dominating_rejects_whole_batch_409(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        self.seed_rich_vs_poor("k2", prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            # Structurally legal but dominates only one of k2's candidates.
            batch_entry("k2", "r3", "f2", {"s9": 4, "r3": 1}, POLICY),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 10)

    def test_batch_mixes_accepted_and_replayed_causal_history_entries(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        first = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(first)[0], 201)
        clock2 = self.seed_rich_vs_poor("k2", prefix="s")
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
            [("k1", "rich"), ("k2", "rich")],
        )

    def test_causal_history_replay_with_other_policy_is_operation_conflict(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        doc = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(doc)[0], 201)
        tampered = self.document(batch_entry("k1", "r3", "f1", clock1, "lowest_value"))
        status, payload = self.post_batch(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanCausalHistoryPolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plan_previews_causal_history_selection_without_writing(self) -> None:
        clock1 = self.seed_rich_vs_poor("k1")
        clock2 = self.seed_rich_vs_poor("k2", prefix="s")
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
            [("k1", "rich", POLICY), ("k2", "rich", POLICY)],
        )
        # Nothing was written: keys still conflict and no log record exists.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 10)

    def test_plan_then_commit_agrees_and_commit_is_still_fresh(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        doc = self.document(batch_entry("k", "r3", "f1", clock, POLICY))
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(planned["resolutions"][0]["value"], "rich")
        status, committed = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])
        # After committing, the same request previews as a replay.
        status, afterwards = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(afterwards["accepted"], 0)
        self.assertEqual(afterwards["replayed"], 1)
        self.assertEqual(afterwards["resolutions"][0]["value"], "rich")

    def test_plan_is_deterministic_across_repeats(self) -> None:
        clock = self.seed_rich_vs_poor("k")
        doc = self.document(batch_entry("k", "r3", "f1", clock, POLICY))
        status, first = self.post_plan(doc)
        self.assertEqual(status, 200)
        for _ in range(3):
            status, again = self.post_plan(json.loads(json.dumps(doc)))
            self.assertEqual(status, 200)
            self.assertEqual(again, first)

    def test_plan_conflict_for_missing_key_is_409(self) -> None:
        doc = self.document(batch_entry("absent", "r3", "f1", {"r3": 1}, POLICY))
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class PersistentCausalHistoryPolicyTestCase(unittest.TestCase):
    """``largest_causal_history`` against a data-file-backed server."""

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
        for replica, op_id, key, value, tick in (
            ("r9", "ra1", "k-hx", "x1", 1),
            ("r9", "ra2", "k-hy", "x2", 2),
            ("r9", "ra3", "k-hz", "x3", 3),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, key, value, {replica: tick}),
            )
            self.assertEqual(status, 201)
        for replica, op_id, value, clock in (
            ("r9", "rc1", "rich", {"r9": 4}),
            ("r1", "rc2", "poor", {"r1": 1}),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", value, clock),
            )
            self.assertEqual(status, 201)
        return {"r1": 1, "r9": 4, "r3": 1}

    def fix(self, clock: dict, policy: str = POLICY) -> dict:
        return auto_request("r3", "fix-1", clock, policy)

    def test_causal_history_binding_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        clock = self.seed(server)
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock)
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")
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
        self.assertEqual(payload["value"], "rich")
        self.assertEqual(payload["policy"], POLICY)
        self.assertEqual(len(load_data_file(str(self.data_file))), 6)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock, "lowest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "rich")

    def test_selection_is_identical_before_and_after_restart(self) -> None:
        server = self.start_server()
        clock = self.seed(server)
        doc = {"resolutions": [batch_entry("k", "r3", "fix-1", clock, POLICY)]}
        status, before = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(before["resolutions"][0]["value"], "rich")

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "POST", PLAN_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        status, committed = self.request(server, "POST", BATCH_PATH, doc)
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], before["resolutions"])

    def test_causal_history_policy_accepted_in_stored_policies_section(self) -> None:
        # Hand-write a data file whose binding uses the causal-history
        # policy name; recovery validates it against the shared constant.
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

    def test_dangling_and_duplicate_bindings_still_refuse_startup(self) -> None:
        # The existing recovery rules apply unchanged to the new policy:
        # a binding naming no accepted operation, or a repeated binding
        # for one identity, rejects the data file.
        dangling = {
            "version": 1,
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "k", "a", {"r1": 1})},
            ],
            "policies": [
                {"replicaId": "r1", "operationId": "ghost", "policy": POLICY},
            ],
        }
        self.data_file.write_text(json.dumps(dangling), encoding="utf-8")
        with self.assertRaises(PersistenceError):
            load_data_file_full(str(self.data_file))

        duplicate = {
            "version": 1,
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "k", "a", {"r1": 1})},
            ],
            "policies": [
                {"replicaId": "r1", "operationId": "o1", "policy": POLICY},
                {"replicaId": "r1", "operationId": "o1", "policy": POLICY},
            ],
        }
        self.data_file.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaises(PersistenceError):
            load_data_file_full(str(self.data_file))


if __name__ == "__main__":
    unittest.main()
