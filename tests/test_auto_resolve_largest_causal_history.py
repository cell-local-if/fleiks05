"""Tests for the ``largest_causal_history`` automatic-resolution policy.

The three automatic-resolution entry points::

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

share one policy vocabulary. ``largest_causal_history`` selects the
current candidate with the richest causal history: the number of
distinct operations in the shared accepted operation log — operations on
**every** key count — whose clock the candidate's clock strictly
dominates, excluding the candidate's own identity. An equal clock is not
an ancestor, a duplicate import of the same ``(replicaId, operationId)``
is one operation, and ties break by the smallest ``(replicaId,
operationId)`` in Unicode code-point order. The selection is a pure
function of the log contents, so the same state selects the same
candidate regardless of log order, batch position, or pagination. The
other five policies are untouched. Everything here goes through the real
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
    _select_auto_resolution_candidate,
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


def candidate(replica: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
    }


class CausalHistoryParseTests(unittest.TestCase):
    def test_constant_lists_causal_history_last(self) -> None:
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

    def test_policy_parses_single_and_batch(self) -> None:
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
        for policy in (
            "causal_history",
            "largest_causal",
            "largest-causal-history",
            "LARGEST_CAUSAL_HISTORY",
            "largest_causal_history ",
            "largest_causal_histories",
            "longest_causal_history",
        ):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_payload(
                        auto_request("r3", "f", {"r3": 1}, policy)
                    )
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(
                        {"resolutions": [batch_entry("k", "r3", "f", {"r3": 1}, policy)]}
                    )

    def test_non_string_policy_is_rejected(self) -> None:
        for policy in (None, 1, True, ["largest_causal_history"], {"p": 1}):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_payload(
                        auto_request("r3", "f", {"r3": 1}, policy)
                    )


class CausalHistorySelectionUnitTests(unittest.TestCase):
    """Direct checks of the counting rule against hand-built logs."""

    def select(self, candidates: list[dict], accepted: list[tuple[str, dict]]) -> dict:
        return _select_auto_resolution_candidate(candidates, POLICY, accepted)

    def test_duplicate_log_records_count_once(self) -> None:
        # The same (replicaId, operationId) appearing twice in the log —
        # a duplicate import — is one operation, not two: the count is 1,
        # so the tie is broken by the smallest identity and (r2, c2) wins.
        dup = ("r1", operation("h1", "other", "x", {"r1": 1}))
        accepted = [dup, dup]
        candidates = [
            candidate("r9", "c9", "vz", {"r1": 1, "r9": 1}),
            candidate("r2", "c2", "va", {"r2": 1}),
        ]
        # r9/c9 dominates the single distinct history op (count 1) and
        # wins outright; with the duplicate counted twice the count would
        # still be 2 > 0 — so also check the symmetric tie case below.
        self.assertEqual(self.select(candidates, accepted)["value"], "vz")

    def test_duplicate_does_not_break_a_tie(self) -> None:
        # Both candidates dominate exactly the same single distinct
        # operation; the duplicated record must not inflate either count,
        # so the tie still falls to the smallest identity.
        dup = ("r1", operation("h1", "other", "x", {"r1": 1}))
        accepted = [dup, dup, dup]
        candidates = [
            candidate("r9", "c9", "vz", {"r1": 1, "r9": 1}),
            candidate("r8", "c8", "va", {"r1": 1, "r8": 1}),
        ]
        self.assertEqual(self.select(candidates, accepted)["value"], "va")

    def test_equal_clock_is_not_an_ancestor(self) -> None:
        accepted = [("r1", operation("h1", "other", "x", {"r1": 1, "r9": 1}))]
        candidates = [
            # Same clock as the history op: equal, not strictly dominated.
            candidate("r9", "c9", "vz", {"r1": 1, "r9": 1}),
            candidate("r2", "c2", "va", {"r2": 1}),
        ]
        # Counts are 0-0; the smallest identity (r2, c2) wins. Counting
        # the equal clock would hand the win to (r9, c9).
        self.assertEqual(self.select(candidates, accepted)["value"], "va")

    def test_own_identity_is_excluded(self) -> None:
        # The candidate's own operation is in the log (it must be: it is
        # an accepted write); it never counts towards its own history.
        own = ("r9", operation("c9", "k", "vz", {"r1": 1, "r9": 1}))
        accepted = [own, ("r1", operation("h1", "other", "x", {"r1": 1}))]
        candidates = [
            candidate("r9", "c9", "vz", {"r1": 1, "r9": 1}),
            candidate("r2", "c2", "va", {"r2": 1}),
        ]
        # r9/c9 counts only h1 (1), not itself; it still wins 1-0.
        self.assertEqual(self.select(candidates, accepted)["value"], "vz")

    def test_cross_key_operations_count(self) -> None:
        accepted = [
            ("r4", operation("h1", "other", "x", {"r4": 1})),
            ("r5", operation("h2", "elsewhere", "y", {"r5": 1})),
        ]
        candidates = [
            candidate("r9", "c9", "vz", {"r4": 1, "r5": 1, "r9": 1}),
            candidate("r2", "c2", "va", {"r4": 1, "r2": 1}),
        ]
        # r9/c9 dominates both foreign-key operations (2); r2/c2 only one.
        self.assertEqual(self.select(candidates, accepted)["value"], "vz")

    def test_tie_break_uses_unicode_identity_order(self) -> None:
        # "r10" sorts before "r2" in Unicode code-point order.
        candidates = [
            candidate("r2", "o1", "vb", {"r2": 1}),
            candidate("r10", "o1", "va", {"r10": 1}),
        ]
        self.assertEqual(self.select(candidates, [])["replicaId"], "r10")

    def test_selection_is_independent_of_log_and_candidate_order(self) -> None:
        accepted = [
            ("r4", operation("h1", "other", "x", {"r4": 1})),
            ("r5", operation("h2", "elsewhere", "y", {"r5": 1})),
            ("r6", operation("h3", "more", "z", {"r6": 1})),
        ]
        candidates = [
            candidate("r9", "c9", "va", {"r4": 1, "r5": 1, "r9": 1}),
            candidate("r8", "c8", "vb", {"r6": 1, "r8": 1}),
        ]
        for accepted_order in (
            accepted,
            list(reversed(accepted)),
            [accepted[1], accepted[2], accepted[0]],
        ):
            for candidate_order in (candidates, list(reversed(candidates))):
                with self.subTest():
                    self.assertEqual(
                        self.select(candidate_order, accepted_order)["value"], "va"
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

    def write(self, replica: str, operation_id: str, key: str, value: str, clock: dict) -> None:
        status, _ = self.post_operation(
            replica, operation(operation_id, key, value, clock)
        )
        self.assertEqual(status, 201)

    def seed_rich_conflict(self, key: str = "k", prefix: str = "") -> dict:
        """Two concurrent candidates on ``key`` with unequal histories.

        ``{prefix}r1`` first commits a history operation on another key,
        then ``{prefix}r9`` writes ``"rich"`` with a clock that dominates
        it (history size 1) while ``{prefix}r2`` writes ``"poor"`` with
        an empty history (size 0). The identities are chosen so the
        identity policies would pick the *other* candidate: ``r2`` sorts
        below ``r9``. Returns the clock dominating both candidates.
        """
        r1, r2, r9 = f"{prefix}r1", f"{prefix}r2", f"{prefix}r9"
        self.write(r1, f"{prefix}h1", f"{prefix}other", "x", {r1: 1})
        self.write(r9, f"{prefix}c9", key, "rich", {r1: 1, r9: 1})
        self.write(r2, f"{prefix}c2", key, "poor", {r2: 1})
        return {r1: 1, r2: 1, r9: 1, "r3": 1}

    def fix(
        self,
        policy: str = POLICY,
        operation_id: str = "fix-1",
        replica: str = "r3",
        clock: dict | None = None,
    ) -> dict:
        return auto_request(replica, operation_id, clock or {"r3": 1}, policy)


class SingleKeyCausalHistorySelectionTests(HttpServerTestCase):
    def test_richer_cross_key_history_wins(self) -> None:
        clock = self.seed_rich_conflict()
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")
        self.assertEqual(payload["policy"], POLICY)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "rich")

    def test_identity_policies_would_pick_the_other_candidate(self) -> None:
        # Guard the fixture: the causal-history winner (r9) is the
        # *larger* identity, so lowest_identity must disagree with it.
        clock = self.seed_rich_conflict()
        status, payload = self.post_auto(
            "k", self.fix("lowest_identity", clock=clock)
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "poor")

    def test_equal_clock_does_not_count_as_ancestor(self) -> None:
        # r9's candidate clock equals the only history operation's clock:
        # equal is not strictly dominated, so both candidates count 0 and
        # the smaller identity (r2, c2) wins.
        self.write("r1", "h1", "other", "x", {"r1": 1, "r9": 1})
        self.write("r9", "c9", "k", "vz", {"r1": 1, "r9": 1})
        self.write("r2", "c2", "k", "va", {"r2": 1})
        clock = {"r1": 1, "r2": 1, "r9": 1, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "va")

    def test_reimported_history_does_not_inflate_the_count(self) -> None:
        # Seed a history operation by sync import, then import the same
        # batch again (a pure replay): the log holds the operation once,
        # so the selection is the same as after the first import.
        records = [
            {"replicaId": "r1", "operation": operation("h1", "other", "x", {"r1": 1})},
        ]
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": records})
        self.assertEqual(status, 201)
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": records})
        self.assertEqual(status, 200)
        self.write("r9", "c9", "k", "rich", {"r1": 1, "r9": 1})
        self.write("r2", "c2", "k", "poor", {"r2": 1})
        clock = {"r1": 1, "r2": 1, "r9": 1, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")

    def test_tie_broken_by_smallest_unicode_identity(self) -> None:
        # No history operations at all: 0-0, and "r10" sorts before "r2".
        self.write("r2", "o1", "k", "vb", {"r2": 1})
        self.write("r10", "o1", "k", "va", {"r10": 1})
        clock = {"r2": 1, "r10": 1, "r3": 1}
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "va")

    def test_selection_ignores_sync_pagination(self) -> None:
        # Paged reads of the accepted log do not move the selection.
        clock = self.seed_rich_conflict()
        _, page1 = self.get_sync("?after=0&limit=1")
        self.assertEqual(len(page1["operations"]), 1)
        _, page2 = self.get_sync("?after=1&limit=100")
        self.assertEqual(len(page2["operations"]), 2)
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "rich")

    def test_chosen_value_becomes_the_only_version(self) -> None:
        clock = self.seed_rich_conflict()
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, state = self.get_state("k")
        self.assertEqual(
            state,
            {"key": "k", "value": "rich", "clock": clock, "status": "resolved"},
        )


class SingleKeyCausalHistoryConflictTests(HttpServerTestCase):
    def test_clock_not_dominating_is_400(self) -> None:
        self.seed_rich_conflict()
        status, payload = self.post_auto(
            "k", self.fix(clock={"r1": 1, "r3": 1})
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_structurally_invalid_clock_is_400(self) -> None:
        self.seed_rich_conflict()
        for clock in (
            {},
            {"r1": 1},  # missing the resolving replica
            {"r3": -1},
            {"r3": True},
            {"r3": 1.0},
        ):
            with self.subTest(clock=clock):
                status, payload = self.post_auto("k", self.fix(clock=clock))
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_unknown_fields_and_bad_json_are_400(self) -> None:
        self.seed_rich_conflict()
        body = self.fix(clock={"r1": 1, "r2": 1, "r9": 1, "r3": 1})
        body["value"] = "rich"
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.post_auto("k", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_key_is_409(self) -> None:
        status, payload = self.post_auto(
            "absent", self.fix(operation_id="fix-1", clock={"r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409(self) -> None:
        self.write("r1", "o1", "k", "same", {"r1": 1})
        self.write("r2", "o2", "k", "same", {"r2": 1})
        status, payload = self.post_auto(
            "k", self.fix(clock={"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_rejects_a_new_identity(self) -> None:
        clock = self.seed_rich_conflict()
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        status, payload = self.post_auto(
            "k", self.fix(operation_id="fix-2", clock=dict(clock, r3=2))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class SingleKeyCausalHistoryIdentityTests(HttpServerTestCase):
    def test_replay_is_200_and_reports_original_value(self) -> None:
        clock = self.seed_rich_conflict()
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
        # No new log record was appended by the replay.
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)

    def test_different_policy_under_same_identity_is_409(self) -> None:
        clock = self.seed_rich_conflict()
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        for policy in (
            "lowest_identity",
            "highest_identity",
            "lowest_value",
            "highest_value",
            "plurality_value",
        ):
            with self.subTest(policy=policy):
                status, payload = self.post_auto("k", self.fix(policy, clock=clock))
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_different_clock_under_same_identity_is_409(self) -> None:
        clock = self.seed_rich_conflict()
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        shifted = dict(clock, r3=2)
        status, payload = self.post_auto("k", self.fix(clock=shifted))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_replay_after_key_moved_on_reports_original_value(self) -> None:
        clock = self.seed_rich_conflict()
        body = self.fix(clock=clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on with a fresh concurrent conflict.
        self.write("r2", "o9", "k", "v9", {"r1": 1, "r2": 2, "r9": 1, "r3": 0})
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "rich")

    def test_plain_write_identity_never_matches_causal_request(self) -> None:
        clock = self.seed_rich_conflict()
        status, _ = self.post_operation("r3", operation("fix-1", "k", "rich", clock))
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix(clock=clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class SingleKeyCausalHistoryIntegrationTests(HttpServerTestCase):
    def test_resolution_is_exported_with_chosen_value(self) -> None:
        clock = self.seed_rich_conflict()
        self.assertEqual(self.post_auto("k", self.fix(clock=clock))[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][3],
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

    def test_imported_resolution_carries_no_binding(self) -> None:
        clock = self.seed_rich_conflict()
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

    def test_batch_entries_select_by_causal_history(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        clock2 = self.seed_rich_conflict("k2", prefix="s")
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

    def test_batch_mixes_causal_history_with_other_policies(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        clock2 = self.seed_rich_conflict("k2", prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, "lowest_identity"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "rich", POLICY), ("k2", "poor", "lowest_identity")],
        )

    def test_earlier_batch_resolution_counts_for_later_entry(self) -> None:
        # k1 is resolved first inside the same batch; its resolution
        # operation enters the staged accepted log, so k2's entry counts
        # it. Candidate A (r9) dominates exactly the two k1 writes plus
        # the staged resolution (3) against B's two foreign operations
        # (2); without the staged record the 2-2 tie would go to the
        # smaller identity (r8, c8) and select "bp".
        self.write("r1", "o1", "k1", "x", {"r1": 1})
        self.write("r2", "o2", "k1", "y", {"r2": 1})
        self.write("r6", "h6", "other", "p", {"r6": 1})
        self.write("r7", "h7", "other", "q", {"r7": 1})
        self.write("r9", "c9", "k2", "ap", {"r1": 1, "r2": 1, "r3": 1, "r9": 1})
        self.write("r8", "c8", "k2", "bp", {"r6": 1, "r7": 1, "r8": 1})
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, POLICY),
            batch_entry(
                "k2",
                "r3",
                "f2",
                {"r1": 1, "r2": 1, "r3": 1, "r6": 1, "r7": 1, "r8": 1, "r9": 1},
                POLICY,
            ),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        # k1: both candidates have empty histories; the tie goes to the
        # smallest identity (r1, o1) and its value "x".
        self.assertEqual(
            [(r["key"], r["value"]) for r in payload["resolutions"]],
            [("k1", "x"), ("k2", "ap")],
        )

    def test_selection_is_independent_of_batch_position(self) -> None:
        # Two keys with disjoint histories: neither key's candidates
        # dominate the other key's resolution, so every entry observes
        # the same relevant state in either request order.
        def seed() -> tuple[dict, dict]:
            self.write("r1", "h1", "other1", "x", {"r1": 1})
            self.write("r9", "c9", "k1", "rich", {"r1": 1, "r9": 1})
            self.write("r2", "c2", "k1", "poor", {"r2": 1})
            self.write("s1", "h1", "other2", "x", {"s1": 1})
            self.write("s9", "c9", "k2", "reicht", {"s1": 1, "s9": 1})
            self.write("s2", "c2", "k2", "arm", {"s2": 1})
            return (
                {"r1": 1, "r2": 1, "r9": 1, "r3": 1},
                {"s1": 1, "s2": 1, "s9": 1, "r3": 1},
            )

        clock1, clock2 = seed()
        forward = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            batch_entry("k2", "r3", "f2", clock2, POLICY),
        )
        status, payload = self.post_batch(forward)
        self.assertEqual(status, 201)
        forward_values = {r["key"]: r["value"] for r in payload["resolutions"]}

        self.server.store = type(self.server.store)()
        clock1, clock2 = seed()
        backward = self.document(
            batch_entry("k2", "r3", "f2", clock2, POLICY),
            batch_entry("k1", "r3", "f1", clock1, POLICY),
        )
        status, payload = self.post_batch(backward)
        self.assertEqual(status, 201)
        backward_values = {r["key"]: r["value"] for r in payload["resolutions"]}
        self.assertEqual(forward_values, backward_values)
        self.assertEqual(forward_values, {"k1": "rich", "k2": "reicht"})

    def test_one_conflict_rejects_whole_batch_unchanged(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
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
            ["h1", "c9", "c2"],
        )
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_legal_clock_not_dominating_rejects_whole_batch_409(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        self.seed_rich_conflict("k2", prefix="s")
        doc = self.document(
            batch_entry("k1", "r3", "f1", clock1, POLICY),
            # Structurally legal but dominates only one of k2's candidates.
            batch_entry("k2", "r3", "f2", {"sr1": 1, "r3": 1}, POLICY),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 6)

    def test_batch_mixes_accepted_and_replayed_entries(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        first = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(first)[0], 201)
        clock2 = self.seed_rich_conflict("k2", prefix="s")
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

    def test_replay_with_other_policy_is_operation_conflict(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        doc = self.document(batch_entry("k1", "r3", "f1", clock1, POLICY))
        self.assertEqual(self.post_batch(doc)[0], 201)
        tampered = self.document(batch_entry("k1", "r3", "f1", clock1, "plurality_value"))
        status, payload = self.post_batch(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanCausalHistoryPolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plan_previews_selection_without_writing(self) -> None:
        clock1 = self.seed_rich_conflict("k1")
        clock2 = self.seed_rich_conflict("k2", prefix="s")
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
        self.assertEqual(len(page["operations"]), 6)

    def test_plan_then_commit_agrees_and_commit_is_still_fresh(self) -> None:
        clock = self.seed_rich_conflict()
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

    def test_plan_counts_staged_earlier_entries_like_the_batch(self) -> None:
        # The same staged-log scenario as the committing batch: the plan
        # must agree with the commit, entry for entry.
        self.write("r1", "o1", "k1", "x", {"r1": 1})
        self.write("r2", "o2", "k1", "y", {"r2": 1})
        self.write("r6", "h6", "other", "p", {"r6": 1})
        self.write("r7", "h7", "other", "q", {"r7": 1})
        self.write("r9", "c9", "k2", "ap", {"r1": 1, "r2": 1, "r3": 1, "r9": 1})
        self.write("r8", "c8", "k2", "bp", {"r6": 1, "r7": 1, "r8": 1})
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, POLICY),
            batch_entry(
                "k2",
                "r3",
                "f2",
                {"r1": 1, "r2": 1, "r3": 1, "r6": 1, "r7": 1, "r8": 1, "r9": 1},
                POLICY,
            ),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            [(r["key"], r["value"]) for r in planned["resolutions"]],
            [("k1", "x"), ("k2", "ap")],
        )
        # The plan wrote nothing: committing the same document is a fresh
        # 201 with the identical resolutions.
        status, committed = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])

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
        """The rich-conflict fixture: r9's candidate dominates one op."""
        for replica, op_id, key, value, clock in (
            ("r1", "h1", "other", "x", {"r1": 1}),
            ("r9", "c9", "k", "rich", {"r1": 1, "r9": 1}),
            ("r2", "c2", "k", "poor", {"r2": 1}),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, key, value, clock),
            )
            self.assertEqual(status, 201)
        return {"r1": 1, "r2": 1, "r9": 1, "r3": 1}

    def fix(self, clock: dict, policy: str = POLICY, operation_id: str = "fix-1") -> dict:
        return auto_request("r3", operation_id, clock, policy)

    def test_binding_is_durable_and_recovers(self) -> None:
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
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix(clock, "lowest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "rich")

    def test_selection_is_identical_after_recovery(self) -> None:
        # A fresh conflict resolved after a restart must select the same
        # candidate as it would have before: the recovered accepted log
        # yields the same causal-history counts.
        server = self.start_server()
        clock = self.seed(server)
        plan_doc = {"resolutions": [batch_entry("k", "r3", "fix-9", clock, POLICY)]}
        status, before = self.request(server, "POST", PLAN_PATH, plan_doc)
        self.assertEqual(status, 200)
        self.assertEqual(before["resolutions"][0]["value"], "rich")

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "POST", PLAN_PATH, plan_doc)
        self.assertEqual(status, 200)
        self.assertEqual(after["resolutions"], before["resolutions"])
        # And the commit after recovery agrees with the preview.
        status, committed = self.request(server, "POST", BATCH_PATH, plan_doc)
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], before["resolutions"])

    def test_policy_accepted_in_stored_policies_section(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
