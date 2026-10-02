"""Tests for the causal_dominant automatic-resolution policy.

The new ``"causal_dominant"`` policy is accepted by the same three public
entries as ``lowest_identity`` / ``highest_identity``:

* ``POST /v1/states/{key}/resolve/auto``
* ``POST /v1/resolve/auto/batch``
* ``POST /v1/resolve/auto/plan`` (read-only)

The server compares the current candidates' vector clocks and, iff exactly
one candidate's clock strictly dominates every other candidate's clock,
takes that candidate's value. Different-valued candidates without a unique
strict dominator are HTTP 409 ``{"error":"causal_ambiguity"}``; a legal
request clock that does not dominate every current candidate is HTTP 409
``{"error":"resolution_conflict"}`` (unlike the identity policies, which
answer that case with 400).

Reachability note: the engine's candidate frontier (``_next_candidates``)
prunes a version whenever an incoming version dominates it, on every state
building path, so a genuine *value* conflict reached over HTTP is always an
antichain (the candidates are mutually concurrent, or carry equal clocks)
and ``causal_dominant`` therefore reports ``causal_ambiguity``. A
comparable frontier with a unique strict dominator is exercised directly at
the ``StateStore`` layer, where the deterministic selection, commit,
replay, batch atomicity, read-only preview, and durable recovery must all
be exact. HTTP-level tests below cover the observable ambiguity, the
precondition and validation errors, and the identity-replay rules.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    load_data_file_full,
    parse_auto_resolve_batch,
    parse_auto_resolve_payload,
)

BATCH_PATH = "/v1/resolve/auto/batch"
PLAN_PATH = "/v1/resolve/auto/plan"
POLICY = "causal_dominant"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def auto_request(replica: str, operation_id: str, clock: dict) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": POLICY,
    }


def batch_entry(key: str, replica: str, operation_id: str, clock: dict) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": POLICY,
    }


def batch_document(*entries: dict) -> dict:
    return {"resolutions": list(entries)}


def candidate(value: str, clock: dict, replica: str, operation_id: str) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica,
        "operationId": operation_id,
    }


# A comparable frontier: o3 causally dominates o1 and o2 (strictly), and the
# three values differ. This cannot be produced by plain writes — the write
# path prunes dominated versions — so tests install it directly.
def dominated_frontier(merged: str = "v3") -> list[dict]:
    return [
        candidate("v1", {"r1": 1}, "r1", "o1"),
        candidate("v2", {"r2": 1}, "r2", "o2"),
        candidate(merged, {"r1": 1, "r2": 1, "r4": 1}, "r4", "o4"),
    ]


DOMINATING_CLOCK = {"r1": 1, "r2": 1, "r4": 1, "r3": 1}


class CausalParseTests(unittest.TestCase):
    def test_single_payload_accepts_causal_dominant(self) -> None:
        payload = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(payload["policy"], "causal_dominant")

    def test_batch_entries_accept_causal_dominant(self) -> None:
        entries = parse_auto_resolve_batch(
            json.dumps(
                batch_document(
                    batch_entry("k1", "r3", "f1", {"r1": 1, "r3": 1}),
                    batch_entry("k2", "r9", "f9", {"r2": 1, "r9": 1}),
                )
            )
        )
        self.assertEqual([e["policy"] for e in entries], ["causal_dominant"] * 2)

    def test_near_miss_policy_strings_are_rejected(self) -> None:
        for raw_policy in ("causal", "Causal_Dominant", "causal-dominant", "dominant"):
            with self.subTest(policy=raw_policy):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_payload(
                        {
                            "replicaId": "r3",
                            "operationId": "fix-1",
                            "clock": {"r3": 1},
                            "policy": raw_policy,
                        }
                    )
        # The batch parser rejects an unknown policy too.
        entry = batch_entry("k", "r3", "f1", {"r3": 1})
        entry["policy"] = "causal"
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch({"resolutions": [entry]})


class CausalStoreSelectionTests(unittest.TestCase):
    """Deterministic selection, commit, and replay on comparable frontiers."""

    def test_single_unique_dominator_is_chosen(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier()
        request = auto_request("r3", "fix-1", DOMINATING_CLOCK)
        status, op, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(op["value"], "v3")
        status, state = store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v3")
        self.assertEqual(state["clock"], DOMINATING_CLOCK)

    def test_single_two_candidate_strict_dominator_is_chosen(self) -> None:
        store = StateStore()
        store._candidates["k"] = [
            candidate("old", {"r1": 1}, "r1", "o1"),
            candidate("new", {"r1": 1, "r2": 1}, "r2", "o2"),
        ]
        status, op, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(op["value"], "new")

    def test_concurrent_candidates_are_causal_ambiguity(self) -> None:
        store = StateStore()
        store._candidates["k"] = [
            candidate("v1", {"r1": 1}, "r1", "o1"),
            candidate("v2", {"r2": 1}, "r2", "o2"),
        ]
        # A dominating request clock cannot make the candidates comparable.
        status, op, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertIsNone(op)
        self.assertEqual(error, "causal_ambiguity")

    def test_two_concurrent_descendants_are_ambiguous(self) -> None:
        # o2 and o3 both descend from o1 but are concurrent with each other;
        # no candidate dominates ALL the others.
        store = StateStore()
        store._candidates["m"] = [
            candidate("v1", {"r1": 1}, "r1", "o1"),
            candidate("v2", {"r1": 1, "r2": 1}, "r2", "o2"),
            candidate("v3", {"r1": 1, "r3": 1}, "r3", "o3"),
        ]
        status, _, error = store.apply_auto_resolution(
            "m", auto_request("r9", "f9", {"r1": 1, "r2": 1, "r3": 1, "r9": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")

    def test_equal_clocks_differing_values_are_ambiguous(self) -> None:
        store = StateStore()
        store._candidates["k"] = [
            candidate("x", {"r1": 1, "r2": 1}, "r1", "a"),
            candidate("y", {"r1": 1, "r2": 1}, "r2", "b"),
        ]
        status, _, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")

    def test_legal_undominating_clock_is_resolution_conflict(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier()
        # Structurally legal (contains r3) but missing the r4 component, so
        # it does not dominate every current candidate: a 409 under causal,
        # not the ValueError->400 the identity policies produce.
        status, op, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertIsNone(op)
        self.assertEqual(error, "resolution_conflict")
        # Nothing committed.
        status, state = store.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)

    def test_undominating_clock_beats_ambiguity(self) -> None:
        # Concurrent candidates would be ambiguous, but the stale clock fails
        # the domination precondition first.
        store = StateStore()
        store._candidates["k"] = [
            candidate("v1", {"r1": 1}, "r1", "o1"),
            candidate("v2", {"r2": 1}, "r2", "o2"),
        ]
        status, _, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix-1", {"r2": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")

    def test_missing_single_and_same_value_are_resolution_conflict(self) -> None:
        store = StateStore()
        status, _, error = store.apply_auto_resolution(
            "absent", auto_request("r3", "fix-1", {"r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")

        # Unique dominator, but every candidate agrees on one value: the key
        # is not in value conflict.
        store._candidates["s"] = [
            candidate("same", {"r7": 1}, "r7", "a1"),
            candidate("same", {"r7": 1, "r8": 1}, "r8", "a2"),
        ]
        status, _, error = store.apply_auto_resolution(
            "s", auto_request("r3", "fix-2", {"r7": 1, "r8": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")

    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier()
        request = auto_request("r3", "fix-1", DOMINATING_CLOCK)
        self.assertIs(
            store.apply_auto_resolution("k", request)[0], HTTPStatus.CREATED
        )
        accepted_before = list(store._accepted)
        status, op, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(op["value"], "v3")
        self.assertEqual(store._accepted, accepted_before)

    def test_replay_reports_original_value_after_frontier_moved(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier()
        request = auto_request("r3", "fix-1", DOMINATING_CLOCK)
        self.assertIs(
            store.apply_auto_resolution("k", request)[0], HTTPStatus.CREATED
        )
        # Reopen a value conflict with a fresh concurrent candidate.
        store._candidates["k"] = [
            candidate("v3", DOMINATING_CLOCK, "r3", "fix-1"),
            candidate("v5", {"r1": 2, "r2": 1, "r4": 1}, "r2", "o5"),
        ]
        status, op, _ = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(op["value"], "v3")

    def test_different_policy_under_same_identity_is_operation_conflict(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier(merged="v1")
        causal = auto_request("r3", "fix-1", DOMINATING_CLOCK)
        for committed, replayed in (
            (
                dict(causal, policy="lowest_identity"),
                causal,
            ),
            (
                causal,
                dict(causal, policy="highest_identity"),
            ),
        ):
            with self.subTest(committed=committed["policy"]):
                local = StateStore()
                local._candidates["k"] = dominated_frontier(merged="v1")
                self.assertIs(
                    local.apply_auto_resolution("k", committed)[0],
                    HTTPStatus.CREATED,
                )
                status, _, error = local.apply_auto_resolution("k", replayed)
                self.assertIs(status, HTTPStatus.CONFLICT)
                self.assertEqual(error, "operation_conflict")

    def test_identity_without_local_binding_is_operation_conflict(self) -> None:
        store = StateStore()
        store._candidates["k"] = dominated_frontier()
        # A plain write (no policy binding) occupies the identity.
        store.apply_operation(
            "r3", operation("fix-1", "other", "v", {"r3": 1})
        )
        status, _, error = store.apply_auto_resolution(
            "k", auto_request("r3", "fix-1", DOMINATING_CLOCK)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")


class CausalStoreBatchTests(unittest.TestCase):
    def install(self, store: StateStore, key: str, frontier: list[dict]) -> None:
        store._candidates[key] = [dict(c, clock=dict(c["clock"])) for c in frontier]

    def test_batch_of_dominating_entries_commits(self) -> None:
        store = StateStore()
        self.install(store, "k1", dominated_frontier())
        self.install(store, "k2", dominated_frontier(merged="w3"))
        entries = [
            batch_entry("k1", "r3", "f1", DOMINATING_CLOCK),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r4": 1, "r3": 2}),
        ]
        status, results, accepted, replayed, error = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(accepted, 2)
        self.assertEqual(replayed, 0)
        self.assertEqual([r["value"] for r in results], ["v3", "w3"])
        self.assertTrue(all(r["policy"] == POLICY for r in results))

    def test_ambiguous_entry_fails_the_whole_batch(self) -> None:
        store = StateStore()
        self.install(store, "k1", dominated_frontier())
        self.install(
            store,
            "k2",
            [
                candidate("v1", {"r1": 1}, "r1", "c1"),
                candidate("v2", {"r2": 1}, "r2", "c2"),
            ],
        )
        entries = [
            batch_entry("k1", "r3", "f1", DOMINATING_CLOCK),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        ]
        status, results, accepted, replayed, error = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")
        self.assertEqual(results, [])
        # Nothing from the batch committed: k1 is still conflicted, log empty.
        _, state1 = store.get_state("k1")
        self.assertEqual(state1["status"], "conflict")
        self.assertEqual(len(store._accepted), 0)
        self.assertEqual(store._policies, {})

    def test_conflict_kind_follows_first_failing_entry(self) -> None:
        ambiguous = [
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}
            ),
        ]
        # k1 stale clock -> resolution_conflict comes first here.
        store = StateStore()
        self.install(
            store,
            "k1",
            [
                candidate("v1", {"r1": 1}, "r1", "c1"),
                candidate("v2", {"r2": 1}, "r2", "c2"),
            ],
        )
        self.install(
            store,
            "k2",
            [
                candidate("v1", {"r1": 1}, "r1", "d1"),
                candidate("v2", {"r2": 1}, "r2", "d2"),
            ],
        )
        entries = [
            batch_entry("k1", "r3", "f1", {"r1": 1, "r3": 1}),  # stale
            ambiguous[0],
        ]
        status, _, _, _, error = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")
        # Reversed: the ambiguous entry surfaces first.
        status, _, _, _, error = store.apply_auto_resolutions(list(reversed(entries)))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")

    def test_replay_after_commit_is_200(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        entries = [batch_entry("k", "r3", "f1", DOMINATING_CLOCK)]
        self.assertIs(store.apply_auto_resolutions(entries)[0], HTTPStatus.CREATED)
        status, results, accepted, replayed, error = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(accepted, 0)
        self.assertEqual(replayed, 1)
        self.assertEqual(results[0]["value"], "v3")
        self.assertEqual(results[0]["policy"], POLICY)

    def test_different_policy_under_known_identity_is_operation_conflict(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        committed = [
            {
                "key": "k",
                "replicaId": "r3",
                "operationId": "f1",
                "clock": DOMINATING_CLOCK,
                "policy": "highest_identity",
            }
        ]
        self.assertIs(
            store.apply_auto_resolutions(committed)[0], HTTPStatus.CREATED
        )
        status, _, _, _, error = store.apply_auto_resolutions(
            [batch_entry("k", "r3", "f1", DOMINATING_CLOCK)]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")


class CausalStorePlanTests(unittest.TestCase):
    def install(self, store: StateStore, key: str, frontier: list[dict]) -> None:
        store._candidates[key] = [dict(c, clock=dict(c["clock"])) for c in frontier]

    def test_plan_selects_dominating_candidate_without_writing(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        entries = [batch_entry("k", "r3", "f1", DOMINATING_CLOCK)]
        status, results, accepted, replayed, error = store.plan_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(accepted, 1)
        self.assertEqual(replayed, 0)
        self.assertEqual(results[0]["value"], "v3")
        self.assertEqual(results[0]["policy"], POLICY)
        # Read-only: no accepted record, no binding, frontier unchanged.
        self.assertEqual(store._accepted, [])
        self.assertEqual(store._policies, {})
        _, state = store.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)
        # The same request can then commit for real (a fresh 201).
        status, _, _, _, _ = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CREATED)

    def test_plan_ambiguity_is_409_and_read_only(self) -> None:
        store = StateStore()
        self.install(
            store,
            "k",
            [
                candidate("v1", {"r1": 1}, "r1", "c1"),
                candidate("v2", {"r2": 1}, "r2", "c2"),
            ],
        )
        entries = [
            batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
        ]
        status, results, _, _, error = store.plan_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")
        self.assertEqual(results, [])
        self.assertEqual(store._accepted, [])

    def test_plan_stale_clock_is_resolution_conflict(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        entries = [batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})]
        status, _, _, _, error = store.plan_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")

    def test_plan_counts_replays_against_committed_binding(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        entries = [batch_entry("k", "r3", "f1", DOMINATING_CLOCK)]
        self.assertIs(store.apply_auto_resolutions(entries)[0], HTTPStatus.CREATED)
        status, results, accepted, replayed, _ = store.plan_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["value"], "v3")

    def test_plan_different_policy_under_bound_identity_is_conflict(self) -> None:
        store = StateStore()
        self.install(store, "k", dominated_frontier())
        self.assertIs(
            store.apply_auto_resolutions(
                [
                    {
                        "key": "k",
                        "replicaId": "r3",
                        "operationId": "f1",
                        "clock": DOMINATING_CLOCK,
                        "policy": "lowest_identity",
                    }
                ]
            )[0],
            HTTPStatus.CREATED,
        )
        status, _, _, _, error = store.plan_auto_resolutions(
            [batch_entry("k", "r3", "f1", DOMINATING_CLOCK)]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")


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
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            conn.request(
                method, path, body=data, headers={"Content-Type": "application/json"}
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

    def get_sync(self) -> tuple[int, object]:
        return self.request("GET", "/v1/sync/operations")

    def seed_concurrent_conflict(self, key: str = "k") -> None:
        salt = getattr(self, "_salt", 0)
        self._salt = salt + 1
        self.assertEqual(
            self.post_operation(
                "r1", operation(f"o1-{key}-{salt}", key, "v1", {"r1": 1})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                "r2", operation(f"o2-{key}-{salt}", key, "v2", {"r2": 1})
            )[0],
            201,
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class CausalHttpTests(HttpServerTestCase):
    def test_concurrent_conflict_is_causal_ambiguity(self) -> None:
        self.seed_concurrent_conflict("k")
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_three_concurrent_candidates_are_ambiguous(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("a", "k", "v1", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("b", "k", "v2", {"r2": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r5", operation("c", "k", "v5", {"r5": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r5": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_equal_clocks_differing_values_are_ambiguous(self) -> None:
        self.assertEqual(
            self.post_operation(
                "r1", operation("a", "k", "x", {"r1": 1, "r2": 1})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                "r2", operation("b", "k", "y", {"r1": 1, "r2": 1})
            )[0],
            201,
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_legal_undominating_clock_is_409_resolution_conflict(self) -> None:
        self.seed_concurrent_conflict("k")
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_structurally_invalid_clock_and_policy_are_400(self) -> None:
        self.seed_concurrent_conflict("k")
        for bad_clock in ({}, {"r2": 1}, {"r3": -1}, {"r3": True}, {"r3": 1.0}):
            with self.subTest(clock=bad_clock):
                status, payload = self.post_auto(
                    "k", auto_request("r3", "fix-1", bad_clock)
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.post_auto(
            "k",
            {
                "replicaId": "r3",
                "operationId": "fix-1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "causal",
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_unconflicted_and_same_value_are_resolution_conflict(self) -> None:
        status, payload = self.post_auto(
            "absent", auto_request("r3", "fix-1", {"r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

        self.assertEqual(
            self.post_operation("r1", operation("solo", "k", "s", {"r1": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-2", {"r1": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

        self.assertEqual(
            self.post_operation("r7", operation("a1", "same", "s", {"r7": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation(
                "r8", operation("a2", "same", "s", {"r7": 1, "r8": 1})
            )[0],
            201,
        )
        _, state = self.get_state("same")
        self.assertEqual(state["status"], "resolved")
        status, payload = self.post_auto(
            "same", auto_request("r3", "fix-3", {"r7": 1, "r8": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_identity_policy_binding_blocks_a_causal_replay(self) -> None:
        self.seed_concurrent_conflict("k")
        # lowest_identity can commit on the concurrent conflict; causal cannot.
        identity_body = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        self.assertEqual(self.post_auto("k", identity_body)[0], 201)
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_plain_write_identity_has_no_binding(self) -> None:
        self.seed_concurrent_conflict("k")
        self.assertEqual(
            self.post_operation(
                "r3", operation("fix-1", "other", "v", {"r3": 1})
            )[0],
            201,
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_ambiguity_appends_nothing_and_keeps_metrics(self) -> None:
        self.seed_concurrent_conflict("k")
        _, metrics_before = self.request("GET", "/v1/metrics")
        status, _ = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        _, metrics_after = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(metrics_after["conflictKeys"], 1)


class CausalHttpBatchTests(HttpServerTestCase):
    def test_ambiguous_batch_fails_whole_and_is_atomic(self) -> None:
        self.seed_concurrent_conflict("k1")
        self.seed_concurrent_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})
        for key in ("k1", "k2"):
            _, state = self.get_state(key)
            self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)

    def test_batch_conflict_kind_follows_first_failing_entry(self) -> None:
        self.seed_concurrent_conflict("k1")
        self.seed_concurrent_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r3": 1}),  # stale clock
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),  # ambiguous
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        status, payload = self.post_batch(
            batch_document(
                batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
                batch_entry("k1", "r3", "f1", {"r1": 1, "r3": 1}),
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_batch_unknown_policy_and_bad_clock_are_400(self) -> None:
        self.seed_concurrent_conflict("k")
        entry = batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
        entry["policy"] = "causal"
        status, payload = self.post_batch({"resolutions": [entry]})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.post_batch(
            batch_document(batch_entry("k", "r3", "f1", {"r3": -1}))
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_batch_identity_bound_under_other_policy_is_operation_conflict(self) -> None:
        self.seed_concurrent_conflict("k")
        committed = {
            "key": "k",
            "replicaId": "r3",
            "operationId": "f1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        self.assertEqual(self.post_batch({"resolutions": [committed]})[0], 201)
        status, payload = self.post_batch(
            batch_document(
                batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class CausalHttpPlanTests(HttpServerTestCase):
    def test_plan_of_concurrent_conflict_is_ambiguity_and_read_only(self) -> None:
        self.seed_concurrent_conflict("k1")
        self.seed_concurrent_conflict("k2")
        _, sync_before = self.get_sync()
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})
        _, sync_after = self.get_sync()
        self.assertEqual(sync_after, sync_before)
        for key in ("k1", "k2"):
            _, state = self.get_state(key)
            self.assertEqual(state["status"], "conflict")

    def test_plan_stale_clock_is_resolution_conflict(self) -> None:
        self.seed_concurrent_conflict("k")
        status, payload = self.post_plan(
            batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_plan_unknown_policy_is_400(self) -> None:
        self.seed_concurrent_conflict("k")
        entry = batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
        entry["policy"] = "causal-dominant"
        status, payload = self.post_plan({"resolutions": [entry]})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_plan_does_not_bind_identity(self) -> None:
        self.seed_concurrent_conflict("k")
        doc = batch_document(
            batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
        )
        # Ambiguous twice: the first preview binds nothing.
        self.assertEqual(self.post_plan(doc)[0], 409)
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "causal_ambiguity"})

    def test_plan_identity_bound_under_other_policy_is_operation_conflict(self) -> None:
        self.seed_concurrent_conflict("k")
        committed = {
            "key": "k",
            "replicaId": "r3",
            "operationId": "f1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "highest_identity",
        }
        self.assertEqual(self.post_batch({"resolutions": [committed]})[0], 201)
        status, payload = self.post_plan(
            batch_document(
                batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PersistentCausalStoreTests(unittest.TestCase):
    """causal_dominant commit, binding, recovery, and durable failure."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def make_store(self) -> StateStore:
        return StateStore(data_file=self.data_file)

    def prepare(self, store: StateStore) -> None:
        # Commit the two concurrent base operations, then install the
        # comparable frontier a plain write sequence could not leave behind.
        store.apply_operation(
            "r1", operation("o1", "k", "v1", {"r1": 1})
        )
        store.apply_operation(
            "r2", operation("o2", "k", "v2", {"r2": 1})
        )
        store._candidates["k"] = dominated_frontier()

    def request(self) -> dict:
        return auto_request("r3", "fix-1", DOMINATING_CLOCK)

    def test_binding_is_durable_and_replays_after_restart(self) -> None:
        store = self.make_store()
        self.prepare(store)
        status, op, error = store.apply_auto_resolution("k", self.request())
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(op["value"], "v3")
        self.assertIsNone(error)
        _, _, policies = load_data_file_full(self.data_file)
        self.assertEqual(policies, {("r3", "fix-1"): "causal_dominant"})
        self.assertEqual(
            [(r, o["operationId"]) for r, o in load_data_file(self.data_file)],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1")],
        )

        restarted = self.make_store()
        status, state = restarted.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v3")
        status, op, _ = restarted.apply_auto_resolution("k", self.request())
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(op["value"], "v3")
        self.assertEqual(len(load_data_file(self.data_file)), 3)

    def test_ambiguity_is_stable_across_restart(self) -> None:
        store = self.make_store()
        store.apply_operation("r5", operation("a1", "m", "x", {"r5": 1}))
        store.apply_operation("r6", operation("a2", "m", "y", {"r6": 1}))
        restarted = self.make_store()
        status, _, error = restarted.apply_auto_resolution(
            "m", auto_request("r3", "fix-2", {"r5": 1, "r6": 1, "r3": 1})
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "causal_ambiguity")
        self.assertEqual(len(load_data_file(self.data_file)), 2)

    def test_persistence_failure_is_500_class_and_changes_nothing(self) -> None:
        store = self.make_store()
        self.prepare(store)
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=server_module.PersistenceError("disk gone"),
        ):
            self.assertRaises(
                server_module.PersistenceError,
                lambda: store.apply_auto_resolution("k", self.request()),
            )
        _, _, policies = load_data_file_full(self.data_file)
        self.assertEqual(policies, {})
        _, state = store.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(load_data_file(self.data_file)), 2)
        # The same request commits cleanly once persistence works again.
        status, op, _ = store.apply_auto_resolution("k", self.request())
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(op["value"], "v3")
        _, _, policies = load_data_file_full(self.data_file)
        self.assertEqual(policies, {("r3", "fix-1"): "causal_dominant"})


if __name__ == "__main__":
    unittest.main()
