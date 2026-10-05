"""Tests for the ``plurality_value`` automatic-resolution policy.

The three automatic-resolution entry points::

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

share one policy vocabulary. ``plurality_value`` groups the current
candidates by their string value, picks the value carried by the most
candidates, and breaks ties by the smallest value in Unicode code-point
order. The outcome depends only on the multiset of candidate values, so it
is independent of the candidate order and of which identities carry each
value. The other four policies are untouched. Everything here goes through
the real HTTP entry point (``SemanticStateServer`` + a request thread);
only the Python standard library is used.
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
    def test_constant_includes_plurality_value(self) -> None:
        self.assertIn("plurality_value", AUTO_RESOLVE_POLICIES)

    def test_plurality_parses_single_and_batch(self) -> None:
        single = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "f1", {"r3": 1}, "plurality_value"))
        )
        self.assertEqual(single["policy"], "plurality_value")
        entries = parse_auto_resolve_batch(
            json.dumps(
                {
                    "resolutions": [
                        batch_entry("k", "r3", "f1", {"r3": 1}, "plurality_value")
                    ]
                }
            )
        )
        self.assertEqual(entries[0]["policy"], "plurality_value")

    def test_near_miss_policies_are_rejected(self) -> None:
        for policy in ("plurality", "PLURALITY_VALUE", "plurality-value",
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

    def seed(self, key: str, values: list[str], first_replica: int = 1) -> dict:
        """Concurrent writes carrying ``values`` on ``key``; returns the
        clock dominating every write plus one fresh fixing replica tick."""
        clock: dict[str, int] = {}
        for index, value in enumerate(values):
            replica = f"r{first_replica + index}"
            self.assertEqual(
                self.post_operation(
                    replica, operation(f"o{first_replica + index}", key, value, {replica: 1})
                )[0],
                201,
            )
            clock[replica] = 1
        return clock

    def fix(
        self,
        clock: dict,
        operation_id: str = "fix-1",
        replica: str = "r9",
        policy: str = "plurality_value",
    ) -> dict:
        return auto_request(replica, operation_id, dict(clock, **{replica: 1}), policy)


class SingleKeyPluralityTests(HttpServerTestCase):
    def test_plurality_picks_value_with_most_votes(self) -> None:
        # Two candidates carry "zzz", one carries "aaa": the smaller string
        # loses because plurality counts votes, not code points.
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zzz")
        self.assertEqual(payload["policy"], "plurality_value")
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "zzz")

    def test_tie_is_broken_by_smallest_code_point(self) -> None:
        # One vote each: the smallest string in Unicode code-point order
        # wins. U+007A ("z") sorts below U+00E9 ("é").
        clock = self.seed("k", ["é", "z"])
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "z")

    def test_tie_break_ignores_identity_and_seed_order(self) -> None:
        # The same value multiset seeded in two different write orders and
        # with the tied values carried by differently ordered identities
        # resolves to the same value both times.
        for values in (["bbb", "aaa"], ["aaa", "bbb"]):
            with self.subTest(values=values):
                self.server.store = type(self.server.store)()
                clock = self.seed("k", values)
                status, payload = self.post_auto("k", self.fix(clock))
                self.assertEqual(status, 201)
                self.assertEqual(payload["value"], "aaa")

    def test_plurality_ignores_identity_order(self) -> None:
        # The lexicographically smallest identity carries a value with only
        # one vote; the majority value belongs to larger identities.
        clock = self.seed("k", ["aaa", "mmm", "mmm"])
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "mmm")

    def test_chosen_value_becomes_the_only_version(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        self.assertEqual(self.post_auto("k", self.fix(clock))[0], 201)
        _, state = self.get_state("k")
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "zzz",
                "clock": {"r1": 1, "r2": 1, "r3": 1, "r9": 1},
                "status": "resolved",
            },
        )


class SingleKeyPluralityConflictTests(HttpServerTestCase):
    def test_missing_key_is_409(self) -> None:
        status, payload = self.post_auto(
            "absent", auto_request("r9", "fix-1", {"r9": 1}, "plurality_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409(self) -> None:
        clock = self.seed("k", ["same", "same"])
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_clock_not_dominating_is_400(self) -> None:
        self.seed("k", ["aaa", "zzz"])
        status, payload = self.post_auto(
            "k",
            auto_request("r9", "fix-1", {"r1": 1, "r9": 1}, "plurality_value"),
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_resolved_key_rejects_a_new_plurality_identity(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        self.assertEqual(self.post_auto("k", self.fix(clock))[0], 201)
        status, payload = self.post_auto("k", self.fix(clock, operation_id="fix-2"))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class SingleKeyPluralityIdentityTests(HttpServerTestCase):
    def test_replay_is_200_and_reports_original_value(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        body = self.fix(clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "key": "k",
                "replicaId": "r9",
                "operationId": "fix-1",
                "value": "zzz",
                "policy": "plurality_value",
            },
        )

    def test_different_policy_under_same_identity_is_409(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        self.assertEqual(self.post_auto("k", self.fix(clock))[0], 201)
        for policy in ("lowest_identity", "highest_identity",
                       "lowest_value", "highest_value"):
            with self.subTest(policy=policy):
                status, payload = self.post_auto("k", self.fix(clock, policy=policy))
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})

    def test_other_policy_then_plurality_under_same_identity_is_409(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        self.assertEqual(
            self.post_auto("k", self.fix(clock, policy="lowest_value"))[0], 201
        )
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_replay_after_key_moved_on_reports_original_value(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        body = self.fix(clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on with a fresh concurrent conflict.
        self.post_operation(
            "r2", operation("o7", "k", "vvv", {"r1": 1, "r2": 2, "r9": 0})
        )
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "zzz")

    def test_plain_write_identity_never_matches_plurality_request(self) -> None:
        clock = self.seed("k", ["aaa", "zzz"])
        status, _ = self.post_operation(
            "r9", operation("fix-1", "k", "aaa", dict(clock, r9=1))
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix(clock))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class SingleKeyPluralityIntegrationTests(HttpServerTestCase):
    def test_plurality_resolution_is_exported_with_chosen_value(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        self.assertEqual(self.post_auto("k", self.fix(clock))[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][3],
            {
                "replicaId": "r9",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "zzz",
                    "clock": {"r1": 1, "r2": 1, "r3": 1, "r9": 1},
                },
            },
        )

    def test_imported_plurality_resolution_carries_no_binding(self) -> None:
        clock = self.seed("k", ["aaa", "zzz", "zzz"])
        body = self.fix(clock)
        self.assertEqual(self.post_auto("k", body)[0], 201)
        _, page = self.get_sync()

        self.server.store = type(self.server.store)()
        status, _ = self.request(
            "POST", "/v1/sync/operations", {"operations": page["operations"]}
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class BatchPluralityTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_batch_plurality_entries_select_by_vote(self) -> None:
        clock1 = self.seed("k1", ["aaa", "zzz", "zzz"])
        clock2 = self.seed("k2", ["mmm", "mmm", "yyy"], first_replica=4)
        doc = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
            batch_entry("k2", "r10", "f2", dict(clock2, r10=1), "plurality_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "zzz", "plurality_value"), ("k2", "mmm", "plurality_value")],
        )

    def test_batch_mixes_plurality_with_other_policies(self) -> None:
        clock1 = self.seed("k1", ["aaa", "zzz", "zzz"])
        clock2 = self.seed("k2", ["aaa", "zzz"], first_replica=4)
        doc = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
            batch_entry("k2", "r10", "f2", dict(clock2, r10=1), "highest_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "zzz", "plurality_value"), ("k2", "zzz", "highest_value")],
        )

    def test_one_conflict_rejects_whole_batch_unchanged(self) -> None:
        clock1 = self.seed("k1", ["aaa", "zzz", "zzz"])
        # k2 has no candidates at all.
        doc = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
            batch_entry("k2", "r9", "f2", {"r9": 2}, "plurality_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "o3"],
        )
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_batch_legal_clock_not_dominating_is_409(self) -> None:
        self.seed("k1", ["aaa", "zzz", "zzz"])
        doc = self.document(
            batch_entry(
                "k1", "r9", "f1", {"r1": 1, "r2": 1, "r9": 1}, "plurality_value"
            ),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_batch_plurality_replay_and_operation_conflict(self) -> None:
        clock1 = self.seed("k1", ["aaa", "zzz", "zzz"])
        doc = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
        )
        self.assertEqual(self.post_batch(doc)[0], 201)
        # Same binding replays from the committed operation.
        status, payload = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(payload["resolutions"][0]["value"], "zzz")
        # A different policy under the same identity conflicts.
        tampered = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "lowest_value"),
        )
        status, payload = self.post_batch(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanPluralityTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plan_previews_plurality_selection_without_writing(self) -> None:
        clock1 = self.seed("k1", ["aaa", "zzz", "zzz"])
        doc = self.document(
            batch_entry("k1", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            payload["resolutions"],
            [
                {
                    "key": "k1",
                    "replicaId": "r9",
                    "operationId": "f1",
                    "value": "zzz",
                    "policy": "plurality_value",
                }
            ],
        )
        # Nothing was written: the key still conflicts, no log record exists.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 3)

    def test_plan_then_commit_agrees_and_commit_is_still_fresh(self) -> None:
        clock1 = self.seed("k", ["aaa", "zzz", "zzz"])
        doc = self.document(
            batch_entry("k", "r9", "f1", dict(clock1, r9=1), "plurality_value"),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(planned["resolutions"][0]["value"], "zzz")
        status, committed = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])
        # After committing, the same request previews as a replay.
        status, afterwards = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(afterwards["accepted"], 0)
        self.assertEqual(afterwards["replayed"], 1)
        self.assertEqual(afterwards["resolutions"][0]["value"], "zzz")

    def test_plan_conflict_for_missing_key_is_409(self) -> None:
        doc = self.document(
            batch_entry("absent", "r9", "f1", {"r9": 1}, "plurality_value"),
        )
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

    def seed(self, server: SemanticStateServer) -> None:
        for replica, value in (("r1", "aaa"), ("r2", "zzz"), ("r3", "zzz")):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(f"o{replica[1:]}", "k", value, {replica: 1}),
            )
            self.assertEqual(status, 201)

    def fix(self) -> dict:
        return auto_request(
            "r9", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r9": 1}, "plurality_value"
        )

    def test_plurality_binding_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed(server)
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zzz")
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r9", "fix-1"): "plurality_value"})

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Same binding replays 200 with the original value.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "zzz")
        self.assertEqual(payload["policy"], "plurality_value")
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server,
            "POST",
            "/v1/states/k/resolve/auto",
            auto_request(
                "r9", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r9": 1}, "highest_value"
            ),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "zzz")

    def test_plurality_accepted_in_stored_policies_section(self) -> None:
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
                        {"replicaId": "r1", "operationId": "o1", "policy": "plurality_value"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r1", "o1"): "plurality_value"})
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
