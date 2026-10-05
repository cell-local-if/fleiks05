"""Tests for the value-based automatic-resolution policies.

The three automatic-resolution entry points::

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

each accept six policies. Alongside the identity policies
(``lowest_identity``/``highest_identity``), ``lowest_value`` selects the
smallest current candidate string value and ``highest_value`` the largest,
compared by Unicode code point in ascending order. A repeated extreme value
changes none of the identity idempotence rules. Everything here goes
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


class ValuePolicyParseTests(unittest.TestCase):
    def test_constant_lists_all_six_policies(self) -> None:
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

    def test_value_policies_parse_single_and_batch(self) -> None:
        for policy in ("lowest_value", "highest_value"):
            with self.subTest(policy=policy):
                single = parse_auto_resolve_payload(
                    json.dumps(auto_request("r3", "f1", {"r3": 1}, policy))
                )
                self.assertEqual(single["policy"], policy)
                entries = parse_auto_resolve_batch(
                    json.dumps(
                        {"resolutions": [batch_entry("k", "r3", "f1", {"r3": 1}, policy)]}
                    )
                )
                self.assertEqual(entries[0]["policy"], policy)

    def test_unknown_or_non_string_policy_is_rejected(self) -> None:
        valid = auto_request("r3", "f", {"r3": 1}, "lowest_value")
        for policy in ("", "lowest", "LOWEST_VALUE", "middle_value", "lowest-value",
                       42, None, [], {}, True):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_payload(dict(valid, policy=policy))
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

    def seed(
        self,
        key: str = "k",
        low: str = "v1",
        high: str = "v2",
        replicas: tuple[str, str] = ("r1", "r2"),
    ) -> None:
        """Two concurrent writes carrying ``low`` and ``high`` on ``key``."""
        first, second = replicas
        self.assertEqual(
            self.post_operation(first, operation("o1", key, low, {first: 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation(second, operation("o2", key, high, {second: 1}))[0], 201
        )

    def fix(
        self,
        policy: str,
        operation_id: str = "fix-1",
        replica: str = "r3",
        clock: dict | None = None,
    ) -> dict:
        return auto_request(
            replica, operation_id, clock or {"r1": 1, "r2": 1, "r3": 1}, policy
        )


class SingleKeyValuePolicyTests(HttpServerTestCase):
    def test_lowest_value_picks_smallest_string(self) -> None:
        # Identity order is deliberately opposite to value order.
        self.seed(low="zzz", high="aaa")
        status, payload = self.post_auto("k", self.fix("lowest_value"))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "aaa")
        self.assertEqual(payload["policy"], "lowest_value")
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "aaa")

    def test_highest_value_picks_largest_string(self) -> None:
        self.seed(low="aaa", high="zzz")
        status, payload = self.post_auto("k", self.fix("highest_value"))
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zzz")
        self.assertEqual(payload["policy"], "highest_value")
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "zzz")

    def test_selection_ignores_identity_order(self) -> None:
        # The lexicographically smallest identity ("r1","o1") carries the
        # largest value; under a value policy identity order decides nothing.
        self.post_operation("r1", operation("o1", "k", "zzz", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "aaa", {"r2": 1}))
        status, low = self.post_auto("k", self.fix("lowest_value"))
        self.assertEqual(status, 201)
        self.assertEqual(low["value"], "aaa")

        self.server.store = type(self.server.store)()
        self.post_operation("r1", operation("o1", "k", "zzz", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "aaa", {"r2": 1}))
        status, high = self.post_auto("k", self.fix("highest_value"))
        self.assertEqual(status, 201)
        self.assertEqual(high["value"], "zzz")

    def test_comparison_is_unicode_code_point_order(self) -> None:
        # U+007A ("z") sorts below U+00E9 ("é"), which sorts below
        # U+4E2D ("中"); plain Python string ordering is code-point order.
        self.assertLess("z", "é")
        self.assertLess("é", "中")
        cases = [
            (("z", "é"), "lowest_value", "z"),
            (("z", "é"), "highest_value", "é"),
            (("é", "中"), "lowest_value", "é"),
            (("é", "中"), "highest_value", "中"),
        ]
        for index, ((first, second), policy, expected) in enumerate(cases):
            with self.subTest(policy=policy, values=(first, second)):
                self.server.store = type(self.server.store)()
                self.seed(key="k", low=first, high=second)
                status, payload = self.post_auto(
                    "k", self.fix(policy, operation_id=f"fix-{index}")
                )
                self.assertEqual(status, 201)
                self.assertEqual(payload["value"], expected)

    def test_repeated_extreme_value_still_resolves_and_replays(self) -> None:
        # Three candidates; two distinct identities carry the same minimum
        # value "aaa". The tie changes nothing: the key resolves to "aaa"
        # and idempotent replay reports the same value.
        self.post_operation("r1", operation("o1", "k", "aaa", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "zzz", {"r2": 1}))
        self.post_operation("r4", operation("o4", "k", "aaa", {"r4": 1}))
        body = self.fix(
            "lowest_value", clock={"r1": 1, "r2": 1, "r4": 1, "r3": 1}
        )
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "aaa")
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "aaa")
        status, replay = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(replay["value"], "aaa")
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "o4", "fix-1"],
        )

    def test_chosen_value_becomes_the_only_version(self) -> None:
        self.seed(low="aaa", high="zzz")
        self.assertEqual(self.post_auto("k", self.fix("lowest_value"))[0], 201)
        _, state = self.get_state("k")
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "aaa",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )


class SingleKeyValuePolicyConflictTests(HttpServerTestCase):
    def test_unknown_or_non_string_policy_is_400(self) -> None:
        self.seed()
        for policy in ("lowest", "HIGHEST_VALUE", 42, None, ["lowest_value"]):
            with self.subTest(policy=policy):
                status, payload = self.post_auto("k", self.fix(policy))
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_not_dominating_is_400(self) -> None:
        self.seed()
        for policy in ("lowest_value", "highest_value"):
            with self.subTest(policy=policy):
                status, payload = self.post_auto(
                    "k",
                    auto_request("r3", "fix-x", {"r1": 1, "r3": 1}, policy),
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_key_is_409(self) -> None:
        for policy in ("lowest_value", "highest_value"):
            status, payload = self.post_auto(
                "absent", auto_request("r3", f"fix-{policy}", {"r3": 1}, policy)
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        for policy in ("lowest_value", "highest_value"):
            status, payload = self.post_auto("k", self.fix(policy))
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_rejects_a_new_value_policy_identity(self) -> None:
        # Candidate set moved: after the first repair the key holds one
        # version, so a second unseen identity is a resolution conflict.
        self.seed()
        self.assertEqual(self.post_auto("k", self.fix("lowest_value"))[0], 201)
        status, payload = self.post_auto("k", self.fix("highest_value", "fix-2"))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class SingleKeyValuePolicyIdentityTests(HttpServerTestCase):
    def test_replay_is_200_and_reports_original_value(self) -> None:
        self.seed(low="aaa", high="zzz")
        body = self.fix("lowest_value")
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
                "value": "aaa",
                "policy": "lowest_value",
            },
        )

    def test_different_policy_under_same_identity_is_409(self) -> None:
        # Every cross-policy pairing is a different binding, including the
        # value policies against the identity policies.
        policies = (
            "lowest_identity",
            "highest_identity",
            "lowest_value",
            "highest_value",
            "plurality_value",
        )
        for committed in policies:
            for replayed in policies:
                if replayed == committed:
                    continue
                with self.subTest(committed=committed, replayed=replayed):
                    self.server.store = type(self.server.store)()
                    self.seed(low="v1", high="v2")
                    self.assertEqual(self.post_auto("k", self.fix(committed))[0], 201)
                    status, payload = self.post_auto("k", self.fix(replayed))
                    self.assertEqual(status, 409)
                    self.assertEqual(payload, {"error": "operation_conflict"})

    def test_value_replay_after_key_moved_on_reports_original_value(self) -> None:
        self.seed(low="v1", high="v2")
        body = self.fix("highest_value")
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on with a fresh concurrent conflict.
        self.post_operation("r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0}))
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v2")

    def test_plain_write_identity_never_matches_value_request(self) -> None:
        self.seed()
        status, _ = self.post_operation(
            "r3", operation("fix-1", "k", "v1", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix("lowest_value"))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class SingleKeyValuePolicyIntegrationTests(HttpServerTestCase):
    def test_value_resolution_is_exported_with_chosen_value(self) -> None:
        self.seed(low="aaa", high="zzz")
        self.assertEqual(self.post_auto("k", self.fix("highest_value"))[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][2],
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "zzz",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                },
            },
        )

    def test_imported_value_resolution_carries_no_binding(self) -> None:
        self.seed(low="aaa", high="zzz")
        self.assertEqual(self.post_auto("k", self.fix("lowest_value"))[0], 201)
        _, page = self.get_sync()

        self.server.store = type(self.server.store)()
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": page["operations"]})
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.fix("lowest_value"))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class BatchValuePolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_mixed_policies_select_by_their_own_rule(self) -> None:
        self.seed("k1", low="v1", high="v2")
        self.seed("k2", low="aaa", high="zzz", replicas=("r5", "r6"))
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
            batch_entry("k2", "r7", "f2", {"r5": 1, "r6": 1, "r7": 1}, "highest_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "v1", "lowest_value"), ("k2", "zzz", "highest_value")],
        )

    def test_unknown_policy_rejects_whole_batch_400(self) -> None:
        self.seed("k1")
        self.seed("k2", replicas=("r5", "r6"))
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
            batch_entry("k2", "r7", "f2", {"r5": 1, "r6": 1, "r7": 1}, "middle_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)

    def test_one_conflict_rejects_whole_batch_unchanged(self) -> None:
        self.seed("k1", low="aaa", high="zzz")
        # k2 has no candidates at all.
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "highest_value"),
            batch_entry("k2", "r3", "f2", {"r3": 2}, "lowest_value"),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2"],
        )
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_batch_mixes_accepted_and_replayed_value_entries(self) -> None:
        self.seed("k1", low="aaa", high="zzz")
        first = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
        )
        self.assertEqual(self.post_batch(first)[0], 201)
        self.seed("k2", low="aaa", high="zzz", replicas=("r5", "r6"))
        again = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
            batch_entry("k2", "r7", "f2", {"r5": 1, "r6": 1, "r7": 1}, "highest_value"),
        )
        status, payload = self.post_batch(again)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(
            [(r["key"], r["value"]) for r in payload["resolutions"]],
            [("k1", "aaa"), ("k2", "zzz")],
        )

    def test_value_replay_with_other_policy_is_operation_conflict(self) -> None:
        self.seed("k1", low="aaa", high="zzz")
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
        )
        self.assertEqual(self.post_batch(doc)[0], 201)
        tampered = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "highest_value"),
        )
        status, payload = self.post_batch(tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanValuePolicyTests(HttpServerTestCase):
    def document(self, *entries: dict) -> dict:
        return {"resolutions": list(entries)}

    def test_plan_previews_value_selection_without_writing(self) -> None:
        self.seed("k1", low="aaa", high="zzz")
        self.seed("k2", low="aaa", high="zzz", replicas=("r5", "r6"))
        doc = self.document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value"),
            batch_entry("k2", "r7", "f2", {"r5": 1, "r6": 1, "r7": 1}, "highest_value"),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            [(r["key"], r["value"], r["policy"]) for r in payload["resolutions"]],
            [("k1", "aaa", "lowest_value"), ("k2", "zzz", "highest_value")],
        )
        # Nothing was written: keys still conflict and no log record exists.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)

    def test_plan_then_commit_agrees_and_commit_is_still_fresh(self) -> None:
        self.seed("k", low="aaa", high="zzz")
        doc = self.document(
            batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "highest_value"),
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

    def test_plan_unknown_policy_is_400(self) -> None:
        self.seed("k")
        doc = self.document(
            batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}, "lowest-values"),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_plan_conflict_for_missing_key_is_409(self) -> None:
        doc = self.document(
            batch_entry("absent", "r3", "f1", {"r3": 1}, "lowest_value"),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})


class PersistentValuePolicyTestCase(unittest.TestCase):
    """Value policies against a data-file-backed server with real HTTP."""

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

    def seed(self, server: SemanticStateServer, low: str = "v1", high: str = "v2") -> None:
        for replica, op in (
            ("r1", operation("o1", "k", low, {"r1": 1})),
            ("r2", operation("o2", "k", high, {"r2": 1})),
        ):
            status, _ = self.request(
                server, "POST", f"/v1/replicas/{replica}/operations", op
            )
            self.assertEqual(status, 201)

    def fix(self, policy: str) -> dict:
        return auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, policy)

    def test_value_binding_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed(server, low="aaa", high="zzz")
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix("highest_value")
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zzz")
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r3", "fix-1"): "highest_value"})

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # Same binding replays 200 with the original value.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix("highest_value")
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "zzz")
        self.assertEqual(payload["policy"], "highest_value")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.fix("lowest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "zzz")

    def test_value_policies_accepted_in_stored_policies_section(self) -> None:
        # Hand-write a data file whose binding uses the value policy name;
        # recovery validates it against the same shared policy constant.
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "a", {"r1": 1})},
                    ],
                    "policies": [
                        {"replicaId": "r1", "operationId": "o1", "policy": "lowest_value"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {("r1", "o1"): "lowest_value"})
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
