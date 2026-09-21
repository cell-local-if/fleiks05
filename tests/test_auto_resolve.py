"""HTTP, sync, and persistence tests for deterministic automatic resolution.

The endpoint is::

    POST /v1/states/{key}/resolve/auto

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
    parse_auto_resolve_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def auto_request(
    replica: str, operation_id: str, clock: dict, policy: str = "lowest_identity"
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


class ParseAutoResolvePayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        payload = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(
            payload,
            {
                "replicaId": "r3",
                "operationId": "fix-1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "lowest_identity",
            },
        )

    def test_rejects_malformed_and_wrong_shapes(self) -> None:
        valid = auto_request("r3", "fix", {"r3": 1})
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"replicaId": "r3"},
            dict(valid, value="v"),
            dict(valid, candidates=[]),
            dict(valid, extra=1),
            {k: v for k, v in valid.items() if k != "clock"},
            {k: v for k, v in valid.items() if k != "policy"},
            dict(valid, replicaId=""),
            dict(valid, replicaId=7),
            dict(valid, operationId=""),
            dict(valid, operationId=9),
            dict(valid, clock={}),
            dict(valid, clock={"r2": 1}),  # clock must contain the replica
            dict(valid, clock={"r3": -1}),
            dict(valid, clock={"r3": True}),
            dict(valid, policy=""),
            dict(valid, policy="highest-identity"),
            dict(valid, policy="lowest-value"),
            dict(valid, policy=42),
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_auto_resolve_payload(body)

    def test_highest_identity_payload_is_normalized(self) -> None:
        payload = parse_auto_resolve_payload(
            json.dumps(auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(payload["policy"], "lowest_identity")
        highest = dict(
            parse_auto_resolve_payload(
                json.dumps(
                    {
                        "replicaId": "r3",
                        "operationId": "fix-2",
                        "clock": {"r1": 1, "r2": 1, "r3": 1},
                        "policy": "highest_identity",
                    }
                )
            )
        )
        self.assertEqual(highest["policy"], "highest_identity")


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

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_audit(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def get_audit_digest(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/digest")

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

    def good_auto_request(self, operation_id: str = "fix-1") -> dict:
        return auto_request("r3", operation_id, {"r1": 1, "r2": 1, "r3": 1})


class AutoResolveHappyPathTests(HttpServerTestCase):
    def test_lowest_identity_value_is_chosen(self) -> None:
        self.seed_conflict()
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "key": "k",
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "v1",
                "policy": "lowest_identity",
            },
        )
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "v1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_highest_identity_value_is_chosen(self) -> None:
        self.seed_conflict()
        body = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "key": "k",
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "v2",
                "policy": "highest_identity",
            },
        )
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "v2",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_highest_identity_ordering_not_value_ordering_decides(self) -> None:
        # The lexicographically largest identity carries the
        # lexicographically smallest value; its value still wins.
        self.post_operation("r1", operation("o1", "k", "zzz", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "aaa", {"r2": 1}))
        body = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "aaa")

    def test_highest_tuple_ordering_uses_replica_before_operation(self) -> None:
        # ("r1", "aaa") sorts before ("r2", "zzz"): replica id is the first
        # tuple component, so the highest identity is the r2 one.
        self.post_operation("r1", operation("aaa", "k", "from-r1", {"r1": 1}))
        self.post_operation("r2", operation("zzz", "k", "from-r2", {"r2": 1}))
        body = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "from-r2")

    def test_three_candidates_pick_the_maximum(self) -> None:
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r9", operation("o9", "k", "v9", {"r9": 1}))
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = auto_request(
            "r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r9": 1}, "highest_identity"
        )
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v9")
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v9")

    def test_identity_ordering_not_value_ordering_decides(self) -> None:
        # The lexicographically smallest identity carries the
        # lexicographically largest value; its value still wins.
        self.post_operation("r1", operation("o1", "k", "zzz", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "aaa", {"r2": 1}))
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zzz")

    def test_tuple_ordering_uses_replica_before_operation(self) -> None:
        # ("r1", "zzz") sorts before ("r2", "aaa"): replica id is the first
        # tuple component.
        self.post_operation("r1", operation("zzz", "k", "from-r1", {"r1": 1}))
        self.post_operation("r2", operation("aaa", "k", "from-r2", {"r2": 1}))
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "from-r1")

    def test_three_candidates_pick_the_minimum(self) -> None:
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r9", operation("o9", "k", "v9", {"r9": 1}))
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r9": 1})
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")


class AutoResolveValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict()
        valid = self.good_auto_request()
        bad_bodies = [
            b"{oops",
            [],
            {},
            dict(valid, extra=1),
            dict(valid, value="v1"),
            dict(valid, candidates=[]),
            dict(valid, replicaId=""),
            dict(valid, operationId=""),
            dict(valid, clock={"r2": 2}),  # missing the resolving replica
            dict(valid, clock={"r1": 1, "r2": 1, "r3": -1}),
            dict(valid, policy=""),
            dict(valid, policy="lowest-value"),
            dict(valid, policy="HIGHEST_IDENTITY"),
        ]
        for body in bad_bodies:
            status, payload = self.post_auto("k", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_highest_policy_invalid_bodies_are_400(self) -> None:
        self.seed_conflict()
        valid = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")
        for body in (
            dict(valid, policy=""),
            dict(valid, policy="highest"),
            dict(valid, clock={"r2": 2}),
        ):
            status, payload = self.post_auto("k", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))

    def test_clock_not_dominating_candidates_is_400(self) -> None:
        self.seed_conflict()
        # Concurrent with the r2 candidate.
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r3": 1})
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # Strictly weaker than one candidate is rejected too.
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 0, "r3": 1})
        )
        self.assertEqual(status, 400)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_highest_policy_clock_not_dominating_candidates_is_400(self) -> None:
        self.seed_conflict()
        status, payload = self.post_auto(
            "k",
            auto_request("r3", "fix-1", {"r1": 1, "r3": 1}, "highest_identity"),
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)

    def test_new_concurrent_candidate_undominated_is_400(self) -> None:
        self.seed_conflict()
        self.post_operation("r4", operation("o4", "k", "v4", {"r4": 1}))
        # The clock that dominated the original two no longer dominates
        # every current candidate: invalid, and nothing changes.
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)

    def test_query_string_does_not_change_handling(self) -> None:
        # The route ignores no and rejects unknown query parameters alike by
        # simply not consulting them; an unknown parameter still routes to
        # the endpoint rather than 404.
        self.seed_conflict()
        status, payload = self.request(
            "POST", "/v1/states/k/resolve/auto?x=1", self.good_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")


class AutoResolveConflictTests(HttpServerTestCase):
    def test_missing_key_is_409(self) -> None:
        # Even a dominating clock is a resolution conflict when there is no
        # candidate set at all.
        status, payload = self.post_auto(
            "absent", auto_request("r3", "fix-1", {"r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_unconflicted_key_is_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_not_a_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_conflict_disappearing_before_commit_is_409(self) -> None:
        self.seed_conflict()
        # A dominating write collapses the candidate set to one agreed
        # value before the auto request arrives: there is nothing left to
        # resolve.
        status, _ = self.post_operation(
            "r4", operation("o4", "k", "v4", {"r1": 1, "r2": 1, "r4": 1})
        )
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v4")


class AutoResolveIdentityTests(HttpServerTestCase):
    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict()
        body = self.good_auto_request()
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
                "value": "v1",
                "policy": "lowest_identity",
            },
        )
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )

    def test_replay_after_key_moved_on_still_reports_original_value(self) -> None:
        self.seed_conflict()
        body = self.good_auto_request()
        self.assertEqual(self.post_auto("k", body)[0], 201)
        # The key moves on: a new candidate concurrent with the repair's
        # clock creates a fresh conflict.
        self.post_operation("r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0}))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        # Replaying the old repair identity is answered from the committed
        # operation: 200 with the original chosen value, no new log record.
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v1")
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1", "o3"],
        )

    def test_same_identity_different_content_is_409_operation_conflict(self) -> None:
        self.seed_conflict()
        body = self.good_auto_request()
        self.assertEqual(self.post_auto("k", body)[0], 201)
        tampered = dict(body, clock={"r1": 2, "r2": 1, "r3": 1})
        status, payload = self.post_auto("k", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "v1")

    def test_same_identity_on_another_key_is_409(self) -> None:
        self.seed_conflict("k")
        # A second conflict using distinct operation identities.
        self.assertEqual(
            self.post_operation("r1", operation("o10", "other", "w1", {"r1": 10}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("o20", "other", "w2", {"r2": 20}))[0],
            201,
        )
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        # The identity is bound to key k with value v1; reusing it for a
        # different key is different content.
        status, payload = self.post_auto("other", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_identity_is_shared_with_plain_writes_and_manual_resolves(self) -> None:
        self.post_operation("r3", operation("fix-1", "k", "v", {"r3": 1}))
        self.seed_conflict()
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class AutoResolvePolicyBindingTests(HttpServerTestCase):
    """The identity binds key, clock, and policy for automatic resolutions."""

    def highest_request(self, operation_id: str = "fix-1") -> dict:
        return auto_request("r3", operation_id, {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")

    def test_highest_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict()
        body = self.highest_request()
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
                "value": "v2",
                "policy": "highest_identity",
            },
        )
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )

    def test_highest_replay_after_key_moved_on_reports_original_value(self) -> None:
        self.seed_conflict()
        body = self.highest_request()
        self.assertEqual(self.post_auto("k", body)[0], 201)
        self.post_operation("r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0}))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v2")
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1", "o3"],
        )

    def test_different_policy_under_same_identity_is_409_even_when_value_agrees(self) -> None:
        # The minimum and maximum identities carry the same value, so both
        # policies would derive the identical value; the policy mismatch is
        # still an operation conflict.
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "other", {"r2": 1}))
        self.post_operation("r4", operation("o4", "k", "same", {"r4": 1}))
        clock = {"r1": 1, "r2": 1, "r3": 1, "r4": 1}
        lowest = auto_request("r3", "fix-1", clock, "lowest_identity")
        highest = auto_request("r3", "fix-1", clock, "highest_identity")
        self.assertEqual(self.post_auto("k", lowest)[0], 201)
        status, payload = self.post_auto("k", highest)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "same")

    def test_highest_then_lowest_under_same_identity_is_409(self) -> None:
        # Reverse ordering on a second, independent conflict: highest
        # commits first, then lowest under the same identity conflicts.
        self.post_operation("r1", operation("o10", "other", "w1", {"r1": 10}))
        self.post_operation("r2", operation("o20", "other", "w2", {"r2": 20}))
        clock = {"r1": 10, "r2": 20, "r6": 1}
        highest = auto_request("r6", "fix-2", clock, "highest_identity")
        lowest = auto_request("r6", "fix-2", clock, "lowest_identity")
        self.assertEqual(self.post_auto("other", highest)[0], 201)
        status, payload = self.post_auto("other", lowest)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("other")
        self.assertEqual(state["value"], "w2")

    def test_highest_same_identity_different_clock_is_409(self) -> None:
        self.seed_conflict()
        body = self.highest_request()
        self.assertEqual(self.post_auto("k", body)[0], 201)
        tampered = dict(body, clock={"r1": 2, "r2": 1, "r3": 1})
        status, payload = self.post_auto("k", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "v2")

    def test_highest_same_identity_on_another_key_is_409(self) -> None:
        self.seed_conflict("k")
        self.assertEqual(
            self.post_operation("r1", operation("o10", "other", "w1", {"r1": 10}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("o20", "other", "w2", {"r2": 20}))[0],
            201,
        )
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        status, payload = self.post_auto("other", self.highest_request())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_policy_switch_after_lowest_commit_is_409(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        # The key moves on into a fresh conflict where the highest policy
        # would now be valid; reusing the bound identity with a different
        # policy still conflicts rather than re-resolving.
        self.post_operation("r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0}))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        switched = auto_request(
            "r3", "fix-1", {"r1": 1, "r2": 2, "r3": 1}, "highest_identity"
        )
        status, payload = self.post_auto("k", switched)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class AutoResolveIntegrationTests(HttpServerTestCase):
    def test_resolution_is_exported_with_chosen_value(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        self.post_operation("r4", operation("after", "k", "v4", {"r4": 1}))
        _, page = self.get_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1"), ("r4", "after")],
        )
        self.assertEqual(
            page["operations"][2],
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "v1",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                },
            },
        )

    def test_audit_stream_carries_the_auto_resolution(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        _, page = self.get_audit("k")
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )
        self.assertEqual(page["operations"][2]["operation"]["value"], "v1")

    def test_metrics_count_the_repair_and_the_resolved_key(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r9", operation("o9", "other", "x", {"r9": 1}))
        status, before = self.get_metrics()
        self.assertEqual(status, 200)
        self.assertEqual(before["acceptedOperations"], 3)
        self.assertEqual(before["conflictKeys"], 1)
        self.assertEqual(before["resolvedKeys"], 1)
        self.assertEqual(before["replicas"], 3)
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        _, after = self.get_metrics()
        self.assertEqual(
            after,
            {
                "acceptedOperations": 4,
                "keys": 2,
                "candidateVersions": 2,
                "conflictKeys": 0,
                "resolvedKeys": 2,
                "replicas": 4,
            },
        )
        # A replay changes none of the counters.
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 200)
        _, replay = self.get_metrics()
        self.assertEqual(replay, after)

    def test_imported_auto_resolution_resolves_the_same_conflict(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        _, page = self.get_sync()

        other = StateStore()
        status, accepted, _ = other.import_operations(
            [(e["replicaId"], e["operation"]) for e in page["operations"]]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 3)
        status, state = other.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "v1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_http_import_then_replay_is_200(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto_request())[0], 201)
        _, page = self.get_sync()

        self.server.store = type(self.server.store)()
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": page["operations"]})
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.good_auto_request())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "v1")


class HighestIdentityIntegrationTests(HttpServerTestCase):
    """The new policy participates in sync, audit, metrics, and digests."""

    def highest_request(self, operation_id: str = "fix-1") -> dict:
        return auto_request("r3", operation_id, {"r1": 1, "r2": 1, "r3": 1}, "highest_identity")

    def test_highest_resolution_is_exported_with_chosen_value(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            page["operations"][2],
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "fix-1",
                    "key": "k",
                    "value": "v2",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                },
            },
        )

    def test_highest_resolution_appears_in_audit_stream(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, page = self.get_audit("k")
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )
        self.assertEqual(page["operations"][2]["operation"]["value"], "v2")

    def test_highest_resolution_counts_in_metrics(self) -> None:
        self.seed_conflict("k")
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, after = self.get_metrics()
        self.assertEqual(after["acceptedOperations"], 3)
        self.assertEqual(after["conflictKeys"], 0)
        self.assertEqual(after["resolvedKeys"], 1)
        self.assertEqual(after["replicas"], 3)
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 200)
        _, replay = self.get_metrics()
        self.assertEqual(replay, after)

    def test_imported_highest_resolution_resolves_the_same_conflict(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, page = self.get_sync()

        other = StateStore()
        status, accepted, _ = other.import_operations(
            [(e["replicaId"], e["operation"]) for e in page["operations"]]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 3)
        status, state = other.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v2")

    def test_http_import_then_highest_replay_is_200(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, page = self.get_sync()

        self.server.store = type(self.server.store)()
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": page["operations"]})
        self.assertEqual(status, 201)
        status, payload = self.post_auto("k", self.highest_request())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "v2")
        self.assertEqual(payload["policy"], "highest_identity")

    def test_highest_resolution_changes_the_verification_digest(self) -> None:
        self.seed_conflict()
        _, before = self.request("GET", "/v1/verification/digest")
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, after = self.request("GET", "/v1/verification/digest")
        self.assertNotEqual(before["digest"], after["digest"])
        self.assertEqual(after["keys"], 1)
        self.assertEqual(after["candidateVersions"], 1)

    def test_highest_resolution_appears_in_audit_digest(self) -> None:
        import hashlib

        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 201)
        _, digest = self.get_audit_digest("k")
        self.assertEqual(digest["operations"], 3)
        _, page = self.get_audit("k")
        records = [(e["replicaId"], e["operation"]) for e in page["operations"]]
        expected = hashlib.sha256(
            server_module._key_audit_digest_input(records)
        ).hexdigest()
        self.assertEqual(digest["digest"], expected)
        # A replay changes neither count nor digest.
        self.assertEqual(self.post_auto("k", self.highest_request())[0], 200)
        _, replayed_digest = self.get_audit_digest("k")
        self.assertEqual(replayed_digest, digest)


class AutoResolveConcurrencyTests(HttpServerTestCase):
    """Competing automatic resolutions serialize on the single commit lock."""

    def highest_request(self, operation_id: str, replica: str = "r3") -> dict:
        return auto_request(
            replica, operation_id, {"r1": 1, "r2": 1, replica: 1}, "highest_identity"
        )

    def lowest_request(self, operation_id: str, replica: str = "r3") -> dict:
        return auto_request(
            replica, operation_id, {"r1": 1, "r2": 1, replica: 1}, "lowest_identity"
        )

    def test_only_one_of_two_competing_repairs_commits(self) -> None:
        self.seed_conflict()
        results: list[tuple[int, str]] = []

        def run(body: dict) -> None:
            status, payload = self.post_auto("k", body)
            results.append((status, payload.get("error") or payload.get("value")))

        first = threading.Thread(target=run, args=(self.lowest_request("fix-a", "r3"),))
        second = threading.Thread(target=run, args=(self.highest_request("fix-b", "r5"),))
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)
        self.assertEqual(len(results), 2)
        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses, [201, 409])
        # The losing request sees the key already resolved: a resolution
        # conflict (distinct identities, never an operation conflict).
        winner = next(value for status, value in results if status == 201)
        loser_error = next(value for status, value in results if status == 409)
        self.assertIn(winner, ("v1", "v2"))
        self.assertEqual(loser_error, "resolution_conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 3)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], winner)

    def test_parallel_replays_of_one_identity_all_return_200(self) -> None:
        self.seed_conflict()
        body = self.highest_request("fix-1")
        self.assertEqual(self.post_auto("k", body)[0], 201)
        results: list[int] = []

        def replay() -> None:
            status, _ = self.post_auto("k", body)
            results.append(status)

        threads = [threading.Thread(target=replay) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(results, [200] * 6)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 3)


class AutoResolveAuthTests(unittest.TestCase):
    """The new POST route keeps the same Content-Length/auth priorities."""

    def test_unauthorized_request_is_401(self) -> None:
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
            "/v1/states/k/resolve/auto",
            body=json.dumps(auto_request("r3", "fix-1", {"r3": 1})),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()


class PersistentAutoResolveTestCase(unittest.TestCase):
    """Automatic resolution against a data-file-backed server with real HTTP."""

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

    def seed_conflict(self, server: SemanticStateServer) -> None:
        for replica, op in (
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
        ):
            status, _ = self.request(
                server, "POST", f"/v1/replicas/{replica}/operations", op
            )
            self.assertEqual(status, 201)

    def good_auto_request(self) -> dict:
        return auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})

    def highest_auto_request(self) -> dict:
        return auto_request(
            "r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity"
        )

    def test_auto_resolution_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"], o["value"]) for r, o in records],
            [("r1", "o1", "v1"), ("r2", "o2", "v2"), ("r3", "fix-1", "v1")],
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")
        # Replay after restart is still a 200 and appends nothing.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "v1")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # A tampered known identity still conflicts after restart.
        tampered = dict(self.good_auto_request(), clock={"r1": 2, "r2": 1, "r3": 1})
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", tampered
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # Memory, identity index, and file are exactly as before.
        self.assertEqual(self.data_file.read_bytes(), before)
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "conflict")
        status, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual([e["operation"]["operationId"] for e in page["operations"]], ["o1", "o2"])
        # The failed resolution commits cleanly once persistence works again.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_highest_auto_resolution_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.highest_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v2")
        records, _, bindings = load_data_file_full(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"], o["value"]) for r, o in records],
            [("r1", "o1", "v1"), ("r2", "o2", "v2"), ("r3", "fix-1", "v2")],
        )
        # The identity binding (key, clock, policy) is in the same file.
        self.assertEqual(
            bindings,
            {
                ("r3", "fix-1"): {
                    "key": "k",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "policy": "highest_identity",
                }
            },
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v2")
        # Same-binding replay after restart is 200 and appends nothing.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.highest_auto_request()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "v2")
        self.assertEqual(payload["policy"], "highest_identity")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # A different policy under the recovered identity conflicts.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # A different clock under the recovered identity conflicts too.
        tampered = dict(self.highest_auto_request(), clock={"r1": 2, "r2": 1, "r3": 1})
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", tampered
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_highest_persistence_failure_is_500_and_rolls_back_everything(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server, "POST", "/v1/states/k/resolve/auto", self.highest_auto_request()
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # Memory, identity index, identity binding, and file are unchanged.
        self.assertEqual(self.data_file.read_bytes(), before)
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "conflict")
        # A same-identity retry is not a replay (the binding rolled back) and
        # commits cleanly once persistence works again.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.highest_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v2")
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v2")
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_old_version1_file_without_bindings_still_recovers(self) -> None:
        # A file written before the bindings section existed: two
        # conflicting operations and no autoResolutions key.
        server = self.start_server()
        self.seed_conflict(server)
        server.shutdown()
        server.server_close()
        document = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.data_file.write_text(
            json.dumps({"version": 1, "operations": document["operations"]}),
            encoding="utf-8",
        )

        server = self.start_server()
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")
        # The recovered conflict resolves normally, and the commit writes
        # the supplemented format including the new binding.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto_request()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        _, _, bindings = load_data_file_full(str(self.data_file))
        self.assertEqual(bindings[("r3", "fix-1")]["policy"], "lowest_identity")

    def test_corrupt_binding_section_refuses_start(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        self.assertEqual(
            self.request(
                server, "POST", "/v1/states/k/resolve/auto", self.highest_auto_request()
            )[0],
            201,
        )
        server.shutdown()
        server.server_close()

        good = json.loads(self.data_file.read_text(encoding="utf-8"))
        corrupt_documents = []
        # Wrong section type.
        wrong_type = json.loads(json.dumps(good))
        wrong_type["autoResolutions"] = {}
        corrupt_documents.append(wrong_type)
        # Unknown policy.
        bad_policy = json.loads(json.dumps(good))
        bad_policy["autoResolutions"][0]["policy"] = "newest_clock"
        corrupt_documents.append(bad_policy)
        # Missing field.
        missing_field = json.loads(json.dumps(good))
        del missing_field["autoResolutions"][0]["clock"]
        corrupt_documents.append(missing_field)
        # Binding disagrees with its operation's clock.
        mismatched_clock = json.loads(json.dumps(good))
        mismatched_clock["autoResolutions"][0]["clock"] = {"r1": 9, "r2": 9, "r3": 9}
        corrupt_documents.append(mismatched_clock)
        # Binding names an unknown identity.
        unknown_identity = json.loads(json.dumps(good))
        unknown_identity["autoResolutions"][0]["operationId"] = "ghost"
        corrupt_documents.append(unknown_identity)
        # Duplicate binding identity.
        duplicate = json.loads(json.dumps(good))
        duplicate["autoResolutions"].append(json.loads(json.dumps(duplicate["autoResolutions"][0])))
        corrupt_documents.append(duplicate)
        for document in corrupt_documents:
            self.data_file.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(
                server_module.PersistenceError, msg=json.dumps(document)[:120]
            ):
                StateStore(data_file=str(self.data_file))



if __name__ == "__main__":
    unittest.main()
