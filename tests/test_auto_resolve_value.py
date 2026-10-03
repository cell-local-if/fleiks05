"""HTTP and persistence tests for value-based automatic resolution policies.

The automatic-resolution endpoints

    POST /v1/states/{key}/resolve/auto
    POST /v1/resolve/auto/batch
    POST /v1/resolve/auto/plan

accept, alongside ``lowest_identity``/``highest_identity``, the policies
``lowest_value`` (the smallest candidate value in Unicode code point order)
and ``highest_value`` (the largest). Everything here goes through the real
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
    RequestHandler,
    SemanticStateServer,
    parse_auto_resolve_batch,
    parse_auto_resolve_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def auto_request(replica: str, operation_id: str, clock: dict, policy: str) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


class ParseValuePolicyTests(unittest.TestCase):
    def test_single_payload_accepts_value_policies(self) -> None:
        for policy in ("lowest_value", "highest_value"):
            payload = parse_auto_resolve_payload(
                json.dumps(auto_request("r3", "fix-1", {"r3": 1}, policy))
            )
            self.assertEqual(payload["policy"], policy)

    def test_batch_accepts_value_policies(self) -> None:
        entries = parse_auto_resolve_batch(
            json.dumps(
                {
                    "resolutions": [
                        dict(
                            auto_request("r3", "fix-1", {"r3": 1}, "lowest_value"),
                            key="a",
                        ),
                        dict(
                            auto_request("r3", "fix-2", {"r3": 1}, "highest_value"),
                            key="b",
                        ),
                    ]
                }
            )
        )
        self.assertEqual([e["policy"] for e in entries], ["lowest_value", "highest_value"])

    def test_unknown_and_non_string_policies_are_rejected(self) -> None:
        valid = auto_request("r3", "fix-1", {"r3": 1}, "lowest_value")
        for bad in ("lowest_Value", "value", "lowest", "", 42, None, True, ["lowest_value"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_auto_resolve_payload(dict(valid, policy=bad))
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_auto_resolve_batch(
                    {"resolutions": [dict(valid, policy=bad, key="k")]}
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

    def seed_conflict(self, key: str, values: tuple[str, str] = ("v1", "v2")) -> None:
        """Two concurrent writes with different values on ``key``."""
        self.assertEqual(
            self.post_operation("r1", operation(f"o1-{key}", key, values[0], {"r1": 1}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation(f"o2-{key}", key, values[1], {"r2": 1}))[0],
            201,
        )


class SingleValuePolicyTests(HttpServerTestCase):
    def test_lowest_value_selects_smallest_string(self) -> None:
        # Code point order, not case-insensitive or identity order: "Zulu"
        # (U+005A) sorts before "apple" (U+0061) even though the "apple"
        # candidate carries the smaller identity.
        self.seed_conflict("k", values=("apple", "Zulu"))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value")
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "key": "k",
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "Zulu",
                "policy": "lowest_value",
            },
        )
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "Zulu")

    def test_highest_value_selects_largest_string(self) -> None:
        self.seed_conflict("k", values=("apple", "Zulu"))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "highest_value")
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "apple")
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "apple")

    def test_comparison_is_by_unicode_code_point(self) -> None:
        # "z" (U+007A) sorts before "é" (U+00E9); identity order disagrees
        # with value order here.
        self.seed_conflict("k", values=("éclair", "zest"))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, "lowest_value")
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "zest")
        status, payload = self.post_auto(
            "k2", auto_request("r3", "fix-2", {"r3": 1}, "highest_value")
        )
        self.assertEqual(status, 409)  # k2 was never written
        self.seed_conflict("k2", values=("éclair", "zest"))
        status, payload = self.post_auto(
            "k2", auto_request("r3", "fix-2", {"r1": 1, "r2": 1, "r3": 1}, "highest_value")
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "éclair")

    def test_tied_extreme_value_still_resolves(self) -> None:
        # Two candidates share the extreme value; the extreme is unambiguous
        # as a value, so the resolution commits and reports it.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r3", operation("o3", "k", "other", {"r3": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k",
            auto_request("r4", "fix-1", {"r1": 1, "r2": 1, "r3": 1, "r4": 1}, "lowest_value"),
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "other")
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "other")

    def test_replay_reports_original_value_after_candidates_move(self) -> None:
        self.seed_conflict("k")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        status, created = self.post_auto("k", auto_request("r3", "fix-1", clock, "lowest_value"))
        self.assertEqual(status, 201)
        self.assertEqual(created["value"], "v1")
        # Move the candidate set with a newer concurrent write.
        self.assertEqual(
            self.post_operation("r4", operation("o9", "k", "v9", {"r4": 5}))[0], 201
        )
        status, replay = self.post_auto("k", auto_request("r3", "fix-1", clock, "lowest_value"))
        self.assertEqual(status, 200)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["value"], "v1")

    def test_same_identity_different_policy_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        self.assertEqual(
            self.post_auto("k", auto_request("r3", "fix-1", clock, "lowest_value"))[0], 201
        )
        for other in ("highest_value", "lowest_identity", "highest_identity"):
            status, payload = self.post_auto(
                "k", auto_request("r3", "fix-1", clock, other)
            )
            self.assertEqual(status, 409, other)
            self.assertEqual(payload, {"error": "operation_conflict"})

    def test_identity_without_policy_binding_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        # A plain write under the identity carries no policy binding.
        self.assertEqual(
            self.post_operation("r3", operation("fix-1", "other", "x", {"r3": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k",
            auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 2}, "lowest_value"),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_resolution_conflict_preconditions(self) -> None:
        clock = {"r1": 1, "r2": 1, "r3": 1}
        # Missing key.
        status, payload = self.post_auto(
            "missing", auto_request("r3", "fix-1", clock, "lowest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        # Candidates agree on the value.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-2", clock, "highest_value")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_non_dominating_clock_is_invalid_request(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix-1", {"r3": 1}, "lowest_value")
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_unknown_policy_is_invalid_request(self) -> None:
        self.seed_conflict("k")
        for bad in ("lowest", "Lowest_Value", "value", 42, None):
            status, payload = self.post_auto(
                "k",
                auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1}, bad),
            )
            self.assertEqual(status, 400, repr(bad))
            self.assertEqual(payload, {"error": "invalid_request"})


class BatchValuePolicyTests(HttpServerTestCase):
    def test_mixed_policies_in_one_batch(self) -> None:
        self.seed_conflict("a", values=("apple", "Zulu"))
        self.seed_conflict("b", values=("apple", "Zulu"))
        self.seed_conflict("c", values=("v1", "v2"))
        clock = {"r1": 1, "r2": 1, "r3": 1}
        status, payload = self.post_batch(
            [
                dict(auto_request("r3", "fix-a", clock, "lowest_value"), key="a"),
                dict(auto_request("r3", "fix-b", clock, "highest_value"), key="b"),
                dict(auto_request("r3", "fix-c", clock, "lowest_identity"), key="c"),
            ]
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["accepted"], 3)
        self.assertEqual(payload["replayed"], 0)
        values = {entry["key"]: entry["value"] for entry in payload["resolutions"]}
        self.assertEqual(values, {"a": "Zulu", "b": "apple", "c": "v1"})
        for key, value in values.items():
            _, state = self.get_state(key)
            self.assertEqual(state["value"], value)

    def test_batch_is_atomic_on_resolution_conflict(self) -> None:
        self.seed_conflict("a")
        self.seed_conflict("b")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        status, payload = self.post_batch(
            [
                dict(auto_request("r3", "fix-a", clock, "lowest_value"), key="a"),
                # Legal clock, but it does not dominate b's candidates.
                dict(auto_request("r3", "fix-b", {"r3": 1}, "highest_value"), key="b"),
            ]
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        # Nothing committed: both keys are still in conflict.
        for key in ("a", "b"):
            _, state = self.get_state(key)
            self.assertEqual(state["status"], "conflict")

    def test_batch_replay_and_operation_conflict(self) -> None:
        self.seed_conflict("a")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        entries = [dict(auto_request("r3", "fix-a", clock, "highest_value"), key="a")]
        self.assertEqual(self.post_batch(entries)[0], 201)
        status, payload = self.post_batch(entries)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(payload["resolutions"][0]["value"], "v2")
        # Same identity, different policy: the whole batch is rejected.
        status, payload = self.post_batch(
            [dict(auto_request("r3", "fix-a", clock, "lowest_value"), key="a")]
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanValuePolicyTests(HttpServerTestCase):
    def test_plan_reports_value_selection_without_committing(self) -> None:
        self.seed_conflict("k", values=("apple", "Zulu"))
        clock = {"r1": 1, "r2": 1, "r3": 1}
        entries = [dict(auto_request("r3", "fix-1", clock, "lowest_value"), key="k")]
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(payload["resolutions"][0]["value"], "Zulu")
        self.assertEqual(payload["resolutions"][0]["policy"], "lowest_value")
        # Nothing was written and no identity was bound: the key is still in
        # conflict and the same request commits as new.
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(self.post_batch(entries)[0], 201)

    def test_plan_counts_replays(self) -> None:
        self.seed_conflict("k")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        entries = [dict(auto_request("r3", "fix-1", clock, "highest_value"), key="k")]
        self.assertEqual(self.post_batch(entries)[0], 201)
        status, payload = self.post_plan(entries)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(payload["resolutions"][0]["value"], "v2")

    def test_plan_conflict_matches_committing_batch(self) -> None:
        self.seed_conflict("k")
        clock = {"r1": 1, "r2": 1, "r3": 1}
        self.assertEqual(
            self.post_batch(
                [dict(auto_request("r3", "fix-1", clock, "lowest_value"), key="k")]
            )[0],
            201,
        )
        status, payload = self.post_plan(
            [dict(auto_request("r3", "fix-1", clock, "highest_value"), key="k")]
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PersistentValuePolicyTests(unittest.TestCase):
    """Value-policy bindings survive a restart with the operation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
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

    def test_binding_and_selection_survive_restart(self) -> None:
        clock = {"r1": 1, "r2": 1, "r3": 1}
        server = self.start_server()
        for replica, op_id, value, op_clock in (
            ("r1", "o1", "apple", {"r1": 1}),
            ("r2", "o2", "Zulu", {"r2": 1}),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", value, op_clock),
            )
            self.assertEqual(status, 201)
        status, created = self.request(
            server,
            "POST",
            "/v1/states/k/resolve/auto",
            auto_request("r3", "fix-1", clock, "lowest_value"),
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["value"], "Zulu")
        server.shutdown()
        server.server_close()

        # The data file records the new policy binding.
        document = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.assertIn(
            {"replicaId": "r3", "operationId": "fix-1", "policy": "lowest_value"},
            [
                {
                    "replicaId": p["replicaId"],
                    "operationId": p["operationId"],
                    "policy": p["policy"],
                }
                for p in document["policies"]
            ],
        )

        restarted = self.start_server()
        # The selection is still in force.
        status, state = self.request(restarted, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "Zulu")
        # The same binding replays with the original value.
        status, replay = self.request(
            restarted,
            "POST",
            "/v1/states/k/resolve/auto",
            auto_request("r3", "fix-1", clock, "lowest_value"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(replay["value"], "Zulu")
        # A different policy under the same identity still conflicts.
        status, payload = self.request(
            restarted,
            "POST",
            "/v1/states/k/resolve/auto",
            auto_request("r3", "fix-1", clock, "highest_value"),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


if __name__ == "__main__":
    unittest.main()
