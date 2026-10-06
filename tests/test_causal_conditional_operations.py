"""Tests for the causal-conditional write endpoint.

`POST /v1/replicas/{replicaId}/operations/causal-conditional` commits an
ordinary write only when every candidate the key currently holds is
covered by the caller-observed causal boundary ``expectedClock``; the
boundary constrains only the first commit, never a replay of an already
accepted identity.
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    parse_causal_conditional_operation_payload,
)


def causal_conditional_body(
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    expected_clock: dict,
) -> dict:
    return {
        "operationId": operation_id,
        "key": key,
        "value": value,
        "clock": clock,
        "expectedClock": expected_clock,
    }


class ParseCausalConditionalOperationPayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        operation, expected_clock = parse_causal_conditional_operation_payload(
            json.dumps(
                causal_conditional_body(
                    "op-1", "color", "blue", {"r1": 2, "r2": 1}, {"r2": 1}
                )
            ),
            "r1",
        )
        self.assertEqual(
            operation,
            {
                "operationId": "op-1",
                "key": "color",
                "value": "blue",
                "clock": {"r1": 2, "r2": 1},
            },
        )
        self.assertEqual(expected_clock, {"r2": 1})

    def test_empty_expected_clock_is_allowed(self) -> None:
        _, expected_clock = parse_causal_conditional_operation_payload(
            causal_conditional_body("op-1", "k", "v", {"r1": 1}, {}), "r1"
        )
        self.assertEqual(expected_clock, {})

    def test_expected_clock_need_not_contain_replica_id(self) -> None:
        _, expected_clock = parse_causal_conditional_operation_payload(
            causal_conditional_body("op-1", "k", "v", {"r1": 1, "r9": 1}, {"r9": 1}),
            "r1",
        )
        self.assertEqual(expected_clock, {"r9": 1})

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(b"{not json", "r1")

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload("[1, 2]", "r1")

    def test_rejects_missing_expected_clock(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                {"operationId": "o", "key": "k", "value": "v", "clock": {"r1": 1}},
                "r1",
            )

    def test_rejects_unknown_field(self) -> None:
        body = causal_conditional_body("o", "k", "v", {"r1": 1}, {})
        body["replicaId"] = "r1"
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(body, "r1")

    def test_rejects_duplicate_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                '{"operationId":"o","operationId":"o2","key":"k","value":"v",'
                '"clock":{"r1":1},"expectedClock":{}}',
                "r1",
            )
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                '{"operationId":"o","key":"k","value":"v","clock":{"r1":1},'
                '"expectedClock":{"r1":1,"r1":1}}',
                "r1",
            )

    def test_rejects_invalid_expected_clock(self) -> None:
        for expected_clock in (
            None,
            [],
            {"": 1},
            {"r1": -1},
            {"r1": 1.5},
            {"r1": True},
            {"r1": "1"},
        ):
            with self.assertRaises(ValueError):
                parse_causal_conditional_operation_payload(
                    causal_conditional_body("o", "k", "v", {"r1": 2}, expected_clock),
                    "r1",
                )

    def test_ordinary_write_constraints_still_apply(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                causal_conditional_body("", "k", "v", {"r1": 1}, {}), "r1"
            )
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                causal_conditional_body("o", "k", "v", {"r2": 1}, {}), "r1"
            )

    def test_clock_must_strictly_dominate_expected_clock(self) -> None:
        # Equal clocks do not dominate.
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                causal_conditional_body("o", "k", "v", {"r1": 1}, {"r1": 1}), "r1"
            )
        # A smaller component does not dominate.
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                causal_conditional_body("o", "k", "v", {"r1": 1}, {"r1": 2}), "r1"
            )
        # An all-zero operation clock does not dominate the empty boundary.
        with self.assertRaises(ValueError):
            parse_causal_conditional_operation_payload(
                causal_conditional_body("o", "k", "v", {"r1": 0}, {}), "r1"
            )
        # Strictly greater on one component, equal on the rest: dominates.
        operation, _ = parse_causal_conditional_operation_payload(
            causal_conditional_body("o", "k", "v", {"r1": 2, "r2": 1}, {"r1": 1, "r2": 1}),
            "r1",
        )
        self.assertEqual(operation["clock"], {"r1": 2, "r2": 1})


class CausalConditionalOperationServerTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_causal_conditional(
        self,
        replica: str,
        operation_id: str,
        key: str,
        value: str,
        clock: dict,
        expected_clock: dict,
    ) -> tuple[int, dict]:
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations/causal-conditional",
            causal_conditional_body(operation_id, key, value, clock, expected_clock),
        )

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            {"operationId": operation_id, "key": key, "value": value, "clock": clock},
        )

    def post_conditional(
        self,
        replica: str,
        operation_id: str,
        key: str,
        value: str,
        clock: dict,
        expected: list,
    ) -> tuple[int, dict]:
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations/conditional",
            {
                "operationId": operation_id,
                "key": key,
                "value": value,
                "clock": clock,
                "expectedCandidates": expected,
            },
        )

    def test_empty_boundary_commits_on_missing_key(self) -> None:
        status, payload = self.post_causal_conditional(
            "r1", "op-1", "color", "blue", {"r1": 1}, {}
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {"status": "created", "replicaId": "r1", "operationId": "op-1", "key": "color"},
        )
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        self.assertEqual(state["status"], "resolved")

    def test_empty_boundary_conflicts_when_key_has_candidates(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, payload = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1}, {}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        # The rejected write added no version.
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        self.assertEqual(state["status"], "resolved")

    def test_covering_boundary_commits(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, _ = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1, "r1": 1}, {"r1": 1}
        )
        self.assertEqual(status, 201)
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "red")
        self.assertEqual(state["status"], "resolved")

    def test_boundary_may_exceed_the_observed_candidates(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        # The boundary covers the candidate with room to spare.
        status, _ = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1, "r1": 5}, {"r1": 5}
        )
        self.assertEqual(status, 201)

    def test_every_candidate_must_be_covered(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.post_operation("r2", "op-2", "color", "red", {"r2": 1})
        # The boundary covers r1's candidate but not r2's.
        status, payload = self.post_causal_conditional(
            "r3", "op-3", "color", "green", {"r3": 1, "r1": 1}, {"r1": 1}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        # Covering both candidates commits.
        status, _ = self.post_causal_conditional(
            "r3", "op-3", "color", "green", {"r3": 1, "r1": 1, "r2": 1}, {"r1": 1, "r2": 1}
        )
        self.assertEqual(status, 201)

    def test_boundary_coverage_counts_missing_components_as_zero(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1, "r2": 1})
        # The boundary names only r1; the candidate's r2 component is not
        # covered (missing counts as 0).
        status, payload = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 2, "r1": 1}, {"r1": 1}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})

    def test_replay_ignores_expected_clock(self) -> None:
        status, _ = self.post_causal_conditional(
            "r1", "op-1", "color", "blue", {"r1": 1}, {}
        )
        self.assertEqual(status, 201)
        # The candidate set moved on, but a same-content replay of the
        # accepted identity is still the original idempotent result — even
        # with a (valid) boundary that covers none of the current
        # candidates.
        self.post_operation("r2", "op-2", "color", "red", {"r2": 1})
        status, payload = self.post_causal_conditional(
            "r1", "op-1", "color", "blue", {"r1": 1}, {"r1": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        # No new version was added by the replay.
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(len(state["candidates"]), 2)

    def test_known_identity_with_different_content_is_operation_conflict(self) -> None:
        status, _ = self.post_causal_conditional(
            "r1", "op-1", "color", "blue", {"r1": 1}, {}
        )
        self.assertEqual(status, 201)
        status, payload = self.post_causal_conditional(
            "r1", "op-1", "color", "red", {"r1": 1}, {}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_identity_binding_is_shared_with_the_other_write_entries(self) -> None:
        # A plain write replays here, whatever boundary it carries.
        status, _ = self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.assertEqual(status, 201)
        status, _ = self.post_causal_conditional(
            "r1", "op-1", "color", "blue", {"r1": 1}, {}
        )
        self.assertEqual(status, 200)
        status, payload = self.post_causal_conditional(
            "r1", "op-1", "color", "red", {"r1": 1}, {}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # A conditional write replays here too.
        status, _ = self.post_conditional(
            "r2",
            "op-2",
            "color",
            "red",
            {"r2": 1},
            [{"replicaId": "r1", "operationId": "op-1"}],
        )
        self.assertEqual(status, 201)
        status, _ = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1}, {}
        )
        self.assertEqual(status, 200)
        # And a causal-conditional commit is visible to both other entries.
        status, _ = self.post_causal_conditional(
            "r3", "op-3", "color", "green", {"r3": 1, "r1": 1, "r2": 1}, {"r1": 1, "r2": 1}
        )
        self.assertEqual(status, 201)
        status, _ = self.post_operation("r3", "op-3", "color", "green", {"r3": 1, "r1": 1, "r2": 1})
        self.assertEqual(status, 200)
        status, _ = self.post_conditional(
            "r3", "op-3", "color", "green", {"r3": 1, "r1": 1, "r2": 1}, []
        )
        self.assertEqual(status, 200)

    def test_causal_conditional_write_enters_sync_stream(self) -> None:
        self.post_causal_conditional("r1", "op-1", "color", "blue", {"r1": 1}, {})
        status, payload = self.request("GET", "/v1/sync/operations?after=0&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["operations"],
            [
                {
                    "replicaId": "r1",
                    "operation": {
                        "operationId": "op-1",
                        "key": "color",
                        "value": "blue",
                        "clock": {"r1": 1},
                    },
                }
            ],
        )

    def test_state_conflict_leaves_no_trace(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, before = self.request("GET", "/v1/sync/operations?after=0&limit=100")
        self.assertEqual(status, 200)
        status, payload = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1}, {}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        status, after = self.request("GET", "/v1/sync/operations?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)
        # The conflicting identity stays unknown: the same write with a
        # covering boundary is still a first commit, not a replay.
        status, _ = self.post_causal_conditional(
            "r2", "op-2", "color", "red", {"r2": 1, "r1": 1}, {"r1": 1}
        )
        self.assertEqual(status, 201)

    def test_invalid_bodies_are_400(self) -> None:
        base = causal_conditional_body("op-1", "k", "v", {"r1": 1}, {})
        bad_bodies = [
            b"{not json",
            b"[1, 2]",
            {k: v for k, v in base.items() if k != "expectedClock"},
            {**base, "extra": 1},
            {**base, "expectedClock": []},
            {**base, "expectedClock": {"r1": -1}},
            {**base, "expectedClock": {"r1": 1.0}},
            {**base, "expectedClock": {"r1": True}},
            {**base, "expectedClock": {"": 1}},
            {**base, "clock": {"r2": 1}},
            {**base, "clock": {}},
            {**base, "value": ""},
            {**base, "operationId": ""},
            {**base, "key": ""},
            # The operation clock must strictly dominate the boundary.
            {**base, "expectedClock": {"r1": 1}},
            {**base, "expectedClock": {"r1": 2}},
        ]
        for body in bad_bodies:
            status, payload = self.request(
                "POST", "/v1/replicas/r1/operations/causal-conditional", body
            )
            self.assertEqual(status, 400, body)
            self.assertEqual(payload, {"error": "invalid_request"})
        # None of the rejected requests committed anything.
        status, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_duplicate_fields_are_400(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            '{"operationId":"o","operationId":"o2","key":"k","value":"v",'
            '"clock":{"r1":1},"expectedClock":{}}',
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_parameters_are_400(self) -> None:
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x"):
            status, payload = self.request(
                "POST",
                "/v1/replicas/r1/operations/causal-conditional" + query,
                causal_conditional_body("op-1", "k", "v", {"r1": 1}, {}),
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"})
        # The query check precedes the body check.
        status, payload = self.request(
            "POST",
            "/v1/replicas/r1/operations/causal-conditional?x=1",
            b"{not json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replicas/r1/operations/causal-conditional/extra",
            "/v1/replicas/r1/operations/causal-conditional/",
            "/v1/replicas/r1/causal-conditional",
        ):
            status, payload = self.request("POST", path, {"operationId": "o"})
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"})
        # The path check precedes the body check: a shape mismatch is 404
        # even with a body that would be a valid write on the real route.
        status, payload = self.request(
            "POST",
            "/v1/replicas/r1/operations/causal-conditional/extra",
            causal_conditional_body("op-1", "k", "v", {"r1": 1}, {}),
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class CausalConditionalOperationPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = f"{self._tmp.name}/state.json"

    def start_server(self) -> SemanticStateServer:
        from semantic_state_engine.server import StateStore

        server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            store=StateStore(data_file=self.data_file),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(
        self, server: SemanticStateServer, method: str, path: str, body: object = None
    ):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
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
        payload = json.loads(response.read().decode("utf-8"))
        self.last_retry_after = response.getheader("Retry-After")
        conn.close()
        return response.status, payload

    def test_commit_is_persisted_and_recovered(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
        )
        self.assertEqual(status, 201)
        server.shutdown()
        server.server_close()

        # The data file holds exactly the one accepted operation.
        with open(self.data_file, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(
            document["operations"],
            [
                {
                    "replicaId": "r1",
                    "operation": {
                        "operationId": "op-1",
                        "key": "color",
                        "value": "blue",
                        "clock": {"r1": 1},
                    },
                }
            ],
        )

        # A restarted service recovers the commit and its idempotence.
        restarted = self.start_server()
        status, payload = self.request(restarted, "GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "blue")
        status, _ = self.request(
            restarted,
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
        )
        self.assertEqual(status, 200)

    def test_state_conflict_and_replay_do_not_touch_the_file(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
        )
        self.assertEqual(status, 201)
        with open(self.data_file, "rb") as handle:
            committed = handle.read()

        # A state conflict changes nothing on disk.
        status, payload = self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations/causal-conditional",
            causal_conditional_body("op-2", "color", "red", {"r2": 1}, {}),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), committed)

        # Neither does a replay.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
        )
        self.assertEqual(status, 200)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), committed)

    def test_persistence_failure_is_503_retryable_and_changes_nothing(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/causal-conditional",
            causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
        )
        self.assertEqual(status, 201)
        with open(self.data_file, "rb") as handle:
            committed = handle.read()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r2/operations/causal-conditional",
                causal_conditional_body(
                    "op-2", "color", "red", {"r2": 1, "r1": 1}, {"r1": 1}
                ),
            )
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "persistence_unavailable"})
            self.assertEqual(self.last_retry_after, "1")
            # Replays and conflicts need no durable write and keep their
            # usual results under the fault.
            status, _ = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations/causal-conditional",
                causal_conditional_body("op-1", "color", "blue", {"r1": 1}, {}),
            )
            self.assertEqual(status, 200)
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations/causal-conditional",
                causal_conditional_body("op-1", "color", "tampered", {"r1": 1}, {}),
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "operation_conflict"})

        # The failed commit left no trace in memory or on disk.
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), committed)
        status, state = self.request(server, "GET", "/v1/states/color")
        self.assertEqual(state["value"], "blue")
        # The same write commits cleanly once persistence is back.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations/causal-conditional",
            causal_conditional_body(
                "op-2", "color", "red", {"r2": 1, "r1": 1}, {"r1": 1}
            ),
        )
        self.assertEqual(status, 201)


if __name__ == "__main__":
    unittest.main()
