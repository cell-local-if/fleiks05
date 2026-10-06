"""Tests for the conditional-write endpoint.

`POST /v1/replicas/{replicaId}/operations/conditional` commits an ordinary
write only when the key's current candidate identity set equals the
request's ``expectedCandidates``; the expectation constrains only the
first commit, never a replay of an already accepted identity.
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
    parse_conditional_operation_payload,
)


def conditional_body(
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    expected: list,
) -> dict:
    return {
        "operationId": operation_id,
        "key": key,
        "value": value,
        "clock": clock,
        "expectedCandidates": expected,
    }


class ParseConditionalOperationPayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        operation, expected = parse_conditional_operation_payload(
            json.dumps(
                conditional_body(
                    "op-1",
                    "color",
                    "blue",
                    {"r1": 1},
                    [{"replicaId": "r2", "operationId": "op-0"}],
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
                "clock": {"r1": 1},
            },
        )
        self.assertEqual(expected, [{"replicaId": "r2", "operationId": "op-0"}])

    def test_empty_expected_set_is_allowed(self) -> None:
        _, expected = parse_conditional_operation_payload(
            conditional_body("op-1", "k", "v", {"r1": 1}, []), "r1"
        )
        self.assertEqual(expected, [])

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(b"{not json", "r1")

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload("[1, 2]", "r1")

    def test_rejects_missing_expected_candidates(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                {"operationId": "o", "key": "k", "value": "v", "clock": {"r1": 1}},
                "r1",
            )

    def test_rejects_unknown_field(self) -> None:
        body = conditional_body("o", "k", "v", {"r1": 1}, [])
        body["replicaId"] = "r1"
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(body, "r1")

    def test_rejects_non_list_expected_candidates(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body("o", "k", "v", {"r1": 1}, {"replicaId": "r1"}),
                "r1",
            )

    def test_rejects_expected_candidate_with_missing_field(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body("o", "k", "v", {"r1": 1}, [{"replicaId": "r1"}]),
                "r1",
            )

    def test_rejects_expected_candidate_with_extra_field(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body(
                    "o",
                    "k",
                    "v",
                    {"r1": 1},
                    [{"replicaId": "r1", "operationId": "a", "value": "x"}],
                ),
                "r1",
            )

    def test_rejects_empty_identity_components(self) -> None:
        for entry in (
            {"replicaId": "", "operationId": "a"},
            {"replicaId": "r1", "operationId": ""},
            {"replicaId": 1, "operationId": "a"},
        ):
            with self.assertRaises(ValueError):
                parse_conditional_operation_payload(
                    conditional_body("o", "k", "v", {"r1": 1}, [entry]), "r1"
                )

    def test_rejects_duplicate_identities(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body(
                    "o",
                    "k",
                    "v",
                    {"r1": 1},
                    [
                        {"replicaId": "r1", "operationId": "a"},
                        {"replicaId": "r1", "operationId": "a"},
                    ],
                ),
                "r1",
            )

    def test_ordinary_write_constraints_still_apply(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body("", "k", "v", {"r1": 1}, []), "r1"
            )
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_body("o", "k", "v", {"r2": 1}, []), "r1"
            )


class ConditionalOperationServerTests(unittest.TestCase):
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
            conditional_body(operation_id, key, value, clock, expected),
        )

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            {"operationId": operation_id, "key": key, "value": value, "clock": clock},
        )

    def test_empty_expectation_commits_on_missing_key(self) -> None:
        status, payload = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {"status": "created", "replicaId": "r1", "operationId": "op-1", "key": "color"},
        )
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        self.assertEqual(state["status"], "resolved")

    def test_empty_expectation_conflicts_when_key_has_candidates(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, payload = self.post_conditional("r2", "op-2", "color", "red", {"r2": 1}, [])
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        # The rejected write added no version.
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        self.assertEqual(state["status"], "resolved")

    def test_matching_expectation_commits(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        status, _ = self.post_conditional(
            "r2",
            "op-2",
            "color",
            "red",
            {"r2": 1},
            [{"replicaId": "r1", "operationId": "op-1"}],
        )
        self.assertEqual(status, 201)
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in state["candidates"]],
            [("r1", "op-1"), ("r2", "op-2")],
        )

    def test_expectation_must_match_exactly(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        # A subset of the current identities is not a match.
        status, payload = self.post_conditional(
            "r2",
            "op-2",
            "color",
            "red",
            {"r2": 1},
            [
                {"replicaId": "r1", "operationId": "op-1"},
                {"replicaId": "r1", "operationId": "op-0"},
            ],
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})

    def test_expectation_matches_regardless_of_order(self) -> None:
        self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.post_operation("r2", "op-2", "color", "red", {"r2": 1})
        status, _ = self.post_conditional(
            "r3",
            "op-3",
            "color",
            "green",
            {"r3": 1},
            [
                {"replicaId": "r2", "operationId": "op-2"},
                {"replicaId": "r1", "operationId": "op-1"},
            ],
        )
        self.assertEqual(status, 201)

    def test_replay_ignores_expected_candidates(self) -> None:
        status, _ = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 201)
        # The candidate set moved on, but a same-content replay of the
        # accepted identity is still the original idempotent result.
        self.post_operation("r2", "op-2", "color", "red", {"r2": 1})
        status, payload = self.post_conditional(
            "r1",
            "op-1",
            "color",
            "blue",
            {"r1": 1},
            [{"replicaId": "r9", "operationId": "other"}],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        # No new version was added by the replay.
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(len(state["candidates"]), 2)

    def test_known_identity_with_different_content_is_operation_conflict(self) -> None:
        status, _ = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 201)
        status, payload = self.post_conditional(
            "r1", "op-1", "color", "red", {"r1": 1}, []
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_identity_committed_by_plain_write_replays_here(self) -> None:
        # The identity binding is key+value+clock, shared with the ordinary
        # write entry point: a conditional retry of a plain write is a
        # replay, whatever it expects.
        status, _ = self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.assertEqual(status, 201)
        status, _ = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 200)
        status, payload = self.post_conditional(
            "r1", "op-1", "color", "red", {"r1": 1}, []
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_conditional_commit_is_visible_to_plain_replay(self) -> None:
        status, _ = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 201)
        status, _ = self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.assertEqual(status, 200)

    def test_conditional_write_enters_sync_stream(self) -> None:
        self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
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
        status, payload = self.post_conditional("r2", "op-2", "color", "red", {"r2": 1}, [])
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        status, after = self.request("GET", "/v1/sync/operations?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)
        # The conflicting identity stays unknown: the same write with the
        # right expectation is still a first commit, not a replay.
        status, _ = self.post_conditional(
            "r2",
            "op-2",
            "color",
            "red",
            {"r2": 1},
            [{"replicaId": "r1", "operationId": "op-1"}],
        )
        self.assertEqual(status, 201)

    def test_invalid_bodies_are_400(self) -> None:
        base = conditional_body("op-1", "k", "v", {"r1": 1}, [])
        bad_bodies = [
            b"{not json",
            b"[1, 2]",
            {k: v for k, v in base.items() if k != "expectedCandidates"},
            {**base, "extra": 1},
            {**base, "expectedCandidates": {}},
            {**base, "expectedCandidates": [{"replicaId": "r1"}]},
            {**base, "expectedCandidates": [{"replicaId": "r1", "operationId": "a", "x": 1}]},
            {**base, "expectedCandidates": [{"replicaId": "", "operationId": "a"}]},
            {**base, "expectedCandidates": [{"replicaId": "r1", "operationId": ""}]},
            {
                **base,
                "expectedCandidates": [
                    {"replicaId": "r1", "operationId": "a"},
                    {"replicaId": "r1", "operationId": "a"},
                ],
            },
            {**base, "clock": {"r2": 1}},
            {**base, "value": ""},
        ]
        for body in bad_bodies:
            status, payload = self.request(
                "POST", "/v1/replicas/r1/operations/conditional", body
            )
            self.assertEqual(status, 400, body)
            self.assertEqual(payload, {"error": "invalid_request"})
        # None of the rejected requests committed anything.
        status, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_route_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replicas/r1/operations/conditional/extra",
            "/v1/replicas/r1/operations/conditional/",
            "/v1/replicas/r1/conditional",
        ):
            status, payload = self.request("POST", path, {"operationId": "o"})
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"})


class ConditionalOperationPersistenceTests(unittest.TestCase):
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
            "/v1/replicas/r1/operations/conditional",
            conditional_body("op-1", "color", "blue", {"r1": 1}, []),
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
            "/v1/replicas/r1/operations/conditional",
            conditional_body("op-1", "color", "blue", {"r1": 1}, []),
        )
        self.assertEqual(status, 200)

    def test_state_conflict_and_replay_do_not_touch_the_file(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/conditional",
            conditional_body("op-1", "color", "blue", {"r1": 1}, []),
        )
        self.assertEqual(status, 201)
        with open(self.data_file, "rb") as handle:
            committed = handle.read()

        # A state conflict changes nothing on disk.
        status, payload = self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations/conditional",
            conditional_body("op-2", "color", "red", {"r2": 1}, []),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), committed)

        # Neither does a replay.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/conditional",
            conditional_body("op-1", "color", "blue", {"r1": 1}, []),
        )
        self.assertEqual(status, 200)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), committed)

    def test_persistence_failure_is_503_retryable_and_changes_nothing(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations/conditional",
            conditional_body("op-1", "color", "blue", {"r1": 1}, []),
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
                "/v1/replicas/r2/operations/conditional",
                conditional_body(
                    "op-2",
                    "color",
                    "red",
                    {"r2": 1},
                    [{"replicaId": "r1", "operationId": "op-1"}],
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
                "/v1/replicas/r1/operations/conditional",
                conditional_body("op-1", "color", "blue", {"r1": 1}, []),
            )
            self.assertEqual(status, 200)
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations/conditional",
                conditional_body("op-1", "color", "tampered", {"r1": 1}, []),
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
            "/v1/replicas/r2/operations/conditional",
            conditional_body(
                "op-2",
                "color",
                "red",
                {"r2": 1},
                [{"replicaId": "r1", "operationId": "op-1"}],
            ),
        )
        self.assertEqual(status, 201)


if __name__ == "__main__":
    unittest.main()
