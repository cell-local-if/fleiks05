"""HTTP, sync, and persistence tests for deterministic auto-resolution.

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
    AUTO_RESOLVE_POLICY,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    parse_auto_resolve_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def auto_request(replica: str, operation_id: str, clock: dict) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock,
        "policy": AUTO_RESOLVE_POLICY,
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
            # The server chooses the value; neither it nor a candidate list
            # may be supplied.
            dict(valid, value="v"),
            dict(valid, candidates=[]),
            {k: v for k, v in valid.items() if k != "clock"},
            {k: v for k, v in valid.items() if k != "policy"},
            dict(valid, replicaId=""),
            dict(valid, replicaId=7),
            dict(valid, operationId=""),
            dict(valid, operationId=9),
            dict(valid, policy=""),
            dict(valid, policy="highest_clock"),
            dict(valid, policy=7),
            dict(valid, clock={}),
            dict(valid, clock={"r2": 1}),  # clock must contain the replica
            dict(valid, clock={"r3": -1}),
            dict(valid, clock={"r3": True}),
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_auto_resolve_payload(body)


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

    def good_auto(self, key: str = "k") -> dict:
        return auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})


class AutoResolveHappyPathTests(HttpServerTestCase):
    def test_lowest_identity_wins_and_state_is_resolved(self) -> None:
        self.seed_conflict()
        status, payload = self.post_auto("k", self.good_auto())
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

    def test_selection_compares_replica_before_operation(self) -> None:
        # (r1, o9) loses on the operation id but wins on the replica id.
        self.assertEqual(
            self.post_operation("r1", operation("o9", "k", "aaa", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o1", "k", "bbb", {"r2": 1}))[0], 201
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "aaa")

    def test_selection_uses_operation_id_when_replica_ties(self) -> None:
        # Two concurrent candidates from the same replica (neither clock
        # dominates the other), distinguished only by operation id.
        self.assertEqual(
            self.post_operation(
                "r1", operation("o2", "k", "second", {"r1": 1, "ry": 2})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_operation(
                "r1", operation("o1", "k", "first", {"r1": 1, "rx": 2})
            )[0],
            201,
        )
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix", {"r1": 1, "rx": 2, "ry": 2, "r3": 1})
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "first")

    def test_keys_are_isolated(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r1", operation("o-other", "other", "x", {"r1": 2}))
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
        _, state = self.get_state("other")
        self.assertEqual(
            state, {"key": "other", "value": "x", "clock": {"r1": 2}, "status": "resolved"}
        )


class AutoResolveValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict()
        valid = self.good_auto()
        bad_bodies = [
            b"{oops",
            [],
            {},
            dict(valid, value="v1"),
            dict(valid, candidates=[]),
            dict(valid, replicaId=""),
            dict(valid, operationId=""),
            dict(valid, policy="highest_clock"),
            dict(valid, clock={"r2": 2}),  # missing the resolving replica
            dict(valid, clock={}),
        ]
        for body in bad_bodies:
            status, payload = self.post_auto("k", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_clock_not_dominating_candidates_is_400(self) -> None:
        self.seed_conflict()
        # Concurrent with the r2 candidate: dominates r1 only.
        body = auto_request("r3", "fix-1", {"r1": 1, "r3": 1})
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # A strictly weaker clock is rejected too.
        body = auto_request("r3", "fix-1", {"r1": 1, "r2": 0, "r3": 1})
        status, payload = self.post_auto("k", body)
        self.assertEqual(status, 400)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_clock_dominance_is_checked_before_conflict_precondition(self) -> None:
        # A resolved (single-candidate) key with a clock that cannot dominate
        # it: validation fails before the not-in-conflict precondition.
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = self.post_auto("k", auto_request("r3", "fix", {"r3": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class AutoResolveConflictTests(HttpServerTestCase):
    def test_missing_key_is_409(self) -> None:
        status, payload = self.post_auto("absent", auto_request("r3", "fix", {"r3": 1}))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_with_dominating_clock_is_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix", {"r1": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_not_a_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix", {"r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_concurrent_candidate_changes_the_deterministic_winner(self) -> None:
        # The request names no candidate set, so it is evaluated atomically
        # against whatever conflict is current: a new lower-identity
        # candidate that the clock dominates simply wins the selection.
        self.seed_conflict()
        self.post_operation("r0", operation("o0", "k", "v0", {"r0": 1}))
        status, payload = self.post_auto(
            "k", auto_request("r3", "fix", {"r0": 1, "r1": 1, "r2": 1, "r3": 1})
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v0")

    def test_concurrent_candidate_not_dominated_is_400(self) -> None:
        # Same movement, but the resolution clock does not dominate the
        # freshly arrived candidate: the atomic validation fails.
        self.seed_conflict()
        self.post_operation("r9", operation("o9", "k", "v9", {"r9": 5}))
        status, payload = self.post_auto("k", self.good_auto())
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)


class AutoResolveIdentityTests(HttpServerTestCase):
    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict()
        body = self.good_auto()
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

    def test_same_identity_different_clock_is_409_operation_conflict(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
        tampered = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 2})
        status, payload = self.post_auto("k", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "v1")
        self.assertEqual(state["clock"], {"r1": 1, "r2": 1, "r3": 1})

    def test_same_identity_different_key_is_409_operation_conflict(self) -> None:
        self.seed_conflict("a")
        self.assertEqual(
            self.post_auto("a", auto_request("r3", "fix", {"r1": 1, "r2": 1, "r3": 1}))[
                0
            ],
            201,
        )
        # Distinct identities on key b, dominated by a different clock under
        # the same resolving identity.
        self.post_operation("r1", operation("o3", "b", "v1", {"r1": 2}))
        self.post_operation("r2", operation("o4", "b", "v2", {"r2": 2}))
        status, payload = self.post_auto(
            "b", auto_request("r3", "fix", {"r1": 2, "r2": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("b")
        self.assertEqual(state["status"], "conflict")

    def test_identity_is_shared_with_plain_writes(self) -> None:
        self.post_operation("r3", operation("fix-1", "k", "v", {"r3": 1}))
        self.seed_conflict()
        status, payload = self.post_auto("k", self.good_auto())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class AutoResolveObservabilityTests(HttpServerTestCase):
    def test_resolution_is_exported_in_commit_order_and_shape(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
        self.post_operation("r4", operation("after", "k", "v4", {"r4": 1}))
        _, page = self.get_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1"), ("r4", "after")],
        )
        self.assertEqual(
            page["operations"][2],
            record(
                "r3",
                operation("fix-1", "k", "v1", {"r1": 1, "r2": 1, "r3": 1}),
            ),
        )

    def test_imported_resolution_resolves_the_same_conflict(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
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

    def test_metrics_count_the_resolution(self) -> None:
        self.seed_conflict()
        status, before = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(before["conflictKeys"], 1)
        self.assertEqual(before["resolvedKeys"], 0)
        self.assertEqual(before["acceptedOperations"], 2)
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
        _, after = self.request("GET", "/v1/metrics")
        self.assertEqual(
            after,
            {
                "acceptedOperations": 3,
                "keys": 1,
                "candidateVersions": 1,
                "conflictKeys": 0,
                "resolvedKeys": 1,
                "replicas": 3,
            },
        )

    def test_audit_stream_contains_the_resolution(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_auto("k", self.good_auto())[0], 201)
        _, page = self.request("GET", "/v1/audit/keys/k/operations")
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )
        self.assertEqual(page["operations"][2], self._expected_resolution_record())
        _, digest = self.request("GET", "/v1/audit/keys/k/digest")
        self.assertEqual(digest["operations"], 3)

    @staticmethod
    def _expected_resolution_record() -> dict:
        return record(
            "r3",
            operation("fix-1", "k", "v1", {"r1": 1, "r2": 1, "r3": 1}),
        )


class PersistentAutoResolveTestCase(unittest.TestCase):
    """Auto-resolution against a data-file-backed server with real HTTP."""

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
            status, _ = self.request(server, "POST", f"/v1/replicas/{replica}/operations", op)
            self.assertEqual(status, 201)

    def good_auto(self) -> dict:
        return auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 1})

    def test_resolution_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        status, payload = self.request(server, "POST", "/v1/states/k/resolve/auto", self.good_auto())
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1")],
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")
        # Replay after restart is still 200 and appends nothing.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # A tampered known identity still conflicts after restart.
        tampered = auto_request("r3", "fix-1", {"r1": 1, "r2": 1, "r3": 2})
        status, payload = self.request(server, "POST", "/v1/states/k/resolve/auto", tampered)
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
                server, "POST", "/v1/states/k/resolve/auto", self.good_auto()
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
            server, "POST", "/v1/states/k/resolve/auto", self.good_auto()
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "v1")
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
