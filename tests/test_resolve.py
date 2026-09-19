"""HTTP, sync, and persistence tests for conflict resolution.

The resolution endpoint is::

    POST /v1/states/{key}/resolve

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
    parse_resolve_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


class ParseResolvePayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        payload = parse_resolve_payload(
            json.dumps(
                resolution(
                    "r3",
                    "fix-1",
                    "blue",
                    {"r1": 1, "r2": 1, "r3": 1},
                    [candidate("r1", "o1"), candidate("r2", "o2")],
                )
            )
        )
        self.assertEqual(
            payload,
            {
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "blue",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "candidates": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r2", "operationId": "o2"},
                ],
            },
        )

    def test_rejects_malformed_and_wrong_shapes(self) -> None:
        valid = resolution("r3", "fix", "v", {"r3": 1}, [candidate("r1", "o1")])
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"replicaId": "r3"},
            dict(valid, extra=1),
            {k: v for k, v in valid.items() if k != "clock"},
            dict(valid, replicaId=""),
            dict(valid, replicaId=7),
            dict(valid, operationId=""),
            dict(valid, value=""),
            dict(valid, clock={}),
            dict(valid, clock={"r2": 1}),  # clock must contain the replica
            dict(valid, clock={"r3": -1}),
            dict(valid, clock={"r3": True}),
            dict(valid, candidates=[]),
            dict(valid, candidates="r1"),
            dict(valid, candidates=[{"replicaId": "r1"}]),
            dict(valid, candidates=[{"replicaId": "r1", "operationId": "o1", "x": 1}]),
            dict(valid, candidates=[{"replicaId": "", "operationId": "o1"}]),
            dict(valid, candidates=[{"replicaId": "r1", "operationId": 3}]),
            dict(valid, candidates=[candidate("r1", "o1"), candidate("r1", "o1")]),
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_resolve_payload(body)


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

    def post_resolve(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve", body)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def post_sync(self, body: object) -> tuple[int, object]:
        return self.request("POST", "/v1/sync/operations", body)

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

    def good_resolution(self, key: str = "k") -> dict:
        return resolution(
            "r3",
            "fix-1",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )


class ResolveHappyPathTests(HttpServerTestCase):
    def test_resolve_conflict_is_201_and_state_is_resolved(self) -> None:
        self.seed_conflict()
        status, payload = self.post_resolve("k", self.good_resolution())
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {"status": "created", "key": "k", "replicaId": "r3", "operationId": "fix-1"},
        )
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "merged",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_resolution_clears_only_dominated_candidates(self) -> None:
        self.seed_conflict()
        # A third candidate concurrent with the resolution clock survives.
        self.post_operation("r9", operation("o9", "k", "other", {"r9": 5}))
        body = resolution(
            "r3",
            "fix-1",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        # The candidate set no longer matches: r9's candidate is current too.
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        # Resolving against the full current set keeps r9's candidate.
        body["candidates"].append(candidate("r9", "o9"))
        body["clock"] = {"r1": 1, "r2": 1, "r3": 1, "r9": 6}
        status, _ = self.post_resolve("k", body)
        self.assertEqual(status, 201)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "merged")

    def test_keys_are_isolated(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r1", operation("o-other", "other", "x", {"r1": 2}))
        self.assertEqual(self.post_resolve("k", self.good_resolution())[0], 201)
        _, state = self.get_state("other")
        self.assertEqual(state, {"key": "other", "value": "x", "clock": {"r1": 2}, "status": "resolved"})


class ResolveValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict()
        valid = self.good_resolution()
        bad_bodies = [
            b"{oops",
            [],
            {},
            dict(valid, extra=1),
            dict(valid, replicaId=""),
            dict(valid, value=""),
            dict(valid, clock={"r2": 2}),  # missing the resolving replica
            dict(valid, candidates=[]),
            dict(valid, candidates=[candidate("r1", "o1"), candidate("r1", "o1")]),
            dict(valid, candidates=[{"replicaId": "r1"}]),
        ]
        for body in bad_bodies:
            status, payload = self.post_resolve("k", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_unknown_candidate_identity_is_400(self) -> None:
        self.seed_conflict()
        body = self.good_resolution()
        body["candidates"] = [candidate("r1", "o1"), candidate("r9", "nope")]
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_clock_not_dominating_candidates_is_400(self) -> None:
        self.seed_conflict()
        # Equal to r1's candidate clock on the r1 component but missing r2:
        # concurrent with the r2 candidate, so it does not dominate both.
        body = self.good_resolution()
        body["clock"] = {"r1": 1, "r3": 1}
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # A clock strictly weaker than one candidate is also rejected.
        body["clock"] = {"r1": 1, "r2": 0, "r3": 1}
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 400)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")


class ResolveConflictTests(HttpServerTestCase):
    def test_missing_key_is_409(self) -> None:
        # The candidate identities are known (they conflict on another key),
        # so the request itself is valid; only the target key is absent.
        self.seed_conflict("other")
        status, payload = self.post_resolve("absent", self.good_resolution())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_is_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = resolution("r3", "fix-1", "merged", {"r1": 2, "r3": 1}, [candidate("r1", "o1")])
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_same_value_candidates_are_not_a_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        body = resolution(
            "r3",
            "fix-1",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_candidate_set_mismatch_is_409(self) -> None:
        self.seed_conflict()
        # Subset of the current candidates.
        body = self.good_resolution()
        body["candidates"] = [candidate("r1", "o1")]
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        # A known operation that is not a current candidate (stale write).
        self.post_operation("r1", operation("o3", "k", "stale", {"r1": 0}))
        body["candidates"] = [candidate("r1", "o1"), candidate("r1", "o3")]
        body["clock"] = {"r1": 1, "r2": 1, "r3": 1}
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_concurrent_change_invalidates_the_request(self) -> None:
        self.seed_conflict()
        body = self.good_resolution()
        # A new concurrent write lands before the resolve: the set moved on.
        self.post_operation("r4", operation("o4", "k", "v4", {"r4": 1}))
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)


class ResolveIdentityTests(HttpServerTestCase):
    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict()
        body = self.good_resolution()
        self.assertEqual(self.post_resolve("k", body)[0], 201)
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {"status": "ok", "key": "k", "replicaId": "r3", "operationId": "fix-1"},
        )
        # The log holds the two writes plus exactly one resolution.
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1", "o2", "fix-1"],
        )

    def test_same_identity_different_content_is_409_operation_conflict(self) -> None:
        self.seed_conflict()
        body = self.good_resolution()
        self.assertEqual(self.post_resolve("k", body)[0], 201)
        tampered = dict(body, value="other")
        status, payload = self.post_resolve("k", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k")
        self.assertEqual(state["value"], "merged")

    def test_identity_is_shared_with_plain_writes(self) -> None:
        self.post_operation("r3", operation("fix-1", "k", "v", {"r3": 1}))
        self.seed_conflict()
        body = self.good_resolution()
        body["clock"] = {"r1": 1, "r2": 1, "r3": 1}
        status, payload = self.post_resolve("k", body)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class ResolveSyncTests(HttpServerTestCase):
    def test_resolution_is_exported_in_commit_order(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_resolve("k", self.good_resolution())[0], 201)
        self.post_operation("r4", operation("after", "k", "v4", {"r4": 1}))
        _, page = self.get_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            [("r1", "o1"), ("r2", "o2"), ("r3", "fix-1"), ("r4", "after")],
        )
        resolution_record = page["operations"][2]
        self.assertEqual(
            resolution_record,
            record(
                "r3",
                operation("fix-1", "k", "merged", {"r1": 1, "r2": 1, "r3": 1}),
            ),
        )

    def test_imported_resolution_resolves_the_same_conflict(self) -> None:
        # This server is the source: conflict plus a resolution.
        self.seed_conflict()
        self.assertEqual(self.post_resolve("k", self.good_resolution())[0], 201)
        _, page = self.get_sync()
        exported = page["operations"]

        # A second, independent store imports the whole log.
        other = StateStore()
        status, accepted, _ = other.import_operations(
            [(e["replicaId"], e["operation"]) for e in exported]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 3)
        status, state = other.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            state,
            {
                "key": "k",
                "value": "merged",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_resolution_via_http_import_into_second_server(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_resolve("k", self.good_resolution())[0], 201)
        _, page = self.get_sync()

        # Fresh store on the same server class: import conflict, then resolve.
        self.server.store = type(self.server.store)()
        writes = [e for e in page["operations"] if e["operation"]["operationId"] != "fix-1"]
        status, _ = self.post_sync({"operations": writes})
        self.assertEqual(status, 201)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        status, payload = self.post_sync({"operations": [page["operations"][2]]})
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "merged")


class PersistentResolveTestCase(unittest.TestCase):
    """Resolution against a data-file-backed server with real HTTP."""

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

    def good_resolution(self) -> dict:
        return resolution(
            "r3",
            "fix-1",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )

    def test_resolution_is_durable_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflict(server)
        status, _ = self.request(server, "POST", "/v1/states/k/resolve", self.good_resolution())
        self.assertEqual(status, 201)
        # The resolution is already in the file, in commit order.
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
        self.assertEqual(state["value"], "merged")
        # Replay after restart is still a 200 and appends nothing.
        status, payload = self.request(
            server, "POST", "/v1/states/k/resolve", self.good_resolution()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        # A tampered known identity still conflicts after restart.
        tampered = dict(self.good_resolution(), value="tampered")
        status, payload = self.request(server, "POST", "/v1/states/k/resolve", tampered)
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
                server, "POST", "/v1/states/k/resolve", self.good_resolution()
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
            server, "POST", "/v1/states/k/resolve", self.good_resolution()
        )
        self.assertEqual(status, 201)
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "merged")
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
