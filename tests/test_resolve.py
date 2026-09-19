"""HTTP, sync, persistence, and recovery tests for conflict resolution.

The resolution endpoint is::

    POST /v1/states/{key}/resolve

A resolution is committed as one accepted operation in the same commit
order as local writes and import batches, so it exports, imports, and
recovers through the existing sync and persistence contracts. Everything
here uses only the Python standard library.
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

from semantic_state_engine.server import (
    PersistenceError,
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


def resolution(
    replica: str = "r3",
    operation_id: str = "resolve-1",
    value: str = "blue",
    clock: dict | None = None,
    candidates: tuple[tuple[str, str], ...] = (("r1", "o1"), ("r2", "o2")),
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": dict(clock) if clock is not None else {"r1": 1, "r2": 1, "r3": 1},
        "candidates": [
            {"replicaId": r, "operationId": o} for r, o in candidates
        ],
    }


class ParseResolvePayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        replica_id, op, candidates = parse_resolve_payload(
            json.dumps(resolution()), "color"
        )
        self.assertEqual(replica_id, "r3")
        self.assertEqual(
            op,
            {
                "operationId": "resolve-1",
                "key": "color",
                "value": "blue",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
            },
        )
        self.assertEqual(candidates, [("r1", "o1"), ("r2", "o2")])

    def test_rejects_bad_shapes(self) -> None:
        valid = resolution()
        bad = [
            b"{not json",
            b"\xff\xfe",
            "[1, 2]",
            {},
            {**valid, "extra": 1},
            {k: v for k, v in valid.items() if k != "candidates"},
            {**valid, "replicaId": ""},
            {**valid, "replicaId": 7},
            {**valid, "operationId": ""},
            {**valid, "value": ""},
            {**valid, "clock": {}},
            {**valid, "clock": {"r1": 1}},  # clock must contain replicaId
            {**valid, "clock": {"r3": -1}},
            {**valid, "clock": {"r3": True}},
            {**valid, "candidates": []},
            {**valid, "candidates": "nope"},
            {**valid, "candidates": [{"replicaId": "r1"}]},
            {**valid, "candidates": [{"replicaId": "r1", "operationId": "o1", "x": 1}]},
            {**valid, "candidates": [{"replicaId": "", "operationId": "o1"}]},
            {**valid, "candidates": [{"replicaId": "r1", "operationId": ""}]},
            {
                **valid,
                "candidates": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r1", "operationId": "o1"},
                ],
            },
        ]
        for body in bad:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_resolve_payload(body, "color")


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

    def seed_conflict(self, key: str = "color") -> None:
        self.assertEqual(
            self.post_operation("r1", operation("o1", key, "blue", {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", key, "green", {"r2": 1}))[0], 201
        )
        _, state = self.get_state(key)
        self.assertEqual(state["status"], "conflict")

    def export_log(self) -> list[dict]:
        status, payload = self.request("GET", "/v1/sync/operations?limit=100")
        self.assertEqual(status, 200)
        return payload["operations"]


class ResolveSuccessTests(HttpServerTestCase):
    def test_resolve_is_201_and_state_becomes_resolved(self) -> None:
        self.seed_conflict()
        status, payload = self.post_resolve("color", resolution())
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "key": "color",
                "replicaId": "r3",
                "operationId": "resolve-1",
            },
        )
        status, state = self.get_state("color")
        self.assertEqual(status, 200)
        self.assertEqual(
            state,
            {
                "key": "color",
                "value": "blue",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_resolution_enters_the_commit_order_like_a_write(self) -> None:
        self.seed_conflict()
        self.post_resolve("color", resolution())
        self.post_operation("r1", operation("o3", "color", "gold", {"r1": 2, "r2": 1, "r3": 1}))
        log = self.export_log()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in log],
            [("r1", "o1"), ("r2", "o2"), ("r3", "resolve-1"), ("r1", "o3")],
        )
        # The exported resolution record is an ordinary operation record.
        self.assertEqual(
            log[2],
            record(
                "r3",
                operation("resolve-1", "color", "blue", {"r1": 1, "r2": 1, "r3": 1}),
            ),
        )

    def test_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_resolve("color", resolution())[0], 201)
        status, payload = self.post_resolve("color", resolution())
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "key": "color",
                "replicaId": "r3",
                "operationId": "resolve-1",
            },
        )
        self.assertEqual(len(self.export_log()), 3)
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "resolved")

    def test_same_identity_different_content_is_409_operation_conflict(self) -> None:
        self.seed_conflict()
        self.assertEqual(self.post_resolve("color", resolution())[0], 201)
        tampered = resolution(value="green")
        status, payload = self.post_resolve("color", tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("color")
        self.assertEqual(state["value"], "blue")
        self.assertEqual(len(self.export_log()), 3)

    def test_resolution_identity_clashes_with_plain_write_identity(self) -> None:
        self.seed_conflict()
        op = operation("resolve-1", "color", "blue", {"r1": 1, "r2": 1, "r3": 1})
        # A plain write already owns the identity with different content.
        self.post_operation("r3", operation("resolve-1", "color", "other", {"r3": 1}))
        status, payload = self.post_resolve("color", resolution())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class ResolveConflictStateTests(HttpServerTestCase):
    def test_missing_key_is_409_resolution_conflict(self) -> None:
        status, payload = self.post_resolve("absent", resolution())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_resolved_key_is_409_resolution_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        status, payload = self.post_resolve(
            "color", resolution(candidates=(("r1", "o1"),), clock={"r1": 2, "r3": 1})
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "blue")

    def test_same_value_candidates_are_not_a_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "blue", {"r2": 1}))
        status, payload = self.post_resolve("color", resolution())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_candidate_set_mismatch_is_409_resolution_conflict(self) -> None:
        self.seed_conflict()
        bad_sets = [
            (("r1", "o1"),),  # subset
            (("r1", "o1"), ("r2", "o2"), ("r3", "o3")),  # superset
            (("r1", "o1"), ("r2", "other")),  # wrong operationId
            (("r1", "o1"), ("r9", "o2")),  # unknown replica
        ]
        for candidates in bad_sets:
            status, payload = self.post_resolve("color", resolution(candidates=candidates))
            self.assertEqual(status, 409, repr(candidates))
            self.assertEqual(payload, {"error": "resolution_conflict"}, repr(candidates))
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(self.export_log()), 2)

    def test_concurrent_change_invalidates_the_candidate_set(self) -> None:
        self.seed_conflict()
        # A third concurrent write arrives after the client read the state.
        self.post_operation("r9", operation("o9", "color", "red", {"r9": 1}))
        status, payload = self.post_resolve("color", resolution())
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)

    def test_clock_not_dominating_candidates_is_400(self) -> None:
        self.seed_conflict()
        bad_clocks = [
            {"r1": 1, "r3": 1},  # does not dominate r2's clock
            {"r2": 2, "r3": 1},  # does not dominate r1's clock
            {"r1": 1, "r2": 0, "r3": 9},  # lower on a candidate component
        ]
        for clock in bad_clocks:
            status, payload = self.post_resolve("color", resolution(clock=clock))
            self.assertEqual(status, 400, repr(clock))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(clock))
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(self.export_log()), 2)

    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict()
        valid = resolution()
        bad_bodies = [
            b"{not json",
            [],
            {},
            {**valid, "extra": 1},
            {**valid, "candidates": []},
            {**valid, "candidates": [{"replicaId": "r1"}]},
            {
                **valid,
                "candidates": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r1", "operationId": "o1"},
                ],
            },
            {**valid, "clock": {"r1": 1, "r2": 1}},  # missing the resolver
            {**valid, "value": ""},
        ]
        for body in bad_bodies:
            status, payload = self.post_resolve("color", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(self.export_log()), 2)

    def test_get_on_resolve_route_is_404(self) -> None:
        self.seed_conflict()
        status, payload = self.request("GET", "/v1/states/color/resolve")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ResolveSyncTests(HttpServerTestCase):
    def test_resolution_imports_like_any_operation(self) -> None:
        self.seed_conflict()
        self.post_resolve("color", resolution())
        log = self.export_log()

        other = StateStore()
        status, accepted, replayed = other.import_operations(
            [(entry["replicaId"], entry["operation"]) for entry in log]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (3, 0))
        http_status, state = other.get_state("color")
        self.assertIs(http_status, HTTPStatus.OK)
        self.assertEqual(
            state,
            {
                "key": "color",
                "value": "blue",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )

    def test_imported_resolution_keeps_unknown_concurrent_candidates(self) -> None:
        self.seed_conflict()
        self.post_resolve("color", resolution())
        log = self.export_log()

        other = StateStore()
        # The other replica has its own concurrent candidate the resolution
        # never saw; importing the log must not silently drop it.
        other.apply_operation("r9", operation("o9", "color", "red", {"r9": 1}))
        other.import_operations(
            [(entry["replicaId"], entry["operation"]) for entry in log]
        )
        _, state = other.get_state("color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(
            {(c["replicaId"], c["operationId"]) for c in state["candidates"]},
            {("r3", "resolve-1"), ("r9", "o9")},
        )


class PersistentResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def seed_conflict(self, store: StateStore) -> None:
        store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "color", "green", {"r2": 1}))

    def commit_resolution(self, store: StateStore) -> tuple[HTTPStatus, object]:
        replica_id, op, candidates = parse_resolve_payload(resolution(), "color")
        return store.resolve("color", replica_id, op, candidates)

    def test_resolution_is_durable_and_recovers(self) -> None:
        store = self.make_store()
        self.seed_conflict(store)
        status, error = self.commit_resolution(store)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        # The file already holds the resolution as an ordinary record.
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records],
            [("r1", "o1"), ("r2", "o2"), ("r3", "resolve-1")],
        )

        reloaded = self.make_store()
        http_status, state = reloaded.get_state("color")
        self.assertIs(http_status, HTTPStatus.OK)
        self.assertEqual(
            state,
            {
                "key": "color",
                "value": "blue",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "status": "resolved",
            },
        )
        # Replay and conflict semantics survive the restart.
        status, error = self.commit_resolution(reloaded)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        _, op, candidates = parse_resolve_payload(resolution(value="green"), "color")
        status, error = reloaded.resolve("color", "r3", op, candidates)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        def req(method: str, path: str, body: object = None) -> tuple[int, object]:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                method,
                path,
                body=json.dumps(body) if body is not None else None,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            return response.status, payload

        req("POST", "/v1/replicas/r1/operations", operation("o1", "color", "blue", {"r1": 1}))
        req("POST", "/v1/replicas/r2/operations", operation("o2", "color", "green", {"r2": 1}))
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = req("POST", "/v1/states/color/resolve", resolution())
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # Memory, the identity index, and the file are unchanged.
        self.assertEqual(self.data_file.read_bytes(), before)
        _, state = req("GET", "/v1/states/color")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)
        # The failed resolution identity was not recorded: it can commit now.
        status, payload = req("POST", "/v1/states/color/resolve", resolution())
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        _, state = req("GET", "/v1/states/color")
        self.assertEqual(state["status"], "resolved")
        reloaded = self.make_store()
        http_status, state = reloaded.get_state("color")
        self.assertIs(http_status, HTTPStatus.OK)
        self.assertEqual(state["value"], "blue")
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
