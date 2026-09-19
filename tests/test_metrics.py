"""Tests for the read-only metrics endpoint::

    GET /v1/metrics

It returns exactly six non-negative integer counters computed from a single
snapshot under the shared commit lock:

    acceptedOperations, keys, candidateVersions,
    conflictKeys, resolvedKeys, replicas

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
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
    parse_metrics_query,
)

METRIC_FIELDS = {
    "acceptedOperations",
    "keys",
    "candidateVersions",
    "conflictKeys",
    "resolvedKeys",
    "replicas",
}

EMPTY_METRICS = {
    "acceptedOperations": 0,
    "keys": 0,
    "candidateVersions": 0,
    "conflictKeys": 0,
    "resolvedKeys": 0,
    "replicas": 0,
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
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


class ParseMetricsQueryTests(unittest.TestCase):
    def test_empty_query_is_accepted(self) -> None:
        self.assertTrue(parse_metrics_query(""))

    def test_any_parameter_is_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x=1"))
        self.assertFalse(parse_metrics_query("after=0"))

    def test_blank_values_are_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x="))
        self.assertFalse(parse_metrics_query("x"))
        self.assertFalse(parse_metrics_query("=1"))

    def test_repeated_parameters_are_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x=1&x=2"))


class MetricsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store_reports_zeroes(self) -> None:
        self.assertEqual(self.store.get_metrics(), EMPTY_METRICS)

    def test_first_accepts_are_counted(self) -> None:
        self.assertIs(
            self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        self.assertIs(
            self.store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1})),
            HTTPStatus.CREATED,
        )
        # Same-value concurrent writes keep the key resolved.
        self.assertIs(
            self.store.apply_operation("r3", operation("o3", "other", "v3", {"r3": 1})),
            HTTPStatus.CREATED,
        )
        self.assertEqual(
            self.store.get_metrics(),
            {
                "acceptedOperations": 3,
                "keys": 2,
                "candidateVersions": 3,
                "conflictKeys": 0,
                "resolvedKeys": 2,
                "replicas": 3,
            },
        )

    def test_stale_write_is_accepted_but_adds_no_candidate(self) -> None:
        self.store.apply_operation("r1", operation("new", "k", "new", {"r1": 2}))
        self.assertIs(
            self.store.apply_operation("r1", operation("stale", "k", "stale", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        metrics = self.store.get_metrics()
        # The stale write counts as a first-accepted operation...
        self.assertEqual(metrics["acceptedOperations"], 2)
        # ...but adds no candidate, so the key still has a single version.
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 1)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)

    def test_replays_conflicts_and_invalid_requests_are_excluded(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.store.apply_operation("r1", op)
        # Identical replay: 200, not counted.
        self.assertIs(self.store.apply_operation("r1", dict(op)), HTTPStatus.OK)
        # Same identity, different content: 409, not counted.
        self.assertIs(
            self.store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        # A conflicting import batch is rejected as a whole: nothing counted.
        status, accepted, replayed = self.store.import_operations(
            [
                ("r2", operation("o2", "k2", "v2", {"r2": 1})),
                ("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual((accepted, replayed), (0, 0))

        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 1)
        self.assertEqual(metrics["replicas"], 1)
        self.assertEqual(metrics["keys"], 1)
        # The accepted count always equals the durable log length.
        page, _, has_more = self.store.get_sync_operations(0, 100)
        self.assertFalse(has_more)
        self.assertEqual(metrics["acceptedOperations"], len(page))

    def test_pure_replay_import_batch_adds_no_accepts(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, accepted, replayed = self.store.import_operations(
            [("r1", operation("o1", "k", "v", {"r1": 1}))]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_conflict_and_resolution_move_keys_between_buckets(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 2)
        self.assertEqual(metrics["conflictKeys"], 1)
        self.assertEqual(metrics["resolvedKeys"], 0)
        self.assertEqual(metrics["replicas"], 2)

        # A repair is a first-accepted operation attributed to its initiator.
        status, error = self.store.apply_resolution(
            "k",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["resolvedKeys"], 1)
        # r3 only appears in the log as the repair initiator and still counts.
        self.assertEqual(metrics["replicas"], 3)

    def test_replaying_a_resolution_is_not_recounted(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        fix = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        self.store.apply_resolution("k", fix)
        status, _ = self.store.apply_resolution("k", fix)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 3)

    def test_import_batch_counts_accepts_and_replicas(self) -> None:
        status, accepted, replayed = self.store.import_operations(
            [
                ("r1", operation("o1", "k", "v1", {"r1": 1})),
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r3", operation("o3", "other", "v3", {"r3": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (3, 0))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["keys"], 2)
        self.assertEqual(metrics["candidateVersions"], 3)
        self.assertEqual(metrics["conflictKeys"], 1)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["replicas"], 3)

    def test_counter_invariants_hold(self) -> None:
        timeline = [
            ("r1", operation("a", "k1", "1", {"r1": 1})),
            ("r2", operation("b", "k1", "2", {"r2": 1})),
            ("r1", operation("c", "k2", "3", {"r1": 2})),
            ("r3", operation("d", "k3", "4", {"r3": 1, "r1": 1})),
            ("r2", operation("e", "k2", "5", {"r2": 2})),
        ]
        for replica, op in timeline:
            self.store.apply_operation(replica, op)
            metrics = self.store.get_metrics()
            self.assertGreaterEqual(metrics["acceptedOperations"], 0)
            self.assertGreaterEqual(metrics["keys"], 0)
            self.assertGreaterEqual(metrics["candidateVersions"], metrics["keys"])
            self.assertEqual(
                metrics["conflictKeys"] + metrics["resolvedKeys"], metrics["keys"]
            )
            self.assertLessEqual(metrics["replicas"], metrics["acceptedOperations"])

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.store.get_sync_operations(0, 100)[0]
        first = self.store.get_metrics()
        second = self.store.get_metrics()
        self.assertEqual(first, second)
        after = self.store.get_sync_operations(0, 100)[0]
        self.assertEqual(before, after)

    def test_failed_durable_commit_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            persistent = StateStore(data_file=str(Path(tmp) / "state.json"))
            with patch.object(
                StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
            ):
                with self.assertRaises(PersistenceError):
                    persistent.apply_operation(
                        "r1", operation("o1", "k", "v", {"r1": 1})
                    )
            self.assertEqual(persistent.get_metrics(), EMPTY_METRICS)


class MetricsRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_metrics_match_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        # Conflict plus a stale write.
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        # A second, independently resolved key.
        store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1}))
        # Repair the conflict.
        store.apply_resolution(
            "k",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        # A replay and a conflict must leave no trace.
        self.assertIs(
            store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1})),
            HTTPStatus.OK,
        )
        self.assertIs(
            store.apply_operation("r4", operation("o4", "other", "z", {"r4": 1})),
            HTTPStatus.CONFLICT,
        )
        # An import batch with one fresh record and one replay.
        store.import_operations(
            [
                ("r5", operation("o5", "third", "t", {"r5": 1})),
                ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ]
        )
        before = store.get_metrics()

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_metrics(), before)
        # And a second restart is identical too.
        del recovered
        self.assertEqual(StateStore(data_file=self.data_file).get_metrics(), before)


class MetricsHttpServerTests(unittest.TestCase):
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

    def raw_request(self, method: str, path: str, body: object = None) -> tuple[int, dict, bytes, list[tuple[str, str]]]:
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
        headers = response.getheaders()
        conn.close()
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def metrics(self, path: str = "/v1/metrics") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_metrics_are_zeroes(self) -> None:
        status, payload = self.metrics()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_METRICS)

    def test_payload_has_exactly_six_non_negative_integer_fields(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), METRIC_FIELDS)
        for name, value in payload.items():
            self.assertIs(type(value), int, f"{name} must be an int, got {type(value)!r}")
            self.assertGreaterEqual(value, 0)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        # The existing explicit-content-length contract is preserved.
        self.assertEqual(header_map["content-length"], str(len(raw)))
        self.assertEqual(len(raw), int(header_map["content-length"]))

    def test_metrics_reflect_writes_conflicts_and_repairs(self) -> None:
        self.assertEqual(self.metrics()[1], EMPTY_METRICS)
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = self.metrics()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "acceptedOperations": 2,
                "keys": 1,
                "candidateVersions": 2,
                "conflictKeys": 1,
                "resolvedKeys": 0,
                "replicas": 2,
            },
        )

        # A replay and a conflict change nothing.
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r1", operation("o1", "k", "other", {"r1": 1}))
        self.assertEqual(self.metrics()[1]["acceptedOperations"], 2)

        status, _ = self.request(
            "POST",
            "/v1/states/k/resolve",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.assertEqual(status, 201)
        status, payload = self.metrics()
        self.assertEqual(
            payload,
            {
                "acceptedOperations": 3,
                "keys": 1,
                "candidateVersions": 1,
                "conflictKeys": 0,
                "resolvedKeys": 1,
                "replicas": 3,
            },
        )

    def test_metrics_reflect_import_batches(self) -> None:
        status, _ = self.request(
            "POST",
            "/v1/sync/operations",
            {
                "operations": [
                    record("r1", operation("o1", "k", "v1", {"r1": 1})),
                    record("r2", operation("o2", "k", "v2", {"r2": 1})),
                ]
            },
        )
        self.assertEqual(status, 201)
        status, payload = self.metrics()
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 2)
        self.assertEqual(payload["replicas"], 2)
        self.assertEqual(payload["conflictKeys"], 1)

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/metrics?x=1",
            "/v1/metrics?after=0",
            "/v1/metrics?x=",
            "/v1/metrics?x",
            "/v1/metrics?=1",
            "/v1/metrics?x=1&x=2",
            "/v1/metrics?acceptedOperations=1",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        # Bare separators carry no parameter, matching the sync/audit parsers.
        status, _ = self.request("GET", "/v1/metrics?")
        self.assertEqual(status, 200)

    def test_extra_path_is_404(self) -> None:
        for path in ("/v1/metrics/extra", "/v1/metrics/operations", "/v1/metric"):
            status, payload = self.metrics(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_extra_path_is_404_even_with_query(self) -> None:
        status, payload = self.metrics("/v1/metrics/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_metrics_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        first_status, first = self.metrics()
        for _ in range(3):
            status, payload = self.metrics()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        # The accepted-operation log is untouched.
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), first["acceptedOperations"])
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                metrics = self.server.store.get_metrics()
                if metrics["conflictKeys"] + metrics["resolvedKeys"] != metrics["keys"]:
                    violations.append("conflict + resolved != keys")
                if metrics["candidateVersions"] < metrics["keys"]:
                    violations.append("candidateVersions < keys")
                if any(not isinstance(v, int) or v < 0 for v in metrics.values()):
                    violations.append("non-integer or negative counter")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                replica = f"r{index}"
                self.post_operation(
                    replica,
                    operation(f"op-{index}", "shared", f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.metrics()
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 40)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 40)
        self.assertEqual(payload["conflictKeys"], 1)
        self.assertEqual(payload["resolvedKeys"], 0)
        self.assertEqual(payload["replicas"], 40)


class PersistentMetricsHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
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
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_metrics_survive_restart(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        status, before = self.request(server, "GET", "/v1/metrics")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_metrics_write_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/metrics")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
