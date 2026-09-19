"""Tests for the read-only metrics endpoint.

Covers::

    GET /v1/metrics

Counter semantics at the store level, HTTP request/response shape, the
no-query and extra-path contracts, snapshot consistency under concurrent
commits, and parity across a persistence restart.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

METRIC_FIELDS = (
    "acceptedOperations",
    "keys",
    "candidateVersions",
    "conflictKeys",
    "resolvedKeys",
    "replicas",
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def resolution(operation_id: str, value: str, clock: dict, candidates: list[tuple[str, str]]) -> dict:
    return {
        "replicaId": "r3",
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": [
            {"replicaId": replica_id, "operationId": op_id}
            for replica_id, op_id in candidates
        ],
    }


class MetricsSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store_is_all_zero(self) -> None:
        self.assertEqual(
            self.store.get_metrics(),
            {
                "acceptedOperations": 0,
                "keys": 0,
                "candidateVersions": 0,
                "conflictKeys": 0,
                "resolvedKeys": 0,
                "replicas": 0,
            },
        )

    def test_single_write(self) -> None:
        self.assertIs(
            self.store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        self.assertEqual(
            self.store.get_metrics(),
            {
                "acceptedOperations": 1,
                "keys": 1,
                "candidateVersions": 1,
                "conflictKeys": 0,
                "resolvedKeys": 1,
                "replicas": 1,
            },
        )

    def test_conflict_and_resolved_keys(self) -> None:
        # "color" is a genuine conflict; "shape" resolves with equal values.
        self.store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "color", "green", {"r2": 1}))
        self.store.apply_operation("r1", operation("o3", "shape", "round", {"r1": 1}))
        self.store.apply_operation("r2", operation("o4", "shape", "round", {"r2": 1}))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 4)
        self.assertEqual(metrics["keys"], 2)
        self.assertEqual(metrics["candidateVersions"], 4)
        self.assertEqual(metrics["conflictKeys"], 1)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["replicas"], 2)

    def test_stale_write_is_accepted_but_adds_no_candidate(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        self.store.apply_operation("r1", operation("o2", "k", "stale", {"r1": 1}))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 2)
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 1)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)

    def test_dominating_write_collapses_candidates(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 2, "r2": 1}))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["replicas"], 2)

    def test_replay_and_conflict_are_excluded(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.store.apply_operation("r1", op)
        self.assertIs(self.store.apply_operation("r1", dict(op)), HTTPStatus.OK)
        self.assertIs(
            self.store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        self.assertEqual(
            self.store.get_metrics(),
            {
                "acceptedOperations": 1,
                "keys": 1,
                "candidateVersions": 1,
                "conflictKeys": 0,
                "resolvedKeys": 1,
                "replicas": 1,
            },
        )

    def test_repair_counts_and_clears_conflict(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, error = self.store.apply_resolution(
            "k",
            resolution(
                "fix-1",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [("r1", "o1"), ("r2", "o2")],
            ),
        )
        self.assertEqual((status, error), (HTTPStatus.CREATED, None))
        metrics = self.store.get_metrics()
        # The repair is a first-accepted operation from a new (initiating)
        # replica, and the key is now resolved with a single candidate.
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["keys"], 1)
        self.assertEqual(metrics["candidateVersions"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["resolvedKeys"], 1)
        self.assertEqual(metrics["replicas"], 3)

    def test_rejected_repair_is_excluded(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        # Key is not in conflict (single agreeing value): resolution_conflict.
        status, error = self.store.apply_resolution(
            "k",
            resolution("fix-1", "merged", {"r1": 1, "r3": 1}, [("r1", "o1")]),
        )
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "resolution_conflict"))
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 1)
        self.assertEqual(metrics["replicas"], 1)
        self.assertEqual(metrics["conflictKeys"], 0)

    def test_distinct_replicas_are_counted_from_the_log(self) -> None:
        for replica, index in (("r2", 1), ("r1", 2), ("r2", 3)):
            self.store.apply_operation(
                replica, operation(f"o{index}", f"k{index}", "v", {replica: index})
            )
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["keys"], 3)
        self.assertEqual(metrics["replicas"], 2)

    def test_invariants_always_hold(self) -> None:
        self.store.apply_operation("r1", operation("o1", "a", "x", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "b", "y", {"r2": 1}))
        self.store.apply_operation("r1", operation("o3", "a", "z", {"r2": 1, "r1": 1}))
        metrics = self.store.get_metrics()
        self.assertEqual(
            set(metrics),
            set(METRIC_FIELDS),
        )
        self.assertTrue(all(isinstance(metrics[name], int) and metrics[name] >= 0 for name in METRIC_FIELDS))
        self.assertEqual(metrics["conflictKeys"] + metrics["resolvedKeys"], metrics["keys"])
        self.assertGreaterEqual(metrics["candidateVersions"], metrics["keys"])
        self.assertGreaterEqual(metrics["acceptedOperations"], metrics["candidateVersions"])


class MetricsConcurrencyTests(unittest.TestCase):
    def test_snapshot_never_observes_half_a_batch(self) -> None:
        store = StateStore()
        stop = threading.Event()
        violations: list[dict] = []

        def reader() -> None:
            while not stop.is_set():
                metrics = store.get_metrics()
                if metrics["conflictKeys"] + metrics["resolvedKeys"] != metrics["keys"]:
                    violations.append(metrics)
                if metrics["candidateVersions"] < metrics["keys"]:
                    violations.append(metrics)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for thread in readers:
            thread.start()
        try:
            for batch in range(20):
                records = [
                    (
                        f"r{index}",
                        operation(
                            f"o{batch}-{index}",
                            f"k{index}",
                            "same",
                            {f"r{index}": batch + 1},
                        ),
                    )
                    for index in range(8)
                ]
                status, accepted, replayed = store.import_operations(records)
                self.assertIs(status, HTTPStatus.CREATED)
                self.assertEqual((accepted, replayed), (8, 0))
        finally:
            stop.set()
            for thread in readers:
                thread.join(timeout=5)
        self.assertEqual(violations, [])
        metrics = store.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 160)
        self.assertEqual(metrics["keys"], 8)
        self.assertEqual(metrics["replicas"], 8)


class MetricsPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def seed(self, store: StateStore) -> dict[str, int]:
        store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "color", "green", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "shape", "round", {"r1": 1}))
        store.apply_operation("r2", operation("o4", "shape", "round", {"r2": 1}))
        store.apply_operation("r1", operation("o5", "color", "old", {"r1": 0}))  # stale: {r1:1} dominates
        store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))  # replay
        store.apply_resolution(
            "color",
            resolution(
                "fix-1",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [("r1", "o1"), ("r2", "o2")],
            ),
        )
        return store.get_metrics()

    def test_metrics_survive_restart(self) -> None:
        store = self.make_store()
        before = self.seed(store)
        del store
        recovered = self.make_store()
        self.assertEqual(recovered.get_metrics(), before)
        self.assertEqual(
            before,
            {
                "acceptedOperations": 6,
                "keys": 2,
                "candidateVersions": 3,
                "conflictKeys": 0,
                "resolvedKeys": 2,
                "replicas": 3,
            },
        )

    def test_metrics_do_not_write_the_data_file(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        raw = self.data_file.read_bytes()
        mtime = self.data_file.stat().st_mtime_ns
        for _ in range(5):
            metrics = store.get_metrics()
            self.assertEqual(metrics["acceptedOperations"], 1)
        self.assertEqual(self.data_file.read_bytes(), raw)
        self.assertEqual(self.data_file.stat().st_mtime_ns, mtime)
        leftovers = [
            p.name
            for p in Path(self._tmp.name).iterdir()
            if p.name != self.data_file.name
        ]
        self.assertEqual(leftovers, [])


class MetricsHttpTests(unittest.TestCase):
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

    def raw_request(self, method: str, path: str, body: object = None) -> tuple[int, bytes, http.client.HTTPResponse]:
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
        headers = response
        conn.close()
        return response.status, raw, headers

    def get(self, path: str) -> tuple[int, object]:
        status, raw, _ = self.raw_request("GET", path)
        return status, json.loads(raw.decode("utf-8"))

    def post_operation(self, replica: str, op: dict) -> int:
        status, _, _ = self.raw_request("POST", f"/v1/replicas/{replica}/operations", op)
        return status

    def test_empty_metrics(self) -> None:
        status, payload = self.get("/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "acceptedOperations": 0,
                "keys": 0,
                "candidateVersions": 0,
                "conflictKeys": 0,
                "resolvedKeys": 0,
                "replicas": 0,
            },
        )

    def test_response_has_exactly_six_non_negative_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, raw, response = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(set(payload), set(METRIC_FIELDS))
        for name in METRIC_FIELDS:
            self.assertIsInstance(payload[name], int, name)
            self.assertNotIsInstance(payload[name], bool, name)
            self.assertGreaterEqual(payload[name], 0, name)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(response.getheader("Content-Length"), str(len(raw)))
        # Compact, key-sorted, UTF-8 JSON, identical encoding to the other routes.
        self.assertEqual(
            raw,
            b'{"acceptedOperations":1,"candidateVersions":1,"conflictKeys":0,'
            b'"keys":1,"replicas":1,"resolvedKeys":1}',
        )

    def test_metrics_reflect_writes_conflicts_repairs(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "green", {"r2": 1}))
        status, payload = self.get("/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 2)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 2)
        self.assertEqual(payload["conflictKeys"], 1)
        self.assertEqual(payload["resolvedKeys"], 0)
        self.assertEqual(payload["replicas"], 2)

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/states/color/resolve",
            body=json.dumps(
                resolution(
                    "fix-1",
                    "merged",
                    {"r1": 1, "r2": 1, "r3": 1},
                    [("r1", "o1"), ("r2", "o2")],
                )
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()

        status, payload = self.get("/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 3)
        self.assertEqual(payload["candidateVersions"], 1)
        self.assertEqual(payload["conflictKeys"], 0)
        self.assertEqual(payload["resolvedKeys"], 1)
        self.assertEqual(payload["replicas"], 3)

    def test_any_query_parameter_is_400(self) -> None:
        for query in (
            "?x=1",
            "?x=1&x=2",
            "?x=",
            "?x",
            "?&",
            "?=1",
            "?%20",
            "?after=0",
            "?limit=100",
        ):
            status, payload = self.get(f"/v1/metrics{query}")
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_bare_question_mark_keeps_empty_query(self) -> None:
        # urlsplit("/v1/metrics?").query is the empty string, not a parameter.
        status, payload = self.get("/v1/metrics?")
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 0)

    def test_extra_path_is_404(self) -> None:
        for path in ("/v1/metrics/extra", "/v2/metrics", "/v1/metrics/extra/operations"):
            status, payload = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_metrics_are_read_only(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, before = self.get("/v1/metrics")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, after = self.get("/v1/metrics")
            self.assertEqual(status, 200)
            self.assertEqual(after, before)
        # State read is unaffected too.
        status, state = self.get("/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")

    def test_other_routes_still_work(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})
        status, payload = self.get("/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(payload["nextCursor"], 1)
        self.assertTrue(payload["operations"])
        status, payload = self.get("/v1/audit/keys/k/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 1)


if __name__ == "__main__":
    unittest.main()
