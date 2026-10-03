"""Tests for the read-only conflict-pressure metrics endpoint::

    GET /v1/metrics/conflicts

It returns exactly six non-negative integer counters computed from a single
snapshot under the shared commit lock:

    keys, conflictKeys, candidatePairs,
    conflictPairs, maxCandidatesInKey, maxDistinctValuesInKey

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
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

CONFLICT_METRIC_FIELDS = {
    "keys",
    "conflictKeys",
    "candidatePairs",
    "conflictPairs",
    "maxCandidatesInKey",
    "maxDistinctValuesInKey",
}

EMPTY_CONFLICT_METRICS = {
    "keys": 0,
    "conflictKeys": 0,
    "candidatePairs": 0,
    "conflictPairs": 0,
    "maxCandidatesInKey": 0,
    "maxDistinctValuesInKey": 0,
}

TOKEN = "s3cret-token_123"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


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


class ConflictMetricsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store_reports_zeroes(self) -> None:
        self.assertEqual(self.store.get_conflict_metrics(), EMPTY_CONFLICT_METRICS)

    def test_single_candidate_key_contributes_only_keys_and_maxima(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 0,
                "conflictPairs": 0,
                "maxCandidatesInKey": 1,
                "maxDistinctValuesInKey": 1,
            },
        )

    def test_same_value_concurrent_candidates_are_not_conflict_pairs(self) -> None:
        # Three pairwise-concurrent writes with the same value: all three
        # unordered pairs count as candidate pairs, none as conflict pairs.
        self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "same", {"r3": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 3,
                "conflictPairs": 0,
                "maxCandidatesInKey": 3,
                "maxDistinctValuesInKey": 1,
            },
        )

    def test_conflict_pairs_count_only_value_mismatches(self) -> None:
        # Candidates (r1,o1)=a, (r2,o2)=b, (r3,o3)=a: three unordered pairs,
        # only the two a-b pairs are conflict pairs.
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "a", {"r3": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 1,
                "candidatePairs": 3,
                "conflictPairs": 2,
                "maxCandidatesInKey": 3,
                "maxDistinctValuesInKey": 2,
            },
        )

    def test_pairs_are_counted_once_across_keys(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k2", "x", {"r3": 1}))
        self.store.apply_operation("r4", operation("o4", "k2", "x", {"r4": 1}))
        metrics = self.store.get_conflict_metrics()
        self.assertEqual(metrics["keys"], 2)
        self.assertEqual(metrics["conflictKeys"], 1)
        # One pair per key; only k1's pair disagrees on the value.
        self.assertEqual(metrics["candidatePairs"], 2)
        self.assertEqual(metrics["conflictPairs"], 1)
        self.assertEqual(metrics["maxCandidatesInKey"], 2)
        self.assertEqual(metrics["maxDistinctValuesInKey"], 2)

    def test_maxima_track_the_largest_key_not_the_totals(self) -> None:
        self.store.apply_operation("r1", operation("o1", "big", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "big", "b", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "big", "c", {"r3": 1}))
        self.store.apply_operation("r4", operation("o4", "small", "x", {"r4": 1}))
        metrics = self.store.get_conflict_metrics()
        self.assertEqual(metrics["keys"], 2)
        self.assertEqual(metrics["candidatePairs"], 3)
        self.assertEqual(metrics["conflictPairs"], 3)
        self.assertEqual(metrics["maxCandidatesInKey"], 3)
        self.assertEqual(metrics["maxDistinctValuesInKey"], 3)

    def test_stale_write_adds_no_candidate_and_no_pairs(self) -> None:
        self.store.apply_operation("r1", operation("new", "k", "new", {"r1": 2}))
        self.store.apply_operation("r1", operation("stale", "k", "stale", {"r1": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 0,
                "conflictPairs": 0,
                "maxCandidatesInKey": 1,
                "maxDistinctValuesInKey": 1,
            },
        )

    def test_resolution_collapses_pairs(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.assertEqual(self.store.get_conflict_metrics()["conflictPairs"], 1)
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
        self.assertEqual((status, error), (201, None))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 0,
                "conflictPairs": 0,
                "maxCandidatesInKey": 1,
                "maxDistinctValuesInKey": 1,
            },
        )

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before = self.store.get_sync_operations(0, 100)[0]
        first = self.store.get_conflict_metrics()
        second = self.store.get_conflict_metrics()
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], before)
        # The existing metrics are untouched by the new read.
        self.assertEqual(self.store.get_metrics()["conflictKeys"], 1)


class ConflictMetricsRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_conflict_metrics_match_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "v1", {"r3": 1}))
        store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1}))
        # A stale write that adds no candidate.
        store.apply_operation("r1", operation("o5", "k", "old", {"r1": 0}))
        before = store.get_conflict_metrics()
        self.assertEqual(
            before,
            {
                "keys": 2,
                "conflictKeys": 1,
                "candidatePairs": 3,
                "conflictPairs": 2,
                "maxCandidatesInKey": 3,
                "maxDistinctValuesInKey": 2,
            },
        )

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_conflict_metrics(), before)
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).get_conflict_metrics(), before
        )


class ConflictMetricsHttpServerTests(unittest.TestCase):
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

    def raw_request(
        self, method: str, path: str, body: object = None
    ) -> tuple[int, dict, bytes, list[tuple[str, str]]]:
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

    def conflicts(self, path: str = "/v1/metrics/conflicts") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_metrics_are_zeroes(self) -> None:
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_CONFLICT_METRICS)

    def test_payload_has_exactly_six_non_negative_integer_fields(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload, raw, headers = self.raw_request("GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), CONFLICT_METRIC_FIELDS)
        for name, value in payload.items():
            self.assertIs(type(value), int, f"{name} must be an int, got {type(value)!r}")
            self.assertGreaterEqual(value, 0)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_counts_reflect_writes_and_repairs(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "keys": 1,
                "conflictKeys": 1,
                "candidatePairs": 1,
                "conflictPairs": 1,
                "maxCandidatesInKey": 2,
                "maxDistinctValuesInKey": 2,
            },
        )
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
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(payload["conflictKeys"], 0)
        self.assertEqual(payload["candidatePairs"], 0)
        self.assertEqual(payload["conflictPairs"], 0)

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/metrics/conflicts?x=1",
            "/v1/metrics/conflicts?after=0",
            "/v1/metrics/conflicts?x=",
            "/v1/metrics/conflicts?x",
            "/v1/metrics/conflicts?=1",
            "/v1/metrics/conflicts?x=1&x=2",
            "/v1/metrics/conflicts?keys=1",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.conflicts("/v1/metrics/conflicts?")
        self.assertEqual(status, 200)

    def test_wrong_path_shape_is_404(self) -> None:
        for path in (
            "/v1/metrics/conflicts/extra",
            "/v1/metrics/conflicts/",
            "/v1/metrics/conflict",
            "/metrics/conflicts",
            "/v1/conflicts",
        ):
            status, payload = self.conflicts(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_wrong_path_shape_is_404_even_with_query(self) -> None:
        status, payload = self.conflicts("/v1/metrics/conflicts/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_non_get_methods_are_404(self) -> None:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            status, payload, _, _ = self.raw_request(method, "/v1/metrics/conflicts")
            self.assertEqual(status, 404, method)
            self.assertEqual(payload, {"error": "not_found"}, method)

    def test_read_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, first = self.conflicts()
        for _ in range(3):
            status, payload = self.conflicts()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        # The accepted-operation log and the existing metrics are untouched.
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 2)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["conflictKeys"], 1)
        self.assertEqual(metrics["candidateVersions"], 2)

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                metrics = self.server.store.get_conflict_metrics()
                if metrics["conflictKeys"] > metrics["keys"]:
                    violations.append("conflictKeys > keys")
                if metrics["conflictPairs"] > metrics["candidatePairs"]:
                    violations.append("conflictPairs > candidatePairs")
                if metrics["maxDistinctValuesInKey"] > metrics["maxCandidatesInKey"]:
                    violations.append("maxDistinctValuesInKey > maxCandidatesInKey")
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
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["conflictKeys"], 1)
        # 40 concurrent distinct-value candidates: C(40, 2) pairs, all conflicts.
        self.assertEqual(payload["candidatePairs"], 780)
        self.assertEqual(payload["conflictPairs"], 780)
        self.assertEqual(payload["maxCandidatesInKey"], 40)
        self.assertEqual(payload["maxDistinctValuesInKey"], 40)


class ConflictMetricsAuthTests(unittest.TestCase):
    """Authentication and scope enforcement for GET /v1/metrics/conflicts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token=TOKEN
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def raw_get(self, path: str, headers: list[tuple[str, str]]) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        conn.close()
        return response.status, payload

    def test_missing_authorization_is_401(self) -> None:
        status, payload = self.raw_get("/v1/metrics/conflicts", [])
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_wrong_token_is_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts", [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_malformed_authorization_is_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts", [("Authorization", TOKEN)]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_duplicate_authorization_headers_are_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts",
            [
                ("Authorization", f"Bearer {TOKEN}"),
                ("Authorization", f"Bearer {TOKEN}"),
            ],
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_token_is_200(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts", [("Authorization", f"Bearer {TOKEN}")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), CONFLICT_METRIC_FIELDS)

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.raw_get("/health", [])
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


class ConflictMetricsScopePolicyTests(unittest.TestCase):
    """Scope-policy mode: read or admin scope is required, write is not enough."""

    POLICY = {
        "reader-token": frozenset({"read"}),
        "writer-token": frozenset({"write"}),
        "admin-token": frozenset({"read", "write", "admin"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(cls.POLICY)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, token: str | None) -> tuple[int, dict]:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/metrics/conflicts", headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        conn.close()
        return response.status, payload

    def test_read_scope_is_200(self) -> None:
        status, payload = self.get("reader-token")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), CONFLICT_METRIC_FIELDS)

    def test_admin_scope_covers_reads(self) -> None:
        status, _ = self.get("admin-token")
        self.assertEqual(status, 200)

    def test_write_only_token_is_403(self) -> None:
        status, payload = self.get("writer-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_unknown_token_is_401(self) -> None:
        status, payload = self.get("unknown-token")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})


class PersistentConflictMetricsHttpServerTests(unittest.TestCase):
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

    def request(
        self, server: SemanticStateServer, method: str, path: str, body: object = None
    ) -> tuple[int, dict]:
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
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_conflict_metrics_survive_restart(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        status, before = self.request(server, "GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_read_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/metrics/conflicts")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
