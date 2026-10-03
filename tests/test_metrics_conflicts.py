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
AUTH_HEADER = f"Bearer {TOKEN}"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
POLICY = {
    READ_TOKEN: frozenset({"read"}),
    WRITE_TOKEN: frozenset({"write"}),
    ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
}


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

    def test_single_candidate_key_contributes_only_keys(self) -> None:
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
        self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 1,
                "conflictPairs": 0,
                "maxCandidatesInKey": 2,
                "maxDistinctValuesInKey": 1,
            },
        )

    def test_different_value_pair_is_a_conflict_pair(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 1,
                "conflictKeys": 1,
                "candidatePairs": 1,
                "conflictPairs": 1,
                "maxCandidatesInKey": 2,
                "maxDistinctValuesInKey": 2,
            },
        )

    def test_pairs_are_counted_once_with_mixed_values(self) -> None:
        # Three pairwise-concurrent candidates with values v1, v1, v2:
        # three unordered pairs, of which the two (v1, v2) pairs conflict.
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v1", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "v2", {"r3": 1}))
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

    def test_maxima_span_keys_and_pairs_accumulate(self) -> None:
        # k1: three candidates, values a/a/b -> 3 pairs, 2 conflict pairs.
        self.store.apply_operation("r1", operation("o1", "k1", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k1", "a", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k1", "b", {"r3": 1}))
        # k2: two candidates, values x/y -> 1 pair, 1 conflict pair.
        self.store.apply_operation("r1", operation("o4", "k2", "x", {"r1": 2}))
        self.store.apply_operation("r2", operation("o5", "k2", "y", {"r2": 2}))
        # k3: one candidate -> only keys.
        self.store.apply_operation("r1", operation("o6", "k3", "z", {"r1": 3}))
        self.assertEqual(
            self.store.get_conflict_metrics(),
            {
                "keys": 3,
                "conflictKeys": 2,
                "candidatePairs": 4,
                "conflictPairs": 3,
                "maxCandidatesInKey": 3,
                "maxDistinctValuesInKey": 2,
            },
        )

    def test_stale_write_adds_no_candidate_and_no_pairs(self) -> None:
        self.store.apply_operation("r1", operation("new", "k", "new", {"r1": 2}))
        self.store.apply_operation("r1", operation("old", "k", "old", {"r1": 1}))
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
        # The classification agrees with the existing metrics snapshot.
        metrics = self.store.get_metrics()
        self.assertEqual(first["keys"], metrics["keys"])
        self.assertEqual(first["conflictKeys"], metrics["conflictKeys"])


class ConflictMetricsRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_counters_match_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "v1", {"r3": 1}))
        store.apply_operation("r1", operation("o4", "other", "x", {"r1": 2}))
        before = store.get_conflict_metrics()

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
    ) -> tuple[int, dict, bytes, list]:
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

    def test_counters_reflect_writes_conflicts_and_repairs(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r3", operation("o3", "k", "v1", {"r3": 1}))
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "keys": 1,
                "conflictKeys": 1,
                "candidatePairs": 3,
                "conflictPairs": 2,
                "maxCandidatesInKey": 3,
                "maxDistinctValuesInKey": 2,
            },
        )
        status, _ = self.request(
            "POST",
            "/v1/states/k/resolve",
            resolution(
                "r4",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1, "r4": 1},
                [candidate("r1", "o1"), candidate("r2", "o2"), candidate("r3", "o3")],
            ),
        )
        self.assertEqual(status, 201)
        status, payload = self.conflicts()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "keys": 1,
                "conflictKeys": 0,
                "candidatePairs": 0,
                "conflictPairs": 0,
                "maxCandidatesInKey": 1,
                "maxDistinctValuesInKey": 1,
            },
        )

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
            "/v1/metrics/",
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
        first_status, first = self.conflicts()
        self.assertEqual(first_status, 200)
        for _ in range(3):
            status, payload = self.conflicts()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["keys"], first["keys"])
        self.assertEqual(metrics["conflictKeys"], first["conflictKeys"])
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(len(state["candidates"]), 2)

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                snapshot = self.server.store.get_conflict_metrics()
                if snapshot["conflictPairs"] > snapshot["candidatePairs"]:
                    violations.append("conflictPairs > candidatePairs")
                if snapshot["conflictKeys"] > snapshot["keys"]:
                    violations.append("conflictKeys > keys")
                if snapshot["maxDistinctValuesInKey"] > snapshot["maxCandidatesInKey"]:
                    violations.append("maxDistinctValuesInKey > maxCandidatesInKey")
                if snapshot["keys"] == 0 and any(snapshot.values()):
                    violations.append("empty store with a non-zero counter")
                if any(not isinstance(v, int) or v < 0 for v in snapshot.values()):
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
        self.assertEqual(payload["candidatePairs"], 40 * 39 // 2)
        self.assertEqual(payload["conflictPairs"], 40 * 39 // 2)
        self.assertEqual(payload["maxCandidatesInKey"], 40)
        self.assertEqual(payload["maxDistinctValuesInKey"], 40)


class ConflictMetricsAuthTests(unittest.TestCase):
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

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def request(
        self, method: str, path: str, auth: str | None = AUTH_HEADER
    ) -> tuple[int, dict, dict]:
        headers = {}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def assert_unauthorized(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_valid_token_gets_the_counters(self) -> None:
        status, payload, _ = self.request("GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_CONFLICT_METRICS)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", auth=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    def test_missing_authorization_header_is_401(self) -> None:
        status, payload, headers = self.request(
            "GET", "/v1/metrics/conflicts", auth=None
        )
        self.assert_unauthorized(status, payload, headers)

    def test_wrong_token_is_401(self) -> None:
        status, payload, headers = self.request(
            "GET", "/v1/metrics/conflicts", auth="Bearer wrong"
        )
        self.assert_unauthorized(status, payload, headers)

    def test_malformed_authorization_values_are_401(self) -> None:
        for value in (
            "Bearer",
            f"Bearer  {TOKEN}",
            f"bearer {TOKEN}",
            TOKEN,
            f"Bearer {TOKEN} extra",
            f"Basic {TOKEN}",
        ):
            with self.subTest(value=value):
                status, payload, headers = self.request(
                    "GET", "/v1/metrics/conflicts", auth=value
                )
                self.assert_unauthorized(status, payload, headers)

    def test_duplicate_authorization_headers_are_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/v1/metrics/conflicts")
        conn.putheader("Authorization", AUTH_HEADER)
        conn.putheader("Authorization", AUTH_HEADER)
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)


class ConflictMetricsScopePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(POLICY)
        )
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

    def request(self, path: str, token: str | None) -> tuple[int, dict, dict]:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def test_read_and_admin_scopes_get_the_counters(self) -> None:
        for token in (READ_TOKEN, ADMIN_TOKEN):
            with self.subTest(token=token):
                status, payload, _ = self.request("/v1/metrics/conflicts", token)
                self.assertEqual(status, 200)
                self.assertEqual(payload, EMPTY_CONFLICT_METRICS)

    def test_write_only_token_is_403(self) -> None:
        status, payload, headers = self.request("/v1/metrics/conflicts", WRITE_TOKEN)
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_missing_token_is_401(self) -> None:
        status, payload, headers = self.request("/v1/metrics/conflicts", None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("/health", None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


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

    def test_counters_survive_restart(self) -> None:
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
        status, before = self.request(server, "GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(before["conflictPairs"], 1)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_read_writes_nothing_to_disk(self) -> None:
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
            status, _ = self.request(server, "GET", "/v1/metrics/conflicts")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
