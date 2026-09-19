"""Tests for the read-only replica-convergence digest endpoint::

    GET /v1/verification/digest

It returns exactly four fields — ``algorithm`` (always ``"sha256"``),
``digest`` (64 lowercase hex chars), ``keys``, and ``candidateVersions`` —
computed from a single snapshot under the shared commit lock. The digest
covers only the current candidate sets, serialized as a compact UTF-8 JSON
array with fixed key/field ordering and minimal string escaping.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _verification_digest_input,
)

DIGEST_FIELDS = {"algorithm", "digest", "keys", "candidateVersions"}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


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


def expected_digest(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DigestInputTests(unittest.TestCase):
    """The canonical digest-input byte format, pinned against literals."""

    def test_empty_store_serializes_to_empty_array(self) -> None:
        self.assertEqual(_verification_digest_input({}), b"[]")

    def test_single_candidate_layout_and_field_order(self) -> None:
        candidates = {
            "k": [
                {
                    "value": "v",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o1",
                }
            ]
        }
        self.assertEqual(
            _verification_digest_input(candidates),
            b'[{"key":"k","candidates":[{"value":"v","clock":{"r1":1},'
            b'"replicaId":"r1","operationId":"o1"}]}]',
        )

    def test_keys_and_clock_components_are_sorted(self) -> None:
        candidates = {
            "b": [
                {
                    "value": "2",
                    "clock": {"z": 1, "a": 3, "m": 2},
                    "replicaId": "r1",
                    "operationId": "o1",
                }
            ],
            "a": [
                {
                    "value": "1",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o2",
                }
            ],
        }
        self.assertEqual(
            _verification_digest_input(candidates),
            b'[{"key":"a","candidates":[{"value":"1","clock":{"r1":1},'
            b'"replicaId":"r1","operationId":"o2"}]},'
            b'{"key":"b","candidates":[{"value":"2","clock":{"a":3,"m":2,"z":1},'
            b'"replicaId":"r1","operationId":"o1"}]}]',
        )

    def test_candidates_are_sorted_by_replica_then_operation(self) -> None:
        candidates = {
            "k": [
                {
                    "value": "x",
                    "clock": {"r2": 1},
                    "replicaId": "r2",
                    "operationId": "o1",
                },
                {
                    "value": "y",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o2",
                },
                {
                    "value": "z",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o1",
                },
            ]
        }
        self.assertEqual(
            _verification_digest_input(candidates),
            b'[{"key":"k","candidates":['
            b'{"value":"z","clock":{"r1":1},"replicaId":"r1","operationId":"o1"},'
            b'{"value":"y","clock":{"r1":1},"replicaId":"r1","operationId":"o2"},'
            b'{"value":"x","clock":{"r2":1},"replicaId":"r2","operationId":"o1"}'
            b"]}]",
        )

    def test_string_escaping_is_minimal(self) -> None:
        candidates = {
            'q"\\': [
                {
                    "value": "line\nbreak\ttabbell",
                    "clock": {"r1": 1},
                    "replicaId": "r1",
                    "operationId": "o1",
                }
            ]
        }
        # Only quotes, backslashes, and control characters are escaped;
        # control characters always use lowercase \u00XX, and every other
        # Unicode code point is written literally as UTF-8.
        expected = (
            '[{"key":"q\\"\\\\","candidates":[{"value":"line\\u000abreak\\u0009tab'
            '\\u0001bell\\u0007","clock":{"r1":1},"replicaId":"r1",'
            '"operationId":"o1"}]}]'
        )
        self.assertEqual(
            _verification_digest_input(candidates), expected.encode("utf-8")
        )

    def test_non_ascii_code_points_are_written_literally(self) -> None:
        candidates = {
            "clé→": [
                {
                    "value": "héllo→世界",
                    "clock": {"r1": 1},
                    "replicaId": "réplica-1",
                    "operationId": "o1",
                }
            ]
        }
        expected = (
            '[{"key":"clé→","candidates":[{"value":"héllo→世界",'
            '"clock":{"r1":1},"replicaId":"réplica-1","operationId":"o1"}]}]'
        )
        self.assertEqual(
            _verification_digest_input(candidates), expected.encode("utf-8")
        )


class DigestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store(self) -> None:
        self.assertEqual(
            self.store.get_verification_digest(),
            {
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "keys": 0,
                "candidateVersions": 0,
            },
        )

    def test_digest_matches_canonical_snapshot(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        result = self.store.get_verification_digest()
        self.assertEqual(result["algorithm"], "sha256")
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"key":"k","candidates":[{"value":"v","clock":{"r1":1},'
                '"replicaId":"r1","operationId":"o1"}]}]'
            ),
        )
        self.assertEqual(result["keys"], 1)
        self.assertEqual(result["candidateVersions"], 1)

    def test_digest_covers_candidates_not_log_stale_writes_or_checkpoints(
        self,
    ) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        before = self.store.get_verification_digest()

        # A stale write is accepted into the log but adds no candidate.
        self.assertIs(
            self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        # A checkpoint moves no candidate either.
        status, error = self.store.save_checkpoint("peer-a", 2)
        self.assertIsNone(error)
        # An identical replay and a conflicting rewrite change nothing.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))

        after = self.store.get_verification_digest()
        self.assertEqual(after, before)
        # The log and checkpoints really did move — only the digest ignores them.
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 2)
        _, checkpoint = self.store.get_checkpoint("peer-a")
        self.assertEqual(checkpoint["cursor"], 2)

    def test_digest_tracks_candidate_changes(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        first = self.store.get_verification_digest()
        # A concurrent write from another replica adds a candidate.
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        second = self.store.get_verification_digest()
        self.assertNotEqual(first["digest"], second["digest"])
        self.assertEqual(second["keys"], 1)
        self.assertEqual(second["candidateVersions"], 2)
        # A dominating write collapses the candidate set back to one.
        self.store.apply_operation(
            "r1", operation("o3", "k", "v3", {"r1": 2, "r2": 1})
        )
        third = self.store.get_verification_digest()
        self.assertNotEqual(second["digest"], third["digest"])
        self.assertEqual(third["candidateVersions"], 1)

    def test_digest_tracks_repairs(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        conflicted = self.store.get_verification_digest()
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
        self.assertIsNone(error)
        resolved = self.store.get_verification_digest()
        self.assertNotEqual(conflicted["digest"], resolved["digest"])
        self.assertEqual(resolved["candidateVersions"], 1)
        self.assertEqual(
            resolved["digest"],
            expected_digest(
                '[{"key":"k","candidates":[{"value":"merged",'
                '"clock":{"r1":1,"r2":1,"r3":1},"replicaId":"r3",'
                '"operationId":"fix-1"}]}]'
            ),
        )

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.store.get_sync_operations(0, 100)[0]
        first = self.store.get_verification_digest()
        second = self.store.get_verification_digest()
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)


class DigestRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_digest_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # A stale write and a checkpoint move the log and the checkpoint
        # section but not the candidate state.
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        store.save_checkpoint("peer-a", 3)
        store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1}))
        before = store.get_verification_digest()

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_verification_digest(), before)
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).get_verification_digest(), before
        )


class DigestHttpServerTests(unittest.TestCase):
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

    def digest(self, path: str = "/v1/verification/digest") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_store_digest(self) -> None:
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "keys": 0,
                "candidateVersions": 0,
            },
        )

    def test_payload_shape_and_headers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request(
            "GET", "/v1/verification/digest"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), DIGEST_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertIs(type(payload["digest"]), str)
        self.assertRegex(payload["digest"], DIGEST_RE)
        for name in ("keys", "candidateVersions"):
            self.assertIs(type(payload[name]), int, f"{name} must be an int")
            self.assertGreaterEqual(payload[name], 0)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_digest_matches_canonical_form_over_http(self) -> None:
        # Multi-component clock sent out of order, a control character, and
        # non-ASCII code points exercise the canonical byte format end to end.
        self.post_operation(
            "r1",
            operation("o1", "clé", "a\nb→", {"r2": 1, "r1": 2}),
        )
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["digest"],
            expected_digest(
                '[{"key":"clé","candidates":[{"value":"a\\u000ab→",'
                '"clock":{"r1":2,"r2":1},"replicaId":"r1","operationId":"o1"}]}]'
            ),
        )
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 1)

    def test_counts_agree_with_metrics(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r3", operation("o3", "other", "x", {"r3": 1}))
        status, payload = self.digest()
        self.assertEqual(status, 200)
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(payload["keys"], metrics["keys"])
        self.assertEqual(payload["candidateVersions"], metrics["candidateVersions"])

    def test_digest_ignores_stale_writes_and_checkpoints_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        _, before = self.digest()
        # Stale write: accepted (201) but adds no candidate.
        status, _ = self.post_operation("r1", operation("o2", "k", "old", {"r1": 1}))
        self.assertEqual(status, 201)
        # Checkpoint registration.
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 2}
        )
        self.assertEqual(status, 200)
        # Replay and conflicting rewrite.
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.post_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))
        _, after = self.digest()
        self.assertEqual(after, before)

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/verification/digest?x=1",
            "/v1/verification/digest?after=0",
            "/v1/verification/digest?x=",
            "/v1/verification/digest?x",
            "/v1/verification/digest?=1",
            "/v1/verification/digest?x=1&x=2",
            "/v1/verification/digest?digest=abc",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.request("GET", "/v1/verification/digest?")
        self.assertEqual(status, 200)

    def test_extra_path_is_404(self) -> None:
        for path in (
            "/v1/verification/digest/extra",
            "/v1/verification",
            "/v1/verification/digest/sha256",
            "/v1/verification/digests",
        ):
            status, payload = self.digest(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_extra_path_is_404_even_with_query(self) -> None:
        status, payload = self.digest("/v1/verification/digest/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/verification/digest", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_digest_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, first = self.digest()
        for _ in range(3):
            status, payload = self.digest()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 1)
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")

    def test_concurrent_commits_observe_consistent_snapshots(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                result = self.server.store.get_verification_digest()
                if result["algorithm"] != "sha256":
                    violations.append("wrong algorithm")
                if not DIGEST_RE.match(result["digest"]):
                    violations.append("malformed digest")
                if result["candidateVersions"] < result["keys"]:
                    violations.append("candidateVersions < keys")
                if result["keys"] < 0 or result["candidateVersions"] < 0:
                    violations.append("negative counter")

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
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 40)
        # The final digest is exactly the hash of the final candidate set.
        self.assertEqual(
            payload["digest"],
            expected_digest(
                '[{"key":"shared","candidates":['
                + ",".join(
                    f'{{"value":"v{index}","clock":{{"r{index}":1}},'
                    f'"replicaId":"r{index}","operationId":"op-{index}"}}'
                    for index in self._sorted_replica_order(40)
                )
                + "]}]"
            ),
        )

    @staticmethod
    def _sorted_replica_order(count: int) -> list[int]:
        return sorted(range(count), key=lambda i: (f"r{i}", f"op-{i}"))


class PersistentDigestHttpServerTests(unittest.TestCase):
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
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_digest_survives_restart(self) -> None:
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
        # A stale write and a checkpoint are persisted too; neither may
        # influence the digest.
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o3", "k", "old", {"r1": 0}),
        )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 3})
        status, before = self.request(server, "GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_digest_writes_nothing_to_disk(self) -> None:
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
            status, _ = self.request(server, "GET", "/v1/verification/digest")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
