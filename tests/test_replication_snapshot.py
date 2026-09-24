"""Tests for the read-only replication snapshot verification endpoint::

    GET /v1/replication/snapshot

It returns exactly seven fields — ``status`` (always ``"ok"``),
``candidateDigest`` and ``snapshotDigest`` (64 lowercase hex chars),
``logCursor``, ``keys``, ``candidateVersions``, and ``checkpoints`` —
computed from a single snapshot under the shared commit lock. The snapshot
digest covers the canonical JSON array ``[candidateDigest, logCursor,
checkpoints]`` with no whitespace, peer ids sorted lexicographically, and
the minimal digest string escaping.

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
    _replication_snapshot_input,
    _verification_digest_input,
)

SNAPSHOT_FIELDS = {
    "status",
    "candidateDigest",
    "snapshotDigest",
    "logCursor",
    "keys",
    "candidateVersions",
    "checkpoints",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

EMPTY_CANDIDATE_DIGEST = hashlib.sha256(b"[]").hexdigest()


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def expected_snapshot_digest(
    candidate_digest: str, log_cursor: int, checkpoints: dict
) -> str:
    canonical = (
        "["
        + json.dumps(candidate_digest, ensure_ascii=False)
        + ","
        + str(log_cursor)
        + ","
        + json.dumps(
            {peer: checkpoints[peer] for peer in sorted(checkpoints)},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "]"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SnapshotInputTests(unittest.TestCase):
    """The canonical snapshot-digest input byte format, pinned to literals."""

    def test_empty_store_input(self) -> None:
        self.assertEqual(
            _replication_snapshot_input(EMPTY_CANDIDATE_DIGEST, 0, {}),
            b'["' + EMPTY_CANDIDATE_DIGEST.encode("ascii") + b'",0,{}]',
        )

    def test_checkpoints_sorted_and_empty_mapping_preserved(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("ab", 3, {"peer-b": 2, "peer-a": 1}),
            b'["ab",3,{"peer-a":1,"peer-b":2}]',
        )
        self.assertEqual(
            _replication_snapshot_input("ab", 0, {}),
            b'["ab",0,{}]',
        )

    def test_string_escaping_is_minimal(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("cd", 1, {'p"\\\né→': 1}),
            '["cd",1,{"p\\"\\\\\\u000aé→":1}]'.encode("utf-8"),
        )


class SnapshotStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store(self) -> None:
        self.assertEqual(
            self.store.get_replication_snapshot(),
            {
                "status": "ok",
                "candidateDigest": EMPTY_CANDIDATE_DIGEST,
                "snapshotDigest": expected_snapshot_digest(
                    EMPTY_CANDIDATE_DIGEST, 0, {}
                ),
                "logCursor": 0,
                "keys": 0,
                "candidateVersions": 0,
                "checkpoints": {},
            },
        )

    def test_fields_track_state(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "other", "x", {"r3": 1}))
        status, error = self.store.save_checkpoint("peer-a", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)

        result = self.store.get_replication_snapshot()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["logCursor"], 3)
        self.assertEqual(result["keys"], 2)
        self.assertEqual(result["candidateVersions"], 3)
        self.assertEqual(result["checkpoints"], {"peer-a": 2})
        # The candidate digest reuses the verification-digest rules exactly.
        self.assertEqual(
            result["candidateDigest"], self.store.get_verification_digest()["digest"]
        )
        self.assertEqual(
            result["snapshotDigest"],
            expected_snapshot_digest(
                result["candidateDigest"], 3, {"peer-a": 2}
            ),
        )

    def test_snapshot_digest_covers_log_and_checkpoints(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        before = self.store.get_replication_snapshot()

        # A stale write moves the log cursor but not the candidates.
        self.assertIs(
            self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        after_stale = self.store.get_replication_snapshot()
        self.assertEqual(after_stale["candidateDigest"], before["candidateDigest"])
        self.assertEqual(after_stale["logCursor"], 2)
        self.assertNotEqual(after_stale["snapshotDigest"], before["snapshotDigest"])

        # A checkpoint moves the mapping but neither the log nor candidates.
        status, error = self.store.save_checkpoint("peer-a", 2)
        self.assertIsNone(error)
        after_checkpoint = self.store.get_replication_snapshot()
        self.assertEqual(after_checkpoint["logCursor"], 2)
        self.assertEqual(
            after_checkpoint["candidateDigest"], before["candidateDigest"]
        )
        self.assertNotEqual(
            after_checkpoint["snapshotDigest"], after_stale["snapshotDigest"]
        )

        # Replays and conflicts move nothing at all.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))
        self.assertEqual(
            self.store.get_replication_snapshot(), after_checkpoint
        )

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.store.save_checkpoint("peer-a", 1)
        first = self.store.get_replication_snapshot()
        second = self.store.get_replication_snapshot()
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        _, checkpoint = self.store.get_checkpoint("peer-a")
        self.assertEqual(checkpoint["cursor"], 1)


class SnapshotRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_snapshot_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        store.save_checkpoint("peer-a", 3)
        store.save_checkpoint("peer-b", 1)
        before = store.get_replication_snapshot()

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_replication_snapshot(), before)
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).get_replication_snapshot(), before
        )


class SnapshotHttpServerTests(unittest.TestCase):
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

    def snapshot(self, path: str = "/v1/replication/snapshot") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_store_snapshot(self) -> None:
        status, payload = self.snapshot()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "candidateDigest": EMPTY_CANDIDATE_DIGEST,
                "snapshotDigest": expected_snapshot_digest(
                    EMPTY_CANDIDATE_DIGEST, 0, {}
                ),
                "logCursor": 0,
                "keys": 0,
                "candidateVersions": 0,
                "checkpoints": {},
            },
        )

    def test_payload_shape_and_headers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request(
            "GET", "/v1/replication/snapshot"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), SNAPSHOT_FIELDS)
        self.assertEqual(payload["status"], "ok")
        for name in ("candidateDigest", "snapshotDigest"):
            self.assertIs(type(payload[name]), str)
            self.assertRegex(payload[name], DIGEST_RE)
        for name in ("logCursor", "keys", "candidateVersions"):
            self.assertIs(type(payload[name]), int, f"{name} must be an int")
            self.assertGreaterEqual(payload[name], 0)
        self.assertEqual(payload["checkpoints"], {})
        # Compact body terminated by exactly one newline, explicit length.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw[:-1])
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_snapshot_matches_components_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r3", operation("o3", "other", "x", {"r3": 1}))
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 2}
        )
        self.assertEqual(status, 200)

        status, payload = self.snapshot()
        self.assertEqual(status, 200)
        _, verification = self.request("GET", "/v1/verification/digest")
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(payload["candidateDigest"], verification["digest"])
        self.assertEqual(payload["keys"], metrics["keys"])
        self.assertEqual(payload["candidateVersions"], metrics["candidateVersions"])
        self.assertEqual(payload["logCursor"], metrics["acceptedOperations"])
        self.assertEqual(payload["checkpoints"], {"peer-a": 2})
        self.assertEqual(
            payload["snapshotDigest"],
            expected_snapshot_digest(
                verification["digest"], 3, {"peer-a": 2}
            ),
        )

    def test_non_ascii_peer_ids_are_written_literally(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _ = self.request(
            "POST", "/v1/sync/peers/p%C3%A9er-%E2%86%92/checkpoint", {"cursor": 1}
        )
        self.assertEqual(status, 200)
        status, payload, raw, _ = self.raw_request("GET", "/v1/replication/snapshot")
        self.assertEqual(status, 200)
        self.assertEqual(payload["checkpoints"], {"péer-→": 1})
        self.assertIn("péer-→".encode("utf-8"), raw)
        self.assertEqual(
            payload["snapshotDigest"],
            expected_snapshot_digest(
                payload["candidateDigest"], 1, {"péer-→": 1}
            ),
        )

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/replication/snapshot?x=1",
            "/v1/replication/snapshot?after=0",
            "/v1/replication/snapshot?x=",
            "/v1/replication/snapshot?x",
            "/v1/replication/snapshot?=1",
            "/v1/replication/snapshot?x=1&x=2",
        ):
            status, payload, raw, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)
            self.assertEqual(raw, b'{"error":"invalid_request"}\n', path)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.request("GET", "/v1/replication/snapshot?")
        self.assertEqual(status, 200)

    def test_extra_path_is_404(self) -> None:
        for path in (
            "/v1/replication/snapshot/extra",
            "/v1/replication",
            "/v1/replication/snapshot/",
            "/v1/replication/snapshots",
        ):
            status, payload = self.snapshot(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_extra_path_is_404_even_with_query(self) -> None:
        status, payload = self.snapshot("/v1/replication/snapshot/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/replication/snapshot", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_snapshot_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.request("POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 1})
        _, first = self.snapshot()
        for _ in range(3):
            status, payload = self.snapshot()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 1)
        status, checkpoint = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        self.assertEqual(status, 200)
        self.assertEqual(checkpoint["cursor"], 1)

    def test_concurrent_commits_observe_consistent_snapshots(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                result = self.server.store.get_replication_snapshot()
                if result["status"] != "ok":
                    violations.append("wrong status")
                if not DIGEST_RE.match(result["candidateDigest"]):
                    violations.append("malformed candidate digest")
                if not DIGEST_RE.match(result["snapshotDigest"]):
                    violations.append("malformed snapshot digest")
                if result["candidateVersions"] < result["keys"]:
                    violations.append("candidateVersions < keys")
                if result["logCursor"] < 0:
                    violations.append("negative log cursor")
                for peer, cursor in result["checkpoints"].items():
                    if cursor > result["logCursor"]:
                        violations.append(f"checkpoint {peer} past the log")

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
                self.request(
                    "POST",
                    f"/v1/sync/peers/peer-{index % 3}/checkpoint",
                    {"cursor": index + 1},
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.snapshot()
        self.assertEqual(status, 200)
        self.assertEqual(payload["logCursor"], 40)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 40)
        self.assertEqual(
            payload["checkpoints"],
            {"peer-0": 40, "peer-1": 38, "peer-2": 39},
        )


class PersistentSnapshotHttpServerTests(unittest.TestCase):
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

    def test_snapshot_survives_restart(self) -> None:
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
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o3", "k", "old", {"r1": 0}),
        )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 3})
        status, before = self.request(server, "GET", "/v1/replication/snapshot")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/replication/snapshot")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_snapshot_writes_nothing_to_disk(self) -> None:
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
            status, _ = self.request(server, "GET", "/v1/replication/snapshot")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
