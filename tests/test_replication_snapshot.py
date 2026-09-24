"""Tests for the read-only replication-snapshot verification endpoint::

    GET /v1/replication/snapshot

It returns exactly seven fields — ``status`` (always ``"ok"``),
``candidateDigest`` and ``snapshotDigest`` (64 lowercase hex chars),
``logCursor``, ``keys``, ``candidateVersions``, and ``checkpoints`` —
computed from a single snapshot under the shared commit lock. The snapshot
digest covers the candidate digest, the log cursor, and the checkpoint
mapping, serialized in that order as a compact UTF-8 JSON array with
minimal string escaping. The success body ends with one newline and
carries an explicit Content-Length.

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
    return hashlib.sha256(
        _replication_snapshot_input(candidate_digest, log_cursor, checkpoints)
    ).hexdigest()


class SnapshotInputTests(unittest.TestCase):
    """The canonical snapshot-digest input byte format, pinned against literals."""

    def test_empty_snapshot_layout(self) -> None:
        self.assertEqual(
            _replication_snapshot_input(EMPTY_CANDIDATE_DIGEST, 0, {}),
            b'["' + EMPTY_CANDIDATE_DIGEST.encode("ascii") + b'",0,{}]',
        )

    def test_element_order_is_digest_cursor_checkpoints(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("ab" * 32, 7, {"peer-a": 2}),
            b'["' + b"ab" * 32 + b'",7,{"peer-a":2}]',
        )

    def test_checkpoints_sorted_by_peer_id_and_empty_mapping_kept(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("cd" * 32, 3, {"b": 1, "a": 2, "m": 0}),
            b'["' + b"cd" * 32 + b'",3,{"a":2,"b":1,"m":0}]',
        )

    def test_peer_id_escaping_is_minimal(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("ef" * 32, 1, {'q"\\\n': 1}),
            b'["' + b"ef" * 32 + b'",1,{"q\\"\\\\\\u000a":1}]',
        )

    def test_non_ascii_peer_ids_are_written_literally(self) -> None:
        self.assertEqual(
            _replication_snapshot_input("01" * 32, 2, {"péer→": 2}),
            '["'.encode("utf-8")
            + b"01" * 32
            + '",2,{"péer→":2}]'.encode("utf-8"),
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

    def test_candidate_digest_matches_verification_digest(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        snapshot = self.store.get_replication_snapshot()
        verification = self.store.get_verification_digest()
        self.assertEqual(snapshot["candidateDigest"], verification["digest"])
        self.assertEqual(snapshot["keys"], verification["keys"])
        self.assertEqual(
            snapshot["candidateVersions"], verification["candidateVersions"]
        )

    def test_log_cursor_tracks_accepted_operations(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        # A stale write is accepted into the log but adds no candidate.
        self.assertIs(
            self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        # Replays and conflicts never enter the log.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))
        snapshot = self.store.get_replication_snapshot()
        self.assertEqual(snapshot["logCursor"], 2)
        self.assertEqual(snapshot["logCursor"], self.store.get_metrics()["acceptedOperations"])

    def test_checkpoints_are_reported_and_move_the_snapshot_digest(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.store.get_replication_snapshot()
        status, error = self.store.save_checkpoint("peer-a", 1)
        self.assertIsNone(error)
        self.assertIs(status, HTTPStatus.OK)
        after = self.store.get_replication_snapshot()
        self.assertEqual(after["checkpoints"], {"peer-a": 1})
        # The candidate digest ignores checkpoints; the snapshot digest does not.
        self.assertEqual(after["candidateDigest"], before["candidateDigest"])
        self.assertNotEqual(after["snapshotDigest"], before["snapshotDigest"])
        self.assertEqual(
            after["snapshotDigest"],
            expected_snapshot_digest(after["candidateDigest"], 1, {"peer-a": 1}),
        )

    def test_snapshot_digest_matches_canonical_form(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "clé", "a\nb→", {"r2": 1, "r1": 2})
        )
        self.store.save_checkpoint("peer-a", 1)
        self.store.save_checkpoint("peer-b", 0)
        snapshot = self.store.get_replication_snapshot()
        candidate_digest = hashlib.sha256(
            _verification_digest_input(self.store._candidates)
        ).hexdigest()
        self.assertEqual(snapshot["candidateDigest"], candidate_digest)
        self.assertEqual(
            snapshot["snapshotDigest"],
            expected_snapshot_digest(
                candidate_digest, 1, {"peer-a": 1, "peer-b": 0}
            ),
        )

    def test_import_batch_moves_cursor_and_snapshot_digest(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.store.get_replication_snapshot()
        status, accepted, replayed = self.store.import_operations(
            [
                ("r2", operation("o2", "k", "w", {"r2": 1})),
                ("r3", operation("o3", "other", "x", {"r3": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (2, 0))
        after = self.store.get_replication_snapshot()
        self.assertEqual(after["logCursor"], 3)
        self.assertNotEqual(after["snapshotDigest"], before["snapshotDigest"])

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

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
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
        # Compact canonical JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(raw[:-1].decode("utf-8"), json.dumps(payload, separators=(",", ":"), sort_keys=True))
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_snapshot_reflects_log_and_checkpoints_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 2}
        )
        self.assertEqual(status, 200)
        status, payload = self.snapshot()
        self.assertEqual(status, 200)
        self.assertEqual(payload["logCursor"], 2)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 2)
        self.assertEqual(payload["checkpoints"], {"peer-a": 2})
        _, verification = self.request("GET", "/v1/verification/digest")
        self.assertEqual(payload["candidateDigest"], verification["digest"])
        self.assertEqual(
            payload["snapshotDigest"],
            expected_snapshot_digest(verification["digest"], 2, {"peer-a": 2}),
        )

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/replication/snapshot?x=1",
            "/v1/replication/snapshot?after=0",
            "/v1/replication/snapshot?x=",
            "/v1/replication/snapshot?x",
            "/v1/replication/snapshot?=1",
            "/v1/replication/snapshot?x=1&x=2",
            "/v1/replication/snapshot?status=ok",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.request("GET", "/v1/replication/snapshot?")
        self.assertEqual(status, 200)

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/snapshot/extra",
            "/v1/replication",
            "/v1/replication/snapshot/",
            "/v1/replication/snapshots",
        ):
            status, payload = self.snapshot(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_check(self) -> None:
        status, payload = self.snapshot("/v1/replication/snapshot/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/replication/snapshot", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_snapshot_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, first = self.snapshot()
        for _ in range(3):
            status, payload = self.snapshot()
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
                    violations.append("negative cursor")
                # The snapshot digest must always agree with the other
                # fields of the same response.
                if result["snapshotDigest"] != expected_snapshot_digest(
                    result["candidateDigest"],
                    result["logCursor"],
                    result["checkpoints"],
                ):
                    violations.append("snapshot digest disagrees with fields")
                for peer_id, cursor in result["checkpoints"].items():
                    if not peer_id or cursor < 0 or cursor > result["logCursor"]:
                        violations.append("checkpoint outside the log")

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
                if index % 8 == 0:
                    self.request(
                        "POST",
                        f"/v1/sync/peers/peer-{index}/checkpoint",
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


class SnapshotAuthTests(unittest.TestCase):
    """The endpoint authenticates like every other non-/health route."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        token_file = Path(self._tmp.name) / "token"
        token_file.write_text("s3cret", encoding="ascii")
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def raw_get(self, path: str, headers: dict | None = None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=5
        )
        conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        www = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, www

    def test_missing_or_mismatched_token_is_401(self) -> None:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
            status, payload, www = self.raw_get("/v1/replication/snapshot", headers)
            self.assertEqual(status, 401, headers)
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertEqual(www, "Bearer")

    def test_valid_token_reaches_the_endpoint(self) -> None:
        status, payload, _ = self.raw_get(
            "/v1/replication/snapshot", {"Authorization": "Bearer s3cret"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), SNAPSHOT_FIELDS)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.raw_get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
