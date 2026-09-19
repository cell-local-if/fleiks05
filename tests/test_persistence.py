"""Tests for optional file-backed persistence and crash recovery."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
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
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"


class DataFileStartupTests(TempDirTestCase):
    def test_missing_file_is_created_on_first_start(self) -> None:
        self.assertFalse(self.data_file.exists())
        store = StateStore(data_file=str(self.data_file))
        try:
            self.assertTrue(self.data_file.is_file())
            records = load_data_file(str(self.data_file))
            self.assertEqual(records, [])
        finally:
            del store

    def test_missing_parent_directory_refuses_start(self) -> None:
        path = self.tmp / "missing" / "dir" / "state.json"
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(path))
        self.assertFalse(Path(str(path)).exists())

    def test_directory_target_refuses_start(self) -> None:
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(self.tmp))

    def test_unreadable_parent_refuses_start(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory permission bits")
        locked = self.tmp / "locked"
        locked.mkdir()
        os.chmod(locked, 0o000)
        self.addCleanup(os.chmod, locked, 0o755)
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(locked / "state.json"))

    def test_existing_empty_file_is_rejected(self) -> None:
        self.data_file.write_bytes(b"")
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(self.data_file))

    def test_explicit_empty_log_is_accepted(self) -> None:
        self.data_file.write_text(json.dumps({"version": 1, "operations": []}), encoding="utf-8")
        store = StateStore(data_file=str(self.data_file))
        status, _ = store.get_state("anything")
        self.assertIs(status, HTTPStatus.NOT_FOUND)

    def test_file_descriptor_target_refuses_start(self) -> None:
        # FIFOs are not regular files and must not be treated as stores.
        fifo = self.tmp / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(fifo))


class CorruptDataFileTests(TempDirTestCase):
    def assert_rejected(self, raw: bytes) -> None:
        self.data_file.write_bytes(raw)
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(self.data_file))

    def test_malformed_json_is_rejected(self) -> None:
        self.assert_rejected(b"{not json")

    def test_truncated_json_is_rejected(self) -> None:
        good = json.dumps(
            {
                "version": 1,
                "operations": [
                    {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                ],
            }
        ).encode("utf-8")
        self.assert_rejected(good[: len(good) // 2])

    def test_invalid_utf8_is_rejected(self) -> None:
        self.assert_rejected(b"\xff\xfe")

    def test_wrong_root_shape_is_rejected(self) -> None:
        self.assert_rejected(b"[]")
        self.assert_rejected(b"{}")
        self.assert_rejected(b'{"version":1}')
        self.assert_rejected(b'{"operations":[]}')
        self.assert_rejected(b'{"version":1,"operations":[],"extra":1}')

    def test_unsupported_version_is_rejected(self) -> None:
        self.assert_rejected(b'{"version":2,"operations":[]}')
        self.assert_rejected(b'{"version":"1","operations":[]}')
        self.assert_rejected(b'{"version":true,"operations":[]}')

    def test_operations_must_be_a_list(self) -> None:
        self.assert_rejected(b'{"version":1,"operations":{}}')

    def test_record_shape_violations_are_rejected(self) -> None:
        valid_op = operation("o1", "k", "v", {"r1": 1})

        def record(raw_replica: object, raw_op: object) -> bytes:
            return json.dumps(
                {"version": 1, "operations": [{"replicaId": raw_replica, "operation": raw_op}]}
            ).encode("utf-8")

        self.assert_rejected(record("r1", ["not", "an", "object"]))
        self.assert_rejected(record("", valid_op))
        self.assert_rejected(record(42, valid_op))
        self.assert_rejected(record("r1", {}))
        self.assert_rejected(record("r1", {**valid_op, "extra": 1}))

    def test_input_constraint_violations_are_rejected(self) -> None:
        def record(raw_replica: str, raw_op: dict) -> bytes:
            return json.dumps(
                {"version": 1, "operations": [{"replicaId": raw_replica, "operation": raw_op}]}
            ).encode("utf-8")

        self.assert_rejected(record("r1", operation("", "k", "v", {"r1": 1})))
        self.assert_rejected(record("r1", operation("o1", "k", "", {"r1": 1})))
        self.assert_rejected(record("r1", operation("o1", "k", "v", {})))
        self.assert_rejected(record("r1", operation("o1", "k", "v", {"r1": -1})))
        self.assert_rejected(record("r1", operation("o1", "k", "v", {"r1": True})))
        self.assert_rejected(record("r1", operation("o1", "k", "v", {"r2": 1})))
        self.assert_rejected(record("r1", operation("o1", "k", 123, {"r1": 1})))  # type: ignore[arg-type]

    def test_duplicate_identities_are_rejected(self) -> None:
        document = {
            "version": 1,
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})},
                {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})},
            ],
        }
        self.assert_rejected(json.dumps(document).encode("utf-8"))


class PersistenceSemanticsTests(TempDirTestCase):
    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def read_records(self) -> list:
        return load_data_file(str(self.data_file))

    def test_first_accept_is_persisted_before_return(self) -> None:
        store = self.make_store()
        status = store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertIs(status, HTTPStatus.CREATED)
        records = self.read_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0], ("r1", operation("o1", "k", "v", {"r1": 1})))

    def test_stale_write_is_persisted_even_without_candidate(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        status = store.apply_operation("r1", operation("o2", "k", "stale", {"r1": 1}))
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(len(self.read_records()), 2)
        _, state = store.get_state("k")
        self.assertEqual(state["value"], "new")

        reloaded = self.make_store()
        status, state = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state, {"key": "k", "value": "new", "clock": {"r1": 2}, "status": "resolved"})
        # The stale operation identity was recovered too.
        replay = reloaded.apply_operation("r1", operation("o2", "k", "stale", {"r1": 1}))
        self.assertIs(replay, HTTPStatus.OK)
        self.assertEqual(len(self.read_records()), 2)

    def test_replay_adds_no_record(self) -> None:
        store = self.make_store()
        op = operation("o1", "k", "v", {"r1": 1})
        store.apply_operation("r1", op)
        status = store.apply_operation("r1", dict(op))
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(self.read_records()), 1)

    def test_conflict_changes_neither_memory_nor_file(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.data_file.read_bytes()
        status = store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1}))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(self.data_file.read_bytes(), before)
        _, state = store.get_state("k")
        self.assertEqual(state["value"], "v")
        # The conflicting content must not have been recorded as the identity.
        reloaded = self.make_store()
        status = reloaded.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertIs(status, HTTPStatus.OK)
        status = reloaded.apply_operation("r1", operation("o1", "k", "other", {"r1": 1}))
        self.assertIs(status, HTTPStatus.CONFLICT)

    def test_persistence_failure_leaves_state_untouched(self) -> None:
        store = self.make_store()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _ = store.get_state("k")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        # The previous (empty) store file is intact and reloadable.
        reloaded = self.make_store()
        status, _ = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        leftovers = [p for p in self.tmp.iterdir() if p.name.startswith(".sestate-")]
        self.assertEqual(leftovers, [])

    def test_no_temp_files_left_behind(self) -> None:
        store = self.make_store()
        for i in range(5):
            store.apply_operation("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i + 1}))
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_concurrent_accepts_keep_one_commit_order(self) -> None:
        store = self.make_store()
        count = 24
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                op = operation(f"op-{index}", "k", f"v{index}", {f"r{index}": 1})
                status = store.apply_operation(f"r{index}", op)
                assert status is HTTPStatus.CREATED
            except BaseException as exc:  # captured and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])

        # The file is a complete log containing exactly every commit.
        records = self.read_records()
        self.assertEqual(len(records), count)
        reloaded = self.make_store()
        status, state = reloaded.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "conflict")
        candidates = state["candidates"]
        self.assertEqual(len(candidates), count)
        self.assertEqual(
            candidates,
            sorted(candidates, key=lambda c: (c["replicaId"], c["operationId"])),
        )

    def test_recovery_matches_uninterrupted_run(self) -> None:
        """Replay the same history against a memory store and a restarting one."""
        timeline = [
            ("r1", operation("op-1", "k", "v1", {"r1": 1}), HTTPStatus.CREATED),
            ("r1", operation("op-1", "k", "v1", {"r1": 1}), HTTPStatus.OK),
            ("r1", operation("op-1", "k", "different", {"r1": 1}), HTTPStatus.CONFLICT),
            ("r2", operation("op-2", "k", "v2", {"r2": 1}), HTTPStatus.CREATED),
            ("r2", operation("op-3", "k", "stale", {"r2": 0}), HTTPStatus.CREATED),
            ("r1", operation("op-4", "k", "v3", {"r1": 2, "r2": 1}), HTTPStatus.CREATED),
        ]
        memory = StateStore()
        persistent = self.make_store()

        restart_after = {"op-2"}
        seen_ids: set[str] = set()
        for replica, op, expected in timeline:
            status_memory = memory.apply_operation(replica, op)
            status_persistent = persistent.apply_operation(replica, op)
            self.assertIs(status_memory, expected)
            self.assertIs(status_persistent, expected)
            self.assertEqual(memory.get_state("k"), persistent.get_state("k"))
            seen_ids.add(op["operationId"])
            if op["operationId"] in restart_after and status_persistent is expected:
                persistent = self.make_store()
                self.assertEqual(memory.get_state("k"), persistent.get_state("k"))

        # Final restart: resolved value, clock, replay 200 and conflict 409.
        persistent = self.make_store()
        self.assertEqual(
            persistent.get_state("k"),
            (
                HTTPStatus.OK,
                {"key": "k", "value": "v3", "clock": {"r1": 2, "r2": 1}, "status": "resolved"},
            ),
        )
        replay = persistent.apply_operation("r2", operation("op-2", "k", "v2", {"r2": 1}))
        self.assertIs(replay, HTTPStatus.OK)
        clash = persistent.apply_operation("r2", operation("op-2", "k", "tampered", {"r2": 1}))
        self.assertIs(clash, HTTPStatus.CONFLICT)
        # Replay/conflict after restart still append nothing.
        self.assertEqual(len(self.read_records()), 4)


class PersistentServerTests(TempDirTestCase):
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
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_state_survives_server_restart(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "color", "blue", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations",
            operation("o2", "color", "green", {"r2": 1}),
        )
        self.assertEqual(status, 201)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.request(server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})
        status, payload = self.request(server, "GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                {"value": "blue", "clock": {"r1": 1}, "replicaId": "r1", "operationId": "o1"},
                {"value": "green", "clock": {"r2": 1}, "replicaId": "r2", "operationId": "o2"},
            ],
        )
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "color", "blue", {"r1": 1}),
        )
        self.assertEqual(status, 200)


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class CommandLineTests(TempDirTestCase):
    def spawn(self, *extra: str) -> subprocess.Popen:
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "semantic_state_engine.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                *extra,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    def wait_for_health(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                conn.request("GET", "/health")
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status == 200:
                    return
            except OSError:
                time.sleep(0.05)
        self.fail("service did not become healthy")

    def post(self, replica: str, body: dict) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        status = conn.getresponse().status
        conn.close()
        return status

    def get(self, path: str) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def setUp(self) -> None:
        super().setUp()
        self.port = _free_port()

    def test_cli_creates_persists_and_recovers(self) -> None:
        proc = self.spawn("--data-file", str(self.data_file))
        try:
            self.wait_for_health()
            self.assertTrue(self.data_file.exists())
            self.assertEqual(
                self.post("r1", operation("o1", "color", "blue", {"r1": 1})), 201
            )
        finally:
            self.stop(proc)

        proc = self.spawn("--data-file", str(self.data_file))
        try:
            self.wait_for_health()
            status, payload = self.get("/v1/states/color")
            self.assertEqual(status, 200)
            self.assertEqual(
                payload,
                {"key": "color", "value": "blue", "clock": {"r1": 1}, "status": "resolved"},
            )
        finally:
            self.stop(proc)

    def test_cli_refuses_corrupt_file(self) -> None:
        self.data_file.write_bytes(b"{broken")
        proc = self.spawn("--data-file", str(self.data_file))
        try:
            stdout, stderr = proc.communicate(timeout=5)
        finally:
            self.stop(proc)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"startup failed", stderr)
        self.assertEqual(stdout, b"")

    def test_cli_refuses_missing_parent_directory(self) -> None:
        proc = self.spawn("--data-file", str(self.tmp / "nope" / "state.json"))
        try:
            _, stderr = proc.communicate(timeout=5)
        finally:
            self.stop(proc)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"startup failed", stderr)


if __name__ == "__main__":
    unittest.main()
