"""Tests for the optional ``--max-clock-components`` admission bound.

With the option absent every behavior is the baseline one. With the
option set, every vector clock participating in a causal decision —
write and sync-import ``operation.clock``, manual and automatic
resolution clocks, transaction and compensation entry clocks,
replication compare/plan/consensus/apply snapshot and action clocks, and
the single-key and cross-key causal-at boundary clocks — may hold at
most N components; an over-wide clock is HTTP 400 with
``{"error":"invalid_request"}`` before any business state is read, and
startup recovery applies the same bound to every stored clock.

The in-process tests enable the bound through the module configuration
exactly as ``main()`` does; the CLI tests spawn the real service.
"""

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
from pathlib import Path

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIMIT = 2


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def wide_clock(*names: str) -> dict:
    names = names or ("r1", "r2", "r3")
    return {name: 1 for name in names}


class MaxClockComponentsValueTests(unittest.TestCase):
    """The argparse type accepts only a decimal integer in 1..1024."""

    def parse(self, token: str) -> int:
        return server_module._max_clock_components_value(token)

    def test_boundaries_and_leading_zeros_are_accepted(self) -> None:
        self.assertEqual(self.parse("1"), 1)
        self.assertEqual(self.parse("1024"), 1024)
        self.assertEqual(self.parse("007"), 7)
        self.assertEqual(self.parse("42"), 42)

    def test_out_of_range_is_rejected(self) -> None:
        for token in ("0", "00", "1025", "99999999999999999999999999"):
            with self.assertRaises(Exception, msg=token):
                self.parse(token)

    def test_non_decimal_tokens_are_rejected(self) -> None:
        for token in ("", "-1", "+1", "1.0", "1e3", "0x10", " 1", "1 ", "1_0", "abc"):
            with self.assertRaises(Exception, msg=repr(token)):
                self.parse(token)


class HttpMaxClockComponentsTests(unittest.TestCase):
    """Endpoint semantics with the admission bound enabled (limit 2)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]
        cls._saved = server_module._MAX_CLOCK_COMPONENTS
        server_module._MAX_CLOCK_COMPONENTS = LIMIT

    @classmethod
    def tearDownClass(cls) -> None:
        server_module._MAX_CLOCK_COMPONENTS = cls._saved
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def request(self, method: str, path: str, body: object = None):
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
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def metrics(self) -> dict:
        status, payload = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        return payload

    def sync_operations(self) -> list:
        status, payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        return payload["operations"]

    def seed_conflict(self, key: str = "color") -> None:
        status, _ = self.post_operation("r1", operation(f"o1-{key}", key, "blue", {"r1": 1}))
        self.assertEqual(status, 201)
        status, _ = self.post_operation("r2", operation(f"o2-{key}", key, "red", {"r2": 1}))
        self.assertEqual(status, 201)

    # --- writes -----------------------------------------------------

    def test_write_at_limit_is_accepted(self) -> None:
        status, _ = self.post_operation("r1", operation("o1", "k", "v", {"r1": 1, "r2": 1}))
        self.assertEqual(status, 201)

    def test_write_over_limit_is_400_and_leaves_no_trace(self) -> None:
        before = self.metrics()
        status, payload = self.post_operation(
            "r1", operation("o1", "k", "v", wide_clock())
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.metrics(), before)
        self.assertEqual(self.sync_operations(), [])
        status, payload = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_replay_idempotency_is_preserved(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        # Identical replay is still answered from the committed binding.
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 200)
        # Same identity with different narrow content is still a conflict.
        status, payload = self.post_operation("r1", operation("o1", "k", "w", {"r1": 1}))
        self.assertEqual(status, 409)
        # Same identity carrying an over-wide clock is a fresh 400 and
        # never rewrites the committed binding.
        status, payload = self.post_operation(
            "r1", operation("o1", "k", "w", wide_clock())
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v")
        self.assertEqual(len(self.sync_operations()), 1)

    # --- sync import -------------------------------------------------

    def test_sync_import_with_one_wide_clock_is_atomic_400(self) -> None:
        body = {
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "a", "v", {"r1": 1})},
                {"replicaId": "r2", "operation": operation("o2", "b", "w", wide_clock("r2", "r3", "r4"))},
            ]
        }
        status, payload = self.request("POST", "/v1/sync/operations", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.sync_operations(), [])

    # --- resolutions -------------------------------------------------

    def test_manual_resolve_with_wide_clock_is_400(self) -> None:
        self.seed_conflict()
        body = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "blue",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1-color"},
                {"replicaId": "r2", "operationId": "o2-color"},
            ],
        }
        status, payload = self.request("POST", "/v1/states/color/resolve", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")

    def test_auto_resolve_with_wide_clock_is_400(self) -> None:
        self.seed_conflict()
        body = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        status, payload = self.request("POST", "/v1/states/color/resolve/auto", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload = self.request("GET", "/v1/states/color")
        self.assertEqual(payload["status"], "conflict")

    def test_auto_resolve_batch_with_one_wide_clock_is_atomic_400(self) -> None:
        self.seed_conflict("color")
        self.seed_conflict("size")
        body = {
            "resolutions": [
                {
                    "key": "color",
                    "replicaId": "r3",
                    "operationId": "fix-1",
                    "clock": {"r1": 1, "r3": 1},
                    "policy": "lowest_identity",
                },
                {
                    "key": "size",
                    "replicaId": "r3",
                    "operationId": "fix-2",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "policy": "lowest_identity",
                },
            ]
        }
        status, payload = self.request("POST", "/v1/resolve/auto/batch", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The narrow entry did not commit either: the batch is atomic.
        status, payload = self.request("GET", "/v1/states/color")
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(len(self.sync_operations()), 4)

    # --- transactions and compensations ------------------------------

    def test_transaction_with_one_wide_entry_is_atomic_400(self) -> None:
        body = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "a",
                    "replicaId": "r1",
                    "operationId": "t-op-1",
                    "value": "v",
                    "clock": {"r1": 1},
                    "candidates": [],
                },
                {
                    "key": "b",
                    "replicaId": "r2",
                    "operationId": "t-op-2",
                    "value": "w",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "candidates": [],
                },
            ],
        }
        status, payload = self.request("POST", "/v1/transactions/apply", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.sync_operations(), [])

    def test_compensation_with_wide_entry_clock_is_400(self) -> None:
        body = {
            "compensationId": "comp-1",
            "expectedPlanDigest": "0" * 64,
            "operations": [
                {
                    "key": "a",
                    "replicaId": "r1",
                    "operationId": "c-op-1",
                    "value": "v",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                }
            ],
        }
        status, payload = self.request("POST", "/v1/transactions/tx-1/compensate", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.sync_operations(), [])

    # --- causal-at boundaries ----------------------------------------

    def test_single_key_causal_at_boundary_over_limit_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        status, _ = self.request(
            "POST", "/v1/states/color/causal-at", {"clock": {"r1": 1, "r2": 1}}
        )
        self.assertEqual(status, 200)
        status, payload = self.request(
            "POST", "/v1/states/color/causal-at", {"clock": wide_clock()}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_cross_key_causal_at_boundary_over_limit_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        status, _ = self.request(
            "POST",
            "/v1/states/causal-at",
            {"clock": {"r1": 1, "r2": 1}, "keys": ["color"]},
        )
        self.assertEqual(status, 200)
        status, payload = self.request(
            "POST",
            "/v1/states/causal-at",
            {"clock": wide_clock(), "keys": ["color"]},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # --- replication endpoints ---------------------------------------

    def snapshot(self, clock: dict) -> dict:
        return {
            "color": [
                {
                    "value": "blue",
                    "clock": clock,
                    "replicaId": "r9",
                    "operationId": "x1",
                }
            ]
        }

    def test_replication_compare_with_wide_snapshot_clock_is_400(self) -> None:
        body = {"replicaId": "r9", "snapshot": self.snapshot({"r9": 1, "r1": 1})}
        status, _ = self.request("POST", "/v1/replication/compare", body)
        self.assertEqual(status, 200)
        body = {"replicaId": "r9", "snapshot": self.snapshot(wide_clock("r9", "r1", "r2"))}
        status, payload = self.request("POST", "/v1/replication/compare", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_replication_plan_with_wide_snapshot_clock_is_400(self) -> None:
        body = {"replicaId": "r9", "snapshot": self.snapshot(wide_clock("r9", "r1", "r2"))}
        status, payload = self.request("POST", "/v1/replication/plan", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_replication_consensus_with_wide_snapshot_clock_is_400(self) -> None:
        body = [
            {"replicaId": "r9", "snapshot": self.snapshot(wide_clock("r9", "r1", "r2"))},
            {"replicaId": "r8", "snapshot": self.snapshot({"r9": 1, "r1": 1})},
        ]
        # The second source must belong to its own replica id.
        body[1]["snapshot"]["color"][0]["replicaId"] = "r8"
        status, payload = self.request("POST", "/v1/replication/consensus", body)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_replication_apply_with_wide_clocks_is_400(self) -> None:
        base = {
            "replicaId": "r9",
            "expectedLocalDigest": "0" * 64,
            "snapshot": self.snapshot({"r9": 1, "r1": 1}),
            "actions": [
                {
                    "action": "fetch_remote",
                    "key": "color",
                    "replicaId": "r9",
                    "operationId": "x1",
                    "value": "blue",
                    "clock": {"r9": 1, "r1": 1},
                }
            ],
        }
        wide_action = json.loads(json.dumps(base))
        wide_action["actions"][0]["clock"] = {"r9": 1, "r1": 1, "r2": 1}
        status, payload = self.request("POST", "/v1/replication/apply", wide_action)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        wide_snapshot = json.loads(json.dumps(base))
        wide_snapshot["snapshot"]["color"][0]["clock"] = {"r9": 1, "r1": 1, "r2": 1}
        status, payload = self.request("POST", "/v1/replication/apply", wide_snapshot)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.sync_operations(), [])


class HttpNoBoundTests(unittest.TestCase):
    """Without the option the baseline admits clocks of any width."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved = server_module._MAX_CLOCK_COMPONENTS
        server_module._MAX_CLOCK_COMPONENTS = None
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        server_module._MAX_CLOCK_COMPONENTS = cls._saved
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_wide_clock_is_accepted_without_the_bound(self) -> None:
        clock = {f"r{i}": 1 for i in range(50)}
        clock["r1"] = 1
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r1/operations",
            body=json.dumps(operation("o1", "k", "v", clock)),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class CommandLineTests(unittest.TestCase):
    """The ``--max-clock-components`` option at the real CLI boundary."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"
        self.port = _free_port()

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

    def assert_not_listening(self) -> None:
        # A startup failure must happen before the socket starts accepting;
        # the process has exited and the port immediately refuses connections.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.5)
                conn.request("GET", "/health")
                conn.getresponse()
                conn.close()
                self.fail("service accepted a connection after startup failure")
            except OSError:
                return

    def post(self, replica: str, body: dict) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def get(self, path: str) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_invalid_values_exit_2_before_any_startup_work(self) -> None:
        for token in ("0", "1025", "-1", "+1", "1.5", "abc", "", " 2"):
            with self.subTest(token=token):
                proc = self.spawn(
                    "--max-clock-components",
                    token,
                    "--data-file",
                    str(self.data_file),
                )
                try:
                    _, stderr = proc.communicate(timeout=5)
                finally:
                    if proc.poll() is None:
                        self.stop(proc)
                self.assertEqual(proc.returncode, 2)
                self.assertIn(b"max-clock-components", stderr)
                # The rejected option never reaches data-file creation.
                self.assertFalse(self.data_file.exists())
                self.assert_not_listening()

    def test_invalid_value_is_rejected_before_auth_files_are_read(self) -> None:
        # A nonexistent token file would fail startup on its own; the
        # argparse rejection of the bound must come first and must not be
        # the auth or data-file failure path.
        proc = self.spawn(
            "--max-clock-components",
            "0",
            "--auth-token-file",
            str(self.tmp / "missing-token"),
            "--data-file",
            str(self.data_file),
        )
        try:
            _, stderr = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"max-clock-components", stderr)
        self.assertNotIn(b"startup failed", stderr)
        self.assertFalse(self.data_file.exists())
        self.assert_not_listening()

    def test_valid_bound_serves_and_enforces(self) -> None:
        proc = self.spawn("--max-clock-components", "2")
        try:
            self.wait_for_health()
            status, _ = self.post("r1", operation("o1", "k", "v", {"r1": 1, "r2": 1}))
            self.assertEqual(status, 201)
            status, payload = self.post("r1", operation("o2", "k", "w", wide_clock()))
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})
        finally:
            self.stop(proc)

    def test_boundary_values_are_accepted(self) -> None:
        for token in ("1", "1024"):
            with self.subTest(token=token):
                proc = self.spawn("--max-clock-components", token)
                try:
                    self.wait_for_health()
                finally:
                    self.stop(proc)

    def write_data_file(self, document: dict) -> None:
        self.data_file.write_text(json.dumps(document), encoding="utf-8")

    def test_recovery_rejects_a_stored_over_wide_operation_clock(self) -> None:
        self.write_data_file(
            {
                "version": 1,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("o1", "k", "v", {"r1": 1, "r2": 1}),
                    }
                ],
            }
        )
        before = self.data_file.read_bytes()
        proc = self.spawn(
            "--max-clock-components", "1", "--data-file", str(self.data_file)
        )
        try:
            _, stderr = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"startup failed", stderr)
        # The rejected file is left byte for byte unchanged.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assert_not_listening()
        # Without the bound the same file still recovers.
        proc = self.spawn("--data-file", str(self.data_file))
        try:
            self.wait_for_health()
            status, payload = self.get("/v1/states/k")
            self.assertEqual(status, 200)
            self.assertEqual(payload["value"], "v")
        finally:
            self.stop(proc)

    def test_recovery_rejects_a_stored_over_wide_transaction_clock(self) -> None:
        entry = {
            "key": "k",
            "replicaId": "r1",
            "operationId": "o1",
            "value": "v",
            "clock": {"r1": 1, "r2": 1},
            "candidates": [],
        }
        self.write_data_file(
            {
                "version": 1,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("o1", "k", "v", {"r1": 1}),
                    }
                ],
                "transactions": [{"transactionId": "tx-1", "operations": [entry]}],
            }
        )
        proc = self.spawn(
            "--max-clock-components", "1", "--data-file", str(self.data_file)
        )
        try:
            _, stderr = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"startup failed", stderr)
        self.assert_not_listening()

    def test_recovery_with_compliant_file_matches_unbounded_behavior(self) -> None:
        self.write_data_file(
            {
                "version": 1,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("o1", "k", "v", {"r1": 1, "r2": 1}),
                    }
                ],
            }
        )
        proc = self.spawn(
            "--max-clock-components", "2", "--data-file", str(self.data_file)
        )
        try:
            self.wait_for_health()
            status, payload = self.get("/v1/states/k")
            self.assertEqual(status, 200)
            self.assertEqual(
                payload,
                {
                    "key": "k",
                    "value": "v",
                    "clock": {"r1": 1, "r2": 1},
                    "status": "resolved",
                },
            )
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
