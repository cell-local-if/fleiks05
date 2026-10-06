"""Tests for the check-only ``--check`` startup entry point.

``--check`` validates the argument combination, the authentication
configuration, the clock-width bound, and the data file exactly as
startup would, but never binds a port and never creates, rewrites, or
deletes any file. On success it prints one compact JSON summary line on
stdout and exits 0; on any failure stdout stays empty and stderr carries
only the ``semantic-state-engine: startup failed: ...`` line with exit
code 2. Without ``--check`` every startup behavior is unchanged.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from semantic_state_engine import server as server_module
from semantic_state_engine.server import StateStore, main, parse_transaction_apply

EXPECTED_PREFIX = "semantic-state-engine: startup failed:"


def empty_data_file_summary(configured: bool = False) -> dict:
    return {
        "configured": configured,
        "operations": 0,
        "checkpoints": 0,
        "transactions": 0,
        "acks": 0,
        "repairs": 0,
        "policyEvents": 0,
        "compensations": 0,
    }


class CheckModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-check-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self._saved_bound = server_module._MAX_CLOCK_COMPONENTS
        self.addCleanup(self._restore_bound)

    def _restore_bound(self) -> None:
        server_module._MAX_CLOCK_COMPONENTS = self._saved_bound

    def run_main(self, argv: list) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def data_path(self, name: str = "state.json") -> str:
        return os.path.join(self.tmpdir, name)

    def write_data_file_with_transaction(self) -> str:
        path = self.data_path()
        store = StateStore(data_file=path)
        _, entries = parse_transaction_apply(
            {
                "transactionId": "tx-1",
                "operations": [
                    {
                        "key": "a",
                        "replicaId": "r1",
                        "operationId": "op-1",
                        "value": "1",
                        "clock": {"r1": 1},
                        "candidates": [],
                    },
                    {
                        "key": "b",
                        "replicaId": "r1",
                        "operationId": "op-2",
                        "value": "2",
                        "clock": {"r1": 2},
                        "candidates": [],
                    },
                ],
            }
        )
        status, *_ = store.apply_transaction("tx-1", entries)
        self.assertEqual(status, 201)
        return path

    def write_token_file(self, content: bytes, name: str = "token.txt") -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def test_plain_check_succeeds_anonymously_with_fixed_shape(self) -> None:
        code, stdout, stderr = self.run_main(["--check"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout,
            '{"status":"ok","authentication":"anonymous","dataFile":'
            '{"configured":false,"operations":0,"checkpoints":0,'
            '"transactions":0,"acks":0,"repairs":0,"policyEvents":0,'
            '"compensations":0},"maxClockComponents":null,'
            '"requestTimeoutSeconds":0}\n',
        )

    def test_check_without_data_file_creates_nothing(self) -> None:
        code, _, _ = self.run_main(["--check"])
        self.assertEqual(code, 0)
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_missing_data_file_fails_and_creates_no_empty_file(self) -> None:
        path = self.data_path()
        code, stdout, stderr = self.run_main(["--check", "--data-file", path])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
        self.assertFalse(os.path.lexists(path))

    def test_data_file_counts_are_the_recovered_totals(self) -> None:
        path = self.write_data_file_with_transaction()
        with open(path, "rb") as handle:
            before = handle.read()
        code, stdout, stderr = self.run_main(["--check", "--data-file", path])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        summary = json.loads(stdout)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["authentication"], "anonymous")
        self.assertEqual(summary["maxClockComponents"], None)
        expected = empty_data_file_summary(configured=True)
        expected["operations"] = 2
        expected["transactions"] = 1
        self.assertEqual(summary["dataFile"], expected)
        # The check never creates, rewrites, or deletes the data file.
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_old_format_file_recovers_missing_sections_as_empty(self) -> None:
        path = self.data_path()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "operations": [
                        {
                            "replicaId": "r1",
                            "operation": {
                                "operationId": "op-9",
                                "key": "z",
                                "value": "9",
                                "clock": {"r1": 1},
                            },
                        }
                    ],
                },
                handle,
            )
        code, stdout, _ = self.run_main(["--check", "--data-file", path])
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        expected = empty_data_file_summary(configured=True)
        expected["operations"] = 1
        self.assertEqual(summary["dataFile"], expected)

    def test_repeated_checks_are_byte_identical(self) -> None:
        path = self.write_data_file_with_transaction()
        argv = ["--check", "--data-file", path]
        code1, stdout1, _ = self.run_main(argv)
        code2, stdout2, _ = self.run_main(argv)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(stdout1, stdout2)

    def test_check_leaves_no_probe_files_behind(self) -> None:
        path = self.write_data_file_with_transaction()
        before = sorted(os.listdir(self.tmpdir))
        code, _, _ = self.run_main(["--check", "--data-file", path])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), before)

    def test_corrupt_data_file_fails_with_empty_stdout(self) -> None:
        path = self.data_path()
        with open(path, "wb") as handle:
            handle.write(b"{not json")
        code, stdout, stderr = self.run_main(["--check", "--data-file", path])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)

    def test_data_file_directory_target_is_rejected(self) -> None:
        code, stdout, stderr = self.run_main(["--check", "--data-file", self.tmpdir])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)

    def test_single_token_mode_is_reported(self) -> None:
        token_path = self.write_token_file(b"tok_abc123")
        with open(token_path, "rb") as handle:
            before = handle.read()
        code, stdout, _ = self.run_main(["--check", "--auth-token-file", token_path])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["authentication"], "single-token")
        # The check never modifies the authentication file.
        with open(token_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_scope_policy_mode_is_reported(self) -> None:
        policy_path = self.write_token_file(
            b'{"t1":["read","admin"],"t2":["write"]}', name="scopes.json"
        )
        code, stdout, _ = self.run_main(["--check", "--scope-policy-file", policy_path])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["authentication"], "scope-policy")

    def test_conflicting_auth_options_fail(self) -> None:
        token_path = self.write_token_file(b"tok_abc123")
        policy_path = self.write_token_file(b'{"t1":["read"]}', name="scopes.json")
        code, stdout, stderr = self.run_main(
            ["--check", "--auth-token-file", token_path, "--scope-policy-file", policy_path]
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)

    def test_missing_token_file_fails(self) -> None:
        code, stdout, stderr = self.run_main(
            ["--check", "--auth-token-file", os.path.join(self.tmpdir, "absent")]
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)

    def test_invalid_token_fails_without_leaking_it(self) -> None:
        secret = "super-secret-token"
        token_path = self.write_token_file(secret.encode("ascii") + b"\n")
        code, stdout, stderr = self.run_main(["--check", "--auth-token-file", token_path])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
        self.assertNotIn(secret, stderr)

    def test_invalid_scope_policy_fails_without_leaking_it(self) -> None:
        policy_path = self.write_token_file(b'{"hidden-token":["nope"]}', name="scopes.json")
        code, stdout, stderr = self.run_main(["--check", "--scope-policy-file", policy_path])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
        self.assertNotIn("hidden-token", stderr)

    def test_max_clock_components_is_reported(self) -> None:
        code, stdout, _ = self.run_main(["--check", "--max-clock-components", "8"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["maxClockComponents"], 8)

    def test_invalid_clock_bound_uses_only_the_startup_channel(self) -> None:
        for token in ("0", "1025", "-1", "1.5", "abc"):
            with self.subTest(token=token):
                code, stdout, stderr = self.run_main(
                    ["--check", "--max-clock-components", token]
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
                self.assertNotIn("usage:", stderr)

    def test_stored_clock_over_the_bound_fails(self) -> None:
        path = self.data_path()
        store = StateStore(data_file=path)
        store.apply_operation(
            "r1",
            {"operationId": "op-1", "key": "k", "value": "v", "clock": {"r1": 1, "r2": 1}},
        )
        code, stdout, stderr = self.run_main(
            ["--check", "--max-clock-components", "1", "--data-file", path]
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)

    def test_unrecognized_argument_uses_only_the_startup_channel(self) -> None:
        code, stdout, stderr = self.run_main(["--check", "--bogus"])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
        self.assertNotIn("usage:", stderr)


if __name__ == "__main__":
    unittest.main()
