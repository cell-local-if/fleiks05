"""Tests for the optional ``--request-timeout-seconds`` receive deadline.

The option adds a configurable cumulative receive deadline per request:
the request line, the request headers, and a legally declared request
body must fully arrive within N seconds of the server starting to
process the request, or the request is answered with HTTP 408
``{"error":"request_timeout"}`` and the connection is closed without any
state being created or modified. A peer dribbling single bytes never
resets the deadline. When the option is omitted every behavior is
identical to the baseline.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    main,
)

EXPECTED_PREFIX = "semantic-state-engine: startup failed:"
TIMEOUT_BODY = b'{"error":"request_timeout"}'

OP_PATH = "/v1/replicas/r1/operations"


def operation_document(op_id: str = "op-1") -> dict:
    return {"operationId": op_id, "key": "k", "value": "v", "clock": {"r1": 1}}


def read_until_close(sock: socket.socket) -> bytes:
    chunks = []
    while True:
        data = sock.recv(65536)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


class ArgumentValidationTests(unittest.TestCase):
    """The bound is validated before any file is read or any port is bound."""

    def setUp(self) -> None:
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

    def test_omitted_option_checks_out_as_zero(self) -> None:
        code, stdout, stderr = self.run_main(["--check"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["requestTimeoutSeconds"], 0)

    def test_valid_bounds_are_reported(self) -> None:
        for token, expected in (("1", 1), ("300", 300), ("007", 7)):
            with self.subTest(token=token):
                code, stdout, stderr = self.run_main(
                    ["--check", "--request-timeout-seconds", token]
                )
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertEqual(json.loads(stdout)["requestTimeoutSeconds"], expected)

    def test_invalid_bounds_fail_with_exit_code_2(self) -> None:
        for token in (
            "0",
            "301",
            "-1",
            "+1",
            "1.5",
            " 1",
            "1 ",
            "",
            "abc",
            "1e2",
            "٣",  # Arabic-Indic digit: non-ASCII numerals are rejected
            "４２",  # full-width digits
        ):
            with self.subTest(token=token):
                code, stdout, stderr = self.run_main(
                    ["--check", "--request-timeout-seconds", token]
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
                self.assertNotIn("usage:", stderr)

    def test_invalid_bound_fails_before_any_file_is_read(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="sestate-timeout-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        missing = os.path.join(tmpdir, "absent")
        code, stdout, stderr = self.run_main(
            [
                "--check",
                "--request-timeout-seconds",
                "0",
                "--data-file",
                missing,
                "--auth-token-file",
                missing,
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
        # The rejected bound is reported, not the missing files.
        self.assertNotIn(missing, stderr)

    def test_repeated_checks_with_the_option_are_byte_identical(self) -> None:
        argv = ["--check", "--request-timeout-seconds", "5"]
        code1, stdout1, _ = self.run_main(argv)
        code2, stdout2, _ = self.run_main(argv)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(stdout1, stdout2)


class ReceiveDeadlineTests(unittest.TestCase):
    """Live-server behavior with a one-second receive deadline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-timeout-live-")
        cls.data_file = os.path.join(cls.tmpdir, "state.json")
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=cls.data_file,
            request_timeout_seconds=1,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, True)

    def connect(self) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.settimeout(5)
        return sock

    def test_stalled_request_line_gets_408_and_close(self) -> None:
        with self.connect() as sock:
            started = time.monotonic()
            response = read_until_close(sock)
            elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 4.0)
        status_line, _, body = response.partition(b"\r\n")
        self.assertIn(b" 408 ", status_line)
        self.assertIn(b"Content-Length: " + str(len(TIMEOUT_BODY)).encode(), response)
        self.assertTrue(response.endswith(TIMEOUT_BODY), response)

    def test_stalled_headers_get_408(self) -> None:
        with self.connect() as sock:
            sock.sendall(b"POST " + OP_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n")
            response = read_until_close(sock)
        self.assertIn(b" 408 ", response.split(b"\r\n", 1)[0])
        self.assertTrue(response.endswith(TIMEOUT_BODY), response)

    def test_dribbled_bytes_do_not_reset_the_deadline(self) -> None:
        with self.connect() as sock:
            started = time.monotonic()
            # One header byte every 0.3s: each byte arrives well inside any
            # per-read budget, so only a cumulative deadline can fire. The
            # server closes the connection once it does, which may break
            # the remaining sends.
            try:
                for _ in range(8):
                    sock.sendall(b"H")
                    time.sleep(0.3)
            except OSError:
                pass
            response = read_until_close(sock)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertIn(b" 408 ", response.split(b"\r\n", 1)[0])
        self.assertTrue(response.endswith(TIMEOUT_BODY), response)

    def test_stalled_body_gets_408_and_creates_nothing(self) -> None:
        with open(self.data_file, "rb") as handle:
            before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        body = json.dumps(operation_document("op-stalled")).encode()
        with self.connect() as sock:
            sock.sendall(
                b"POST "
                + OP_PATH.encode()
                + b" HTTP/1.1\r\nHost: x\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body[:5]
            )
            response = read_until_close(sock)
        self.assertIn(b" 408 ", response.split(b"\r\n", 1)[0])
        self.assertTrue(response.endswith(TIMEOUT_BODY), response)
        # The timed-out request created no state and left no files behind.
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)

    def test_complete_request_within_the_deadline_is_unchanged(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(
                "POST",
                OP_PATH,
                body=json.dumps(operation_document("op-fast")),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 201)
        finally:
            conn.close()

    def test_health_probe_within_the_deadline_is_unchanged(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["status"], "ok")
        finally:
            conn.close()

    def test_unknown_path_stays_404(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", "/nope")
            response = conn.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            conn.close()

    def test_invalid_content_length_is_400_without_waiting(self) -> None:
        with self.connect() as sock:
            started = time.monotonic()
            sock.sendall(
                b"POST "
                + OP_PATH.encode()
                + b" HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n"
            )
            response = read_until_close(sock)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.9)
        self.assertIn(b" 400 ", response.split(b"\r\n", 1)[0])
        self.assertTrue(response.endswith(b'{"error":"invalid_request"}'), response)

    def test_over_limit_content_length_is_413_without_waiting(self) -> None:
        with self.connect() as sock:
            started = time.monotonic()
            sock.sendall(
                b"POST "
                + OP_PATH.encode()
                + b" HTTP/1.1\r\nHost: x\r\nContent-Length: "
                + str(MAX_BODY_BYTES + 1).encode()
                + b"\r\n\r\n"
            )
            response = read_until_close(sock)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.9)
        self.assertIn(b" 413 ", response.split(b"\r\n", 1)[0])
        self.assertTrue(response.endswith(b'{"error":"payload_too_large"}'), response)

    def test_server_keeps_serving_after_a_timeout(self) -> None:
        with self.connect() as sock:
            response = read_until_close(sock)
        self.assertIn(b" 408 ", response.split(b"\r\n", 1)[0])
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(
                "POST",
                OP_PATH,
                body=json.dumps(operation_document("op-after-timeout")),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 201)
        finally:
            conn.close()


class ReceiveDeadlineAuthTests(unittest.TestCase):
    """Authentication and scope outcomes are unchanged under the deadline."""

    TOKEN = "s3cret-token_123"

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_token=cls.TOKEN,
            request_timeout_seconds=1,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, headers: dict) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(
                "POST",
                OP_PATH,
                body=json.dumps(operation_document()),
                headers=headers,
            )
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def test_missing_credentials_stay_401(self) -> None:
        status, payload = self.request({"Content-Type": "application/json"})
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_credentials_still_accept(self) -> None:
        status, _ = self.request(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.TOKEN}",
            }
        )
        self.assertEqual(status, 201)


class DisabledTimeoutTests(unittest.TestCase):
    """Without the option the server behaves exactly as the baseline."""

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

    def test_no_deadline_reader_is_installed(self) -> None:
        self.assertIsNone(self.server.request_timeout_seconds)

    def test_normal_request_is_unaffected(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(
                "POST",
                OP_PATH,
                body=json.dumps(operation_document()),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 201)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
