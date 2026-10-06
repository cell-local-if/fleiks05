"""Tests for the optional ``--request-timeout-seconds`` receive deadline.

The option sets one cumulative receive deadline per request: the request
line, the headers, and a legally declared body must fully arrive within
the configured number of seconds, counted from when the server starts
processing the request. An overdue request is answered with HTTP 408 and
a fixed ``{"error":"request_timeout"}`` body, the connection is closed,
and no state is created or modified. A missing, malformed, or
conflicting Content-Length is still an immediate HTTP 400 and an
over-limit declared length an immediate HTTP 413 — neither waits for the
deadline. Without the option every behavior is exactly the baseline's.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    main,
)

EXPECTED_PREFIX = "semantic-state-engine: startup failed:"

OP_PATH = "/v1/replicas/r1/operations"
TOKEN = "s3cret-token_123"

TIMEOUT_BODY = b'{"error":"request_timeout"}'


def operation_body(op_id: str = "op-1", key: str = "k", value: str = "v") -> bytes:
    return json.dumps(
        {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}
    ).encode("utf-8")


def post_request_bytes(body: bytes, extra_headers: bytes = b"") -> bytes:
    return (
        f"POST {OP_PATH} HTTP/1.0\r\n".encode("ascii")
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + extra_headers
        + b"\r\n"
        + body
    )


def read_until_close(sock: socket.socket, limit: float = 10.0) -> bytes:
    """Read until the server closes the connection."""
    sock.settimeout(limit)
    chunks = []
    while True:
        data = sock.recv(65536)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


def parse_response(raw: bytes) -> tuple[int, dict, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split(b" ", 2)[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(b":")
        headers[name.strip().decode("ascii").lower()] = value.strip().decode("ascii")
    return status, headers, body


class CliValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-timeout-cli-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def run_main(self, argv: list) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_omitted_option_reports_zero(self) -> None:
        code, stdout, stderr = self.run_main(["--check"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["requestTimeoutSeconds"], 0)

    def test_valid_values_are_reported(self) -> None:
        for token, expected in (("1", 1), ("300", 300), ("42", 42), ("007", 7)):
            with self.subTest(token=token):
                code, stdout, stderr = self.run_main(
                    ["--check", "--request-timeout-seconds", token]
                )
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertEqual(json.loads(stdout)["requestTimeoutSeconds"], expected)

    def test_repeated_checks_are_byte_identical(self) -> None:
        argv = ["--check", "--request-timeout-seconds", "5"]
        code1, stdout1, _ = self.run_main(argv)
        code2, stdout2, _ = self.run_main(argv)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(stdout1, stdout2)

    def test_invalid_values_fail_before_any_file_or_port(self) -> None:
        for token in (
            "0",
            "301",
            "1000",
            "-1",
            "+5",
            "1.5",
            "1e2",
            "abc",
            "",
            " 5",
            "5 ",
            "\t5",
            "1 0",
            "٣",  # Arabic-Indic digit three
            "５",  # fullwidth digit five
            "999999999999999999999999",
        ):
            with self.subTest(token=token):
                code, stdout, stderr = self.run_main(
                    ["--check", "--request-timeout-seconds", token]
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertTrue(stderr.startswith(EXPECTED_PREFIX), stderr)
                self.assertNotIn("usage:", stderr)

    def test_invalid_value_fails_without_check_too(self) -> None:
        code, stdout, stderr = self.run_main(["--request-timeout-seconds", "0"])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")

    def test_missing_value_fails(self) -> None:
        code, stdout, _ = self.run_main(["--check", "--request-timeout-seconds"])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")

    def test_summary_shape_with_timeout_configured(self) -> None:
        code, stdout, _ = self.run_main(["--check", "--request-timeout-seconds", "7"])
        self.assertEqual(code, 0)
        self.assertEqual(
            stdout,
            '{"status":"ok","authentication":"anonymous","dataFile":'
            '{"configured":false,"operations":0,"checkpoints":0,'
            '"transactions":0,"acks":0,"repairs":0,"policyEvents":0,'
            '"compensations":0},"maxClockComponents":null,'
            '"requestTimeoutSeconds":7}\n',
        )


class TimedServerTests(unittest.TestCase):
    """Live-server tests with a one-second receive deadline."""

    def start_server(self, **kwargs) -> None:
        self.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.addCleanup(self.stop_server)

    def stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def connect(self) -> socket.socket:
        return socket.create_connection(("127.0.0.1", self.port), timeout=10)

    def get_state(self, key: str) -> int:
        with self.connect() as sock:
            sock.sendall(f"GET /v1/states/{key} HTTP/1.0\r\n\r\n".encode("ascii"))
            status, _, _ = parse_response(read_until_close(sock))
            return status

    def test_stalled_headers_get_408_and_close(self) -> None:
        self.start_server(request_timeout_seconds=1)
        started = time.monotonic()
        with self.connect() as sock:
            sock.sendall(b"POST /v1/replicas/r1/operations HTTP/1.0\r\n")
            raw = read_until_close(sock)
        elapsed = time.monotonic() - started
        status, headers, body = parse_response(raw)
        self.assertEqual(status, 408)
        self.assertEqual(body, TIMEOUT_BODY)
        self.assertEqual(headers["content-length"], str(len(TIMEOUT_BODY)))
        self.assertGreaterEqual(elapsed, 0.8)
        self.assertLess(elapsed, 5.0)
        # Nothing was created: the key the operation targeted stays absent.
        self.assertEqual(self.get_state("k"), 404)

    def test_dribbling_bytes_do_not_reset_the_deadline(self) -> None:
        self.start_server(request_timeout_seconds=1)
        request = post_request_bytes(operation_body())
        started = time.monotonic()
        with self.connect() as sock:
            # One byte every 0.1s for 0.9s, then silence: a per-byte reset
            # would only fire a full second after the last byte (~1.9s);
            # the cumulative deadline must fire at ~1s from request start.
            while time.monotonic() - started < 0.9:
                sock.sendall(request[:1])
                request = request[1:]
                time.sleep(0.1)
            raw = read_until_close(sock)
        elapsed = time.monotonic() - started
        status, _, body = parse_response(raw)
        self.assertEqual(status, 408)
        self.assertEqual(body, TIMEOUT_BODY)
        self.assertGreaterEqual(elapsed, 0.8)
        self.assertLess(elapsed, 1.6)
        self.assertEqual(self.get_state("k"), 404)

    def test_stalled_body_gets_408_and_creates_nothing(self) -> None:
        self.start_server(request_timeout_seconds=1)
        body = operation_body()
        head = post_request_bytes(body)[: -len(body)]
        with self.connect() as sock:
            # Headers (with a legal Content-Length) arrive in full, then
            # only half of the declared body ever shows up.
            sock.sendall(head + body[: len(body) // 2])
            raw = read_until_close(sock)
        status, headers, received = parse_response(raw)
        self.assertEqual(status, 408)
        self.assertEqual(received, TIMEOUT_BODY)
        self.assertEqual(headers["content-length"], str(len(TIMEOUT_BODY)))
        self.assertEqual(self.get_state("k"), 404)

    def test_timed_out_request_leaves_no_persistent_trace(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="sestate-timeout-data-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        path = os.path.join(tmpdir, "state.json")
        store = StateStore(data_file=path)
        status = store.apply_operation(
            "r1",
            {"operationId": "op-0", "key": "seed", "value": "0", "clock": {"r1": 1}},
        )
        self.assertEqual(status, 201)
        with open(path, "rb") as handle:
            before = handle.read()
        self.start_server(data_file=path, request_timeout_seconds=1)
        with self.connect() as sock:
            sock.sendall(b"POST /v1/replicas/r1/operations HTTP/1.0\r\n")
            raw = read_until_close(sock)
        status, _, body = parse_response(raw)
        self.assertEqual((status, body), (408, TIMEOUT_BODY))
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(sorted(os.listdir(tmpdir)), ["state.json"])

    def test_over_limit_declared_length_is_413_without_waiting(self) -> None:
        self.start_server(request_timeout_seconds=2)
        started = time.monotonic()
        with self.connect() as sock:
            sock.sendall(
                f"POST {OP_PATH} HTTP/1.0\r\n"
                f"Content-Length: {MAX_BODY_BYTES + 1}\r\n\r\n".encode("ascii")
            )
            raw = read_until_close(sock)
        elapsed = time.monotonic() - started
        status, _, body = parse_response(raw)
        self.assertEqual(status, 413)
        self.assertEqual(body, b'{"error":"payload_too_large"}')
        self.assertLess(elapsed, 1.5)

    def test_invalid_content_length_is_400_without_waiting(self) -> None:
        self.start_server(request_timeout_seconds=2)
        started = time.monotonic()
        with self.connect() as sock:
            sock.sendall(
                f"POST {OP_PATH} HTTP/1.0\r\nContent-Length: abc\r\n\r\n".encode("ascii")
            )
            raw = read_until_close(sock)
        elapsed = time.monotonic() - started
        status, _, body = parse_response(raw)
        self.assertEqual(status, 400)
        self.assertEqual(body, b'{"error":"invalid_request"}')
        self.assertLess(elapsed, 1.5)

    def test_complete_request_within_deadline_is_unchanged(self) -> None:
        self.start_server(request_timeout_seconds=5)
        with self.connect() as sock:
            sock.sendall(post_request_bytes(operation_body()))
            raw = read_until_close(sock)
        status, _, _ = parse_response(raw)
        self.assertEqual(status, 201)
        self.assertEqual(self.get_state("k"), 200)

    def test_missing_credential_is_still_401(self) -> None:
        self.start_server(auth_token=TOKEN, request_timeout_seconds=5)
        with self.connect() as sock:
            sock.sendall(post_request_bytes(operation_body()))
            raw = read_until_close(sock)
        status, _, body = parse_response(raw)
        self.assertEqual(status, 401)
        self.assertEqual(body, b'{"error":"unauthorized"}')

    def test_unknown_path_is_still_404(self) -> None:
        self.start_server(request_timeout_seconds=5)
        with self.connect() as sock:
            sock.sendall(b"GET /v1/no-such-route HTTP/1.0\r\n\r\n")
            raw = read_until_close(sock)
        status, _, body = parse_response(raw)
        self.assertEqual(status, 404)
        self.assertEqual(body, b'{"error":"not_found"}')

    def test_disabled_timeout_tolerates_slow_arrival(self) -> None:
        # Without the option the baseline behavior is unchanged: a request
        # arriving in pieces over more than a second is still served.
        self.start_server()
        request = post_request_bytes(operation_body())
        with self.connect() as sock:
            for index in range(0, len(request), 16):
                sock.sendall(request[index : index + 16])
                time.sleep(0.05)
            raw = read_until_close(sock)
        status, _, _ = parse_response(raw)
        self.assertEqual(status, 201)
        self.assertEqual(self.get_state("k"), 200)


if __name__ == "__main__":
    unittest.main()
