"""Tests for optional bearer token authentication (``--auth-token-file``).

Covers authentication enabled and disabled, token-file startup failures,
exact header handling, the rule that an unauthorized POST with a legal
Content-Length is rejected without the body being read, Content-Length's
400/413 precedence over 401, and the invariance of memory, the data file,
and persistence/restart. All requests here are real HTTP over sockets.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    AuthTokenError,
    RequestHandler,
    SemanticStateServer,
    load_auth_token,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN = "a-long-random-token-42"
AUTH_HEADER = f"Bearer {TOKEN}"

OP_PATH = "/v1/replicas/r1/operations"
SYNC_PATH = "/v1/sync/operations"
RESOLVE_PATH = "/v1/states/k/resolve"
CHECKPOINT_PATH = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_PATHS = (OP_PATH, SYNC_PATH, RESOLVE_PATH, CHECKPOINT_PATH)

EXISTING_GET_PATHS = (
    "/v1/metrics",
    "/v1/verification/digest",
    "/v1/states/k",
    "/v1/sync/operations",
    "/v1/audit/keys/k/operations",
    "/v1/audit/keys/k/digest",
    "/v1/sync/peers/peer-a/checkpoint",
)

OPERATION_BODY = json.dumps(
    {"operationId": "op-1", "key": "k", "value": "v", "clock": {"r1": 1}}
).encode("utf-8")


# --------------------------------------------------------------------------- #
# Token-file loading
# --------------------------------------------------------------------------- #


class LoadAuthTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write(self, name: str, raw: bytes) -> str:
        path = self.tmp / name
        path.write_bytes(raw)
        return str(path)

    def test_plain_printable_token_is_returned(self) -> None:
        path = self.write("t", b"a-long-random-token-42")
        self.assertEqual(load_auth_token(path), "a-long-random-token-42")

    def test_printable_boundary_bytes_are_accepted(self) -> None:
        self.assertEqual(load_auth_token(self.write("a", b"!")), "!")  # 0x21
        self.assertEqual(load_auth_token(self.write("b", b"~")), "~")  # 0x7E
        self.assertEqual(
            load_auth_token(self.write("c", bytes(range(0x21, 0x7F)))),
            bytes(range(0x21, 0x7F)).decode("ascii"),
        )

    def test_trailing_newline_is_rejected(self) -> None:
        for raw in (b"token\n", b"token\r\n", b"token\r"):
            with self.subTest(raw=raw):
                with self.assertRaises(AuthTokenError):
                    load_auth_token(self.write("nl", raw))

    def test_whitespace_anywhere_is_rejected(self) -> None:
        for raw in (b" token", b"token ", b"a b", b"a\tb", b"a\vb", b"a\fb"):
            with self.subTest(raw=raw):
                with self.assertRaises(AuthTokenError):
                    load_auth_token(self.write("ws", raw))

    def test_empty_file_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(self.write("empty", b""))

    def test_non_printable_and_non_ascii_are_rejected(self) -> None:
        for raw in (b"a\x00b", b"a\x1fb", b"a\x7fb", b"a\x80b", b"a\xffb", b"tok\xc3\xa9n"):
            with self.subTest(raw=raw):
                with self.assertRaises(AuthTokenError):
                    load_auth_token(self.write("ctl", raw))

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(str(self.tmp / "does-not-exist"))

    def test_directory_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(str(self.tmp))

    def test_fifo_is_rejected(self) -> None:
        fifo = self.tmp / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(AuthTokenError):
            load_auth_token(str(fifo))

    @unittest.skipIf(os.geteuid() == 0, "root bypasses file permission bits")
    def test_unreadable_regular_file_is_rejected(self) -> None:
        path = self.write("locked", b"token")
        os.chmod(path, 0o000)
        self.addCleanup(os.chmod, path, 0o600)
        with self.assertRaises(AuthTokenError):
            load_auth_token(path)

    def test_error_message_never_contains_the_token(self) -> None:
        path = self.write("t", b"supersecretvalue\n")
        try:
            load_auth_token(path)
        except AuthTokenError as exc:
            self.assertNotIn("supersecretvalue", str(exc))
            self.assertIn(os.path.basename(path), str(exc))
        else:  # pragma: no cover - the loader must reject the newline
            self.fail("token with newline was accepted")


# --------------------------------------------------------------------------- #
# Shared real-HTTP fixtures
# --------------------------------------------------------------------------- #


def parse_raw_response(data: bytes) -> tuple[int, dict[str, str], bytes]:
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return status, headers, body


def raw_request(port: int, request: bytes) -> tuple[int, dict[str, str], bytes]:
    """Send raw bytes, half-close the write side, and read until server EOF."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()
    return parse_raw_response(data)


class HttpServerFixture:
    """Start a SemanticStateServer on an ephemeral port for real HTTP tests."""

    data_file: str | None = None

    @classmethod
    def setUpClass(cls) -> None:  # noqa: N802 - unittest API
        cls._tmp = tempfile.mkdtemp(prefix="sestate-auth-")
        data_path = os.path.join(cls._tmp, "state.json") if cls.use_data_file else None
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=data_path,
            auth_token=cls.auth_token,
        )
        cls.data_path = data_path
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:  # noqa: N802 - unittest API
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def request(
        self, method: str, path: str, body: bytes | None = None, token: str | None = False,
        headers: dict | None = None,
    ) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = dict(headers or {})
        if token is not False:
            hdrs["Authorization"] = f"Bearer {token if token is not None else TOKEN}"
        if body is not None:
            hdrs["Content-Length"] = str(len(body))
        conn.request(method, path, body=body, headers=hdrs)
        response = conn.getresponse()
        raw = response.read()
        www_auth = response.getheader("WWW-Authenticate")
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        if www_auth is not None:
            payload = dict(payload)
            payload["__www_authenticate"] = www_auth
        return response.status, payload

    def get(self, path: str, token: str | None = False) -> tuple[int, dict]:
        return self.request("GET", path, token=token)

    def post(
        self, path: str, body: bytes, token: str | None = False, headers: dict | None = None
    ) -> tuple[int, dict]:
        return self.request("POST", path, body=body, token=token, headers=headers)


# --------------------------------------------------------------------------- #
# Authentication enabled
# --------------------------------------------------------------------------- #


class AuthEnabledTests(HttpServerFixture, unittest.TestCase):
    auth_token = TOKEN
    use_data_file = False

    def setUp(self) -> None:
        # Fresh in-memory state per test (the server itself is shared).
        self.server.store = type(self.server.store)()

    # -- /health stays anonymous ------------------------------------------- #

    def test_health_is_anonymous(self) -> None:
        status, payload = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    def test_health_ignores_garbage_or_duplicate_authorization(self) -> None:
        for raw in (
            b"GET /health HTTP/1.1\r\nHost: x\r\nAuthorization: junk\r\n\r\n",
            (
                b"GET /health HTTP/1.1\r\nHost: x\r\n"
                b"Authorization: Bearer " + TOKEN.encode() + b"\r\n"
                b"Authorization: Bearer other\r\n\r\n"
            ),
        ):
            with self.subTest(raw=raw):
                status, _, body = raw_request(self.port, raw)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), health())

    # -- Every existing GET route requires authentication ------------------ #

    def test_all_existing_get_routes_are_401_without_a_token(self) -> None:
        for path in EXISTING_GET_PATHS:
            with self.subTest(path=path):
                status, payload = self.get(path)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized", "__www_authenticate": "Bearer"})

    def test_unknown_get_route_is_401_before_404(self) -> None:
        status, payload = self.get("/totally/unknown/route")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "unauthorized")

    def test_401_is_checked_before_query_parsing(self) -> None:
        # Malformed query would be 400 after authentication; without a token
        # the request is rejected on auth alone.
        status, payload = self.get("/v1/metrics?x=1&x=2")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "unauthorized")
        status, payload = self.get("/v1/sync/operations?limit=9999")
        self.assertEqual(status, 401)

    def test_authenticated_get_routes_reach_their_normal_statuses(self) -> None:
        status, payload = self.get("/v1/metrics", token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 0)
        status, _ = self.get("/v1/states/absent", token=TOKEN)
        self.assertEqual(status, 404)
        status, payload = self.get("/v1/metrics?x=1", token=TOKEN)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- Exact Authorization-header semantics ------------------------------ #

    def test_authorization_value_must_match_exactly(self) -> None:
        cases = [
            b"",  # empty value
            b"Bearer",  # no space, no token
            b"Bearer" + TOKEN.encode(),  # no separating space
            b"Bearer  " + TOKEN.encode(),  # two spaces
            b"Bearer " + TOKEN.encode() + b" ",  # trailing space
            b"Bearer " + TOKEN.encode() + b" x",  # extra token
            b"bearer " + TOKEN.encode(),  # lowercase scheme
            b"Basic " + TOKEN.encode(),  # wrong scheme
            b"Bearer not-the-token",  # wrong token
            b"Bearer " + TOKEN[:-1].encode(),  # prefix of the token
            TOKEN.encode(),  # token with no scheme
        ]
        for value in cases:
            with self.subTest(value=value):
                raw = (
                    b"GET /v1/metrics HTTP/1.1\r\nHost: x\r\nAuthorization: "
                    + value
                    + b"\r\n\r\n"
                )
                status, headers, body = raw_request(self.port, raw)
                self.assertEqual(status, 401)
                self.assertEqual(headers.get("www-authenticate"), "Bearer")
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_duplicate_authorization_headers_are_401_even_when_correct(self) -> None:
        raw = (
            b"GET /v1/metrics HTTP/1.1\r\nHost: x\r\n"
            b"Authorization: Bearer " + TOKEN.encode() + b"\r\n"
            b"Authorization: Bearer " + TOKEN.encode() + b"\r\n\r\n"
        )
        status, headers, body = raw_request(self.port, raw)
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_one_correct_authorization_header_passes(self) -> None:
        status, _, body = raw_request(
            self.port,
            b"GET /v1/metrics HTTP/1.1\r\nHost: x\r\n"
            b"Authorization: Bearer " + TOKEN.encode() + b"\r\n\r\n",
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["keys"], 0)

    # -- POST precedence and unread body ----------------------------------- #

    def test_post_missing_or_bad_content_length_is_400_even_unauthenticated(self) -> None:
        # 400 takes precedence over 401, on all four endpoints.
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, _, body = raw_request(
                    self.port,
                    b"POST " + path.encode() + b" HTTP/1.1\r\nHost: x\r\n\r\n",
                )
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body), {"error": "invalid_request"})

    def test_post_over_limit_declaration_is_413_before_auth(self) -> None:
        for auth in (None, TOKEN):
            header = b"" if auth is None else b"Authorization: Bearer " + auth.encode() + b"\r\n"
            with self.subTest(auth=auth):
                status, _, body = raw_request(
                    self.port,
                    b"POST " + SYNC_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Length: 1048577\r\n" + header + b"\r\n",
                )
                self.assertEqual(status, 413)
                self.assertEqual(json.loads(body), {"error": "payload_too_large"})

    def test_legal_length_but_unauthorized_is_401_on_all_post_endpoints(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, headers, body = raw_request(
                    self.port,
                    b"POST " + path.encode() + b" HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Length: 2\r\n\r\n",
                )
                self.assertEqual(status, 401)
                self.assertEqual(headers.get("www-authenticate"), "Bearer")
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_401_post_responds_before_reading_the_body_and_closes(self) -> None:
        # Declare a body but never send it. A server that waited for the body
        # would time out instead of answering; the 401 proves the body was not
        # read. The server then closes the connection (EOF).
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            sock.sendall(
                b"POST " + OP_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n"
                b"Content-Length: 50\r\n\r\n"
            )
            sock.settimeout(3)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(4096)
                self.assertTrue(chunk, "server closed before sending headers")
                data += chunk
            head, _, body = data.partition(b"\r\n\r\n")
            self.assertEqual(int(head.split()[1]), 401)
            # Drain the declared response body, then observe the server EOF.
            while len(body) < len(b'{"error":"unauthorized"}'):
                body += sock.recv(4096)
            self.assertEqual(body, b'{"error":"unauthorized"}')
            self.assertEqual(sock.recv(16), b"")
        finally:
            sock.close()
        # The unread body certainly never committed an operation.
        status, _ = self.get("/v1/states/k", token=TOKEN)
        self.assertEqual(status, 404)

    def test_unknown_post_route_auth_precedes_404(self) -> None:
        # Legal Content-Length, no body needed to observe the auth decision.
        status, _, body = raw_request(
            self.port,
            b"POST /v1/nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})
        status, _, body = raw_request(
            self.port,
            b"POST /v1/nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n"
            b"Authorization: Bearer " + TOKEN.encode() + b"\r\n\r\n",
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body), {"error": "not_found"})

    # -- Authenticated behavior is otherwise unchanged --------------------- #

    def test_authenticated_writes_and_reads_work_normally(self) -> None:
        status, payload = self.post(OP_PATH, OPERATION_BODY, token=TOKEN)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        # Identical replay -> 200.
        status, payload = self.post(OP_PATH, OPERATION_BODY, token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        status, payload = self.get("/v1/states/k", token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v")
        status, payload = self.get("/v1/metrics", token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 1)

    def test_authenticated_conflict_is_still_409(self) -> None:
        self.post(OP_PATH, OPERATION_BODY, token=TOKEN)
        clash = json.dumps(
            {"operationId": "op-1", "key": "k", "value": "other", "clock": {"r1": 1}}
        ).encode("utf-8")
        status, payload = self.post(OP_PATH, clash, token=TOKEN)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_authenticated_malformed_body_is_still_400(self) -> None:
        status, payload = self.post(OP_PATH, b"not json", token=TOKEN)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


def health() -> dict:
    return {"service": "semantic-state-engine", "status": "ok"}


# --------------------------------------------------------------------------- #
# Authentication disabled: published semantics are unchanged
# --------------------------------------------------------------------------- #


class AuthDisabledTests(HttpServerFixture, unittest.TestCase):
    auth_token = None
    use_data_file = False

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def test_routes_are_anonymous(self) -> None:
        status, payload = self.get("/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 0)
        status, _ = self.get("/totally/unknown")
        self.assertEqual(status, 404)

    def test_posts_work_and_size_contract_is_unchanged(self) -> None:
        status, payload = self.post(OP_PATH, OPERATION_BODY)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        status, _, body = raw_request(
            self.port,
            b"POST " + OP_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n\r\n",
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "invalid_request"})


# --------------------------------------------------------------------------- #
# A 401 must not touch memory, the data file, checkpoints, audit, or temp files
# --------------------------------------------------------------------------- #


class AuthFailureInvarianceTests(HttpServerFixture, unittest.TestCase):
    auth_token = TOKEN
    use_data_file = True

    def test_unauthorized_requests_change_nothing(self) -> None:
        # Seed committed state and a checkpoint, all authenticated.
        self.assertEqual(self.post(OP_PATH, OPERATION_BODY, token=TOKEN)[0], 201)
        self.assertEqual(
            self.post(CHECKPOINT_PATH, b'{"cursor":1}', token=TOKEN)[0], 200
        )

        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self._tmp))
        _, metrics_before = self.get("/v1/metrics", token=TOKEN)
        _, state_before = self.get("/v1/states/k", token=TOKEN)
        _, checkpoint_before = self.get(CHECKPOINT_PATH, token=TOKEN)
        _, audit_before = self.get("/v1/audit/keys/k/operations", token=TOKEN)

        # A battery of unauthorized traffic, including a valid operation that
        # must not commit and over/malformed-length declarations.
        unauthenticated_gets = list(EXISTING_GET_PATHS) + [
            "/totally/unknown",
            "/v1/metrics?x=1",
        ]
        for path in unauthenticated_gets:
            self.assertEqual(self.get(path)[0], 401)
        for path in ALL_POST_PATHS:
            self.assertEqual(
                raw_request(
                    self.port,
                    b"POST " + path.encode() + b" HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Length: 50\r\n\r\n",
                )[0],
                401,
            )
        # Missing Content-Length (400) and over-limit (413) precede auth.
        self.assertEqual(
            raw_request(
                self.port,
                b"POST " + OP_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n\r\n",
            )[0],
            400,
        )
        self.assertEqual(
            raw_request(
                self.port,
                b"POST " + SYNC_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n"
                b"Content-Length: 1048577\r\n\r\n",
            )[0],
            413,
        )

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self._tmp)), listing_before)
        _, metrics_after = self.get("/v1/metrics", token=TOKEN)
        self.assertEqual(metrics_after, metrics_before)
        _, state_after = self.get("/v1/states/k", token=TOKEN)
        self.assertEqual(state_after, state_before)
        _, checkpoint_after = self.get(CHECKPOINT_PATH, token=TOKEN)
        self.assertEqual(checkpoint_after, checkpoint_before)
        _, audit_after = self.get("/v1/audit/keys/k/operations", token=TOKEN)
        self.assertEqual(audit_after, audit_before)

        # The configured token is never persisted.
        self.assertNotIn(TOKEN.encode(), bytes_before)
        document = json.loads(bytes_before)
        self.assertTrue(set(document) <= {"version", "operations", "checkpoints"})

        # Service still serves and commits authenticated requests afterwards.
        follow_up = json.dumps(
            {"operationId": "op-2", "key": "k", "value": "v2", "clock": {"r2": 1}}
        ).encode("utf-8")
        self.assertEqual(
            self.post("/v1/replicas/r2/operations", follow_up, token=TOKEN)[0], 201
        )


# --------------------------------------------------------------------------- #
# Command line: flag wiring, startup failure, persistence across auth toggles
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class CommandLineAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"
        self.token_file = self.tmp / "auth.token"
        self.token_file.write_text(TOKEN, encoding="ascii")  # no trailing newline
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
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

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

    def http(self, method: str, path: str, body: bytes | None = None, authorized: bool = True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if authorized:
            headers["Authorization"] = f"Bearer {TOKEN}"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, (json.loads(raw) if raw else None)

    def assert_not_listening(self) -> None:
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

    def test_cli_enforces_token_and_keeps_health_anonymous(self) -> None:
        proc = self.spawn("--auth-token-file", str(self.token_file))
        try:
            self.wait_for_health()
            status, _ = self.http("GET", "/v1/metrics", authorized=False)
            self.assertEqual(status, 401)
            status, payload = self.http("GET", "/v1/metrics", authorized=True)
            self.assertEqual(status, 200)
            self.assertEqual(payload["keys"], 0)
        finally:
            self.stop(proc)

    def test_cli_startup_failures(self) -> None:
        bad_content = self.tmp / "bad.token"
        cases: list[tuple[str, bytes | None, str]] = [
            (str(self.tmp / "missing.token"), None, "missing file"),
            (str(self.tmp), None, "directory"),
        ]
        content_cases = [
            (b"", "empty"),
            (b"realsecrettoken\n", "trailing newline"),
            (b"realsecrettoken\r\n", "CRLF"),
            (b"has space", "embedded space"),
            (b"\xff\xfe", "non-ascii bytes"),
        ]
        fifo = self.tmp / "fifo.token"
        os.mkfifo(fifo)
        cases.append((str(fifo), None, "fifo"))
        for content, label in content_cases:
            path = self.tmp / f"token-{label.replace(' ', '-')}"
            path.write_bytes(content)
            cases.append((str(path), content, label))

        for path, content, label in cases:
            with self.subTest(label=label):
                proc = self.spawn("--auth-token-file", path)
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                finally:
                    self.stop(proc)
                self.assertEqual(proc.returncode, 2, label)
                self.assertIn(b"startup failed", stderr)
                self.assertEqual(stdout, b"")
                # The token value must never be printed.
                if content is not None:
                    secret = content.strip(b"\r\n").strip()
                    if secret and all(0x21 <= b <= 0x7E for b in secret):
                        self.assertNotIn(secret, stderr)
                self.assert_not_listening()

    def test_persisted_state_survives_auth_toggle_and_never_stores_the_token(self) -> None:
        # First run: no authentication, commit an operation.
        proc = self.spawn("--data-file", str(self.data_file))
        try:
            self.wait_for_health()
            status, _ = self.http(
                "POST", OP_PATH, body=OPERATION_BODY, authorized=False
            )
            self.assertEqual(status, 201)
        finally:
            self.stop(proc)

        raw_first = self.data_file.read_bytes()
        self.assertNotIn(TOKEN.encode(), raw_first)

        # Second run: authentication enabled; recovered state is visible only
        # with the token, and an anonymous read is 401.
        proc = self.spawn(
            "--data-file", str(self.data_file),
            "--auth-token-file", str(self.token_file),
        )
        try:
            self.wait_for_health()
            status, _ = self.http("GET", "/v1/states/k", authorized=False)
            self.assertEqual(status, 401)
            status, payload = self.http("GET", "/v1/states/k", authorized=True)
            self.assertEqual(status, 200)
            self.assertEqual(payload["value"], "v")
            # A recovered replay still behaves identically.
            status, payload = self.http("POST", OP_PATH, body=OPERATION_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")
        finally:
            self.stop(proc)

        document = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.assertTrue(set(document) <= {"version", "operations", "checkpoints"})
        self.assertEqual(len(document["operations"]), 1)
        self.assertNotIn(TOKEN.encode(), self.data_file.read_bytes())

        # Third run: authentication disabled again; the same state is served
        # anonymously, proving auth config is process-local, not persisted.
        proc = self.spawn("--data-file", str(self.data_file))
        try:
            self.wait_for_health()
            status, payload = self.http("GET", "/v1/states/k", authorized=False)
            self.assertEqual(status, 200)
            self.assertEqual(payload["value"], "v")
        finally:
            self.stop(proc)


if __name__ == "__main__":
    unittest.main()
