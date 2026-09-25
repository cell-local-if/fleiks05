import contextlib
import http.client
import io
import json
import os
import shutil
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    AuthTokenError,
    RequestHandler,
    SemanticStateServer,
    load_auth_token,
    main,
)

TOKEN = "s3cret-token_123"
AUTH_HEADER = f"Bearer {TOKEN}"

OP_PATH = "/v1/replicas/r1/operations"
SYNC_PATH = "/v1/sync/operations"
RESOLVE_PATH = "/v1/states/k/resolve"
CHECKPOINT_PATH = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_PATHS = (OP_PATH, SYNC_PATH, RESOLVE_PATH, CHECKPOINT_PATH)

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class LoadAuthTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-auth-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def write_token_file(self, content: bytes) -> str:
        path = os.path.join(self.tmpdir, "token.txt")
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def test_valid_token_is_loaded(self) -> None:
        path = self.write_token_file(TOKEN.encode("ascii"))
        self.assertEqual(load_auth_token(path), TOKEN)

    def test_single_character_token_is_valid(self) -> None:
        path = self.write_token_file(b"x")
        self.assertEqual(load_auth_token(path), "x")

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(os.path.join(self.tmpdir, "absent"))

    def test_directory_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(self.tmpdir)

    def test_empty_file_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(self.write_token_file(b""))

    def test_trailing_newline_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(self.write_token_file(TOKEN.encode("ascii") + b"\n"))

    def test_embedded_whitespace_is_rejected(self) -> None:
        for content in (b"two tokens", b"tok\ten", b" token", b"token ", b"\ntoken"):
            with self.subTest(content=content):
                with self.assertRaises(AuthTokenError):
                    load_auth_token(self.write_token_file(content))

    def test_non_ascii_is_rejected(self) -> None:
        with self.assertRaises(AuthTokenError):
            load_auth_token(self.write_token_file("tökén".encode("utf-8")))

    def test_error_never_echoes_the_token(self) -> None:
        secret = "do-not-leak-me"
        path = self.write_token_file(secret.encode("ascii") + b"\n")
        with self.assertRaises(AuthTokenError) as caught:
            load_auth_token(path)
        self.assertNotIn(secret, str(caught.exception))


class StartupFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-auth-cli-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def run_main(self, argv: list) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        return caught.exception.code, stderr.getvalue()

    def test_missing_token_file_fails_startup_like_a_data_file_error(self) -> None:
        code, stderr = self.run_main(
            ["--auth-token-file", os.path.join(self.tmpdir, "absent")]
        )
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)

    def test_invalid_token_format_fails_without_leaking_the_token(self) -> None:
        secret = "super-secret-token"
        path = os.path.join(self.tmpdir, "token.txt")
        with open(path, "wb") as handle:
            handle.write(secret.encode("ascii") + b"\n")
        code, stderr = self.run_main(["--auth-token-file", path])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)
        self.assertNotIn(secret, stderr)

    def test_directory_token_path_fails_startup(self) -> None:
        code, stderr = self.run_main(["--auth-token-file", self.tmpdir])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)


class AuthDisabledTests(unittest.TestCase):
    """Without --auth-token-file the documented behavior is unchanged."""

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

    def request(self, method: str, path: str, headers: dict | None = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_no_authorization_header_is_required(self) -> None:
        self.assertEqual(self.server.auth_token, None)
        status, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        status, payload = self.request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_an_authorization_header_is_simply_ignored(self) -> None:
        status, _ = self.request("GET", "/v1/metrics", {"Authorization": "Bearer wrong"})
        self.assertEqual(status, 200)


class AuthEnabledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, auth_token=TOKEN)
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
        self, method: str, path: str, body: object = None, auth: str | None = AUTH_HEADER
    ) -> tuple[int, dict, dict]:
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def post_raw(self, path: str, headers: list, body: bytes | None = None) -> tuple[int, dict]:
        """POST with exact control over the header lines that are sent."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    # -- /health stays anonymous --

    def test_health_is_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", auth=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    # -- 401 shape --

    def assert_unauthorized(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_missing_authorization_header_is_401(self) -> None:
        status, payload, headers = self.request("GET", "/v1/metrics", auth=None)
        self.assert_unauthorized(status, payload, headers)

    def test_wrong_token_is_401(self) -> None:
        status, payload, headers = self.request("GET", "/v1/metrics", auth="Bearer wrong")
        self.assert_unauthorized(status, payload, headers)

    def test_malformed_authorization_values_are_401(self) -> None:
        bad_values = [
            "Bearer",  # scheme only
            f"Bearer  {TOKEN}",  # two spaces
            f"bearer {TOKEN}",  # wrong scheme case
            TOKEN,  # no scheme
            f"Bearer {TOKEN} extra",  # trailing content
            f"Bearer {TOKEN} ",  # trailing space
            f"Basic {TOKEN}",  # wrong scheme
        ]
        for value in bad_values:
            with self.subTest(value=value):
                status, payload, headers = self.request("GET", "/v1/metrics", auth=value)
                self.assert_unauthorized(status, payload, headers)

    def test_duplicate_authorization_headers_are_401_even_when_one_matches(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/v1/metrics")
        conn.putheader("Authorization", AUTH_HEADER)
        conn.putheader("Authorization", AUTH_HEADER)
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)

    def test_unknown_routes_require_auth_before_404(self) -> None:
        status, payload, headers = self.request("GET", "/nope", auth=None)
        self.assert_unauthorized(status, payload, headers)
        status, payload, _ = self.request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unknown_post_routes_require_auth_before_404(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        status, payload = self.post_raw(
            "/v1/unknown", [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.post_raw(
            "/v1/unknown",
            [("Content-Length", str(len(body))), ("Authorization", AUTH_HEADER)],
            body,
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_authorized_requests_behave_as_before(self) -> None:
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        # An identical replay is still 200, a conflicting identity still 409.
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 200)
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="red")
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        status, state, _ = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        status, metrics, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["acceptedOperations"], 1)

    # -- Content-Length keeps priority over authentication on POST endpoints --

    def test_missing_content_length_is_400_not_401_on_all_post_endpoints(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(path, [], b"{}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_content_length_is_413_not_401_on_all_post_endpoints(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(
                    path, [("Content-Length", OVER_LIMIT)], b"junk"
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_malformed_content_length_is_400_not_401(self) -> None:
        status, payload = self.post_raw(OP_PATH, [("Content-Length", "abc")], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_valid_length_without_auth_is_401_and_body_is_not_read(self) -> None:
        # Declare a valid length but never send the body: the 401 must arrive
        # without the server waiting to read it.
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.putrequest("POST", path)
                conn.putheader("Content-Length", "64")
                conn.endheaders()
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.getheaders())
                conn.close()
                self.assert_unauthorized(response.status, payload, headers)

    def test_valid_length_with_wrong_token_is_401_on_all_post_endpoints(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(
                    path,
                    [("Content-Length", str(len(body))), ("Authorization", "Bearer nope")],
                    body,
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})

    # -- Failed authentication changes nothing --

    def test_failed_auth_changes_no_state(self) -> None:
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        _, metrics_before, _ = self.request("GET", "/v1/metrics")
        _, digest_before, _ = self.request("GET", "/v1/verification/digest")
        _, audit_before, _ = self.request("GET", "/v1/audit/keys/color/digest")

        body = json.dumps(operation_document(op_id="op-2")).encode("utf-8")
        rejections = 0
        for path in ALL_POST_PATHS:
            status, _ = self.post_raw(path, [("Content-Length", str(len(body)))], body)
            self.assertEqual(status, 401)
            rejections += 1
        for method, path in (
            ("GET", "/v1/metrics"),
            ("GET", "/v1/states/color"),
            ("GET", "/v1/sync/operations"),
            ("GET", "/v1/audit/keys/color/operations"),
            ("GET", "/v1/audit/keys/color/digest"),
            ("GET", "/v1/verification/digest"),
            ("GET", "/v1/sync/peers/peer-a/checkpoint"),
        ):
            status, _, _ = self.request(method, path, auth=None)
            self.assertEqual(status, 401)
            rejections += 1
        self.assertGreater(rejections, 0)

        _, metrics_after, _ = self.request("GET", "/v1/metrics")
        _, digest_after, _ = self.request("GET", "/v1/verification/digest")
        _, audit_after, _ = self.request("GET", "/v1/audit/keys/color/digest")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)
        self.assertEqual(audit_after, audit_before)


class AuthPersistenceTests(unittest.TestCase):
    """With --data-file, failed auth touches neither the file nor the directory."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-auth-data-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=cls.data_path, auth_token=TOKEN,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def request(
        self, method: str, path: str, document: dict = None, auth: str | None = AUTH_HEADER
    ) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if document is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(document), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_failed_auth_leaves_file_directory_and_state_untouched(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        status, _ = self.request("POST", CHECKPOINT_PATH, {"cursor": 1})
        self.assertEqual(status, 200)

        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        _, metrics_before = self.request("GET", "/v1/metrics")
        _, checkpoint_before = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")

        body = json.dumps(operation_document(op_id="op-2")).encode("utf-8")
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.request(
                    "POST", path, json.loads(body), auth="Bearer wrong"
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
        status, _ = self.request("GET", "/v1/metrics", auth=None)
        self.assertEqual(status, 401)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        _, metrics_after = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        _, checkpoint_after = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(checkpoint_after, checkpoint_before)

    def test_data_file_never_contains_auth_configuration(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-10", key="k10", value="v")
        )
        self.assertEqual(status, 201)
        with open(self.data_path, "rb") as handle:
            document = json.loads(handle.read().decode("utf-8"))
        self.assertLessEqual(
            set(document.keys()),
            {"version", "operations", "checkpoints", "policies", "transactions", "acks",
            "scopePolicyEvents"},
        )
        self.assertNotIn(TOKEN, json.dumps(document))

    def test_restart_recovers_state_and_still_requires_auth(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-20", key="k20", value="kept")
        )
        self.assertEqual(status, 201)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

        restarted = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path, auth_token=TOKEN,
        )
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        try:
            port = restarted.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/states/k20")
            response = conn.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            conn.close()

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/states/k20", headers={"Authorization": AUTH_HEADER})
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["value"], "kept")
        finally:
            restarted.shutdown()
            restarted.server_close()
            thread.join(timeout=5)

        # Reopen the original server for any later tests in this class.
        type(self).server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path, auth_token=TOKEN,
        )
        type(self).thread = threading.Thread(
            target=type(self).server.serve_forever, daemon=True
        )
        type(self).thread.start()
        type(self).port = type(self).server.server_address[1]


if __name__ == "__main__":
    unittest.main()
