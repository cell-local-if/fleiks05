"""Tests for runtime single-token hot rotation.

``POST /v1/admin/auth-token/reload`` atomically replaces the live bearer
token by re-reading the file supplied with ``--auth-token-file`` at
startup, without a restart. The tests cover the full request precedence
chain (path shape, declared length, authentication, mode gate, query,
body), the 503/409 failure split with the old token kept whole, immediate
effect of the rotation on new requests, snapshot-scoped authentication,
and the absence of any business-state, data-file, or temporary-file
effects. The endpoint is published only in single-token mode.
"""

import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    AuthTokenManager,
    AuthTokenReloadError,
    RequestHandler,
    SemanticStateServer,
    load_auth_token,
    load_scope_policy,
)

TOKEN = "s3cret-token_123"
NEW_TOKEN = "r0tated-token_456"
RELOAD_PATH = "/v1/admin/auth-token/reload"
METRICS_PATH = "/v1/metrics"
OP_PATH = "/v1/replicas/r1/operations"

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class AuthTokenManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-token-reload-unit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "token.txt")
        self.write(TOKEN.encode("ascii"))
        self.manager = AuthTokenManager(self.path, load_auth_token(self.path))

    def write(self, content: bytes) -> str:
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        with open(self.path, "wb") as handle:
            handle.write(content)
        return self.path

    def test_reload_swaps_the_live_token(self) -> None:
        self.write(NEW_TOKEN.encode("ascii"))
        self.manager.reload()
        self.assertEqual(self.manager.snapshot(), NEW_TOKEN)

    def test_reload_of_an_unchanged_file_succeeds(self) -> None:
        self.manager.reload()
        self.assertEqual(self.manager.snapshot(), TOKEN)

    def test_missing_file_is_unavailable_and_keeps_the_old_token(self) -> None:
        os.unlink(self.path)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(self.manager.snapshot(), TOKEN)

    def test_directory_is_unavailable_and_keeps_the_old_token(self) -> None:
        os.unlink(self.path)
        os.mkdir(self.path)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(self.manager.snapshot(), TOKEN)

    def test_unreadable_file_is_unavailable(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses file permission bits")
        os.chmod(self.path, 0o000)
        self.addCleanup(os.chmod, self.path, 0o600)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(self.manager.snapshot(), TOKEN)

    def test_invalid_readable_content_is_conflict_and_keeps_the_old_token(self) -> None:
        for content in (
            b"",
            b"two tokens",
            b"tok\ten",
            b" token",
            b"token ",
            b"\ntoken",
            TOKEN.encode("ascii") + b"\n",
            "tökén".encode("utf-8"),
        ):
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(AuthTokenReloadError) as caught:
                    self.manager.reload()
                self.assertEqual(caught.exception.kind, "conflict")
                self.assertEqual(self.manager.snapshot(), TOKEN)

    def test_successful_reload_recovers_after_a_conflict(self) -> None:
        self.write(b"bad token")
        with self.assertRaises(AuthTokenReloadError):
            self.manager.reload()
        self.write(NEW_TOKEN.encode("ascii"))
        self.manager.reload()
        self.assertEqual(self.manager.snapshot(), NEW_TOKEN)

    def test_manager_without_a_configured_file_is_unavailable(self) -> None:
        with self.assertRaises(AuthTokenReloadError) as caught:
            AuthTokenManager(None, TOKEN).reload()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_errors_never_echo_the_token_or_file_content(self) -> None:
        secret = "do-not-leak-me"
        self.write(secret.encode("ascii") + b"\n")
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertNotIn(secret, str(caught.exception))
        os.unlink(self.path)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertNotIn(secret, str(caught.exception))


class AuthTokenReloadHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-token-reload-http-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.token_path = os.path.join(cls.tmpdir, "token.txt")
        with open(cls.token_path, "wb") as handle:
            handle.write(TOKEN.encode("ascii"))
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=cls.data_path,
            auth_token=TOKEN,
            auth_token_file=cls.token_path,
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

    def setUp(self) -> None:
        # Every test starts from the initial token both on disk and in the
        # live manager, regardless of what an earlier test left behind.
        self.write_token(TOKEN.encode("ascii"))
        self.server.auth_token_manager.reload()
        # Cleanups run last-added first: reload the live manager only after
        # the restored file is back on disk.
        self.addCleanup(self.server.auth_token_manager.reload)
        self.addCleanup(self.write_token, TOKEN.encode("ascii"))

    def write_token(self, content: bytes) -> None:
        # A previous test may have turned the path into a directory; clear
        # whatever occupies it so each call starts from a regular file.
        if os.path.isdir(self.token_path):
            shutil.rmtree(self.token_path)
        with open(self.token_path, "wb") as handle:
            handle.write(content)

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = TOKEN,
    ) -> tuple[int, object, bytes, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def raw_request(
        self, method: str, path: str, headers: list, body: bytes | None = None
    ) -> tuple[int, object, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest(method, path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, (json.loads(raw) if raw else None), response_headers

    def reload(self, body: object = {}, token: str | None = TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    # -- success contract --

    def test_success_returns_exactly_the_status_field(self) -> None:
        status, payload, raw, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "reloaded"})
        self.assertEqual(raw.decode("utf-8"), '{"status":"reloaded"}')

    def test_rotation_takes_effect_immediately(self) -> None:
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _, _ = self.reload()
        self.assertEqual(status, 200)
        # The new token authenticates at once...
        status, _, _, _ = self.request("GET", METRICS_PATH, token=NEW_TOKEN)
        self.assertEqual(status, 200)
        # ...and the old token is rejected at once, with the challenge.
        status, payload, _, headers = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_reload_of_an_unchanged_file_keeps_the_token_working(self) -> None:
        status, _, _, _ = self.reload()
        self.assertEqual(status, 200)
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 200)

    def test_successive_rotations_each_take_effect(self) -> None:
        current = TOKEN
        for token in ("second-token", "third-token"):
            self.write_token(token.encode("ascii"))
            status, _, _, _ = self.reload(token=current)
            self.assertEqual(status, 200)
            status, _, _, _ = self.request("GET", METRICS_PATH, token=token)
            self.assertEqual(status, 200)
            current = token
        status, _, _, _ = self.request("GET", METRICS_PATH, token="second-token")
        self.assertEqual(status, 401)

    def test_response_never_contains_old_or_new_token(self) -> None:
        self.write_token(NEW_TOKEN.encode("ascii"))
        _, _, raw, _ = self.reload()
        self.assertNotIn(TOKEN.encode("ascii"), raw)
        self.assertNotIn(NEW_TOKEN.encode("ascii"), raw)

    # -- 503 unavailable keeps the old token --

    def assert_unavailable_keeps_old_token(self) -> None:
        status, payload, _, _ = self.reload()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "auth_token_unavailable"})
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 200)

    def test_missing_file_is_503_and_keeps_the_old_token(self) -> None:
        os.unlink(self.token_path)
        self.assert_unavailable_keeps_old_token()

    def test_directory_is_503_and_keeps_the_old_token(self) -> None:
        os.unlink(self.token_path)
        os.mkdir(self.token_path)
        self.assert_unavailable_keeps_old_token()

    def test_unreadable_file_is_503_and_keeps_the_old_token(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses file permission bits")
        os.chmod(self.token_path, 0o000)
        self.addCleanup(os.chmod, self.token_path, 0o600)
        self.assert_unavailable_keeps_old_token()

    # -- 409 conflict keeps the old token --

    def test_invalid_readable_content_is_409_and_keeps_the_old_token(self) -> None:
        for content in (
            b"",
            b"two tokens",
            b"token ",
            TOKEN.encode("ascii") + b"\n",
            "tökén".encode("utf-8"),
        ):
            with self.subTest(content=content):
                self.write_token(content)
                status, payload, _, _ = self.reload()
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "auth_token_conflict"})
                status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
                self.assertEqual(status, 200)

    def test_successful_reload_recovers_after_a_conflict(self) -> None:
        self.write_token(b"bad token")
        status, _, _, _ = self.reload()
        self.assertEqual(status, 409)
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _, _ = self.reload()
        self.assertEqual(status, 200)
        status, _, _, _ = self.request("GET", METRICS_PATH, token=NEW_TOKEN)
        self.assertEqual(status, 200)
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 401)

    # -- request precedence: path shape, declared length, auth, query, body --

    def test_route_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/admin/auth-token/reload/",
            "/v1/admin/auth-token/reload/extra",
            "/v1/admin/auth-token",
            "/v1/admin/auth-token/",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request("POST", path, body={})
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_the_route_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", RELOAD_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_content_length_is_400_before_auth(self) -> None:
        status, payload, _ = self.raw_request("POST", RELOAD_PATH, [], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_content_length_is_400_before_auth(self) -> None:
        status, payload, _ = self.raw_request(
            "POST", RELOAD_PATH, [("Content-Length", "abc")], b"{}"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_content_length_is_413_before_auth(self) -> None:
        status, payload, _ = self.raw_request(
            "POST", RELOAD_PATH, [("Content-Length", OVER_LIMIT)], b"junk"
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_length_rejections_neither_read_the_file_nor_change_the_token(self) -> None:
        # With invalid content on disk, a length rejection must still be
        # 400/413 — never 409 — and the live token must stay in force.
        self.write_token(b"bad token")
        status, _, _ = self.raw_request("POST", RELOAD_PATH, [], b"{}")
        self.assertEqual(status, 400)
        status, _, _ = self.raw_request(
            "POST", RELOAD_PATH, [("Content-Length", OVER_LIMIT)], b"junk"
        )
        self.assertEqual(status, 413)
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 200)

    def test_missing_or_wrong_token_is_401(self) -> None:
        status, payload, _, headers = self.reload(token=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, payload, _, _ = self.reload(token="wrong")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_failed_auth_never_reads_the_file(self) -> None:
        # With invalid content on disk, an unauthenticated request is 401,
        # not 409: the file is not touched before authentication passes.
        self.write_token(b"bad token")
        status, _, _, _ = self.reload(token=None)
        self.assertEqual(status, 401)
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 200)

    def test_any_query_parameter_is_400(self) -> None:
        status, payload, _, _ = self.request("POST", RELOAD_PATH + "?x=1", body={})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_non_empty_or_non_object_bodies_are_400(self) -> None:
        for body in (
            {"token": NEW_TOKEN},
            {"expectedPolicyDigest": "0" * 64},
            ["x"],
            "text",
            0,
            None,
            True,
        ):
            with self.subTest(body=body):
                status, payload, _, _ = self.reload(body=body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        status, payload, _ = self.raw_request(
            "POST",
            RELOAD_PATH,
            [("Content-Length", "1"), ("Authorization", f"Bearer {TOKEN}")],
            b"{",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The token is untouched by every rejection.
        status, _, _, _ = self.request("GET", METRICS_PATH, token=TOKEN)
        self.assertEqual(status, 200)

    def test_body_may_carry_json_whitespace(self) -> None:
        for raw_body in (b"{}", b"{ }", b"{\n\t}"):
            with self.subTest(raw_body=raw_body):
                status, payload, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", str(len(raw_body))),
                        ("Authorization", f"Bearer {TOKEN}"),
                    ],
                    raw_body,
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload, {"status": "reloaded"})

    # -- rotation changes no business state --

    def test_reload_changes_no_state_and_writes_no_files(self) -> None:
        status, _, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        _, metrics_before, _, _ = self.request("GET", METRICS_PATH)
        _, digest_before, _, _ = self.request("GET", "/v1/verification/digest")
        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))

        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _, _ = self.reload()
        self.assertEqual(status, 200)

        _, metrics_after, _, _ = self.request("GET", METRICS_PATH, token=NEW_TOKEN)
        _, digest_after, _, _ = self.request(
            "GET", "/v1/verification/digest", token=NEW_TOKEN
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)

    def test_failed_reload_changes_no_state(self) -> None:
        _, metrics_before, _, _ = self.request("GET", METRICS_PATH)
        self.write_token(b"bad token")
        status, _, _, _ = self.reload()
        self.assertEqual(status, 409)
        _, metrics_after, _, _ = self.request("GET", METRICS_PATH)
        self.assertEqual(metrics_after, metrics_before)

    def test_data_file_never_contains_the_token(self) -> None:
        status, _, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-2", key="k2", value="v")
        )
        self.assertEqual(status, 201)
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _, _ = self.reload()
        self.assertEqual(status, 200)
        with open(self.data_path, "rb") as handle:
            document = handle.read().decode("utf-8")
        self.assertNotIn(TOKEN, document)
        self.assertNotIn(NEW_TOKEN, document)


class AuthTokenReloadModeGateTests(unittest.TestCase):
    """The endpoint exists only in single-token mode."""

    def serve(self, server: SemanticStateServer) -> tuple[int, threading.Thread]:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server.server_address[1], thread

    def post_reload(self, port: int, token: str | None) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", RELOAD_PATH, body="{}", headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_anonymous_mode_answers_404(self) -> None:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        port, thread = self.serve(server)
        try:
            status, payload = self.post_reload(port, None)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
            status, payload = self.post_reload(port, TOKEN)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_scope_policy_mode_answers_404_after_authentication(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="sestate-token-reload-gate-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        policy_path = os.path.join(tmpdir, "scopes.json")
        with open(policy_path, "wb") as handle:
            handle.write(json.dumps({"admin-token": ["admin"]}).encode("utf-8"))
        server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(policy_path)),
            scope_policy_file=policy_path,
        )
        port, thread = self.serve(server)
        try:
            # A missing or bad credential is still 401 in every mode.
            status, payload = self.post_reload(port, None)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
            # An authenticated admin token gets the unpublished-route 404.
            status, payload = self.post_reload(port, "admin-token")
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_single_token_server_without_a_file_is_503(self) -> None:
        # A server assembled directly with a token but no token file is in
        # single-token mode, so the route exists; the reload itself reports
        # the token as unavailable and keeps the live token.
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, auth_token=TOKEN)
        port, thread = self.serve(server)
        try:
            status, payload = self.post_reload(port, TOKEN)
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "auth_token_unavailable"})
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "GET", METRICS_PATH, headers={"Authorization": f"Bearer {TOKEN}"}
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
