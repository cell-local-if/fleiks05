"""Tests for runtime single-token hot reload.

``POST /v1/admin/auth-token/reload`` atomically replaces the live bearer
token in single-token mode by re-reading the file supplied at startup with
``--auth-token-file``, without a restart. The tests cover the full request
precedence chain (path shape, declared length, authentication, the
single-token mode gate, query, body), the 503/409 failure split with the
old token kept whole, the immediate old-token/new-token swap, serialized
reloads and snapshot-scoped authentication, restart still initializing from
the command-line file, and the absence of any business-state, data-file,
audit, or temporary-file effects. Neither token nor file content may appear
in any response.
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
    parse_auth_token,
    parse_empty_object_payload,
    read_auth_token_bytes,
)

OLD_TOKEN = "old-token-123"
NEW_TOKEN = "new-token-456"
RELOAD_PATH = "/v1/admin/auth-token/reload"
METRICS_PATH = "/v1/metrics"
OP_PATH = "/v1/replicas/r1/operations"

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class AuthTokenSplitLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-token-reload-unit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "token")

    def write(self, content: bytes) -> str:
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        with open(self.path, "wb") as handle:
            handle.write(content)
        return self.path

    def test_read_bytes_returns_raw_content_without_validation(self) -> None:
        # A regular file is returned byte-for-byte even when its content
        # would fail token validation, so the reload boundary can classify
        # read failures (503) separately from format failures (409).
        invalid = OLD_TOKEN.encode("ascii") + b"\n"
        self.assertEqual(read_auth_token_bytes(self.write(invalid)), invalid)

    def test_read_bytes_missing_file_is_auth_token_error(self) -> None:
        from semantic_state_engine.server import AuthTokenError

        with self.assertRaises(AuthTokenError):
            read_auth_token_bytes(os.path.join(self.tmpdir, "absent"))

    def test_read_bytes_directory_is_auth_token_error(self) -> None:
        from semantic_state_engine.server import AuthTokenError

        with self.assertRaises(AuthTokenError):
            read_auth_token_bytes(self.tmpdir)

    def test_parse_accepts_the_single_token_shapes(self) -> None:
        self.assertEqual(parse_auth_token(b"x"), "x")
        self.assertEqual(parse_auth_token(b"~!@#$%^&*()_+-="), "~!@#$%^&*()_+-=")
        self.assertEqual(parse_auth_token(OLD_TOKEN.encode("ascii")), OLD_TOKEN)

    def test_parse_rejects_every_invalid_shape(self) -> None:
        from semantic_state_engine.server import AuthTokenError

        for raw in (
            b"",
            b"two tokens",
            OLD_TOKEN.encode("ascii") + b"\n",
            b" token",
            b"token ",
            b"tok\ten",
            b"\x7f",
            b"\x20",
            "tökén".encode("utf-8"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(AuthTokenError):
                    parse_auth_token(raw)

    def test_load_still_combines_read_and_parse(self) -> None:
        self.assertEqual(load_auth_token(self.write(OLD_TOKEN.encode())), OLD_TOKEN)
        with self.assertRaises(Exception):
            load_auth_token(self.write(OLD_TOKEN.encode() + b"\n"))


class AuthTokenManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-token-mgr-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "token")
        with open(self.path, "wb") as handle:
            handle.write(OLD_TOKEN.encode("ascii"))
        self.manager = AuthTokenManager(self.path, OLD_TOKEN)

    def write(self, content: bytes) -> None:
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        with open(self.path, "wb") as handle:
            handle.write(content)

    def test_snapshot_returns_the_live_token(self) -> None:
        self.assertEqual(self.manager.snapshot(), OLD_TOKEN)
        self.assertEqual(self.manager.path, os.path.abspath(self.path))

    def test_reload_swaps_to_the_file_token(self) -> None:
        self.write(NEW_TOKEN.encode("ascii"))
        self.manager.reload()
        self.assertEqual(self.manager.snapshot(), NEW_TOKEN)

    def test_missing_file_is_unavailable_and_keeps_the_old_token(self) -> None:
        os.remove(self.path)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(self.manager.snapshot(), OLD_TOKEN)

    def test_directory_is_unavailable_and_keeps_the_old_token(self) -> None:
        os.remove(self.path)
        os.mkdir(self.path)
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(self.manager.snapshot(), OLD_TOKEN)

    @unittest.skipIf(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        "permission denials are bypassened for root",
    )
    def test_unreadable_file_is_unavailable_and_keeps_the_old_token(self) -> None:
        os.chmod(self.path, 0)
        try:
            with self.assertRaises(AuthTokenReloadError) as caught:
                self.manager.reload()
            self.assertEqual(caught.exception.kind, "unavailable")
        finally:
            os.chmod(self.path, 0o600)
        self.assertEqual(self.manager.snapshot(), OLD_TOKEN)

    def test_invalid_content_is_conflict_and_keeps_the_old_token(self) -> None:
        for content in (
            b"",
            OLD_TOKEN.encode("ascii") + b"\n",
            b"two tokens",
            "tök".encode("utf-8"),
        ):
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(AuthTokenReloadError) as caught:
                    self.manager.reload()
                self.assertEqual(caught.exception.kind, "conflict")
                self.assertEqual(self.manager.snapshot(), OLD_TOKEN)

    def test_reload_without_a_configured_path_is_unavailable(self) -> None:
        with self.assertRaises(AuthTokenReloadError) as caught:
            AuthTokenManager(None, OLD_TOKEN).reload()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_errors_never_echo_the_candidate_token(self) -> None:
        secret = "do-not-leak-me"
        self.write(secret.encode("ascii") + b"\n")
        with self.assertRaises(AuthTokenReloadError) as caught:
            self.manager.reload()
        self.assertNotIn(secret, str(caught.exception))


class AuthTokenReloadHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-token-reload-http-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.token_path = os.path.join(cls.tmpdir, "token")
        with open(cls.token_path, "wb") as handle:
            handle.write(OLD_TOKEN.encode("ascii"))
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=cls.data_path,
            auth_token=OLD_TOKEN,
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
        # Every test starts with the old token both on disk and live,
        # regardless of what an earlier test left on either side. Business
        # state accumulates in the one data file, which the restart test
        # relies on; the state-effect tests compare before/after snapshots
        # within the same test.
        self.write_token(OLD_TOKEN.encode("ascii"))
        self.server.auth_token_manager.reload()
        # Cleanups run last-added first: restore the file first, then reload
        # the live manager from it.
        self.addCleanup(self.server.auth_token_manager.reload)
        self.addCleanup(self.write_token, OLD_TOKEN.encode("ascii"))

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
        token: str | None = OLD_TOKEN,
    ) -> tuple[int, object, bytes]:
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
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw

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

    def reload(self, body: object = {}, token: str | None = OLD_TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    # -- success contract --

    def test_reload_returns_exactly_the_contracted_body(self) -> None:
        status, payload, raw = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "reloaded"})
        # No extension fields, no trailing line terminator, no token.
        self.assertEqual(raw.decode("utf-8"), '{"status":"reloaded"}')
        self.assertNotIn(OLD_TOKEN, raw.decode("utf-8"))
        self.assertNotIn(NEW_TOKEN, raw.decode("utf-8"))

    def test_new_token_authenticates_immediately_and_old_one_is_rejected(self) -> None:
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 401)
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, payload, _ = self.reload()
        self.assertEqual((status, payload), (200, {"status": "reloaded"}))
        # The rotation is immediate in both directions.
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 200)
        status, rejected, _ = self.request("GET", METRICS_PATH, token=OLD_TOKEN)
        self.assertEqual(status, 401)
        self.assertEqual(rejected, {"error": "unauthorized"})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", METRICS_PATH, headers={"Authorization": f"Bearer {OLD_TOKEN}"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        # A second reload from the same file is an idempotent no-op swap.
        status, _, _ = self.reload(token=NEW_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 200)

    def test_reload_must_authenticate_with_the_pre_rotation_token(self) -> None:
        self.write_token(NEW_TOKEN.encode("ascii"))
        # The token only in the not-yet-read file cannot drive the reload.
        status, payload, _ = self.reload(token=NEW_TOKEN)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, _, response_headers = self.raw_request(
            "POST",
            RELOAD_PATH,
            [("Content-Length", "2"), ("Authorization", f"Bearer {NEW_TOKEN}")],
            b"{}",
        )
        self.assertEqual(status, 401)
        self.assertEqual(response_headers.get("WWW-Authenticate"), "Bearer")
        # The failed attempt changed nothing: the old token is still live and
        # the file still waiting.
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)
        status, _, _ = self.reload(token=OLD_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 200)

    def test_missing_or_malformed_credential_is_401(self) -> None:
        for auth in (None, "Bearer", "Bearer ", "Bearer nope", f"Bearer {OLD_TOKEN} "):
            with self.subTest(auth=auth):
                headers = [("Content-Length", "2")]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, response_headers = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"{}"
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(response_headers.get("WWW-Authenticate"), "Bearer")

    # -- body validation --

    def test_body_must_be_exactly_the_empty_object(self) -> None:
        for body in (
            None,
            {"a": 1},
            {"x": {}},
            {"status": "reloaded"},
            {"token": NEW_TOKEN},
            [],
            "{}",
            0,
            True,
        ):
            with self.subTest(body=body):
                status, payload, _ = self.reload(body=body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        # None means no body at all: exercise the raw wire for that and the
        # other non-object JSON documents.
        for raw in (b"", b"junk", b"{", b"[]", b"null", b'"{}"', b"0", b"true"):
            with self.subTest(raw=raw):
                status, payload, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", str(len(raw))),
                        ("Authorization", f"Bearer {OLD_TOKEN}"),
                    ],
                    raw,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_object_with_json_whitespace_is_accepted(self) -> None:
        # Re-reloading an unchanged valid file succeeds; the live token is
        # already the new one after the first call, so every request in the
        # loop authenticates with it.
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _ = self.reload(token=OLD_TOKEN)
        self.assertEqual(status, 200)
        for raw in (b"{}", b"{ }", b"{\n\t}"):
            with self.subTest(raw=raw):
                status, _, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", str(len(raw))),
                        ("Authorization", f"Bearer {NEW_TOKEN}"),
                    ],
                    raw,
                )
                self.assertEqual(status, 200)

    def test_body_failure_reads_no_file_and_changes_no_token(self) -> None:
        # Even with the configured file missing, a bad body is 400, not 503:
        # the body is validated before the file is read.
        os.remove(self.token_path)
        status, payload, _ = self.reload(body={"a": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # Restore the file; the live token never moved.
        self.write_token(OLD_TOKEN.encode("ascii"))
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 401)

    # -- query parameters are rejected before the body --

    def test_any_query_parameter_is_400_even_with_a_valid_body(self) -> None:
        for query in ("?x", "?x=", "?x=1", "?x=1&y=2", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload, _ = self.request(
                    "POST", RELOAD_PATH + query, body={}, token=OLD_TOKEN
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_check_precedes_the_body_and_file_checks(self) -> None:
        # Malformed query plus malformed body plus a missing file still
        # reports the query 400 first.
        os.remove(self.token_path)
        status, payload, _ = self.raw_request(
            "POST",
            RELOAD_PATH + "?bogus=1",
            [
                ("Content-Length", "4"),
                ("Authorization", f"Bearer {OLD_TOKEN}"),
            ],
            b"junk",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.write_token(OLD_TOKEN.encode("ascii"))
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)

    # -- Content-Length keeps priority over everything --

    def test_missing_content_length_is_400_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {OLD_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = []
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_content_length_is_400(self) -> None:
        # Header values http.client transmits verbatim; the parser's own
        # whitespace/blank handling is covered by the request-limit tests.
        for value in ("abc", "-2", "1.0", "2 3"):
            with self.subTest(value=value):
                status, payload, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", value),
                        ("Authorization", f"Bearer {OLD_TOKEN}"),
                    ],
                    b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {OLD_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", OVER_LIMIT)]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"junk"
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_length_rejections_read_no_file_and_change_no_token(self) -> None:
        # Point the file at a valid new token: the 400/413 must still run
        # first, so the live token stays old afterwards.
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _ = self.raw_request(
            "POST", RELOAD_PATH, [("Authorization", f"Bearer {OLD_TOKEN}")], b"{}"
        )
        self.assertEqual(status, 400)
        status, _, _ = self.raw_request(
            "POST",
            RELOAD_PATH,
            [
                ("Content-Length", OVER_LIMIT),
                ("Authorization", f"Bearer {OLD_TOKEN}"),
            ],
            b"junk",
        )
        self.assertEqual(status, 413)
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 401)

    # -- file failures: 503 vs 409, old token always retained --

    def test_missing_file_is_503_and_keeps_the_old_token(self) -> None:
        os.remove(self.token_path)
        status, payload, _ = self.reload()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "auth_token_unavailable"})
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)
        self.write_token(OLD_TOKEN.encode("ascii"))

    def test_directory_target_is_503_and_keeps_the_old_token(self) -> None:
        os.remove(self.token_path)
        os.mkdir(self.token_path)
        status, payload, _ = self.reload()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "auth_token_unavailable"})
        self.assertEqual(self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200)

    def test_invalid_content_is_409_and_keeps_the_old_token(self) -> None:
        for content in (
            b"",
            NEW_TOKEN.encode("ascii") + b"\n",
            b"two tokens",
            b" leading",
            "nön-ascii".encode("utf-8"),
        ):
            with self.subTest(content=content):
                self.write_token(content)
                status, payload, raw = self.reload()
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "auth_token_conflict"})
                decoded = raw.decode("utf-8")
                self.assertNotIn(NEW_TOKEN, decoded)
                self.assertNotIn("two tokens", decoded)
                self.assertEqual(
                    self.request("GET", METRICS_PATH, token=OLD_TOKEN)[0], 200
                )
                self.assertEqual(
                    self.request("GET", METRICS_PATH, token=NEW_TOKEN)[0], 401
                )

    def test_server_built_without_a_token_file_reports_503(self) -> None:
        # Single-token mode is still active so the route is published, but no
        # startup file means the swap has no source.
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="standalone-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST",
                RELOAD_PATH,
                body=b"{}",
                headers={
                    "Content-Length": "2",
                    "Authorization": "Bearer standalone-token",
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 503)
            self.assertEqual(payload, {"error": "auth_token_unavailable"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # -- route shape and method --

    def test_wrong_path_shapes_are_404_after_authentication(self) -> None:
        for path in (
            "/v1/admin/auth-token",
            "/v1/admin/auth-token/reload/extra",
            "/v1/admin/auth-token/reload/",
            "/v1/admin/other/reload",
            "/v1/admin/auth-token/rotate",
        ):
            with self.subTest(path=path):
                status, payload, _ = self.request("POST", path, body={})
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
                # Without the credential the same shapes are 401, never 404.
                status, payload, _ = self.request("POST", path, body={}, token=None)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})

    def test_get_on_the_reload_path_is_404_after_authentication(self) -> None:
        status, payload, _ = self.request("GET", RELOAD_PATH, token=OLD_TOKEN)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # -- no business-state, persistence, or audit effects --

    def test_successful_rotation_changes_no_state(self) -> None:
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-1", key="color", value="blue"),
        )
        self.assertEqual(status, 201)
        _, metrics_before, _ = self.request("GET", "/v1/metrics")
        _, digest_before, _ = self.request("GET", "/v1/verification/digest")
        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))

        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _ = self.reload()
        self.assertEqual(status, 200)
        # A failed reload afterwards changes nothing either.
        self.write_token(b"bad token\n")
        status, _, _ = self.reload(token=NEW_TOKEN)
        self.assertEqual(status, 409)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        _, metrics_after, _ = self.request("GET", "/v1/metrics", token=NEW_TOKEN)
        _, digest_after, _ = self.request(
            "GET", "/v1/verification/digest", token=NEW_TOKEN
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)
        _, state, _ = self.request("GET", "/v1/states/color", token=NEW_TOKEN)
        self.assertEqual(state["value"], "blue")

    def test_neither_token_enters_the_data_file(self) -> None:
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-99", key="k99", value="v"),
        )
        self.assertEqual(status, 201)
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _ = self.reload()
        self.assertEqual(status, 200)
        with open(self.data_path, "rb") as handle:
            document_text = handle.read().decode("utf-8")
        self.assertNotIn(OLD_TOKEN, document_text)
        self.assertNotIn(NEW_TOKEN, document_text)

    def test_restart_still_initializes_auth_from_the_command_line_file(self) -> None:
        # Rotate the live token to the file's new value, then recover the
        # persisted state into a fresh server pointed at the same token file
        # and data file, exactly like a process restart.
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-restart", key="k-restart", value="kept"),
        )
        self.assertEqual(status, 201)
        self.write_token(NEW_TOKEN.encode("ascii"))
        status, _, _ = self.reload()
        self.assertEqual(status, 200)
        restarted = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=self.data_path,
            auth_token=load_auth_token(self.token_path),
            auth_token_file=self.token_path,
        )
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        try:
            port = restarted.server_address[1]

            def get(path: str, token: str | None) -> int:
                headers = {}
                if token is not None:
                    headers["Authorization"] = f"Bearer {token}"
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", path, headers=headers)
                response = conn.getresponse()
                response.read()
                conn.close()
                return response.status

            self.assertEqual(get("/v1/states/k-restart", None), 401)
            self.assertEqual(get("/v1/states/k-restart", OLD_TOKEN), 401)
            self.assertEqual(get("/v1/states/k-restart", NEW_TOKEN), 200)
        finally:
            restarted.shutdown()
            restarted.server_close()
            thread.join(timeout=5)

    # -- concurrency --

    def test_concurrent_reloads_serialize_without_partial_state(self) -> None:
        token_a = "concurrent-token-aaaa"
        token_b = "concurrent-token-bbbb"

        manager = self.server.auth_token_manager
        errors: list[Exception] = []

        def rotate(candidate: str, temp_name: str) -> None:
            # Publish each candidate with an atomic rename so a reload never
            # observes a truncated or empty file; the manager lock then
            # serializes the read-validate-swap units themselves.
            staging = os.path.join(self.tmpdir, temp_name)
            for _ in range(25):
                try:
                    with open(staging, "wb") as handle:
                        handle.write(candidate.encode("ascii"))
                    os.replace(staging, self.token_path)
                    manager.reload()
                except Exception as exc:  # noqa: BLE001 - recorded for assertion
                    errors.append(exc)

        threads = [
            threading.Thread(target=rotate, args=(token_a, "tmp-a")),
            threading.Thread(target=rotate, args=(token_b, "tmp-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        live = manager.snapshot()
        self.assertIn(live, (token_a, token_b))
        # Exactly one of the two tokens authenticates; the other is rejected.
        other = token_b if live == token_a else token_a
        self.assertEqual(self.request("GET", METRICS_PATH, token=live)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token=other)[0], 401)


class AuthTokenReloadModeGatingTests(unittest.TestCase):
    def start_server(self, **kwargs) -> tuple[SemanticStateServer, threading.Thread, int]:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, server.server_address[1]

    def post(
        self, port: int, token: str | None, body: bytes = b"{}"
    ) -> tuple[int, object]:
        headers = {"Content-Length": str(len(body))}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", RELOAD_PATH, body=body, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_endpoint_is_404_when_authentication_is_disabled(self) -> None:
        server, thread, port = self.start_server()
        try:
            status, payload = self.post(port, None)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_endpoint_is_404_in_scope_policy_mode_but_401_without_a_token(self) -> None:
        scopes = {
            "admin-token": frozenset({"read", "write", "admin"}),
            "reader-token": frozenset({"read"}),
        }
        server, thread, port = self.start_server(auth_scopes=dict(scopes))
        try:
            # A valid admin policy token still sees an unpublished route:
            # this is single-token mode's endpoint.
            status, payload = self.post(port, "admin-token")
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
            # Authentication runs before the mode gate.
            status, payload = self.post(port, None)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
            status, payload = self.post(port, "stranger")
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
            # The scope decision precedes the mode gate, exactly like every
            # other admin endpoint: an authenticated token without admin is
            # 403, never a route-level 404.
            status, payload = self.post(port, "reader-token")
            self.assertEqual(status, 403)
            self.assertEqual(payload, {"error": "forbidden"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_health_stays_anonymous_through_a_rotation(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="sestate-token-health-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        token_path = os.path.join(tmpdir, "token")
        with open(token_path, "wb") as handle:
            handle.write(OLD_TOKEN.encode("ascii"))
        server, thread, port = self.start_server(
            auth_token=OLD_TOKEN, auth_token_file=token_path
        )
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["status"], "ok")

            with open(token_path, "wb") as handle:
                handle.write(NEW_TOKEN.encode("ascii"))
            status, payload = self.post(port, OLD_TOKEN)
            self.assertEqual((status, payload), (200, {"status": "reloaded"}))

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
