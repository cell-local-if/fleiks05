"""Tests for the scope-policy authentication mode.

The scope policy is an opt-in second authentication configuration:
``--scope-policy-file`` maps bearer tokens to read/write/admin scopes. It is
mutually exclusive with the legacy single token, validates strictly at
startup, and once running enforces authentication before the method scope
and the scope before route/query handling.
"""

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
    RequestHandler,
    ScopePolicyError,
    SemanticStateServer,
    load_scope_policy,
    main,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
RW_TOKEN = "read-write-token"

POLICY = {
    READ_TOKEN: frozenset({"read"}),
    WRITE_TOKEN: frozenset({"write"}),
    ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
    RW_TOKEN: frozenset({"read", "write"}),
}

OP_PATH = "/v1/replicas/r1/operations"
SYNC_PATH = "/v1/sync/operations"
METRICS_PATH = "/v1/metrics"
STATE_PATH = "/v1/states/color"
CHECKPOINT_PATH = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_PATHS = (OP_PATH, SYNC_PATH, "/v1/states/k/resolve", CHECKPOINT_PATH)

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


def policy_file(tmpdir: str, content: bytes, name: str = "scopes.json") -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


class LoadScopePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-scope-load-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def load_bytes(self, content: bytes) -> dict[str, frozenset[str]]:
        return load_scope_policy(policy_file(self.tmpdir, content))

    def test_valid_policy_is_loaded(self) -> None:
        loaded = self.load_bytes(
            json.dumps(
                {
                    READ_TOKEN: ["read"],
                    WRITE_TOKEN: ["write"],
                    ADMIN_TOKEN: ["read", "write", "admin"],
                }
            ).encode("utf-8")
        )
        self.assertEqual(
            loaded,
            {
                READ_TOKEN: frozenset({"read"}),
                WRITE_TOKEN: frozenset({"write"}),
                ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
            },
        )
        self.assertIsInstance(loaded[READ_TOKEN], frozenset)

    def test_single_scope_each_and_reordered_arrays_load(self) -> None:
        loaded = self.load_bytes(b'{"t":["admin"],"u":["write","read"]}')
        self.assertEqual(loaded, {"t": frozenset({"admin"}), "u": frozenset({"read", "write"})})

    def test_empty_object_is_a_valid_policy_with_no_tokens(self) -> None:
        self.assertEqual(self.load_bytes(b"{}"), {})

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(ScopePolicyError):
            load_scope_policy(os.path.join(self.tmpdir, "absent"))

    def test_directory_is_rejected(self) -> None:
        with self.assertRaises(ScopePolicyError):
            load_scope_policy(self.tmpdir)

    def test_empty_file_is_rejected(self) -> None:
        with self.assertRaises(ScopePolicyError):
            self.load_bytes(b"")

    def test_incomplete_json_is_rejected(self) -> None:
        for content in (b'{"t":', b'{"t":[', b'{"t":["read"]', b"not json", b"[", b'"'):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyError):
                    self.load_bytes(content)

    def test_non_object_json_is_rejected(self) -> None:
        for content in (b'["read"]', b'"read"', b"42", b"true", b"null"):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyError):
                    self.load_bytes(content)

    def test_non_utf8_file_is_rejected(self) -> None:
        with self.assertRaises(ScopePolicyError):
            self.load_bytes('{"tök":["read"]}'.encode("latin-1") + b"\xff")

    def test_duplicate_token_key_is_rejected(self) -> None:
        with self.assertRaises(ScopePolicyError):
            self.load_bytes(b'{"t":["read"],"t":["write"]}')

    def test_illegal_token_keys_are_rejected(self) -> None:
        for content in (
            b'{"":["read"]}',
            b'{"two tokens":["read"]}',
            b'{"tok\ten":["read"]}',
            b'{" token":["read"]}',
            b'{"token\n":["read"]}',
            '{"tök":["read"]}'.encode("utf-8"),
        ):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyError):
                    self.load_bytes(content)

    def test_value_must_be_a_non_empty_array(self) -> None:
        for content in (
            b'{"t":[]}',
            b'{"t":"read"}',
            b'{"t":null}',
            b'{"t":{}}',
            b'{"t":42}',
            b'{"t":true}',
        ):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyError):
                    self.load_bytes(content)

    def test_unknown_and_duplicate_scopes_are_rejected(self) -> None:
        for content in (
            b'{"t":["administer"]}',
            b'{"t":["READ"]}',
            b'{"t":[null]}',
            b'{"t":[42]}',
            b'{"t":["read","read"]}',
            b'{"t":["read",null]}',
            b'{"t":["read","write","read"]}',
        ):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyError):
                    self.load_bytes(content)

    def test_error_never_echoes_tokens_or_scopes(self) -> None:
        secret = "do-not-leak-this-token"
        with self.assertRaises(ScopePolicyError) as caught:
            self.load_bytes(json.dumps({secret: ["bogus"]}).encode("utf-8"))
        self.assertNotIn(secret, str(caught.exception))
        with self.assertRaises(ScopePolicyError) as caught:
            self.load_bytes(json.dumps({secret: ["read", "read"]}).encode("utf-8"))
        self.assertNotIn(secret, str(caught.exception))


class StartupFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-scope-cli-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def run_main(self, argv: list) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        return caught.exception.code, stderr.getvalue()

    def test_mutually_exclusive_auth_arguments_fail_with_exit_2(self) -> None:
        # Neither path needs to exist: the mutual-exclusion check fails the
        # startup before any file is read.
        code, stderr = self.run_main(
            [
                "--auth-token-file",
                os.path.join(self.tmpdir, "token"),
                "--scope-policy-file",
                os.path.join(self.tmpdir, "scopes.json"),
                "--port",
                "0",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)

    def test_mutual_exclusion_leaks_no_content(self) -> None:
        token_path = os.path.join(self.tmpdir, "token")
        policy_path = os.path.join(self.tmpdir, "scopes.json")
        with open(token_path, "wb") as handle:
            handle.write(b"super-secret-token-value")
        with open(policy_path, "wb") as handle:
            handle.write(b'{"super-secret-policy-token":["read"]}')
        code, stderr = self.run_main(
            ["--auth-token-file", token_path, "--scope-policy-file", policy_path]
        )
        self.assertEqual(code, 2)
        self.assertNotIn("super-secret-token-value", stderr)
        self.assertNotIn("super-secret-policy-token", stderr)

    def test_missing_policy_file_fails_startup(self) -> None:
        code, stderr = self.run_main(
            ["--scope-policy-file", os.path.join(self.tmpdir, "absent")]
        )
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)

    def test_directory_policy_path_fails_startup(self) -> None:
        code, stderr = self.run_main(["--scope-policy-file", self.tmpdir])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)

    def test_malformed_policy_fails_without_leaking_tokens(self) -> None:
        secret = "super-secret-scope-token"
        path = policy_file(self.tmpdir, json.dumps({secret: ["nope"]}).encode("utf-8"))
        code, stderr = self.run_main(["--scope-policy-file", path])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)
        self.assertNotIn(secret, stderr)

    def test_truncated_json_policy_fails_startup(self) -> None:
        path = policy_file(self.tmpdir, b'{"t":["read"]')
        code, stderr = self.run_main(["--scope-policy-file", path])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)


class ScopePolicyServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(POLICY)
        )
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
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = READ_TOKEN,
    ) -> tuple[int, dict, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def put_raw(self, method: str, path: str, headers: list, body: bytes | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest(method, path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def assert_unauthorized(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def assert_forbidden(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 403)
        # The error body carries only the error field and there is no
        # Bearer challenge on an authenticated-but-unauthorized request.
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("Www-Authenticate", headers)
        self.assertNotIn("WWW-Authenticate", headers)

    # -- /health stays anonymous in scope mode --

    def test_health_is_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    # -- the authorization matrix --

    def test_read_scope_reaches_every_get(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        self.assertEqual(status, 200)
        status, _, _ = self.request("GET", STATE_PATH, token=READ_TOKEN)
        self.assertEqual(status, 404)

    def test_read_scope_cannot_post(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload, headers = self.request(
                    "POST", path, operation_document(), token=READ_TOKEN
                )
                self.assert_forbidden(status, payload, headers)

    def test_write_scope_submits_posts(self) -> None:
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")

    def test_write_scope_cannot_get(self) -> None:
        for path in (METRICS_PATH, STATE_PATH, SYNC_PATH, "/nope"):
            with self.subTest(path=path):
                status, payload, headers = self.request("GET", path, token=WRITE_TOKEN)
                self.assert_forbidden(status, payload, headers)

    def test_admin_scope_covers_both_classes(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-9", key="a", value="b"),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 201)

    def test_read_write_scope_covers_both_classes(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=RW_TOKEN)
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-9", key="a", value="b"),
            token=RW_TOKEN,
        )
        self.assertEqual(status, 201)

    # -- authentication failures keep the 401/challenge contract --

    def test_missing_header_is_401(self) -> None:
        status, payload, headers = self.request("GET", METRICS_PATH, token=None)
        self.assert_unauthorized(status, payload, headers)

    def test_unknown_token_is_401(self) -> None:
        status, payload, headers = self.request("GET", METRICS_PATH, token="not-configured")
        self.assert_unauthorized(status, payload, headers)

    def test_malformed_authorization_is_401(self) -> None:
        for value in ("Bearer", "Basic x", f"Bearer  {READ_TOKEN}", READ_TOKEN, "bearer " + READ_TOKEN):
            with self.subTest(value=value):
                status, payload, headers = self.put_raw(
                    "GET", METRICS_PATH, [("Authorization", value)]
                )
                self.assert_unauthorized(status, payload, headers)

    def test_duplicate_authorization_headers_are_401_even_with_matching_tokens(self) -> None:
        status, payload, headers = self.put_raw(
            "GET",
            METRICS_PATH,
            [("Authorization", f"Bearer {ADMIN_TOKEN}"), ("Authorization", f"Bearer {ADMIN_TOKEN}")],
        )
        self.assert_unauthorized(status, payload, headers)

    # -- scope decision precedes routing and query validation --

    def test_missing_scope_on_get_is_403_before_404_and_400(self) -> None:
        # Unknown route: scope failure must not surface as 404.
        status, payload, headers = self.request("GET", "/nope", token=WRITE_TOKEN)
        self.assert_forbidden(status, payload, headers)
        # Known route with a malformed paging query: scope failure must not
        # surface as 400.
        status, payload, headers = self.request(
            "GET", "/v1/sync/operations?limit=bogus", token=WRITE_TOKEN
        )
        self.assert_forbidden(status, payload, headers)

    def test_missing_scope_on_post_is_403_before_404(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        status, payload, headers = self.put_raw(
            "POST",
            "/v1/unknown",
            [("Content-Length", str(len(body))), ("Authorization", f"Bearer {READ_TOKEN}")],
            body,
        )
        self.assert_forbidden(status, payload, headers)

    def test_correct_scope_still_runs_route_and_query_validation(self) -> None:
        # The read token passes the scope gate, so a malformed query gives
        # the normal 400 rather than 403.
        status, payload, _ = self.request(
            "GET", "/v1/sync/operations?limit=bogus", token=READ_TOKEN
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- Content-Length keeps priority over both auth and scope --

    def test_missing_content_length_is_400_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {READ_TOKEN}", f"Bearer {WRITE_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = []
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.put_raw("POST", OP_PATH, headers, b"{}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {READ_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", OVER_LIMIT)]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.put_raw("POST", OP_PATH, headers, b"junk")
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_valid_length_but_missing_scope_is_403_without_reading_body(self) -> None:
        # Declare a length but never send the body: the 403 must arrive
        # without the server waiting for those bytes.
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.putrequest("POST", path)
                conn.putheader("Content-Length", "64")
                conn.putheader("Authorization", f"Bearer {READ_TOKEN}")
                conn.endheaders()
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.getheaders())
                conn.close()
                self.assert_forbidden(response.status, payload, headers)

    def test_valid_length_but_bad_token_is_401_without_reading_body(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", OP_PATH)
        conn.putheader("Content-Length", "64")
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)

    # -- rejected requests have no effects --

    def test_rejected_requests_change_no_state(self) -> None:
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)
        _, metrics_before, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        _, digest_before, _ = self.request(
            "GET", "/v1/verification/digest", token=ADMIN_TOKEN
        )

        # 403s with a valid but under-scoped token across POST endpoints.
        body = json.dumps(operation_document(op_id="op-2")).encode("utf-8")
        for path in ALL_POST_PATHS:
            status, _, _ = self.put_raw(
                "POST",
                path,
                [
                    ("Content-Length", str(len(body))),
                    ("Authorization", f"Bearer {READ_TOKEN}"),
                ],
                body,
            )
            self.assertEqual(status, 403)
        # 403s with the write token across GETs, plus 401s with no/unknown
        # credentials across both classes.
        for path in (METRICS_PATH, STATE_PATH, SYNC_PATH, "/nope"):
            status, _, _ = self.request("GET", path, token=WRITE_TOKEN)
            self.assertEqual(status, 403)
            status, _, _ = self.request("GET", path, token=None)
            self.assertEqual(status, 401)
            status, _, _ = self.request("GET", path, token="stranger")
            self.assertEqual(status, 401)

        _, metrics_after, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        _, digest_after, _ = self.request(
            "GET", "/v1/verification/digest", token=ADMIN_TOKEN
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)

    # -- authorized traffic keeps every existing semantic --

    def test_authorized_write_keeps_idempotency_conflict_and_read_semantics(self) -> None:
        doc = operation_document(op_id="op-1", key="color", value="blue")
        status, _, _ = self.request("POST", OP_PATH, doc, token=WRITE_TOKEN)
        self.assertEqual(status, 201)
        status, _, _ = self.request("POST", OP_PATH, doc, token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="red"),
            token=RW_TOKEN,
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        status, state, _ = self.request("GET", STATE_PATH, token=READ_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")


class ScopePolicyPersistenceTests(unittest.TestCase):
    """The policy is not persisted and rejected requests never touch disk."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-scope-data-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.policy = dict(POLICY)
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=cls.data_path, auth_scopes=cls.policy,
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
        self, method: str, path: str, document: dict = None, token: str | None = WRITE_TOKEN
    ) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if document is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(document), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_rejected_requests_leave_file_and_directory_untouched(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)

        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))

        for path in ALL_POST_PATHS:
            status, payload = self.request(
                "POST", path, operation_document(op_id="op-2"), token=READ_TOKEN
            )
            self.assertEqual(status, 403)
            self.assertEqual(payload, {"error": "forbidden"})
            status, payload = self.request(
                "POST", path, operation_document(op_id="op-2"), token="stranger"
            )
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
        status, _ = self.request("GET", METRICS_PATH, token=WRITE_TOKEN)
        self.assertEqual(status, 403)
        status, _ = self.request("GET", METRICS_PATH, token=None)
        self.assertEqual(status, 401)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)

    def test_data_file_never_contains_policy_tokens_or_scopes(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-10", key="k10", value="v"),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 201)
        with open(self.data_path, "rb") as handle:
            raw = handle.read()
        document = json.loads(raw.decode("utf-8"))
        self.assertLessEqual(
            set(document.keys()),
            {"version", "operations", "checkpoints", "policies", "transactions", "acks"},
        )
        for configured_token in POLICY:
            self.assertNotIn(configured_token.encode("utf-8"), raw)

    def test_restart_requires_the_policy_to_be_supplied_again(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-20", key="k20", value="kept"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

        # Restart with the policy re-supplied: data recovered, enforcement
        # is back in force.
        locked = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path, auth_scopes=dict(POLICY),
        )
        thread = threading.Thread(target=locked.serve_forever, daemon=True)
        thread.start()
        try:
            port = locked.server_address[1]

            def call(method: str, path: str, token: str | None) -> int:
                headers = {} if token is None else {"Authorization": f"Bearer {token}"}
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(method, path, headers=headers)
                response = conn.getresponse()
                response.read()
                conn.close()
                return response.status

            self.assertEqual(call("GET", "/v1/states/k20", None), 401)
            self.assertEqual(call("GET", "/v1/states/k20", WRITE_TOKEN), 403)
            self.assertEqual(call("GET", "/v1/states/k20", READ_TOKEN), 200)
        finally:
            locked.shutdown()
            locked.server_close()
            thread.join(timeout=5)

        # Restart without re-supplying any authentication configuration:
        # the policy is not recovered from the data file, so the service is
        # anonymous again while the data itself still recovers.
        reopened = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_path
        )
        thread = threading.Thread(target=reopened.serve_forever, daemon=True)
        thread.start()
        try:
            port = reopened.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/states/k20")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["value"], "kept")
        finally:
            reopened.shutdown()
            reopened.server_close()
            thread.join(timeout=5)

        # Reopen the configured server for any later tests in this class.
        type(self).server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path, auth_scopes=self.policy,
        )
        type(self).thread = threading.Thread(
            target=type(self).server.serve_forever, daemon=True
        )
        type(self).thread.start()
        type(self).port = type(self).server.server_address[1]


if __name__ == "__main__":
    unittest.main()
