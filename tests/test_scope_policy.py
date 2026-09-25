"""Tests for scope-policy bearer authentication (``--auth-policy-file``)."""

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
    AuthPolicyError,
    RequestHandler,
    SemanticStateServer,
    load_auth_policy,
    main,
)

READ_TOKEN = "read-token"
WRITE_TOKEN = "write-token"
ADMIN_TOKEN = "admin-token"
RW_TOKEN = "rw-token"

POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["admin"],
    RW_TOKEN: ["read", "write"],
}
# The server stores each token's scopes as a frozenset.
POLICY_SETS = {token: frozenset(scopes) for token, scopes in POLICY.items()}

OP_PATH = "/v1/replicas/r1/operations"
SYNC_PATH = "/v1/sync/operations"
METRICS_PATH = "/v1/metrics"
CHECKPOINT_PATH = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_PATHS = (OP_PATH, SYNC_PATH, "/v1/states/k/resolve", CHECKPOINT_PATH)

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class LoadAuthPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def write_policy(self, content: bytes) -> str:
        path = os.path.join(self.tmpdir, "policy.json")
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def load_bytes(self, content: bytes) -> dict:
        return load_auth_policy(self.write_policy(content))

    def test_valid_policy_is_loaded(self) -> None:
        policy = self.load_bytes(
            json.dumps({"a": ["read"], "b": ["write", "admin"]}).encode("utf-8")
        )
        self.assertEqual(policy["a"], frozenset({"read"}))
        self.assertEqual(policy["b"], frozenset({"write", "admin"}))

    def test_all_three_scopes_accepted(self) -> None:
        for scope in ("read", "write", "admin"):
            with self.subTest(scope=scope):
                policy = self.load_bytes(json.dumps({"t": [scope]}).encode("utf-8"))
                self.assertEqual(policy["t"], frozenset({scope}))

    def test_single_character_token_key_is_valid(self) -> None:
        policy = self.load_bytes(b'{"x":["read"]}')
        self.assertIn("x", policy)

    def test_punctuation_token_keys_are_valid(self) -> None:
        policy = self.load_bytes(b'{"a.b-c_d~!@#$%^&*()+":["read"]}')
        self.assertIn("a.b-c_d~!@#$%^&*()+", policy)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            load_auth_policy(os.path.join(self.tmpdir, "absent"))

    def test_directory_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            load_auth_policy(self.tmpdir)

    def test_non_utf8_file_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t\xf6":["read"]}')

    def test_non_object_root_is_rejected(self) -> None:
        for content in (b"[1, 2]", b'"read"', b"42", b"null", b"true"):
            with self.subTest(content=content):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(content)

    def test_empty_file_and_whitespace_only_are_rejected(self) -> None:
        for content in (b"", b"   ", b"\n"):
            with self.subTest(content=content):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(content)

    def test_truncated_json_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t":["read"]')

    def test_trailing_content_is_rejected(self) -> None:
        for content in (b'{"t":["read"]} garbage', b'{"t":["read"]}{}', b'{"t":["read"]}\n{}'):
            with self.subTest(content=content):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(content)

    def test_duplicate_token_key_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t":["read"],"t":["write"]}')

    def test_empty_token_key_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"":["read"]}')

    def test_whitespace_token_keys_are_rejected(self) -> None:
        for key in ("a b", "a\tb", " a", "a ", "a\nb"):
            with self.subTest(key=key):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(json.dumps({key: ["read"]}).encode("utf-8"))

    def test_non_ascii_token_key_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(json.dumps({"tök": ["read"]}).encode("utf-8"))

    def test_null_scope_value_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t":null}')

    def test_non_array_scope_value_is_rejected(self) -> None:
        for value in (b'"read"', b"42", b'{"read":true}', b"true"):
            content = b'{"t":' + value + b"}"
            with self.subTest(content=content):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(content)

    def test_empty_scope_array_is_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t":[]}')

    def test_duplicate_scopes_are_rejected(self) -> None:
        with self.assertRaises(AuthPolicyError):
            self.load_bytes(b'{"t":["read","read"]}')

    def test_unknown_scope_is_rejected(self) -> None:
        for scopes in (b'["execute"]', b'["Read"]', b'["READ"]', b'["admin","root"]'):
            with self.subTest(scopes=scopes):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(b'{"t":' + scopes + b"}")

    def test_non_string_scope_is_rejected(self) -> None:
        for scopes in (b"[1]", b"[null]", b"[true]", b'[["read"]]'):
            with self.subTest(scopes=scopes):
                with self.assertRaises(AuthPolicyError):
                    self.load_bytes(b'{"t":' + scopes + b"}")

    def test_empty_object_is_a_valid_document(self) -> None:
        self.assertEqual(self.load_bytes(b"{}"), {})

    def test_error_never_echoes_tokens_or_scopes(self) -> None:
        secret = "do-not-leak-this-token"
        with self.assertRaises(AuthPolicyError) as caught:
            self.load_bytes(json.dumps({secret: ["bogus"]}).encode("utf-8"))
        self.assertNotIn(secret, str(caught.exception))


class PolicyStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-cli-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.policy_path = os.path.join(self.tmpdir, "policy.json")
        with open(self.policy_path, "wb") as handle:
            handle.write(json.dumps(POLICY).encode("utf-8"))
        self.token_path = os.path.join(self.tmpdir, "token")
        with open(self.token_path, "wb") as handle:
            handle.write(b"single-token")

    def run_main(self, argv: list) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        return caught.exception.code, stderr.getvalue()

    def test_missing_policy_file_fails_startup_like_a_data_file_error(self) -> None:
        code, stderr = self.run_main(
            ["--auth-policy-file", os.path.join(self.tmpdir, "absent")]
        )
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)

    def test_malformed_policy_fails_without_leaking_tokens(self) -> None:
        secret = "super-secret-policy-token"
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "wb") as handle:
            handle.write(json.dumps({secret: ["nope"]}).encode("utf-8"))
        code, stderr = self.run_main(["--auth-policy-file", path])
        self.assertEqual(code, 2)
        self.assertIn("startup failed", stderr)
        self.assertNotIn(secret, stderr)

    def test_token_and_policy_options_are_mutually_exclusive(self) -> None:
        code, stderr = self.run_main(
            ["--auth-token-file", self.token_path, "--auth-policy-file", self.policy_path]
        )
        self.assertEqual(code, 2)
        self.assertNotIn("single-token", stderr)

    def test_server_constructor_rejects_both_configurations(self) -> None:
        with self.assertRaises(ValueError):
            SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler,
                auth_token="single-token", auth_policy={"t": frozenset({"read"})},
            )

    def test_valid_policy_does_not_leak_tokens_on_failure_elsewhere(self) -> None:
        code, stderr = self.run_main(
            [
                "--auth-policy-file",
                self.policy_path,
                "--data-file",
                os.path.join(self.tmpdir, "no-such-dir", "state.json"),
            ]
        )
        self.assertEqual(code, 2)
        self.assertNotIn(READ_TOKEN, stderr)
        self.assertNotIn(WRITE_TOKEN, stderr)


class ScopePolicyServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_policy=dict(POLICY_SETS)
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
        payload = json.loads(response.read().decode("utf-8"))
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def post_raw(self, path: str, headers: list, body: bytes | None = None) -> tuple[int, dict, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")), response_headers

    def assert_unauthorized(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def assert_forbidden(self, status: int, payload: dict, headers: dict) -> None:
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("Www-Authenticate", headers)
        self.assertNotIn("WWW-Authenticate", headers)

    # -- health stays anonymous --

    def test_health_is_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    # -- scope matrix --

    def test_read_scope_allows_get(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        self.assertEqual(status, 200)

    def test_read_scope_denies_post(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload, headers = self.request(
                    "POST", path, operation_document(), token=READ_TOKEN
                )
                self.assert_forbidden(status, payload, headers)

    def test_write_scope_allows_post(self) -> None:
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)

    def test_write_scope_denies_get(self) -> None:
        status, payload, headers = self.request("GET", METRICS_PATH, token=WRITE_TOKEN)
        self.assert_forbidden(status, payload, headers)

    def test_admin_scope_covers_get_and_post(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 201)

    def test_read_write_scope_covers_both_methods(self) -> None:
        status, _, _ = self.request("GET", METRICS_PATH, token=RW_TOKEN)
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=RW_TOKEN,
        )
        self.assertEqual(status, 201)

    # -- authentication failures stay 401 even in policy mode --

    def test_missing_header_is_401_for_get_and_post(self) -> None:
        status, payload, headers = self.request("GET", METRICS_PATH, token=None)
        self.assert_unauthorized(status, payload, headers)
        body = json.dumps(operation_document()).encode("utf-8")
        status, payload, headers = self.post_raw(
            OP_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assert_unauthorized(status, payload, headers)

    def test_unknown_token_is_401(self) -> None:
        status, payload, headers = self.request("GET", METRICS_PATH, token="nobody")
        self.assert_unauthorized(status, payload, headers)

    def test_malformed_authorization_is_401(self) -> None:
        for value in ("Bearer", f"Bearer  {READ_TOKEN}", f"bearer {READ_TOKEN}",
                      READ_TOKEN, f"Bearer {READ_TOKEN} extra", "Basic x"):
            with self.subTest(value=value):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request("GET", METRICS_PATH, headers={"Authorization": value})
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.getheaders())
                conn.close()
                self.assert_unauthorized(response.status, payload, headers)

    def test_duplicate_authorization_headers_are_401_even_with_scope(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", METRICS_PATH)
        conn.putheader("Authorization", f"Bearer {READ_TOKEN}")
        conn.putheader("Authorization", f"Bearer {READ_TOKEN}")
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)

    # -- the scope decision precedes routing and query validation --

    def test_missing_scope_on_unknown_get_is_403_not_404(self) -> None:
        # write token has no read scope: the 403 is answered before route
        # matching even though the path does not exist.
        status, payload, headers = self.request("GET", "/nope", token=WRITE_TOKEN)
        self.assert_forbidden(status, payload, headers)
        status, _, _ = self.request("GET", "/nope", token=READ_TOKEN)
        self.assertEqual(status, 404)

    def test_missing_scope_on_unknown_post_is_403_not_404(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        status, payload, headers = self.post_raw(
            "/v1/unknown",
            [("Content-Length", str(len(body))), ("Authorization", f"Bearer {READ_TOKEN}")],
            body,
        )
        self.assert_forbidden(status, payload, headers)
        status, payload, _ = self.post_raw(
            "/v1/unknown",
            [("Content-Length", str(len(body))), ("Authorization", f"Bearer {WRITE_TOKEN}")],
            body,
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_scope_precedes_query_validation(self) -> None:
        # /v1/metrics rejects any query parameter with 400, but the scope
        # check comes first.
        status, payload, headers = self.request(
            "GET", "/v1/metrics?x=1", token=WRITE_TOKEN
        )
        self.assert_forbidden(status, payload, headers)
        status, _, _ = self.request("GET", "/v1/metrics?x=1", token=READ_TOKEN)
        self.assertEqual(status, 400)

    # -- Content-Length keeps priority over the scope check --

    def test_missing_content_length_is_400_regardless_of_scope(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload, _ = self.post_raw(path, [], b"{}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_even_without_write_scope(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload, _ = self.post_raw(
                    path,
                    [
                        ("Content-Length", OVER_LIMIT),
                        ("Authorization", f"Bearer {READ_TOKEN}"),
                    ],
                    b"junk",
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_valid_length_without_scope_is_403_without_reading_body(self) -> None:
        # Declare a length but never send the body: the 403 must arrive
        # without the server waiting for body bytes.
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

    # -- rejected and authorized requests change nothing --

    def test_forbidden_request_changes_no_state(self) -> None:
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 201)
        _, metrics_before, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        _, digest_before, _ = self.request(
            "GET", "/v1/verification/digest", token=READ_TOKEN
        )

        # write-less tokens attempt POSTs; read-less tokens attempt GETs.
        body = json.dumps(operation_document(op_id="op-2")).encode("utf-8")
        for path in ALL_POST_PATHS:
            status, _, _ = self.post_raw(
                path,
                [
                    ("Content-Length", str(len(body))),
                    ("Authorization", f"Bearer {READ_TOKEN}"),
                ],
                body,
            )
            self.assertEqual(status, 403)
        for path in (METRICS_PATH, "/v1/sync/operations", "/v1/states/color"):
            status, _, _ = self.request("GET", path, token=WRITE_TOKEN)
            self.assertEqual(status, 403)

        _, metrics_after, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        _, digest_after, _ = self.request(
            "GET", "/v1/verification/digest", token=READ_TOKEN
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)

    def test_authorized_writes_keep_existing_semantics(self) -> None:
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        # Identical replay stays 200, a conflicting identity stays 409.
        status, _, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue"),
            token=RW_TOKEN,
        )
        self.assertEqual(status, 200)
        status, payload, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="red"),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        status, state, _ = self.request("GET", "/v1/states/color", token=READ_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")
        status, metrics, _ = self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(metrics["acceptedOperations"], 1)


class ScopePolicyPersistenceTests(unittest.TestCase):
    """With --data-file, policy failures touch neither the file nor state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-data-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.policy = {token: frozenset(scopes) for token, scopes in POLICY.items()}
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=cls.data_path, auth_policy=cls.policy,
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
        self, method: str, path: str, document: dict | None = None, token: str | None = ADMIN_TOKEN
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

    def test_forbidden_and_unauthorized_leave_file_untouched(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        status, _ = self.request("POST", CHECKPOINT_PATH, {"cursor": 1})
        self.assertEqual(status, 200)

        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        _, metrics_before = self.request("GET", METRICS_PATH)
        _, checkpoint_before = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )

        # read token on a POST -> 403; write token on a GET -> 403; no
        # Authorization at all -> 401.
        status, payload = self.request(
            "POST", OP_PATH, operation_document(op_id="op-2"), token=READ_TOKEN
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        status, payload = self.request("GET", METRICS_PATH, token=WRITE_TOKEN)
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        status, payload = self.request("GET", METRICS_PATH, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        _, metrics_after = self.request("GET", METRICS_PATH)
        self.assertEqual(metrics_after, metrics_before)
        _, checkpoint_after = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )
        self.assertEqual(checkpoint_after, checkpoint_before)

    def test_policy_is_never_written_to_data_file(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-10", key="k10", value="v")
        )
        self.assertEqual(status, 201)
        with open(self.data_path, "rb") as handle:
            document = json.loads(handle.read().decode("utf-8"))
        self.assertLessEqual(
            set(document.keys()),
            {"version", "operations", "checkpoints", "policies", "transactions", "acks"},
        )
        serialized = json.dumps(document)
        for token in POLICY:
            self.assertNotIn(token, serialized)

    def test_restart_recovers_with_the_policy_supplied_again(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-20", key="k20", value="kept")
        )
        self.assertEqual(status, 201)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

        policy = {token: frozenset(scopes) for token, scopes in POLICY.items()}
        restarted = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path, auth_policy=policy,
        )
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        try:
            port = restarted.server_address[1]

            # No header: still 401.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/states/k20")
            response = conn.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            conn.close()

            # Read scope recovers the persisted state.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "GET", "/v1/states/k20",
                headers={"Authorization": f"Bearer {READ_TOKEN}"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["value"], "kept")

            # Write scope can keep committing after recovery.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST", OP_PATH,
                body=json.dumps(operation_document(op_id="op-21", key="k21", value="v21")),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {WRITE_TOKEN}",
                },
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.close()
        finally:
            restarted.shutdown()
            restarted.server_close()
            thread.join(timeout=5)

        # Reopen the original server for any later tests in this class.
        type(self).server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler,
            data_file=self.data_path,
            auth_policy={token: frozenset(scopes) for token, scopes in POLICY.items()},
        )
        type(self).thread = threading.Thread(
            target=type(self).server.serve_forever, daemon=True
        )
        type(self).thread.start()
        type(self).port = type(self).server.server_address[1]


if __name__ == "__main__":
    unittest.main()
