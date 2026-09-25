"""Tests for runtime scope-policy hot reload.

``POST /v1/admin/scope-policy/reload`` is open only in scope-policy mode
and only to the admin scope. It rereads the exact file supplied at
startup, atomically swaps the whole token/scope mapping, and reports the
new file's SHA-256 digest and entry count. These tests cover the error
precedence chain (length before authentication; route shape before
query and body; query before body), the 404 mode gate, the 503/409
failure split, atomic serial commits, the authenticated-request
snapshot, and the no-side-effects guarantees.
"""

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from email.message import Message

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    SCOPE_ADMIN,
    RequestHandler,
    ScopePolicyConflict,
    ScopePolicyUnavailable,
    SemanticStateServer,
    load_scope_policy,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
NEW_ADMIN_TOKEN = "brand-new-admin"

POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
POLICY_B = {NEW_ADMIN_TOKEN: ["read", "write", "admin"]}

RELOAD_PATH = "/v1/admin/scope-policy/reload"
METRICS_PATH = "/v1/metrics"
OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def policy_bytes(document: dict) -> bytes:
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


class ReloadServerTestBase(unittest.TestCase):
    auth_token: str | None = None
    scope_mode: bool = True
    with_policy_file: bool = True
    use_data_file: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-reload-")
        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "wb") as handle:
            handle.write(policy_bytes(POLICY))
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        kwargs: dict = {}
        if cls.use_data_file:
            kwargs["data_file"] = cls.data_path
        if cls.auth_token is not None:
            kwargs["auth_token"] = cls.auth_token
        elif cls.scope_mode:
            kwargs["auth_scopes"] = dict(load_scope_policy(cls.policy_path))
            if cls.with_policy_file:
                kwargs["scope_policy_file"] = cls.policy_path
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def write_policy(self, content: bytes) -> None:
        with open(self.policy_path, "wb") as handle:
            handle.write(content)

    def write_policy_document(self, document: dict) -> None:
        self.write_policy(policy_bytes(document))

    def restore_policy(self) -> None:
        self.write_policy_document(POLICY)

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = ADMIN_TOKEN,
        raw_body: bytes | None = None,
    ) -> tuple[int, dict | None, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw_body is not None:
            conn.request(method, path, body=raw_body, headers=headers)
        elif body is None:
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
        conn.endheaders(body)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def reload(self, body: object = None, token: str | None = ADMIN_TOKEN, query: str = ""):
        path = RELOAD_PATH + query
        if body is None:
            return self.request("POST", path, body={}, token=token)
        return self.request("POST", path, body=body, token=token)

    def assert_unauthorized(self, status, payload, headers) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def assert_forbidden(self, status, payload, headers) -> None:
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)


class ScopeModeReloadTests(ReloadServerTestBase):
    def setUp(self) -> None:
        # The server is shared across the class; reset business state like
        # the existing scope-policy tests so identity/key reuse in one
        # test cannot answer 200/409 in another.
        self.server.store = type(self.server.store)()

    def tearDown(self) -> None:
        # Restore both the file and the live mapping; a missing file (the
        # 503 case) is recreated first.
        self.restore_policy()
        self.server.reload_scope_policy()

    # -- success contract --

    def test_reload_succeeds_with_exactly_three_fields(self) -> None:
        raw = policy_bytes(POLICY)
        self.write_policy(raw)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "policyDigest", "tokens"})
        self.assertEqual(payload["status"], "reloaded")
        self.assertEqual(payload["tokens"], 3)
        self.assertEqual(payload["policyDigest"], hashlib.sha256(raw).hexdigest())
        self.assertRegex(payload["policyDigest"], r"^[0-9a-f]{64}$")

    def test_digest_covers_raw_bytes_including_trailing_newline(self) -> None:
        raw = policy_bytes({READ_TOKEN: ["read"]}) + b"\n"
        self.write_policy(raw)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens"], 1)
        self.assertEqual(payload["policyDigest"], hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(
            payload["policyDigest"], hashlib.sha256(raw.rstrip()).hexdigest()
        )

    def test_empty_object_policy_reports_zero_tokens(self) -> None:
        self.write_policy(b"{}")
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {
            "status": "reloaded",
            "policyDigest": hashlib.sha256(b"{}").hexdigest(),
            "tokens": 0,
        })
        # Every previously configured token is now rejected.
        status, payload, headers = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        self.assert_unauthorized(status, payload, headers)

    def test_whitespace_padded_empty_body_object_is_accepted(self) -> None:
        status, payload, _ = self.request(
            "POST", RELOAD_PATH, raw_body=b"  {  }\n", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "reloaded")

    def test_bare_question_mark_query_is_empty(self) -> None:
        status, payload, _ = self.request(
            "POST", RELOAD_PATH + "?", raw_body=b"{}", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "reloaded")

    # -- the reload actually swaps the live mapping --

    def test_reload_replaces_whole_policy_atomically(self) -> None:
        self.write_policy_document(POLICY_B)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens"], 1)
        # Old tokens are gone; the new admin token works everywhere.
        self.assert_unauthorized(
            *self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)
        )
        self.assert_unauthorized(
            *self.request("GET", METRICS_PATH, token=READ_TOKEN)
        )
        status, _, _ = self.request("GET", METRICS_PATH, token=NEW_ADMIN_TOKEN)
        self.assertEqual(status, 200)

    def test_scope_change_takes_effect_after_reload(self) -> None:
        # The read-only token becomes write-only.
        self.write_policy_document({READ_TOKEN: ["write"], ADMIN_TOKEN: ["admin"]})
        self.assertEqual(self.reload()[0], 200)
        self.assert_forbidden(*self.request("GET", METRICS_PATH, token=READ_TOKEN))
        doc = {"operationId": "op-1", "key": "k", "value": "v", "clock": {"r1": 1}}
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", body=doc, token=READ_TOKEN
        )
        self.assertEqual(status, 201)

    # -- body must be exactly an empty object --

    def test_malformed_or_non_object_bodies_are_400(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{",
            b"junk",
            b"\xff\xfe",
            b"[]",
            b"null",
            b"42",
            b'"{}"',
            b"true",
            b'{"a":1}',
            b'{"x":null}',
            b'{"a":1,"a":2}',
        ):
            with self.subTest(raw=raw):
                status, payload, _ = self.request(
                    "POST", RELOAD_PATH, raw_body=raw, token=ADMIN_TOKEN
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    # -- query parameters are rejected and beat body validation --

    def test_any_query_parameter_is_400_even_with_valid_body(self) -> None:
        for query in ("?x=1", "?x=", "?x", "?=1", "?x=1&y=2", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload, _ = self.request(
                    "POST", RELOAD_PATH + query, raw_body=b"{}", token=ADMIN_TOKEN
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_validation_has_priority_over_body_validation(self) -> None:
        status, payload, _ = self.request(
            "POST", RELOAD_PATH + "?x=1", raw_body=b"not json", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- route shape is 404 before query and body checks --

    def test_shape_errors_are_404_even_with_bad_query_body_and_length(self) -> None:
        bad_shapes = (
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/reload/extra",
            "/v1/admin/scope-policy/reload/",
            "/v1/admin/scope-policy/RELOAD",
            "/v1/admin/scope-policy/reload;x",
        )
        for path in bad_shapes:
            with self.subTest(path=path):
                # A declared over-limit length and a bad query must not
                # surface: shape failure stays 404.
                status, payload, _ = self.put_raw(
                    "POST",
                    path + "?x=1",
                    [
                        ("Content-Length", OVER_LIMIT),
                        ("Authorization", f"Bearer {ADMIN_TOKEN}"),
                    ],
                    b"junk",
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_errors_are_404_without_content_length_for_admin(self) -> None:
        for path in (
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/reload/extra",
            "/v1/admin/scope-policy/reload/",
        ):
            with self.subTest(path=path):
                status, payload, _ = self.put_raw(
                    "POST",
                    path,
                    [("Authorization", f"Bearer {ADMIN_TOKEN}")],
                    b"{}",
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_reload_path_is_not_published(self) -> None:
        status, payload, _ = self.request("GET", RELOAD_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # -- Content-Length priority over authentication --

    def test_missing_or_malformed_length_is_400_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {ADMIN_TOKEN}", "Bearer unknown", f"Bearer {READ_TOKEN}"):
            with self.subTest(auth=auth):
                headers = []
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.put_raw("POST", RELOAD_PATH, headers, b"{}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for value in ("abc", "-1", "1.5", "1 2"):
            with self.subTest(value=value):
                status, payload, _ = self.put_raw(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", value),
                        ("Authorization", f"Bearer {ADMIN_TOKEN}"),
                    ],
                    b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_lengths_are_400(self) -> None:
        status, payload, _ = self.put_raw(
            "POST",
            RELOAD_PATH,
            [
                ("Content-Length", "2"),
                ("Content-Length", "3"),
                ("Authorization", f"Bearer {ADMIN_TOKEN}"),
            ],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {ADMIN_TOKEN}", "Bearer unknown", f"Bearer {READ_TOKEN}"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", OVER_LIMIT)]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.put_raw("POST", RELOAD_PATH, headers, b"junk")
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_length_check_beats_query_check(self) -> None:
        status, payload, _ = self.put_raw(
            "POST",
            RELOAD_PATH + "?x=1",
            [("Authorization", f"Bearer {ADMIN_TOKEN}")],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload, _ = self.put_raw(
            "POST",
            RELOAD_PATH + "?x=1",
            [("Content-Length", OVER_LIMIT), ("Authorization", f"Bearer {ADMIN_TOKEN}")],
            b"junk",
        )
        self.assertEqual(status, 413)

    # -- authentication and admin scope, without reading the body --

    def test_authentication_runs_after_length_but_before_body(self) -> None:
        # Missing credential: 401 even though the body would be 400.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", RELOAD_PATH)
        conn.putheader("Content-Length", "64")
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)

    def test_unknown_token_is_401_without_body_read(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", RELOAD_PATH)
        conn.putheader("Content-Length", "64")
        conn.putheader("Authorization", "Bearer stranger")
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        self.assert_unauthorized(response.status, payload, headers)

    def test_read_and_write_scopes_are_403_without_body_read(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.putrequest("POST", RELOAD_PATH)
                conn.putheader("Content-Length", "64")
                conn.putheader("Authorization", f"Bearer {token}")
                conn.endheaders()
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.getheaders())
                conn.close()
                self.assert_forbidden(response.status, payload, headers)

    def test_admin_scope_is_required_even_for_a_400_query(self) -> None:
        # Without admin the scope gate fires first; the bad query never
        # surfaces as 400.
        status, payload, headers = self.request(
            "POST", RELOAD_PATH + "?x=1", body={}, token=READ_TOKEN
        )
        self.assert_forbidden(status, payload, headers)

    def test_admin_scope_reaches_query_validation(self) -> None:
        status, payload, _ = self.request(
            "POST", RELOAD_PATH + "?x=1", body={}, token=ADMIN_TOKEN
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- 503 policy_unavailable --

    def test_missing_file_is_503_and_old_policy_survives(self) -> None:
        os.unlink(self.policy_path)
        try:
            status, payload, _ = self.reload()
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "policy_unavailable"})
        finally:
            self.restore_policy()
        # Old mapping kept serving throughout and still does.
        status, _, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
        self.assertEqual(status, 200)
        status, payload, _ = self.request("GET", METRICS_PATH, token="stranger")
        self.assert_unauthorized(status, payload, _)

    def test_directory_in_place_of_file_is_503(self) -> None:
        os.unlink(self.policy_path)
        os.mkdir(self.policy_path)
        try:
            status, payload, _ = self.reload()
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "policy_unavailable"})
        finally:
            os.rmdir(self.policy_path)
            self.restore_policy()
        self.assertEqual(self.reload()[0], 200)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses file modes")
    def test_unreadable_file_is_503(self) -> None:
        os.chmod(self.policy_path, 0o000)
        try:
            status, payload, _ = self.reload()
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "policy_unavailable"})
        finally:
            os.chmod(self.policy_path, 0o644)

    # -- 409 policy_conflict --

    def test_illegal_content_is_409_and_old_policy_survives(self) -> None:
        old_digest = hashlib.sha256(policy_bytes(POLICY)).hexdigest()
        bad_contents = (
            b"",
            b"{",
            b"not json",
            b"\xff",
            b'{"t":["read"]',
            b'["read"]',
            b"42",
            b"null",
            b'{"t":["read"],"t":["write"]}',
            b'{"":["read"]}',
            b'{"two tokens":["read"]}',
            b'{"t":[]}',
            b'{"t":"read"}',
            b'{"t":["administrate"]}',
            b'{"t":["read","read"]}',
        )
        for content in bad_contents:
            with self.subTest(content=content):
                self.write_policy(content)
                status, payload, _ = self.reload()
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "policy_conflict"})
                # The old permission mapping is still fully in force.
                status, _, _ = self.request("GET", METRICS_PATH, token=READ_TOKEN)
                self.assertEqual(status, 200)
                status, _, _ = self.request(
                    "POST",
                    "/v1/replicas/r1/operations",
                    body={"operationId": "op", "key": "k", "value": "v", "clock": {"r1": 1}},
                    token=WRITE_TOKEN,
                )
                self.assertIn(status, (200, 201))
        # The file was never overwritten: once valid content returns, the
        # reload commits it.
        self.restore_policy()
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], old_digest)

    def test_503_then_409_then_success_leave_consistent_policy(self) -> None:
        digest_b = hashlib.sha256(policy_bytes(POLICY_B)).hexdigest()
        self.write_policy(b'{"t":["nope"]}')
        self.assertEqual(self.reload()[0], 409)
        os.unlink(self.policy_path)
        self.assertEqual(self.reload()[0], 503)
        self.write_policy_document(POLICY_B)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], digest_b)
        status, _, _ = self.request("GET", METRICS_PATH, token=NEW_ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assert_unauthorized(*self.request("GET", METRICS_PATH, token=ADMIN_TOKEN))
        self.restore_policy()
        self.reload()

    # -- no side effects --

    def test_rejected_requests_change_no_business_state(self) -> None:
        doc = {"operationId": "op-1", "key": "color", "value": "blue", "clock": {"r1": 1}}
        self.assertEqual(
            self.request("POST", "/v1/replicas/r1/operations", body=doc, token=WRITE_TOKEN)[0],
            201,
        )
        _, metrics_before, _ = self.request("GET", "/v1/metrics", token=READ_TOKEN)
        _, digest_before, _ = self.request("GET", "/v1/verification/digest", token=ADMIN_TOKEN)

        # Authentication/permission rejections and malformed requests.
        self.request("POST", RELOAD_PATH, body={"x": 1}, token=None)
        self.request("POST", RELOAD_PATH, body={"x": 1}, token="stranger")
        self.request("POST", RELOAD_PATH, body={"x": 1}, token=READ_TOKEN)
        self.request("POST", RELOAD_PATH, raw_body=b"bad", token=ADMIN_TOKEN)
        self.request("POST", RELOAD_PATH + "?x=1", raw_body=b"{}", token=ADMIN_TOKEN)
        # Failed reloads: unreadable-content conflict then missing file.
        self.write_policy(b'{"t":["bogus"]}')
        self.assertEqual(self.reload()[0], 409)
        os.unlink(self.policy_path)
        self.assertEqual(self.reload()[0], 503)
        self.restore_policy()

        _, metrics_after, _ = self.request("GET", "/v1/metrics", token=READ_TOKEN)
        _, digest_after, _ = self.request("GET", "/v1/verification/digest", token=ADMIN_TOKEN)
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(digest_after, digest_before)

    def test_no_temp_files_are_created_anywhere(self) -> None:
        self.write_policy(b'{"t":["nope"]}')
        listing = sorted(os.listdir(self.tmpdir))
        for _ in range(3):
            self.reload()
        self.write_policy_document(POLICY_B)
        self.reload()
        self.restore_policy()
        self.reload()
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing)

    # -- serial whole commits under concurrency --

    def test_concurrent_reloads_commit_whole_serialized_policies(self) -> None:
        raw_a = policy_bytes(POLICY)
        raw_b = policy_bytes(POLICY_B)
        digest_a, digest_b = hashlib.sha256(raw_a).hexdigest(), hashlib.sha256(raw_b).hexdigest()
        committed = {digest_a: 3, digest_b: 1}
        stop = threading.Event()
        results: list[tuple[int, str | None, int | None]] = []
        results_lock = threading.Lock()

        def writer(index: int) -> None:
            current = raw_a
            swap = os.path.join(self.tmpdir, f"swap-{index}.tmp")
            while not stop.is_set():
                current = raw_b if current == raw_a else raw_a
                # Atomic inode replacement: a reader either sees the full
                # old file or the full new file.
                with open(swap, "wb") as handle:
                    handle.write(current)
                os.replace(swap, self.policy_path)

        # Drive reloads directly on the server (HTTP auth would itself
        # flip with the policy) and assert every response describes one
        # of the two complete committed policies — never a mix.
        errors: list[BaseException] = []

        def reload_worker() -> None:
            try:
                while not stop.is_set():
                    payload = self.server.reload_scope_policy()
                    with results_lock:
                        results.append(
                            (200, payload["policyDigest"], payload["tokens"])
                        )
            except BaseException as exc:  # noqa: BLE001 - recorded for the test
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(index,), daemon=True) for index in range(2)
        ]
        threads += [threading.Thread(target=reload_worker, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
        stop.wait(0.4)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertGreater(len(results), 4)
        for status_code, digest, tokens in results:
            self.assertEqual(status_code, 200)
            self.assertIn(digest, committed)
            self.assertEqual(tokens, committed[digest])
        self.restore_policy()
        self.reload()

    def test_reload_lock_serializes_file_reads(self) -> None:
        import semantic_state_engine.server as server_module

        original = server_module.read_scope_policy_file
        in_progress = {"value": 0}
        guard = threading.Lock()

        def tracked_read(path):
            with guard:
                in_progress["value"] += 1
            try:
                # Hold the read briefly; the watcher must never observe
                # two reads in flight at once.
                stop = threading.Event()
                stop.wait(0.02)
                return original(path)
            finally:
                with guard:
                    in_progress["value"] -= 1

        observed_peaks: list[int] = []

        def watcher() -> None:
            for _ in range(40):
                with guard:
                    observed_peaks.append(in_progress["value"])
                stop = threading.Event()
                stop.wait(0.005)

        server_module.read_scope_policy_file = tracked_read
        try:
            watch = threading.Thread(target=watcher, daemon=True)
            watch.start()
            workers = [
                threading.Thread(target=self.server.reload_scope_policy, daemon=True)
                for _ in range(4)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)
            watch.join(timeout=5)
        finally:
            server_module.read_scope_policy_file = original
        self.assertLessEqual(max(observed_peaks), 1)

    # -- authenticated requests keep their auth-time scopes --

    def _authenticate_directly(self, token: str):
        headers = Message()
        headers["Authorization"] = f"Bearer {token}"
        handler = RequestHandler.__new__(RequestHandler)
        handler.headers = headers
        handler.server = self.server
        handler.close_connection = False
        rejected: dict[str, int] = {}
        handler._json = lambda status, payload, extra=None: rejected.setdefault(
            "status", int(status)
        )
        return (*handler._authenticate(), rejected.get("status"))

    def test_authenticated_request_keeps_old_scopes_after_reload(self) -> None:
        ok, old_scopes, rejected = self._authenticate_directly(ADMIN_TOKEN)
        self.assertTrue(ok)
        self.assertIsNotNone(old_scopes)
        self.assertIn(SCOPE_ADMIN, old_scopes)
        self.assertIsNone(rejected)

        # Swap the admin token out from under the already-authenticated
        # request: its captured scope set is unchanged.
        self.write_policy_document(POLICY_B)
        self.assertEqual(self.reload()[0], 200)
        self.assertIn(SCOPE_ADMIN, old_scopes)

        # The old token is rejected by the fresh mapping...
        ok, scopes, rejected = self._authenticate_directly(ADMIN_TOKEN)
        self.assertFalse(ok)
        self.assertIsNone(scopes)
        self.assertEqual(rejected, 401)
        # ...and the new admin token authenticates with it.
        ok, scopes, rejected = self._authenticate_directly(NEW_ADMIN_TOKEN)
        self.assertTrue(ok)
        self.assertIn(SCOPE_ADMIN, scopes)
        self.assertIsNone(rejected)

    # -- anonymous health stays untouched --

    def test_health_remains_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


class OtherModesReloadTests(ReloadServerTestBase):
    """The route is not published without a configured scope policy."""

    scope_mode = False

    def test_anonymous_mode_returns_404(self) -> None:
        status, payload, _ = self.request(
            "POST", RELOAD_PATH, raw_body=b"{}", token=None
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_404_comes_before_length_and_query_checks(self) -> None:
        # No Content-Length at all: still 404, never 400.
        status, payload, _ = self.put_raw(
            "POST", RELOAD_PATH + "?x=1", [], b"{}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # An over-limit declaration is still 404, never 413.
        status, payload, _ = self.put_raw(
            "POST", RELOAD_PATH, [("Content-Length", OVER_LIMIT)], b"junk"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class LegacyTokenModeReloadTests(ReloadServerTestBase):
    auth_token = "legacy-token"

    def test_legacy_token_mode_returns_404_even_with_the_token(self) -> None:
        status, payload, _ = self.put_raw(
            "POST",
            RELOAD_PATH,
            [
                ("Content-Length", "2"),
                ("Authorization", "Bearer legacy-token"),
            ],
            b"{}",
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_legacy_mode_404_beats_missing_length(self) -> None:
        status, payload, _ = self.put_raw(
            "POST",
            RELOAD_PATH + "?x=1",
            [("Authorization", "Bearer legacy-token")],
            b"{}",
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ScopesWithoutReloadPathTests(ReloadServerTestBase):
    """auth_scopes set directly but no policy file configured: route closed."""

    with_policy_file = False

    def test_route_is_closed_without_configured_file(self) -> None:
        status, payload, _ = self.request("POST", RELOAD_PATH, raw_body=b"{}")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ReloadPersistenceTests(ReloadServerTestBase):
    use_data_file = True

    def tearDown(self) -> None:
        self.restore_policy()

    def test_reload_touches_no_file_but_the_policy_file(self) -> None:
        doc = {"operationId": "op-1", "key": "color", "value": "blue", "clock": {"r1": 1}}
        self.assertEqual(
            self.request("POST", "/v1/replicas/r1/operations", body=doc, token=WRITE_TOKEN)[0],
            201,
        )
        with open(self.data_path, "rb") as handle:
            data_before = handle.read()

        def policy_snapshot():
            stat_result = os.stat(self.policy_path)
            with open(self.policy_path, "rb") as handle:
                return stat_result.st_mtime_ns, handle.read()

        # Success and both kinds of failure. Authentication always runs
        # against the currently-live policy, and only a successful reload
        # swaps it: the first reload is authorized by the old admin token
        # and commits B; later ones are authorized by B's admin token.
        self.write_policy_document(POLICY_B)
        snapshot = policy_snapshot()
        self.assertEqual(self.reload(token=ADMIN_TOKEN)[0], 200)
        # The service only read the file: bytes and mtime are unchanged.
        self.assertEqual(policy_snapshot(), snapshot)

        self.write_policy(b"not a policy")
        snapshot = policy_snapshot()
        self.assertEqual(self.reload(token=NEW_ADMIN_TOKEN)[0], 409)
        self.assertEqual(policy_snapshot(), snapshot)

        os.unlink(self.policy_path)
        self.assertEqual(self.reload(token=NEW_ADMIN_TOKEN)[0], 503)
        # The failed reload never recreates the file itself.
        self.assertFalse(os.path.exists(self.policy_path))

        self.restore_policy()
        snapshot = policy_snapshot()
        self.assertEqual(self.reload(token=NEW_ADMIN_TOKEN)[0], 200)
        self.assertEqual(policy_snapshot(), snapshot)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), data_before)

    def test_restart_recovers_data_and_rereads_same_configured_path(self) -> None:
        self.write_policy_document(POLICY_B)
        self.assertEqual(self.reload()[0], 200)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

        # Restart still points at the same configured path, whose content
        # is now policy B; the mapping is recovered from the file, never
        # from the data file.
        restarted = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=self.data_path,
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        try:
            port = restarted.server_address[1]

            def call(method, path, token):
                headers = {"Authorization": f"Bearer {token}"}
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(method, path, headers=headers)
                response = conn.getresponse()
                response.read()
                conn.close()
                return response.status

            self.assertEqual(call("GET", METRICS_PATH, ADMIN_TOKEN), 401)
            self.assertEqual(call("GET", METRICS_PATH, NEW_ADMIN_TOKEN), 200)
            # Business data recovered independently of the policy.
            self.assertEqual(call("GET", "/v1/states/color", NEW_ADMIN_TOKEN), 200)
        finally:
            restarted.shutdown()
            restarted.server_close()
            thread.join(timeout=5)

        # Bring the class server back for tearDownClass.
        type(self).server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=self.data_path,
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        type(self).thread = threading.Thread(
            target=type(self).server.serve_forever, daemon=True
        )
        type(self).thread.start()
        type(self).port = type(self).server.server_address[1]
        self.restore_policy()
        type(self).server.reload_scope_policy()


class LoaderClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-reload-load-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "scopes.json")

    def load(self, content: bytes | None = None, directory: bool = False):
        import semantic_state_engine.server as server_module

        if directory:
            os.mkdir(self.path)
        elif content is not None:
            with open(self.path, "wb") as handle:
                handle.write(content)
        return server_module.read_scope_policy_file(self.path)

    def test_missing_and_directory_are_unavailable(self) -> None:
        with self.assertRaises(ScopePolicyUnavailable):
            self.load()
        with self.assertRaises(ScopePolicyUnavailable):
            self.load(directory=True)

    def test_readable_but_invalid_content_is_conflict(self) -> None:
        for content in (b"", b"{", b"[]", b'{"t":["nope"]}', b"\xff"):
            with self.subTest(content=content):
                with self.assertRaises(ScopePolicyConflict):
                    self.load(content)

    def test_success_returns_raw_bytes_and_mapping(self) -> None:
        raw = policy_bytes({"t": ["read", "admin"]})
        returned_raw, policy = self.load(raw)
        self.assertEqual(returned_raw, raw)
        self.assertEqual(policy, {"t": frozenset({"read", "admin"})})

    def test_both_classifications_are_scope_policy_errors(self) -> None:
        self.assertTrue(issubclass(ScopePolicyUnavailable, Exception))
        with self.assertRaises(Exception):
            self.load()
        with self.assertRaises(Exception):
            self.load(b"bad")


if __name__ == "__main__":
    unittest.main()
