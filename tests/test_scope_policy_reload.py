"""Tests for runtime scope-policy hot reload.

``POST /v1/admin/scope-policy/reload`` atomically replaces the live
token-to-scopes policy by re-reading the file supplied at startup, without a
restart. The tests cover the full request precedence chain (path shape,
declared length, authentication, admin scope, mode gate, query, body), the
503/409 failure split with the old policy kept whole, the SHA-256 digest
contract, atomic/serialized reloads, snapshot-scoped authentication, restart
recovery, and the absence of any business-state or temporary-file effects.
"""

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    ScopePolicyManager,
    ScopePolicyReloadError,
    SemanticStateServer,
    load_scope_policy,
    parse_empty_object_payload,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
RELOAD_PATH = "/v1/admin/scope-policy/reload"
METRICS_PATH = "/v1/metrics"
OP_PATH = "/v1/replicas/r1/operations"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class ParseEmptyObjectPayloadTests(unittest.TestCase):
    def test_empty_object_in_its_shapes_is_accepted(self) -> None:
        for raw in (b"{}", "{}", {}, b"{ }", b"{\n\t}"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_empty_object_payload(raw))

    def test_anything_else_is_rejected(self) -> None:
        for raw in (
            b"",
            b"junk",
            b"{",
            b'{"a":1}',
            b'{"x":{}}',
            b'{"status":"reloaded"}',
            b"[]",
            b"null",
            b'"{}"',
            b"0",
            b"true",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_empty_object_payload(raw)
        for value in ({"a": 1}, [], None, 0, True, ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_empty_object_payload(value)


class ScopePolicyManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-reload-unit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "scopes.json")
        with open(self.path, "wb") as handle:
            handle.write(INITIAL_BYTES)
        self.manager = ScopePolicyManager(self.path, load_scope_policy(self.path))

    def write(self, content: bytes) -> str:
        with open(self.path, "wb") as handle:
            handle.write(content)
        return self.path

    def test_reload_returns_raw_byte_digest_and_token_count(self) -> None:
        digest, tokens = self.manager.reload()
        self.assertEqual(tokens, 3)
        self.assertEqual(digest, hashlib.sha256(INITIAL_BYTES).hexdigest())
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_digest_covers_the_raw_utf8_bytes_not_canonicalized_content(self) -> None:
        # Semantically identical files with different raw bytes must hash
        # differently while reporting the same entry count.
        spaced = b'{ "reader-token": ["read"] }'
        self.write(spaced)
        digest, tokens = self.manager.reload()
        self.assertEqual(digest, hashlib.sha256(spaced).hexdigest())
        self.assertEqual(tokens, 1)
        self.write(INITIAL_BYTES)
        digest, tokens = self.manager.reload()
        self.assertEqual(digest, hashlib.sha256(INITIAL_BYTES).hexdigest())
        self.assertEqual(tokens, 3)

    def test_empty_object_policy_reloads_with_zero_tokens(self) -> None:
        self.write(b"{}")
        digest, tokens = self.manager.reload()
        self.assertEqual(tokens, 0)
        self.assertEqual(digest, hashlib.sha256(b"{}").hexdigest())
        self.assertEqual(self.manager.snapshot(), {})

    def test_reload_swaps_the_live_mapping(self) -> None:
        self.write(json.dumps({"new-admin": ["admin"]}).encode("utf-8"))
        _, tokens = self.manager.reload()
        self.assertEqual(tokens, 1)
        self.assertEqual(set(self.manager.snapshot()), {"new-admin"})
        self.assertEqual(self.manager.snapshot()["new-admin"], frozenset({"admin"}))

    def test_missing_file_is_unavailable_and_keeps_the_old_policy(self) -> None:
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))

    def test_directory_is_unavailable_and_keeps_the_old_policy(self) -> None:
        os.unlink(self.path)
        os.mkdir(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))

    def test_unreadable_file_is_unavailable(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses file permission bits")
        os.chmod(self.path, 0o000)
        self.addCleanup(os.chmod, self.path, 0o600)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload()
        self.assertEqual(caught.exception.kind, "unavailable")
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))

    def test_invalid_readable_content_is_conflict_and_keeps_the_old_policy(self) -> None:
        for content in (
            b"",
            b"not json",
            b'{"t":["read"]',
            b'["read"]',
            b'{"t":["bogus"]}',
            b'{"t":[]}',
            b'{"t":["read","read"]}',
            b'{"t":["read"],"t":["write"]}',
            b'{"tok\xffen":["read"]}',
        ):
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(ScopePolicyReloadError) as caught:
                    self.manager.reload()
                self.assertEqual(caught.exception.kind, "conflict")
                self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))

    def test_successful_reload_recovers_after_a_conflict(self) -> None:
        self.write(b'{"t":["nope"]}')
        with self.assertRaises(ScopePolicyReloadError):
            self.manager.reload()
        self.write(INITIAL_BYTES)
        digest, tokens = self.manager.reload()
        self.assertEqual(tokens, 3)
        self.assertEqual(digest, hashlib.sha256(INITIAL_BYTES).hexdigest())

    def test_authentication_snapshot_survives_later_reloads(self) -> None:
        # A snapshot taken at authentication time is its own mapping: a later
        # reload never changes what the already-authorized request executes
        # under.
        at_authentication = self.manager.snapshot()
        self.write(json.dumps({"completely-different-token": ["admin"]}).encode("utf-8"))
        self.manager.reload()
        self.assertEqual(set(at_authentication), set(INITIAL_POLICY))
        self.assertEqual(
            at_authentication[ADMIN_TOKEN], frozenset({"read", "write", "admin"})
        )

    def test_path_is_resolved_and_fixed_at_construction(self) -> None:
        relative = ScopePolicyManager(
            os.path.relpath(self.path), load_scope_policy(self.path)
        )
        self.assertEqual(relative.path, os.path.abspath(self.path))

    def test_manager_without_a_path_reports_unavailable(self) -> None:
        with self.assertRaises(ScopePolicyReloadError) as caught:
            ScopePolicyManager(None, {}).reload()
        self.assertEqual(caught.exception.kind, "unavailable")


class ScopePolicyReloadHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-reload-http-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "wb") as handle:
            handle.write(INITIAL_BYTES)
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=cls.data_path,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
            scope_policy_file=cls.policy_path,
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
        # Every test starts from the initial policy both on disk and in the
        # live manager, regardless of what an earlier test left behind.
        self.write_policy(INITIAL_BYTES)
        self.server.scope_policy.reload()
        # Cleanups run last-added first: reload the live manager only after
        # the restored file is back on disk.
        self.addCleanup(self.server.scope_policy.reload)
        self.addCleanup(self.write_policy, INITIAL_BYTES)

    def write_policy(self, content: bytes) -> None:
        # A previous test may have turned the path into a directory; clear
        # whatever occupies it so each call starts from a regular file.
        if os.path.isdir(self.policy_path):
            shutil.rmtree(self.policy_path)
        with open(self.policy_path, "wb") as handle:
            handle.write(content)

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = ADMIN_TOKEN,
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

    def reload(self, body: object = {}, token: str | None = ADMIN_TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    # -- success contract --

    def test_reload_returns_exactly_the_three_contracted_fields(self) -> None:
        status, payload, raw = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "policyDigest", "tokens"})
        self.assertEqual(payload["status"], "reloaded")
        self.assertEqual(payload["tokens"], 3)
        digest = payload["policyDigest"]
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, hashlib.sha256(INITIAL_BYTES).hexdigest())
        # Re-reloading an unchanged file returns the same digest.
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], digest)
        # Field order is fixed and there is no trailing line terminator.
        self.assertEqual(
            raw.decode("utf-8"),
            f'{{"status":"reloaded","policyDigest":"{digest}","tokens":3}}',
        )

    def test_digest_follows_the_file_bytes_after_a_change(self) -> None:
        content = b'{ "reader-token" : [ "read" ] }'
        self.write_policy(content)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], hashlib.sha256(content).hexdigest())
        self.assertEqual(payload["tokens"], 1)

    def test_empty_object_policy_reports_zero_tokens(self) -> None:
        self.write_policy(b"{}")
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "reloaded")
        self.assertEqual(payload["tokens"], 0)
        # With no tokens configured, every credential is now rejected.
        status, _, _ = self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 401)

    def test_reload_atomically_replaces_the_whole_boundary(self) -> None:
        new_policy = {
            "new-admin": ["read", "write", "admin"],
            READ_TOKEN: ["read", "write", "admin"],
        }
        self.write_policy(json.dumps(new_policy).encode("utf-8"))
        status, _, _ = self.reload()
        self.assertEqual(status, 200)
        # A token absent from the new mapping is rejected even though it was
        # an admin under the old one.
        status, _, _ = self.request("GET", METRICS_PATH, token=WRITE_TOKEN)
        self.assertEqual(status, 401)
        # The reader token was promoted and now writes.
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-reload-swap", key="swap-color", value="blue"),
            token=READ_TOKEN,
        )
        self.assertEqual(status, 201)
        # The brand-new admin works end to end.
        status, state, _ = self.request(
            "GET", "/v1/states/swap-color", token="new-admin"
        )
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")

    def test_failed_reload_keeps_the_old_boundary_fully_in_force(self) -> None:
        self.write_policy(b'{"new-admin":["admin"]')  # truncated JSON
        status, payload, _ = self.reload()
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "policy_conflict"})
        # Every old boundary is intact: reader reads, writer writes,
        # stranger is rejected, and the file-only new token does not exist.
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(
            self.request(
                "POST", OP_PATH, operation_document(op_id="op-old-boundary"), token=WRITE_TOKEN
            )[0],
            201,
        )
        self.assertEqual(self.request("GET", METRICS_PATH, token="new-admin")[0], 401)

    # -- body validation --

    def test_body_must_be_exactly_the_empty_object(self) -> None:
        for body in (
            None,
            {"a": 1},
            {"x": {}},
            {"status": "reloaded"},
            {"policyDigest": "x"},
            [],
            "{}",
            0,
            True,
        ):
            with self.subTest(body=body):
                status, payload, _ = self.reload(body=body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_json_body_is_400(self) -> None:
        for raw in (b"", b"junk", b"{", b"[]", b"null"):
            with self.subTest(raw=raw):
                status, payload, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", str(len(raw))),
                        ("Authorization", f"Bearer {ADMIN_TOKEN}"),
                    ],
                    raw,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_object_with_whitespace_is_accepted(self) -> None:
        for raw in (b"{}", b"{ }", b"{\n\t}"):
            with self.subTest(raw=raw):
                status, _, _ = self.raw_request(
                    "POST",
                    RELOAD_PATH,
                    [
                        ("Content-Length", str(len(raw))),
                        ("Authorization", f"Bearer {ADMIN_TOKEN}"),
                    ],
                    raw,
                )
                self.assertEqual(status, 200)

    # -- query parameters are rejected before the body --

    def test_any_query_parameter_is_400_even_with_a_valid_body(self) -> None:
        for query in ("?x", "?x=", "?x=1", "?x=1&y=2", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload, _ = self.request(
                    "POST", RELOAD_PATH + query, body={}, token=ADMIN_TOKEN
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_check_precedes_the_body_check(self) -> None:
        # Malformed query and a malformed body together still report the
        # query rejection, and the file is never reloaded.
        status, _, _ = self.raw_request(
            "POST",
            RELOAD_PATH + "?x=1",
            [
                ("Content-Length", "4"),
                ("Authorization", f"Bearer {ADMIN_TOKEN}"),
            ],
            b"junk",
        )
        self.assertEqual(status, 400)
        self.assertEqual(set(self.server.scope_policy.snapshot()), set(INITIAL_POLICY))

    # -- Content-Length keeps priority over authentication --

    def test_missing_content_length_is_400_for_any_credential(self) -> None:
        for auth in (
            None,
            f"Bearer {ADMIN_TOKEN}",
            f"Bearer {READ_TOKEN}",
            "Bearer unknown",
        ):
            with self.subTest(auth=auth):
                headers = []
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_for_any_credential(self) -> None:
        for auth in (None, f"Bearer {ADMIN_TOKEN}", f"Bearer {READ_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", OVER_LIMIT)]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"junk"
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    # -- authentication and scope failures never read the body --

    def test_401_and_403_arrive_without_waiting_for_the_declared_body(self) -> None:
        for auth, expected_status in (
            (None, 401),
            ("Bearer unknown", 401),
            (f"Bearer {READ_TOKEN}", 403),
            (f"Bearer {WRITE_TOKEN}", 403),
        ):
            with self.subTest(auth=auth):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.putrequest("POST", RELOAD_PATH)
                conn.putheader("Content-Length", "64")
                if auth is not None:
                    conn.putheader("Authorization", auth)
                conn.endheaders()  # declare 64 bytes but never send them
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                self.assertEqual(response.status, expected_status)
                self.assertEqual(
                    payload,
                    {"error": "unauthorized" if expected_status == 401 else "forbidden"},
                )

    # -- the endpoint exists only in scope-policy mode --

    def test_endpoint_is_404_in_single_token_mode_but_401_without_a_token(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="legacy-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            def call(token: str | None, body: bool = True) -> int:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                headers = {"Content-Type": "application/json"}
                if token is not None:
                    headers["Authorization"] = f"Bearer {token}"
                if body:
                    conn.request("POST", RELOAD_PATH, body="{}", headers=headers)
                else:
                    conn.request("POST", RELOAD_PATH, headers=headers)
                response = conn.getresponse()
                response.read()
                conn.close()
                return response.status

            # Authentication runs before the mode gate, so a missing or bad
            # credential is still 401 rather than 404.
            self.assertEqual(call(None), 401)
            self.assertEqual(call("wrong-token"), 401)
            # The valid legacy token authorizes everything, but the route is
            # not published outside scope-policy mode.
            self.assertEqual(call("legacy-token"), 404)
            # A length error still precedes authentication on this shape.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.putrequest("POST", RELOAD_PATH)
            conn.putheader("Content-Length", OVER_LIMIT)
            conn.endheaders(b"junk")
            response = conn.getresponse()
            response.read()
            conn.close()
            self.assertEqual(response.status, 413)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_endpoint_is_404_in_anonymous_mode(self) -> None:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", RELOAD_PATH, body="{}", headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # -- path shape precedes query and body checks --

    def test_path_shape_mismatches_are_404_even_with_bad_query_and_body(self) -> None:
        for path in (
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/reload/",
            "/v1/admin/scope-policy/reload/extra",
            "/v1/admin/scopepolicy/reload",
            "/v1/admin/scope-policy/other",
            "/admin/scope-policy/reload",
        ):
            with self.subTest(path=path):
                # Admin passes the method-wide gate; the wrong shape must
                # still be a not-found rather than the route's 400 for the
                # bogus query and body, and no length is declared at all.
                status, payload, _ = self.raw_request(
                    "POST",
                    path + "?bogus=1",
                    [("Authorization", f"Bearer {ADMIN_TOKEN}")],
                    b"not even json",
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_other_methods_on_the_path_are_not_published(self) -> None:
        status, _, _ = self.request("GET", RELOAD_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status, 404)

    # -- 503 policy_unavailable --

    def test_missing_file_is_503_and_leaves_the_old_policy_in_force(self) -> None:
        os.unlink(self.policy_path)
        status, payload, _ = self.reload()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(
            self.request(
                "POST", OP_PATH, operation_document(op_id="op-503-write"), token=WRITE_TOKEN
            )[0],
            201,
        )

    def test_directory_target_is_503(self) -> None:
        os.unlink(self.policy_path)
        os.mkdir(self.policy_path)
        status, payload, _ = self.reload()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})

    def test_recovering_the_file_makes_the_next_reload_succeed(self) -> None:
        os.unlink(self.policy_path)
        self.assertEqual(self.reload()[0], 503)
        self.write_policy(INITIAL_BYTES)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens"], 3)

    # -- 409 policy_conflict --

    def test_readable_but_invalid_content_is_409(self) -> None:
        for content in (
            b"",
            b"junk",
            b'{"t":["read"]',
            b'["read"]',
            b'"read"',
            b'{"t":["read","read"]}',
            b'{"t":["administer"]}',
            b'{"t":[]}',
            b'{"":["read"]}',
            b'{"t ok":["read"]}',
            b'{"t":[\xff]}',
        ):
            with self.subTest(content=content):
                self.write_policy(content)
                status, payload, _ = self.reload()
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "policy_conflict"})

    # -- no side effects from failures --

    def test_failures_create_no_files_and_leave_the_data_file_untouched(self) -> None:
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-no-side-effects", key="color", value="blue"),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 201)
        with open(self.data_path, "rb") as handle:
            data_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))

        self.write_policy(b'{"t":["nope"]}')
        self.assertEqual(self.reload()[0], 409)
        os.unlink(self.policy_path)
        self.assertEqual(self.reload()[0], 503)
        # Auth and scope rejections touch neither business state nor disk.
        # A declared (but unsent) length keeps the 401 ahead of the missing
        # file: authentication never reaches the reload.
        status, _, _ = self.raw_request(
            "POST",
            RELOAD_PATH,
            [("Content-Length", "2")],
            None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(self.reload(token=READ_TOKEN)[0], 403)
        # Restore the file so the directory listing comparison sees the same
        # two files the test started with.
        self.write_policy(INITIAL_BYTES)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), data_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)

    # -- concurrency: reloads commit serially and requests see one revision --

    def test_concurrent_reloads_commit_as_complete_serialized_swaps(self) -> None:
        concurrent_policy = dict(INITIAL_POLICY)
        concurrent_policy.update({f"extra-token-{i}": ["read"] for i in range(8)})
        content = json.dumps(concurrent_policy).encode("utf-8")
        self.write_policy(content)

        reload_errors: list[Exception] = []
        results: list[dict] = []
        results_lock = threading.Lock()
        start = threading.Barrier(16)

        def do_reload() -> None:
            try:
                start.wait()
                # Retry only transport-level teardown races of rapid loopback
                # short connections (TCP RST); the status contract itself is
                # asserted strictly once a response arrives.
                for attempt in range(10):
                    try:
                        status, payload, _ = self.reload()
                        break
                    except ConnectionError:
                        if attempt == 9:
                            raise
                else:  # pragma: no cover - loop always breaks or raises
                    return
                if status != 200:
                    reload_errors.append(AssertionError(f"reload status {status}"))
                    return
                with results_lock:
                    results.append(payload)
            except Exception as exc:  # pragma: no cover - surfaced below
                reload_errors.append(exc)

        def do_authenticate() -> None:
            try:
                start.wait()
                for _ in range(50):
                    # The reader exists in both revisions with read scope;
                    # every observation must be a clean 200, never a mixed or
                    # half-applied policy surfacing as 401/403/500.
                    for attempt in range(10):
                        try:
                            status, _, _ = self.request(
                                "GET", METRICS_PATH, token=READ_TOKEN
                            )
                            break
                        except ConnectionError:
                            if attempt == 9:
                                raise
                    else:  # pragma: no cover - loop always breaks or raises
                        return
                    if status != 200:
                        reload_errors.append(AssertionError(f"metrics status {status}"))
            except Exception as exc:  # pragma: no cover - surfaced below
                reload_errors.append(exc)

        threads = [threading.Thread(target=do_reload) for _ in range(8)]
        threads += [threading.Thread(target=do_authenticate) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(reload_errors, reload_errors)
        self.assertEqual(len(results), 8)
        expected_digest = hashlib.sha256(content).hexdigest()
        self.assertTrue(all(r["status"] == "reloaded" for r in results))
        self.assertTrue(all(r["policyDigest"] == expected_digest for r in results))
        self.assertTrue(all(r["tokens"] == 11 for r in results))
        live = self.server.scope_policy.snapshot()
        self.assertEqual(set(live), set(concurrent_policy))

    # -- restart still recovers the policy from the configured file --

    def test_restart_reloads_policy_from_the_configured_file(self) -> None:
        restart_policy = {
            "restart-admin": ["read", "write", "admin"],
            READ_TOKEN: ["read"],
        }
        self.write_policy(json.dumps(restart_policy).encode("utf-8"))
        self.assertEqual(self.reload()[0], 200)
        # Business state committed before the restart survives as well. The
        # new policy has no ADMIN_TOKEN, so the write uses its admin. A
        # unique key and clock keep this observation independent of the
        # class-shared store.
        status, _, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(op_id="op-9", key="restart-color", value="kept", ),
            token="restart-admin",
        )
        self.assertEqual(status, 201)

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        try:
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

                def call(method: str, path: str, token: str | None, body: object = None):
                    headers = {"Content-Type": "application/json"}
                    if token is not None:
                        headers["Authorization"] = f"Bearer {token}"
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    if body is None:
                        conn.request(method, path, headers=headers)
                    else:
                        conn.request(method, path, body=json.dumps(body), headers=headers)
                    response = conn.getresponse()
                    raw = response.read()
                    conn.close()
                    payload = json.loads(raw.decode("utf-8")) if raw else None
                    return response.status, payload

                # The policy comes from the same configured file (now holding
                # the reloaded content), not from any persisted copy.
                self.assertEqual(call("POST", RELOAD_PATH, "restart-admin", {})[0], 200)
                self.assertEqual(call("GET", METRICS_PATH, ADMIN_TOKEN)[0], 401)
                self.assertEqual(call("GET", METRICS_PATH, READ_TOKEN)[0], 200)
                # Business data recovered independently of the policy.
                status, state = call(
                    "GET", "/v1/states/restart-color", "restart-admin"
                )
                self.assertEqual(status, 200)
                self.assertEqual(state["value"], "kept")
            finally:
                restarted.shutdown()
                restarted.server_close()
                thread.join(timeout=5)
        finally:
            # Rebuild the class server for any later test in this class.
            self.write_policy(INITIAL_BYTES)
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

    # -- everything else stays as it was --

    def test_health_stays_anonymous_and_existing_traffic_is_unchanged(self) -> None:
        self.write_policy(
            json.dumps({**INITIAL_POLICY, "extra": ["read"]}).encode("utf-8")
        )
        self.assertEqual(self.reload()[0], 200)
        # Health stays anonymous in scope mode.
        status, payload, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})
        # Existing write/query/idempotency semantics are unchanged; unique ids
        # and key keep this test independent of the class-shared store.
        doc = operation_document(
            op_id="op-health-check", key="health-color", value="blue"
        )
        self.assertEqual(self.request("POST", OP_PATH, doc, token=WRITE_TOKEN)[0], 201)
        self.assertEqual(self.request("POST", OP_PATH, doc, token=ADMIN_TOKEN)[0], 200)
        status, conflicting, _ = self.request(
            "POST",
            OP_PATH,
            operation_document(
                op_id="op-health-check", key="health-color", value="red"
            ),
            token=WRITE_TOKEN,
        )
        self.assertEqual(status, 409)
        self.assertEqual(conflicting, {"error": "operation_conflict"})
        status, state, _ = self.request(
            "GET", "/v1/states/health-color", token="extra"
        )
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "blue")


if __name__ == "__main__":
    unittest.main()
