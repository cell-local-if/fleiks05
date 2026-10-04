"""Tests for the scope-policy reload preflight.

``GET /v1/admin/scope-policy/preview`` reads the ``--scope-policy-file``
file once, validates it under the startup/reload constraints, and diffs it
against one snapshot of the live policy — reporting the candidate and
current digests, token counts, and the added/removed/changed/unchanged
breakdown — without swapping the boundary, recording an audit event, or
writing the data file or a temporary file. The endpoint is published only
in scope-policy mode and is gated by the admin scope.
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
    RequestHandler,
    ScopePolicyManager,
    ScopePolicyReloadError,
    SemanticStateServer,
    load_scope_policy,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
PREVIEW_PATH = "/v1/admin/scope-policy/preview"
STATUS_PATH = "/v1/admin/scope-policy/status"
AUDIT_PATH = "/v1/admin/scope-policy/audit"
METRICS_PATH = "/v1/metrics"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")
INITIAL_DIGEST = hashlib.sha256(INITIAL_BYTES).hexdigest()

# One token only in the candidate (added), one only in the live policy
# (removed), one shared with a different scope set (changed), one shared
# with the same scope set (unchanged).
OTHER_POLICY = {
    ADMIN_TOKEN: ["read", "write", "admin"],
    WRITE_TOKEN: ["write", "read"],
    "new-admin": ["admin"],
}
OTHER_BYTES = json.dumps(OTHER_POLICY).encode("utf-8")
OTHER_DIGEST = hashlib.sha256(OTHER_BYTES).hexdigest()


class ScopePolicyManagerPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-preview-unit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "scopes.json")
        with open(self.path, "wb") as handle:
            handle.write(INITIAL_BYTES)
        self.manager = ScopePolicyManager(self.path, load_scope_policy(self.path))

    def write(self, content: bytes) -> None:
        with open(self.path, "wb") as handle:
            handle.write(content)

    def test_identical_file_reports_all_unchanged(self) -> None:
        digest, tokens, current_digest, current_tokens, changes = self.manager.preview()
        self.assertEqual(digest, INITIAL_DIGEST)
        self.assertEqual(tokens, 3)
        self.assertEqual(current_digest, INITIAL_DIGEST)
        self.assertEqual(current_tokens, 3)
        self.assertEqual(
            changes, {"added": 0, "removed": 0, "changed": 0, "unchanged": 3}
        )

    def test_diff_counts_each_bucket_once(self) -> None:
        self.write(OTHER_BYTES)
        digest, tokens, current_digest, current_tokens, changes = self.manager.preview()
        self.assertEqual(digest, OTHER_DIGEST)
        self.assertEqual(current_digest, INITIAL_DIGEST)
        self.assertEqual(
            changes, {"added": 1, "removed": 1, "changed": 1, "unchanged": 1}
        )
        # candidateTokens = added + changed + unchanged.
        self.assertEqual(tokens, 3)
        # currentTokens = removed + changed + unchanged.
        self.assertEqual(current_tokens, 3)

    def test_scope_array_order_is_irrelevant(self) -> None:
        reordered = json.dumps({
            ADMIN_TOKEN: ["admin", "write", "read"],
            WRITE_TOKEN: ["write"],
            READ_TOKEN: ["read"],
        }).encode("utf-8")
        self.write(reordered)
        _, _, _, _, changes = self.manager.preview()
        self.assertEqual(
            changes, {"added": 0, "removed": 0, "changed": 0, "unchanged": 3}
        )

    def test_preview_never_swaps_the_live_policy(self) -> None:
        self.write(OTHER_BYTES)
        self.manager.preview()
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_preview_is_stable_when_nothing_changes(self) -> None:
        self.write(OTHER_BYTES)
        self.assertEqual(self.manager.preview(), self.manager.preview())

    def test_missing_file_is_unavailable(self) -> None:
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.preview()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_non_regular_file_is_unavailable(self) -> None:
        os.unlink(self.path)
        os.mkdir(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.preview()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_invalid_content_is_conflict(self) -> None:
        for content in (
            b'{"t":["nope"]}',
            b'{"t":[]}',
            b'{"t":["read","read"]}',
            b'{"t":"read"}',
            b"[]",
            b"{",
            b"\xff\xfe",
        ):
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(ScopePolicyReloadError) as caught:
                    self.manager.preview()
                self.assertEqual(caught.exception.kind, "conflict")
                # The live policy survives every failed preview.
                self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_preview_after_a_reload_compares_against_the_new_policy(self) -> None:
        self.write(OTHER_BYTES)
        self.manager.reload()
        _, _, current_digest, current_tokens, changes = self.manager.preview()
        self.assertEqual(current_digest, OTHER_DIGEST)
        self.assertEqual(current_tokens, 3)
        self.assertEqual(
            changes, {"added": 0, "removed": 0, "changed": 0, "unchanged": 3}
        )


class ScopePolicyPreviewHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-preview-http-")
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
        self.addCleanup(self.server.scope_policy.reload)
        self.addCleanup(self.write_policy, INITIAL_BYTES)

    def write_policy(self, content: bytes) -> None:
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
    ) -> tuple[int, object, bytes, object]:
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

    def preview(self, token: str | None = ADMIN_TOKEN, query: str = ""):
        return self.request("GET", PREVIEW_PATH + query, token=token)

    def events_count(self) -> int:
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        return payload["eventsCount"]

    # -- success contract --

    def test_preview_returns_the_contracted_fields_in_order_with_newline(self) -> None:
        code, payload, raw, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(
            raw.decode("utf-8"),
            f'{{"status":"valid","candidateDigest":"{INITIAL_DIGEST}",'
            f'"candidateTokens":3,"currentDigest":"{INITIAL_DIGEST}",'
            f'"currentTokens":3,"changes":{{"added":0,"removed":0,'
            f'"changed":0,"unchanged":3}}}}\n',
        )
        self.assertEqual(payload["status"], "valid")

    def test_preview_reports_the_diff_against_the_live_policy(self) -> None:
        self.write_policy(OTHER_BYTES)
        code, payload, raw, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(payload["candidateDigest"], OTHER_DIGEST)
        self.assertEqual(payload["candidateTokens"], 3)
        self.assertEqual(payload["currentDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["currentTokens"], 3)
        self.assertEqual(
            payload["changes"],
            {"added": 1, "removed": 1, "changed": 1, "unchanged": 1},
        )
        # The response never carries tokens or scopes.
        self.assertNotIn(READ_TOKEN, raw.decode("utf-8"))
        self.assertNotIn("new-admin", raw.decode("utf-8"))
        self.assertNotIn("read", raw.decode("utf-8"))

    def test_preview_scope_array_order_is_irrelevant(self) -> None:
        self.write_policy(json.dumps({
            ADMIN_TOKEN: ["admin", "write", "read"],
            WRITE_TOKEN: ["write"],
            READ_TOKEN: ["read"],
        }).encode("utf-8"))
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(
            payload["changes"],
            {"added": 0, "removed": 0, "changed": 0, "unchanged": 3},
        )

    def test_preview_is_stable_when_nothing_changes(self) -> None:
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.preview()[2], self.preview()[2])

    def test_preview_reflects_a_committed_reload_as_current(self) -> None:
        self.write_policy(OTHER_BYTES)
        code, _, _, _ = self.request(
            "POST", "/v1/admin/scope-policy/reload", body={}, token=ADMIN_TOKEN
        )
        self.assertEqual(code, 200)
        code, payload, _, _ = self.preview(token="new-admin")
        self.assertEqual(code, 200)
        self.assertEqual(payload["currentDigest"], OTHER_DIGEST)
        self.assertEqual(
            payload["changes"],
            {"added": 0, "removed": 0, "changed": 0, "unchanged": 3},
        )

    # -- the preview changes nothing --

    def test_preview_does_not_swap_the_boundary(self) -> None:
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.preview()[0], 200)
        # The old boundary is still fully in force.
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token="new-admin")[0], 401)
        _, status_payload, _, _ = self.request("GET", STATUS_PATH, token=ADMIN_TOKEN)
        self.assertEqual(status_payload["policyDigest"], INITIAL_DIGEST)

    def test_preview_records_no_audit_event(self) -> None:
        count_before = self.events_count()
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.preview()[0], 200)
        self.assertEqual(self.events_count(), count_before)

    def test_preview_does_not_touch_the_data_file(self) -> None:
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.preview()[0], 200)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    # -- request validation and failure classification --

    def test_preview_rejects_any_query_parameter_without_reading_the_file(self) -> None:
        # An invalid file would be 409 and a missing file 503 if the query
        # check did not run first; any parameter must be 400 regardless.
        self.write_policy(b'{"t":["nope"]}')
        for query in ("?x", "?x=", "?x=1", "?after=0&limit=1"):
            with self.subTest(query=query):
                code, payload, _, _ = self.preview(query=query)
                self.assertEqual(code, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        os.unlink(self.policy_path)
        code, payload, _, _ = self.preview(query="?x=1")
        self.assertEqual(code, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_preview_missing_file_is_503(self) -> None:
        os.unlink(self.policy_path)
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})

    def test_preview_non_regular_file_is_503(self) -> None:
        os.unlink(self.policy_path)
        os.mkdir(self.policy_path)
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})

    def test_preview_invalid_content_is_409(self) -> None:
        for content in (
            b'{"t":["nope"]}',
            b'{"t":[]}',
            b'{"t":["read","read"]}',
            b"[]",
            b"{",
            b"\xff\xfe",
        ):
            with self.subTest(content=content):
                self.write_policy(content)
                code, payload, raw, _ = self.preview()
                self.assertEqual(code, 409)
                self.assertEqual(payload, {"error": "policy_conflict"})
                # The error never echoes content, tokens, scopes, or
                # validation details.
                self.assertNotIn("nope", raw.decode("utf-8"))
                # The live boundary survives the failed preview.
                self.assertEqual(
                    self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200
                )

    def test_preview_path_shape_mismatches_are_404(self) -> None:
        for path in (
            PREVIEW_PATH + "/",
            PREVIEW_PATH + "/extra",
            "/v1/admin/scope-policy",
            "/v1/admin/scopepolicy/preview",
        ):
            with self.subTest(path=path):
                code, payload, _, _ = self.request(
                    "GET", path + "?x=1", token=ADMIN_TOKEN
                )
                self.assertEqual(code, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_preview_post_is_not_published(self) -> None:
        code, payload, _, _ = self.request(
            "POST", PREVIEW_PATH, body={}, token=ADMIN_TOKEN
        )
        self.assertEqual(code, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # -- authentication and authorization --

    def test_preview_requires_authentication_with_a_challenge(self) -> None:
        for token in (None, "unknown"):
            with self.subTest(token=token):
                code, payload, _, headers = self.preview(token=token)
                self.assertEqual(code, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_preview_requires_the_admin_scope_without_a_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                code, payload, _, headers = self.preview(token=token)
                self.assertEqual(code, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)

    def test_preview_is_not_published_outside_scope_policy_mode(self) -> None:
        for kwargs, token in (
            ({"auth_token": "legacy-token"}, "legacy-token"),
            ({}, None),
        ):
            with self.subTest(kwargs=kwargs):
                server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    port = server.server_address[1]
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    headers = {}
                    if token is not None:
                        headers["Authorization"] = f"Bearer {token}"
                    conn.request("GET", PREVIEW_PATH, headers=headers)
                    response = conn.getresponse()
                    payload = json.loads(response.read().decode("utf-8"))
                    conn.close()
                    self.assertEqual(response.status, 404)
                    self.assertEqual(payload, {"error": "not_found"})
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
