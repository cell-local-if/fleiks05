"""Tests for the scope-policy preview query.

``GET /v1/admin/scope-policy/preview`` validates the file configured with
``--scope-policy-file`` and quantifies its impact against the live policy
without swapping anything: one file read, one live-policy snapshot, no
audit event, no data-file or temporary-file write. The endpoint is
published only in scope-policy mode and requires the admin scope.
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

# One added token, one removed token, one changed scope set, one unchanged.
CANDIDATE_POLICY = {
    ADMIN_TOKEN: ["admin", "write", "read"],  # same set, different order
    WRITE_TOKEN: ["write", "read"],  # changed
    "new-token": ["read"],  # added
}
CANDIDATE_BYTES = json.dumps(CANDIDATE_POLICY).encode("utf-8")
CANDIDATE_DIGEST = hashlib.sha256(CANDIDATE_BYTES).hexdigest()


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

    def test_preview_counts_each_change_class(self) -> None:
        self.write(CANDIDATE_BYTES)
        candidate_digest, candidate_tokens, current_digest, current_tokens, changes = (
            self.manager.preview()
        )
        self.assertEqual(candidate_digest, CANDIDATE_DIGEST)
        self.assertEqual(candidate_tokens, 3)
        self.assertEqual(current_digest, INITIAL_DIGEST)
        self.assertEqual(current_tokens, 3)
        self.assertEqual(
            changes, {"added": 1, "removed": 1, "changed": 1, "unchanged": 1}
        )
        # candidateTokens == added + changed + unchanged;
        # currentTokens == removed + changed + unchanged.
        self.assertEqual(candidate_tokens, 1 + 1 + 1)
        self.assertEqual(current_tokens, 1 + 1 + 1)

    def test_preview_of_the_live_content_is_all_unchanged(self) -> None:
        _, _, _, _, changes = self.manager.preview()
        self.assertEqual(
            changes, {"added": 0, "removed": 0, "changed": 0, "unchanged": 3}
        )

    def test_preview_leaves_the_live_state_untouched(self) -> None:
        self.write(CANDIDATE_BYTES)
        self.manager.preview()
        # The live mapping and its digest are exactly as before, so a
        # following reload is still the only source of change.
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_preview_unavailable_when_the_file_cannot_be_read(self) -> None:
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.preview()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_preview_unavailable_for_a_non_regular_file(self) -> None:
        os.unlink(self.path)
        os.mkdir(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.preview()
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_preview_conflict_for_invalid_content(self) -> None:
        for content in (
            b'{"t":["nope"]}',
            b'{"t":[]}',
            b'{"t":["read","read"]}',
            b'{"":["read"]}',
            b'{"a b":["read"]}',
            b'{"t":"read"}',
            b"[1]",
            b"{",
            b'{"t":["read"]}{"u":["read"]}',
            b'{"t":["read"],"t":["read"]}',
            b"\xff",
        ):
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(ScopePolicyReloadError) as caught:
                    self.manager.preview()
                self.assertEqual(caught.exception.kind, "conflict")
                # The live policy stays fully in force.
                self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))
                self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))

    def test_preview_error_never_echoes_tokens_or_scopes(self) -> None:
        self.write(b'{"secret-token":["nope-scope"]}')
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.preview()
        message = str(caught.exception)
        self.assertNotIn("secret-token", message)
        self.assertNotIn("nope-scope", message)

    def test_preview_is_stable_for_unchanged_input(self) -> None:
        self.write(CANDIDATE_BYTES)
        first = self.manager.preview()
        second = self.manager.preview()
        self.assertEqual(first, second)

    def test_preview_observes_one_committed_revision(self) -> None:
        # After a committed reload the preview diffs against the new live
        # policy, never a mix of old and new.
        self.write(CANDIDATE_BYTES)
        self.manager.reload()
        candidate_digest, candidate_tokens, current_digest, current_tokens, changes = (
            self.manager.preview()
        )
        self.assertEqual(candidate_digest, CANDIDATE_DIGEST)
        self.assertEqual(current_digest, CANDIDATE_DIGEST)
        self.assertEqual(
            changes, {"added": 0, "removed": 0, "changed": 0, "unchanged": 3}
        )
        self.assertEqual(candidate_tokens, current_tokens)


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
    ) -> tuple[int, object, bytes, dict[str, str]]:
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
        response_headers = {name.lower(): value for name, value in response.getheaders()}
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

    # -- preview success contract --

    def test_preview_returns_the_contracted_fields_in_order_with_newline(self) -> None:
        self.write_policy(CANDIDATE_BYTES)
        code, payload, raw, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(
            raw.decode("utf-8"),
            '{"status":"valid",'
            f'"candidateDigest":"{CANDIDATE_DIGEST}",'
            '"candidateTokens":3,'
            f'"currentDigest":"{INITIAL_DIGEST}",'
            '"currentTokens":3,'
            '"changes":{"added":1,"removed":1,"changed":1,"unchanged":1}}\n',
        )
        self.assertEqual(payload["status"], "valid")

    def test_preview_of_the_live_content_is_all_unchanged(self) -> None:
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(payload["candidateDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["currentDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["candidateTokens"], 3)
        self.assertEqual(payload["currentTokens"], 3)
        self.assertEqual(
            payload["changes"],
            {"added": 0, "removed": 0, "changed": 0, "unchanged": 3},
        )

    def test_preview_scope_arrays_compare_as_sets(self) -> None:
        # Same tokens, same scope sets, different array order: all unchanged.
        reordered = json.dumps(
            {
                ADMIN_TOKEN: ["admin", "write", "read"],
                WRITE_TOKEN: ["write"],
                READ_TOKEN: ["read"],
            }
        ).encode("utf-8")
        self.write_policy(reordered)
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 200)
        self.assertEqual(
            payload["changes"],
            {"added": 0, "removed": 0, "changed": 0, "unchanged": 3},
        )

    def test_preview_token_counts_follow_the_change_counts(self) -> None:
        self.write_policy(CANDIDATE_BYTES)
        code, payload, _, _ = self.preview()
        self.assertEqual(code, 200)
        changes = payload["changes"]
        self.assertEqual(
            payload["candidateTokens"],
            changes["added"] + changes["changed"] + changes["unchanged"],
        )
        self.assertEqual(
            payload["currentTokens"],
            changes["removed"] + changes["changed"] + changes["unchanged"],
        )

    def test_preview_is_stable_for_unchanged_input(self) -> None:
        self.write_policy(CANDIDATE_BYTES)
        first = self.preview()
        second = self.preview()
        self.assertEqual(first[0], 200)
        self.assertEqual(first[2], second[2])

    def test_preview_swaps_nothing_and_records_no_event(self) -> None:
        count_before = self.events_count()
        self.write_policy(CANDIDATE_BYTES)
        self.assertEqual(self.preview()[0], 200)
        # The live boundary is untouched: the old tokens still authenticate
        # exactly as before and the new token is not live.
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token="new-token")[0], 401)
        code, payload, _, _ = self.request("GET", STATUS_PATH)
        self.assertEqual(code, 200)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["tokens"], 3)
        # No audit event was recorded.
        self.assertEqual(self.events_count(), count_before)

    def test_preview_does_not_touch_the_data_file(self) -> None:
        self.write_policy(CANDIDATE_BYTES)
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        self.assertEqual(self.preview()[0], 200)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_preview_creates_no_temporary_file(self) -> None:
        self.write_policy(CANDIDATE_BYTES)
        before = set(os.listdir(self.tmpdir))
        self.assertEqual(self.preview()[0], 200)
        self.assertEqual(set(os.listdir(self.tmpdir)), before)

    # -- preview request validation and gating --

    def test_preview_rejects_any_query_parameter_without_reading_the_file(self) -> None:
        # An invalid file would be 409 if it were read; a query-parameter
        # rejection must short-circuit before the read.
        self.write_policy(b'{"t":["nope"]}')
        for query in ("?x", "?x=", "?x=1", "?after=0&limit=1"):
            with self.subTest(query=query):
                code, payload, _, _ = self.preview(query=query)
                self.assertEqual(code, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

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

    def test_preview_requires_the_admin_scope(self) -> None:
        code, payload, _, headers = self.preview(token=None)
        self.assertEqual(code, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        code, payload, _, headers = self.preview(token="unknown")
        self.assertEqual(code, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("www-authenticate"), "Bearer")
        for token in (READ_TOKEN, WRITE_TOKEN):
            code, payload, _, headers = self.preview(token=token)
            self.assertEqual(code, 403)
            self.assertEqual(payload, {"error": "forbidden"})
            self.assertNotIn("www-authenticate", headers)

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

    def test_preview_post_is_not_published(self) -> None:
        code, _, _, _ = self.request("POST", PREVIEW_PATH, body={}, token=ADMIN_TOKEN)
        self.assertEqual(code, 404)

    # -- preview failure modes --

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
            b'{"":["read"]}',
            b'{"t":"read"}',
            b"[1]",
            b"{",
            b"\xff",
        ):
            with self.subTest(content=content):
                self.write_policy(content)
                code, payload, _, _ = self.preview()
                self.assertEqual(code, 409)
                self.assertEqual(payload, {"error": "policy_conflict"})

    def test_preview_error_never_echoes_content(self) -> None:
        self.write_policy(b'{"secret-token":["nope-scope"]}')
        code, _, raw, _ = self.preview()
        self.assertEqual(code, 409)
        self.assertNotIn(b"secret-token", raw)
        self.assertNotIn(b"nope-scope", raw)

    def test_preview_failure_leaves_the_live_policy_in_force(self) -> None:
        count_before = self.events_count()
        self.write_policy(b'{"t":["nope"]}')
        self.assertEqual(self.preview()[0], 409)
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token="t")[0], 401)
        code, payload, _, _ = self.request("GET", STATUS_PATH)
        self.assertEqual(code, 200)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)
        self.assertEqual(self.events_count(), count_before)

    def test_preview_follows_a_committed_reload(self) -> None:
        # After a reload commits, the preview diffs against the new live
        # policy: the same file now previews as all-unchanged. The
        # reloaded policy still grants ADMIN_TOKEN the admin scope.
        self.write_policy(CANDIDATE_BYTES)
        code, _, _, _ = self.request(
            "POST", "/v1/admin/scope-policy/reload", body={}, token=ADMIN_TOKEN
        )
        self.assertEqual(code, 200)
        code, payload, _, _ = self.preview(token=ADMIN_TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual(
            payload["changes"],
            {"added": 0, "removed": 0, "changed": 0, "unchanged": 3},
        )
        self.assertEqual(payload["currentDigest"], CANDIDATE_DIGEST)
        self.assertEqual(payload["candidateDigest"], CANDIDATE_DIGEST)


if __name__ == "__main__":
    unittest.main()
