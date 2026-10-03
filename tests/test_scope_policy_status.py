"""Tests for the scope-policy status query and conditional hot reload.

``GET /v1/admin/scope-policy/status`` reports the live policy's digest and
token count without re-reading the configured file or writing the data
file. ``POST /v1/admin/scope-policy/reload`` additionally accepts
``{"expectedPolicyDigest": "<64 lowercase hex>"}`` as a conditional
compare-then-swap reload: the expectation is checked against the live
digest inside the reload serialization, and a mismatch is HTTP 409
``policy_state_conflict`` without reading the file, swapping the policy,
or recording an audit event.
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
    parse_scope_policy_reload_payload,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
STATUS_PATH = "/v1/admin/scope-policy/status"
RELOAD_PATH = "/v1/admin/scope-policy/reload"
AUDIT_PATH = "/v1/admin/scope-policy/audit"
METRICS_PATH = "/v1/metrics"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")
INITIAL_DIGEST = hashlib.sha256(INITIAL_BYTES).hexdigest()

OTHER_POLICY = {
    "new-admin": ["read", "write", "admin"],
    READ_TOKEN: ["read"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
OTHER_BYTES = json.dumps(OTHER_POLICY).encode("utf-8")
OTHER_DIGEST = hashlib.sha256(OTHER_BYTES).hexdigest()

STALE_DIGEST = hashlib.sha256(b'{"stale":["admin"]}').hexdigest()


class ParseScopePolicyReloadPayloadTests(unittest.TestCase):
    def test_empty_object_means_unconditional(self) -> None:
        for raw in (b"{}", "{}", {}, b"{ }", b"{\n\t}"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_scope_policy_reload_payload(raw))

    def test_conditional_shape_returns_the_digest(self) -> None:
        body = {"expectedPolicyDigest": INITIAL_DIGEST}
        self.assertEqual(parse_scope_policy_reload_payload(body), INITIAL_DIGEST)
        self.assertEqual(
            parse_scope_policy_reload_payload(json.dumps(body).encode("utf-8")),
            INITIAL_DIGEST,
        )

    def test_anything_else_is_rejected(self) -> None:
        for raw in (
            b"",
            b"junk",
            b"{",
            b"[]",
            b"null",
            b"0",
            b"true",
            b'{"a":1}',
            b'{"expectedPolicyDigest":"%s","extra":1}' % INITIAL_DIGEST.encode(),
            b'{"expectedPolicyDigest":null}',
            b'{"expectedPolicyDigest":64}',
            b'{"expectedPolicyDigest":true}',
            b'{"expectedPolicyDigest":["%s"]}' % INITIAL_DIGEST.encode(),
            # Wrong length, uppercase, and non-hex digests are bad shapes.
            b'{"expectedPolicyDigest":"%s"}' % INITIAL_DIGEST[:63].encode(),
            b'{"expectedPolicyDigest":"%s0"}' % INITIAL_DIGEST.encode(),
            b'{"expectedPolicyDigest":"%s"}' % INITIAL_DIGEST.upper().encode(),
            b'{"expectedPolicyDigest":"%s"}' % (b"g" * 64),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_scope_policy_reload_payload(raw)


class ScopePolicyManagerStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-status-unit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "scopes.json")
        with open(self.path, "wb") as handle:
            handle.write(INITIAL_BYTES)
        self.manager = ScopePolicyManager(self.path, load_scope_policy(self.path))

    def write(self, content: bytes) -> None:
        with open(self.path, "wb") as handle:
            handle.write(content)

    def test_status_reports_the_construction_digest_without_rereading(self) -> None:
        digest, tokens = self.manager.status()
        self.assertEqual(digest, INITIAL_DIGEST)
        self.assertEqual(tokens, 3)
        # Changing the file on disk changes nothing until a reload commits.
        self.write(OTHER_BYTES)
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_status_follows_successful_reloads_only(self) -> None:
        self.write(OTHER_BYTES)
        self.manager.reload()
        self.assertEqual(self.manager.status(), (OTHER_DIGEST, 3))
        # A failed reload leaves the reported state untouched.
        self.write(b'{"t":["nope"]}')
        with self.assertRaises(ScopePolicyReloadError):
            self.manager.reload()
        self.assertEqual(self.manager.status(), (OTHER_DIGEST, 3))

    def test_conditional_reload_matches_and_swaps(self) -> None:
        self.write(OTHER_BYTES)
        digest, tokens = self.manager.reload(expected_digest=INITIAL_DIGEST)
        self.assertEqual((digest, tokens), (OTHER_DIGEST, 3))
        self.assertEqual(set(self.manager.snapshot()), set(OTHER_POLICY))
        self.assertEqual(self.manager.status(), (OTHER_DIGEST, 3))

    def test_conditional_reload_mismatch_never_reads_the_file(self) -> None:
        # An unreadable file proves the read never happens: a mismatch must
        # be state_conflict, not unavailable.
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload(expected_digest=STALE_DIGEST)
        self.assertEqual(caught.exception.kind, "state_conflict")
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_conditional_reload_mismatch_records_no_event(self) -> None:
        recorded: list[tuple[str, int]] = []
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload(
                recorder=lambda d, t: recorded.append((d, t)),
                expected_digest=STALE_DIGEST,
            )
        self.assertEqual(caught.exception.kind, "state_conflict")
        self.assertEqual(recorded, [])

    def test_conditional_reload_still_validates_the_file(self) -> None:
        self.write(b'{"t":["nope"]}')
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload(expected_digest=INITIAL_DIGEST)
        self.assertEqual(caught.exception.kind, "conflict")
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_unconditional_reload_ignores_no_expectation(self) -> None:
        self.write(OTHER_BYTES)
        digest, _ = self.manager.reload()
        self.assertEqual(digest, OTHER_DIGEST)


class ScopePolicyStatusHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-status-http-")
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

    def status(self, token: str | None = ADMIN_TOKEN, query: str = ""):
        return self.request("GET", STATUS_PATH + query, token=token)

    def reload(self, body: object = {}, token: str | None = ADMIN_TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    def events_count(self) -> int:
        status, payload, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        return payload["eventsCount"]

    # -- status success contract --

    def test_status_returns_the_three_contracted_fields_in_order(self) -> None:
        code, payload, raw = self.status()
        self.assertEqual(code, 200)
        self.assertEqual(
            raw.decode("utf-8"),
            f'{{"status":"active","policyDigest":"{INITIAL_DIGEST}","tokens":3}}',
        )
        self.assertEqual(payload, {
            "status": "active",
            "policyDigest": INITIAL_DIGEST,
            "tokens": 3,
        })

    def test_status_reflects_a_committed_reload(self) -> None:
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.reload()[0], 200)
        code, payload, _ = self.status(token="new-admin")
        self.assertEqual(code, 200)
        self.assertEqual(payload["policyDigest"], OTHER_DIGEST)
        self.assertEqual(payload["tokens"], 3)

    def test_status_never_rereads_the_file(self) -> None:
        # Rewriting — even deleting — the configured file changes nothing
        # until a reload commits.
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.status()[1]["policyDigest"], INITIAL_DIGEST)
        os.unlink(self.policy_path)
        code, payload, _ = self.status()
        self.assertEqual(code, 200)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["tokens"], 3)

    def test_status_does_not_touch_the_data_file(self) -> None:
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        self.assertEqual(self.status()[0], 200)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    # -- status request validation and gating --

    def test_status_rejects_any_query_parameter(self) -> None:
        for query in ("?x", "?x=", "?x=1", "?after=0&limit=1"):
            with self.subTest(query=query):
                code, payload, _ = self.status(query=query)
                self.assertEqual(code, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_status_path_shape_mismatches_are_404(self) -> None:
        for path in (
            STATUS_PATH + "/",
            STATUS_PATH + "/extra",
            "/v1/admin/scope-policy",
            "/v1/admin/scopepolicy/status",
        ):
            with self.subTest(path=path):
                code, payload, _ = self.request("GET", path + "?x=1", token=ADMIN_TOKEN)
                self.assertEqual(code, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_status_requires_the_admin_scope(self) -> None:
        self.assertEqual(self.status(token=None)[0], 401)
        self.assertEqual(self.status(token="unknown")[0], 401)
        self.assertEqual(self.status(token=READ_TOKEN)[0], 403)
        self.assertEqual(self.status(token=WRITE_TOKEN)[0], 403)

    def test_status_is_not_published_outside_scope_policy_mode(self) -> None:
        for kwargs, token, expected in (
            ({"auth_token": "legacy-token"}, "legacy-token", 404),
            ({}, None, 404),
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
                    conn.request("GET", STATUS_PATH, headers=headers)
                    response = conn.getresponse()
                    payload = json.loads(response.read().decode("utf-8"))
                    conn.close()
                    self.assertEqual(response.status, expected)
                    self.assertEqual(payload, {"error": "not_found"})
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_status_post_is_not_published(self) -> None:
        code, _, _ = self.request("POST", STATUS_PATH, body={}, token=ADMIN_TOKEN)
        self.assertEqual(code, 404)

    # -- conditional reload --

    def test_conditional_reload_with_the_live_digest_succeeds(self) -> None:
        self.write_policy(OTHER_BYTES)
        code, payload, _ = self.reload({"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(code, 200)
        self.assertEqual(payload["status"], "reloaded")
        self.assertEqual(payload["policyDigest"], OTHER_DIGEST)
        self.assertEqual(payload["tokens"], 3)
        # The status query now reports the new digest.
        self.assertEqual(self.status(token="new-admin")[1]["policyDigest"], OTHER_DIGEST)

    def test_conditional_reload_with_a_stale_digest_is_409_and_changes_nothing(self) -> None:
        count_before = self.events_count()
        self.write_policy(OTHER_BYTES)
        code, payload, _ = self.reload({"expectedPolicyDigest": STALE_DIGEST})
        self.assertEqual(code, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})
        # The old boundary is fully in force and the file was not loaded.
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(self.request("GET", METRICS_PATH, token="new-admin")[0], 401)
        self.assertEqual(self.status()[1]["policyDigest"], INITIAL_DIGEST)
        # No audit event was recorded.
        self.assertEqual(self.events_count(), count_before)

    def test_conditional_reload_mismatch_never_reads_the_file(self) -> None:
        # An invalid file would be 409 policy_conflict if it were read; a
        # digest mismatch must short-circuit before the read.
        self.write_policy(b'{"t":["nope"]}')
        code, payload, _ = self.reload({"expectedPolicyDigest": STALE_DIGEST})
        self.assertEqual(code, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})
        # A missing file would be 503 if it were read.
        os.unlink(self.policy_path)
        code, payload, _ = self.reload({"expectedPolicyDigest": STALE_DIGEST})
        self.assertEqual(code, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})

    def test_conditional_reload_matching_digest_still_validates_the_file(self) -> None:
        self.write_policy(b'{"t":["nope"]}')
        code, payload, _ = self.reload({"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(code, 409)
        self.assertEqual(payload, {"error": "policy_conflict"})
        os.unlink(self.policy_path)
        code, payload, _ = self.reload({"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(code, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})

    def test_conditional_reload_after_a_reload_observes_the_new_digest(self) -> None:
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.reload()[0], 200)
        # The pre-reload digest is now stale.
        code, payload, _ = self.reload({"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(code, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})
        # The fresh digest matches.
        code, _, _ = self.reload({"expectedPolicyDigest": OTHER_DIGEST}, token="new-admin")
        self.assertEqual(code, 200)

    def test_reload_body_shape_validation(self) -> None:
        for body in (
            {"expectedPolicyDigest": STALE_DIGEST.upper()},
            {"expectedPolicyDigest": STALE_DIGEST[:-1]},
            {"expectedPolicyDigest": STALE_DIGEST + "0"},
            {"expectedPolicyDigest": None},
            {"expectedPolicyDigest": 64},
            {"expectedPolicyDigest": [STALE_DIGEST]},
            {"expectedPolicyDigest": STALE_DIGEST, "extra": 1},
            {"policyDigest": STALE_DIGEST},
            {"a": 1},
            [],
            0,
            True,
        ):
            with self.subTest(body=body):
                code, payload, _ = self.reload(body)
                self.assertEqual(code, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_unconditional_reload_still_works(self) -> None:
        self.write_policy(OTHER_BYTES)
        code, payload, _ = self.reload({})
        self.assertEqual(code, 200)
        self.assertEqual(payload["policyDigest"], OTHER_DIGEST)

    def test_conditional_reload_success_records_exactly_one_event(self) -> None:
        count_before = self.events_count()
        self.write_policy(OTHER_BYTES)
        self.assertEqual(self.reload({"expectedPolicyDigest": INITIAL_DIGEST})[0], 200)
        self.assertEqual(self.events_count(), count_before + 1)

    # -- concurrency: conditional reloads are one compare-then-swap --

    def test_concurrent_conditional_reloads_commit_exactly_one_swap(self) -> None:
        self.write_policy(OTHER_BYTES)
        outcomes: list[int] = []
        outcomes_lock = threading.Lock()
        start = threading.Barrier(8)

        def do_reload() -> None:
            start.wait()
            for attempt in range(10):
                try:
                    code, _, _ = self.reload({"expectedPolicyDigest": INITIAL_DIGEST})
                    break
                except ConnectionError:
                    if attempt == 9:
                        raise
            with outcomes_lock:
                outcomes.append(code)

        threads = [threading.Thread(target=do_reload) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        # Every conditional reload expected the same pre-change digest:
        # exactly one compare-then-swap matched and committed; the rest
        # observed the new digest and conflicted.
        self.assertEqual(sorted(outcomes), [200] + [409] * 7)
        self.assertEqual(self.status(token="new-admin")[1]["policyDigest"], OTHER_DIGEST)


if __name__ == "__main__":
    unittest.main()
