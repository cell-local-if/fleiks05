"""Tests for the scope-policy status query and conditional hot reload.

``GET /v1/admin/scope-policy/status`` reports the live policy's digest and
token count without re-reading the configured file, and
``POST /v1/admin/scope-policy/reload`` additionally accepts
``{"expectedPolicyDigest": "<64 lowercase hex>"}`` for a conditional
compare-and-swap reload. The tests cover the status contract (field order,
digest source, no file or data-file effects, request precedence), the
conditional-reload contract (match, stale-expectation 409 without a file
read or an event, body validation), and the mode gates.
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
RELOAD_PATH = "/v1/admin/scope-policy/reload"
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

STALE_DIGEST = hashlib.sha256(b'{"other":["admin"]}').hexdigest()


class ParseScopePolicyReloadPayloadTests(unittest.TestCase):
    def test_empty_object_is_unconditional(self) -> None:
        for raw in (b"{}", "{}", {}, b"{ }", b"{\n\t}"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_scope_policy_reload_payload(raw))

    def test_expected_digest_shape_is_accepted(self) -> None:
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
            b'{"expectedPolicyDigest":""}',
            b'{"expectedPolicyDigest":null}',
            b'{"expectedPolicyDigest":0}',
            b'{"expectedPolicyDigest":true}',
            b'{"expectedPolicyDigest":["' + INITIAL_DIGEST.encode("ascii") + b'"]}',
            # Too short, too long, uppercase, and non-hex characters.
            b'{"expectedPolicyDigest":"' + INITIAL_DIGEST[:63].encode("ascii") + b'"}',
            b'{"expectedPolicyDigest":"' + INITIAL_DIGEST.encode("ascii") + b'0"}',
            b'{"expectedPolicyDigest":"' + INITIAL_DIGEST.upper().encode("ascii") + b'"}',
            b'{"expectedPolicyDigest":"' + (b"g" * 64) + b'"}',
            # Unknown or extra fields.
            b'{"expectedPolicyDigest":"' + INITIAL_DIGEST.encode("ascii") + b'","x":1}',
            b'{"policyDigest":"' + INITIAL_DIGEST.encode("ascii") + b'"}',
            # A duplicated key is rejected like everywhere else.
            b'{"expectedPolicyDigest":"' + INITIAL_DIGEST.encode("ascii")
            + b'","expectedPolicyDigest":"' + INITIAL_DIGEST.encode("ascii") + b'"}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_scope_policy_reload_payload(raw)
        for value in (
            [],
            None,
            0,
            True,
            {"a": 1},
            {"expectedPolicyDigest": None},
            {"expectedPolicyDigest": INITIAL_DIGEST.upper()},
            {"expectedPolicyDigest": INITIAL_DIGEST, "x": 1},
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_scope_policy_reload_payload(value)


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

    def test_initial_status_is_the_startup_file_digest(self) -> None:
        digest, tokens = self.manager.status()
        self.assertEqual(digest, INITIAL_DIGEST)
        self.assertEqual(tokens, 3)

    def test_status_does_not_re_read_the_file(self) -> None:
        # Rewriting (and even removing) the file changes nothing until a
        # reload commits: the status reports the revision in force.
        self.write(b'{"swapped":["admin"]}')
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))
        os.unlink(self.path)
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_status_follows_successful_reloads(self) -> None:
        content = b'{ "reader-token" : [ "read" ] }'
        self.write(content)
        self.manager.reload()
        self.assertEqual(
            self.manager.status(), (hashlib.sha256(content).hexdigest(), 1)
        )

    def test_failed_reload_leaves_status_untouched(self) -> None:
        self.write(b'{"t":["nope"]}')
        with self.assertRaises(ScopePolicyReloadError):
            self.manager.reload()
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_conditional_reload_matches_the_live_digest(self) -> None:
        content = json.dumps({"new-admin": ["admin"]}).encode("utf-8")
        self.write(content)
        digest, tokens = self.manager.reload(expected=INITIAL_DIGEST)
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertEqual(tokens, 1)
        self.assertEqual(self.manager.status(), (digest, 1))

    def test_stale_expectation_conflicts_without_reading_the_file(self) -> None:
        # The file is gone: a mismatched expectation must still be a state
        # conflict, proving the comparison completes before any file access.
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload(expected=STALE_DIGEST)
        self.assertEqual(caught.exception.kind, "state_conflict")
        self.assertEqual(set(self.manager.snapshot()), set(INITIAL_POLICY))
        self.assertEqual(self.manager.status(), (INITIAL_DIGEST, 3))

    def test_matching_expectation_still_reads_the_file(self) -> None:
        os.unlink(self.path)
        with self.assertRaises(ScopePolicyReloadError) as caught:
            self.manager.reload(expected=INITIAL_DIGEST)
        self.assertEqual(caught.exception.kind, "unavailable")

    def test_conditional_reload_records_nothing_on_conflict(self) -> None:
        recorded: list[tuple[str, int]] = []
        with self.assertRaises(ScopePolicyReloadError):
            self.manager.reload(recorded.append, STALE_DIGEST)
        self.assertEqual(recorded, [])

    def test_manager_without_a_path_reports_status_unavailable(self) -> None:
        with self.assertRaises(ScopePolicyReloadError) as caught:
            ScopePolicyManager(None, {}).status()
        self.assertEqual(caught.exception.kind, "unavailable")


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

    def status(self, path: str = STATUS_PATH, token: str | None = ADMIN_TOKEN):
        return self.request("GET", path, token=token)

    def reload(self, body: object = {}, token: str | None = ADMIN_TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    def audit_events_count(self) -> int:
        status, payload, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        return payload["eventsCount"]

    # -- status success contract --

    def test_status_returns_exactly_the_three_contracted_fields(self) -> None:
        status, payload, raw = self.status()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "policyDigest", "tokens"})
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["tokens"], 3)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)
        # Field order is fixed and there is no trailing line terminator.
        self.assertEqual(
            raw.decode("utf-8"),
            f'{{"status":"active","policyDigest":"{INITIAL_DIGEST}","tokens":3}}',
        )

    def test_status_does_not_re_read_the_configured_file(self) -> None:
        self.write_policy(b'{"swapped":["admin"]}')
        status, payload, _ = self.status()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)
        self.assertEqual(payload["tokens"], 3)
        os.unlink(self.policy_path)
        status, payload, _ = self.status()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], INITIAL_DIGEST)

    def test_status_follows_committed_reloads(self) -> None:
        content = b'{ "admin-token" : [ "admin" ] }'
        self.write_policy(content)
        status, payload, _ = self.reload()
        self.assertEqual(status, 200)
        status, payload, _ = self.status()
        self.assertEqual(status, 200)
        self.assertEqual(payload["policyDigest"], hashlib.sha256(content).hexdigest())
        self.assertEqual(payload["tokens"], 1)
        # The reader token no longer exists under the new revision.
        self.assertEqual(self.status(token=READ_TOKEN)[0], 401)

    def test_status_touches_neither_the_data_file_nor_the_audit_history(self) -> None:
        with open(self.data_path, "rb") as handle:
            data_before = handle.read()
        events_before = self.audit_events_count()
        self.assertEqual(self.status()[0], 200)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), data_before)
        self.assertEqual(self.audit_events_count(), events_before)

    # -- status request precedence --

    def test_status_rejects_any_query_parameter(self) -> None:
        for query in ("?x", "?x=", "?x=1", "?x=1&y=2", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload, _ = self.status(path=STATUS_PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_status_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/status/",
            "/v1/admin/scope-policy/status/extra",
            "/v1/admin/scopepolicy/status",
            "/admin/scope-policy/status",
        ):
            with self.subTest(path=path):
                status, payload, _ = self.status(path=path + "?bogus=1")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_status_requires_authentication_and_the_admin_scope(self) -> None:
        status, payload, _ = self.status(token=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload, _ = self.status(token="unknown")
        self.assertEqual(status, 401)
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                status, payload, _ = self.status(token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})

    def test_status_is_404_in_single_token_and_anonymous_modes(self) -> None:
        for server in (
            SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler, auth_token="legacy-token"
            ),
            SemanticStateServer(("127.0.0.1", 0), RequestHandler),
        ):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                for token in (None, "legacy-token"):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    headers = {}
                    if token is not None:
                        headers["Authorization"] = f"Bearer {token}"
                    conn.request("GET", STATUS_PATH, headers=headers)
                    response = conn.getresponse()
                    payload = json.loads(response.read().decode("utf-8"))
                    conn.close()
                    if token is None and server.auth_token is not None:
                        # Authentication still precedes the mode gate.
                        self.assertEqual(response.status, 401)
                    else:
                        self.assertEqual(response.status, 404)
                        self.assertEqual(payload, {"error": "not_found"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_other_methods_on_the_status_path_are_not_published(self) -> None:
        status, _, _ = self.request("POST", STATUS_PATH, body={}, token=ADMIN_TOKEN)
        self.assertEqual(status, 404)

    # -- conditional reload --

    def test_conditional_reload_with_the_live_digest_succeeds(self) -> None:
        content = json.dumps({"new-admin": ["read", "write", "admin"]}).encode("utf-8")
        self.write_policy(content)
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "policyDigest", "tokens"})
        self.assertEqual(payload["status"], "reloaded")
        self.assertEqual(payload["policyDigest"], hashlib.sha256(content).hexdigest())
        self.assertEqual(payload["tokens"], 1)
        # The status query observes the committed revision.
        status, current, _ = self.status(token="new-admin")
        self.assertEqual(status, 200)
        self.assertEqual(current["policyDigest"], payload["policyDigest"])

    def test_stale_expectation_is_409_without_read_swapping_or_recording(self) -> None:
        # The file holds invalid content: a stale expectation must still be
        # the state conflict, proving the file was never read.
        self.write_policy(b'{"t":["nope"]}')
        events_before = self.audit_events_count()
        with open(self.data_path, "rb") as handle:
            data_before = handle.read()
        status, payload, _ = self.reload(body={"expectedPolicyDigest": STALE_DIGEST})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})
        # The old boundary stays fully in force and nothing was recorded.
        self.assertEqual(self.request("GET", METRICS_PATH, token=READ_TOKEN)[0], 200)
        self.assertEqual(self.audit_events_count(), events_before)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), data_before)
        status, current, _ = self.status()
        self.assertEqual(current["policyDigest"], INITIAL_DIGEST)

    def test_conditional_reload_after_a_success_uses_the_new_digest(self) -> None:
        content = json.dumps(
            {"new-admin": ["admin"], ADMIN_TOKEN: ["read", "write", "admin"]}
        ).encode("utf-8")
        self.write_policy(content)
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 200)
        new_digest = payload["policyDigest"]
        # The old expectation is now stale; the new digest matches.
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "policy_state_conflict"})
        status, payload, _ = self.reload(body={"expectedPolicyDigest": new_digest})
        self.assertEqual(status, 200)

    def test_matching_expectation_still_validates_the_file(self) -> None:
        self.write_policy(b'{"t":["nope"]}')
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "policy_conflict"})
        os.unlink(self.policy_path)
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "policy_unavailable"})

    def test_unconditional_reload_stays_unconditional(self) -> None:
        # {} never compares digests, even with a changed file.
        self.write_policy(json.dumps({"other-admin": ["admin"]}).encode("utf-8"))
        status, payload, _ = self.reload(body={})
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens"], 1)

    def test_invalid_conditional_bodies_are_400(self) -> None:
        for body in (
            {"expectedPolicyDigest": ""},
            {"expectedPolicyDigest": None},
            {"expectedPolicyDigest": 0},
            {"expectedPolicyDigest": INITIAL_DIGEST.upper()},
            {"expectedPolicyDigest": INITIAL_DIGEST[:-1]},
            {"expectedPolicyDigest": INITIAL_DIGEST + "0"},
            {"expectedPolicyDigest": INITIAL_DIGEST, "x": 1},
            {"policyDigest": INITIAL_DIGEST},
            {"a": 1},
            [],
            0,
            True,
        ):
            with self.subTest(body=body):
                status, payload, _ = self.reload(body=body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conditional_reload_records_exactly_one_event_on_success(self) -> None:
        events_before = self.audit_events_count()
        content = json.dumps({"event-admin": ["admin"]}).encode("utf-8")
        self.write_policy(content)
        status, payload, _ = self.reload(body={"expectedPolicyDigest": INITIAL_DIGEST})
        self.assertEqual(status, 200)
        status, audit, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token="event-admin"
        )
        self.assertEqual(status, 200)
        self.assertEqual(audit["eventsCount"], events_before + 1)
        event = audit["events"][-1]
        self.assertEqual(event["digest"], payload["policyDigest"])
        self.assertEqual(event["tokens"], 1)
        self.assertEqual(set(event), {"sequence", "digest", "tokens"})

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


if __name__ == "__main__":
    unittest.main()
