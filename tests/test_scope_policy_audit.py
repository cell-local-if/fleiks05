"""Tests for the scope-policy change audit.

``GET /v1/admin/scope-policy/audit`` is the read-only, admin-only view of
the committed policy-change history: every successful hot reload appends
one event (sequence, raw-byte SHA-256 digest, token count) in the same
atomic commit as the policy swap. The tests cover the required-parameter
paging contract, the route-shape/mode/scope gates and their priorities,
the full-history digest and count, the atomicity of the reload commit
(including the 500 internal_error durable-failure path), ``--data-file``
persistence and restart recovery, and the empty Content-Length priority
on the reload endpoint.
"""

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    PersistenceError,
    RequestHandler,
    ScopePolicyManager,
    SemanticStateServer,
    StateStore,
    load_data_file_scope_policy_events,
    load_scope_policy,
    parse_scope_policy_audit_query,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
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

EMPTY_HISTORY_DIGEST = hashlib.sha256(b"[]").hexdigest()
OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def events_digest_input(events: list) -> bytes:
    """The canonical digest input over a committed event history."""
    parts = ["["]
    for index, event in enumerate(events):
        if index:
            parts.append(",")
        parts.append(
            '{"sequence":%d,"policyDigest":"%s","tokens":%d}'
            % (event["sequence"], event["policyDigest"], event["tokens"])
        )
    parts.append("]")
    return "".join(parts).encode("utf-8")


class ParseScopePolicyAuditQueryTests(unittest.TestCase):
    def test_both_parameters_are_required(self) -> None:
        for query in ("", "after=0", "limit=10", "x=1&after=0"):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_query(query))
        self.assertEqual(parse_scope_policy_audit_query("after=0&limit=1"), (0, 1))

    def test_malformed_values_are_rejected(self) -> None:
        for query in (
            "after=-1&limit=1",
            "after=+1&limit=1",
            "after=&limit=1",
            "after= &limit=1",
            "after=1 &limit=1",
            "after=1.0&limit=1",
            "after=²&limit=1",
            "after=0&limit=",
            "after=0&limit=0",
            "after=0&limit=101",
            "after=0&limit=-5",
            "after=0&limit=1&after=0",
            "after=0&limit=1&limit=2",
            "after=0&limit=1&x=2",
            "after=0&limit=1&x",
            "after=0&limit=1&=2",
        ):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_query(query))

    def test_boundaries_are_accepted(self) -> None:
        self.assertEqual(parse_scope_policy_audit_query("after=0&limit=1"), (0, 1))
        self.assertEqual(parse_scope_policy_audit_query("after=17&limit=100"), (17, 100))
        self.assertEqual(parse_scope_policy_audit_query("limit=3&after=4"), (4, 3))


class ScopePolicyAuditHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-audit-http-")
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
        # Every test starts from the initial policy on disk and a clean,
        # empty audit history, regardless of what an earlier test left.
        self.write_policy(INITIAL_BYTES)
        self.server.scope_policy.reload()
        self.server.scope_policy._events = []
        self.server.store._scope_policy_events = []
        self.server.store._persist_locked()
        self.addCleanup(self.reset_history)

    def reset_history(self) -> None:
        self.write_policy(INITIAL_BYTES)
        self.server.scope_policy.reload()
        self.server.scope_policy._events = []
        self.server.store._scope_policy_events = []
        self.server.store._persist_locked()

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

    def audit(self, query: str = "after=0&limit=100", token: str | None = ADMIN_TOKEN):
        return self.request("GET", AUDIT_PATH + "?" + query, token=token)

    def reload(self, body: object = {}, token: str | None = ADMIN_TOKEN):
        return self.request("POST", RELOAD_PATH, body=body, token=token)

    # -- success contract --

    def test_empty_history_reports_the_empty_array_digest(self) -> None:
        status, payload, raw, _ = self.audit()
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {"events", "nextCursor", "hasMore", "algorithm", "digest", "eventsCount"},
        )
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_HISTORY_DIGEST)
        self.assertEqual(payload["eventsCount"], 0)
        # Compact JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"}\n"))
        self.assertNotIn(b" ", raw)

    def test_successful_reloads_append_events_in_order(self) -> None:
        first = b'{ "admin-token": ["read","write","admin"] }'
        self.write_policy(first)
        self.assertEqual(self.reload()[0], 200)
        second = json.dumps({ADMIN_TOKEN: ["admin"], READ_TOKEN: ["read"]}).encode()
        self.write_policy(second)
        self.assertEqual(self.reload()[0], 200)

        status, payload, _, _ = self.audit()
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(
            payload["events"],
            [
                {
                    "sequence": 1,
                    "policyDigest": hashlib.sha256(first).hexdigest(),
                    "tokens": 1,
                },
                {
                    "sequence": 2,
                    "policyDigest": hashlib.sha256(second).hexdigest(),
                    "tokens": 2,
                },
            ],
        )
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(
            payload["digest"],
            hashlib.sha256(events_digest_input(payload["events"])).hexdigest(),
        )

    def test_digest_and_count_cover_the_whole_history_on_every_page(self) -> None:
        policies = []
        for index in range(3):
            content = json.dumps(
                {ADMIN_TOKEN: ["read", "write", "admin"], f"extra-{index}": ["read"]}
            ).encode()
            policies.append(content)
            self.write_policy(content)
            self.assertEqual(self.reload()[0], 200)
        expected_events = [
            {
                "sequence": index + 1,
                "policyDigest": hashlib.sha256(content).hexdigest(),
                "tokens": 2,
            }
            for index, content in enumerate(policies)
        ]
        expected_digest = hashlib.sha256(
            events_digest_input(expected_events)
        ).hexdigest()

        seen = []
        after = 0
        while True:
            status, payload, _, _ = self.audit(f"after={after}&limit=2")
            self.assertEqual(status, 200)
            self.assertEqual(payload["eventsCount"], 3)
            self.assertEqual(payload["digest"], expected_digest)
            seen.extend(payload["events"])
            self.assertEqual(payload["nextCursor"], after + len(payload["events"]))
            self.assertIs(payload["hasMore"], payload["nextCursor"] < 3)
            if not payload["hasMore"]:
                break
            after = payload["nextCursor"]
        self.assertEqual(seen, expected_events)
        # An after equal to the history length is a valid empty page.
        status, payload, _, _ = self.audit("after=3&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["eventsCount"], 3)
        self.assertEqual(payload["digest"], expected_digest)

    # -- query validation --

    def test_query_parameter_violations_are_400(self) -> None:
        for query in (
            "",
            "after=0",
            "limit=10",
            "after=-1&limit=1",
            "after=+1&limit=1",
            "after=&limit=1",
            "after=0&limit=",
            "after=0&limit=0",
            "after=0&limit=101",
            "after=0&limit=1&after=2",
            "after=0&limit=1&unknown=1",
            "after=0&limit=1&unknown",
            "after=1&limit=1",  # past the empty history
        ):
            with self.subTest(query=query):
                status, payload, _, _ = self.audit(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_after_past_the_history_is_400(self) -> None:
        self.write_policy(json.dumps({ADMIN_TOKEN: ["admin"]}).encode())
        self.assertEqual(self.reload()[0], 200)
        self.assertEqual(self.audit("after=1&limit=1")[0], 200)
        status, payload, _, _ = self.audit("after=2&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- path shape precedes the query check --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/audit/",
            "/v1/admin/scope-policy/audit/extra",
            "/v1/admin/scope-policy/reload",
            "/v1/admin/scopepolicy/audit",
            "/admin/scope-policy/audit",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request(
                    "GET", path + "?bogus", token=ADMIN_TOKEN
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_other_methods_on_the_path_are_not_published(self) -> None:
        status, _, _, _ = self.request(
            "POST", AUDIT_PATH, body={}, token=ADMIN_TOKEN
        )
        self.assertEqual(status, 404)

    # -- authentication and authorization --

    def test_authentication_failures_are_401_with_the_bearer_challenge(self) -> None:
        for headers in (
            [],
            [("Authorization", "Bearer unknown")],
            [("Authorization", ADMIN_TOKEN)],
            [("Authorization", f"Bearer {ADMIN_TOKEN} "),],
            [("Authorization", f"Bearer {ADMIN_TOKEN}"), ("Authorization", f"Bearer {ADMIN_TOKEN}")],
        ):
            with self.subTest(headers=headers):
                status, payload, response_headers = self.raw_request(
                    "GET", AUDIT_PATH + "?after=0&limit=1", headers
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(response_headers.get("WWW-Authenticate"), "Bearer")

    def test_tokens_without_the_admin_scope_are_403_without_a_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                status, payload, _, response_headers = self.audit(token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", response_headers)

    def test_admin_scope_covers_the_audit(self) -> None:
        self.assertEqual(self.audit()[0], 200)

    # -- the endpoint exists only in scope-policy mode --

    def test_endpoint_is_404_in_single_token_mode_but_401_without_a_token(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="legacy-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            def call(token: str | None) -> int:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                headers = {}
                if token is not None:
                    headers["Authorization"] = f"Bearer {token}"
                conn.request("GET", AUDIT_PATH + "?after=0&limit=1", headers=headers)
                response = conn.getresponse()
                response.read()
                conn.close()
                return response.status

            self.assertEqual(call(None), 401)
            self.assertEqual(call("wrong-token"), 401)
            self.assertEqual(call("legacy-token"), 404)
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
            conn.request("GET", AUDIT_PATH + "?after=0&limit=1")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # -- failed reloads record no event --

    def test_failed_reloads_record_no_event(self) -> None:
        self.write_policy(b'{"t":["nope"]}')
        self.assertEqual(self.reload()[0], 409)
        os.unlink(self.policy_path)
        self.assertEqual(self.reload()[0], 503)
        self.write_policy(INITIAL_BYTES)
        # Authentication and permission rejections record nothing either.
        self.assertEqual(self.reload(token=None)[0], 401)
        self.assertEqual(self.reload(token=READ_TOKEN)[0], 403)
        status, payload, _, _ = self.audit()
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], EMPTY_HISTORY_DIGEST)

    # -- the reload commit is atomic with its event --

    def test_durable_commit_failure_is_500_and_keeps_the_old_policy_and_history(self) -> None:
        self.write_policy(json.dumps({ADMIN_TOKEN: ["admin"]}).encode())
        self.assertEqual(self.reload()[0], 200)
        _, before, _, _ = self.audit()
        self.assertEqual(before["eventsCount"], 1)

        replacement = json.dumps(
            {ADMIN_TOKEN: ["read", "write", "admin"], "new-reader": ["read"]}
        ).encode()
        self.write_policy(replacement)
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload, _, _ = self.reload()
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "internal_error"})
        # The old policy stays fully in force: the file-only new token does
        # not exist and the old admin still authenticates.
        self.assertEqual(self.request("GET", METRICS_PATH, token="new-reader")[0], 401)
        self.assertEqual(self.request("GET", METRICS_PATH, token=ADMIN_TOKEN)[0], 200)
        # The old history is untouched.
        _, after, _, _ = self.audit()
        self.assertEqual(after, before)
        # The failure is recoverable: the retried reload commits once.
        self.assertEqual(self.reload()[0], 200)
        _, payload, _, _ = self.audit()
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(
            payload["events"][1]["policyDigest"],
            hashlib.sha256(replacement).hexdigest(),
        )

    # -- persistence and recovery --

    def test_events_are_persisted_and_recovered_across_a_restart(self) -> None:
        first = json.dumps({ADMIN_TOKEN: ["read", "write", "admin"]}).encode()
        self.write_policy(first)
        self.assertEqual(self.reload()[0], 200)
        second = json.dumps({ADMIN_TOKEN: ["admin"], READ_TOKEN: ["read"]}).encode()
        self.write_policy(second)
        self.assertEqual(self.reload()[0], 200)
        _, before, _, _ = self.audit()

        stored = load_data_file_scope_policy_events(self.data_path)
        self.assertEqual(stored, before["events"])

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
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "GET",
                    AUDIT_PATH + "?after=0&limit=100",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(payload, before)
                # The sequence continues after the recovered history.
                third = json.dumps({ADMIN_TOKEN: ["admin"]}).encode()
                self.write_policy(third)
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST",
                    RELOAD_PATH,
                    body="{}",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                )
                self.assertEqual(conn.getresponse().status, 200)
                conn.close()
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "GET",
                    AUDIT_PATH + "?after=2&limit=1",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                self.assertEqual(
                    payload["events"],
                    [
                        {
                            "sequence": 3,
                            "policyDigest": hashlib.sha256(third).hexdigest(),
                            "tokens": 1,
                        }
                    ],
                )
                self.assertEqual(payload["eventsCount"], 3)
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

    def test_data_file_without_the_section_recovers_an_empty_history(self) -> None:
        # A data file written before policy-change auditing existed carries
        # no scopePolicyEvents section and recovers with an empty history.
        document = {
            "version": 1,
            "operations": [],
            "checkpoints": {},
            "policies": [],
            "transactions": [],
            "acks": [],
        }
        with open(self.data_path, "wb") as handle:
            handle.write(json.dumps(document).encode("utf-8"))
        self.assertEqual(load_data_file_scope_policy_events(self.data_path), [])
        server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=self.data_path,
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "GET",
                AUDIT_PATH + "?after=0&limit=1",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["events"], [])
            self.assertEqual(payload["eventsCount"], 0)
            self.assertEqual(payload["digest"], EMPTY_HISTORY_DIGEST)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_corrupt_event_sections_fail_startup(self) -> None:
        base = {"version": 1, "operations": []}
        for events in (
            [{"sequence": 0, "policyDigest": "0" * 64, "tokens": 1}],
            [{"sequence": 2, "policyDigest": "0" * 64, "tokens": 1}],
            [{"sequence": 1, "policyDigest": "0" * 63, "tokens": 1}],
            [{"sequence": 1, "policyDigest": "0" * 64, "tokens": -1}],
            [{"sequence": 1, "policyDigest": "0" * 64, "tokens": 1, "x": 1}],
            [{"sequence": True, "policyDigest": "0" * 64, "tokens": 1}],
            [{"sequence": 1, "policyDigest": "0" * 64}],
            {"sequence": 1},
        ):
            with self.subTest(events=events):
                document = dict(base, scopePolicyEvents=events)
                with open(self.data_path, "wb") as handle:
                    handle.write(json.dumps(document).encode("utf-8"))
                with self.assertRaises(PersistenceError):
                    StateStore(data_file=self.data_path)
        # Restore a valid empty store for the rest of the class.
        self.write_policy(INITIAL_BYTES)
        with open(self.data_path, "wb") as handle:
            handle.write(json.dumps(base).encode("utf-8"))

    # -- the audit query is strictly read-only --

    def test_queries_leave_state_file_and_directory_untouched(self) -> None:
        self.write_policy(json.dumps({ADMIN_TOKEN: ["read", "write", "admin"]}).encode())
        self.assertEqual(self.reload()[0], 200)
        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        _, before, _, _ = self.audit()

        self.assertEqual(self.audit("after=0&limit=1")[0], 200)
        self.assertEqual(self.audit("after=1&limit=1")[0], 200)
        self.assertEqual(self.audit("after=99&limit=1")[0], 400)
        self.assertEqual(self.audit("")[0], 400)

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        _, after, _, _ = self.audit()
        self.assertEqual(after, before)

    # -- empty Content-Length keeps its priority on the reload endpoint --

    def test_empty_content_length_is_400_before_authentication(self) -> None:
        for auth in (None, f"Bearer {ADMIN_TOKEN}", "Bearer unknown"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", "")]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_content_length_does_not_read_the_body(self) -> None:
        # An empty declaration is rejected without waiting for body bytes.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", RELOAD_PATH)
        conn.putheader("Content-Length", "")
        conn.putheader("Authorization", f"Bearer {ADMIN_TOKEN}")
        conn.endheaders()  # no body is ever sent
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The policy and the history are untouched.
        _, payload, _, _ = self.audit()
        self.assertEqual(payload["eventsCount"], 0)

    def test_over_limit_declaration_is_413_before_authentication(self) -> None:
        for auth in (None, f"Bearer {ADMIN_TOKEN}"):
            with self.subTest(auth=auth):
                headers = [("Content-Length", OVER_LIMIT)]
                if auth is not None:
                    headers.append(("Authorization", auth))
                status, payload, _ = self.raw_request(
                    "POST", RELOAD_PATH, headers, b"junk"
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})


if __name__ == "__main__":
    unittest.main()
