"""Tests for the scope-policy change audit.

``GET /v1/admin/scope-policy/audit`` pages the history of successful
scope-policy hot reloads. The tests cover the request precedence chain
(404 path shape, 401 authentication with a Bearer challenge, 403 without
one, the scope-policy-only mode gate, 400 query validation), the paging
contract, the full-history SHA-256 digest, atomic commit of a reload
together with its event (a durable failure is 500 and keeps the old
policy and history), ``--data-file`` persistence and recovery, and the
read-only/no-temp-file guarantees.
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
    SemanticStateServer,
    StateStore,
    load_data_file_policy_events,
    load_scope_policy,
    parse_scope_policy_audit_query,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
AUDIT_PATH = "/v1/admin/scope-policy/audit"
RELOAD_PATH = "/v1/admin/scope-policy/reload"
METRICS_PATH = "/v1/metrics"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")


class ParseScopePolicyAuditQueryTests(unittest.TestCase):
    def test_requires_both_after_and_limit(self) -> None:
        for query in ("", "after=0", "limit=1"):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_query(query))

    def test_accepts_plain_ascii_decimal_pairs(self) -> None:
        self.assertEqual(parse_scope_policy_audit_query("after=0&limit=1"), (0, 1))
        self.assertEqual(parse_scope_policy_audit_query("after=00&limit=0100"), (0, 100))
        self.assertEqual(parse_scope_policy_audit_query("after=12&limit=7"), (12, 7))

    def test_rejects_bad_values(self) -> None:
        bad = [
            "after=&limit=1",
            "after=0&limit=",
            "after=-1&limit=1",
            "after=0&limit=-1",
            "after=%200&limit=1",
            "after=0&limit=1%20",
            "after=+0&limit=1",
            "after=0&limit=+1",
            "after=1.0&limit=1",
            "after=0&limit=1.0",
            "after=%C2%B2&limit=1",
            "after=0&limit=0",
            "after=0&limit=101",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_query(query))

    def test_rejects_unknown_and_repeated_parameters(self) -> None:
        bad = [
            "after=0&limit=1&x=1",
            "after=0&limit=1&after=2",
            "after=0&after=1&limit=1",
            "after=0&limit=1&limit=2",
            "x=0&limit=1",
            "after=0&x=1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_query(query))


class PolicyEventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-events-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def digest_input(self, events: list[dict]) -> bytes:
        return json.dumps(
            [
                {"sequence": e["sequence"], "digest": e["digest"], "tokens": e["tokens"]}
                for e in events
            ],
            separators=(",", ":"),
        ).encode("utf-8")

    def test_empty_history_pages_and_hashes_the_empty_array(self) -> None:
        store = StateStore()
        status, payload = store.get_policy_events(0, 100)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(
            payload["digest"], hashlib.sha256(b"[]").hexdigest()
        )

    def test_events_are_sequential_and_digest_covers_the_whole_history(self) -> None:
        store = StateStore()
        first = store.record_policy_reload("a" * 64, 3)
        second = store.record_policy_reload("b" * 64, 1)
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        expected = [
            {"sequence": 1, "digest": "a" * 64, "tokens": 3},
            {"sequence": 2, "digest": "b" * 64, "tokens": 1},
        ]
        status, payload = store.get_policy_events(0, 1)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], expected[:1])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertTrue(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(
            payload["digest"], hashlib.sha256(self.digest_input(expected)).hexdigest()
        )
        # The full-history digest is identical on every page.
        status, page_two = store.get_policy_events(1, 1)
        self.assertEqual(page_two["events"], expected[1:])
        self.assertEqual(page_two["nextCursor"], 2)
        self.assertFalse(page_two["hasMore"])
        self.assertEqual(page_two["digest"], payload["digest"])

    def test_after_equal_to_count_is_an_empty_tail_past_count_is_value_error(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        status, payload = store.get_policy_events(1, 10)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        with self.assertRaises(ValueError):
            store.get_policy_events(2, 10)

    def test_persistence_failure_rolls_the_event_back(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 1)
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            with self.assertRaises(PersistenceError):
                store.record_policy_reload("b" * 64, 2)
        self.assertEqual([e["digest"] for e in store._policy_events], ["a" * 64])

    def test_events_persist_and_recover_in_order(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 3)
        store.record_policy_reload("b" * 64, 0)
        recovered = StateStore(data_file=data_file)
        self.assertEqual(
            recovered._policy_events,
            [
                {"sequence": 1, "digest": "a" * 64, "tokens": 3},
                {"sequence": 2, "digest": "b" * 64, "tokens": 0},
            ],
        )
        self.assertEqual(
            load_data_file_policy_events(data_file), recovered._policy_events
        )

    def test_old_file_without_policy_events_recovers_empty(self) -> None:
        data_file = os.path.join(self.tmpdir, "old.json")
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=data_file)
        self.assertEqual(store._policy_events, [])
        # The supplemented format is only rewritten on the next commit.
        status, payload = store.get_policy_events(0, 100)
        self.assertEqual(payload["eventsCount"], 0)


class CorruptPolicyEventsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-events-bad-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = os.path.join(self.tmpdir, "state.json")

    def write(self, document: dict) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def test_corrupt_policy_events_make_startup_fail(self) -> None:
        good = {"sequence": 1, "digest": "a" * 64, "tokens": 1}
        cases = [
            {"version": 1, "operations": [], "policyEvents": {}},
            {"version": 1, "operations": [], "policyEvents": [{}]},
            {"version": 1, "operations": [], "policyEvents": [{"sequence": 0, "digest": "a" * 64, "tokens": 1}]},
            {"version": 1, "operations": [], "policyEvents": [{"sequence": 2, "digest": "a" * 64, "tokens": 1}]},
            {"version": 1, "operations": [], "policyEvents": [dict(good, digest="A" * 64)]},
            {"version": 1, "operations": [], "policyEvents": [dict(good, digest="a" * 63)]},
            {"version": 1, "operations": [], "policyEvents": [dict(good, tokens=-1)]},
            {"version": 1, "operations": [], "policyEvents": [dict(good, tokens=True)]},
        ]
        for document in cases:
            with self.subTest(document=document):
                self.write(document)
                with self.assertRaises(PersistenceError):
                    StateStore(data_file=self.data_file)


class ScopePolicyAuditHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-audit-http-")
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
        # Reset business state and the policy history for every test,
        # keeping the same data path (but a fresh empty file) so reload
        # durability and recovery are exercised for real.
        if os.path.lexists(self.data_path):
            os.unlink(self.data_path)
        self.server.store = StateStore(data_file=self.data_path)
        self.write_policy(INITIAL_BYTES)
        self.server.scope_policy.reload()

    def write_policy(self, content: bytes) -> None:
        if os.path.isdir(self.policy_path):
            shutil.rmtree(self.policy_path)
        with open(self.policy_path, "wb") as handle:
            handle.write(content)

    def request(
        self, method: str, path: str, token: str | None = ADMIN_TOKEN, body: object = None
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

    def audit(self, query: str, token: str | None = ADMIN_TOKEN):
        return self.request("GET", AUDIT_PATH + query, token=token)

    def reload(self, content: bytes, token: str | None = ADMIN_TOKEN) -> str:
        self.write_policy(content)
        status, payload, _, _ = self.request(
            "POST", RELOAD_PATH, token=token, body={}
        )
        self.assertEqual(status, 200, payload)
        return payload["policyDigest"]

    # -- success contract on an empty history --

    def test_empty_history_is_compact_ordered_json_with_a_single_newline(self) -> None:
        status, payload, raw, headers = self.audit("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            ["events", "nextCursor", "hasMore", "algorithm", "digest", "eventsCount"],
        )
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(
            raw,
            b'{"events":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + payload["digest"].encode() + b'","eventsCount":0}\n',
        )
        self.assertTrue(raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    # -- events and the full-history digest --

    def test_events_follow_successful_reloads_and_digest_covers_full_history(self) -> None:
        first_bytes = b'{ "admin-token" : [ "admin" ] }'
        first_digest = self.reload(first_bytes)
        second_bytes = b'{"z":["admin"],"r":["read"]}'
        second_digest = self.reload(second_bytes)

        status, payload, _, _ = self.audit("?after=0&limit=100", token="z")
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(
            payload["events"],
            [
                {"sequence": 1, "digest": first_digest, "tokens": 1},
                {"sequence": 2, "digest": second_digest, "tokens": 2},
            ],
        )
        self.assertEqual(first_digest, hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(second_digest, hashlib.sha256(second_bytes).hexdigest())
        digest_input = json.dumps(
            [
                {"sequence": 1, "digest": first_digest, "tokens": 1},
                {"sequence": 2, "digest": second_digest, "tokens": 2},
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(payload["digest"], hashlib.sha256(digest_input).hexdigest())

    def test_digest_and_count_cover_the_whole_history_on_every_page(self) -> None:
        # Keep a stable admin credential across reloads while changing the
        # raw bytes and the entry count each time.
        digests = [
            self.reload(
                json.dumps(
                    {"admin-token": ["admin"], **{f"t{i}": ["read"] for i in range(k + 1)}}
                ).encode("utf-8")
            )
            for k in range(3)
        ]
        pages = []
        for after in range(4):
            status, payload, _, _ = self.audit(f"?after={after}&limit=1")
            self.assertEqual(status, 200)
            self.assertEqual(payload["eventsCount"], 3)
            pages.append(payload)
        full_digest = pages[0]["digest"]
        self.assertTrue(all(page["digest"] == full_digest for page in pages))
        self.assertEqual([p["events"][0]["sequence"] for p in pages[:3]], [1, 2, 3])
        self.assertEqual(pages[3]["events"], [])
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])
        self.assertEqual([p["events"][0]["digest"] for p in pages[:3]], digests)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.reload(b'{"x":["admin"]}')
        status, payload, _, _ = self.audit("?after=1&limit=10", token="x")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 1)

    # -- query validation --

    def test_invalid_queries_are_400(self) -> None:
        bad_queries = [
            "",
            "?after=0",
            "?limit=1",
            "?after=&limit=1",
            "?after=0&limit=",
            "?after=-1&limit=1",
            "?after=0&limit=-1",
            "?after=%20&limit=1",
            "?after=0&limit=1%20",
            "?after=1.0&limit=1",
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=1&x=1",
            "?after=0&after=1&limit=1",
            "?after=0&limit=1&limit=2",
            "?x=0&limit=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.audit(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_the_event_count_is_400(self) -> None:
        status, payload, _, _ = self.audit("?after=1&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- route shape precedes the query check --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            "/v1/admin/scope-policy/audit/",
            "/v1/admin/scope-policy/audit/extra",
            "/v1/admin/scope-policy",
            "/v1/admin/scope-policy/other",
            "/v1/admin/scopepolicy/audit",
            "/admin/scope-policy/audit",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request(
                    "GET", path + "?bogus=1&after=0", token=ADMIN_TOKEN
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    # -- authentication and scope --

    def test_missing_or_bad_credential_is_401_with_a_bearer_challenge(self) -> None:
        conn_factory = lambda: http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        # Missing header.
        conn = conn_factory()
        conn.request("GET", AUDIT_PATH + "?after=0&limit=1")
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        # Token mismatch.
        status, _, _, headers = self.audit("?after=0&limit=1", token="nobody")
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Malformed header value.
        conn = conn_factory()
        conn.putrequest("GET", AUDIT_PATH + "?after=0&limit=1")
        conn.putheader("Authorization", "Token admin-token")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        # Duplicated header.
        conn = conn_factory()
        conn.putrequest("GET", AUDIT_PATH + "?after=0&limit=1")
        conn.putheader("Authorization", f"Bearer {ADMIN_TOKEN}")
        conn.putheader("Authorization", f"Bearer {ADMIN_TOKEN}")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_authenticated_without_admin_is_403_without_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                status, payload, _, headers = self.audit("?after=0&limit=1", token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        # A bad query for a read-only token is still 403, never downgraded
        # to 400.
        status, _, _, _ = self.audit("?after=nope&limit=1", token=READ_TOKEN)
        self.assertEqual(status, 403)

    # -- the entry is published only in scope-policy mode --

    def test_entry_is_404_in_single_token_and_anonymous_modes(self) -> None:
        for mode, kwargs, creds in [
            ("single", {"auth_token": "legacy-token"}, [None, "Bearer wrong", "Bearer legacy-token"]),
            ("anonymous", {}, [None]),
        ]:
            server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                for cred in creds:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    headers = {}
                    if cred is not None:
                        headers["Authorization"] = cred
                    conn.request("GET", AUDIT_PATH + "?after=0&limit=1", headers=headers)
                    response = conn.getresponse()
                    payload = json.loads(response.read().decode("utf-8"))
                    conn.close()
                    if mode == "single" and cred != "Bearer legacy-token":
                        self.assertEqual(response.status, 401, (mode, cred))
                        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
                    else:
                        self.assertEqual(response.status, 404, (mode, cred))
                        self.assertEqual(payload, {"error": "not_found"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    # -- failed reloads and rejected requests leave no event --

    def test_failed_and_rejected_reloads_never_enter_the_history(self) -> None:
        self.write_policy(b'{"t":["bogus"]}')
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 409)
        os.unlink(self.policy_path)
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 503)
        self.write_policy(INITIAL_BYTES)
        status, _, _, _ = self.audit("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(self.server.store._policy_events, [])

    def test_durable_reload_failure_is_500_and_keeps_old_policy_and_history(self) -> None:
        self.reload(b'{"keep":["admin"]}')
        self.write_policy(b'{"keep":["admin"],"next":["read"]}')
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload, _, _ = self.request(
                "POST", RELOAD_PATH, token="keep", body={}
            )
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "internal_error"})
        # The old policy stays in force: the new token is unknown, the old
        # one still reads, and the history did not gain the failed event.
        self.assertEqual(self.audit("?after=0&limit=100", token="next")[0], 401)
        status, payload, _, _ = self.request("GET", METRICS_PATH, token="keep")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.server.store._policy_events), 1)

    def test_rejected_reload_creates_no_temp_file_or_event(self) -> None:
        # A 400 (empty Content-Length) and 413 (over-limit) are answered
        # before authentication and the reload, and leave no temp file.
        listing_before = sorted(os.listdir(self.tmpdir))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", RELOAD_PATH)
        conn.putheader("Content-Length", "")
        conn.endheaders(b"{}")
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", RELOAD_PATH)
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.putheader("Authorization", f"Bearer {ADMIN_TOKEN}")
        conn.endheaders(b"junk")
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        conn.close()
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        self.assertEqual(self.server.store._policy_events, [])

    # -- the audit query is strictly read-only --

    def test_audit_query_persists_nothing(self) -> None:
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in ("?after=0&limit=100", "?after=0&limit=1&x=1"):
            self.audit(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
