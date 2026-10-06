"""Tests for the scope-policy token-change audit.

``GET /v1/admin/scope-policy/audit/changes`` pages the same history of
successful scope-policy hot reloads as the plain change audit, but each
event carries the reload's token-change detail — the ascending,
deduplicated SHA-256 fingerprints of the tokens added, removed, and
scope-changed — instead of the entry count, so a successful
privilege change can be traced without ever exposing a raw token, a
scope, or the policy file's content. The tests cover the request
precedence chain (404 path shape, 401 authentication with a Bearer
challenge, 403 without one, the scope-policy-only mode gate, 400 query
validation), the paging contract, the full-history digest, the
fingerprint diff semantics, the null detail of events that cannot be
reconstructed, ``--data-file`` persistence and recovery (including old
files without the detail), and the read-only guarantee.
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
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file_policy_events,
    load_scope_policy,
    parse_scope_policy_audit_changes_query,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
CHANGES_PATH = "/v1/admin/scope-policy/audit/changes"
RELOAD_PATH = "/v1/admin/scope-policy/reload"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")


def fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ParseScopePolicyAuditChangesQueryTests(unittest.TestCase):
    def test_requires_both_after_and_limit(self) -> None:
        for query in ("", "after=0", "limit=1"):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_changes_query(query))

    def test_accepts_plain_ascii_decimal_pairs(self) -> None:
        self.assertEqual(
            parse_scope_policy_audit_changes_query("after=0&limit=1"), (0, 1)
        )
        self.assertEqual(
            parse_scope_policy_audit_changes_query("after=00&limit=0100"), (0, 100)
        )
        self.assertEqual(
            parse_scope_policy_audit_changes_query("after=12&limit=7"), (12, 7)
        )

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
                self.assertIsNone(parse_scope_policy_audit_changes_query(query))

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
                self.assertIsNone(parse_scope_policy_audit_changes_query(query))


class PolicyEventChangesStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-changes-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def digest_input(self, events: list[dict]) -> bytes:
        return json.dumps(
            [
                {
                    "sequence": e["sequence"],
                    "policyDigest": e["digest"],
                    "tokenChanges": e.get("tokenChanges"),
                }
                for e in events
            ],
            separators=(",", ":"),
        ).encode("utf-8")

    def test_empty_history_pages_and_hashes_the_empty_array(self) -> None:
        store = StateStore()
        status, payload = store.get_policy_event_changes(0, 100)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())

    def test_events_carry_the_detail_and_digest_covers_the_whole_history(
        self,
    ) -> None:
        store = StateStore()
        first_changes = {
            "added": sorted([fingerprint("new-token")]),
            "removed": sorted([fingerprint("old-token")]),
            "changed": [],
        }
        second_changes = {
            "added": [],
            "removed": [],
            "changed": sorted([fingerprint("new-token")]),
        }
        store.record_policy_reload("a" * 64, 3, first_changes)
        store.record_policy_reload("b" * 64, 1, second_changes)
        store.record_policy_reload("c" * 64, 2)
        events = [
            {"sequence": 1, "digest": "a" * 64, "tokens": 3,
             "tokenChanges": first_changes},
            {"sequence": 2, "digest": "b" * 64, "tokens": 1,
             "tokenChanges": second_changes},
            {"sequence": 3, "digest": "c" * 64, "tokens": 2},
        ]
        status, payload = store.get_policy_event_changes(0, 2)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["events"],
            [
                {"sequence": 1, "policyDigest": "a" * 64,
                 "tokenChanges": first_changes},
                {"sequence": 2, "policyDigest": "b" * 64,
                 "tokenChanges": second_changes},
            ],
        )
        self.assertEqual(payload["nextCursor"], 2)
        self.assertTrue(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 3)
        self.assertEqual(
            payload["digest"], hashlib.sha256(self.digest_input(events)).hexdigest()
        )
        # The event recorded without a detail reports null, and the
        # full-history digest is identical on every page.
        status, page_two = store.get_policy_event_changes(2, 2)
        self.assertEqual(
            page_two["events"],
            [{"sequence": 3, "policyDigest": "c" * 64, "tokenChanges": None}],
        )
        self.assertEqual(page_two["nextCursor"], 3)
        self.assertFalse(page_two["hasMore"])
        self.assertEqual(page_two["digest"], payload["digest"])

    def test_after_equal_to_count_is_an_empty_tail_past_count_is_value_error(
        self,
    ) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        status, payload = store.get_policy_event_changes(1, 10)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        with self.assertRaises(ValueError):
            store.get_policy_event_changes(2, 10)

    def test_details_persist_and_recover_in_order(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        changes = {
            "added": sorted([fingerprint("b-token"), fingerprint("a-token")]),
            "removed": [],
            "changed": [fingerprint("c-token")],
        }
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 3, changes)
        store.record_policy_reload("b" * 64, 0)
        recovered = StateStore(data_file=data_file)
        self.assertEqual(
            recovered._policy_events,
            [
                {"sequence": 1, "digest": "a" * 64, "tokens": 3,
                 "tokenChanges": changes},
                {"sequence": 2, "digest": "b" * 64, "tokens": 0},
            ],
        )
        self.assertEqual(
            load_data_file_policy_events(data_file), recovered._policy_events
        )
        # The recovered history pages and hashes identically.
        status, payload = recovered.get_policy_event_changes(0, 100)
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(payload["events"][0]["tokenChanges"], changes)
        self.assertIsNone(payload["events"][1]["tokenChanges"])

    def test_old_file_without_details_recovers_with_null_and_new_reload_records(
        self,
    ) -> None:
        data_file = os.path.join(self.tmpdir, "old.json")
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "operations": [],
                    "policyEvents": [
                        {"sequence": 1, "digest": "a" * 64, "tokens": 2}
                    ],
                },
                handle,
            )
        store = StateStore(data_file=data_file)
        status, payload = store.get_policy_event_changes(0, 100)
        self.assertEqual(
            payload["events"],
            [{"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": None}],
        )
        # A later reload appends its detail; the old event stays null.
        changes = {"added": [fingerprint("t")], "removed": [], "changed": []}
        store.record_policy_reload("b" * 64, 1, changes)
        recovered = StateStore(data_file=data_file)
        status, payload = recovered.get_policy_event_changes(0, 100)
        self.assertEqual(payload["eventsCount"], 2)
        self.assertIsNone(payload["events"][0]["tokenChanges"])
        self.assertEqual(payload["events"][1]["tokenChanges"], changes)

    def test_null_detail_in_file_recovers_as_null(self) -> None:
        data_file = os.path.join(self.tmpdir, "null.json")
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "operations": [],
                    "policyEvents": [
                        {
                            "sequence": 1,
                            "digest": "a" * 64,
                            "tokens": 2,
                            "tokenChanges": None,
                        }
                    ],
                },
                handle,
            )
        store = StateStore(data_file=data_file)
        status, payload = store.get_policy_event_changes(0, 100)
        self.assertEqual(
            payload["events"],
            [{"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": None}],
        )


class CorruptPolicyEventChangesFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-changes-bad-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = os.path.join(self.tmpdir, "state.json")

    def write(self, document: dict) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def test_corrupt_token_changes_make_startup_fail(self) -> None:
        good = {"sequence": 1, "digest": "a" * 64, "tokens": 1}
        good_changes = {"added": [], "removed": [], "changed": []}
        cases = [
            # Not an object.
            dict(good, tokenChanges=[]),
            dict(good, tokenChanges="added"),
            # Missing or extra keys.
            dict(good, tokenChanges={"added": [], "removed": []}),
            dict(good, tokenChanges=dict(good_changes, extra=[])),
            # Lists must contain only 64 lowercase hex fingerprints.
            dict(good, tokenChanges=dict(good_changes, added=["x"])),
            dict(good, tokenChanges=dict(good_changes, removed=["A" * 64])),
            dict(good, tokenChanges=dict(good_changes, changed=["a" * 63])),
            dict(good, tokenChanges=dict(good_changes, added=[1])),
            dict(good, tokenChanges=dict(good_changes, added="ab")),
            # Lists must be ascending and deduplicated.
            dict(good, tokenChanges=dict(good_changes, added=["b" * 64, "a" * 64])),
            dict(good, tokenChanges=dict(good_changes, removed=["a" * 64, "a" * 64])),
        ]
        for changes_document in cases:
            document = {
                "version": 1,
                "operations": [],
                "policyEvents": [changes_document],
            }
            with self.subTest(document=document):
                self.write(document)
                with self.assertRaises(PersistenceError):
                    StateStore(data_file=self.data_file)


class ScopePolicyAuditChangesHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-changes-http-")
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

    def changes(self, query: str, token: str | None = ADMIN_TOKEN):
        return self.request("GET", CHANGES_PATH + query, token=token)

    def reload(self, content: bytes, token: str | None = ADMIN_TOKEN) -> str:
        self.write_policy(content)
        status, payload, _, _ = self.request(
            "POST", RELOAD_PATH, token=token, body={}
        )
        self.assertEqual(status, 200, payload)
        return payload["policyDigest"]

    def digest_input(self, events: list[dict]) -> bytes:
        return json.dumps(events, separators=(",", ":")).encode("utf-8")

    # -- success contract on an empty history --

    def test_empty_history_is_compact_ordered_json_with_a_single_newline(
        self,
    ) -> None:
        status, payload, raw, headers = self.changes("?after=0&limit=100")
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

    # -- the token-change detail --

    def test_reload_records_added_removed_changed_fingerprints(self) -> None:
        first_bytes = b'{"keep":["admin"],"drop":["read"],"swap":["read"]}'
        first_digest = self.reload(first_bytes)
        second_bytes = b'{"keep":["admin"],"swap":["write"],"new":["read"]}'
        second_digest = self.reload(second_bytes, token="keep")

        status, payload, raw, _ = self.changes("?after=0&limit=100", token="keep")
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 2)
        events = [
            {
                "sequence": 1,
                "policyDigest": first_digest,
                "tokenChanges": {
                    "added": sorted(
                        fingerprint(t) for t in ("keep", "drop", "swap")
                    ),
                    "removed": sorted(
                        fingerprint(t)
                        for t in (READ_TOKEN, WRITE_TOKEN, ADMIN_TOKEN)
                    ),
                    "changed": [],
                },
            },
            {
                "sequence": 2,
                "policyDigest": second_digest,
                "tokenChanges": {
                    "added": [fingerprint("new")],
                    "removed": [fingerprint("drop")],
                    "changed": [fingerprint("swap")],
                },
            },
        ]
        self.assertEqual(payload["events"], events)
        # Per-event field order is sequence, policyDigest, tokenChanges;
        # the detail's order is added, removed, changed.
        self.assertEqual(list(payload["events"][0]), ["sequence", "policyDigest", "tokenChanges"])
        self.assertEqual(
            list(payload["events"][0]["tokenChanges"]),
            ["added", "removed", "changed"],
        )
        self.assertEqual(
            payload["digest"],
            hashlib.sha256(self.digest_input(events)).hexdigest(),
        )
        # The response never contains a raw token, a scope name inside the
        # detail, or the policy file's content.
        for token in ("keep", "drop", "swap", "new", READ_TOKEN, WRITE_TOKEN):
            self.assertNotIn(token.encode(), raw)

    def test_identical_reload_records_three_empty_lists(self) -> None:
        self.reload(b'{"same":["admin"]}')
        digest = self.reload(b'{"same":["admin"]}', token="same")
        status, payload, _, _ = self.changes("?after=1&limit=1", token="same")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["events"],
            [
                {
                    "sequence": 2,
                    "policyDigest": digest,
                    "tokenChanges": {"added": [], "removed": [], "changed": []},
                }
            ],
        )

    def test_paging_matches_the_plain_audit_semantics(self) -> None:
        self.reload(b'{"a":["admin"]}')
        self.reload(b'{"a":["admin"],"b":["read"]}', token="a")
        self.reload(b'{"a":["admin"],"b":["write"]}', token="a")
        status, page_one = self.changes("?after=0&limit=2", token="a")[0:2]
        self.assertEqual(status, 200)
        self.assertEqual([e["sequence"] for e in page_one["events"]], [1, 2])
        self.assertEqual(page_one["nextCursor"], 2)
        self.assertTrue(page_one["hasMore"])
        self.assertEqual(page_one["eventsCount"], 3)
        status, page_two = self.changes("?after=2&limit=2", token="a")[0:2]
        self.assertEqual([e["sequence"] for e in page_two["events"]], [3])
        self.assertEqual(page_two["nextCursor"], 3)
        self.assertFalse(page_two["hasMore"])
        self.assertEqual(page_two["digest"], page_one["digest"])
        # The tail page at the event count is a valid empty page.
        status, tail = self.changes("?after=3&limit=1", token="a")[0:2]
        self.assertEqual(status, 200)
        self.assertEqual(tail["events"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertFalse(tail["hasMore"])

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
                status, payload, raw, _ = self.changes(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_the_event_count_is_400(self) -> None:
        status, payload, _, _ = self.changes("?after=1&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- route shape precedes the query check --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            "/v1/admin/scope-policy/audit/changes/",
            "/v1/admin/scope-policy/audit/changes/extra",
            "/v1/admin/scope-policy/audit/other",
            "/v1/admin/scope-policy",
            "/admin/scope-policy/audit/changes",
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
        conn.request("GET", CHANGES_PATH + "?after=0&limit=1")
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        # Token mismatch.
        status, _, _, headers = self.changes("?after=0&limit=1", token="nobody")
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Malformed header value.
        conn = conn_factory()
        conn.putrequest("GET", CHANGES_PATH + "?after=0&limit=1")
        conn.putheader("Authorization", "Token admin-token")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_authenticated_without_admin_is_403_without_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN):
            with self.subTest(token=token):
                status, payload, _, headers = self.changes("?after=0&limit=1", token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, _, _ = self.changes("?after=nope&limit=1", token=READ_TOKEN)
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
                    conn.request("GET", CHANGES_PATH + "?after=0&limit=1", headers=headers)
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

    # -- failed reloads leave no event --

    def test_failed_and_rejected_reloads_never_enter_the_history(self) -> None:
        self.write_policy(b'{"t":["bogus"]}')
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 409)
        os.unlink(self.policy_path)
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 503)
        self.write_policy(INITIAL_BYTES)
        status, payload, _, _ = self.changes("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["eventsCount"], 0)

    # -- persistence across a restart --

    def test_details_survive_a_restart(self) -> None:
        self.reload(b'{"keep":["admin"],"drop":["read"]}')
        self.reload(b'{"keep":["admin"],"new":["write"]}', token="keep")
        events_before = self.changes("?after=0&limit=100", token="keep")[1]["events"]
        digest_before = self.changes("?after=0&limit=100", token="keep")[1]["digest"]
        self.server.store = StateStore(data_file=self.data_path)
        status, payload, _, _ = self.changes("?after=0&limit=100", token="keep")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], events_before)
        self.assertEqual(payload["digest"], digest_before)

    # -- the audit query is strictly read-only --

    def test_changes_query_persists_nothing(self) -> None:
        self.reload(b'{"keep":["admin"]}')
        listing_before = sorted(os.listdir(self.tmpdir))
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in ("?after=0&limit=100", "?after=0&limit=1&x=1"):
            self.changes(query, token="keep")
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)


if __name__ == "__main__":
    unittest.main()
