"""Tests for the scope-policy token-change audit detail.

``GET /v1/admin/scope-policy/audit/changes`` pages the same successful
hot-reload history as ``GET /v1/admin/scope-policy/audit`` but each
event carries ``sequence``, ``policyDigest``, and ``tokenChanges`` —
added/removed/changed token SHA-256 fingerprints, or null for an older
event whose detail cannot be rebuilt. The tests cover the query parser,
the store paging/digest contract, fingerprint computation, the HTTP
request precedence chain (404 path shape, 401 authentication with a
Bearer challenge, 403 without one, the scope-policy-only mode gate, 400
query validation), compact ordered response bodies, atomic commit with
the new policy, ``--data-file`` persistence and recovery of both new
and old files, corrupt-file rejection, and the read-only / no-leak
guarantees (no raw token, scope, or policy bytes ever appear).
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
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _scope_token_fingerprint,
    load_scope_policy,
    parse_scope_policy_audit_changes_query,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
CHANGES_PATH = "/v1/admin/scope-policy/audit/changes"
AUDIT_PATH = "/v1/admin/scope-policy/audit"
RELOAD_PATH = "/v1/admin/scope-policy/reload"
METRICS_PATH = "/v1/metrics"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")


def fp(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def detail(added=(), removed=(), changed=()) -> dict:
    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
    }


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


class PolicyChangeEventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-changes-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def digest_input(self, events: list[dict]) -> bytes:
        return json.dumps(
            [
                {
                    "sequence": e["sequence"],
                    "policyDigest": e["policyDigest"],
                    "tokenChanges": e["tokenChanges"],
                }
                for e in events
            ],
            separators=(",", ":"),
        ).encode("utf-8")

    def test_empty_history_pages_and_hashes_the_empty_array(self) -> None:
        store = StateStore()
        status, payload = store.get_policy_change_events(0, 100)
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

    def test_events_page_with_fingerprints_and_digest_covers_full_history(self) -> None:
        store = StateStore()
        first_detail = detail(added=[fp("a"), fp("b")])
        second_detail = detail(added=[fp("c")], removed=[fp("a")], changed=[fp("b")])
        store.record_policy_reload("a" * 64, 2, first_detail)
        store.record_policy_reload("b" * 64, 2, second_detail)
        expected = [
            {"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": first_detail},
            {"sequence": 2, "policyDigest": "b" * 64, "tokenChanges": second_detail},
        ]
        status, payload = store.get_policy_change_events(0, 1)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], expected[:1])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertTrue(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(
            payload["digest"], hashlib.sha256(self.digest_input(expected)).hexdigest()
        )
        status, page_two = store.get_policy_change_events(1, 1)
        self.assertEqual(page_two["events"], expected[1:])
        self.assertEqual(page_two["nextCursor"], 2)
        self.assertFalse(page_two["hasMore"])
        self.assertEqual(page_two["digest"], payload["digest"])

    def test_old_three_field_events_report_null_detail(self) -> None:
        store = StateStore()
        # Simulate an event recovered from a file written before the
        # detail existed: no tokenChanges key at all.
        store.record_policy_reload("a" * 64, 1)
        # A subsequent reload supplies its detail; the old event stays null.
        new_detail = detail(added=[fp("x")])
        store.record_policy_reload("c" * 64, 1, new_detail)
        status, payload = store.get_policy_change_events(0, 100)
        self.assertEqual(
            payload["events"],
            [
                {"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": None},
                {"sequence": 2, "policyDigest": "c" * 64, "tokenChanges": new_detail},
            ],
        )
        expected = [
            {"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": None},
            {"sequence": 2, "policyDigest": "c" * 64, "tokenChanges": new_detail},
        ]
        self.assertEqual(
            payload["digest"], hashlib.sha256(self.digest_input(expected)).hexdigest()
        )

    def test_after_equal_to_count_is_empty_tail_past_count_is_value_error(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1, detail())
        status, payload = store.get_policy_change_events(1, 10)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        with self.assertRaises(ValueError):
            store.get_policy_change_events(2, 10)

    def test_plain_audit_page_never_carries_the_detail(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1, detail(added=[fp("x")]))
        _, detailed = store.get_policy_change_events(0, 100)
        _, plain = store.get_policy_events(0, 100)
        self.assertEqual(
            plain["events"], [{"sequence": 1, "digest": "a" * 64, "tokens": 1}]
        )
        self.assertNotIn("tokenChanges", plain["events"][0])
        # The two surfaces hash different field sets.
        self.assertNotEqual(plain["digest"], detailed["digest"])

    def test_persistence_failure_rolls_the_event_back(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 1, detail(added=[fp("x")]))
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            with self.assertRaises(PersistenceError):
                store.record_policy_reload("b" * 64, 2, detail(added=[fp("y")]))
        self.assertEqual([e["digest"] for e in store._policy_events], ["a" * 64])

    def test_details_persist_and_recover(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 1, detail(added=[fp("x")]))
        store.record_policy_reload(
            "b" * 64,
            3,
            detail(added=[fp("y")], removed=[fp("x")], changed=[fp("z")]),
        )
        recovered = StateStore(data_file=data_file)
        _, payload = recovered.get_policy_change_events(0, 100)
        self.assertEqual(payload["eventsCount"], 2)
        self.assertEqual(payload["events"][0]["tokenChanges"], detail(added=[fp("x")]))
        self.assertEqual(
            payload["events"][1]["tokenChanges"],
            detail(added=[fp("y")], removed=[fp("x")], changed=[fp("z")]),
        )
        with open(data_file, "rb") as handle:
            raw = handle.read()
        # Fingerprints are persisted; raw tokens never are.
        self.assertNotIn(b'"x"', raw)
        self.assertIn(fp("x").encode("utf-8"), raw)

    def test_old_file_recovers_null_and_later_reload_supplements(self) -> None:
        data_file = os.path.join(self.tmpdir, "old.json")
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "operations": [],
                    "policyEvents": [
                        {"sequence": 1, "digest": "a" * 64, "tokens": 1}
                    ],
                },
                handle,
            )
        store = StateStore(data_file=data_file)
        _, payload = store.get_policy_change_events(0, 100)
        self.assertEqual(
            payload["events"],
            [{"sequence": 1, "policyDigest": "a" * 64, "tokenChanges": None}],
        )
        # A later reload supplements the history; the old null stays.
        store.record_policy_reload("d" * 64, 2, detail(added=[fp("k")]))
        _, payload = store.get_policy_change_events(0, 100)
        self.assertEqual(payload["events"][0]["tokenChanges"], None)
        self.assertEqual(
            payload["events"][1]["tokenChanges"], detail(added=[fp("k")])
        )
        # The supplemented file keeps the old event shape (no null key
        # written) and the new event's detail.
        with open(data_file, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(
            set(document["policyEvents"][0]), {"sequence", "digest", "tokens"}
        )
        self.assertEqual(
            set(document["policyEvents"][1]),
            {"sequence", "digest", "tokens", "tokenChanges"},
        )

    def test_explicit_null_detail_recovers_as_null(self) -> None:
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
                            "tokens": 1,
                            "tokenChanges": None,
                        }
                    ],
                },
                handle,
            )
        store = StateStore(data_file=data_file)
        _, payload = store.get_policy_change_events(0, 100)
        self.assertIsNone(payload["events"][0]["tokenChanges"])


class CorruptPolicyChangeEventsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-changes-bad-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = os.path.join(self.tmpdir, "state.json")

    def write(self, policy_events: object) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump(
                {"version": 1, "operations": [], "policyEvents": policy_events},
                handle,
            )

    def test_corrupt_token_changes_make_startup_fail(self) -> None:
        good_fingerprint = "a" * 64
        good_event = {"sequence": 1, "digest": "b" * 64, "tokens": 1}
        cases = [
            [dict(good_event, tokenChanges=[])],
            [dict(good_event, tokenChanges="null")],
            [dict(good_event, tokenChanges={"added": [], "removed": []})],
            [
                dict(
                    good_event,
                    tokenChanges={
                        "added": [],
                        "removed": [],
                        "changed": [],
                        "x": [],
                    },
                )
            ],
            [dict(good_event, tokenChanges={"added": [good_fingerprint], "removed": [], "changed": {}})],
            [dict(good_event, tokenChanges={"added": ["A" * 64], "removed": [], "changed": []})],
            [dict(good_event, tokenChanges={"added": ["a" * 63], "removed": [], "changed": []})],
            # Duplicate fingerprints.
            [dict(good_event, tokenChanges={"added": [good_fingerprint, good_fingerprint], "removed": [], "changed": []})],
            # Not ascending.
            [dict(good_event, tokenChanges={"added": ["b" * 64, "a" * 64], "removed": [], "changed": []})],
            [dict(good_event, tokenChanges={"added": [], "removed": [123], "changed": []})],
        ]
        for policy_events in cases:
            with self.subTest(policy_events=policy_events):
                self.write(policy_events)
                with self.assertRaises(PersistenceError):
                    StateStore(data_file=self.data_file)


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_sha256_utf8_lowercase_hex64(self) -> None:
        for token in ("", "a", "admin-token", "tökën", "x" * 100):
            with self.subTest(token=token):
                self.assertEqual(
                    _scope_token_fingerprint(token),
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                )
                self.assertRegex(_scope_token_fingerprint(token), r"^[0-9a-f]{64}$")

    def test_distinct_tokens_have_distinct_fingerprints(self) -> None:
        self.assertNotEqual(fp("a"), fp("b"))


class ScopePolicyChangesHttpTests(unittest.TestCase):
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
        if os.path.lexists(self.data_path):
            os.unlink(self.data_path)
        self.server.store = StateStore(data_file=self.data_path)
        self.write_policy(INITIAL_BYTES)
        # No recorder: the reset reload leaves no audit event, exactly
        # like the plain audit test harness.
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

    # -- success contract on an empty history --

    def test_empty_history_is_compact_ordered_json_with_a_single_newline(self) -> None:
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

    # -- the diff and fingerprints --

    def test_token_changes_fingerprint_added_removed_changed(self) -> None:
        # Initial live mapping: reader-token, writer-token, admin-token.
        # New mapping:
        #   admin-token -> ["admin"]                 (changed)
        #   writer-token unchanged                    (not reported)
        #   fresh-token only in new                   (added)
        #   reader-token dropped                      (removed)
        new_policy = {
            ADMIN_TOKEN: ["admin"],
            WRITE_TOKEN: ["write"],
            "fresh-token": ["read", "write"],
        }
        first_digest = self.reload(json.dumps(new_policy).encode("utf-8"))
        status, payload, raw, _ = self.changes("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 1)
        event = payload["events"][0]
        self.assertEqual(
            list(event), ["sequence", "policyDigest", "tokenChanges"]
        )
        self.assertEqual(event["sequence"], 1)
        self.assertEqual(event["policyDigest"], first_digest)
        self.assertEqual(
            event["tokenChanges"],
            {
                "added": [fp("fresh-token")],
                "removed": [fp(READ_TOKEN)],
                "changed": [fp(ADMIN_TOKEN)],
            },
        )
        # The unchanged token appears in no list.
        reported = set().union(*(event["tokenChanges"][k] for k in ("added", "removed", "changed")))
        self.assertNotIn(fp(WRITE_TOKEN), reported)
        # Arrays are ascending, 64 lowercase hex each.
        for name in ("added", "removed", "changed"):
            values = event["tokenChanges"][name]
            self.assertEqual(values, sorted(set(values)))
            self.assertTrue(all(len(v) == 64 and v == v.lower() for v in values))
        # No raw token or scope value or policy bytes in the response.
        for secret in (READ_TOKEN, WRITE_TOKEN, ADMIN_TOKEN, "fresh-token", "read", "write"):
            self.assertNotIn(secret.encode("utf-8"), raw)

    def test_repeated_reloads_accumulate_details_and_digest_covers_history(self) -> None:
        first_bytes = json.dumps({ADMIN_TOKEN: ["admin"], "a": ["read"]}).encode()
        first_digest = self.reload(first_bytes)
        second_bytes = json.dumps(
            {ADMIN_TOKEN: ["admin"], "a": ["write"], "b": ["read"]}
        ).encode()
        second_digest = self.reload(second_bytes)
        status, payload, _, _ = self.changes("?after=0&limit=100", token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["events"],
            [
                {
                    "sequence": 1,
                    "policyDigest": first_digest,
                    "tokenChanges": {
                        "added": sorted([fp("a")]),
                        "removed": sorted([fp(READ_TOKEN), fp(WRITE_TOKEN)]),
                        "changed": [fp(ADMIN_TOKEN)],
                    },
                },
                {
                    "sequence": 2,
                    "policyDigest": second_digest,
                    "tokenChanges": {
                        "added": [fp("b")],
                        "removed": [],
                        "changed": [fp("a")],
                    },
                },
            ],
        )
        digest_input = json.dumps(
            [
                {
                    "sequence": 1,
                    "policyDigest": first_digest,
                    "tokenChanges": {
                        "added": sorted([fp("a")]),
                        "removed": sorted([fp(READ_TOKEN), fp(WRITE_TOKEN)]),
                        "changed": [fp(ADMIN_TOKEN)],
                    },
                },
                {
                    "sequence": 2,
                    "policyDigest": second_digest,
                    "tokenChanges": {
                        "added": [fp("b")],
                        "removed": [],
                        "changed": [fp("a")],
                    },
                },
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(payload["digest"], hashlib.sha256(digest_input).hexdigest())

    def test_scope_order_does_not_matter_for_changed(self) -> None:
        # Same scope set, different array order: not a change.
        self.reload(json.dumps({ADMIN_TOKEN: ["admin", "write", "read"]}).encode())
        status, payload, _, _ = self.changes("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"][0]["tokenChanges"]["changed"], [])

    def test_paging_digest_and_count_cover_the_whole_history(self) -> None:
        for i in range(3):
            self.reload(
                json.dumps({ADMIN_TOKEN: ["admin"], f"t{i}": ["read"]}).encode()
            )
        pages = [self.changes(f"?after={after}&limit=1")[1] for after in range(4)]
        full_digest = pages[0]["digest"]
        self.assertTrue(all(page["digest"] == full_digest for page in pages))
        self.assertEqual([p["eventsCount"] for p in pages], [3, 3, 3, 3])
        self.assertEqual([p["events"][0]["sequence"] for p in pages[:3]], [1, 2, 3])
        self.assertEqual(pages[3]["events"], [])
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])

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

    # -- route shape --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            "/v1/admin/scope-policy/audit/changes/",
            "/v1/admin/scope-policy/audit/changes/extra",
            "/v1/admin/scope-policy/audit/other",
            "/v1/admin/scope-policy/changes",
            "/v1/admin/scopepolicy/audit/changes",
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
        conn = conn_factory()
        conn.request("GET", CHANGES_PATH + "?after=0&limit=1")
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        status, _, _, headers = self.changes("?after=0&limit=1", token="nobody")
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
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
                status, payload, _, headers = self.changes(
                    "?after=0&limit=1", token=token
                )
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

    # -- atomicity: failed reloads add nothing --

    def test_failed_reloads_never_enter_the_history(self) -> None:
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

    def test_durable_failure_keeps_old_policy_and_history(self) -> None:
        self.reload(b'{"keep":["admin"]}')
        self.write_policy(b'{"keep":["admin"],"next":["read"]}')
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload, _, headers = self.request(
                "POST", RELOAD_PATH, token="keep", body={}
            )
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "persistence_unavailable"})
        self.assertEqual(headers.get("Retry-After"), "1")
        self.assertEqual(self.changes("?after=0&limit=100", token="next")[0], 401)
        status, _, _, _ = self.request("GET", METRICS_PATH, token="keep")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.server.store._policy_events), 1)

    # -- persistence across a real restart --

    def test_details_round_trip_through_the_data_file(self) -> None:
        self.reload(b'{"only":["admin"]}')
        with open(self.data_path, "rb") as handle:
            persisted_raw = handle.read()
        document = json.loads(persisted_raw.decode("utf-8"))
        self.assertEqual(len(document["policyEvents"]), 1)
        stored = document["policyEvents"][0]
        self.assertEqual(
            set(stored), {"sequence", "digest", "tokens", "tokenChanges"}
        )
        self.assertEqual(
            stored["tokenChanges"],
            {
                "added": [fp("only")],
                "removed": sorted([fp(READ_TOKEN), fp(WRITE_TOKEN), fp(ADMIN_TOKEN)]),
                "changed": [],
            },
        )
        # No raw token or scope value in the data file.
        for secret in (READ_TOKEN, WRITE_TOKEN, ADMIN_TOKEN, "only", "admin", "read", "write"):
            self.assertNotIn(secret.encode("utf-8"), persisted_raw)
        recovered_store = StateStore(data_file=self.data_path)
        self.server.store = recovered_store
        status, payload, _, _ = self.changes("?after=0&limit=100", token="only")
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 1)
        self.assertEqual(
            payload["events"][0]["tokenChanges"], stored["tokenChanges"]
        )

    def test_old_data_file_recovers_null_until_the_next_reload(self) -> None:
        self.reload(b'{"oldadmin":["admin"]}')
        # Rewrite the persisted file dropping the detail, as an older
        # build would have written it.
        with open(self.data_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        for event in document["policyEvents"]:
            event.pop("tokenChanges", None)
        with open(self.data_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        self.server.store = StateStore(data_file=self.data_path)
        status, payload, _, _ = self.changes(
            "?after=0&limit=100", token="oldadmin"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["eventsCount"], 1)
        self.assertIsNone(payload["events"][0]["tokenChanges"])
        # The next reload supplements the detail; the old event stays null.
        self.write_policy(b'{"oldadmin":["admin"],"new":["read"]}')
        status, _, _, _ = self.request(
            "POST", RELOAD_PATH, token="oldadmin", body={}
        )
        self.assertEqual(status, 200)
        status, payload, _, _ = self.changes("?after=0&limit=100", token="oldadmin")
        self.assertEqual(payload["eventsCount"], 2)
        self.assertIsNone(payload["events"][0]["tokenChanges"])
        self.assertEqual(
            payload["events"][1]["tokenChanges"],
            {"added": [fp("new")], "removed": [], "changed": []},
        )

    # -- read-only --

    def test_changes_query_persists_nothing(self) -> None:
        self.reload(b'{"only":["admin"]}')
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in ("?after=0&limit=100", "?after=0&limit=1&x=1", "?after=9&limit=1"):
            self.changes(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_plain_audit_and_verify_are_unchanged(self) -> None:
        self.reload(b'{"z":["admin"]}')
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token="z"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            ["events", "nextCursor", "hasMore", "algorithm", "digest", "eventsCount"],
        )
        self.assertEqual(
            set(payload["events"][0]), {"sequence", "digest", "tokens"}
        )
        status, verify, _, _ = self.request(
            "GET", AUDIT_PATH + "/verify?after=0&limit=100", token="z"
        )
        self.assertEqual(status, 200)
        self.assertEqual(verify["verification"]["status"], "ok")
        self.assertEqual(set(verify["events"][0]), {"sequence", "digest", "tokens"})


if __name__ == "__main__":
    unittest.main()
