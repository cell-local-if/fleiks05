"""Tests for the scope-policy change-history verification export.

``GET /v1/admin/scope-policy/audit/verify`` pages the history of
successful scope-policy hot reloads exactly like
``GET /v1/admin/scope-policy/audit`` (same required ``after``/``limit``
incremental-export rules, same event page, cursor, remaining flag,
full-history digest, and count) and additionally carries an independent
``verification`` conclusion over the complete history. The tests cover
the query parser, the integrity scan (missing, duplicate, out-of-range
sequences and digest-shape mismatches marked with a 0-based
``eventsIndex`` and the 1-based ``sequence``), the HTTP request
precedence chain (404 path shape, 401 authentication with a Bearer
challenge, 403 without one, the scope-policy-only mode gate, 400 query
validation), the compact ordered response body, snapshot/full-history
semantics on every page, ``--data-file`` recovery, and the
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

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _policy_events_verification_locked,
    load_scope_policy,
    parse_scope_policy_audit_verify_query,
)

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
VERIFY_PATH = "/v1/admin/scope-policy/audit/verify"
AUDIT_PATH = "/v1/admin/scope-policy/audit"
RELOAD_PATH = "/v1/admin/scope-policy/reload"

INITIAL_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}
INITIAL_BYTES = json.dumps(INITIAL_POLICY).encode("utf-8")

OK_VERIFICATION = {
    "status": "ok",
    "missingSequences": [],
    "duplicateSequences": [],
    "outOfRangeSequences": [],
    "digestMismatches": [],
}


class ParseScopePolicyAuditVerifyQueryTests(unittest.TestCase):
    def test_requires_both_after_and_limit(self) -> None:
        for query in ("", "after=0", "limit=1"):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_verify_query(query))

    def test_accepts_plain_ascii_decimal_pairs(self) -> None:
        self.assertEqual(
            parse_scope_policy_audit_verify_query("after=0&limit=1"), (0, 1)
        )
        self.assertEqual(
            parse_scope_policy_audit_verify_query("after=00&limit=0100"), (0, 100)
        )
        self.assertEqual(
            parse_scope_policy_audit_verify_query("after=12&limit=7"), (12, 7)
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
                self.assertIsNone(parse_scope_policy_audit_verify_query(query))

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
                self.assertIsNone(parse_scope_policy_audit_verify_query(query))


def event(sequence: object, digest: object, tokens: int) -> dict:
    return {"sequence": sequence, "digest": digest, "tokens": tokens}


class PolicyEventsVerificationTests(unittest.TestCase):
    def test_empty_history_is_intact(self) -> None:
        self.assertEqual(_policy_events_verification_locked([]), OK_VERIFICATION)

    def test_continuous_hex_history_is_ok(self) -> None:
        events = [event(1, "a" * 64, 1), event(2, "b" * 64, 0)]
        self.assertEqual(_policy_events_verification_locked(events)["status"], "ok")

    def test_missing_sequence_is_marked_with_events_index_and_sequence(self) -> None:
        # Two events claiming sequences 1 and 3: position 2 is unclaimed
        # (its marker sits at history index 1), and the claim of 3 is past
        # the two-event range, so it is out of range rather than claimed.
        events = [event(1, "a" * 64, 1), event(3, "c" * 64, 1)]
        verification = _policy_events_verification_locked(events)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["missingSequences"], [{"eventsIndex": 1, "sequence": 2}]
        )
        self.assertEqual(
            verification["outOfRangeSequences"], [{"eventsIndex": 1, "sequence": 3}]
        )

    def test_pure_missing_when_a_later_duplicate_frees_a_position(self) -> None:
        # Three events; sequences claimed are {1, 3}, so 2 is missing and
        # nothing is out of range.
        events = [
            event(1, "a" * 64, 1),
            event(3, "b" * 64, 1),
            event(1, "c" * 64, 1),
        ]
        verification = _policy_events_verification_locked(events)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["missingSequences"], [{"eventsIndex": 1, "sequence": 2}]
        )
        self.assertEqual(
            verification["duplicateSequences"], [{"eventsIndex": 2, "sequence": 1}]
        )
        self.assertEqual(verification["outOfRangeSequences"], [])

    def test_duplicate_marks_only_the_later_occurrence(self) -> None:
        events = [
            event(1, "a" * 64, 1),
            event(1, "b" * 64, 1),
            event(3, "c" * 64, 1),
        ]
        verification = _policy_events_verification_locked(events)
        self.assertEqual(
            verification["duplicateSequences"], [{"eventsIndex": 1, "sequence": 1}]
        )
        self.assertEqual(
            verification["missingSequences"], [{"eventsIndex": 1, "sequence": 2}]
        )

    def test_out_of_range_values(self) -> None:
        cases = [0, -1, 4, "1", 1.0, True, False, None]
        for bad_sequence in cases:
            with self.subTest(bad_sequence=bad_sequence):
                events = [event(bad_sequence, "a" * 64, 1)]
                verification = _policy_events_verification_locked(events)
                self.assertEqual(
                    verification["outOfRangeSequences"],
                    [{"eventsIndex": 0, "sequence": bad_sequence}],
                )
                self.assertEqual(verification["status"], "broken")

    def test_digest_mismatch_carries_null_expected_and_observed_value(self) -> None:
        for bad_digest in ("a" * 63, "A" * 64, "g" * 64, "", 1, None):
            with self.subTest(bad_digest=bad_digest):
                events = [event(1, bad_digest, 1)]
                verification = _policy_events_verification_locked(events)
                self.assertEqual(
                    verification["digestMismatches"],
                    [
                        {
                            "eventsIndex": 0,
                            "sequence": 1,
                            "expected": None,
                            "observed": bad_digest,
                        }
                    ],
                )
                self.assertEqual(verification["status"], "broken")

    def test_independent_anomaly_lists_combine(self) -> None:
        events = [
            event(1, "a" * 64, 1),
            event(1, "b" * 64, 1),  # duplicate of 1; frees position 2
            event(2, "nope", 1),  # bad digest shape
        ]
        verification = _policy_events_verification_locked(events)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["duplicateSequences"], [{"eventsIndex": 1, "sequence": 1}]
        )
        self.assertEqual(
            verification["digestMismatches"],
            [
                {
                    "eventsIndex": 2,
                    "sequence": 2,
                    "expected": None,
                    "observed": "nope",
                }
            ],
        )
        # Claimed positions are {1, 2}; with three events, position 3 is
        # unclaimed.
        self.assertEqual(
            verification["missingSequences"], [{"eventsIndex": 2, "sequence": 3}]
        )
        self.assertEqual(verification["outOfRangeSequences"], [])


class PolicyEventsVerifyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def digest_input(self, events: list[dict]) -> bytes:
        return json.dumps(
            [
                {"sequence": e["sequence"], "digest": e["digest"], "tokens": e["tokens"]}
                for e in events
            ],
            separators=(",", ":"),
        ).encode("utf-8")

    def test_empty_history_verifies_ok_and_hashes_empty_array(self) -> None:
        store = StateStore()
        status, payload = store.get_policy_events_verify(0, 100)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], OK_VERIFICATION)

    def test_page_summary_and_verification_cover_the_whole_history(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 3)
        store.record_policy_reload("b" * 64, 1)
        expected = [
            {"sequence": 1, "digest": "a" * 64, "tokens": 3},
            {"sequence": 2, "digest": "b" * 64, "tokens": 1},
        ]
        first_status, first = store.get_policy_events_verify(0, 1)
        second_status, second = store.get_policy_events_verify(1, 1)
        self.assertEqual(first_status, 200)
        self.assertEqual(first["events"], expected[:1])
        self.assertEqual(first["nextCursor"], 1)
        self.assertTrue(first["hasMore"])
        self.assertEqual(second["events"], expected[1:])
        self.assertEqual(second["nextCursor"], 2)
        self.assertFalse(second["hasMore"])
        full_digest = hashlib.sha256(self.digest_input(expected)).hexdigest()
        for page in (first, second):
            self.assertEqual(page["eventsCount"], 2)
            self.assertEqual(page["digest"], full_digest)
            self.assertEqual(page["verification"]["status"], "ok")
            self.assertEqual(page["verification"]["missingSequences"], [])

    def test_empty_tail_still_verifies_the_complete_history(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        status, payload = store.get_policy_events_verify(1, 10)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 1)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_after_past_count_raises_value_error(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        with self.assertRaises(ValueError):
            store.get_policy_events_verify(2, 10)

    def test_damaged_history_is_reported_broken_on_every_page(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        store.record_policy_reload("b" * 64, 1)
        store.record_policy_reload("c" * 64, 1)
        # Damage the snapshot: event 2 re-claims sequence 1 (so 2 goes
        # missing) and carries a non-hex digest.
        store._policy_events[1]["sequence"] = 1
        store._policy_events[1]["digest"] = "broken"
        for after, page_events in ((0, 3), (2, 1)):
            with self.subTest(after=after):
                status, payload = store.get_policy_events_verify(after, 100)
                self.assertEqual(status, 200)
                self.assertEqual(len(payload["events"]), page_events)
                # The summary still covers the whole damaged history.
                self.assertEqual(payload["eventsCount"], 3)
                self.assertEqual(
                    payload["digest"],
                    hashlib.sha256(
                        self.digest_input(store._policy_events)
                    ).hexdigest(),
                )
                verification = payload["verification"]
                self.assertEqual(verification["status"], "broken")
                self.assertEqual(
                    verification["duplicateSequences"],
                    [{"eventsIndex": 1, "sequence": 1}],
                )
                self.assertEqual(
                    verification["missingSequences"],
                    [{"eventsIndex": 1, "sequence": 2}],
                )
                self.assertEqual(
                    verification["digestMismatches"],
                    [
                        {
                            "eventsIndex": 1,
                            "sequence": 1,
                            "expected": None,
                            "observed": "broken",
                        }
                    ],
                )

    def test_recovery_reproduces_events_digest_count_and_verification(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 3)
        store.record_policy_reload("b" * 64, 0)
        _, before = store.get_policy_events_verify(0, 1)
        recovered = StateStore(data_file=data_file)
        _, after = recovered.get_policy_events_verify(0, 1)
        self.assertEqual(after["events"], before["events"])
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["eventsCount"], before["eventsCount"])
        self.assertEqual(after["verification"], before["verification"])
        _, tail = recovered.get_policy_events_verify(2, 10)
        self.assertEqual(tail["events"], [])
        self.assertEqual(tail["verification"]["status"], "ok")

    def test_old_file_without_policy_events_verifies_empty(self) -> None:
        data_file = os.path.join(self.tmpdir, "old.json")
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=data_file)
        status, payload = store.get_policy_events_verify(0, 100)
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], OK_VERIFICATION)


class ScopePolicyAuditVerifyHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-policy-verify-http-")
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

    def verify(self, query: str, token: str | None = ADMIN_TOKEN):
        return self.request("GET", VERIFY_PATH + query, token=token)

    def reload(self, content: bytes, token: str | None = ADMIN_TOKEN) -> str:
        self.write_policy(content)
        status, payload, _, _ = self.request(
            "POST", RELOAD_PATH, token=token, body={}
        )
        self.assertEqual(status, 200, payload)
        return payload["policyDigest"]

    # -- success contract on an empty history --

    def test_empty_history_is_compact_ordered_json_with_a_single_newline(self) -> None:
        status, payload, raw, headers = self.verify("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            [
                "events",
                "nextCursor",
                "hasMore",
                "algorithm",
                "digest",
                "eventsCount",
                "verification",
            ],
        )
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(
            list(payload["verification"]),
            [
                "status",
                "missingSequences",
                "duplicateSequences",
                "outOfRangeSequences",
                "digestMismatches",
            ],
        )
        self.assertEqual(
            raw,
            b'{"events":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + payload["digest"].encode()
            + b'","eventsCount":0,"verification":{"status":"ok",'
            b'"missingSequences":[],"duplicateSequences":[],'
            b'"outOfRangeSequences":[],"digestMismatches":[]}}\n',
        )
        self.assertTrue(raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    # -- events, paging, and the full-history summary --

    def test_events_and_full_history_summary_match_plain_audit(self) -> None:
        first_bytes = b'{ "admin-token" : [ "admin" ] }'
        first_digest = self.reload(first_bytes)
        second_bytes = b'{"z":["admin"],"r":["read"]}'
        second_digest = self.reload(second_bytes)

        status, verify_payload, _, _ = self.verify(
            "?after=0&limit=100", token="z"
        )
        audit_status, audit_payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token="z"
        )
        self.assertEqual(status, 200)
        self.assertEqual(audit_status, 200)
        self.assertEqual(verify_payload["events"], audit_payload["events"])
        self.assertEqual(
            verify_payload["events"],
            [
                {"sequence": 1, "digest": first_digest, "tokens": 1},
                {"sequence": 2, "digest": second_digest, "tokens": 2},
            ],
        )
        for field in ("nextCursor", "hasMore", "algorithm", "digest", "eventsCount"):
            self.assertEqual(verify_payload[field], audit_payload[field])
        self.assertEqual(verify_payload["verification"]["status"], "ok")

    def test_verification_digest_and_count_are_page_independent(self) -> None:
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
            status, payload, _, _ = self.verify(f"?after={after}&limit=1")
            self.assertEqual(status, 200)
            pages.append(payload)
        full_digest = pages[0]["digest"]
        for page in pages:
            self.assertEqual(page["digest"], full_digest)
            self.assertEqual(page["eventsCount"], 3)
            self.assertEqual(page["verification"]["status"], "ok")
        self.assertEqual([p["events"][0]["sequence"] for p in pages[:3]], [1, 2, 3])
        self.assertEqual([p["events"][0]["digest"] for p in pages[:3]], digests)
        self.assertEqual(pages[3]["events"], [])
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.reload(b'{"x":["admin"]}')
        status, payload, raw, _ = self.verify("?after=1&limit=10", token="x")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["eventsCount"], 1)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_damaged_history_is_reported_broken_over_http(self) -> None:
        self.reload(b'{"admin-token":["admin"]}')
        self.reload(b'{"admin-token":["admin"],"z":["read"]}')
        # Tamper with the live history in place: the second event re-claims
        # sequence 1 and records a malformed digest. The HTTP endpoint must
        # surface the broken conclusion while still covering the complete
        # (damaged) history with digest and count.
        self.server.store._policy_events[1]["sequence"] = 1
        self.server.store._policy_events[1]["digest"] = "nope"
        status, payload, _, _ = self.verify("?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(payload["eventsCount"], 2)
        verification = payload["verification"]
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["duplicateSequences"],
            [{"eventsIndex": 1, "sequence": 1}],
        )
        self.assertEqual(
            verification["missingSequences"],
            [{"eventsIndex": 1, "sequence": 2}],
        )
        self.assertEqual(
            verification["digestMismatches"],
            [
                {
                    "eventsIndex": 1,
                    "sequence": 1,
                    "expected": None,
                    "observed": "nope",
                }
            ],
        )

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
            "?after=%C2%B9&limit=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.verify(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_the_event_count_is_400(self) -> None:
        status, payload, _, _ = self.verify("?after=1&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- route shape precedes the query check --

    def test_path_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        for path in (
            VERIFY_PATH + "/",
            VERIFY_PATH + "/extra",
            "/v1/admin/scope-policy/audit/verify/extra",
            "/v1/admin/scope-policy/audit/verifyx",
            "/v1/admin/scope-policy/other/verify",
            "/v1/admin/scope-policy",
            "/v1/admin/scopepolicy/audit/verify",
            "/admin/scope-policy/audit/verify",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request(
                    "GET", path + "?bogus=1", token=ADMIN_TOKEN
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
        # The plain audit route is a different published route, so an
        # unknown query there keeps its own 400 rather than becoming a 404.
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?bogus=1", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_plain_audit_route_still_serves_its_own_contract(self) -> None:
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=100", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        self.assertNotIn("verification", payload)

    # -- authentication and scope --

    def test_missing_or_bad_credential_is_401_with_a_bearer_challenge(self) -> None:
        conn_factory = lambda: http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn = conn_factory()
        conn.request("GET", VERIFY_PATH + "?after=0&limit=1")
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        status, _, _, headers = self.verify("?after=0&limit=1", token="nobody")
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        conn = conn_factory()
        conn.putrequest("GET", VERIFY_PATH + "?after=0&limit=1")
        conn.putheader("Authorization", "Token admin-token")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()
        conn = conn_factory()
        conn.putrequest("GET", VERIFY_PATH + "?after=0&limit=1")
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
                status, payload, _, headers = self.verify("?after=0&limit=1", token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, _, _ = self.verify("?after=nope&limit=1", token=READ_TOKEN)
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
                    conn.request("GET", VERIFY_PATH + "?after=0&limit=1", headers=headers)
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

    # -- failed reloads leave no trace; the query is strictly read-only --

    def test_failed_reloads_never_enter_the_verified_history(self) -> None:
        self.write_policy(b'{"t":["bogus"]}')
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 409)
        os.unlink(self.policy_path)
        status, _, _, _ = self.request("POST", RELOAD_PATH, token=ADMIN_TOKEN, body={})
        self.assertEqual(status, 503)
        status, payload, _, _ = self.verify("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)

    def test_verify_query_persists_nothing(self) -> None:
        self.reload(b'{"keep":["admin"]}')
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in (
            "?after=0&limit=100",
            "?after=1&limit=1",
            "?after=0&limit=1&x=1",
            "?after=99&limit=1",
        ):
            self.verify(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
