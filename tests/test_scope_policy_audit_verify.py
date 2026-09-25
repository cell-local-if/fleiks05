"""Tests for the scope-policy change-audit integrity verification.

``GET /v1/admin/scope-policy/audit/verify`` is the operational sibling of
``GET /v1/admin/scope-policy/audit``: it exports the same successful-reload
history one incremental page at a time (required ``after``/``limit``) and
additionally carries an independent integrity conclusion over the whole
history. The tests cover the query parser, the store-level verification
(missing, duplicate, and out-of-range sequence numbers, digest-shape
mismatches, page-independent summary and conclusion), the response byte
contract, the request precedence chain (404 path shape, 401 with a Bearer
challenge, 403 without one, the scope-policy-only mode gate, 400 query
validation), snapshot consistency under concurrent reloads,
``--data-file`` recovery, and the read-only/no-temp-file guarantee.
"""

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http import HTTPStatus
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
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


def event(sequence: int, digest: str = "a" * 64, tokens: int = 1) -> dict:
    return {"sequence": sequence, "digest": digest, "tokens": tokens}


def verification_ok() -> dict:
    return {
        "status": "ok",
        "missingSequences": [],
        "duplicates": [],
        "outOfRange": [],
        "digestMismatches": [],
    }


class ParseScopePolicyAuditVerifyQueryTests(unittest.TestCase):
    def test_requires_both_after_and_limit(self) -> None:
        for query in ("", "after=0", "limit=1"):
            with self.subTest(query=query):
                self.assertIsNone(parse_scope_policy_audit_verify_query(query))

    def test_accepts_plain_ascii_decimal_pairs(self) -> None:
        self.assertEqual(parse_scope_policy_audit_verify_query("after=0&limit=1"), (0, 1))
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


class PolicyEventsVerificationFunctionTests(unittest.TestCase):
    def test_empty_history_is_ok(self) -> None:
        self.assertEqual(_policy_events_verification_locked([]), verification_ok())

    def test_continuous_one_based_history_is_ok(self) -> None:
        events = [event(1, "a" * 64), event(2, "b" * 64), event(3, "c" * 64)]
        self.assertEqual(_policy_events_verification_locked(events), verification_ok())

    def test_missing_sequence_is_reported_at_the_gap_position(self) -> None:
        # History length 3 claims sequences 1 and 3: number 2 is missing.
        result = _policy_events_verification_locked([event(1), event(3), event(4)])
        self.assertEqual(result["status"], "broken")
        # Missing numbers 2 and 3? No: 1 present, 3 present, 4 present; the
        # history length is 3, so only 2 is missing (4 is out of range).
        self.assertEqual(
            result["missingSequences"], [{"eventsIndex": 1, "sequence": 2}]
        )
        self.assertEqual(result["duplicates"], [])
        self.assertEqual(result["outOfRange"], [{"eventsIndex": 2, "sequence": 4}])
        self.assertEqual(result["digestMismatches"], [])

    def test_duplicate_sequence_marks_every_later_occurrence(self) -> None:
        result = _policy_events_verification_locked(
            [event(1), event(1), event(1)]
        )
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["duplicates"],
            [
                {"eventsIndex": 1, "sequence": 1},
                {"eventsIndex": 2, "sequence": 1},
            ],
        )
        # A length-3 history also never claims 2 or 3.
        self.assertEqual(
            result["missingSequences"],
            [
                {"eventsIndex": 1, "sequence": 2},
                {"eventsIndex": 2, "sequence": 3},
            ],
        )

    def test_out_of_range_covers_zero_and_past_end(self) -> None:
        result = _policy_events_verification_locked(
            [event(0, "a" * 64), event(2, "b" * 64)]
        )
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["outOfRange"],
            [
                {"eventsIndex": 0, "sequence": 0},
            ],
        )
        self.assertEqual(
            result["missingSequences"], [{"eventsIndex": 0, "sequence": 1}]
        )

    def test_digest_shape_mismatch_is_reported(self) -> None:
        good = event(1)
        cases = ["A" * 64, "a" * 63, "g" * 64, ""]
        for bad_digest in cases:
            with self.subTest(bad_digest=bad_digest):
                result = _policy_events_verification_locked(
                    [good, event(2, bad_digest)]
                )
                self.assertEqual(result["status"], "broken")
                self.assertEqual(
                    result["digestMismatches"],
                    [{"eventsIndex": 1, "sequence": 2}],
                )

    def test_one_event_can_carry_several_anomalies(self) -> None:
        # Sequence 4 is past the length 1, out of range, and leaves 1
        # missing; its digest is also malformed.
        result = _policy_events_verification_locked([event(4, "zz")])
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["missingSequences"], [{"eventsIndex": 0, "sequence": 1}]
        )
        self.assertEqual(result["outOfRange"], [{"eventsIndex": 0, "sequence": 4}])
        self.assertEqual(
            result["digestMismatches"], [{"eventsIndex": 0, "sequence": 4}]
        )
        self.assertEqual(result["duplicates"], [])


class PolicyEventsVerificationStoreTests(unittest.TestCase):
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

    def test_empty_history_pages_and_verifies_ok(self) -> None:
        store = StateStore()
        status, payload = store.get_policy_events_verification(0, 100)
        self.assertIs(status, HTTPStatus.OK)
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
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["eventsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], verification_ok())

    def test_healthy_history_summary_and_conclusion_cover_all_pages(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 3)
        store.record_policy_reload("b" * 64, 1)
        expected = [
            {"sequence": 1, "digest": "a" * 64, "tokens": 3},
            {"sequence": 2, "digest": "b" * 64, "tokens": 1},
        ]
        pages = [store.get_policy_events_verification(after, 1) for after in range(3)]
        for status, page in pages:
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(page["eventsCount"], 2)
            self.assertEqual(
                page["digest"],
                hashlib.sha256(self.digest_input(expected)).hexdigest(),
            )
            self.assertEqual(page["verification"], verification_ok())
        self.assertEqual(pages[0][1]["events"], expected[:1])
        self.assertEqual(pages[0][1]["nextCursor"], 1)
        self.assertIs(pages[0][1]["hasMore"], True)
        self.assertEqual(pages[1][1]["events"], expected[1:])
        self.assertEqual(pages[1][1]["nextCursor"], 2)
        self.assertIs(pages[1][1]["hasMore"], False)
        # The boundary request is a stable empty page over the same history.
        status, tail = store.get_policy_events_verification(2, 10)
        self.assertEqual(tail["events"], [])
        self.assertEqual(tail["nextCursor"], 2)
        self.assertIs(tail["hasMore"], False)
        self.assertEqual(tail["verification"], verification_ok())

    def test_after_past_count_is_value_error(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        with self.assertRaises(ValueError):
            store.get_policy_events_verification(2, 10)

    def test_broken_history_is_reported_but_still_exported_and_hashed(self) -> None:
        store = StateStore()
        tampered = [
            event(1, "a" * 64, 3),
            event(1, "b" * 64, 1),
            event(4, "BAD", 0),
        ]
        store._policy_events = [dict(item) for item in tampered]
        status, payload = store.get_policy_events_verification(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["events"], tampered)
        self.assertEqual(payload["eventsCount"], 3)
        # The digest is recomputed over the actual (tampered) full history.
        self.assertEqual(
            payload["digest"],
            hashlib.sha256(self.digest_input(tampered)).hexdigest(),
        )
        verification = payload["verification"]
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["missingSequences"],
            [
                {"eventsIndex": 1, "sequence": 2},
                {"eventsIndex": 2, "sequence": 3},
            ],
        )
        self.assertEqual(
            verification["duplicates"], [{"eventsIndex": 1, "sequence": 1}]
        )
        self.assertEqual(
            verification["outOfRange"], [{"eventsIndex": 2, "sequence": 4}]
        )
        self.assertEqual(
            verification["digestMismatches"],
            [{"eventsIndex": 2, "sequence": 4}],
        )

    def test_paging_trims_only_the_event_page(self) -> None:
        store = StateStore()
        store._policy_events = [
            event(1, "a" * 64),
            event(1, "b" * 64),  # duplicate, leaves sequence 2 missing
        ]
        status, page = store.get_policy_events_verification(1, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(page["events"], [event(1, "b" * 64)])
        self.assertEqual(page["nextCursor"], 2)
        self.assertIs(page["hasMore"], False)
        self.assertEqual(page["eventsCount"], 2)
        self.assertEqual(page["verification"]["status"], "broken")
        self.assertEqual(
            page["verification"]["duplicates"],
            [{"eventsIndex": 1, "sequence": 1}],
        )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.record_policy_reload("a" * 64, 1)
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("verify query must not persist"),
        ):
            status, payload = store.get_policy_events_verification(0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["eventsCount"], 1)

    def test_recovered_history_verifies_identically(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")
        store = StateStore(data_file=data_file)
        store.record_policy_reload("a" * 64, 3)
        store.record_policy_reload("b" * 64, 0)
        recovered = StateStore(data_file=data_file)
        for after in (0, 1, 2):
            status, before = store.get_policy_events_verification(after, 1)
            self.assertIs(status, HTTPStatus.OK)
            status, after_page = recovered.get_policy_events_verification(after, 1)
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(after_page["events"], before["events"])
            self.assertEqual(after_page["digest"], before["digest"])
            self.assertEqual(after_page["eventsCount"], before["eventsCount"])
            self.assertEqual(
                after_page["verification"], before["verification"]
            )
            self.assertEqual(after_page["verification"]["status"], "ok")


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

    # -- success byte contract on an empty history --

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
        self.assertEqual(list(payload["verification"]), [
            "status",
            "missingSequences",
            "duplicates",
            "outOfRange",
            "digestMismatches",
        ])
        self.assertEqual(payload["verification"], verification_ok())
        self.assertEqual(
            raw,
            b'{"events":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + payload["digest"].encode()
            + b'","eventsCount":0,"verification":{"status":"ok",'
            b'"missingSequences":[],"duplicates":[],"outOfRange":[],'
            b'"digestMismatches":[]}}\n',
        )
        self.assertTrue(raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    # -- healthy history end to end --

    def test_healthy_history_pages_with_a_full_history_conclusion(self) -> None:
        first_digest = self.reload(b'{ "admin-token" : [ "admin" ] }')
        second_digest = self.reload(b'{"z":["admin"],"r":["read"]}')
        status, first, _, _ = self.verify("?after=0&limit=1", token="z")
        self.assertEqual(status, 200)
        self.assertEqual(first["eventsCount"], 2)
        self.assertEqual(
            first["events"],
            [{"sequence": 1, "digest": first_digest, "tokens": 1}],
        )
        self.assertEqual(first["nextCursor"], 1)
        self.assertIs(first["hasMore"], True)
        self.assertEqual(first["verification"], verification_ok())
        status, second, raw, _ = self.verify("?after=1&limit=1", token="z")
        self.assertEqual(status, 200)
        self.assertEqual(
            second["events"],
            [{"sequence": 2, "digest": second_digest, "tokens": 2}],
        )
        self.assertEqual(second["nextCursor"], 2)
        self.assertIs(second["hasMore"], False)
        # The full-history digest and conclusion are page independent.
        self.assertEqual(second["digest"], first["digest"])
        self.assertEqual(second["verification"], first["verification"])
        digest_input = json.dumps(
            [
                {"sequence": 1, "digest": first_digest, "tokens": 1},
                {"sequence": 2, "digest": second_digest, "tokens": 2},
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(first["digest"], hashlib.sha256(digest_input).hexdigest())

    def test_boundary_request_is_an_empty_page_over_the_full_history(self) -> None:
        self.reload(b'{"x":["admin"]}')
        status, payload, _, _ = self.verify("?after=1&limit=10", token="x")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["eventsCount"], 1)
        self.assertEqual(payload["verification"], verification_ok())

    # -- broken history surfaced over HTTP --

    def test_broken_history_is_200_with_a_broken_conclusion(self) -> None:
        self.server.store._policy_events = [
            event(1, "a" * 64, 2),
            event(1, "b" * 64, 1),
            event(4, "not-hex", 0),
        ]
        status, payload, raw, _ = self.verify("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["eventsCount"], 3)
        verification = payload["verification"]
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["missingSequences"],
            [
                {"eventsIndex": 1, "sequence": 2},
                {"eventsIndex": 2, "sequence": 3},
            ],
        )
        self.assertEqual(
            verification["duplicates"], [{"eventsIndex": 1, "sequence": 1}]
        )
        self.assertEqual(
            verification["outOfRange"], [{"eventsIndex": 2, "sequence": 4}]
        )
        self.assertEqual(
            verification["digestMismatches"],
            [{"eventsIndex": 2, "sequence": 4}],
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
            "/v1/admin/scope-policy/audit/verify/",
            "/v1/admin/scope-policy/audit/verify/extra",
            "/v1/admin/scope-policy/audit/",
            "/v1/admin/scope-policy/verify",
            "/v1/admin/scope-policy",
            "/v1/admin/scopepolicy/audit/verify",
            "/admin/scope-policy/audit/verify",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.request(
                    "GET", path + "?bogus=1&after=0", token=ADMIN_TOKEN
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_bare_audit_sibling_still_serves_its_own_query_contract(self) -> None:
        # The bare change-audit route remains a distinct published route:
        # a verify-only bad query on it is its own 400, not a verify 404.
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?after=0&limit=1", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            ["events", "nextCursor", "hasMore", "algorithm", "digest", "eventsCount"],
        )
        status, payload, _, _ = self.request(
            "GET", AUDIT_PATH + "?bogus=1&after=0", token=ADMIN_TOKEN
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

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

    # -- read-only --

    def test_verify_query_persists_nothing(self) -> None:
        self.reload(b'{"keep":["admin"]}')
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in ("?after=0&limit=100", "?after=0&limit=1&x=1"):
            self.verify(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    # -- restart gives identical pages, digest, and conclusion --

    def test_history_survives_restart_identically(self) -> None:
        self.reload(b'{"admin-token":["admin"],"r1":["read"]}')
        self.reload(b'{"admin-token":["admin"],"r2":["read"],"r3":["read"]}')
        before = self.verify("?after=0&limit=1")[1]
        restarted_store = StateStore(data_file=self.data_path)
        restarted = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=self.data_path,
            store=restarted_store,
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        try:
            port = restarted.server_address[1]
            for query in ("?after=0&limit=1", "?after=1&limit=1", "?after=2&limit=10"):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "GET",
                    VERIFY_PATH + query,
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                conn.close()
                self.assertEqual(response.status, 200, query)
                self.assertEqual(payload["eventsCount"], before["eventsCount"])
                self.assertEqual(payload["digest"], before["digest"])
                self.assertEqual(payload["verification"], verification_ok())
        finally:
            restarted.shutdown()
            restarted.server_close()
            thread.join(timeout=5)


class PolicyEventsVerificationConcurrencyTests(unittest.TestCase):
    def test_concurrent_reloads_and_verifies_observe_consistent_snapshots(self) -> None:
        store = StateStore()
        stop = threading.Event()
        errors: list[BaseException] = []

        def reload_worker() -> None:
            try:
                for index in range(40):
                    store.record_policy_reload(f"{index:064x}", index % 5)
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        def verify_worker() -> None:
            try:
                while not stop.is_set():
                    status, payload = store.get_policy_events_verification(0, 100)
                    assert status is HTTPStatus.OK
                    count = payload["eventsCount"]
                    assert len(payload["events"]) == min(100, count)
                    assert payload["verification"]["status"] == "ok"
                    seen = [item["sequence"] for item in payload["events"]]
                    assert seen == list(range(1, len(seen) + 1))
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        verifiers = [threading.Thread(target=verify_worker) for _ in range(3)]
        for thread in verifiers:
            thread.start()
        reload_worker()
        stop.set()
        for thread in verifiers:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        status, payload = store.get_policy_events_verification(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["eventsCount"], 40)
        self.assertEqual(payload["verification"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
