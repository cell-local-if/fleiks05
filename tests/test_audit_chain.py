"""Tests for the global read-only audit-chain endpoint::

    GET /v1/audit/log/chain?after=N&limit=N

It returns exactly four fields — ``entries`` (a page of the shared
accepted-operation log in global commit order, each linking to its
predecessor by digest), ``nextCursor`` (the 0-based resume cursor),
``hasMore`` (the remainder flag), and ``head`` (the tail-of-chain
summary, 64 zeros for an empty log) — all computed from one snapshot
under the shared commit lock.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot and recovery
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    ZERO_DIGEST,
    _audit_record_bytes,
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

CHAIN_PATH = "/v1/audit/log/chain"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


def expected_link_digest(prev: str, sequence: int, replica: str, op: dict) -> str:
    """Independently compute one chain-link digest from literal pieces."""
    return hashlib.sha256(
        prev.encode("ascii") + str(sequence).encode("ascii") + _audit_record_bytes(replica, op)
    ).hexdigest()


def expected_head(records: list[tuple[str, dict]]) -> str:
    """Walk the whole chain the same way and return the tail digest."""
    prev = ZERO_DIGEST
    for index, (replica, op) in enumerate(records):
        prev = expected_link_digest(prev, index + 1, replica, op)
    return prev


class HttpServerTestCase(unittest.TestCase):
    """Spin up one in-memory server per class; reset the store per test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def raw_request(self, path: str, auth: str | None = None) -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if auth is not None:
            headers["Authorization"] = auth
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def request(self, path: str, auth: str | None = None) -> tuple[int, object]:
        status, raw, _ = self.raw_request(path, auth)
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return status, payload

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            body=json.dumps(op),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_sync(self, body: object) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/sync/operations",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_resolve(self, key: str, body: object) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/states/{key}/resolve",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_auto_resolve(self, key: str, body: object) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/states/{key}/resolve/auto",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def chain(self, query: str = "") -> tuple[int, object]:
        return self.request(f"{CHAIN_PATH}{query}")

    def drain_chain(self, limit: int = 2) -> list[dict]:
        """Page through the whole chain using the public cursor."""
        after = 0
        entries: list[dict] = []
        while True:
            status, payload = self.chain(f"?after={after}&limit={limit}")
            assert status == 200
            page = payload["entries"]
            entries.extend(page)
            assert payload["nextCursor"] == after + len(page)
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return entries


class EmptyChainTests(HttpServerTestCase):
    def test_empty_log_shape(self) -> None:
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"entries", "nextCursor", "hasMore", "head"})
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["head"], ZERO_DIGEST)

    def test_empty_log_head_is_64_zeros_and_does_not_change_with_paging(self) -> None:
        for query in ("", "?after=0", "?after=0&limit=1", "?after=0&limit=100"):
            status, payload = self.chain(query)
            self.assertEqual(status, 200)
            self.assertEqual(payload["head"], ZERO_DIGEST)
            self.assertEqual(payload["entries"], [])

    def test_success_body_is_compact_json_with_single_newline(self) -> None:
        status, raw, headers = self.raw_request(CHAIN_PATH)
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        # Compact: no insignificant whitespace.
        body = raw[:-1].decode("utf-8")
        self.assertNotIn(" ", body)
        self.assertNotIn("\n", body)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(headers.get("Content-Length"), str(len(raw)))

    def test_after_equal_to_chain_length_is_empty_page_but_keeps_head(self) -> None:
        # Empty log: after == length (0) is the default empty page.
        status, payload = self.chain("?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])


class ChainLinkTests(HttpServerTestCase):
    def test_entries_sequence_prev_and_digest_link_correctly(self) -> None:
        ops = [
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
            ("r7", operation("x1", "x", "x1", {"r7": 1})),
        ]
        for replica, op in ops:
            status, _ = self.post_operation(replica, op)
            self.assertEqual(status, 201)

        status, payload = self.chain()
        self.assertEqual(status, 200)
        entries = payload["entries"]
        self.assertEqual(len(entries), len(ops))

        prev = ZERO_DIGEST
        for index, entry in enumerate(entries):
            replica, op = ops[index]
            self.assertEqual(set(entry), {"sequence", "prevDigest", "digest"})
            self.assertEqual(entry["sequence"], index + 1)
            self.assertEqual(entry["prevDigest"], prev)
            self.assertRegex(entry["digest"], DIGEST_RE)
            want = expected_link_digest(prev, index + 1, replica, op)
            self.assertEqual(entry["digest"], want)
            prev = entry["digest"]

        # head is the tail digest and independent of paging.
        self.assertEqual(payload["head"], prev)
        status, paged = self.chain("?after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(paged["head"], prev)

    def test_first_entry_prev_digest_is_64_zeros(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"][0]["prevDigest"], ZERO_DIGEST)
        self.assertEqual(payload["head"], payload["entries"][0]["digest"])

    def test_link_input_concatenation_order(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.post_operation("r1", op)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        record_bytes = _audit_record_bytes("r1", op)
        manual = hashlib.sha256(
            ZERO_DIGEST.encode("ascii") + b"1" + record_bytes
        ).hexdigest()
        self.assertEqual(payload["entries"][0]["digest"], manual)

    def test_record_bytes_match_key_audit_canonical_rules(self) -> None:
        # Control characters and quotes/backslashes use the same minimal
        # escaping; non-ASCII is written literally as UTF-8; clock
        # components sort lexicographically.
        op = operation(
            'o"1', "k", "line\nx\ty", {"z": 1, "a": 2, "r1": 3}
        )
        self.post_operation("r1", op)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        entry = payload["entries"][0]
        self.assertEqual(
            entry["digest"],
            expected_link_digest(ZERO_DIGEST, 1, "r1", op),
        )

    def test_non_ascii_literal_utf8_in_record_bytes(self) -> None:
        op = operation("o1", "clé", "héllo→世界", {"réplica": 1, "r1": 2})
        self.post_operation("r1", op)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["entries"][0]["digest"],
            expected_link_digest(ZERO_DIGEST, 1, "r1", op),
        )

    def test_stale_writes_and_accepted_fixes_enter_the_chain(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        status, _ = self.post_resolve("k", body)
        self.assertEqual(status, 201)
        # A stale (already-dominated) write still enters the chain.
        status, _ = self.post_operation(
            "r1", operation("stale", "k", "old", {"r1": 0})
        )
        self.assertEqual(status, 201)

        entries = self.drain_chain()
        self.assertEqual([e["sequence"] for e in entries], [1, 2, 3, 4])
        # Verify each link independently against the committed log.
        with self.server.store._lock:
            accepted = list(self.server.store._accepted)
        prev = ZERO_DIGEST
        for index, ((rid, op), entry) in enumerate(zip(accepted, entries)):
            self.assertEqual(entry["sequence"], index + 1)
            self.assertEqual(entry["prevDigest"], prev)
            self.assertEqual(
                entry["digest"], expected_link_digest(prev, index + 1, rid, op)
            )
            prev = entry["digest"]
        status, payload = self.chain()
        self.assertEqual(payload["head"], prev)

    def test_sync_import_enters_chain_in_batch_order(self) -> None:
        batch = {
            "operations": [
                record("r4", operation("b1", "k", "b1", {"r4": 1})),
                record("r9", operation("bx", "x", "bx", {"r9": 1})),
                record("r4", operation("b2", "k", "b2", {"r4": 2})),
            ]
        }
        status, payload = self.post_sync(batch)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 3)
        entries = self.drain_chain(limit=1)
        self.assertEqual([e["sequence"] for e in entries], [1, 2, 3])
        with self.server.store._lock:
            accepted = list(self.server.store._accepted)
        self.assertEqual(
            [(r, o["operationId"]) for r, o in accepted],
            [("r4", "b1"), ("r9", "bx"), ("r4", "b2")],
        )
        # Links chain across the batch boundary.
        for index, entry in enumerate(entries):
            if index > 0:
                self.assertEqual(entry["prevDigest"], entries[index - 1]["digest"])

    def test_replays_rejects_and_failures_never_enter_chain(self) -> None:
        op = operation("o1", "k", "v1", {"r1": 1})
        self.post_operation("r1", op)
        # Identical replay -> 200, no record.
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 200)
        # Same identity, different content -> 409, no record.
        status, _ = self.post_operation(
            "r1", operation("o1", "k", "other", {"r1": 2})
        )
        self.assertEqual(status, 409)
        # Malformed -> 400, no record.
        status, _ = self.post_operation("r1", {"operationId": "", "key": "k"})
        self.assertEqual(status, 400)
        entries = self.drain_chain()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["sequence"], 1)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(payload["head"], entries[0]["digest"])

    def test_rejected_sync_batch_enters_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        batch = {
            "operations": [
                record("r2", operation("ok", "k", "a", {"r2": 1})),
                record("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        }
        status, payload = self.post_sync(batch)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        entries = self.drain_chain()
        self.assertEqual(len(entries), 1)

    def test_accepted_automatic_fix_enters_chain(self) -> None:
        self.post_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        body = {
            "replicaId": "r3",
            "operationId": "auto-1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        status, payload = self.post_auto_resolve("k", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["value"], "blue")
        entries = self.drain_chain()
        self.assertEqual([e["sequence"] for e in entries], [1, 2, 3])
        with self.server.store._lock:
            accepted = list(self.server.store._accepted)
        self.assertEqual([(r, o["operationId"]) for r, o in accepted][-1], ("r3", "auto-1"))

    def test_unfinished_transaction_conflict_enters_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        # The transaction expects an empty key, but k already holds a
        # candidate -> 409 transaction_conflict, the batch never completes.
        body = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "k",
                    "replicaId": "r9",
                    "operationId": "tx-op",
                    "value": "new",
                    "clock": {"r9": 1},
                    "candidates": [],
                }
            ],
        }
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/transactions/apply",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})
        entries = self.drain_chain()
        self.assertEqual([e["sequence"] for e in entries], [1])

    def test_rejected_auto_resolve_batch_enters_nothing(self) -> None:
        # A batch on a resolved (single-value) key fails resolution; the
        # whole batch is rejected and adds no chain entry.
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        body = {
            "resolutions": [
                {
                    "key": "k",
                    "replicaId": "r3",
                    "operationId": "auto-x",
                    "clock": {"r1": 1, "r3": 1},
                    "policy": "lowest_identity",
                }
            ]
        }
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/resolve/auto/batch",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        entries = self.drain_chain()
        self.assertEqual([e["sequence"] for e in entries], [1])


class ChainPaginationTests(HttpServerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ops = []
        for i in range(5):
            op = operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            self.post_operation(f"r{i}", op)
            self.ops.append((f"r{i}", op))

    def test_default_paging(self) -> None:
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["entries"]), 5)
        self.assertEqual(payload["nextCursor"], 5)
        self.assertFalse(payload["hasMore"])

    def test_pages_stitch_into_full_chain(self) -> None:
        seen: list[dict] = []
        after = 0
        while True:
            status, payload = self.chain(f"?after={after}&limit=2")
            self.assertEqual(status, 200)
            page = payload["entries"]
            seen.extend(page)
            self.assertEqual(payload["nextCursor"], after + len(page))
            if page:
                self.assertTrue(payload["hasMore"] or payload["nextCursor"] == 5)
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual([e["sequence"] for e in seen], [1, 2, 3, 4, 5])
        # Digest links survive the page boundaries.
        for index, entry in enumerate(seen):
            self.assertEqual(
                entry["prevDigest"],
                ZERO_DIGEST if index == 0 else seen[index - 1]["digest"],
            )
        self.assertEqual(seen[-1]["digest"], payload["head"])

    def test_head_identical_across_pages(self) -> None:
        heads = set()
        for after in range(6):
            status, payload = self.chain(f"?after={after}&limit=2")
            self.assertEqual(status, 200)
            heads.add(payload["head"])
        self.assertEqual(len(heads), 1)

    def test_resume_cursor_round_trip(self) -> None:
        status, first = self.chain("?after=0&limit=2")
        self.assertEqual(status, 200)
        cursor = first["nextCursor"]
        status, second = self.chain(f"?after={cursor}&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["sequence"] for e in second["entries"]], [3, 4])
        self.assertEqual(second["entries"][0]["prevDigest"], first["entries"][1]["digest"])

    def test_limit_boundaries_1_and_100(self) -> None:
        status, one = self.chain("?limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(len(one["entries"]), 1)
        self.assertTrue(one["hasMore"])
        status, hundred = self.chain("?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(hundred["entries"]), 5)
        self.assertFalse(hundred["hasMore"])

    def test_after_equal_to_length_returns_empty_page(self) -> None:
        status, payload = self.chain("?after=5")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 5)
        self.assertFalse(payload["hasMore"])
        self.assertRegex(payload["head"], DIGEST_RE)

    def test_full_page_with_remainder_reports_has_more(self) -> None:
        status, payload = self.chain("?after=0&limit=5")
        self.assertEqual(status, 200)
        self.assertFalse(payload["hasMore"])
        status, payload = self.chain("?after=0&limit=4")
        self.assertEqual(status, 200)
        self.assertTrue(payload["hasMore"])


class ChainValidationTests(HttpServerTestCase):
    def _expect_400(self, query: str) -> None:
        status, payload = self.chain(query)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_negative_after_is_400(self) -> None:
        self._expect_400("?after=-1")

    def test_blank_after_is_400(self) -> None:
        self._expect_400("?after=")
        self._expect_400("?after&limit=1")

    def test_non_ascii_digit_after_is_400(self) -> None:
        # U+2160 ROMAN NUMERAL ONE and Arabic-Indic digit, percent encoded.
        self._expect_400("?after=%E2%85%A0")
        self._expect_400("?after=%D9%A1")  # ١ U+0661

    def test_encoded_whitespace_after_is_400(self) -> None:
        self._expect_400("?after=%201")
        self._expect_400("?after=1%20")
        self._expect_400("?after=%09")

    def test_signs_decimals_and_leading_plus_are_400(self) -> None:
        self._expect_400("?after=+1")
        self._expect_400("?after=1.0")
        self._expect_400("?after=0x1")
        self._expect_400("?after=1e0")

    def test_after_past_chain_length_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self._expect_400("?after=2")

    def test_limit_boundaries_outside_1_100_are_400(self) -> None:
        self._expect_400("?limit=0")
        self._expect_400("?limit=101")
        self._expect_400("?limit=-1")
        self._expect_400("?limit=")
        self._expect_400("?limit=1.0")
        self._expect_400("?limit=%D9%A1")

    def test_unknown_parameter_is_400(self) -> None:
        self._expect_400("?x=1")
        self._expect_400("?after=0&foo=bar")

    def test_repeated_after_is_400(self) -> None:
        self._expect_400("?after=0&after=1")

    def test_repeated_limit_is_400(self) -> None:
        self._expect_400("?limit=1&limit=2")

    def test_blank_name_parameter_is_400(self) -> None:
        self._expect_400("?=1")

    def test_error_body_is_compact_json_with_explicit_length(self) -> None:
        # Validation errors share the endpoint's compact-single-line
        # contract: the JSON error envelope plus one trailing newline and
        # an explicit Content-Length.
        status, raw, headers = self.raw_request(f"{CHAIN_PATH}?after=-1")
        self.assertEqual(status, 400)
        self.assertEqual(raw, b'{"error":"invalid_request"}\n')
        self.assertEqual(headers.get("Content-Length"), str(len(raw)))


class ChainRouteShapeTests(HttpServerTestCase):
    def _expect_404(self, path: str) -> None:
        status, payload = self.raw_request(path)[:2]
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(payload), {"error": "not_found"})

    def test_trailing_slash_is_404(self) -> None:
        self._expect_404(f"{CHAIN_PATH}/")

    def test_extra_segment_is_404(self) -> None:
        self._expect_404(f"{CHAIN_PATH}/extra")

    def test_missing_segment_is_404(self) -> None:
        self._expect_404("/v1/audit/log")
        self._expect_404("/v1/audit/log/")
        self._expect_404("/v1/audit")

    def test_unknown_route_is_404(self) -> None:
        self._expect_404("/v1/audit/log/notchain")
        self._expect_404("/v1/nope")

    def test_shape_check_precedes_query_check(self) -> None:
        # A malformed path together with a malformed query is still 404.
        self._expect_404(f"{CHAIN_PATH}/extra?after=-1")
        self._expect_404(f"{CHAIN_PATH}/?after=notanint")
        self._expect_404("/v1/audit/log?after=1&x=1")

    def test_existing_audit_routes_still_resolve(self) -> None:
        # The new route must not shadow the per-key audit routes.
        status, payload = self.request("/v1/audit/keys/k/operations")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        status, payload = self.request("/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertIn("digest", payload)


class ChainConcurrencyTests(HttpServerTestCase):
    def test_concurrent_commits_and_chain_reads_stay_consistent(self) -> None:
        thread_count = 12
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                status, _ = self.post_operation(
                    f"l{index}",
                    operation(f"local-{index}", "k", "l", {f"l{index}": 1}),
                )
                assert status == 201
                batch = {
                    "operations": [
                        record(
                            f"s{index}a",
                            operation(f"sync-{index}a", "k", "a", {f"s{index}a": 1}),
                        ),
                        record(
                            f"s{index}b",
                            operation(f"sync-{index}b", "x", "b", {f"s{index}b": 1}),
                        ),
                    ]
                }
                status, payload = self.post_sync(batch)
                assert status == 201 and payload["accepted"] == 2
            except BaseException as exc:  # reported below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(80):
                    after = 0
                    while True:
                        status, payload = self.chain(f"?after={after}&limit=3")
                        assert status == 200
                        page = payload["entries"]
                        # One snapshot per request: cursor math agrees, a
                        # hasMore page is exactly full, sequences are
                        # contiguous, and both digests are well formed.
                        assert payload["nextCursor"] == after + len(page)
                        if payload["hasMore"]:
                            assert len(page) == 3
                        for position, entry in enumerate(page):
                            assert entry["sequence"] == after + position + 1
                            assert DIGEST_RE.match(entry["digest"])
                            assert DIGEST_RE.match(entry["prevDigest"])
                        after = payload["nextCursor"]
                        if not payload["hasMore"]:
                            break
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        threads.append(threading.Thread(target=reader))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])

        # Final state: one global order with a fully linked digest chain.
        entries = self.drain_chain(limit=7)
        self.assertEqual(len(entries), 3 * thread_count)
        self.assertEqual([e["sequence"] for e in entries], list(range(1, len(entries) + 1)))
        for index, entry in enumerate(entries):
            self.assertEqual(
                entry["prevDigest"],
                ZERO_DIGEST if index == 0 else entries[index - 1]["digest"],
            )
        status, payload = self.chain()
        self.assertEqual(payload["head"], entries[-1]["digest"])


class ChainSnapshotStoreTests(unittest.TestCase):
    """Direct store-level snapshot and link semantics."""

    def test_entries_page_cursor_and_head_come_from_one_call(self) -> None:
        store = StateStore()
        for i in range(3):
            store.apply_operation(
                f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
            )
        entries, next_cursor, has_more, head = store.get_audit_chain(1, 1)
        self.assertEqual([e["sequence"] for e in entries], [2])
        self.assertEqual(next_cursor, 2)
        self.assertTrue(has_more)
        # The page's link reaches the tail head through the full chain.
        full, _, _, full_head = store.get_audit_chain(0, 100)
        self.assertEqual(head, full_head)
        self.assertEqual(full[-1]["digest"], full_head)

    def test_empty_store_head_is_zero(self) -> None:
        store = StateStore()
        entries, next_cursor, has_more, head = store.get_audit_chain(0, 100)
        self.assertEqual(entries, [])
        self.assertEqual(next_cursor, 0)
        self.assertFalse(has_more)
        self.assertEqual(head, ZERO_DIGEST)

    def test_after_past_length_raises(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            store.get_audit_chain(2, 100)

    def test_head_matches_committed_log_independent_walk(self) -> None:
        store = StateStore()
        committed: list[tuple[str, dict]] = []
        for i in range(4):
            rid, op = f"r{i}", operation(f"o{i}", f"k{i % 2}", f"v{i}", {f"r{i}": 1})
            store.apply_operation(rid, op)
            committed.append((rid, op))
        _, _, _, head = store.get_audit_chain(0, 1)
        self.assertEqual(head, expected_head(committed))


class ChainPersistenceTests(unittest.TestCase):
    """The chain over a data-file-backed server, including recovery."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def chain_pages(self, server: SemanticStateServer, limit: int) -> list[dict]:
        pages = []
        after = 0
        while True:
            status, payload = self.request(
                server, "GET", f"{CHAIN_PATH}?after={after}&limit={limit}"
            )
            assert status == 200
            pages.append(payload)
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return pages

    def seed(self, server: SemanticStateServer) -> None:
        for replica, op in (
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
        ):
            status, _ = self.request(
                server, "POST", f"/v1/replicas/{replica}/operations", op
            )
            assert status == 201
        body = resolution(
            "r3",
            "fix-1",
            "k",
            "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )
        status, _ = self.request(server, "POST", "/v1/states/k/resolve", body)
        assert status == 201
        # Stale write and a sync import.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("stale", "k", "old", {"r1": 0}),
        )
        assert status == 201
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/operations",
            {
                "operations": [
                    record("r4", operation("b1", "k", "b1", {"r4": 1})),
                    record("r9", operation("bx", "x", "bx", {"r9": 1})),
                ]
            },
        )
        assert status == 201 and payload["accepted"] == 2

    def _summarize(self, pages: list[dict]) -> list[tuple]:
        return [
            (
                [
                    (e["sequence"], e["prevDigest"], e["digest"])
                    for e in page["entries"]
                ],
                page["nextCursor"],
                page["hasMore"],
                page["head"],
            )
            for page in pages
        ]

    def test_restart_preserves_order_links_cursors_and_head(self) -> None:
        server = self.start_server()
        self.seed(server)

        before_limit1 = self.chain_pages(server, 1)
        before_limit3 = self.chain_pages(server, 3)
        status, before_mid = self.request(
            server, "GET", f"{CHAIN_PATH}?after=2&limit=2"
        )
        self.assertEqual(status, 200)

        server.shutdown()
        server.server_close()
        server = self.start_server()

        after_limit1 = self.chain_pages(server, 1)
        after_limit3 = self.chain_pages(server, 3)
        status, after_mid = self.request(
            server, "GET", f"{CHAIN_PATH}?after=2&limit=2"
        )
        self.assertEqual(status, 200)

        # Same record order, link digests, page boundaries, cursors, head.
        self.assertEqual(self._summarize(after_limit1), self._summarize(before_limit1))
        self.assertEqual(self._summarize(after_limit3), self._summarize(before_limit3))
        self.assertEqual(after_mid, before_mid)

        # The tail head is 64 lowercase hex and the recovered chain has
        # all six records linked from the zero genesis.
        status, full = self.request(server, "GET", CHAIN_PATH)
        self.assertEqual(status, 200)
        self.assertEqual(len(full["entries"]), 6)
        self.assertEqual([e["sequence"] for e in full["entries"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(full["entries"][0]["prevDigest"], ZERO_DIGEST)
        for index, entry in enumerate(full["entries"]):
            if index > 0:
                self.assertEqual(
                    entry["prevDigest"], full["entries"][index - 1]["digest"]
                )
        self.assertEqual(full["head"], full["entries"][-1]["digest"])
        self.assertRegex(full["head"], DIGEST_RE)

        # Empty tail after restart.
        status, tail = self.request(server, "GET", f"{CHAIN_PATH}?after=6")
        self.assertEqual(status, 200)
        self.assertEqual(tail["entries"], [])
        self.assertEqual(tail["nextCursor"], 6)
        self.assertEqual(tail["head"], full["head"])
        status, _ = self.request(server, "GET", f"{CHAIN_PATH}?after=7")
        self.assertEqual(status, 400)

    def test_query_is_read_only_and_creates_no_temp_files(self) -> None:
        server = self.start_server()
        self.seed(server)
        before = self.data_file.read_bytes()
        names_before = set(p.name for p in self.tmp.iterdir())
        for query in ("", "?after=0&limit=1", "?after=3", "?after=0&unknown=1"):
            status, _ = self.request(server, "GET", f"{CHAIN_PATH}{query}")
            assert status in (200, 400)
        # The data file is byte-for-byte unchanged and no temp files appear.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(set(p.name for p in self.tmp.iterdir()), names_before)

    def test_failed_durable_commit_enters_no_chain_entry_or_head_change(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r0/operations",
            operation("o0", "k", "v0", {"r0": 1}),
        )
        self.assertEqual(status, 201)
        status, before = self.request(server, "GET", CHAIN_PATH)
        self.assertEqual(status, 200)
        self.assertEqual(len(before["entries"]), 1)

        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=server_module.PersistenceError("disk gone"),
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations",
                operation("o1", "k", "v1", {"r1": 1}),
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # The failed operation is neither an entry nor the head; the chain
        # is exactly as it was.
        status, after = self.request(server, "GET", CHAIN_PATH)
        self.assertEqual(status, 200)
        self.assertEqual(len(after["entries"]), 1)
        self.assertEqual(after["head"], before["head"])
        self.assertEqual([e["digest"] for e in after["entries"]],
                         [e["digest"] for e in before["entries"]])
        # It commits cleanly once persistence works again and extends the chain.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        status, final = self.request(server, "GET", CHAIN_PATH)
        self.assertEqual(status, 200)
        self.assertEqual([e["sequence"] for e in final["entries"]], [1, 2])


class ChainAuthTests(unittest.TestCase):
    TOKEN = "chain-secret-token"

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token=cls.TOKEN
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def get_raw(self, headers: list[tuple[str, str]]) -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", CHAIN_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def _assert_unauthorized(self, status: int, raw: bytes, headers: dict) -> None:
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_missing_header_is_401(self) -> None:
        status, raw, headers = self.get_raw([])
        self._assert_unauthorized(status, raw, headers)

    def test_wrong_token_is_401(self) -> None:
        status, raw, headers = self.get_raw([("Authorization", "Bearer nope")])
        self._assert_unauthorized(status, raw, headers)

    def test_malformed_header_is_401(self) -> None:
        for value in ("Bearer", f"Bearer  {self.TOKEN}", "bearer " + self.TOKEN, self.TOKEN):
            with self.subTest(value=value):
                status, raw, headers = self.get_raw([("Authorization", value)])
                self._assert_unauthorized(status, raw, headers)

    def test_duplicate_header_is_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", CHAIN_PATH)
        conn.putheader("Authorization", f"Bearer {self.TOKEN}")
        conn.putheader("Authorization", f"Bearer {self.TOKEN}")
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read()
        headers = dict(response.getheaders())
        conn.close()
        self._assert_unauthorized(response.status, raw, headers)

    def test_health_stays_anonymous_but_chain_requires_token(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        response.read()
        conn.close()
        status, raw, headers = self.get_raw([])
        self._assert_unauthorized(status, raw, headers)

    def test_authorized_chain_succeeds(self) -> None:
        status, raw, headers = self.get_raw([("Authorization", f"Bearer {self.TOKEN}")])
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["head"], ZERO_DIGEST)


if __name__ == "__main__":
    unittest.main()
