"""Tests for the global audit-chain integrity-verification endpoint::

    GET /v1/audit/log/verify?after=N&limit=N&head=H&count=N

It pages the shared accepted-operation log exactly like
``GET /v1/audit/log/chain`` (same link page, resume cursor, remaining
flag, and chain-tail ``head``) but with **required** ``after``/``limit``
paging (no defaults), and additionally requires two external expectations
— ``head`` (exactly 64 lowercase hexadecimal characters) and ``count``
(a non-negative ASCII decimal integer) — and carries an independent
``verification`` conclusion over the *complete* log. The scan checks
sequence continuity, predecessor closure, per-link digest recomputation,
and chain-tail/head-plus-count agreement; the status is ``"ok"`` exactly
when the internal chain is intact and both expectations match, otherwise
``"broken"``.

The tests cover the query parser, the independent anomaly scan (missing,
duplicate, and out-of-range sequences, broken predecessor closure and
digest recomputation, head/count expectation mismatches, each marked with
the 0-based ``linkIndex``, the 1-based ``sequence``, and the observed
value), store snapshot/paging/full-history semantics, the HTTP request
precedence chain (404 path shape, 401 authentication with a Bearer
challenge, 403 in scope mode without one, 400 query validation), the
compact canonical response body, ``--data-file`` recovery, and the
read-only/no-temp-file guarantee. The plain chain endpoint's contract is
unchanged.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _AUDIT_CHAIN_GENESIS,
    _audit_log_verification_locked,
    _audit_record_bytes,
    load_scope_policy,
    parse_audit_log_verify_query,
)

VERIFY_FIELDS = {"entries", "nextCursor", "hasMore", "head", "verification"}
ENTRY_FIELDS = {"sequence", "prevDigest", "digest"}
VERIFICATION_FIELDS = {
    "status",
    "missingSequences",
    "duplicateSequences",
    "outOfRangeSequences",
    "brokenLinks",
    "digestMismatches",
    "headMismatches",
    "countMismatches",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

GENESIS = _AUDIT_CHAIN_GENESIS

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


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


def expected_chain(records: list[tuple[str, dict]]) -> list[dict]:
    """Compute the full expected chain for (replica, operation) pairs."""
    entries = []
    previous = GENESIS
    for index, (replica, op) in enumerate(records):
        sequence = index + 1
        digest = hashlib.sha256(
            previous.encode("ascii")
            + str(sequence).encode("ascii")
            + _audit_record_bytes(replica, op)
        ).hexdigest()
        entries.append(
            {"sequence": sequence, "prevDigest": previous, "digest": digest}
        )
        previous = digest
    return entries


def materialized_entries(records: list[tuple[str, dict]]) -> list[dict]:
    """Build the claimed link list the store materializes for a raw log."""
    return expected_chain(records)


class ParseAuditLogVerifyQueryTests(unittest.TestCase):
    def test_requires_all_four_parameters(self) -> None:
        head = "a" * 64
        for query in (
            "",
            f"after=0&limit=1&head={head}",
            f"after=0&limit=1&count=0",
            f"after=0&head={head}&count=0",
            f"limit=1&head={head}&count=0",
            f"head={head}&count=0",
            f"after=0&limit=1",
        ):
            with self.subTest(query=query):
                self.assertIsNone(parse_audit_log_verify_query(query))

    def test_accepts_all_four_parameters(self) -> None:
        self.assertEqual(
            parse_audit_log_verify_query(
                "after=0&limit=100&head=" + "a" * 64 + "&count=0"
            ),
            (0, 100, "a" * 64, 0),
        )
        self.assertEqual(
            parse_audit_log_verify_query(
                "after=12&limit=7&head=" + "b" * 64 + "&count=13"
            ),
            (12, 7, "b" * 64, 13),
        )
        # Leading zeros are legal ASCII decimal; genesis head is legal.
        self.assertEqual(
            parse_audit_log_verify_query(
                "after=00&limit=0100&head=" + "0" * 64 + "&count=00"
            ),
            (0, 100, "0" * 64, 0),
        )

    def test_rejects_malformed_head(self) -> None:
        bad_heads = [
            "",  # blank
            "a" * 63,  # too short
            "a" * 65,  # too long
            "A" * 64,  # uppercase
            "g" * 64,  # non-hex
            "a" * 63 + " ",  # raw trailing space
            "a" * 63 + "%20",  # percent-decodes to a trailing space
            "%61%61" + "a" * 60,  # decodes to 62 a's — wrong length
        ]
        for head in bad_heads:
            with self.subTest(head=head):
                self.assertIsNone(
                    parse_audit_log_verify_query(f"after=0&limit=1&head={head}&count=0")
                )

    def test_rejects_malformed_count(self) -> None:
        for count in ("", "-1", "+1", "1.0", " 1", "1 ", "%C2%B9", "0x1"):
            with self.subTest(count=count):
                self.assertIsNone(
                    parse_audit_log_verify_query(
                        f"after=0&limit=1&head={'a' * 64}&count={count}"
                    )
                )

    def test_rejects_bad_after_and_limit(self) -> None:
        bad = [
            "after=&limit=1&head=" + "a" * 64 + "&count=0",
            "after=-1&limit=1&head=" + "a" * 64 + "&count=0",
            "after=1.0&limit=1&head=" + "a" * 64 + "&count=0",
            "after=+1&limit=1&head=" + "a" * 64 + "&count=0",
            "after=%C2%B9&limit=1&head=" + "a" * 64 + "&count=0",
            "after=0&limit=0&head=" + "a" * 64 + "&count=0",
            "after=0&limit=101&head=" + "a" * 64 + "&count=0",
            "after=0&limit=&head=" + "a" * 64 + "&count=0",
            "after=0&limit=-1&head=" + "a" * 64 + "&count=0",
            "after=0&limit=1.0&head=" + "a" * 64 + "&count=0",
            "after=0&after=1&limit=1&head=" + "a" * 64 + "&count=0",
            "after=0&limit=1&limit=2&head=" + "a" * 64 + "&count=0",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_audit_log_verify_query(query))

    def test_rejects_repeated_and_unknown_parameters(self) -> None:
        bad = [
            "after=0&limit=1&head=" + "a" * 64 + "&head=" + "b" * 64 + "&count=0",
            "after=0&limit=1&head=" + "a" * 64 + "&count=0&count=1",
            "after=0&limit=1&head=" + "a" * 64 + "&count=0&x=1",
            "x=1&after=0&limit=1&head=" + "a" * 64 + "&count=0",
            "after=0&limit=1&head=" + "a" * 64 + "&count=0&=",
            "?after=0&limit=1&head=" + "a" * 64 + "&count=0",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_audit_log_verify_query(query))


def build_log(store: StateStore, records: list[tuple[str, dict]]) -> None:
    for replica, op in records:
        status = store.apply_operation(replica, op)
        assert status in (201, 200), status


class AuditLogVerificationScanTests(unittest.TestCase):
    """The independent scan against a raw log and a claimed link list."""

    def setUp(self) -> None:
        self.store = StateStore()
        self.records = [
            ("r1", operation("o1", "k", "a", {"r1": 1})),
            ("r2", operation("o2", "k", "b", {"r2": 1})),
            ("r3", operation("o3", "k", "c", {"r3": 1})),
        ]
        build_log(self.store, self.records)
        self.accepted = list(self.store._accepted)
        self.correct = materialized_entries(self.accepted)
        self.head = self.correct[-1]["digest"]

    def verify(self, entries: object, head: object, count: int) -> dict:
        return _audit_log_verification_locked(self.accepted, entries, head, count)

    def test_intact_chain_matching_expectations_is_ok(self) -> None:
        result = self.verify(self.correct, self.head, 3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            set(result),
            VERIFICATION_FIELDS,
        )
        self.assertEqual(result["missingSequences"], [])
        self.assertEqual(result["duplicateSequences"], [])
        self.assertEqual(result["outOfRangeSequences"], [])
        self.assertEqual(result["brokenLinks"], [])
        self.assertEqual(result["digestMismatches"], [])
        self.assertEqual(result["headMismatches"], [])
        self.assertEqual(result["countMismatches"], [])

    def test_empty_log_verifies_ok_for_genesis_head_and_zero_count(self) -> None:
        result = _audit_log_verification_locked([], [], GENESIS, 0)
        self.assertEqual(result["status"], "ok")
        for field in (
            "missingSequences",
            "duplicateSequences",
            "outOfRangeSequences",
            "brokenLinks",
            "digestMismatches",
            "headMismatches",
            "countMismatches",
        ):
            self.assertEqual(result[field], [])

    def test_wrong_head_is_one_mismatch_external_first(self) -> None:
        result = self.verify(self.correct, "f" * 64, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["headMismatches"],
            [{"expected": "f" * 64, "observed": self.head}],
        )
        # An internal-intact chain with only a wrong head has neither a
        # broken predecessor closure nor a recomputed digest mismatch.
        self.assertEqual(result["brokenLinks"], [])
        self.assertEqual(result["digestMismatches"], [])
        self.assertEqual(result["countMismatches"], [])

    def test_wrong_count_is_one_mismatch_external_first(self) -> None:
        result = self.verify(self.correct, self.head, 2)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(result["countMismatches"], [{"expected": 2, "observed": 3}])
        self.assertEqual(result["headMismatches"], [])

    def test_empty_log_wrong_head_reports_genesis_as_observed(self) -> None:
        result = _audit_log_verification_locked([], [], "a" * 64, 0)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["headMismatches"], [{"expected": "a" * 64, "observed": GENESIS}]
        )

    def test_broken_record_digest_is_marked_with_position_and_values(self) -> None:
        tampered = [dict(link) for link in self.correct]
        tampered[1]["digest"] = "f" * 64
        result = self.verify(tampered, self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(result["brokenLinks"], [])
        self.assertEqual(
            result["digestMismatches"],
            [
                {
                    "linkIndex": 1,
                    "sequence": 2,
                    "expected": self.correct[1]["digest"],
                    "observed": "f" * 64,
                }
            ],
        )

    def test_broken_predecessor_closure_is_marked(self) -> None:
        tampered = [dict(link) for link in self.correct]
        tampered[2]["prevDigest"] = "1" * 64
        result = self.verify(tampered, self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["brokenLinks"],
            [
                {
                    "linkIndex": 2,
                    "sequence": 3,
                    "expected": self.correct[1]["digest"],
                    "observed": "1" * 64,
                }
            ],
        )

    def test_first_link_must_close_against_genesis(self) -> None:
        tampered = [dict(link) for link in self.correct]
        tampered[0]["prevDigest"] = "1" * 64
        result = self.verify(tampered, self.head, 3)
        self.assertEqual(
            result["brokenLinks"],
            [
                {
                    "linkIndex": 0,
                    "sequence": 1,
                    "expected": GENESIS,
                    "observed": "1" * 64,
                }
            ],
        )

    def test_duplicate_and_missing_sequence_are_both_marked(self) -> None:
        tampered = [dict(link) for link in self.correct]
        tampered[1]["sequence"] = 1  # duplicate of link 0; frees position 2
        result = self.verify(tampered, self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["duplicateSequences"], [{"linkIndex": 1, "sequence": 1}]
        )
        self.assertEqual(
            result["missingSequences"], [{"linkIndex": 1, "sequence": 2}]
        )
        self.assertEqual(result["outOfRangeSequences"], [])

    def test_out_of_range_sequence(self) -> None:
        for bad in (0, -1, 4, "1", 1.0, True, False, None):
            with self.subTest(bad=bad):
                tampered = [dict(link) for link in self.correct]
                tampered[0]["sequence"] = bad
                result = self.verify(tampered, self.head, 3)
                self.assertEqual(result["status"], "broken")
                self.assertEqual(
                    result["outOfRangeSequences"],
                    [{"linkIndex": 0, "sequence": bad}],
                )

    def test_missing_claimed_link_is_out_of_range_and_leaves_a_gap(self) -> None:
        # A claimed chain shorter than the raw log: the absent third link
        # has no sequence (None) and leaves position 3 unclaimed.
        result = self.verify(self.correct[:2], self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["outOfRangeSequences"], [{"linkIndex": 2, "sequence": None}]
        )
        self.assertEqual(
            result["missingSequences"], [{"linkIndex": 2, "sequence": 3}]
        )

    def test_extra_claimed_link_beyond_the_log_is_out_of_range(self) -> None:
        # A claimed chain longer than the raw log: the fourth link claims a
        # sequence past the 3-link range and there is no record to recompute.
        extra = [dict(link) for link in self.correct]
        extra.append(
            {
                "sequence": 4,
                "prevDigest": self.head,
                "digest": "d" * 64,
            }
        )
        result = self.verify(extra, self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["outOfRangeSequences"], [{"linkIndex": 3, "sequence": 4}]
        )
        self.assertEqual(result["countMismatches"], [])

    def test_multiple_anomaly_lists_combine_independently(self) -> None:
        tampered = [dict(link) for link in self.correct]
        tampered[0]["digest"] = "0" * 64  # digest mismatch
        tampered[2]["sequence"] = 2  # duplicate of 1; frees position 3
        result = self.verify(tampered, "9" * 64, 7)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(result["brokenLinks"], [])
        self.assertEqual(
            result["digestMismatches"],
            [
                {
                    "linkIndex": 0,
                    "sequence": 1,
                    "expected": self.correct[0]["digest"],
                    "observed": "0" * 64,
                }
            ],
        )
        self.assertEqual(
            result["duplicateSequences"], [{"linkIndex": 2, "sequence": 2}]
        )
        self.assertEqual(
            result["missingSequences"], [{"linkIndex": 2, "sequence": 3}]
        )
        self.assertEqual(
            result["headMismatches"], [{"expected": "9" * 64, "observed": self.head}]
        )
        self.assertEqual(result["countMismatches"], [{"expected": 7, "observed": 3}])

    def test_closure_break_and_digest_mismatch_are_reported_separately(self) -> None:
        tampered = [dict(link) for link in self.correct]
        # Link 2 closes off a false predecessor and carries a bad digest.
        tampered[1]["prevDigest"] = "1" * 64
        tampered[1]["digest"] = "f" * 64
        result = self.verify(tampered, self.head, 3)
        self.assertEqual(result["status"], "broken")
        self.assertEqual(
            result["brokenLinks"],
            [
                {
                    "linkIndex": 1,
                    "sequence": 2,
                    "expected": self.correct[0]["digest"],
                    "observed": "1" * 64,
                }
            ],
        )
        self.assertEqual(
            result["digestMismatches"],
            [
                {
                    "linkIndex": 1,
                    "sequence": 2,
                    "expected": self.correct[1]["digest"],
                    "observed": "f" * 64,
                }
            ],
        )


class AuditLogVerifyStoreTests(unittest.TestCase):
    """Snapshot, paging, and full-history semantics against StateStore."""

    def setUp(self) -> None:
        self.store = StateStore()

    def commit(self, replica: str, op: dict) -> None:
        status = self.store.apply_operation(replica, op)
        self.assertIn(status, (201, 200))

    def test_empty_log_matching_expectations_verifies_ok(self) -> None:
        report = self.store.get_audit_log_verify(0, 100, GENESIS, 0)
        self.assertEqual(set(report), VERIFY_FIELDS)
        self.assertEqual(report["entries"], [])
        self.assertEqual(report["nextCursor"], 0)
        self.assertIs(report["hasMore"], False)
        self.assertEqual(report["head"], GENESIS)
        self.assertEqual(report["verification"]["status"], "ok")

    def test_page_matches_chain_query_and_verifies_ok(self) -> None:
        records = [
            ("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i}))
            for i in range(1, 6)
        ]
        build_log(self.store, records)
        chain_entries, _, _, head = self.store.get_audit_log_chain(0, 100)
        report = self.store.get_audit_log_verify(0, 100, head, 5)
        self.assertEqual(report["entries"], chain_entries)
        self.assertEqual(report["entries"], expected_chain(records))
        self.assertEqual(report["nextCursor"], 5)
        self.assertIs(report["hasMore"], False)
        self.assertEqual(report["head"], head)
        self.assertEqual(report["verification"]["status"], "ok")

    def test_paging_trims_only_entries_not_the_conclusion(self) -> None:
        records = [
            ("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i}))
            for i in range(1, 5)
        ]
        build_log(self.store, records)
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        seen: list[dict] = []
        after = 0
        while True:
            report = self.store.get_audit_log_verify(after, 2, head, 4)
            self.assertEqual(report["head"], head)
            self.assertEqual(report["verification"]["status"], "ok")
            self.assertEqual(report["verification"]["brokenLinks"], [])
            self.assertEqual(report["verification"]["digestMismatches"], [])
            seen.extend(report["entries"])
            after = report["nextCursor"]
            if not report["hasMore"]:
                break
        self.assertEqual(seen, expected_chain(records))

    def test_empty_tail_still_verifies_the_full_log(self) -> None:
        self.commit("r1", operation("o1", "k", "v", {"r1": 1}))
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        report = self.store.get_audit_log_verify(1, 10, head, 1)
        self.assertEqual(report["entries"], [])
        self.assertEqual(report["nextCursor"], 1)
        self.assertIs(report["hasMore"], False)
        self.assertEqual(report["head"], head)
        self.assertEqual(report["verification"]["status"], "ok")

    def test_wrong_expectations_report_broken_on_every_page(self) -> None:
        records = [
            ("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i}))
            for i in range(1, 4)
        ]
        build_log(self.store, records)
        for after, page_size in ((0, 2), (2, 2)):
            report = self.store.get_audit_log_verify(after, page_size, "a" * 64, 9)
            self.assertEqual(report["head"], expected_chain(records)[-1]["digest"])
            verification = report["verification"]
            self.assertEqual(verification["status"], "broken")
            self.assertEqual(
                verification["headMismatches"],
                [{"expected": "a" * 64, "observed": report["head"]}],
            )
            self.assertEqual(verification["countMismatches"], [{"expected": 9, "observed": 3}])

    def test_after_past_chain_length_raises(self) -> None:
        self.commit("r1", operation("o1", "k", "v", {"r1": 1}))
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        with self.assertRaises(ValueError):
            self.store.get_audit_log_verify(2, 100, head, 1)

    def test_stale_writes_repairs_and_imports_are_verified(self) -> None:
        self.commit("r1", operation("o1", "k", "a", {"r1": 2, "r2": 1}))
        self.commit("r2", operation("o2", "k", "b", {"r2": 1}))  # stale
        self.commit("r3", operation("o3", "k", "c", {"r3": 5}))
        status, _ = self.store.apply_resolution(
            "k",
            resolution(
                "r4",
                "o4",
                "k",
                "fixed",
                {"r1": 2, "r2": 1, "r3": 5, "r4": 1},
                [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r3", "operationId": "o3"},
                ],
            ),
        )
        self.assertEqual(status, 201)
        imported = [
            ("r5", operation("o5", "other", "x", {"r5": 1})),
            ("r6", operation("o6", "other", "y", {"r6": 1})),
        ]
        self.assertEqual(self.store.import_operations(imported), (201, 2, 0))
        records = [
            ("r1", operation("o1", "k", "a", {"r1": 2, "r2": 1})),
            ("r2", operation("o2", "k", "b", {"r2": 1})),
            ("r3", operation("o3", "k", "c", {"r3": 5})),
            (
                "r4",
                operation(
                    "o4",
                    "k",
                    "fixed",
                    {"r1": 2, "r2": 1, "r3": 5, "r4": 1},
                ),
            ),
            ("r5", operation("o5", "other", "x", {"r5": 1})),
            ("r6", operation("o6", "other", "y", {"r6": 1})),
        ]
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        report = self.store.get_audit_log_verify(0, 100, head, 6)
        self.assertEqual(report["entries"], expected_chain(records))
        self.assertEqual(report["verification"]["status"], "ok")

    def test_replays_and_rejections_never_enter_verification(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.commit("r1", op)
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        before = self.store.get_audit_log_verify(0, 100, head, 1)
        self.assertEqual(self.store.apply_operation("r1", op), 200)  # replay
        self.assertEqual(
            self.store.apply_operation(
                "r1", operation("o1", "k", "other", {"r1": 1})
            ),
            409,
        )
        after = self.store.get_audit_log_verify(0, 100, head, 1)
        self.assertEqual(after, before)

    def test_recovery_reproduces_page_head_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            records = [
                ("r1", operation("o1", "a", "x", {"r1": 1})),
                ("r2", operation("o2", "b", "y", {"r2": 1})),
                ("r1", operation("o3", "a", "z", {"r1": 2})),
            ]
            build_log(store, records)
            _, _, _, head = store.get_audit_log_chain(0, 100)
            before = store.get_audit_log_verify(1, 1, head, 3)
            recovered = StateStore(data_file=data_file)
            after = recovered.get_audit_log_verify(1, 1, head, 3)
            self.assertEqual(after, before)
            self.assertEqual(after["verification"]["status"], "ok")


class AuditLogVerifyHttpTests(unittest.TestCase):
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

    def raw_request(
        self, method: str, path: str, body: object = None, headers: dict | None = None
    ) -> tuple[int, object, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=request_headers)
        elif isinstance(body, (bytes, str)):
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers=request_headers,
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def verify(self, query: str) -> tuple[int, dict, bytes, dict]:
        return self.raw_request("GET", f"/v1/audit/log/verify{query}")

    def chain(self, query: str = "") -> tuple[int, dict]:
        return self.request("GET", f"/v1/audit/log/chain{query}")

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_sync(self, body: object) -> tuple[int, dict]:
        return self.request("POST", "/v1/sync/operations", body)

    def seed(self, count: int = 5) -> tuple[list[tuple[str, dict]], str]:
        records = [
            ("r1", operation(f"o{i}", f"k{i}", f"v{i}", {"r1": i}))
            for i in range(1, count + 1)
        ]
        for replica, op in records:
            status, _ = self.post_operation(replica, op)
            assert status == 201
        _, chain_payload = self.chain()
        return records, chain_payload["head"]

    def test_empty_log_matching_genesis_is_compact_canonical_json(self) -> None:
        status, payload, raw, headers = self.verify(
            f"?after=0&limit=100&head={GENESIS}&count=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), VERIFY_FIELDS)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["head"], GENESIS)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertEqual(set(payload["verification"]), VERIFICATION_FIELDS)
        # Compact canonical JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(
            raw[:-1],
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        )
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_verify_page_matches_chain_and_concludes_ok(self) -> None:
        records, head = self.seed(3)
        status, payload, _, _ = self.verify(f"?after=0&limit=100&head={head}&count=3")
        self.assertEqual(status, 200)
        chain_status, chain_payload = self.chain()
        self.assertEqual(chain_status, 200)
        for field in ("entries", "nextCursor", "hasMore", "head"):
            self.assertEqual(payload[field], chain_payload[field])
        self.assertEqual(payload["entries"], expected_chain(records))
        self.assertEqual(payload["head"], head)
        self.assertEqual(payload["verification"]["status"], "ok")
        for entry in payload["entries"]:
            self.assertEqual(set(entry), ENTRY_FIELDS)
            self.assertRegex(entry["prevDigest"], DIGEST_RE)
            self.assertRegex(entry["digest"], DIGEST_RE)

    def test_paging_walks_with_a_page_independent_conclusion(self) -> None:
        records, head = self.seed(5)
        seen: list[dict] = []
        after = 0
        while True:
            status, payload, _, _ = self.verify(
                f"?after={after}&limit=2&head={head}&count=5"
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload["head"], head)
            self.assertEqual(payload["verification"]["status"], "ok")
            seen.extend(payload["entries"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual(seen, expected_chain(records))
        self.assertEqual(after, 5)

    def test_after_equal_to_chain_length_is_an_empty_verified_page(self) -> None:
        _, head = self.seed(2)
        status, payload, raw, _ = self.verify(f"?after=2&limit=100&head={head}&count=2")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["head"], head)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_wrong_head_reports_broken_with_external_and_observed(self) -> None:
        _, head = self.seed(2)
        status, payload, _, _ = self.verify(f"?after=0&limit=100&head={'a' * 64}&count=2")
        self.assertEqual(status, 200)
        verification = payload["verification"]
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["headMismatches"],
            [{"expected": "a" * 64, "observed": head}],
        )
        self.assertEqual(verification["countMismatches"], [])
        self.assertEqual(verification["brokenLinks"], [])
        self.assertEqual(verification["digestMismatches"], [])

    def test_wrong_count_reports_broken(self) -> None:
        _, head = self.seed(2)
        status, payload, _, _ = self.verify(f"?after=0&limit=100&head={head}&count=3")
        self.assertEqual(status, 200)
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(
            payload["verification"]["countMismatches"],
            [{"expected": 3, "observed": 2}],
        )

    def test_stale_writes_repairs_and_imports_are_verified_over_http(self) -> None:
        records: list[tuple[str, dict]] = []
        for replica, op in (
            ("r1", operation("o1", "k", "a", {"r1": 2, "r2": 1})),
            ("r2", operation("o2", "k", "b", {"r2": 1})),
            ("r3", operation("o3", "k", "c", {"r3": 5})),
        ):
            status, _ = self.post_operation(replica, op)
            assert status == 201
            records.append((replica, op))
        repair = resolution(
            "r4",
            "o4",
            "k",
            "fixed",
            {"r1": 2, "r2": 1, "r3": 5, "r4": 1},
            [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r3", "operationId": "o3"},
            ],
        )
        status, _ = self.request("POST", "/v1/states/k/resolve", repair)
        self.assertEqual(status, 201)
        records.append(
            (
                "r4",
                operation(
                    "o4", "k", "fixed", {"r1": 2, "r2": 1, "r3": 5, "r4": 1}
                ),
            )
        )
        imported = [
            record("r5", operation("o5", "other", "x", {"r5": 1})),
            record("r6", operation("o6", "other", "y", {"r6": 1})),
        ]
        status, _ = self.post_sync({"operations": imported})
        self.assertEqual(status, 201)
        records.extend(
            [("r5", imported[0]["operation"]), ("r6", imported[1]["operation"])]
        )
        _, chain_payload = self.chain()
        head = chain_payload["head"]
        status, payload, _, _ = self.verify(f"?after=0&limit=100&head={head}&count=6")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], expected_chain(records))
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_replays_and_rejections_never_enter_verification_over_http(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        _, chain_payload = self.chain()
        head = chain_payload["head"]
        status, before, _, _ = self.verify(f"?after=0&limit=100&head={head}&count=1")
        self.assertEqual(status, 200)
        self.post_operation("r1", op)  # replay -> 200
        self.post_operation("r1", operation("o1", "k", "x", {"r1": 1}))  # 409
        self.post_operation("r1", {"operationId": "bad"})  # 400
        self.post_sync(
            {"operations": [record("r1", operation("o1", "k", "y", {"r1": 1}))]}
        )  # 409
        status, after, _, _ = self.verify(f"?after=0&limit=100&head={head}&count=1")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        self.assertEqual(len(after["entries"]), 1)

    # -- query validation --

    def test_missing_or_malformed_parameters_are_400(self) -> None:
        _, head = self.seed(1)
        bad_queries = [
            "",
            f"?head={head}&count=1",  # missing after and limit
            f"?after=0&head={head}&count=1",  # missing limit
            f"?limit=1&head={head}&count=1",  # missing after
            f"?after=0&limit=1&head={head}",  # missing count
            f"?after=0&limit=1&count=1",  # missing head
            f"?after=0&limit=1&head=&count=1",
            f"?after=0&limit=1&head={head}&count=",
            f"?after=0&limit=1&head={'A' * 64}&count=1",  # uppercase
            f"?after=0&limit=1&head={'g' * 64}&count=1",  # non-hex
            f"?after=0&limit=1&head={head[:-1]}&count=1",  # 63 chars
            f"?after=0&limit=1&head={head}x&count=1",  # 65 chars
            f"?after=0&limit=1&head={head}&count=-1",
            f"?after=0&limit=1&head={head}&count=1.0",
            f"?after=0&limit=1&head={head}&count=%201",
            f"?after=&limit=1&head={head}&count=1",
            f"?after=-1&limit=1&head={head}&count=1",
            f"?after=0&limit=0&head={head}&count=1",
            f"?after=0&limit=101&head={head}&count=1",
            f"?after=0&limit=1&head={head}&count=1&x=1",  # unknown
            f"?after=0&limit=1&head={head}&head={'b' * 64}&count=1",  # repeated head
            f"?after=0&limit=1&head={head}&count=1&count=2",  # repeated count
            f"?after=0&after=1&limit=1&head={head}&count=1",
            f"?after=0&limit=1&limit=2&head={head}&count=1",
            f"?x&after=0&limit=1&head={head}&count=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.verify(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_chain_length_is_400(self) -> None:
        _, head = self.seed(1)
        status, payload, _, _ = self.verify(f"?after=2&limit=100&head={head}&count=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- route shape precedes the query check --

    def test_route_shape_mismatches_are_404(self) -> None:
        _, head = self.seed(1)
        bad_paths = [
            "/v1/audit/log",
            "/v1/audit/log/verify/",
            "/v1/audit/log/verify/extra",
            "/v1/audit/verify",
            "/v1/audit/log/unknown",
            "/v1/audit/log/chainx",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?head={head}&count=1"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/audit/log/verify/extra?head=x&count=nope"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_chain_route_keeps_its_own_contract(self) -> None:
        self.seed(1)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"entries", "nextCursor", "hasMore", "head"})
        self.assertNotIn("verification", payload)
        # A verify-only parameter on the plain chain route is still rejected.
        status, payload = self.chain("?head=" + "a" * 64)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_post_to_verify_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/audit/log/verify", {"operations": []}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only(self) -> None:
        _, head = self.seed(2)
        status, metrics_before = self.request("GET", "/v1/metrics")
        status, sync_before = self.request("GET", "/v1/sync/operations")
        self.verify(f"?after=0&limit=100&head={head}&count=2")
        self.verify(f"?after=1&limit=1&head={head}&count=2")
        self.verify(f"?after=0&limit=100&head={'a' * 64}&count=2")
        self.verify(f"?after=0&limit=100&head={head}&count=99")
        status, metrics_after = self.request("GET", "/v1/metrics")
        status, sync_after = self.request("GET", "/v1/sync/operations")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(sync_after, sync_before)
        _, chain_payload = self.chain()
        self.assertEqual(chain_payload["head"], head)


class AuditLogVerifyAuthTests(unittest.TestCase):
    """The verify endpoint authenticates like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-verify-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret-token"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]

        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(SCOPE_POLICY, handle)
        cls.scope_server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
        )
        cls.scope_thread = threading.Thread(
            target=cls.scope_server.serve_forever, daemon=True
        )
        cls.scope_thread.start()
        cls.scope_port = cls.scope_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.single_server.shutdown()
        cls.single_server.server_close()
        cls.scope_server.shutdown()
        cls.scope_server.server_close()
        cls.single_thread.join(timeout=5)
        cls.scope_thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def get(
        self, port: int, path: str, headers: list[tuple[str, str]] | None = None
    ) -> tuple[int, object, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers=dict(headers or []))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, response_headers

    def test_single_token_missing_duplicate_bad_or_wrong_is_401(self) -> None:
        path = f"/v1/audit/log/verify?head={'a' * 64}&count=0"
        # Missing.
        status, payload, headers = self.get(self.single_port, path)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Wrong token.
        status, _, headers = self.get(
            self.single_port, path, [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Malformed scheme.
        status, _, headers = self.get(
            self.single_port, path, [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # Duplicated header (sent without collapsing via putheader).
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("GET", path)
        conn.putheader("Authorization", "Bearer s3cret-token")
        conn.putheader("Authorization", "Bearer s3cret-token")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_single_token_valid_is_200(self) -> None:
        path = f"/v1/audit/log/verify?after=0&limit=100&head={'0' * 64}&count=0"
        status, payload, _ = self.get(
            self.single_port, path, [("Authorization", "Bearer s3cret-token")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), VERIFY_FIELDS)

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        path = f"/v1/audit/log/verify?head={'0' * 64}&count=0"
        status, payload, headers = self.get(
            self.scope_port, path, [("Authorization", f"Bearer {WRITE_TOKEN}")]
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_is_200(self) -> None:
        path = f"/v1/audit/log/verify?after=0&limit=100&head={'0' * 64}&count=0"
        status, payload, _ = self.get(
            self.scope_port, path, [("Authorization", f"Bearer {READ_TOKEN}")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_scope_decision_precedes_the_query_check(self) -> None:
        # A malformed query is still 403 for a token lacking the read scope.
        status, _, headers = self.get(
            self.scope_port,
            "/v1/audit/log/verify?head=nope",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
