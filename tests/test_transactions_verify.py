"""Tests for the read-only transaction-ledger audit::

    GET /v1/transactions/verify

There is no plain transaction-history query: this verify endpoint pages
the committed atomic-transaction bindings in creation order through the
required ``after``/``limit`` paging parameters and additionally requires
two external expectations — ``expectedDigest`` (exactly 64 lowercase
hexadecimal characters) and ``expectedCount`` (a non-negative ASCII
decimal integer). Its ``verification`` independently scans the
**complete** ledger:

- ``duplicateTransactionIds``, ``recordViolations``,
  ``identityMismatches``, and ``batchViolations`` are the internal
  anomaly lists (each internal marker keeps the 0-based
  ``transactionIndex`` and the ``transactionId``; an identity problem
  additionally carries the 0-based ``operationIndex`` and the
  ``expected``/``observed`` operation content);
- ``digestMismatches`` and ``countMismatches`` report external
  disagreement with at most one ``{"expected", "observed"}`` marker
  each (the caller's expectation first, the independently recomputed
  value second).

``status`` is ``ok`` exactly when all six lists are empty. Paging trims
only the exported page; the digest, count, and conclusion always cover
the complete ledger. The tests cover the query parser, the independent
scan and canonical digest (sorted clock components and candidate
identities, original operation order), the store's
paging/snapshot/recovery semantics, the HTTP request precedence chain
(404 path shape before the query check, 401 authentication with a
Bearer challenge, 403 in scope mode without one, 400 query validation
including an ``after`` past the transaction count), the compact ordered
single-newline response body with its explicit Content-Length, restart
consistency, and the strict read-only guarantee. Only the Python
standard library is used.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _transactions_digest_input,
    _transactions_verification_locked,
    load_scope_policy,
    parse_transactions_verify_query,
)

VERIFY_PATH = "/v1/transactions/verify"
APPLY_PATH = "/v1/transactions/apply"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

DIGEST_64_A = "a" * 64
DIGEST_64_B = "b" * 64
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()

RESPONSE_FIELDS = [
    "transactions",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "transactionsCount",
    "verification",
]
VERIFICATION_FIELDS = [
    "status",
    "duplicateTransactionIds",
    "recordViolations",
    "identityMismatches",
    "batchViolations",
    "digestMismatches",
    "countMismatches",
]
ANOMALY_FIELDS = VERIFICATION_FIELDS[1:]
OK_VERIFICATION = {name: [] for name in ANOMALY_FIELDS}
OK_VERIFICATION["status"] = "ok"
OK_VERIFICATION = {name: OK_VERIFICATION[name] for name in VERIFICATION_FIELDS}


def entry(
    key: str = "k1",
    replica: str = "r3",
    operation_id: str = "o1",
    value: str = "blue",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {replica: 1},
        "candidates": list(candidates) if candidates is not None else [],
    }


def tx_record(transaction_id: str, operations: list) -> dict:
    return {"transactionId": transaction_id, "operations": operations}


def accepted_op(replica: str, operation_id: str, key: str, value: str, clock: dict):
    return replica, {
        "operationId": operation_id,
        "key": key,
        "value": value,
        "clock": clock,
    }


def verify_query(after: int, limit: int, digest: str, count: int) -> str:
    return (
        f"?after={after}&limit={limit}"
        f"&expectedDigest={digest}&expectedCount={count}"
    )


class ParseTransactionsVerifyQueryTests(unittest.TestCase):
    def test_accepts_all_four_required_parameters(self) -> None:
        self.assertEqual(
            parse_transactions_verify_query(
                f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
            ),
            (0, 1, DIGEST_64_A, 0),
        )
        # Parameter order is insignificant; leading zeros parse normally.
        self.assertEqual(
            parse_transactions_verify_query(
                f"expectedCount=09&limit=09&after=007&expectedDigest={DIGEST_64_B}"
            ),
            (7, 9, DIGEST_64_B, 9),
        )

    def test_rejects_missing_repeated_unknown_and_blank(self) -> None:
        base = f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
        bad = [
            "",
            "?",
            "after=0&limit=1",
            "after=0&limit=1&expectedCount=0",
            "after=0&limit=1&expectedDigest=" + DIGEST_64_A,
            f"limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            base + "&after=2",
            base + "&limit=2",
            base + "&expectedDigest=" + DIGEST_64_B,
            base + "&expectedCount=1",
            base + "&x=1",
            "x=1&" + base,
            base.replace("after=0", "after="),
            base.replace("limit=1", "limit="),
            base.replace("expectedCount=0", "expectedCount="),
            base.replace("expectedDigest=" + DIGEST_64_A, "expectedDigest="),
            base.replace("after=0", "after"),
            base.replace("&expectedCount=0", "&expectedCount"),
            base + "&=",
            "=1&" + base,
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_transactions_verify_query(query))

    def test_rejects_malformed_expected_digest(self) -> None:
        bad_digests = [
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,  # uppercase rejected
            "g" * 64,  # non-hex rejected
            "a" * 63 + "G",
            " " + "a" * 63,
        ]
        for digest in bad_digests:
            with self.subTest(digest=digest[:10]):
                query = f"after=0&limit=1&expectedDigest={digest}&expectedCount=0"
                self.assertIsNone(parse_transactions_verify_query(query))

    def test_rejects_signed_decimal_non_ascii_and_out_of_range(self) -> None:
        base = f"limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
        bad = [
            f"after=-1&{base}",
            f"after=+0&{base}",
            f"after=1.0&{base}",
            f"after=%200&{base}",
            f"after=%C2%B2&{base}",
            f"after=0&limit=0&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=101&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=-1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=1.0&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=%D9%A1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=-1",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=+0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=1.0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0%20",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_transactions_verify_query(query))


class TransactionsVerificationScanTests(unittest.TestCase):
    def test_empty_ledger_ok_for_empty_array_digest_and_zero_count(self) -> None:
        self.assertEqual(
            _transactions_verification_locked([], [], EMPTY_DIGEST, 0),
            OK_VERIFICATION,
        )

    def test_well_formed_ledger_matching_expectations_is_ok(self) -> None:
        accepted = [
            accepted_op("r3", "o1", "k1", "blue", {"r3": 1}),
            accepted_op("r3", "o2", "k2", "red", {"r3": 2}),
        ]
        records = [
            tx_record(
                "tx-1",
                [entry("k1", "r3", "o1", "blue", {"r3": 1})],
            ),
            tx_record(
                "tx-2",
                [entry("k2", "r3", "o2", "red", {"r3": 2})],
            ),
        ]
        digest = hashlib.sha256(_transactions_digest_input(records)).hexdigest()
        verdict = _transactions_verification_locked(records, accepted, digest, 2)
        self.assertEqual(verdict, OK_VERIFICATION)
        self.assertEqual(list(verdict), VERIFICATION_FIELDS)

    def test_digest_mismatch_marks_expected_first_observed_second(self) -> None:
        verdict = _transactions_verification_locked([], [], DIGEST_64_B, 0)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": EMPTY_DIGEST}],
        )
        self.assertEqual(verdict["countMismatches"], [])

    def test_count_mismatch_marks_expected_first_observed_second(self) -> None:
        accepted = [accepted_op("r3", "o1", "k1", "blue", {"r3": 1})]
        records = [tx_record("tx-1", [entry()])]
        verdict = _transactions_verification_locked(records, accepted, EMPTY_DIGEST, 5)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["countMismatches"], [{"expected": 5, "observed": 1}])
        self.assertEqual(len(verdict["digestMismatches"]), 1)
        self.assertEqual(verdict["digestMismatches"][0]["expected"], EMPTY_DIGEST)

    def test_duplicate_transaction_id_marks_only_the_later_occurrence(self) -> None:
        accepted = [
            accepted_op("r3", "o1", "k1", "blue", {"r3": 1}),
            accepted_op("r3", "o2", "k2", "red", {"r3": 2}),
        ]
        records = [
            tx_record("tx-1", [entry("k1", "r3", "o1")]),
            tx_record("tx-1", [entry("k2", "r3", "o2", "red", {"r3": 2})]),
        ]
        digest = hashlib.sha256(_transactions_digest_input(records)).hexdigest()
        verdict = _transactions_verification_locked(records, accepted, digest, 2)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["duplicateTransactionIds"],
            [{"transactionIndex": 1, "transactionId": "tx-1"}],
        )
        self.assertEqual(verdict["recordViolations"], [])

    def test_record_violation_covers_bad_shapes(self) -> None:
        cases = [
            tx_record("", [entry()]),  # empty transaction id
            tx_record("tx-1", []),  # empty operations list
            {"transactionId": "tx-1", "operations": [entry()]},  # placeholder
            "not-a-record",
            None,
        ]
        # Replace the placeholder with an entry carrying an illegal clock.
        cases[2] = tx_record(
            "tx-1", [entry(clock={"r3": -1})]
        )
        for index, record in enumerate(cases):
            with self.subTest(index=index):
                digest = hashlib.sha256(
                    _transactions_digest_input([record])
                ).hexdigest()
                verdict = _transactions_verification_locked(
                    [record], [], digest, 1
                )
                self.assertEqual(verdict["status"], "broken")
                self.assertEqual(len(verdict["recordViolations"]), 1)
                marker = verdict["recordViolations"][0]
                self.assertEqual(marker["transactionIndex"], 0)
                self.assertIn("transactionId", marker)
        # A non-object record carries a null transaction id.
        digest = hashlib.sha256(_transactions_digest_input(["x"])).hexdigest()
        verdict = _transactions_verification_locked(["x"], [], digest, 1)
        self.assertIsNone(verdict["recordViolations"][0]["transactionId"])

    def test_identity_mismatch_names_operation_index_and_both_sides(self) -> None:
        # The identity was never accepted: expected is null.
        records = [tx_record("tx-1", [entry("k1", "r3", "o1")])]
        digest = hashlib.sha256(_transactions_digest_input(records)).hexdigest()
        verdict = _transactions_verification_locked(records, [], digest, 1)
        self.assertEqual(verdict["status"], "broken")
        marker = verdict["identityMismatches"][0]
        self.assertEqual(marker["transactionIndex"], 0)
        self.assertEqual(marker["transactionId"], "tx-1")
        self.assertEqual(marker["operationIndex"], 0)
        self.assertIsNone(marker["expected"])
        self.assertEqual(
            marker["observed"],
            {
                "key": "k1",
                "replicaId": "r3",
                "operationId": "o1",
                "value": "blue",
                "clock": {"r3": 1},
            },
        )

        # The identity is accepted but the stored content drifted: the
        # accepted content is expected first, the ledger content observed
        # second; the expected candidate set is part of neither side.
        accepted = [accepted_op("r3", "o1", "k1", "blue", {"r3": 1})]
        drifted = tx_record(
            "tx-1",
            [entry("k1", "r3", "o1", "green", {"r3": 1}, candidates=[])],
        )
        digest = hashlib.sha256(_transactions_digest_input([drifted])).hexdigest()
        verdict = _transactions_verification_locked([drifted], accepted, digest, 1)
        marker = verdict["identityMismatches"][0]
        self.assertEqual(
            marker["expected"],
            {
                "key": "k1",
                "replicaId": "r3",
                "operationId": "o1",
                "value": "blue",
                "clock": {"r3": 1},
            },
        )
        self.assertEqual(marker["observed"]["value"], "green")
        self.assertEqual(verdict["batchViolations"], [])

    def test_batch_violation_flags_repeated_key_or_identity_once(self) -> None:
        accepted = [
            accepted_op("r3", "o1", "k1", "blue", {"r3": 1}),
            accepted_op("r3", "o2", "k1", "red", {"r3": 2}),
        ]
        # Same key twice (distinct identities) — well formed as individual
        # records, illegal as one transaction batch.
        records = [
            tx_record(
                "tx-1",
                [
                    entry("k1", "r3", "o1", "blue", {"r3": 1}),
                    entry("k1", "r3", "o2", "red", {"r3": 2}),
                ],
            )
        ]
        digest = hashlib.sha256(_transactions_digest_input(records)).hexdigest()
        verdict = _transactions_verification_locked(records, accepted, digest, 1)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["batchViolations"],
            [{"transactionIndex": 0, "transactionId": "tx-1"}],
        )
        # Exactly one batch marker no matter how many repeats.
        self.assertEqual(len(verdict["batchViolations"]), 1)

    def test_internal_anomaly_breaks_even_when_expectations_agree(self) -> None:
        records = [tx_record("tx-1", [entry("k1", "r3", "o1")])]
        digest = hashlib.sha256(_transactions_digest_input(records)).hexdigest()
        verdict = _transactions_verification_locked(records, [], digest, 1)
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])
        self.assertTrue(verdict["identityMismatches"])


class TransactionsDigestInputTests(unittest.TestCase):
    def test_empty_ledger_serializes_to_empty_array(self) -> None:
        self.assertEqual(_transactions_digest_input([]), b"[]")

    def test_fields_are_fixed_order_and_operations_keep_their_order(self) -> None:
        records = [
            tx_record(
                "tx-1",
                [
                    entry("k1", "r3", "o1", "blue", {"r3": 1}),
                    entry("k2", "r3", "o2", "red", {"r3": 2}),
                ],
            )
        ]
        self.assertEqual(
            _transactions_digest_input(records),
            b'[{"transactionId":"tx-1","operations":['
            b'{"key":"k1","replicaId":"r3","operationId":"o1","value":"blue",'
            b'"clock":{"r3":1},"candidates":[]},'
            b'{"key":"k2","replicaId":"r3","operationId":"o2","value":"red",'
            b'"clock":{"r3":2},"candidates":[]}]}]',
        )

    def test_clock_components_and_candidate_identities_are_sorted(self) -> None:
        # The clock arrives out of component order and the candidate set
        # out of identity order; the digest input normalizes both
        # lexicographically while keeping the operation order.
        records = [
            tx_record(
                "tx-1",
                [
                    entry(
                        "k1",
                        "r3",
                        "o1",
                        "blue",
                        {"r3": 1, "r1": 2, "r2": 1},
                        [
                            {"replicaId": "r9", "operationId": "op-9"},
                            {"replicaId": "r1", "operationId": "op-1"},
                        ],
                    )
                ],
            )
        ]
        self.assertEqual(
            _transactions_digest_input(records),
            b'[{"transactionId":"tx-1","operations":['
            b'{"key":"k1","replicaId":"r3","operationId":"o1","value":"blue",'
            b'"clock":{"r1":2,"r2":1,"r3":1},"candidates":['
            b'{"replicaId":"r1","operationId":"op-1"},'
            b'{"replicaId":"r9","operationId":"op-9"}]}]}]',
        )

    def test_control_characters_and_quotes_are_escaped_compactly(self) -> None:
        records = [tx_record('tx-"1"\n', [entry('k\t1', "r3", "o1", "v", {"r3": 1})])]
        raw = _transactions_digest_input(records)
        # The quote and backslash get a one-character escape; every
        # control character including newline and tab is the lowercase
        # \u00XX form.
        self.assertIn(b'tx-\\"1\\"\\u000a', raw)
        self.assertIn(b'k\\u00091', raw)
        self.assertNotIn(b" ", raw)


class TransactionVerifyStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed_transaction(
        self,
        transaction_id: str,
        ops: list[tuple[str, str, str]],
        *,
        replica: str = "r3",
    ) -> None:
        """Commit one transaction of empty-expectation writes to fresh keys.

        Each ``ops`` triple is ``(operationId, key, value)``; the clock is
        one tick per distinct identity under one replica.
        """
        entries = [
            {
                "key": key,
                "replicaId": replica,
                "operationId": operation_id,
                "value": value,
                "clock": {replica: tick},
                "candidates": [],
            }
            for tick, (operation_id, key, value) in enumerate(ops, start=1)
        ]
        status, _, _, _, error = self.store.apply_transaction(
            transaction_id, entries
        )
        self.assertIs(status, HTTPStatus.CREATED, error)

    def seed_three_transactions(self) -> dict:
        self.seed_transaction("tx-1", [("o1", "k1", "blue")])
        self.seed_transaction(
            "tx-2", [("o2", "k2", "red"), ("o3", "k3", "green")]
        )
        self.seed_transaction("tx-3", [("o4", "k4", "yellow")])
        _, audit = self.store.get_transactions_verify(0, 100, DIGEST_64_A, 0)
        return {
            "digest": audit["digest"],
            "count": audit["transactionsCount"],
        }

    def verify(self, after: int, limit: int, digest: str, count: int):
        return self.store.get_transactions_verify(after, limit, digest, count)


class TransactionsVerifyStoreTests(TransactionVerifyStoreFixture):
    def test_empty_ledger_matching_expectations_is_ok(self) -> None:
        status, payload = self.verify(0, 100, EMPTY_DIGEST, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["transactionsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)

    def test_populated_ledger_counts_transactions_not_operations(self) -> None:
        expected = self.seed_three_transactions()
        status, payload = self.verify(
            0, 100, expected["digest"], expected["count"]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(
            [record["transactionId"] for record in payload["transactions"]],
            ["tx-1", "tx-2", "tx-3"],
        )
        # Records keep creation order and each operations array keeps its
        # original transaction order.
        second = payload["transactions"][1]
        self.assertEqual(
            [op["operationId"] for op in second["operations"]],
            ["o2", "o3"],
        )
        # Each page record carries exactly transactionId and operations.
        for record in payload["transactions"]:
            self.assertEqual(set(record), {"transactionId", "operations"})

    def test_paging_trims_only_the_page_never_the_conclusion(self) -> None:
        expected = self.seed_three_transactions()
        pages = []
        for after in range(4):
            status, page = self.verify(
                after, 1, expected["digest"], expected["count"]
            )
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
        self.assertEqual(
            [
                [record["transactionId"] for record in p["transactions"]]
                for p in pages
            ],
            [["tx-1"], ["tx-2"], ["tx-3"], []],
        )
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual(
            [p["hasMore"] for p in pages], [True, True, False, False]
        )
        for page in pages:
            self.assertEqual(page["digest"], expected["digest"])
            self.assertEqual(page["transactionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_digest_and_count_mismatches_are_page_independent(self) -> None:
        expected = self.seed_three_transactions()
        for after in (0, 2, 3):
            status, page = self.verify(after, 1, DIGEST_64_B, 9)
            self.assertIs(status, HTTPStatus.OK)
            verdict = page["verification"]
            self.assertEqual(verdict["status"], "broken")
            self.assertEqual(
                verdict["digestMismatches"],
                [{"expected": DIGEST_64_B, "observed": expected["digest"]}],
            )
            self.assertEqual(
                verdict["countMismatches"],
                [{"expected": 9, "observed": 3}],
            )
            # The true summary is reported alongside the mismatch.
            self.assertEqual(page["digest"], expected["digest"])
            self.assertEqual(page["transactionsCount"], 3)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        expected = self.seed_three_transactions()
        status, first = self.verify(
            3, 10, expected["digest"], expected["count"]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(first["transactions"], [])
        self.assertEqual(first["nextCursor"], 3)
        self.assertFalse(first["hasMore"])
        self.assertEqual(first["transactionsCount"], 3)
        self.assertEqual(first["verification"]["status"], "ok")
        status, second = self.verify(
            3, 10, expected["digest"], expected["count"]
        )
        self.assertEqual(second, first)

    def test_after_past_count_raises_value_error(self) -> None:
        expected = self.seed_three_transactions()
        with self.assertRaises(ValueError):
            self.verify(4, 1, expected["digest"], expected["count"])

    def test_identical_replay_appends_no_record(self) -> None:
        self.seed_transaction("tx-1", [("o1", "k1", "blue")])
        entries = [
            {
                "key": "k1",
                "replicaId": "r3",
                "operationId": "o1",
                "value": "blue",
                "clock": {"r3": 1},
                "candidates": [],
            }
        ]
        status, _, accepted, replayed, error = self.store.apply_transaction(
            "tx-1", entries
        )
        self.assertIs(status, HTTPStatus.OK, error)
        self.assertEqual((accepted, replayed), (0, 1))
        _, audit = self.store.get_transactions_verify(0, 100, DIGEST_64_A, 0)
        self.assertEqual(audit["transactionsCount"], 1)

    def test_query_is_strictly_read_only(self) -> None:
        expected = self.seed_three_transactions()
        transactions_before = copy.deepcopy(dict(self.store._transactions))
        accepted_before = copy.deepcopy(list(self.store._accepted))
        metrics_before = self.store.get_metrics()
        for query in (
            (0, 100, expected["digest"], expected["count"]),
            (0, 1, DIGEST_64_B, 9),
            (3, 1, expected["digest"], expected["count"]),
        ):
            self.verify(*query)
        self.assertEqual(self.store._transactions, transactions_before)
        self.assertEqual(self.store._accepted, accepted_before)
        self.assertEqual(self.store.get_metrics(), metrics_before)

    def test_damaged_ledger_is_broken_even_when_expectations_agree(self) -> None:
        expected = self.seed_three_transactions()
        # Damage the binding of tx-2 by rewriting one operation's value so
        # it drifts from the accepted log; the independently recomputed
        # digest then disagrees too, but a structural-only break is
        # demonstrated separately.
        self.store._transactions["tx-2"][0]["key"] = "k9"
        status, payload = self.verify(
            0, 100, expected["digest"], expected["count"]
        )
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertTrue(verdict["identityMismatches"])

    def test_structural_damage_is_a_record_violation_without_raising(self) -> None:
        self.seed_transaction("tx-1", [("o1", "k1", "blue")])
        # Replace a binding with a structurally illegal value; the digest
        # recomputation and scan must both stay well defined.
        self.store._transactions["tx-1"] = []  # empty operations list
        status, payload = self.verify(0, 100, DIGEST_64_B, 1)
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["recordViolations"],
            [{"transactionIndex": 0, "transactionId": "tx-1"}],
        )


class TransactionsVerifyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-transactions-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        store = StateStore(data_file=self.data_file)
        helper = TransactionVerifyStoreFixture()
        helper.store = store
        expected = helper.seed_three_transactions()
        _, before = store.get_transactions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_transactions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        self.assertEqual(after, before)
        # Every page boundary of the recovered store agrees.
        for cursor in range(4):
            _, page = recovered.get_transactions_verify(
                cursor, 1, expected["digest"], expected["count"]
            )
            self.assertEqual(page["digest"], expected["digest"])
            self.assertEqual(page["transactionsCount"], expected["count"])
            self.assertEqual(page["verification"]["status"], "ok")


class TransactionsVerifyHttpFixture(unittest.TestCase):
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
        self.server.store = StateStore()

    def raw_request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=body if isinstance(body, (bytes, str)) else json.dumps(body),
                headers=request_headers,
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def get(self, query: str, path: str = VERIFY_PATH):
        return self.raw_request("GET", path + query)

    def apply(self, transaction_id: str, ops: list[dict]) -> None:
        status, payload, _, _ = self.raw_request(
            "POST",
            APPLY_PATH,
            {"transactionId": transaction_id, "operations": ops},
        )
        self.assertEqual(status, 201, payload)

    def seed_three(self) -> tuple[str, int]:
        self.apply(
            "tx-1",
            [entry("k1", "r3", "o1", "blue", {"r3": 1})],
        )
        self.apply(
            "tx-2",
            [
                entry("k2", "r3", "o2", "red", {"r3": 1}),
                entry("k3", "r3", "o3", "green", {"r3": 2}),
            ],
        )
        self.apply(
            "tx-3",
            [entry("k4", "r3", "o4", "yellow", {"r3": 1})],
        )
        # Fetch the true summary with placeholder expectations; the
        # verdict is broken, but digest/count are page-independent facts.
        status, audit, _, _ = self.get(
            verify_query(0, 100, DIGEST_64_A, 0)
        )
        self.assertEqual(status, 200)
        return audit["digest"], audit["transactionsCount"]


class TransactionsVerifyHttpTests(TransactionsVerifyHttpFixture):
    def test_empty_ledger_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get(
            verify_query(0, 100, EMPTY_DIGEST, 0)
        )
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            raw,
            b'{"transactions":[],"nextCursor":0,"hasMore":false,'
            b'"algorithm":"sha256","digest":"'
            + EMPTY_DIGEST.encode("ascii")
            + b'","transactionsCount":0,"verification":{"status":"ok",'
            b'"duplicateTransactionIds":[],"recordViolations":[],'
            b'"identityMismatches":[],"batchViolations":[],'
            b'"digestMismatches":[],"countMismatches":[]}}\n',
        )

    def test_matching_expectations_page_and_stay_ok(self) -> None:
        digest, count = self.seed_three()
        status, payload, raw, headers = self.get(
            verify_query(0, 100, digest, count)
        )
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(payload["digest"], digest)
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        # Operations keep their original order inside one transaction.
        self.assertEqual(
            [op["operationId"] for op in payload["transactions"][1]["operations"]],
            ["o2", "o3"],
        )
        # limit=1 pages: summary and conclusion stay identical.
        pages = []
        for after in range(4):
            status, page, _, _ = self.get(verify_query(after, 1, digest, count))
            self.assertEqual(status, 200)
            pages.append(page)
        self.assertEqual(
            [
                [record["transactionId"] for record in p["transactions"]]
                for p in pages
            ],
            [["tx-1"], ["tx-2"], ["tx-3"], []],
        )
        for page in pages:
            self.assertEqual(page["digest"], digest)
            self.assertEqual(page["transactionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_wrong_digest_and_count_report_expected_then_observed(self) -> None:
        digest, count = self.seed_three()
        status, payload, _, _ = self.get(
            verify_query(0, 1, DIGEST_64_B, count + 1)
        )
        self.assertEqual(status, 200)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": digest}],
        )
        self.assertEqual(
            verdict["countMismatches"],
            [{"expected": count + 1, "observed": count}],
        )
        status, payload, _, _ = self.get(verify_query(0, 1, DIGEST_64_B, count))
        self.assertEqual(payload["verification"]["countMismatches"], [])

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        digest, count = self.seed_three()
        status, payload, raw, _ = self.get(verify_query(3, 10, digest, count))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_bad_queries_are_400_and_change_nothing(self) -> None:
        bad_queries = [
            "",
            "?",
            "?after=0&limit=1",
            "?after=0&limit=1&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0&after=2",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0&x=1",
            "?x=1",
            f"?after=&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest=&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=",
            f"?after=-1&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=0&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=101&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=-1",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=1.0",
            f"?after=0&limit=1&expectedDigest={'A' * 64}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={'a' * 63}&expectedCount=0",
            "?after=0&limit=1&expectedDigest=" + "g" * 64 + "&expectedCount=0",
            f"?after&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
        ]
        for query in bad_queries:
            with self.subTest(query=query[:60]):
                status, bad_payload, raw, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(bad_payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))
        # No rejected query created a transaction.
        self.assertEqual(self.server.store._transactions, {})

    def test_after_past_the_transaction_count_is_400(self) -> None:
        digest, count = self.seed_three()
        status, payload, raw, _ = self.get(verify_query(4, 1, digest, count))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        bad_paths = [
            VERIFY_PATH + "/",
            VERIFY_PATH + "/extra",
            "/v1/transactions/verif",
            "/v1/transaction/verify",
            "/v1/transactions",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=%ZZ&expectedDigest=x"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_apply_route_does_not_accept_the_verify_query(self) -> None:
        # The sibling POST route is unchanged: GET on it is an unknown
        # route, and the verify parameters are never its query contract.
        status, payload, _, _ = self.raw_request(
            "GET", APPLY_PATH + verify_query(0, 1, DIGEST_64_A, 0)
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_verify_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request("POST", VERIFY_PATH, {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only_over_http(self) -> None:
        digest, count = self.seed_three()
        status, first, _, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        transactions_before = copy.deepcopy(dict(self.server.store._transactions))
        for query in (
            verify_query(0, 100, digest, count),
            verify_query(0, 1, DIGEST_64_B, 9),
            verify_query(3, 1, digest, count),
            "?after=0&limit=x",
        ):
            self.get(query)
        status, second, _, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(second, first)
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(self.server.store._transactions, transactions_before)


class TransactionsVerifyPersistenceHttpTests(TransactionsVerifyHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-transactions-verify-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_query_persists_nothing(self) -> None:
        digest, count = self.seed_three()
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in (
            verify_query(0, 100, digest, count),
            verify_query(3, 1, digest, count),
            verify_query(0, 1, DIGEST_64_B, 9),
            "?after=99&limit=1",
        ):
            self.get(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class TransactionsVerifyAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-transactions-verify-auth-")
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

    def get(self, port, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers=dict(headers or []))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, response_headers

    QUERY = verify_query(0, 1, DIGEST_64_A, 0)
    PATH = VERIFY_PATH + QUERY

    def test_single_token_missing_bad_or_wrong_is_401(self) -> None:
        status, payload, headers = self.get(self.single_port, self.PATH)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, self.PATH, [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, headers = self.get(
            self.single_port, self.PATH, [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_single_token_valid_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.single_port,
            self.PATH,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["transactionsCount"], 0)
        # The supplied digest disagrees with the empty-array digest.
        self.assertEqual(payload["verification"]["status"], "broken")

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            self.PATH,
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        status, _, headers = self.get(
            self.scope_port,
            VERIFY_PATH + "?after=%ZZ",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            VERIFY_PATH + verify_query(0, 1, EMPTY_DIGEST, 0),
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["transactionsCount"], 0)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_unknown_shape_still_authenticates_before_the_404(self) -> None:
        status, payload, headers = self.get(
            self.single_port,
            VERIFY_PATH + "/extra" + self.QUERY,
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra" + self.QUERY,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
