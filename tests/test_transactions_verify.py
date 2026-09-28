"""Tests for the read-only transaction-ledger audit::

    GET /v1/transactions/verify

The endpoint pages the committed atomic transactions in creation order
(each item carrying ``transactionId`` and ``operations``, the latter in
the transaction's original operation order with candidate identities
and clock components canonicalized lexicographically) and requires four
parameters — the ``after``/``limit`` page pair plus two external
expectations, ``expectedDigest`` (exactly 64 lowercase hexadecimal
characters) and ``expectedCount`` (a non-negative ASCII decimal
integer). Its ``verification`` independently scans the **complete**
history:

- ``duplicateTransactionIds``, ``recordViolations``,
  ``identityMismatches``, and ``batchViolations`` keep the internal
  anomaly judgement;
- ``digestMismatches`` (at most one ``{"expected", "observed"}`` marker,
  the caller's digest first and the recomputed full-history digest
  second) and ``countMismatches`` (at most one marker with the expected
  count first and the actual full count second) report external
  disagreement.

``status`` is ``"ok"`` exactly when all six lists are empty. Paging
trims only the exported page; the digest, count, and conclusion always
cover the complete history. The tests cover the query parser, the
independent scan and external comparisons, the store's
paging/snapshot/recovery semantics, the HTTP precedence chain (404
path shape before the query check, 401 authentication with a Bearer
challenge, 403 in scope mode without read/admin, 400 query validation
including an ``after`` past the transaction count), the compact
field-ordered single-newline response body with its explicit
Content-Length, restart consistency, and the strict read-only
guarantee. The committing ``POST /v1/transactions/apply`` route is
unchanged. Only the Python standard library is used.
"""

from __future__ import annotations

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
    _transactions_external_verification_locked,
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
OK_VERIFICATION = {name: [] for name in VERIFICATION_FIELDS if name != "status"}
OK_VERIFICATION["status"] = "ok"
OK_VERIFICATION = {name: OK_VERIFICATION[name] for name in VERIFICATION_FIELDS}


def verify_query(after: int, limit: int, digest: str, count: int) -> str:
    return (
        f"?after={after}&limit={limit}"
        f"&expectedDigest={digest}&expectedCount={count}"
    )


def tx_entry(
    key: str,
    replica: str = "r1",
    operation_id: str = "t1",
    value: str = "v",
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


def tx_document(*entries: dict, transaction_id: str = "tx-1") -> dict:
    return {"transactionId": transaction_id, "operations": list(entries)}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def page_operation(entry: dict) -> dict:
    """The exported page form of a normalized transaction entry."""
    return {
        "key": entry["key"],
        "replicaId": entry["replicaId"],
        "operationId": entry["operationId"],
        "value": entry["value"],
        "clock": dict(sorted(entry["clock"].items())),
        "candidates": [
            {"replicaId": c["replicaId"], "operationId": c["operationId"]}
            for c in sorted(
                entry["candidates"],
                key=lambda c: (c["replicaId"], c["operationId"]),
            )
        ],
    }


class ParseTransactionsVerifyQueryTests(unittest.TestCase):
    def test_accepts_all_four_required_parameters(self) -> None:
        self.assertEqual(
            parse_transactions_verify_query(
                f"after=2&limit=50&expectedDigest={DIGEST_64_A}&expectedCount=7"
            ),
            (2, 50, DIGEST_64_A, 7),
        )

    def test_accepts_zero_after_and_count(self) -> None:
        self.assertEqual(
            parse_transactions_verify_query(
                f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
            ),
            (0, 1, DIGEST_64_A, 0),
        )

    def test_leading_zeros_are_plain_decimal(self) -> None:
        self.assertEqual(
            parse_transactions_verify_query(
                f"after=007&limit=09&expectedDigest={DIGEST_64_A}"
                "&expectedCount=00"
            ),
            (7, 9, DIGEST_64_A, 0),
        )

    def test_rejects_missing_parameters(self) -> None:
        bad_queries = [
            "",
            "?",
            "after=0&limit=1",
            f"after=0&limit=1&expectedCount=0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}",
            f"limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&expectedDigest={DIGEST_64_A}&expectedCount=0",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                self.assertIsNone(parse_transactions_verify_query(query))

    def test_rejects_repeated_and_unknown_parameters(self) -> None:
        base = f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
        bad_queries = [
            base + "&after=2",
            base + "&x=1",
            "x=1",
            f"after&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount",
        ]
        for query in bad_queries:
            with self.subTest(query=query[:60]):
                self.assertIsNone(parse_transactions_verify_query(query))

    def test_rejects_blank_or_malformed_integer_values(self) -> None:
        base = f"limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
        bad_queries = [
            "after=&" + base,
            "after=-1&" + base,
            "after=+0&" + base,
            "after=1.0&" + base,
            "after=0%20&" + base,
            "after=0&limit=&"
            f"expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=0&"
            f"expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=101&"
            f"expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=-1",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=1.0",
        ]
        for query in bad_queries:
            with self.subTest(query=query[:60]):
                self.assertIsNone(parse_transactions_verify_query(query))

    def test_rejects_malformed_digest(self) -> None:
        base = "after=0&limit=1&expectedCount=0"
        bad_digests = [
            "",
            "A" * 64,
            "a" * 63,
            "g" * 64,
            DIGEST_64_A[:63] + "G",
        ]
        for digest in bad_digests:
            with self.subTest(digest=digest[:10]):
                self.assertIsNone(
                    parse_transactions_verify_query(
                        f"{base}&expectedDigest={digest}"
                    )
                )


class TransactionsExternalVerificationTests(unittest.TestCase):
    def history_digest(self, history):
        return hashlib.sha256(_transactions_digest_input(history)).hexdigest()

    def test_empty_history_ok_for_empty_array_digest_and_zero_count(self) -> None:
        self.assertEqual(
            _transactions_external_verification_locked(
                [], {}, EMPTY_DIGEST, 0
            ),
            OK_VERIFICATION,
        )

    def test_well_formed_history_matching_expectations_is_ok(self) -> None:
        entries = [tx_entry("k", "r1", "t1")]
        history = [("tx-1", entries)]
        operations_index = {
            ("r1", "t1"): {
                "operationId": "t1",
                "key": "k",
                "value": "v",
                "clock": {"r1": 1},
            }
        }
        verdict = _transactions_external_verification_locked(
            history, operations_index, self.history_digest(history), 1
        )
        self.assertEqual(verdict, OK_VERIFICATION)
        self.assertEqual(list(verdict), VERIFICATION_FIELDS)

    def test_digest_mismatch_marks_expected_first_observed_second(self) -> None:
        verdict = _transactions_external_verification_locked(
            [], {}, DIGEST_64_B, 0
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": EMPTY_DIGEST}],
        )
        self.assertEqual(verdict["countMismatches"], [])

    def test_count_mismatch_marks_expected_first_observed_second(self) -> None:
        history = [("tx-1", [tx_entry("k")])]
        verdict = _transactions_external_verification_locked(
            history, {}, EMPTY_DIGEST, 5
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["countMismatches"], [{"expected": 5, "observed": 1}]
        )

    def test_duplicate_transaction_id_marks_the_later_occurrence(self) -> None:
        history = [
            ("tx-1", [tx_entry("k1", "r1", "t1")]),
            ("tx-1", [tx_entry("k2", "r1", "t2")]),
        ]
        operations_index = {
            ("r1", "t1"): {
                "operationId": "t1",
                "key": "k1",
                "value": "v",
                "clock": {"r1": 1},
            },
            ("r1", "t2"): {
                "operationId": "t2",
                "key": "k2",
                "value": "v",
                "clock": {"r1": 1},
            },
        }
        verdict = _transactions_external_verification_locked(
            history,
            operations_index,
            self.history_digest(history),
            2,
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["duplicateTransactionIds"],
            [{"transactionIndex": 1, "transactionId": "tx-1"}],
        )
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])

    def test_record_violation_marks_malformed_records(self) -> None:
        history = [
            ("", [tx_entry("k")]),
            ("tx-2", "not-a-list"),
            ("tx-3", []),
            ("tx-4", [{"key": "k"}]),
        ]
        verdict = _transactions_external_verification_locked(
            history, {}, self.history_digest(history), 4
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["recordViolations"],
            [
                {"transactionIndex": 0, "transactionId": ""},
                {"transactionIndex": 1, "transactionId": "tx-2"},
                {"transactionIndex": 2, "transactionId": "tx-3"},
                {"transactionIndex": 3, "transactionId": "tx-4"},
            ],
        )

    def test_identity_mismatch_against_accepted_operation(self) -> None:
        entry = tx_entry("k", "r1", "t1", value="stored")
        history = [("tx-1", [entry])]
        operations_index = {
            ("r1", "t1"): {
                "operationId": "t1",
                "key": "k",
                "value": "accepted",
                "clock": {"r1": 1},
            }
        }
        verdict = _transactions_external_verification_locked(
            history,
            operations_index,
            self.history_digest(history),
            1,
        )
        self.assertEqual(verdict["status"], "broken")
        mismatches = verdict["identityMismatches"]
        self.assertEqual(len(mismatches), 1)
        marker = mismatches[0]
        self.assertEqual(marker["transactionIndex"], 0)
        self.assertEqual(marker["transactionId"], "tx-1")
        self.assertEqual(marker["operationIndex"], 0)
        self.assertEqual(marker["replicaId"], "r1")
        self.assertEqual(marker["operationId"], "t1")
        self.assertEqual(
            marker["expected"],
            {
                "operationId": "t1",
                "key": "k",
                "value": "accepted",
                "clock": {"r1": 1},
            },
        )
        self.assertEqual(
            marker["observed"],
            {
                "operationId": "t1",
                "key": "k",
                "value": "stored",
                "clock": {"r1": 1},
            },
        )

    def test_identity_mismatch_expected_null_when_operation_missing(self) -> None:
        entry = tx_entry("k", "r9", "t9")
        history = [("tx-1", [entry])]
        verdict = _transactions_external_verification_locked(
            history, {}, self.history_digest(history), 1
        )
        self.assertEqual(verdict["status"], "broken")
        marker = verdict["identityMismatches"][0]
        self.assertIsNone(marker["expected"])
        self.assertEqual(marker["observed"]["operationId"], "t9")

    def test_batch_violation_marks_repeated_keys_and_identities(self) -> None:
        history = [
            (
                "tx-1",
                [
                    tx_entry("k", "r1", "t1"),
                    tx_entry("k", "r2", "t2"),
                    tx_entry("other", "r2", "t2"),
                ],
            )
        ]
        operations_index = {
            ("r1", "t1"): {
                "operationId": "t1",
                "key": "k",
                "value": "v",
                "clock": {"r1": 1},
            },
            ("r2", "t2"): {
                "operationId": "t2",
                "key": "k",
                "value": "v",
                "clock": {"r2": 1},
            },
        }
        verdict = _transactions_external_verification_locked(
            history,
            operations_index,
            self.history_digest(history),
            1,
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["batchViolations"],
            [
                {
                    "transactionIndex": 0,
                    "transactionId": "tx-1",
                    "operationIndex": 1,
                    "replicaId": "r2",
                    "operationId": "t2",
                },
                {
                    "transactionIndex": 0,
                    "transactionId": "tx-1",
                    "operationIndex": 2,
                    "replicaId": "r2",
                    "operationId": "t2",
                },
            ],
        )

    def test_internal_anomaly_keeps_judgement_even_when_expectations_match(
        self,
    ) -> None:
        entry = tx_entry("k", "r1", "t1", value="stored")
        history = [("tx-1", [entry])]
        verdict = _transactions_external_verification_locked(
            history, {}, self.history_digest(history), 1
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])
        self.assertEqual(len(verdict["identityMismatches"]), 1)


class TransactionsVerifyStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def apply_tx(
        self,
        *entries: dict,
        transaction_id: str = "tx-1",
        assert_status: int = 201,
    ):
        status, results, accepted, replayed, error = self.store.apply_transaction(
            transaction_id, list(entries)
        )
        self.assertEqual(status, assert_status, error)
        return status, results, accepted, replayed

    def seed_three(self) -> dict:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.apply_tx(
            tx_entry("k2", "r2", "t2", clock={"r2": 1}),
            transaction_id="tx-2",
        )
        self.apply_tx(
            tx_entry("k3", "r3", "t3", clock={"r3": 1}),
            transaction_id="tx-3",
        )
        status, payload = self.store.get_transactions_verify(
            0, 100, "0" * 64, -1
        )
        self.assertIs(status, HTTPStatus.OK)
        return {"digest": payload["digest"], "count": payload["transactionsCount"]}

    def verify(self, after: int, limit: int, digest: str, count: int):
        return self.store.get_transactions_verify(after, limit, digest, count)

    def history(self):
        return list(self.store._transactions.items())


class TransactionsVerifyStoreTests(TransactionsVerifyStoreFixture):
    def test_empty_history_matching_expectations_is_ok(self) -> None:
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

    def test_populated_history_page_shape_and_order(self) -> None:
        expected = self.seed_three()
        status, payload = self.verify(0, 100, expected["digest"], expected["count"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(
            [t["transactionId"] for t in payload["transactions"]],
            ["tx-1", "tx-2", "tx-3"],
        )
        first = payload["transactions"][0]
        self.assertEqual(list(first), ["transactionId", "operations"])
        self.assertEqual(first["operations"], [page_operation(tx_entry("k1"))])
        second_entry = tx_entry("k2", "r2", "t2", clock={"r2": 1})
        self.assertEqual(
            payload["transactions"][1]["operations"],
            [page_operation(second_entry)],
        )

    def test_candidate_set_and_clock_components_are_canonicalized(self) -> None:
        # Commit a transaction whose candidate set and clock arrive
        # unordered; the page and digest use the lexicographic order.
        self.apply_tx(
            tx_entry(
                "k1", "r1", "t1", clock={"r2": 1, "r1": 1}
            ),
            transaction_id="first",
        )
        self.apply_tx(
            tx_entry(
                "k1",
                "r2",
                "t2",
                clock={"r2": 2, "r1": 1},
                candidates=[candidate("r1", "t1")],
            ),
            transaction_id="second",
        )
        _, payload = self.verify(0, 100, "0" * 64, 0)
        operation = payload["transactions"][1]["operations"][0]
        self.assertEqual(list(operation["clock"]), ["r1", "r2"])
        self.assertEqual(
            operation["candidates"], [candidate("r1", "t1")]
        )
        # The independently recomputed digest matches the same canonical
        # ordering, so matching expectations verify ok.
        digest = payload["digest"]
        _, matching = self.verify(0, 100, digest, 2)
        self.assertEqual(matching["verification"]["status"], "ok")

    def test_digest_covers_candidates_and_full_history(self) -> None:
        expected = self.seed_three()
        history = self.history()
        self.assertEqual(
            expected["digest"],
            hashlib.sha256(_transactions_digest_input(history)).hexdigest(),
        )

    def test_paging_trims_only_the_page(self) -> None:
        expected = self.seed_three()
        pages = []
        for after in range(4):
            status, page = self.verify(after, 1, expected["digest"], 3)
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
        self.assertEqual(
            [[t["transactionId"] for t in p["transactions"]] for p in pages],
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

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        expected = self.seed_three()
        status, payload = self.verify(3, 10, expected["digest"], 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_after_past_the_count_raises(self) -> None:
        self.seed_three()
        with self.assertRaises(ValueError):
            self.verify(4, 1, EMPTY_DIGEST, 3)

    def test_wrong_digest_and_count_break_with_expected_first(self) -> None:
        expected = self.seed_three()
        status, payload = self.verify(0, 1, DIGEST_64_B, 9)
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": expected["digest"]}],
        )
        self.assertEqual(
            verdict["countMismatches"], [{"expected": 9, "observed": 3}]
        )

    def test_injected_identity_mismatch_is_detected_against_the_archive(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        # Damage the binding while keeping the accepted operation intact.
        self.store._transactions["tx-1"] = [
            tx_entry("k1", "r1", "t1", value="tampered")
        ]
        _, payload = self.verify(0, 100, "0" * 64, 1)
        marker = payload["verification"]["identityMismatches"][0]
        self.assertEqual(marker["expected"]["value"], "v")
        self.assertEqual(marker["observed"]["value"], "tampered")
        self.assertEqual(payload["verification"]["status"], "broken")

    def test_injected_batch_violation_is_detected(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.store._transactions["tx-batch"] = [
            tx_entry("k", "r9", "a"),
            tx_entry("k", "r9", "b"),
        ]
        _, payload = self.verify(0, 100, "0" * 64, 2)
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(
            payload["verification"]["batchViolations"],
            [
                {
                    "transactionIndex": 1,
                    "transactionId": "tx-batch",
                    "operationIndex": 1,
                    "replicaId": "r9",
                    "operationId": "b",
                }
            ],
        )

    def test_damaged_history_still_pages_and_reports_a_record_violation(
        self,
    ) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        # Inject a structurally malformed record behind the good one.
        self.store._transactions["tx-bad"] = "not-a-list"
        status, payload = self.verify(0, 100, "0" * 64, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [t["transactionId"] for t in payload["transactions"]],
            ["tx-1", "tx-bad"],
        )
        self.assertEqual(payload["transactions"][1]["operations"], "not-a-list")
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(
            payload["verification"]["recordViolations"],
            [{"transactionIndex": 1, "transactionId": "tx-bad"}],
        )
        # A concrete observed digest is still recomputed deterministically.
        observed = payload["digest"]
        _, again = self.verify(0, 100, "0" * 64, 2)
        self.assertEqual(again["digest"], observed)


class TransactionsVerifyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-transactions-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        store = StateStore(data_file=self.data_file)
        helper = TransactionsVerifyStoreFixture()
        helper.store = store
        expected = helper.seed_three()
        _, before = store.get_transactions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_transactions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        self.assertEqual(after, before)
        _, broken = recovered.get_transactions_verify(0, 2, DIGEST_64_B, 9)
        self.assertEqual(
            broken["verification"]["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": expected["digest"]}],
        )
        self.assertEqual(
            broken["verification"]["countMismatches"],
            [{"expected": 9, "observed": 3}],
        )

    def test_old_file_without_section_verifies_empty_intact(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=self.data_file)
        status, payload = store.get_transactions_verify(0, 100, EMPTY_DIGEST, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["transactionsCount"], 0)
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["verification"], OK_VERIFICATION)


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

    def apply_tx(self, *entries: dict, transaction_id: str = "tx-1") -> None:
        status, _, _, _ = self.raw_request(
            "POST",
            APPLY_PATH,
            tx_document(*entries, transaction_id=transaction_id),
        )
        self.assertEqual(status, 201)

    def seed_three(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")
        self.apply_tx(
            tx_entry("k2", "r2", "t2", clock={"r2": 1}),
            transaction_id="tx-2",
        )
        self.apply_tx(
            tx_entry("k3", "r3", "t3", clock={"r3": 1}),
            transaction_id="tx-3",
        )

    def current_expectations(self) -> tuple[str, int]:
        # Read the recomputed summary straight from the store; the query
        # is exercised over HTTP with matching expectations.
        _, payload = self.server.store.get_transactions_verify(
            0, 100, "0" * 64, -1
        )
        return payload["digest"], payload["transactionsCount"]


class TransactionsVerifyHttpTests(TransactionsVerifyHttpFixture):
    def test_empty_history_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get(verify_query(0, 100, EMPTY_DIGEST, 0))
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
            b'"algorithm":"sha256","digest":"' + EMPTY_DIGEST.encode("ascii")
            + b'","transactionsCount":0,"verification":{"status":"ok",'
            b'"duplicateTransactionIds":[],"recordViolations":[],'
            b'"identityMismatches":[],"batchViolations":[],'
            b'"digestMismatches":[],"countMismatches":[]}}\n',
        )

    def test_matching_expectations_page_and_stay_ok(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
        status, payload, raw, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(payload["digest"], digest)
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        pages = []
        for after in range(4):
            status, page, _, _ = self.get(verify_query(after, 1, digest, count))
            self.assertEqual(status, 200)
            pages.append(page)
        self.assertEqual(
            [[t["transactionId"] for t in p["transactions"]] for p in pages],
            [["tx-1"], ["tx-2"], ["tx-3"], []],
        )
        for page in pages:
            self.assertEqual(page["digest"], digest)
            self.assertEqual(page["transactionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_page_record_shape_keeps_operation_order_and_candidates(self) -> None:
        self.apply_tx(tx_entry("k1", "r1", "t1", clock={"r2": 1, "r1": 1}), transaction_id="first")
        self.apply_tx(
            tx_entry(
                "k1",
                "r2",
                "t2",
                clock={"r2": 2, "r1": 1},
                candidates=[candidate("r1", "t1")],
            ),
            transaction_id="second",
        )
        digest, count = self.current_expectations()
        status, payload, _, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        operation = payload["transactions"][1]["operations"][0]
        self.assertEqual(
            list(operation),
            ["key", "replicaId", "operationId", "value", "clock", "candidates"],
        )
        self.assertEqual(list(operation["clock"]), ["r1", "r2"])
        self.assertEqual(operation["candidates"], [candidate("r1", "t1")])

    def test_wrong_digest_and_count_report_expected_then_observed(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
        status, payload, _, _ = self.get(verify_query(0, 1, DIGEST_64_B, count + 1))
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

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
        status, payload, raw, _ = self.get(verify_query(3, 10, digest, count))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["transactionsCount"], 3)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_bad_queries_are_400_and_change_nothing(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
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
        # No rejected query changed the summary.
        self.assertEqual(self.current_expectations(), (digest, count))

    def test_after_past_the_transaction_count_is_400(self) -> None:
        self.seed_three()
        digest, _ = self.current_expectations()
        status, payload, raw, _ = self.get(verify_query(4, 1, digest, 3))
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

    def test_apply_route_is_unchanged(self) -> None:
        # GET on the commit route is an unknown route, and the four
        # verification parameters are not accepted there.
        status, payload, _, _ = self.raw_request(
            "GET", APPLY_PATH + verify_query(0, 1, DIGEST_64_A, 0)
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        self.apply_tx(tx_entry("k1", "r1", "t1"), transaction_id="tx-1")

    def test_post_to_verify_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request("POST", VERIFY_PATH, {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only_over_http(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
        status, first, _, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
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
        self.assertEqual(len(self.server.store._transactions), 3)


class TransactionsVerifyPersistenceHttpTests(TransactionsVerifyHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-transactions-verify-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_query_persists_nothing(self) -> None:
        self.seed_three()
        digest, count = self.current_expectations()
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in (
            verify_query(0, 100, digest, count),
            verify_query(1, 1, digest, count),
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
            self.single_port, VERIFY_PATH + "/extra" + self.QUERY
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
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra?after=%ZZ&expectedDigest=x",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
