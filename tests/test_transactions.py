"""HTTP, sync, and persistence tests for atomic multi-key transactions.

The endpoint is::

    POST /v1/transactions/apply

It accepts a ``{"transactionId","operations"}`` document with 1-100
conditional writes to distinct keys, validates every entry's expected
candidate set against the current state, and commits the whole batch
atomically into the shared accepted log. Everything here goes through the
real HTTP entry point (``SemanticStateServer`` + a request thread); only
the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    load_data_file_full,
    load_data_file_transactions,
    parse_transaction_apply,
)

APPLY_PATH = "/v1/transactions/apply"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def tx_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "tx-op-1",
    value: str = "v",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock if clock is not None else {"r3": 1},
        "candidates": list(candidates) if candidates is not None else [],
    }


def tx_document(*entries: dict, transaction_id: str = "tx-1") -> dict:
    return {"transactionId": transaction_id, "operations": list(entries)}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


class ParseTransactionApplyTests(unittest.TestCase):
    def test_valid_transaction_is_normalized_in_order(self) -> None:
        transaction_id, entries = parse_transaction_apply(
            json.dumps(
                tx_document(
                    tx_entry(
                        "k1",
                        "r3",
                        "t1",
                        "v1",
                        {"r1": 1, "r3": 1},
                        [candidate("r1", "o1")],
                    ),
                    tx_entry("k2", "r9", "t2", "v2", {"r9": 2}, []),
                )
            )
        )
        self.assertEqual(transaction_id, "tx-1")
        self.assertEqual(
            entries,
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "t1",
                    "value": "v1",
                    "clock": {"r1": 1, "r3": 1},
                    "candidates": [{"replicaId": "r1", "operationId": "o1"}],
                },
                {
                    "key": "k2",
                    "replicaId": "r9",
                    "operationId": "t2",
                    "value": "v2",
                    "clock": {"r9": 2},
                    "candidates": [],
                },
            ],
        )

    def test_candidates_are_normalized_to_set_order(self) -> None:
        _, entries = parse_transaction_apply(
            tx_document(
                tx_entry(
                    "k",
                    candidates=[candidate("r2", "o2"), candidate("r1", "o1")],
                )
            )
        )
        self.assertEqual(
            entries[0]["candidates"],
            [candidate("r1", "o1"), candidate("r2", "o2")],
        )

    def test_one_and_one_hundred_entries_are_accepted(self) -> None:
        self.assertEqual(len(parse_transaction_apply(tx_document(tx_entry("k")))[1]), 1)
        entries = [
            tx_entry(f"k{i:03d}", "r3", f"t{i:03d}", "v", {"r3": 1})
            for i in range(100)
        ]
        _, parsed = parse_transaction_apply(tx_document(*entries))
        self.assertEqual(len(parsed), 100)
        self.assertEqual([e["key"] for e in parsed], [f"k{i:03d}" for i in range(100)])

    def test_rejects_malformed_and_wrong_root_shapes(self) -> None:
        valid = tx_document(tx_entry("k"))
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"transactionId": "tx-1"},
            {"operations": valid["operations"]},
            dict(valid, extra=1),
            {"transactionId": "tx-1", "operations": valid["operations"], "x": 2},
            {"transactionId": "", "operations": valid["operations"]},
            {"transactionId": 7, "operations": valid["operations"]},
            {"transactionId": None, "operations": valid["operations"]},
            {"transactionId": "tx-1", "operations": {}},
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_transaction_apply(body)

    def test_rejects_empty_and_oversized_batches(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(tx_document())
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(
                    *[tx_entry(f"k{i}", "r3", f"t{i}", "v", {"r3": 1}) for i in range(101)]
                )
            )

    def test_rejects_bad_entry_shapes(self) -> None:
        good = tx_entry("k")
        bad_entries = [
            [],
            "x",
            {},
            *[
                {key: value for key, value in good.items() if key != field}
                for field in ("key", "replicaId", "operationId", "value", "clock", "candidates")
            ],
            dict(good, extra=1),
            dict(good, key=""),
            dict(good, key=7),
            dict(good, replicaId=""),
            dict(good, operationId=""),
            dict(good, value=""),
            dict(good, value=3),
            dict(good, clock={}),
            dict(good, clock={"r2": 1}),  # must contain the entry replica
            dict(good, clock={"r3": -1}),
            dict(good, clock={"r3": True}),
            dict(good, candidates={}),
            dict(good, candidates="r1"),
        ]
        for entry in bad_entries:
            with self.assertRaises(ValueError, msg=repr(entry)):
                parse_transaction_apply(tx_document(entry))

    def test_rejects_bad_candidate_shapes(self) -> None:
        good_candidate = candidate("r1", "o1")
        bad_candidates = [
            ["x"],
            [{}],
            [{"replicaId": "r1"}],
            [{"operationId": "o1"}],
            [dict(good_candidate, extra=1)],
            [candidate("", "o1")],
            [candidate("r1", "")],
            [candidate(1, "o1")],
            [candidate("r1", 2)],
            [candidate("r1", "o1"), candidate("r1", "o1")],  # duplicate
        ]
        for candidates in bad_candidates:
            with self.assertRaises(ValueError, msg=repr(candidates)):
                parse_transaction_apply(tx_document(tx_entry("k", candidates=candidates)))

    def test_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(
                    tx_entry("k", "r3", "t1"),
                    tx_entry("k", "r4", "t2", "v", {"r4": 1}),
                )
            )

    def test_rejects_duplicate_identities_even_on_different_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(
                    tx_entry("k1", "r3", "same-op"),
                    tx_entry("k2", "r3", "same-op"),
                )
            )

    def test_rejects_floats_and_non_finite_clock_values(self) -> None:
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                with self.assertRaises(ValueError):
                    parse_transaction_apply(
                        tx_document(tx_entry("k", clock={"r3": tick}))
                    )
        for literal in ("NaN", "Infinity", "-Infinity", "1e3"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"transactionId":"tx-1","operations":[{"key":"k","replicaId":"r3",'
                    b'"operationId":"t1","value":"v","clock":{"r3":'
                    + literal.encode("ascii")
                    + b'},"candidates":[]}]}'
                )
                with self.assertRaises(ValueError):
                    parse_transaction_apply(raw)


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

    def request_bytes(
        self, method: str, path: str, body: bytes | None = None, headers: dict | None = None
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(method, path, body=body, headers=merged)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        if body is None:
            status, raw = self.request_bytes(method, path)
        elif isinstance(body, (bytes, str)):
            data = body.encode("utf-8") if isinstance(body, str) else body
            status, raw = self.request_bytes(method, path, data)
        else:
            status, raw = self.request_bytes(method, path, json.dumps(body).encode("utf-8"))
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_apply(self, body: object, path: str = APPLY_PATH) -> tuple[int, object]:
        return self.request("POST", path, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_audit(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_key(
        self, key: str, replica: str = "r1", operation_id: str | None = None, value: str = "v1"
    ) -> str:
        """One write on ``key``; returns the operation id used."""
        op_id = operation_id or f"o-{key}"
        status, _ = self.post_operation(replica, operation(op_id, key, value, {replica: 1}))
        self.assertEqual(status, 201)
        return op_id


class TransactionHappyPathTests(HttpServerTestCase):
    def test_transaction_commits_distinct_keys_in_order_with_counts(self) -> None:
        self.seed_key("k1", "r1", "o1")
        self.seed_key("k2", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r2": 1, "r3": 1}, [candidate("r2", "o2")]),
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r3", "operationId": "t1", "value": "n1"},
                    {"key": "k2", "replicaId": "r3", "operationId": "t2", "value": "n2"},
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        _, state1 = self.get_state("k1")
        self.assertEqual((state1["status"], state1["value"]), ("resolved", "n1"))
        _, state2 = self.get_state("k2")
        self.assertEqual((state2["status"], state2["value"]), ("resolved", "n2"))

    def test_empty_expected_set_matches_a_key_without_candidates(self) -> None:
        doc = tx_document(tx_entry("fresh", "r3", "t1", "n", {"r3": 1}, []))
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        _, state = self.get_state("fresh")
        self.assertEqual(state["value"], "n")

    def test_response_is_compact_json_ending_with_a_newline(self) -> None:
        body = json.dumps(tx_document(tx_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", APPLY_PATH, body)
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw[:-1])

    def test_one_hundred_entries_commit_together(self) -> None:
        entries = [
            tx_entry(f"k{i:03d}", "r3", f"t{i:03d}", f"n{i:03d}", {"r3": 1}, [])
            for i in range(100)
        ]
        status, payload = self.post_apply(tx_document(*entries))
        self.assertEqual(status, 201)
        self.assertEqual((payload["accepted"], payload["replayed"]), (100, 0))
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 100)
        self.assertEqual(metrics["keys"], 100)

    def test_records_enter_sync_audit_metrics_and_archive(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
        )
        status, _ = self.post_apply(doc)
        self.assertEqual(status, 201)

        _, page = self.get_sync()
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in page["operations"]],
            [("r1", "o1"), ("r3", "t1"), ("r4", "t2")],
        )
        _, audit = self.get_audit("k1")
        self.assertEqual(
            [r["operation"]["operationId"] for r in audit["operations"]], ["o1", "t1"]
        )
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 3)
        self.assertEqual(metrics["replicas"], 3)
        status, record = self.request("GET", "/v1/replicas/r3/operations/t1")
        self.assertEqual(status, 200)
        self.assertEqual(record["operation"]["value"], "n1")
        status, digest = self.request("GET", "/v1/audit/keys/k1/digest")
        self.assertEqual(status, 200)
        self.assertEqual(digest["operations"], 2)

    def test_transaction_operations_import_cleanly_elsewhere(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        status, _ = self.post_apply(doc)
        self.assertEqual(status, 201)
        _, page = self.get_sync()
        # Re-importing the exported records (the transaction binding is not
        # exported) is a pure replay with no new log records.
        status, payload = self.request(
            "POST", "/v1/sync/operations", {"operations": page["operations"]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 2)


class TransactionReplayTests(HttpServerTestCase):
    def test_identical_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        status, created = self.post_apply(doc)
        self.assertEqual(status, 201)
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r3", "operationId": "t1", "value": "n1"}
                ],
                "accepted": 0,
                "replayed": 1,
            },
        )
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_replay_skips_the_state_check_after_the_key_moved_on(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        # The key moves on: the original expected set no longer matches.
        self.assertEqual(
            self.post_operation("r4", operation("o4", "k1", "later", {"r1": 1, "r3": 1, "r4": 1}))[0],
            201,
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 1)

    def test_replay_matches_candidates_as_a_set(self) -> None:
        self.seed_key("k1", "r1", "o1")
        self.seed_key("k1", "r2", "o2", "v2")
        doc = tx_document(
            tx_entry(
                "k1",
                "r3",
                "t1",
                "n1",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r2", "o2"), candidate("r1", "o1")],
            )
        )
        self.assertEqual(self.post_apply(doc)[0], 201)
        reordered = tx_document(
            tx_entry(
                "k1",
                "r3",
                "t1",
                "n1",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            )
        )
        status, payload = self.post_apply(reordered)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 1)

    def test_same_transaction_id_with_different_entries_is_409(self) -> None:
        self.assertEqual(self.post_apply(tx_document(tx_entry("k1", "r3", "t1")))[0], 201)
        variants = [
            tx_document(tx_entry("k1", "r3", "t1", "other")),  # different value
            tx_document(tx_entry("k1", "r3", "t1", "v", {"r3": 2})),  # different clock
            tx_document(tx_entry("k2", "r3", "t1")),  # different key
            tx_document(tx_entry("k1", "r3", "t1"), tx_entry("k2", "r3", "t2")),  # more entries
            tx_document(  # different expected set
                tx_entry("k1", "r3", "t1", "v", {"r3": 1}, [candidate("r9", "o9")])
            ),
        ]
        for doc in variants:
            with self.subTest(doc=doc):
                status, payload = self.post_apply(doc)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_mixed_new_and_replayed_entries_is_201(self) -> None:
        self.seed_key("k1", "r1", "o1")
        # "t1" is committed by an earlier transaction; the second
        # transaction replays it and adds one new entry.
        first = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            transaction_id="tx-a",
        )
        self.assertEqual(self.post_apply(first)[0], 201)
        second = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
            transaction_id="tx-b",
        )
        status, payload = self.post_apply(second)
        self.assertEqual(status, 201)
        self.assertEqual((payload["accepted"], payload["replayed"]), (1, 1))
        self.assertEqual(
            [r["operationId"] for r in payload["operations"]], ["t1", "t2"]
        )
        # The mixed transaction itself replays as a pure replay.
        status, payload = self.post_apply(second)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 2))

    def test_known_identity_with_different_content_is_operation_conflict(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r1", "o1", "n", {"r1": 1}, []),
            transaction_id="tx-new",
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)


class TransactionValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        bad_bodies = [
            b"{not json",
            b"[]",
            b"{}",
            json.dumps({"operations": []}).encode(),
            json.dumps(tx_document()).encode(),
            json.dumps({"transactionId": "", "operations": [tx_entry("k")]}).encode(),
            json.dumps(tx_document(tx_entry("k", candidates="x"))).encode(),
            json.dumps(
                tx_document(tx_entry("k"), tx_entry("k", "r4", "t2", "v", {"r4": 1}))
            ).encode(),
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload = self.post_apply(body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_query_parameter_is_400(self) -> None:
        status, payload = self.post_apply(tx_document(tx_entry("k")), path=f"{APPLY_PATH}?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, page = self.get_sync()
        self.assertEqual(page["operations"], [])

    def test_extra_path_and_trailing_slash_are_404(self) -> None:
        for path in (f"{APPLY_PATH}/extra", f"{APPLY_PATH}/"):
            status, payload = self.post_apply(tx_document(tx_entry("k")), path=path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", APPLY_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_clock_missing_own_replica_is_400(self) -> None:
        status, payload = self.post_apply(
            tx_document(tx_entry("k", "r3", "t1", "v", {"r9": 1}, []))
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_not_dominating_expected_candidates_is_400_and_unchanged(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            # The clock does not dominate {"r1": 1}.
            tx_entry("k1", "r3", "t1", "n1", {"r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # The whole transaction is unchanged: neither entry committed.
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "v1")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)
        # The transaction id was not bound: a corrected transaction commits.
        fixed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        status, payload = self.post_apply(fixed)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)


class TransactionConflictTests(HttpServerTestCase):
    def test_expected_set_mismatch_is_409_and_whole_transaction_unchanged(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),  # would commit
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, []),  # wrong expectation
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})
        # Nothing committed, including the earlier valid entry.
        status, _ = self.get_state("k2")
        self.assertEqual(status, 404)
        _, state = self.get_state("k1")
        self.assertEqual(state["value"], "v1")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_nonempty_expectation_on_candidateless_key_is_409(self) -> None:
        doc = tx_document(
            tx_entry("missing", "r3", "t1", "n", {"r3": 1}, [candidate("r1", "o1")])
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})

    def test_unknown_expected_identity_is_409(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry(
                "k1", "r3", "t1", "n", {"r1": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r9", "o9")],
            )
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "transaction_conflict"})

    def test_expectation_checked_against_staged_state_of_earlier_entries(self) -> None:
        # Two entries in one transaction target different keys, but the
        # second entry's expectation must match the state as the first
        # entry left it — here, simply the pre-existing candidate.
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k2", "r3", "t1", "n2", {"r3": 1}, []),
            tx_entry("k1", "r3", "t2", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        status, payload = self.post_apply(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)

    def test_conflict_leaves_transaction_id_unbound(self) -> None:
        self.seed_key("k1", "r1", "o1")
        bad = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, []),
        )
        self.assertEqual(self.post_apply(bad)[0], 409)
        fixed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
        )
        status, _ = self.post_apply(fixed)
        self.assertEqual(status, 201)


class TransactionConcurrencyTests(HttpServerTestCase):
    def test_concurrent_identical_transactions_commit_exactly_once(self) -> None:
        self.seed_key("k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        body = json.dumps(doc).encode("utf-8")
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            status, _ = self.request_bytes("POST", APPLY_PATH, body)
            with lock:
                results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(sorted(results), [200] * 7 + [201])
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)


class TransactionRequestLimitTests(unittest.TestCase):
    """The transaction route keeps the shared Content-Length/auth priority."""

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

    def post_raw(self, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_length_is_400(self) -> None:
        for value in ("", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            status, payload = self.post_raw([("Content-Length", value)], b"{}")
            self.assertEqual(status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_over_limit_declaration_is_413(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_declared_length_exactly_at_the_limit_is_processed(self) -> None:
        template = {
            "transactionId": "tx-1",
            "operations": [
                {
                    "key": "",
                    "replicaId": "r3",
                    "operationId": "t1",
                    "value": "v",
                    "clock": {"r3": 1},
                    "candidates": [],
                }
            ],
        }
        base = json.dumps(template, separators=(",", ":")).encode("utf-8")
        key_length = MAX_BODY_BYTES - len(base)
        self.assertGreater(key_length, 0)
        key = "k" + "x" * (key_length - 1)
        doc = json.dumps(tx_document(tx_entry(key, "r3", "t1")), separators=(",", ":")).encode(
            "utf-8"
        )
        self.assertEqual(len(doc), MAX_BODY_BYTES)
        status, payload = self.post_raw([("Content-Length", str(len(doc)))], doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["operations"][0]["key"], key)


class TransactionAuthTests(unittest.TestCase):
    def test_unauthorized_transaction_is_401_without_reading_the_body(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            APPLY_PATH,
            body=json.dumps(tx_document(tx_entry("k"))),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_authorized_transaction_succeeds_and_health_stays_anonymous(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        response.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "POST",
            APPLY_PATH,
            body=json.dumps(tx_document(tx_entry("k"))),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token",
            },
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()


class PersistentTransactionTestCase(unittest.TestCase):
    """Atomic transactions against a data-file-backed server."""

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
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            conn.request(
                method,
                path,
                body=data,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_key(self, server: SemanticStateServer, key: str, replica: str, op_id: str) -> None:
        status, _ = self.request(
            server,
            "POST",
            f"/v1/replicas/{replica}/operations",
            operation(op_id, key, f"v-{key}", {replica: 1}),
        )
        self.assertEqual(status, 201)

    def test_transaction_is_durable_with_its_binding_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
        )
        status, payload = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)

        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records],
            [("r1", "o1"), ("r3", "t1"), ("r4", "t2")],
        )
        transactions = load_data_file_transactions(str(self.data_file))
        self.assertEqual(list(transactions), ["tx-1"])
        self.assertEqual(
            transactions["tx-1"],
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "t1",
                    "value": "n1",
                    "clock": {"r1": 1, "r3": 1},
                    "candidates": [{"replicaId": "r1", "operationId": "o1"}],
                },
                {
                    "key": "k2",
                    "replicaId": "r4",
                    "operationId": "t2",
                    "value": "n2",
                    "clock": {"r4": 1},
                    "candidates": [],
                },
            ],
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The identical transaction replays as 200 after restart and
        # appends nothing; a different body under the same id conflicts.
        status, payload = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 2))
        changed = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, []),
            tx_entry("k2", "r4", "t2", "n2", {"r4": 1}, []),
        )
        status, payload = self.request(server, "POST", APPLY_PATH, changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(len(load_data_file(str(self.data_file))), 3)
        _, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "n1")

    def test_pure_replay_transaction_still_persists_its_binding(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        # A new transaction id whose only operation is already committed:
        # the response is 200, but the binding must still be durable.
        doc = tx_document(
            tx_entry("k1", "r1", "o1", "v-k1", {"r1": 1}, [candidate("r1", "o1")]),
        )
        # The entry's content equals the committed operation, so it replays.
        status, payload = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual((payload["accepted"], payload["replayed"]), (0, 1))
        self.assertEqual(list(load_data_file_transactions(str(self.data_file))), ["tx-1"])

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 1)
        changed = tx_document(tx_entry("k1", "r1", "o1", "v-k1", {"r1": 1}, []))
        status, _ = self.request(server, "POST", APPLY_PATH, changed)
        self.assertEqual(status, 409)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        before = self.data_file.read_bytes()
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")]),
            tx_entry("k2", "r3", "t2", "n2", {"r3": 1}, []),
        )
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(server, "POST", APPLY_PATH, doc)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File, memory, identity index, and bindings are exactly as before.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(load_data_file_transactions(str(self.data_file)), {})
        _, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "v-k1")
        status, _ = self.request(server, "GET", "/v1/states/k2")
        self.assertEqual(status, 404)
        status, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual(len(page["operations"]), 1)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # The same transaction commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", APPLY_PATH, doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(list(load_data_file_transactions(str(self.data_file))), ["tx-1"])

    def test_corrupt_transaction_section_fails_startup(self) -> None:
        server = self.start_server()
        self.seed_key(server, "k1", "r1", "o1")
        doc = tx_document(
            tx_entry("k1", "r3", "t1", "n1", {"r1": 1, "r3": 1}, [candidate("r1", "o1")])
        )
        self.assertEqual(self.request(server, "POST", APPLY_PATH, doc)[0], 201)
        server.shutdown()
        server.server_close()

        document = json.loads(self.data_file.read_text())
        # A binding naming an identity that was never accepted is corrupt.
        document["transactions"][0]["operations"][0]["operationId"] = "ghost"
        self.data_file.write_text(json.dumps(document))
        with self.assertRaises(server_module.PersistenceError):
            self.start_server()


if __name__ == "__main__":
    unittest.main()
