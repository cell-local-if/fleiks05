"""HTTP, sync, and persistence tests for cross-key conditional-write transactions.

The endpoint is::

    POST /v1/transactions/apply

It accepts a ``{"transactionId", "operations"}`` document with 1-100
conditional writes on distinct keys, validates every entry's expected
candidate set and dominating clock, and commits all new operations plus the
local transaction binding in one atomic commit. Everything here goes through
the real HTTP entry point (``SemanticStateServer`` + a request thread); only
the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
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
    load_data_file,
    load_data_file_state,
    parse_transaction_apply,
)

APPLY_PATH = "/v1/transactions/apply"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def tx_entry(
    key: str,
    replica: str = "r1",
    operation_id: str = "tx-op-1",
    value: str = "v",
    clock: dict | None = None,
    candidates: list | None = None,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "key": key,
        "value": value,
        "clock": clock if clock is not None else {replica: 1},
        "candidates": [] if candidates is None else candidates,
    }


def tx_document(*entries: dict, transaction_id: str = "tx-1") -> dict:
    return {"transactionId": transaction_id, "operations": list(entries)}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


class ParseTransactionApplyTests(unittest.TestCase):
    def test_valid_transaction_is_normalized_in_order(self) -> None:
        transaction_id, entries = parse_transaction_apply(
            json.dumps(
                tx_document(
                    tx_entry("k1", "r1", "o1", "v1", {"r1": 2}, [identity("r0", "o0")]),
                    tx_entry("k2", "r2", "o2", "v2", {"r2": 1}),
                    transaction_id="tx-9",
                )
            )
        )
        self.assertEqual(transaction_id, "tx-9")
        self.assertEqual(
            entries,
            [
                {
                    "replicaId": "r1",
                    "operationId": "o1",
                    "key": "k1",
                    "value": "v1",
                    "clock": {"r1": 2},
                    "candidates": [{"replicaId": "r0", "operationId": "o0"}],
                },
                {
                    "replicaId": "r2",
                    "operationId": "o2",
                    "key": "k2",
                    "value": "v2",
                    "clock": {"r2": 1},
                    "candidates": [],
                },
            ],
        )

    def test_one_hundred_entries_are_accepted(self) -> None:
        entries = [tx_entry(f"k{i:03d}", "r1", f"o{i:03d}") for i in range(100)]
        transaction_id, parsed = parse_transaction_apply(
            json.dumps(tx_document(*entries))
        )
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
            {"operations": []},
            dict(valid, extra=1),
            {"transactionId": "tx-1", "operations": valid["operations"], "x": 2},
            {"transactionId": "", "operations": valid["operations"]},
            {"transactionId": 7, "operations": valid["operations"]},
            {"transactionId": None, "operations": valid["operations"]},
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_transaction_apply(body)

    def test_rejects_empty_and_oversized_batches(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(tx_document())
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(*[tx_entry(f"k{i}", "r1", f"o{i}") for i in range(101)])
            )

    def test_rejects_bad_entry_shapes(self) -> None:
        good = tx_entry("k")
        bad_entries = [
            [],
            "x",
            {},
            *(
                {key: value for key, value in good.items() if key != field}
                for field in ("replicaId", "operationId", "key", "value", "clock", "candidates")
            ),
            dict(good, extra=1),
            dict(good, replicaId=""),
            dict(good, replicaId=3),
            dict(good, operationId=""),
            dict(good, key=""),
            dict(good, key=1),
            dict(good, value=""),
            dict(good, clock={}),
            dict(good, clock={"other": 1}),  # must contain the replica
            dict(good, clock={"r1": -1}),
            dict(good, clock={"r1": True}),
            dict(good, clock={"r1": 1.0}),
            dict(good, clock={"r1": "1"}),
            dict(good, candidates={}),
            dict(good, candidates="r1"),
            dict(good, candidates=[{"replicaId": "r1"}]),
            dict(good, candidates=[{"operationId": "o1"}]),
            dict(good, candidates=[{"replicaId": "r1", "operationId": "o1", "x": 1}]),
            dict(good, candidates=[{"replicaId": "", "operationId": "o1"}]),
            dict(good, candidates=[{"replicaId": "r1", "operationId": ""}]),
        ]
        for entry in bad_entries:
            with self.assertRaises(ValueError, msg=repr(entry)):
                parse_transaction_apply(tx_document(entry))

    def test_rejects_duplicate_candidates_within_an_entry(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(
                    tx_entry("k", candidates=[identity("r1", "o1"), identity("r1", "o1")])
                )
            )

    def test_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(tx_entry("k", "r1", "o1"), tx_entry("k", "r2", "o2", clock={"r2": 1}))
            )

    def test_rejects_duplicate_identities_even_on_different_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_transaction_apply(
                tx_document(
                    tx_entry("k1", "r1", "same-op"),
                    tx_entry("k2", "r1", "same-op"),
                )
            )

    def test_rejects_non_finite_clock_values(self) -> None:
        for literal in ("NaN", "Infinity", "-Infinity", "1e3"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"transactionId":"tx-1","operations":[{"replicaId":"r1",'
                    b'"operationId":"o1","key":"k","value":"v","clock":{"r1":'
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

    def post_apply(self, body: object) -> tuple[int, object]:
        return self.request("POST", APPLY_PATH, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def seed(self, replica: str, op_id: str, key: str, value: str, clock: dict) -> None:
        status, _ = self.post_operation(replica, operation(op_id, key, value, clock))
        self.assertEqual(status, 201)


class TransactionApplyHttpTests(HttpServerTestCase):
    def test_create_on_empty_keys_returns_201_and_commits_all_entries(self) -> None:
        status, payload = self.post_apply(
            tx_document(
                tx_entry("k1", "r1", "o1", "blue", {"r1": 1}),
                tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
                transaction_id="tx-1",
            )
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r1", "operationId": "o1"},
                    {"key": "k2", "replicaId": "r2", "operationId": "o2"},
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        for key, value in (("k1", "blue"), ("k2", "large")):
            status, state = self.get_state(key)
            self.assertEqual(status, 200)
            self.assertEqual(state["status"], "resolved")
            self.assertEqual(state["value"], value)

    def test_create_with_matching_expectation_replaces_candidates(self) -> None:
        self.seed("r1", "o1", "color", "red", {"r1": 1})
        status, payload = self.post_apply(
            tx_document(
                tx_entry(
                    "color", "r2", "o2", "blue", {"r1": 1, "r2": 1},
                    [identity("r1", "o1")],
                )
            )
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        status, state = self.get_state("color")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "blue")

    def test_expectation_mismatch_is_transaction_conflict_and_changes_nothing(self) -> None:
        self.seed("r1", "o1", "color", "red", {"r1": 1})
        mismatches = [
            # Expects no candidates, but the key has one.
            tx_document(tx_entry("color", "r2", "o2", "blue", {"r1": 1, "r2": 1})),
            # Expects a different identity than the current candidate.
            tx_document(
                tx_entry(
                    "color", "r2", "o2", "blue", {"r1": 1, "r2": 1},
                    [identity("r1", "other")],
                )
            ),
            # Expects an extra candidate the key does not hold.
            tx_document(
                tx_entry(
                    "color", "r2", "o2", "blue", {"r1": 1, "r2": 1},
                    [identity("r1", "o1"), identity("r3", "o3")],
                )
            ),
            # Expects a candidate on a key that has none.
            tx_document(
                tx_entry(
                    "missing", "r2", "o2", "blue", {"r2": 1}, [identity("r1", "o1")]
                )
            ),
        ]
        for document in mismatches:
            with self.subTest(document=document):
                status, payload = self.post_apply(document)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "transaction_conflict"})
        _, state = self.get_state("color")
        self.assertEqual(state["value"], "red")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_non_dominating_clock_is_invalid_request_and_changes_nothing(self) -> None:
        self.seed("r1", "o1", "color", "red", {"r1": 2})
        status, payload = self.post_apply(
            tx_document(
                tx_entry(
                    "color", "r2", "o2", "blue", {"r2": 1}, [identity("r1", "o1")]
                )
            )
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("color")
        self.assertEqual(state["value"], "red")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_batch_is_atomic_when_a_later_entry_fails(self) -> None:
        self.seed("r1", "o1", "k2", "red", {"r1": 5})
        status, payload = self.post_apply(
            tx_document(
                tx_entry("k1", "r2", "o2", "blue", {"r2": 1}),
                # Legal shape, but the clock does not dominate the candidate.
                tx_entry("k2", "r2", "o3", "green", {"r2": 1}, [identity("r1", "o1")]),
            )
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, _ = self.get_state("k1")
        self.assertEqual(status, 404)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_replay_same_transaction_returns_200_without_rechecking_state(self) -> None:
        document = tx_document(
            tx_entry("k1", "r1", "o1", "blue", {"r1": 1}),
            tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
        )
        status, _ = self.post_apply(document)
        self.assertEqual(status, 201)
        # Move the state on so a state re-check would fail.
        self.seed("r3", "o3", "k1", "green", {"r1": 1, "r3": 1})
        status, payload = self.post_apply(document)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "ok",
                "transactionId": "tx-1",
                "operations": [
                    {"key": "k1", "replicaId": "r1", "operationId": "o1"},
                    {"key": "k2", "replicaId": "r2", "operationId": "o2"},
                ],
                "accepted": 0,
                "replayed": 2,
            },
        )
        # No new log records: the two transaction operations plus the later write.
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 3)

    def test_same_transaction_id_with_different_entries_is_operation_conflict(self) -> None:
        status, _ = self.post_apply(tx_document(tx_entry("k1", "r1", "o1")))
        self.assertEqual(status, 201)
        variants = [
            tx_document(tx_entry("k1", "r1", "o1", "other")),  # different value
            tx_document(tx_entry("k2", "r1", "o1")),  # different key
            tx_document(tx_entry("k1", "r1", "o1", clock={"r1": 2})),  # different clock
            tx_document(  # different expected set
                tx_entry("k1", "r1", "o1", candidates=[identity("r9", "o9")])
            ),
            tx_document(  # different entry count
                tx_entry("k1", "r1", "o1"), tx_entry("k2", "r2", "o2", clock={"r2": 1})
            ),
            tx_document(tx_entry("k1", "r1", "o2")),  # different operation id
        ]
        for document in variants:
            with self.subTest(document=document):
                status, payload = self.post_apply(document)
                self.assertEqual(status, 409)
                self.assertEqual(payload, {"error": "operation_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_entry_identity_with_different_committed_content_is_operation_conflict(self) -> None:
        self.seed("r1", "o1", "color", "red", {"r1": 1})
        status, payload = self.post_apply(
            tx_document(tx_entry("other", "r1", "o1", "blue", {"r1": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)

    def test_entry_identity_with_identical_content_is_a_replay(self) -> None:
        self.seed("r1", "o1", "color", "red", {"r1": 1})
        # A new transaction whose only entry repeats a committed operation.
        status, payload = self.post_apply(
            tx_document(tx_entry("color", "r1", "o1", "red", {"r1": 1}))
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 1)
        # The binding is still recorded: a different reuse of the id conflicts.
        status, payload = self.post_apply(
            tx_document(tx_entry("color", "r1", "o1", "blue", {"r1": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_mixed_new_and_replayed_entries_commit_together(self) -> None:
        self.seed("r1", "o1", "k1", "red", {"r1": 1})
        status, payload = self.post_apply(
            tx_document(
                tx_entry("k1", "r1", "o1", "red", {"r1": 1}),
                tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
            )
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_operations_flow_to_sync_audit_metrics_and_archive(self) -> None:
        status, _ = self.post_apply(
            tx_document(
                tx_entry("k1", "r1", "o1", "blue", {"r1": 1}),
                tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
            )
        )
        self.assertEqual(status, 201)
        status, page = self.get_sync()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in page["operations"]],
            [("r1", "o1"), ("r2", "o2")],
        )
        status, audit = self.request("GET", "/v1/audit/keys/k1/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(audit["operations"]), 1)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["acceptedOperations"], 2)
        self.assertEqual(metrics["keys"], 2)
        status, record = self.request("GET", "/v1/replicas/r1/operations/o1")
        self.assertEqual(status, 200)
        self.assertEqual(record["operation"]["key"], "k1")

    def test_malformed_body_and_fields_are_400(self) -> None:
        bad_bodies = [
            b"{not json",
            b"[]",
            b"{}",
            json.dumps({"transactionId": "tx-1"}).encode(),
            json.dumps(tx_document()).encode(),  # empty batch
            json.dumps(tx_document(tx_entry("k", clock={"other": 1}))).encode(),
            json.dumps(tx_document(tx_entry("k", candidates=[identity("a", "b"), identity("a", "b")]))).encode(),
            json.dumps(
                tx_document(tx_entry("k", "r1", "o1"), tx_entry("k", "r2", "o2", clock={"r2": 1}))
            ).encode(),
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload = self.request("POST", APPLY_PATH, body)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        _, page = self.get_sync()
        self.assertEqual(page["operations"], [])

    def test_query_parameters_are_400_and_extra_paths_404(self) -> None:
        document = tx_document(tx_entry("k"))
        for path in (
            "/v1/transactions/apply?x=1",
            "/v1/transactions/apply?x=1&x=2",
            "/v1/transactions/apply?x",
        ):
            with self.subTest(path=path):
                status, payload = self.request("POST", path, document)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for path in (
            "/v1/transactions/apply/extra",
            "/v1/transactions/apply/",
            "/v1/transactions",
        ):
            with self.subTest(path=path):
                status, payload = self.request("POST", path, document)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})
        _, page = self.get_sync()
        self.assertEqual(page["operations"], [])

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", APPLY_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_content_length_contract_keeps_priority(self) -> None:
        # A missing Content-Length is a 400 before anything else.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        conn.close()
        # An over-limit declaration is a 413 before the body is read.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        conn.putheader("Content-Length", str(1_048_577))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read().decode("utf-8")), {"error": "payload_too_large"})
        conn.close()


class PersistentTransactionTestCase(unittest.TestCase):
    """Cross-key transactions against a data-file-backed server."""

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

    def test_transaction_and_binding_are_durable_and_recover(self) -> None:
        server = self.start_server()
        document = tx_document(
            tx_entry("k1", "r1", "o1", "blue", {"r1": 1}),
            tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
        )
        status, _ = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 201)

        records, _, _, transactions = load_data_file_state(str(self.data_file))
        self.assertEqual(
            [(r, o["operationId"]) for r, o in records], [("r1", "o1"), ("r2", "o2")]
        )
        self.assertEqual(set(transactions), {"tx-1"})
        self.assertEqual(
            transactions["tx-1"],
            [
                {
                    "replicaId": "r1",
                    "operationId": "o1",
                    "key": "k1",
                    "value": "blue",
                    "clock": {"r1": 1},
                    "candidates": [],
                },
                {
                    "replicaId": "r2",
                    "operationId": "o2",
                    "key": "k2",
                    "value": "large",
                    "clock": {"r2": 1},
                    "candidates": [],
                },
            ],
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The 201/200/409 judgments are identical after the restart.
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual(len(load_data_file(str(self.data_file))), 2)
        status, payload = self.request(
            server, "POST", APPLY_PATH, tx_document(tx_entry("k1", "r1", "o1", "other"))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        status, state = self.request(server, "GET", "/v1/states/k1")
        self.assertEqual(state["value"], "blue")

    def test_pure_replay_transaction_binding_is_persisted(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k1", "red", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        # A new transaction whose only entry replays the committed operation
        # returns 200 but still records (and persists) its binding.
        document = tx_document(tx_entry("k1", "r1", "o1", "red", {"r1": 1}))
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        _, _, _, transactions = load_data_file_state(str(self.data_file))
        self.assertEqual(set(transactions), {"tx-1"})

        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.request(
            server, "POST", APPLY_PATH, tx_document(tx_entry("k1", "r1", "o1", "blue", {"r1": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        before = self.data_file.read_bytes()
        document = tx_document(
            tx_entry("k1", "r1", "o1", "blue", {"r1": 1}),
            tx_entry("k2", "r2", "o2", "large", {"r2": 1}),
        )
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(server, "POST", APPLY_PATH, document)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File, memory, identity index, and bindings are exactly as before.
        self.assertEqual(self.data_file.read_bytes(), before)
        _, _, _, transactions = load_data_file_state(str(self.data_file))
        self.assertEqual(transactions, {})
        for key in ("k1", "k2"):
            status, _ = self.request(server, "GET", f"/v1/states/{key}")
            self.assertEqual(status, 404)
        status, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual(page["operations"], [])
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # The same transaction commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)

    def test_corrupt_transactions_section_fails_startup(self) -> None:
        base_operations = [
            {"replicaId": "r1", "operation": operation("o1", "k1", "v", {"r1": 1})}
        ]
        good_entry = tx_entry("k1", "r1", "o1", "v", {"r1": 1})
        bad_sections = [
            {},  # must be a list
            [{"operations": [good_entry]}],  # missing transactionId
            [{"transactionId": "tx-1"}],  # missing operations
            [{"transactionId": "", "operations": [good_entry]}],
            [{"transactionId": "tx-1", "operations": []}],  # empty batch
            [{"transactionId": "tx-1", "operations": [dict(good_entry, clock={"r9": 1})]}],
            # A binding entry naming no accepted operation.
            [{"transactionId": "tx-1", "operations": [tx_entry("k9", "r9", "o9", clock={"r9": 1})]}],
            # Duplicate transaction identifiers.
            [
                {"transactionId": "tx-1", "operations": [good_entry]},
                {"transactionId": "tx-1", "operations": [good_entry]},
            ],
        ]
        for section in bad_sections:
            with self.subTest(section=section):
                self.data_file.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "operations": base_operations,
                            "transactions": section,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(server_module.PersistenceError):
                    StateStore(data_file=str(self.data_file))

    def test_file_without_transactions_section_recovers_empty(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k1", "v", {"r1": 1})}
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(data_file=str(self.data_file))
        self.assertEqual(store._transactions, {})
        # A transaction on the recovered state behaves normally.
        status, results, accepted, replayed, error = store.apply_transaction(
            "tx-1",
            [
                {
                    "replicaId": "r1",
                    "operationId": "o2",
                    "key": "k1",
                    "value": "blue",
                    "clock": {"r1": 2},
                    "candidates": [{"replicaId": "r1", "operationId": "o1"}],
                }
            ],
        )
        self.assertEqual(status, 201)
        self.assertEqual(accepted, 1)
        self.assertEqual(replayed, 0)
        self.assertIsNone(error)
        self.assertEqual(results, [{"key": "k1", "replicaId": "r1", "operationId": "o2"}])


class AuthTransactionTestCase(unittest.TestCase):
    """The transaction endpoint authenticates like every other POST route."""

    def setUp(self) -> None:
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, path: str, body: object, token: str | None) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")) if raw else None

    def test_unauthorized_request_is_401_and_changes_nothing(self) -> None:
        document = tx_document(tx_entry("k1", "r1", "o1"))
        for token in (None, "wrong"):
            with self.subTest(token=token):
                status, payload = self.request(APPLY_PATH, document, token)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.request(APPLY_PATH, document, "secret-token")
        self.assertEqual(status, 201)
        # The committed transaction replays under the valid token.
        status, payload = self.request(APPLY_PATH, document, "secret-token")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
