"""Tests for the read-only repair-execution external verification::

    GET /v1/replication/repairs/executions/verify?after=N&limit=N
        &expectedDigest=H&expectedCount=C

The endpoint is the external-verification companion of the repair
execution audit: it pages the committed history of conditional repair
executions exactly like ``GET /v1/replication/repairs/executions``
(same field order, record structure, and after/limit paging) and adds
two external comparisons — ``expectedDigest`` against the digest
independently recomputed over the complete creation-order history, and
``expectedCount`` against the full history length. The ``verification``
conclusion keeps the audit's five internal anomaly lists and appends
``digestMismatches`` and ``countMismatches``; ``status`` is ``"ok"``
exactly when all seven lists are empty.

The tests cover the query parser, the external verification scan, the
store's paging/snapshot/recovery semantics, the HTTP request precedence
chain (404 path shape, 401 authentication with a Bearer challenge, 403
in scope mode without one, 400 query validation including an ``after``
past the execution count), the compact ordered single-newline response
body with its explicit Content-Length, restart consistency, and the
strict read-only guarantee. Only the Python standard library is used.
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
    _repair_executions_digest_input,
    _repair_executions_external_verification_locked,
    _repair_receipts_digest,
    load_scope_policy,
    parse_replication_repair_executions_verify_query,
)

VERIFY_PATH = "/v1/replication/repairs/executions/verify"
EXECUTIONS_PATH = "/v1/replication/repairs/executions"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
OTHER_DIGEST = "b" * 64

RESPONSE_FIELDS = [
    "executions",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "executionsCount",
    "verification",
]
EXECUTION_FIELDS = [
    "peerId",
    "ackId",
    "expectedCheckpoint",
    "expectedReceipts",
    "suggestions",
    "results",
    "cursor",
]
VERIFICATION_FIELDS = [
    "status",
    "duplicateBindings",
    "outOfOrderActions",
    "boundaryViolations",
    "checkpointViolations",
    "recordViolations",
    "digestMismatches",
    "countMismatches",
]
OK_VERIFICATION = {name: [] for name in VERIFICATION_FIELDS if name != "status"}
OK_VERIFICATION["status"] = "ok"
OK_VERIFICATION = {name: OK_VERIFICATION[name] for name in VERIFICATION_FIELDS}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def interval(action: str, ack_id: str, start: int, end: int) -> dict:
    return {
        "action": action,
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def gap_record(
    peer_id: str = "peer-a",
    ack_id: str = "exec-1",
    *,
    expected_checkpoint: int = 4,
    cursor: int = 4,
    digest: str = "a" * 64,
) -> dict:
    """One well-formed stored execution carrying a single resend action."""
    suggestion = interval("resend", "ack-2", 1, 2)
    return {
        "peerId": peer_id,
        "ackId": ack_id,
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": digest,
        "suggestions": [suggestion],
        "results": [{"action": "resend", "boundary": {"start": 1, "end": 2}}],
        "cursor": cursor,
    }


def history_digest(entries: list) -> str:
    return hashlib.sha256(_repair_executions_digest_input(entries)).hexdigest()


class ParseRepairExecutionsVerifyQueryTests(unittest.TestCase):
    def test_accepts_all_four_required_parameters(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            parse_replication_repair_executions_verify_query(
                f"after=0&limit=1&expectedDigest={digest}&expectedCount=0"
            ),
            (0, 1, digest, 0),
        )
        self.assertEqual(
            parse_replication_repair_executions_verify_query(
                f"expectedCount=42&expectedDigest={digest}&limit=100&after=7"
            ),
            (7, 100, digest, 42),
        )
        self.assertEqual(
            parse_replication_repair_executions_verify_query(
                f"after=007&limit=09&expectedDigest={digest}&expectedCount=003"
            ),
            (7, 9, digest, 3),
        )

    def test_rejects_missing_repeated_unknown_and_blank(self) -> None:
        digest = "a" * 64
        good_tail = f"&expectedDigest={digest}&expectedCount=0"
        bad = [
            "",
            "after=0&limit=1",
            f"after=0&limit=1&expectedDigest={digest}",
            f"after=0&limit=1&expectedCount=0",
            f"after=0&limit=1{good_tail}&after=2",
            f"after=0&limit=1{good_tail}&limit=2",
            f"after=0&limit=1{good_tail}&expectedDigest={digest}",
            f"after=0&limit=1{good_tail}&expectedCount=1",
            f"after=0&limit=1{good_tail}&x=1",
            f"x=1&after=0&limit=1{good_tail}",
            f"after=&limit=1{good_tail}",
            f"after=0&limit={good_tail}",
            f"after=0&limit=1&expectedDigest=&expectedCount=0",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=",
            f"after&limit=1{good_tail}",
            f"after=0&limit=1{good_tail}&=",
            f"=1&after=0&limit=1{good_tail}",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(query)
                )

    def test_rejects_malformed_digest(self) -> None:
        tail = "&after=0&limit=1&expectedCount=0"
        bad = [
            "a" * 63,  # too short
            "a" * 65,  # too long
            "A" * 64,  # uppercase
            "g" * 64,  # non-hex
            "",  # blank
            "%20" + "a" * 63,  # whitespace
        ]
        for digest in bad:
            with self.subTest(digest=digest):
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(
                        f"expectedDigest={digest}{tail}"
                    )
                )

    def test_rejects_signed_decimal_and_non_ascii_numbers(self) -> None:
        digest = "a" * 64
        bad = [
            f"after=-1&limit=1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=-1&expectedDigest={digest}&expectedCount=0",
            f"after=+0&limit=1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=+1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=-1",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=+1",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=1.0",
            f"after=1.0&limit=1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=1.0&expectedDigest={digest}&expectedCount=0",
            f"after=%200&limit=1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=1%20&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=%200",
            f"after=%C2%B2&limit=1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=%D9%A1&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={digest}&expectedCount=%D9%A1",
            f"after=0&limit=0&expectedDigest={digest}&expectedCount=0",
            f"after=0&limit=101&expectedDigest={digest}&expectedCount=0",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(query)
                )


class RepairExecutionsExternalVerificationTests(unittest.TestCase):
    def test_empty_history_matches_empty_expectations(self) -> None:
        self.assertEqual(
            _repair_executions_external_verification_locked(
                [], {}, 0, EMPTY_DIGEST, 0
            ),
            OK_VERIFICATION,
        )

    def test_empty_history_reports_mismatched_expectations(self) -> None:
        verdict = _repair_executions_external_verification_locked(
            [], {}, 0, OTHER_DIGEST, 1
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": OTHER_DIGEST, "observed": EMPTY_DIGEST}],
        )
        self.assertEqual(
            verdict["countMismatches"], [{"expected": 1, "observed": 0}]
        )

    def test_well_formed_history_with_matching_expectations_is_ok(self) -> None:
        entries = [gap_record(), gap_record("peer-b", "exec-b")]
        verdict = _repair_executions_external_verification_locked(
            entries,
            {"peer-a": 4, "peer-b": 4},
            4,
            history_digest(entries),
            2,
        )
        self.assertEqual(verdict, OK_VERIFICATION)
        self.assertEqual(list(verdict), VERIFICATION_FIELDS)

    def test_digest_mismatch_keeps_expected_first(self) -> None:
        entries = [gap_record()]
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4}, 4, OTHER_DIGEST, 1
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": OTHER_DIGEST, "observed": history_digest(entries)}],
        )
        self.assertEqual(verdict["countMismatches"], [])

    def test_count_mismatch_keeps_expected_first(self) -> None:
        entries = [gap_record()]
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4}, 4, history_digest(entries), 5
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [{"expected": 5, "observed": 1}])

    def test_internal_damage_and_external_mismatch_combine(self) -> None:
        entries = [gap_record(cursor=2)]  # checkpoint advancement violation
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4}, 6, OTHER_DIGEST, 9
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(len(verdict["checkpointViolations"]), 1)
        self.assertEqual(len(verdict["digestMismatches"]), 1)
        self.assertEqual(verdict["countMismatches"], [{"expected": 9, "observed": 1}])

    def test_internal_damage_alone_is_broken_even_with_matching_expectations(
        self,
    ) -> None:
        entries = [gap_record(cursor=2)]
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4}, 6, history_digest(entries), 1
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])


class RepairExecutionsVerifyStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int, key: str = "k") -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", key, f"v{index}", {f"r{index}": 1}),
            )
            self.assertIs(status, HTTPStatus.CREATED)

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.store.save_checkpoint(peer, cursor)
        self.assertIs(status, HTTPStatus.OK)

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def advice(self, peer: str) -> list[dict]:
        status, payload = self.store.get_replication_repairs(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        return [item for item in payload["suggestions"] if item["peer"] == peer]

    def receipts_digest(self, peer: str) -> str:
        committed = [
            (ack_id, receipt)
            for (receipt_peer, ack_id), receipt in self.store._acks.items()
            if receipt_peer == peer
        ]
        return _repair_receipts_digest(peer, committed)

    @staticmethod
    def suggestion(item: dict) -> dict:
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        return suggestion

    def gap_repair(self, peer: str, ack_id: str):
        """Commit one resend-gap repair for a peer with the standard seed."""
        item = self.advice(peer)[0]
        checkpoint = self.store.get_checkpoint(peer)[1]["cursor"]
        return self.store.apply_replication_repairs(
            peer,
            ack_id,
            checkpoint,
            self.receipts_digest(peer),
            [self.suggestion(item)],
        )

    def seed_gap(self, peer: str, *, first: str = "ack-1", second: str = "ack-2") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, first, 1, [identity("r0", "o0")])
        self.seed_receipt(peer, second, 3, [identity("r2", "o2")])

    def committed_digest(self) -> str:
        history = [
            {"peerId": peer_id, "ackId": ack_id, **binding}
            for (peer_id, ack_id), binding in self.store._repairs.items()
        ]
        return history_digest(history)


class RepairExecutionsVerifyStoreTests(RepairExecutionsVerifyStoreFixture):
    def verify(self, after, limit, expected_digest, expected_count):
        return self.store.verify_replication_repair_executions(
            after, limit, expected_digest, expected_count
        )

    def test_empty_history_matches_empty_expectations(self) -> None:
        status, payload = self.verify(0, 100, EMPTY_DIGEST, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)

    def test_matching_expectations_are_ok(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        status, _, error = self.gap_repair("peer-a", "exec-1")
        self.assertIs(status, HTTPStatus.CREATED, error)
        status, payload = self.verify(0, 100, self.committed_digest(), 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(payload["executionsCount"], 1)
        execution = payload["executions"][0]
        self.assertEqual(list(execution), EXECUTION_FIELDS)

    def test_page_fields_match_the_plain_audit_query(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.gap_repair("peer-a", "exec-a1")
        self.gap_repair("peer-b", "exec-b1")
        self.gap_repair("peer-a", "exec-a2")
        digest = self.committed_digest()
        for after, limit in ((0, 100), (0, 1), (1, 1), (2, 1), (3, 10)):
            with self.subTest(after=after, limit=limit):
                _, plain = self.store.get_replication_repair_executions(after, limit)
                status, verified = self.verify(after, limit, digest, 3)
                self.assertIs(status, HTTPStatus.OK)
                for field in RESPONSE_FIELDS[:-1]:
                    self.assertEqual(verified[field], plain[field])
                self.assertEqual(verified["verification"], OK_VERIFICATION)

    def test_digest_mismatch_is_broken_with_expected_first(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        status, payload = self.verify(0, 100, OTHER_DIGEST, 1)
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": OTHER_DIGEST, "observed": payload["digest"]}],
        )
        self.assertEqual(verdict["countMismatches"], [])

    def test_count_mismatch_is_broken_with_expected_first(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        status, payload = self.verify(0, 100, self.committed_digest(), 7)
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [{"expected": 7, "observed": 1}])

    def test_paging_never_changes_digest_count_or_conclusion(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.gap_repair("peer-a", "exec-a1")
        self.gap_repair("peer-b", "exec-b1")
        self.gap_repair("peer-a", "exec-a2")
        digest = self.committed_digest()
        _, full = self.verify(0, 100, digest, 3)
        pages = []
        for after in range(4):
            status, page = self.verify(after, 1, digest, 3)
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
        self.assertEqual(
            [[e["ackId"] for e in p["executions"]] for p in pages],
            [["exec-a1"], ["exec-a2"], ["exec-b1"], []],
        )
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])
        for page in pages:
            self.assertEqual(page["digest"], full["digest"])
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"], full["verification"])

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        digest = self.committed_digest()
        status, first = self.verify(1, 10, digest, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(first["executions"], [])
        self.assertEqual(first["nextCursor"], 1)
        self.assertFalse(first["hasMore"])
        self.assertEqual(first["executionsCount"], 1)
        self.assertEqual(first["verification"], OK_VERIFICATION)
        status, second = self.verify(1, 10, digest, 1)
        self.assertEqual(second, first)

    def test_after_past_count_raises_value_error(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        with self.assertRaises(ValueError):
            self.verify(2, 1, self.committed_digest(), 1)

    def test_query_is_strictly_read_only(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        digest = self.committed_digest()
        repairs_before = copy.deepcopy(dict(self.store._repairs))
        checkpoints_before = dict(self.store._checkpoints)
        acks_before = copy.deepcopy(dict(self.store._acks))
        metrics_before = self.store.get_metrics()
        advice_before = self.store.get_replication_repairs(0, 100)[1]
        for query in (
            (0, 100, digest, 1),
            (0, 1, digest, 1),
            (1, 1, digest, 1),
            (0, 100, OTHER_DIGEST, 9),
        ):
            self.verify(*query)
        self.assertEqual(self.store._repairs, repairs_before)
        self.assertEqual(self.store._checkpoints, checkpoints_before)
        self.assertEqual(self.store._acks, acks_before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_replication_repairs(0, 100)[1], advice_before)

    def test_internal_damage_stays_broken_with_matching_expectations(self) -> None:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.gap_repair("peer-a", "exec-1")
        # Roll the registered checkpoint behind the execution's restored
        # cursor: the internal scan must flag it even though the external
        # expectations are recomputed from the same damaged snapshot.
        self.store._checkpoints["peer-a"] = 0
        status, payload = self.verify(0, 100, self.committed_digest(), 1)
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(len(verdict["checkpointViolations"]), 1)
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])


class RepairExecutionsVerifyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        store = StateStore(data_file=self.data_file)
        for index in range(4):
            store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
        helper = RepairExecutionsVerifyStoreFixture()
        helper.store = store
        helper.seed_gap("peer-a")
        helper.seed_gap("peer-b", first="ack-3", second="ack-4")
        helper.gap_repair("peer-a", "exec-a1")
        helper.gap_repair("peer-b", "exec-b1")
        helper.gap_repair("peer-a", "exec-a2")
        digest = helper.committed_digest()
        _, before = store.verify_replication_repair_executions(0, 2, digest, 3)
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.verify_replication_repair_executions(0, 2, digest, 3)
        self.assertEqual(after, before)
        self.assertEqual(after["verification"], OK_VERIFICATION)

    def test_old_file_without_section_recovers_empty(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=self.data_file)
        status, payload = store.verify_replication_repair_executions(
            0, 100, EMPTY_DIGEST, 0
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["verification"], OK_VERIFICATION)


class RepairExecutionsVerifyHttpFixture(unittest.TestCase):
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

    def get(self, query: str):
        return self.raw_request("GET", VERIFY_PATH + query)

    def good_query(self, digest: str = EMPTY_DIGEST, count: int = 0) -> str:
        return f"?after=0&limit=100&expectedDigest={digest}&expectedCount={count}"

    def seed(self, count: int = 4) -> None:
        for index in range(count):
            status, _, _, _ = self.raw_request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int) -> None:
        status, _, _, _ = self.raw_request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status == 200

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.server.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def apply(self, peer: str, ack_id: str) -> None:
        status, advice, _, _ = self.raw_request(
            "GET", "/v1/replication/repairs?after=0&limit=100"
        )
        assert status == 200
        item = next(entry for entry in advice["suggestions"] if entry["peer"] == peer)
        status, receipts, _, _ = self.raw_request(
            "GET", f"/v1/sync/peers/{peer}/receipts?after=0&limit=100"
        )
        assert status == 200
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        status, payload, _, _ = self.raw_request(
            "POST",
            "/v1/replication/repairs/apply",
            {
                "peerId": peer,
                "ackId": ack_id,
                "expectedCheckpoint": self.server.store.get_checkpoint(peer)[1][
                    "cursor"
                ],
                "expectedReceipts": receipts["digest"],
                "suggestions": [suggestion],
            },
        )
        self.assertEqual(status, 201, payload)

    def seed_gap(self, peer: str, *, first: str = "ack-1", second: str = "ack-2") -> None:
        self.register(peer, 4)
        self.seed_receipt(peer, first, 1, [identity("r0", "o0")])
        self.seed_receipt(peer, second, 3, [identity("r2", "o2")])

    def committed_digest(self) -> str:
        history = [
            {"peerId": peer_id, "ackId": ack_id, **binding}
            for (peer_id, ack_id), binding in self.server.store._repairs.items()
        ]
        return history_digest(history)


class RepairExecutionsVerifyHttpTests(RepairExecutionsVerifyHttpFixture):
    def test_empty_history_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get(self.good_query())
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            raw,
            b'{"executions":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + EMPTY_DIGEST.encode("ascii")
            + b'","executionsCount":0,"verification":{"status":"ok",'
            b'"duplicateBindings":[],"outOfOrderActions":[],'
            b'"boundaryViolations":[],"checkpointViolations":[],'
            b'"recordViolations":[],"digestMismatches":[],'
            b'"countMismatches":[]}}\n',
        )

    def test_matching_expectations_verify_ok(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.apply("peer-a", "exec-a1")
        self.apply("peer-b", "exec-b1")
        self.apply("peer-a", "exec-a2")
        digest = self.committed_digest()
        status, payload, raw, headers = self.get(self.good_query(digest, 3))
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(
            [(e["peerId"], e["ackId"]) for e in payload["executions"]],
            [("peer-a", "exec-a1"), ("peer-a", "exec-a2"), ("peer-b", "exec-b1")],
        )
        self.assertEqual(payload["nextCursor"], 3)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 3)
        self.assertEqual(payload["digest"], digest)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        execution = payload["executions"][0]
        self.assertEqual(list(execution), EXECUTION_FIELDS)
        # Page through with limit=1: digest, count, and verification are
        # page-independent.
        for after in range(4):
            status, page, _, _ = self.get(
                f"?after={after}&limit=1&expectedDigest={digest}&expectedCount=3"
            )
            self.assertEqual(status, 200)
            self.assertEqual(page["digest"], digest)
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_mismatched_expectations_are_broken_with_expected_first(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest = self.committed_digest()
        status, payload, _, _ = self.get(self.good_query(OTHER_DIGEST, 9))
        self.assertEqual(status, 200)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"], [{"expected": OTHER_DIGEST, "observed": digest}]
        )
        self.assertEqual(verdict["countMismatches"], [{"expected": 9, "observed": 1}])
        # The page, digest, and count are unaffected by the mismatch.
        self.assertEqual(payload["digest"], digest)
        self.assertEqual(payload["executionsCount"], 1)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest = self.committed_digest()
        status, payload, raw, _ = self.get(
            f"?after=1&limit=10&expectedDigest={digest}&expectedCount=1"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 1)
        self.assertEqual(payload["verification"], OK_VERIFICATION)

    def test_bad_queries_are_400_and_change_nothing(self) -> None:
        self.seed(1)
        self.register("peer-a", 1)
        digest = "a" * 64
        tail = f"&expectedDigest={digest}&expectedCount=0"
        bad_queries = [
            "",
            "?",
            "?after=0&limit=1",
            f"?after=0&limit=1&expectedDigest={digest}",
            f"?after=0&limit=1&expectedCount=0",
            f"?after=0&limit=1{tail}&after=2",
            f"?after=0&limit=1{tail}&limit=2",
            f"?after=0&limit=1{tail}&expectedDigest={digest}",
            f"?after=0&limit=1{tail}&expectedCount=1",
            f"?after=0&limit=1{tail}&x=1",
            "?x=1",
            f"?after=&limit=1{tail}",
            f"?after=0&limit={tail}",
            f"?after=0&limit=1&expectedDigest=&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=",
            f"?after=-1&limit=1{tail}",
            f"?after=+1&limit=1{tail}",
            f"?after=1.0&limit=1{tail}",
            f"?after=%201&limit=1{tail}",
            f"?after=0&limit=0{tail}",
            f"?after=0&limit=101{tail}",
            f"?after=0&limit=-5{tail}",
            f"?after=0&limit=%D9%A1{tail}",
            f"?after&limit=1{tail}",
            f"?after=0&limit=1&expectedDigest={'A' * 64}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={'a' * 63}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={'g' * 64}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=-1",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=+1",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=1.0",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=%200",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, bad_payload, raw, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(bad_payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))
        # No rejected query created an execution.
        status, payload, _, _ = self.get(self.good_query())
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(self.server.store._repairs, {})

    def test_after_past_the_execution_count_is_400(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest = self.committed_digest()
        status, payload, raw, _ = self.get(
            f"?after=2&limit=1&expectedDigest={digest}&expectedCount=1"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        bad_paths = [
            VERIFY_PATH + "/",
            VERIFY_PATH + "/extra",
            "/v1/replication/repairs/executions/verif",
            "/v1/replication/repairs/execution/verify",
            "/v1/replication/repairs/verify",
            "/v1/replication/executions/verify",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=%ZZ&limit=x"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_plain_executions_route_keeps_its_own_contract(self) -> None:
        # The plain audit route is unchanged: it keeps its five-list
        # verification and its two-parameter query contract.
        status, payload, _, _ = self.raw_request(
            "GET", EXECUTIONS_PATH + "?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload["verification"]),
            [
                "status",
                "duplicateBindings",
                "outOfOrderActions",
                "boundaryViolations",
                "checkpointViolations",
                "recordViolations",
            ],
        )
        status, payload, _, _ = self.raw_request(
            "GET",
            EXECUTIONS_PATH + f"?after=0&limit=1&expectedDigest={'a' * 64}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_post_to_verify_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", VERIFY_PATH, {"cursor": 0}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only_over_http(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest = self.committed_digest()
        status, first, _, _ = self.get(self.good_query(digest, 1))
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        checkpoint_before = self.server.store.get_checkpoint("peer-a")[1]
        for query in (
            self.good_query(digest, 1),
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=1",
            f"?after=1&limit=1&expectedDigest={digest}&expectedCount=1",
            f"?after=0&limit=1&expectedDigest={OTHER_DIGEST}&expectedCount=9",
            "?after=0&limit=x",
        ):
            self.get(query)
        status, second, _, _ = self.get(self.good_query(digest, 1))
        self.assertEqual(second, first)
        _, metrics_after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(
            self.server.store.get_checkpoint("peer-a")[1], checkpoint_before
        )


class RepairExecutionsVerifyPersistenceHttpTests(RepairExecutionsVerifyHttpFixture):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-verify-http-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        data_path = os.path.join(self.tmpdir, "state.json")
        self.server.store = StateStore(data_file=data_path)
        self.data_path = data_path

    def test_query_persists_nothing(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest = self.committed_digest()
        with open(self.data_path, "rb") as handle:
            before = handle.read()
        for query in (
            self.good_query(digest, 1),
            f"?after=1&limit=1&expectedDigest={digest}&expectedCount=1",
            f"?after=0&limit=1&expectedDigest={digest}&expectedCount=1&x=1",
            f"?after=99&limit=1&expectedDigest={digest}&expectedCount=1",
        ):
            self.get(query)
        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), before)


class RepairExecutionsVerifyAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-verify-auth-")
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

    QUERY = f"?after=0&limit=1&expectedDigest={EMPTY_DIGEST}&expectedCount=0"
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
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["verification"]["status"], "ok")

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
            self.PATH,
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)

    def test_unknown_shape_still_authenticates_before_the_404(self) -> None:
        # Like every other route, an unknown path authenticates first on
        # an auth-enabled server; a missing credential is 401 rather than
        # a shape 404.
        status, payload, headers = self.get(
            self.single_port,
            VERIFY_PATH + "/extra" + self.QUERY,
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        # A valid credential then sees the shape mismatch.
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra" + self.QUERY,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # The shape decision still takes priority over the query check.
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra?after=%ZZ&limit=x",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
