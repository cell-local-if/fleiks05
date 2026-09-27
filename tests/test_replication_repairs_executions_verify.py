"""Tests for the read-only external verification of the replication-repair
execution audit::

    GET /v1/replication/repairs/executions/verify

The endpoint pages the committed history of conditional repair executions
exactly like ``GET /v1/replication/repairs/executions`` (same field order,
record structure, and ``after``/``limit`` paging) but requires two extra
external expectations — ``expectedDigest`` (exactly 64 lowercase
hexadecimal characters) and ``expectedCount`` (a non-negative ASCII
decimal integer) — and its ``verification`` independently scans the
**complete** history: the five internal anomaly lists keep their original
judgement, and two new lists report external disagreement:

- ``digestMismatches``: at most one ``{"expected", "observed"}`` marker
  (the caller's digest first, the independently recomputed full-history
  digest second);
- ``countMismatches``: at most one ``{"expected", "observed"}`` marker
  (the expected count first, the actual full history length second).

``status`` is ``"ok"`` exactly when all seven lists are empty. Paging
trims only the exported page; the digest, count, and conclusion always
cover the complete history. The tests cover the query parser, the
independent scan and external comparisons, the store's
paging/snapshot/recovery semantics, the HTTP request precedence chain
(404 path shape before the query check, 401 authentication with a Bearer
challenge, 403 in scope mode without one, 400 query validation including
an ``after`` past the execution count), the compact ordered
single-newline response body with its explicit Content-Length, restart
consistency, and the strict read-only guarantee. The plain audit route
is unchanged. Only the Python standard library is used.
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
    _repair_executions_external_verification_locked,
    _repair_receipts_digest,
    load_scope_policy,
    parse_replication_repair_executions_verify_query,
)

EXECUTIONS_PATH = "/v1/replication/repairs/executions"
VERIFY_PATH = EXECUTIONS_PATH + "/verify"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

DIGEST_64_A = "a" * 64
DIGEST_64_B = "b" * 64
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()

RESPONSE_FIELDS = [
    "executions",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "executionsCount",
    "verification",
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
    digest: str = DIGEST_64_A,
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


def verify_query(after: int, limit: int, digest: str, count: int) -> str:
    return (
        f"?after={after}&limit={limit}"
        f"&expectedDigest={digest}&expectedCount={count}"
    )


class ParseRepairExecutionsVerifyQueryTests(unittest.TestCase):
    def test_accepts_all_four_required_parameters(self) -> None:
        self.assertEqual(
            parse_replication_repair_executions_verify_query(
                f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
            ),
            (0, 1, DIGEST_64_A, 0),
        )
        # Parameter order is insignificant; leading zeros parse normally.
        self.assertEqual(
            parse_replication_repair_executions_verify_query(
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
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(query)
                )

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
                query = (
                    f"after=0&limit=1&expectedDigest={digest}&expectedCount=0"
                )
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(query)
                )

    def test_rejects_signed_decimal_non_ascii_and_out_of_range(self) -> None:
        base = f"limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0"
        bad = [
            # Malformed integer parameters (after, limit, expectedCount).
            f"after=-1&{base}",
            f"after=+0&{base}",
            f"after=1.0&{base}",
            f"after=%200&{base}",
            f"after=%C2%B2&{base}",
            "after=0&limit=0"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=101"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=-1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=1.0"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "after=0&limit=%D9%A1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=-1",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=+0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=1.0",
            f"after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0%20",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_verify_query(query)
                )


class RepairExecutionsExternalVerificationTests(unittest.TestCase):
    def test_empty_history_ok_for_empty_array_digest_and_zero_count(self) -> None:
        self.assertEqual(
            _repair_executions_external_verification_locked(
                [], {}, 0, EMPTY_DIGEST, 0
            ),
            OK_VERIFICATION,
        )

    def test_well_formed_history_matching_expectations_is_ok(self) -> None:
        entries = [gap_record(), gap_record("peer-b", "exec-b")]
        from semantic_state_engine.server import _repair_executions_digest_input

        digest = hashlib.sha256(_repair_executions_digest_input(entries)).hexdigest()
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4, "peer-b": 4}, 4, digest, 2
        )
        self.assertEqual(verdict, OK_VERIFICATION)
        self.assertEqual(list(verdict), VERIFICATION_FIELDS)

    def test_digest_mismatch_marks_expected_first_observed_second(self) -> None:
        verdict = _repair_executions_external_verification_locked(
            [], {}, 0, DIGEST_64_B, 0
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["digestMismatches"],
            [{"expected": DIGEST_64_B, "observed": EMPTY_DIGEST}],
        )
        self.assertEqual(verdict["countMismatches"], [])

    def test_count_mismatch_marks_expected_first_observed_second(self) -> None:
        entries = [gap_record(), gap_record("peer-b", "exec-b")]
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 4, "peer-b": 4}, 4, EMPTY_DIGEST, 5
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(
            verdict["countMismatches"], [{"expected": 5, "observed": 2}]
        )
        # The observed digest is reported alongside the digest mismatch.
        self.assertEqual(len(verdict["digestMismatches"]), 1)
        self.assertEqual(verdict["digestMismatches"][0]["expected"], EMPTY_DIGEST)

    def test_internal_anomaly_keeps_its_judgement_even_if_expectations_match(
        self,
    ) -> None:
        # Rolling the registered checkpoint behind the restored cursor
        # does not change the history digest: both external expectations
        # can agree while the internal checkpoint scan still breaks.
        entries = [gap_record()]
        from semantic_state_engine.server import _repair_executions_digest_input

        digest = hashlib.sha256(_repair_executions_digest_input(entries)).hexdigest()
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 0}, 4, digest, 1
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])
        self.assertEqual(len(verdict["checkpointViolations"]), 1)
        self.assertEqual(
            verdict["checkpointViolations"][0]["observed"], {"cursor": 4}
        )

    def test_internal_and_external_anomalies_combine(self) -> None:
        entries = [gap_record()]
        verdict = _repair_executions_external_verification_locked(
            entries, {"peer-a": 0}, 4, DIGEST_64_B, 9
        )
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(len(verdict["checkpointViolations"]), 1)
        self.assertEqual(verdict["digestMismatches"][0]["expected"], DIGEST_64_B)
        self.assertEqual(verdict["countMismatches"], [{"expected": 9, "observed": 1}])


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

    def gap_repair(self, peer: str, ack_id: str, *, checkpoint: int | None = None):
        """Commit one resend-gap repair for a peer with the standard seed."""
        item = self.advice(peer)[0]
        if checkpoint is None:
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

    def seed_three_executions(self) -> dict:
        self.seed_operations(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.gap_repair("peer-a", "exec-a1")
        self.gap_repair("peer-b", "exec-b1")
        self.gap_repair("peer-a", "exec-a2")
        audit = self.store.get_replication_repair_executions(0, 100)[1]
        return {"digest": audit["digest"], "count": audit["executionsCount"]}

    def verify(self, after: int, limit: int, digest: str, count: int):
        return self.store.get_replication_repair_executions_verify(
            after, limit, digest, count
        )


class RepairExecutionsVerifyStoreTests(RepairExecutionsVerifyStoreFixture):
    def test_empty_history_matching_expectations_is_ok(self) -> None:
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

    def test_matching_expectations_over_populated_history_is_ok(self) -> None:
        expected = self.seed_three_executions()
        status, payload = self.verify(0, 100, expected["digest"], expected["count"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertEqual(
            [(e["peerId"], e["ackId"]) for e in payload["executions"]],
            [("peer-a", "exec-a1"), ("peer-a", "exec-a2"), ("peer-b", "exec-b1")],
        )

    def test_paging_trims_only_the_page_never_the_conclusion(self) -> None:
        expected = self.seed_three_executions()
        pages = []
        for after in range(4):
            status, page = self.verify(after, 1, expected["digest"], expected["count"])
            self.assertIs(status, HTTPStatus.OK)
            pages.append(page)
        self.assertEqual(
            [[e["ackId"] for e in p["executions"]] for p in pages],
            [["exec-a1"], ["exec-a2"], ["exec-b1"], []],
        )
        self.assertEqual([p["nextCursor"] for p in pages], [1, 2, 3, 3])
        self.assertEqual([p["hasMore"] for p in pages], [True, True, False, False])
        for page in pages:
            self.assertEqual(page["digest"], expected["digest"])
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_digest_and_count_mismatches_are_page_independent(self) -> None:
        expected = self.seed_three_executions()
        for after in (0, 1, 3):
            status, page = self.verify(after, 1, DIGEST_64_B, 9)
            self.assertIs(status, HTTPStatus.OK)
            verdict = page["verification"]
            self.assertEqual(verdict["status"], "broken")
            self.assertEqual(
                verdict["digestMismatches"],
                [{"expected": DIGEST_64_B, "observed": expected["digest"]}],
            )
            self.assertEqual(
                verdict["countMismatches"], [{"expected": 9, "observed": 3}]
            )
            # The true summary is still reported alongside the mismatch.
            self.assertEqual(page["digest"], expected["digest"])
            self.assertEqual(page["executionsCount"], 3)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        expected = self.seed_three_executions()
        status, first = self.verify(3, 10, expected["digest"], expected["count"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(first["executions"], [])
        self.assertEqual(first["nextCursor"], 3)
        self.assertFalse(first["hasMore"])
        self.assertEqual(first["executionsCount"], 3)
        self.assertEqual(first["verification"]["status"], "ok")
        status, second = self.verify(3, 10, expected["digest"], expected["count"])
        self.assertEqual(second, first)
        # Zero executions: after=0 is already the stable empty tail.
        fresh = StateStore()
        status, empty = fresh.get_replication_repair_executions_verify(
            0, 1, EMPTY_DIGEST, 0
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(empty["executions"], [])
        self.assertEqual(empty["executionsCount"], 0)
        self.assertEqual(empty["verification"], OK_VERIFICATION)

    def test_after_past_count_raises_value_error(self) -> None:
        expected = self.seed_three_executions()
        with self.assertRaises(ValueError):
            self.verify(4, 1, expected["digest"], expected["count"])

    def test_replay_appends_nothing_and_keeps_the_conclusion(self) -> None:
        expected = self.seed_three_executions()
        item = self.advice("peer-a")[0]
        status, replay, error = self.store.apply_replication_repairs(
            "peer-a",
            "exec-a1",
            4,
            self.receipts_digest("peer-a"),
            [self.suggestion(item)],
        )
        self.assertIs(status, HTTPStatus.OK, error)
        status, payload = self.verify(0, 100, expected["digest"], expected["count"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["executionsCount"], 3)
        self.assertEqual(payload["digest"], expected["digest"])
        self.assertEqual(payload["verification"], OK_VERIFICATION)

    def test_query_is_strictly_read_only(self) -> None:
        expected = self.seed_three_executions()
        repairs_before = copy.deepcopy(dict(self.store._repairs))
        checkpoints_before = dict(self.store._checkpoints)
        acks_before = copy.deepcopy(dict(self.store._acks))
        metrics_before = self.store.get_metrics()
        for query in (
            (0, 100, expected["digest"], expected["count"]),
            (0, 1, DIGEST_64_B, 9),
            (3, 1, expected["digest"], expected["count"]),
        ):
            self.verify(*query)
        self.assertEqual(self.store._repairs, repairs_before)
        self.assertEqual(self.store._checkpoints, checkpoints_before)
        self.assertEqual(self.store._acks, acks_before)
        self.assertEqual(self.store.get_metrics(), metrics_before)

    def test_damaged_history_is_broken_even_when_expectations_agree(self) -> None:
        # The internal scan runs independently: a rolled-back registered
        # checkpoint leaves digest/count unchanged yet must break status.
        expected = self.seed_three_executions()
        self.store._checkpoints["peer-a"] = 0
        status, payload = self.verify(0, 100, expected["digest"], expected["count"])
        self.assertIs(status, HTTPStatus.OK)
        verdict = payload["verification"]
        self.assertEqual(verdict["status"], "broken")
        self.assertEqual(verdict["digestMismatches"], [])
        self.assertEqual(verdict["countMismatches"], [])
        self.assertTrue(verdict["checkpointViolations"])


class RepairExecutionsVerifyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-executions-verify-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        store = StateStore(data_file=self.data_file)
        helper = RepairExecutionsVerifyStoreFixture()
        helper.store = store
        expected = helper.seed_three_executions()
        _, before = store.get_replication_repair_executions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_replication_repair_executions_verify(
            0, 2, expected["digest"], expected["count"]
        )
        self.assertEqual(after, before)
        # A wrong expectation reproduces after restart as well.
        _, broken = recovered.get_replication_repair_executions_verify(
            0, 2, DIGEST_64_B, 9
        )
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
        status, payload = store.get_replication_repair_executions_verify(
            0, 100, EMPTY_DIGEST, 0
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["executionsCount"], 0)
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

    def get(self, query: str, path: str = VERIFY_PATH):
        return self.raw_request("GET", path + query)

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

    def current_expectations(self) -> tuple[str, int]:
        status, audit, _, _ = self.get("?after=0&limit=100", path=EXECUTIONS_PATH)
        self.assertEqual(status, 200)
        return audit["digest"], audit["executionsCount"]


class RepairExecutionsVerifyHttpTests(RepairExecutionsVerifyHttpFixture):
    def test_empty_history_is_compact_ordered_json_with_single_newline(self) -> None:
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
            b'{"executions":[],"nextCursor":0,"hasMore":false,"algorithm":"sha256",'
            b'"digest":"' + EMPTY_DIGEST.encode("ascii")
            + b'","executionsCount":0,"verification":{"status":"ok",'
            b'"duplicateBindings":[],"outOfOrderActions":[],'
            b'"boundaryViolations":[],"checkpointViolations":[],'
            b'"recordViolations":[],"digestMismatches":[],'
            b'"countMismatches":[]}}\n',
        )

    def test_matching_expectations_page_and_stay_ok(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.seed_gap("peer-b", first="ack-3", second="ack-4")
        self.apply("peer-a", "exec-a1")
        self.apply("peer-b", "exec-b1")
        self.apply("peer-a", "exec-a2")
        digest, count = self.current_expectations()
        status, payload, raw, headers = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), RESPONSE_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        self.assertEqual(payload["digest"], digest)
        self.assertEqual(payload["executionsCount"], 3)
        self.assertEqual(payload["verification"], OK_VERIFICATION)
        # Page with limit=1: summary and the conclusion stay identical.
        pages = []
        for after in range(4):
            status, page, _, _ = self.get(verify_query(after, 1, digest, count))
            self.assertEqual(status, 200)
            pages.append(page)
        self.assertEqual(
            [[e["ackId"] for e in p["executions"]] for p in pages],
            [["exec-a1"], ["exec-a2"], ["exec-b1"], []],
        )
        for page in pages:
            self.assertEqual(page["digest"], digest)
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"], OK_VERIFICATION)

    def test_wrong_digest_and_count_report_expected_then_observed(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
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
        # Only the digest wrong: count list stays empty.
        status, payload, _, _ = self.get(verify_query(0, 1, DIGEST_64_B, count))
        self.assertEqual(payload["verification"]["countMismatches"], [])

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest, count = self.current_expectations()
        status, payload, raw, _ = self.get(verify_query(1, 10, digest, count))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 1)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_bad_queries_are_400_and_change_nothing(self) -> None:
        self.seed(1)
        self.register("peer-a", 1)
        bad_queries = [
            "",
            "?",
            "?after=0&limit=1",
            "?after=0&limit=1&expectedCount=0",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0&after=2",
            f"?after=0&limit=1&expectedDigest={DIGEST_64_A}&expectedCount=0&x=1",
            "?x=1",
            "?after=&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "?after=0&limit="
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            f"?after=0&limit=1&expectedDigest=&expectedCount=0",
            "?after=0&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=",
            "?after=-1&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "?after=0&limit=0"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "?after=0&limit=101"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
            "?after=0&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=-1",
            "?after=0&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=1.0",
            "?after=0&limit=1"
            f"&expectedDigest={'A' * 64}&expectedCount=0",
            "?after=0&limit=1"
            f"&expectedDigest={'a' * 63}&expectedCount=0",
            "?after=0&limit=1&expectedDigest=" + "g" * 64 + "&expectedCount=0",
            "?after&limit=1"
            f"&expectedDigest={DIGEST_64_A}&expectedCount=0",
        ]
        for query in bad_queries:
            with self.subTest(query=query[:60]):
                status, bad_payload, raw, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(bad_payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))
        # No rejected query created an execution.
        digest, count = self.current_expectations()
        self.assertEqual(count, 0)
        self.assertEqual(self.server.store._repairs, {})

    def test_after_past_the_execution_count_is_400(self) -> None:
        self.seed(4)
        self.seed_gap("peer-a")
        self.apply("peer-a", "exec-1")
        digest, _ = self.current_expectations()
        status, payload, raw, _ = self.get(verify_query(2, 1, digest, 1))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404_even_with_a_bad_query(self) -> None:
        bad_paths = [
            VERIFY_PATH + "/",
            VERIFY_PATH + "/extra",
            "/v1/replication/repairs/executions/verif",
            "/v1/replication/repairs/verify",
            "/v1/replication/repair/executions/verify",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=%ZZ&expectedDigest=x"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_plain_audit_route_is_unchanged(self) -> None:
        # The sibling route keeps its own two-parameter contract: the two
        # extra expectations are unknown parameters there.
        status, payload, _, _ = self.raw_request(
            "GET", EXECUTIONS_PATH + verify_query(0, 1, DIGEST_64_A, 0)
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload, _, _ = self.get("?after=0&limit=100", path=EXECUTIONS_PATH)
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)
        # And the plain audit's verification still carries five lists.
        self.assertEqual(
            set(payload["verification"]),
            {
                "status",
                "duplicateBindings",
                "outOfOrderActions",
                "boundaryViolations",
                "checkpointViolations",
                "recordViolations",
            },
        )

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
        digest, count = self.current_expectations()
        status, first, _, _ = self.get(verify_query(0, 100, digest, count))
        self.assertEqual(status, 200)
        _, metrics_before, _, _ = self.raw_request("GET", "/v1/metrics")
        checkpoint_before = self.server.store.get_checkpoint("peer-a")[1]
        for query in (
            verify_query(0, 100, digest, count),
            verify_query(0, 1, DIGEST_64_B, 9),
            verify_query(1, 1, digest, count),
            "?after=0&limit=x",
        ):
            self.get(query)
        status, second, _, _ = self.get(verify_query(0, 100, digest, count))
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
        self.assertEqual(payload["executionsCount"], 0)
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
        self.assertEqual(payload["executionsCount"], 0)
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
        # The shape decision still takes priority over the query check.
        status, payload, _ = self.get(
            self.single_port,
            VERIFY_PATH + "/extra?after=%ZZ&expectedDigest=x",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
