"""Tests for the unified audit-root endpoint::

    GET /v1/integrity/root?expectedDigest=<64 lowercase hex>

From one committed snapshot the endpoint binds every persisted audit
stream under one SHA-256 root: the accepted-operation log (the global
audit chain), atomic-transaction bindings, conditional replication-repair
executions, per-peer consumption receipts, scope-policy change events,
and the registered sender checkpoint mapping. The success body is a
compact UTF-8 JSON object terminated by a single newline with exactly
five fields in order: ``algorithm``, ``digest``, ``evidenceCount`` (with
``streams``, ``records``, and ``mappings``), ``streams`` (the six fixed
streams, each carrying ``name``, ``digest``, ``count``, ``status``, and
the existing anomaly locations), and ``verification`` (``status`` plus
``rootMismatches`` reporting ``expected`` and ``observed``). The root
digest is the SHA-256 of the whitespace-free UTF-8 JSON array of
``{"name","digest","count"}`` stream summaries in the fixed stream
order. Receipts aggregate by sender in ascending ``peerId`` order (each
peer's receipts in first-commit order); checkpoints cover the sorted
peer-to-cursor mapping; the other streams keep first-commit order.

The tests cover the query contract (a single 64-lowercase-hex
``expectedDigest``; missing, repeated, unknown, blank, empty, uppercase,
non-hex, and wrong-length values are 400), shape-first 404 routing and
non-GET methods, store-level stream summaries and root recomputation,
multi-peer receipt aggregation with peer-qualified anomaly markers, the
checkpoints mapping digest, broken-stream propagation, the 401/403
authentication contract, restart consistency under ``--data-file``, the
strict read-only guarantee, and snapshot consistency under concurrent
commits. Only the Python standard library is used.
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
    _checkpoints_digest_input,
    _ordered_json_bytes,
    load_scope_policy,
    parse_integrity_root_query,
)

ROOT_PATH = "/v1/integrity/root"
HEX64_ZERO = "0" * 64

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
SCOPE_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
    ADMIN_TOKEN: ["read", "write", "admin"],
}

EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()
EMPTY_MAPPING_DIGEST = hashlib.sha256(b"{}").hexdigest()
EMPTY_LOG_HEAD = "0" * 64

ROOT_FIELDS = ["algorithm", "digest", "evidenceCount", "streams", "verification"]
EVIDENCE_FIELDS = ["streams", "records", "mappings"]
STREAM_FIELDS = ["name", "digest", "count", "status", "anomalies"]
STREAM_NAMES = [
    "acceptedOperations",
    "transactions",
    "repairExecutions",
    "receipts",
    "policyEvents",
    "checkpoints",
]
VERIFICATION_FIELDS = ["status", "rootMismatches"]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def root_summary(root: dict) -> list[dict]:
    return [
        {"name": stream["name"], "digest": stream["digest"], "count": stream["count"]}
        for stream in root["streams"]
    ]


class IntegrityRootQueryTests(unittest.TestCase):
    def test_single_lowercase_hex64_value_is_accepted(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(parse_integrity_root_query("expectedDigest=" + digest), digest)

    def test_missing_parameter_is_rejected(self) -> None:
        self.assertIsNone(parse_integrity_root_query(""))

    def test_blank_and_empty_values_are_rejected(self) -> None:
        self.assertIsNone(parse_integrity_root_query("expectedDigest="))
        self.assertIsNone(parse_integrity_root_query("expectedDigest"))
        self.assertIsNone(parse_integrity_root_query("=x"))

    def test_unknown_parameter_is_rejected(self) -> None:
        self.assertIsNone(parse_integrity_root_query("x=" + HEX64_ZERO))
        self.assertIsNone(
            parse_integrity_root_query(
                "expectedDigest=" + HEX64_ZERO + "&x=1"
            )
        )

    def test_repeated_parameter_is_rejected(self) -> None:
        self.assertIsNone(
            parse_integrity_root_query(
                "expectedDigest=" + HEX64_ZERO + "&expectedDigest=" + HEX64_ZERO
            )
        )

    def test_case_length_and_non_hex_values_are_rejected(self) -> None:
        self.assertIsNone(parse_integrity_root_query("expectedDigest=" + "A" * 64))
        self.assertIsNone(parse_integrity_root_query("expectedDigest=" + "g" * 64))
        self.assertIsNone(parse_integrity_root_query("expectedDigest=" + "0" * 63))
        self.assertIsNone(parse_integrity_root_query("expectedDigest=" + "0" * 65))
        self.assertIsNone(parse_integrity_root_query("expectedDigest=12345"))
        self.assertIsNone(
            parse_integrity_root_query("expectedDigest=" + "0" * 32 + "F" * 32)
        )


class IntegrityRootStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def root(self, expected: str = HEX64_ZERO):
        return self.store.get_integrity_root(expected)

    def seed_write(self, replica: str, operation_id: str, key: str, clock: dict) -> None:
        status = self.store.apply_operation(
            replica, operation(operation_id, key, "v", clock)
        )
        self.assertIs(status, HTTPStatus.CREATED)


class IntegrityRootEmptyTests(IntegrityRootStoreFixture):
    def test_empty_store_six_streams_in_fixed_order(self) -> None:
        status, payload = self.root()
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(list(payload["evidenceCount"]), EVIDENCE_FIELDS)
        self.assertEqual(
            payload["evidenceCount"], {"streams": 6, "records": 0, "mappings": 0}
        )
        self.assertEqual([s["name"] for s in payload["streams"]], STREAM_NAMES)
        for stream in payload["streams"]:
            self.assertEqual(list(stream), STREAM_FIELDS)
            self.assertEqual(stream["count"], 0)
            self.assertEqual(stream["status"], "ok")
        self.assertEqual(payload["streams"][0]["digest"], EMPTY_LOG_HEAD)
        self.assertEqual(payload["streams"][1]["digest"], EMPTY_ARRAY_DIGEST)
        self.assertEqual(payload["streams"][2]["digest"], EMPTY_ARRAY_DIGEST)
        self.assertEqual(payload["streams"][3]["digest"], EMPTY_ARRAY_DIGEST)
        self.assertEqual(payload["streams"][4]["digest"], EMPTY_ARRAY_DIGEST)
        self.assertEqual(payload["streams"][5]["digest"], EMPTY_MAPPING_DIGEST)

    def test_empty_store_root_is_hash_of_six_empty_summaries(self) -> None:
        _, payload = self.root()
        expected_input = _ordered_json_bytes(root_summary(payload))
        self.assertEqual(
            payload["digest"], hashlib.sha256(expected_input).hexdigest()
        )

    def test_empty_store_zero_expectation_is_a_mismatch_but_streams_intact(self) -> None:
        _, payload = self.root(HEX64_ZERO)
        self.assertEqual(payload["verification"]["status"], "broken")
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertEqual(
            payload["verification"]["rootMismatches"],
            [{"expected": HEX64_ZERO, "observed": payload["digest"]}],
        )

    def test_matching_digest_verifies_ok(self) -> None:
        _, first = self.root(HEX64_ZERO)
        status, payload = self.root(first["digest"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertEqual(payload["verification"]["rootMismatches"], [])

    def test_repeated_reads_are_identical(self) -> None:
        _, first = self.root()
        for _ in range(3):
            _, again = self.root()
            self.assertEqual(again, first)


class IntegrityRootPopulatedTests(IntegrityRootStoreFixture):
    def test_accepted_log_stream_uses_chain_head_and_count(self) -> None:
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.seed_write("r2", "o2", "k2", {"r2": 1})
        _, payload = self.root()
        log_stream = payload["streams"][0]
        self.assertEqual(log_stream["name"], "acceptedOperations")
        self.assertEqual(log_stream["count"], 2)
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(log_stream["digest"], head)
        self.assertNotEqual(head, EMPTY_LOG_HEAD)
        self.assertEqual(log_stream["status"], "ok")

    def test_checkpoint_stream_uses_sorted_mapping_digest(self) -> None:
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.store.save_checkpoint("peer-b", 0)
        self.store.save_checkpoint("peer-a", 1)
        _, payload = self.root()
        checkpoints = payload["streams"][5]
        self.assertEqual(checkpoints["count"], 2)
        self.assertEqual(
            checkpoints["digest"],
            hashlib.sha256(_checkpoints_digest_input({"peer-a": 1, "peer-b": 0})).hexdigest(),
        )
        self.assertEqual(
            payload["evidenceCount"],
            {"streams": 6, "records": 1, "mappings": 2},
        )

    def test_receipts_stream_aggregates_peers_sorted_with_creation_order(self) -> None:
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.seed_write("r2", "o2", "k2", {"r2": 1})
        self.seed_write("r3", "o3", "k3", {"r3": 1})
        self.store.save_checkpoint("peer-b", 0)
        self.store.save_checkpoint("peer-a", 0)
        status, _ = self.store.acknowledge_operations(
            "peer-b", "ack-b1", 1, [identity("r1", "o1")]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, _ = self.store.acknowledge_operations(
            "peer-a", "ack-a1", 1, [identity("r1", "o1")]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, _ = self.store.acknowledge_operations(
            "peer-b",
            "ack-b2",
            3,
            [identity("r2", "o2"), identity("r3", "o3")],
        )
        self.assertIs(status, HTTPStatus.CREATED)

        _, payload = self.root()
        receipts = payload["streams"][3]
        self.assertEqual(receipts["count"], 3)
        aggregate = [
            {
                "peerId": "peer-a",
                "ackId": "ack-a1",
                "cursor": 1,
                "operations": [identity("r1", "o1")],
            },
            {
                "peerId": "peer-b",
                "ackId": "ack-b1",
                "cursor": 1,
                "operations": [identity("r1", "o1")],
            },
            {
                "peerId": "peer-b",
                "ackId": "ack-b2",
                "cursor": 3,
                "operations": [identity("r2", "o2"), identity("r3", "o3")],
            },
        ]
        self.assertEqual(
            receipts["digest"],
            hashlib.sha256(_ordered_json_bytes(aggregate)).hexdigest(),
        )
        self.assertEqual(receipts["status"], "ok")
        self.assertEqual(
            payload["evidenceCount"],
            {"streams": 6, "records": 3 + 3, "mappings": 2},
        )

    def test_receipt_anomaly_markers_are_peer_qualified(self) -> None:
        # One accepted record, and a receipt that names the wrong identity
        # for the position it confirms: the receipts stream reports the
        # chain audit's identity mismatch prefixed with the peerId.
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.store.save_checkpoint("peer-a", 0)
        status, _ = self.store.acknowledge_operations(
            "peer-a", "ack-1", 1, [identity("r9", "o9")]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        # The conflicting acknowledgement never commits, so force the
        # damaged receipt into the committed set to exercise aggregation
        # over an anomalous chain (as a corrupted history would present).
        self.store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r9", "o9")],
        }
        _, payload = self.root()
        receipts = payload["streams"][3]
        self.assertEqual(receipts["status"], "broken")
        mismatch = receipts["anomalies"]["identityMismatches"]
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch[0]["peerId"], "peer-a")
        self.assertEqual(mismatch[0]["ackId"], "ack-1")
        self.assertEqual(mismatch[0]["position"], 0)

    def test_damaged_transaction_stream_makes_verification_broken(self) -> None:
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.store._transactions["tx-bad"] = "damaged"
        _, current = self.root()
        # Pass the current recomputed root (no root mismatch): the broken
        # transaction stream alone must still fail verification.
        _, payload = self.root(current["digest"])
        transactions = payload["streams"][1]
        self.assertEqual(transactions["status"], "broken")
        self.assertEqual(
            transactions["anomalies"]["recordViolations"][0]["transactionId"],
            "tx-bad",
        )
        self.assertEqual(payload["verification"]["rootMismatches"], [])
        self.assertEqual(payload["verification"]["status"], "broken")
        # A stale expectation additionally reports expected/observed.
        _, stale = self.root("0" * 64)
        self.assertEqual(stale["verification"]["status"], "broken")
        self.assertEqual(
            stale["verification"]["rootMismatches"],
            [{"expected": "0" * 64, "observed": current["digest"]}],
        )

    def test_root_digest_binds_all_six_summaries(self) -> None:
        self.seed_write("r1", "o1", "k1", {"r1": 1})
        self.store.save_checkpoint("peer-a", 1)
        _, payload = self.root()
        recomputed = hashlib.sha256(
            _ordered_json_bytes(root_summary(payload))
        ).hexdigest()
        self.assertEqual(recomputed, payload["digest"])
        status, matched = self.root(recomputed)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(matched["verification"]["status"], "ok")


class IntegrityRootSnapshotTests(IntegrityRootStoreFixture):
    def test_concurrent_readers_observe_only_consistent_roots(self) -> None:
        for i in range(10):
            replica = f"r{i}"
            self.seed_write(replica, f"o{i}", f"k{i}", {replica: 1})
        seen: set[str] = set()
        stop = False

        def reader() -> None:
            while not stop:
                _, report = self.store.get_integrity_root(HEX64_ZERO)
                recomputed = hashlib.sha256(
                    _ordered_json_bytes(root_summary(report))
                ).hexdigest()
                self.assertEqual(recomputed, report["digest"])
                seen.add(report["digest"])

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for i in range(10, 40):
            replica = f"r{i}"
            self.seed_write(replica, f"o{i}", f"k{i}", {replica: 1})
        stop = True
        for thread in threads:
            thread.join()
        self.assertGreater(len(seen), 1)


class IntegrityRootRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-root-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def test_restart_reports_identical_root(self) -> None:
        store = StateStore(data_file=self.data_file)
        status = store.apply_operation(
            "r1", operation("o1", "k1", "v1", {"r1": 1})
        )
        self.assertIs(status, HTTPStatus.CREATED)
        store.save_checkpoint("peer-a", 0)
        status, _ = store.acknowledge_operations(
            "peer-a", "ack-1", 1, [identity("r1", "o1")]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        _, before = store.get_integrity_root(HEX64_ZERO)
        recovered = StateStore(data_file=self.data_file)
        _, after = recovered.get_integrity_root(HEX64_ZERO)
        self.assertEqual(after, before)
        # And with the reported digest, the recovered state verifies ok.
        _, verified = recovered.get_integrity_root(after["digest"])
        self.assertEqual(verified["verification"]["status"], "ok")

    def test_old_file_without_new_sections_verifies_intact(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "operations": []}, handle)
        store = StateStore(data_file=self.data_file)
        _, payload = store.get_integrity_root(HEX64_ZERO)
        self.assertEqual(
            [stream["name"] for stream in payload["streams"]], STREAM_NAMES
        )
        for stream in payload["streams"]:
            self.assertEqual(stream["count"], 0)
            self.assertEqual(stream["status"], "ok")

    def test_query_is_strictly_read_only(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        size = os.path.getsize(self.data_file)
        mtime = os.path.getmtime(self.data_file)
        for _ in range(5):
            store.get_integrity_root(HEX64_ZERO)
        self.assertEqual(os.path.getsize(self.data_file), size)
        self.assertEqual(os.path.getmtime(self.data_file), mtime)


class IntegrityRootHttpFixture(unittest.TestCase):
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

    def raw_request(self, method, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=dict(headers or {}))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def get(self, query: str = "expectedDigest=" + HEX64_ZERO, path: str = ROOT_PATH):
        return self.raw_request("GET", path + "?" + query)


class IntegrityRootHttpTests(IntegrityRootHttpFixture):
    def test_empty_store_success_body_is_compact_ordered_single_newline(self) -> None:
        status, payload, raw, headers = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(list(payload["evidenceCount"]), EVIDENCE_FIELDS)
        self.assertEqual([s["name"] for s in payload["streams"]], STREAM_NAMES)
        for stream in payload["streams"]:
            self.assertEqual(list(stream), STREAM_FIELDS)
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        # All counts are JSON integers (compact form, no decimal points).
        self.assertNotIn(b".", raw)

    def test_matching_digest_verifies_ok(self) -> None:
        _, first, _, _ = self.get()
        status, payload, _, _ = self.get("expectedDigest=" + first["digest"])
        self.assertEqual(status, 200)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertEqual(payload["verification"]["rootMismatches"], [])

    def test_missing_digest_is_400(self) -> None:
        status, payload, raw, _ = self.raw_request("GET", ROOT_PATH)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_malformed_digest_values_are_400(self) -> None:
        bad = [
            "expectedDigest=",
            "expectedDigest",
            "expectedDigest=" + "a" * 63,
            "expectedDigest=" + "a" * 65,
            "expectedDigest=" + "A" * 64,
            "expectedDigest=" + "g" * 64,
            "expectedDigest=12345",
            "x=" + HEX64_ZERO,
            "expectedDigest=" + HEX64_ZERO + "&x=1",
            "expectedDigest=" + HEX64_ZERO
            + "&expectedDigest=" + HEX64_ZERO,
        ]
        for query in bad:
            with self.subTest(query=query):
                status, payload, _, _ = self.get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_and_method_mismatches_are_404(self) -> None:
        cases = [
            ("GET", ROOT_PATH + "/"),
            ("GET", ROOT_PATH + "/extra"),
            ("GET", "/v1/integrity"),
            ("GET", "/v1/integrity/rot"),
            ("GET", "/v1/integrities/root"),
            ("POST", ROOT_PATH),
            ("PUT", ROOT_PATH),
            ("DELETE", ROOT_PATH),
            ("PATCH", ROOT_PATH),
            ("OPTIONS", ROOT_PATH),
        ]
        for method, path in cases:
            with self.subTest(method=method, path=path):
                status, payload, _, _ = self.raw_request(
                    method, path + "?expectedDigest=" + HEX64_ZERO
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_404_precedes_the_query_check(self) -> None:
        # A wrong path shape together with an invalid query is still 404.
        status, payload, _, _ = self.raw_request("GET", ROOT_PATH + "/extra?bogus")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_repeated_gets_are_identical(self) -> None:
        _, _, first, _ = self.get()
        _, _, second, _ = self.get()
        self.assertEqual(first, second)


class IntegrityRootAuthHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-root-auth-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.policy_path = os.path.join(self.tmpdir, "scopes.json")
        with open(self.policy_path, "w", encoding="utf-8") as handle:
            json.dump(SCOPE_POLICY, handle)

    def serve(self, **kwargs):
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, **kwargs
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        server.store = StateStore()
        return server

    def request(self, server, headers=None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.request(
            "GET", ROOT_PATH + "?expectedDigest=" + HEX64_ZERO, headers=headers or {}
        )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def test_single_token_mode_401_carries_bearer_challenge(self) -> None:
        server = self.serve(auth_token="legacy-token")
        status, raw, headers = self.request(server)
        self.assertEqual(status, 401)
        self.assertEqual(raw, b'{"error":"unauthorized"}')
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, _, headers = self.request(
            server, {"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, _, _ = self.request(server, {"Authorization": "Bearer legacy-token"})
        self.assertEqual(status, 200)

    def test_scope_mode_403_has_no_challenge_and_read_suffices(self) -> None:
        server = self.serve(
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        status, raw, headers = self.request(
            server, {"Authorization": "Bearer " + WRITE_TOKEN}
        )
        self.assertEqual(status, 403)
        self.assertEqual(raw, b'{"error":"forbidden"}')
        self.assertNotIn("WWW-Authenticate", headers)
        status, _, _ = self.request(
            server, {"Authorization": "Bearer " + READ_TOKEN}
        )
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            server, {"Authorization": "Bearer " + ADMIN_TOKEN}
        )
        self.assertEqual(status, 200)

    def test_health_stays_anonymous_in_scope_mode(self) -> None:
        server = self.serve(
            auth_scopes=dict(load_scope_policy(self.policy_path)),
            scope_policy_file=self.policy_path,
        )
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        conn.close()


if __name__ == "__main__":
    unittest.main()
