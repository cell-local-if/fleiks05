"""Tests for the unified audit root over all persisted audit streams::

    GET /v1/integrity/root?expectedDigest=<64 lowercase hex>

The endpoint reads every persisted audit stream in one committed snapshot
— the shared accepted-operation log (the global audit chain), the
atomic-transaction bindings, the conditional replication-repair
executions, the per-peer consumption receipts aggregated by sender, the
scope-policy change events, and the registered sender checkpoint mapping
— and binds them under one SHA-256 root. A successful response is a
compact UTF-8 JSON object terminated by a single newline with exactly
five fields in order: ``algorithm`` (always ``"sha256"``), ``digest``
(the root), ``evidenceCount`` (``streams``/``records``/``mappings``
counts), ``streams`` (the six fixed-order stream reports, each with a
stable ``name``, the stream's existing full-history ``digest``, its
record or mapping ``count``, its existing integrity ``status`` and
anomaly locations), and ``verification`` (``"ok"`` exactly when all six
streams are intact and the recomputed root equals ``expectedDigest``,
otherwise ``"broken"`` with a ``rootMismatches`` expected/observed
marker).

The tests cover the query parser, the per-stream digest/count/status
reuse and ordering (receipts sorted by sender, checkpoints sorted by
peer, the other histories in first-commit order), the root-digest
encoding, independent broken-stream and wrong-digest conclusions, the
HTTP precedence chain (404 path shape before the 400 query check, 401
authentication with a Bearer challenge, 403 in scope mode without
read/admin), the compact ordered single-newline body with explicit
Content-Length, restart consistency under ``--data-file``, and the
strict read-only guarantee. Only the Python standard library is used.
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

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _checkpoints_digest_input,
    _ordered_json_bytes,
    _repair_executions_digest_input,
    load_scope_policy,
    parse_integrity_root_query,
)

ROOT_PATH = "/v1/integrity/root"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()
EMPTY_OBJECT_DIGEST = hashlib.sha256(b"{}").hexdigest()
GENESIS = "0" * 64

ROOT_FIELDS = ["algorithm", "digest", "evidenceCount", "streams", "verification"]
EVIDENCE_FIELDS = ["streams", "records", "mappings"]
STREAM_ORDER = [
    "acceptedOperations",
    "transactions",
    "repairExecutions",
    "receipts",
    "policyEvents",
    "checkpoints",
]
STREAM_FIELDS = ["name", "digest", "count", "status", "anomalies"]
VERIFICATION_FIELDS = ["status", "rootMismatches"]


def empty_root_digest() -> str:
    """Recompute the empty-store root digest from the fixed stream order."""
    elements = [
        {"name": "acceptedOperations", "digest": GENESIS, "count": 0},
        {"name": "transactions", "digest": EMPTY_ARRAY_DIGEST, "count": 0},
        {"name": "repairExecutions", "digest": EMPTY_ARRAY_DIGEST, "count": 0},
        {"name": "receipts", "digest": EMPTY_ARRAY_DIGEST, "count": 0},
        {"name": "policyEvents", "digest": EMPTY_ARRAY_DIGEST, "count": 0},
        {"name": "checkpoints", "digest": EMPTY_OBJECT_DIGEST, "count": 0},
    ]
    return hashlib.sha256(_ordered_json_bytes(elements)).hexdigest()


def repair_record(
    peer_id: str = "peer-a",
    ack_id: str = "exec-1",
    *,
    expected_checkpoint: int = 2,
    cursor: int = 2,
) -> dict:
    """One well-formed stored execution carrying a single resend action."""
    return {
        "peerId": peer_id,
        "ackId": ack_id,
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": EMPTY_ARRAY_DIGEST,
        "suggestions": [
            {
                "action": "resend",
                "ackId": "ack-1",
                "location": {"start": 1, "end": 2},
                "target": {"start": 1, "end": 2},
            }
        ],
        "results": [
            {"action": "resend", "boundary": {"start": 1, "end": 2}}
        ],
        "cursor": cursor,
    }


class ParseIntegrityRootQueryTests(unittest.TestCase):
    def test_accepts_one_lowercase_hex64_value(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            parse_integrity_root_query(f"expectedDigest={digest}"), digest
        )

    def test_rejects_missing_unknown_repeated_and_blank(self) -> None:
        bad = [
            "",
            "?",
            "expectedDigest",
            "expectedDigest=",
            f"expectedDigest={'a' * 63}",
            f"expectedDigest={'a' * 65}",
            f"expectedDigest={'A' * 64}",
            f"expectedDigest={'g' * 64}",
            f"expectedDigest={'0' * 64} ",
            f"expectedDigest={'a' * 64}&expectedDigest={'b' * 64}",
            f"expectedDigest={'a' * 64}&x=1",
            f"x={'a' * 64}",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_integrity_root_query(query))


class IntegrityRootStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def root(self, expected: str):
        return self.store.get_integrity_root(expected)

    def streams(self, report) -> dict:
        return {stream["name"]: stream for stream in report["streams"]}

    def test_empty_store_six_fixed_streams_and_counts(self) -> None:
        status, report = self.root(GENESIS)
        self.assertEqual(status, 200)
        self.assertEqual(list(report), ROOT_FIELDS)
        self.assertEqual(report["algorithm"], "sha256")
        self.assertEqual(list(report["evidenceCount"]), EVIDENCE_FIELDS)
        self.assertEqual(
            report["evidenceCount"],
            {"streams": 6, "records": 0, "mappings": 0},
        )
        self.assertEqual(
            [stream["name"] for stream in report["streams"]], STREAM_ORDER
        )
        for stream in report["streams"]:
            self.assertEqual(list(stream), STREAM_FIELDS)
            self.assertEqual(stream["count"], 0)
            self.assertEqual(stream["status"], "ok")
        by = self.streams(report)
        self.assertEqual(by["acceptedOperations"]["digest"], GENESIS)
        for name in (
            "transactions",
            "repairExecutions",
            "receipts",
            "policyEvents",
        ):
            self.assertEqual(by[name]["digest"], EMPTY_ARRAY_DIGEST, name)
        self.assertEqual(by["checkpoints"]["digest"], EMPTY_OBJECT_DIGEST)

    def test_empty_store_root_digest_and_verification(self) -> None:
        expected = empty_root_digest()
        _, matching = self.root(expected)
        self.assertEqual(matching["digest"], expected)
        self.assertEqual(list(matching["verification"]), VERIFICATION_FIELDS)
        self.assertEqual(matching["verification"]["status"], "ok")
        self.assertEqual(matching["verification"]["rootMismatches"], [])
        _, wrong = self.root("f" * 64)
        self.assertEqual(wrong["verification"]["status"], "broken")
        self.assertEqual(
            wrong["verification"]["rootMismatches"],
            [{"expected": "f" * 64, "observed": expected}],
        )

    def test_root_input_is_whitespace_free_name_digest_count_array(self) -> None:
        _, report = self.root(GENESIS)
        observed_root = report["digest"]
        elements = [
            {
                "name": stream["name"],
                "digest": stream["digest"],
                "count": stream["count"],
            }
            for stream in report["streams"]
        ]
        raw = _ordered_json_bytes(elements)
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b"\t", raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), observed_root)

    def _seed_populated(self) -> None:
        self.store.apply_operation(
            "r1",
            {"operationId": "o1", "key": "k", "value": "a", "clock": {"r1": 1}},
        )
        self.store.apply_operation(
            "r2",
            {"operationId": "o2", "key": "k", "value": "b", "clock": {"r2": 1}},
        )
        self.store.apply_transaction(
            "tx-1",
            [
                {
                    "key": "t",
                    "replicaId": "r1",
                    "operationId": "o3",
                    "value": "x",
                    "clock": {"r1": 2},
                    "candidates": [],
                }
            ],
        )
        self.store.save_checkpoint("peer-a", 0)
        self.store.save_checkpoint("peer-b", 0)
        self.store.acknowledge_operations(
            "peer-a",
            "ack-a1",
            2,
            [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        )
        self.store.acknowledge_operations(
            "peer-b",
            "ack-b1",
            1,
            [{"replicaId": "r1", "operationId": "o1"}],
        )
        self.store.record_policy_reload("d" * 64, 1)

    def test_populated_counts_records_and_mappings(self) -> None:
        self._seed_populated()
        _, report = self.root(GENESIS)
        by = self.streams(report)
        self.assertEqual(by["acceptedOperations"]["count"], 3)
        self.assertEqual(by["transactions"]["count"], 1)
        self.assertEqual(by["repairExecutions"]["count"], 0)
        self.assertEqual(by["receipts"]["count"], 2)
        self.assertEqual(by["policyEvents"]["count"], 1)
        self.assertEqual(by["checkpoints"]["count"], 2)
        # records covers the five record streams but not the checkpoint map.
        self.assertEqual(
            report["evidenceCount"],
            {"streams": 6, "records": 3 + 1 + 0 + 2 + 1, "mappings": 2},
        )
        for stream in report["streams"]:
            self.assertEqual(stream["status"], "ok", stream["name"])

    def test_receipts_aggregate_senders_sorted_and_digest_is_canonical(self) -> None:
        self._seed_populated()
        _, report = self.root(GENESIS)
        by = self.streams(report)
        peer_a = [
            (ack_id, receipt)
            for (peer_id, ack_id), receipt in self.store._acks.items()
            if peer_id == "peer-a"
        ]
        peer_b = [
            (ack_id, receipt)
            for (peer_id, ack_id), receipt in self.store._acks.items()
            if peer_id == "peer-b"
        ]
        from semantic_state_engine.server import _receipts_digest_input

        inner_a = _receipts_digest_input("peer-a", peer_a)[1:-1]
        inner_b = _receipts_digest_input("peer-b", peer_b)[1:-1]
        aggregate = b"[" + inner_a + b"," + inner_b + b"]"
        self.assertEqual(
            by["receipts"]["digest"],
            hashlib.sha256(aggregate).hexdigest(),
        )

    def test_checkpoint_digest_covers_peer_to_cursor_map_sorted(self) -> None:
        self._seed_populated()
        _, report = self.root(GENESIS)
        by = self.streams(report)
        self.assertEqual(
            by["checkpoints"]["digest"],
            hashlib.sha256(
                _checkpoints_digest_input({"peer-a": 2, "peer-b": 1})
            ).hexdigest(),
        )

    def test_repair_stream_reuses_existing_execution_digest(self) -> None:
        self.store.apply_operation(
            "r1",
            {"operationId": "o1", "key": "k", "value": "a", "clock": {"r1": 1}},
        )
        self.store.apply_operation(
            "r2",
            {"operationId": "o2", "key": "k", "value": "b", "clock": {"r2": 1}},
        )
        self.store.save_checkpoint("peer-a", 2)
        record = repair_record()
        self.store._repairs[("peer-a", "exec-1")] = {
            "expectedCheckpoint": record["expectedCheckpoint"],
            "expectedReceipts": record["expectedReceipts"],
            "suggestions": record["suggestions"],
            "results": record["results"],
            "cursor": record["cursor"],
        }
        _, report = self.root(GENESIS)
        by = self.streams(report)
        self.assertEqual(by["repairExecutions"]["count"], 1)
        self.assertEqual(by["repairExecutions"]["status"], "ok")
        self.assertEqual(
            by["repairExecutions"]["digest"],
            hashlib.sha256(
                _repair_executions_digest_input([record])
            ).hexdigest(),
        )

    def test_a_damaged_stream_makes_the_root_broken_even_with_matching_digest(self) -> None:
        # A policy event with an illegal digest shape breaks the
        # policyEvents stream independently of the root comparison.
        self.store._policy_events = [
            {"sequence": 1, "digest": "not-a-digest", "tokens": 1}
        ]
        _, observed = self.root(GENESIS)
        by = self.streams(observed)
        self.assertEqual(by["policyEvents"]["status"], "broken")
        self.assertTrue(
            by["policyEvents"]["anomalies"]["digestMismatches"]
        )
        # Passing the recomputed root digest clears the root mismatch but
        # the broken stream still makes the overall verification broken.
        _, matching = self.root(observed["digest"])
        self.assertEqual(matching["verification"]["rootMismatches"], [])
        self.assertEqual(matching["verification"]["status"], "broken")

    def test_repeated_queries_are_stable(self) -> None:
        self._seed_populated()
        _, first = self.root(GENESIS)
        _, second = self.root(GENESIS)
        self.assertEqual(first, second)


class IntegrityRootRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-root-")
        self.data_file = os.path.join(self.tmpdir, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restart_gives_identical_report(self) -> None:
        store = StateStore(self.data_file)
        store.apply_operation(
            "r1",
            {"operationId": "o1", "key": "k", "value": "a", "clock": {"r1": 1}},
        )
        store.save_checkpoint("peer-a", 0)
        store.acknowledge_operations(
            "peer-a",
            "ack-1",
            1,
            [{"replicaId": "r1", "operationId": "o1"}],
        )
        _, before = store.get_integrity_root(GENESIS)
        restarted = StateStore(self.data_file)
        _, after = restarted.get_integrity_root(GENESIS)
        self.assertEqual(before, after)

    def test_query_is_strictly_read_only_on_disk(self) -> None:
        store = StateStore(self.data_file)
        store.apply_operation(
            "r1",
            {"operationId": "o1", "key": "k", "value": "a", "clock": {"r1": 1}},
        )
        with open(self.data_file, "rb") as handle:
            persisted = handle.read()
        store.get_integrity_root(GENESIS)
        with open(self.data_file, "rb") as handle:
            self.assertEqual(handle.read(), persisted)


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
        conn.request(method, path, headers=dict(headers or []))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def get(self, path: str):
        return self.raw_request("GET", path)


class IntegrityRootHttpTests(IntegrityRootHttpFixture):
    def test_empty_store_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.get(
            f"{ROOT_PATH}?expectedDigest={GENESIS}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(list(payload["evidenceCount"]), EVIDENCE_FIELDS)
        self.assertEqual(
            [stream["name"] for stream in payload["streams"]], STREAM_ORDER
        )
        self.assertEqual(list(payload["verification"]), VERIFICATION_FIELDS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_matching_digest_verifies_ok(self) -> None:
        _, first, _, _ = self.get(f"{ROOT_PATH}?expectedDigest={GENESIS}")
        status, payload, _, _ = self.get(
            f"{ROOT_PATH}?expectedDigest={first['digest']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertEqual(payload["verification"]["rootMismatches"], [])

    def test_bad_query_is_400_invalid_request(self) -> None:
        bad_queries = [
            ROOT_PATH,
            ROOT_PATH + "?",
            ROOT_PATH + "?expectedDigest",
            ROOT_PATH + "?expectedDigest=",
            ROOT_PATH + f"?expectedDigest={'a' * 63}",
            ROOT_PATH + f"?expectedDigest={'a' * 65}",
            ROOT_PATH + f"?expectedDigest={'A' * 64}",
            ROOT_PATH + f"?expectedDigest={'g' * 64}",
            ROOT_PATH + f"?expectedDigest={'a' * 64}&x=1",
            ROOT_PATH
            + f"?expectedDigest={'a' * 64}&expectedDigest={'b' * 64}",
            ROOT_PATH + f"?x={'a' * 64}",
        ]
        for path in bad_queries:
            with self.subTest(path=path):
                status, payload, raw, _ = self.get(path)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_and_method_mismatches_are_404(self) -> None:
        bad_paths = [
            ("GET", ROOT_PATH + "/"),
            ("GET", ROOT_PATH + "/extra"),
            ("GET", "/v1/integrity"),
            ("GET", "/v1/integrity/roo"),
            ("GET", "/v1/integrities/root"),
            ("POST", ROOT_PATH),
            ("PUT", ROOT_PATH),
            ("DELETE", ROOT_PATH),
            ("PATCH", ROOT_PATH),
            ("OPTIONS", ROOT_PATH),
        ]
        for method, path in bad_paths:
            with self.subTest(method=method, path=path):
                status, payload, _, _ = self.raw_request(
                    method, path + f"?expectedDigest={'a' * 64}"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_path_shape_404_precedes_query_check(self) -> None:
        status, payload, _, _ = self.get(ROOT_PATH + "/extra?expectedDigest=bogus")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_health_stays_anonymous_and_unaffected(self) -> None:
        status, payload, _, _ = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class IntegrityRootAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-integrity-root-auth-")
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

    PATH = f"{ROOT_PATH}?expectedDigest={GENESIS}"

    def test_single_token_requires_bearer_with_challenge(self) -> None:
        status, payload, headers = self.get(self.single_port, self.PATH)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, _, _ = self.get(
            self.single_port, self.PATH, {"Authorization": "Bearer wrong"}
        )
        self.assertEqual(status, 401)
        status, _, _ = self.get(
            self.single_port, self.PATH, {"Authorization": "Bearer s3cret-token"}
        )
        self.assertEqual(status, 200)

    def test_scope_mode_401_challenge_and_403_without_challenge(self) -> None:
        status, _, headers = self.get(self.scope_port, self.PATH)
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, payload, headers = self.get(
            self.scope_port, self.PATH, {"Authorization": f"Bearer {WRITE_TOKEN}"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        self.assertNotIn("Www-Authenticate", headers)
        status, _, _ = self.get(
            self.scope_port, self.PATH, {"Authorization": f"Bearer {READ_TOKEN}"}
        )
        self.assertEqual(status, 200)

    def test_scope_decision_precedes_query_and_route_checks(self) -> None:
        status, _, _ = self.get(
            self.scope_port,
            f"{ROOT_PATH}?bogus",
            {"Authorization": f"Bearer {WRITE_TOKEN}"},
        )
        self.assertEqual(status, 403)
        status, _, headers = self.get(
            self.scope_port,
            ROOT_PATH + f"/extra?expectedDigest={'a' * 64}",
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.get(port, "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
