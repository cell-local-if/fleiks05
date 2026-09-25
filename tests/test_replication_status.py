"""Tests for the sender-side replication delivery-status endpoint::

    GET /v1/replication/status?peerId=P

The endpoint is a strictly read-only status query over one registered
sender-side replication peer: it reports the decoded peer id (``peer``),
the registered checkpoint cursor (``pos``), the number of accepted
records the peer has not yet consumed (``left``), the peer's committed
receipt count (``acks``), and the receipt chain-audit conclusion
(``chain`` — status, coverage interval, and the gap, overlap,
identity-mismatch, and cursor-regression lists), all from one committed
snapshot.

The tests cover the query parser (required, non-repeated, non-empty,
percent-decoded ``peerId``; illegal encodings rejected), the store's
snapshot and recovery semantics, the HTTP request precedence chain
(401 authentication with a Bearer challenge, 403 in scope mode without
one, 404 path shape and unregistered peer, 400 query validation), the
compact ordered response body with its single trailing newline, and the
read-only guarantee.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot and recovery
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import http.client
from http import HTTPStatus
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_replication_status_query,
)

STATUS_FIELDS = ["peer", "pos", "left", "acks", "chain"]
CHAIN_FIELDS = [
    "status",
    "coverage",
    "gaps",
    "overlaps",
    "identityMismatches",
    "cursorRegressions",
]

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"

SCOPE_POLICY = {
    READ_TOKEN: ["read"],
    WRITE_TOKEN: ["write"],
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def empty_chain() -> dict:
    return {
        "status": "ok",
        "coverage": {"start": 0, "end": 0},
        "gaps": [],
        "overlaps": [],
        "identityMismatches": [],
        "cursorRegressions": [],
    }


class ParseReplicationStatusQueryTests(unittest.TestCase):
    def test_accepts_a_single_non_empty_peer_id(self) -> None:
        self.assertEqual(parse_replication_status_query("peerId=peer-a"), "peer-a")
        # Percent decoding follows the replication routes' path rules.
        self.assertEqual(
            parse_replication_status_query("peerId=peer%20one"), "peer one"
        )
        self.assertEqual(parse_replication_status_query("peerId=%2F"), "/")
        self.assertEqual(parse_replication_status_query("peerId=%C3%A9"), "é")
        # A plus is a space in the query encoding.
        self.assertEqual(parse_replication_status_query("peerId=a+b"), "a b")

    def test_rejects_missing_repeated_empty_and_unknown(self) -> None:
        bad = [
            "",
            "after=0",
            "peerId=",
            "peerId",
            "peerId=a&peerId=b",
            "peerId=a&peerId=a",
            "peerId=a&x=1",
            "x=1&peerId=a",
            "peerId=a&=",
            "=1&peerId=a",
            "?peerId=a",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))

    def test_rejects_illegal_encodings(self) -> None:
        bad = [
            "peerId=%",  # lone percent
            "peerId=%2",  # truncated escape
            "peerId=%ZZ",  # non-hex escape
            "peerId=a%2G",  # non-hex second digit
            "peerId=%FF",  # not valid UTF-8
            "peerId=%C3",  # truncated UTF-8 sequence
            "peerId=a%",  # trailing percent
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))


class ReplicationStatusStoreTests(unittest.TestCase):
    """Snapshot, progress, and chain-conclusion semantics against StateStore."""

    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int) -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            self.assertIn(status, (200, 201))

    def ack(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, error = self.store.acknowledge_operations(
            peer, ack_id, cursor, operations
        )
        self.assertEqual(status, 201, error)

    def test_unregistered_peer_is_404(self) -> None:
        status, payload = self.store.get_replication_status("peer-a")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts_reports_empty_chain(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 1)
        status, payload = self.store.get_replication_status("peer-a")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), STATUS_FIELDS)
        self.assertEqual(payload["peer"], "peer-a")
        self.assertEqual(payload["pos"], 1)
        self.assertEqual(payload["left"], 2)
        self.assertEqual(payload["acks"], 0)
        self.assertEqual(list(payload["chain"]), CHAIN_FIELDS)
        self.assertEqual(payload["chain"], empty_chain())

    def test_progress_and_chain_follow_commits(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        status, payload = self.store.get_replication_status("peer-a")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["pos"], 2)
        self.assertEqual(payload["left"], 1)
        self.assertEqual(payload["acks"], 1)
        self.assertEqual(payload["chain"]["status"], "ok")
        self.assertEqual(payload["chain"]["coverage"], {"start": 0, "end": 2})
        self.ack("peer-a", "ack-2", 3, [identity("r2", "o2")])
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(payload["pos"], 3)
        self.assertEqual(payload["left"], 0)
        self.assertEqual(payload["acks"], 2)
        self.assertEqual(payload["chain"]["coverage"], {"start": 0, "end": 3})

    def test_chain_carries_the_receipt_audit_conclusion(self) -> None:
        self.seed_operations(4)
        self.store.save_checkpoint("peer-a", 4)
        self.store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        # [2,3): accepted record 1 is confirmed by no receipt.
        self.store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        status, payload = self.store.get_replication_status("peer-a")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["acks"], 2)
        chain = payload["chain"]
        self.assertEqual(chain["status"], "broken")
        self.assertEqual(chain["coverage"], {"start": 0, "end": 3})
        self.assertEqual(
            chain["gaps"],
            [{"receiptIndex": 1, "ackId": "ack-2", "from": 1, "to": 2}],
        )
        self.assertEqual(chain["overlaps"], [])
        self.assertEqual(chain["identityMismatches"], [])
        self.assertEqual(chain["cursorRegressions"], [])

    def test_peer_ids_are_independent(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.store.save_checkpoint("peer-b", 2)
        self.ack("peer-a", "ack-1", 1, [identity("r0", "o0")])
        _, payload_a = self.store.get_replication_status("peer-a")
        _, payload_b = self.store.get_replication_status("peer-b")
        self.assertEqual((payload_a["pos"], payload_a["left"], payload_a["acks"]), (1, 1, 1))
        self.assertEqual((payload_b["pos"], payload_b["left"], payload_b["acks"]), (2, 0, 0))
        self.assertEqual(payload_b["chain"], empty_chain())

    def test_query_is_strictly_read_only(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 1, [identity("r0", "o0")])
        before = self.store.get_replication_status("peer-a")
        metrics_before = self.store.get_metrics()
        checkpoint_before = self.store.get_checkpoint("peer-a")
        for _ in range(3):
            after = self.store.get_replication_status("peer-a")
        self.assertEqual(after, before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_checkpoint("peer-a"), checkpoint_before)

    def test_recovery_reproduces_the_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            for index in range(3):
                store.apply_operation(
                    f"r{index}",
                    operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
                )
            store.save_checkpoint("peer-a", 0)
            status, error = store.acknowledge_operations(
                "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
            )
            self.assertEqual(status, 201, error)
            before = store.get_replication_status("peer-a")
            recovered = StateStore(data_file=data_file)
            after = recovered.get_replication_status("peer-a")
            self.assertEqual(after, before)
            self.assertEqual(after[1]["chain"]["status"], "ok")


class ReplicationStatusHttpTests(unittest.TestCase):
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

    def status_get(self, query: str) -> tuple[int, object, bytes, dict]:
        return self.raw_request("GET", f"/v1/replication/status{query}")

    def seed(self, count: int = 3) -> None:
        for index in range(count):
            status, _ = self.request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int = 0) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status in (200, 201), status

    def acknowledge(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, payload = self.request(
            "POST",
            f"/v1/sync/peers/{peer}/acknowledge",
            {"ackId": ack_id, "cursor": cursor, "operations": operations},
        )
        assert status == 201, payload

    def test_status_is_compact_ordered_json_with_one_newline(self) -> None:
        self.seed(3)
        self.register("peer-a")
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        status, payload, raw, headers = self.status_get("?peerId=peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), STATUS_FIELDS)
        self.assertEqual(list(payload["chain"]), CHAIN_FIELDS)
        self.assertEqual(
            payload,
            {
                "peer": "peer-a",
                "pos": 2,
                "left": 1,
                "acks": 1,
                "chain": {
                    "status": "ok",
                    "coverage": {"start": 0, "end": 2},
                    "gaps": [],
                    "overlaps": [],
                    "identityMismatches": [],
                    "cursorRegressions": [],
                },
            },
        )
        # Compact JSON in the contracted field order, one trailing newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(
            raw[:-1],
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        # Numbers are JSON integers, never floats or strings.
        for field in ("pos", "left", "acks"):
            self.assertIsInstance(payload[field], int)
            self.assertNotIsInstance(payload[field], bool)

    def test_empty_receipt_set_reports_ok_empty_coverage(self) -> None:
        self.seed(2)
        self.register("peer-a", cursor=1)
        status, payload, _, _ = self.status_get("?peerId=peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["pos"], 1)
        self.assertEqual(payload["left"], 1)
        self.assertEqual(payload["acks"], 0)
        self.assertEqual(payload["chain"], empty_chain())

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(1)
        self.register("peer%20one")
        status, payload, _, _ = self.status_get("?peerId=peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peer"], "peer one")

    def test_unregistered_peer_is_404(self) -> None:
        self.seed(1)
        status, payload, raw, _ = self.status_get("?peerId=ghost")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_missing_repeated_empty_unknown_and_illegal_query_is_400(self) -> None:
        self.register("peer-a")
        bad_queries = [
            "",
            "?",
            "?peerId=",
            "?peerId",
            "?peerId=peer-a&peerId=peer-a",
            "?peerId=peer-a&peerId=peer-b",
            "?peerId=peer-a&after=0",
            "?after=0&peerId=peer-a",
            "?peerId=peer-a&=",
            "?peerId=%",
            "?peerId=%2",
            "?peerId=%ZZ",
            "?peerId=%FF",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.status_get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404(self) -> None:
        self.register("peer-a")
        bad_paths = [
            "/v1/replication",
            "/v1/replication/status/",
            "/v1/replication/status/extra",
            "/v1/replication/unknown",
            "/v1/status",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?peerId=peer-a"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/status/extra?peerId=%ZZ"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_status_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/status", {"cursor": 0}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only(self) -> None:
        self.seed(2)
        self.register("peer-a")
        self.acknowledge("peer-a", "ack-1", 1, [identity("r0", "o0")])
        _, metrics_before = self.request("GET", "/v1/metrics")
        _, checkpoint_before = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        _, receipts_before = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        _, first, _, _ = self.status_get("?peerId=peer-a")
        self.status_get("?peerId=peer-a")
        self.status_get("?peerId=ghost")
        self.status_get("?peerId=%ZZ")
        _, second, _, _ = self.status_get("?peerId=peer-a")
        self.assertEqual(second, first)
        _, metrics_after = self.request("GET", "/v1/metrics")
        _, checkpoint_after = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        _, receipts_after = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(checkpoint_after, checkpoint_before)
        self.assertEqual(receipts_after, receipts_before)


class ReplicationStatusAuthTests(unittest.TestCase):
    """The status endpoint authenticates like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-status-auth-")
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
        path = "/v1/replication/status?peerId=peer-a"
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

    def test_single_token_valid_reaches_the_route(self) -> None:
        # No peer is registered, so a valid token gets the route's 404.
        status, payload, _ = self.get(
            self.single_port,
            "/v1/replication/status?peerId=peer-a",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            "/v1/replication/status?peerId=peer-a",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        # A malformed query is still 403 for a token lacking the read scope.
        status, _, headers = self.get(
            self.scope_port,
            "/v1/replication/status?peerId=%ZZ",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            "/v1/replication/status?peerId=peer-a",
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
