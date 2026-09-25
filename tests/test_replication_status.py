"""Tests for the sender-side replication delivery-status endpoint::

    GET /v1/replication/status?peerId=P

It is a strictly read-only query over one committed snapshot: the peer's
registered checkpoint cursor (``pos``), the number of unconsumed accepted
records (``left``), the peer's committed receipt count (``acks``), and
the chain-integrity conclusion over the peer's whole confirmation chain
(``chain``, the same audit conclusion as the receipts-audit route) are
assembled under the shared commit lock, so they always describe a single
commit. The success body keeps the contracted field order
``peer,pos,left,acks,chain`` as compact UTF-8 JSON terminated by one
newline, with numbers only as JSON integers. ``peerId`` is a required
query parameter following the replication routes' percent-decoding and
non-empty rules; an unregistered peer and any path-shape error are 404,
a missing/repeated/blank/unknown/badly-encoded parameter is 400, and the
route authenticates like every other non-``/health`` route.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

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
    load_scope_policy,
    parse_replication_status_query,
)

STATUS_FIELDS = {"peer", "pos", "left", "acks", "chain"}
CHAIN_FIELDS = {
    "status",
    "coverage",
    "gaps",
    "overlaps",
    "identityMismatches",
    "cursorRegressions",
}

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


def ok_chain(start: int, end: int) -> dict:
    return {
        "status": "ok",
        "coverage": {"start": start, "end": end},
        "gaps": [],
        "overlaps": [],
        "identityMismatches": [],
        "cursorRegressions": [],
    }


class ParseReplicationStatusQueryTests(unittest.TestCase):
    def test_accepts_a_single_non_empty_peer_id(self) -> None:
        self.assertEqual(parse_replication_status_query("peerId=peer-a"), "peer-a")
        # Percent decoding follows the replication path-segment rules.
        self.assertEqual(parse_replication_status_query("peerId=peer%2Da"), "peer-a")
        self.assertEqual(parse_replication_status_query("peerId=%70%65%65%72"), "peer")
        self.assertEqual(parse_replication_status_query("peerId=%C3%A9"), "é")
        self.assertEqual(parse_replication_status_query("peerId=a%20b"), "a b")

    def test_missing_repeated_or_blank_peer_id_is_rejected(self) -> None:
        for query in ("", "x=1", "peerId=", "peerId", "peerId=a&peerId=b"):
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))

    def test_unknown_parameters_are_rejected(self) -> None:
        for query in (
            "peerId=a&x=1",
            "x=1&peerId=a",
            "peerId=a&=",
            "peerId=a&peerid=b",
            "PEERID=a",
        ):
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))

    def test_malformed_percent_escapes_are_rejected(self) -> None:
        for query in (
            "peerId=%",
            "peerId=%2",
            "peerId=%ZZ",
            "peerId=a%",
            "peerId=a%2G",
            "peerId=%%32",
            "x=%&peerId=a",
        ):
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))

    def test_non_utf8_percent_bytes_are_rejected(self) -> None:
        for query in ("peerId=%FF", "peerId=%C3%28", "peerId=%80"):
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_query(query))


class ReplicationStatusStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int, key: str = "k") -> list[tuple[str, dict]]:
        records = []
        for i in range(count):
            replica = f"r{i}"
            op = operation(f"o{i}", key, f"v{i}", {replica: 1})
            self.assertIn(self.store.apply_operation(replica, op), (200, 201))
            records.append((replica, op))
        return records

    def ack(self, peer: str, ack_id: str, cursor: int, ids: list) -> None:
        status, error = self.store.acknowledge_operations(peer, ack_id, cursor, ids)
        self.assertEqual((status, error), (201, None))

    def test_unregistered_peer_is_not_found(self) -> None:
        self.seed_operations(1)
        status, payload = self.store.get_replication_status("nobody")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 1)
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "peer": "peer-a",
                "pos": 1,
                "left": 2,
                "acks": 0,
                "chain": ok_chain(0, 0),
            },
        )

    def test_status_tracks_checkpoint_and_receipts(self) -> None:
        records = self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 2, [identity(r, o["operationId"]) for r, o in records[:2]])
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peer"], "peer-a")
        self.assertEqual(payload["pos"], 2)
        self.assertEqual(payload["left"], 1)
        self.assertEqual(payload["acks"], 1)
        self.assertEqual(payload["chain"], ok_chain(0, 2))
        self.ack(
            "peer-a",
            "ack-2",
            3,
            [identity(records[2][0], records[2][1]["operationId"])],
        )
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual((payload["pos"], payload["left"], payload["acks"]), (3, 0, 2))
        self.assertEqual(payload["chain"], ok_chain(0, 3))

    def test_other_peers_receipts_are_not_counted(self) -> None:
        records = self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.store.save_checkpoint("peer-b", 0)
        self.ack("peer-b", "ack-1", 2, [identity(r, o["operationId"]) for r, o in records])
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual((payload["pos"], payload["left"], payload["acks"]), (0, 2, 0))
        self.assertEqual(payload["chain"], ok_chain(0, 0))
        status, payload = self.store.get_replication_status("peer-b")
        self.assertEqual(status, 200)
        self.assertEqual((payload["pos"], payload["left"], payload["acks"]), (2, 0, 1))

    def test_chain_reflects_the_receipt_audit_conclusion(self) -> None:
        records = self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 2, [identity(r, o["operationId"]) for r, o in records])
        # Tamper with the committed receipt behind the store's back: the
        # status query must report the same broken conclusion the
        # receipts-audit route derives from the same snapshot.
        self.store._acks[("peer-a", "ack-1")]["operations"][0] = identity("rx", "ox")
        status, payload = self.store.get_replication_status("peer-a")
        self.assertEqual(status, 200)
        chain = payload["chain"]
        self.assertEqual(chain["status"], "broken")
        self.assertEqual(len(chain["identityMismatches"]), 1)
        self.assertEqual(chain["identityMismatches"][0]["position"], 0)
        _, audit_payload = self.store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertEqual(chain, audit_payload["audit"])

    def test_query_is_strictly_read_only(self) -> None:
        records = self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 1, [identity(records[0][0], "o0")])
        before = self.store.get_replication_status("peer-a")
        metrics_before = self.store.get_metrics()
        for _ in range(3):
            self.assertEqual(self.store.get_replication_status("peer-a"), before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        _, checkpoint = self.store.get_checkpoint("peer-a")
        self.assertEqual(checkpoint["cursor"], 1)
        # The unconsumed pickup stream is untouched by the status query.
        status, pickup = self.store.get_peer_operations("peer-a", 0, 100)
        self.assertEqual(status, 200)
        self.assertEqual(len(pickup["operations"]), 1)


class ReplicationStatusRecoveryTests(unittest.TestCase):
    def test_status_matches_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            records = []
            for i in range(3):
                replica = f"r{i}"
                op = operation(f"o{i}", "k", f"v{i}", {replica: 1})
                store.apply_operation(replica, op)
                records.append((replica, op))
            store.save_checkpoint("peer-a", 0)
            store.acknowledge_operations(
                "peer-a",
                "ack-1",
                2,
                [identity(r, o["operationId"]) for r, o in records[:2]],
            )
            before = store.get_replication_status("peer-a")
            del store
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_replication_status("peer-a"), before)


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
        self, method: str, path: str, body: object = None
    ) -> tuple[int, object, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def status_query(self, query: str) -> tuple[int, dict, bytes, dict]:
        return self.raw_request("GET", f"/v1/replication/status{query}")

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def seed(self, count: int = 2) -> list[tuple[str, dict]]:
        records = []
        for i in range(count):
            replica = f"r{i}"
            op = operation(f"o{i}", "k", f"v{i}", {replica: 1})
            status, _ = self.post_operation(replica, op)
            assert status == 201
            records.append((replica, op))
        return records

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status == 200

    def acknowledge(self, peer: str, ack_id: str, cursor: int, ids: list) -> None:
        status, _ = self.request(
            "POST",
            f"/v1/sync/peers/{peer}/acknowledge",
            {"ackId": ack_id, "cursor": cursor, "operations": ids},
        )
        assert status == 201

    def test_unregistered_peer_is_404(self) -> None:
        status, payload, raw, _ = self.status_query("?peerId=nobody")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_registered_peer_without_receipts(self) -> None:
        self.seed(3)
        self.register("peer-a", 1)
        status, payload, raw, headers = self.status_query("?peerId=peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), STATUS_FIELDS)
        self.assertEqual(payload["peer"], "peer-a")
        self.assertEqual(payload["pos"], 1)
        self.assertEqual(payload["left"], 2)
        self.assertEqual(payload["acks"], 0)
        self.assertEqual(set(payload["chain"]), CHAIN_FIELDS)
        self.assertEqual(payload["chain"], ok_chain(0, 0))
        for name in ("pos", "left", "acks"):
            self.assertIs(type(payload[name]), int, f"{name} must be an int")
        # Compact JSON terminated by exactly one newline, fields in the
        # contracted order peer,pos,left,acks,chain.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        expected = {
            "peer": "peer-a",
            "pos": 1,
            "left": 2,
            "acks": 0,
            "chain": ok_chain(0, 0),
        }
        self.assertEqual(
            raw[:-1],
            json.dumps(expected, separators=(",", ":")).encode("utf-8"),
        )
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_status_after_acknowledgement(self) -> None:
        records = self.seed(3)
        self.register("peer-a", 0)
        self.acknowledge(
            "peer-a",
            "ack-1",
            2,
            [identity(r, o["operationId"]) for r, o in records[:2]],
        )
        status, payload, _, _ = self.status_query("?peerId=peer-a")
        self.assertEqual(status, 200)
        self.assertEqual((payload["pos"], payload["left"], payload["acks"]), (2, 1, 1))
        self.assertEqual(payload["chain"], ok_chain(0, 2))
        # The chain conclusion matches the receipts-audit route's audit.
        _, audit = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts/audit?after=0&limit=100"
        )
        self.assertEqual(payload["chain"], audit["audit"])

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed(1)
        self.register("p%C3%A9er", 1)
        status, payload, _, _ = self.status_query("?peerId=p%C3%A9er")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peer"], "péer")
        self.assertEqual((payload["pos"], payload["left"]), (1, 0))

    def test_status_agrees_with_snapshot_and_checkpoint(self) -> None:
        self.seed(2)
        self.register("peer-a", 1)
        status, payload, _, _ = self.status_query("?peerId=peer-a")
        self.assertEqual(status, 200)
        _, snapshot = self.request("GET", "/v1/replication/snapshot")
        _, checkpoint = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(payload["pos"], checkpoint["cursor"])
        self.assertEqual(payload["left"], snapshot["logCursor"] - checkpoint["cursor"])

    # -- query validation --

    def test_bad_queries_are_400(self) -> None:
        self.seed(1)
        self.register("peer-a", 0)
        bad_queries = [
            "",
            "?",
            "?x=1",
            "?peerId=",
            "?peerId",
            "?peerId=peer-a&peerId=peer-a",
            "?peerId=peer-a&x=1",
            "?peerId=peer-a&=",
            "?peerId=%",
            "?peerId=%2",
            "?peerId=%ZZ",
            "?peerId=%FF",
            "?peerid=peer-a",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.status_query(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    # -- route shape precedes the query check --

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication",
            "/v1/replication/status/",
            "/v1/replication/status/extra",
            "/v1/replication/statusx",
            "/v1/replication/snapshot/status",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request("GET", f"{path}?peerId=peer-a")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/status/extra?peerId=%"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request(
            "POST", "/v1/replication/status", {"peerId": "peer-a"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_strictly_read_only(self) -> None:
        records = self.seed(2)
        self.register("peer-a", 0)
        self.acknowledge("peer-a", "ack-1", 1, [identity(records[0][0], "o0")])
        _, metrics_before = self.request("GET", "/v1/metrics")
        _, sync_before = self.request("GET", "/v1/sync/operations")
        _, checkpoint_before = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        status, first, _, _ = self.status_query("?peerId=peer-a")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, again, _, _ = self.status_query("?peerId=peer-a")
            self.assertEqual(status, 200)
            self.assertEqual(again, first)
        _, metrics_after = self.request("GET", "/v1/metrics")
        _, sync_after = self.request("GET", "/v1/sync/operations")
        _, checkpoint_after = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(sync_after, sync_before)
        self.assertEqual(checkpoint_after, checkpoint_before)
        # No receipt was created: the receipts stream is unchanged.
        _, receipts = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(receipts["receiptsCount"], 1)


class ReplicationStatusPersistenceHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_status_survives_restart(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 0})
        self.request(
            server,
            "POST",
            "/v1/sync/peers/peer-a/acknowledge",
            {"ackId": "ack-1", "cursor": 1, "operations": [identity("r1", "o1")]},
        )
        status, before = self.request(server, "GET", "/v1/replication/status?peerId=peer-a")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/replication/status?peerId=peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        self.assertEqual((after["pos"], after["left"], after["acks"]), (1, 1, 1))

    def test_status_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.request(server, "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 1})
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/replication/status?peerId=peer-a")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


class ReplicationStatusAuthTests(unittest.TestCase):
    """The endpoint authenticates like every other non-/health route."""

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

    def test_missing_bad_or_wrong_token_is_401_with_challenge(self) -> None:
        path = "/v1/replication/status?peerId=peer-a"
        for headers in (
            [],
            [("Authorization", "Bearer wrong")],
            [("Authorization", "s3cret-token")],
        ):
            with self.subTest(headers=headers):
                status, payload, response_headers = self.get(
                    self.single_port, path, headers
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(response_headers["WWW-Authenticate"], "Bearer")
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

    def test_single_token_valid_reaches_the_endpoint(self) -> None:
        status, payload, _ = self.get(
            self.single_port,
            "/v1/replication/status?peerId=peer-a",
            [("Authorization", "Bearer s3cret-token")],
        )
        # Authenticated; the peer itself is unregistered.
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
            "/v1/replication/status?peerId=%",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_endpoint(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            "/v1/replication/status?peerId=peer-a",
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unauthorized_request_reads_no_state(self) -> None:
        # Register a peer on the scope server with an admin-less write? The
        # write routes need the write scope; use the reader's 404 boundary
        # instead: an unauthenticated request must not reveal registration.
        status, payload, headers = self.get(
            self.scope_port, "/v1/replication/status?peerId=peer-a"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
