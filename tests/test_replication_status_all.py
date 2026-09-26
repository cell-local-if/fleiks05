"""Tests for the all-peers sender-side replication delivery overview::

    GET /v1/replication/status/all?after=N&limit=N

The endpoint is a strictly read-only delivery-status overview across
every registered sender-side replication peer: one page of per-peer
details (``peer``, ``pos``, ``left``, ``acks``, ``chainStatus``) in
ascending ``peerId`` order, plus ``nextCursor``/``hasMore`` paging, the
complete registered count (``peerCount``), the ``pos``/``left``/``acks``
sums over the complete registered set (``totals``), and the per-class
anomaly peer counts over the complete registered set (``anomalies`` —
``gaps``, ``overlaps``, ``identityMismatches``, ``cursorRegressions``),
all from one committed snapshot.

The tests cover the query parser (required, non-repeated, ASCII-decimal
``after``/``limit``; ``limit`` between 1 and 100), the store's paging,
sorting, summary, and recovery semantics, the HTTP request precedence
chain (401 authentication with a Bearer challenge, 403 in scope mode
without one, 404 path shape, 400 query validation including an ``after``
past the registered count), the compact ordered response body with its
single trailing newline, and the read-only guarantee.

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
    parse_replication_status_all_query,
)

OVERVIEW_FIELDS = ["peers", "nextCursor", "hasMore", "peerCount", "totals", "anomalies"]
PEER_FIELDS = ["peer", "pos", "left", "acks", "chainStatus"]
TOTALS_FIELDS = ["pos", "left", "acks"]
ANOMALY_FIELDS = ["gaps", "overlaps", "identityMismatches", "cursorRegressions"]

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


def zero_totals() -> dict:
    return {"pos": 0, "left": 0, "acks": 0}


def zero_anomalies() -> dict:
    return {"gaps": 0, "overlaps": 0, "identityMismatches": 0, "cursorRegressions": 0}


class ParseReplicationStatusAllQueryTests(unittest.TestCase):
    def test_accepts_required_after_and_limit(self) -> None:
        self.assertEqual(parse_replication_status_all_query("after=0&limit=1"), (0, 1))
        self.assertEqual(
            parse_replication_status_all_query("limit=100&after=42"), (42, 100)
        )
        self.assertEqual(
            parse_replication_status_all_query("after=007&limit=09"), (7, 9)
        )

    def test_rejects_missing_repeated_unknown_and_blank(self) -> None:
        bad = [
            "",
            "after=0",
            "limit=1",
            "after=0&limit=1&after=2",
            "after=0&limit=1&limit=2",
            "after=0&limit=1&x=1",
            "x=1&after=0&limit=1",
            "after=&limit=1",
            "after=0&limit=",
            "after&limit=1",
            "after=0&limit=1&=",
            "=1&after=0&limit=1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_all_query(query))

    def test_rejects_non_ascii_decimal_signed_and_out_of_range(self) -> None:
        bad = [
            "after=-1&limit=1",
            "after=+1&limit=1",
            "after=1.0&limit=1",
            "after= 1&limit=1",
            "after=1 &limit=1",
            "after=١&limit=1",  # Arabic-Indic digit
            "after=１&limit=1",  # fullwidth digit
            "after=0&limit=0",
            "after=0&limit=101",
            "after=0&limit=-1",
            "after=0&limit=1.5",
            "after=0&limit=١",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_all_query(query))


class ReplicationStatusAllStoreTests(unittest.TestCase):
    """Paging, sorting, summary, and recovery semantics against StateStore."""

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

    def test_empty_registry_reports_an_empty_all_zero_overview(self) -> None:
        status, payload = self.store.get_replication_status_all(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), OVERVIEW_FIELDS)
        self.assertEqual(
            payload,
            {
                "peers": [],
                "nextCursor": 0,
                "hasMore": False,
                "peerCount": 0,
                "totals": zero_totals(),
                "anomalies": zero_anomalies(),
            },
        )

    def test_after_equal_to_the_count_is_a_valid_empty_page(self) -> None:
        self.store.save_checkpoint("peer-a", 0)
        status, payload = self.store.get_replication_status_all(1, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertEqual(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 1)

    def test_after_past_the_count_is_rejected(self) -> None:
        self.store.save_checkpoint("peer-a", 0)
        # Equal to the count is a valid empty page; only past it raises.
        status, payload = self.store.get_replication_status_all(1, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["peers"], [])
        with self.assertRaises(ValueError):
            self.store.get_replication_status_all(2, 100)
        empty = StateStore()
        with self.assertRaises(ValueError):
            empty.get_replication_status_all(1, 100)

    def test_peers_are_sorted_and_paged(self) -> None:
        self.seed_operations(3)
        for peer in ("peer-c", "peer-a", "peer-b"):
            self.store.save_checkpoint(peer, 0)
        status, payload = self.store.get_replication_status_all(0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([item["peer"] for item in payload["peers"]], ["peer-a", "peer-b"])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertEqual(payload["hasMore"], True)
        self.assertEqual(payload["peerCount"], 3)
        status, payload = self.store.get_replication_status_all(2, 2)
        self.assertEqual([item["peer"] for item in payload["peers"]], ["peer-c"])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertEqual(payload["hasMore"], False)

    def test_peer_items_carry_progress_and_chain_status(self) -> None:
        self.seed_operations(3)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        self.store.save_checkpoint("peer-b", 3)
        status, payload = self.store.get_replication_status_all(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(payload["peers"]), 2)
        for item in payload["peers"]:
            self.assertEqual(list(item), PEER_FIELDS)
        self.assertEqual(
            payload["peers"][0],
            {"peer": "peer-a", "pos": 2, "left": 1, "acks": 1, "chainStatus": "ok"},
        )
        self.assertEqual(
            payload["peers"][1],
            {"peer": "peer-b", "pos": 3, "left": 0, "acks": 0, "chainStatus": "ok"},
        )
        self.assertEqual(payload["totals"], {"pos": 5, "left": 1, "acks": 1})
        self.assertEqual(payload["anomalies"], zero_anomalies())

    def test_totals_and_anomalies_cover_the_complete_set_not_the_page(self) -> None:
        self.seed_operations(4)
        self.store.save_checkpoint("peer-a", 1)
        self.store.save_checkpoint("peer-b", 4)
        self.store.save_checkpoint("peer-c", 2)
        # peer-b's committed chain skips accepted record 2: a gap.
        self.store._acks[("peer-b", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        self.store._acks[("peer-b", "ack-2")] = {
            "cursor": 4,
            "operations": [identity("r3", "o3")],
        }
        # The page holds only peer-a, but the summary covers all three.
        status, payload = self.store.get_replication_status_all(0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([item["peer"] for item in payload["peers"]], ["peer-a"])
        self.assertEqual(payload["peerCount"], 3)
        self.assertEqual(payload["totals"], {"pos": 7, "left": 5, "acks": 2})
        self.assertEqual(
            payload["anomalies"],
            {"gaps": 1, "overlaps": 0, "identityMismatches": 0, "cursorRegressions": 0},
        )

    def test_anomalies_count_peers_per_class(self) -> None:
        self.seed_operations(4)
        for peer in ("peer-a", "peer-b", "peer-c", "peer-d", "peer-e"):
            self.store.save_checkpoint(peer, 4)
        # peer-a: a gap between the first and second receipt.
        self.store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        self.store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        # peer-b: the second receipt starts before the first one ended.
        self.store._acks[("peer-b", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        self.store._acks[("peer-b", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r1", "o1"), identity("r2", "o2")],
        }
        # peer-c: one confirmed position names the wrong identity.
        self.store._acks[("peer-c", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r9", "o9")],
        }
        # peer-d: the confirmation cursor regresses (which also overlaps).
        self.store._acks[("peer-d", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        self.store._acks[("peer-d", "ack-2")] = {"cursor": 1, "operations": []}
        # peer-e: a clean, seamless chain.
        self.store._acks[("peer-e", "ack-1")] = {
            "cursor": 2,
            "operations": [identity("r0", "o0"), identity("r1", "o1")],
        }
        status, payload = self.store.get_replication_status_all(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [item["chainStatus"] for item in payload["peers"]],
            ["broken", "broken", "broken", "broken", "ok"],
        )
        self.assertEqual(
            payload["anomalies"],
            {"gaps": 1, "overlaps": 2, "identityMismatches": 1, "cursorRegressions": 1},
        )
        self.assertEqual(payload["totals"]["acks"], 8)

    def test_query_is_strictly_read_only(self) -> None:
        self.seed_operations(2)
        self.store.save_checkpoint("peer-a", 0)
        self.ack("peer-a", "ack-1", 1, [identity("r0", "o0")])
        before = self.store.get_replication_status_all(0, 100)
        metrics_before = self.store.get_metrics()
        checkpoint_before = self.store.get_checkpoint("peer-a")
        for _ in range(3):
            after = self.store.get_replication_status_all(0, 100)
        self.assertEqual(after, before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_checkpoint("peer-a"), checkpoint_before)

    def test_recovery_reproduces_the_overview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            for index in range(3):
                store.apply_operation(
                    f"r{index}",
                    operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
                )
            store.save_checkpoint("peer-a", 0)
            store.save_checkpoint("peer-b", 3)
            status, error = store.acknowledge_operations(
                "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
            )
            self.assertEqual(status, 201, error)
            before = store.get_replication_status_all(0, 100)
            recovered = StateStore(data_file=data_file)
            after = recovered.get_replication_status_all(0, 100)
            self.assertEqual(after, before)
            self.assertEqual(after[1]["peerCount"], 2)


class ReplicationStatusAllHttpTests(unittest.TestCase):
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

    def overview_get(self, query: str) -> tuple[int, object, bytes, dict]:
        return self.raw_request("GET", f"/v1/replication/status/all{query}")

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

    def test_overview_is_compact_ordered_json_with_one_newline(self) -> None:
        self.seed(3)
        self.register("peer-b", cursor=3)
        self.register("peer-a")
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        status, payload, raw, headers = self.overview_get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), OVERVIEW_FIELDS)
        self.assertEqual(
            payload,
            {
                "peers": [
                    {
                        "peer": "peer-a",
                        "pos": 2,
                        "left": 1,
                        "acks": 1,
                        "chainStatus": "ok",
                    },
                    {
                        "peer": "peer-b",
                        "pos": 3,
                        "left": 0,
                        "acks": 0,
                        "chainStatus": "ok",
                    },
                ],
                "nextCursor": 2,
                "hasMore": False,
                "peerCount": 2,
                "totals": {"pos": 5, "left": 1, "acks": 1},
                "anomalies": {
                    "gaps": 0,
                    "overlaps": 0,
                    "identityMismatches": 0,
                    "cursorRegressions": 0,
                },
            },
        )
        for item in payload["peers"]:
            self.assertEqual(list(item), PEER_FIELDS)
        self.assertEqual(list(payload["totals"]), TOTALS_FIELDS)
        self.assertEqual(list(payload["anomalies"]), ANOMALY_FIELDS)
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
        for field in ("nextCursor", "peerCount"):
            self.assertIsInstance(payload[field], int)
            self.assertNotIsInstance(payload[field], bool)
        for field in TOTALS_FIELDS:
            self.assertIsInstance(payload["totals"][field], int)
            self.assertNotIsInstance(payload["totals"][field], bool)
        for field in ANOMALY_FIELDS:
            self.assertIsInstance(payload["anomalies"][field], int)
            self.assertNotIsInstance(payload["anomalies"][field], bool)

    def test_empty_registry_reports_an_empty_all_zero_overview(self) -> None:
        status, payload, raw, _ = self.overview_get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "peers": [],
                "nextCursor": 0,
                "hasMore": False,
                "peerCount": 0,
                "totals": zero_totals(),
                "anomalies": zero_anomalies(),
            },
        )
        self.assertTrue(raw.endswith(b"\n"))

    def test_paging_walks_the_sorted_registry(self) -> None:
        self.seed(1)
        for peer in ("peer-c", "peer-a", "peer-b"):
            self.register(peer)
        seen = []
        after = 0
        for _ in range(3):
            status, payload, _, _ = self.overview_get(f"?after={after}&limit=2")
            self.assertEqual(status, 200)
            seen.extend(item["peer"] for item in payload["peers"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual(seen, ["peer-a", "peer-b", "peer-c"])
        self.assertEqual(after, 3)

    def test_after_equal_to_the_count_is_an_empty_page_past_it_is_400(self) -> None:
        self.register("peer-a")
        status, payload, _, _ = self.overview_get("?after=1&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertEqual(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 1)
        status, payload, raw, _ = self.overview_get("?after=2&limit=100")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_missing_repeated_unknown_and_malformed_query_is_400(self) -> None:
        self.register("peer-a")
        bad_queries = [
            "",
            "?",
            "?after=0",
            "?limit=1",
            "?after=0&limit=1&after=0",
            "?after=0&limit=1&limit=1",
            "?after=0&limit=1&x=1",
            "?after=&limit=1",
            "?after=0&limit=",
            "?after=-1&limit=1",
            "?after=+1&limit=1",
            "?after=1.0&limit=1",
            "?after=%201&limit=1",  # percent-encoded leading space
            "?after=%D9%A1&limit=1",  # percent-encoded Arabic-Indic digit
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=-5",
            "?after=0&limit=%D9%A1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.overview_get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_route_shape_mismatches_are_404(self) -> None:
        self.register("peer-a")
        bad_paths = [
            "/v1/replication",
            "/v1/replication/status/all/",
            "/v1/replication/status/all/extra",
            "/v1/replication/status/unknown",
            "/v1/replication/unknown",
            "/v1/status/all",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=0&limit=1"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/status/all/extra?after=%ZZ&limit=x"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_status_all_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/status/all", {"cursor": 0}
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
        _, first, _, _ = self.overview_get("?after=0&limit=100")
        self.overview_get("?after=0&limit=1")
        self.overview_get("?after=5&limit=100")
        self.overview_get("?after=%ZZ")
        _, second, _, _ = self.overview_get("?after=0&limit=100")
        self.assertEqual(second, first)
        _, metrics_after = self.request("GET", "/v1/metrics")
        _, checkpoint_after = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        _, receipts_after = self.request(
            "GET", "/v1/sync/peers/peer-a/receipts?after=0&limit=100"
        )
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(checkpoint_after, checkpoint_before)
        self.assertEqual(receipts_after, receipts_before)


class ReplicationStatusAllAuthTests(unittest.TestCase):
    """The overview endpoint authenticates like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-status-all-auth-")
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
        path = "/v1/replication/status/all?after=0&limit=1"
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
        status, payload, _ = self.get(
            self.single_port,
            "/v1/replication/status/all?after=0&limit=1",
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["peerCount"], 0)

    def test_scope_mode_write_only_is_403_without_challenge(self) -> None:
        status, payload, headers = self.get(
            self.scope_port,
            "/v1/replication/status/all?after=0&limit=1",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_decision_precedes_the_query_check(self) -> None:
        # A malformed query is still 403 for a token lacking the read scope.
        status, _, headers = self.get(
            self.scope_port,
            "/v1/replication/status/all?after=%ZZ",
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
        )
        self.assertEqual(status, 403)
        self.assertNotIn("WWW-Authenticate", headers)

    def test_scope_mode_reader_reaches_the_route(self) -> None:
        status, payload, _ = self.get(
            self.scope_port,
            "/v1/replication/status/all?after=0&limit=1",
            [("Authorization", f"Bearer {READ_TOKEN}")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["peerCount"], 0)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
