"""Tests for the all-peers sender-side replication delivery overview::

    GET /v1/replication/status/all?after=N&limit=N

The endpoint is a strictly read-only, paginated status query covering
**every** registered sender-side replication peer from one committed
snapshot: a detail page (``peers`` — ``peer``, ``pos``, ``left``,
``acks``, and the receipt-chain ``chainStatus`` conclusion per peer),
the resume cursor (``nextCursor``/``hasMore``), the complete registered
count (``peerCount``), the totals over the complete registered set
(``totals`` — summed ``pos``, ``left``, ``acks``), and the chain-anomaly
aggregate (``anomalies`` — the receipt audit's ``gaps``, ``overlaps``,
``identityMismatches``, and ``cursorRegressions`` lists, each marker
prefixed with its owning peer).

The tests cover the query parser (both ``after`` and ``limit`` required,
non-repeated ASCII decimal integers, limit 1-100), the store's paging,
aggregation, snapshot, and recovery semantics, the HTTP request
precedence chain (401 authentication with a Bearer challenge, 403 in
scope mode without one, 404 path shape, 400 query and out-of-range
after), the compact ordered response body with its single trailing
newline, and the read-only guarantee.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot and recovery
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

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
    load_scope_policy,
    parse_replication_status_all_query,
)

OVERVIEW_FIELDS = [
    "peers",
    "nextCursor",
    "hasMore",
    "peerCount",
    "totals",
    "anomalies",
]
PEER_FIELDS = ["peer", "pos", "left", "acks", "chainStatus"]
TOTAL_FIELDS = ["pos", "left", "acks"]
ANOMALY_FIELDS = [
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


class ParseReplicationStatusAllQueryTests(unittest.TestCase):
    def test_accepts_required_after_and_limit(self) -> None:
        self.assertEqual(parse_replication_status_all_query("after=0&limit=1"), (0, 1))
        self.assertEqual(parse_replication_status_all_query("limit=100&after=7"), (7, 100))
        # Leading zeroes are still plain ASCII decimal integers.
        self.assertEqual(parse_replication_status_all_query("after=00&limit=01"), (0, 1))

    def test_rejects_missing_repeated_unknown_and_blank(self) -> None:
        bad = [
            "",
            "after=0",
            "limit=1",
            "after",
            "limit",
            "after=",
            "limit=",
            "after=0&limit=1&x=1",
            "x=1&after=0&limit=1",
            "after=0&after=0&limit=1",
            "after=0&limit=1&limit=1",
            "after=0&limit=1&=",
            "=1&after=0&limit=1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_all_query(query))

    def test_rejects_signs_whitespace_and_non_ascii_digits(self) -> None:
        bad = [
            "after=-1&limit=1",
            "after=+1&limit=1",
            "after=%200&limit=1",
            "after=0%20&limit=1",
            "after=1.0&limit=1",
            # Fullwidth digit one (U+FF11), percent-encoded.
            "after=%EF%BC%91&limit=1",
            "after=0&limit=-1",
            "after=0&limit=+1",
            "after=0&limit=%201",
            "after=0&limit=0x1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_all_query(query))

    def test_rejects_limit_outside_one_to_one_hundred(self) -> None:
        bad = [
            "after=0&limit=0",
            "after=0&limit=101",
            "after=0&limit=1000",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_replication_status_all_query(query))


class ReplicationStatusAllStoreTests(unittest.TestCase):
    """Paging, aggregation, snapshot, and recovery against StateStore."""

    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int) -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            self.assertIn(status, (200, 201))

    def register(self, peer: str, cursor: int = 0) -> None:
        status, error = self.store.save_checkpoint(peer, cursor)
        self.assertIs(status, HTTPStatus.OK, error)

    def ack(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, error = self.store.acknowledge_operations(
            peer, ack_id, cursor, operations
        )
        self.assertEqual(status, 201, error)

    def overview(self, after: int = 0, limit: int = 100) -> dict:
        return self.store.get_replication_status_all(after, limit)

    def test_empty_registered_set_is_an_empty_all_zero_overview(self) -> None:
        payload = self.overview()
        self.assertEqual(list(payload), OVERVIEW_FIELDS)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 0)
        self.assertEqual(list(payload["totals"]), TOTAL_FIELDS)
        self.assertEqual(payload["totals"], {"pos": 0, "left": 0, "acks": 0})
        self.assertEqual(list(payload["anomalies"]), ANOMALY_FIELDS)
        self.assertEqual(
            payload["anomalies"],
            {"gaps": [], "overlaps": [], "identityMismatches": [], "cursorRegressions": []},
        )

    def test_peers_are_sorted_by_peer_id_with_contracted_fields(self) -> None:
        self.seed_operations(3)
        self.register("peer-c", cursor=3)
        self.register("peer-a", cursor=1)
        self.register("peer-b", cursor=2)
        payload = self.overview()
        self.assertEqual([entry["peer"] for entry in payload["peers"]],
                         ["peer-a", "peer-b", "peer-c"])
        for entry in payload["peers"]:
            self.assertEqual(list(entry), PEER_FIELDS)
        by_peer = {entry["peer"]: entry for entry in payload["peers"]}
        self.assertEqual(
            (by_peer["peer-a"]["pos"], by_peer["peer-a"]["left"],
             by_peer["peer-a"]["acks"], by_peer["peer-a"]["chainStatus"]),
            (1, 2, 0, "ok"),
        )
        self.assertEqual(
            (by_peer["peer-b"]["pos"], by_peer["peer-b"]["left"],
             by_peer["peer-b"]["acks"]),
            (2, 1, 0),
        )
        self.assertEqual(
            (by_peer["peer-c"]["pos"], by_peer["peer-c"]["left"],
             by_peer["peer-c"]["acks"]),
            (3, 0, 0),
        )

    def test_paging_next_cursor_and_has_more(self) -> None:
        self.seed_operations(1)
        for name in ("peer-a", "peer-b", "peer-c"):
            self.register(name)
        first = self.overview(after=0, limit=2)
        self.assertEqual([entry["peer"] for entry in first["peers"]],
                         ["peer-a", "peer-b"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        # peerCount and totals describe the complete set on every page.
        self.assertEqual(first["peerCount"], 3)
        self.assertEqual(first["totals"], {"pos": 0, "left": 3, "acks": 0})
        second = self.overview(after=2, limit=2)
        self.assertEqual([entry["peer"] for entry in second["peers"]], ["peer-c"])
        self.assertEqual(second["nextCursor"], 3)
        self.assertIs(second["hasMore"], False)
        self.assertEqual(second["peerCount"], 3)
        self.assertEqual(second["totals"], first["totals"])

    def test_after_equal_to_count_is_an_empty_tail_page(self) -> None:
        self.register("peer-a")
        payload = self.overview(after=1, limit=10)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 1)
        # Totals still cover the complete registered set on an empty page.
        self.assertEqual(payload["totals"], {"pos": 0, "left": 0, "acks": 0})

    def test_after_past_the_count_is_rejected(self) -> None:
        self.register("peer-a")
        with self.assertRaises(ValueError):
            self.overview(after=2, limit=10)
        # An empty store rejects any positive after.
        with self.assertRaises(ValueError):
            StateStore().get_replication_status_all(1, 10)

    def test_totals_sum_the_complete_registered_set(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", cursor=0)
        self.register("peer-b", cursor=4)
        self.ack("peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")])
        self.ack("peer-a", "ack-2", 3, [identity("r2", "o2")])
        # A page that hides peer-a still reports the complete totals.
        payload = self.overview(after=1, limit=10)
        self.assertEqual([entry["peer"] for entry in payload["peers"]], ["peer-b"])
        self.assertEqual(payload["peerCount"], 2)
        # peer-a: pos 3, left 1, acks 2; peer-b: pos 4, left 0, acks 0.
        self.assertEqual(payload["totals"], {"pos": 7, "left": 1, "acks": 2})
        full = self.overview()
        self.assertEqual(full["totals"], {"pos": 7, "left": 1, "acks": 2})

    def test_chain_status_and_anomalies_follow_the_receipt_audit(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", cursor=4)
        self.register("peer-b", cursor=4)
        self.store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        # [2,3): accepted record 1 is confirmed by no peer-a receipt.
        self.store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        payload = self.overview()
        by_peer = {entry["peer"]: entry for entry in payload["peers"]}
        self.assertEqual(by_peer["peer-a"]["chainStatus"], "broken")
        self.assertEqual(by_peer["peer-a"]["acks"], 2)
        self.assertEqual(by_peer["peer-b"]["chainStatus"], "ok")
        anomalies = payload["anomalies"]
        self.assertEqual(
            anomalies["gaps"],
            [{"peer": "peer-a", "receiptIndex": 1, "ackId": "ack-2",
              "from": 1, "to": 2}],
        )
        self.assertEqual(anomalies["overlaps"], [])
        self.assertEqual(anomalies["identityMismatches"], [])
        self.assertEqual(anomalies["cursorRegressions"], [])

    def test_identity_mismatch_marker_is_aggregated_with_owning_peer(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", cursor=2)
        self.store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r9", "ghost")],
        }
        payload = self.overview()
        markers = payload["anomalies"]["identityMismatches"]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["peer"], "peer-a")
        self.assertEqual(markers[0]["receiptIndex"], 0)
        self.assertEqual(markers[0]["ackId"], "ack-1")
        self.assertEqual(markers[0]["position"], 0)
        self.assertEqual(markers[0]["expected"],
                         {"replicaId": "r0", "operationId": "o0"})
        self.assertEqual(markers[0]["observed"],
                         {"replicaId": "r9", "operationId": "ghost"})
        self.assertEqual(payload["peers"][0]["chainStatus"], "broken")

    def test_anomaly_lists_are_grouped_by_kind_then_peer_id(self) -> None:
        self.seed_operations(4)
        # Both peers get a gapped confirmation chain.
        for peer in ("peer-a", "peer-b"):
            self.register(peer, cursor=4)
            self.store._acks[(peer, "ack-1")] = {
                "cursor": 1,
                "operations": [identity("r0", "o0")],
            }
            self.store._acks[(peer, "ack-2")] = {
                "cursor": 3,
                "operations": [identity("r2", "o2")],
            }
        payload = self.overview()
        gaps = payload["anomalies"]["gaps"]
        self.assertEqual([marker["peer"] for marker in gaps], ["peer-a", "peer-b"])
        for marker in gaps:
            self.assertEqual(marker["from"], 1)
            self.assertEqual(marker["to"], 2)

    def test_numbers_are_plain_integers(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", cursor=1)
        payload = self.overview()
        for field in ("nextCursor", "peerCount"):
            self.assertIsInstance(payload[field], int)
            self.assertNotIsInstance(payload[field], bool)
        for field in TOTAL_FIELDS:
            self.assertIsInstance(payload["totals"][field], int)
            self.assertNotIsInstance(payload["totals"][field], bool)
        for entry in payload["peers"]:
            for field in ("pos", "left", "acks"):
                self.assertIsInstance(entry[field], int)
                self.assertNotIsInstance(entry[field], bool)

    def test_query_is_strictly_read_only(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", cursor=0)
        self.ack("peer-a", "ack-1", 1, [identity("r0", "o0")])
        before = self.overview()
        metrics_before = self.store.get_metrics()
        checkpoint_before = self.store.get_checkpoint("peer-a")
        for _ in range(3):
            after = self.overview()
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
            store.save_checkpoint("peer-b", 3)
            store.save_checkpoint("peer-a", 0)
            status, error = store.acknowledge_operations(
                "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
            )
            self.assertEqual(status, 201, error)
            before = store.get_replication_status_all(0, 100)
            recovered = StateStore(data_file=data_file)
            after = recovered.get_replication_status_all(0, 100)
            self.assertEqual(after, before)
            self.assertEqual([entry["peer"] for entry in after["peers"]],
                             ["peer-a", "peer-b"])
            self.assertEqual(after["totals"], {"pos": 5, "left": 1, "acks": 1})


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
            conn.request(method, path, body=json.dumps(body), headers=request_headers)
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
        self.register("peer-a")
        self.acknowledge(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        status, payload, raw, headers = self.overview_get("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), OVERVIEW_FIELDS)
        self.assertEqual(list(payload["totals"]), TOTAL_FIELDS)
        self.assertEqual(list(payload["anomalies"]), ANOMALY_FIELDS)
        self.assertEqual(list(payload["peers"][0]), PEER_FIELDS)
        self.assertEqual(
            payload,
            {
                "peers": [
                    {"peer": "peer-a", "pos": 2, "left": 1,
                     "acks": 1, "chainStatus": "ok"}
                ],
                "nextCursor": 1,
                "hasMore": False,
                "peerCount": 1,
                "totals": {"pos": 2, "left": 1, "acks": 1},
                "anomalies": {
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

    def test_empty_registered_set_returns_empty_page_and_zero_totals(self) -> None:
        status, payload, _, _ = self.overview_get("?after=0&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 0)
        self.assertEqual(payload["totals"], {"pos": 0, "left": 0, "acks": 0})
        self.assertEqual(
            payload["anomalies"],
            {"gaps": [], "overlaps": [], "identityMismatches": [], "cursorRegressions": []},
        )

    def test_paging_and_sort_order(self) -> None:
        self.seed(2)
        self.register("peer-c", cursor=2)
        self.register("peer-a")
        self.register("peer-b", cursor=1)
        status, first, _, _ = self.overview_get("?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([entry["peer"] for entry in first["peers"]],
                         ["peer-a", "peer-b"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        self.assertEqual(first["peerCount"], 3)
        self.assertEqual(first["totals"], {"pos": 3, "left": 3, "acks": 0})
        status, second, _, _ = self.overview_get("?after=2&limit=2")
        self.assertEqual([entry["peer"] for entry in second["peers"]], ["peer-c"])
        self.assertEqual(second["nextCursor"], 3)
        self.assertIs(second["hasMore"], False)

    def test_after_equal_to_count_is_an_empty_tail_page(self) -> None:
        self.register("peer-a")
        self.register("peer-b")
        status, payload, raw, _ = self.overview_get("?after=2&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(payload["peers"], [])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["peerCount"], 2)
        self.assertTrue(raw.endswith(b"\n"))

    def test_missing_repeated_unknown_blank_signed_and_non_ascii_is_400(self) -> None:
        self.register("peer-a")
        bad_queries = [
            "",
            "?",
            "?after=0",
            "?limit=10",
            "?after=&limit=10",
            "?after=0&limit=",
            "?after&limit=10",
            "?after=0&limit=10&x=1",
            "?after=0&after=0&limit=10",
            "?after=0&limit=10&limit=10",
            "?after=-1&limit=10",
            "?after=0&limit=-1",
            "?after=%20&limit=10",
            "?after=0&limit=1.0",
            "?after=%EF%BC%91&limit=10",
            "?after=0&limit=0",
            "?after=0&limit=101",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.overview_get(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_the_count_is_400_against_the_snapshot(self) -> None:
        # An empty store: every positive after is out of range.
        status, payload, _, _ = self.overview_get("?after=1&limit=10")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.register("peer-a")
        status, payload, _, _ = self.overview_get("?after=2&limit=10")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_mismatches_are_404(self) -> None:
        self.register("peer-a")
        bad_paths = [
            "/v1/replication",
            "/v1/replication/status/all/",
            "/v1/replication/status/all/extra",
            "/v1/replication/unknown",
            "/v1/status",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(
                    "GET", f"{path}?after=0&limit=10"
                )
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_shape_check_precedes_query_check(self) -> None:
        status, payload, _, _ = self.raw_request(
            "GET", "/v1/replication/status/all/extra?after=xx&limit=0"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_overview_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/status/all", {"after": 0, "limit": 10}
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
        self.overview_get("?after=0&limit=100")
        self.overview_get("?after=1&limit=100")
        self.overview_get("?after=2&limit=100")
        self.overview_get("?after=xx&limit=0")
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
    """The overview authenticates like every other non-/health route."""

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

    PATH = "/v1/replication/status/all?after=0&limit=10"

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
        # Duplicated Authorization header.
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("GET", self.PATH)
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
            self.PATH,
            [("Authorization", "Bearer s3cret-token")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["peerCount"], 0)

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
            "/v1/replication/status/all?after=xx",
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
        self.assertEqual(payload["peerCount"], 0)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.get(self.single_port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
