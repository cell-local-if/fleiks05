"""Tests for per-peer consumption fetch.

The endpoint is::

    GET /v1/sync/peers/{peerId}/operations?after=N&limit=N

It pages the shared accepted-operation log from a peer's registered
checkpoint onward, in global commit order. The query is strictly
read-only: it neither advances nor writes the checkpoint and changes no
other state. Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    main,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"replicaId": replica_id, "operation": operation(operation_id, key, value, clock)}


class PeerOperationsStoreTests(unittest.TestCase):
    """Store-level paging semantics over the unconsumed tail."""

    def fill(self, store: StateStore, count: int) -> None:
        for index in range(1, count + 1):
            replica = f"r{index}"
            store.apply_operation(
                replica, operation(f"o{index}", f"k{index}", f"v{index}", {replica: 1})
            )

    def test_unknown_peer_returns_none(self) -> None:
        store = StateStore()
        self.fill(store, 2)
        self.assertIsNone(store.get_peer_operations("nope", 0, 100))

    def test_checkpoint_zero_sees_the_whole_log(self) -> None:
        store = StateStore()
        self.fill(store, 3)
        store.save_checkpoint("p1", 0)
        page, next_cursor, has_more = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(
            page,
            [
                record("r1", "o1", "k1", "v1", {"r1": 1}),
                record("r2", "o2", "k2", "v2", {"r2": 1}),
                record("r3", "o3", "k3", "v3", {"r3": 1}),
            ],
        )
        self.assertEqual(next_cursor, 3)
        self.assertFalse(has_more)

    def test_checkpoint_skips_consumed_records(self) -> None:
        store = StateStore()
        self.fill(store, 4)
        store.save_checkpoint("p1", 2)
        page, next_cursor, has_more = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(
            page,
            [
                record("r3", "o3", "k3", "v3", {"r3": 1}),
                record("r4", "o4", "k4", "v4", {"r4": 1}),
            ],
        )
        self.assertEqual(next_cursor, 2)
        self.assertFalse(has_more)

    def test_paging_walks_the_unconsumed_tail(self) -> None:
        store = StateStore()
        self.fill(store, 5)
        store.save_checkpoint("p1", 1)
        seen = []
        after = 0
        while True:
            page, after, has_more = store.get_peer_operations("p1", after, 2)
            seen.extend(page)
            if not has_more:
                break
        self.assertEqual([entry["operation"]["operationId"] for entry in seen], ["o2", "o3", "o4", "o5"])
        self.assertEqual(after, 4)

    def test_after_equal_to_total_is_an_empty_tail(self) -> None:
        store = StateStore()
        self.fill(store, 3)
        store.save_checkpoint("p1", 1)
        page, next_cursor, has_more = store.get_peer_operations("p1", 2, 100)
        self.assertEqual(page, [])
        self.assertEqual(next_cursor, 2)
        self.assertFalse(has_more)

    def test_after_past_total_raises(self) -> None:
        store = StateStore()
        self.fill(store, 3)
        store.save_checkpoint("p1", 1)
        with self.assertRaises(ValueError):
            store.get_peer_operations("p1", 3, 100)

    def test_checkpoint_at_log_tail_yields_an_empty_stream(self) -> None:
        store = StateStore()
        self.fill(store, 2)
        store.save_checkpoint("p1", 2)
        page, next_cursor, has_more = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(page, [])
        self.assertEqual(next_cursor, 0)
        self.assertFalse(has_more)
        with self.assertRaises(ValueError):
            store.get_peer_operations("p1", 1, 100)

    def test_fetch_does_not_move_the_checkpoint(self) -> None:
        store = StateStore()
        self.fill(store, 3)
        store.save_checkpoint("p1", 1)
        store.get_peer_operations("p1", 0, 100)
        store.get_peer_operations("p1", 1, 1)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 1})
        )
        # A repeated fetch returns the same page.
        first = store.get_peer_operations("p1", 0, 100)
        second = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(first, second)

    def test_stale_writes_imports_and_repairs_are_visible(self) -> None:
        store = StateStore()
        # An ordinary write, then a stale write (its clock is dominated).
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 2}))
        store.apply_operation("r1", operation("o2", "k", "stale", {"r1": 1}))
        # A sync-imported record.
        store.import_operations([("r2", operation("o3", "k", "v2", {"r2": 1}))])
        # A conflict on the key, repaired by an automatic resolution.
        store.apply_operation("r3", operation("o4", "k", "v3", {"r3": 1}))
        status, _, _ = store.apply_auto_resolution(
            "k",
            {
                "replicaId": "r4",
                "operationId": "fix-1",
                "clock": {"r1": 2, "r2": 1, "r3": 1, "r4": 1},
                "policy": "lowest_identity",
            },
        )
        self.assertIs(status, HTTPStatus.CREATED)
        store.save_checkpoint("p1", 0)
        page, _, _ = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(
            [entry["operation"]["operationId"] for entry in page],
            ["o1", "o2", "o3", "o4", "fix-1"],
        )

    def test_replays_conflicts_and_rejected_requests_leave_no_record(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        # Identical replay (200) and conflicting content (409).
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1}))
        # A rejected import batch (one conflicting record) commits nothing.
        status, _, _ = store.import_operations(
            [
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r1", operation("o1", "k", "changed", {"r1": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        store.save_checkpoint("p1", 0)
        page, next_cursor, has_more = store.get_peer_operations("p1", 0, 100)
        self.assertEqual(page, [record("r1", "o1", "k", "v1", {"r1": 1})])
        self.assertEqual((next_cursor, has_more), (1, False))

    def test_peers_page_independently(self) -> None:
        store = StateStore()
        self.fill(store, 4)
        store.save_checkpoint("p1", 0)
        store.save_checkpoint("p2", 3)
        page1, _, _ = store.get_peer_operations("p1", 0, 100)
        page2, _, _ = store.get_peer_operations("p2", 0, 100)
        self.assertEqual(len(page1), 4)
        self.assertEqual(len(page2), 1)


class HttpPeerOperationsTests(unittest.TestCase):
    """HTTP contract against an in-memory server."""

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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, bytes, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        elif isinstance(body, (bytes, str)):
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
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
        return response.status, raw, payload

    def get_page(self, peer: str, query: str = "") -> tuple[int, bytes, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/operations{query}")

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict) -> None:
        status, _, _ = self.request(
            "POST", f"/v1/replicas/{replica}/operations",
            operation(operation_id, key, value, clock),
        )
        self.assertEqual(status, 201)

    def post_checkpoint(self, peer: str, cursor: int) -> None:
        status, _, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        self.assertEqual(status, 200)

    def fill(self, count: int) -> None:
        for index in range(1, count + 1):
            replica = f"r{index}"
            self.post_operation(replica, f"o{index}", f"k{index}", f"v{index}", {replica: 1})

    def test_success_shape_is_compact_newline_terminated_json(self) -> None:
        self.post_operation("r1", "o1", "k", "v", {"r1": 1})
        self.post_checkpoint("p1", 0)
        status, raw, payload = self.get_page("p1")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertEqual(set(payload), {"operations", "nextCursor", "hasMore"})
        self.assertEqual(
            payload,
            {
                "operations": [record("r1", "o1", "k", "v", {"r1": 1})],
                "nextCursor": 1,
                "hasMore": False,
            },
        )

    def test_page_follows_the_checkpoint_in_commit_order(self) -> None:
        self.fill(4)
        self.post_checkpoint("p1", 2)
        status, _, payload = self.get_page("p1")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "operations": [
                    record("r3", "o3", "k3", "v3", {"r3": 1}),
                    record("r4", "o4", "k4", "v4", {"r4": 1}),
                ],
                "nextCursor": 2,
                "hasMore": False,
            },
        )

    def test_paging_resume_with_next_cursor(self) -> None:
        self.fill(5)
        self.post_checkpoint("p1", 0)
        status, _, first = self.get_page("p1", "?limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(first["nextCursor"], 2)
        self.assertTrue(first["hasMore"])
        status, _, second = self.get_page("p1", f"?after={first['nextCursor']}&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(second["nextCursor"], 4)
        self.assertTrue(second["hasMore"])
        status, _, third = self.get_page("p1", f"?after={second['nextCursor']}&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(third["nextCursor"], 5)
        self.assertFalse(third["hasMore"])
        seen = first["operations"] + second["operations"] + third["operations"]
        self.assertEqual(
            [entry["operation"]["operationId"] for entry in seen],
            ["o1", "o2", "o3", "o4", "o5"],
        )

    def test_after_equal_to_total_returns_an_empty_page(self) -> None:
        self.fill(2)
        self.post_checkpoint("p1", 0)
        status, _, payload = self.get_page("p1", "?after=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 2, "hasMore": False})

    def test_unregistered_peer_is_404(self) -> None:
        self.fill(1)
        status, _, payload = self.get_page("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_percent_decoded_peer_id(self) -> None:
        self.post_operation("r1", "o1", "k", "v", {"r1": 1})
        self.post_checkpoint("peer%20one", 0)
        status, _, payload = self.get_page("peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 1)
        # A differently-encoded peer is a different, unregistered peer.
        status, _, payload = self.get_page("peer+one")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_queries_are_400(self) -> None:
        self.fill(2)
        self.post_checkpoint("p1", 0)
        bad_queries = [
            "?after=-1",
            "?after=",
            "?after",
            "?after=1.0",
            "?after=+1",
            "?after=%201",  # whitespace
            "?after=%D9%A1",  # non-ASCII digit (Arabic-Indic ١)
            "?after=3",  # past the unconsumed total of 2
            "?limit=0",
            "?limit=101",
            "?limit=-1",
            "?limit=",
            "?limit=1.5",
            "?limit=%D9%A2",  # non-ASCII digit (Arabic-Indic ٢)
            "?after=1&after=1",
            "?limit=1&limit=2",
            "?cursor=1",
            "?after=0&unknown=1",
            "?=1",
        ]
        for query in bad_queries:
            status, _, payload = self.get_page("p1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # Rejections changed nothing: the stream is still fully available.
        status, _, payload = self.get_page("p1")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 2)

    def test_route_shape_mismatches_are_404(self) -> None:
        self.fill(1)
        self.post_checkpoint("p1", 0)
        for path in (
            "/v1/sync/peers//operations",
            "/v1/sync/peers/p1/operations/",
            "/v1/sync/peers/p1/operations/extra",
            "/v1/sync/peers/p1",
            "/v1/sync/peers",
            "/v1/sync/operations/p1",
        ):
            status, _, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_check_precedes_query_check(self) -> None:
        status, _, payload = self.request("GET", "/v1/sync/peers/p1/operations/extra?after=-1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, _, payload = self.request("GET", "/v1/sync/peers/p1/operations/?limit=0")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_fetch_does_not_change_checkpoint_or_other_state(self) -> None:
        self.fill(3)
        self.post_checkpoint("p1", 1)
        before = {
            "checkpoint": self.request("GET", "/v1/sync/peers/p1/checkpoint")[2],
            "metrics": self.request("GET", "/v1/metrics")[2],
            "sync": self.request("GET", "/v1/sync/operations")[2],
            "audit": self.request("GET", "/v1/audit/keys/k1/operations")[2],
            "digest": self.request("GET", "/v1/verification/digest")[2],
            "snapshot": self.request("GET", "/v1/replication/snapshot")[2],
        }
        self.assertEqual(self.get_page("p1")[0], 200)
        self.assertEqual(self.get_page("p1", "?limit=1")[0], 200)
        self.assertEqual(self.get_page("p1", "?after=9")[0], 400)
        self.assertEqual(self.request("GET", "/v1/sync/peers/p1/checkpoint")[2], before["checkpoint"])
        self.assertEqual(self.request("GET", "/v1/metrics")[2], before["metrics"])
        self.assertEqual(self.request("GET", "/v1/sync/operations")[2], before["sync"])
        self.assertEqual(self.request("GET", "/v1/audit/keys/k1/operations")[2], before["audit"])
        self.assertEqual(self.request("GET", "/v1/verification/digest")[2], before["digest"])
        self.assertEqual(self.request("GET", "/v1/replication/snapshot")[2], before["snapshot"])

    def test_strings_escape_like_the_other_canonical_endpoints(self) -> None:
        self.post_operation("r1", 'o"1', "k\\", "vé\n", {"r1": 1})
        self.post_checkpoint("p1", 0)
        status, raw, payload = self.get_page("p1")
        self.assertEqual(status, 200)
        text = raw.decode("utf-8")
        self.assertIn('\\"', text)
        self.assertIn("\\\\", text)
        self.assertIn("\\u000a", text)
        self.assertIn("é", text)  # non-ASCII code points are written literally
        self.assertEqual(payload["operations"][0]["operation"]["value"], "vé\n")


class PersistentPeerOperationsTests(unittest.TestCase):
    """Recovery keeps pages, resume cursors, and error boundaries identical."""

    def serve(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._running.append((server, thread))
        return server

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"
        self._running: list[tuple[SemanticStateServer, threading.Thread]] = []
        self.addCleanup(self._stop_all)

    def _stop_all(self) -> None:
        for server, thread in self._running:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self._running.clear()

    def stop(self, server: SemanticStateServer) -> None:
        for pair in list(self._running):
            if pair[0] is server:
                self._running.remove(pair)
                server.shutdown()
                server.server_close()
                pair[1].join(timeout=5)
                return
        raise AssertionError("unknown server")

    @staticmethod
    def get(port: int, path: str) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    @staticmethod
    def post(port: int, path: str, body: object) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_pages_survive_a_restart(self) -> None:
        server = self.serve()
        port = server.server_address[1]
        for index in range(1, 5):
            replica = f"r{index}"
            status, _ = self.post(
                port,
                f"/v1/replicas/{replica}/operations",
                operation(f"o{index}", f"k{index}", f"v{index}", {replica: 1}),
            )
            self.assertEqual(status, 201)
        status, _ = self.post(port, "/v1/sync/peers/p1/checkpoint", {"cursor": 2})
        self.assertEqual(status, 200)
        before_first = self.get(port, "/v1/sync/peers/p1/operations?limit=1")
        before_rest = self.get(port, "/v1/sync/peers/p1/operations?after=1")
        before_tail = self.get(port, "/v1/sync/peers/p1/operations?after=2")
        before_out = self.get(port, "/v1/sync/peers/p1/operations?after=3")
        self.stop(server)

        restarted = self.serve()
        port = restarted.server_address[1]
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations?limit=1"), before_first)
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations?after=1"), before_rest)
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations?after=2"), before_tail)
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations?after=3"), before_out)
        self.assertEqual(
            before_tail, (200, {"operations": [], "nextCursor": 2, "hasMore": False})
        )
        self.assertEqual(before_out, (400, {"error": "invalid_request"}))

    def test_fetch_creates_no_temporary_files(self) -> None:
        server = self.serve()
        port = server.server_address[1]
        replica = "r1"
        self.post(port, f"/v1/replicas/{replica}/operations", operation("o1", "k", "v", {"r1": 1}))
        self.post(port, "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        before = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations")[0], 200)
        self.assertEqual(self.get(port, "/v1/sync/peers/p1/operations?after=5")[0], 400)
        after = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(before, after)

    def test_recovered_cursor_beyond_log_length_refuses_startup(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {
                            "replicaId": "r1",
                            "operation": {
                                "operationId": "o1",
                                "key": "k",
                                "value": "v",
                                "clock": {"r1": 1},
                            },
                        }
                    ],
                    "checkpoints": {"p1": 2},
                }
            ),
            encoding="utf-8",
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["--data-file", str(self.data_file)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("startup failed", stderr.getvalue())


class AuthPeerOperationsTests(unittest.TestCase):
    """Bearer authentication applies like every other non-/health route."""

    TOKEN = "peer-fetch-token"

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token=cls.TOKEN
        )
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

    def request(
        self, path: str, authorization: object = "valid"
    ) -> tuple[int, object, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if authorization == "valid":
            headers["Authorization"] = f"Bearer {self.TOKEN}"
        elif authorization is not None:
            headers["Authorization"] = authorization
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        authenticate = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, authenticate

    def register(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/sync/peers/p1/checkpoint",
            body=json.dumps({"cursor": 0}),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.TOKEN}",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 200)

    def test_missing_header_is_401(self) -> None:
        status, payload, authenticate = self.request(
            "/v1/sync/peers/p1/operations", authorization=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(authenticate, "Bearer")

    def test_malformed_and_mismatched_headers_are_401(self) -> None:
        for header in ("Bearer", "bearer " + self.TOKEN, "Bearer wrong", self.TOKEN):
            status, payload, authenticate = self.request(
                "/v1/sync/peers/p1/operations", authorization=header
            )
            self.assertEqual(status, 401, header)
            self.assertEqual(payload, {"error": "unauthorized"}, header)
            self.assertEqual(authenticate, "Bearer", header)

    def test_duplicate_header_is_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/v1/sync/peers/p1/operations")
        conn.putheader("Authorization", f"Bearer {self.TOKEN}")
        conn.putheader("Authorization", f"Bearer {self.TOKEN}")
        conn.endheaders()
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_token_reaches_the_endpoint(self) -> None:
        self.register()
        status, payload, _ = self.request("/v1/sync/peers/p1/operations")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"operations": [], "nextCursor": 0, "hasMore": False})

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("/health", authorization=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
