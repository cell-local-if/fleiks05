"""Tests for conditional sender-side replication checkpoints.

The conditional checkpoint endpoint is::

    POST /v1/sync/peers/{peerId}/checkpoint/conditional
    {"expectedCursor": N, "cursor": M}

The commit only happens when the peer's registered cursor equals
``expectedCursor`` at commit time, so concurrent coordinators and retried
requests can never overwrite newer progress with a stale read. Like the
unconditional checkpoint it is not an operation: it must not change the
accepted log, sync export, the per-key audit, candidate state, the receipt
chain, or the metrics counters, and with ``--data-file`` it shares the
operation commit lock and the atomic-commit protocol.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file_full,
    parse_conditional_checkpoint_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class ParseConditionalCheckpointPayloadTests(unittest.TestCase):
    def test_accepts_valid_pairs(self) -> None:
        for raw, expected in [
            (b'{"expectedCursor":0,"cursor":0}', (0, 0)),
            ('{"cursor": 5, "expectedCursor": 3}', (3, 5)),
            ({"expectedCursor": 12, "cursor": 12}, (12, 12)),
        ]:
            self.assertEqual(parse_conditional_checkpoint_payload(raw), expected)

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_checkpoint_payload(b"{not json")

    def test_rejects_non_object_and_empty_body(self) -> None:
        for raw in (b"", b"[]", b"5", b"null", b'"cursor"'):
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(raw)

    def test_rejects_wrong_keys(self) -> None:
        for raw in (
            {},
            {"cursor": 0},
            {"expectedCursor": 0},
            {"expectedCursor": 0, "cursor": 0, "extra": 1},
            {"expectedCursor": 0, "cursor": 0, "peerId": "p"},
        ):
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(raw)

    def test_rejects_non_integer_values(self) -> None:
        for bad in (True, False, -1, 1.0, "1", None, [], {}):
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(
                    {"expectedCursor": bad, "cursor": 0}
                )
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(
                    {"expectedCursor": 0, "cursor": bad}
                )

    def test_rejects_expected_cursor_above_cursor(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_checkpoint_payload({"expectedCursor": 2, "cursor": 1})


class ConditionalCheckpointStoreTests(unittest.TestCase):
    """Store-level semantics, both in memory and file backed."""

    def test_register_replay_advance_and_conflict(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # An unregistered peer only matches the zero expectation.
        status, error = store.save_checkpoint_conditional("p1", 1, 1)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(store.get_checkpoint("p1")[0], HTTPStatus.NOT_FOUND)
        # First registration with expectedCursor 0.
        status, error = store.save_checkpoint_conditional("p1", 0, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(
            store.get_checkpoint("p1"),
            (HTTPStatus.OK, {"peerId": "p1", "cursor": 0}),
        )
        # Equal-value replay.
        self.assertIs(store.save_checkpoint_conditional("p1", 0, 0)[0], HTTPStatus.OK)
        store.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        # A stale expectation is a conflict and moves nothing.
        status, error = store.save_checkpoint_conditional("p1", 1, 2)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 0)
        # The matching expectation advances.
        status, _ = store.save_checkpoint_conditional("p1", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 2)
        # Peers are independent.
        self.assertEqual(store.get_checkpoint("other")[0], HTTPStatus.NOT_FOUND)

    def test_first_registration_may_advance_directly(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _ = store.save_checkpoint_conditional("p1", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)

    def test_cursor_cannot_pass_accepted_log_length(self) -> None:
        store = StateStore()
        with self.assertRaises(ValueError):
            store.save_checkpoint_conditional("p1", 0, 1)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertIs(store.save_checkpoint_conditional("p1", 0, 1)[0], HTTPStatus.OK)
        with self.assertRaises(ValueError):
            store.save_checkpoint_conditional("p1", 1, 2)

    def test_replay_does_not_persist(self) -> None:
        store = StateStore()
        store.save_checkpoint_conditional("p1", 0, 0)
        # An equal-value replay must not attempt a durable write.
        with patch.object(
            StateStore, "_persist_locked", side_effect=AssertionError("replay must not persist")
        ):
            status, _ = store.save_checkpoint_conditional("p1", 0, 0)
            self.assertIs(status, HTTPStatus.OK)

    def test_conflict_does_not_persist(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint_conditional("p1", 0, 0)
        with patch.object(
            StateStore, "_persist_locked", side_effect=AssertionError("conflict must not persist")
        ):
            self.assertIs(
                store.save_checkpoint_conditional("p1", 1, 1)[0], HTTPStatus.CONFLICT
            )
            self.assertIs(
                store.save_checkpoint_conditional("p2", 1, 1)[0], HTTPStatus.CONFLICT
            )

    def test_conditional_checkpoints_are_not_operations(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics = store.get_metrics()
        page_before, _, _ = store.get_sync_operations(0, 100)
        audit_before, _, _ = store.get_key_operations("k", 0, 100)
        state_before = store.get_state("k")

        store.save_checkpoint_conditional("p1", 0, 0)
        store.save_checkpoint_conditional("p1", 0, 1)
        store.save_checkpoint_conditional("p2", 0, 2)
        # Rejected requests are likewise invisible to everything.
        with self.assertRaises(ValueError):
            store.save_checkpoint_conditional("p3", 0, 3)
        self.assertEqual(store.save_checkpoint_conditional("p1", 0, 1)[0], HTTPStatus.CONFLICT)

        self.assertEqual(store.get_metrics(), before_metrics)
        page_after, next_cursor, has_more = store.get_sync_operations(0, 100)
        self.assertEqual(page_after, page_before)
        self.assertEqual((next_cursor, has_more), (2, False))
        audit_after, _, _ = store.get_key_operations("k", 0, 100)
        self.assertEqual(audit_after, audit_before)
        self.assertEqual(store.get_state("k"), state_before)

    def test_unconditional_checkpoint_semantics_unchanged(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # The unconditional endpoint keeps its own register/replay/advance
        # and rollback-conflict semantics alongside the conditional one.
        self.assertIs(store.save_checkpoint("p1", 0)[0], HTTPStatus.OK)
        self.assertIs(store.save_checkpoint("p1", 1)[0], HTTPStatus.OK)
        self.assertIs(store.save_checkpoint("p1", 0)[0], HTTPStatus.CONFLICT)
        # Both endpoints share the one registration.
        self.assertIs(store.save_checkpoint_conditional("p1", 1, 1)[0], HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)


class PersistentConditionalCheckpointStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def read_checkpoints(self) -> dict[str, int]:
        return load_data_file_full(str(self.data_file))[1]

    def test_registration_is_durable_before_return(self) -> None:
        store = self.make_store()
        status, _ = store.save_checkpoint_conditional("p1", 0, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 0})

    def test_advance_is_durable_and_replay_rewrites_nothing(self) -> None:
        store = self.make_store()
        store.save_checkpoint_conditional("p1", 0, 0)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint_conditional("p1", 0, 1)
        self.assertEqual(self.read_checkpoints(), {"p1": 1})
        before = self.data_file.read_bytes()
        status, _ = store.save_checkpoint_conditional("p1", 1, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.data_file.read_bytes(), before)

    def test_persistence_failure_on_register_leaves_nothing(self) -> None:
        store = self.make_store()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.save_checkpoint_conditional("p1", 0, 0)
        self.assertEqual(store.get_checkpoint("p1")[0], HTTPStatus.NOT_FOUND)
        reloaded = self.make_store()
        self.assertEqual(reloaded.get_checkpoint("p1")[0], HTTPStatus.NOT_FOUND)
        self.assertEqual(store.save_checkpoint_conditional("p1", 0, 0)[0], HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 0})

    def test_persistence_failure_on_advance_keeps_old_cursor(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint_conditional("p1", 0, 1)
        store.apply_operation("r1", operation("o2", "k", "w", {"r1": 2}))
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.save_checkpoint_conditional("p1", 1, 2)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)
        self.assertEqual(self.data_file.read_bytes(), before)
        # The failed advance is retryable and then commits.
        self.assertEqual(store.save_checkpoint_conditional("p1", 1, 2)[0], HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 2})

    def test_restart_preserves_conditional_semantics(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.save_checkpoint_conditional("p1", 0, 2)
        del store

        reloaded = self.make_store()
        self.assertEqual(
            reloaded.get_checkpoint("p1"),
            (HTTPStatus.OK, {"peerId": "p1", "cursor": 2}),
        )
        # Replay stays 200, a stale expectation stays 409.
        self.assertIs(reloaded.save_checkpoint_conditional("p1", 2, 2)[0], HTTPStatus.OK)
        self.assertIs(
            reloaded.save_checkpoint_conditional("p1", 1, 2)[0], HTTPStatus.CONFLICT
        )
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 2)


class HttpConditionalCheckpointTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object, bytes]:
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
        return response.status, payload, raw

    def post_conditional(self, peer: str, body: object) -> tuple[int, object, bytes]:
        return self.request("POST", f"/v1/sync/peers/{peer}/checkpoint/conditional", body)

    def get_checkpoint(self, peer: str) -> tuple[int, object, bytes]:
        return self.request("GET", f"/v1/sync/peers/{peer}/checkpoint")

    def add_operation(self, operation_id: str = "o1") -> None:
        status, _, _ = self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation(operation_id, "k", "v", {"r1": 1}),
        )
        self.assertEqual(status, 201)

    def test_register_replay_advance_flow(self) -> None:
        status, payload, raw = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # The success body is exactly the two fields and one newline.
        self.assertEqual(raw, b'{"peerId":"p1","cursor":0}\n')
        self.assertEqual(
            self.get_checkpoint("p1")[1], {"peerId": "p1", "cursor": 0}
        )
        # Equal replay.
        status, payload, _ = self.post_conditional("p1", {"expectedCursor": 0, "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # Grow the accepted log and advance.
        self.add_operation()
        status, payload, raw = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 1}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 1})
        self.assertEqual(raw, b'{"peerId":"p1","cursor":1}\n')
        self.assertEqual(
            self.get_checkpoint("p1")[1], {"peerId": "p1", "cursor": 1}
        )

    def test_stale_expectation_is_409_and_does_not_move(self) -> None:
        self.add_operation()
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})[0], 200
        )
        status, payload, _ = self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("p1")[1], {"peerId": "p1", "cursor": 1})

    def test_unregistered_peer_with_nonzero_expectation_is_409(self) -> None:
        self.add_operation()
        status, payload, _ = self.post_conditional("p1", {"expectedCursor": 1, "cursor": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_cursor_past_accepted_log_is_400(self) -> None:
        status, payload, _ = self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_invalid_bodies_are_400(self) -> None:
        bad_bodies = [
            b"",
            b"{not json",
            [],
            {},
            {"cursor": 0},
            {"expectedCursor": 0},
            {"expectedCursor": 0, "cursor": 0, "extra": 1},
            {"expectedCursor": -1, "cursor": 0},
            {"expectedCursor": 0, "cursor": -1},
            {"expectedCursor": True, "cursor": 0},
            {"expectedCursor": 0, "cursor": False},
            {"expectedCursor": 0, "cursor": 1.0},
            {"expectedCursor": "0", "cursor": 0},
            {"expectedCursor": 0, "cursor": None},
            {"expectedCursor": 2, "cursor": 1},
        ]
        for body in bad_bodies:
            status, payload, _ = self.post_conditional("p1", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_rejects_query_parameters(self) -> None:
        for query in ("?x=1", "?x=", "?=1", "?cursor=0", "?x=1&x=2"):
            status, payload, _ = self.request(
                "POST",
                f"/v1/sync/peers/p1/checkpoint/conditional{query}",
                {"expectedCursor": 0, "cursor": 0},
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_path_shape_failures_are_404(self) -> None:
        body = {"expectedCursor": 0, "cursor": 0}
        # Empty peer segment, extra segments, missing segments, and a
        # trailing slash are all route-shape failures.
        for path in (
            "/v1/sync/peers//checkpoint/conditional",
            "/v1/sync/peers/p1/checkpoint/conditional/extra",
            "/v1/sync/peers/p1/checkpoint/conditional/",
            "/v1/sync/peers/p1/checkpoint/conditional/extra/",
            "/v1/sync/peers/checkpoint/conditional",
        ):
            status, payload, _ = self.request("POST", path, body)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
        # The shape check takes precedence over a malformed query.
        status, _, _ = self.request(
            "POST", "/v1/sync/peers//checkpoint/conditional?x=1", body
        )
        self.assertEqual(status, 404)
        # GET on the conditional path is not published either.
        self.assertEqual(
            self.request("GET", "/v1/sync/peers/p1/checkpoint/conditional")[0], 404
        )

    def test_peer_id_is_percent_decoded(self) -> None:
        status, payload, _ = self.request(
            "POST",
            "/v1/sync/peers/peer%20one/checkpoint/conditional",
            {"expectedCursor": 0, "cursor": 0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer one", "cursor": 0})
        status, payload, _ = self.request(
            "GET", "/v1/sync/peers/peer%20one/checkpoint"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer one", "cursor": 0})

    def test_conditional_checkpoints_do_not_change_metrics_log_or_audit(self) -> None:
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            "POST",
            "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        metrics_before = self.request("GET", "/v1/metrics")[1]
        sync_before = self.request("GET", "/v1/sync/operations")[1]
        audit_before = self.request("GET", "/v1/audit/keys/k/operations")[1]

        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 0})[0], 200
        )
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 2})[0], 200
        )
        self.assertEqual(
            self.post_conditional("p2", {"expectedCursor": 0, "cursor": 1})[0], 200
        )
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})[0], 409
        )

        self.assertEqual(self.request("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(self.request("GET", "/v1/sync/operations")[1], sync_before)
        self.assertEqual(
            self.request("GET", "/v1/audit/keys/k/operations")[1], audit_before
        )

    def test_unconditional_endpoint_is_unaffected(self) -> None:
        self.add_operation()
        self.assertEqual(
            self.request("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0], 200
        )
        self.assertEqual(
            self.request("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1})[0], 200
        )
        self.assertEqual(
            self.request("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0], 409
        )
        # Both endpoints share the one registration.
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 1, "cursor": 1})[0], 200
        )
        self.assertEqual(self.get_checkpoint("p1")[1], {"peerId": "p1", "cursor": 1})


class PersistentConditionalCheckpointHttpTests(unittest.TestCase):
    """Conditional-checkpoint durability, 503 handling, and recovery."""

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
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        self.last_retry_after = response.getheader("Retry-After")
        conn.close()
        return response.status, payload

    def post_conditional(self, server, peer: str, expected: int, cursor: int):
        return self.request(
            server,
            "POST",
            f"/v1/sync/peers/{peer}/checkpoint/conditional",
            {"expectedCursor": expected, "cursor": cursor},
        )

    def test_file_is_committed_before_200(self) -> None:
        server = self.start_server()
        status, payload = self.post_conditional(server, "p1", 0, 0)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 0})
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_persistence_failure_is_503_retryable_and_leaves_everything(self) -> None:
        server = self.start_server()
        self.assertEqual(self.post_conditional(server, "p1", 0, 0)[0], 200)
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.post_conditional(server, "p1", 0, 1)
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "persistence_unavailable"})
            self.assertEqual(self.last_retry_after, "1")
            status, payload = self.post_conditional(server, "p2", 0, 0)
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "persistence_unavailable"})
            self.assertEqual(self.last_retry_after, "1")
            # A replay needs no write and still succeeds under the fault.
            status, payload = self.post_conditional(server, "p1", 0, 0)
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"peerId": "p1", "cursor": 0})

        # File and visible memory are exactly the pre-failure state.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")[1],
            {"peerId": "p1", "cursor": 0},
        )
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p2/checkpoint")[0], 404
        )
        reloaded = StateStore(data_file=str(self.data_file))
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 0)
        self.assertEqual(reloaded.get_checkpoint("p2")[0], HTTPStatus.NOT_FOUND)
        # Both failed requests commit cleanly once persistence is back.
        self.assertEqual(self.post_conditional(server, "p1", 0, 1)[0], 200)
        self.assertEqual(self.post_conditional(server, "p2", 0, 0)[0], 200)
        self.assertEqual(
            load_data_file_full(str(self.data_file))[1], {"p1": 1, "p2": 0}
        )
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_restart_preserves_checkpoints(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.post_conditional(server, "p1", 0, 1)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 1})
        # Replay 200 and stale-expectation 409 survive the restart.
        self.assertEqual(self.post_conditional(server, "p1", 1, 1)[0], 200)
        self.assertEqual(self.post_conditional(server, "p1", 0, 1)[0], 409)


class ConditionalCheckpointScopePolicyTests(unittest.TestCase):
    """The endpoint requires the write (or admin) scope in policy mode."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.policy_path = cls.tmp / "scopes.json"
        cls.policy_path.write_text(
            json.dumps(
                {
                    "reader-token": ["read"],
                    "writer-token": ["write"],
                    "admin-token": ["read", "write", "admin"],
                }
            ),
            encoding="utf-8",
        )
        from semantic_state_engine.server import load_scope_policy

        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(str(cls.policy_path))),
            scope_policy_file=str(cls.policy_path),
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def post(self, token: str | None) -> tuple[int, object]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/sync/peers/p1/checkpoint/conditional",
            body=json.dumps({"expectedCursor": 0, "cursor": 0}),
            headers=headers,
        )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")) if raw else None

    def test_write_and_admin_scopes_may_commit(self) -> None:
        self.assertEqual(self.post("writer-token")[0], 200)
        self.assertEqual(self.post("admin-token")[0], 200)

    def test_read_scope_is_403(self) -> None:
        status, payload = self.post("reader-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_missing_or_bad_token_is_401(self) -> None:
        status, payload = self.post(None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.post("wrong-token")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})


if __name__ == "__main__":
    unittest.main()
