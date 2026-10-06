"""Tests for conditional sender-side replication checkpoints.

The conditional checkpoint endpoint is::

    POST /v1/sync/peers/{peerId}/checkpoint/conditional
    {"expectedCursor": N, "cursor": M}

The stored cursor advances to ``cursor`` only when the registered cursor
equals ``expectedCursor`` at commit time (or the peer is unregistered and
``expectedCursor`` is 0, which performs the first registration), so a
concurrent coordinator or a retried request holding a stale read can
never overwrite newer progress. Like an unconditional checkpoint it is
not an operation: it never touches the accepted log, sync export,
candidate state, the receipt chain, audit streams, or the metrics
counters, and with ``--data-file`` it shares the atomic-commit protocol.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
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
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file_full,
    load_scope_policy,
    parse_conditional_checkpoint_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class ParseConditionalCheckpointPayloadTests(unittest.TestCase):
    def test_accepts_expected_and_cursor_pairs(self) -> None:
        for raw, expected in [
            (b'{"expectedCursor":0,"cursor":0}', (0, 0)),
            ('{"cursor": 5, "expectedCursor": 2}', (2, 5)),
            ({"expectedCursor": 3, "cursor": 3}, (3, 3)),
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
        for value in (True, False, -1, 1.0, "1", None, [], {}):
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(
                    {"expectedCursor": value, "cursor": 0}
                )
            with self.assertRaises(ValueError):
                parse_conditional_checkpoint_payload(
                    {"expectedCursor": 0, "cursor": value}
                )

    def test_rejects_expected_greater_than_cursor(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_checkpoint_payload({"expectedCursor": 2, "cursor": 1})


class ConditionalCheckpointStoreTests(unittest.TestCase):
    """Store-level compare-and-set semantics, in memory and file backed."""

    def test_first_registration_requires_expected_zero(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, error = store.save_checkpoint_conditional("p1", 1, 1)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.NOT_FOUND, {"error": "not_found"})
        )
        # expectedCursor 0 performs the first registration.
        status, error = store.save_checkpoint_conditional("p1", 0, 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 0})
        )

    def test_first_registration_with_advance(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _ = store.save_checkpoint_conditional("p1", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 1})
        )

    def test_matching_expected_replays_and_advances(self) -> None:
        store = StateStore()
        for i in (1, 2, 3):
            store.apply_operation(f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}))
        store.save_checkpoint_conditional("p1", 0, 1)
        # Equal-value replay: registered cursor equals both expectations.
        status, _ = store.save_checkpoint_conditional("p1", 1, 1)
        self.assertIs(status, HTTPStatus.OK)
        # Advance.
        status, _ = store.save_checkpoint_conditional("p1", 1, 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 3})
        )

    def test_stale_expected_is_conflict_and_never_overwrites(self) -> None:
        store = StateStore()
        for i in (1, 2, 3):
            store.apply_operation(f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}))
        store.save_checkpoint_conditional("p1", 0, 2)
        # A coordinator holding a stale read cannot move the cursor,
        # neither backwards nor forwards.
        status, error = store.save_checkpoint_conditional("p1", 1, 1)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        status, error = store.save_checkpoint_conditional("p1", 0, 3)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        status, error = store.save_checkpoint_conditional("p1", 3, 3)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 2})
        )

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
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("replay must not persist"),
        ):
            status, _ = store.save_checkpoint_conditional("p1", 0, 0)
            self.assertIs(status, HTTPStatus.OK)

    def test_conflict_does_not_persist(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint_conditional("p1", 0, 0)
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("conflict must not persist"),
        ):
            self.assertIs(
                store.save_checkpoint_conditional("p1", 1, 1)[0], HTTPStatus.CONFLICT
            )
            self.assertIs(
                store.save_checkpoint_conditional("other", 1, 1)[0],
                HTTPStatus.CONFLICT,
            )

    def test_conditional_checkpoints_are_not_operations(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics = store.get_metrics()
        page_before, _, _ = store.get_sync_operations(0, 100)
        audit_before, _, _ = store.get_key_operations("k", 0, 100)
        state_before = store.get_state("k")

        store.save_checkpoint_conditional("p1", 0, 1)
        store.save_checkpoint_conditional("p1", 1, 2)
        store.save_checkpoint_conditional("p2", 0, 0)
        # A rejected conditional write is likewise invisible to everything.
        self.assertIs(
            store.save_checkpoint_conditional("p1", 0, 2)[0], HTTPStatus.CONFLICT
        )
        with self.assertRaises(ValueError):
            store.save_checkpoint_conditional("p3", 0, 3)

        self.assertEqual(store.get_metrics(), before_metrics)
        page_after, next_cursor, has_more = store.get_sync_operations(0, 100)
        self.assertEqual(page_after, page_before)
        self.assertEqual((next_cursor, has_more), (2, False))
        audit_after, _, _ = store.get_key_operations("k", 0, 100)
        self.assertEqual(audit_after, audit_before)
        self.assertEqual(store.get_state("k"), state_before)


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
        # Neither memory nor the file gained the checkpoint, and it can be
        # registered again once persistence works.
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
            reloaded.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 2})
        )
        # Stale expectation stays a conflict after restart; replay and
        # advance keep working.
        self.assertIs(
            reloaded.save_checkpoint_conditional("p1", 1, 2)[0], HTTPStatus.CONFLICT
        )
        self.assertIs(reloaded.save_checkpoint_conditional("p1", 2, 2)[0], HTTPStatus.OK)
        self.assertIs(
            reloaded.save_checkpoint_conditional("other", 1, 1)[0], HTTPStatus.CONFLICT
        )


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

    def request(
        self, method: str, path: str, body: object = None
    ) -> tuple[int, object, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        elif isinstance(body, (bytes, str)):
            conn.request(
                method, path, body=body, headers={"Content-Type": "application/json"}
            )
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
        return self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint/conditional", body
        )

    def get_checkpoint(self, peer: str) -> tuple[int, object]:
        status, payload, _ = self.request("GET", f"/v1/sync/peers/{peer}/checkpoint")
        return status, payload

    def test_register_replay_advance_flow(self) -> None:
        status, payload, raw = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # The body is exactly the two fields in order plus one newline.
        self.assertEqual(raw, b'{"peerId":"p1","cursor":0}\n')
        self.assertEqual(self.get_checkpoint("p1"), (200, {"peerId": "p1", "cursor": 0}))
        # Equal-value replay.
        status, payload, raw = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"peerId":"p1","cursor":0}\n')
        # Grow the accepted log and advance.
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        status, payload, raw = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 1}
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"peerId":"p1","cursor":1}\n')
        self.assertEqual(self.get_checkpoint("p1"), (200, {"peerId": "p1", "cursor": 1}))

    def test_stale_expected_is_409_and_does_not_move(self) -> None:
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})[0], 200
        )
        status, payload, _ = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 1}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        status, payload, _ = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 0}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("p1"), (200, {"peerId": "p1", "cursor": 1}))

    def test_unregistered_peer_with_nonzero_expected_is_409(self) -> None:
        for i in (1, 2, 3):
            self.request(
                "POST",
                "/v1/replicas/r1/operations",
                operation(f"o{i}", "k", f"v{i}", {"r1": i}),
            )
        status, payload, _ = self.post_conditional(
            "ghost", {"expectedCursor": 3, "cursor": 3}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("ghost")[0], 404)

    def test_cursor_past_accepted_log_is_400(self) -> None:
        status, payload, _ = self.post_conditional(
            "p1", {"expectedCursor": 0, "cursor": 1}
        )
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

    def test_empty_peer_segment_is_404(self) -> None:
        status, payload, _ = self.request(
            "POST",
            "/v1/sync/peers//checkpoint/conditional",
            {"expectedCursor": 0, "cursor": 0},
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_extra_segments_and_trailing_slash_are_404(self) -> None:
        body = {"expectedCursor": 0, "cursor": 0}
        for path in (
            "/v1/sync/peers/p1/checkpoint/conditional/extra",
            "/v1/sync/peers/p1/checkpoint/conditional/",
            "/v1/sync/peers/p1",
            "/v1/sync/peers",
        ):
            status, payload, _ = self.request("POST", path, body)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_query_parameters_are_400(self) -> None:
        for query in ("?x=1", "?x=", "?=1", "?cursor=0", "?x=1&x=2"):
            status, payload, _ = self.request(
                "POST",
                f"/v1/sync/peers/p1/checkpoint/conditional{query}",
                {"expectedCursor": 0, "cursor": 0},
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A rejected query registers nothing.
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_path_shape_is_checked_before_query(self) -> None:
        # An empty peer id stays 404 even when the query is malformed.
        status, payload, _ = self.request(
            "POST",
            "/v1/sync/peers//checkpoint/conditional?x=1",
            {"expectedCursor": 0, "cursor": 0},
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_peer_id_is_percent_decoded(self) -> None:
        status, payload, raw = self.request(
            "POST",
            "/v1/sync/peers/peer%20one/checkpoint/conditional",
            {"expectedCursor": 0, "cursor": 0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"peerId":"peer one","cursor":0}\n')
        self.assertEqual(
            self.get_checkpoint("peer%20one"), (200, {"peerId": "peer one", "cursor": 0})
        )

    def test_get_on_conditional_path_is_404(self) -> None:
        status, payload, _ = self.request(
            "GET", "/v1/sync/peers/p1/checkpoint/conditional"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unconditional_checkpoint_semantics_unchanged(self) -> None:
        # The plain POST keeps its own registration, replay, advance, and
        # conflict behavior alongside the conditional route.
        status, payload, _ = self.request(
            "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        # The conditional route observes the unconditionally registered
        # cursor, and vice versa.
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})[0], 200
        )
        status, payload, _ = self.request(
            "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("p1"), (200, {"peerId": "p1", "cursor": 1}))

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
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 1})[0], 200
        )
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 1, "cursor": 2})[0], 200
        )
        self.assertEqual(
            self.post_conditional("p1", {"expectedCursor": 0, "cursor": 2})[0], 409
        )

        self.assertEqual(self.request("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(self.request("GET", "/v1/sync/operations")[1], sync_before)
        self.assertEqual(
            self.request("GET", "/v1/audit/keys/k/operations")[1], audit_before
        )


class PersistentConditionalCheckpointHttpTests(unittest.TestCase):
    """Conditional-checkpoint durability and 503 handling over real HTTP."""

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
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
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

    def test_registration_and_advance_are_committed_before_200(self) -> None:
        server = self.start_server()
        status, payload = self.post_conditional(server, "p1", 0, 0)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 0})
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        status, _ = self.post_conditional(server, "p1", 0, 1)
        self.assertEqual(status, 200)
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 1})
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
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            # A real advance fails with 503 and Retry-After: 1.
            status, payload = self.post_conditional(server, "p1", 0, 1)
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "persistence_unavailable"})
            self.assertEqual(self.last_retry_after, "1")
            # A first registration fails the same way.
            status, payload = self.post_conditional(server, "p2", 0, 0)
            self.assertEqual(status, 503)
            self.assertEqual(payload, {"error": "persistence_unavailable"})
            self.assertEqual(self.last_retry_after, "1")
            # An idempotent replay writes nothing and still succeeds.
            status, payload = self.post_conditional(server, "p1", 0, 0)
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"peerId": "p1", "cursor": 0})

        # Memory and the file are exactly as before the failed requests.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint"),
            (200, {"peerId": "p1", "cursor": 0}),
        )
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p2/checkpoint")[0], 404
        )
        # Both failed requests are retryable once persistence recovers.
        self.assertEqual(self.post_conditional(server, "p1", 0, 1)[0], 200)
        self.assertEqual(self.post_conditional(server, "p2", 0, 0)[0], 200)
        self.assertEqual(
            load_data_file_full(str(self.data_file))[1], {"p1": 1, "p2": 0}
        )

    def test_restart_preserves_conditional_flow(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.assertEqual(self.post_conditional(server, "p1", 0, 1)[0], 200)
        server.shutdown()
        server.server_close()

        restarted = self.start_server()
        self.assertEqual(
            self.request(restarted, "GET", "/v1/sync/peers/p1/checkpoint"),
            (200, {"peerId": "p1", "cursor": 1}),
        )
        self.assertEqual(self.post_conditional(restarted, "p1", 0, 1)[0], 409)
        self.assertEqual(self.post_conditional(restarted, "p1", 1, 1)[0], 200)


class ConditionalCheckpointAuthTests(unittest.TestCase):
    """Scope-policy mode requires the write (or admin) scope."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-cond-checkpoint-auth-")
        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reader": ["read"],
                    "writer": ["write"],
                    "admin": ["read", "write", "admin"],
                },
                handle,
            )
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def request(self, path: str, body: object, auth: str | None) -> tuple[int, object]:
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_conditional(self, peer: str, auth: str | None) -> tuple[int, object]:
        return self.request(
            f"/v1/sync/peers/{peer}/checkpoint/conditional",
            {"expectedCursor": 0, "cursor": 0},
            auth,
        )

    def test_missing_or_bad_token_is_401(self) -> None:
        status, payload = self.post_conditional("p1", None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.post_conditional("p1", "Bearer wrong")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        # A rejected request registers nothing.
        status, payload = self.request(
            "/v1/sync/peers/p1/checkpoint/conditional",
            {"expectedCursor": 0, "cursor": 0},
            "Bearer reader",
        )
        self.assertEqual(status, 403)

    def test_read_scope_is_403(self) -> None:
        status, payload = self.post_conditional("p1", "Bearer reader")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_write_and_admin_scopes_are_accepted(self) -> None:
        status, payload = self.post_conditional("p1", "Bearer writer")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        status, payload = self.post_conditional("p2", "Bearer admin")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p2", "cursor": 0})


if __name__ == "__main__":
    unittest.main()
