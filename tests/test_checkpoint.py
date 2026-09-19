"""Tests for sender-side replication checkpoints.

The checkpoint endpoints are::

    POST /v1/sync/peers/{peerId}/checkpoint   {"cursor": N}
    GET  /v1/sync/peers/{peerId}/checkpoint

Checkpoints record a sending peer's consumption progress into the
accepted-operation log. They are not operations: they must not change the
accepted log, sync export, the per-key audit, candidate state, or the six
metrics counters. With ``--data-file`` they share the operation commit
lock and the atomic-commit protocol.

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
    parse_checkpoint_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class ParseCheckpointPayloadTests(unittest.TestCase):
    def test_accepts_non_negative_integers(self) -> None:
        for raw, expected in [
            (b'{"cursor":0}', 0),
            ('{"cursor": 5}', 5),
            ({"cursor": 12}, 12),
        ]:
            self.assertEqual(parse_checkpoint_payload(raw), expected)

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_checkpoint_payload(b"{not json")

    def test_rejects_non_object_and_empty_body(self) -> None:
        for raw in (b"", b"[]", b"5", b"null", b'"cursor"'):
            with self.assertRaises(ValueError):
                parse_checkpoint_payload(raw)

    def test_rejects_wrong_keys(self) -> None:
        for raw in (
            {},
            {"cursor": 0, "extra": 1},
            {"peerId": "p", "cursor": 0},
        ):
            with self.assertRaises(ValueError):
                parse_checkpoint_payload(raw)

    def test_rejects_non_integer_cursors(self) -> None:
        for cursor in (True, False, -1, 1.0, "1", None, [], {}):
            with self.assertRaises(ValueError):
                parse_checkpoint_payload({"cursor": cursor})


class CheckpointStoreTests(unittest.TestCase):
    """Store-level semantics, both in memory and file backed."""

    def test_register_replay_advance_and_get(self) -> None:
        store = StateStore()
        # First registration, even at cursor 0, is a stored object.
        status, error = store.save_checkpoint("p1", 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 0}))
        # Equal-value replay.
        status, _ = store.save_checkpoint("p1", 0)
        self.assertIs(status, HTTPStatus.OK)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "x", {"r3": 1}))
        # Advance.
        status, _ = store.save_checkpoint("p1", 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 3}))
        # Peers are independent.
        self.assertEqual(store.get_checkpoint("other"), (HTTPStatus.NOT_FOUND, {"error": "not_found"}))

    def test_stored_cursor_never_moves_backwards(self) -> None:
        store = StateStore()
        for i in (1, 2, 3):
            store.apply_operation(f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}))
        store.save_checkpoint("p1", 3)
        status, error = store.save_checkpoint("p1", 2)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 3}))
        # Zero must not clobber an advanced cursor either.
        status, _ = store.save_checkpoint("p1", 0)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 3)

    def test_cursor_cannot_pass_accepted_log_length(self) -> None:
        store = StateStore()
        with self.assertRaises(ValueError):
            store.save_checkpoint("p1", 1)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _ = store.save_checkpoint("p1", 1)
        self.assertIs(status, HTTPStatus.OK)
        with self.assertRaises(ValueError):
            store.save_checkpoint("p1", 2)

    def test_replay_does_not_persist(self) -> None:
        store = StateStore()
        store.save_checkpoint("p1", 0)
        # An equal-value replay must not attempt a durable write.
        with patch.object(
            StateStore, "_persist_locked", side_effect=AssertionError("replay must not persist")
        ):
            status, _ = store.save_checkpoint("p1", 0)
            self.assertIs(status, HTTPStatus.OK)

    def test_checkpoints_are_not_operations(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics = store.get_metrics()
        page_before, _, _ = store.get_sync_operations(0, 100)
        audit_before, _, _ = store.get_key_operations("k", 0, 100)
        state_before = store.get_state("k")

        store.save_checkpoint("p1", 0)
        store.save_checkpoint("p1", 1)
        store.save_checkpoint("p2", 2)
        # A rejected rollback is likewise invisible to everything.
        with self.assertRaises(ValueError):
            store.save_checkpoint("p3", 3)
        self.assertEqual(store.save_checkpoint("p1", 0)[0], HTTPStatus.CONFLICT)

        self.assertEqual(store.get_metrics(), before_metrics)
        page_after, next_cursor, has_more = store.get_sync_operations(0, 100)
        self.assertEqual(page_after, page_before)
        self.assertEqual((next_cursor, has_more), (2, False))
        audit_after, _, _ = store.get_key_operations("k", 0, 100)
        self.assertEqual(audit_after, audit_before)
        self.assertEqual(store.get_state("k"), state_before)


class PersistentCheckpointStoreTests(unittest.TestCase):
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
        status, _ = store.save_checkpoint("p1", 0)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 0})

    def test_advance_is_durable_and_replay_rewrites_nothing(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint("p1", 1)
        self.assertEqual(self.read_checkpoints(), {"p1": 1})
        before = self.data_file.read_bytes()
        status, _ = store.save_checkpoint("p1", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.data_file.read_bytes(), before)

    def test_persistence_failure_on_register_leaves_nothing(self) -> None:
        store = self.make_store()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.save_checkpoint("p1", 0)
        # Neither memory nor the file gained the checkpoint, and it can be
        # registered again once persistence works.
        self.assertEqual(store.get_checkpoint("p1")[0], HTTPStatus.NOT_FOUND)
        reloaded = self.make_store()
        self.assertEqual(reloaded.get_checkpoint("p1")[0], HTTPStatus.NOT_FOUND)
        self.assertEqual(store.save_checkpoint("p1", 0)[0], HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 0})

    def test_persistence_failure_on_advance_keeps_old_cursor(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint("p1", 1)
        store.apply_operation("r1", operation("o2", "k", "w", {"r1": 2}))
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.save_checkpoint("p1", 2)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)
        self.assertEqual(self.data_file.read_bytes(), before)
        # The failed advance is retryable and then commits.
        self.assertEqual(store.save_checkpoint("p1", 2)[0], HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 2})

    def test_conflict_does_not_persist(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=AssertionError("conflict must not persist")
        ):
            self.assertIs(store.save_checkpoint("p1", 0)[0], HTTPStatus.OK)  # replay
            self.assertIs(store.save_checkpoint("p1", 0)[0], HTTPStatus.OK)
        self.assertEqual(self.data_file.read_bytes(), before)

    def test_checkpoints_share_the_file_with_operations(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.save_checkpoint("p1", 1)
        store.save_checkpoint("p2", 0)
        records, checkpoints = load_data_file_full(str(self.data_file))
        self.assertEqual([r[0] for r in records], ["r1"])
        self.assertEqual(checkpoints, {"p1": 1, "p2": 0})

    def test_old_version1_file_recovers_without_checkpoints(self) -> None:
        # A file written before checkpoints existed has no checkpoints key.
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = self.make_store()
        self.assertEqual(store.get_checkpoint("p1"), (HTTPStatus.NOT_FOUND, {"error": "not_found"}))
        # Existing semantics survive unchanged.
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1})),
            HTTPStatus.OK,
        )
        # Registering a checkpoint upgrades the file to the supplemented
        # format on its next commit.
        self.assertIs(store.save_checkpoint("p1", 1)[0], HTTPStatus.OK)
        self.assertEqual(self.read_checkpoints(), {"p1": 1})

    def test_restart_preserves_checkpoints_and_semantics(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.save_checkpoint("p1", 2)
        store.save_checkpoint("p2", 0)
        # A durable 2-commit checkpoint.
        del store

        reloaded = self.make_store()
        self.assertEqual(
            reloaded.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 2})
        )
        self.assertEqual(
            reloaded.get_checkpoint("p2"), (HTTPStatus.OK, {"peerId": "p2", "cursor": 0})
        )
        # Rollback stays rejected after restart; replay stays 200.
        self.assertIs(reloaded.save_checkpoint("p1", 1)[0], HTTPStatus.CONFLICT)
        self.assertIs(reloaded.save_checkpoint("p2", 0)[0], HTTPStatus.OK)
        # Operations and their semantics were recovered too.
        self.assertEqual(reloaded.get_metrics()["acceptedOperations"], 2)
        self.assertEqual(reloaded.get_state("k")[1]["status"], "conflict")

    def test_corrupt_checkpoint_sections_are_rejected(self) -> None:
        def reject(section_raw: str) -> None:
            self.data_file.write_text(
                f'{{"version":1,"operations":[],{section_raw}}}',
                encoding="utf-8",
            )
            with self.assertRaises(PersistenceError):
                self.make_store()

        reject('"checkpoints":[]')
        reject('"checkpoints":{"":0}')
        reject('"checkpoints":{"p":-1}')
        reject('"checkpoints":{"p":true}')
        reject('"checkpoints":{"p":"1"}')
        reject('"checkpoints":{"p":1.5}')
        # A cursor naming an unaccepted record is a corrupt file.
        self.data_file.write_text(
            '{"version":1,"operations":[],'
            '"checkpoints":{"p":1}}',
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()

    def test_unknown_root_key_is_still_rejected(self) -> None:
        self.data_file.write_text(
            '{"version":1,"operations":[],"checkpoints":{},"extra":1}',
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()


class HttpCheckpointTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
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
        return response.status, payload

    def post_checkpoint(self, peer: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/sync/peers/{peer}/checkpoint", body)

    def get_checkpoint(self, peer: str, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/checkpoint{query}")

    def test_get_unknown_peer_is_404(self) -> None:
        status, payload = self.get_checkpoint("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_register_replay_advance_flow(self) -> None:
        status, payload = self.post_checkpoint("p1", {"cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        status, payload = self.get_checkpoint("p1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # Equal replay.
        status, payload = self.post_checkpoint("p1", {"cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # Grow the accepted log and advance.
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        status, payload = self.post_checkpoint("p1", {"cursor": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 1})
        status, payload = self.get_checkpoint("p1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 1})

    def test_rollback_is_409_and_does_not_move(self) -> None:
        self.assertEqual(self.post_checkpoint("p1", {"cursor": 0})[0], 200)
        self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        self.assertEqual(self.post_checkpoint("p1", {"cursor": 1})[0], 200)
        status, payload = self.post_checkpoint("p1", {"cursor": 0})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(self.get_checkpoint("p1")[1], {"peerId": "p1", "cursor": 1})

    def test_cursor_past_accepted_log_is_400(self) -> None:
        status, payload = self.post_checkpoint("p1", {"cursor": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_invalid_bodies_are_400(self) -> None:
        bad_bodies = [
            b"",
            b"{not json",
            [],
            {},
            {"extra": 1},
            {"cursor": 0, "extra": 1},
            {"cursor": -1},
            {"cursor": True},
            {"cursor": False},
            {"cursor": 1.0},
            {"cursor": "1"},
            {"cursor": None},
        ]
        for body in bad_bodies:
            status, payload = self.post_checkpoint("p1", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        self.assertEqual(self.get_checkpoint("p1")[0], 404)

    def test_empty_peer_segment_is_400(self) -> None:
        for method, body in (("GET", None), ("POST", {"cursor": 0})):
            status, payload = self.request(method, "/v1/sync/peers//checkpoint", body)
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})

    def test_extra_path_segments_are_404(self) -> None:
        for method, body in (("GET", None), ("POST", {"cursor": 0})):
            status, payload = self.request(
                method, "/v1/sync/peers/p1/checkpoint/extra", body
            )
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
        # Missing the checkpoint segment or peer is also 404.
        for path in ("/v1/sync/peers", "/v1/sync/peers/p1"):
            self.assertEqual(self.request("GET", path)[0], 404)
            self.assertEqual(self.request("POST", path, {"cursor": 0})[0], 404)

    def test_get_rejects_query_parameters(self) -> None:
        for query in ("?x=1", "?x=", "?=1", "?cursor=0", "?x=1&x=2"):
            status, payload = self.get_checkpoint("p1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_peer_id_is_percent_decoded_and_distinct(self) -> None:
        status, payload = self.request(
            "POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer one", "cursor": 0})
        status, payload = self.request("GET", "/v1/sync/peers/peer%20one/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer one", "cursor": 0})

    def test_checkpoints_do_not_change_metrics_log_or_audit(self) -> None:
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

        self.assertEqual(self.post_checkpoint("p1", {"cursor": 0})[0], 200)
        self.assertEqual(self.post_checkpoint("p1", {"cursor": 2})[0], 200)
        self.assertEqual(self.post_checkpoint("p2", {"cursor": 1})[0], 200)
        self.assertEqual(self.post_checkpoint("p1", {"cursor": 0})[0], 409)

        self.assertEqual(self.request("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(self.request("GET", "/v1/sync/operations")[1], sync_before)
        self.assertEqual(self.request("GET", "/v1/audit/keys/k/operations")[1], audit_before)


class PersistentCheckpointHttpTests(unittest.TestCase):
    """Checkpoint durability, 500 handling, and recovery over real HTTP."""

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
        conn.close()
        return response.status, payload

    def test_checkpoint_file_is_committed_before_200(self) -> None:
        server = self.start_server()
        status, payload = self.request(
            server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        # The 200 already has the checkpoint durably on disk.
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 0})
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_persistence_failure_is_500_retryable_and_leaves_everything(self) -> None:
        server = self.start_server()
        # Seed a checkpoint and an operation so both register and advance
        # failure paths are observable.
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0],
            200,
        )
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
            status, payload = self.request(
                server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1}
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            status, payload = self.request(
                server, "POST", "/v1/sync/peers/p2/checkpoint", {"cursor": 0}
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # A replay needs no write and still succeeds under the fault.
            status, payload = self.request(
                server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0}
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"peerId": "p1", "cursor": 0})

        # File and visible memory are exactly the pre-failure state.
        self.assertEqual(self.data_file.read_bytes(), before)
        status, payload = self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")
        self.assertEqual(payload, {"peerId": "p1", "cursor": 0})
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p2/checkpoint")[0], 404
        )
        reloaded = StateStore(data_file=str(self.data_file))
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 0)
        self.assertEqual(reloaded.get_checkpoint("p2")[0], HTTPStatus.NOT_FOUND)
        # Both failed requests commit cleanly once persistence is back.
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1})[0],
            200,
        )
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p2/checkpoint", {"cursor": 0})[0],
            200,
        )
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
        self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1})
        self.request(server, "POST", "/v1/sync/peers/p2/checkpoint", {"cursor": 0})
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 1})
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p2/checkpoint")[1],
            {"peerId": "p2", "cursor": 0},
        )
        # Replay 200, rollback 409, and the recovered log still bound cursors.
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1})[0],
            200,
        )
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0],
            409,
        )
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p2/checkpoint", {"cursor": 2})[0],
            400,
        )
        # The operation itself was recovered as well.
        metrics = self.request(server, "GET", "/v1/metrics")[1]
        self.assertEqual(metrics["acceptedOperations"], 1)

    def test_old_version1_file_is_recovered_without_checkpoints(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        status, payload = self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # Operations from the old file are intact.
        sync = self.request(server, "GET", "/v1/sync/operations")[1]
        self.assertEqual(len(sync["operations"]), 1)
        # Registering persists the supplemented format, which re-recovers.
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 1})[0],
            200,
        )
        server.shutdown()
        server.server_close()
        server = self.start_server()
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")[1],
            {"peerId": "p1", "cursor": 1},
        )


class CheckpointConcurrencyTests(unittest.TestCase):
    """Checkpoint commits share the operation commit lock."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=None
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

    def post(self, peer: str, cursor: int) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            f"/v1/sync/peers/{peer}/checkpoint",
            body=json.dumps({"cursor": cursor}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()
        return response.status

    def get(self, peer: str) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", f"/v1/sync/peers/{peer}/checkpoint")
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        return payload["cursor"]

    def test_concurrent_registrations_against_empty_log(self) -> None:
        peer_count = 8
        rounds = 20
        errors: list[BaseException] = []
        results: list[list[int]] = [[] for _ in range(peer_count)]

        def worker(peer_index: int) -> None:
            # A small fixed number of threads (each sequential) keeps the
            # HTTP accept queue comfortably below its limit; requests from
            # the threads still interleave on the server.
            local: list[int] = []
            try:
                for _ in range(rounds):
                    # Cursor 0 is always valid on the empty log: 200 for
                    # the first registration and for every replay. Cursor 1
                    # is always invalid there: 400, never a 5xx or a 409.
                    local.append(self.post(f"p{peer_index}", 0))
                    local.append(self.post(f"p{peer_index}", 1))
                results[peer_index] = local
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(peer_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])
        statuses = [status for local in results for status in local]
        self.assertTrue(statuses)
        self.assertEqual(set(statuses), {200, 400})
        for peer_index in range(peer_count):
            self.assertEqual(self.get(f"p{peer_index}"), 0)


class CheckpointConcurrencyWithLogTests(unittest.TestCase):
    """Advances against a populated log serialize to the maximum cursor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.data_file = cls.tmp / "state.json"
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(cls.data_file)
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

    def post(self, peer: str, cursor: int) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            f"/v1/sync/peers/{peer}/checkpoint",
            body=json.dumps({"cursor": cursor}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()
        return response.status

    def get(self, peer: str) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", f"/v1/sync/peers/{peer}/checkpoint")
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        return payload["cursor"]

    def test_concurrent_mixed_writes_and_checkpoint_advances_commit_cleanly(self) -> None:
        size = 30

        def seed() -> None:
            for i in range(size):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                conn.request(
                    "POST",
                    f"/v1/replicas/r{i}/operations",
                    body=json.dumps(operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})),
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                assert response.status == 201
                response.read()
                conn.close()

        seed_thread = threading.Thread(target=seed)
        errors: list[BaseException] = []

        def advance(peer: str) -> None:
            try:
                for cursor in range(size + 1):
                    status = self.post(peer, cursor)
                    # 200 registers/replays/advances; 400 means the log had
                    # not yet reached that cursor; 409 means the cursor had
                    # already advanced past it. None of these is an error.
                    assert status in (200, 400, 409), (peer, cursor, status)
            except BaseException as exc:  # reported below
                errors.append(exc)

        advancers = [
            threading.Thread(target=advance, args=(f"peer-{i}",)) for i in range(6)
        ]
        seed_thread.start()
        for thread in advancers:
            thread.start()
        seed_thread.join(timeout=30)
        for thread in advancers:
            thread.join(timeout=30)
        self.assertEqual(errors, [])

        # Once the whole log is committed, every peer confirms the full
        # length: it advances if the racing loop stopped early, and replays
        # if it already reached the end. Nothing ever overshot.
        for i in range(6):
            status = self.post(f"peer-{i}", size)
            self.assertEqual(status, 200)
            self.assertEqual(self.get(f"peer-{i}"), size)
        reloaded = StateStore(data_file=str(self.data_file))
        for i in range(6):
            self.assertEqual(
                reloaded.get_checkpoint(f"peer-{i}"),
                (HTTPStatus.OK, {"peerId": f"peer-{i}", "cursor": size}),
            )
        self.assertEqual(reloaded.get_metrics()["acceptedOperations"], size)


if __name__ == "__main__":
    unittest.main()
