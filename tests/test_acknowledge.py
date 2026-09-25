"""Tests for verifiable consumption receipts (acknowledgements).

The acknowledge endpoint is::

    POST /v1/sync/peers/{peerId}/acknowledge
    {"ackId": "...", "cursor": N, "operations": [{"replicaId","operationId"}, ...]}

A receipt confirms, segment by segment, exactly which accepted records a
sending peer consumed: starting from the peer's registered checkpoint,
``operations`` must exactly cover the contiguous accepted records up to
``cursor``, and the checkpoint advances with the confirmation. A receipt
is not an operation: it must not change the accepted log, sync export,
the per-key audit, candidate state, or the six metrics counters. With
``--data-file`` the receipt, the checkpoint advance, and the
``(peerId, ackId)`` binding commit together under the shared commit lock
and the atomic-commit protocol.

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
    load_data_file_acks,
    load_data_file_full,
    parse_acknowledge_payload,
)

TOKEN = "s3cret-token_123"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identities(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"replicaId": replica, "operationId": op} for replica, op in pairs]


class ParseAcknowledgePayloadTests(unittest.TestCase):
    def test_accepts_valid_payloads(self) -> None:
        ack_id, cursor, operations = parse_acknowledge_payload(
            b'{"ackId":"a1","cursor":2,"operations":'
            b'[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r2","operationId":"o2"}]}'
        )
        self.assertEqual(ack_id, "a1")
        self.assertEqual(cursor, 2)
        self.assertEqual(operations, identities(("r1", "o1"), ("r2", "o2")))
        # An empty segment is a valid payload.
        self.assertEqual(
            parse_acknowledge_payload({"ackId": "a", "cursor": 0, "operations": []}),
            ("a", 0, []),
        )

    def test_rejects_malformed_json_and_non_objects(self) -> None:
        for raw in (b"", b"{not json", b"[]", b"5", b"null", b'"ack"'):
            with self.assertRaises(ValueError):
                parse_acknowledge_payload(raw)

    def test_rejects_missing_and_unknown_keys(self) -> None:
        valid = {"ackId": "a", "cursor": 0, "operations": []}
        for key in ("ackId", "cursor", "operations"):
            body = {k: v for k, v in valid.items() if k != key}
            with self.assertRaises(ValueError):
                parse_acknowledge_payload(body)
        with self.assertRaises(ValueError):
            parse_acknowledge_payload({**valid, "extra": 1})

    def test_rejects_invalid_ack_id(self) -> None:
        for ack_id in ("", 1, None, True, [], {}):
            with self.assertRaises(ValueError):
                parse_acknowledge_payload(
                    {"ackId": ack_id, "cursor": 0, "operations": []}
                )

    def test_rejects_invalid_cursor(self) -> None:
        for cursor in (True, False, -1, 1.0, "1", None, [], {}):
            with self.assertRaises(ValueError):
                parse_acknowledge_payload(
                    {"ackId": "a", "cursor": cursor, "operations": []}
                )

    def test_rejects_invalid_operations_shape(self) -> None:
        for operations in (None, {}, "x", 1, True):
            with self.assertRaises(ValueError):
                parse_acknowledge_payload(
                    {"ackId": "a", "cursor": 0, "operations": operations}
                )
        # Entries must carry exactly replicaId and operationId.
        for entry in (
            {},
            {"replicaId": "r1"},
            {"operationId": "o1"},
            {"replicaId": "r1", "operationId": "o1", "extra": 1},
            {"replicaId": "", "operationId": "o1"},
            {"replicaId": "r1", "operationId": ""},
            {"replicaId": 1, "operationId": "o1"},
            {"replicaId": "r1", "operationId": None},
        ):
            with self.assertRaises(ValueError, msg=repr(entry)):
                parse_acknowledge_payload(
                    {"ackId": "a", "cursor": 1, "operations": [entry]}
                )

    def test_rejects_duplicate_identities(self) -> None:
        with self.assertRaises(ValueError):
            parse_acknowledge_payload(
                {
                    "ackId": "a",
                    "cursor": 2,
                    "operations": [
                        {"replicaId": "r1", "operationId": "o1"},
                        {"replicaId": "r1", "operationId": "o1"},
                    ],
                }
            )

    def test_rejects_segments_longer_than_one_hundred(self) -> None:
        operations = [
            {"replicaId": "r", "operationId": f"o{i}"} for i in range(101)
        ]
        with self.assertRaises(ValueError):
            parse_acknowledge_payload(
                {"ackId": "a", "cursor": 101, "operations": operations}
            )
        # Exactly one hundred is accepted.
        ack_id, cursor, parsed = parse_acknowledge_payload(
            {"ackId": "a", "cursor": 100, "operations": operations[:100]}
        )
        self.assertEqual(len(parsed), 100)


class AcknowledgeStoreTests(unittest.TestCase):
    """Store-level semantics, in memory."""

    def make_store(self) -> StateStore:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 2, "r2": 1}))
        return store

    def test_unregistered_peer_is_not_found(self) -> None:
        store = StateStore()
        status, error = store.acknowledge_operations("p1", "a1", 0, [])
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(error, "not_found")

    def test_create_advances_checkpoint_and_replays(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 1)
        status, error = store.acknowledge_operations(
            "p1", "a1", 3, identities(("r2", "o2"), ("r1", "o3"))
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(
            store.get_checkpoint("p1"), (HTTPStatus.OK, {"peerId": "p1", "cursor": 3})
        )
        # Identical replay: 200, no state change.
        status, error = store.acknowledge_operations(
            "p1", "a1", 3, identities(("r2", "o2"), ("r1", "o3"))
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        # Other peers are independent, and other ackIds are independent.
        self.assertEqual(store.get_checkpoint("p2")[0], HTTPStatus.NOT_FOUND)
        status, _ = store.acknowledge_operations("p1", "a2", 3, [])
        self.assertIs(status, HTTPStatus.CREATED)

    def test_empty_segment_at_current_checkpoint(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 3)
        status, _ = store.acknowledge_operations("p1", "a1", 3, [])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 3)

    def test_binding_change_is_operation_conflict(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        self.assertIs(
            store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.CREATED,
        )
        # Same ackId, different cursor.
        status, error = store.acknowledge_operations("p1", "a1", 2, identities(("r1", "o1")))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")
        # Same ackId, different operations.
        status, error = store.acknowledge_operations(
            "p1", "a1", 1, identities(("r2", "o2"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")
        # State is unchanged: the original binding still replays.
        self.assertIs(
            store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.OK,
        )
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)

    def test_cursor_below_checkpoint_is_checkpoint_conflict(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 2)
        status, error = store.acknowledge_operations(
            "p1", "a1", 1, identities(("r1", "o1"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 2)

    def test_log_mismatch_is_ack_conflict(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        # Wrong identity.
        status, error = store.acknowledge_operations(
            "p1", "a1", 1, identities(("r2", "o2"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # Wrong record count (segment longer than the identity list).
        status, error = store.acknowledge_operations(
            "p1", "a2", 2, identities(("r1", "o1"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # Cursor past the accepted log end.
        status, error = store.acknowledge_operations(
            "p1", "a3", 4, identities(("r1", "o1"), ("r2", "o2"), ("r1", "o3"), ("r9", "o9"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # Non-empty list against the empty segment at the checkpoint.
        store.save_checkpoint("p2", 3)
        status, error = store.acknowledge_operations(
            "p2", "a4", 3, identities(("r1", "o1"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # Nothing moved and no binding was recorded.
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 0)
        self.assertIs(
            store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.CREATED,
        )

    def test_receipts_are_not_operations(self) -> None:
        store = self.make_store()
        before_metrics = store.get_metrics()
        page_before, _, _ = store.get_sync_operations(0, 100)
        audit_before, _, _ = store.get_key_operations("k", 0, 100)
        state_before = store.get_state("k")
        digest_before = store.get_verification_digest()

        store.save_checkpoint("p1", 0)
        store.acknowledge_operations(
            "p1", "a1", 2, identities(("r1", "o1"), ("r2", "o2"))
        )
        store.acknowledge_operations("p1", "a1", 2, identities(("r1", "o1"), ("r2", "o2")))
        store.acknowledge_operations("p1", "a2", 1, identities(("r1", "o1")))

        self.assertEqual(store.get_metrics(), before_metrics)
        page_after, next_cursor, has_more = store.get_sync_operations(0, 100)
        self.assertEqual(page_after, page_before)
        self.assertEqual((next_cursor, has_more), (3, False))
        audit_after, _, _ = store.get_key_operations("k", 0, 100)
        self.assertEqual(audit_after, audit_before)
        self.assertEqual(store.get_state("k"), state_before)
        self.assertEqual(store.get_verification_digest(), digest_before)

    def test_replay_does_not_persist(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))
        with patch.object(
            StateStore, "_persist_locked", side_effect=AssertionError("replay must not persist")
        ):
            status, _ = store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))
            self.assertIs(status, HTTPStatus.OK)


class PersistentAcknowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def seed(self, store: StateStore) -> None:
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))

    def test_receipt_is_durable_with_the_checkpoint_advance(self) -> None:
        store = self.make_store()
        self.seed(store)
        store.save_checkpoint("p1", 0)
        status, _ = store.acknowledge_operations(
            "p1", "a1", 2, identities(("r1", "o1"), ("r2", "o2"))
        )
        self.assertIs(status, HTTPStatus.CREATED)
        _, checkpoints, _ = load_data_file_full(str(self.data_file))
        self.assertEqual(checkpoints, {"p1": 2})
        self.assertEqual(
            load_data_file_acks(str(self.data_file)),
            {("p1", "a1"): {"cursor": 2, "operations": identities(("r1", "o1"), ("r2", "o2"))}},
        )

    def test_persistence_failure_leaves_everything_and_is_retryable(self) -> None:
        store = self.make_store()
        self.seed(store)
        store.save_checkpoint("p1", 0)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))
        # Memory and file are exactly the pre-failure state.
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 0)
        self.assertEqual(self.data_file.read_bytes(), before)
        reloaded = self.make_store()
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 0)
        # The failed receipt left no binding: the same ackId commits cleanly.
        self.assertIs(
            store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.CREATED,
        )
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 1)

    def test_restart_preserves_replay_conflict_and_mismatch(self) -> None:
        store = self.make_store()
        self.seed(store)
        store.save_checkpoint("p1", 0)
        store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))
        del store

        reloaded = self.make_store()
        # The checkpoint advance and the binding survived.
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 1)
        # Identical replay stays 200.
        self.assertIs(
            reloaded.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.OK,
        )
        # Binding change stays an operation conflict.
        self.assertIs(
            reloaded.acknowledge_operations("p1", "a1", 1, identities(("r2", "o2")))[0],
            HTTPStatus.CONFLICT,
        )
        # Rollback stays a checkpoint conflict, mismatch an ack conflict.
        status, error = reloaded.acknowledge_operations("p1", "a2", 0, [])
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        status, error = reloaded.acknowledge_operations(
            "p1", "a3", 2, identities(("r1", "o1"))
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # A fresh valid receipt still commits after the restart.
        self.assertIs(
            reloaded.acknowledge_operations("p1", "a4", 2, identities(("r2", "o2")))[0],
            HTTPStatus.CREATED,
        )

    def test_old_version1_file_recovers_without_receipts(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                    "checkpoints": {"p1": 0},
                }
            ),
            encoding="utf-8",
        )
        store = self.make_store()
        # No receipt bindings exist, so the first acknowledgement creates.
        self.assertIs(
            store.acknowledge_operations("p1", "a1", 1, identities(("r1", "o1")))[0],
            HTTPStatus.CREATED,
        )
        # The commit upgraded the file to the supplemented format.
        self.assertEqual(
            load_data_file_acks(str(self.data_file)),
            {("p1", "a1"): {"cursor": 1, "operations": identities(("r1", "o1"))}},
        )

    def test_corrupt_ack_sections_are_rejected(self) -> None:
        def reject(section_raw: str) -> None:
            self.data_file.write_text(
                '{"version":1,"operations":['
                '{"replicaId":"r1","operation":'
                '{"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}],'
                f'"checkpoints":{{"p1":1}},{section_raw}}}',
                encoding="utf-8",
            )
            with self.assertRaises(PersistenceError):
                self.make_store()

        reject('"acks":{}')
        reject('"acks":[{"peerId":"p1","ackId":"a1","cursor":1}]')
        reject('"acks":[{"peerId":"","ackId":"a1","cursor":1,"operations":[]}]')
        reject('"acks":[{"peerId":"p1","ackId":"","cursor":1,"operations":[]}]')
        reject('"acks":[{"peerId":"p1","ackId":"a1","cursor":true,"operations":[]}]')
        reject('"acks":[{"peerId":"p1","ackId":"a1","cursor":-1,"operations":[]}]')
        reject(
            '"acks":[{"peerId":"p1","ackId":"a1","cursor":1,'
            '"operations":[{"replicaId":"r1","operationId":"o1"},'
            '{"replicaId":"r1","operationId":"o1"}]}]'
        )
        # A duplicate (peerId, ackId) binding is corrupt.
        reject(
            '"acks":[{"peerId":"p1","ackId":"a1","cursor":1,'
            '"operations":[{"replicaId":"r1","operationId":"o1"}]},'
            '{"peerId":"p1","ackId":"a1","cursor":1,'
            '"operations":[{"replicaId":"r1","operationId":"o1"}]}]'
        )
        # A binding whose peer never registered is corrupt.
        reject('"acks":[{"peerId":"p9","ackId":"a1","cursor":0,"operations":[]}]')
        # A binding past the peer checkpoint is corrupt.
        reject('"acks":[{"peerId":"p1","ackId":"a1","cursor":2,'
               '"operations":[{"replicaId":"r1","operationId":"o1"},'
               '{"replicaId":"r1","operationId":"o1"}]}]')
        # A binding that does not match the accepted log segment is corrupt.
        reject('"acks":[{"peerId":"p1","ackId":"a1","cursor":1,'
               '"operations":[{"replicaId":"r2","operationId":"o1"}]}]')


class HttpAcknowledgeTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, bytes]:
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
        conn.close()
        return response.status, raw

    def payload(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        status, raw = self.request(method, path, body)
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_ack(self, peer: str, body: object) -> tuple[int, object]:
        return self.payload("POST", f"/v1/sync/peers/{peer}/acknowledge", body)

    def seed_operations(self, count: int = 2) -> list[dict[str, str]]:
        pairs = []
        for index in range(1, count + 1):
            replica = f"r{index}"
            op_id = f"o{index}"
            status, _ = self.payload(
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", f"v{index}", {replica: 1}),
            )
            self.assertEqual(status, 201)
            pairs.append({"replicaId": replica, "operationId": op_id})
        return pairs

    def test_create_replay_and_response_shape(self) -> None:
        pairs = self.seed_operations(2)
        self.assertEqual(
            self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0], 200
        )
        body = {"ackId": "a1", "cursor": 2, "operations": pairs}
        status, raw = self.request("POST", "/v1/sync/peers/p1/acknowledge", body)
        self.assertEqual(status, 201)
        # Exactly four fields, compact JSON, one trailing newline.
        self.assertEqual(
            raw,
            b'{"ackId":"a1","cursor":2,"peerId":"p1","status":"created"}\n',
        )
        # The checkpoint advanced with the confirmation.
        self.assertEqual(
            self.payload("GET", "/v1/sync/peers/p1/checkpoint")[1],
            {"peerId": "p1", "cursor": 2},
        )
        # Identical replay: 200 with status ok, same shape, no new record.
        status, payload = self.post_ack("p1", body)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload, {"status": "ok", "peerId": "p1", "ackId": "a1", "cursor": 2}
        )
        self.assertEqual(
            self.payload("GET", "/v1/metrics")[1]["acceptedOperations"], 2
        )

    def test_unregistered_peer_is_404(self) -> None:
        status, payload = self.post_ack("nope", {"ackId": "a1", "cursor": 0, "operations": []})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_binding_change_is_409_operation_conflict(self) -> None:
        pairs = self.seed_operations(2)
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        body = {"ackId": "a1", "cursor": 1, "operations": pairs[:1]}
        self.assertEqual(self.post_ack("p1", body)[0], 201)
        for changed in (
            {"ackId": "a1", "cursor": 2, "operations": pairs},
            {"ackId": "a1", "cursor": 1, "operations": [pairs[1]]},
            {"ackId": "a1", "cursor": 1, "operations": []},
        ):
            status, payload = self.post_ack("p1", changed)
            self.assertEqual(status, 409, repr(changed))
            self.assertEqual(payload, {"error": "operation_conflict"}, repr(changed))
        # State is unchanged: the original binding still replays.
        self.assertEqual(self.post_ack("p1", body)[0], 200)
        self.assertEqual(
            self.payload("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 1
        )

    def test_cursor_below_checkpoint_is_409_checkpoint_conflict(self) -> None:
        pairs = self.seed_operations(2)
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 2})
        status, payload = self.post_ack(
            "p1", {"ackId": "a1", "cursor": 1, "operations": pairs[:1]}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        self.assertEqual(
            self.payload("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 2
        )

    def test_log_mismatch_is_409_ack_conflict(self) -> None:
        pairs = self.seed_operations(2)
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        mismatches = (
            # Wrong identity.
            {"ackId": "a1", "cursor": 1, "operations": [pairs[1]]},
            # Wrong record count.
            {"ackId": "a2", "cursor": 2, "operations": pairs[:1]},
            # Cursor past the log end.
            {
                "ackId": "a3",
                "cursor": 3,
                "operations": pairs + [{"replicaId": "r9", "operationId": "o9"}],
            },
        )
        for body in mismatches:
            status, payload = self.post_ack("p1", body)
            self.assertEqual(status, 409, repr(body))
            self.assertEqual(payload, {"error": "ack_conflict"}, repr(body))
        # Nothing was committed: the same ackIds are still usable.
        self.assertEqual(
            self.post_ack("p1", {"ackId": "a1", "cursor": 1, "operations": pairs[:1]})[0],
            201,
        )

    def test_invalid_bodies_are_400(self) -> None:
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        bad_bodies = [
            b"",
            b"{not json",
            [],
            {},
            {"ackId": "a1"},
            {"ackId": "a1", "cursor": 0},
            {"ackId": "a1", "cursor": 0, "operations": [], "extra": 1},
            {"ackId": "", "cursor": 0, "operations": []},
            {"ackId": 1, "cursor": 0, "operations": []},
            {"ackId": "a1", "cursor": True, "operations": []},
            {"ackId": "a1", "cursor": -1, "operations": []},
            {"ackId": "a1", "cursor": 1.0, "operations": []},
            {"ackId": "a1", "cursor": "0", "operations": []},
            {"ackId": "a1", "cursor": 0, "operations": {}},
            {"ackId": "a1", "cursor": 0, "operations": [{}]},
            {"ackId": "a1", "cursor": 0, "operations": [{"replicaId": "r1"}]},
            {
                "ackId": "a1",
                "cursor": 0,
                "operations": [{"replicaId": "r1", "operationId": "o1", "x": 1}],
            },
            {
                "ackId": "a1",
                "cursor": 0,
                "operations": [{"replicaId": "", "operationId": "o1"}],
            },
            {
                "ackId": "a1",
                "cursor": 0,
                "operations": [{"replicaId": "r1", "operationId": ""}],
            },
            {
                "ackId": "a1",
                "cursor": 0,
                "operations": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r1", "operationId": "o1"},
                ],
            },
        ]
        for body in bad_bodies:
            status, payload = self.post_ack("p1", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        # No binding was recorded by any rejected body.
        self.assertEqual(
            self.post_ack("p1", {"ackId": "a1", "cursor": 0, "operations": []})[0], 201
        )

    def test_segment_longer_than_one_hundred_is_400(self) -> None:
        pairs = self.seed_operations(3)
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        operations = [
            {"replicaId": "r", "operationId": f"o{i}"} for i in range(101)
        ]
        status, payload = self.post_ack(
            "p1", {"ackId": "a1", "cursor": 101, "operations": operations}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertEqual(
            self.payload("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 0
        )

    def test_query_parameters_are_400(self) -> None:
        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        for query in ("?x=1", "?x=", "?=1", "?cursor=0", "?x=1&x=2"):
            status, payload = self.payload(
                "POST",
                f"/v1/sync/peers/p1/acknowledge{query}",
                {"ackId": "a1", "cursor": 0, "operations": []},
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_route_shape_failures_are_404(self) -> None:
        body = {"ackId": "a1", "cursor": 0, "operations": []}
        for path in (
            "/v1/sync/peers//acknowledge",
            "/v1/sync/peers/p1/acknowledge/extra",
            "/v1/sync/peers/p1/acknowledge/",
            "/v1/sync/peers/p1",
        ):
            status, payload = self.payload("POST", path, body)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
        # GET is not published on the acknowledge route.
        self.assertEqual(self.payload("GET", "/v1/sync/peers/p1/acknowledge")[0], 404)
        # The route-shape check beats the query check.
        status, _ = self.payload("POST", "/v1/sync/peers/p1/acknowledge/extra?x=1", body)
        self.assertEqual(status, 404)

    def test_peer_id_is_percent_decoded(self) -> None:
        self.assertEqual(
            self.payload("POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0})[0],
            200,
        )
        status, payload = self.post_ack(
            "peer%20one", {"ackId": "a1", "cursor": 0, "operations": []}
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {"status": "created", "peerId": "peer one", "ackId": "a1", "cursor": 0},
        )

    def test_receipts_do_not_change_metrics_log_or_audit(self) -> None:
        pairs = self.seed_operations(2)
        metrics_before = self.payload("GET", "/v1/metrics")[1]
        sync_before = self.payload("GET", "/v1/sync/operations")[1]
        audit_before = self.payload("GET", "/v1/audit/keys/k/operations")[1]

        self.payload("POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        self.assertEqual(
            self.post_ack("p1", {"ackId": "a1", "cursor": 2, "operations": pairs})[0], 201
        )
        self.assertEqual(
            self.post_ack("p1", {"ackId": "a1", "cursor": 2, "operations": pairs})[0], 200
        )
        self.assertEqual(
            self.post_ack("p1", {"ackId": "a2", "cursor": 1, "operations": pairs[:1]})[0],
            409,
        )

        self.assertEqual(self.payload("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(self.payload("GET", "/v1/sync/operations")[1], sync_before)
        self.assertEqual(
            self.payload("GET", "/v1/audit/keys/k/operations")[1], audit_before
        )
        # The pickup view is anchored at the advanced checkpoint.
        status, payload = self.payload(
            "GET", "/v1/sync/peers/p1/operations?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])


class PersistentAcknowledgeHttpTests(unittest.TestCase):
    """Receipt durability, 500 handling, and recovery over real HTTP."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def start_server(self, auth_token: str | None = None) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=str(self.data_file),
            auth_token=auth_token,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(
        self,
        server: SemanticStateServer,
        method: str,
        path: str,
        body: object = None,
        auth: str | None = None,
    ):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers=headers,
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed(self, server: SemanticStateServer) -> list[dict[str, str]]:
        pairs = []
        for replica, op_id in (("r1", "o1"), ("r2", "o2")):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", f"v{op_id}", {replica: 1}),
            )
            self.assertEqual(status, 201)
            pairs.append({"replicaId": replica, "operationId": op_id})
        self.assertEqual(
            self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})[0],
            200,
        )
        return pairs

    def test_receipt_is_committed_before_201(self) -> None:
        server = self.start_server()
        pairs = self.seed(server)
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a1", "cursor": 2, "operations": pairs},
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            payload, {"status": "created", "peerId": "p1", "ackId": "a1", "cursor": 2}
        )
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 2})
        self.assertEqual(
            load_data_file_acks(str(self.data_file)),
            {("p1", "a1"): {"cursor": 2, "operations": pairs}},
        )
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_persistence_failure_is_500_retryable_and_leaves_everything(self) -> None:
        server = self.start_server()
        pairs = self.seed(server)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                {"ackId": "a1", "cursor": 2, "operations": pairs},
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File and visible memory are exactly the pre-failure state.
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 0
        )
        reloaded = StateStore(data_file=str(self.data_file))
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 0)
        # The failed receipt left no binding: the same request now commits.
        status, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a1", "cursor": 2, "operations": pairs},
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 2
        )
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_restart_preserves_receipts_and_semantics(self) -> None:
        server = self.start_server()
        pairs = self.seed(server)
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                {"ackId": "a1", "cursor": 1, "operations": pairs[:1]},
            )[0],
            201,
        )
        server.shutdown()
        server.server_close()

        server = self.start_server()
        # The checkpoint advance survived.
        self.assertEqual(
            self.request(server, "GET", "/v1/sync/peers/p1/checkpoint")[1],
            {"peerId": "p1", "cursor": 1},
        )
        # Replay stays 200, binding change stays 409 operation_conflict.
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                {"ackId": "a1", "cursor": 1, "operations": pairs[:1]},
            )[0],
            200,
        )
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a1", "cursor": 1, "operations": [pairs[1]]},
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Rollback stays 409 checkpoint_conflict, mismatch 409 ack_conflict.
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a2", "cursor": 0, "operations": []},
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a3", "cursor": 2, "operations": [pairs[0]]},
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "ack_conflict"})
        # A fresh valid receipt still commits after the restart.
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                {"ackId": "a4", "cursor": 2, "operations": [pairs[1]]},
            )[0],
            201,
        )

    def test_old_version1_file_recovers_without_receipts(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                    "checkpoints": {"p1": 0},
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            {"ackId": "a1", "cursor": 1, "operations": [{"replicaId": "r1", "operationId": "o1"}]},
        )
        self.assertEqual(status, 201)
        server.shutdown()
        server.server_close()
        server = self.start_server()
        self.assertEqual(
            self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                {"ackId": "a1", "cursor": 1, "operations": [{"replicaId": "r1", "operationId": "o1"}]},
            )[0],
            200,
        )

    def test_authentication_applies_to_acknowledge(self) -> None:
        server = self.start_server(auth_token=TOKEN)
        body = {"ackId": "a1", "cursor": 0, "operations": []}
        # A missing or wrong token is 401 before anything else.
        status, payload = self.request(
            server, "POST", "/v1/sync/peers/p1/acknowledge", body
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            body,
            auth="Bearer wrong",
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        # The health probe stays anonymous.
        self.assertEqual(self.request(server, "GET", "/health")[0], 200)
        # Authenticated requests reach the endpoint semantics (peer unknown).
        status, payload = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            body,
            auth=f"Bearer {TOKEN}",
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
