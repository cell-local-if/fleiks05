"""Tests for verifiable per-segment consumption receipts.

The receipt endpoint is::

    POST /v1/sync/peers/{peerId}/acknowledge
    {"ackId": A, "cursor": N, "operations": [{"replicaId", "operationId"}, ...]}

A receipt starts at the peer's registered checkpoint: ``operations`` must
exactly cover, in order, the contiguous accepted records before
``cursor``. Committing a receipt both advances the peer's checkpoint and
binds the peer's ``ackId`` to that exact cursor and identity list. A
receipt is not an operation: it never enters the accepted log, sync
export, the audit streams, or the metrics counters. With ``--data-file``
the binding, the checkpoint advance, and their persistence are one
atomic commit.

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
    MAX_BODY_BYTES,
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _load_data_file_complete,
    load_data_file_full,
    parse_acknowledgement_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica_id: str, operation_id: str) -> dict:
    return {"replicaId": replica_id, "operationId": operation_id}


def seed_store(store: StateStore, count: int) -> None:
    """Commit ``count`` accepted records with distinct identities r{i}/o{i}."""
    for i in range(1, count + 1):
        status = store.apply_operation(
            f"r{i}", operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1})
        )
        assert status is HTTPStatus.CREATED


def identities_for(count: int, start: int = 1) -> list[dict]:
    return [identity(f"r{i}", f"o{i}") for i in range(start, start + count)]


class ParseAcknowledgementPayloadTests(unittest.TestCase):
    def test_accepts_valid_bodies_including_empty_segment(self) -> None:
        cases = [
            (b'{"ackId":"a","cursor":0,"operations":[]}', ("a", 0, [])),
            (
                '{"ackId": "a", "cursor": 2, "operations": '
                '[{"replicaId":"r1","operationId":"o1"}]}',
                ("a", 2, [identity("r1", "o1")]),
            ),
            (
                {"ackId": "a", "cursor": 1, "operations": [identity("r1", "o1")]},
                ("a", 1, [identity("r1", "o1")]),
            ),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_acknowledgement_payload(raw), expected)

    def test_accepts_up_to_100_identities(self) -> None:
        operations = [identity("r", f"o{i}") for i in range(100)]
        _, cursor, parsed = parse_acknowledgement_payload(
            {"ackId": "a", "cursor": 100, "operations": operations}
        )
        self.assertEqual(cursor, 100)
        self.assertEqual(len(parsed), 100)

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_acknowledgement_payload(b"{not json")

    def test_rejects_non_object_and_empty_body(self) -> None:
        for raw in (b"", b"[]", b"5", b"null", b'"ack"'):
            with self.assertRaises(ValueError):
                parse_acknowledgement_payload(raw)

    def test_rejects_wrong_keys(self) -> None:
        good_ops = [identity("r1", "o1")]
        for raw in (
            {},
            {"ackId": "a", "cursor": 1},
            {"ackId": "a", "cursor": 1, "operations": good_ops, "extra": 1},
            {"ackid": "a", "cursor": 1, "operations": good_ops},
            {"ackId": "", "cursor": 1, "operations": good_ops},
        ):
            with self.assertRaises(ValueError):
                parse_acknowledgement_payload(raw)

    def test_rejects_bad_ack_id(self) -> None:
        for ack_id in (1, True, None, [], {}):
            with self.assertRaises(ValueError):
                parse_acknowledgement_payload(
                    {"ackId": ack_id, "cursor": 0, "operations": []}
                )

    def test_rejects_bad_cursor(self) -> None:
        for cursor in (True, False, -1, 1.0, "1", None, [], {}):
            with self.assertRaises(ValueError):
                parse_acknowledgement_payload(
                    {"ackId": "a", "cursor": cursor, "operations": []}
                )

    def test_rejects_bad_operations(self) -> None:
        for operations in (
            None,
            {},
            "x",
            [identity("r", f"o{i}") for i in range(101)],
            [{}],
            [{"replicaId": "r1"}],
            [{"replicaId": "r1", "operationId": "o1", "extra": 1}],
            [{"replicaId": "", "operationId": "o1"}],
            [{"replicaId": "r1", "operationId": ""}],
            [{"replicaId": 1, "operationId": "o1"}],
            [{"replicaId": "r1", "operationId": 1}],
        ):
            with self.assertRaises(ValueError):
                parse_acknowledgement_payload(
                    {"ackId": "a", "cursor": 1, "operations": operations}
                )

    def test_rejects_duplicate_identities(self) -> None:
        with self.assertRaises(ValueError):
            parse_acknowledgement_payload(
                {
                    "ackId": "a",
                    "cursor": 2,
                    "operations": [identity("r1", "o1"), identity("r1", "o1")],
                }
            )


class AcknowledgeStoreTests(unittest.TestCase):
    """Store-level semantics against an in-memory store."""

    def setUp(self) -> None:
        self.store = StateStore()
        seed_store(self.store, 5)

    def register(self, peer: str = "p1", cursor: int = 0) -> None:
        status, _ = self.store.save_checkpoint(peer, cursor)
        self.assertIs(status, HTTPStatus.OK)

    def acknowledge(self, peer: str, ack_id: str, cursor: int, operations: list):
        return self.store.acknowledge(peer, ack_id, cursor, operations)

    def test_create_advances_checkpoint_and_replays(self) -> None:
        self.register()
        status, error = self.acknowledge("p1", "a1", 2, identities_for(2))
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(
            self.store.get_checkpoint("p1"),
            (HTTPStatus.OK, {"peerId": "p1", "cursor": 2}),
        )
        # The next segment starts at the advanced checkpoint.
        status, _ = self.acknowledge("p1", "a2", 5, identities_for(3, start=3))
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 5)
        # Exact replays are answered from the binding without moving.
        status, _ = self.acknowledge("p1", "a1", 2, identities_for(2))
        self.assertIs(status, HTTPStatus.OK)
        status, _ = self.acknowledge("p1", "a2", 5, identities_for(3, start=3))
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 5)

    def test_unregistered_peer_is_not_found(self) -> None:
        status, error = self.acknowledge("ghost", "a1", 0, [])
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(error, "not_found")
        self.assertEqual(self.store.get_checkpoint("ghost")[0], HTTPStatus.NOT_FOUND)

    def test_log_mismatch_is_ack_conflict_and_changes_nothing(self) -> None:
        self.register()
        # Wrong identity in the middle.
        status, error = self.acknowledge(
            "p1", "a1", 2, [identity("r1", "o1"), identity("r9", "o2")]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        # Swapped order.
        status, error = self.acknowledge(
            "p1", "a1", 2, [identity("r2", "o2"), identity("r1", "o1")]
        )
        self.assertEqual(error, "ack_conflict")
        # Length shorter than the cursor span.
        status, error = self.acknowledge("p1", "a1", 2, [identity("r1", "o1")])
        self.assertEqual(error, "ack_conflict")
        # Length longer than the cursor span.
        status, error = self.acknowledge(
            "p1", "a1", 1, identities_for(2)
        )
        self.assertEqual(error, "ack_conflict")
        # Unknown identity that exists nowhere.
        status, error = self.acknowledge(
            "p1", "a1", 1, [identity("r1", "nope")]
        )
        self.assertEqual(error, "ack_conflict")
        # The binding was never created and the checkpoint never moved; the
        # same ackId with correct content still creates.
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 0)
        status, _ = self.acknowledge("p1", "a1", 1, identities_for(1))
        self.assertIs(status, HTTPStatus.CREATED)

    def test_cursor_past_the_log_is_ack_conflict(self) -> None:
        self.register()
        # The segment clips to the accepted tail, so its length never
        # matches cursor - checkpoint and every such request fails as a log
        # mismatch (whatever identities are offered).
        status, error = self.acknowledge("p1", "a1", 6, identities_for(5))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "ack_conflict")
        status, error = self.acknowledge(
            "p1", "a2", 6, identities_for(5) + [identity("r6", "o6")]
        )
        self.assertEqual(error, "ack_conflict")

    def test_cursor_below_checkpoint_is_checkpoint_conflict(self) -> None:
        self.register()
        self.assertIs(self.acknowledge("p1", "a1", 2, identities_for(2))[0], HTTPStatus.CREATED)
        status, error = self.acknowledge("p1", "a2", 1, identities_for(1))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "checkpoint_conflict")
        status, error = self.acknowledge("p1", "a2", 0, [])
        self.assertEqual(error, "checkpoint_conflict")
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 2)

    def test_binding_change_is_operation_conflict_and_state_unchanged(self) -> None:
        self.register()
        self.assertIs(self.acknowledge("p1", "a1", 1, identities_for(1))[0], HTTPStatus.CREATED)
        # Same ackId, different cursor/identities.
        status, error = self.acknowledge("p1", "a1", 2, identities_for(2))
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")
        status, error = self.acknowledge("p1", "a1", 1, [])
        self.assertEqual(error, "operation_conflict")
        # The binding takes precedence over a rollback cursor.
        status, error = self.acknowledge("p1", "a1", 0, [])
        self.assertEqual(error, "operation_conflict")
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 1)
        # A replay even after the checkpoint moved on stays 200.
        self.acknowledge("p1", "a2", 2, identities_for(1, start=2))
        status, _ = self.acknowledge("p1", "a1", 1, identities_for(1))
        self.assertIs(status, HTTPStatus.OK)

    def test_ack_ids_are_scoped_per_peer(self) -> None:
        self.register("p1")
        self.register("p2")
        self.assertIs(self.acknowledge("p1", "shared", 1, identities_for(1))[0], HTTPStatus.CREATED)
        # The same ackId for a different peer is a fresh receipt.
        self.assertIs(self.acknowledge("p2", "shared", 1, identities_for(1))[0], HTTPStatus.CREATED)
        # Each peer's binding is independent.
        self.assertIs(
            self.acknowledge("p1", "shared", 2, identities_for(2))[0], HTTPStatus.CONFLICT
        )
        self.assertIs(
            self.acknowledge("p2", "shared", 2, identities_for(2))[0], HTTPStatus.CONFLICT
        )

    def test_zero_length_receipt_at_current_cursor_creates_without_moving(self) -> None:
        self.register()
        status, _ = self.acknowledge("p1", "a0", 0, [])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 0)
        # Its replay is idempotent, and a second ackId at the same cursor
        # is likewise a distinct receipt.
        self.assertIs(self.acknowledge("p1", "a0", 0, [])[0], HTTPStatus.OK)
        self.assertIs(self.acknowledge("p1", "a0b", 0, [])[0], HTTPStatus.CREATED)
        # A real segment still advances afterwards.
        self.assertIs(self.acknowledge("p1", "a1", 1, identities_for(1))[0], HTTPStatus.CREATED)
        self.assertEqual(self.store.get_checkpoint("p1")[1]["cursor"], 1)

    def test_segment_may_span_stale_and_imported_records(self) -> None:
        store = StateStore()
        # A stale write (clock dominated) still occupies a log position and
        # must be acknowledged by its committed identity.
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        self.assertIs(
            store.apply_operation("r1", operation("o2", "k", "v0", {"r1": 0})),
            HTTPStatus.CREATED,
        )
        store.save_checkpoint("p1", 0)
        status, _ = store.acknowledge(
            "p1", "a1", 2, [identity("r1", "o1"), identity("r1", "o2")]
        )
        self.assertIs(status, HTTPStatus.CREATED)

    def test_receipts_are_not_operations(self) -> None:
        self.register()
        metrics_before = self.store.get_metrics()
        page_before, _, _ = self.store.get_sync_operations(0, 100)
        audit_before, _, _ = self.store.get_key_operations("k", 0, 100)

        self.acknowledge("p1", "a1", 2, identities_for(2))
        # Conflicts likewise leave no trace anywhere.
        self.assertEqual(
            self.acknowledge("p1", "a1", 3, identities_for(3))[1], "operation_conflict"
        )
        self.assertEqual(
            self.acknowledge("p1", "a2", 0, [])[1], "checkpoint_conflict"
        )
        self.assertEqual(
            self.acknowledge("p1", "a3", 5, identities_for(4))[1], "ack_conflict"
        )

        self.assertEqual(self.store.get_metrics(), metrics_before)
        page_after, _, _ = self.store.get_sync_operations(0, 100)
        self.assertEqual(page_after, page_before)
        audit_after, _, _ = self.store.get_key_operations("k", 0, 100)
        self.assertEqual(audit_after, audit_before)

    def test_replay_performs_no_persist_even_with_a_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            store = StateStore(data_file=path)
            seed_store(store, 2)
            store.save_checkpoint("p1", 0)
            self.assertIs(store.acknowledge("p1", "a1", 2, identities_for(2))[0], HTTPStatus.CREATED)
            with patch.object(
                StateStore, "_persist_locked", side_effect=AssertionError("replay must not persist")
            ):
                self.assertIs(
                    store.acknowledge("p1", "a1", 2, identities_for(2))[0], HTTPStatus.OK
                )


class PersistentAcknowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def read_section(self) -> list:
        return _load_data_file_complete(str(self.data_file))[4]

    def test_receipt_and_checkpoint_advance_are_durable_before_return(self) -> None:
        store = self.make_store()
        seed_store(store, 2)
        store.save_checkpoint("p1", 0)
        status, _ = store.acknowledge("p1", "a1", 2, identities_for(2))
        self.assertIs(status, HTTPStatus.CREATED)
        acks = self.read_section()
        self.assertEqual(
            acks,
            {"p1": {"a1": {"cursor": 2, "operations": identities_for(2)}}},
        )
        self.assertEqual(load_data_file_full(str(self.data_file))[1], {"p1": 2})

    def test_persistence_failure_leaves_binding_and_checkpoint_unchanged(self) -> None:
        store = self.make_store()
        seed_store(store, 2)
        store.save_checkpoint("p1", 0)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.acknowledge("p1", "a1", 2, identities_for(2))
        # Memory, binding, and file are exactly as before; retry commits.
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 0)
        self.assertEqual(self.data_file.read_bytes(), before)
        reloaded = self.make_store()
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 0)
        status, _ = store.acknowledge("p1", "a1", 2, identities_for(2))
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(self.read_section()["p1"]["a1"]["cursor"], 2)

    def test_restart_preserves_receipts_and_all_decisions(self) -> None:
        store = self.make_store()
        seed_store(store, 4)
        store.save_checkpoint("p1", 0)
        store.acknowledge("p1", "a1", 2, identities_for(2))
        store.acknowledge("p1", "a2", 4, identities_for(2, start=3))
        del store

        reloaded = self.make_store()
        self.assertEqual(reloaded.get_checkpoint("p1")[1]["cursor"], 4)
        # Exact replay stays 200 after restart.
        self.assertIs(
            reloaded.acknowledge("p1", "a1", 2, identities_for(2))[0], HTTPStatus.OK
        )
        self.assertIs(
            reloaded.acknowledge("p1", "a2", 4, identities_for(2, start=3))[0],
            HTTPStatus.OK,
        )
        # A changed binding is still an operation_conflict.
        self.assertEqual(
            reloaded.acknowledge("p1", "a1", 2, [])[1], "operation_conflict"
        )
        # Rollback stays checkpoint_conflict.
        self.assertEqual(
            reloaded.acknowledge("p1", "a3", 3, identities_for(1))[1],
            "checkpoint_conflict",
        )
        # Log mismatch stays ack_conflict (a wrong identity at the tail).
        reloaded.save_checkpoint("p2", 0)
        wrong = identities_for(4)
        wrong[3] = identity("r9", "o4")
        self.assertEqual(
            reloaded.acknowledge("p2", "b1", 4, wrong)[1], "ack_conflict"
        )
        # And a correct receipt for that fresh peer still commits.
        self.assertIs(
            reloaded.acknowledge("p2", "b1", 4, identities_for(4))[0],
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
        self.assertEqual(self.read_section(), {})
        # The recovered checkpoint anchors a fresh receipt normally.
        self.assertEqual(store.get_checkpoint("p1")[1]["cursor"], 0)
        status, _ = store.acknowledge("p1", "a1", 1, identities_for(1))
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(self.read_section()["p1"]["a1"]["cursor"], 1)

    def test_corrupt_acknowledgment_sections_are_rejected(self) -> None:
        op_record = json.dumps(
            {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
        )
        good_ops = '[{"replicaId":"r1","operationId":"o1"}]'

        def reject(section_raw: str) -> None:
            self.data_file.write_text(
                '{"version":1,"operations":['
                + op_record
                + "],"
                + '"checkpoints":{"p1":1},'
                + section_raw
                + "}",
                encoding="utf-8",
            )
            with self.assertRaises(PersistenceError):
                self.make_store()

        reject('"acknowledgments":{}')  # wrong top type
        reject(f'"acknowledgments":[{{"peerId":"p1","ackId":"a","cursor":1}}]')
        reject(
            '"acknowledgments":['
            f'{{"peerId":"","ackId":"a","cursor":1,"operations":{good_ops}}}]'
        )
        reject(
            '"acknowledgments":['
            f'{{"peerId":"p1","ackId":"","cursor":1,"operations":{good_ops}}}]'
        )
        reject(
            '"acknowledgments":['
            f'{{"peerId":"p1","ackId":"a","cursor":true,"operations":{good_ops}}}]'
        )
        reject(
            '"acknowledgments":['
            f'{{"peerId":"p1","ackId":"a","cursor":-1,"operations":{good_ops}}}]'
        )
        # Cursor past the recovered log length.
        reject(
            '"acknowledgments":['
            f'{{"peerId":"p1","ackId":"a","cursor":2,"operations":{good_ops}}}]'
        )
        # Identity does not match the committed log at that position.
        reject(
            '"acknowledgments":['
            '{"peerId":"p1","ackId":"a","cursor":1,"operations":'
            '[{"replicaId":"r9","operationId":"o1"}]}]'
        )
        # Receipt names a peer with no recovered checkpoint.
        reject(
            '"acknowledgments":['
            '{"peerId":"ghost","ackId":"a","cursor":0,"operations":[]}]'
        )
        # Listing more identities than the log contains before the cursor.
        reject(
            '"acknowledgments":['
            '{"peerId":"p1","ackId":"a","cursor":1,"operations":'
            '[{"replicaId":"r1","operationId":"o1"},'
            '{"replicaId":"r1","operationId":"o2"}]}]'
        )

    def test_receipt_cursor_outrunning_recovered_checkpoint_is_rejected(self) -> None:
        # The peer's recovered checkpoint (0) lags a receipt cursor (1):
        # a receipt can only exist together with the advance it performed.
        self.data_file.write_text(
            '{"version":1,"operations":['
            '{"replicaId":"r1","operation":'
            + json.dumps(operation("o1", "k", "v", {"r1": 1}))
            + '}],"checkpoints":{"p1":0},'
            '"acknowledgments":['
            '{"peerId":"p1","ackId":"a","cursor":1,'
            '"operations":[{"replicaId":"r1","operationId":"o1"}]}]}',
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()

    def test_unknown_root_key_is_still_rejected(self) -> None:
        self.data_file.write_text(
            '{"version":1,"operations":['
            '{"replicaId":"r1","operation":'
            + json.dumps(operation("o1", "k", "v", {"r1": 1}))
            + '],"extra":1}',
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()

    def test_duplicate_receipt_binding_in_file_is_rejected(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                    "checkpoints": {"p1": 1},
                    "acknowledgments": [
                        {
                            "peerId": "p1",
                            "ackId": "a",
                            "cursor": 1,
                            "operations": [identity("r1", "o1")],
                        },
                        {
                            "peerId": "p1",
                            "ackId": "a",
                            "cursor": 1,
                            "operations": [identity("r1", "o1")],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(PersistenceError):
            self.make_store()


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

    def request_raw(self, method: str, path: str, raw: bytes | None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, body=raw, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def request(self, method: str, path: str, body: object = None):
        if body is None:
            status, data = self.request_raw(method, path, None)
        elif isinstance(body, (bytes, str)):
            raw = body if isinstance(body, bytes) else body.encode("utf-8")
            status, data = self.request_raw(
                method, path, raw, {"Content-Type": "application/json"}
            )
        else:
            status, data = self.request_raw(
                method,
                path,
                json.dumps(body).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
        payload = json.loads(data.decode("utf-8")) if data else None
        return status, payload

    def write_ops(self, count: int) -> None:
        for i in range(1, count + 1):
            status, _ = self.request(
                "POST",
                f"/v1/replicas/r{i}/operations",
                operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}),
            )
            self.assertEqual(status, 201)

    def register(self, peer: str = "p1", cursor: int = 0) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        self.assertEqual(status, 200)

    def ack(self, peer: str, ack_id: str, cursor: int, operations: list, query: str = ""):
        return self.request(
            "POST",
            f"/v1/sync/peers/{peer}/acknowledge{query}",
            {"ackId": ack_id, "cursor": cursor, "operations": operations},
        )

    def test_created_body_has_exactly_four_fields_and_one_newline(self) -> None:
        self.write_ops(2)
        self.register()
        status, raw = self.request_raw(
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            json.dumps(
                {"ackId": "a1", "cursor": 2, "operations": identities_for(2)}
            ).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            raw,
            b'{"ackId":"a1","cursor":2,"peerId":"p1","status":"created"}\n',
        )

    def test_replay_is_200_ok_and_creates_nothing(self) -> None:
        self.write_ops(1)
        self.register()
        body = {"ackId": "a1", "cursor": 1, "operations": identities_for(1)}
        self.assertEqual(self.request("POST", "/v1/sync/peers/p1/acknowledge", body)[0], 201)
        status, raw = self.request_raw(
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            json.dumps(body).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"ackId":"a1","cursor":1,"peerId":"p1","status":"ok"}\n')
        # Metrics and the log are unchanged by create and replay.
        metrics = self.request("GET", "/v1/metrics")[1]
        self.assertEqual(metrics["acceptedOperations"], 1)

    def test_full_create_advance_flow_and_checkpoint_visibility(self) -> None:
        self.write_ops(3)
        self.register("p1", 0)
        status, payload = self.ack("p1", "a1", 2, identities_for(2))
        self.assertEqual(status, 201)
        self.assertEqual(
            payload, {"status": "created", "peerId": "p1", "ackId": "a1", "cursor": 2}
        )
        # The pickup endpoint immediately reflects the advanced checkpoint.
        status, payload = self.request(
            "GET", "/v1/sync/peers/p1/operations?after=0&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual([op["operation"]["operationId"] for op in payload["operations"]], ["o3"])
        # Acknowledge the final record.
        status, payload = self.ack("p1", "a2", 3, identities_for(1, start=3))
        self.assertEqual(status, 201)
        self.assertEqual(payload["cursor"], 3)

    def test_binding_change_is_409_and_state_unchanged(self) -> None:
        self.write_ops(2)
        self.register()
        self.assertEqual(self.ack("p1", "a1", 1, identities_for(1))[0], 201)
        status, payload = self.ack("p1", "a1", 2, identities_for(2))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.assertEqual(self.request("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 1)

    def test_ack_conflict_variants_are_409(self) -> None:
        self.write_ops(3)
        self.register()
        wrong_order = [identity("r2", "o2"), identity("r1", "o1")]
        wrong_identity = [identity("r1", "o1"), identity("r9", "o2")]
        too_short = [identity("r1", "o1")]
        too_long = identities_for(3)
        past_tail = identities_for(3) + [identity("r4", "o4")]
        for cursor, operations, label in (
            (2, wrong_order, "order"),
            (2, wrong_identity, "identity"),
            (2, too_short, "short"),
            (2, too_long, "long"),
            (4, past_tail, "past tail"),
        ):
            status, payload = self.ack("p1", f"a-{label}", cursor, operations)
            self.assertEqual(status, 409, label)
            self.assertEqual(payload, {"error": "ack_conflict"}, label)
        # Nothing moved and no ackId got bound.
        self.assertEqual(self.request("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 0)
        status, _ = self.ack("p1", "identity", 2, wrong_identity)
        self.assertEqual(status, 409)
        status, _ = self.ack("p1", "identity", 2, identities_for(2))
        self.assertEqual(status, 201)

    def test_cursor_below_checkpoint_is_409(self) -> None:
        self.write_ops(2)
        self.register()
        self.assertEqual(self.ack("p1", "a1", 2, identities_for(2))[0], 201)
        status, payload = self.ack("p1", "a2", 1, identities_for(1))
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})

    def test_unregistered_peer_is_404(self) -> None:
        self.write_ops(1)
        status, payload = self.request(
            "POST",
            "/v1/sync/peers/ghost/acknowledge",
            {"ackId": "a1", "cursor": 0, "operations": []},
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_invalid_bodies_are_400(self) -> None:
        self.write_ops(1)
        self.register()
        good_ops = identities_for(1)
        bad_bodies = [
            b"",
            b"{not json",
            [],
            {},
            {"ackId": "a", "cursor": 1},
            {"ackId": "a", "cursor": 1, "operations": good_ops, "extra": 1},
            {"ackId": 1, "cursor": 1, "operations": good_ops},
            {"ackId": "", "cursor": 1, "operations": good_ops},
            {"ackId": "a", "cursor": -1, "operations": good_ops},
            {"ackId": "a", "cursor": True, "operations": good_ops},
            {"ackId": "a", "cursor": 1.0, "operations": good_ops},
            {"ackId": "a", "cursor": "1", "operations": good_ops},
            {"ackId": "a", "cursor": 1, "operations": None},
            {"ackId": "a", "cursor": 1, "operations": [{}]},
            {"ackId": "a", "cursor": 1, "operations": [{"replicaId": "", "operationId": "o1"}]},
            {"ackId": "a", "cursor": 1, "operations": [{"replicaId": "r1", "operationId": ""}]},
            {
                "ackId": "a",
                "cursor": 2,
                "operations": [identity("r1", "o1"), identity("r1", "o1")],
            },
            {
                "ackId": "a",
                "cursor": 101,
                "operations": [identity("r", f"o{i}") for i in range(101)],
            },
        ]
        for body in bad_bodies:
            status, payload = self.request(
                "POST", "/v1/sync/peers/p1/acknowledge", body
            )
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        # State stayed at the registered checkpoint.
        self.assertEqual(self.request("GET", "/v1/sync/peers/p1/checkpoint")[1]["cursor"], 0)

    def test_query_parameters_are_400(self) -> None:
        for query in ("?x=1", "?x=", "?=1", "?cursor=0", "?x=1&x=2", "?ackId=a"):
            status, payload = self.ack("p1", "a", 0, [], query=query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_path_shape_takes_precedence_over_query(self) -> None:
        self.write_ops(1)
        self.register()
        # Extra segment and trailing slash are 404 even with a bad query.
        for path in (
            "/v1/sync/peers/p1/acknowledge/extra?x=1",
            "/v1/sync/peers/p1/acknowledge/?x=1",
            "/v1/sync/peers?x=1",
        ):
            status, payload = self.request(
                "POST", path, {"ackId": "a", "cursor": 0, "operations": []}
            )
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_empty_peer_segment_is_404_even_with_query(self) -> None:
        for path in (
            "/v1/sync/peers//acknowledge",
            "/v1/sync/peers//acknowledge?x=1",
        ):
            status, payload = self.request(
                "POST", path, {"ackId": "a", "cursor": 0, "operations": []}
            )
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_and_other_methods_on_the_route_are_not_served(self) -> None:
        status, _ = self.request("GET", "/v1/sync/peers/p1/acknowledge")
        self.assertEqual(status, 404)

    def test_receipts_do_not_change_metrics_log_or_audit(self) -> None:
        self.write_ops(2)
        self.register()
        metrics_before = self.request("GET", "/v1/metrics")[1]
        sync_before = self.request("GET", "/v1/sync/operations")[1]
        audit_before = self.request("GET", "/v1/audit/keys/k/operations")[1]

        self.assertEqual(self.ack("p1", "a1", 2, identities_for(2))[0], 201)
        self.assertEqual(self.ack("p1", "a1", 2, identities_for(2))[0], 200)
        self.assertEqual(self.ack("p1", "a2", 0, [])[0], 409)
        # An empty list at a cursor inside the covered range cannot match
        # the one-record segment.
        self.assertEqual(self.ack("p1", "a3", 1, [])[0], 409)

        self.assertEqual(self.request("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(self.request("GET", "/v1/sync/operations")[1], sync_before)
        self.assertEqual(self.request("GET", "/v1/audit/keys/k/operations")[1], audit_before)

    def test_peer_id_is_percent_decoded(self) -> None:
        self.write_ops(1)
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer%20one/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        status, payload = self.request(
            "POST",
            "/v1/sync/peers/peer%20one/acknowledge",
            {"ackId": "a1", "cursor": 1, "operations": identities_for(1)},
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["peerId"], "peer one")


class PersistentAcknowledgeHttpTests(unittest.TestCase):
    """Durability, 500 handling, and recovery over real HTTP."""

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
        auth: str | None = "none",
        raw_body: bytes | None = None,
        extra_headers: dict | None = None,
    ):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        headers = dict(extra_headers or {})
        if auth != "none":
            headers["Authorization"] = auth
        if raw_body is not None:
            payload = raw_body
        elif body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        else:
            payload = None
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        result_headers = response.getheaders()
        conn.close()
        parsed = json.loads(data.decode("utf-8")) if data else None
        return response.status, parsed, result_headers

    def seed(self, server: SemanticStateServer, count: int) -> None:
        for i in range(1, count + 1):
            status, _, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/r{i}/operations",
                operation(f"o{i}", "k", f"v{i}", {f"r{i}": 1}),
            )
            self.assertEqual(status, 201)

    def ack_body(self, cursor: int, operations: list, ack_id: str = "a1") -> dict:
        return {"ackId": ack_id, "cursor": cursor, "operations": operations}

    def test_receipt_is_durable_before_the_201(self) -> None:
        server = self.start_server()
        self.seed(server, 1)
        self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(1, identities_for(1)),
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        acks = _load_data_file_complete(str(self.data_file))[4]
        self.assertEqual(acks["p1"]["a1"]["cursor"], 1)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_persistence_failure_is_500_and_replay_still_200(self) -> None:
        server = self.start_server()
        self.seed(server, 2)
        self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        status, _, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(1, identities_for(1)),
        )
        self.assertEqual(status, 201)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload, _ = self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                self.ack_body(2, identities_for(1, start=2), ack_id="a2"),
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # The exact replay needs no durable write and still succeeds.
            status, payload, _ = self.request(
                server,
                "POST",
                "/v1/sync/peers/p1/acknowledge",
                self.ack_body(1, identities_for(1)),
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")
        self.assertEqual(self.data_file.read_bytes(), before)
        status, payload, _ = self.request(
            server, "GET", "/v1/sync/peers/p1/checkpoint"
        )
        self.assertEqual(payload["cursor"], 1)
        # The failed receipt commits cleanly once persistence is back.
        status, _, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(2, identities_for(1, start=2), ack_id="a2"),
        )
        self.assertEqual(status, 201)

    def test_restart_replay_conflict_and_mismatch_are_identical(self) -> None:
        server = self.start_server()
        self.seed(server, 2)
        self.request(server, "POST", "/v1/sync/peers/p1/checkpoint", {"cursor": 0})
        status, _, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(2, identities_for(2)),
        )
        self.assertEqual(status, 201)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(2, identities_for(2)),
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(2, []),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        self.request(server, "POST", "/v1/sync/peers/p2/checkpoint", {"cursor": 0})
        wrong = identities_for(2)
        wrong[1] = identity("r9", "o2")
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p2/acknowledge",
            self.ack_body(2, wrong),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "ack_conflict"})

    def test_old_file_without_receipts_recovers_as_no_receipts(self) -> None:
        # A genuinely old file: version, operations, checkpoint only, and
        # none of the later sections.
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {
                            "replicaId": "r1",
                            "operation": operation("o1", "k", "v1", {"r1": 1}),
                        }
                    ],
                    "checkpoints": {"p1": 0},
                }
            ),
            encoding="utf-8",
        )
        server = self.start_server()
        # The ackId is unknown: it creates rather than replays.
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(1, identities_for(1)),
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        # Restart: it is now a binding and the identical post replays.
        server.shutdown()
        server.server_close()
        server = self.start_server()
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/sync/peers/p1/acknowledge",
            self.ack_body(1, identities_for(1)),
        )
        self.assertEqual(status, 200)


class AcknowledgeAuthPrecedenceTests(unittest.TestCase):
    """Length validation precedes authentication; shape precedes both."""

    TOKEN = "s3cret-ack-token"

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

    def raw_request(self, path: str, raw: bytes | None, headers: dict):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read()
        result_headers = response.getheaders()
        conn.close()
        return response.status, data, result_headers

    def test_missing_content_length_is_400_before_auth(self) -> None:
        # http.client always sends Content-Length for a bytes body, so use a
        # low-level request without it; identity framing aside, the 400 is
        # decided before the auth header is inspected.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/v1/sync/peers/p1/acknowledge")
        conn.endheaders()
        response = conn.getresponse()
        data = response.read()
        conn.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(data), {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_auth(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/v1/sync/peers/p1/acknowledge")
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.endheaders()
        response = conn.getresponse()
        data = response.read()
        conn.close()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(data), {"error": "payload_too_large"})

    def test_valid_length_with_bad_auth_is_401_without_reading_body(self) -> None:
        body = json.dumps({"ackId": "a", "cursor": 0, "operations": []}).encode("utf-8")
        status, data, headers = self.raw_request(
            "/v1/sync/peers/p1/acknowledge",
            body,
            {"Content-Type": "application/json", "Authorization": "Bearer wrong"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(data), {"error": "unauthorized"})
        self.assertIn(("WWW-Authenticate", "Bearer"), headers)

    def test_valid_length_with_valid_auth_passes_to_404_for_unknown_peer(self) -> None:
        body = json.dumps({"ackId": "a", "cursor": 0, "operations": []}).encode("utf-8")
        status, data, _ = self.raw_request(
            "/v1/sync/peers/ghost/acknowledge",
            body,
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(data), {"error": "not_found"})

    def test_shape_error_is_404_even_without_auth(self) -> None:
        body = b'{"ackId":"a","cursor":0,"operations":[]}'
        # An unknown route shape authenticates first (like every unknown
        # route), but a shape on a published POST path still demands valid
        # length first; assert the route is not served as an acknowledgement
        # either way.
        status, _, _ = self.raw_request(
            "/v1/sync/peers/p1/acknowledge/extra",
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
