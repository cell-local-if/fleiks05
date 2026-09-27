"""Tests for the conditional replication-repair preflight and execution::

    POST /v1/replication/repairs/plan   (read-only preflight)
    POST /v1/replication/repairs/apply  (committing execution)

Each request carries ``peerId``, an ``ackId`` naming the execution, the
``expectedCheckpoint`` the peer currently holds, the
``expectedReceipts`` digest the peer's committed receipt set must match,
and the ordered ``suggestions`` — together they lock the target. The
preflight only checks a staged view and returns a per-item conclusion
with the predicted boundary, committing nothing; the apply processes the
batch in the fixed action order (resend, deduplicate, correct_identity,
correct_cursor), commits the repair record and advances the checkpoint
together once every guard passes, and is idempotent by
``(peerId, ackId)``.

The tests cover the body parser, the shared staged guards (anchor
mismatch -> apply_conflict; stale log location/identity ->
repair_conflict), the corrected empty-receipt classification (a later
same-cursor empty receipt yields only a correct_cursor suggestion), the
201/200 split with per-item results and counts, operation/conflict
replay rules, persistence and recovery, the HTTP precedence chain
(404 path shape, 400 length/body, 413 over-limit, 401, 403), the compact
ordered single-newline response, and the read-only guarantee of the
preflight. Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
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
    parse_replication_repairs_batch,
)

PLAN_PATH = "/v1/replication/repairs/plan"
APPLY_PATH = "/v1/replication/repairs/apply"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def interval(action: str, ack_id: str, start: int, end: int) -> dict:
    return {
        "action": action,
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def identity_fix(
    ack_id: str,
    position: int,
    expected: dict | None,
    observed: dict,
) -> dict:
    suggestion = {
        "action": "correct_identity",
        "ackId": ack_id,
        "location": {"position": position},
        "target": {"position": position},
        "expected": expected,
        "observed": observed,
    }
    return suggestion


class ParseRepairsBatchTests(unittest.TestCase):
    def base(self) -> dict:
        return {
            "peerId": "peer-a",
            "ackId": "exec-1",
            "expectedCheckpoint": 4,
            "expectedReceipts": "a" * 64,
            "suggestions": [interval("resend", "ack-2", 1, 2)],
        }

    def test_valid_interval_and_identity_entries_round_trip(self) -> None:
        document = self.base()
        document["suggestions"].append(
            identity_fix(
                "ack-3",
                3,
                identity("r3", "o3"),
                identity("r9", "bogus"),
            )
        )
        document["suggestions"].append(interval("correct_cursor", "ack-4", 2, 2))
        peer_id, ack_id, checkpoint, digest, suggestions = (
            parse_replication_repairs_batch(document)
        )
        self.assertEqual(peer_id, "peer-a")
        self.assertEqual(ack_id, "exec-1")
        self.assertEqual(checkpoint, 4)
        self.assertEqual(digest, "a" * 64)
        self.assertEqual([item["action"] for item in suggestions], [
            "resend",
            "correct_identity",
            "correct_cursor",
        ])

    def test_invalid_documents_raise(self) -> None:
        base = self.base()
        bad: list[object] = [
            b"{not json",
            [],
            "x",
            5,
            None,
            {},
            {"peerId": "peer-a"},
            {**base, "extra": 1},
            {**base, "peerId": ""},
            {**base, "peerId": 7},
            {**base, "ackId": ""},
            {**base, "expectedCheckpoint": -1},
            {**base, "expectedCheckpoint": True},
            {**base, "expectedCheckpoint": 1.0},
            {**base, "expectedCheckpoint": "4"},
            {**base, "expectedReceipts": "0" * 63},
            {**base, "expectedReceipts": "A" * 64},
            {**base, "suggestions": []},
            {**base, "suggestions": [{}]},
            {**base, "suggestions": [{"action": "teleport"}]},
            {**base, "suggestions": [{"action": "resend", "ackId": ""}]},
            {**base, "suggestions": [{"action": "resend", "ackId": "a",
                                      "location": {"start": 1}, "target": {"start": 1, "end": 2}}]},
            {**base, "suggestions": [interval("resend", "a", 2, 1)]},
            {**base, "suggestions": [interval("resend", "a", -1, 2)]},
            {**base, "suggestions": [interval("resend", "a", True, 2)]},
        ]
        for document in bad:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_repairs_batch(document)

    def test_invalid_identity_fix_raises(self) -> None:
        good = identity_fix("ack-3", 3, identity("r3", "o3"), identity("r9", "x"))
        for tweak in (
            lambda d: d.update(location={"position": -1}),
            lambda d: d.update(target={"position": 4}),
            lambda d: d.update(expected={"replicaId": "r3"}),
            lambda d: d.update(observed=identity("", "x")),
        ):
            document = self.base()
            entry = json.loads(json.dumps(good))
            tweak(entry)
            document["suggestions"] = [entry]
            with self.assertRaises(ValueError):
                parse_replication_repairs_batch(document)

    def test_duplicate_root_field_raises(self) -> None:
        raw = (
            '{"peerId":"p","peerId":"p","ackId":"a","expectedCheckpoint":0,'
            '"expectedReceipts":"' + "a" * 64 + '","suggestions":['
            '{"action":"resend","ackId":"a","location":{"start":0,"end":0},'
            '"target":{"start":0,"end":0}}]}'
        )
        with self.assertRaises(ValueError):
            parse_replication_repairs_batch(raw)


class RepairStoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed_operations(self, count: int, key: str = "k") -> None:
        for index in range(count):
            status = self.store.apply_operation(
                f"r{index}",
                operation(f"o{index}", key, f"v{index}", {f"r{index}": 1}),
            )
            self.assertIs(status, HTTPStatus.CREATED)

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.store.save_checkpoint(peer, cursor)
        self.assertIs(status, HTTPStatus.OK)

    def ack(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        status, error = self.store.acknowledge_operations(
            peer, ack_id, cursor, operations
        )
        self.assertIs(status, HTTPStatus.CREATED, error)

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def advice(self, peer: str = "peer-a") -> list[dict]:
        status, payload = self.store.get_replication_repairs(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        return [item for item in payload["suggestions"] if item["peer"] == peer]

    def receipts_digest(self, peer: str = "peer-a") -> str:
        committed = [
            (ack_id, receipt)
            for (receipt_peer, ack_id), receipt in self.store._acks.items()
            if receipt_peer == peer
        ]
        from semantic_state_engine.server import _repair_receipts_digest

        return _repair_receipts_digest(peer, committed)

    def suggestion_from_advice(self, item: dict) -> dict:
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        return suggestion

    def batch(
        self,
        suggestions: list[dict],
        *,
        peer: str = "peer-a",
        ack_id: str = "exec-1",
        checkpoint: int | None = None,
        digest: str | None = None,
    ) -> tuple:
        cursor = (
            self.store.get_checkpoint(peer)[1]["cursor"]
            if checkpoint is None
            else checkpoint
        )
        return (
            peer,
            ack_id,
            cursor,
            self.receipts_digest(peer) if digest is None else digest,
            suggestions,
        )


class RepairPreflightStoreTests(RepairStoreFixture):
    def test_preflight_reports_executable_and_predicted_boundary(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        item = self.advice()[0]
        status, payload = self.store.preflight_replication_repairs(
            *self.batch([self.suggestion_from_advice(item)])
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(list(payload), ["status", "peerId", "ackId", "results"])
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["peerId"], "peer-a")
        self.assertEqual(payload["ackId"], "exec-1")
        self.assertEqual(
            payload["results"],
            [
                {
                    "action": "resend",
                    "executable": True,
                    "boundary": {"start": 1, "end": 2},
                }
            ],
        )

    def test_preflight_is_strictly_read_only(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        suggestion = self.suggestion_from_advice(self.advice()[0])
        before = self.store.get_replication_repairs(0, 100)[1]
        metrics_before = self.store.get_metrics()
        for _ in range(3):
            status, payload = self.store.preflight_replication_repairs(
                *self.batch([suggestion])
            )
            self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(self.store.get_replication_repairs(0, 100)[1], before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_checkpoint("peer-a")[1]["cursor"], 4)
        self.assertNotIn(("peer-a", "exec-1"), self.store._repairs)

    def test_preflight_anchor_mismatch_is_apply_conflict(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        suggestion = interval("resend", "ack-x", 0, 1)
        # Unknown peer.
        status, payload = self.store.preflight_replication_repairs(
            *self.batch([suggestion], peer="nobody", checkpoint=0)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "apply_conflict"})
        # Wrong expected checkpoint.
        status, payload = self.store.preflight_replication_repairs(
            *self.batch([suggestion], checkpoint=1)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "apply_conflict"})
        # Wrong expected receipts digest.
        status, payload = self.store.preflight_replication_repairs(
            *self.batch([suggestion], digest="b" * 64)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "apply_conflict"})

    def test_preflight_reports_stale_item_as_not_executable(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        # A resend suggestion for a receipt that is not currently faulty.
        status, payload = self.store.preflight_replication_repairs(
            *self.batch([interval("resend", "ack-missing", 0, 1)])
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["results"],
            [{"action": "resend", "executable": False, "boundary": None}],
        )

    def test_preflight_out_of_order_is_apply_conflict(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", 4)
        self.seed_receipt(
            "peer-a", "ack-1", 2, [identity("r0", "o0"), identity("r1", "o1")]
        )
        self.seed_receipt("peer-a", "ack-2", 2, [])
        items = self.advice()
        # Only a cursor regression exists here; build an artificial
        # out-of-order batch (cursor before resend).
        suggestions = [
            interval("correct_cursor", "ack-2", 2, 2),
            interval("resend", "ack-9", 0, 1),
        ]
        status, payload = self.store.preflight_replication_repairs(
            *self.batch(suggestions)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "apply_conflict"})
        self.assertEqual(len(items), 1)


class RepairApplyStoreTests(RepairStoreFixture):
    def apply_advice(
        self,
        items: list[dict],
        *,
        ack_id: str = "exec-1",
        peer: str = "peer-a",
    ):
        suggestions = [self.suggestion_from_advice(item) for item in items]
        return self.store.apply_replication_repairs(
            *self.batch(suggestions, ack_id=ack_id, peer=peer)
        )

    def test_gap_resend_is_created_and_idempotent(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        item = self.advice()[0]
        status, payload, error = self.apply_advice([item])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["peerId"], "peer-a")
        self.assertEqual(payload["ackId"], "exec-1")
        self.assertEqual(payload["newExecutions"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            payload["results"],
            [{"action": "resend", "boundary": {"start": 1, "end": 2}}],
        )
        # The repair restored the cursor to the boundary end; here it was
        # already 4, so the checkpoint is unchanged but recorded.
        self.assertEqual(self.store.get_checkpoint("peer-a")[1]["cursor"], 4)
        # Replaying the same execution returns 200 and advances nothing.
        status2, payload2, error2 = self.apply_advice([item])
        self.assertIs(status2, HTTPStatus.OK)
        self.assertIsNone(error2)
        self.assertEqual(payload2["status"], "ok")
        self.assertEqual(payload2["newExecutions"], 0)
        self.assertEqual(payload2["replayed"], 1)
        self.assertEqual(payload2["results"], payload["results"])

    def test_repair_advances_checkpoint_to_boundary_end(self) -> None:
        self.seed_operations(4)
        # The peer's registered checkpoint sits behind the restored end.
        self.register("peer-a", 1)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        item = self.advice()[0]
        status, _, _ = self.apply_advice([item])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(self.store.get_checkpoint("peer-a")[1]["cursor"], 2)

    def test_same_ack_id_with_different_content_is_operation_conflict(self) -> None:
        self.seed_operations(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        item = self.advice()[0]
        status, _, _ = self.apply_advice([item], ack_id="exec-1")
        self.assertIs(status, HTTPStatus.CREATED)
        # A different suggestion set under the same execution id conflicts.
        different = [interval("resend", "ack-2", 3, 4)]
        status, payload, error = self.store.apply_replication_repairs(
            *self.batch(different, ack_id="exec-1")
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")
        self.assertEqual(payload, {})

    def test_expected_checkpoint_or_digest_mismatch_is_apply_conflict(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        suggestions = [interval("resend", "ack-x", 0, 1)]
        status, _, error = self.store.apply_replication_repairs(
            *self.batch(suggestions, checkpoint=0)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")
        status, _, error = self.store.apply_replication_repairs(
            *self.batch(suggestions, digest="c" * 64)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")
        self.assertNotIn(("peer-a", "exec-1"), self.store._repairs)

    def test_stale_location_is_repair_conflict_without_half_execution(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        # The receipt's mismatching identity sits at position 1, so the
        # current (correct_identity, ack-1) slot aligns to position 1.
        # A suggestion still naming that receipt's slot but the old
        # position 0 has a log location that no longer matches: a repair
        # conflict, not a position mismatch.
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        current = self.advice()
        self.assertEqual(current[0]["location"], {"position": 1})
        suggestions = [
            identity_fix("ack-1", 0, identity("r0", "o0"), identity("r9", "bogus"))
        ]
        status, _, error = self.store.apply_replication_repairs(
            *self.batch(suggestions)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "repair_conflict")
        self.assertNotIn(("peer-a", "exec-1"), self.store._repairs)
        self.assertEqual(self.store.get_checkpoint("peer-a")[1]["cursor"], 2)

    def test_missing_position_is_apply_conflict(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 2)
        # No faulty receipt at all -> the named resend slot does not exist.
        suggestions = [interval("resend", "ack-missing", 0, 1)]
        status, _, error = self.store.apply_replication_repairs(
            *self.batch(suggestions)
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")

    def test_batch_processes_all_four_actions_in_order(self) -> None:
        self.seed_operations(6)
        self.register("peer-a", 6)
        # ack-1: correct identities at positions 0 and 1, leaving an
        # identity mismatch pair; build mismatches + a gap + an empty
        # receipt regression.
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r9", "bogus0"), identity("r8", "bogus1")],
        )
        self.seed_receipt("peer-a", "ack-2", 4, [identity("r3", "o3")])  # gap [2,3)
        self.seed_receipt("peer-a", "ack-3", 4, [])  # same-cursor empty -> cursor fix
        items = self.advice()
        actions = [item["action"] for item in items]
        # Stable advice order lists gap then the two identity mismatches;
        # reorder into the execution's fixed order.
        order = {"resend": 0, "deduplicate": 1, "correct_identity": 2, "correct_cursor": 3}
        ordered = sorted(
            [self.suggestion_from_advice(item) for item in items],
            key=lambda entry: order[entry["action"]],
        )
        self.assertIn("resend", actions)
        self.assertIn("correct_identity", actions)
        self.assertIn("correct_cursor", actions)
        status, payload, error = self.store.apply_replication_repairs(
            *self.batch(ordered)
        )
        self.assertIs(status, HTTPStatus.CREATED, error)
        self.assertEqual(
            [result["action"] for result in payload["results"]],
            [entry["action"] for entry in ordered],
        )
        self.assertEqual(payload["newExecutions"], len(ordered))
        # The boundaries (3 and 4) sit below the registered cursor 6, so
        # the repair keeps the cursor at 6 while recording the execution.
        self.assertEqual(self.store.get_checkpoint("peer-a")[1]["cursor"], 6)

    def test_empty_receipt_head_is_legal_and_yields_no_suggestion(self) -> None:
        self.seed_operations(2)
        self.register("peer-a", 0)
        # An empty receipt at the chain head is legal and reports nothing.
        self.seed_receipt("peer-a", "ack-0", 0, [])
        self.assertEqual(self.advice(), [])
        status, payload = self.store.get_peer_receipts_audit("peer-a", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["audit"]["status"], "ok")


class RepairPersistenceTests(unittest.TestCase):
    def write_state(self, data_file: str) -> tuple[StateStore, dict]:
        store = StateStore(data_file=data_file)
        for index in range(4):
            store.apply_operation(
                f"r{index}",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
        store.save_checkpoint("peer-a", 4)
        store._acks[("peer-a", "ack-1")] = {
            "cursor": 1,
            "operations": [identity("r0", "o0")],
        }
        store._acks[("peer-a", "ack-2")] = {
            "cursor": 3,
            "operations": [identity("r2", "o2")],
        }
        status, payload = store.get_replication_repairs(0, 100)
        assert status is HTTPStatus.OK
        return store, payload

    def test_committed_repair_survives_restart_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store, repairs = self.write_state(data_file)
            item = repairs["suggestions"][0]
            suggestion = {
                "action": item["action"],
                "ackId": item["ackId"],
                "location": item["location"],
                "target": item["target"],
            }
            committed = [
                (ack_id, receipt)
                for (peer, ack_id), receipt in store._acks.items()
                if peer == "peer-a"
            ]
            from semantic_state_engine.server import _repair_receipts_digest

            digest = _repair_receipts_digest("peer-a", committed)
            status, first, error = store.apply_replication_repairs(
                "peer-a", "exec-1", 4, digest, [suggestion]
            )
            self.assertIs(status, HTTPStatus.CREATED, error)
            recovered = StateStore(data_file=data_file)
            # The identical execution replays against the recovered file.
            status, second, error = recovered.apply_replication_repairs(
                "peer-a", "exec-1", 4, digest, [suggestion]
            )
            self.assertIs(status, HTTPStatus.OK, error)
            self.assertEqual(second["results"], first["results"])
            self.assertEqual(recovered.get_checkpoint("peer-a")[1]["cursor"], 4)

    def test_persistence_failure_rolls_back_completely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store, repairs = self.write_state(data_file)
            item = repairs["suggestions"][0]
            suggestion = {
                "action": item["action"],
                "ackId": item["ackId"],
                "location": item["location"],
                "target": item["target"],
            }
            from semantic_state_engine.server import (
                PersistenceError,
                _repair_receipts_digest,
            )

            committed = [
                (ack_id, receipt)
                for (peer, ack_id), receipt in store._acks.items()
                if peer == "peer-a"
            ]
            digest = _repair_receipts_digest("peer-a", committed)
            original = store._persist_locked

            def fail_persist() -> None:
                raise PersistenceError("disk unavailable")

            store._persist_locked = fail_persist  # type: ignore[assignment]
            with self.assertRaises(PersistenceError):
                store.apply_replication_repairs("peer-a", "exec-1", 4, digest, [suggestion])
            store._persist_locked = original  # type: ignore[assignment]
            self.assertNotIn(("peer-a", "exec-1"), store._repairs)
            self.assertEqual(store.get_checkpoint("peer-a")[1]["cursor"], 4)
            # The request is safely retryable after the fault.
            status, payload, error = store.apply_replication_repairs(
                "peer-a", "exec-1", 4, digest, [suggestion]
            )
            self.assertIs(status, HTTPStatus.CREATED, error)


class RepairHttpFixture(unittest.TestCase):
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

    def raw_request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        if body is None:
            conn.request(method, path, headers=request_headers)
        else:
            request_headers.setdefault("Content-Type", "application/json")
            conn.request(
                method,
                path,
                body=body if isinstance(body, (bytes, str)) else json.dumps(body),
                headers=request_headers,
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def request(self, method, path, body=None, headers=None):
        status, payload, _, _ = self.raw_request(method, path, body, headers)
        return status, payload

    def seed(self, count: int = 4) -> None:
        for index in range(count):
            status, _ = self.request(
                "POST",
                f"/v1/replicas/r{index}/operations",
                operation(f"o{index}", "k", f"v{index}", {f"r{index}": 1}),
            )
            assert status == 201

    def register(self, peer: str, cursor: int) -> None:
        status, _ = self.request(
            "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor}
        )
        assert status == 200

    def seed_receipt(self, peer: str, ack_id: str, cursor: int, operations: list) -> None:
        self.server.store._acks[(peer, ack_id)] = {
            "cursor": cursor,
            "operations": [dict(item) for item in operations],
        }

    def advice(self, peer: str = "peer-a") -> list[dict]:
        status, payload = self.request(
            "GET", "/v1/replication/repairs?after=0&limit=100"
        )
        assert status == 200
        return [item for item in payload["suggestions"] if item["peer"] == peer]

    def receipt_digest(self, peer: str = "peer-a") -> str:
        status, payload = self.request(
            "GET", f"/v1/sync/peers/{peer}/receipts?after=0&limit=100"
        )
        assert status == 200
        return payload["digest"]

    @staticmethod
    def suggestion(item: dict) -> dict:
        suggestion = {
            "action": item["action"],
            "ackId": item["ackId"],
            "location": item["location"],
            "target": item["target"],
        }
        if item["action"] == "correct_identity":
            suggestion["expected"] = item["expected"]
            suggestion["observed"] = item["observed"]
        return suggestion

    def body(
        self,
        suggestions: list[dict],
        *,
        peer: str = "peer-a",
        ack_id: str = "exec-1",
        checkpoint: int | None = None,
        digest: str | None = None,
    ) -> dict:
        cursor = checkpoint
        if cursor is None:
            status, payload = self.request(
                "GET", f"/v1/sync/peers/{peer}/checkpoint"
            )
            assert status == 200
            cursor = payload["cursor"]
        return {
            "peerId": peer,
            "ackId": ack_id,
            "expectedCheckpoint": cursor,
            "expectedReceipts": digest or self.receipt_digest(peer),
            "suggestions": suggestions,
        }


class RepairPreflightHttpTests(RepairHttpFixture):
    def test_preflight_success_body_is_compact_ordered_with_newline(self) -> None:
        self.seed(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        document = self.body([self.suggestion(self.advice()[0])])
        status, payload, raw, headers = self.raw_request("POST", PLAN_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ["status", "peerId", "ackId", "results"])
        self.assertEqual(payload["status"], "planned")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))
        result = payload["results"][0]
        self.assertEqual(list(result), ["action", "executable", "boundary"])
        for value in result["boundary"].values():
            self.assertIsInstance(value, int)
            self.assertNotIsInstance(value, bool)

    def test_preflight_does_not_commit(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        document = self.body([interval("resend", "ack-missing", 0, 1)])
        status, payload = self.request("POST", PLAN_PATH, document)
        self.assertEqual(status, 200)
        self.assertFalse(payload["results"][0]["executable"])
        self.assertNotIn(("peer-a", "exec-1"), self.server.store._repairs)
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 2)

    def test_invalid_body_is_400(self) -> None:
        self.register("peer-a", 0)
        for document in ({}, {"peerId": "peer-a"}, b"{not json", {"a": 1}):
            status, payload, raw, _ = self.raw_request("POST", PLAN_PATH, document)
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})
            self.assertTrue(raw.endswith(b"\n"))

    def test_query_parameter_is_400(self) -> None:
        self.register("peer-a", 0)
        # A declared-but-unwritten body: the query check precedes the
        # body read, so the client does not push bytes the server
        # refuses to read.
        status, payload, _, _ = self.raw_request(
            "POST", f"{PLAN_PATH}?x=1", headers={"Content-Length": "0"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_path_shape_mismatches_are_404(self) -> None:
        self.register("peer-a", 0)
        for path in (
            "/v1/replication/repairs",
            "/v1/replication/repairs/",
            "/v1/replication/repairs/plan/",
            "/v1/replication/repairs/plan/extra",
            "/v1/replication/repairs/other",
        ):
            # No body: an unknown route rejects before reading it, so the
            # client never writes into a closing socket.
            status, payload, _, _ = self.raw_request("POST", f"{path}?x=%ZZ")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_plan_route_is_404(self) -> None:
        status, payload, _, _ = self.raw_request("GET", PLAN_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class RepairApplyHttpTests(RepairHttpFixture):
    def test_apply_created_then_replayed_ok(self) -> None:
        self.seed(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        document = self.body([self.suggestion(self.advice()[0])])
        status, payload, raw, _ = self.raw_request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        self.assertEqual(
            list(payload),
            ["status", "peerId", "ackId", "results", "newExecutions", "replayed"],
        )
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["newExecutions"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        status, payload = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["newExecutions"], 0)
        self.assertEqual(payload["replayed"], 1)

    def test_operation_conflict_is_409(self) -> None:
        self.seed(4)
        self.register("peer-a", 4)
        self.seed_receipt("peer-a", "ack-1", 1, [identity("r0", "o0")])
        self.seed_receipt("peer-a", "ack-2", 3, [identity("r2", "o2")])
        item = self.advice()[0]
        document = self.body([self.suggestion(item)])
        status, _ = self.request("POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        changed = dict(document)
        changed["expectedCheckpoint"] = 3
        status, payload, _, _ = self.raw_request("POST", APPLY_PATH, changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_apply_conflict_on_anchor_mismatch(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        document = self.body([interval("resend", "ack-x", 0, 1)], checkpoint=0)
        status, payload, _, _ = self.raw_request("POST", APPLY_PATH, document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "apply_conflict"})

    def test_repair_conflict_on_stale_identity(self) -> None:
        self.seed(2)
        self.register("peer-a", 2)
        # The current mismatch sits at position 1; a suggestion still
        # naming the same receipt slot but the older position 0 is a
        # repair conflict (the log location moved).
        self.seed_receipt(
            "peer-a",
            "ack-1",
            2,
            [identity("r0", "o0"), identity("r9", "bogus")],
        )
        document = self.body(
            [identity_fix("ack-1", 0, identity("r0", "o0"), identity("r9", "bogus"))]
        )
        status, payload, _, _ = self.raw_request("POST", APPLY_PATH, document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "repair_conflict"})

    def test_invalid_body_is_400(self) -> None:
        self.register("peer-a", 0)
        for document in ({}, b"{bad", {"peerId": "peer-a"}):
            status, payload, _, _ = self.raw_request("POST", APPLY_PATH, document)
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/repairs",
            "/v1/replication/repairs/apply/",
            "/v1/replication/repairs/apply/extra",
        ):
            status, payload, _, _ = self.raw_request("POST", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"})

    def test_query_parameter_is_400(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", f"{APPLY_PATH}?x=1", headers={"Content-Length": "0"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class RepairRequestLimitsTests(RepairHttpFixture):
    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_over_limit_declaration_is_413_without_reading(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", APPLY_PATH)
        conn.putheader("Content-Length", str(1_048_577))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read()), {"error": "payload_too_large"})
        conn.close()


class RepairAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-repairs2-auth-")
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

    def post(self, port, path, headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body is None:
            conn.request("POST", path, headers=dict(headers or {}))
        else:
            conn.request(
                "POST",
                path,
                body=body,
                headers={**dict(headers or {}), "Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw), response_headers

    def test_single_token_required(self) -> None:
        for path in (PLAN_PATH, APPLY_PATH):
            status, payload, headers = self.post(self.single_port, path)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertEqual(headers["WWW-Authenticate"], "Bearer")
            status, payload, _ = self.post(
                self.single_port,
                path,
                [("Authorization", "Bearer s3cret-token")],
                b"{}",
            )
            # Authenticated; body {} is a 400, never 401/403.
            self.assertEqual(status, 400)

    def test_plan_needs_read_apply_needs_write(self) -> None:
        # A write-only token cannot reach the read-only preflight.
        status, payload, headers = self.post(
            self.scope_port, PLAN_PATH, [("Authorization", f"Bearer {WRITE_TOKEN}")]
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        # A read-only token cannot reach the committing apply.
        status, payload, headers = self.post(
            self.scope_port, APPLY_PATH, [("Authorization", f"Bearer {READ_TOKEN}")]
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        # A read token reaches the preflight (body {} -> 400).
        status, _, _ = self.post(
            self.scope_port,
            PLAN_PATH,
            [("Authorization", f"Bearer {READ_TOKEN}")],
            b"{}",
        )
        self.assertEqual(status, 400)
        # A write token reaches the apply (body {} -> 400).
        status, _, _ = self.post(
            self.scope_port,
            APPLY_PATH,
            [("Authorization", f"Bearer {WRITE_TOKEN}")],
            b"{}",
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
