"""Tests for the executable replica repair batch endpoint.

The endpoint is::

    POST /v1/replication/apply

with a body of exactly ``{"replicaId", "snapshot", "expectedDigest",
"actions"}``: the remote replica's identifier and complete candidate
snapshot under the comparison's constraints, the local candidate digest the
plan was computed against, and an ordered list of 1-100 actions. Each
action carries ``action`` plus the candidate identity, value, and clock it
acts on; a ``semantic_resolution`` action additionally carries the merged
value's expected conflicting ``candidates`` set. Actions are validated in
request order against a staged view:

- ``send_local`` only confirms the local side of the plan direction (the
  identity must be a current local candidate and the remote side must
  still need it) and commits nothing, so it counts as replayed;
- ``fetch_remote`` imports the named remote candidate as one operation;
- ``semantic_resolution`` commits a manual resolution for the key.

A committing batch requires ``expectedDigest`` to match the current
committed candidate snapshot, commits atomically, and answers 201; a pure
replay answers 200. Known identities with different content are
``operation_conflict``; a stale direction or changed candidate is
``apply_conflict``; a missing/non-conflicting/mismatched resolution target
is ``resolution_conflict``.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for apply semantics. Only the
Python standard library is used.
"""

from __future__ import annotations

import hashlib
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

import semantic_state_engine.server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _verification_digest_input,
    load_scope_policy,
    parse_replication_apply_payload,
)

APPLY_PATH = "/v1/replication/apply"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def action(
    kind: str,
    key: str,
    replica_id: str,
    operation_id: str,
    value: str,
    clock: dict,
    candidates: list | None = None,
) -> dict:
    entry = {
        "action": kind,
        "key": key,
        "replicaId": replica_id,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
    }
    if candidates is not None:
        entry["candidates"] = candidates
    return entry


def identities(*pairs: tuple[str, str]) -> list:
    return [{"replicaId": r, "operationId": o} for r, o in pairs]


def digest_of(store: StateStore) -> str:
    return hashlib.sha256(_verification_digest_input(store._candidates)).hexdigest()


def body(
    replica_id: str = "remote",
    snapshot: dict | None = None,
    expected_digest: str = "0" * 64,
    actions: list | None = None,
) -> dict:
    return {
        "replicaId": replica_id,
        "snapshot": {} if snapshot is None else snapshot,
        "expectedDigest": expected_digest,
        "actions": [] if actions is None else actions,
    }


class ApplyParserTests(unittest.TestCase):
    def parse(self, document):
        return parse_replication_apply_payload(document)

    def test_valid_body_parses(self) -> None:
        document = body(
            "rb",
            {"color": [candidate("r2", "o2", "red", {"r2": 1})]},
            "a" * 64,
            [
                action("fetch_remote", "color", "r2", "o2", "red", {"r2": 1}),
                action(
                    "semantic_resolution",
                    "color",
                    "r3",
                    "fix",
                    "merged",
                    {"r2": 1, "r3": 1},
                    identities(("r2", "o2")),
                ),
            ],
        )
        replica_id, snapshot, expected_digest, actions = self.parse(
            json.dumps(document)
        )
        self.assertEqual(replica_id, "rb")
        self.assertEqual(expected_digest, "a" * 64)
        self.assertEqual([a["action"] for a in actions], ["fetch_remote", "semantic_resolution"])
        self.assertEqual(actions[1]["candidates"], [{"replicaId": "r2", "operationId": "o2"}])
        self.assertEqual(snapshot["color"][0]["value"], "red")

    def test_missing_or_unknown_root_fields_are_rejected(self) -> None:
        valid = body("rb", {}, "a" * 64, [action("send_local", "k", "r1", "o1", "v", {"r1": 1})])
        for document in (
            {},
            {"replicaId": "rb"},
            {**valid, "x": 1},
            {k: v for k, v in valid.items() if k != "expectedDigest"},
            {k: v for k, v in valid.items() if k != "actions"},
            {k: v for k, v in valid.items() if k != "snapshot"},
            [],
            "text",
        ):
            with self.assertRaises(ValueError, msg=repr(document)):
                self.parse(document)

    def test_malformed_json_and_duplicated_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.parse(b"{not json")
        with self.assertRaises(ValueError):
            self.parse('{"replicaId":"a","replicaId":"b","snapshot":{},"expectedDigest":"%s","actions":[]}' % ("a" * 64))
        with self.assertRaises(ValueError):
            self.parse(b"\xff\xfe")

    def test_digest_must_be_64_lowercase_hex(self) -> None:
        valid_action = action("send_local", "k", "r1", "o1", "v", {"r1": 1})
        for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 5, None, ["a" * 64]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.parse(body("rb", {}, bad, [valid_action]))

    def test_action_list_bounds_and_shape(self) -> None:
        valid_action = action("send_local", "k", "r1", "o1", "v", {"r1": 1})
        for bad_actions in (
            [],
            [valid_action] * 0,
            "x",
            [{}],
            [{"action": "send_local"}],
            [{**valid_action, "extra": 1}],
            [{k: v for k, v in valid_action.items() if k != "clock"}],
            [{**valid_action, "candidates": identities(("r1", "o1"))}],
        ):
            with self.assertRaises(ValueError, msg=repr(bad_actions)):
                self.parse(body("rb", {}, "a" * 64, bad_actions))
        too_many = [
            action("send_local", "k", "r1", f"o{i}", "v", {"r1": 1}) for i in range(101)
        ]
        with self.assertRaises(ValueError):
            self.parse(body("rb", {}, "a" * 64, too_many))

    def test_unknown_action_direction_is_rejected(self) -> None:
        for kind in ("", "send", "fetch", "semantic", "SEND_LOCAL", "resolve", 1, None):
            entry = action("send_local", "k", "r1", "o1", "v", {"r1": 1})
            entry["action"] = kind
            with self.assertRaises(ValueError, msg=repr(kind)):
                self.parse(body("rb", {}, "a" * 64, [entry]))

    def test_duplicate_action_identity_is_rejected(self) -> None:
        first = action("send_local", "k", "r1", "o1", "v", {"r1": 1})
        second = action("fetch_remote", "k2", "r1", "o1", "v", {"r1": 1})
        with self.assertRaises(ValueError):
            self.parse(body("rb", {}, "a" * 64, [first, second]))

    def test_illegal_entry_fields_and_clocks_are_rejected(self) -> None:
        for entry in (
            action("send_local", "", "r1", "o1", "v", {"r1": 1}),
            action("send_local", "k", "", "o1", "v", {"r1": 1}),
            action("send_local", "k", "r1", "", "v", {"r1": 1}),
            action("send_local", "k", "r1", "o1", "", {"r1": 1}),
            action("send_local", "k", "r1", "o1", "v", {}),
            action("send_local", "k", "r1", "o1", "v", {"r2": 1}),
            action("send_local", "k", "r1", "o1", "v", {"r1": -1}),
            action("send_local", "k", "r1", "o1", "v", {"r1": 1.0}),
            action("send_local", "k", "r1", "o1", "v", {"r1": True}),
        ):
            with self.assertRaises(ValueError, msg=repr(entry)):
                self.parse(body("rb", {}, "a" * 64, [entry]))

    def test_resolution_requires_a_valid_candidate_set(self) -> None:
        base = action(
            "semantic_resolution", "k", "r3", "f", "m", {"r3": 1},
            identities(("r1", "o1")),
        )
        for candidates in (
            [],
            "x",
            [{}],
            [{"replicaId": "r1"}],
            [{"replicaId": "r1", "operationId": "o1", "x": 1}],
            identities(("r1", "o1"), ("r1", "o1")),
        ):
            entry = {**base, "candidates": candidates}
            with self.assertRaises(ValueError, msg=repr(candidates)):
                self.parse(body("rb", {}, "a" * 64, [entry]))
        # send/fetch entries must not carry a candidate set.
        fetch = action("fetch_remote", "k", "r1", "o1", "v", {"r1": 1})
        fetch["candidates"] = identities(("r1", "o1"))
        with self.assertRaises(ValueError):
            self.parse(body("rb", {}, "a" * 64, [fetch]))

    def test_snapshot_uses_the_comparison_constraints(self) -> None:
        valid_action = action("fetch_remote", "k", "r1", "o1", "v", {"r1": 1})
        for snapshot in (
            [],
            {"k": []},
            {"k": [{}]},
            {"k": [candidate("r1", "o1", "v", {"r1": 1.0})]},
            {"k": [candidate("r1", "o1", "v", {"r2": 1})]},
            {"k": [candidate("r1", "o1", "a", {"r1": 1}), candidate("r1", "o1", "b", {"r1": 2})]},
        ):
            with self.assertRaises(ValueError, msg=repr(snapshot)):
                self.parse(body("rb", snapshot, "a" * 64, [valid_action]))


class ApplyStoreTests(unittest.TestCase):
    """Apply semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def apply(self, document_actions, snapshot=None, expected=None, replica_id="rb"):
        if expected is None:
            expected = digest_of(self.store)
        return self.store.apply_replication_plan(
            replica_id, {} if snapshot is None else snapshot, expected, document_actions
        )

    def test_fetch_remote_imports_the_candidate(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        status, results, accepted, replayed, error = self.apply(
            [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})], snapshot
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual((accepted, replayed), (1, 0))
        self.assertEqual(
            results,
            [{"action": "fetch_remote", "key": "k", "replicaId": "r2",
              "operationId": "o2", "value": "red"}],
        )
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "red")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_fetch_replay_is_idempotent_even_with_stale_digest(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        entries = [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})]
        first_digest = digest_of(self.store)
        status, _, _, _, _ = self.apply(entries, snapshot, first_digest)
        self.assertIs(status, HTTPStatus.CREATED)
        # The first apply moved the digest; an identical retry is still 200.
        status, results, accepted, replayed, error = self.apply(entries, snapshot, first_digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["value"], "red")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_fetch_with_changed_remote_candidate_is_apply_conflict(self) -> None:
        entries = [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})]
        for snapshot in (
            {},
            {"k": [candidate("r2", "o2", "blue", {"r2": 1})]},
            {"k": [candidate("r2", "o2", "red", {"r2": 2})]},
            {"other": [candidate("r2", "o2", "red", {"r2": 1})]},
        ):
            status, _, _, _, error = self.apply(entries, snapshot)
            self.assertIs(status, HTTPStatus.CONFLICT, snapshot)
            self.assertEqual(error, "apply_conflict", snapshot)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 0)

    def test_send_local_confirms_and_commits_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        status, results, accepted, replayed, error = self.apply(
            [action("send_local", "k", "r1", "o1", "blue", {"r1": 1})]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["action"], "send_local")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_send_local_direction_failures_are_apply_conflict(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 2}))
        entry = action("send_local", "k", "r1", "o1", "blue", {"r1": 2})
        # The remote already holds the identity converged.
        converged = {"k": [candidate("r1", "o1", "blue", {"r1": 2})]}
        # The remote holds a conflicting value.
        conflicted = {"k": [candidate("r1", "o1", "red", {"r1": 1})]}
        # The remote holds a clock the local one does not dominate.
        concurrent = {"k": [candidate("r1", "o1", "blue", {"r9": 1})]}
        for snapshot in (converged, conflicted, concurrent):
            status, _, _, _, error = self.apply([entry], snapshot)
            self.assertIs(status, HTTPStatus.CONFLICT, snapshot)
            self.assertEqual(error, "apply_conflict", snapshot)
        # An identity the local replica never held.
        ghost = action("send_local", "k", "r9", "o9", "x", {"r9": 1})
        status, _, _, _, error = self.apply([ghost], {})
        self.assertEqual(error, "apply_conflict")
        # The candidate was overwritten by a newer local write.
        self.store.apply_operation("r1", operation("o2", "k", "green", {"r1": 3}))
        status, _, _, _, error = self.apply([entry], {})
        self.assertEqual(error, "apply_conflict")

    def test_send_local_with_dominated_remote_clock_holds(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 2}))
        snapshot = {"k": [candidate("r1", "o1", "blue", {"r1": 1})]}
        status, _, accepted, replayed, _ = self.apply(
            [action("send_local", "k", "r1", "o1", "blue", {"r1": 2})], snapshot
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))

    def test_known_identity_with_different_content_is_operation_conflict(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        for entry in (
            action("send_local", "k", "r1", "o1", "red", {"r1": 1}),
            action("send_local", "k", "r1", "o1", "blue", {"r1": 2}),
            action("send_local", "other", "r1", "o1", "blue", {"r1": 1}),
            action("fetch_remote", "k", "r1", "o1", "red", {"r1": 1}),
            action("semantic_resolution", "k", "r1", "o1", "red", {"r1": 1},
                   identities(("r1", "o1"))),
        ):
            status, _, _, _, error = self.apply([entry], {})
            self.assertIs(status, HTTPStatus.CONFLICT, entry)
            self.assertEqual(error, "operation_conflict", entry)

    def test_semantic_resolution_commits_and_resolves(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        entry = action(
            "semantic_resolution", "k", "r3", "fix-1", "merged",
            {"r1": 1, "r2": 1, "r3": 1},
            identities(("r1", "o1"), ("r2", "o2")),
        )
        status, results, accepted, replayed, error = self.apply([entry])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (1, 0))
        self.assertEqual(results[0]["value"], "merged")
        status, state = self.store.get_state("k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "merged")
        # Identical replay: 200, no new record, stale digest tolerated.
        status, _, accepted, replayed, _ = self.apply([entry], expected="0" * 64)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 3)

    def test_semantic_resolution_state_failures_are_resolution_conflict(self) -> None:
        entry = action(
            "semantic_resolution", "k", "r3", "fix-1", "merged", {"r3": 1},
            identities(("r1", "o1")),
        )
        # Missing key.
        status, _, _, _, error = self.apply([entry])
        self.assertEqual(error, "resolution_conflict")
        # Key not in conflict.
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        status, _, _, _, error = self.apply([entry])
        self.assertEqual(error, "resolution_conflict")
        # Candidate set mismatch.
        self.store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        status, _, _, _, error = self.apply([entry])
        self.assertEqual(error, "resolution_conflict")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 2)

    def test_semantic_resolution_non_dominating_clock_is_invalid_request(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        entry = action(
            "semantic_resolution", "k", "r3", "fix-1", "merged", {"r3": 1},
            identities(("r1", "o1"), ("r2", "o2")),
        )
        with self.assertRaises(ValueError):
            self.apply([entry])
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 2)

    def test_digest_guard_rejects_committing_batches(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        entries = [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})]
        status, _, _, _, error = self.apply(entries, snapshot, expected="f" * 64)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 0)
        # A pure send-only batch commits nothing and skips the digest guard.
        self.store.apply_operation("r1", operation("o1", "j", "v", {"r1": 1}))
        status, _, _, _, _ = self.apply(
            [action("send_local", "j", "r1", "o1", "v", {"r1": 1})], {}, expected="f" * 64
        )
        self.assertIs(status, HTTPStatus.OK)

    def test_mixed_batch_applies_in_order_and_counts(self) -> None:
        self.store.apply_operation("r1", operation("o1", "a", "blue", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "b", "x", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "b", "y", {"r3": 1}))
        snapshot = {
            "b": [candidate("r4", "o4", "z", {"r4": 1})],
            "c": [candidate("r5", "o5", "w", {"r5": 1})],
        }
        entries = [
            action("send_local", "a", "r1", "o1", "blue", {"r1": 1}),
            action("fetch_remote", "c", "r5", "o5", "w", {"r5": 1}),
            action(
                "semantic_resolution", "b", "r6", "fix", "merged",
                {"r2": 1, "r3": 1, "r6": 1},
                identities(("r2", "o2"), ("r3", "o3")),
            ),
        ]
        status, results, accepted, replayed, error = self.apply(entries, snapshot)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (2, 1))
        self.assertEqual(
            [r["action"] for r in results],
            ["send_local", "fetch_remote", "semantic_resolution"],
        )
        self.assertEqual(results[2]["value"], "merged")
        _, state_b = self.store.get_state("b")
        self.assertEqual(state_b["value"], "merged")
        _, state_c = self.store.get_state("c")
        self.assertEqual(state_c["value"], "w")

    def test_failed_batch_changes_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "a", "blue", {"r1": 1}))
        before = digest_of(self.store)
        snapshot = {"c": [candidate("r5", "o5", "w", {"r5": 1})]}
        entries = [
            action("fetch_remote", "c", "r5", "o5", "w", {"r5": 1}),
            action(
                "semantic_resolution", "missing", "r6", "fix", "m", {"r6": 1},
                identities(("r1", "o1")),
            ),
        ]
        status, _, _, _, error = self.apply(entries, snapshot)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "resolution_conflict")
        self.assertEqual(digest_of(self.store), before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, _ = self.store.get_state("c")
        self.assertIs(status, HTTPStatus.NOT_FOUND)


class ApplyHttpServerTests(unittest.TestCase):
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

    def raw_request(self, method: str, path: str, body: object = None) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, bytes) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _ = self.raw_request(method, path, body)
        return status, payload

    def apply(self, document: dict, path: str = APPLY_PATH) -> tuple[int, dict]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def local_digest(self) -> str:
        status, payload = self.request("GET", "/v1/replication/snapshot")
        self.assertEqual(status, 200)
        return payload["candidateDigest"]

    def test_fetch_apply_created_then_replayed_over_http(self) -> None:
        snapshot = {"color": [candidate("r2", "op-2", "red", {"r2": 1})]}
        document = body(
            "replica-b", snapshot, self.local_digest(),
            [action("fetch_remote", "color", "r2", "op-2", "red", {"r2": 1})],
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["replicaId"], "replica-b")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(
            payload["actions"],
            [{"action": "fetch_remote", "key": "color", "operationId": "op-2",
              "replicaId": "r2", "value": "red"}],
        )
        # The imported candidate is visible through the ordinary reads.
        status, state = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "red")
        # An identical retry is a replay even though the digest moved.
        status, payload = self.apply(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)

    def test_response_shape_headers_and_trailing_newline(self) -> None:
        document = body("rb", {}, self.local_digest(),
                        [action("send_local", "k", "r1", "o1", "v", {"r1": 1})])
        # send_local of an unknown identity conflicts; use a committed one.
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document["expectedDigest"] = self.local_digest()
        status, payload, raw = self.raw_request("POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload), {"status", "replicaId", "actions", "accepted", "replayed"}
        )
        self.assertEqual(
            set(payload["actions"][0]),
            {"action", "key", "replicaId", "operationId", "value"},
        )
        for name in ("accepted", "replayed"):
            self.assertIs(type(payload[name]), int, name)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(
            raw[:-1].decode("utf-8"),
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )

    def test_semantic_resolution_over_http(self) -> None:
        self.post_operation("r1", operation("op-1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("op-2", "color", "red", {"r2": 1}))
        document = body(
            "rb", {}, self.local_digest(),
            [action(
                "semantic_resolution", "color", "r3", "fix-1", "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                identities(("r1", "op-1"), ("r2", "op-2")),
            )],
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        _, state = self.request("GET", "/v1/states/color")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "merged")
        # The resolution flows through the sync export like any operation.
        _, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync_payload["operations"]), 3)

    def test_digest_mismatch_is_409_apply_conflict(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        document = body(
            "rb", snapshot, "f" * 64,
            [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})],
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "apply_conflict"})
        status, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_operation_and_resolution_conflicts_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        digest = self.local_digest()
        document = body("rb", {}, digest,
                        [action("send_local", "k", "r1", "o1", "other", {"r1": 1})])
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        document = body(
            "rb", {}, digest,
            [action("semantic_resolution", "k", "r3", "f", "m", {"r3": 1},
                    identities(("r1", "o1")))],
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_invalid_bodies_are_400(self) -> None:
        valid_action = action("send_local", "k", "r1", "o1", "v", {"r1": 1})
        for document in (
            {},
            {"replicaId": "rb"},
            body("rb", {}, "a" * 64, []),
            body("rb", {}, "not-a-digest", [valid_action]),
            body("rb", {}, "a" * 64, [{**valid_action, "action": "send"}]),
            body("rb", {}, "a" * 64, [valid_action, valid_action]),
            body("", {}, "a" * 64, [valid_action]),
            {**body("rb", {}, "a" * 64, [valid_action]), "x": 1},
        ):
            status, payload = self.apply(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)

    def test_malformed_json_body_is_400(self) -> None:
        status, payload, _ = self.raw_request("POST", APPLY_PATH, b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_any_query_parameter_is_400(self) -> None:
        document = body("rb", {}, "a" * 64,
                        [action("send_local", "k", "r1", "o1", "v", {"r1": 1})])
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.apply(document, APPLY_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_path_shape_mismatches_are_404(self) -> None:
        document = body("rb", {}, "a" * 64,
                        [action("send_local", "k", "r1", "o1", "v", {"r1": 1})])
        for path in (
            "/v1/replication/apply/extra",
            "/v1/replication",
            "/v1/replication/apply/",
            "/v1/replication/apply2",
            "/v1/replication/apply//",
        ):
            status, payload = self.apply(document, path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_and_body_checks(self) -> None:
        status, payload, _ = self.raw_request(
            "POST", "/v1/replication/apply/extra?x=1", {"not": "valid"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", APPLY_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejected_requests_change_no_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_digest = self.request("GET", "/v1/verification/digest")
        snapshot = {"k2": [candidate("r2", "o2", "w", {"r2": 1})]}
        for document in (
            body("rb", snapshot, "f" * 64,
                 [action("fetch_remote", "k2", "r2", "o2", "w", {"r2": 1})]),
            body("rb", {}, self.local_digest(),
                 [action("fetch_remote", "k2", "r2", "o2", "w", {"r2": 1})]),
            body("rb", {}, self.local_digest(), []),
        ):
            self.apply(document)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)


class ApplyHttpRequestLimitTests(unittest.TestCase):
    """The apply route keeps the shared Content-Length contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

        cls.auth_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.auth_thread = threading.Thread(
            target=cls.auth_server.serve_forever, daemon=True
        )
        cls.auth_thread.start()
        cls.auth_port = cls.auth_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.auth_server.shutdown()
        cls.auth_server.server_close()
        cls.thread.join(timeout=5)
        cls.auth_thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()
        self.auth_server.store = type(self.auth_server.store)()

    def post_raw(self, port: int, path: str, headers: list, body: bytes = b""):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, APPLY_PATH, [("Content-Length", value)], b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw(
            self.port,
            APPLY_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", APPLY_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        status, payload = self.post_raw(
            self.auth_port,
            APPLY_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        body_bytes = b"x" * MAX_BODY_BYTES
        status, payload = self.post_raw(
            self.port,
            APPLY_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class ApplyHttpAuthTests(unittest.TestCase):
    """The apply endpoint authenticates as a state-changing write endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-apply-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]

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

    def setUp(self) -> None:
        self.single_server.store = type(self.single_server.store)()
        self.scope_server.store = type(self.scope_server.store)()

    def request(self, port: int, method: str, path: str, body: object = None,
                auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, challenge

    def document(self) -> dict:
        return body("rb", {}, "a" * 64,
                    [action("send_local", "k", "r1", "o1", "v", {"r1": 1})])

    def test_single_token_mode_requires_bearer_token(self) -> None:
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", APPLY_PATH, self.document(), auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", APPLY_PATH, self.document(), auth="Bearer sekret"
        )
        self.assertEqual(status, 409)  # send_local of an unknown identity
        self.assertEqual(payload, {"error": "apply_conflict"})

    def test_scope_mode_requires_write_or_admin(self) -> None:
        status, payload, challenge = self.request(
            self.scope_port, "POST", APPLY_PATH, self.document(), auth="Bearer reader"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer writer", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", APPLY_PATH, self.document(), auth=token
            )
            self.assertEqual(status, 409, token)
            self.assertEqual(payload, {"error": "apply_conflict"}, token)

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")


class ApplyHttpPersistenceTests(unittest.TestCase):
    """With --data-file the apply commits atomically and recovers."""

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

    def request(
        self, server: SemanticStateServer, method: str, path: str, body: object = None
    ):
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

    def local_digest(self, server: SemanticStateServer) -> str:
        status, payload = self.request(server, "GET", "/v1/replication/snapshot")
        self.assertEqual(status, 200)
        return payload["candidateDigest"]

    def test_apply_is_durable_and_recovers_identically(self) -> None:
        server = self.start_server()
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        document = body(
            "rb", snapshot, self.local_digest(server),
            [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})],
        )
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "red")
        # The replay is answered exactly as before the restart.
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 1)
        # Different content under the same identity still conflicts.
        changed = body(
            "rb", snapshot, self.local_digest(server),
            [action("fetch_remote", "k", "r2", "o2", "blue", {"r2": 1})],
        )
        status, payload = self.request(server, "POST", APPLY_PATH, changed)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_persistence_failure_is_500_and_rolls_back(self) -> None:
        server = self.start_server()
        before = self.data_file.read_bytes()
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        document = body(
            "rb", snapshot, self.local_digest(server),
            [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})],
        )
        with patch.object(
            StateStore, "_persist_locked",
            side_effect=server_module.PersistenceError("disk gone"),
        ):
            status, payload = self.request(server, "POST", APPLY_PATH, document)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File, memory, and the identity index are exactly as before.
        self.assertEqual(self.data_file.read_bytes(), before)
        status, _ = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(status, 404)
        _, metrics = self.request(server, "GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 0)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # The same batch commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)

    def test_rejected_batches_create_no_temporary_file(self) -> None:
        server = self.start_server()
        before_entries = set(os.listdir(self.tmp))
        before_bytes = self.data_file.read_bytes()
        snapshot = {"k": [candidate("r2", "o2", "red", {"r2": 1})]}
        documents = [
            # Digest mismatch on a committing batch.
            body("rb", snapshot, "f" * 64,
                 [action("fetch_remote", "k", "r2", "o2", "red", {"r2": 1})]),
            # Direction failure.
            body("rb", {}, self.local_digest(server),
                 [action("send_local", "k", "r9", "o9", "x", {"r9": 1})]),
            # Malformed.
            body("rb", {}, "bad-digest",
                 [action("send_local", "k", "r9", "o9", "x", {"r9": 1})]),
        ]
        for document in documents:
            self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(set(os.listdir(self.tmp)), before_entries)
        self.assertEqual(self.data_file.read_bytes(), before_bytes)


if __name__ == "__main__":
    unittest.main()
