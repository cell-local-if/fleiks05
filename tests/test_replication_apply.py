"""Tests for the executable replica repair batch endpoint.

The endpoint is::

    POST /v1/replication/apply

with a body of exactly ``{"replicaId", "expectedLocalDigest", "snapshot",
"actions"}`` — the remote replica identifier, the digest the caller
expects the current committed local candidate snapshot to have, the
remote's complete candidate snapshot (the comparison's constraints), and
the ordered repair actions (``send_local``, ``fetch_remote``, or
``semantic_resolution``). The batch is validated in request order against
a staged view and commits atomically:

- a digest mismatch is 409 ``apply_conflict`` and changes nothing;
- a send/fetch whose plan direction no longer holds or whose candidate
  changed is 409 ``apply_conflict``;
- a semantic repair on a missing key, a key no longer in conflict, or a
  mismatched candidate set is 409 ``resolution_conflict``; a repair clock
  that does not dominate the expected candidates is 400
  ``invalid_request``;
- a known identity with identical content is a replay, with different
  content 409 ``operation_conflict``;
- at least one new action commits atomically and answers 201, an
  all-replay batch answers 200.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for batch semantics. Only the
Python standard library is used.
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
    MAX_BODY_BYTES,
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
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


def send_action(key: str, replica: str, op: str, value: str, clock: dict) -> dict:
    return {
        "action": "send_local",
        "key": key,
        "replicaId": replica,
        "operationId": op,
        "value": value,
        "clock": clock,
    }


def fetch_action(key: str, replica: str, op: str, value: str, clock: dict) -> dict:
    return {
        "action": "fetch_remote",
        "key": key,
        "replicaId": replica,
        "operationId": op,
        "value": value,
        "clock": clock,
    }


def resolve_action(
    key: str, replica: str, op: str, value: str, clock: dict, candidates: list
) -> dict:
    return {
        "action": "semantic_resolution",
        "key": key,
        "replicaId": replica,
        "operationId": op,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


def identities(*pairs: tuple) -> list:
    return [{"replicaId": r, "operationId": o} for r, o in pairs]


class ApplyParserTests(unittest.TestCase):
    def test_valid_payload_round_trip(self) -> None:
        document = {
            "replicaId": "remote",
            "expectedLocalDigest": "a" * 64,
            "snapshot": {"k": [candidate("r1", "o1", "v", {"r1": 1})]},
            "actions": [fetch_action("k", "r1", "o1", "v", {"r1": 1})],
        }
        replica_id, digest, snapshot, actions = parse_replication_apply_payload(
            json.dumps(document)
        )
        self.assertEqual(replica_id, "remote")
        self.assertEqual(digest, "a" * 64)
        self.assertEqual(snapshot["k"][0]["value"], "v")
        self.assertEqual(actions[0]["action"], "fetch_remote")

    def test_invalid_payloads_raise(self) -> None:
        valid = {
            "replicaId": "remote",
            "expectedLocalDigest": "a" * 64,
            "snapshot": {},
            "actions": [fetch_action("k", "r1", "o1", "v", {"r1": 1})],
        }
        bad_documents = [
            {},
            {"replicaId": "remote"},
            {**valid, "extra": 1},
            {**valid, "replicaId": ""},
            {**valid, "expectedLocalDigest": "a" * 63},
            {**valid, "expectedLocalDigest": "A" * 64},
            {**valid, "expectedLocalDigest": "g" * 64},
            {**valid, "expectedLocalDigest": 5},
            {**valid, "snapshot": []},
            {**valid, "actions": []},
            {**valid, "actions": {}},
            {**valid, "actions": [{}]},
            {**valid, "actions": [{"action": "move"}]},
            {**valid, "actions": [{**fetch_action("k", "r1", "o1", "v", {"r1": 1}), "x": 1}]},
            {**valid, "actions": [{k: v for k, v in fetch_action("k", "r1", "o1", "v", {"r1": 1}).items() if k != "clock"}]},
            {**valid, "actions": [fetch_action("", "r1", "o1", "v", {"r1": 1})]},
            {**valid, "actions": [fetch_action("k", "", "o1", "v", {"r1": 1})]},
            {**valid, "actions": [fetch_action("k", "r1", "", "v", {"r1": 1})]},
            {**valid, "actions": [fetch_action("k", "r1", "o1", "", {"r1": 1})]},
            {**valid, "actions": [fetch_action("k", "r1", "o1", "v", {"r1": 1.0})]},
            {**valid, "actions": [fetch_action("k", "r1", "o1", "v", {"r2": 1})]},
            {**valid, "actions": [fetch_action("k", "r1", "o1", "v", {})]},
            # Duplicate action identity.
            {
                **valid,
                "actions": [
                    fetch_action("k", "r1", "o1", "v", {"r1": 1}),
                    fetch_action("k2", "r1", "o1", "v", {"r1": 1}),
                ],
            },
            # semantic_resolution without candidates / with bad candidates.
            {
                **valid,
                "actions": [
                    {k: v for k, v in resolve_action("k", "r3", "o3", "m", {"r3": 2}, identities(("r1", "o1"))).items() if k != "candidates"}
                ],
            },
            {
                **valid,
                "actions": [resolve_action("k", "r3", "o3", "m", {"r3": 2}, [])],
            },
            {
                **valid,
                "actions": [
                    resolve_action(
                        "k", "r3", "o3", "m", {"r3": 2},
                        identities(("r1", "o1"), ("r1", "o1")),
                    )
                ],
            },
            # send/fetch must not carry candidates.
            {
                **valid,
                "actions": [
                    {**fetch_action("k", "r1", "o1", "v", {"r1": 1}), "candidates": []}
                ],
            },
        ]
        for document in bad_documents:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_apply_payload(json.dumps(document))

    def test_duplicate_fields_raise(self) -> None:
        with self.assertRaises(ValueError):
            parse_replication_apply_payload(
                b'{"replicaId":"a","replicaId":"b","expectedLocalDigest":"'
                + b"a" * 64
                + b'","snapshot":{},"actions":[]}'
            )

    def test_malformed_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_replication_apply_payload(b"{not json")


class ApplyStoreTests(unittest.TestCase):
    """Batch semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def digest(self) -> str:
        return self.store.get_verification_digest()["digest"]

    def apply(self, actions: list, snapshot: dict | None = None,
              digest: str | None = None, replica_id: str = "remote"):
        return self.store.apply_replication_plan(
            replica_id,
            self.digest() if digest is None else digest,
            {} if snapshot is None else snapshot,
            actions,
        )

    def test_digest_mismatch_is_apply_conflict_and_changes_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, results, accepted, replayed, error = self.apply(
            [fetch_action("k2", "r2", "o2", "w", {"r2": 1})],
            snapshot={"k2": [candidate("r2", "o2", "w", {"r2": 1})]},
            digest="0" * 64,
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")
        self.assertEqual((results, accepted, replayed), ([], 0, 0))
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_fetch_remote_imports_candidate(self) -> None:
        status, results, accepted, replayed, error = self.apply(
            [fetch_action("k", "r2", "o2", "w", {"r2": 1})],
            snapshot={"k": [candidate("r2", "o2", "w", {"r2": 1})]},
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual((accepted, replayed), (1, 0))
        self.assertEqual(
            results,
            [
                {
                    "action": "fetch_remote",
                    "key": "k",
                    "replicaId": "r2",
                    "operationId": "o2",
                    "value": "w",
                }
            ],
        )
        status_code, state = self.store.get_state("k")
        self.assertIs(status_code, HTTPStatus.OK)
        self.assertEqual(state["value"], "w")
        # The import is an ordinary operation: exported by sync.
        page, _, _ = self.store.get_sync_operations(0, 10)
        self.assertEqual(len(page), 1)
        self.assertEqual(page[0]["replicaId"], "r2")

    def test_send_local_is_replay_and_changes_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.digest()
        status, results, accepted, replayed, error = self.apply(
            [send_action("k", "r1", "o1", "v", {"r1": 1})]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["action"], "send_local")
        self.assertEqual(results[0]["value"], "v")
        self.assertEqual(self.digest(), before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_send_local_with_unknown_identity_is_apply_conflict(self) -> None:
        status, _, _, _, error = self.apply(
            [send_action("k", "r1", "o1", "v", {"r1": 1})]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "apply_conflict")

    def test_send_local_with_different_content_is_operation_conflict(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _, _, _, error = self.apply(
            [send_action("k", "r1", "o1", "other", {"r1": 1})]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")

    def test_fetch_remote_candidate_mismatch_is_apply_conflict(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        for action in (
            fetch_action("k", "r2", "o2", "changed", {"r2": 1}),
            fetch_action("k", "r2", "o2", "w", {"r2": 2}),
            fetch_action("k", "r2", "o3", "w", {"r2": 1}),
            fetch_action("missing", "r2", "o2", "w", {"r2": 1}),
        ):
            with self.subTest(action=action):
                status, _, _, _, error = self.apply([action], snapshot=snapshot)
                self.assertIs(status, HTTPStatus.CONFLICT)
                self.assertEqual(error, "apply_conflict")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 0)

    def test_fetch_replay_is_idempotent(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        actions = [fetch_action("k", "r2", "o2", "w", {"r2": 1})]
        status, _, _, _, _ = self.apply(actions, snapshot=snapshot)
        self.assertIs(status, HTTPStatus.CREATED)
        # A replay against the moved state (fresh digest) is all-replay.
        status, results, accepted, replayed, error = self.apply(
            actions, snapshot=snapshot
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["value"], "w")
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_fetch_with_known_identity_different_content_is_operation_conflict(
        self,
    ) -> None:
        self.store.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        status, _, _, _, error = self.apply(
            [fetch_action("k", "r2", "o2", "w", {"r2": 2})],
            snapshot={"k": [candidate("r2", "o2", "w", {"r2": 2})]},
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "operation_conflict")

    def test_semantic_resolution_commits_merged_value(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        status, results, accepted, replayed, error = self.apply(
            [
                resolve_action(
                    "k", "r3", "o3", "merged", {"r3": 1, "r1": 1, "r2": 1},
                    identities(("r1", "o1"), ("r2", "o2")),
                )
            ]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual((accepted, replayed), (1, 0))
        self.assertEqual(results[0]["action"], "semantic_resolution")
        self.assertEqual(results[0]["value"], "merged")
        status_code, state = self.store.get_state("k")
        self.assertIs(status_code, HTTPStatus.OK)
        self.assertEqual(state["value"], "merged")
        self.assertEqual(state["status"], "resolved")

    def test_semantic_resolution_state_conflicts(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        clock = {"r3": 1, "r1": 1, "r2": 1}
        # Missing target key.
        status, _, _, _, error = self.apply(
            [resolve_action("nope", "r3", "o3", "m", clock, identities(("r1", "o1")))]
        )
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "resolution_conflict"))
        # Candidate set mismatch.
        status, _, _, _, error = self.apply(
            [resolve_action("k", "r3", "o3", "m", clock, identities(("r1", "o1")))]
        )
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "resolution_conflict"))
        # Resolve for real, then the key is no longer in conflict.
        status, _, _, _, _ = self.apply(
            [resolve_action("k", "r3", "o3", "m", clock, identities(("r1", "o1"), ("r2", "o2")))]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, _, _, _, error = self.apply(
            [
                resolve_action(
                    "k", "r3", "o4", "m2", {"r3": 2, "r1": 1, "r2": 1},
                    identities(("r3", "o3")),
                )
            ]
        )
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "resolution_conflict"))

    def test_semantic_resolution_non_dominating_clock_is_invalid(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 2}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        with self.assertRaises(ValueError):
            self.apply(
                [
                    resolve_action(
                        "k", "r3", "o3", "m", {"r3": 1},
                        identities(("r1", "o1"), ("r2", "o2")),
                    )
                ]
            )
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 2)

    def test_semantic_resolution_replay_and_conflict(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        action = resolve_action(
            "k", "r3", "o3", "m", {"r3": 1, "r1": 1, "r2": 1},
            identities(("r1", "o1"), ("r2", "o2")),
        )
        status, _, _, _, _ = self.apply([action])
        self.assertIs(status, HTTPStatus.CREATED)
        # Identical replay (fresh digest) is a replay even though the key
        # is no longer in conflict.
        status, results, accepted, replayed, error = self.apply([action])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results[0]["value"], "m")
        # Same identity, different merged value: operation conflict.
        changed = dict(action, value="different")
        status, _, _, _, error = self.apply([changed])
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "operation_conflict"))

    def test_batch_runs_in_order_against_staged_view(self) -> None:
        # The fetch makes the remote candidate visible to the later
        # resolution inside the same batch.
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        snapshot = {"k": [candidate("r2", "o2", "b", {"r2": 1})]}
        status, results, accepted, replayed, error = self.apply(
            [
                fetch_action("k", "r2", "o2", "b", {"r2": 1}),
                resolve_action(
                    "k", "r3", "o3", "m", {"r3": 1, "r1": 1, "r2": 1},
                    identities(("r1", "o1"), ("r2", "o2")),
                ),
            ],
            snapshot=snapshot,
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (2, 0))
        self.assertEqual([r["action"] for r in results],
                         ["fetch_remote", "semantic_resolution"])
        _, state = self.store.get_state("k")
        self.assertEqual(state["value"], "m")

    def test_failed_batch_commits_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        before = self.digest()
        status, _, _, _, error = self.apply(
            [
                fetch_action("k2", "r2", "o2", "b", {"r2": 1}),
                send_action("k", "r9", "o9", "x", {"r9": 1}),
            ],
            snapshot={"k2": [candidate("r2", "o2", "b", {"r2": 1})]},
        )
        self.assertEqual((status, error), (HTTPStatus.CONFLICT, "apply_conflict"))
        self.assertEqual(self.digest(), before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status_code, _ = self.store.get_state("k2")
        self.assertIs(status_code, HTTPStatus.NOT_FOUND)


class ApplyPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_committed_batch_survives_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        digest = store.get_verification_digest()["digest"]
        snapshot = {"k": [candidate("r2", "o2", "b", {"r2": 1})]}
        actions = [fetch_action("k", "r2", "o2", "b", {"r2": 1})]
        status, before_results, _, _, _ = store.apply_replication_plan(
            "remote", digest, snapshot, actions
        )
        self.assertIs(status, HTTPStatus.CREATED)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_metrics()["acceptedOperations"], 2)
        # The replay decision is identical after recovery.
        digest = recovered.get_verification_digest()["digest"]
        status, results, accepted, replayed, _ = recovered.apply_replication_plan(
            "remote", digest, snapshot, actions
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual((accepted, replayed), (0, 1))
        self.assertEqual(results, before_results)

    def test_persistence_failure_rolls_back(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        digest = store.get_verification_digest()["digest"]
        snapshot = {"k": [candidate("r2", "o2", "b", {"r2": 1})]}
        actions = [fetch_action("k", "r2", "o2", "b", {"r2": 1})]

        def boom() -> None:
            raise PersistenceError("injected failure")

        store._persist_locked = boom  # type: ignore[assignment]
        with self.assertRaises(PersistenceError):
            store.apply_replication_plan("remote", digest, snapshot, actions)
        # Memory, the identity index, and the file are unchanged.
        self.assertEqual(store.get_metrics()["acceptedOperations"], 1)
        status_code, _ = store.get_state("k")
        self.assertIs(status_code, HTTPStatus.OK)
        _, state = store.get_state("k")
        self.assertEqual(state["value"], "a")
        # No temporary file was left behind.
        self.assertEqual(os.listdir(self._tmp.name), ["state.json"])
        # The batch can be retried once persistence works again.
        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_metrics()["acceptedOperations"], 1)


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

    def raw_request(self, method: str, path: str, body: object = None):
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

    def request(self, method: str, path: str, body: object = None):
        status, payload, _ = self.raw_request(method, path, body)
        return status, payload

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def local_digest(self) -> str:
        status, payload = self.request("GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        return payload["digest"]

    def apply_body(self, actions: list, snapshot: dict | None = None,
                   digest: str | None = None, replica_id: str = "remote") -> dict:
        return {
            "replicaId": replica_id,
            "expectedLocalDigest": self.local_digest() if digest is None else digest,
            "snapshot": {} if snapshot is None else snapshot,
            "actions": actions,
        }

    def apply(self, document: dict, path: str = APPLY_PATH):
        return self.request("POST", path, document)

    def test_fetch_over_http_created_then_ok(self) -> None:
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        document = self.apply_body(
            [fetch_action("k", "r2", "o2", "w", {"r2": 1})], snapshot=snapshot
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["replicaId"], "remote")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(len(payload["actions"]), 1)
        # Replay with a fresh digest is all-replay 200.
        document["expectedLocalDigest"] = self.local_digest()
        status, payload = self.apply(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = self.apply_body([send_action("k", "r1", "o1", "v", {"r1": 1})])
        status, payload, raw = self.raw_request("POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload), {"status", "replicaId", "actions", "accepted", "replayed"}
        )
        (result,) = payload["actions"]
        self.assertEqual(
            set(result), {"action", "key", "replicaId", "operationId", "value"}
        )
        self.assertIs(type(payload["accepted"]), int)
        self.assertIs(type(payload["replayed"]), int)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(
            raw[:-1].decode("utf-8"),
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )

    def test_digest_mismatch_is_409_apply_conflict(self) -> None:
        document = self.apply_body(
            [fetch_action("k", "r2", "o2", "w", {"r2": 1})],
            snapshot={"k": [candidate("r2", "o2", "w", {"r2": 1})]},
            digest="0" * 64,
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "apply_conflict"})
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 0)

    def test_invalid_bodies_are_400(self) -> None:
        digest = self.local_digest()
        base = self.apply_body([fetch_action("k", "r2", "o2", "w", {"r2": 1})])
        for document in (
            {},
            {"replicaId": "remote"},
            {**base, "x": 1},
            {**base, "expectedLocalDigest": "zz"},
            {**base, "actions": []},
            {**base, "actions": [{"action": "teleport"}]},
            {
                **base,
                "actions": [
                    fetch_action("k", "r2", "o2", "w", {"r2": 1}),
                    fetch_action("k", "r2", "o2", "w", {"r2": 1}),
                ],
            },
        ):
            status, payload = self.apply(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)

    def test_malformed_json_body_is_400(self) -> None:
        status, payload, _ = self.raw_request("POST", APPLY_PATH, b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_non_dominating_repair_clock_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "a", {"r1": 2}))
        self.post_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        document = self.apply_body(
            [
                resolve_action(
                    "k", "r3", "o3", "m", {"r3": 1},
                    identities(("r1", "o1"), ("r2", "o2")),
                )
            ]
        )
        status, payload = self.apply(document)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_resolution_conflict_is_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        document = self.apply_body(
            [resolve_action("k", "r3", "o3", "m", {"r3": 2, "r1": 1}, identities(("r1", "o1")))]
        )
        # Only one candidate value: the key is not in conflict.
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_operation_conflict_is_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = self.apply_body([send_action("k", "r1", "o1", "changed", {"r1": 1})])
        status, payload = self.apply(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_any_query_parameter_is_400(self) -> None:
        document = self.apply_body([send_action("k", "r1", "o1", "v", {"r1": 1})])
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.apply(document, APPLY_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_path_shape_mismatches_are_404(self) -> None:
        document = self.apply_body([send_action("k", "r1", "o1", "v", {"r1": 1})])
        for path in (
            "/v1/replication/apply/extra",
            "/v1/replication",
            "/v1/replication/apply/",
            "/v1/replication/applies",
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
        documents = [
            {"replicaId": "p"},
            self.apply_body(
                [fetch_action("k2", "r2", "o2", "w", {"r2": 1})],
                snapshot={"k2": [candidate("r2", "o2", "w", {"r2": 1})]},
                digest="0" * 64,
            ),
            self.apply_body([send_action("k", "r9", "o9", "x", {"r9": 1})]),
        ]
        for document in documents:
            status, _ = self.apply(document)
            self.assertIn(status, (400, 409))
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


class ApplyHttpAuthTests(unittest.TestCase):
    """The apply endpoint authenticates as a write endpoint."""

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

    def document(self, port: int, read_token: str) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "GET",
            "/v1/verification/digest",
            headers={"Authorization": f"Bearer {read_token}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertIn("digest", payload)
        return {
            "replicaId": "remote",
            "expectedLocalDigest": payload["digest"],
            "snapshot": {},
            "actions": [send_action("k", "r1", "o1", "v", {"r1": 1})],
        }

    def test_single_token_mode_requires_bearer_token(self) -> None:
        document = self.document(self.single_port, "sekret")
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", APPLY_PATH, document, auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", APPLY_PATH, document, auth="Bearer sekret"
        )
        # The send_local direction does not hold (empty store), but the
        # request is authenticated and authorized.
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "apply_conflict"})

    def test_scope_mode_requires_write_or_admin(self) -> None:
        document = self.document(self.scope_port, "reader")
        status, payload, challenge = self.request(
            self.scope_port, "POST", APPLY_PATH, document, auth="Bearer reader"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer writer", "Bearer admin"):
            status, _, _ = self.request(
                self.scope_port, "POST", APPLY_PATH, document, auth=token
            )
            # Authenticated and authorized; the send direction itself does
            # not hold against the empty store.
            self.assertEqual(status, 409, token)

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")


class ApplyHttpPersistenceTests(unittest.TestCase):
    """With --data-file the committed batch survives restarts."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str,
                body: object = None):
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

    def test_apply_survives_restart_and_rejected_batches_write_nothing(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "a", {"r1": 1}),
        )
        _, digest_payload = self.request(server, "GET", "/v1/verification/digest")
        snapshot = {"k": [candidate("r2", "o2", "b", {"r2": 1})]}
        document = {
            "replicaId": "remote",
            "expectedLocalDigest": digest_payload["digest"],
            "snapshot": snapshot,
            "actions": [fetch_action("k", "r2", "o2", "b", {"r2": 1})],
        }
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        _, digest_payload = self.request(server, "GET", "/v1/verification/digest")
        document["expectedLocalDigest"] = digest_payload["digest"]
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["replayed"], 1)

        # A rejected batch writes nothing and creates no temporary file.
        before_bytes = Path(self.data_file).read_bytes()
        before_entries = set(os.listdir(self._tmp.name))
        document["expectedLocalDigest"] = "0" * 64
        status, payload = self.request(server, "POST", APPLY_PATH, document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "apply_conflict"})
        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        self.assertEqual(set(os.listdir(self._tmp.name)), before_entries)


if __name__ == "__main__":
    unittest.main()
