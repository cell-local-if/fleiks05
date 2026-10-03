"""Tests for the conditional-write endpoint.

`POST /v1/replicas/{replicaId}/operations/conditional` commits an ordinary
write only when the key's current candidate identity set equals the
request's `expectedCandidates`; the idempotence rules of the plain write
endpoint run first and are shared with it.
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
    load_data_file,
    parse_conditional_operation_payload,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def conditional_document(
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    expected: list,
) -> dict:
    return {
        "operationId": operation_id,
        "key": key,
        "value": value,
        "clock": clock,
        "expectedCandidates": expected,
    }


def identity(replica_id: str, operation_id: str) -> dict:
    return {"replicaId": replica_id, "operationId": operation_id}


class ParseConditionalOperationPayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        operation_out, expected = parse_conditional_operation_payload(
            json.dumps(
                conditional_document(
                    "op-1",
                    "color",
                    "blue",
                    {"r1": 1},
                    [identity("r2", "op-2"), identity("r1", "op-0")],
                )
            ),
            "r1",
        )
        self.assertEqual(operation_out, operation("op-1", "color", "blue", {"r1": 1}))
        # Identities are normalized to a deterministic sorted order.
        self.assertEqual(expected, [identity("r1", "op-0"), identity("r2", "op-2")])

    def test_empty_expected_candidates_is_valid(self) -> None:
        _, expected = parse_conditional_operation_payload(
            conditional_document("op-1", "k", "v", {"r1": 1}, []), "r1"
        )
        self.assertEqual(expected, [])

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(b"{not json", "r1")

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload("[1, 2]", "r1")

    def test_rejects_missing_expected_candidates(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                operation("op-1", "k", "v", {"r1": 1}), "r1"
            )

    def test_rejects_extra_field(self) -> None:
        document = conditional_document("op-1", "k", "v", {"r1": 1}, [])
        document["replicaId"] = "r1"
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(document, "r1")

    def test_rejects_non_list_expected_candidates(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document("op-1", "k", "v", {"r1": 1}, {}), "r1"
            )

    def test_rejects_element_with_missing_field(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document("op-1", "k", "v", {"r1": 1}, [{"replicaId": "r2"}]),
                "r1",
            )

    def test_rejects_element_with_extra_field(self) -> None:
        entry = identity("r2", "op-2")
        entry["value"] = "v"
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document("op-1", "k", "v", {"r1": 1}, [entry]), "r1"
            )

    def test_rejects_empty_identity_fields(self) -> None:
        for entry in (identity("", "op-2"), identity("r2", "")):
            with self.assertRaises(ValueError):
                parse_conditional_operation_payload(
                    conditional_document("op-1", "k", "v", {"r1": 1}, [entry]), "r1"
                )

    def test_rejects_duplicate_identities(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document(
                    "op-1",
                    "k",
                    "v",
                    {"r1": 1},
                    [identity("r2", "op-2"), identity("r2", "op-2")],
                ),
                "r1",
            )

    def test_write_constraints_are_reused(self) -> None:
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document("", "k", "v", {"r1": 1}, []), "r1"
            )
        with self.assertRaises(ValueError):
            parse_conditional_operation_payload(
                conditional_document("op-1", "k", "v", {"r2": 1}, []), "r1"
            )


class ConditionalWriteServerTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            {"operationId": operation_id, "key": key, "value": value, "clock": clock},
        )

    def post_conditional(
        self, replica: str, operation_id: str, key: str, value: str, clock: dict, expected: list
    ):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations/conditional",
            conditional_document(operation_id, key, value, clock, expected),
        )

    def test_empty_expected_on_absent_key_is_201(self) -> None:
        status, payload = self.post_conditional("r1", "op-1", "color", "blue", {"r1": 1}, [])
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {"status": "created", "replicaId": "r1", "operationId": "op-1", "key": "color"},
        )
        _, state = self.request("GET", "/v1/states/color")
        self.assertEqual(
            state,
            {"key": "color", "value": "blue", "clock": {"r1": 1}, "status": "resolved"},
        )

    def test_nonempty_expected_on_absent_key_is_state_conflict(self) -> None:
        status, payload = self.post_conditional(
            "r1", "op-1", "k", "v", {"r1": 1}, [identity("r2", "op-2")]
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "state_conflict"})
        status, payload = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_matching_expected_set_commits(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})[0], 201)
        self.assertEqual(self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})[0], 201)
        status, _ = self.post_conditional(
            "r3",
            "op-3",
            "k",
            "v3",
            {"r3": 1},
            # Order inside expectedCandidates does not matter.
            [identity("r2", "op-2"), identity("r1", "op-1")],
        )
        self.assertEqual(status, 201)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)

    def test_mismatched_expected_set_is_state_conflict_and_commits_nothing(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})[0], 201)
        for expected in (
            [],
            [identity("r1", "op-1"), identity("r2", "op-2")],
            [identity("r1", "op-9")],
        ):
            status, payload = self.post_conditional(
                "r3", "op-3", "k", "v3", {"r3": 1}, expected
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "state_conflict"})
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(
            state, {"key": "k", "value": "v1", "clock": {"r1": 1}, "status": "resolved"}
        )
        # The rejected identity was not recorded: it can still commit.
        status, _ = self.post_conditional(
            "r3", "op-3", "k", "v3", {"r3": 1}, [identity("r1", "op-1")]
        )
        self.assertEqual(status, 201)

    def test_state_conflict_does_not_move_metrics(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})[0], 201)
        _, before = self.request("GET", "/v1/metrics")
        status, _ = self.post_conditional("r3", "op-3", "k", "v3", {"r3": 1}, [])
        self.assertEqual(status, 409)
        _, after = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_replay_is_200_regardless_of_expected_set(self) -> None:
        document = conditional_document("op-1", "k", "v", {"r1": 1}, [])
        status, _ = self.post_conditional("r1", "op-1", "k", "v", {"r1": 1}, [])
        self.assertEqual(status, 201)
        # Move the candidate set so the original expectation no longer holds.
        self.assertEqual(self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})[0], 201)
        for expected in (
            [],
            [identity("r1", "op-1")],
            [identity("r9", "op-9")],
        ):
            status, payload = self.post_conditional("r1", "op-1", "k", "v", {"r1": 1}, expected)
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(len(state["candidates"]), 2)

    def test_same_identity_different_content_is_operation_conflict(self) -> None:
        self.assertEqual(self.post_conditional("r1", "op-1", "k", "v", {"r1": 1}, [])[0], 201)
        status, payload = self.post_conditional("r1", "op-1", "k", "other", {"r1": 1}, [])
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v")

    def test_idempotence_is_shared_with_plain_writes(self) -> None:
        # Accepted via the plain endpoint, replayed on the conditional one.
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 201)
        status, _ = self.post_conditional("r1", "op-1", "k", "v", {"r1": 1}, [])
        self.assertEqual(status, 200)
        status, payload = self.post_conditional("r1", "op-1", "k", "other", {"r1": 1}, [])
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Accepted via the conditional endpoint, replayed on the plain one.
        self.assertEqual(self.post_conditional("r1", "op-2", "j", "w", {"r1": 1}, [])[0], 201)
        status, _ = self.post_operation("r1", "op-2", "j", "w", {"r1": 1})
        self.assertEqual(status, 200)

    def test_malformed_body_is_400(self) -> None:
        status, payload = self.request(
            "POST", "/v1/replicas/r1/operations/conditional", b"{oops"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_invalid_expected_candidates_is_400(self) -> None:
        bad_bodies = (
            conditional_document("op-1", "k", "v", {"r1": 1}, [{}]),
            conditional_document("op-1", "k", "v", {"r1": 1}, [identity("r2", "op-2"), identity("r2", "op-2")]),
            {"operationId": "op-1", "key": "k", "value": "v", "clock": {"r1": 1}},
        )
        for body in bad_bodies:
            status, payload = self.request(
                "POST", "/v1/replicas/r1/operations/conditional", body
            )
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})
        status, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)

    def test_plain_write_endpoint_is_unchanged(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 201)
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 200)
        status, payload = self.post_operation("r1", "op-1", "k", "other", {"r1": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_trailing_segment_is_404(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/replicas/r1/operations/conditional/extra",
            conditional_document("op-1", "k", "v", {"r1": 1}, []),
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ConditionalWritePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def test_first_accept_is_persisted_and_recovered(self) -> None:
        store = self.make_store()
        status, error = store.apply_conditional_operation(
            "r1", operation("o1", "k", "v", {"r1": 1}), []
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        records = load_data_file(str(self.data_file))
        self.assertEqual(records, [("r1", operation("o1", "k", "v", {"r1": 1}))])

        reloaded = self.make_store()
        status, _ = reloaded.apply_conditional_operation(
            "r1", operation("o1", "k", "v", {"r1": 1}), [identity("r9", "op-9")]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)

    def test_state_conflict_changes_neither_memory_nor_file(self) -> None:
        store = self.make_store()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.data_file.read_bytes()
        status, error = store.apply_conditional_operation(
            "r2", operation("o2", "k", "w", {"r2": 1}), []
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(error, "state_conflict")
        self.assertEqual(self.data_file.read_bytes(), before)
        _, state = store.get_state("k")
        self.assertEqual(state["value"], "v")

    def test_persistence_failure_is_raised_and_leaves_state_untouched(self) -> None:
        store = self.make_store()
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            with self.assertRaises(PersistenceError):
                store.apply_conditional_operation(
                    "r1", operation("o1", "k", "v", {"r1": 1}), []
                )
        status, _ = store.get_state("k")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        # The failed identity was not recorded: the write can be retried.
        status, _ = store.apply_conditional_operation(
            "r1", operation("o1", "k", "v", {"r1": 1}), []
        )
        self.assertIs(status, HTTPStatus.CREATED)


class ConditionalWritePersistentServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_durable_failure_is_500_and_memory_is_unchanged(self) -> None:
        body = conditional_document("op-1", "k", "v", {"r1": 1}, [])
        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
        ):
            status, payload = self.request(
                "POST", "/v1/replicas/r1/operations/conditional", body
            )
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "internal_error"})
        status, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 404)
        # The write can be retried once persistence works again.
        status, _ = self.request("POST", "/v1/replicas/r1/operations/conditional", body)
        self.assertEqual(status, 201)


if __name__ == "__main__":
    unittest.main()
