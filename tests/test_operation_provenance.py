"""Tests for the read-only per-operation provenance query.

The provenance endpoint is::

    GET /v1/replicas/{replicaId}/operations/{operationId}/provenance

It explains one first-accepted operation's immediate semantic effect on its
target key. The success body is one newline-terminated compact JSON line
with the fields ``replicaId``, ``operationId``, ``operation``, ``origin``,
``policy``, ``before``, and ``after`` in that order; ``before`` and
``after`` are ``{"status", "candidates"}`` snapshots of the target key
replayed from the shared accepted log immediately before and after the
target record. ``origin`` is ``automatic_resolution`` (with one of the six
published policies) only for an operation carrying a local automatic
resolution policy binding; every other record is ``other`` with a null
policy. The query is strictly read-only and shares the commit lock.

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

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(value: str, clock: dict, replica_id: str, operation_id: str) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def make_conflict(store: StateStore, key: str = "k") -> None:
    """Commit two concurrent writes with different values on ``key``."""
    store.apply_operation("r1", operation("o1", key, "v1", {"r1": 1}))
    store.apply_operation("r2", operation("o2", key, "v2", {"r2": 1}))


class OperationProvenanceStoreTests(unittest.TestCase):
    """Store-level semantics, both in memory and file backed."""

    def test_plain_write_from_empty(self) -> None:
        store = StateStore()
        op = operation("o1", "k", "v", {"r1": 1})
        self.assertIs(store.apply_operation("r1", op), HTTPStatus.CREATED)
        status, payload = store.get_operation_provenance("r1", "o1")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "replicaId": "r1",
                "operationId": "o1",
                "operation": op,
                "origin": "other",
                "policy": None,
                "before": {"status": "absent", "candidates": []},
                "after": {
                    "status": "resolved",
                    "candidates": [candidate("v", {"r1": 1}, "r1", "o1")],
                },
            },
        )

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica_id, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            self.assertEqual(
                store.get_operation_provenance(replica_id, operation_id),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )

    def test_second_concurrent_write_moves_resolved_to_conflict(self) -> None:
        store = StateStore()
        make_conflict(store)
        status, payload = store.get_operation_provenance("r2", "o2")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])
        self.assertEqual(
            payload["before"],
            {
                "status": "resolved",
                "candidates": [candidate("v1", {"r1": 1}, "r1", "o1")],
            },
        )
        self.assertEqual(
            payload["after"],
            {
                "status": "conflict",
                "candidates": [
                    candidate("v1", {"r1": 1}, "r1", "o1"),
                    candidate("v2", {"r2": 1}, "r2", "o2"),
                ],
            },
        )

    def test_stale_write_leaves_snapshots_identical(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        stale = operation("o2", "k", "old", {"r1": 1})
        self.assertIs(store.apply_operation("r1", stale), HTTPStatus.CREATED)
        status, payload = store.get_operation_provenance("r1", "o2")
        self.assertIs(status, HTTPStatus.OK)
        snapshot = {
            "status": "resolved",
            "candidates": [candidate("new", {"r1": 2}, "r1", "o1")],
        }
        self.assertEqual(payload["before"], snapshot)
        self.assertEqual(payload["after"], snapshot)

    def test_manual_resolution_is_other(self) -> None:
        store = StateStore()
        make_conflict(store)
        resolution = {
            "replicaId": "r1",
            "operationId": "fix",
            "value": "merged",
            "clock": {"r1": 2, "r2": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, error = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_operation_provenance("r1", "fix")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(
            payload["after"],
            {
                "status": "resolved",
                "candidates": [
                    candidate("merged", {"r1": 2, "r2": 1}, "r1", "fix")
                ],
            },
        )

    def test_auto_resolution_lowest_identity(self) -> None:
        store = StateStore()
        make_conflict(store)
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        status, _, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "lowest_identity")
        self.assertEqual(
            payload["before"],
            {
                "status": "conflict",
                "candidates": [
                    candidate("v1", {"r1": 1}, "r1", "o1"),
                    candidate("v2", {"r2": 1}, "r2", "o2"),
                ],
            },
        )
        self.assertEqual(
            payload["after"],
            {
                "status": "resolved",
                "candidates": [
                    candidate(
                        "v1",
                        {"r3": 1, "r1": 1, "r2": 1},
                        "r3",
                        "auto",
                    )
                ],
            },
        )
        # The chosen value is the policy-selected candidate's value.
        self.assertEqual(payload["operation"]["value"], "v1")

    def test_auto_resolution_highest_value(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v9", {"r2": 1}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "highest_value",
        }
        status, _, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "highest_value")
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["after"]["status"], "resolved")
        self.assertEqual(payload["after"]["candidates"][0]["value"], "v9")
        self.assertEqual(payload["operation"]["value"], "v9")

    def test_auto_resolution_batch_binds_policy(self) -> None:
        store = StateStore()
        make_conflict(store, "k")
        store.apply_operation("a", operation("p1", "other", "x", {"a": 1}))
        store.apply_operation("b", operation("p2", "other", "y", {"b": 1}))
        entries = [
            {
                "key": "k",
                "replicaId": "r3",
                "operationId": "auto-k",
                "clock": {"r3": 1, "r1": 1, "r2": 1},
                "policy": "highest_identity",
            },
            {
                "key": "other",
                "replicaId": "r3",
                "operationId": "auto-o",
                "clock": {"r3": 2, "a": 1, "b": 1},
                "policy": "lowest_value",
            },
        ]
        status, _, accepted, _, error = store.apply_auto_resolutions(entries)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 2)
        self.assertIsNone(error)
        status, payload = store.get_operation_provenance("r3", "auto-k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "highest_identity")
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["after"]["candidates"][0]["value"], "v2")
        status, payload = store.get_operation_provenance("r3", "auto-o")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "lowest_value")
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["after"]["candidates"][0]["value"], "x")

    def test_synced_operation_is_other(self) -> None:
        store = StateStore()
        make_conflict(store)
        # An operation that looks exactly like a repair, but arrives via sync.
        record = operation("synced", "k", "v1", {"r9": 1, "r1": 1, "r2": 1})
        status, accepted, _ = store.import_operations([("r9", record)])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 1)
        status, payload = store.get_operation_provenance("r9", "synced")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])
        self.assertEqual(payload["before"]["status"], "conflict")

    def test_synced_auto_resolution_keeps_no_local_binding(self) -> None:
        # A locally bound automatic resolution is exported as an ordinary
        # operation; the importing replica holds no binding, so there it is
        # "other" even though its before/after snapshots are unchanged.
        source = StateStore()
        make_conflict(source)
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        status, committed, error = source.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        importer = StateStore()
        status, accepted, _ = importer.import_operations(
            [
                ("r1", operation("o1", "k", "v1", {"r1": 1})),
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r3", committed),
            ]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 3)
        status, payload = importer.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["after"]["status"], "resolved")

    def test_later_overwrite_does_not_change_response(self) -> None:
        store = StateStore()
        make_conflict(store)
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        status, _, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        _, at_resolution = store.get_operation_provenance("r3", "auto")
        # A later concurrent write puts the key back into conflict.
        store.apply_operation("r4", operation("later", "k", "v2", {"r4": 1}))
        self.assertEqual(store.get_state("k")[1]["status"], "conflict")
        status, after_overwrite = store.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(after_overwrite, at_resolution)
        # The later write, however, sees the resolved key before itself.
        status, later = store.get_operation_provenance("r4", "later")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(later["before"]["status"], "resolved")
        self.assertEqual(later["after"]["status"], "conflict")

    def test_operations_on_other_keys_are_ignored(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r9", operation("noise", "other", "z", {"r9": 9}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = store.get_operation_provenance("r2", "o2")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [c["operationId"] for c in payload["before"]["candidates"]],
            ["o1"],
        )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        make_conflict(store)
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_export = store.get_store_export()
        store.get_operation_provenance("r1", "o1")
        store.get_operation_provenance("r3", "absent")
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_store_export(), before_export)

    def test_data_file_restart_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            make_conflict(store)
            request = {
                "replicaId": "r3",
                "operationId": "auto",
                "clock": {"r3": 1, "r1": 1, "r2": 1},
                "policy": "lowest_identity",
            }
            status, _, error = store.apply_auto_resolution("k", request)
            self.assertIs(status, HTTPStatus.CREATED)
            self.assertIsNone(error)
            expected_auto = store.get_operation_provenance("r3", "auto")[1]
            expected_write = store.get_operation_provenance("r2", "o2")[1]
            expected_miss = store.get_operation_provenance("r9", "nope")
            recovered = StateStore(data_file=data_file)
            self.assertEqual(
                recovered.get_operation_provenance("r3", "auto"),
                (HTTPStatus.OK, expected_auto),
            )
            self.assertEqual(
                recovered.get_operation_provenance("r2", "o2"),
                (HTTPStatus.OK, expected_write),
            )
            self.assertEqual(
                recovered.get_operation_provenance("r9", "nope"), expected_miss
            )

    def test_old_data_file_without_policies_section_binds_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "state.json"
            # A version:1 file written before the policies section existed:
            # two conflicting writes plus a resolution-looking record, no
            # policies segment at all.
            document = {
                "version": 1,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("o1", "k", "v1", {"r1": 1}),
                    },
                    {
                        "replicaId": "r2",
                        "operation": operation("o2", "k", "v2", {"r2": 1}),
                    },
                    {
                        "replicaId": "r3",
                        "operation": operation(
                            "fix", "k", "v1", {"r3": 1, "r1": 1, "r2": 1}
                        ),
                    },
                ],
                "checkpoints": {},
            }
            data_file.write_text(json.dumps(document), encoding="utf-8")
            raw_before = data_file.read_bytes()
            store = StateStore(data_file=str(data_file))
            status, payload = store.get_operation_provenance("r3", "fix")
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(payload["origin"], "other")
            self.assertIsNone(payload["policy"])
            self.assertEqual(payload["before"]["status"], "conflict")
            self.assertEqual(payload["after"]["status"], "resolved")
            # The read-only query must not have rewritten the old file.
            self.assertEqual(data_file.read_bytes(), raw_before)


class HttpOperationProvenanceTests(unittest.TestCase):
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

    def request_raw(self, method: str, path: str, body: object = None):
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
        raw = response.read()
        headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, headers

    def request(self, method: str, path: str, body: object = None):
        status, raw, headers = self.request_raw(method, path, body)
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return status, payload, headers

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_auto_resolution(self, key: str, request_body: dict):
        return self.request("POST", f"/v1/states/{key}/resolve/auto", request_body)

    def get_provenance(self, replica: str, operation_id: str, query: str = ""):
        return self.request(
            "GET",
            f"/v1/replicas/{replica}/operations/{operation_id}/provenance{query}",
        )

    def test_plain_write_round_trip(self) -> None:
        op = operation("o1", "color", "blue", {"r1": 1})
        status, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, raw, headers = self.request_raw(
            "GET", "/v1/replicas/r1/operations/o1/provenance"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact ordered JSON, exactly one trailing newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        expected = (
            b'{"replicaId":"r1","operationId":"o1",'
            b'"operation":{"operationId":"o1","key":"color","value":"blue","clock":{"r1":1}},'
            b'"origin":"other","policy":null,'
            b'"before":{"status":"absent","candidates":[]},'
            b'"after":{"status":"resolved","candidates":['
            b'{"value":"blue","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}'
            b"]}}\n"
        )
        self.assertEqual(raw, expected)
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_auto_resolution_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        status, _, _ = self.post_auto_resolution("k", body)
        self.assertEqual(status, 201)
        status, payload, headers = self.get_provenance("r3", "auto")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), [
            "replicaId",
            "operationId",
            "operation",
            "origin",
            "policy",
            "before",
            "after",
        ])
        self.assertEqual(payload["replicaId"], "r3")
        self.assertEqual(payload["operationId"], "auto")
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "lowest_identity")
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["after"]["status"], "resolved")
        self.assertTrue(all(list(c) == ["value", "clock", "replicaId", "operationId"]
                            for c in payload["before"]["candidates"]))
        self.assertTrue(all(list(c) == ["value", "clock", "replicaId", "operationId"]
                            for c in payload["after"]["candidates"]))
        self.assertEqual(list(payload["before"]), ["status", "candidates"])
        self.assertEqual(list(payload["after"]), ["status", "candidates"])

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            status, payload, _ = self.get_provenance(replica, operation_id)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("op 1", "k", "v", {"r/1": 1})
        status, _, _ = self.post_operation("r%2F1", op)
        self.assertEqual(status, 201)
        status, payload, _ = self.get_provenance("r%2F1", "op%201")
        self.assertEqual(status, 200)
        self.assertEqual(payload["replicaId"], "r/1")
        self.assertEqual(payload["operationId"], "op 1")
        self.assertEqual(payload["operation"], op)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations/o1/provenance/extra",
            "/v1/replicas/r1/operations",
            "/v1/replicas//operations/o1/provenance",
            "/v1/replicas/r1/operations//provenance",
            "/v1/replicas/r1/operations/o1/provenance/",
        ):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1"):
            status, payload, _ = self.get_provenance("r1", "o1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations/o1/provenance/extra?x=1",
            "/v1/replicas//operations/o1/provenance?x=1",
            "/v1/replicas/r1/operations?x=1",
        ):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_non_get_method_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        path = "/v1/replicas/r1/operations/o1/provenance"
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            status, payload, _ = self.request(method, path, {})
            self.assertEqual(status, 404, method)
            self.assertEqual(payload, {"error": "not_found"}, method)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _ = self.request("GET", "/v1/metrics")
        self.get_provenance("r1", "o1")
        self.get_provenance("r1", "absent")
        after, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)


class HttpOperationProvenanceAuthTests(unittest.TestCase):
    """With the legacy token the endpoint authenticates like any GET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
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

    def request(self, method: str, path: str, body: object = None, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, headers

    def test_requires_bearer_token_with_challenge(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        status, payload, headers = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, _, headers = self.request(
            "GET",
            "/v1/replicas/r1/operations/o1/provenance",
            auth="Bearer wrong",
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, payload, headers = self.request(
            "GET",
            "/v1/replicas/r1/operations/o1/provenance",
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertNotIn("WWW-Authenticate", headers)
        self.assertEqual(payload["origin"], "other")

    def test_shape_404_still_authenticates(self) -> None:
        status, payload, headers = self.request(
            "GET", "/v1/replicas//operations/o1/provenance"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")


class HttpOperationProvenanceScopePolicyTests(unittest.TestCase):
    """Scope-policy mode: read/admin allowed, write-only gets 403."""

    POLICY = {
        "reader-token": frozenset({"read"}),
        "writer-token": frozenset({"write"}),
        "admin-token": frozenset({"read", "write", "admin"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(cls.POLICY),
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

    def request(self, path: str, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = f"Bearer {auth}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/replicas/r1/operations",
                     body=json.dumps(operation("o1", "k", "v", {"r1": 1})),
                     headers={"Authorization": "Bearer admin-token",
                              "Content-Type": "application/json"})
        conn.getresponse().read()
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        headers_out = dict(response.getheaders())
        conn.close()
        return response.status, payload, headers_out

    def test_scopes(self) -> None:
        path = "/v1/replicas/r1/operations/o1/provenance"
        status, payload, headers = self.request(path)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, payload, headers = self.request(path, auth="writer-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        status, payload, headers = self.request(path, auth="reader-token")
        self.assertEqual(status, 200)
        self.assertNotIn("WWW-Authenticate", headers)
        self.assertEqual(payload["origin"], "other")
        status, payload, _ = self.request(path, auth="admin-token")
        self.assertEqual(status, 200)
        self.assertEqual(payload["origin"], "other")

    def test_write_only_token_gets_403_before_query_check(self) -> None:
        status, payload, headers = self.request(
            "/v1/replicas/r1/operations/o1/provenance?x=1", auth="writer-token"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)


class HttpOperationProvenancePersistenceTests(unittest.TestCase):
    """The provenance response survives a data-file restart unchanged."""

    def test_restart_preserves_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")

            def serve_once(actions):
                server = SemanticStateServer(
                    ("127.0.0.1", 0), RequestHandler, data_file=data_file
                )
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                port = server.server_address[1]
                try:
                    results = []
                    for method, path, body in actions:
                        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                        headers = {"Content-Type": "application/json"}
                        if body is None:
                            conn.request(method, path, headers=headers)
                        else:
                            conn.request(method, path, body=json.dumps(body), headers=headers)
                        response = conn.getresponse()
                        raw = response.read()
                        results.append((response.status, raw))
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            serve_once(
                [
                    (
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o1", "k", "v1", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "k", "v2", {"r2": 1}),
                    ),
                    (
                        "POST",
                        "/v1/states/k/resolve/auto",
                        {
                            "replicaId": "r3",
                            "operationId": "auto",
                            "clock": {"r3": 1, "r1": 1, "r2": 1},
                            "policy": "lowest_identity",
                        },
                    ),
                ]
            )
            first = serve_once(
                [("GET", "/v1/replicas/r3/operations/auto/provenance", None)]
            )
            second = serve_once(
                [
                    ("GET", "/v1/replicas/r3/operations/auto/provenance", None),
                    ("GET", "/v1/replicas/r9/operations/nope/provenance", None),
                    ("GET", "/v1/replicas/r3/operations/auto/provenance?x=1", None),
                ]
            )
            self.assertEqual(first[0][0], 200)
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[0][1])
            self.assertTrue(second[0][1].endswith(b"\n"))
            self.assertEqual(second[1][0], 404)
            self.assertEqual(json.loads(second[1][1]), {"error": "not_found"})
            self.assertEqual(second[2][0], 400)
            self.assertEqual(json.loads(second[2][1]), {"error": "invalid_request"})


if __name__ == "__main__":
    unittest.main()
