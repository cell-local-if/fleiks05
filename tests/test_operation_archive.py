"""Tests for the read-only operation archive query.

The archive endpoint is::

    GET /v1/replicas/{replicaId}/operations/{operationId}

It locates one first-accepted operation by its ``(replicaId, operationId)``
identity and returns ``{"replicaId", "operation"}`` where ``operation``
carries exactly ``operationId``, ``key``, ``value``, and ``clock``. Every
accepted record is addressable — ordinary writes, stale writes, manual and
automatic resolutions, and sync imports — while replays, conflicts, invalid
requests, and undurably-committed operations never produce an archive
record. The query is strictly read-only and shares the commit lock.

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
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class OperationArchiveStoreTests(unittest.TestCase):
    """Store-level semantics, both in memory and file backed."""

    def test_plain_write_is_queryable(self) -> None:
        store = StateStore()
        op = operation("o1", "k", "v", {"r1": 1})
        self.assertIs(store.apply_operation("r1", op), HTTPStatus.CREATED)
        status, payload = store.get_operation("r1", "o1")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica_id, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            self.assertEqual(
                store.get_operation(replica_id, operation_id),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )

    def test_stale_write_is_queryable(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        stale = operation("o2", "k", "old", {"r1": 1})
        self.assertIs(store.apply_operation("r1", stale), HTTPStatus.CREATED)
        status, payload = store.get_operation("r1", "o2")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["operation"], stale)

    def test_manual_resolution_is_queryable(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
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
        status, payload = store.get_operation("r1", "fix")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "replicaId": "r1",
                "operation": operation("fix", "k", "merged", {"r1": 2, "r2": 1}),
            },
        )

    def test_auto_resolution_is_queryable(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        status, committed, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_operation("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload, {"replicaId": "r3", "operation": committed})

    def test_sync_import_is_queryable(self) -> None:
        store = StateStore()
        record = operation("o9", "k", "v", {"r9": 3})
        status, accepted, _ = store.import_operations([("r9", record)])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 1)
        self.assertEqual(
            store.get_operation("r9", "o9"),
            (HTTPStatus.OK, {"replicaId": "r9", "operation": record}),
        )

    def test_replay_adds_no_record(self) -> None:
        store = StateStore()
        op = operation("o1", "k", "v", {"r1": 1})
        store.apply_operation("r1", op)
        self.assertIs(store.apply_operation("r1", op), HTTPStatus.OK)
        status, payload = store.get_operation("r1", "o1")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["operation"], op)

    def test_conflict_and_invalid_requests_leave_no_record(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # Same identity, different content: a conflict, nothing committed.
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        # A failed import batch commits nothing.
        status, _, _ = store.import_operations(
            [("r2", operation("o1", "k", "w", {"r2": 1})), ("r1", operation("o1", "k", "x", {"r1": 1}))]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(store.get_operation("r2", "o1")[0], HTTPStatus.NOT_FOUND)
        # The original record is untouched.
        self.assertEqual(
            store.get_operation("r1", "o1"),
            (HTTPStatus.OK, {"replicaId": "r1", "operation": operation("o1", "k", "v", {"r1": 1})}),
        )

    def test_persistence_failure_leaves_no_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(data_file=str(Path(tmp) / "state.json"))
            with patch.object(store, "_persist_locked", side_effect=PersistenceError("boom")):
                with self.assertRaises(PersistenceError):
                    store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
            self.assertEqual(
                store.get_operation("r1", "o1"),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("k")
        store.get_operation("r1", "o1")
        store.get_operation("r1", "absent")
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("k"), before_audit)

    def test_data_file_restart_preserves_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            op = operation("o1", "k", "v", {"r1": 1})
            store.apply_operation("r1", op)
            expected = store.get_operation("r1", "o1")
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_operation("r1", "o1"), expected)
            self.assertEqual(
                recovered.get_operation("r1", "absent"),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpOperationArchiveTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None):
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
        headers = dict(response.getheaders())
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload, headers

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_archive(self, replica: str, operation_id: str, query: str = ""):
        return self.request("GET", f"/v1/replicas/{replica}/operations/{operation_id}{query}")

    def test_accepted_operation_round_trip(self) -> None:
        op = operation("o1", "color", "blue", {"r1": 1})
        status, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, payload, headers = self.get_archive("r1", "o1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})
        # The response carries exactly these two keys and the operation
        # exactly its four committed fields.
        self.assertEqual(set(payload), {"replicaId", "operation"})
        self.assertEqual(set(payload["operation"]), {"operationId", "key", "value", "clock"})
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(
            int(headers.get("Content-Length")),
            len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")),
        )

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            status, payload, _ = self.get_archive(replica, operation_id)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("op 1", "k", "v", {"r/1": 1})
        status, _, _ = self.post_operation("r%2F1", op)
        self.assertEqual(status, 201)
        status, payload, _ = self.get_archive("r%2F1", "op%201")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r/1", "operation": op})

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations",
            "/v1/replicas/r1/operations/o1/extra",
            "/v1/replicas/r1",
            "/v1/replicas",
            "/v1/replicas//operations/o1",
            "/v1/replicas/r1/operations/",
            "/v1/replicas/r1/operations//",
        ):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?after=0"):
            status, payload, _ = self.get_archive("r1", "o1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_route_shape_error_beats_query_error(self) -> None:
        # A route that does not match the archive shape is 404 even when it
        # also carries a query parameter.
        for path in (
            "/v1/replicas/r1/operations?x=1",
            "/v1/replicas/r1/operations/o1/extra?x=1",
            "/v1/replicas//operations/o1?x=1",
        ):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _ = self.request("GET", "/v1/metrics")
        self.get_archive("r1", "o1")
        self.get_archive("r1", "absent")
        after, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)


class HttpOperationArchiveAuthTests(unittest.TestCase):
    """With auth enabled the archive query authenticates like any GET."""

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
        conn.close()
        return response.status, payload

    def test_archive_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        status, payload = self.request("GET", "/v1/replicas/r1/operations/o1")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.request(
            "GET", "/v1/replicas/r1/operations/o1", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


class HttpOperationArchivePersistenceTests(unittest.TestCase):
    """The archive survives a data-file restart unchanged."""

    def test_restart_preserves_hits_and_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            op = operation("o1", "k", "v", {"r1": 1})

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
                        results.append(
                            (response.status, json.loads(raw.decode("utf-8")) if raw else None)
                        )
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            first = serve_once([("POST", "/v1/replicas/r1/operations", op)])
            self.assertEqual(first[0][0], 201)
            second = serve_once(
                [
                    ("GET", "/v1/replicas/r1/operations/o1", None),
                    ("GET", "/v1/replicas/r1/operations/absent", None),
                    ("GET", "/v1/replicas/r1/operations/o1?x=1", None),
                ]
            )
            self.assertEqual(second[0], (200, {"replicaId": "r1", "operation": op}))
            self.assertEqual(second[1], (404, {"error": "not_found"}))
            self.assertEqual(second[2], (400, {"error": "invalid_request"}))


if __name__ == "__main__":
    unittest.main()
