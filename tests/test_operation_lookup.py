"""HTTP, read-only, auth, and recovery tests for the operation archive lookup.

The operation-archive endpoint is::

    GET /v1/replicas/{replicaId}/operations/{operationId}

It returns one first-accepted operation addressed by its ``(replicaId,
operationId)`` identity. Everything here goes through the real HTTP entry
point (``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import quote

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

TOKEN = "s3cret-token_123"
AUTH_HEADER = f"Bearer {TOKEN}"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class HttpServerTestCase(unittest.TestCase):
    """Spin up one in-memory server per class; reset the store per test."""

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

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_archive(self, replica: str, operation_id: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/replicas/{replica}/operations/{operation_id}")


class ArchiveContentTests(HttpServerTestCase):
    def test_accepted_write_is_returned_with_full_operation(self) -> None:
        op = operation("op-1", "color", "blue", {"r1": 1})
        self.assertEqual(self.post_operation("r1", op)[0], 201)
        status, payload = self.get_archive("r1", "op-1")
        self.assertEqual(status, 200)
        # The response carries exactly replicaId and the full operation.
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})
        self.assertEqual(set(payload["operation"]), {"operationId", "key", "value", "clock"})

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations/absent",
            "/v1/replicas/absent/operations/op-1",
            "/v1/replicas/absent/operations/absent",
        ):
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_stale_write_is_queryable(self) -> None:
        self.post_operation("r1", operation("new", "k", "new", {"r1": 2}))
        stale = operation("stale", "k", "stale", {"r1": 1})
        self.assertEqual(self.post_operation("r1", stale)[0], 201)
        status, payload = self.get_archive("r1", "stale")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": stale})

    def test_manual_resolution_is_queryable(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        self.assertEqual(self.request("POST", "/v1/states/k/resolve", body)[0], 201)
        status, payload = self.get_archive("r3", "fix-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "replicaId": "r3",
                "operation": operation("fix-1", "k", "merged", {"r1": 1, "r2": 1, "r3": 1}),
            },
        )

    def test_auto_resolution_is_queryable(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        body = {
            "replicaId": "r3",
            "operationId": "auto-1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        self.assertEqual(self.request("POST", "/v1/states/k/resolve/auto", body)[0], 201)
        status, payload = self.get_archive("r3", "auto-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "replicaId": "r3",
                "operation": operation("auto-1", "k", "v1", {"r1": 1, "r2": 1, "r3": 1}),
            },
        )

    def test_sync_imported_record_is_queryable(self) -> None:
        op = operation("imp-1", "k", "v", {"r9": 4})
        status, _ = self.request(
            "POST", "/v1/sync/operations", {"operations": [{"replicaId": "r9", "operation": op}]}
        )
        self.assertEqual(status, 201)
        status, payload = self.get_archive("r9", "imp-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r9", "operation": op})

    def test_replay_adds_no_new_record(self) -> None:
        op = operation("op-1", "k", "v", {"r1": 1})
        self.assertEqual(self.post_operation("r1", op)[0], 201)
        self.assertEqual(self.post_operation("r1", op)[0], 200)
        status, payload = self.get_archive("r1", "op-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})

    def test_conflict_keeps_original_archive(self) -> None:
        op = operation("op-1", "k", "v", {"r1": 1})
        self.post_operation("r1", op)
        status, _ = self.post_operation("r1", operation("op-1", "k", "other", {"r1": 1}))
        self.assertEqual(status, 409)
        status, payload = self.get_archive("r1", "op-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})

    def test_invalid_request_creates_no_archive(self) -> None:
        status, _ = self.post_operation("r1", operation("bad", "", "v", {"r1": 1}))
        self.assertEqual(status, 400)
        status, payload = self.get_archive("r1", "bad")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_lookup_is_read_only(self) -> None:
        self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_state = self.request("GET", "/v1/states/k")
        _, before_digest = self.request("GET", "/v1/verification/digest")
        self.assertEqual(self.get_archive("r1", "op-1")[0], 200)
        self.assertEqual(self.get_archive("r1", "absent")[0], 404)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_state = self.request("GET", "/v1/states/k")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_digest, after_digest)


class ArchiveRoutingTests(HttpServerTestCase):
    def test_route_shape_errors_are_404(self) -> None:
        self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations",  # missing operationId
            "/v1/replicas/operations/op-1",  # missing replicaId shape
            "/v1/replicas/r1/operations/op-1/extra",  # extra segment
            "/v1/replicas//operations/op-1",  # empty replica segment
            "/v1/replicas/r1/operations/",  # empty operation segment
            "/v1/replicas/r1/archive/op-1",  # unknown literal
            "/v1/replicas",
        ):
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("op-1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?after=0&limit=1"):
            with self.subTest(query=query):
                status, payload = self.request(
                    "GET", f"/v1/replicas/r1/operations/op-1{query}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_error_beats_query_error(self) -> None:
        # A malformed route with a query string is a 404, never a 400.
        for path in (
            "/v1/replicas/r1/operations?x=1",
            "/v1/replicas/r1/operations/op-1/extra?x=1",
            "/nope?x=1",
        ):
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_segments_are_percent_decoded(self) -> None:
        op = operation("op 1/ü", "k", "v", {"r 1": 1})
        self.assertEqual(self.post_operation("r%201", op)[0], 201)
        status, payload = self.request(
            "GET",
            "/v1/replicas/r%201/operations/" + quote("op 1/ü", safe=""),
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r 1", "operation": op})

    def test_response_uses_utf8_and_explicit_content_length(self) -> None:
        op = operation("op-ü", "k", "välue-雪", {"r1": 1})
        self.post_operation("r1", op)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/replicas/r1/operations/" + quote("op-ü"))
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.getheader("Content-Type"), "application/json; charset=utf-8"
        )
        self.assertEqual(int(response.getheader("Content-Length")), len(raw))
        self.assertEqual(
            json.loads(raw.decode("utf-8")), {"replicaId": "r1", "operation": op}
        )


class ArchiveAuthTests(unittest.TestCase):
    """With authentication enabled the archive route follows the GET rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, auth_token=TOKEN)
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
        self, method: str, path: str, body: object = None, auth: str | None = AUTH_HEADER
    ) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health", auth=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_archive_requires_bearer_token(self) -> None:
        op = operation("op-1", "k", "v", {"r1": 1})
        self.assertEqual(
            self.request("POST", "/v1/replicas/r1/operations", op)[0], 201
        )
        # Missing and wrong tokens are rejected before any state access.
        for auth in (None, "Bearer wrong", "basic x"):
            with self.subTest(auth=auth):
                status, payload = self.request(
                    "GET", "/v1/replicas/r1/operations/op-1", auth=auth
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
        status, payload = self.request("GET", "/v1/replicas/r1/operations/op-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"replicaId": "r1", "operation": op})

    def test_unauthenticated_unknown_route_is_401_not_404(self) -> None:
        status, payload = self.request("GET", "/v1/replicas/r1/operations", auth=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})


class ArchivePersistenceTests(unittest.TestCase):
    """A data-file restart preserves archive hits, 404s, and error states."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-archive-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.data_file = str(Path(self.tmpdir) / "state.json")

    def run_server(self, actions) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            actions(server.server_address[1])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @staticmethod
    def request(port: int, method: str, path: str, body: object = None) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method, path, body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_restart_preserves_archive_results(self) -> None:
        op = operation("op-1", "k", "v", {"r1": 1})

        def first(port: int) -> None:
            self.assertEqual(
                self.request(port, "POST", "/v1/replicas/r1/operations", op)[0], 201
            )
            # A conflicting retry is rejected and creates no second record.
            self.assertEqual(
                self.request(
                    port, "POST", "/v1/replicas/r1/operations",
                    operation("op-1", "k", "other", {"r1": 1}),
                )[0],
                409,
            )

        def second(port: int) -> None:
            status, payload = self.request(
                port, "GET", "/v1/replicas/r1/operations/op-1"
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"replicaId": "r1", "operation": op})
            status, payload = self.request(
                port, "GET", "/v1/replicas/r1/operations/absent"
            )
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})
            status, payload = self.request(
                port, "GET", "/v1/replicas/r1/operations/op-1?x=1"
            )
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})

        self.run_server(first)
        self.run_server(second)


if __name__ == "__main__":
    unittest.main()
