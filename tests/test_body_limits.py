"""Tests for the shared POST request-body size boundary.

Every POST interface (local operations, sync import, conflict resolution,
checkpoints) rejects a missing/malformed/conflicting Content-Length with
HTTP 400 and a declared length above 1,048,576 bytes with HTTP 413 before
reading the body, while a body declared at exactly the cap is processed by
the endpoint's usual semantics. Rejected requests must leave memory, the
data file, and the data file's directory untouched.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

LOCAL = "/v1/replicas/r1/operations"
SYNC = "/v1/sync/operations"
RESOLVE = "/v1/states/color/resolve"
CHECKPOINT = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_ROUTES = (LOCAL, SYNC, RESOLVE, CHECKPOINT)


def operation_json(operation_id: str = "op-1", key: str = "color", value: str = "blue") -> str:
    return json.dumps(
        {
            "operationId": operation_id,
            "key": key,
            "value": value,
            "clock": {"r1": 1},
        }
    )


def pad_to_limit(document: str) -> bytes:
    """Pad a JSON document with trailing whitespace to exactly the cap."""
    raw = document.encode("utf-8")
    assert len(raw) <= MAX_BODY_BYTES
    return raw + b" " * (MAX_BODY_BYTES - len(raw))


class BodyLimitTestBase(unittest.TestCase):
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
        self.server.store = StateStore()

    def raw_post(
        self,
        path: str,
        body: bytes = b"",
        content_lengths: tuple[str, ...] = (),
        send_body: bool = True,
    ) -> tuple[int, dict]:
        """POST with explicit control over the Content-Length header(s).

        ``send_body=False`` sends only the headers, which lets a test prove
        the server answers an oversized declaration without waiting for the
        body bytes.
        """
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for value in content_lengths:
            conn.putheader("Content-Length", value)
        conn.endheaders(body if send_body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post(self, path: str, body: bytes) -> tuple[int, dict]:
        return self.raw_post(path, body, (str(len(body)),))

    def raw_get(self, path: str) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload


class ContentLengthValidationTests(BodyLimitTestBase):
    def test_missing_content_length_is_400_on_all_post_routes(self) -> None:
        for path in ALL_POST_ROUTES:
            with self.subTest(path=path):
                status, payload = self.raw_post(path, content_lengths=())
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_content_length_is_400_on_all_post_routes(self) -> None:
        for token in ("", "abc", "+5", "-1", "1.0", "1 0", "0x10"):
            for path in ALL_POST_ROUTES:
                with self.subTest(token=token, path=path):
                    status, payload = self.raw_post(path, content_lengths=(token,))
                    self.assertEqual(status, 400)
                    self.assertEqual(payload, {"error": "invalid_request"})

    def test_non_ascii_content_length_is_400(self) -> None:
        # http.client cannot send non-latin-1 header values, so this case
        # is driven over a raw socket with the fullwidth digits "１２".
        request = (
            "POST /v1/replicas/r1/operations HTTP/1.0\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Length: １２\r\n"
            "\r\n"
        ).encode("utf-8")
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(request)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
        self.assertIn(b" 400 ", head.split(b"\r\n", 1)[0])
        self.assertEqual(json.loads(body.decode("utf-8")), {"error": "invalid_request"})

    def test_conflicting_content_lengths_are_400_on_all_post_routes(self) -> None:
        for path in ALL_POST_ROUTES:
            with self.subTest(path=path):
                status, payload = self.raw_post(path, content_lengths=("5", "6"))
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_identical_repeated_content_length_is_accepted(self) -> None:
        body = operation_json().encode("utf-8")
        status, payload = self.raw_post(
            LOCAL, body, (str(len(body)), str(len(body)))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")

    def test_unknown_post_route_stays_404(self) -> None:
        status, payload = self.raw_post("/nope", content_lengths=(str(MAX_BODY_BYTES + 1),))
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class PayloadTooLargeTests(BodyLimitTestBase):
    def test_over_limit_declaration_is_413_on_all_post_routes(self) -> None:
        for path in ALL_POST_ROUTES:
            with self.subTest(path=path):
                status, payload = self.raw_post(
                    path, content_lengths=(str(MAX_BODY_BYTES + 1),), send_body=False
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_413_is_answered_without_reading_the_body(self) -> None:
        # Only headers are sent; if the server tried to read the declared
        # body the response would not arrive until the client timed out.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", LOCAL, skip_accept_encoding=True)
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(
            json.loads(response.read().decode("utf-8")), {"error": "payload_too_large"}
        )
        conn.close()

    def test_oversized_invalid_content_still_gets_413(self) -> None:
        # The over-limit declaration wins over any content-level problem:
        # the body is never read, parsed, or validated.
        status, payload = self.raw_post(
            LOCAL, content_lengths=(str(MAX_BODY_BYTES + 100),), send_body=False
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_forged_short_length_truncates_to_400(self) -> None:
        # A declared length smaller than the shipped body makes the server
        # read only the declared prefix, which fails the usual JSON checks.
        body = operation_json().encode("utf-8")
        status, payload = self.raw_post(LOCAL, body, (str(len(body) - 1),))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class BoundaryLengthTests(BodyLimitTestBase):
    def test_local_operation_at_exact_limit_is_created(self) -> None:
        status, payload = self.post(LOCAL, pad_to_limit(operation_json()))
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")
        _, state = self.raw_get("/v1/states/color")
        self.assertEqual(state["value"], "blue")

    def test_sync_import_at_exact_limit_is_created(self) -> None:
        document = json.dumps(
            {"operations": [{"replicaId": "r1", "operation": json.loads(operation_json())}]}
        )
        status, payload = self.post(SYNC, pad_to_limit(document))
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 1, "replayed": 0})

    def test_resolve_at_exact_limit_is_created(self) -> None:
        self.post(LOCAL, operation_json("op-1", "color", "v1").encode("utf-8"))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r2/operations",
            body=json.dumps(
                {
                    "operationId": "op-2",
                    "key": "color",
                    "value": "v2",
                    "clock": {"r2": 1},
                }
            ),
        )
        self.assertEqual(conn.getresponse().status, 201)
        conn.close()
        document = json.dumps(
            {
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "merged",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "candidates": [
                    {"replicaId": "r1", "operationId": "op-1"},
                    {"replicaId": "r2", "operationId": "op-2"},
                ],
            }
        )
        status, payload = self.post(RESOLVE, pad_to_limit(document))
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")

    def test_checkpoint_at_exact_limit_is_ok(self) -> None:
        status, payload = self.post(CHECKPOINT, pad_to_limit('{"cursor":0}'))
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 0})

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        status, payload = self.post(LOCAL, pad_to_limit("{oops"))
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class RejectionLeavesNoTraceTests(BodyLimitTestBase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_path = os.path.join(self.tmpdir.name, "state.json")
        self.server.store = StateStore(data_file=self.data_path)

    def snapshot(self) -> tuple[bytes, list[str], dict]:
        with open(self.data_path, "rb") as handle:
            file_bytes = handle.read()
        listing = sorted(os.listdir(self.tmpdir.name))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/metrics")
        metrics = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return file_bytes, listing, metrics

    def test_rejections_leave_state_and_data_file_unchanged(self) -> None:
        status, _ = self.post(LOCAL, operation_json().encode("utf-8"))
        self.assertEqual(status, 201)
        before = self.snapshot()

        rejections = [
            # Oversized declarations (no body is ever read).
            (LOCAL, (str(MAX_BODY_BYTES + 1),), b"", False, 413),
            (SYNC, (str(MAX_BODY_BYTES + 1),), b"", False, 413),
            (RESOLVE, (str(MAX_BODY_BYTES + 1),), b"", False, 413),
            (CHECKPOINT, (str(MAX_BODY_BYTES + 1),), b"", False, 413),
            # Missing and malformed Content-Length.
            (LOCAL, (), b"", True, 400),
            (SYNC, ("not-a-number",), b"", True, 400),
            (RESOLVE, ("-3",), b"", True, 400),
            (CHECKPOINT, ("2", "3"), b"", True, 400),
            # Forged short length truncating a valid document.
            (LOCAL, (str(len(operation_json()) - 1),), operation_json().encode("utf-8"), True, 400),
        ]
        for path, lengths, body, send_body, expected in rejections:
            with self.subTest(path=path, lengths=lengths):
                status, payload = self.raw_post(
                    path, body, lengths, send_body=send_body
                )
                self.assertEqual(status, expected)
                self.assertEqual(payload["error"], "invalid_request" if expected == 400 else "payload_too_large")

        after = self.snapshot()
        self.assertEqual(before, after)

        # The one accepted operation is still the only visible state.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/states/color")
        state = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        self.assertEqual(
            state,
            {"key": "color", "value": "blue", "clock": {"r1": 1}, "status": "resolved"},
        )
        status, payload = self.raw_get(CHECKPOINT)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
