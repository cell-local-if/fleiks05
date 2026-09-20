import http.client
import json
import os
import shutil
import socket
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
)

OP_PATH = "/v1/replicas/r1/operations"
SYNC_PATH = "/v1/sync/operations"
RESOLVE_PATH = "/v1/states/k/resolve"
RESOLVE_AUTO_PATH = "/v1/states/k/resolve/auto"
CHECKPOINT_PATH = "/v1/sync/peers/peer-a/checkpoint"
ALL_POST_PATHS = (OP_PATH, SYNC_PATH, RESOLVE_PATH, RESOLVE_AUTO_PATH, CHECKPOINT_PATH)

OVER_LIMIT = str(MAX_BODY_BYTES + 1)


def padded_document(document: dict, value_path: list, target_length: int) -> bytes:
    """Return ``document`` serialized to exactly ``target_length`` bytes.

    The string at ``value_path`` is padded with ``x`` characters so the
    compact JSON serialization reaches the requested raw byte length.
    """
    base = json.dumps(document, separators=(",", ":")).encode("utf-8")
    pad = target_length - len(base)
    assert pad >= 0, "document is already larger than the target"
    node = document
    for key in value_path[:-1]:
        node = node[key]
    node[value_path[-1]] = "x" * pad
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


def operation_document(op_id: str = "op-1", key: str = "k", value: str = "v") -> dict:
    return {"operationId": op_id, "key": key, "value": value, "clock": {"r1": 1}}


class RequestLimitTests(unittest.TestCase):
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

    def post_raw(self, path: str, headers: list, body: bytes = b"") -> tuple[int, dict]:
        """POST with exact control over the header lines that are sent."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_json(self, path: str, document: dict) -> tuple[int, dict]:
        body = json.dumps(document).encode("utf-8")
        return self.post_raw(path, [("Content-Length", str(len(body)))], body)

    def post_operation(self, op_id: str, key: str, value: str, clock: dict) -> tuple[int, dict]:
        return self.post_json(
            OP_PATH, {"operationId": op_id, "key": key, "value": value, "clock": clock}
        )

    # -- Content-Length validation, uniform across all five POST endpoints --

    def test_missing_content_length_is_400_on_all_post_endpoints(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(path, [], b"{}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_content_lengths_are_400_on_all_post_endpoints(self) -> None:
        bad_values = [
            "abc",  # not a number
            "",  # empty value
            "+5",  # explicit sign
            "-5",  # negative
            "5 ",  # trailing whitespace
            "1 2",  # inner whitespace
            "5.0",  # decimal
            "²",  # non-ASCII digit
            "5,5",  # folded list is not a plain integer
        ]
        for path in ALL_POST_PATHS:
            for value in bad_values:
                with self.subTest(path=path, value=value):
                    status, payload = self.post_raw(
                        path, [("Content-Length", value)], b"{}"
                    )
                    self.assertEqual(status, 400)
                    self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400_on_all_post_endpoints(self) -> None:
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(
                    path,
                    [("Content-Length", "2"), ("Content-Length", "3")],
                    b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_identical_duplicate_content_length_headers_are_accepted(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        status, payload = self.post_raw(
            OP_PATH,
            [("Content-Length", str(len(body))), ("Content-Length", str(len(body)))],
            body,
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")

    def test_leading_zeros_are_a_plain_decimal_length(self) -> None:
        body = json.dumps(operation_document()).encode("utf-8")
        status, _ = self.post_raw(
            OP_PATH, [("Content-Length", "000" + str(len(body)))], body
        )
        self.assertEqual(status, 201)

    def test_zero_content_length_keeps_empty_body_semantics(self) -> None:
        status, payload = self.post_raw(OP_PATH, [("Content-Length", "0")])
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    # -- Over-limit declarations: 413 before the body is read or parsed --

    def test_over_limit_declaration_is_413_on_all_post_endpoints(self) -> None:
        # The content itself is invalid JSON; the declared size wins.
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(
                    path, [("Content-Length", OVER_LIMIT)], b"not json"
                )
                self.assertEqual(status, 413)
                self.assertEqual(payload, {"error": "payload_too_large"})

    def test_over_limit_is_413_even_with_a_full_invalid_body(self) -> None:
        body = b"x" * (MAX_BODY_BYTES + 1)
        status, payload = self.post_raw(
            OP_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_absurdly_long_content_length_digits_are_413(self) -> None:
        status, payload = self.post_raw(
            OP_PATH, [("Content-Length", "9" * 5000)], b"{}"
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def get_state(self, key: str) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", f"/v1/states/{key}")
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_over_limit_resolve_does_not_consume_state(self) -> None:
        # A resolve over the limit must not disturb an existing conflict.
        self.assertEqual(self.post_operation("op-1", "k", "v1", {"r1": 1})[0], 201)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = json.dumps(
            {"operationId": "op-2", "key": "k", "value": "v2", "clock": {"r2": 1}}
        ).encode("utf-8")
        conn.putrequest("POST", "/v1/replicas/r2/operations")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        self.assertEqual(conn.getresponse().status, 201)
        conn.close()
        status, _ = self.post_raw(
            RESOLVE_PATH, [("Content-Length", OVER_LIMIT)], b"junk"
        )
        self.assertEqual(status, 413)
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)

    # -- Boundary: exactly MAX_BODY_BYTES is processed normally --

    def test_operation_body_at_exact_limit_is_accepted(self) -> None:
        document = operation_document(op_id="op-big", value="")
        base_len = len(json.dumps(document, separators=(",", ":")).encode("utf-8"))
        body = padded_document(document, ["value"], MAX_BODY_BYTES)
        self.assertEqual(len(body), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            OP_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["operationId"], "op-big")
        status, state = self.get_state("k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "x" * (MAX_BODY_BYTES - base_len))

    def test_sync_import_body_at_exact_limit_is_accepted(self) -> None:
        document = {
            "operations": [
                {"replicaId": "r1", "operation": operation_document(op_id="op-s", key="ks", value="")}
            ]
        }
        body = padded_document(document, ["operations", 0, "operation", "value"], MAX_BODY_BYTES)
        self.assertEqual(len(body), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            SYNC_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload, {"status": "created", "accepted": 1, "replayed": 0})

    def test_resolve_body_at_exact_limit_is_accepted(self) -> None:
        self.assertEqual(self.post_operation("op-1", "k", "v1", {"r1": 1})[0], 201)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body2 = json.dumps(
            {"operationId": "op-2", "key": "k", "value": "v2", "clock": {"r2": 1}}
        ).encode("utf-8")
        conn.putrequest("POST", "/v1/replicas/r2/operations")
        conn.putheader("Content-Length", str(len(body2)))
        conn.endheaders(body2)
        self.assertEqual(conn.getresponse().status, 201)
        conn.close()
        document = {
            "replicaId": "r3",
            "operationId": "fix-big",
            "value": "",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "op-1"},
                {"replicaId": "r2", "operationId": "op-2"},
            ],
        }
        body = padded_document(document, ["value"], MAX_BODY_BYTES)
        self.assertEqual(len(body), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            RESOLVE_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["status"], "created")

    def test_invalid_body_at_exact_limit_gets_normal_400_not_413(self) -> None:
        body = b"x" * MAX_BODY_BYTES
        for path in ALL_POST_PATHS:
            with self.subTest(path=path):
                status, payload = self.post_raw(
                    path, [("Content-Length", str(len(body)))], body
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    # -- Forged lengths: exactly the declared bytes are read --

    def test_declared_length_limits_how_much_is_read(self) -> None:
        valid = json.dumps(operation_document(op_id="op-trim")).encode("utf-8")
        body = valid + b"GARBAGE-THAT-MUST-NOT-BE-READ"
        status, payload = self.post_raw(
            OP_PATH, [("Content-Length", str(len(valid)))], body
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["operationId"], "op-trim")

    def test_truncated_body_against_declaration_is_400(self) -> None:
        # Declare more than is ever sent, then close the write side: the
        # server reads only what arrives and validates those bytes.
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            sock.sendall(
                b"POST /v1/replicas/r1/operations HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
                b"{}"
            )
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        status_line = data.split(b"\r\n", 1)[0]
        self.assertIn(b"400", status_line)
        self.assertIn(b'"invalid_request"', data)


class DataFileRejectionInvarianceTests(unittest.TestCase):
    """Rejected requests leave memory, the data file, and the directory untouched."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-limits-")
        cls.data_path = os.path.join(cls.tmpdir, "state.json")
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=cls.data_path
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def request(self, method: str, path: str, document: dict = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        if document is None:
            conn.request(method, path)
        else:
            conn.request(
                method, path, body=json.dumps(document),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_raw(self, path: str, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_rejections_leave_state_file_and_directory_untouched(self) -> None:
        status, _ = self.request(
            "POST", OP_PATH, operation_document(op_id="op-1", key="color", value="blue")
        )
        self.assertEqual(status, 201)
        status, _ = self.request(
            "POST", CHECKPOINT_PATH, {"cursor": 1}
        )
        self.assertEqual(status, 200)

        with open(self.data_path, "rb") as handle:
            bytes_before = handle.read()
        listing_before = sorted(os.listdir(self.tmpdir))
        _, metrics_before = self.request("GET", "/v1/metrics")
        _, state_before = self.request("GET", "/v1/states/color")
        _, checkpoint_before = self.request(
            "GET", "/v1/sync/peers/peer-a/checkpoint"
        )

        rejections = [
            ([], b"{}", 400),  # missing Content-Length
            ([("Content-Length", "abc")], b"{}", 400),  # malformed
            ([("Content-Length", "-1")], b"{}", 400),  # negative
            ([("Content-Length", "2"), ("Content-Length", "9")], b"{}", 400),  # conflict
            ([("Content-Length", OVER_LIMIT)], b"junk", 413),  # over the limit
        ]
        for path in ALL_POST_PATHS:
            for headers, body, expected in rejections:
                with self.subTest(path=path, headers=headers):
                    status, payload = self.post_raw(path, headers, body)
                    self.assertEqual(status, expected)
                    self.assertEqual(
                        payload,
                        {"error": "payload_too_large" if expected == 413 else "invalid_request"},
                    )

        with open(self.data_path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), listing_before)
        _, metrics_after = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics_after, metrics_before)
        _, state_after = self.request("GET", "/v1/states/color")
        self.assertEqual(state_after, state_before)
        _, checkpoint_after = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(checkpoint_after, checkpoint_before)

        # The service keeps processing valid requests after the rejections.
        status, _ = self.request(
            "POST",
            OP_PATH,
            {"operationId": "op-2", "key": "color", "value": "green", "clock": {"r1": 2}},
        )
        self.assertEqual(status, 201)
        _, state = self.request("GET", "/v1/states/color")
        self.assertEqual(state["value"], "green")


if __name__ == "__main__":
    unittest.main()
