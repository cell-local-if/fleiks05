"""Tests for the per-key audit integrity digest endpoint::

    GET /v1/audit/keys/{key}/digest

It returns exactly three fields — ``algorithm`` (always ``"sha256"``),
``digest`` (64 lowercase hex chars), and ``operations`` (the number of
accepted operations for the key) — computed from a single snapshot under
the shared commit lock. The digest covers the key's complete audit stream
in global commit order, including stale writes that added no candidate and
accepted conflict repairs, serialized as a compact UTF-8 JSON array whose
entries have a fixed shape and field order and whose strings use minimal
escaping.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _key_audit_digest_input,
)

DIGEST_FIELDS = {"algorithm", "digest", "operations"}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


def expected_digest(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DigestInputTests(unittest.TestCase):
    """The canonical audit-digest-input byte format, pinned against literals."""

    def test_empty_stream_serializes_to_empty_array(self) -> None:
        self.assertEqual(_key_audit_digest_input([]), b"[]")

    def test_single_record_layout_and_field_order(self) -> None:
        records = [("r1", operation("o1", "k", "v", {"r1": 1}))]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r1","operation":{'
            b'"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}]',
        )

    def test_records_follow_global_commit_order(self) -> None:
        records = [
            ("r2", operation("a", "k", "1", {"r2": 1})),
            ("r1", operation("b", "k", "2", {"r1": 1})),
            ("r3", operation("c", "k", "3", {"r3": 1})),
        ]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r2","operation":{'
            b'"operationId":"a","key":"k","value":"1","clock":{"r2":1}}},'
            b'{"replicaId":"r1","operation":{'
            b'"operationId":"b","key":"k","value":"2","clock":{"r1":1}}},'
            b'{"replicaId":"r3","operation":{'
            b'"operationId":"c","key":"k","value":"3","clock":{"r3":1}}}]',
        )

    def test_clock_components_are_sorted(self) -> None:
        records = [
            ("r1", operation("o1", "k", "v", {"z": 1, "a": 3, "m": 2})),
        ]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r1","operation":{'
            b'"operationId":"o1","key":"k","value":"v",'
            b'"clock":{"a":3,"m":2,"z":1}}}]',
        )

    def test_string_escaping_is_minimal(self) -> None:
        records = [
            ('q"\\', operation("o\b1", 'k"\\', "line\nbreak\ttab\x01bell\x07", {"q": 1})),
        ]
        # Only quotes, backslashes, and control characters are escaped;
        # control characters always use lowercase \u00XX, and every other
        # Unicode code point is written literally as UTF-8.
        expected = (
            '[{"replicaId":"q\\"\\\\","operation":{'
            '"operationId":"o\\u00081","key":"k\\"\\\\",'
            '"value":"line\\u000abreak\\u0009tab\\u0001bell\\u0007",'
            '"clock":{"q":1}}}]'
        )
        self.assertEqual(_key_audit_digest_input(records), expected.encode("utf-8"))

    def test_non_ascii_code_points_are_written_literally(self) -> None:
        records = [
            ("réplica-1", operation("o1", "clé", "héllo→世界", {"réplica-1": 1})),
        ]
        expected = (
            '[{"replicaId":"réplica-1","operation":{'
            '"operationId":"o1","key":"clé","value":"héllo→世界",'
            '"clock":{"réplica-1":1}}}]'
        )
        self.assertEqual(_key_audit_digest_input(records), expected.encode("utf-8"))


class DigestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_key_without_history_uses_empty_array(self) -> None:
        self.assertEqual(
            self.store.get_key_audit_digest("absent"),
            {"algorithm": "sha256", "digest": EMPTY_DIGEST, "operations": 0},
        )

    def test_digest_matches_canonical_stream(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["algorithm"], "sha256")
        self.assertRegex(result["digest"], DIGEST_RE)
        self.assertEqual(result["operations"], 1)
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":{'
                '"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}]'
            ),
        )

    def test_digest_covers_stale_writes_and_repairs_in_commit_order(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # A stale write is recorded although it adds no candidate.
        self.assertIs(
            self.store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0})),
            HTTPStatus.CREATED,
        )
        status, error = self.store.apply_resolution(
            "k",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.assertIsNone(error)
        self.assertIs(status, HTTPStatus.CREATED)
        # Other keys, replays, conflicts, and checkpoints never enter the stream.
        self.store.apply_operation("r4", operation("x1", "x", "x", {"r4": 1}))
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 1}))
        self.store.save_checkpoint("peer-a", 4)

        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["operations"], 4)
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":{'
                '"operationId":"o1","key":"k","value":"v1","clock":{"r1":1}}},'
                '{"replicaId":"r2","operation":{'
                '"operationId":"o2","key":"k","value":"v2","clock":{"r2":1}}},'
                '{"replicaId":"r1","operation":{'
                '"operationId":"o3","key":"k","value":"old","clock":{"r1":0}}},'
                '{"replicaId":"r3","operation":{"operationId":"fix-1",'
                '"key":"k","value":"merged",'
                '"clock":{"r1":1,"r2":1,"r3":1}}}]'
            ),
        )
        # Other keys have isolated streams.
        self.assertEqual(self.store.get_key_audit_digest("x")["operations"], 1)
        self.assertEqual(self.store.get_key_audit_digest("absent")["operations"], 0)

    def test_imported_batch_records_land_in_commit_order(self) -> None:
        self.store.import_operations(
            [
                ("r1", operation("i1", "k", "v1", {"r1": 1})),
                ("r2", operation("ix", "x", "vx", {"r2": 1})),
                ("r1", operation("i2", "k", "v2", {"r1": 2})),
            ]
        )
        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["operations"], 2)
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":{'
                '"operationId":"i1","key":"k","value":"v1","clock":{"r1":1}}},'
                '{"replicaId":"r1","operation":{'
                '"operationId":"i2","key":"k","value":"v2","clock":{"r1":2}}}]'
            ),
        )

    def test_digest_changes_with_every_accepted_record(self) -> None:
        first = self.store.get_key_audit_digest("k")
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        second = self.store.get_key_audit_digest("k")
        self.assertNotEqual(first["digest"], second["digest"])
        # A stale write still moves the audit digest (unlike the candidate
        # verification digest).
        self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 0}))
        third = self.store.get_key_audit_digest("k")
        self.assertNotEqual(second["digest"], third["digest"])
        self.assertEqual(third["operations"], 2)

    def test_rejected_batch_changes_nothing(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        before = self.store.get_key_audit_digest("k")
        status, _, _ = self.store.import_operations(
            [
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(self.store.get_key_audit_digest("k"), before)

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.store.get_sync_operations(0, 100)[0]
        first = self.store.get_key_audit_digest("k")
        second = self.store.get_key_audit_digest("k")
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)


class DigestRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_digest_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_resolution(
            "k",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        store.apply_operation("r4", operation("x1", "x", "x", {"r4": 1}))
        store.import_operations(
            [("r5", operation("i1", "k", "iv", {"r5": 1}))]
        )
        store.save_checkpoint("peer-a", 5)
        before = {
            "k": store.get_key_audit_digest("k"),
            "x": store.get_key_audit_digest("x"),
            "absent": store.get_key_audit_digest("absent"),
        }

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_key_audit_digest("k"), before["k"])
        self.assertEqual(recovered.get_key_audit_digest("x"), before["x"])
        self.assertEqual(recovered.get_key_audit_digest("absent"), before["absent"])


class DigestHttpServerTests(unittest.TestCase):
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

    def raw_request(
        self, method: str, path: str, body: object = None
    ) -> tuple[int, dict, bytes, list[tuple[str, str]]]:
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
        payload = json.loads(raw.decode("utf-8")) if raw else None
        headers = response.getheaders()
        conn.close()
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def digest(self, key: str = "k", suffix: str = "") -> tuple[int, dict]:
        return self.request("GET", f"/v1/audit/keys/{key}/digest{suffix}")

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_missing_key_history(self) -> None:
        status, payload = self.digest("absent")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {"algorithm": "sha256", "digest": EMPTY_DIGEST, "operations": 0},
        )

    def test_payload_shape_and_headers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request("GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), DIGEST_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertRegex(payload["digest"], DIGEST_RE)
        self.assertIs(type(payload["operations"]), int)
        self.assertEqual(payload["operations"], 1)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_digest_matches_canonical_form_over_http(self) -> None:
        # Multi-component clock sent out of order, a control character, and
        # non-ASCII code points exercise the canonical byte format end to end.
        self.post_operation(
            "r1", operation("o1", "clé", "a\nb→", {"r2": 1, "r1": 2})
        )
        self.post_operation("r1", operation("o2", "clé", "old", {"r1": 1}))
        status, payload = self.digest("cl%C3%A9")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 2)
        self.assertEqual(
            payload["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":{'
                '"operationId":"o1","key":"clé","value":"a\\u000ab→",'
                '"clock":{"r1":2,"r2":1}}},'
                '{"replicaId":"r1","operation":{'
                '"operationId":"o2","key":"clé","value":"old",'
                '"clock":{"r1":1}}}]'
            ),
        )

    def test_operations_count_matches_audit_stream(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        self.post_operation("r4", operation("x1", "x", "x", {"r4": 1}))
        status, digest_payload = self.digest("k")
        self.assertEqual(status, 200)
        status, audit_payload = self.request("GET", "/v1/audit/keys/k/operations")
        self.assertEqual(status, 200)
        self.assertEqual(
            digest_payload["operations"], len(audit_payload["operations"])
        )

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/audit/keys/k/digest?x=1",
            "/v1/audit/keys/k/digest?after=0",
            "/v1/audit/keys/k/digest?x=",
            "/v1/audit/keys/k/digest?x",
            "/v1/audit/keys/k/digest?=1",
            "/v1/audit/keys/k/digest?x=1&x=2",
            "/v1/audit/keys/k/digest?operations=1",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.request("GET", "/v1/audit/keys/k/digest?")
        self.assertEqual(status, 200)

    def test_missing_or_extra_path_segments_are_404(self) -> None:
        for path in (
            "/v1/audit/keys/k/digest/extra",
            "/v1/audit/keys/k",
            "/v1/audit/keys",
            "/v1/audit",
            "/v1/audit/keys",
            "/v1/audit/keys//digest",
            "/v1/audit/keys//operations",
            "/v1/audit/k/digest",
            "/v1/audit/keys/k/other",
            "/v1/audit/keys/k/operations/digest",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_404_wins_over_query_validation_for_bad_shape(self) -> None:
        status, payload = self.request("GET", "/v1/audit/keys/k/digest/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/audit/keys/k/digest", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_digest_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, first = self.digest()
        for _ in range(3):
            status, payload = self.digest()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 1)
        status, audit_payload = self.request("GET", "/v1/audit/keys/k/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(audit_payload["operations"]), 1)

    def test_concurrent_commits_observe_consistent_snapshots(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                result = self.server.store.get_key_audit_digest("k")
                if result["algorithm"] != "sha256":
                    violations.append("wrong algorithm")
                if not DIGEST_RE.match(result["digest"]):
                    violations.append("malformed digest")
                if result["operations"] < 0:
                    violations.append("negative operations")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                replica = f"r{index}"
                self.post_operation(
                    replica,
                    operation(f"op-{index}", "k", f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 40)
        expected = (
            "["
            + ",".join(
                '{"replicaId":"r' + str(i) + '","operation":{'
                '"operationId":"op-' + str(i) + '","key":"k","value":"v' + str(i)
                + '","clock":{"r' + str(i) + '":1}}}'
                for i in range(40)
            )
            + "]"
        )
        self.assertEqual(payload["digest"], expected_digest(expected))


class PersistentDigestHttpServerTests(unittest.TestCase):
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

    def seed(self, server: SemanticStateServer) -> None:
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        self.request(
            server,
            "POST",
            "/v1/states/k/resolve",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o3", "k", "old", {"r1": 0}),
        )
        self.request(
            server,
            "POST",
            "/v1/sync/operations",
            {"operations": [record("r4", operation("i1", "k", "iv", {"r4": 1}))]},
        )

    def test_digest_survives_restart(self) -> None:
        server = self.start_server()
        self.seed(server)
        status, before = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(before["operations"], 5)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_digest_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.seed(server)
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/audit/keys/k/digest")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)

    def test_failed_commit_and_rejected_batch_do_not_affect_digest(self) -> None:
        server = self.start_server()
        self.seed(server)
        status, before = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)

        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=server_module.PersistenceError("disk gone"),
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r9/operations",
                operation("p9", "k", "v9", {"r9": 1}),
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            status, during = self.request(server, "GET", "/v1/audit/keys/k/digest")
            self.assertEqual(status, 200)
            self.assertEqual(during, before)

            status, payload = self.request(
                server,
                "POST",
                "/v1/sync/operations",
                {
                    "operations": [
                        record("r8", operation("p8", "k", "v8", {"r8": 1})),
                        record("r1", operation("o1", "k", "tampered", {"r1": 1})),
                    ]
                },
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "operation_conflict"})

        status, after = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
