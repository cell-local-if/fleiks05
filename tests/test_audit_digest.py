"""Tests for the per-key audit-integrity digest endpoint::

    GET /v1/audit/keys/{key}/digest

It returns exactly three fields — ``algorithm`` (always ``"sha256"``),
``digest`` (64 lowercase hex chars), and ``operations`` (the number of
accepted operations for that key) — computed from one snapshot under the
shared commit lock. The digest covers the key's whole audit stream in
global commit order, including stale writes and accepted repairs,
serialized as a compact UTF-8 JSON array with fixed field ordering and
minimal string escaping; a key with no history hashes ``[]``.

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


def digest_of_audit(operations: list[dict]) -> str:
    """SHA-256 of audit-endpoint records, converted to (replica, op) pairs."""
    return hashlib.sha256(
        _key_audit_digest_input(
            [(entry["replicaId"], entry["operation"]) for entry in operations]
        )
    ).hexdigest()


class DigestInputTests(unittest.TestCase):
    """The canonical digest-input byte format, pinned against literals."""

    def test_empty_stream_serializes_to_empty_array(self) -> None:
        self.assertEqual(_key_audit_digest_input([]), b"[]")

    def test_single_record_layout_and_field_order(self) -> None:
        records = [("r1", operation("o1", "k", "v", {"r1": 1}))]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r1","operation":'
            b'{"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}]',
        )

    def test_records_keep_commit_order_and_no_sorting(self) -> None:
        records = [
            ("r2", operation("o9", "k", "z", {"r2": 1})),
            ("r1", operation("o1", "k", "a", {"r1": 1})),
            ("r1", operation("o5", "k", "m", {"r1": 2})),
        ]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r2","operation":'
            b'{"operationId":"o9","key":"k","value":"z","clock":{"r2":1}}},'
            b'{"replicaId":"r1","operation":'
            b'{"operationId":"o1","key":"k","value":"a","clock":{"r1":1}}},'
            b'{"replicaId":"r1","operation":'
            b'{"operationId":"o5","key":"k","value":"m","clock":{"r1":2}}}]',
        )

    def test_clock_components_are_sorted(self) -> None:
        records = [
            ("r1", operation("o1", "k", "v", {"z": 1, "a": 3, "m": 2, "r1": 4}))
        ]
        self.assertEqual(
            _key_audit_digest_input(records),
            b'[{"replicaId":"r1","operation":'
            b'{"operationId":"o1","key":"k","value":"v",'
            b'"clock":{"a":3,"m":2,"r1":4,"z":1}}}]',
        )

    def test_string_escaping_is_minimal(self) -> None:
        records = [
            ('q"\\r', operation('o"1', 'k"\\', "line\nbreak\ttabbell", {"r1": 1}))
        ]
        # Only quotes, backslashes, and control characters are escaped;
        # control characters always use lowercase \u00XX, and every other
        # Unicode code point is written literally as UTF-8.
        expected = (
            '[{"replicaId":"q\\"\\\\r","operation":{'
            '"operationId":"o\\"1","key":"k\\"\\\\",'
            '"value":"line\\u000abreak\\u0009tab\\u0001bell\\u0007",'
            '"clock":{"r1":1}}}]'
        )
        self.assertEqual(_key_audit_digest_input(records), expected.encode("utf-8"))

    def test_non_ascii_code_points_are_written_literally(self) -> None:
        records = [
            ("réplica-1", operation("o1", "clé", "héllo→世界", {"réplica-1": 1}))
        ]
        expected = (
            '[{"replicaId":"réplica-1","operation":'
            '{"operationId":"o1","key":"clé","value":"héllo→世界",'
            '"clock":{"réplica-1":1}}}]'
        )
        self.assertEqual(_key_audit_digest_input(records), expected.encode("utf-8"))


class DigestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_unknown_key_is_hash_of_empty_array(self) -> None:
        self.assertEqual(
            self.store.get_key_audit_digest("absent"),
            {"algorithm": "sha256", "digest": EMPTY_DIGEST, "operations": 0},
        )

    def test_digest_matches_canonical_snapshot(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["algorithm"], "sha256")
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":'
                '{"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}]'
            ),
        )
        self.assertEqual(result["operations"], 1)

    def test_digest_covers_stale_writes_and_repairs_in_commit_order(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # A stale write is accepted but adds no candidate; it must hash.
        self.assertIs(
            self.store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0})),
            server_module.HTTPStatus.CREATED,
        )
        # A resolution repair is an ordinary accepted operation.
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
        # A checkpoint never enters the audit stream.
        self.store.save_checkpoint("peer-a", 4)
        # Identical replays and conflicting rewrites add no record.
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 1}))
        # Other keys are isolated.
        self.store.apply_operation("r9", operation("x1", "x", "x", {"r9": 1}))

        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["operations"], 4)
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":'
                '{"operationId":"o1","key":"k","value":"v1","clock":{"r1":1}}},'
                '{"replicaId":"r2","operation":'
                '{"operationId":"o2","key":"k","value":"v2","clock":{"r2":1}}},'
                '{"replicaId":"r1","operation":'
                '{"operationId":"o3","key":"k","value":"old","clock":{"r1":0}}},'
                '{"replicaId":"r3","operation":'
                '{"operationId":"fix-1","key":"k","value":"merged",'
                '"clock":{"r1":1,"r2":1,"r3":1}}}]'
            ),
        )
        self.assertEqual(
            self.store.get_key_audit_digest("x"),
            {
                "algorithm": "sha256",
                "digest": expected_digest(
                    '[{"replicaId":"r9","operation":'
                    '{"operationId":"x1","key":"x","value":"x","clock":{"r9":1}}}]'
                ),
                "operations": 1,
            },
        )
        self.assertEqual(
            self.store.get_key_audit_digest("absent")["digest"], EMPTY_DIGEST
        )

    def test_imported_batch_records_hash_in_batch_order(self) -> None:
        records = [
            ("r1", operation("i1", "k", "v1", {"r1": 1})),
            ("r2", operation("ix", "x", "vx", {"r2": 1})),
            ("r1", operation("i2", "k", "v2", {"r1": 2})),
        ]
        status, accepted, replayed = self.store.import_operations(records)
        self.assertIs(status, server_module.HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (3, 0))
        result = self.store.get_key_audit_digest("k")
        self.assertEqual(result["operations"], 2)
        self.assertEqual(
            result["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":'
                '{"operationId":"i1","key":"k","value":"v1","clock":{"r1":1}}},'
                '{"replicaId":"r1","operation":'
                '{"operationId":"i2","key":"k","value":"v2","clock":{"r1":2}}}]'
            ),
        )

    def test_conflicting_batch_leaves_digest_untouched(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        before = self.store.get_key_audit_digest("k")
        status, _, _ = self.store.import_operations(
            [
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r1", operation("o1", "k", "tampered", {"r1": 1})),
            ]
        )
        self.assertIs(status, server_module.HTTPStatus.CONFLICT)
        self.assertEqual(self.store.get_key_audit_digest("k"), before)

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        audit_before = self.store.get_key_operations("k", 0, 100)[0]
        first = self.store.get_key_audit_digest("k")
        second = self.store.get_key_audit_digest("k")
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_key_operations("k", 0, 100)[0], audit_before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        # An untouched key stays empty however often it is read.
        self.assertEqual(
            self.store.get_key_audit_digest("absent")["digest"], EMPTY_DIGEST
        )


class DigestRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def seed(self, store: StateStore) -> None:
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
        # Stale write and checkpoint move the log/checkpoint sections.
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        store.save_checkpoint("peer-a", 4)
        store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1}))
        store.import_operations(
            [
                ("r5", operation("b1", "k", "b1", {"r5": 1})),
                ("r6", operation("bx", "x", "bx", {"r6": 1})),
                ("r5", operation("b2", "k", "b2", {"r5": 2})),
            ]
        )

    def test_digest_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        self.seed(store)
        before_k = store.get_key_audit_digest("k")
        before_x = store.get_key_audit_digest("x")
        before_empty = store.get_key_audit_digest("absent")

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_key_audit_digest("k"), before_k)
        self.assertEqual(recovered.get_key_audit_digest("x"), before_x)
        self.assertEqual(recovered.get_key_audit_digest("absent"), before_empty)
        del recovered
        store = StateStore(data_file=self.data_file)
        self.assertEqual(store.get_key_audit_digest("k"), before_k)
        self.assertEqual(store.get_key_audit_digest("k")["operations"], 6)


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
        elif isinstance(body, (bytes, str)):
            conn.request(
                method, path, body=body, headers={"Content-Type": "application/json"}
            )
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

    def get_digest(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)

    def digest(self, key: str = "k", query: str = "") -> tuple[int, dict]:
        return self.request("GET", f"/v1/audit/keys/{key}/digest{query}")

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_sync(self, body: object) -> tuple[int, dict]:
        return self.request("POST", "/v1/sync/operations", body)

    def post_resolve(self, key: str, body: object) -> tuple[int, dict]:
        return self.request("POST", f"/v1/states/{key}/resolve", body)

    def drain_audit(self, key: str, limit: int = 3) -> list[dict]:
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.request(
                "GET", f"/v1/audit/keys/{key}/operations?after={after}&limit={limit}"
            )
            assert status == 200
            seen.extend(payload["operations"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        return seen

    def test_unknown_key_is_200_empty_digest(self) -> None:
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
        self.assertIs(type(payload["digest"]), str)
        self.assertRegex(payload["digest"], DIGEST_RE)
        self.assertIs(type(payload["operations"]), int)
        self.assertEqual(payload["operations"], 1)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_digest_matches_audit_stream_end_to_end(self) -> None:
        # Out-of-order clock components, a control character, and non-ASCII
        # code points exercise the canonical byte format end to end.
        self.post_operation(
            "r1", operation("o1", "clé", "a\nb→", {"r2": 1, "r1": 2})
        )
        self.post_operation("r2", operation("o2", "clé", "c", {"r2": 2}))
        # Unrelated key must not contribute.
        self.post_operation("r9", operation("x1", "other", "x", {"r9": 1}))
        status, payload = self.digest("cl%C3%A9")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 2)
        drained = self.drain_audit("cl%C3%A9", limit=1)
        self.assertEqual(payload["digest"], digest_of_audit(drained))
        self.assertEqual(
            payload["digest"],
            expected_digest(
                '[{"replicaId":"r1","operation":'
                '{"operationId":"o1","key":"clé","value":"a\\u000ab→",'
                '"clock":{"r1":2,"r2":1}}},'
                '{"replicaId":"r2","operation":'
                '{"operationId":"o2","key":"clé","value":"c",'
                '"clock":{"r2":2}}}]'
            ),
        )

    def test_stale_writes_and_repairs_are_covered(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r1", operation("stale", "k", "old", {"r1": 0}))
        self.post_resolve(
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
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 4)
        drained = self.drain_audit("k", limit=1)
        self.assertEqual(payload["digest"], digest_of_audit(drained))

    def test_replays_conflicts_and_malformed_requests_do_not_move_digest(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        _, before = self.digest()
        op = operation("o1", "k", "v1", {"r1": 1})
        self.assertEqual(self.post_operation("r1", op)[0], 200)
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "x", {"r1": 1}))[0], 409
        )
        self.assertEqual(
            self.request("POST", "/v1/replicas/r1/operations", b"{not json")[0], 400
        )
        self.assertEqual(
            self.post_sync(
                {
                    "operations": [
                        record("r2", operation("o2", "k", "v2", {"r2": 1})),
                        record("r1", operation("o1", "k", "bad", {"r1": 1})),
                    ]
                }
            )[0],
            409,
        )
        _, after = self.digest()
        self.assertEqual(after, before)

    def test_import_batch_is_one_indivisible_segment(self) -> None:
        self.post_operation("r0", operation("before", "k", "v0", {"r0": 1}))
        status, _ = self.post_sync(
            {
                "operations": [
                    record("r1", operation("i1", "k", "v1", {"r1": 1})),
                    record("r2", operation("ix", "x", "vx", {"r2": 1})),
                    record("r1", operation("i2", "k", "v2", {"r1": 2})),
                ]
            }
        )
        self.assertEqual(status, 201)
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 3)
        drained = self.drain_audit("k", limit=1)
        self.assertEqual(payload["digest"], digest_of_audit(drained))

    def test_url_encoded_key_with_slash_and_space(self) -> None:
        key = "names/color wheel"
        self.post_operation("r1", operation("o1", key, "blue", {"r1": 1}))
        status, payload = self.get_digest("/v1/audit/keys/names%2Fcolor%20wheel/digest")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 1)
        self.assertEqual(payload["digest"], digest_of_audit([record("r1", operation("o1", key, "blue", {"r1": 1}))]))

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/audit/keys/k/digest?x=1",
            "/v1/audit/keys/k/digest?after=0",
            "/v1/audit/keys/k/digest?operations=0",
            "/v1/audit/keys/k/digest?x=",
            "/v1/audit/keys/k/digest?x",
            "/v1/audit/keys/k/digest?=1",
            "/v1/audit/keys/k/digest?x=1&x=2",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)
        # The same rejection applies to a key with no history.
        status, payload = self.get_digest("/v1/audit/keys/absent/digest?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.request("GET", "/v1/audit/keys/k/digest?")
        self.assertEqual(status, 200)

    def test_missing_or_extra_segments_are_404(self) -> None:
        for path in (
            "/v1/audit/keys/k/digest/extra",
            "/v1/audit/keys/k",
            "/v1/audit/keys",
            "/v1/audit",
            "/v1/audit/k/digest",
            "/v1/audit/key/k/digest",
            "/v1/audit/keys//digest",
        ):
            status, payload = self.get_digest(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_extra_path_is_404_even_with_query(self) -> None:
        status, payload = self.get_digest("/v1/audit/keys/k/digest/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_segment_with_query_is_404_not_400(self) -> None:
        status, payload = self.get_digest("/v1/audit/keys//digest?x=1")
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
        self.assertEqual(len(self.drain_audit("k")), 1)
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["acceptedOperations"], 1)

    def test_concurrent_commits_observe_consistent_digests(self) -> None:
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
            for index in range(20):
                replica = f"l{index}"
                status, _ = self.post_operation(
                    replica, operation(f"local-{index}", "k", f"v{index}", {replica: 1})
                )
                assert status == 201
                status, payload = self.post_sync(
                    {
                        "operations": [
                            record(f"s{index}a", operation(f"s{index}a", "k", "a", {f"s{index}a": 1})),
                            record(f"s{index}b", operation(f"s{index}b", "x", "b", {f"s{index}b": 1})),
                            record(f"s{index}c", operation(f"s{index}c", "k", "c", {f"s{index}c": 1})),
                        ]
                    }
                )
                assert status == 201 and payload["accepted"] == 3
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])

        # The final digest equals the hash of the final audit stream, and its
        # operations count agrees with the stream length.
        drained = self.drain_audit("k", limit=4)
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], 3 * 20)
        self.assertEqual(len(drained), 3 * 20)
        self.assertEqual(payload["digest"], digest_of_audit(drained))
        # Each batch's two key-k records are adjacent in commit order.
        identities = [(e["replicaId"], e["operation"]["operationId"]) for e in drained]
        positions = {identity: i for i, identity in enumerate(identities)}
        for index in range(20):
            self.assertEqual(
                positions[(f"s{index}a", f"s{index}a")] + 1,
                positions[(f"s{index}c", f"s{index}c")],
            )
        # The other key's stream is independent and complete.
        x_status, x_payload = self.digest("x")
        self.assertEqual(x_status, 200)
        self.assertEqual(x_payload["operations"], 20)


class PersistentDigestHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"

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
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def drain(self, server: SemanticStateServer, key: str, limit: int = 2) -> list[dict]:
        after = 0
        seen: list[dict] = []
        while True:
            status, payload = self.request(
                server, "GET", f"/v1/audit/keys/{key}/operations?after={after}&limit={limit}"
            )
            assert status == 200
            seen.extend(payload["operations"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                return seen

    def seed_mixed_history(self, server: SemanticStateServer) -> None:
        for replica, op in (
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
        ):
            status, _ = self.request(
                server, "POST", f"/v1/replicas/{replica}/operations", op
            )
            assert status == 201
        status, _ = self.request(
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
        assert status == 201
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("stale", "k", "old", {"r1": 0}),
        )
        assert status == 201
        status, _ = self.request(
            server, "POST", "/v1/sync/operations",
            {
                "operations": [
                    record("r4", operation("b1", "k", "b1", {"r4": 1})),
                    record("r9", operation("bx", "x", "bx", {"r9": 1})),
                    record("r4", operation("b2", "k", "b2", {"r4": 2})),
                ]
            },
        )
        assert status == 201

    def test_digest_survives_restart(self) -> None:
        server = self.start_server()
        self.seed_mixed_history(server)
        status, before = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(before["operations"], 6)
        status, before_x = self.request(server, "GET", "/v1/audit/keys/x/digest")
        self.assertEqual(status, 200)
        status, before_empty = self.request(server, "GET", "/v1/audit/keys/absent/digest")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        status, after_x = self.request(server, "GET", "/v1/audit/keys/x/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after_x, before_x)
        status, after_empty = self.request(server, "GET", "/v1/audit/keys/absent/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after_empty, before_empty)
        # And it still matches the recovered audit stream byte for byte.
        drained = self.drain(server, "k", limit=1)
        self.assertEqual(after["digest"], digest_of_audit(drained))

    def test_digest_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = self.data_file.read_bytes()
        before_stat = self.data_file.stat()

        for path in (
            "/v1/audit/keys/k/digest",
            "/v1/audit/keys/absent/digest",
            "/v1/audit/keys/k/digest?",
        ):
            for _ in range(3):
                status, _ = self.request(server, "GET", path)
                self.assertEqual(status, 200)

        self.assertEqual(self.data_file.read_bytes(), before_bytes)
        after_stat = self.data_file.stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)

    def test_persistence_failure_and_rejected_batch_do_not_move_digest(self) -> None:
        server = self.start_server()
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r0/operations",
            operation("o0", "k", "v0", {"r0": 1}),
        )
        self.assertEqual(status, 201)
        status, before = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        before_bytes = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server,
                "POST",
                "/v1/replicas/r1/operations",
                operation("o1", "k", "v1", {"r1": 1}),
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            status, payload = self.request(
                server,
                "POST",
                "/v1/sync/operations",
                {
                    "operations": [
                        record("r2", operation("o2", "k", "v2", {"r2": 1})),
                        record("r0", operation("o0", "k", "tampered", {"r0": 1})),
                    ]
                },
            )
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "operation_conflict"})
            # Reads still succeed during the fault and report the old commit.
            status, during = self.request(server, "GET", "/v1/audit/keys/k/digest")
            self.assertEqual(status, 200)
            self.assertEqual(during, before)

        self.assertEqual(self.data_file.read_bytes(), before_bytes)
        # The failed operation commits cleanly once persistence works again.
        status, _ = self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.assertEqual(status, 201)
        status, after = self.request(server, "GET", "/v1/audit/keys/k/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after["operations"], before["operations"] + 1)
        self.assertNotEqual(after["digest"], before["digest"])
        drained = self.drain(server, "k")
        self.assertEqual(after["digest"], digest_of_audit(drained))


if __name__ == "__main__":
    unittest.main()
