"""Tests for the read-only replica-convergence verification digest::

    GET /v1/verification/digest

It returns exactly ``algorithm`` (always ``"sha256"``), ``digest`` (64
lowercase hex characters), and the non-negative ``keys`` /
``candidateVersions`` counts, all from a single snapshot under the shared
commit lock. The digest input is a compact UTF-8 JSON array of
``{"key":K,"candidates":C}`` entries ordered by key, with ``C`` ordered by
``(replicaId, operationId)`` and each candidate carrying exactly ``value``,
``clock`` (components ordered by name), ``replicaId``, ``operationId`` in
that field order; strings escape only the quotation mark, the reverse
solidus, and control characters, and every other Unicode code point is
written as-is. Only current candidates are covered — never the accepted
log, stale writes, or checkpoints.

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

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    parse_metrics_query,
)

DIGEST_FIELDS = {"algorithm", "digest", "keys", "candidateVersions"}

DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

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


class DigestQueryTests(unittest.TestCase):
    def test_empty_query_is_accepted(self) -> None:
        self.assertTrue(parse_metrics_query(""))

    def test_any_parameter_is_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x=1"))
        self.assertFalse(parse_metrics_query("after=0"))

    def test_blank_values_are_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x="))
        self.assertFalse(parse_metrics_query("x"))
        self.assertFalse(parse_metrics_query("=1"))

    def test_repeated_parameters_are_rejected(self) -> None:
        self.assertFalse(parse_metrics_query("x=1&x=2"))


class DigestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def digest(self) -> dict:
        result = self.store.get_verification_digest()
        self.assertEqual(set(result), DIGEST_FIELDS)
        self.assertEqual(result["algorithm"], "sha256")
        self.assertRegex(result["digest"], DIGEST_PATTERN)
        self.assertIs(type(result["keys"]), int)
        self.assertIs(type(result["candidateVersions"]), int)
        self.assertGreaterEqual(result["keys"], 0)
        self.assertGreaterEqual(result["candidateVersions"], 0)
        return result

    def test_empty_store_digests_the_empty_array(self) -> None:
        result = self.digest()
        self.assertEqual(result["digest"], EMPTY_DIGEST)
        self.assertEqual(result["keys"], 0)
        self.assertEqual(result["candidateVersions"], 0)

    def test_digest_matches_the_canonical_form(self) -> None:
        self.store.apply_operation("r1", operation("op-1", "color", "blue", {"r1": 1}))
        self.store.apply_operation("r2", operation("op-2", "color", "red", {"r2": 1}))
        result = self.digest()
        self.assertEqual(result["keys"], 1)
        self.assertEqual(result["candidateVersions"], 2)
        canonical = (
            '[{"key":"color","candidates":['
            '{"value":"blue","clock":{"r1":1},"replicaId":"r1","operationId":"op-1"},'
            '{"value":"red","clock":{"r2":1},"replicaId":"r2","operationId":"op-2"}'
            "]}]"
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_candidates_are_ordered_by_replica_then_operation(self) -> None:
        # Concurrent writes committed in reverse identity order.
        self.store.apply_operation("r2", operation("op-2", "k", "v2", {"r2": 1}))
        self.store.apply_operation("r1", operation("op-9", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r1", operation("op-3", "k", "v0", {"r1": 0}))
        # op-3 is stale (dominated by op-9's clock) and adds no candidate.
        result = self.digest()
        self.assertEqual(result["candidateVersions"], 2)
        canonical = (
            '[{"key":"k","candidates":['
            '{"value":"v1","clock":{"r1":1},"replicaId":"r1","operationId":"op-9"},'
            '{"value":"v2","clock":{"r2":1},"replicaId":"r2","operationId":"op-2"}'
            "]}]"
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_keys_are_ordered_lexicographically(self) -> None:
        self.store.apply_operation("r1", operation("o1", "b", "1", {"r1": 1}))
        self.store.apply_operation("r1", operation("o2", "a", "2", {"r1": 2}))
        result = self.digest()
        self.assertEqual(result["keys"], 2)
        canonical = (
            '[{"key":"a","candidates":['
            '{"value":"2","clock":{"r1":2},"replicaId":"r1","operationId":"o2"}]},'
            '{"key":"b","candidates":['
            '{"value":"1","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}]'
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_clock_components_are_ordered_by_name(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "k", "v", {"r2": 1, "r1": 2, "r10": 3})
        )
        result = self.digest()
        canonical = (
            '[{"key":"k","candidates":['
            '{"value":"v","clock":{"r1":2,"r10":3,"r2":1},'
            '"replicaId":"r1","operationId":"o1"}]}]'
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_unicode_is_written_as_is_and_only_json_escapes_apply(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "café", 'naïve "quoted"\n→\\', {"r1": 1})
        )
        result = self.digest()
        canonical = (
            '[{"key":"café","candidates":['
            '{"value":"naïve \\"quoted\\"\\n→\\\\",'
            '"clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}]'
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_stale_writes_and_the_log_do_not_change_the_digest(self) -> None:
        self.store.apply_operation("r1", operation("new", "k", "v", {"r1": 2}))
        before = self.digest()
        # A stale write is first-accepted (201) and grows the log...
        self.assertIs(
            self.store.apply_operation("r1", operation("stale", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        page, _, _ = self.store.get_sync_operations(0, 100)
        self.assertEqual(len(page), 2)
        # ...but adds no candidate, so the digest is unchanged.
        self.assertEqual(self.digest(), before)

    def test_checkpoints_do_not_change_the_digest(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.digest()
        status, error = self.store.save_checkpoint("peer-a", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIsNone(error)
        self.assertEqual(self.digest(), before)

    def test_replays_and_conflicts_do_not_change_the_digest(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.store.apply_operation("r1", op)
        before = self.digest()
        self.assertIs(self.store.apply_operation("r1", dict(op)), HTTPStatus.OK)
        self.assertIs(
            self.store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        self.assertEqual(self.digest(), before)

    def test_resolution_moves_the_digest_to_the_single_candidate(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        conflict_digest = self.digest()
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
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        result = self.digest()
        self.assertNotEqual(result["digest"], conflict_digest["digest"])
        self.assertEqual(result["keys"], 1)
        self.assertEqual(result["candidateVersions"], 1)
        canonical = (
            '[{"key":"k","candidates":['
            '{"value":"merged","clock":{"r1":1,"r2":1,"r3":1},'
            '"replicaId":"r3","operationId":"fix-1"}]}]'
        )
        self.assertEqual(result["digest"], expected_digest(canonical))

    def test_import_batch_commits_as_one_digest_step(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        before = self.digest()
        status, accepted, replayed = self.store.import_operations(
            [
                ("r2", operation("o2", "k", "v2", {"r2": 1})),
                ("r3", operation("o3", "other", "v3", {"r3": 1})),
            ]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual((accepted, replayed), (2, 0))
        result = self.digest()
        self.assertNotEqual(result["digest"], before["digest"])
        self.assertEqual(result["keys"], 2)
        self.assertEqual(result["candidateVersions"], 3)

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        log_before = self.store.get_sync_operations(0, 100)[0]
        first = self.digest()
        second = self.digest()
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], log_before)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)

    def test_failed_durable_commit_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            persistent = StateStore(data_file=str(Path(tmp) / "state.json"))
            empty = persistent.get_verification_digest()
            with patch.object(
                StateStore, "_persist_locked", side_effect=PersistenceError("disk full")
            ):
                with self.assertRaises(PersistenceError):
                    persistent.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
            self.assertEqual(persistent.get_verification_digest(), empty)


class DigestRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_digest_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        # A conflict, a stale write, and a checkpoint.
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        store.save_checkpoint("peer-a", 3)
        # A repair that collapses the conflict.
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
        store.apply_operation("r4", operation("o4", "café", "naïve→", {"r4": 1}))
        before = store.get_verification_digest()

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.get_verification_digest(), before)
        # And a second restart is identical too.
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).get_verification_digest(), before
        )


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

    def digest(self, path: str = "/v1/verification/digest") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_store_returns_the_empty_array_digest(self) -> None:
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "algorithm": "sha256",
                "digest": EMPTY_DIGEST,
                "keys": 0,
                "candidateVersions": 0,
            },
        )

    def test_payload_shape_and_response_contract(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request("GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), DIGEST_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertIs(type(payload["digest"]), str)
        self.assertRegex(payload["digest"], DIGEST_PATTERN)
        for name in ("keys", "candidateVersions"):
            self.assertIs(type(payload[name]), int, name)
            self.assertGreaterEqual(payload[name], 0)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        # The existing explicit-content-length contract is preserved.
        self.assertEqual(header_map["content-length"], str(len(raw)))
        self.assertEqual(len(raw), int(header_map["content-length"]))

    def test_digest_matches_the_canonical_form_over_http(self) -> None:
        self.post_operation("r1", operation("op-1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("op-2", "color", "red", {"r2": 1}))
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 2)
        canonical = (
            '[{"key":"color","candidates":['
            '{"value":"blue","clock":{"r1":1},"replicaId":"r1","operationId":"op-1"},'
            '{"value":"red","clock":{"r2":1},"replicaId":"r2","operationId":"op-2"}'
            "]}]"
        )
        self.assertEqual(payload["digest"], expected_digest(canonical))

    def test_digest_excludes_stale_writes_repairs_collapse_conflicts(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, conflict = self.digest()
        # A stale write grows the accepted log but not the digest.
        status, _ = self.post_operation("r1", operation("o3", "k", "old", {"r1": 0}))
        self.assertEqual(status, 201)
        status, sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync["operations"]), 3)
        self.assertEqual(self.digest()[1], conflict)
        # A checkpoint changes nothing either.
        status, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 3}
        )
        self.assertEqual(status, 200)
        self.assertEqual(self.digest()[1], conflict)
        # A repair collapses the conflict to one candidate.
        status, _ = self.request(
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
        self.assertEqual(status, 201)
        status, resolved = self.digest()
        self.assertEqual(status, 200)
        self.assertNotEqual(resolved["digest"], conflict["digest"])
        self.assertEqual(resolved["candidateVersions"], 1)
        canonical = (
            '[{"key":"k","candidates":['
            '{"value":"merged","clock":{"r1":1,"r2":1,"r3":1},'
            '"replicaId":"r3","operationId":"fix-1"}]}]'
        )
        self.assertEqual(resolved["digest"], expected_digest(canonical))

    def test_any_query_parameter_is_400(self) -> None:
        for path in (
            "/v1/verification/digest?x=1",
            "/v1/verification/digest?after=0",
            "/v1/verification/digest?x=",
            "/v1/verification/digest?x",
            "/v1/verification/digest?=1",
            "/v1/verification/digest?x=1&x=2",
            "/v1/verification/digest?digest=abc",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_empty_query_separator_is_still_200(self) -> None:
        # Bare separators carry no parameter, matching the metrics parser.
        status, _ = self.request("GET", "/v1/verification/digest?")
        self.assertEqual(status, 200)

    def test_extra_path_is_404(self) -> None:
        for path in (
            "/v1/verification/digest/extra",
            "/v1/verification/digest/sha256",
            "/v1/verification",
            "/v1/verification/diges",
        ):
            status, payload = self.digest(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_extra_path_is_404_even_with_query(self) -> None:
        status, payload = self.digest("/v1/verification/digest/extra?x=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_the_digest_route_is_404(self) -> None:
        status, payload = self.request("POST", "/v1/verification/digest", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_digest_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        first_status, first = self.digest()
        self.assertEqual(first_status, 200)
        for _ in range(3):
            status, payload = self.digest()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        # The accepted-operation log and the visible state are untouched.
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 1)
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                result = self.server.store.get_verification_digest()
                if result["algorithm"] != "sha256":
                    violations.append("unexpected algorithm")
                if not DIGEST_PATTERN.match(result["digest"]):
                    violations.append("malformed digest")
                if result["candidateVersions"] < result["keys"]:
                    violations.append("candidateVersions < keys")
                if result["keys"] < 0 or result["candidateVersions"] < 0:
                    violations.append("negative count")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                replica = f"r{index}"
                self.post_operation(
                    replica,
                    operation(f"op-{index}", "shared", f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.digest()
        self.assertEqual(status, 200)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 40)


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
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
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

    def test_digest_survives_restart(self) -> None:
        server = self.start_server()
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
        status, before = self.request(server, "GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_digest_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/verification/digest")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
