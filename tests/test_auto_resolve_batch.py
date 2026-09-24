"""HTTP, sync, and persistence tests for batched automatic resolution.

The endpoint is::

    POST /v1/resolve/auto/batch

It accepts 1-100 automatic-resolution entries (each naming its own key) in
one JSON object ``{"resolutions":[...]}``, processes them sequentially in
request order under one commit, and reports one result per entry. Everything
here goes through the real HTTP entry point (``SemanticStateServer`` + a
request thread); only the Python standard library is used.
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

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    load_data_file_full,
    parse_auto_resolve_batch,
)

BATCH_PATH = "/v1/resolve/auto/batch"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def batch_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "fix-1",
    clock: dict | None = None,
    policy: str = "lowest_identity",
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock if clock is not None else {"r1": 1, "r2": 1, "r3": 1},
        "policy": policy,
    }


def batch_document(*entries: dict) -> dict:
    return {"resolutions": list(entries)}


class ParseAutoResolveBatchTests(unittest.TestCase):
    def test_valid_batch_is_normalized_in_order(self) -> None:
        entries = parse_auto_resolve_batch(
            json.dumps(
                batch_document(
                    batch_entry("k1", "r3", "f1"),
                    batch_entry(
                        "k2", "r9", "f9", {"r1": 1, "r9": 1}, "highest_identity"
                    ),
                )
            )
        )
        self.assertEqual(
            entries,
            [
                {
                    "key": "k1",
                    "replicaId": "r3",
                    "operationId": "f1",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "policy": "lowest_identity",
                },
                {
                    "key": "k2",
                    "replicaId": "r9",
                    "operationId": "f9",
                    "clock": {"r1": 1, "r9": 1},
                    "policy": "highest_identity",
                },
            ],
        )

    def test_one_hundred_entries_are_accepted(self) -> None:
        entries = [
            batch_entry(f"k{i:03d}", f"r{i:03d}", f"f{i:03d}", {f"r{i:03d}": 1})
            for i in range(100)
        ]
        parsed = parse_auto_resolve_batch(json.dumps(batch_document(*entries)))
        self.assertEqual(len(parsed), 100)
        self.assertEqual([e["key"] for e in parsed], [f"k{i:03d}" for i in range(100)])

    def test_rejects_malformed_and_wrong_root_shapes(self) -> None:
        valid = batch_document(batch_entry("k"))
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"operations": []},
            {"resolutions": []},  # present but wrong type
            dict(valid, extra=1),
            {"resolutions": valid["resolutions"], "x": 2},
        ]
        for body in bad_bodies:
            with self.assertRaises(ValueError, msg=repr(body)):
                parse_auto_resolve_batch(body)

    def test_rejects_empty_and_oversized_batches(self) -> None:
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(batch_document())
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(
                batch_document(
                    *[batch_entry(f"k{i}", "r3", f"f{i}", {"r3": 1}) for i in range(101)]
                )
            )

    def test_rejects_bad_entry_shapes(self) -> None:
        good = batch_entry("k")
        bad_entries = [
            [],
            "x",
            {},
            {k: v for k, v in good.items() if k != "key"},
            {k: v for k, v in good.items() if k != "replicaId"},
            {k: v for k, v in good.items() if k != "operationId"},
            {k: v for k, v in good.items() if k != "clock"},
            {k: v for k, v in good.items() if k != "policy"},
            dict(good, extra=1),
            dict(good, key=""),
            dict(good, key=7),
            dict(good, replicaId=""),
            dict(good, operationId=""),
            dict(good, clock={}),
            dict(good, clock={"r2": 1}),  # must contain the replica
            dict(good, clock={"r3": -1}),
            dict(good, clock={"r3": True}),
            dict(good, policy=""),
            dict(good, policy="lowest"),
            dict(good, policy=42),
        ]
        for entry in bad_entries:
            with self.assertRaises(ValueError, msg=repr(entry)):
                parse_auto_resolve_batch(batch_document(entry))

    def test_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(
                batch_document(
                    batch_entry("k", "r3", "f1"),
                    batch_entry("k", "r4", "f2", {"r1": 1, "r2": 1, "r4": 1}),
                )
            )

    def test_rejects_duplicate_identities_even_on_different_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(
                batch_document(
                    batch_entry("k1", "r3", "same-fix"),
                    batch_entry("k2", "r3", "same-fix", {"r3": 2}),
                )
            )

    def test_rejects_floats_and_non_finite_clock_values(self) -> None:
        # Python floats (including 1.0 and -0.0) are not integers.
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(
                        batch_document(batch_entry("k", clock={"r3": tick}))
                    )
        # NaN and Infinity parse as floats under the stdlib decoder and are
        # rejected just like any other non-integer value.
        for literal in ("NaN", "Infinity", "-Infinity", "1e3"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
                    b'"clock":{"r3":' + literal.encode("ascii") + b'},"policy":"lowest_identity"}]}'
                )
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(raw)


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

    def request_bytes(
        self, method: str, path: str, body: bytes | None = None, headers: dict | None = None
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(method, path, body=body, headers=merged)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        if body is None:
            status, raw = self.request_bytes(method, path)
        elif isinstance(body, (bytes, str)):
            data = body.encode("utf-8") if isinstance(body, str) else body
            status, raw = self.request_bytes(method, path, data)
        else:
            status, raw = self.request_bytes(method, path, json.dumps(body).encode("utf-8"))
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_batch(self, body: object) -> tuple[int, object]:
        return self.request("POST", BATCH_PATH, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_auto(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve/auto", body)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_audit(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_conflict(self, key: str = "k", low: str = "v1", high: str = "v2") -> None:
        """Two concurrent writes with distinct global identities on ``key``."""
        # Identities are global across keys, so derive the operation ids from
        # the key (and salt colliding keys with a per-test counter).
        salt = getattr(self, "_seed_salt", 0)
        self._seed_salt = salt + 1
        id1, id2 = f"o1-{key}-{salt}", f"o2-{key}-{salt}"
        self.assertEqual(
            self.post_operation("r1", operation(id1, key, low, {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation(id2, key, high, {"r2": 1}))[0], 201
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class BatchHappyPathTests(HttpServerTestCase):
    def test_batch_resolves_distinct_keys_in_order_with_counts(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2", low="aaa", high="zzz")
        doc = batch_document(
            batch_entry("k1", "r3", "f1"),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "resolutions": [
                    {
                        "key": "k1",
                        "replicaId": "r3",
                        "operationId": "f1",
                        "value": "v1",
                        "policy": "lowest_identity",
                    },
                    {
                        "key": "k2",
                        "replicaId": "r3",
                        "operationId": "f2",
                        "value": "zzz",
                        "policy": "highest_identity",
                    },
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        self.assertIsInstance(payload["accepted"], int)
        self.assertIsInstance(payload["replayed"], int)
        _, state1 = self.get_state("k1")
        self.assertEqual(state1["status"], "resolved")
        self.assertEqual(state1["value"], "v1")
        _, state2 = self.get_state("k2")
        self.assertEqual(state2["status"], "resolved")
        self.assertEqual(state2["value"], "zzz")

    def test_response_is_compact_json_ending_with_a_newline(self) -> None:
        self.seed_conflict("k")
        body = json.dumps(batch_document(batch_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", BATCH_PATH, body)
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        # Compact encoding: no whitespace around separators.
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["resolutions"][0]["value"], "v1")
        # Error responses for the route carry the same terminator.
        status, raw = self.request_bytes("POST", BATCH_PATH, b"{oops")
        self.assertEqual(status, 400)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"error": "invalid_request"})

    def test_single_entry_batch_is_supported(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_batch(batch_document(batch_entry("k")))
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)

    def test_one_hundred_entries_commit_together(self) -> None:
        doc_entries = []
        for i in range(100):
            key = f"k{i:03d}"
            pa, pb = f"p{i:03d}a", f"p{i:03d}b"
            self.assertEqual(
                self.post_operation(pa, operation("o1", key, "lo", {pa: 1}))[0], 201
            )
            self.assertEqual(
                self.post_operation(pb, operation("o2", key, "hi", {pb: 1}))[0], 201
            )
            doc_entries.append(
                batch_entry(
                    key,
                    f"r{i:03d}",
                    f"f{i:03d}",
                    {pa: 1, pb: 1, f"r{i:03d}": 1},
                    "highest_identity" if i % 2 else "lowest_identity",
                )
            )
        status, payload = self.post_batch(batch_document(*doc_entries))
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 100)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(len(payload["resolutions"]), 100)
        self.assertEqual([r["key"] for r in payload["resolutions"]], [f"k{i:03d}" for i in range(100)])
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 300)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["resolvedKeys"], 100)

    def test_mixed_new_and_replayed_entries_is_201(self) -> None:
        # k2 is resolved first via the single-key endpoint; that resolution
        # is later replayed inside a batch alongside one new repair.
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        self.assertEqual(
            self.post_auto("k2", {"replicaId": "r3", "operationId": "f2",
                                  "clock": {"r1": 1, "r2": 1, "r3": 2},
                                  "policy": "lowest_identity"})[0],
            201,
        )
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["v1", "v1"]
        )

    def test_all_replay_batch_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        self.assertEqual(self.post_batch(doc)[0], 201)
        _, before = self.get_sync()
        self.assertEqual(len(before["operations"]), 6)
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        _, after = self.get_sync()
        self.assertEqual(after, before)

    def test_results_follow_request_order_not_identity_order(self) -> None:
        self.seed_conflict("k1", low="a", high="b")
        self.seed_conflict("k2", low="c", high="d")
        doc = batch_document(
            batch_entry("k2", "r9", "zzz-fix", {"r1": 1, "r2": 1, "r9": 1}),
            batch_entry("k1", "r3", "aaa-fix", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 201)
        self.assertEqual([r["key"] for r in payload["resolutions"]], ["k2", "k1"])

    def test_records_enter_sync_audit_metrics_and_digests(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        _, digest_before = self.request("GET", "/v1/verification/digest")
        _, audit_before = self.request("GET", "/v1/audit/keys/k1/digest")
        _, metrics_before = self.get_metrics()
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        self.assertEqual(self.post_batch(doc)[0], 201)

        _, page = self.get_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"], e["operation"]["key"])
             for e in page["operations"]],
            [
                ("r1", "o1-k1-0", "k1"),
                ("r2", "o2-k1-0", "k1"),
                ("r1", "o1-k2-1", "k2"),
                ("r2", "o2-k2-1", "k2"),
                ("r3", "f1", "k1"),
                ("r3", "f2", "k2"),
            ],
        )
        _, audit = self.get_audit("k2")
        self.assertEqual(
            [e["operation"]["operationId"] for e in audit["operations"]],
            ["o1-k2-1", "o2-k2-1", "f2"],
        )
        self.assertEqual(audit["operations"][2]["operation"]["value"], "v1")
        # The per-operation archive addresses the batch-committed repair.
        status, archived = self.request("GET", "/v1/replicas/r3/operations/f2")
        self.assertEqual(status, 200)
        self.assertEqual(
            archived,
            {
                "replicaId": "r3",
                "operation": {
                    "operationId": "f2",
                    "key": "k2",
                    "value": "v1",
                    "clock": {"r1": 1, "r2": 1, "r3": 2},
                },
            },
        )
        _, metrics_after = self.get_metrics()
        self.assertEqual(
            metrics_after["acceptedOperations"], metrics_before["acceptedOperations"] + 2
        )
        self.assertEqual(metrics_after["conflictKeys"], 0)
        self.assertEqual(metrics_after["resolvedKeys"], 2)
        _, digest_after = self.request("GET", "/v1/verification/digest")
        _, audit_after = self.request("GET", "/v1/audit/keys/k1/digest")
        self.assertNotEqual(digest_before["digest"], digest_after["digest"])
        self.assertEqual(audit_after["operations"], audit_before["operations"] + 1)
        self.assertNotEqual(audit_before["digest"], audit_after["digest"])


class BatchValidationTests(HttpServerTestCase):
    def assert_unchanged_conflict(self, key: str = "k", candidates: int = 2) -> None:
        _, state = self.get_state(key)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), candidates)

    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict("k")
        valid = batch_document(batch_entry("k"))
        bad_bodies = [
            b"{oops",
            [],
            {},
            {"operations": [batch_entry("k")]},
            dict(valid, extra=1),
            batch_document(),
            batch_document(
                *[batch_entry(f"k{i}", "r3", f"f{i}", {"r3": 1}) for i in range(101)]
            ),
            batch_document(dict(batch_entry("k"), extra=1)),
            batch_document({k: v for k, v in batch_entry("k").items() if k != "key"}),
            batch_document(dict(batch_entry("k"), key="")),
            batch_document(dict(batch_entry("k"), replicaId="")),
            batch_document(dict(batch_entry("k"), operationId="")),
            batch_document(dict(batch_entry("k"), clock={"r2": 2})),
            batch_document(dict(batch_entry("k"), clock={"r3": -1})),
            batch_document(dict(batch_entry("k"), policy="middle")),
        ]
        for body in bad_bodies:
            status, payload = self.post_batch(body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        self.assert_unchanged_conflict()

    def test_float_clocks_are_400(self) -> None:
        self.seed_conflict("k")
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                doc = batch_document(batch_entry("k", clock={"r1": 1, "r2": 1, "r3": tick}))
                status, payload = self.post_batch(doc)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for literal in ("NaN", "Infinity", "-Infinity", "-0.0"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f1",'
                    b'"clock":{"r1":1,"r2":1,"r3":' + literal.encode("ascii")
                    + b'},"policy":"lowest_identity"}]}'
                )
                status, payload = self.post_batch(raw)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        self.assert_unchanged_conflict()

    def test_duplicate_keys_are_400(self) -> None:
        self.seed_conflict("k")
        doc = batch_document(
            batch_entry("k", "r3", "f1"),
            batch_entry("k", "r4", "f2", {"r1": 1, "r2": 1, "r4": 1}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assert_unchanged_conflict()

    def test_duplicate_identities_are_400_even_across_keys(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "same", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "same", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assert_unchanged_conflict("k1")
        self.assert_unchanged_conflict("k2")

    def test_clock_not_dominating_is_409_and_batch_is_unchanged(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            # Concurrent with k2's r2 candidate: a legal clock that does not
            # dominate the live candidates.
            batch_entry("k2", "r3", "f2", {"r1": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        # The valid first entry is not partially committed.
        self.assert_unchanged_conflict("k1")
        self.assert_unchanged_conflict("k2")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 4)

    def test_clock_not_dominating_after_earlier_entry_changed_candidates(self) -> None:
        # Processing is sequential: a legal entry clock is checked against
        # the live candidates of its own key at its position in the batch.
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        # A third concurrent candidate strengthens k2's candidate set.
        self.post_operation("r4", operation("o4-k2", "k2", "v4", {"r4": 1}))
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            # Does not dominate the new r4 candidate on k2.
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_unchanged_conflict("k1")
        _, state = self.get_state("k2")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 3)

    def test_extra_path_is_404(self) -> None:
        for path in (
            "/v1/resolve/auto/batch/extra",
            "/v1/resolve/auto",
            "/v1/resolve",
            "/v1/resolve/auto/batch/extra/two",
        ):
            status, payload = self.request("POST", path, batch_document(batch_entry("k")))
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)


class BatchConflictTests(HttpServerTestCase):
    def test_missing_key_entry_is_409_and_whole_batch_unchanged(self) -> None:
        self.seed_conflict("k1")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("absent", "r3", "f2", {"r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_unconflicted_key_entry_is_409(self) -> None:
        self.seed_conflict("k1")
        self.post_operation("r1", operation("o9", "k2", "solo", {"r1": 9}))
        doc = batch_document(
            batch_entry("k2", "r3", "f2", {"r1": 10, "r3": 2}),
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_same_value_candidates_are_409(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        status, payload = self.post_batch(
            batch_document(batch_entry("k", "r3", "f", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_operation_conflict_on_rebound_identity(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        # The identity is first committed under one binding (key k1,
        # lowest_identity); the batch reuses it with a different clock.
        self.assertEqual(
            self.post_auto(
                "k1",
                {"replicaId": "r3", "operationId": "f1",
                 "clock": {"r1": 1, "r2": 1, "r3": 1},
                 "policy": "lowest_identity"},
            )[0],
            201,
        )
        doc = batch_document(
            batch_entry("k2", "r3", "f9", {"r1": 1, "r2": 1, "r3": 9}),
            batch_entry("k2", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

        # With distinct keys the batch parses, and the rebound identity then
        # fails the whole batch as an operation conflict at commit time.
        doc = batch_document(
            batch_entry("k2", "r3", "f9", {"r1": 1, "r2": 1, "r3": 9}),
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Nothing in the batch committed.
        _, state = self.get_state("k2")
        self.assertEqual(state["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 5)

    def test_identity_bound_to_plain_write_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r3", operation("f1", "other", "v", {"r3": 1}))
        status, payload = self.post_batch(
            batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_different_policy_under_known_identity_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        good = {"replicaId": "r3", "operationId": "f1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "lowest_identity"}
        self.assertEqual(self.post_auto("k", good)[0], 201)
        status, payload = self.post_batch(
            batch_document(batch_entry("k", "r3", "f1", good["clock"], "highest_identity"))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_conflict_kind_follows_first_failing_entry(self) -> None:
        # Processing is sequential: the first entry that fails fixes the
        # error code, and the batch fails as a whole either way.
        self.seed_conflict("k1")
        self.post_operation("r3", operation("bound", "x", "v", {"r3": 1}))

        # First entry: resolution conflict; second would be an operation
        # conflict — resolution_conflict wins by order.
        doc = batch_document(
            batch_entry("missing-a", "r3", "fa", {"r3": 2}),
            batch_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

        # Reversed order: operation_conflict surfaces first.
        doc = batch_document(
            batch_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
            batch_entry("missing-a", "r3", "fa", {"r3": 2}),
        )
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_replay_after_key_moved_on_reports_original_value(self) -> None:
        self.seed_conflict("k")
        doc = batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        self.assertEqual(self.post_batch(doc)[0], 201)
        # A later concurrent write reopens the conflict on k.
        self.post_operation(
            "r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0})
        )
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        # The replay is answered from the committed operation and appends
        # nothing.
        status, payload = self.post_batch(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["resolutions"][0]["value"], "v1")
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1-k-0", "o2-k-0", "f1", "o3"],
        )

    def test_replay_requires_same_binding_key(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        self.assertEqual(
            self.post_batch(
                batch_document(batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
            )[0],
            201,
        )
        # Same identity, different target key: not a replay.
        status, payload = self.post_batch(
            batch_document(batch_entry("k2", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class BatchConcurrencyTests(HttpServerTestCase):
    def test_concurrent_identical_batches_commit_exactly_once(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        results: list[tuple[int, object] | None] = [None] * 8

        def worker(index: int) -> None:
            results[index] = self.post_batch(json.loads(json.dumps(doc)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        statuses = sorted(status for status, _ in results if status is not None)
        self.assertEqual(statuses, [200] * 7 + [201])
        created = [p for s, p in results if s == 201][0]
        self.assertEqual(created["accepted"], 2)
        self.assertEqual(created["replayed"], 0)
        for _, payload in results:
            if payload is not None and "resolutions" in payload:
                self.assertEqual(
                    [r["value"] for r in payload["resolutions"]], ["v1", "v2"]
                )
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 6)
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["resolvedKeys"], 2)
        self.assertEqual(metrics["conflictKeys"], 0)


class BatchImportInteractionTests(HttpServerTestCase):
    def test_imported_resolution_carries_no_policy_binding(self) -> None:
        self.seed_conflict("k")
        self.assertEqual(
            self.post_batch(
                batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
            )[0],
            201,
        )
        _, page = self.get_sync()
        self.server.store = type(self.server.store)()
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": page["operations"]})
        self.assertEqual(status, 201)
        # The importing replica holds the operation but no binding, so
        # replaying the batch entry conflicts rather than replaying.
        status, payload = self.post_batch(
            batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_batch_repairs_flow_through_sync_and_resolve_elsewhere(self) -> None:
        self.seed_conflict("k")
        self.assertEqual(
            self.post_batch(
                batch_document(batch_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
            )[0],
            201,
        )
        _, page = self.get_sync()
        other = StateStore()
        status, accepted, _ = other.import_operations(
            [(e["replicaId"], e["operation"]) for e in page["operations"]]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 3)
        status, state = other.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "v1")


class BatchRequestLimitTests(unittest.TestCase):
    """The batch route keeps the shared Content-Length/auth priority."""

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

    def post_raw(self, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", BATCH_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", BATCH_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_length_is_400(self) -> None:
        for value in ("", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            status, payload = self.post_raw([("Content-Length", value)], b"{}")
            self.assertEqual(status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_over_limit_declaration_is_413(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_declared_length_exactly_at_the_limit_is_processed(self) -> None:
        # A batch document padded to exactly MAX_BODY_BYTES is processed by
        # the normal endpoint semantics rather than rejected as too large.
        template = {
            "key": "",
            "replicaId": "r3",
            "operationId": "f1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        base = json.dumps(
            {"resolutions": [template]}, separators=(",", ":")
        ).encode("utf-8")
        key_length = MAX_BODY_BYTES - len(base)
        self.assertGreater(key_length, 0)
        key = "k" + "x" * (key_length - 1)
        doc = json.dumps(
            batch_document(batch_entry(key, "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})),
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(len(doc), MAX_BODY_BYTES)
        for replica, op_id, clock in (
            ("r1", "o1", {"r1": 1}),
            ("r2", "o2", {"r2": 1}),
        ):
            body = json.dumps(operation(op_id, key, f"v{op_id}", clock), separators=(",", ":")).encode()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            conn.request(
                "POST",
                f"/v1/replicas/{replica}/operations",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201, response.read())
            conn.close()
        status, payload = self.post_raw([("Content-Length", str(len(doc)))], doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["resolutions"][0]["key"], key)


class BatchAuthTests(unittest.TestCase):
    def test_unauthorized_batch_is_401_without_reading_the_body(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            BATCH_PATH,
            body=json.dumps(batch_document(batch_entry("k"))),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_authorized_batch_succeeds(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        def authed(method: str, path: str, payload: dict) -> tuple[int, dict]:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                method,
                path,
                body=json.dumps(payload),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer secret-token",
                },
            )
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            conn.close()
            return response.status, data

        for replica, op_id, clock in (
            ("r1", "o1", {"r1": 1}),
            ("r2", "o2", {"r2": 1}),
        ):
            status, _ = authed(
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", f"v{op_id[-1]}", clock),
            )
            self.assertEqual(status, 201)
        status, payload = authed(
            "POST", BATCH_PATH, batch_document(batch_entry("k"))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["resolutions"][0]["value"], "v1")


class PersistentBatchTestCase(unittest.TestCase):
    """Batched automatic resolution against a data-file-backed server."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            conn.request(
                method,
                path,
                body=data,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_conflicts(self, server: SemanticStateServer, *keys: str) -> None:
        for index, key in enumerate(keys):
            for replica, op_id in (("r1", f"o1-{key}"), ("r2", f"o2-{key}")):
                status, _ = self.request(
                    server,
                    "POST",
                    f"/v1/replicas/{replica}/operations",
                    operation(op_id, key, f"v-{key}-{replica}", {replica: 1}),
                )
                self.assertEqual(status, 201, index)

    def test_batch_is_durable_in_one_commit_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, payload = self.request(server, "POST", BATCH_PATH, doc)
        self.assertEqual(status, 201)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["v-k1-r1", "v-k2-r2"]
        )
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["key"], o["operationId"]) for r, o in records],
            [
                ("r1", "k1", "o1-k1"),
                ("r2", "k1", "o2-k1"),
                ("r1", "k2", "o1-k2"),
                ("r2", "k2", "o2-k2"),
                ("r3", "k1", "f1"),
                ("r3", "k2", "f2"),
            ],
        )
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(
            policies,
            {
                ("r3", "f1"): "lowest_identity",
                ("r3", "f2"): "highest_identity",
            },
        )

        server.shutdown()
        server.server_close()

        server = self.start_server()
        for key, value in (("k1", "v-k1-r1"), ("k2", "v-k2-r2")):
            status, state = self.request(server, "GET", f"/v1/states/{key}")
            self.assertEqual(status, 200)
            self.assertEqual(state["status"], "resolved")
            self.assertEqual(state["value"], value)
        # The full batch replays as 200 after restart and appends nothing.
        status, payload = self.request(server, "POST", BATCH_PATH, doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual(len(load_data_file(str(self.data_file))), 6)

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        before = self.data_file.read_bytes()
        doc = batch_document(
            batch_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            batch_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(server, "POST", BATCH_PATH, doc)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})

        # File and memory are exactly as they were before the batch.
        self.assertEqual(self.data_file.read_bytes(), before)
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {})
        for key in ("k1", "k2"):
            _, state = self.request(server, "GET", f"/v1/states/{key}")
            self.assertEqual(state["status"], "conflict")
        status, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual(len(page["operations"]), 4)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])
        # The same batch commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", BATCH_PATH, doc)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 2)
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(
            policies,
            {("r3", "f1"): "lowest_identity", ("r3", "f2"): "lowest_identity"},
        )


if __name__ == "__main__":
    unittest.main()
