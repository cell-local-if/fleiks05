"""Tests for the read-only cross-key causal snapshot endpoint.

The endpoint is::

    POST /v1/states/causal-at

with a body of exactly ``{"clock": {...}, "keys": [...]}`` naming one
vector-clock boundary (the same boundary rules as the single-key
``POST /v1/states/{key}/causal-at``) and between 1 and 100 distinct
non-empty keys in response order. Starting from the empty state the
accepted log is replayed once in global commit order, keeping one
candidate set per requested key under the same componentwise coverage
and domination rules as the single-key query. The report is exactly
``clock`` (the requested boundary), ``results`` (one entry per requested
key, in order), ``found``, and ``missing``; each result entry carries
exactly ``key``, ``status`` (``resolved``/``conflict``/``absent``), and
``candidates`` (sorted by ``(replicaId, operationId)``). The whole batch
is one committed-snapshot read and is strictly read-only.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_causal_snapshot_payload,
)

CAUSAL_SNAPSHOT_PATH = "/v1/states/causal-at"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def entry(key: str, status: str, candidates: list[dict] | None = None) -> dict:
    return {"key": key, "status": status, "candidates": candidates or []}


class ParseCausalSnapshotPayloadTests(unittest.TestCase):
    """Body validation: exactly clock + a 1-100 list of distinct keys."""

    def test_minimal_document_passes(self) -> None:
        self.assertEqual(
            parse_causal_snapshot_payload(b'{"clock":{},"keys":["k"]}'),
            ({}, ["k"]),
        )
        self.assertEqual(
            parse_causal_snapshot_payload({"clock": {"r1": 2}, "keys": ["a", "b"]}),
            ({"r1": 2}, ["a", "b"]),
        )

    def test_key_order_is_preserved(self) -> None:
        _, keys = parse_causal_snapshot_payload(
            b'{"clock":{},"keys":["z","a","m"]}'
        )
        self.assertEqual(keys, ["z", "a", "m"])

    def test_up_to_one_hundred_keys_pass(self) -> None:
        keys = [f"k{i}" for i in range(100)]
        boundary, parsed = parse_causal_snapshot_payload(
            {"clock": {}, "keys": keys}
        )
        self.assertEqual(boundary, {})
        self.assertEqual(parsed, keys)

    def test_malformed_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}x",
            b"[]",
            b"null",
            b'""',
            b"42",
            b'{"clock":{},"keys":["k"]}\n{}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_missing_extra_and_unknown_fields_are_rejected(self) -> None:
        for raw in (
            b"{}",
            b'{"clock":{}}',
            b'{"keys":["k"]}',
            b'{"clock":{},"keys":["k"],"x":1}',
            b'{"clock":{},"keys":["k"],"keys":["j"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_duplicate_fields_inside_clock_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{"r1":1,"r1":2},"keys":["k"]}',
            b'{"clock":{},"keys":["k"],"clock":{"r1":1}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_clock_follows_the_single_key_boundary_rules(self) -> None:
        for raw in (
            b'{"clock":null,"keys":["k"]}',
            b'{"clock":[],"keys":["k"]}',
            b'{"clock":1,"keys":["k"]}',
            b'{"clock":{"":1},"keys":["k"]}',
            b'{"clock":{"r1":-1},"keys":["k"]}',
            b'{"clock":{"r1":true},"keys":["k"]}',
            b'{"clock":{"r1":"1"},"keys":["k"]}',
            b'{"clock":{"r1":1.0},"keys":["k"]}',
            b'{"clock":{"r1":-0.0},"keys":["k"]}',
            b'{"clock":{"r1":1e3},"keys":["k"]}',
            b'{"clock":{"r1":NaN},"keys":["k"]}',
            b'{"clock":{"r1":Infinity},"keys":["k"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_keys_must_be_a_non_empty_list(self) -> None:
        for raw in (
            b'{"clock":{},"keys":[]}',
            b'{"clock":{},"keys":"k"}',
            b'{"clock":{},"keys":null}',
            b'{"clock":{},"keys":{}}',
            b'{"clock":{},"keys":1}',
            b'{"clock":{},"keys":true}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_over_one_hundred_keys_are_rejected(self) -> None:
        keys = [f"k{i}" for i in range(101)]
        with self.assertRaises(ValueError):
            parse_causal_snapshot_payload({"clock": {}, "keys": keys})

    def test_each_key_must_be_a_non_empty_string(self) -> None:
        for keys in (
            [""],
            ["ok", ""],
            [1],
            [True],
            [None],
            [["k"]],
            [{"k": 1}],
        ):
            with self.subTest(keys=keys):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload({"clock": {}, "keys": keys})

    def test_duplicate_keys_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{},"keys":["a","a"]}',
            b'{"clock":{},"keys":["a","b","a"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_snapshot_payload(raw)

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(
            parse_causal_snapshot_payload(b'  { "clock": { }, "keys": [ "k" ] }\n'),
            ({}, ["k"]),
        )


class StatesCausalAtStoreTests(unittest.TestCase):
    """Store-level semantics of the one-pass cross-key replay."""

    def test_all_absent_on_a_fresh_store(self) -> None:
        store = StateStore()
        payload = store.get_states_causal_at({}, ["a", "b"])
        self.assertEqual(
            payload,
            {
                "clock": {},
                "results": [entry("a", "absent"), entry("b", "absent")],
                "found": 0,
                "missing": 2,
            },
        )

    def test_results_follow_request_order_not_name_order(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "z", "z1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "a", "a1", {"r1": 2}))
        payload = store.get_states_causal_at({"r1": 2}, ["z", "a", "missing"])
        self.assertEqual(
            [r["key"] for r in payload["results"]], ["z", "a", "missing"]
        )
        self.assertEqual(payload["found"], 2)
        self.assertEqual(payload["missing"], 1)
        self.assertEqual(payload["results"][0], entry("z", "resolved", [
            candidate("r1", "o1", "z1", {"r1": 1}),
        ]))
        self.assertEqual(payload["results"][1], entry("a", "resolved", [
            candidate("r1", "o2", "a1", {"r1": 2}),
        ]))
        self.assertEqual(payload["results"][2], entry("missing", "absent"))

    def test_one_boundary_classifies_every_key(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "shape", "round", {"r1": 3}))
        # Within {"r1":1,"r2":1}: color conflicts, shape is still absent
        # (its only record ticks r1=3), and an unknown key is absent.
        payload = store.get_states_causal_at(
            {"r1": 1, "r2": 1}, ["color", "shape", "nope"]
        )
        self.assertEqual(payload["clock"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 2)
        color, shape, nope = payload["results"]
        self.assertEqual(color["status"], "conflict")
        self.assertEqual(
            color["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )
        self.assertEqual(shape, entry("shape", "absent"))
        self.assertEqual(nope, entry("nope", "absent"))
        # At the wider boundary shape resolves and color stays in conflict;
        # both keys are now found.
        payload = store.get_states_causal_at(
            {"r1": 3, "r2": 1}, ["color", "shape", "nope"]
        )
        self.assertEqual(payload["found"], 2)
        self.assertEqual(payload["missing"], 1)
        self.assertEqual(payload["results"][0]["status"], "conflict")
        self.assertEqual(
            payload["results"][1],
            entry("shape", "resolved", [
                candidate("r1", "o3", "round", {"r1": 3})
            ]),
        )

    def test_each_key_matches_the_single_key_query(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "a", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "b", "w", {"r1": 3}))
        store.apply_operation("r1", operation("o4", "c", "x", {"r1": 4}))
        boundary = {"r1": 4, "r2": 1}
        payload = store.get_states_causal_at(boundary, ["a", "b", "c", "d"])
        for result in payload["results"]:
            status, single = store.get_state_causal_at(result["key"], boundary)
            if status is HTTPStatus.NOT_FOUND:
                self.assertEqual(result["status"], "absent")
                self.assertEqual(result["candidates"], [])
            else:
                self.assertEqual(result["status"], single["status"])
                self.assertEqual(result["candidates"], single["candidates"])

    def test_conflict_candidates_are_sorted_per_key(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1b", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1a", {"r1": 1}))
        payload = store.get_states_causal_at({"r1": 1, "r2": 1}, ["k"])
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["results"][0]["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o2")],
        )

    def test_dominated_and_stale_records_follow_the_replay_rules(self) -> None:
        store = StateStore()
        # Committed first, this clock dominates the later r2 write.
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        store.apply_operation("r2", operation("o2", "k", "old", {"r1": 1}))
        payload = store.get_states_causal_at({"r1": 2, "r2": 1}, ["k"])
        self.assertEqual(
            payload["results"][0]["candidates"],
            [candidate("r1", "o1", "new", {"r1": 2})],
        )

    def test_repeated_keys_in_a_key_list_are_a_caller_choice_not_an_error(self) -> None:
        # Caller-chosen order also means a caller asking for the same key
        # twice is legal at the store level; validation of distinct keys
        # happens in the payload parser, not the replay.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        payload = store.get_states_causal_at({"r1": 1}, ["k", "k"])
        self.assertEqual([r["key"] for r in payload["results"]], ["k", "k"])
        self.assertEqual(payload["found"], 2)
        self.assertEqual(payload["missing"], 0)

    def test_replay_does_not_move_metrics(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = store.get_metrics()
        store.get_states_causal_at({}, ["k", "absent"])
        store.get_states_causal_at({"r1": 1}, ["k"])
        self.assertEqual(store.get_metrics(), before)


class HttpStatesCausalAtTests(unittest.TestCase):
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
        return response.status, payload, headers, raw

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def snapshot(self, clock: object, keys: list, query: str = ""):
        return self.request(
            "POST", f"{CAUSAL_SNAPSHOT_PATH}{query}", {"clock": clock, "keys": keys}
        )

    def snapshot_raw(self, raw: bytes, query: str = ""):
        return self.request("POST", f"{CAUSAL_SNAPSHOT_PATH}{query}", raw)

    def seed(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        self.post_operation("r1", operation("o3", "shape", "round", {"r1": 3}))

    def test_round_trip_shape_and_encoding(self) -> None:
        self.seed()
        status, payload, headers, raw = self.snapshot(
            {"r1": 1, "r2": 1}, ["color", "shape", "nope"]
        )
        self.assertEqual(status, 200)
        # Top-level fields are exactly clock, results, found, missing, in
        # that order, compact JSON with one trailing newline.
        self.assertEqual(list(payload), ["clock", "results", "found", "missing"])
        self.assertEqual(payload["clock"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 2)
        self.assertEqual(
            payload["results"][0],
            entry("color", "conflict", [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ]),
        )
        self.assertEqual(set(payload["results"][1]), {"key", "status", "candidates"})
        self.assertEqual(payload["results"][1], entry("shape", "absent"))
        self.assertEqual(payload["results"][2], entry("nope", "absent"))
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        # Field order is the contracted order, not sorted (found precedes
        # results alphabetically, so a sorted encoding would differ).
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n",
        )
        self.assertIn(b'"clock":', raw)
        self.assertLess(raw.index(b'"results"'), raw.index(b'"found"'))
        self.assertLess(raw.index(b'"found"'), raw.index(b'"missing"'))
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        status, _, _, raw = self.snapshot({"r1": 3}, ["k"])
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"found":1', body)
        self.assertIn('"missing":0', body)

    def test_absent_keys_do_not_fail_the_batch(self) -> None:
        self.seed()
        status, payload, _, _ = self.snapshot({}, ["color", "shape", "ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(payload["found"], 0)
        self.assertEqual(payload["missing"], 3)
        self.assertTrue(all(r["status"] == "absent" for r in payload["results"]))
        self.assertTrue(all(r["candidates"] == [] for r in payload["results"]))

    def test_result_order_is_request_order(self) -> None:
        self.seed()
        status, payload, _, _ = self.snapshot(
            {"r1": 3, "r2": 1}, ["nope", "shape", "color"]
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["key"] for r in payload["results"]], ["nope", "shape", "color"]
        )
        self.assertEqual([r["status"] for r in payload["results"]], [
            "absent", "resolved", "conflict"
        ])

    def test_single_key_matches_the_single_key_endpoint(self) -> None:
        self.seed()
        boundary = {"r1": 1, "r2": 1}
        status, single, _, _ = self.request(
            "POST", "/v1/states/color/causal-at", {"clock": boundary}
        )
        self.assertEqual(status, 200)
        status, batch, _, _ = self.snapshot(boundary, ["color"])
        self.assertEqual(status, 200)
        self.assertEqual(batch["results"][0]["status"], single["status"])
        self.assertEqual(batch["results"][0]["candidates"], single["candidates"])
        self.assertEqual(batch["clock"], single["clock"])
        self.assertEqual(batch["found"], 1)
        self.assertEqual(batch["missing"], 0)

    def test_hundred_key_batch(self) -> None:
        self.post_operation("r1", operation("o1", "k0", "v", {"r1": 1}))
        keys = [f"k{i}" for i in range(100)]
        status, payload, _, _ = self.snapshot({"r1": 1}, keys)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["results"]), 100)
        self.assertEqual([r["key"] for r in payload["results"]], keys)
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 99)

    def test_malformed_bodies_are_400(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"[]",
            b"null",
            b'{"clock":{}}',
            b'{"keys":["k"]}',
            b'{"clock":{},"keys":["k"],"x":1}',
            b'{"x":{},"keys":["k"]}',
            b'{"clock":null,"keys":["k"]}',
            b'{"clock":[],"keys":["k"]}',
            b'{"clock":{"r1":-1},"keys":["k"]}',
            b'{"clock":{"r1":true},"keys":["k"]}',
            b'{"clock":{"r1":1.0},"keys":["k"]}',
            b'{"clock":{"r1":NaN},"keys":["k"]}',
            b'{"clock":{},"keys":[]}',
            b'{"clock":{},"keys":"k"}',
            b'{"clock":{},"keys":null}',
            b'{"clock":{},"keys":[1]}',
            b'{"clock":{},"keys":[true]}',
            b'{"clock":{},"keys":[""]}',
            b'{"clock":{},"keys":["a","a"]}',
            b'{"clock":{},"keys":' + b"[" + b",".join([b'"x"'] * 101) + b"]}",
            b'{"clock":{},"keys":["k"],"clock":{"r1":1}}',
            b'{"clock":{"r1":1,"r1":2},"keys":["k"]}',
        ):
            with self.subTest(raw=raw):
                status, payload, _, _ = self.snapshot_raw(raw)
                self.assertEqual(status, 400, raw)
                self.assertEqual(payload, {"error": "invalid_request"}, raw)

    def test_any_query_parameter_is_400(self) -> None:
        self.seed()
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?x&y="):
            with self.subTest(query=query):
                status, payload, _, _ = self.snapshot({"r1": 1}, ["color"], query)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A bare '?' is fine.
        status, _, _, _ = self.snapshot({"r1": 1}, ["color"], "?")
        self.assertEqual(status, 200)

    def test_bad_query_is_rejected_before_the_body_is_validated(self) -> None:
        status, payload, _, _ = self.snapshot_raw(b"{not json", "?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_wrong_path_shapes_are_404(self) -> None:
        self.seed()
        for path in (
            "/v1/states/causal-at/extra",
            "/v1/states/causal-at/",
            "/v1/states//causal-at",
            "/v1/causal-at",
            "/v2/states/causal-at",
            "/v1/state/causal-at",
            "/v1/states/color/causal-at-snapshot",
        ):
            status, payload, _, _ = self.request(
                "POST", path, {"clock": {"r1": 1}, "keys": ["color"]}
            )
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_single_key_route_still_requires_four_segments(self) -> None:
        self.seed()
        # The two routes coexist: the single-key body on the batch path is
        # a 400, and the batch body on the single-key path is a 400 —
        # neither route silently serves the other.
        status, payload, _, _ = self.request(
            "POST", CAUSAL_SNAPSHOT_PATH, {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        status, payload, _, _ = self.request(
            "POST", "/v1/states/color/causal-at",
            {"clock": {"r1": 1}, "keys": ["color"]},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_error_beats_query_and_body_errors(self) -> None:
        self.seed()
        for path, raw in (
            ("/v1/states/causal-at/extra?x=1", b"{not json"),
            ("/v1/states/causal-at/?x=1", b"{not json"),
            ("/v2/states/causal-at?x=1", b"{not json"),
        ):
            status, payload, _, _ = self.request("POST", path, raw)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_route_is_not_served(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("GET", CAUSAL_SNAPSHOT_PATH)
        # A GET never matches the POST route; it falls through to the
        # ordinary key lookup, which knows no such key.
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_is_read_only_over_http(self) -> None:
        self.seed()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        before_color, _, _, _ = self.request("GET", "/v1/states/color")
        self.snapshot({}, ["color", "ghost"])
        self.snapshot({"r1": 1, "r2": 1}, ["color", "shape", "nope"])
        self.snapshot_raw(b"{not json")
        self.snapshot({"r1": 1}, ["color"], "?x=1")
        self.snapshot({"r1": 1}, ["a", "a"])
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        after_color, _, _, _ = self.request("GET", "/v1/states/color")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_sync, after_sync)
        self.assertEqual(before_color, after_color)

    def test_whole_batch_is_one_snapshot_under_contention(self) -> None:
        # Hammer writes concurrently with batch reads over many requests;
        # every response must be internally consistent: its boundary is
        # echoed and its counts match its entries. A torn read would show
        # up as a candidate set that no committed prefix could produce —
        # the per-key replay is deterministic, so instead we assert the
        # structural invariants the snapshot guarantee implies.
        self.seed()
        stop = threading.Event()

        def writer() -> None:
            tick = 10
            while not stop.is_set():
                tick += 1
                self.request(
                    "POST",
                    "/v1/replicas/r3/operations",
                    operation(f"w{tick}", "color", f"c{tick}", {"r3": tick}),
                )

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            for _ in range(40):
                status, payload, _, _ = self.snapshot(
                    {"r1": 3, "r2": 1, "r3": 1_000_000_000},
                    ["color", "shape", "nope"],
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["clock"], {
                    "r1": 3, "r2": 1, "r3": 1_000_000_000
                })
                self.assertEqual(
                    [r["key"] for r in payload["results"]],
                    ["color", "shape", "nope"],
                )
                statuses = [r["status"] for r in payload["results"]]
                self.assertEqual(payload["found"], sum(s != "absent" for s in statuses))
                self.assertEqual(payload["missing"], sum(s == "absent" for s in statuses))
                for result in payload["results"]:
                    if result["status"] == "absent":
                        self.assertEqual(result["candidates"], [])
                    else:
                        self.assertIn(result["status"], ("resolved", "conflict"))
                        self.assertTrue(result["candidates"])
                        identities = [
                            (c["replicaId"], c["operationId"])
                            for c in result["candidates"]
                        ]
                        self.assertEqual(identities, sorted(identities))
        finally:
            stop.set()
            thread.join(timeout=5)


class HttpStatesCausalAtAuthTests(unittest.TestCase):
    """The batch endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-snapshot-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]

        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reader": ["read"],
                    "writer": ["write"],
                    "admin": ["read", "write", "admin"],
                },
                handle,
            )
        cls.scope_server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
        )
        cls.scope_thread = threading.Thread(
            target=cls.scope_server.serve_forever, daemon=True
        )
        cls.scope_thread.start()
        cls.scope_port = cls.scope_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.single_server.shutdown()
        cls.single_server.server_close()
        cls.scope_server.shutdown()
        cls.scope_server.server_close()
        cls.single_thread.join(timeout=5)
        cls.scope_thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        self.single_server.store = type(self.single_server.store)()
        self.scope_server.store = type(self.scope_server.store)()

    def request(self, port: int, method: str, path: str, body: object = None,
                auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, challenge

    def seed(self, port: int, token: str) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            port, "POST", "/v1/replicas/r1/operations", op, auth=token
        )
        self.assertEqual(status, 201)

    def body(self) -> dict:
        return {"clock": {"r1": 1}, "keys": ["k"]}

    def test_single_token_mode_requires_bearer_token(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body(),
                auth=auth,
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body(),
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["results"][0]["key"], "k")

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body(),
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body(),
                auth=token,
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["results"][0]["key"], "k", token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_SNAPSHOT_PATH + "?x=1", {"nope": {}},
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_rejected_auth_reads_and_changes_nothing(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        before, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.request(self.single_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body())
        self.request(
            self.single_port, "POST", CAUSAL_SNAPSHOT_PATH, self.body(),
            auth="Bearer nope",
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class HttpStatesCausalAtRequestLimitTests(unittest.TestCase):
    """The batch route keeps the shared Content-Length contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

        cls.auth_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.auth_thread = threading.Thread(
            target=cls.auth_server.serve_forever, daemon=True
        )
        cls.auth_thread.start()
        cls.auth_port = cls.auth_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.auth_server.shutdown()
        cls.auth_server.server_close()
        cls.thread.join(timeout=5)
        cls.auth_thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()
        self.auth_server.store = type(self.auth_server.store)()

    def post_raw(self, port: int, path: str, headers: list, body: bytes = b""):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", CAUSAL_SNAPSHOT_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, CAUSAL_SNAPSHOT_PATH,
                    [("Content-Length", value)], b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CAUSAL_SNAPSHOT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", CAUSAL_SNAPSHOT_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        status, payload = self.post_raw(
            self.auth_port,
            CAUSAL_SNAPSHOT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_body_at_exact_limit_is_processed_normally(self) -> None:
        template = b'{"clock":{},"keys":[""]}'
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"k" + b"x" * (pad - 1)
        body = b'{"clock":{},"keys":["' + name + b'"]}'
        self.assertEqual(len(body), MAX_BODY_BYTES)
        # The valid at-limit document is processed by the endpoint's
        # normal semantics: an unknown key is absent, not a rejection.
        status, payload = self.post_raw(
            self.port, CAUSAL_SNAPSHOT_PATH,
            [("Content-Length", str(len(body)))], body,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["found"], 0)
        self.assertEqual(payload["missing"], 1)

    def test_rejections_change_no_state(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST", "/v1/replicas/r1/operations",
            body=json.dumps(operation("o1", "k", "v1", {"r1": 1})),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/states/k")
        response = conn.getresponse()
        before = response.read()
        conn.close()
        for headers, body in (
            ([], b"{}"),
            ([("Content-Length", "abc")], b"{}"),
            ([("Content-Length", str(MAX_BODY_BYTES + 1))], b"junk"),
        ):
            status, _ = self.post_raw(self.port, CAUSAL_SNAPSHOT_PATH, headers, body)
            self.assertIn(status, (400, 413))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/states/k")
        response = conn.getresponse()
        self.assertEqual(response.read(), before)
        conn.close()


class HttpStatesCausalAtPersistenceTests(unittest.TestCase):
    """The same boundary and keys answer identically across a restart."""

    def test_restart_preserves_the_batch(self) -> None:
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
                        results.append((response.status, response.read()))
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            batch = {"clock": {"r1": 1, "r2": 1}, "keys": ["color", "shape"]}
            first = serve_once(
                [
                    ("POST", "/v1/replicas/r1/operations",
                     operation("o1", "color", "blue", {"r1": 1})),
                    ("POST", "/v1/replicas/r2/operations",
                     operation("o2", "color", "red", {"r2": 1})),
                    ("POST", CAUSAL_SNAPSHOT_PATH, batch),
                ]
            )
            self.assertEqual([status for status, _ in first], [201, 201, 200])
            second = serve_once([("POST", CAUSAL_SNAPSHOT_PATH, batch)])
            # Identical bytes, including the trailing newline.
            self.assertEqual(second[0], first[2])

    def test_query_never_touches_the_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "state.json"
            server = SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler, data_file=str(data_file)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                def call(method, path, body=None):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    if body is None:
                        conn.request(method, path)
                    else:
                        conn.request(
                            method, path, body=json.dumps(body),
                            headers={"Content-Type": "application/json"},
                        )
                    response = conn.getresponse()
                    result = (response.status, response.read())
                    conn.close()
                    return result

                self.assertEqual(
                    call(
                        "POST", "/v1/replicas/r1/operations",
                        operation("o1", "k", "v", {"r1": 1}),
                    )[0],
                    201,
                )
                before = data_file.read_bytes()
                before_listing = sorted(os.listdir(tmp))
                for body in (
                    {"clock": {}, "keys": ["k"]},
                    {"clock": {"r1": 1}, "keys": ["k", "absent"]},
                ):
                    status, _ = call("POST", CAUSAL_SNAPSHOT_PATH, body)
                    self.assertEqual(status, 200)
                self.assertEqual(data_file.read_bytes(), before)
                self.assertEqual(sorted(os.listdir(tmp)), before_listing)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
