"""Tests for the multi-key causal-boundary state query endpoint.

The endpoint is::

    POST /v1/states/causal-at

with a body of exactly ``{"clock": {...}, "keys": [...]}`` naming one
vector-clock boundary and 1-100 distinct non-empty string keys in
caller-specified order. It answers every key from the same causal slice
as the single-key ``POST /v1/states/{key}/causal-at`` route: starting
from the empty state it replays, in global commit order, only the
first-accepted records whose clock is componentwise no greater than the
boundary, and reports exactly ``clock``, ``results`` (one entry per
requested key, in request order, each with exactly ``key``, ``status``,
and ``candidates``), ``found``, and ``missing``. A key with no candidate
inside the boundary is ``"absent"`` with an empty candidate array and
never fails the batch. The query is strictly read-only, runs against one
committed snapshot, and answers with compact UTF-8 JSON terminated by
one newline.

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
    parse_causal_at_batch_payload,
)

CAUSAL_AT_BATCH_PATH = "/v1/states/causal-at"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


class ParseCausalAtBatchPayloadTests(unittest.TestCase):
    """Body validation: exactly ``clock`` plus a 1-100 distinct key list."""

    def test_valid_bodies_pass(self) -> None:
        self.assertEqual(
            parse_causal_at_batch_payload(b'{"clock":{},"keys":["a"]}'),
            ({}, ["a"]),
        )
        self.assertEqual(
            parse_causal_at_batch_payload({"clock": {"r1": 2}, "keys": ["a", "b"]}),
            ({"r1": 2}, ["a", "b"]),
        )
        # The caller-specified order is preserved.
        self.assertEqual(
            parse_causal_at_batch_payload(b'{"keys":["b","a"],"clock":{"r1":1}}'),
            ({"r1": 1}, ["b", "a"]),
        )

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(
            parse_causal_at_batch_payload(b'  { "clock": { }, "keys": [ "k" ] }\n'),
            ({}, ["k"]),
        )

    def test_malformed_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b'{"clock":{},"keys":["a"]}x',
            b"[]",
            b"null",
            b'""',
            b"42",
            b'["a"]',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        for raw in (
            b"{}",
            b'{"clock":{}}',
            b'{"keys":["a"]}',
            b'{"clock":{},"keys":["a"],"x":1}',
            b'{"clock":{},"keys":["a"]}\n{"clock":{},"keys":["a"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)

    def test_duplicate_fields_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{},"clock":{},"keys":["a"]}',
            b'{"clock":{},"keys":["a"],"keys":["b"]}',
            b'{"clock":{"r1":1,"r1":2},"keys":["a"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)

    def test_clock_constraints_match_the_single_key_route(self) -> None:
        for raw in (
            b'{"clock":null,"keys":["a"]}',
            b'{"clock":[],"keys":["a"]}',
            b'{"clock":1,"keys":["a"]}',
            b'{"clock":true,"keys":["a"]}',
            b'{"clock":{"":1},"keys":["a"]}',
            b'{"clock":{"r1":-1},"keys":["a"]}',
            b'{"clock":{"r1":true},"keys":["a"]}',
            b'{"clock":{"r1":"1"},"keys":["a"]}',
            b'{"clock":{"r1":1.0},"keys":["a"]}',
            b'{"clock":{"r1":-0.0},"keys":["a"]}',
            b'{"clock":{"r1":1e3},"keys":["a"]}',
            b'{"clock":{"r1":NaN},"keys":["a"]}',
            b'{"clock":{"r1":Infinity},"keys":["a"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)

    def test_keys_must_be_a_list_of_non_empty_strings(self) -> None:
        for raw in (
            b'{"clock":{},"keys":null}',
            b'{"clock":{},"keys":"a"}',
            b'{"clock":{},"keys":{}}',
            b'{"clock":{},"keys":1}',
            b'{"clock":{},"keys":[""]}',
            b'{"clock":{},"keys":[1]}',
            b'{"clock":{},"keys":[true]}',
            b'{"clock":{},"keys":[null]}',
            b'{"clock":{},"keys":[["a"]]}',
            b'{"clock":{},"keys":[{"a":1}]}',
            b'{"clock":{},"keys":["a",1]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)

    def test_empty_and_overlong_key_lists_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_at_batch_payload(b'{"clock":{},"keys":[]}')
        too_many = ["k%d" % i for i in range(101)]
        with self.assertRaises(ValueError):
            parse_causal_at_batch_payload({"clock": {}, "keys": too_many})
        exactly = ["k%d" % i for i in range(100)]
        self.assertEqual(
            parse_causal_at_batch_payload({"clock": {}, "keys": exactly}),
            ({}, exactly),
        )

    def test_duplicate_keys_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{},"keys":["a","a"]}',
            b'{"clock":{},"keys":["a","b","a"]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_batch_payload(raw)


class StatesCausalAtStoreTests(unittest.TestCase):
    """Store-level semantics of the multi-key causal-boundary replay."""

    def test_empty_store_reports_every_key_absent(self) -> None:
        store = StateStore()
        status, payload = store.get_states_causal_at({"r1": 1}, ["a", "b"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "clock": {"r1": 1},
                "results": [
                    {"key": "a", "status": "absent", "candidates": []},
                    {"key": "b", "status": "absent", "candidates": []},
                ],
                "found": 0,
                "missing": 2,
            },
        )

    def test_empty_boundary_is_absent_for_every_key(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_states_causal_at({}, ["k", "absent"])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["found"], 0)
        self.assertEqual(payload["missing"], 2)
        self.assertEqual(
            [r["status"] for r in payload["results"]], ["absent", "absent"]
        )

    def test_results_follow_request_order_not_coverage(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "b", "vb", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "a", "va", {"r1": 2}))
        _, payload = store.get_states_causal_at({"r1": 2}, ["z", "a", "b"])
        self.assertEqual([r["key"] for r in payload["results"]], ["z", "a", "b"])
        self.assertEqual(
            [r["status"] for r in payload["results"]],
            ["absent", "resolved", "resolved"],
        )
        self.assertEqual((payload["found"], payload["missing"]), (2, 1))

    def test_found_and_missing_counts_cover_every_key(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, payload = store.get_states_causal_at({"r1": 1}, ["k", "x", "y"])
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 2)
        self.assertEqual(payload["found"] + payload["missing"], 3)

    def test_per_key_results_match_the_single_key_query(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "j", "one", {"r1": 2}))
        boundary = {"r1": 2, "r2": 1}
        _, batch = store.get_states_causal_at(boundary, ["k", "j", "absent"])
        for entry in batch["results"]:
            if entry["status"] == "absent":
                self.assertEqual(entry, {"key": "absent", "status": "absent", "candidates": []})
                continue
            _, single = store.get_state_causal_at(entry["key"], boundary)
            self.assertEqual(entry["status"], single["status"], entry["key"])
            self.assertEqual(entry["candidates"], single["candidates"], entry["key"])

    def test_conflict_candidates_sorted_by_identity(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1b", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1a", {"r1": 1}))
        _, payload = store.get_states_causal_at({"r1": 1, "r2": 1}, ["k"])
        entry = payload["results"][0]
        self.assertEqual(entry["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in entry["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o2")],
        )

    def test_boundary_selects_each_key_independently(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        _, early = store.get_states_causal_at({"r1": 1}, ["a", "b"])
        self.assertEqual(
            [r["status"] for r in early["results"]], ["resolved", "absent"]
        )
        _, late = store.get_states_causal_at({"r1": 2}, ["a", "b"])
        self.assertEqual(
            [r["status"] for r in late["results"]], ["resolved", "resolved"]
        )

    def test_stale_writes_and_overwrites_follow_single_key_rules(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        store.apply_operation("r2", operation("o2", "k", "old", {"r1": 1}))
        _, payload = store.get_states_causal_at({"r1": 2, "r2": 1}, ["k"])
        self.assertEqual(
            payload["results"][0]["candidates"],
            [candidate("r1", "o1", "new", {"r1": 2})],
        )
        _, earlier = store.get_states_causal_at({"r1": 1}, ["k"])
        self.assertEqual(
            earlier["results"][0]["candidates"],
            [candidate("r2", "o2", "old", {"r1": 1})],
        )

    def test_boundary_is_echoed_back_verbatim(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, payload = store.get_states_causal_at({"r1": 1, "r9": 9}, ["k"])
        self.assertEqual(payload["clock"], {"r1": 1, "r9": 9})

    def test_query_reads_nothing_but_the_snapshot(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = store.get_metrics()
        store.get_states_causal_at({}, ["k"])
        store.get_states_causal_at({"r1": 1}, ["k", "absent"])
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

    def causal_at(self, clock: object, keys: object, query: str = ""):
        return self.request(
            "POST", f"{CAUSAL_AT_BATCH_PATH}{query}", {"clock": clock, "keys": keys}
        )

    def causal_at_raw(self, raw: bytes, query: str = ""):
        return self.request("POST", f"{CAUSAL_AT_BATCH_PATH}{query}", raw)

    def test_round_trip_shape_and_encoding(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, payload, headers, raw = self.causal_at(
            {"r1": 1, "r2": 1}, ["color", "absent"]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"clock", "results", "found", "missing"})
        self.assertEqual(payload["clock"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 1)
        self.assertEqual(
            payload["results"],
            [
                {
                    "key": "color",
                    "status": "conflict",
                    "candidates": [
                        candidate("r1", "o1", "blue", {"r1": 1}),
                        candidate("r2", "o2", "red", {"r2": 1}),
                    ],
                },
                {"key": "absent", "status": "absent", "candidates": []},
            ],
        )
        for entry in payload["results"]:
            self.assertEqual(set(entry), {"key", "status", "candidates"})
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact JSON terminated by exactly one newline, with the declared
        # length covering the terminator.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n",
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        status, _, _, raw = self.causal_at({"r1": 3}, ["k", "absent"])
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"found":1', body)
        self.assertIn('"missing":1', body)

    def test_results_keep_request_order(self) -> None:
        self.post_operation("r1", operation("o1", "b", "vb", {"r1": 1}))
        self.post_operation("r1", operation("o2", "a", "va", {"r1": 2}))
        status, payload, _, _ = self.causal_at({"r1": 2}, ["z", "a", "b"])
        self.assertEqual(status, 200)
        self.assertEqual([r["key"] for r in payload["results"]], ["z", "a", "b"])
        self.assertEqual(
            [r["status"] for r in payload["results"]],
            ["absent", "resolved", "resolved"],
        )

    def test_absent_keys_never_fail_the_batch(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.causal_at({"r1": 1}, ["x", "k", "y"])
        self.assertEqual(status, 200)
        self.assertEqual(payload["found"], 1)
        self.assertEqual(payload["missing"], 2)
        self.assertEqual(
            payload["results"][0],
            {"key": "x", "status": "absent", "candidates": []},
        )
        self.assertEqual(payload["results"][1]["status"], "resolved")
        self.assertEqual(
            payload["results"][2],
            {"key": "y", "status": "absent", "candidates": []},
        )

    def test_empty_clock_reports_all_absent_with_200(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.causal_at({}, ["k", "absent"])
        self.assertEqual(status, 200)
        self.assertEqual(payload["found"], 0)
        self.assertEqual(payload["missing"], 2)

    def test_batch_matches_single_key_answers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.post_operation("r1", operation("o3", "j", "w", {"r1": 2}))
        boundary = {"r1": 2, "r2": 1}
        status, payload, _, _ = self.causal_at(boundary, ["k", "j"])
        self.assertEqual(status, 200)
        for entry in payload["results"]:
            single_status, single, _, _ = self.request(
                "POST", f"/v1/states/{entry['key']}/causal-at", {"clock": boundary}
            )
            self.assertEqual(single_status, 200)
            self.assertEqual(entry["status"], single["status"], entry["key"])
            self.assertEqual(entry["candidates"], single["candidates"], entry["key"])

    def test_malformed_bodies_are_400(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"[]",
            b"null",
            b'{"clock":{}}',
            b'{"keys":["a"]}',
            b'{"clock":{},"keys":["a"],"x":1}',
            b'{"clock":null,"keys":["a"]}',
            b'{"clock":{"r1":1.0},"keys":["a"]}',
            b'{"clock":{"r1":-1},"keys":["a"]}',
            b'{"clock":{"r1":NaN},"keys":["a"]}',
            b'{"clock":{},"keys":[]}',
            b'{"clock":{},"keys":[""]}',
            b'{"clock":{},"keys":[1]}',
            b'{"clock":{},"keys":["a","a"]}',
            b'{"clock":{},"clock":{},"keys":["a"]}',
            b'{"clock":{},"keys":["a"],"keys":["b"]}',
        ):
            with self.subTest(raw=raw):
                status, payload, _, _ = self.causal_at_raw(raw)
                self.assertEqual(status, 400, raw)
                self.assertEqual(payload, {"error": "invalid_request"}, raw)

    def test_overlong_key_list_is_400(self) -> None:
        keys = ["k%d" % i for i in range(101)]
        status, payload, _, _ = self.causal_at({}, keys)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        keys = ["k%d" % i for i in range(100)]
        status, payload, _, _ = self.causal_at({}, keys)
        self.assertEqual(status, 200)
        self.assertEqual(payload["missing"], 100)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?x&y="):
            with self.subTest(query=query):
                status, payload, _, _ = self.causal_at({"r1": 1}, ["k"], query)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A bare '?' is fine.
        status, _, _, _ = self.causal_at({"r1": 1}, ["k"], "?")
        self.assertEqual(status, 200)

    def test_bad_query_is_rejected_before_the_body_is_validated(self) -> None:
        status, payload, _, _ = self.causal_at_raw(b"{not json", "?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/causal-at/extra",
            "/v1/states/causal-at/",
            "/v1/causal-at",
            "/v2/states/causal-at",
        ):
            status, payload, _, _ = self.request(
                "POST", path, {"clock": {"r1": 1}, "keys": ["k"]}
            )
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_and_body_errors(self) -> None:
        for path in (
            "/v1/states/causal-at/extra?x=1",
            "/v1/states/causal-at/?x=1",
            "/v2/states/causal-at?x=1",
        ):
            status, payload, _, _ = self.request("POST", path, b"{not json")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_batch_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # GET /v1/states/causal-at is the single-key current-state route
        # for the key "causal-at", which holds nothing: 404 either way.
        status, payload, _, _ = self.request("GET", CAUSAL_AT_BATCH_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_single_key_route_is_unaffected(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request(
            "POST", "/v1/states/k/causal-at", {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["status"], "resolved")

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        self.causal_at({}, ["k"])
        self.causal_at({"r1": 1}, ["k"])
        self.causal_at({"r1": 1, "r2": 1}, ["k", "absent"])
        self.causal_at_raw(b"{not json")
        self.causal_at({"r1": 1}, ["k"], "?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_sync, after_sync)


class HttpStatesCausalAtAuthTests(unittest.TestCase):
    """The batch causal-at endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-causal-batch-auth-")
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

    def test_single_token_mode_requires_bearer_token(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        body = {"clock": {"r1": 1}, "keys": ["k"]}
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", CAUSAL_AT_BATCH_PATH, body, auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", CAUSAL_AT_BATCH_PATH, body,
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["found"], 1)

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        body = {"clock": {"r1": 1}, "keys": ["k"]}
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_AT_BATCH_PATH, body,
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", CAUSAL_AT_BATCH_PATH, body, auth=token
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["found"], 1, token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_AT_BATCH_PATH + "?x=1", {"nope": {}},
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
        body = {"clock": {"r1": 1}, "keys": ["k"]}
        self.request(self.single_port, "POST", CAUSAL_AT_BATCH_PATH, body)
        self.request(
            self.single_port, "POST", CAUSAL_AT_BATCH_PATH, body,
            auth="Bearer nope",
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class HttpStatesCausalAtRequestLimitTests(unittest.TestCase):
    """The batch causal-at route keeps the shared Content-Length contract."""

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
        conn.putrequest("POST", CAUSAL_AT_BATCH_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, CAUSAL_AT_BATCH_PATH,
                    [("Content-Length", value)], b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CAUSAL_AT_BATCH_PATH,
            [("Content-Length", "2"), ("Content-Length", "3")],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        # The content itself is invalid JSON; the declared size wins.
        status, payload = self.post_raw(
            self.port,
            CAUSAL_AT_BATCH_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        # Missing declaration: 400 even without a bearer token.
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", CAUSAL_AT_BATCH_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        # Over-limit declaration: 413, not 401, even with no token.
        status, payload = self.post_raw(
            self.auth_port,
            CAUSAL_AT_BATCH_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_rejections_change_no_state(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST", "/v1/replicas/r1/operations",
            body=json.dumps(operation("o1", "k", "v1", {"r1": 1})),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(conn.getresponse().status, 201)
        conn.request("GET", "/v1/states/k")
        before = conn.getresponse().read()
        conn.close()
        for headers, body in (
            ([], b"{}"),
            ([("Content-Length", "abc")], b"{}"),
            ([("Content-Length", str(MAX_BODY_BYTES + 1))], b"junk"),
        ):
            status, _ = self.post_raw(self.port, CAUSAL_AT_BATCH_PATH, headers, body)
            self.assertIn(status, (400, 413))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/states/k")
        after = conn.getresponse().read()
        conn.close()
        self.assertEqual(before, after)


class HttpStatesCausalAtPersistenceTests(unittest.TestCase):
    """The same boundary answers identically across a data-file restart."""

    def test_restart_preserves_the_causal_slice(self) -> None:
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
                        raw = response.read()
                        results.append((response.status, raw))
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            first = serve_once(
                [
                    (
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o1", "k", "v1", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "j", "v2", {"r2": 1}),
                    ),
                    (
                        "POST",
                        CAUSAL_AT_BATCH_PATH,
                        {"clock": {"r1": 1}, "keys": ["k", "j"]},
                    ),
                    (
                        "POST",
                        CAUSAL_AT_BATCH_PATH,
                        {"clock": {"r1": 1, "r2": 1}, "keys": ["k", "j", "x"]},
                    ),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    (
                        "POST",
                        CAUSAL_AT_BATCH_PATH,
                        {"clock": {"r1": 1}, "keys": ["k", "j"]},
                    ),
                    (
                        "POST",
                        CAUSAL_AT_BATCH_PATH,
                        {"clock": {"r1": 1, "r2": 1}, "keys": ["k", "j", "x"]},
                    ),
                ]
            )
            # Same boundaries before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0], first[2])
            self.assertEqual(second[1], first[3])

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
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o1", "k", "v", {"r1": 1}),
                    )[0],
                    201,
                )
                before = data_file.read_bytes()
                before_listing = sorted(os.listdir(tmp))
                for body in (
                    {"clock": {}, "keys": ["k"]},
                    {"clock": {"r1": 1}, "keys": ["k", "absent"]},
                    {"clock": {"r1": 1, "r2": 1}, "keys": ["k"]},
                ):
                    status, _ = call("POST", CAUSAL_AT_BATCH_PATH, body)
                    self.assertEqual(status, 200)
                self.assertEqual(data_file.read_bytes(), before)
                self.assertEqual(sorted(os.listdir(tmp)), before_listing)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
