"""Tests for the read-only causal-slice state query endpoint.

The endpoint is::

    POST /v1/states/{key}/causal-at

with a body of exactly ``{"clock": {...}}`` naming a vector-clock boundary.
The query starts from the empty state and replays only the first-accepted
operations whose clock is no later than the boundary on every component
(missing boundary components count as zero), in the shared global commit
order. Ordinary writes, stale writes, sync imports, and accepted repairs
all participate; candidate adds and deletes keep the existing vector-clock
domination semantics. The success report carries exactly the requested
``boundary``, the ``key``, the ``status``, and ``candidates`` (always an
array). Everything here goes through the real HTTP entry point
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
    parse_causal_at_payload,
)

CAUSAL_PATH = "/v1/states/k/causal-at"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


class ParseCausalAtPayloadTests(unittest.TestCase):
    """Strict validation of the ``{"clock": ...}`` request body."""

    def test_empty_clock_is_the_causal_origin(self) -> None:
        self.assertEqual(parse_causal_at_payload(b'{"clock":{}}'), {})
        self.assertEqual(parse_causal_at_payload({"clock": {}}), {})

    def test_valid_clocks_pass(self) -> None:
        self.assertEqual(
            parse_causal_at_payload(b'{"clock":{"r1":0,"r2":42}}'),
            {"r1": 0, "r2": 42},
        )

    def test_malformed_and_wrong_typed_documents_are_rejected(self) -> None:
        for raw in (
            b"{oops",
            b"[]",
            b'"x"',
            b"42",
            b"null",
            b"",
            b"{}",
            b'{"other":{}}',
            b'{"clock":{}}x',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_unknown_or_missing_fields_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{"r1":1},"x":1}',
            b'{"clock":{"r1":1},"clock2":{}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_clock_must_be_an_object(self) -> None:
        for raw in (
            b'{"clock":[]}',
            b'{"clock":null}',
            b'{"clock":1}',
            b'{"clock":"r1:1"}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_illegal_components_and_ticks_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{"":1}}',
            b'{"clock":{"r1":-1}}',
            b'{"clock":{"r1":true}}',
            b'{"clock":{"r1":false}}',
            b'{"clock":{"r1":"1"}}',
            b'{"clock":{"r1":null}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_floats_and_non_finite_values_are_rejected(self) -> None:
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload({"clock": {"r1": tick}})
        for literal in ("NaN", "Infinity", "-Infinity", "1e3", "-0.0"):
            with self.subTest(literal=literal):
                raw = b'{"clock":{"r1":' + literal.encode("ascii") + b"}}"
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_duplicate_fields_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{"r1":1},"clock":{"r2":2}}',
            b'{"clock":{"r1":1,"r1":2}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)


class StateCausalAtStoreTests(unittest.TestCase):
    """Store-level semantics of the causal replay."""

    def test_empty_boundary_is_always_404(self) -> None:
        store = StateStore()
        self.assertEqual(
            store.get_state_causal_at("k", {}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_causal_at("k", {}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_unknown_key_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_causal_at("absent", {"r1": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_key_only_past_the_boundary_is_404_even_though_it_appears_later(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "x", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "k", "v", {"r1": 2}))
        self.assertEqual(
            store.get_state_causal_at("k", {"r1": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        status, payload = store.get_state_causal_at("k", {"r1": 2})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["boundary"], {"r1": 2})

    def test_minimal_boundary_covering_a_clock_includes_the_operation(self) -> None:
        store = StateStore()
        store.apply_operation(
            "r2", operation("o1", "k", "v", {"r1": 1, "r2": 1})
        )
        # A boundary missing one component treats it as zero, so the clock
        # is not yet covered.
        self.assertEqual(
            store.get_state_causal_at("k", {"r1": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_state_causal_at("k", {"r2": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        # The clock itself is the minimal covering boundary.
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["candidates"], [candidate("r2", "o1", "v", {"r1": 1, "r2": 1})])
        # A later boundary still covers it.
        status, payload = store.get_state_causal_at("k", {"r1": 2, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(payload["candidates"]), 1)

    def test_report_shape_and_sorting(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(set(payload), {"boundary", "key", "status", "candidates"})
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["boundary"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "v1", {"r1": 1}),
                candidate("r2", "o2", "v2", {"r2": 1}),
            ],
        )

    def test_same_value_candidates_are_resolved_but_all_kept(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o1", "k", "blue", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(len(payload["candidates"]), 2)

    def test_selection_is_causal_not_a_global_prefix(self) -> None:
        # The first committed record is excluded by the boundary while the
        # second one is included: the replay follows causality, not the log
        # prefix.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "x", {"r1": 1}))
        store.apply_operation("r2", operation("o1", "k", "a", {"r2": 1}))
        status, payload = store.get_state_causal_at("k", {"r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["candidates"], [candidate("r2", "o1", "a", {"r2": 1})])

    def test_excluded_dominator_lets_a_stale_write_survive(self) -> None:
        # o1 dominates o2 in the current state, but a boundary that covers
        # only o2 replays it from the empty state as a live candidate. The
        # stale write participates like every accepted record.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        store.apply_operation("r2", operation("o2", "k", "old", {"r1": 1}))
        status, payload = store.get_state_causal_at("k", {"r1": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r2", "o2", "old", {"r1": 1})]
        )
        status, payload = store.get_state_causal_at("k", {"r1": 2})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "new", {"r1": 2})]
        )

    def test_domination_applies_within_the_replayed_subset(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r1": 1, "r2": 1}))
        # Both covered: o2 dominates o1, leaving only v2.
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [candidate("r2", "o2", "v2", {"r1": 1, "r2": 1})],
        )

    def test_sync_imports_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "local", {"r1": 1}))
        status, accepted, _ = store.import_operations(
            [
                ("r3", operation("o9", "k", "imported", {"r3": 1})),
                ("r3", operation("o10", "other", "x", {"r3": 2})),
            ]
        )
        self.assertEqual((status, accepted), (HTTPStatus.CREATED, 2))
        status, payload = store.get_state_causal_at("k", {"r3": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r3", "o9", "imported", {"r3": 1})]
        )
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r3": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(len(payload["candidates"]), 2)

    def test_repairs_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        resolution = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        # Before the repair the conflict is visible.
        status, before = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(before["status"], "conflict")
        # At the repair boundary the key is resolved.
        status, after = store.get_state_causal_at(
            "k", {"r1": 1, "r2": 1, "r3": 1}
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(after["status"], "resolved")
        self.assertEqual(
            after["candidates"],
            [candidate("r3", "fix-1", "merged", {"r1": 1, "r2": 1, "r3": 1})],
        )

    def test_replays_and_rejections_are_not_in_the_log(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1})),
            HTTPStatus.OK,
        )
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        status, payload = store.get_state_causal_at("k", {"r1": 10})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(payload["candidates"]), 1)

    def test_read_is_strictly_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        before = store.get_metrics()
        store.get_state_causal_at("k", {"r1": 1})
        store.get_state_causal_at("k", {})
        store.get_state_causal_at("absent", {"r1": 1, "r2": 1})
        self.assertEqual(store.get_metrics(), before)


class HttpStateCausalAtTests(unittest.TestCase):
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

    def request_raw(
        self, method: str, path: str, raw: bytes | None, headers: dict | None = None
    ) -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        if raw is None:
            conn.request(method, path, headers=merged)
        else:
            conn.request(method, path, body=raw, headers=merged)
        response = conn.getresponse()
        data = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, data, response_headers

    def request(self, method: str, path: str, body: object = None):
        if isinstance(body, (bytes, str)):
            raw = body.encode("utf-8") if isinstance(body, str) else body
        elif body is None:
            raw = None
        else:
            raw = json.dumps(body).encode("utf-8")
        status, data, headers = self.request_raw(method, path, raw)
        payload = json.loads(data.decode("utf-8")) if data else None
        return status, payload, headers, data

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def causal_at(self, key: str, clock: object, query: str = ""):
        return self.request("POST", f"/v1/states/{key}/causal-at{query}", {"clock": clock})

    def seed_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))

    def test_round_trip_shape_and_encoding(self) -> None:
        self.seed_conflict()
        status, payload, headers, raw = self.causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"boundary", "key", "status", "candidates"})
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["boundary"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "v1", {"r1": 1}),
                candidate("r2", "o2", "v2", {"r2": 1}),
            ],
        )
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n",
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        status, _, _, raw = self.causal_at("k", {"r1": 3})
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)

    def test_empty_clock_is_404_for_every_key(self) -> None:
        self.seed_conflict()
        for key in ("k", "absent"):
            status, payload, _, raw = self.causal_at(key, {})
            self.assertEqual(status, 404, key)
            self.assertEqual(payload, {"error": "not_found"}, key)
            # Even the business 404 keeps the single-newline contract.
            self.assertEqual(raw, b'{"error":"not_found"}\n', key)

    def test_boundary_walks_the_causal_slice(self) -> None:
        self.seed_conflict()
        # Only r1 has happened: one resolved candidate.
        status, payload, _, _ = self.causal_at("k", {"r1": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "v1", {"r1": 1})])
        # Both sides covered: the conflict is visible.
        status, payload, _, _ = self.causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(len(payload["candidates"]), 2)
        # Boundary short of the key's clocks: 404 even though records exist.
        status, payload, _, _ = self.causal_at("k", {"r9": 9})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_bodies_are_400(self) -> None:
        self.seed_conflict()
        bad_bodies = [
            b"{oops",
            b"[]",
            b"{}",
            b"null",
            b'{"other":{}}',
            b'{"clock":{"r1":1},"x":1}',
            b'{"clock":[]}',
            b'{"clock":null}',
            b'{"clock":{"":1}}',
            b'{"clock":{"r1":-1}}',
            b'{"clock":{"r1":true}}',
            b'{"clock":{"r1":"1"}}',
        ]
        for raw_body in bad_bodies:
            status, payload, _, _ = self.request("POST", CAUSAL_PATH, raw_body)
            self.assertEqual(status, 400, repr(raw_body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(raw_body))

    def test_float_and_non_finite_clocks_are_400(self) -> None:
        self.seed_conflict()
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                status, payload, _, _ = self.causal_at("k", {"r1": tick})
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for literal in ("NaN", "Infinity", "-Infinity", "-0.0", "1e3"):
            with self.subTest(literal=literal):
                raw_body = b'{"clock":{"r1":' + literal.encode("ascii") + b"}}"
                status, payload, _, _ = self.request("POST", CAUSAL_PATH, raw_body)
                self.assertEqual(status, 400, literal)
                self.assertEqual(payload, {"error": "invalid_request"}, literal)

    def test_duplicate_fields_are_400(self) -> None:
        self.seed_conflict()
        for raw_body in (
            b'{"clock":{"r1":1},"clock":{"r2":2}}',
            b'{"clock":{"r1":1,"r1":2}}',
        ):
            status, payload, _, _ = self.request("POST", CAUSAL_PATH, raw_body)
            self.assertEqual(status, 400, raw_body)
            self.assertEqual(payload, {"error": "invalid_request"}, raw_body)

    def test_any_query_parameter_is_400(self) -> None:
        self.seed_conflict()
        for query in ("?x=1", "?x=1&x=2", "?x=", "?=1", "?x&y=", "?clock=1"):
            status, payload, _, _ = self.request(
                "POST", CAUSAL_PATH + query, {"clock": {"r1": 1}}
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A bare '?' is fine.
        status, _, _, _ = self.request("POST", CAUSAL_PATH + "?", {"clock": {"r1": 1}})
        self.assertEqual(status, 200)

    def test_bad_query_beats_bad_body(self) -> None:
        status, payload, _, _ = self.request(
            "POST", CAUSAL_PATH + "?x=1", b"{not json"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        self.seed_conflict()
        for path in (
            "/v1/states/k/causal-at/extra",
            "/v1/states/k/causal-at/extra/two",
            "/v1/states/k/causal-at/",
            "/v1/states//causal-at",
            "/v1/states/causal-at",
            "/v1/causal-at",
            "/v2/states/k/causal-at",
        ):
            status, payload, _, _ = self.request("POST", path, {"clock": {}})
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_404_takes_priority_over_query_and_body(self) -> None:
        self.seed_conflict()
        for path in (
            "/v1/states/k/causal-at/extra?x=1",
            "/v1/states/k/causal-at/?x=1",
            "/v2/states/k/causal-at?x=1",
        ):
            status, payload, _, _ = self.request("POST", path, b"{not json")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_causal_at_path_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", CAUSAL_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        self.post_operation("r1", operation("o1", "k/1", "v", {"r1": 1}))
        status, payload, _, _ = self.causal_at("k%2F1", {"r1": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_stale_import_and_repair_records_participate_over_http(self) -> None:
        # A stale write (its clock already dominated) is in the log: r1
        # first accepts a tick-2 write, then a lagging tick-1 write whose
        # clock is dominated and therefore adds no current candidate.
        self.post_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        self.post_operation("r1", operation("o2", "k", "old", {"r1": 1}))
        # At the earlier boundary o1 has not happened, so the otherwise
        # stale lagging write is the only replayed candidate.
        status, payload, _, _ = self.causal_at("k", {"r1": 1})
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o2", "old", {"r1": 1})]
        )
        # At the later boundary the dominating o1 wins and the stale write
        # disappears from the candidate set.
        status, payload, _, _ = self.causal_at("k", {"r1": 2})
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "new", {"r1": 2})]
        )
        # A sync import participates.
        import_body = {
            "operations": [
                {"replicaId": "r3", "operation": operation("o9", "k", "imp", {"r3": 1})}
            ]
        }
        status, _, _, _ = self.request("POST", "/v1/sync/operations", import_body)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.causal_at("k", {"r3": 1})
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["candidates"], [candidate("r3", "o9", "imp", {"r3": 1})]
        )
        # The current candidates are the concurrent o1 (new) and o9 (imp),
        # so a repair naming exactly that set dominates and resolves them.
        resolve_body = {
            "replicaId": "r4",
            "operationId": "fix-1",
            "value": "done",
            "clock": {"r1": 2, "r3": 1, "r4": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r3", "operationId": "o9"},
            ],
        }
        status, _, _, _ = self.request("POST", "/v1/states/k/resolve", resolve_body)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.causal_at(
            "k", {"r1": 2, "r3": 1, "r4": 1}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [
                candidate(
                    "r4",
                    "fix-1",
                    "done",
                    {"r1": 2, "r3": 1, "r4": 1},
                )
            ],
        )

    def test_query_is_read_only_over_http(self) -> None:
        self.seed_conflict()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.causal_at("k", {"r1": 1})
        self.causal_at("k", {"r1": 1, "r2": 1})
        self.causal_at("k", {})
        self.causal_at("absent", {"r1": 1})
        self.request("POST", CAUSAL_PATH, b"{bad")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)

    def test_malformed_content_length_is_400_without_reading_body(self) -> None:
        for value in ("abc", "", "+5", "-5", "5.0"):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.putrequest("POST", CAUSAL_PATH)
            conn.putheader("Content-Length", value)
            conn.endheaders(b"{}")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
            self.assertEqual(response.status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload, _ = self.request_raw(
            "POST",
            CAUSAL_PATH,
            b"not json",
            {"Content-Length": str(MAX_BODY_BYTES + 1)},
        )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(payload.decode("utf-8")), {"error": "payload_too_large"})


class HttpStateCausalAtAuthTests(unittest.TestCase):
    """Authentication and scope behavior of the causal-slice endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-causal-auth-")
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

    def request(self, port: int, path: str, auth: str | None = None, body: object = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        raw_body = None if body is None else json.dumps(body).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if raw_body is None:
            conn.request("POST", path, headers=headers)
        else:
            conn.request("POST", path, body=raw_body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, json.loads(data.decode("utf-8")), challenge

    def seed(self, port: int, auth: str) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r1/operations",
            body=json.dumps(op),
            headers={"Content-Type": "application/json", "Authorization": auth},
        )
        self.assertEqual(conn.getresponse().status, 201)
        conn.close()

    def test_single_token_mode_requires_bearer_token(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload, challenge = self.request(
                self.single_port, CAUSAL_PATH, auth, {"clock": {"r1": 1}}
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, CAUSAL_PATH, "Bearer sekret", {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        # A write-only token is 403 without a challenge and the body is
        # never read.
        status, payload, challenge = self.request(
            self.scope_port, CAUSAL_PATH, "Bearer writer", {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        status, payload, _ = self.request(
            self.scope_port, CAUSAL_PATH, "Bearer reader", {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")
        status, payload, _ = self.request(
            self.scope_port, CAUSAL_PATH, "Bearer admin", {"clock": {"r1": 1}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_content_length_priority_beats_authentication(self) -> None:
        # An over-limit declaration is 413 even with no token at all, and
        # the body is never read.
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("POST", CAUSAL_PATH)
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.endheaders(b"not json")
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(
            json.loads(response.read().decode("utf-8")), {"error": "payload_too_large"}
        )
        self.assertIsNone(response.getheader("WWW-Authenticate"))
        conn.close()


class HttpStateCausalAtPersistenceTests(unittest.TestCase):
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
                        results.append((response.status, response.read()))
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
                        operation("o2", "k", "v2", {"r2": 1}),
                    ),
                    ("POST", CAUSAL_PATH, {"clock": {}}),
                    ("POST", CAUSAL_PATH, {"clock": {"r1": 1}}),
                    ("POST", CAUSAL_PATH, {"clock": {"r1": 1, "r2": 1}}),
                ]
            )
            self.assertEqual([status for status, _ in first], [201, 201, 404, 200, 200])
            second = serve_once(
                [
                    ("POST", CAUSAL_PATH, {"clock": {}}),
                    ("POST", CAUSAL_PATH, {"clock": {"r1": 1}}),
                    ("POST", CAUSAL_PATH, {"clock": {"r1": 1, "r2": 1}}),
                ]
            )
            # Identical bytes, including the single trailing newline.
            self.assertEqual(second[0], first[2])
            self.assertEqual(second[1], first[3])
            self.assertEqual(second[2], first[4])


if __name__ == "__main__":
    unittest.main()
