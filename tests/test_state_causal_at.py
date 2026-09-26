"""Tests for the causal-boundary historical state query endpoint.

The endpoint is::

    POST /v1/states/{key}/causal-at

with a body of exactly ``{"clock": {...}}`` naming a vector-clock
boundary. Starting from the empty state it replays, in global commit
order, only the first-accepted records whose clock is componentwise no
greater than the boundary (missing components count as 0), and reports
one key's candidate state: exactly ``clock`` (the requested boundary),
``key``, ``status``, and ``candidates`` (always an array, even when
resolved). The empty clock is the causal origin. The replay covers every
first-accepted record that falls inside the boundary — ordinary writes,
stale writes, accepted repairs, and sync imports — and nothing else. The
query is strictly read-only, runs against one committed snapshot, and
answers with compact UTF-8 JSON terminated by one newline.

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
    parse_causal_at_payload,
)

CAUSAL_AT_PATH = "/v1/states/k/causal-at"


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
    """Body validation: exactly one ``clock`` object of integer components."""

    def test_empty_and_non_empty_clocks_pass(self) -> None:
        self.assertEqual(parse_causal_at_payload(b'{"clock":{}}'), {})
        self.assertEqual(parse_causal_at_payload({"clock": {}}), {})
        self.assertEqual(
            parse_causal_at_payload(b'{"clock":{"r1":0,"r2":3}}'),
            {"r1": 0, "r2": 3},
        )

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(parse_causal_at_payload(b'  { "clock": { } }\n'), {})

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
            b'"{"clock":{}}"',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        for raw in (
            b"{}",
            b'{"clock":{},"x":1}',
            b'{"x":{}}',
            b'{"clock":{}}\n{"clock":{}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_duplicate_fields_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{},"clock":{"r1":1}}',
            b'{"clock":{"r1":1,"r1":2}}',
            b'{"clock":{"r1":1},"x":1,"x":2}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_clock_must_be_an_object(self) -> None:
        for raw in (
            b'{"clock":null}',
            b'{"clock":[]}',
            b'{"clock":"r1:1"}',
            b'{"clock":1}',
            b'{"clock":true}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_clock_component_names_must_be_non_empty_strings(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_at_payload(b'{"clock":{"":1}}')

    def test_clock_values_must_be_non_negative_integers(self) -> None:
        for raw in (
            b'{"clock":{"r1":-1}}',
            b'{"clock":{"r1":true}}',
            b'{"clock":{"r1":false}}',
            b'{"clock":{"r1":null}}',
            b'{"clock":{"r1":"1"}}',
            b'{"clock":{"r1":[1]}}',
            b'{"clock":{"r1":{}}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_floats_negative_zero_and_non_finite_values_are_rejected(self) -> None:
        for raw in (
            b'{"clock":{"r1":1.0}}',
            b'{"clock":{"r1":1.5}}',
            b'{"clock":{"r1":-0.0}}',
            b'{"clock":{"r1":1e3}}',
            b'{"clock":{"r1":NaN}}',
            b'{"clock":{"r1":Infinity}}',
            b'{"clock":{"r1":-Infinity}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_at_payload(raw)

    def test_large_integer_components_are_accepted(self) -> None:
        self.assertEqual(
            parse_causal_at_payload(b'{"clock":{"r1":1000000000000}}'),
            {"r1": 1_000_000_000_000},
        )


class StateCausalAtStoreTests(unittest.TestCase):
    """Store-level semantics of the causal-boundary replay."""

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

    def test_componentwise_zero_ticks_stay_below_the_empty_boundary(self) -> None:
        # Coverage is a componentwise clock comparison (missing components
        # count as 0): an operation clock carrying a positive tick is never
        # covered by the empty boundary, while an all-zero clock is.
        store = StateStore()
        store.apply_operation("r1", operation("o0", "k", "zero", {"r1": 0}))
        status, payload = store.get_state_causal_at("k", {})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o0", "zero", {"r1": 0})]
        )
        store.apply_operation("r1", operation("o1", "k", "one", {"r1": 1}))
        status, payload = store.get_state_causal_at("k", {})
        self.assertIs(status, HTTPStatus.OK)
        # The positive-tick write dominates the zero-tick candidate only
        # when it is inside the slice; it is not, so the zero candidate is
        # exactly what the empty boundary still shows.
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o0", "zero", {"r1": 0})]
        )

    def test_unknown_key_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_causal_at("absent", {"r1": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_boundary_naming_no_covering_record_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # A component no record ticks never covers the operation.
        self.assertEqual(
            store.get_state_causal_at("k", {"r9": 9}),
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
        self.assertEqual(payload["clock"], {"r1": 2})

    def test_report_shape_and_first_candidate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_state_causal_at("k", {"r1": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "clock": {"r1": 1},
                "key": "k",
                "status": "resolved",
                "candidates": [candidate("r1", "o1", "v", {"r1": 1})],
            },
        )

    def test_boundary_equal_to_an_operation_clock_covers_it(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1, "r2": 2}))
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 2})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "v", {"r1": 1, "r2": 2})])
        # One tick short on any component leaves the operation uncovered.
        self.assertEqual(
            store.get_state_causal_at("k", {"r1": 1, "r2": 1}),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_boundary_selects_concurrent_branches_independently(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        status, payload = store.get_state_causal_at("k", {"r1": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "blue", {"r1": 1})])
        status, payload = store.get_state_causal_at("k", {"r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["candidates"], [candidate("r2", "o2", "red", {"r2": 1})])
        status, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )

    def test_conflict_candidates_sorted_by_identity(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1b", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1a", {"r1": 1}))
        _, payload = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o2")],
        )

    def test_overwriting_writes_delete_dominated_candidates(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "k", "v2", {"r1": 2}))
        _, before = store.get_state_causal_at("k", {"r1": 1})
        self.assertEqual(before["candidates"], [candidate("r1", "o1", "v1", {"r1": 1})])
        _, after = store.get_state_causal_at("k", {"r1": 2})
        self.assertEqual(after["status"], "resolved")
        self.assertEqual(after["candidates"], [candidate("r1", "o2", "v2", {"r1": 2})])

    def test_dominated_record_inside_the_boundary_is_stale_within_the_slice(self) -> None:
        store = StateStore()
        # Committed first, this clock dominates the later r2 write.
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        store.apply_operation("r2", operation("o2", "k", "old", {"r1": 1}))
        # Both clocks sit inside this boundary, but the stale write adds
        # nothing.
        _, payload = store.get_state_causal_at("k", {"r1": 2, "r2": 1})
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "new", {"r1": 2})]
        )
        # The earlier slice contains only the stale record, which is a
        # ordinary candidate there.
        _, earlier = store.get_state_causal_at("k", {"r1": 1})
        self.assertEqual(
            earlier["candidates"], [candidate("r2", "o2", "old", {"r1": 1})]
        )

    def test_filtering_independent_of_global_interleaving(self) -> None:
        # The later-covering record is committed first in the global log;
        # the causal slice must still answer from the clocks alone.
        store = StateStore()
        store.apply_operation(
            "r2", operation("o9", "k", "large", {"r1": 1, "r2": 1})
        )
        store.apply_operation("r1", operation("o1", "k", "small", {"r1": 1}))
        _, only_r1 = store.get_state_causal_at("k", {"r1": 1})
        self.assertEqual(
            only_r1["candidates"], [candidate("r1", "o1", "small", {"r1": 1})]
        )
        _, both = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        # The r2 clock dominates the r1 clock, so the r1 candidate is
        # deleted during the in-order replay.
        self.assertEqual(
            both["candidates"],
            [candidate("r2", "o9", "large", {"r1": 1, "r2": 1})],
        )

    def test_repairs_participate_in_the_slice(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        resolution = {
            "replicaId": "r3",
            "operationId": "fix-1",
            "value": "fixed",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        _, conflicted = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(conflicted["status"], "conflict")
        _, repaired = store.get_state_causal_at(
            "k", {"r1": 1, "r2": 1, "r3": 1}
        )
        self.assertEqual(repaired["status"], "resolved")
        self.assertEqual(
            repaired["candidates"],
            [candidate("r3", "fix-1", "fixed", {"r1": 1, "r2": 1, "r3": 1})],
        )

    def test_sync_imports_participate_in_the_slice(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "local", {"r1": 1}))
        records = [
            ("r2", operation("o1", "k", "imported", {"r2": 1})),
            ("r2", operation("o2", "other", "x", {"r2": 2})),
        ]
        status, accepted, _ = store.import_operations(records)
        self.assertEqual((status, accepted), (HTTPStatus.CREATED, 2))
        _, local = store.get_state_causal_at("k", {"r1": 1})
        self.assertEqual(
            local["candidates"], [candidate("r1", "o1", "local", {"r1": 1})]
        )
        _, merged = store.get_state_causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(merged["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in merged["candidates"]],
            [("r1", "o1"), ("r2", "o1")],
        )

    def test_replays_and_rejections_do_not_move_the_slice(self) -> None:
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
        _, payload = store.get_state_causal_at("k", {"r1": 1})
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "v", {"r1": 1})]
        )

    def test_operations_on_other_keys_never_enter_the_slice(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "other", "x", {"r1": 2}))
        _, payload = store.get_state_causal_at("k", {"r1": 2})
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "v", {"r1": 1})]
        )

    def test_boundary_is_echoed_back_verbatim(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, payload = store.get_state_causal_at("k", {"r1": 1, "r9": 9})
        self.assertEqual(payload["clock"], {"r1": 1, "r9": 9})

    def test_query_reads_nothing_but_the_snapshot(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = store.get_metrics()
        store.get_state_causal_at("k", {})
        store.get_state_causal_at("k", {"r1": 1})
        store.get_state_causal_at("absent", {"r1": 1})
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

    def causal_at(self, key: str, clock: object, query: str = ""):
        return self.request("POST", f"/v1/states/{key}/causal-at{query}", {"clock": clock})

    def causal_at_raw(self, key: str, raw: bytes, query: str = ""):
        return self.request("POST", f"/v1/states/{key}/causal-at{query}", raw)

    def causal_at_raw_path(self, path: str, raw: bytes):
        return self.request("POST", path, raw)

    def test_round_trip_shape_and_encoding(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, payload, headers, raw = self.causal_at(
            "color", {"r1": 1, "r2": 1}
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"clock", "key", "status", "candidates"})
        self.assertEqual(payload["clock"], {"r1": 1, "r2": 1})
        self.assertEqual(payload["key"], "color")
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )
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
        status, _, _, raw = self.causal_at("k", {"r1": 3})
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)

    def test_empty_clock_is_404_for_any_key(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for key in ("k", "absent"):
            status, payload, _, raw = self.causal_at(key, {})
            self.assertEqual(status, 404, key)
            self.assertEqual(payload, {"error": "not_found"}, key)
            self.assertEqual(raw, b'{"error":"not_found"}\n')

    def test_causal_boundaries_walk_the_branches(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload, _, _ = self.causal_at("k", {"r1": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "v1", {"r1": 1})])
        status, payload, _, _ = self.causal_at("k", {"r1": 1, "r2": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(len(payload["candidates"]), 2)

    def test_key_only_past_boundary_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        status, payload, _, _ = self.causal_at("k", {"r1": 1})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_bodies_are_400(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"[]",
            b"null",
            b'{"x":{}}',
            b'{"clock":{},"x":1}',
            b'{"clock":null}',
            b'{"clock":[]}',
            b'{"clock":1}',
            b'{"clock":{"":1}}',
            b'{"clock":{"r1":-1}}',
            b'{"clock":{"r1":true}}',
            b'{"clock":{"r1":"1"}}',
            b'{"clock":{"r1":1.0}}',
            b'{"clock":{"r1":-0.0}}',
            b'{"clock":{"r1":1e3}}',
            b'{"clock":{"r1":NaN}}',
            b'{"clock":{"r1":Infinity}}',
            b'{"clock":{"r1":-Infinity}}',
            b'{"clock":{},"clock":{"r1":1}}',
            b'{"clock":{"r1":1,"r1":2}}',
        ):
            with self.subTest(raw=raw):
                status, payload, _, _ = self.causal_at_raw("k", raw)
                self.assertEqual(status, 400, raw)
                self.assertEqual(payload, {"error": "invalid_request"}, raw)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?x&y="):
            with self.subTest(query=query):
                status, payload, _, _ = self.causal_at("k", {"r1": 1}, query)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)
        # A bare '?' is fine.
        status, _, _, _ = self.causal_at("k", {"r1": 1}, "?")
        self.assertEqual(status, 200)

    def test_bad_query_is_rejected_before_the_body_is_validated(self) -> None:
        status, payload, _, _ = self.causal_at_raw("k", b"{not json", "?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/causal-at/extra",
            "/v1/states/k/causal-at/extra/more",
            "/v1/states/k/causal-at/",
            "/v1/states/causal-at",
            "/v1/causal-at",
            "/v2/states/k/causal-at",
            "/v1/states//causal-at",
        ):
            status, payload, _, _ = self.request("POST", path, {"clock": {"r1": 1}})
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_and_body_errors(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path, raw in (
            ("/v1/states/k/causal-at/extra?x=1", b"{not json"),
            ("/v1/states/k/causal-at/?x=1", b"{not json"),
            ("/v2/states/k/causal-at?x=1", b"{not json"),
        ):
            status, payload, _, _ = self.causal_at_raw_path(path, raw)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_causal_at_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("GET", "/v1/states/k/causal-at")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        status, _, _, _ = self.post_operation(
            "r1", operation("o1", "k/1", "v", {"r1": 1})
        )
        self.assertEqual(status, 201)
        status, payload, _, _ = self.causal_at("k%2F1", {"r1": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        self.causal_at("k", {})
        self.causal_at("k", {"r1": 1})
        self.causal_at("k", {"r1": 1, "r2": 1})
        self.causal_at("absent", {"r1": 1})
        self.causal_at_raw("k", b"{not json")
        self.causal_at("k", {"r1": 1}, "?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_sync, _, _, _ = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_sync, after_sync)


class HttpStateCausalAtAuthTests(unittest.TestCase):
    """The causal-at endpoint authenticates as a read endpoint."""

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
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}},
                auth=auth,
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}},
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_duplicate_bearer_headers_are_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("POST", CAUSAL_AT_PATH)
        conn.putheader("Content-Length", "16")
        conn.putheader("Authorization", "Bearer sekret")
        conn.putheader("Authorization", "Bearer sekret")
        conn.endheaders(b'{"clock":{}}')
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(response.read()), {"error": "unauthorized"})
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}},
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}},
                auth=token,
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["key"], "k", token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CAUSAL_AT_PATH + "?x=1", {"nope": {}},
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")

    def test_rejected_auth_reads_and_changes_nothing(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        before, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.request(
            self.single_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}}
        )
        self.request(
            self.single_port, "POST", CAUSAL_AT_PATH, {"clock": {"r1": 1}},
            auth="Bearer nope",
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class HttpStateCausalAtRequestLimitTests(unittest.TestCase):
    """The causal-at route keeps the shared Content-Length contract."""

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
        conn.putrequest("POST", CAUSAL_AT_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, CAUSAL_AT_PATH, [("Content-Length", value)], b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CAUSAL_AT_PATH,
            [("Content-Length", "2"), ("Content-Length", "3")],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        # The content itself is invalid JSON; the declared size wins.
        status, payload = self.post_raw(
            self.port,
            CAUSAL_AT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_absurdly_long_content_length_digits_are_413(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CAUSAL_AT_PATH,
            [("Content-Length", "9" * 5000)],
            b"{}",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        # Missing declaration: 400 even without a bearer token.
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", CAUSAL_AT_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        # Over-limit declaration: 413, not 401, even with no token.
        status, payload = self.post_raw(
            self.auth_port,
            CAUSAL_AT_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_body_at_exact_limit_is_processed_normally(self) -> None:
        template = b'{"clock":{"":0}}'
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"r" + b"x" * (pad - 1)
        body = b'{"clock":{' + b'"' + name + b'":0}}'
        self.assertEqual(len(body), MAX_BODY_BYTES)
        # No record ticks that component, so the valid at-limit document is
        # processed by the endpoint's normal semantics and answers 404.
        status, payload = self.post_raw(
            self.port, CAUSAL_AT_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        body = b"x" * MAX_BODY_BYTES
        status, payload = self.post_raw(
            self.port, CAUSAL_AT_PATH, [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_rejections_change_no_state(self) -> None:
        post = lambda replica, oid, value, clock: self.request_helper(
            "POST", f"/v1/replicas/{replica}/operations",
            operation(oid, "k", value, clock),
        )
        self.assertEqual(post("r1", "o1", "v1", {"r1": 1})[0], 201)
        self.assertEqual(post("r2", "o2", "v2", {"r2": 1})[0], 201)
        before = self.request_helper("GET", "/v1/states/k")
        for headers, body in (
            ([], b"{}"),
            ([("Content-Length", "abc")], b"{}"),
            ([("Content-Length", str(MAX_BODY_BYTES + 1))], b"junk"),
        ):
            status, _ = self.post_raw(self.port, CAUSAL_AT_PATH, headers, body)
            self.assertIn(status, (400, 413))
        after = self.request_helper("GET", "/v1/states/k")
        self.assertEqual(before, after)

    def request_helper(self, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method, path, body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload


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
                        operation("o2", "k", "v2", {"r2": 1}),
                    ),
                    ("POST", "/v1/states/k/causal-at", {"clock": {"r1": 1}}),
                    (
                        "POST",
                        "/v1/states/k/causal-at",
                        {"clock": {"r1": 1, "r2": 1}},
                    ),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    ("POST", "/v1/states/k/causal-at", {"clock": {}}),
                    ("POST", "/v1/states/k/causal-at", {"clock": {"r1": 1}}),
                    (
                        "POST",
                        "/v1/states/k/causal-at",
                        {"clock": {"r1": 1, "r2": 1}},
                    ),
                ]
            )
            # Same boundaries before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[1], first[2])
            self.assertEqual(second[2], first[3])

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
                    {"clock": {}},
                    {"clock": {"r1": 1}},
                    {"clock": {"r1": 1, "r2": 1}},
                ):
                    status, _ = call("POST", "/v1/states/k/causal-at", body)
                    self.assertIn(status, (200, 404))
                self.assertEqual(data_file.read_bytes(), before)
                self.assertEqual(sorted(os.listdir(tmp)), before_listing)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
