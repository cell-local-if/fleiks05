"""Tests for the read-only historical state endpoint.

The historical state endpoint is::

    GET /v1/states/{key}/at?cursor=N

It replays the first ``cursor`` first-accepted operations from the empty
state in commit order and reports one key's candidate state as it stood at
that position. ``cursor`` is required, counted from zero (zero replays
nothing and every key is 404), and must not exceed the current log length.
The response carries exactly cursor, key, status, and candidates (always
an array, even when resolved), as compact ordered UTF-8 JSON terminated
by one newline. The query is strictly read-only, shares the commit lock
with writes/imports/repairs/checkpoints, and the same cursor yields the
same state after a data-file restart.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


class StateAtStoreTests(unittest.TestCase):
    """Store-level semantics of the historical replay."""

    def test_cursor_zero_is_404_for_any_key(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for key in ("k", "absent", ""):
            self.assertEqual(
                store.get_state_at(key, 0),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
                key,
            )

    def test_empty_store_only_accepts_cursor_zero(self) -> None:
        store = StateStore()
        self.assertEqual(
            store.get_state_at("k", 0),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        with self.assertRaises(ValueError):
            store.get_state_at("k", 1)

    def test_cursor_past_log_length_raises(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            store.get_state_at("k", 2)

    def test_single_write_resolved_keeps_candidate_array(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_state_at("k", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "cursor": 1,
                "key": "k",
                "status": "resolved",
                # Resolved history still carries every candidate; it never
                # copies the current resolved response's value/clock shape.
                "candidates": [candidate("r1", "o1", "v", {"r1": 1})],
            },
        )

    def test_replay_advances_position_by_position(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        _, at_one = store.get_state_at("k", 1)
        self.assertEqual(at_one["status"], "resolved")
        self.assertEqual(
            at_one["candidates"], [candidate("r1", "o1", "blue", {"r1": 1})]
        )
        _, at_two = store.get_state_at("k", 2)
        self.assertEqual(at_two["status"], "conflict")
        self.assertEqual(
            at_two["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )

    def test_key_absent_then_present_is_404_before_it_appears(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "later", "v", {"r2": 1}))
        # The target has no candidate at this position even though it shows
        # up later in the log.
        self.assertEqual(
            store.get_state_at("later", 1),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        status, payload = store.get_state_at("later", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r2", "o2", "v", {"r2": 1})]
        )

    def test_stale_write_is_replayed_but_adds_no_candidate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        # Already dominated: recorded as an accepted log entry but adds no
        # candidate.
        stale_status = store.apply_operation(
            "r2", operation("o9", "k", "old", {"r1": 1})
        )
        self.assertIs(stale_status, HTTPStatus.CREATED)
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "new", {"r1": 2})]
        )
        # The cursor still counts the stale record: position 1 has the first
        # candidate, position 2 leaves it unchanged.
        _, before = store.get_state_at("k", 1)
        self.assertEqual(before["candidates"], payload["candidates"])
        self.assertEqual(before["cursor"], 1)
        self.assertEqual(payload["cursor"], 2)

    def test_dominated_candidate_disappears_from_later_history(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        store.apply_operation(
            "r3",
            operation("o3", "k", "merged", {"r1": 1, "r2": 1, "r3": 1}),
        )
        _, conflict = store.get_state_at("k", 2)
        self.assertEqual(conflict["status"], "conflict")
        status, resolved = store.get_state_at("k", 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(
            resolved["candidates"],
            [candidate("r3", "o3", "merged", {"r1": 1, "r2": 1, "r3": 1})],
        )

    def test_resolved_with_multiple_same_value_candidates(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "blue", {"r2": 1}))
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "blue", {"r2": 1}),
            ],
        )

    def test_tail_matches_current_query_classification_and_selection(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r9", operation("o9", "other", "x", {"r9": 1}))
        tail_cursor = len(store.get_sync_operations(0, 100)[0])
        status, historical = store.get_state_at("k", tail_cursor)
        self.assertIs(status, HTTPStatus.OK)
        current_status, current = store.get_state("k")
        self.assertIs(current_status, HTTPStatus.OK)
        self.assertEqual(historical["status"], "conflict")
        self.assertEqual(current["status"], "conflict")
        self.assertEqual(historical["candidates"], current["candidates"])

        # After a repair the current query is resolved: the first historical
        # candidate is exactly the selected value and clock.
        resolution = {
            "replicaId": "r3",
            "operationId": "fix",
            "value": "blue",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        tail_cursor = len(store.get_sync_operations(0, 100)[0])
        status, historical = store.get_state_at("k", tail_cursor)
        self.assertIs(status, HTTPStatus.OK)
        current_status, current = store.get_state("k")
        self.assertIs(current_status, HTTPStatus.OK)
        self.assertEqual(historical["status"], current["status"])
        self.assertEqual(historical["candidates"][0]["value"], current["value"])
        self.assertEqual(historical["candidates"][0]["clock"], current["clock"])

    def test_imports_and_repairs_are_part_of_the_replay(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        status, _, _ = store.import_operations(
            [("r2", operation("o2", "k", "red", {"r2": 1}))]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        resolution = {
            "replicaId": "r3",
            "operationId": "fix",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        # Cursor 3 includes the write, the import, and the repair.
        status, payload = store.get_state_at("k", 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [candidate("r3", "fix", "merged", {"r1": 1, "r2": 1, "r3": 1})],
        )
        # Cursor 2 stops right after the import: conflict with both values.
        _, imported = store.get_state_at("k", 2)
        self.assertEqual(imported["status"], "conflict")
        self.assertEqual([c["value"] for c in imported["candidates"]], ["blue", "red"])

    def test_replays_do_not_extend_the_log(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # Identical replay: 200, no new record.
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1})),
            HTTPStatus.OK,
        )
        self.assertEqual(len(store.get_sync_operations(0, 100)[0]), 1)
        status, payload = store.get_state_at("k", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["cursor"], 1)

    def test_other_keys_are_isolated_in_the_replay(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "other", "x", {"r2": 1}))
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "v", {"r1": 1})]
        )
        self.assertEqual(
            store.get_state_at("other", 1),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_state = store.get_state("k")
        for cursor in range(3):
            store.get_state_at("k", cursor)
        store.get_state_at("absent", 2)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_state("k"), before_state)

    def test_data_file_restart_same_cursor_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "k", "red", {"r2": 1}))
            expected = [(c, store.get_state_at("k", c)) for c in range(3)]
            recovered = StateStore(data_file=data_file)
            for cursor, result in expected:
                self.assertEqual(recovered.get_state_at("k", cursor), result)
            with self.assertRaises(ValueError):
                recovered.get_state_at("k", 3)


class HttpStateAtTests(unittest.TestCase):
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

    def get_at(self, key: str, query: str):
        return self.request("GET", f"/v1/states/{key}/at{query}")

    def test_conflict_history_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, payload, headers, raw = self.get_at("color", "?cursor=2")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ["cursor", "key", "status", "candidates"])
        self.assertEqual(payload["cursor"], 2)
        self.assertEqual(payload["key"], "color")
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )
        self.assertIsInstance(payload["candidates"], list)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact JSON with the fixed field order, terminated by exactly one
        # newline covered by the declared length.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertTrue(raw.startswith(b'{"cursor":2,"key":"color"'))
        self.assertEqual(
            raw,
            b'{"cursor":2,"key":"color","status":"conflict","candidates":'
            b'[{"value":"blue","clock":{"r1":1},"replicaId":"r1","operationId":"o1"},'
            b'{"value":"red","clock":{"r2":1},"replicaId":"r2","operationId":"o2"}]}\n',
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_resolved_history_keeps_all_candidates(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "blue", {"r2": 1}))
        status, payload, _, _ = self.get_at("color", "?cursor=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertNotIn("value", payload)
        self.assertNotIn("clock", payload)

    def test_cursor_zero_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, raw = self.get_at("k", "?cursor=0")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        self.assertEqual(raw, b'{"error":"not_found"}\n')

    def test_unknown_key_at_tail_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("absent", "?cursor=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_tail_matches_current_query(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, historical, _, _ = self.get_at("color", "?cursor=2")
        self.assertEqual(status, 200)
        status, current, _, _ = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(historical["status"], current["status"])
        self.assertEqual(historical["candidates"], current["candidates"])

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 3}))
        status, _, _, raw = self.get_at("color", "?cursor=1")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)

    def test_path_segments_are_percent_decoded(self) -> None:
        self.post_operation("r1", operation("o1", "k/1", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("k%2F1", "?cursor=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_missing_cursor_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("", "?"):
            status, payload, _, _ = self.get_at("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_invalid_cursor_shapes_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "?cursor=",
            "?cursor=-1",
            "?cursor=+1",
            "?cursor=1.0",
            "?cursor=.5",
            "?cursor=1%20",
            "?cursor=%201",
            "?cursor=1+",
            "?cursor=0x1",
            "?cursor=1e2",
            "?cursor=*",
            "?cursor=%E0%A9%B5",  # U+0A75 non-ASCII digit
            "?cursor=%EF%BC%91",  # fullwidth digit one
            "?cursor=1&cursor=2",
            "?cursor=1&x=2",
            "?x=1",
            "?cursor",
            "?=1",
        ):
            status, payload, _, _ = self.get_at("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_leading_zero_is_numeric_like_other_endpoints(self) -> None:
        # The existing query parsers accept ASCII decimal runs with leading
        # zeros by their numeric value; the historical cursor keeps the same
        # contract rather than introducing a stricter one.
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=01")
        self.assertEqual(status, 200)
        self.assertEqual(payload["cursor"], 1)
        status, _, _, _ = self.get_at("k", "?cursor=00")
        self.assertEqual(status, 404)

    def test_cursor_past_log_length_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_cursor_equal_log_length_is_tail(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["cursor"], 1)

    def test_route_shape_failures_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/at/extra",
            "/v1/states/k/at/extra/more",
            "/v1/states//at?cursor=1",
            "/v1/at?cursor=1",
            "/v2/states/k/at?cursor=1",
            "/v1/states/k/at/?cursor=1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_beats_query_shape(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/at/extra?cursor=1",
            "/v1/states//at?cursor=1",
            "/v1/states/k/at/?cursor=1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_unknown_route_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", "/v1/states/k/when?cursor=1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_to_at_route_is_404(self) -> None:
        status, payload, _, _ = self.request("POST", "/v1/states/k/at", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_tail, _, _, _ = self.get_at("k", "?cursor=2")
        for query in ("", "?cursor=9", "?cursor=x", "?x=1", "?cursor=1&cursor=2"):
            self.get_at("k", query)
        self.get_at("absent", "?cursor=x")
        self.request("GET", "/v1/states/k/at/extra?cursor=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_tail, _, _, _ = self.get_at("k", "?cursor=2")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_tail, after_tail)


class HttpStateAtAuthTests(unittest.TestCase):
    """With auth enabled the historical endpoint authenticates like any GET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
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

    def request(self, method: str, path: str, body: object = None, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_at_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload = self.request(
                "GET", "/v1/states/k/at?cursor=1", auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request(
            "GET", "/v1/states/k/at?cursor=1", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_bad_auth_beats_bad_query_and_shape(self) -> None:
        for path in (
            "/v1/states/k/at",
            "/v1/states/k/at?cursor=x",
            "/v1/states/k/at/extra?cursor=1",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 401, path)
            self.assertEqual(payload, {"error": "unauthorized"}, path)

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpStateAtScopeTests(unittest.TestCase):
    """Scope-policy mode: the endpoint needs the read scope (or admin)."""

    POLICY = {
        "reader-token": frozenset({"read"}),
        "writer-token": frozenset({"write"}),
        "admin-token": frozenset({"read", "write", "admin"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(cls.POLICY)
        )
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

    def request(self, path: str, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        www_auth = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, www_auth

    def seed(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/v1/replicas/r1/operations",
            body=json.dumps(operation("o1", "k", "v", {"r1": 1})),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer admin-token",
            },
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()

    def test_read_and_admin_scopes_allowed(self) -> None:
        self.seed()
        for auth in ("Bearer reader-token", "Bearer admin-token"):
            status, payload, _ = self.request("/v1/states/k/at?cursor=1", auth)
            self.assertEqual(status, 200, auth)
            self.assertEqual(payload["key"], "k")

    def test_write_scope_is_403_without_challenge(self) -> None:
        self.seed()
        status, payload, www_auth = self.request(
            "/v1/states/k/at?cursor=1", "Bearer writer-token"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(www_auth)

    def test_scope_decision_beats_query_validation(self) -> None:
        self.seed()
        status, payload, _ = self.request(
            "/v1/states/k/at?cursor=nope", "Bearer writer-token"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_missing_token_is_401(self) -> None:
        self.seed()
        status, payload, www_auth = self.request("/v1/states/k/at?cursor=1")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(www_auth, "Bearer")


class HttpStateAtPersistenceTests(unittest.TestCase):
    """The same cursor returns identical history across a data-file restart."""

    def test_restart_preserves_historical_states(self) -> None:
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

            writes = [
                (
                    "POST",
                    "/v1/replicas/r1/operations",
                    operation("o1", "k", "blue", {"r1": 1}),
                ),
                (
                    "POST",
                    "/v1/replicas/r2/operations",
                    operation("o2", "k", "red", {"r2": 1}),
                ),
            ]
            first = serve_once(
                writes
                + [
                    ("GET", "/v1/states/k/at?cursor=0", None),
                    ("GET", "/v1/states/k/at?cursor=1", None),
                    ("GET", "/v1/states/k/at?cursor=2", None),
                    ("GET", "/v1/states/k/at?cursor=3", None),
                ]
            )
            self.assertEqual([r[0] for r in first], [201, 201, 404, 200, 200, 400])
            second = serve_once(
                [
                    ("GET", "/v1/states/k/at?cursor=0", None),
                    ("GET", "/v1/states/k/at?cursor=1", None),
                    ("GET", "/v1/states/k/at?cursor=2", None),
                    ("GET", "/v1/states/k/at?cursor=3", None),
                ]
            )
            self.assertEqual(second, first[2:])
            # Identical bytes, including the single trailing newline.
            self.assertEqual(second[1][1], first[3][1])
            self.assertTrue(second[1][1].endswith(b"\n"))
            self.assertFalse(second[1][1].endswith(b"\n\n"))


if __name__ == "__main__":
    unittest.main()
