"""Tests for the read-only historical state query endpoint.

The history endpoint is::

    GET /v1/states/{key}/at?cursor=N

It replays the shared accepted-operation log from the empty state up to
``cursor`` records and reports one key's candidate state at that
position: exactly ``cursor``, ``key``, ``status``, and ``candidates``
(always an array, even when resolved). The replay covers every
first-accepted record — ordinary writes, stale writes, accepted repairs,
and sync imports — and nothing else. The query is strictly read-only,
runs against one committed snapshot, and answers with compact UTF-8 JSON
terminated by one newline.

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
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_state_at_query,
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


class ParseStateAtQueryTests(unittest.TestCase):
    """Query-string validation: exactly one required ``cursor`` parameter."""

    def test_valid_cursor(self) -> None:
        self.assertEqual(parse_state_at_query("cursor=0"), 0)
        self.assertEqual(parse_state_at_query("cursor=42"), 42)

    def test_missing_repeated_unknown_and_blank_are_rejected(self) -> None:
        for query in ("", "x=1", "cursor=1&x=2", "cursor=1&cursor=2", "cursor=", "cursor"):
            self.assertIsNone(parse_state_at_query(query), query)

    def test_malformed_values_are_rejected(self) -> None:
        for query in (
            "cursor=-1",
            "cursor=+1",
            "cursor=1.0",
            "cursor= 1",
            "cursor=1 ",
            "cursor=1%20",
            "cursor=%EF%BC%91",  # fullwidth digit 1
            "cursor=abc",
        ):
            self.assertIsNone(parse_state_at_query(query), query)


class StateAtStoreTests(unittest.TestCase):
    """Store-level semantics of the historical replay."""

    def test_cursor_zero_is_always_404(self) -> None:
        store = StateStore()
        self.assertEqual(
            store.get_state_at("k", 0),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_at("k", 0),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_unknown_key_is_404_at_every_position(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_at("absent", 1),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_key_not_yet_present_is_404_even_though_it_appears_later(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "x", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "k", "v", {"r1": 2}))
        self.assertEqual(
            store.get_state_at("k", 1),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["cursor"], 2)

    def test_cursor_past_log_length_raises(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            store.get_state_at("k", 2)

    def test_report_shape_and_first_candidate(self) -> None:
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
                "candidates": [candidate("r1", "o1", "v", {"r1": 1})],
            },
        )

    def test_resolved_history_keeps_every_candidate(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o1", "k", "blue", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        # Unlike the current-state resolved view, all candidates survive.
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o1", "blue", {"r2": 1}),
            ],
        )

    def test_conflict_history_sorted_by_identity(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1b", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1a", {"r1": 1}))
        status, payload = store.get_state_at("k", 3)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o2")],
        )

    def test_first_candidate_matches_current_query_choice(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o1", "k", "v", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, history = store.get_state_at("k", 2)
        _, current = store.get_state("k")
        # The current query chooses the first identity-ordered candidate.
        self.assertEqual(history["candidates"][0]["value"], current["value"])
        self.assertEqual(history["candidates"][0]["clock"], current["clock"])

    def test_stale_writes_replay_without_adding_candidates(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        # Stale: its clock is already dominated, so it is accepted into the
        # log but adds no candidate at its position.
        store.apply_operation("r2", operation("o2", "k", "old", {"r1": 1}))
        status, payload = store.get_state_at("k", 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "new", {"r1": 2})]
        )

    def test_resolution_replays_as_part_of_the_log(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # Position 2: still in conflict.
        _, before = store.get_state_at("k", 2)
        self.assertEqual(before["status"], "conflict")
        resolution = {
            "replicaId": "r1",
            "operationId": "o3",
            "value": "fixed",
            "clock": {"r1": 2, "r2": 2},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, _ = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        # Position 3: the repair is replayed and resolves the key.
        _, after = store.get_state_at("k", 3)
        self.assertEqual(after["status"], "resolved")
        self.assertEqual(
            after["candidates"],
            [candidate("r1", "o3", "fixed", {"r1": 2, "r2": 2})],
        )

    def test_imports_replay_as_part_of_the_log(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "local", {"r1": 1}))
        records = [
            ("r2", operation("o1", "k", "imported", {"r2": 1})),
            ("r2", operation("o2", "other", "x", {"r2": 2})),
        ]
        status, accepted, _ = store.import_operations(records)
        self.assertEqual((status, accepted), (HTTPStatus.CREATED, 2))
        _, payload = store.get_state_at("k", 3)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["candidates"]],
            [("r1", "o1"), ("r2", "o1")],
        )

    def test_replays_and_rejections_do_not_advance_the_log(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # Identical replay: 200, no new log record.
        self.assertIs(store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1})), HTTPStatus.OK)
        # Conflicting identity: 409, no new log record.
        self.assertIs(
            store.apply_operation("r1", operation("o1", "k", "other", {"r1": 1})),
            HTTPStatus.CONFLICT,
        )
        # The log still holds exactly one record.
        with self.assertRaises(ValueError):
            store.get_state_at("k", 2)
        status, payload = store.get_state_at("k", 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["cursor"], 1)

    def test_tail_position_matches_current_query(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, history = store.get_state_at("k", 2)
        _, current = store.get_state("k")
        self.assertEqual(history["status"], current["status"])
        self.assertEqual(history["candidates"], current["candidates"])

    def test_replay_reads_nothing_but_the_snapshot(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = store.get_metrics()
        store.get_state_at("k", 1)
        store.get_state_at("absent", 1)
        self.assertEqual(store.get_metrics(), before)


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

    def get_at(self, key: str, query: str = ""):
        return self.request("GET", f"/v1/states/{key}/at{query}")

    def test_round_trip_shape_and_encoding(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, payload, headers, raw = self.get_at("color", "?cursor=2")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"cursor", "key", "status", "candidates"})
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
        status, _, _, raw = self.get_at("k", "?cursor=1")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"cursor":1', body)
        self.assertIn('"r1":3', body)

    def test_cursor_zero_is_404_for_any_key(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for key in ("k", "absent"):
            status, payload, _, _ = self.get_at(key, "?cursor=0")
            self.assertEqual(status, 404, key)
            self.assertEqual(payload, {"error": "not_found"}, key)

    def test_history_positions_walk_the_log(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "v1", {"r1": 1})])
        status, payload, _, _ = self.get_at("k", "?cursor=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(len(payload["candidates"]), 2)

    def test_resolved_history_keeps_all_candidates(self) -> None:
        self.post_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "blue", {"r2": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(len(payload["candidates"]), 2)
        # The current-state query collapses the same set to one value.
        _, current, _, _ = self.request("GET", "/v1/states/k")
        self.assertNotIn("candidates", current)
        self.assertEqual(current["value"], payload["candidates"][0]["value"])

    def test_cursor_beyond_log_length_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_at("k", "?cursor=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_missing_repeated_unknown_and_blank_cursor_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("", "?x=1", "?cursor=1&x=2", "?cursor=1&cursor=0", "?cursor=", "?cursor"):
            status, payload, _, _ = self.get_at("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_malformed_cursor_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "?cursor=-1",
            "?cursor=+1",
            "?cursor=1.0",
            "?cursor=%201",
            "?cursor=1%20",
            "?cursor=%EF%BC%91",
            "?cursor=abc",
        ):
            status, payload, _, _ = self.get_at("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/at/extra",
            "/v1/states/k/at/extra/more",
            "/v1/states/k/at/",
            "/v1/states/at",
            "/v1/at",
            "/v2/states/k/at",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/at/extra?cursor=x",
            "/v1/states/k/at/?cursor=x",
            "/v2/states/k/at?cursor=x",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_segments_are_percent_decoded(self) -> None:
        status, _, _, _ = self.post_operation("r1", operation("o1", "k/1", "v", {"r1": 1}))
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_at("k%2F1", "?cursor=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.get_at("k", "?cursor=1")
        self.get_at("k", "?cursor=2")
        self.get_at("absent", "?cursor=2")
        self.get_at("k", "?cursor=x")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)

    def test_post_to_at_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/states/k/at", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpStateAtAuthTests(unittest.TestCase):
    """The history endpoint authenticates like every other non-/health GET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-at-auth-")
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
            json.dump({"reader": ["read"], "writer": ["write"]}, handle)
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

    def request(self, port: int, method: str, path: str, body: object = None, auth: str | None = None):
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

    def test_single_token_mode_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            self.single_port, "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload, challenge = self.request(
                self.single_port, "GET", "/v1/states/k/at?cursor=1", auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "GET", "/v1/states/k/at?cursor=1", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_scope_mode_requires_read_or_admin_scope(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            self.scope_port, "POST", "/v1/replicas/r1/operations", op, auth="Bearer writer"
        )
        self.assertEqual(status, 201)
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "GET", "/v1/states/k/at?cursor=1", auth="Bearer writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        # A read token passes.
        status, payload, _ = self.request(
            self.scope_port, "GET", "/v1/states/k/at?cursor=1", auth="Bearer reader"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")

    def test_rejected_auth_reads_and_changes_nothing(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.request(
            self.single_port, "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        before, _, _ = self.request(self.single_port, "GET", "/v1/metrics", auth="Bearer sekret")
        self.request(self.single_port, "GET", "/v1/states/k/at?cursor=1")
        self.request(self.single_port, "GET", "/v1/states/k/at?cursor=1", auth="Bearer nope")
        after, _, _ = self.request(self.single_port, "GET", "/v1/metrics", auth="Bearer sekret")
        self.assertEqual(before, after)


class HttpStateAtPersistenceTests(unittest.TestCase):
    """The same cursor answers identically across a data-file restart."""

    def test_restart_preserves_history(self) -> None:
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
                    ("GET", "/v1/states/k/at?cursor=1", None),
                    ("GET", "/v1/states/k/at?cursor=2", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/states/k/at?cursor=0", None),
                    ("GET", "/v1/states/k/at?cursor=1", None),
                    ("GET", "/v1/states/k/at?cursor=2", None),
                    ("GET", "/v1/states/k/at?cursor=3", None),
                ]
            )
            # Same cursors before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[1], first[2])
            self.assertEqual(second[2], first[3])
            self.assertEqual(second[3], (400, b'{"error":"invalid_request"}\n'))


if __name__ == "__main__":
    unittest.main()
