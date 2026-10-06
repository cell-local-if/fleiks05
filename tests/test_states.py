"""Tests for the read-only paginated current-state overview endpoint.

The overview endpoint is::

    GET /v1/states[?afterKey=K&limit=N]

It pages every key that currently holds at least one candidate, ordered
by Unicode code point ascending. Each page entry is exactly what
``GET /v1/states/{key}`` reports for the same key on the same snapshot —
``{"key","value","clock","status":"resolved"}`` or
``{"key","status":"conflict","candidates"}``. The response carries
exactly ``keys``, ``cursor`` (the page's last key, null on an empty
page), and ``more`` (whether any key follows in the same order), all
from one committed snapshot under the commit lock. The query is
strictly read-only and answers with compact UTF-8 JSON terminated by
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
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_states_query,
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


class ParseStatesQueryTests(unittest.TestCase):
    """Query-string validation: optional ``afterKey`` and ``limit``."""

    def test_defaults(self) -> None:
        self.assertEqual(parse_states_query(""), (None, 100))

    def test_valid_parameters(self) -> None:
        self.assertEqual(parse_states_query("limit=1"), (None, 1))
        self.assertEqual(parse_states_query("limit=100"), (None, 100))
        self.assertEqual(parse_states_query("afterKey=k"), ("k", 100))
        self.assertEqual(parse_states_query("afterKey=k&limit=7"), ("k", 7))
        self.assertEqual(parse_states_query("limit=7&afterKey=k"), ("k", 7))
        # The boundary is any non-empty string, percent-decoded as usual.
        self.assertEqual(parse_states_query("afterKey=a%2Fb"), ("a/b", 100))
        self.assertEqual(parse_states_query("afterKey=%E4%B8%AD"), ("中", 100))

    def test_unknown_and_repeated_parameters_are_rejected(self) -> None:
        for query in (
            "x=1",
            "limit=5&x=1",
            "afterKey=k&x=1",
            "limit=5&limit=6",
            "afterKey=a&afterKey=b",
            "after=a",
            "cursor=1",
        ):
            self.assertIsNone(parse_states_query(query), query)

    def test_blank_and_valueless_parameters_are_rejected(self) -> None:
        for query in (
            "afterKey=",
            "afterKey",
            "limit=",
            "limit",
            "afterKey=&limit=5",
            "limit=5&afterKey=",
        ):
            self.assertIsNone(parse_states_query(query), query)

    def test_malformed_and_out_of_range_limits_are_rejected(self) -> None:
        for query in (
            "limit=0",
            "limit=101",
            "limit=-1",
            "limit=+1",
            "limit=1.0",
            "limit=%201",
            "limit=1%20",
            "limit=%EF%BC%91",  # fullwidth digit 1
            "limit=abc",
        ):
            self.assertIsNone(parse_states_query(query), query)


class StatesStoreTests(unittest.TestCase):
    """Store-level semantics of the current-state overview."""

    def test_empty_store_is_an_empty_page(self) -> None:
        store = StateStore()
        self.assertEqual(
            store.get_states(None, 100),
            {"keys": [], "cursor": None, "more": False},
        )

    def test_keys_are_sorted_by_unicode_code_point(self) -> None:
        store = StateStore()
        for key in ("b", "A", "中", "a", "z", "é"):
            store.apply_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        page = store.get_states(None, 100)
        self.assertEqual(
            [entry["key"] for entry in page["keys"]],
            sorted(["b", "A", "中", "a", "z", "é"]),
        )
        self.assertEqual(page["cursor"], page["keys"][-1]["key"])
        self.assertIs(page["more"], False)

    def test_resolved_entries_match_the_single_key_query(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v", {"r2": 1}))
        page = store.get_states(None, 100)
        _, single = store.get_state("k")
        self.assertEqual(page["keys"], [single])
        self.assertEqual(
            page["keys"][0],
            {"key": "k", "value": "v", "clock": {"r1": 1}, "status": "resolved"},
        )

    def test_conflict_entries_match_the_single_key_query(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1b", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1a", {"r1": 1}))
        page = store.get_states(None, 100)
        _, single = store.get_state("k")
        self.assertEqual(page["keys"], [single])
        self.assertEqual(page["keys"][0]["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in page["keys"][0]["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o2")],
        )

    def test_after_key_is_an_exclusive_boundary(self) -> None:
        store = StateStore()
        for key in ("a", "b", "c"):
            store.apply_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        page = store.get_states("a", 100)
        self.assertEqual([entry["key"] for entry in page["keys"]], ["b", "c"])
        self.assertEqual(page["cursor"], "c")
        self.assertIs(page["more"], False)

    def test_after_key_need_not_name_an_existing_key(self) -> None:
        store = StateStore()
        for key in ("a", "c"):
            store.apply_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        page = store.get_states("b", 100)
        self.assertEqual([entry["key"] for entry in page["keys"]], ["c"])
        self.assertEqual(page["cursor"], "c")
        self.assertIs(page["more"], False)

    def test_after_key_past_every_key_is_an_empty_page(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        page = store.get_states("zzz", 100)
        self.assertEqual(page, {"keys": [], "cursor": None, "more": False})

    def test_pages_walk_the_whole_order(self) -> None:
        store = StateStore()
        keys = [f"k{i}" for i in range(5)]
        for key in keys:
            store.apply_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        seen: list[str] = []
        after: str | None = None
        rounds = 0
        while True:
            page = store.get_states(after, 2)
            self.assertLessEqual(len(page["keys"]), 2)
            seen.extend(entry["key"] for entry in page["keys"])
            after = page["cursor"]
            rounds += 1
            if not page["more"]:
                break
        self.assertEqual(seen, keys)
        self.assertEqual(rounds, 3)

    def test_limit_one_reports_more_until_the_last_key(self) -> None:
        store = StateStore()
        for key in ("a", "b"):
            store.apply_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        first = store.get_states(None, 1)
        self.assertEqual([entry["key"] for entry in first["keys"]], ["a"])
        self.assertIs(first["more"], True)
        second = store.get_states(first["cursor"], 1)
        self.assertEqual([entry["key"] for entry in second["keys"]], ["b"])
        self.assertIs(second["more"], False)

    def test_overview_reads_nothing_but_the_snapshot(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = store.get_metrics()
        store.get_states(None, 100)
        store.get_states("k", 1)
        self.assertEqual(store.get_metrics(), before)


class HttpStatesTests(unittest.TestCase):
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

    def get_overview(self, query: str = ""):
        return self.request("GET", f"/v1/states{query}")

    def test_empty_store_round_trip(self) -> None:
        status, payload, headers, raw = self.get_overview()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"keys": [], "cursor": None, "more": False})
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

    def test_entries_match_the_single_key_query(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        self.post_operation("r1", operation("o3", "size", "xl", {"r1": 2}))
        status, payload, _, _ = self.get_overview()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"keys", "cursor", "more"})
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["color", "size"])
        _, conflict, _, _ = self.request("GET", "/v1/states/color")
        _, resolved, _, _ = self.request("GET", "/v1/states/size")
        self.assertEqual(payload["keys"], [conflict, resolved])
        self.assertEqual(payload["keys"][0]["status"], "conflict")
        self.assertEqual(
            payload["keys"][0]["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )
        self.assertEqual(payload["keys"][1], {"key": "size", "value": "xl", "clock": {"r1": 2}, "status": "resolved"})
        self.assertEqual(payload["cursor"], "size")
        self.assertIs(payload["more"], False)

    def test_paging_walks_the_whole_order(self) -> None:
        for i in range(5):
            self.post_operation("r1", operation(f"o{i}", f"k{i}", "v", {"r1": i + 1}))
        seen: list[str] = []
        query = "?limit=2"
        for _ in range(4):
            status, payload, _, _ = self.get_overview(query)
            self.assertEqual(status, 200)
            seen.extend(entry["key"] for entry in payload["keys"])
            if not payload["more"]:
                break
            query = f"?afterKey={payload['cursor']}&limit=2"
        self.assertEqual(seen, [f"k{i}" for i in range(5)])

    def test_after_key_boundary_is_exclusive_and_need_not_exist(self) -> None:
        for key in ("a", "c"):
            self.post_operation("r1", operation(f"o-{key}", key, "v", {"r1": 1}))
        status, payload, _, _ = self.get_overview("?afterKey=b")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["c"])
        self.assertEqual(payload["cursor"], "c")
        self.assertIs(payload["more"], False)
        status, payload, _, _ = self.get_overview("?afterKey=zzz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"keys": [], "cursor": None, "more": False})

    def test_percent_encoded_after_key_boundary(self) -> None:
        self.post_operation("r1", operation("o1", "k/1", "v", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k/2", "v", {"r1": 2}))
        status, payload, _, _ = self.get_overview("?afterKey=k%2F1")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["k/2"])

    def test_invalid_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "?x=1",
            "?limit=5&x=1",
            "?limit=5&limit=6",
            "?afterKey=a&afterKey=b",
            "?afterKey=",
            "?afterKey",
            "?limit=",
            "?limit",
            "?limit=0",
            "?limit=101",
            "?limit=-1",
            "?limit=+1",
            "?limit=1.0",
            "?limit=%201",
            "?limit=1%20",
            "?limit=%EF%BC%91",
            "?limit=abc",
        ):
            status, payload, _, _ = self.get_overview(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_invalid_query_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.get_overview()
        self.get_overview("?limit=0")
        self.get_overview("?afterKey=")
        after, _, _, _ = self.get_overview()
        self.assertEqual(before, after)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/",
            "/v1/states/a/b",
            "/v1/states/a/b/c",
            "/v1/",
            "/v2/states",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/a/b?limit=x",
            "/v2/states?limit=x",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_overview_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.get_overview()
        self.get_overview("?limit=1")
        self.get_overview("?afterKey=k")
        self.get_overview("?limit=x")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)

    def test_post_to_overview_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/states", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpStatesAuthTests(unittest.TestCase):
    """The overview endpoint authenticates like every other non-/health GET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestates-auth-")
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
                self.single_port, "GET", "/v1/states", auth=auth
            )
            self.assertEqual(status, 401, auth)
            # The failure body carries only the error, never page content.
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "GET", "/v1/states", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["k"])

    def test_scope_mode_requires_read_or_admin_scope(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            self.scope_port, "POST", "/v1/replicas/r1/operations", op, auth="Bearer writer"
        )
        self.assertEqual(status, 201)
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "GET", "/v1/states", auth="Bearer writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        # A read token passes.
        status, payload, _ = self.request(
            self.scope_port, "GET", "/v1/states", auth="Bearer reader"
        )
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["k"])

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
        self.request(self.single_port, "GET", "/v1/states")
        self.request(self.single_port, "GET", "/v1/states", auth="Bearer nope")
        after, _, _ = self.request(self.single_port, "GET", "/v1/metrics", auth="Bearer sekret")
        self.assertEqual(before, after)


class HttpStatesPersistenceTests(unittest.TestCase):
    """The recovered state pages identically across a data-file restart."""

    def test_restart_preserves_the_overview(self) -> None:
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
                        operation("o1", "a", "v1", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "b", "v2", {"r2": 1}),
                    ),
                    ("GET", "/v1/states", None),
                    ("GET", "/v1/states?limit=1", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            self.assertEqual(first[3][0], 200)
            size_before = os.path.getsize(data_file)
            second = serve_once(
                [
                    ("GET", "/v1/states", None),
                    ("GET", "/v1/states?limit=1", None),
                    ("GET", "/v1/states?afterKey=a&limit=1", None),
                ]
            )
            # Same state after the restart: identical bytes, including the
            # trailing newline, and the read never touched the data file.
            self.assertEqual(second[0], first[2])
            self.assertEqual(second[1], first[3])
            self.assertEqual(
                second[2],
                (
                    200,
                    b'{"cursor":"b","keys":[{"clock":{"r2":1},"key":"b",'
                    b'"status":"resolved","value":"v2"}],"more":false}\n',
                ),
            )
            self.assertEqual(os.path.getsize(data_file), size_before)


if __name__ == "__main__":
    unittest.main()
