"""Tests for the read-only paged current-state overview.

The overview endpoint is::

    GET /v1/states?afterKey=K&limit=N

It reports every key that currently holds at least one candidate, in
Unicode code-point ascending key order, one page at a time. Each page
entry is exactly what ``GET /v1/states/{key}`` reports for the same key
on the same snapshot — ``{"key","value","clock","status":"resolved"}``
when every candidate agrees on the value, ``{"key","status":"conflict",
"candidates":[...]}`` otherwise — and the response carries only ``keys``,
``cursor`` (the page's last key, null for an empty page), and ``more``
(whether any key follows the page in the same order). ``afterKey`` is an
exclusive lower bound that keeps its meaning even when the boundary key
does not currently exist; ``limit`` defaults to 100 and accepts only an
ASCII decimal integer between 1 and 100.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    parse_states_query,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class StatesStoreTests(unittest.TestCase):
    """Store-level semantics of the paged current-state overview."""

    def build_store(self) -> StateStore:
        # Keys in commit order: "b" (resolved), "a" (conflict), "é"
        # (resolved). Code-point order is a < b < é.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "b", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "a", "x", {"r1": 2}))
        store.apply_operation("r2", operation("o3", "a", "y", {"r2": 1}))
        store.apply_operation("r1", operation("o4", "é", "z", {"r1": 3}))
        return store

    def test_empty_store_reports_an_empty_page(self) -> None:
        store = StateStore()
        status, payload = store.get_states(None, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload, {"keys": [], "cursor": None, "more": False})

    def test_full_page_in_code_point_order(self) -> None:
        store = self.build_store()
        status, payload = store.get_states(None, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(set(payload), {"keys", "cursor", "more"})
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["a", "b", "é"])
        self.assertEqual(payload["cursor"], "é")
        self.assertIs(payload["more"], False)
        self.assertEqual(
            payload["keys"][0],
            {
                "key": "a",
                "status": "conflict",
                "candidates": [
                    {
                        "value": "x",
                        "clock": {"r1": 2},
                        "replicaId": "r1",
                        "operationId": "o2",
                    },
                    {
                        "value": "y",
                        "clock": {"r2": 1},
                        "replicaId": "r2",
                        "operationId": "o3",
                    },
                ],
            },
        )
        self.assertEqual(
            payload["keys"][1],
            {"key": "b", "value": "v1", "clock": {"r1": 1}, "status": "resolved"},
        )
        self.assertEqual(
            payload["keys"][2],
            {"key": "é", "value": "z", "clock": {"r1": 3}, "status": "resolved"},
        )

    def test_entries_match_the_single_key_read(self) -> None:
        store = self.build_store()
        _, payload = store.get_states(None, 100)
        for entry in payload["keys"]:
            single_status, single = store.get_state(entry["key"])
            self.assertIs(single_status, HTTPStatus.OK)
            self.assertEqual(entry, single)

    def test_unicode_code_point_order_not_utf16(self) -> None:
        # U+4E00 (CJK) sorts below U+10000 (astral) in code-point order;
        # a UTF-16 comparison would reverse them.
        store = StateStore()
        store.apply_operation("r1", operation("o1", "\U00010000", "a", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "一", "b", {"r1": 2}))
        _, payload = store.get_states(None, 100)
        self.assertEqual(
            [entry["key"] for entry in payload["keys"]], ["一", "\U00010000"]
        )

    def test_after_key_is_an_exclusive_bound(self) -> None:
        store = self.build_store()
        _, payload = store.get_states("a", 100)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["b", "é"])
        self.assertEqual(payload["cursor"], "é")
        self.assertIs(payload["more"], False)

    def test_after_key_need_not_exist(self) -> None:
        store = self.build_store()
        # "aa" sorts between "a" and "b" but holds no candidate.
        _, payload = store.get_states("aa", 100)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["b", "é"])
        # A boundary past every key yields the stable empty page ("é" is
        # U+00E9, so the boundary must sort above it in code-point order).
        _, tail = store.get_states("ÿ", 100)
        self.assertEqual(tail, {"keys": [], "cursor": None, "more": False})

    def test_limit_pages_the_ordering(self) -> None:
        store = self.build_store()
        _, page1 = store.get_states(None, 2)
        self.assertEqual([entry["key"] for entry in page1["keys"]], ["a", "b"])
        self.assertEqual(page1["cursor"], "b")
        self.assertIs(page1["more"], True)
        _, page2 = store.get_states(page1["cursor"], 2)
        self.assertEqual([entry["key"] for entry in page2["keys"]], ["é"])
        self.assertEqual(page2["cursor"], "é")
        self.assertIs(page2["more"], False)
        _, page3 = store.get_states(page2["cursor"], 2)
        self.assertEqual(page3, {"keys": [], "cursor": None, "more": False})
        # The pages partition the unpaged overview.
        _, whole = store.get_states(None, 100)
        self.assertEqual(page1["keys"] + page2["keys"], whole["keys"])

    def test_limit_one_walks_every_key(self) -> None:
        store = self.build_store()
        seen: list[str] = []
        after = None
        while True:
            _, payload = store.get_states(after, 1)
            seen.extend(entry["key"] for entry in payload["keys"])
            if not payload["more"]:
                break
            after = payload["cursor"]
        self.assertEqual(seen, ["a", "b", "é"])

    def test_query_is_read_only(self) -> None:
        store = self.build_store()
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_snapshot = store.get_replication_snapshot()
        store.get_states(None, 100)
        store.get_states("a", 1)
        store.get_states("zzz", 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_replication_snapshot(), before_snapshot)

    def test_data_file_restart_preserves_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            for args in (
                ("r1", operation("o1", "b", "v1", {"r1": 1})),
                ("r1", operation("o2", "a", "x", {"r1": 2})),
                ("r2", operation("o3", "a", "y", {"r2": 1})),
                ("r1", operation("o4", "é", "z", {"r1": 3})),
            ):
                store.apply_operation(*args)
            expected = store.get_states(None, 100)
            paged = store.get_states("a", 1)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_states(None, 100), expected)
            self.assertEqual(recovered.get_states("a", 1), paged)


class ParseStatesQueryTests(unittest.TestCase):
    """Unit-level query validation for the overview endpoint."""

    def test_empty_query_uses_defaults(self) -> None:
        self.assertEqual(parse_states_query(""), (None, 100))

    def test_limit_only(self) -> None:
        self.assertEqual(parse_states_query("limit=7"), (None, 7))
        self.assertEqual(parse_states_query("limit=1"), (None, 1))
        self.assertEqual(parse_states_query("limit=100"), (None, 100))

    def test_after_key_only(self) -> None:
        self.assertEqual(parse_states_query("afterKey=color"), ("color", 100))

    def test_full_query(self) -> None:
        self.assertEqual(parse_states_query("afterKey=a&limit=2"), ("a", 2))
        self.assertEqual(parse_states_query("limit=2&afterKey=a"), ("a", 2))

    def test_percent_encoded_after_key(self) -> None:
        self.assertEqual(parse_states_query("afterKey=a%2Fb"), ("a/b", 100))
        self.assertEqual(
            parse_states_query("afterKey=%C3%A9"), ("é", 100)
        )

    def test_unknown_parameter(self) -> None:
        self.assertIsNone(parse_states_query("x=1"))
        self.assertIsNone(parse_states_query("afterKey=a&x=1"))
        self.assertIsNone(parse_states_query("after=a"))

    def test_repeated_parameter(self) -> None:
        self.assertIsNone(parse_states_query("afterKey=a&afterKey=b"))
        self.assertIsNone(parse_states_query("limit=1&limit=2"))

    def test_blank_and_valueless_parameters(self) -> None:
        self.assertIsNone(parse_states_query("afterKey="))
        self.assertIsNone(parse_states_query("afterKey"))
        self.assertIsNone(parse_states_query("limit="))
        self.assertIsNone(parse_states_query("limit"))
        self.assertIsNone(parse_states_query("=1"))

    def test_malformed_and_out_of_range_limit(self) -> None:
        for query in (
            "limit=0",
            "limit=101",
            "limit=-1",
            "limit=1.0",
            "limit=%201",
            "limit=%EF%BC%91",  # fullwidth digit one
            "limit=" + "9" * 5000,  # beyond the int() digit conversion cap
        ):
            self.assertIsNone(parse_states_query(query), query)


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

    def overview(self, query: str = ""):
        suffix = f"?{query}" if query else ""
        return self.request("GET", f"/v1/states{suffix}")

    def seed(self) -> None:
        self.post_operation("r1", operation("o1", "b", "v1", {"r1": 1}))
        self.post_operation("r1", operation("o2", "a", "x", {"r1": 2}))
        self.post_operation("r2", operation("o3", "a", "y", {"r2": 1}))
        self.post_operation("r1", operation("o4", "é", "z", {"r1": 3}))

    def test_round_trip(self) -> None:
        self.seed()
        status, payload, headers, raw = self.overview()
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"keys", "cursor", "more"})
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["a", "b", "é"])
        self.assertEqual(payload["cursor"], "é")
        self.assertIs(payload["more"], False)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact JSON terminated by exactly one newline, with the
        # declared length covering the terminator.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n",
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_entries_match_the_single_key_endpoint(self) -> None:
        self.seed()
        status, payload, _, _ = self.overview()
        self.assertEqual(status, 200)
        for entry in payload["keys"]:
            single_status, single, _, _ = self.request(
                "GET", f"/v1/states/{quote(entry['key'])}"
            )
            self.assertEqual(single_status, 200)
            self.assertEqual(entry, single)

    def test_empty_store(self) -> None:
        status, payload, _, raw = self.overview()
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"keys": [], "cursor": None, "more": False})
        self.assertEqual(raw, b'{"cursor":null,"keys":[],"more":false}\n')

    def test_paging_over_http(self) -> None:
        self.seed()
        status, page1, _, _ = self.overview("limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in page1["keys"]], ["a", "b"])
        self.assertEqual(page1["cursor"], "b")
        self.assertIs(page1["more"], True)
        status, page2, _, _ = self.overview("afterKey=b&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in page2["keys"]], ["é"])
        self.assertEqual(page2["cursor"], "é")
        self.assertIs(page2["more"], False)
        status, page3, _, _ = self.overview("afterKey=%C3%A9")
        self.assertEqual(status, 200)
        self.assertEqual(page3, {"keys": [], "cursor": None, "more": False})

    def test_after_key_boundary_key_need_not_exist(self) -> None:
        self.seed()
        status, payload, _, _ = self.overview("afterKey=aa")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["b", "é"])

    def test_bad_query_parameters_are_400(self) -> None:
        self.seed()
        for query in (
            "x=1",  # unknown
            "after=a",  # unknown name
            "afterKey=a&afterKey=b",  # repeated
            "limit=1&limit=2",  # repeated
            "afterKey=",  # blank
            "afterKey",  # valueless
            "limit=",  # blank
            "limit",  # valueless
            "limit=0",
            "limit=101",
            "limit=-1",
            "limit=1.5",
            "limit=%201",
            "limit=%EF%BC%91",
            "limit=" + "9" * 5000,  # beyond the int() digit conversion cap
        ):
            status, payload, _, raw = self.overview(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
            self.assertEqual(raw, b'{"error":"invalid_request"}\n', query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.seed()
        for path in (
            "/v1/states/",
            "/v1/states/overview",
            "/v1/states/overview/extra",
            "/v2/states",
            "/v1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.seed()
        for path in (
            "/v1/states/?limit=0",
            "/v1/states/overview?limit=0",
            "/v2/states?afterKey=",
        ):
            status, payload, _, raw = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
            # A shape 404 is answered by the generic route fallthrough
            # without a trailing newline, like every other unknown route.
            self.assertEqual(raw, b'{"error":"not_found"}', path)

    def test_single_key_route_still_matches(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("GET", "/v1/states/a")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "conflict")
        status, payload, _, _ = self.request("GET", "/v1/states/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.seed()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_overview, _, _, _ = self.overview()
        for query in ("x=1", "limit=0", "afterKey=", "afterKey=a&afterKey=b"):
            self.overview(query)
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_overview, _, _, _ = self.overview()
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_overview, after_overview)

    def test_post_to_overview_route_is_404(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("POST", "/v1/states", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpStatesAuthTests(unittest.TestCase):
    """With auth enabled the overview authenticates like any read GET."""

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
        response_headers = dict(response.getheaders())
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload, response_headers

    def test_overview_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Sekret"):
            status, payload, headers = self.request("GET", "/v1/states", auth=auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(headers.get("WWW-Authenticate"), "Bearer", auth)
        status, payload, _ = self.request("GET", "/v1/states", auth="Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual([entry["key"] for entry in payload["keys"]], ["k"])

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpStatesPersistenceTests(unittest.TestCase):
    """The overview survives a data-file restart unchanged."""

    def test_restart_preserves_overview(self) -> None:
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
                        operation("o1", "b", "v1", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o2", "a", "x", {"r1": 2}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o3", "a", "y", {"r2": 1}),
                    ),
                    ("GET", "/v1/states", None),
                ]
            )
            for status, _ in first[:3]:
                self.assertEqual(status, 201)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/states", None),
                    ("GET", "/v1/states?afterKey=a&limit=1", None),
                    ("GET", "/v1/states?limit=0", None),
                    ("GET", "/v1/states/extra", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[3][1])
            self.assertEqual(second[1][0], 200)
            page = json.loads(second[1][1].decode("utf-8"))
            self.assertEqual([entry["key"] for entry in page["keys"]], ["b"])
            self.assertIs(page["more"], False)
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))
            # A shape 404 is answered by the generic route fallthrough
            # without a trailing newline, like every other unknown route.
            self.assertEqual(second[3], (404, b'{"error":"not_found"}'))
            # The read-only queries created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
