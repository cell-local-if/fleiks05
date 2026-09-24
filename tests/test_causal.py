"""Tests for the read-only per-operation causal ancestry endpoint.

The causal endpoint is::

    GET /v1/causal/{replicaId}/{operationId}?after=N&limit=N

It locates one first-accepted operation by identity and reports its strict
causal predecessors — the first-accepted records committed before it whose
clocks are strictly less than the source clock — in the shared log's global
commit order. Each ancestor preserves the archive record content plus a
``relation`` of ``direct`` (dominated by no other strict predecessor) or
``transitive``. The response carries the source operation in the archive
shape, one page of ancestors, the resume cursor, and the more flag as
compact canonical JSON terminated by one newline.

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

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
    }


def ancestor(
    replica_id: str, operation_id: str, key: str, value: str, clock: dict, relation: str
) -> dict:
    return {**record(replica_id, operation_id, key, value, clock), "relation": relation}


class CausalStoreTests(unittest.TestCase):
    """Store-level semantics of the causal ancestry snapshot."""

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_causal_ancestors("r1", "nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_ancestors("nope", "o1", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_no_predecessors_reports_empty_page(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_causal_ancestors("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["operation"], record("r1", "o1", "k", "v", {"r1": 1})
        )
        self.assertEqual(payload["ancestors"], [])
        self.assertEqual(payload["cursor"], 0)
        self.assertIs(payload["more"], False)

    def test_strict_predecessors_are_reported_in_commit_order(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "v2", {"r1": 1, "r2": 1}))
        # Concurrent with the source clock: not a strict predecessor.
        store.apply_operation("r3", operation("o3", "c", "v3", {"r3": 1}))
        store.apply_operation(
            "r1", operation("o4", "d", "v4", {"r1": 2, "r2": 1})
        )
        status, payload = store.get_causal_ancestors("r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["ancestors"],
            [
                ancestor("r1", "o1", "a", "v1", {"r1": 1}, "transitive"),
                ancestor("r2", "o2", "b", "v2", {"r1": 1, "r2": 1}, "direct"),
            ],
        )
        self.assertEqual(payload["cursor"], 2)
        self.assertIs(payload["more"], False)

    def test_records_committed_after_the_source_are_excluded(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 2}))
        # Committed after the source and dominated by it: never an ancestor.
        store.apply_operation("r2", operation("o2", "b", "v2", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_ancestors("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["ancestors"], [])

    def test_equal_clock_is_not_strictly_less(self) -> None:
        store = StateStore()
        # A different identity carrying a clock equal to the source's.
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1, "r2": 1}))
        store.apply_operation("r2", operation("o2", "b", "v2", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_ancestors("r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["ancestors"], [])

    def test_missing_clock_components_count_as_zero(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        # {"r1": 1} is strictly less than {"r1": 1, "r2": 1}.
        store.apply_operation("r2", operation("o2", "b", "v2", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_ancestors("r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["ancestors"],
            [ancestor("r1", "o1", "a", "v1", {"r1": 1}, "direct")],
        )

    def test_dominated_predecessor_is_transitive(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "c", "v3", {"r1": 3}))
        status, payload = store.get_causal_ancestors("r1", "o3", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["ancestors"]],
            [("o1", "transitive"), ("o2", "direct")],
        )

    def test_concurrent_predecessors_are_both_direct(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "v2", {"r2": 1}))
        store.apply_operation(
            "r3", operation("o3", "c", "v3", {"r1": 1, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_ancestors("r3", "o3", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["ancestors"]],
            [("o1", "direct"), ("o2", "direct")],
        )

    def test_stale_writes_and_repairs_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # Stale write: dominated by the o1 candidate, adds no candidate, but
        # is a first-accepted record and therefore a potential ancestor.
        store.apply_operation("r2", operation("o3", "k", "v3", {"r2": 1}))
        # An accepted repair of the conflict.
        store.apply_operation(
            "r3", operation("fix", "k", "merged", {"r1": 1, "r2": 1, "r3": 1})
        )
        # The source observes every earlier clock.
        store.apply_operation(
            "r1", operation("o4", "other", "x", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_ancestors("r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["ancestors"]],
            [
                ("o1", "transitive"),
                ("o2", "transitive"),
                ("o3", "transitive"),
                ("fix", "direct"),
            ],
        )

    def test_replays_and_rejected_requests_never_appear(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        # Identical replay: no new record.
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        # Conflicting identity: rejected, no record.
        store.apply_operation("r1", operation("o1", "a", "other", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        status, payload = store.get_causal_ancestors("r1", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["ancestors"]], ["o1"]
        )

    def test_paging_boundaries(self) -> None:
        store = StateStore()
        for index in range(3):
            store.apply_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        store.apply_operation("r1", operation("src", "k", "v", {"r1": 4}))
        status, page1 = store.get_causal_ancestors("r1", "src", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page1["ancestors"]), 2)
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2 = store.get_causal_ancestors("r1", "src", 2, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page2["ancestors"]), 1)
        self.assertEqual(page2["cursor"], 3)
        self.assertIs(page2["more"], False)
        # after equal to the total is a valid empty tail.
        status, tail = store.get_causal_ancestors("r1", "src", 3, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["ancestors"], [])
        self.assertEqual(tail["cursor"], 3)
        self.assertIs(tail["more"], False)
        # The two pages partition the unpaged list in commit order.
        _, whole = store.get_causal_ancestors("r1", "src", 0, 100)
        self.assertEqual(page1["ancestors"] + page2["ancestors"], whole["ancestors"])

    def test_after_past_the_ancestor_count_is_rejected(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        with self.assertRaises(ValueError):
            store.get_causal_ancestors("r1", "o2", 2, 100)

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("a")
        before_state = store.get_state("a")
        before_sync = store.get_sync_operations(0, 100)
        store.get_causal_ancestors("r1", "o2", 0, 100)
        store.get_causal_ancestors("absent", "o2", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("a"), before_audit)
        self.assertEqual(store.get_state("a"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)

    def test_data_file_restart_preserves_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
            store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
            expected = store.get_causal_ancestors("r1", "o2", 0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_causal_ancestors("r1", "o2", 0, 100), expected)
            self.assertEqual(
                recovered.get_causal_ancestors("r1", "absent", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpCausalTests(unittest.TestCase):
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

    def get_causal(self, replica: str, operation_id: str, query: str = ""):
        return self.request("GET", f"/v1/causal/{replica}/{operation_id}{query}")

    def test_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1}))
        status, payload, headers, raw = self.get_causal("r2", "o2")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"operation", "ancestors", "cursor", "more"})
        self.assertEqual(
            payload["operation"],
            record("r2", "o2", "size", "large", {"r1": 1, "r2": 1}),
        )
        self.assertEqual(
            payload["ancestors"],
            [ancestor("r1", "o1", "color", "blue", {"r1": 1}, "direct")],
        )
        self.assertEqual(payload["cursor"], 1)
        self.assertIs(payload["more"], False)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact canonical JSON terminated by exactly one newline, with
        # the declared length covering the terminator.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n",
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_non_ascii_strings_are_written_literally(self) -> None:
        self.post_operation("r1", operation("o1", "clé", "bléu", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k", "v", {"r1": 2}))
        status, _, _, raw = self.get_causal("r1", "o2")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 3, "r2": 2}))
        status, _, _, raw = self.get_causal("r2", "o2")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"r2":2', body)
        self.assertIn('"cursor":1', body)

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in ("/v1/causal/r1/nope", "/v1/causal/nope/o1", "/v1/causal/nope/nope"):
            status, payload, _, raw = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
            self.assertEqual(raw, b'{"error":"not_found"}\n', path)

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("o/1", "k", "v", {"r1": 1})
        status, _, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_causal("r1", "o%2F1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operation"]["operation"]["operationId"], "o/1")

    def test_paging_over_http(self) -> None:
        for index in range(3):
            self.post_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        self.post_operation("r1", operation("src", "k", "v", {"r1": 4}))
        status, page1, _, _ = self.get_causal("r1", "src", "?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["ancestors"]), 2)
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2, _, _ = self.get_causal("r1", "src", "?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page2["ancestors"]), 1)
        self.assertEqual(page2["cursor"], 3)
        self.assertIs(page2["more"], False)
        status, tail, _, _ = self.get_causal("r1", "src", "?after=3")
        self.assertEqual(status, 200)
        self.assertEqual(tail["ancestors"], [])
        self.assertIs(tail["more"], False)

    def test_bad_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "?x=1",
            "?after=1&after=2",
            "?after=",
            "?after",
            "?after=-1",
            "?after=1.0",
            "?after=%201",  # whitespace
            "?after=%EF%BC%91",  # non-ASCII digit
            "?after=1",  # past the ancestor count of 0
            "?limit=0",
            "?limit=101",
            "?limit=",
            "?limit=-1",
            "?limit=1.5",
        ):
            status, payload, _, _ = self.get_causal("r1", "o1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/r1/o1/extra",
            "/v1/causal/r1",
            "/v1/causal//o1",
            "/v1/causal/r1/",
            "/v1/causal",
            "/v2/causal/r1/o1",
            "/v1/causal/r1/o1/",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/r1/o1/extra?x=1",
            "/v1/causal//o1?x=1",
            "/v2/causal/r1/o1?x=1",
            "/v1/causal/r1/o1/?after=0",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r1", operation("o2", "a", "x", {"r1": 2}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_causal, _, _, _ = self.get_causal("r1", "o2")
        self.get_causal("r1", "o2", "?x=1")
        self.get_causal("r1", "o2", "?after=9")
        self.get_causal("r1", "absent", "?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_causal, _, _, _ = self.get_causal("r1", "o2")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_causal, after_causal)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_causal("r1", "o1")
        self.get_causal("r1", "absent")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_causal_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/causal/r1/o1", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpCausalAuthTests(unittest.TestCase):
    """With auth enabled the causal endpoint authenticates like any GET."""

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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_causal_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request("GET", "/v1/causal/r1/o1", auth=auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request("GET", "/v1/causal/r1/o1", auth="Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operation"]["replicaId"], "r1")
        self.assertEqual(payload["ancestors"], [])

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalPersistenceTests(unittest.TestCase):
    """The causal ancestry report survives a data-file restart unchanged."""

    def test_restart_preserves_ancestors(self) -> None:
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
                        operation("o1", "k", "v", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "a", "x", {"r1": 1, "r2": 1}),
                    ),
                    ("GET", "/v1/causal/r2/o2", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/causal/r2/o2", None),
                    ("GET", "/v1/causal/r2/absent", None),
                    ("GET", "/v1/causal/r2/o2?x=1", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[2][1])
            self.assertEqual(second[1], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))
            # The read-only query created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
