"""Tests for the read-only cross-key causal impact endpoint.

The impact endpoint is::

    GET /v1/states/{key}/impact?after=N&limit=N

It reads one key's current candidates and reports the accepted operations
on *other* keys whose clocks dominate those candidates — the operations
causally later than the key's current state — in the shared log's global
commit order. The response carries the key, the conflict status, the basis
candidates, one page of impact records, the resume cursor, and the
has-more flag as compact canonical JSON terminated by one newline.

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


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
    }


class StateImpactStoreTests(unittest.TestCase):
    """Store-level semantics of the impact snapshot."""

    def test_unknown_key_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_impact("nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_no_impacts_reports_empty_page(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # A concurrent operation on another key does not dominate the
        # candidate, so it is not an impact.
        store.apply_operation("r2", operation("o2", "other", "x", {"r2": 1}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "v", {"r1": 1})])
        self.assertEqual(payload["impacts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)

    def test_cross_key_dominating_operations_are_impacts(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        # Causally later than the candidate of k: dominates {"r1": 2}.
        store.apply_operation("r2", operation("o2", "a", "x", {"r1": 2, "r2": 1}))
        # Concurrent with the candidate: not an impact.
        store.apply_operation("r3", operation("o3", "b", "y", {"r3": 1}))
        # Causally earlier: dominated by the candidate, not an impact.
        store.apply_operation("r1", operation("o5", "c", "z", {"r1": 1}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["impacts"],
            [record("r2", "o2", "a", "x", {"r1": 2, "r2": 1})],
        )
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["nextCursor"], 1)

    def test_impacts_follow_global_commit_order(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "y", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "a", "x", {"r1": 1, "r3": 1}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["impacts"]],
            [("r2", "o2"), ("r3", "o3")],
        )

    def test_own_key_operations_never_impact(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # A later write on the same key dominates the old candidate but
        # belongs to the target key, so it is never an impact.
        store.apply_operation("r1", operation("o2", "k", "v2", {"r1": 2}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["impacts"], [])

    def test_stale_write_on_other_key_can_impact(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "other", "x", {"r1": 2, "r2": 5}))
        # Stale on its own key (dominated by {"r1": 2, "r2": 5}, so it adds
        # no candidate there) but causally later than the candidate of k:
        # it is an accepted operation, so it is an impact.
        store.apply_operation("r2", operation("o3", "other", "y", {"r1": 1, "r2": 2}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["impacts"],
            [
                record("r2", "o2", "other", "x", {"r1": 2, "r2": 5}),
                record("r2", "o3", "other", "y", {"r1": 1, "r2": 2}),
            ],
        )

    def test_status_reports_conflict_when_values_disagree(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "v1", {"r1": 1}),
                candidate("r2", "o2", "v2", {"r2": 1}),
            ],
        )

    def test_impact_dominating_any_candidate_counts(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # Dominates only the r1 candidate: still an impact.
        store.apply_operation("r3", operation("o3", "other", "x", {"r1": 1, "r3": 1}))
        status, payload = store.get_state_impact("k", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["impacts"],
            [record("r3", "o3", "other", "x", {"r1": 1, "r3": 1})],
        )

    def test_paging_boundaries(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for index in range(3):
            store.apply_operation(
                "r2", operation(f"o{index}", f"k{index}", "x", {"r1": 1, "r2": index + 1})
            )
        status, page1 = store.get_state_impact("k", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page1["impacts"]), 2)
        self.assertEqual(page1["nextCursor"], 2)
        self.assertIs(page1["hasMore"], True)
        status, page2 = store.get_state_impact("k", page1["nextCursor"], 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page2["impacts"]), 1)
        self.assertEqual(page2["nextCursor"], 3)
        self.assertIs(page2["hasMore"], False)
        # after equal to the total is a valid empty tail.
        status, tail = store.get_state_impact("k", 3, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["impacts"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)
        # The two pages partition the unpaged list in commit order.
        _, whole = store.get_state_impact("k", 0, 100)
        self.assertEqual(page1["impacts"] + page2["impacts"], whole["impacts"])

    def test_after_past_the_impact_count_is_rejected(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        with self.assertRaises(ValueError):
            store.get_state_impact("k", 2, 100)

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("k")
        before_state = store.get_state("k")
        before_sync = store.get_sync_operations(0, 100)
        store.get_state_impact("k", 0, 100)
        store.get_state_impact("absent", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("k"), before_audit)
        self.assertEqual(store.get_state("k"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)

    def test_data_file_restart_preserves_impact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
            expected = store.get_state_impact("k", 0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_state_impact("k", 0, 100), expected)
            self.assertEqual(
                recovered.get_state_impact("absent", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpStateImpactTests(unittest.TestCase):
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

    def get_impact(self, key: str, query: str = ""):
        return self.request("GET", f"/v1/states/{key}/impact{query}")

    def test_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1}))
        status, payload, headers, raw = self.get_impact("color")
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {"key", "status", "candidates", "impacts", "nextCursor", "hasMore"},
        )
        self.assertEqual(payload["key"], "color")
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"], [candidate("r1", "o1", "blue", {"r1": 1})]
        )
        self.assertEqual(
            payload["impacts"],
            [record("r2", "o2", "size", "large", {"r1": 1, "r2": 1})],
        )
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], False)
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
        status, _, _, raw = self.get_impact("cl%C3%A9")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 3, "r2": 2}))
        status, _, _, raw = self.get_impact("k")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"r2":2', body)
        self.assertIn('"nextCursor":1', body)

    def test_unknown_key_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, raw = self.get_impact("absent")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        self.assertEqual(raw, b'{"error":"not_found"}\n')

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("o1", "k/1", "v", {"r1": 1})
        status, _, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_impact("k%2F1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_paging_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for index in range(3):
            self.post_operation(
                "r2", operation(f"o{index}", f"k{index}", "x", {"r1": 1, "r2": index + 1})
            )
        status, page1, _, _ = self.get_impact("k", "?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["impacts"]), 2)
        self.assertEqual(page1["nextCursor"], 2)
        self.assertIs(page1["hasMore"], True)
        status, page2, _, _ = self.get_impact("k", "?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page2["impacts"]), 1)
        self.assertEqual(page2["nextCursor"], 3)
        self.assertIs(page2["hasMore"], False)
        status, tail, _, _ = self.get_impact("k", "?after=3")
        self.assertEqual(status, 200)
        self.assertEqual(tail["impacts"], [])
        self.assertIs(tail["hasMore"], False)

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
            "?after=1",  # past the impact count of 0
            "?limit=0",
            "?limit=101",
            "?limit=",
            "?limit=-1",
            "?limit=1.5",
        ):
            status, payload, _, _ = self.get_impact("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/impact/extra",
            "/v1/states/k/impact/extra/more",
            "/v1/states//impact",
            "/v1/impact",
            "/v2/states/k/impact",
            "/v1/states/k/impact/",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/impact/extra?x=1",
            "/v1/states//impact?x=1",
            "/v2/states/k/impact?x=1",
            "/v1/states/k/impact/?after=0",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_impact, _, _, _ = self.get_impact("k")
        self.get_impact("k", "?x=1")
        self.get_impact("k", "?after=9")
        self.get_impact("absent", "?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_impact, _, _, _ = self.get_impact("k")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_impact, after_impact)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_impact("k")
        self.get_impact("absent")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_impact_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/states/k/impact", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpTrailingSlashTests(unittest.TestCase):
    """A trailing slash on any published path is a 404 boundary."""

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
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_get_trailing_slashes_are_404(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request("POST", "/v1/replicas/r1/operations", op)
        self.assertEqual(status, 201)
        for path in (
            "/health/",
            "/v1/metrics/",
            "/v1/verification/digest/",
            "/v1/states/k/",
            "/v1/states/k/why/",
            "/v1/states/k/impact/",
            "/v1/sync/operations/",
            "/v1/audit/keys/k/operations/",
            "/v1/audit/keys/k/digest/",
            "/v1/replicas/r1/operations/o1/",
            "/v1/sync/peers/peer-a/checkpoint/",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_post_trailing_slashes_are_404_and_change_nothing(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        for path in (
            "/v1/replicas/r1/operations/",
            "/v1/sync/operations/",
            "/v1/states/k/resolve/",
            "/v1/states/k/resolve/auto/",
            "/v1/resolve/auto/batch/",
            "/v1/sync/peers/peer-a/checkpoint/",
        ):
            status, payload = self.request("POST", path, op)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
        # None of the rejected requests committed anything.
        status, payload = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(payload["acceptedOperations"], 0)
        status, payload = self.request("GET", "/v1/sync/peers/peer-a/checkpoint")
        self.assertEqual(status, 404)


class HttpStateImpactAuthTests(unittest.TestCase):
    """With auth enabled the impact endpoint authenticates like any GET."""

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

    def test_impact_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request("GET", "/v1/states/k/impact", auth=auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request("GET", "/v1/states/k/impact", auth="Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["status"], "resolved")

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpStateImpactPersistenceTests(unittest.TestCase):
    """The impact report survives a data-file restart unchanged."""

    def test_restart_preserves_impact(self) -> None:
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
                    ("GET", "/v1/states/k/impact", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/states/k/impact", None),
                    ("GET", "/v1/states/absent/impact", None),
                    ("GET", "/v1/states/k/impact?x=1", None),
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
