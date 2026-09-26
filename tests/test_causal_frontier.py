"""Tests for the read-only causal frontier overview endpoint.

The frontier endpoint is::

    GET /v1/causal/frontier?after=N&limit=N

It reports the causal frontier of the complete accepted-operation log:
the maximal set of first-accepted operations — the accepted records whose
clock no *other* accepted record's clock strictly dominates (missing
components count as 0). Equal or concurrent clocks both stay on the
frontier; ordinary writes, stale writes, sync imports, and accepted
repairs all participate, while replays, rejected requests, uncommitted
requests, and persistence failures never enter the log. The response
carries one page of frontier records in global commit order (each in the
per-operation archive shape), the sync-export paging fields, the complete
frontier count, and the lowercase hexadecimal SHA-256 digest of the
canonical frontier array as compact canonical JSON terminated by one
newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
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


def archive_record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
    }


def frontier_digest(records: list) -> str:
    """Recompute the expected digest from (replica_id, operation) records."""
    parts = []
    for replica_id, op in records:
        clock = ",".join(
            f'{json.dumps(name)}:{tick}' for name, tick in sorted(op["clock"].items())
        )
        parts.append(
            '{"replicaId":%s,"operation":{"operationId":%s,"key":%s,"value":%s,"clock":{%s}}}'
            % (
                json.dumps(replica_id),
                json.dumps(op["operationId"]),
                json.dumps(op["key"]),
                json.dumps(op["value"]),
                clock,
            )
        )
    return hashlib.sha256(("[" + ",".join(parts) + "]").encode("utf-8")).hexdigest()


EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


class CausalFrontierStoreTests(unittest.TestCase):
    """Store-level semantics of the causal-frontier snapshot."""

    def test_empty_log_reports_empty_frontier(self) -> None:
        store = StateStore()
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["operations"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_DIGEST)
        self.assertEqual(payload["frontierCount"], 0)

    def test_dominated_operations_leave_the_frontier(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Strictly dominates o1 (missing components count as 0).
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        # Concurrent with both: stays.
        store.apply_operation("r3", operation("o3", "c", "x", {"r3": 2}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["operations"],
            [
                archive_record("r2", "o2", "b", "w", {"r1": 1, "r2": 1}),
                archive_record("r3", "o3", "c", "x", {"r3": 2}),
            ],
        )
        self.assertEqual(payload["frontierCount"], 2)
        self.assertEqual(
            payload["digest"],
            frontier_digest(
                [
                    ("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1})),
                    ("r3", operation("o3", "c", "x", {"r3": 2})),
                ]
            ),
        )

    def test_equal_clocks_both_stay(self) -> None:
        store = StateStore()
        # Two distinct operations with identical clocks: neither strictly
        # dominates the other, so both stay on the frontier.
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 1}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["operations"]],
            [("r1", "o1"), ("r1", "o2")],
        )
        self.assertEqual(payload["frontierCount"], 2)

    def test_frontier_keeps_global_commit_order(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o1", "a", "v", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 1}))
        store.apply_operation("r3", operation("o3", "c", "x", {"r3": 1}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["operations"]],
            [("r2", "o1"), ("r1", "o2"), ("r3", "o3")],
        )

    def test_stale_writes_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "w", {"r1": 2, "r2": 1}))
        # Stale on its key (dominated by o2's clock): accepted but adds no
        # candidate. It is still an accepted record — but o2 dominates it,
        # so it leaves the frontier while o2 stays.
        store.apply_operation("r3", operation("o3", "k", "z", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["operations"]],
            [("r2", "o2")],
        )
        # A stale write that nothing dominates stays on the frontier.
        store.apply_operation("r4", operation("o4", "other", "y", {"r4": 1}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["operations"]],
            [("r2", "o2"), ("r4", "o4")],
        )

    def test_sync_imports_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        status, accepted, replayed = store.import_operations(
            [
                ("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1})),
                ("r3", operation("o3", "c", "x", {"r3": 1})),
            ]
        )
        self.assertEqual((status, accepted, replayed), (HTTPStatus.CREATED, 2, 0))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["replicaId"], r["operation"]["operationId"]) for r in payload["operations"]],
            [("r2", "o2"), ("r3", "o3")],
        )

    def test_accepted_repairs_participate(self) -> None:
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
        status, error = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        # The repair dominates both conflicting writes: it alone remains.
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["operations"],
            [archive_record("r3", "fix-1", "k", "merged", {"r1": 1, "r2": 1, "r3": 1})],
        )
        self.assertEqual(payload["frontierCount"], 1)

    def test_replays_and_rejected_requests_never_appear(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Identical replay: no new record.
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Conflicting identity: rejected, no record.
        store.apply_operation("r1", operation("o1", "a", "other", {"r1": 1}))
        status, payload = store.get_causal_frontier(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["frontierCount"], 1)
        self.assertEqual(
            payload["operations"],
            [archive_record("r1", "o1", "a", "v", {"r1": 1})],
        )

    def test_paging_boundaries(self) -> None:
        store = StateStore()
        for index in range(3):
            store.apply_operation(
                f"r{index}", operation(f"o{index}", f"k{index}", "x", {f"r{index}": 1})
            )
        status, page1 = store.get_causal_frontier(0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page1["operations"]), 2)
        self.assertEqual(page1["nextCursor"], 2)
        self.assertIs(page1["hasMore"], True)
        self.assertEqual(page1["frontierCount"], 3)
        status, page2 = store.get_causal_frontier(page1["nextCursor"], 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page2["operations"]), 1)
        self.assertEqual(page2["nextCursor"], 3)
        self.assertIs(page2["hasMore"], False)
        # after equal to the total is a valid stable empty page.
        status, tail = store.get_causal_frontier(3, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["operations"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)
        # The two pages partition the unpaged frontier in commit order, and
        # the digest and count are identical on every page.
        _, whole = store.get_causal_frontier(0, 100)
        self.assertEqual(page1["operations"] + page2["operations"], whole["operations"])
        self.assertEqual(page1["digest"], whole["digest"])
        self.assertEqual(page2["digest"], whole["digest"])
        self.assertEqual(tail["digest"], whole["digest"])

    def test_after_past_the_frontier_size_is_rejected(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            store.get_causal_frontier(2, 100)

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("a")
        before_state = store.get_state("a")
        before_sync = store.get_sync_operations(0, 100)
        before_archive = store.get_operation("r2", "o2")
        store.get_causal_frontier(0, 100)
        store.get_causal_frontier(0, 1)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("a"), before_audit)
        self.assertEqual(store.get_state("a"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)
        self.assertEqual(store.get_operation("r2", "o2"), before_archive)

    def test_data_file_restart_preserves_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
            store.apply_operation("r3", operation("o3", "c", "x", {"r3": 1}))
            expected = store.get_causal_frontier(0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_causal_frontier(0, 100), expected)


class HttpCausalFrontierTests(unittest.TestCase):
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

    def get_frontier(self, query: str = ""):
        return self.request("GET", f"/v1/causal/frontier{query}")

    def test_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1}))
        self.post_operation("r3", operation("o3", "mood", "calm", {"r3": 2}))
        status, payload, headers, raw = self.get_frontier("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {"operations", "nextCursor", "hasMore", "algorithm", "digest", "frontierCount"},
        )
        self.assertEqual(
            payload["operations"],
            [
                archive_record("r2", "o2", "size", "large", {"r1": 1, "r2": 1}),
                archive_record("r3", "o3", "mood", "calm", {"r3": 2}),
            ],
        )
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["frontierCount"], 2)
        self.assertEqual(
            payload["digest"],
            frontier_digest(
                [
                    ("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1})),
                    ("r3", operation("o3", "mood", "calm", {"r3": 2})),
                ]
            ),
        )
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

    def test_empty_frontier_hashes_the_empty_array(self) -> None:
        status, payload, _, _ = self.get_frontier("?after=0&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        self.assertEqual(payload["frontierCount"], 0)
        self.assertEqual(payload["digest"], EMPTY_DIGEST)

    def test_digest_is_identical_on_every_page(self) -> None:
        for index in range(3):
            self.post_operation(
                f"r{index}", operation(f"o{index}", f"k{index}", "x", {f"r{index}": 1})
            )
        status, page1, _, _ = self.get_frontier("?after=0&limit=2")
        status, page2, _, _ = self.get_frontier("?after=2&limit=2")
        status, tail, _, _ = self.get_frontier("?after=3&limit=100")
        self.assertEqual(page1["digest"], page2["digest"])
        self.assertEqual(page2["digest"], tail["digest"])
        self.assertEqual(page1["frontierCount"], 3)
        self.assertEqual(page2["frontierCount"], 3)
        self.assertEqual(tail["frontierCount"], 3)
        self.assertEqual(tail["operations"], [])
        self.assertIs(tail["hasMore"], False)

    def test_paging_over_http(self) -> None:
        for index in range(3):
            self.post_operation(
                f"r{index}", operation(f"o{index}", f"k{index}", "x", {f"r{index}": 1})
            )
        status, page1, _, _ = self.get_frontier("?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["operations"]), 2)
        self.assertEqual(page1["nextCursor"], 2)
        self.assertIs(page1["hasMore"], True)
        status, page2, _, _ = self.get_frontier("?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page2["operations"]), 1)
        self.assertEqual(page2["nextCursor"], 3)
        self.assertIs(page2["hasMore"], False)

    def test_bad_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "",
            "?after=0",
            "?limit=1",
            "?after=&limit=1",
            "?after=0&limit=",
            "?after&limit=1",
            "?after=0&after=1&limit=1",
            "?after=0&limit=1&limit=2",
            "?after=0&limit=1&x=1",
            "?after=-1&limit=1",
            "?after=+1&limit=1",
            "?after=1.0&limit=1",
            "?after=%201&limit=1",  # whitespace
            "?after=%EF%BC%91&limit=1",  # non-ASCII digit
            "?after=2&limit=1",  # past the frontier size of 1
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=-1",
            "?after=0&limit=1.5",
        ):
            status, payload, _, _ = self.get_frontier(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_after_equal_to_the_frontier_size_is_a_stable_empty_page(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, raw = self.get_frontier("?after=1&limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["frontierCount"], 1)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/frontier/extra",
            "/v1/causal/frontier/",
            "/v1/causal",
            "/v2/causal/frontier",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/frontier/extra?x=1",
            "/v1/causal/frontier/?after=0&limit=1",
            "/v2/causal/frontier?after=0&limit=1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        before_frontier, _, _, _ = self.get_frontier("?after=0&limit=100")
        self.get_frontier("?after=0&limit=1&x=1")
        self.get_frontier("?after=9&limit=1")
        self.get_frontier("?limit=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        after_frontier, _, _, _ = self.get_frontier("?after=0&limit=100")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_archive, after_archive)
        self.assertEqual(before_frontier, after_frontier)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_frontier("?after=0&limit=100")
        self.get_frontier("?after=1&limit=100")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_frontier_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/causal/frontier", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpCausalFrontierAuthTests(unittest.TestCase):
    """With auth enabled the frontier endpoint authenticates like any GET."""

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

    def test_frontier_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request(
                "GET", "/v1/causal/frontier?after=0&limit=1", auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request(
            "GET", "/v1/causal/frontier?after=0&limit=1", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["frontierCount"], 1)

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalFrontierScopeTests(unittest.TestCase):
    """In scope-policy mode the frontier endpoint needs read or admin."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes={
                "reader": frozenset({"read"}),
                "writer": frozenset({"write"}),
                "admin": frozenset({"admin"}),
            },
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

    def request(self, method: str, path: str, body: object = None, token: str | None = None):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        challenge = response.getheader("WWW-Authenticate")
        payload = json.loads(raw.decode("utf-8"))
        conn.close()
        return response.status, payload, challenge

    def test_read_and_admin_scopes_may_query(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, token="writer"
        )
        self.assertEqual(status, 201)
        for token in ("reader", "admin"):
            status, payload, _ = self.request(
                "GET", "/v1/causal/frontier?after=0&limit=1", token=token
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["frontierCount"], 1, token)

    def test_write_only_token_is_forbidden_without_challenge(self) -> None:
        status, payload, challenge = self.request(
            "GET", "/v1/causal/frontier?after=0&limit=1", token="writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_missing_or_bad_token_is_unauthorized_with_challenge(self) -> None:
        for token in (None, "stranger"):
            status, payload, challenge = self.request(
                "GET", "/v1/causal/frontier?after=0&limit=1", token=token
            )
            self.assertEqual(status, 401, token)
            self.assertEqual(payload, {"error": "unauthorized"}, token)
            self.assertEqual(challenge, "Bearer", token)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalFrontierPersistenceTests(unittest.TestCase):
    """The frontier report survives a data-file restart unchanged."""

    def test_restart_preserves_frontier(self) -> None:
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
                        operation("o1", "a", "v", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "b", "w", {"r1": 1, "r2": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r3/operations",
                        operation("o3", "c", "x", {"r3": 1}),
                    ),
                    ("GET", "/v1/causal/frontier?after=0&limit=100", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 201)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/causal/frontier?after=0&limit=100", None),
                    ("GET", "/v1/causal/frontier?after=0&limit=1", None),
                    ("GET", "/v1/causal/frontier?after=0", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[3][1])
            self.assertEqual(second[1][0], 200)
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))
            # The read-only query created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
