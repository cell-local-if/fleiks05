"""Tests for the read-only per-operation causal descendants endpoint.

The descendants endpoint is::

    GET /v1/causal/descendants?replicaId=R&operationId=O&after=N&limit=N

It locates one first-accepted operation by identity and reports its strict
causal descendants: the accepted records committed after it in the shared
log whose clocks strictly dominate the source operation's clock. Each
descendant is classified as ``direct`` (not dominated by any other strict
descendant) or ``transitive``. The response carries the source operation in
the archive shape, one page of descendants in global commit order, the
resume cursor, and the more flag as compact canonical JSON terminated by
one newline.

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


def descendant(replica_id: str, operation_id: str, key: str, value: str, clock: dict, relation: str) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
        "relation": relation,
    }


def source_record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
    }


class CausalDescendantsStoreTests(unittest.TestCase):
    """Store-level semantics of the causal-descendants snapshot."""

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_causal_descendants("r1", "nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_descendants("nope", "o1", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_no_descendants_reports_empty_page(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["operation"], source_record("r1", "o1", "k", "v", {"r1": 1})
        )
        self.assertEqual(payload["descendants"], [])
        self.assertEqual(payload["cursor"], 0)
        self.assertIs(payload["more"], False)

    def test_strict_descendants_are_reported_in_commit_order(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        # Concurrent with the source clock: not a strict descendant.
        store.apply_operation("r3", operation("o3", "c", "x", {"r3": 5}))
        store.apply_operation("r1", operation("o4", "d", "y", {"r1": 2, "r2": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(d["replicaId"], d["operation"]["operationId"]) for d in payload["descendants"]],
            [("r2", "o2"), ("r1", "o4")],
        )
        self.assertEqual(payload["cursor"], 2)
        self.assertIs(payload["more"], False)

    def test_source_operation_never_lists_itself(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["descendants"], [])

    def test_records_committed_before_the_source_are_excluded(self) -> None:
        store = StateStore()
        # Committed before the source and strictly larger: still excluded,
        # because only records after the source commit count.
        store.apply_operation("r2", operation("o2", "a", "x", {"r1": 4, "r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["descendants"], [])

    def test_equal_clock_is_not_strictly_larger(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Same clock components as o1 (missing r2 counts as 0): the clocks
        # are equal, so o2 is not a strict descendant of o1.
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 0}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["descendants"], [])

    def test_missing_components_count_as_zero(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # {"r1": 1, "r2": 1} strictly dominates {"r1": 1}.
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["descendants"],
            [descendant("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "direct")],
        )

    def test_direct_and_transitive_classification(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Dominates o1; dominated by o4: a transitive descendant of o1.
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        # Concurrent with o2: another direct descendant of o1.
        store.apply_operation("r3", operation("o3", "c", "x", {"r1": 1, "r3": 1}))
        store.apply_operation("r1", operation("o4", "d", "y", {"r1": 2, "r2": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["descendants"],
            [
                descendant("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "transitive"),
                descendant("r3", "o3", "c", "x", {"r1": 1, "r3": 1}, "direct"),
                descendant("r1", "o4", "d", "y", {"r1": 2, "r2": 1}, "direct"),
            ],
        )

    def test_relation_is_computed_on_the_complete_set_not_the_page(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "c", "x", {"r1": 1, "r2": 1, "r3": 1}))
        # Paging off the dominated record must not flip it to "direct".
        status, page = store.get_causal_descendants("r1", "o1", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            page["descendants"],
            [descendant("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "transitive")],
        )
        self.assertIs(page["more"], True)

    def test_stale_writes_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "w", {"r1": 2, "r2": 1}))
        # Stale on its key (dominated by {"r1": 2, "r2": 1}): accepted but
        # adds no candidate. It is still a strict descendant of o1 — and a
        # transitive one, since o2 dominates it.
        store.apply_operation("r3", operation("o3", "k", "z", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(d["operation"]["operationId"], d["relation"]) for d in payload["descendants"]],
            [("o2", "direct"), ("o3", "transitive")],
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
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["descendants"],
            [descendant("r3", "fix-1", "k", "merged", {"r1": 1, "r2": 1, "r3": 1}, "direct")],
        )

    def test_replays_and_rejected_requests_never_appear(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Identical replay: no new record.
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Conflicting identity: rejected, no record.
        store.apply_operation("r1", operation("o1", "a", "other", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_descendants("r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["descendants"],
            [descendant("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "direct")],
        )

    def test_paging_boundaries(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o0", "k0", "x", {"r1": 1}))
        for index in range(1, 4):
            store.apply_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        status, page1 = store.get_causal_descendants("r1", "o0", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page1["descendants"]), 2)
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2 = store.get_causal_descendants("r1", "o0", page1["cursor"], 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page2["descendants"]), 1)
        self.assertEqual(page2["cursor"], 3)
        self.assertIs(page2["more"], False)
        # after equal to the total is a valid empty tail.
        status, tail = store.get_causal_descendants("r1", "o0", 3, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["descendants"], [])
        self.assertEqual(tail["cursor"], 3)
        self.assertIs(tail["more"], False)
        # The two pages partition the unpaged list in commit order.
        _, whole = store.get_causal_descendants("r1", "o0", 0, 100)
        self.assertEqual(page1["descendants"] + page2["descendants"], whole["descendants"])

    def test_after_past_the_descendant_count_is_rejected(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        with self.assertRaises(ValueError):
            store.get_causal_descendants("r1", "o1", 2, 100)

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
        store.get_causal_descendants("r1", "o1", 0, 100)
        store.get_causal_descendants("absent", "absent", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("a"), before_audit)
        self.assertEqual(store.get_state("a"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)
        self.assertEqual(store.get_operation("r2", "o2"), before_archive)

    def test_data_file_restart_preserves_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
            expected = store.get_causal_descendants("r1", "o1", 0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_causal_descendants("r1", "o1", 0, 100), expected)
            self.assertEqual(
                recovered.get_causal_descendants("r2", "absent", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpCausalDescendantsTests(unittest.TestCase):
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

    def get_descendants(self, query: str = ""):
        return self.request("GET", f"/v1/causal/descendants{query}")

    def test_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1}))
        status, payload, headers, raw = self.get_descendants("?replicaId=r1&operationId=o1")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"operation", "descendants", "cursor", "more"})
        self.assertEqual(
            payload["operation"],
            source_record("r1", "o1", "color", "blue", {"r1": 1}),
        )
        self.assertEqual(
            payload["descendants"],
            [descendant("r2", "o2", "size", "large", {"r1": 1, "r2": 1}, "direct")],
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
        self.post_operation("r2", operation("o2", "taille", "grand", {"r1": 1, "r2": 1}))
        status, _, _, raw = self.get_descendants("?replicaId=r1&operationId=o1")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 3, "r2": 2}))
        status, _, _, raw = self.get_descendants("?replicaId=r1&operationId=o1")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"r2":2', body)
        self.assertIn('"cursor":1', body)

    def test_identity_values_are_percent_decoded(self) -> None:
        op = operation("o/1", "k", "v", {"r/1": 1})
        status, _, _, _ = self.request("POST", "/v1/replicas/r%2F1/operations", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_descendants("?replicaId=r%2F1&operationId=o%2F1")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["operation"],
            source_record("r/1", "o/1", "k", "v", {"r/1": 1}),
        )

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in (
            "?replicaId=r1&operationId=absent",
            "?replicaId=absent&operationId=o1",
            "?replicaId=absent&operationId=absent",
        ):
            status, payload, _, raw = self.get_descendants(query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)
            self.assertEqual(raw, b'{"error":"not_found"}\n', query)

    def test_paging_over_http(self) -> None:
        self.post_operation("r1", operation("o0", "k0", "x", {"r1": 1}))
        for index in range(1, 4):
            self.post_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        base = "?replicaId=r1&operationId=o0"
        status, page1, _, _ = self.get_descendants(f"{base}&after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["descendants"]), 2)
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2, _, _ = self.get_descendants(f"{base}&after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page2["descendants"]), 1)
        self.assertEqual(page2["cursor"], 3)
        self.assertIs(page2["more"], False)
        status, tail, _, _ = self.get_descendants(f"{base}&after=3")
        self.assertEqual(status, 200)
        self.assertEqual(tail["descendants"], [])
        self.assertIs(tail["more"], False)

    def test_bad_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        for query in (
            "",
            "?replicaId=r1",
            "?operationId=o1",
            "?replicaId=&operationId=o1",
            "?replicaId=r1&operationId=",
            "?replicaId=r1&replicaId=r1&operationId=o1",
            "?replicaId=r1&operationId=o1&operationId=o1",
            "?replicaId=r1&operationId=o1&x=1",
            "?replicaId=r1&operationId=o1&after=1&after=2",
            "?replicaId=r1&operationId=o1&after=",
            "?replicaId=r1&operationId=o1&after",
            "?replicaId=r1&operationId=o1&after=-1",
            "?replicaId=r1&operationId=o1&after=1.0",
            "?replicaId=r1&operationId=o1&after=%201",  # whitespace
            "?replicaId=r1&operationId=o1&after=%EF%BC%91",  # non-ASCII digit
            "?replicaId=r1&operationId=o1&after=2",  # past the descendant count of 1
            "?replicaId=r1&operationId=o1&limit=0",
            "?replicaId=r1&operationId=o1&limit=101",
            "?replicaId=r1&operationId=o1&limit=",
            "?replicaId=r1&operationId=o1&limit=-1",
            "?replicaId=r1&operationId=o1&limit=1.5",
        ):
            status, payload, _, _ = self.get_descendants(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/descendants/extra",
            "/v1/causal/descendants/",
            "/v1/causal",
            "/v2/causal/descendants",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/descendants/extra?x=1",
            "/v1/causal/descendants/?replicaId=r1&operationId=o1",
            "/v2/causal/descendants?replicaId=r1&operationId=o1",
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
        base = "?replicaId=r1&operationId=o1"
        before_descendants, _, _, _ = self.get_descendants(base)
        self.get_descendants(f"{base}&x=1")
        self.get_descendants(f"{base}&after=9")
        self.get_descendants("?replicaId=absent&operationId=absent&x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        after_descendants, _, _, _ = self.get_descendants(base)
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_archive, after_archive)
        self.assertEqual(before_descendants, after_descendants)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_descendants("?replicaId=r1&operationId=o1")
        self.get_descendants("?replicaId=absent&operationId=absent")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_descendants_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/causal/descendants", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpCausalDescendantsAuthTests(unittest.TestCase):
    """With auth enabled the descendants endpoint authenticates like any GET."""

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

    def test_descendants_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request(
                "GET", "/v1/causal/descendants?replicaId=r1&operationId=o1", auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request(
            "GET", "/v1/causal/descendants?replicaId=r1&operationId=o1", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["descendants"], [])

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalDescendantsPersistenceTests(unittest.TestCase):
    """The descendants report survives a data-file restart unchanged."""

    def test_restart_preserves_descendants(self) -> None:
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
                    ("GET", "/v1/causal/descendants?replicaId=r1&operationId=o1", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/causal/descendants?replicaId=r1&operationId=o1", None),
                    ("GET", "/v1/causal/descendants?replicaId=r2&operationId=absent", None),
                    ("GET", "/v1/causal/descendants?replicaId=r1&operationId=o1&x=1", None),
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
