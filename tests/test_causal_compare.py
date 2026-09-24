"""Tests for the read-only two-operation causal comparison endpoint.

The comparison endpoint is::

    GET /v1/causal/compare?leftReplicaId=R&leftOperationId=O&rightReplicaId=R&rightOperationId=O&after=N&limit=N

It locates two first-accepted operations by identity and reports each
side's strict causal predecessors exactly as the single-operation chain
does (archive record shape plus ``direct``/``transitive``), together with
the dominance relation between the two source clocks and the deduplicated
identity difference of the two full predecessor sets. Paging trims only
the two predecessor lists; the relation and the difference are computed
over the complete sets. The response is compact canonical JSON terminated
by one newline.

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


def ancestor(replica_id: str, operation_id: str, key: str, value: str, clock: dict, relation: str) -> dict:
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


def compare_query(left: tuple[str, str], right: tuple[str, str], extra: str = "") -> str:
    return (
        f"?leftReplicaId={left[0]}&leftOperationId={left[1]}"
        f"&rightReplicaId={right[0]}&rightOperationId={right[1]}{extra}"
    )


class CausalCompareStoreTests(unittest.TestCase):
    """Store-level semantics of the two-operation causal comparison."""

    def test_unknown_identity_on_either_side_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_causal_compare("r1", "nope", "r1", "o1", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_compare("r1", "o1", "nope", "o1", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_compare("nope", "nope", "nope", "nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_no_predecessors_reports_empty_pages(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "a", "w", {"r2": 1}))
        status, payload = store.get_causal_compare("r1", "o1", "r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["left"]["operation"], source_record("r1", "o1", "k", "v", {"r1": 1})
        )
        self.assertEqual(
            payload["right"]["operation"], source_record("r2", "o2", "a", "w", {"r2": 1})
        )
        self.assertEqual(payload["left"]["ancestors"], [])
        self.assertEqual(payload["right"]["ancestors"], [])
        self.assertEqual(payload["left"]["cursor"], 0)
        self.assertEqual(payload["right"]["cursor"], 0)
        self.assertIs(payload["left"]["more"], False)
        self.assertIs(payload["right"]["more"], False)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

    def test_left_dominates_right(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 2}))
        status, payload = store.get_causal_compare("r1", "o2", "r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "left_dominates_right")
        self.assertEqual(
            payload["left"]["ancestors"],
            [ancestor("r1", "o1", "a", "v", {"r1": 1}, "direct")],
        )
        self.assertEqual(payload["right"]["ancestors"], [])
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 1, "rightOnly": 0}
        )

    def test_right_dominates_left(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 2}))
        status, payload = store.get_causal_compare("r1", "o1", "r1", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "right_dominates_left")
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 1}
        )

    def test_concurrent_sources(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        status, payload = store.get_causal_compare("r1", "o1", "r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")

    def test_equal_clocks_are_concurrent(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Same identity on both sides: the clocks are equal, so neither
        # dominates, and both sides share the same (empty) slice.
        status, payload = store.get_causal_compare("r1", "o1", "r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

    def test_difference_counts_deduplicated_identities(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "c", "x", {"r1": 2}))
        # Left source: strict predecessors o1, o2, o3.
        store.apply_operation(
            "r3", operation("o4", "d", "y", {"r1": 2, "r2": 1, "r3": 1})
        )
        # Right source: strict predecessors o1, o2.
        store.apply_operation("r2", operation("o5", "e", "z", {"r1": 1, "r2": 2}))
        status, payload = store.get_causal_compare("r3", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            [
                (a["replicaId"], a["operation"]["operationId"], a["relation"])
                for a in payload["left"]["ancestors"]
            ],
            [("r1", "o1", "transitive"), ("r2", "o2", "direct"), ("r1", "o3", "direct")],
        )
        self.assertEqual(
            [
                (a["replicaId"], a["operation"]["operationId"], a["relation"])
                for a in payload["right"]["ancestors"]
            ],
            [("r1", "o1", "direct"), ("r2", "o2", "direct")],
        )
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 1, "rightOnly": 0}
        )

    def test_stale_writes_and_repairs_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2, "r2": 1}))
        # Stale on its key: accepted but adds no candidate.
        store.apply_operation("r2", operation("o2", "k", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "a", "x", {"r3": 1}))
        store.apply_operation("r1", operation("o4", "b", "y", {"r1": 3, "r2": 1}))
        # An accepted repair is an ordinary committed record and
        # participates in both sides' slices like any other operation.
        store2 = StateStore()
        store2.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store2.apply_operation("r2", operation("o2", "k", "w", {"r2": 1}))
        status, error = store2.apply_resolution(
            "k",
            {
                "replicaId": "r3",
                "operationId": "fix-1",
                "value": "merged",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "candidates": [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r2", "operationId": "o2"},
                ],
            },
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        store2.apply_operation(
            "r1", operation("o3", "a", "x", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store2.get_causal_compare("r1", "o3", "r3", "fix-1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        # The accepted repair is an ordinary committed record: it is a
        # strict predecessor of o3 and the right side's source.
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["left"]["ancestors"]],
            ["o1", "o2", "fix-1"],
        )
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["right"]["ancestors"]],
            ["o1", "o2"],
        )
        self.assertEqual(payload["relation"], "left_dominates_right")
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 1, "rightOnly": 0}
        )
        # The stale write participates on the first store scenario.
        store.apply_operation(
            "r2", operation("o5", "c", "z", {"r1": 3, "r2": 2})
        )
        status, payload = store.get_causal_compare("r2", "o5", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertIn(
            ("r2", "o2"),
            [(a["replicaId"], a["operation"]["operationId"]) for a in payload["left"]["ancestors"]],
        )

    def test_paging_trims_only_the_predecessor_pages(self) -> None:
        store = StateStore()
        for index in range(3):
            store.apply_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        store.apply_operation("r2", operation("o9", "k", "y", {"r1": 3, "r2": 1}))
        store.apply_operation("r3", operation("o8", "j", "z", {"r1": 2, "r3": 1}))
        status, page1 = store.get_causal_compare("r2", "o9", "r3", "o8", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        # Left has 3 predecessors, right has 2; both pages share after/limit.
        self.assertEqual(len(page1["left"]["ancestors"]), 2)
        self.assertEqual(page1["left"]["cursor"], 2)
        self.assertIs(page1["left"]["more"], True)
        self.assertEqual(len(page1["right"]["ancestors"]), 2)
        self.assertEqual(page1["right"]["cursor"], 2)
        self.assertIs(page1["right"]["more"], False)
        # Relation and difference reflect the complete sets, not the page.
        self.assertEqual(page1["relation"], "concurrent")
        self.assertEqual(
            page1["difference"], {"shared": 2, "leftOnly": 1, "rightOnly": 0}
        )
        status, page2 = store.get_causal_compare("r2", "o9", "r3", "o8", 2, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page2["left"]["ancestors"]), 1)
        self.assertEqual(page2["left"]["cursor"], 3)
        self.assertIs(page2["left"]["more"], False)
        # The right side is exactly at its total: a valid empty tail page.
        self.assertEqual(page2["right"]["ancestors"], [])
        self.assertEqual(page2["right"]["cursor"], 2)
        self.assertIs(page2["right"]["more"], False)
        self.assertEqual(page2["relation"], page1["relation"])
        self.assertEqual(page2["difference"], page1["difference"])
        # The pages partition each side's unpaged list in commit order.
        _, whole = store.get_causal_compare("r2", "o9", "r3", "o8", 0, 100)
        self.assertEqual(
            page1["left"]["ancestors"] + page2["left"]["ancestors"],
            whole["left"]["ancestors"],
        )
        self.assertEqual(
            page1["right"]["ancestors"] + page2["right"]["ancestors"],
            whole["right"]["ancestors"],
        )

    def test_after_past_either_side_is_rejected(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "c", "x", {"r3": 1}))
        # after=2 is past both sides; after=1 is past the right side only.
        with self.assertRaises(ValueError):
            store.get_causal_compare("r2", "o2", "r3", "o3", 2, 100)
        with self.assertRaises(ValueError):
            store.get_causal_compare("r2", "o2", "r3", "o3", 1, 100)
        with self.assertRaises(ValueError):
            store.get_causal_compare("r3", "o3", "r2", "o2", 1, 100)

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
        store.get_causal_compare("r2", "o2", "r1", "o1", 0, 100)
        store.get_causal_compare("absent", "absent", "r1", "o1", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("a"), before_audit)
        self.assertEqual(store.get_state("a"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)
        self.assertEqual(store.get_operation("r2", "o2"), before_archive)

    def test_data_file_restart_preserves_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
            store.apply_operation("r3", operation("o3", "c", "x", {"r3": 1}))
            expected = store.get_causal_compare("r2", "o2", "r3", "o3", 0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(
                recovered.get_causal_compare("r2", "o2", "r3", "o3", 0, 100), expected
            )
            self.assertEqual(
                recovered.get_causal_compare("r2", "absent", "r3", "o3", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpCausalCompareTests(unittest.TestCase):
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

    def get_compare(self, left: tuple[str, str], right: tuple[str, str], extra: str = ""):
        return self.request("GET", f"/v1/causal/compare{compare_query(left, right, extra)}")

    def test_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "size", "large", {"r1": 1, "r2": 1}))
        self.post_operation("r3", operation("o3", "shape", "round", {"r3": 1}))
        status, payload, headers, raw = self.get_compare(("r2", "o2"), ("r3", "o3"))
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"left", "right", "relation", "difference"})
        self.assertEqual(
            set(payload["left"]), {"operation", "ancestors", "cursor", "more"}
        )
        self.assertEqual(
            set(payload["right"]), {"operation", "ancestors", "cursor", "more"}
        )
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r2", "o2", "size", "large", {"r1": 1, "r2": 1}),
        )
        self.assertEqual(
            payload["left"]["ancestors"],
            [ancestor("r1", "o1", "color", "blue", {"r1": 1}, "direct")],
        )
        self.assertEqual(payload["left"]["cursor"], 1)
        self.assertIs(payload["left"]["more"], False)
        self.assertEqual(payload["right"]["ancestors"], [])
        self.assertEqual(payload["right"]["cursor"], 0)
        self.assertIs(payload["right"]["more"], False)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 1, "rightOnly": 0}
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

    def test_non_ascii_strings_are_written_literally(self) -> None:
        self.post_operation("r1", operation("o1", "clé", "bléu", {"r1": 1}))
        self.post_operation("r2", operation("o2", "taille", "grand", {"r1": 1, "r2": 1}))
        status, _, _, raw = self.get_compare(("r2", "o2"), ("r1", "o1"))
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 3, "r2": 2}))
        status, _, _, raw = self.get_compare(("r2", "o2"), ("r1", "o1"))
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"shared":0', body)
        self.assertIn('"leftOnly":1', body)
        self.assertIn('"rightOnly":0', body)

    def test_dominance_relation_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r1", operation("o2", "a", "x", {"r1": 2}))
        status, payload, _, _ = self.get_compare(("r1", "o2"), ("r1", "o1"))
        self.assertEqual(status, 200)
        self.assertEqual(payload["relation"], "left_dominates_right")
        status, payload, _, _ = self.get_compare(("r1", "o1"), ("r1", "o2"))
        self.assertEqual(status, 200)
        self.assertEqual(payload["relation"], "right_dominates_left")

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        base = "leftReplicaId=r1&leftOperationId=o1"
        for query in (
            f"?{base}&rightReplicaId=r1&rightOperationId=absent",
            f"?leftReplicaId=absent&leftOperationId=o1&rightReplicaId=r1&rightOperationId=o1",
            "?leftReplicaId=absent&leftOperationId=absent&rightReplicaId=absent&rightOperationId=absent",
        ):
            status, payload, _, raw = self.request("GET", f"/v1/causal/compare{query}")
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)
            self.assertEqual(raw, b'{"error":"not_found"}\n', query)

    def test_query_values_are_percent_decoded(self) -> None:
        op = operation("o/1", "k", "v", {"r/1": 1})
        status, _, _, _ = self.request("POST", "/v1/replicas/r%2F1/operations", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.request(
            "GET",
            "/v1/causal/compare?leftReplicaId=r%2F1&leftOperationId=o%2F1"
            "&rightReplicaId=r%2F1&rightOperationId=o%2F1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r/1", "o/1", "k", "v", {"r/1": 1}),
        )

    def test_paging_over_http(self) -> None:
        for index in range(3):
            self.post_operation(
                "r1", operation(f"o{index}", f"k{index}", "x", {"r1": index + 1})
            )
        self.post_operation("r2", operation("o9", "k", "y", {"r1": 3, "r2": 1}))
        self.post_operation("r3", operation("o8", "j", "z", {"r1": 3, "r3": 1}))
        status, page1, _, _ = self.get_compare(("r2", "o9"), ("r3", "o8"), "&after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["left"]["ancestors"]), 2)
        self.assertEqual(page1["left"]["cursor"], 2)
        self.assertIs(page1["left"]["more"], True)
        status, page2, _, _ = self.get_compare(("r2", "o9"), ("r3", "o8"), "&after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page2["left"]["ancestors"]), 1)
        self.assertEqual(page2["left"]["cursor"], 3)
        self.assertIs(page2["left"]["more"], False)
        status, tail, _, _ = self.get_compare(("r2", "o9"), ("r3", "o8"), "&after=3")
        self.assertEqual(status, 200)
        self.assertEqual(tail["left"]["ancestors"], [])
        self.assertIs(tail["left"]["more"], False)
        self.assertEqual(tail["right"]["ancestors"], [])

    def test_bad_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 1, "r2": 1}))
        good = "leftReplicaId=r2&leftOperationId=o2&rightReplicaId=r1&rightOperationId=o1"
        for query in (
            "",  # all four identities missing
            "?leftReplicaId=r2&leftOperationId=o2&rightReplicaId=r1",  # one missing
            f"?{good}&x=1",  # unknown parameter
            f"?{good}&after=1&after=2",  # repeated
            f"?{good}&leftReplicaId=r1",  # repeated identity
            "?leftReplicaId=&leftOperationId=o2&rightReplicaId=r1&rightOperationId=o1",
            f"?{good}&after=",
            f"?{good}&after",
            f"?{good}&after=-1",
            f"?{good}&after=1.0",
            f"?{good}&after=%201",  # whitespace
            f"?{good}&after=%EF%BC%91",  # non-ASCII digit
            f"?{good}&after=2",  # past the left ancestor count of 1
            f"?{good}&after=1",  # past the right ancestor count of 0
            f"?{good}&limit=0",
            f"?{good}&limit=101",
            f"?{good}&limit=",
            f"?{good}&limit=-1",
            f"?{good}&limit=1.5",
        ):
            status, payload, _, _ = self.request("GET", f"/v1/causal/compare{query}")
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        query = "?leftReplicaId=r1&leftOperationId=o1&rightReplicaId=r1&rightOperationId=o1"
        for path in (
            f"/v1/causal/compare/extra/extra2{query}",
            f"/v1/causal{query}",
            f"/v2/causal/compare{query}",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
        # A fourth segment after "compare" is the single-operation route
        # for the identity ("compare", <segment>): the shape is valid, so
        # an unaccepted identity is 404 and a trailing slash (empty
        # operationId) is 404 the same way.
        for path in (
            "/v1/causal/compare/extra",
            "/v1/causal/compare/",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/compare/extra/extra2?x=1",
            "/v1/causal/compare/extra/extra2?leftReplicaId=r1",
            "/v2/causal/compare?x=1",
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
        before_compare, _, _, _ = self.get_compare(("r2", "o2"), ("r1", "o1"))
        self.get_compare(("r2", "o2"), ("r1", "o1"), "&x=1")
        self.get_compare(("r2", "o2"), ("r1", "o1"), "&after=9")
        self.request("GET", "/v1/causal/compare?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        after_compare, _, _, _ = self.get_compare(("r2", "o2"), ("r1", "o1"))
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_archive, after_archive)
        self.assertEqual(before_compare, after_compare)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_compare(("r1", "o1"), ("r1", "o1"))
        self.request(
            "GET",
            "/v1/causal/compare?leftReplicaId=absent&leftOperationId=absent"
            "&rightReplicaId=absent&rightOperationId=absent",
        )
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_compare_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/causal/compare", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpCausalCompareAuthTests(unittest.TestCase):
    """With auth enabled the compare endpoint authenticates like any GET."""

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

    def test_compare_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        query = (
            "?leftReplicaId=r1&leftOperationId=o1"
            "&rightReplicaId=r1&rightOperationId=o1"
        )
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request("GET", f"/v1/causal/compare{query}", auth=auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request(
            "GET", f"/v1/causal/compare{query}", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["left"]["ancestors"], [])

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalComparePersistenceTests(unittest.TestCase):
    """The comparison survives a data-file restart unchanged."""

    def test_restart_preserves_comparison(self) -> None:
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

            query = (
                "/v1/causal/compare?leftReplicaId=r2&leftOperationId=o2"
                "&rightReplicaId=r3&rightOperationId=o3"
            )
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
                    ("GET", query, None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 201)
            self.assertEqual(first[3][0], 200)
            second = serve_once(
                [
                    ("GET", query, None),
                    (
                        "GET",
                        "/v1/causal/compare?leftReplicaId=r2&leftOperationId=absent"
                        "&rightReplicaId=r3&rightOperationId=o3",
                        None,
                    ),
                    ("GET", "/v1/causal/compare?x=1", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[3][1])
            self.assertEqual(second[1], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))
            # The read-only query created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
