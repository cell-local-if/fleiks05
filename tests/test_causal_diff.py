"""Tests for the read-only two-operation causal difference endpoint.

The difference endpoint is::

    GET /v1/causal/diff?leftReplicaId=R&leftOperationId=O\
&rightReplicaId=R&rightOperationId=O&after=N&limit=N

It locates two first-accepted operations by identity — the same four
identity parameters and strict predecessor rules as the comparison
endpoint — and makes the causal difference directly visible as three
predecessor evidence groups (``shared``/``leftOnly``/``rightOnly``), each
in global commit order with the comparison endpoint's predecessor record
shape, plus the complete-set difference counts and a compressed
``explanation`` that locates only the minimal one-sided boundary: a
one-sided predecessor that no other one-sided predecessor on the same
side dominates. Paging runs over one stable merge (shared, then
left-only, then right-only); the counts and the explanation always come
from the complete sets.

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


def evidence(
    replica_id: str, operation_id: str, key: str, value: str, clock: dict, relation: str
) -> dict:
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


def explanation_entry(from_replica: str, from_operation: str, side: str,
                      to_replica: str, to_operation: str) -> dict:
    return {
        "from": {"replicaId": from_replica, "operationId": from_operation},
        "to": {"replicaId": to_replica, "operationId": to_operation},
        "side": side,
    }


class CausalDiffStoreTests(unittest.TestCase):
    """Store-level semantics of the grouped causal difference snapshot."""

    def build_chain(self) -> StateStore:
        # Commit order and clocks:
        #   r1/o1 {r1:1}
        #   r2/o2 {r1:1,r2:1}        dominates o1
        #   r3/o3 {r3:1}             concurrent with the r1/r2 chain
        #   r1/o4 {r1:2,r2:1}        dominates o1, o2
        #   r2/o5 {r1:2,r2:2,r3:1}   dominates o1..o4
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "c", "x", {"r3": 1}))
        store.apply_operation("r1", operation("o4", "d", "y", {"r1": 2, "r2": 1}))
        store.apply_operation(
            "r2", operation("o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1})
        )
        return store

    def test_unknown_identity_on_either_side_is_404(self) -> None:
        store = self.build_chain()
        self.assertEqual(
            store.get_causal_diff("r1", "nope", "r2", "o5", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_diff("r1", "o4", "r2", "nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_diff("nope", "o4", "also-nope", "o5", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_groups_counts_and_sources(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            set(payload),
            {
                "left",
                "right",
                "difference",
                "shared",
                "leftOnly",
                "rightOnly",
                "explanation",
                "cursor",
                "more",
            },
        )
        self.assertEqual(
            payload["left"],
            {
                "operation": source_record(
                    "r1", "o4", "d", "y", {"r1": 2, "r2": 1}
                )
            },
        )
        self.assertEqual(
            payload["right"],
            {
                "operation": source_record(
                    "r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}
                )
            },
        )
        # o1 and o2 are predecessors of both; o3 and o4 only of o5.
        self.assertEqual(
            payload["shared"],
            [
                evidence("r1", "o1", "a", "v", {"r1": 1}, "transitive"),
                evidence("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "transitive"),
            ],
        )
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(
            payload["rightOnly"],
            [
                evidence("r3", "o3", "c", "x", {"r3": 1}, "direct"),
                evidence("r1", "o4", "d", "y", {"r1": 2, "r2": 1}, "direct"),
            ],
        )
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(payload["cursor"], 4)
        self.assertIs(payload["more"], False)

    def test_explanation_locates_minimal_one_sided_boundary(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        # o3 and o4 are both right-only and neither dominates the other
        # (concurrent clocks), so both survive the compression.
        self.assertEqual(
            payload["explanation"],
            [
                explanation_entry("r3", "o3", "right", "r2", "o5"),
                explanation_entry("r1", "o4", "right", "r2", "o5"),
            ],
        )

    def test_explanation_drops_a_one_sided_record_dominated_on_that_side(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("a", "k", "v", {"r1": 1}))
        store.apply_operation(
            "r2", operation("b", "k", "w", {"r1": 1, "r2": 1})
        )  # dominates a
        store.apply_operation(
            "r3", operation("L", "k", "x", {"r1": 1, "r2": 1, "r3": 1})
        )  # left source: predecessors a, b
        store.apply_operation("r4", operation("R", "k", "z", {"r4": 1}))  # concurrent
        status, payload = store.get_causal_diff("r3", "L", "r4", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 2, "rightOnly": 0}
        )
        # a is left-only but dominated by another left-only record (b), so
        # only the minimal boundary b is located; the right side has no
        # one-sided evidence at all.
        self.assertEqual(
            payload["explanation"],
            [explanation_entry("r2", "b", "left", "r3", "L")],
        )
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["leftOnly"]],
            ["a", "b"],
        )
        self.assertEqual(
            [e["relation"] for e in payload["leftOnly"]],
            ["transitive", "direct"],
        )

    def test_shared_evidence_never_appears_in_the_explanation(self) -> None:
        # A shared predecessor may dominate a one-sided predecessor, but it
        # can never cover it in the explanation: only same-side one-sided
        # evidence counts.
        store = StateStore()
        store.apply_operation("r1", operation("s", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("l", "k", "w", {"r2": 1}))
        # Left source dominates s and l; right source dominates only s.
        store.apply_operation(
            "r1", operation("L", "k", "x", {"r1": 1, "r2": 1})
        )
        store.apply_operation(
            "r3", operation("R", "k", "z", {"r1": 1, "r3": 1})
        )
        status, payload = store.get_causal_diff("r1", "L", "r3", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["difference"], {"shared": 1, "leftOnly": 1, "rightOnly": 0}
        )
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["shared"]], ["s"]
        )
        # l is left-only and not dominated by any other left-only record
        # (s is shared and does not dominate l's clock anyway), so it is
        # located.
        self.assertEqual(
            payload["explanation"],
            [explanation_entry("r2", "l", "left", "r1", "L")],
        )

    def test_explanation_entries_keep_global_commit_order(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("a", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("b", "k", "w", {"r2": 1}))
        store.apply_operation("r3", operation("c", "k", "x", {"r3": 1}))
        # Left source dominates a; right source dominates b and c. All
        # three are one-sided and mutually concurrent, so every boundary is
        # located, in commit order a, b, c across the two sides.
        store.apply_operation(
            "r1", operation("L", "k", "p", {"r1": 1, "r9": 1})
        )
        store.apply_operation(
            "r2", operation("R", "k", "q", {"r2": 1, "r3": 1, "r9": 1})
        )
        status, payload = store.get_causal_diff("r1", "L", "r2", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["from"]["operationId"], e["side"]) for e in payload["explanation"]],
            [("a", "left"), ("b", "right"), ("c", "right")],
        )

    def test_reverse_argument_order_flips_the_groups(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r2", "o5", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 2, "rightOnly": 0}
        )
        self.assertEqual(
            [(e["from"]["operationId"], e["side"]) for e in payload["explanation"]],
            [("o3", "left"), ("o4", "left")],
        )
        self.assertEqual(
            [(e["to"]["operationId"]) for e in payload["explanation"]],
            ["o5", "o5"],
        )

    def test_comparing_an_operation_with_itself_is_empty(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o1", "r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(payload["explanation"], [])
        self.assertEqual(payload["cursor"], 0)
        self.assertIs(payload["more"], False)
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

    def test_stale_writes_and_accepted_repairs_participate(self) -> None:
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
        # A stale write after the repair (the repair clock dominates it).
        store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 1, "r2": 1}))
        store.apply_operation(
            "r1", operation("o4", "a", "x", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_diff("r3", "fix-1", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        # o1, o2 are shared; fix-1 and the stale o3 are right-only. fix-1
        # dominates o3, so o3 is compressed out of the explanation.
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["shared"]],
            ["o1", "o2"],
        )
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["rightOnly"]],
            [("fix-1", "direct"), ("o3", "transitive")],
        )
        self.assertEqual(
            payload["explanation"],
            [explanation_entry("r3", "fix-1", "right", "r1", "o4")],
        )

    def test_counts_and_explanation_use_the_full_sets_not_the_page(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            len(payload["shared"]) + len(payload["leftOnly"])
            + len(payload["rightOnly"]),
            1,
        )
        # The counts and the explanation still reflect all four
        # predecessors even though the page holds one.
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(len(payload["explanation"]), 2)
        self.assertEqual(payload["cursor"], 1)
        self.assertIs(payload["more"], True)

    def test_paging_walks_the_merged_groups_in_stable_order(self) -> None:
        store = self.build_chain()
        # Merge order is shared (o1, o2), then right-only (o3, o4).
        status, page1 = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [e["operation"]["operationId"] for e in page1["shared"]], ["o1", "o2"]
        )
        self.assertEqual(page1["leftOnly"], [])
        self.assertEqual(page1["rightOnly"], [])
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2 = store.get_causal_diff("r1", "o4", "r2", "o5", 2, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(page2["shared"], [])
        self.assertEqual(
            [e["operation"]["operationId"] for e in page2["rightOnly"]],
            ["o3", "o4"],
        )
        self.assertEqual(page2["cursor"], 4)
        self.assertIs(page2["more"], False)

    def test_paging_window_is_partitioned_back_into_groups(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("s1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("l1", "k", "w", {"r2": 1}))
        store.apply_operation("r1", operation("s2", "k", "v", {"r1": 2}))
        store.apply_operation("r2", operation("l2", "k", "w", {"r2": 2}))
        # Left source dominates s1, l1, s2; right dominates s1, l1, l2.
        store.apply_operation(
            "r1", operation("L", "k", "p", {"r1": 2, "r2": 1, "r9": 1})
        )
        store.apply_operation(
            "r3", operation("R", "k", "q", {"r1": 1, "r2": 2, "r3": 1, "r9": 1})
        )
        # Merge: shared s1, l1 (commit order), left-only s2, right-only l2.
        # A window straddling the shared/left-only boundary must split into
        # the right arrays by identity, not list position.
        status, payload = store.get_causal_diff("r1", "L", "r3", "R", 1, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["shared"]], ["l1"]
        )
        self.assertEqual(
            [e["operation"]["operationId"] for e in payload["leftOnly"]], ["s2"]
        )
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(payload["cursor"], 3)
        self.assertIs(payload["more"], True)

    def test_after_equal_to_merged_count_is_an_empty_page(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o4", "r2", "o5", 4, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(payload["cursor"], 4)
        self.assertIs(payload["more"], False)

    def test_after_past_the_merged_count_is_rejected(self) -> None:
        store = self.build_chain()
        with self.assertRaises(ValueError):
            store.get_causal_diff("r1", "o4", "r2", "o5", 5, 100)

    def test_query_is_read_only(self) -> None:
        store = self.build_chain()
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_snapshot = store.get_replication_snapshot()
        before_compare = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        store.get_causal_diff("r1", "o4", "r2", "o5", 3, 1)
        store.get_causal_diff("absent", "x", "r2", "o5", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_replication_snapshot(), before_snapshot)
        self.assertEqual(
            store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100),
            before_compare,
        )

    def test_data_file_restart_preserves_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            for args in (
                ("r1", operation("o1", "a", "v", {"r1": 1})),
                ("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1})),
                ("r3", operation("o3", "c", "x", {"r3": 1})),
                ("r1", operation("o4", "d", "y", {"r1": 2, "r2": 1})),
                ("r2", operation("o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1})),
            ):
                store.apply_operation(*args)
            expected = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 2)
            paged = store.get_causal_diff("r1", "o4", "r2", "o5", 2, 2)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(
                recovered.get_causal_diff("r1", "o4", "r2", "o5", 0, 2), expected
            )
            self.assertEqual(
                recovered.get_causal_diff("r1", "o4", "r2", "o5", 2, 2), paged
            )
            self.assertEqual(
                recovered.get_causal_diff("r1", "nope", "r2", "o5", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpCausalDiffTests(unittest.TestCase):
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

    def diff(self, query: str):
        return self.request("GET", f"/v1/causal/diff?{query}")

    def diff_path(self, path: str):
        return self.request("GET", path)

    def seed(self) -> None:
        self.post_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        self.post_operation("r3", operation("o3", "c", "x", {"r3": 1}))
        self.post_operation("r1", operation("o4", "d", "y", {"r1": 2, "r2": 1}))
        self.post_operation(
            "r2", operation("o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1})
        )

    BASE_QUERY = (
        "leftReplicaId=r1&leftOperationId=o4&"
        "rightReplicaId=r2&rightOperationId=o5"
    )

    def test_round_trip(self) -> None:
        self.seed()
        status, payload, headers, raw = self.diff(self.BASE_QUERY)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {
                "left",
                "right",
                "difference",
                "shared",
                "leftOnly",
                "rightOnly",
                "explanation",
                "cursor",
                "more",
            },
        )
        self.assertEqual(set(payload["left"]), {"operation"})
        self.assertEqual(set(payload["right"]), {"operation"})
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r1", "o4", "d", "y", {"r1": 2, "r2": 1}),
        )
        self.assertEqual(
            payload["right"]["operation"],
            source_record("r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
        )
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["shared"]],
            [("o1", "transitive"), ("o2", "transitive")],
        )
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["rightOnly"]],
            [("o3", "direct"), ("o4", "direct")],
        )
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(
            payload["explanation"],
            [
                explanation_entry("r3", "o3", "right", "r2", "o5"),
                explanation_entry("r1", "o4", "right", "r2", "o5"),
            ],
        )
        self.assertEqual(payload["cursor"], 4)
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
        status, _, _, raw = self.diff(
            "leftReplicaId=r1&leftOperationId=o1&rightReplicaId=r2&rightOperationId=o2"
        )
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.seed()
        status, _, _, raw = self.diff(self.BASE_QUERY + "&limit=2")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"cursor":2', body)
        self.assertIn('"shared":2', body)

    def test_unknown_identity_is_404(self) -> None:
        self.seed()
        for query in (
            "leftReplicaId=r1&leftOperationId=nope&rightReplicaId=r2&rightOperationId=o5",
            "leftReplicaId=r1&leftOperationId=o4&rightReplicaId=nope&rightOperationId=o5",
            "leftReplicaId=r1&leftOperationId=o4&rightReplicaId=r2&rightOperationId=nope",
        ):
            status, payload, _, raw = self.diff(query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)
            self.assertEqual(raw, b'{"error":"not_found"}\n', query)

    def test_identity_parameters_are_percent_decoded(self) -> None:
        op = operation("o/1", "k", "v", {"r/1": 1})
        status, _, _, _ = self.request("POST", "/v1/replicas/r%2F1/operations", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.diff(
            "leftReplicaId=r%2F1&leftOperationId=o%2F1&"
            "rightReplicaId=r%2F1&rightOperationId=o%2F1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r/1", "o/1", "k", "v", {"r/1": 1}),
        )

    def test_default_paging(self) -> None:
        self.seed()
        status, payload, _, _ = self.diff(self.BASE_QUERY)
        self.assertEqual(status, 200)
        self.assertEqual(payload["cursor"], 4)
        self.assertIs(payload["more"], False)

    def test_paging_over_http(self) -> None:
        self.seed()
        status, page1, _, _ = self.diff(self.BASE_QUERY + "&after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [e["operation"]["operationId"] for e in page1["shared"]], ["o1", "o2"]
        )
        self.assertEqual(page1["rightOnly"], [])
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)
        status, page2, _, _ = self.diff(self.BASE_QUERY + "&after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(page2["shared"], [])
        self.assertEqual(
            [e["operation"]["operationId"] for e in page2["rightOnly"]],
            ["o3", "o4"],
        )
        self.assertEqual(page2["cursor"], 4)
        self.assertIs(page2["more"], False)
        status, tail, _, _ = self.diff(self.BASE_QUERY + "&after=4")
        self.assertEqual(status, 200)
        self.assertEqual(tail["shared"], [])
        self.assertEqual(tail["leftOnly"], [])
        self.assertEqual(tail["rightOnly"], [])
        self.assertIs(tail["more"], False)

    def test_bad_query_parameters_are_400(self) -> None:
        self.seed()
        base = self.BASE_QUERY
        for query in (
            "leftReplicaId=r1&leftOperationId=o4&rightReplicaId=r2",  # missing
            base + "&x=1",  # unknown
            base.replace("leftReplicaId=r1", "leftReplicaId=r1&leftReplicaId=r9"),  # repeated
            base + "&after=1&after=2",  # repeated paging
            "leftReplicaId=&leftOperationId=o4&rightReplicaId=r2&rightOperationId=o5",  # blank
            base + "&rightOperationId=",  # blank trailing
            base + "&after=",  # blank after
            base + "&after",  # valueless after
            base + "&after=-1",
            base + "&after=1.0",
            base + "&after=%201",
            base + "&after=%EF%BC%91",
            base + "&after=5",  # past the merged predecessor count of 4
            base + "&limit=0",
            base + "&limit=101",
            base + "&limit=",
            base + "&limit=-1",
            base + "&limit=1.5",
            base + "&after=" + "9" * 5000,  # beyond int() digit conversion cap
        ):
            status, payload, _, _ = self.diff(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.seed()
        query = (
            "?leftReplicaId=r1&leftOperationId=o4&"
            "rightReplicaId=r2&rightOperationId=o5"
        )
        for path in (
            "/v1/causal/diff/",
            "/v1/causal/diff/extra",
            "/v1/causal",
            "/v2/causal/diff",
            "/v1/causal/diff" + query.replace("?", "/x?"),
        ):
            status, payload, _, _ = self.diff_path(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_and_identity_errors(self) -> None:
        self.seed()
        for path in (
            "/v1/causal/diff/?x=1",
            "/v1/causal/diff/extra?x=1",
            "/v1/causal/diff/extra?leftReplicaId=r1",
            "/v2/causal/diff?x=1",
        ):
            status, payload, _, _ = self.diff_path(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_single_operation_and_compare_routes_still_match(self) -> None:
        self.seed()
        status, payload, _, _ = self.diff_path("/v1/causal/r2/o5")
        self.assertEqual(status, 200)
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["ancestors"]],
            ["o1", "o2", "o3", "o4"],
        )
        status, payload, _, _ = self.diff_path(
            "/v1/causal/compare?" + self.BASE_QUERY
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["relation"], "right_dominates_left")

    def test_post_to_diff_route_is_404(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("POST", "/v1/causal/diff", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.seed()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_diff, _, _, _ = self.diff(self.BASE_QUERY)
        for query in (
            self.BASE_QUERY + "&x=1",
            self.BASE_QUERY + "&after=9",
        ):
            self.diff(query)
        self.diff(
            "leftReplicaId=r1&leftOperationId=nope&rightReplicaId=r2&rightOperationId=o5"
            "&x=1"
        )
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_diff, _, _, _ = self.diff(self.BASE_QUERY)
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_diff, after_diff)


class HttpCausalDiffAuthTests(unittest.TestCase):
    """With auth enabled the diff endpoint authenticates like any GET."""

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

    def diff_path(self, auth):
        return self.request(
            "GET",
            "/v1/causal/diff?leftReplicaId=r1&leftOperationId=o1"
            "&rightReplicaId=r1&rightOperationId=o1",
            auth=auth,
        )

    def test_diff_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Sekret"):
            status, payload, headers = self.diff_path(auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(headers.get("WWW-Authenticate"), "Bearer", auth)
        status, payload, _ = self.diff_path("Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0})

    def test_duplicate_authorization_header_is_401(self) -> None:
        path = (
            "/v1/causal/diff?leftReplicaId=r1&leftOperationId=o1"
            "&rightReplicaId=r1&rightOperationId=o1"
        )
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", path)
        conn.putheader("Authorization", "Bearer sekret")
        conn.putheader("Authorization", "Bearer sekret")
        conn.endheaders()
        response = conn.getresponse()
        status = response.status
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalDiffPersistenceTests(unittest.TestCase):
    """The diff report survives a data-file restart unchanged."""

    def test_restart_preserves_diff(self) -> None:
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
                "/v1/causal/diff?leftReplicaId=r1&leftOperationId=o4"
                "&rightReplicaId=r2&rightOperationId=o5"
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
                    (
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o4", "d", "y", {"r1": 2, "r2": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
                    ),
                    ("GET", query, None),
                ]
            )
            for status, _ in first[:5]:
                self.assertEqual(status, 201)
            self.assertEqual(first[5][0], 200)
            second = serve_once(
                [
                    ("GET", query, None),
                    ("GET", query + "&after=1&limit=2", None),
                    (
                        "GET",
                        "/v1/causal/diff?leftReplicaId=r1&leftOperationId=nope"
                        "&rightReplicaId=r2&rightOperationId=o5",
                        None,
                    ),
                    ("GET", query + "&x=1", None),
                    ("GET", "/v1/causal/diff/extra", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[5][1])
            self.assertEqual(second[1][0], 200)
            page = json.loads(second[1][1].decode("utf-8"))
            # after=1 skips o1 in the merged (shared-first) order: the
            # window holds o2 (shared) and o3 (right-only).
            self.assertEqual(
                [e["operation"]["operationId"] for e in page["shared"]], ["o2"]
            )
            self.assertEqual(
                [e["operation"]["operationId"] for e in page["rightOnly"]], ["o3"]
            )
            self.assertEqual(page["cursor"], 3)
            self.assertIs(page["more"], True)
            self.assertEqual(second[2], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[3], (400, b'{"error":"invalid_request"}\n'))
            # A shape 404 is answered by the generic route fallthrough
            # without a trailing newline, like every other unknown route.
            self.assertEqual(second[4], (404, b'{"error":"not_found"}'))
            # The read-only queries created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
