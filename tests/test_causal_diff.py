"""Tests for the read-only two-operation causal-difference explanation.

The difference endpoint is::

    GET /v1/causal/diff?leftReplicaId=R&leftOperationId=O\
&rightReplicaId=R&rightOperationId=O&after=N&limit=N

It locates two first-accepted operations by identity (the same four query
parameters as ``GET /v1/causal/compare``), splits their strict causal
predecessor identities into three groups — ``shared``, ``leftOnly``, and
``rightOnly`` — and renders one merged page in global commit order with
the comparison predecessor shape. ``difference`` reports the full-set
counts, and ``explanation`` compresses the one-sided evidence to the
minimal boundary each side alone introduces: the exclusive records no
other same-side exclusive evidence dominates, each as a
``{"from","to","side"}`` item in global commit order. Paging only trims
the merged evidence page; the counts and explanation always come from the
complete sets.

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
    parse_causal_diff_query,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def entry(replica_id: str, operation_id: str, key: str, value: str, clock: dict, relation: str) -> dict:
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


def explanation_item(from_replica: str, from_operation: str, to_replica: str, to_operation: str, side: str) -> dict:
    return {
        "from": {"replicaId": from_replica, "operationId": from_operation},
        "to": {"replicaId": to_replica, "operationId": to_operation},
        "side": side,
    }


class CausalDiffStoreTests(unittest.TestCase):
    """Store-level semantics of the causal-difference snapshot."""

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

    def test_full_report_shape_and_groups(self) -> None:
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
            source_record("r1", "o4", "d", "y", {"r1": 2, "r2": 1}),
        )
        self.assertEqual(
            payload["right"],
            source_record("r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
        )
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(
            payload["shared"],
            [
                entry("r1", "o1", "a", "v", {"r1": 1}, "transitive"),
                entry("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "direct"),
            ],
        )
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(
            payload["rightOnly"],
            [
                entry("r3", "o3", "c", "x", {"r3": 1}, "direct"),
                entry("r1", "o4", "d", "y", {"r1": 2, "r2": 1}, "direct"),
            ],
        )
        # Both right-only records are minimal: neither dominates the other
        # ({r3:1} vs {r1:2,r2:1} are concurrent clocks).
        self.assertEqual(
            payload["explanation"],
            [
                explanation_item("r3", "o3", "r2", "o5", "right"),
                explanation_item("r1", "o4", "r2", "o5", "right"),
            ],
        )
        self.assertEqual(payload["cursor"], 4)
        self.assertIs(payload["more"], False)

    def test_groups_follow_global_commit_order(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r2", "o5", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        # Flipping the arguments swaps left/right but keeps commit order.
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 2, "rightOnly": 0}
        )
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["shared"]],
            [("r1", "o1"), ("r2", "o2")],
        )
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in payload["leftOnly"]],
            [("r3", "o3"), ("r1", "o4")],
        )
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(
            payload["explanation"],
            [
                explanation_item("r3", "o3", "r2", "o5", "left"),
                explanation_item("r1", "o4", "r2", "o5", "left"),
            ],
        )

    def test_explanation_drops_exclusive_records_dominated_on_the_same_side(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        # o3 dominates both o1 and o2.
        store.apply_operation("r1", operation("o3", "c", "x", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o4", "d", "q", {"r3": 1}))
        # Left source dominates o1..o3; right source dominates only o4.
        store.apply_operation("r1", operation("L", "e", "l", {"r1": 2, "r2": 1}))
        store.apply_operation("r3", operation("R", "f", "r", {"r3": 2}))
        status, payload = store.get_causal_diff("r1", "L", "r3", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["leftOnly"]],
            [("o1", "transitive"), ("o2", "transitive"), ("o3", "direct")],
        )
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["rightOnly"]],
            [("o4", "direct")],
        )
        self.assertEqual(payload["difference"], {"shared": 0, "leftOnly": 3, "rightOnly": 1})
        # o1 and o2 are explained away by the same-side exclusive o3; the
        # minimal left boundary is o3 alone, followed by the right one.
        self.assertEqual(
            payload["explanation"],
            [
                explanation_item("r1", "o3", "r1", "L", "left"),
                explanation_item("r3", "o4", "r3", "R", "right"),
            ],
        )

    def test_shared_evidence_never_dominates_an_exclusive_boundary_away(self) -> None:
        store = StateStore()
        # o1 ends up shared; o2 (which dominates o1) ends up left-only.
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        # Left source dominates both o1 and o2; the right source dominates
        # only o1 (it carries no r2 component).
        store.apply_operation("r1", operation("L", "c", "l", {"r1": 2, "r2": 1}))
        store.apply_operation("r3", operation("R", "d", "r", {"r1": 1, "r3": 1}))
        status, payload = store.get_causal_diff("r1", "L", "r3", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["operation"]["operationId"]) for e in payload["shared"]], ["o1"]
        )
        self.assertEqual(
            [(e["operation"]["operationId"]) for e in payload["leftOnly"]], ["o2"]
        )
        # Even though shared o2 dominates o1, domination by a *shared*
        # record does not explain a left-only boundary: o2 is still the
        # minimal left frontier because no other left-only record
        # dominates it.
        self.assertEqual(
            payload["explanation"],
            [explanation_item("r2", "o2", "r1", "L", "left")],
        )

    def test_explanation_merges_both_sides_in_global_commit_order(self) -> None:
        store = StateStore()
        # b1 is shared; a1/a2 are left-only; c1/b2 are right-only.
        store.apply_operation("r1", operation("a1", "k", "1", {"r1": 1}))
        store.apply_operation("r2", operation("b1", "k", "2", {"r2": 1}))
        store.apply_operation("r1", operation("a2", "k", "3", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("c1", "k", "4", {"r3": 1}))
        store.apply_operation("r2", operation("b2", "k", "5", {"r2": 2}))
        store.apply_operation("r1", operation("L", "k", "L", {"r1": 2, "r2": 1}))
        store.apply_operation("r2", operation("R", "k", "R", {"r2": 3, "r3": 1}))
        status, payload = store.get_causal_diff("r1", "L", "r2", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["difference"], {"shared": 1, "leftOnly": 2, "rightOnly": 2})
        # a1 is dominated by same-side exclusive a2, so the left frontier
        # is a2 alone; the right frontier is c1 and b2 (concurrent clocks).
        # Global commit order interleaves the two sides: a2, c1, b2.
        self.assertEqual(
            payload["explanation"],
            [
                explanation_item("r1", "a2", "r1", "L", "left"),
                explanation_item("r3", "c1", "r2", "R", "right"),
                explanation_item("r2", "b2", "r2", "R", "right"),
            ],
        )

    def test_comparing_an_operation_with_itself_has_no_difference(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o1", "r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(payload["explanation"], [])
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )
        self.assertEqual(payload["cursor"], 0)
        self.assertIs(payload["more"], False)

    def test_records_committed_after_a_source_are_never_evidence(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o1", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(payload["leftOnly"], [])
        # From o1's viewpoint the later records do not exist; o5 still sees
        # the prefix, so the whole prefix is right-only.
        self.assertEqual(len(payload["rightOnly"]), 4)
        self.assertEqual(payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 4})

    def test_stale_writes_participate_like_any_committed_record(self) -> None:
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
        # A stale write: dominated by the repair clock, accepted with no
        # candidate, but still an ordinary committed record.
        store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 1, "r2": 1}))
        store.apply_operation(
            "r1", operation("o4", "a", "x", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_diff("r3", "fix-1", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["operation"]["operationId"]) for e in payload["shared"]],
            ["o1", "o2"],
        )
        # The repair and the stale write exist only on the later source's side.
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in payload["rightOnly"]],
            [("fix-1", "direct"), ("o3", "transitive")],
        )
        # The stale write is dominated by the same-side exclusive repair,
        # so it is never a minimal boundary.
        self.assertEqual(
            payload["explanation"],
            [explanation_item("r3", "fix-1", "r1", "o4", "right")],
        )

    def test_counts_and_explanation_use_the_full_sets_not_the_page(self) -> None:
        store = self.build_chain()
        status, full = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        for after, limit in ((0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2)):
            status, paged = store.get_causal_diff(
                "r1", "o4", "r2", "o5", after, limit
            )
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(paged["difference"], full["difference"])
            self.assertEqual(paged["explanation"], full["explanation"])
            self.assertEqual(paged["left"], full["left"])
            self.assertEqual(paged["right"], full["right"])

    def test_relation_labels_are_computed_before_paging(self) -> None:
        store = self.build_chain()
        # Shared o1 is transitive only because shared o2 (on a different
        # page position) dominates it; a one-entry first page must still
        # label o1 "transitive".
        status, page1 = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(e["operation"]["operationId"], e["relation"]) for e in page1["shared"]],
            [("o1", "transitive")],
        )

    def test_paging_walks_shared_then_left_then_right(self) -> None:
        store = self.build_chain()
        # Merged order is o1, o2 (shared) then o3, o4 (right-only).
        status, page1 = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [e["operation"]["operationId"] for e in page1["shared"]], ["o1", "o2"]
        )
        self.assertEqual(page1["rightOnly"], [])
        self.assertEqual(page1["cursor"], 2)
        self.assertIs(page1["more"], True)

        status, page2 = store.get_causal_diff("r1", "o4", "r2", "o5", 1, 2)
        self.assertIs(status, HTTPStatus.OK)
        # The page straddles the group boundary: position 1 is shared o2,
        # position 2 is right-only o3.
        self.assertEqual(
            [e["operation"]["operationId"] for e in page2["shared"]], ["o2"]
        )
        self.assertEqual(page2["leftOnly"], [])
        self.assertEqual(
            [e["operation"]["operationId"] for e in page2["rightOnly"]], ["o3"]
        )
        self.assertEqual(page2["cursor"], 3)
        self.assertIs(page2["more"], True)

        status, page3 = store.get_causal_diff("r1", "o4", "r2", "o5", 3, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(page3["shared"], [])
        self.assertEqual(
            [e["operation"]["operationId"] for e in page3["rightOnly"]], ["o4"]
        )
        self.assertEqual(page3["cursor"], 4)
        self.assertIs(page3["more"], False)

    def test_pages_partition_the_merged_evidence(self) -> None:
        store = self.build_chain()
        status, full = store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        merged = full["shared"] + full["leftOnly"] + full["rightOnly"]
        collected = []
        after = 0
        while True:
            status, page = store.get_causal_diff("r1", "o4", "r2", "o5", after, 1)
            self.assertIs(status, HTTPStatus.OK)
            collected += page["shared"] + page["leftOnly"] + page["rightOnly"]
            after = page["cursor"]
            if not page["more"]:
                break
        self.assertEqual(collected, merged)
        self.assertEqual(after, len(merged))

    def test_after_at_the_merged_length_is_an_empty_page(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_diff("r1", "o4", "r2", "o5", 4, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["shared"], [])
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(payload["rightOnly"], [])
        self.assertEqual(payload["cursor"], 4)
        self.assertIs(payload["more"], False)

    def test_after_past_the_merged_length_is_rejected(self) -> None:
        store = self.build_chain()
        with self.assertRaises(ValueError):
            store.get_causal_diff("r1", "o4", "r2", "o5", 5, 100)

    def test_after_bound_is_the_merged_length_not_the_larger_side(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("a1", "k", "1", {"r1": 1}))
        store.apply_operation("r2", operation("b1", "k", "2", {"r2": 1}))
        store.apply_operation("r1", operation("a2", "k", "3", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("c1", "k", "4", {"r3": 1}))
        store.apply_operation("r2", operation("b2", "k", "5", {"r2": 2}))
        store.apply_operation("r1", operation("L", "k", "L", {"r1": 2, "r2": 1}))
        store.apply_operation("r2", operation("R", "k", "R", {"r2": 3, "r3": 1}))
        # Each side has 3 predecessors (b1 shared, two exclusive), but the
        # merged evidence is 5 long: shared b1 + left-only a1,a2 +
        # right-only c1,b2.
        status, full = store.get_causal_diff("r1", "L", "r2", "R", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(full["difference"], {"shared": 1, "leftOnly": 2, "rightOnly": 2})
        # after=4 is past the larger side's count of 3 but still inside the
        # merged sequence of 5: the last right-only record remains.
        status, page = store.get_causal_diff("r1", "L", "r2", "R", 4, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["rightOnly"]], ["b2"]
        )
        self.assertEqual(page["cursor"], 5)
        self.assertIs(page["more"], False)
        # after == merged length is the valid empty tail ...
        status, tail = store.get_causal_diff("r1", "L", "r2", "R", 5, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(tail["shared"] + tail["leftOnly"] + tail["rightOnly"], [])
        self.assertEqual(tail["cursor"], 5)
        # ... and only past it is the request rejected.
        with self.assertRaises(ValueError):
            store.get_causal_diff("r1", "L", "r2", "R", 6, 100)

    def test_query_is_read_only(self) -> None:
        store = self.build_chain()
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_snapshot = store.get_replication_snapshot()
        before_compare = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        store.get_causal_diff("r1", "o4", "r2", "o5", 0, 100)
        store.get_causal_diff("r1", "o4", "r2", "o5", 3, 1)
        store.get_causal_diff("absent", "x", "r2", "o5", 0, 100)
        with self.assertRaises(ValueError):
            store.get_causal_diff("r1", "o4", "r2", "o5", 99, 100)
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
            expected = store.get_causal_diff("r1", "o4", "r2", "o5", 1, 2)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(
                recovered.get_causal_diff("r1", "o4", "r2", "o5", 1, 2),
                expected,
            )
            self.assertEqual(
                recovered.get_causal_diff("r1", "nope", "r2", "o5", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class ParseCausalDiffQueryTests(unittest.TestCase):
    """Unit-level query validation for the diff endpoint."""

    BASE = (
        "leftReplicaId=r1&leftOperationId=o1&"
        "rightReplicaId=r2&rightOperationId=o2"
    )

    def test_minimal_query(self) -> None:
        self.assertEqual(
            parse_causal_diff_query(self.BASE),
            ("r1", "o1", "r2", "o2", 0, 100),
        )

    def test_full_query(self) -> None:
        self.assertEqual(
            parse_causal_diff_query(self.BASE + "&after=3&limit=7"),
            ("r1", "o1", "r2", "o2", 3, 7),
        )

    def test_invalid_queries(self) -> None:
        for query in (
            "",
            "leftReplicaId=r1&leftOperationId=o1&rightReplicaId=r2",
            self.BASE + "&x=1",
            self.BASE + "&after=1&after=2",
            "leftReplicaId=&leftOperationId=o1&rightReplicaId=r2&rightOperationId=o2",
            self.BASE + "&after=",
            self.BASE + "&after=-1",
            self.BASE + "&after=1.0",
            self.BASE + "&limit=0",
            self.BASE + "&limit=101",
            self.BASE + "&after=" + "9" * 5000,
        ):
            self.assertIsNone(parse_causal_diff_query(query), query)


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
        self.assertEqual(
            payload["left"],
            source_record("r1", "o4", "d", "y", {"r1": 2, "r2": 1}),
        )
        self.assertEqual(
            payload["right"],
            source_record("r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
        )
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(
            [
                (e["replicaId"], e["operation"]["operationId"], e["relation"])
                for e in payload["shared"]
            ],
            [("r1", "o1", "transitive"), ("r2", "o2", "direct")],
        )
        self.assertEqual(payload["leftOnly"], [])
        self.assertEqual(
            [
                (e["replicaId"], e["operation"]["operationId"], e["relation"])
                for e in payload["rightOnly"]
            ],
            [("r3", "o3", "direct"), ("r1", "o4", "direct")],
        )
        self.assertEqual(
            payload["explanation"],
            [
                explanation_item("r3", "o3", "r2", "o5", "right"),
                explanation_item("r1", "o4", "r2", "o5", "right"),
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
            payload["left"],
            source_record("r/1", "o/1", "k", "v", {"r/1": 1}),
        )

    def test_paging_straddles_the_group_boundary(self) -> None:
        self.seed()
        status, page1, _, _ = self.diff(self.BASE_QUERY + "&after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["shared"]), 2)
        self.assertEqual(page1["leftOnly"], [])
        self.assertEqual(page1["rightOnly"], [])
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
        self.assertEqual(tail["cursor"], 4)
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
            base + "&after=5",  # past the merged evidence length of 4
            base + "&limit=0",
            base + "&limit=101",
            base + "&limit=",
            base + "&limit=-1",
            base + "&limit=1.5",
            base + "&after=" + "9" * 5000,  # beyond int() digit conversion cap
        ):
            status, payload, _, raw = self.diff(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
            self.assertEqual(raw, b'{"error":"invalid_request"}\n', query)

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
            status, payload, _, raw = self.diff_path(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)
            # A shape 404 is the generic route fallthrough without the
            # single-newline terminator used by error payloads of known
            # routes.
            self.assertEqual(raw, b'{"error":"not_found"}', path)

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

    def test_single_operation_route_still_matches(self) -> None:
        self.seed()
        # The diff reservation on the third segment must not shadow the
        # single-operation chain for an unrelated replica id.
        status, payload, _, _ = self.diff_path("/v1/causal/r2/o5")
        self.assertEqual(status, 200)
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["ancestors"]],
            ["o1", "o2", "o3", "o4"],
        )

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

    def test_post_to_diff_route_is_404(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("POST", "/v1/causal/diff", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


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
        return response.status, payload, response_headers, raw

    def diff_path(self, auth):
        return self.request(
            "GET",
            "/v1/causal/diff?leftReplicaId=r1&leftOperationId=o1"
            "&rightReplicaId=r1&rightOperationId=o1",
            auth=auth,
        )

    def test_diff_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Sekret"):
            status, payload, headers, _ = self.diff_path(auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(headers.get("WWW-Authenticate"), "Bearer", auth)
        status, payload, _, _ = self.diff_path("Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

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
        status, payload, _, _ = self.request("GET", "/health")
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
            # Merged positions 1-2 straddle the boundary: shared o2 and
            # right-only o3, with the counts still from the full sets.
            self.assertEqual(
                [e["operation"]["operationId"] for e in page["shared"]],
                ["o2"],
            )
            self.assertEqual(
                [e["operation"]["operationId"] for e in page["rightOnly"]],
                ["o3"],
            )
            self.assertEqual(page["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2})
            self.assertEqual(second[2], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[3], (400, b'{"error":"invalid_request"}\n'))
            # A shape 404 is answered by the generic route fallthrough
            # without a trailing newline, like every other unknown route.
            self.assertEqual(second[4], (404, b'{"error":"not_found"}'))
            # The read-only queries created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
