"""Tests for the read-only two-operation causal slice comparison.

The comparison endpoint is::

    GET /v1/causal/compare?leftReplicaId=R&leftOperationId=O\
&rightReplicaId=R&rightOperationId=O&after=N&limit=N

It locates two first-accepted operations by identity and compares their
strict causal predecessor slices — the same predecessors the single
operation chain reports — classifies the relation between the two source
clocks (``left_dominates_right``, ``right_dominates_left``, or
``concurrent``), and reports the de-duplicated identity difference
(``shared``/``leftOnly``/``rightOnly``). Each side carries its source
archive record, one page of predecessors in global commit order with the
``direct``/``transitive`` relation, a resume cursor, and a more flag. The
shared ``after``/``limit`` only trims the two pages; the relation and the
difference counts always come from the complete sets.

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
    parse_causal_compare_query,
    parse_paging_query,
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


class CausalCompareStoreTests(unittest.TestCase):
    """Store-level semantics of the two-operation comparison snapshot."""

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
            store.get_causal_comparison("r1", "nope", "r2", "o5", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_comparison("r1", "o1", "r2", "nope", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_comparison("nope", "o1", "also-nope", "o5", 0, 100),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_dominating_relation_and_full_sides(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(set(payload), {"left", "right", "relation", "difference"})
        self.assertEqual(payload["relation"], "right_dominates_left")
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r1", "o4", "d", "y", {"r1": 2, "r2": 1}),
        )
        self.assertEqual(
            payload["right"]["operation"],
            source_record("r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
        )
        self.assertEqual(
            payload["left"]["predecessors"],
            [
                ancestor("r1", "o1", "a", "v", {"r1": 1}, "transitive"),
                ancestor("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "direct"),
            ],
        )
        self.assertEqual(
            payload["right"]["predecessors"],
            [
                ancestor("r1", "o1", "a", "v", {"r1": 1}, "transitive"),
                ancestor("r2", "o2", "b", "w", {"r1": 1, "r2": 1}, "transitive"),
                ancestor("r3", "o3", "c", "x", {"r3": 1}, "direct"),
                ancestor("r1", "o4", "d", "y", {"r1": 2, "r2": 1}, "direct"),
            ],
        )
        self.assertEqual(payload["left"]["cursor"], 2)
        self.assertIs(payload["left"]["more"], False)
        self.assertEqual(payload["right"]["cursor"], 4)
        self.assertIs(payload["right"]["more"], False)
        # The two left predecessors are shared; o3 and o4 are right-only.
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )

    def test_reverse_argument_order_flips_relation(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_comparison("r2", "o5", "r1", "o4", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "left_dominates_right")
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 2, "rightOnly": 0}
        )

    def test_concurrent_sources(self) -> None:
        store = self.build_chain()
        # o2 {r1:1,r2:1} and o3 {r3:1} are concurrent.
        status, payload = store.get_causal_comparison("r2", "o2", "r3", "o3", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            [(a["operation"]["operationId"]) for a in payload["left"]["predecessors"]],
            ["o1"],
        )
        self.assertEqual(payload["right"]["predecessors"], [])
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 1, "rightOnly": 0}
        )

    def test_equal_clocks_are_concurrent(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Equal clocks (missing r2 counts as 0): neither dominates.
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 0}))
        status, payload = store.get_causal_comparison("r1", "o1", "r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")

    def test_comparing_an_operation_with_itself_is_concurrent(self) -> None:
        store = self.build_chain()
        status, payload = store.get_causal_comparison("r1", "o1", "r1", "o1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(payload["left"]["predecessors"], [])
        self.assertEqual(payload["right"]["predecessors"], [])
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

    def test_identities_are_de_duplicated_in_the_difference(self) -> None:
        # The same accepted identity can only appear once in the log, but a
        # predecessor set is built per side from identities: an identity
        # present on both sides counts once as shared regardless of order.
        store = self.build_chain()
        status, payload = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        shared_ids = {
            (a["replicaId"], a["operation"]["operationId"])
            for a in payload["left"]["predecessors"]
        } & {
            (a["replicaId"], a["operation"]["operationId"])
            for a in payload["right"]["predecessors"]
        }
        self.assertEqual(len(shared_ids), payload["difference"]["shared"])

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
        # A stale write committed after the repair: the repair clock
        # dominates {"r1":1,"r2":1}, so it is accepted but adds no
        # candidate. It is still a predecessor of a later operation — a
        # transitive one, since the repair dominates it.
        store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 1, "r2": 1}))
        store.apply_operation(
            "r1", operation("o4", "a", "x", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_comparison(
            "r3", "fix-1", "r1", "o4", 0, 100
        )
        self.assertIs(status, HTTPStatus.OK)
        # The repair sees only the records committed before it: o1 and o2.
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["left"]["predecessors"]],
            [("o1", "direct"), ("o2", "direct")],
        )
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["right"]["predecessors"]],
            [
                ("o1", "transitive"),
                ("o2", "transitive"),
                ("fix-1", "direct"),
                ("o3", "transitive"),
            ],
        )
        self.assertEqual(payload["relation"], "right_dominates_left")
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )

    def test_records_committed_after_a_source_are_excluded_on_that_side(self) -> None:
        store = self.build_chain()
        # From o1's viewpoint none of the later records exist, even though
        # some carry smaller clocks; o5's side still sees the whole prefix.
        status, payload = store.get_causal_comparison("r1", "o1", "r2", "o5", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["left"]["predecessors"], [])
        self.assertEqual(
            len(payload["right"]["predecessors"]), payload["right"]["cursor"]
        )

    def test_relation_and_difference_use_the_full_sets_not_the_page(self) -> None:
        store = self.build_chain()
        # limit=1 trims each page to one predecessor, but the right side's
        # full set of four still drives relation and the counts.
        status, payload = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(payload["left"]["predecessors"]), 1)
        self.assertEqual(len(payload["right"]["predecessors"]), 1)
        self.assertEqual(payload["relation"], "right_dominates_left")
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
        )
        self.assertEqual(payload["left"]["cursor"], 1)
        self.assertIs(payload["left"]["more"], True)
        self.assertEqual(payload["right"]["cursor"], 1)
        self.assertIs(payload["right"]["more"], True)

    def test_shared_after_skips_both_sides_together(self) -> None:
        store = self.build_chain()
        status, page1 = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [a["operation"]["operationId"] for a in page1["right"]["predecessors"]],
            ["o1", "o2"],
        )
        status, page2 = store.get_causal_comparison("r1", "o4", "r2", "o5", 2, 2)
        self.assertIs(status, HTTPStatus.OK)
        # The left side has only two predecessors: its page is empty while
        # the right side continues with o3 and o4.
        self.assertEqual(page2["left"]["predecessors"], [])
        self.assertEqual(page2["left"]["cursor"], 2)
        self.assertIs(page2["left"]["more"], False)
        self.assertEqual(
            [a["operation"]["operationId"] for a in page2["right"]["predecessors"]],
            ["o3", "o4"],
        )
        self.assertEqual(page2["right"]["cursor"], 4)
        self.assertIs(page2["right"]["more"], False)
        # The full right pages partition the unpaged list in commit order.
        _, whole = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        self.assertEqual(
            page1["right"]["predecessors"] + page2["right"]["predecessors"],
            whole["right"]["predecessors"],
        )

    def test_after_at_the_larger_count_is_valid_for_both_sides(self) -> None:
        store = self.build_chain()
        # Left has 2 predecessors, right has 4: after=4 is the valid empty
        # tail for the larger side and an empty page for the smaller one.
        status, payload = store.get_causal_comparison("r1", "o4", "r2", "o5", 4, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["left"]["predecessors"], [])
        self.assertEqual(payload["right"]["predecessors"], [])
        self.assertIs(payload["left"]["more"], False)
        self.assertIs(payload["right"]["more"], False)

    def test_after_past_the_larger_count_is_rejected(self) -> None:
        store = self.build_chain()
        with self.assertRaises(ValueError):
            store.get_causal_comparison("r1", "o4", "r2", "o5", 5, 100)

    def test_empty_predecessor_sides_report_empty_arrays(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        status, payload = store.get_causal_comparison("r1", "o1", "r2", "o2", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["left"]["predecessors"], [])
        self.assertEqual(payload["right"]["predecessors"], [])
        self.assertEqual(payload["left"]["cursor"], 0)
        self.assertEqual(payload["right"]["cursor"], 0)
        self.assertEqual(payload["relation"], "concurrent")
        self.assertEqual(
            payload["difference"], {"shared": 0, "leftOnly": 0, "rightOnly": 0}
        )

    def test_query_is_read_only(self) -> None:
        store = self.build_chain()
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_snapshot = store.get_replication_snapshot()
        before_left = store.get_causal_ancestors("r1", "o4", 0, 100)
        store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 100)
        store.get_causal_comparison("absent", "x", "r2", "o5", 0, 100)
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_replication_snapshot(), before_snapshot)
        self.assertEqual(store.get_causal_ancestors("r1", "o4", 0, 100), before_left)

    def test_data_file_restart_preserves_comparison(self) -> None:
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
            expected = store.get_causal_comparison("r1", "o4", "r2", "o5", 0, 2)
            paged = store.get_causal_comparison("r1", "o4", "r2", "o5", 2, 2)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(
                recovered.get_causal_comparison("r1", "o4", "r2", "o5", 0, 2),
                expected,
            )
            self.assertEqual(
                recovered.get_causal_comparison("r1", "o4", "r2", "o5", 2, 2),
                paged,
            )
            self.assertEqual(
                recovered.get_causal_comparison("r1", "nope", "r2", "o5", 0, 100),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class ParseCausalCompareQueryTests(unittest.TestCase):
    """Unit-level query validation for the comparison endpoint."""

    BASE = (
        "leftReplicaId=r1&leftOperationId=o1&"
        "rightReplicaId=r2&rightOperationId=o2"
    )

    def test_minimal_query(self) -> None:
        self.assertEqual(
            parse_causal_compare_query(self.BASE),
            ("r1", "o1", "r2", "o2", 0, 100),
        )

    def test_full_query(self) -> None:
        self.assertEqual(
            parse_causal_compare_query(self.BASE + "&after=3&limit=7"),
            ("r1", "o1", "r2", "o2", 3, 7),
        )

    def test_missing_identity_parameter(self) -> None:
        self.assertIsNone(
            parse_causal_compare_query(
                "leftReplicaId=r1&leftOperationId=o1&rightReplicaId=r2"
            )
        )

    def test_unknown_parameter(self) -> None:
        self.assertIsNone(parse_causal_compare_query(self.BASE + "&x=1"))

    def test_repeated_parameter(self) -> None:
        self.assertIsNone(parse_causal_compare_query(self.BASE + "&after=1&after=2"))
        self.assertIsNone(
            parse_causal_compare_query(
                self.BASE.replace("leftReplicaId=r1", "leftReplicaId=r1&leftReplicaId=r9")
            )
        )

    def test_blank_values(self) -> None:
        self.assertIsNone(
            parse_causal_compare_query(
                "leftReplicaId=&leftOperationId=o1&rightReplicaId=r2&rightOperationId=o2"
            )
        )
        self.assertIsNone(parse_causal_compare_query(self.BASE + "&after="))
        self.assertIsNone(parse_causal_compare_query(self.BASE + "&limit="))

    def test_non_ascii_and_negative_numbers(self) -> None:
        for tail in (
            "&after=-1",
            "&after=1.0",
            "&after=%201",
            "&after=%EF%BC%91",  # fullwidth digit one
            "&limit=0",
            "&limit=101",
            "&limit=-1",
            "&limit=1.5",
        ):
            self.assertIsNone(parse_causal_compare_query(self.BASE + tail), tail)

    def test_empty_query(self) -> None:
        self.assertIsNone(parse_causal_compare_query(""))

    def test_percent_encoded_identity(self) -> None:
        query = (
            "leftReplicaId=r%2F1&leftOperationId=o%2F1&"
            "rightReplicaId=r2&rightOperationId=o2"
        )
        self.assertEqual(
            parse_causal_compare_query(query),
            ("r/1", "o/1", "r2", "o2", 0, 100),
        )

    def test_digit_run_past_the_interpreter_conversion_limit(self) -> None:
        # Python 3.11+ raises ValueError inside int() for runs over 4300
        # digits; such a token is a malformed paging value, not a server
        # error, for both the comparison parser and the shared paging rules.
        huge = "9" * 5000
        self.assertIsNone(parse_causal_compare_query(self.BASE + f"&after={huge}"))
        self.assertIsNone(parse_causal_compare_query(self.BASE + f"&limit={huge}"))
        self.assertIsNone(parse_paging_query(f"after={huge}"))
        self.assertIsNone(parse_paging_query(f"limit={huge}"))


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

    def compare(self, query: str):
        return self.request("GET", f"/v1/causal/compare?{query}")

    def compare_path(self, path: str):
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
        status, payload, headers, raw = self.compare(self.BASE_QUERY)
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"left", "right", "relation", "difference"})
        self.assertEqual(set(payload["left"]), {"operation", "predecessors", "cursor", "more"})
        self.assertEqual(set(payload["right"]), {"operation", "predecessors", "cursor", "more"})
        self.assertEqual(payload["relation"], "right_dominates_left")
        self.assertEqual(
            payload["left"]["operation"],
            source_record("r1", "o4", "d", "y", {"r1": 2, "r2": 1}),
        )
        self.assertEqual(
            payload["right"]["operation"],
            source_record("r2", "o5", "e", "z", {"r1": 2, "r2": 2, "r3": 1}),
        )
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["left"]["predecessors"]],
            [("o1", "transitive"), ("o2", "direct")],
        )
        self.assertEqual(
            [(a["operation"]["operationId"], a["relation"]) for a in payload["right"]["predecessors"]],
            [("o1", "transitive"), ("o2", "transitive"), ("o3", "direct"), ("o4", "direct")],
        )
        self.assertEqual(payload["left"]["cursor"], 2)
        self.assertEqual(payload["right"]["cursor"], 4)
        self.assertIs(payload["left"]["more"], False)
        self.assertIs(payload["right"]["more"], False)
        self.assertEqual(
            payload["difference"], {"shared": 2, "leftOnly": 0, "rightOnly": 2}
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
        status, _, _, raw = self.compare(
            "leftReplicaId=r1&leftOperationId=o1&rightReplicaId=r2&rightOperationId=o2"
        )
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertIn('"clé"', body)
        self.assertIn('"bléu"', body)
        self.assertNotIn("\\u", body)

    def test_numbers_are_json_integers(self) -> None:
        self.seed()
        status, _, _, raw = self.compare(self.BASE_QUERY + "&limit=2")
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
            status, payload, _, raw = self.compare(query)
            self.assertEqual(status, 404, query)
            self.assertEqual(payload, {"error": "not_found"}, query)
            self.assertEqual(raw, b'{"error":"not_found"}\n', query)

    def test_identity_parameters_are_percent_decoded(self) -> None:
        op = operation("o/1", "k", "v", {"r/1": 1})
        status, _, _, _ = self.request("POST", "/v1/replicas/r%2F1/operations", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.compare(
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
        status, payload, _, _ = self.compare(self.BASE_QUERY)
        self.assertEqual(status, 200)
        self.assertEqual(payload["left"]["cursor"], 2)
        self.assertEqual(payload["right"]["cursor"], 4)

    def test_shared_paging_over_http(self) -> None:
        self.seed()
        status, page1, _, _ = self.compare(self.BASE_QUERY + "&after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(page1["left"]["predecessors"]), 2)
        self.assertEqual(len(page1["right"]["predecessors"]), 2)
        self.assertIs(page1["left"]["more"], False)
        self.assertIs(page1["right"]["more"], True)
        status, page2, _, _ = self.compare(self.BASE_QUERY + "&after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(page2["left"]["predecessors"], [])
        self.assertEqual(
            [a["operation"]["operationId"] for a in page2["right"]["predecessors"]],
            ["o3", "o4"],
        )
        self.assertEqual(page2["right"]["cursor"], 4)
        self.assertIs(page2["right"]["more"], False)
        status, tail, _, _ = self.compare(self.BASE_QUERY + "&after=4")
        self.assertEqual(status, 200)
        self.assertEqual(tail["left"]["predecessors"], [])
        self.assertEqual(tail["right"]["predecessors"], [])

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
            base + "&after=5",  # past the larger predecessor count of 4
            base + "&limit=0",
            base + "&limit=101",
            base + "&limit=",
            base + "&limit=-1",
            base + "&limit=1.5",
            base + "&after=" + "9" * 5000,  # beyond int() digit conversion cap
        ):
            status, payload, _, _ = self.compare(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.seed()
        query = (
            "?leftReplicaId=r1&leftOperationId=o4&"
            "rightReplicaId=r2&rightOperationId=o5"
        )
        for path in (
            "/v1/causal/compare/",
            "/v1/causal/compare/extra",
            "/v1/causal",
            "/v2/causal/compare",
            "/v1/causal/compare" + query.replace("?", "/x?"),
        ):
            status, payload, _, _ = self.compare_path(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_query_and_identity_errors(self) -> None:
        self.seed()
        for path in (
            "/v1/causal/compare/?x=1",
            "/v1/causal/compare/extra?x=1",
            "/v1/causal/compare/extra?leftReplicaId=r1",
            "/v2/causal/compare?x=1",
        ):
            status, payload, _, _ = self.compare_path(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_single_operation_route_still_matches(self) -> None:
        self.seed()
        # The compare reservation on the third segment must not shadow the
        # single-operation chain for an unrelated replica id.
        status, payload, _, _ = self.compare_path("/v1/causal/r2/o5")
        self.assertEqual(status, 200)
        self.assertEqual(
            [a["operation"]["operationId"] for a in payload["ancestors"]],
            ["o1", "o2", "o3", "o4"],
        )

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.seed()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_compare, _, _, _ = self.compare(self.BASE_QUERY)
        for query in (
            self.BASE_QUERY + "&x=1",
            self.BASE_QUERY + "&after=9",
        ):
            self.compare(query)
        self.compare(
            "leftReplicaId=r1&leftOperationId=nope&rightReplicaId=r2&rightOperationId=o5"
            "&x=1"
        )
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_compare, _, _, _ = self.compare(self.BASE_QUERY)
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_compare, after_compare)

    def test_post_to_compare_route_is_404(self) -> None:
        self.seed()
        status, payload, _, _ = self.request("POST", "/v1/causal/compare", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpCausalCompareAuthTests(unittest.TestCase):
    """With auth enabled the comparison endpoint authenticates like any GET."""

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

    def compare_path(self, auth):
        return self.request(
            "GET",
            "/v1/causal/compare?leftReplicaId=r1&leftOperationId=o1"
            "&rightReplicaId=r1&rightOperationId=o1",
            auth=auth,
        )

    def test_compare_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Sekret"):
            status, payload, headers = self.compare_path(auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(headers.get("WWW-Authenticate"), "Bearer", auth)
        status, payload, _ = self.compare_path("Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(payload["relation"], "concurrent")

    def test_duplicate_authorization_header_is_401(self) -> None:
        path = (
            "/v1/causal/compare?leftReplicaId=r1&leftOperationId=o1"
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


class HttpCausalComparePersistenceTests(unittest.TestCase):
    """The comparison report survives a data-file restart unchanged."""

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
                "/v1/causal/compare?leftReplicaId=r1&leftOperationId=o4"
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
                    ("GET", query + "&after=1&limit=1", None),
                    (
                        "GET",
                        "/v1/causal/compare?leftReplicaId=r1&leftOperationId=nope"
                        "&rightReplicaId=r2&rightOperationId=o5",
                        None,
                    ),
                    ("GET", query + "&x=1", None),
                    ("GET", "/v1/causal/compare/extra", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[5][1])
            self.assertEqual(second[1][0], 200)
            page = json.loads(second[1][1].decode("utf-8"))
            # after=1 skips o1 on both sides, so both one-entry pages show o2.
            self.assertEqual(
                [a["operation"]["operationId"] for a in page["left"]["predecessors"]],
                ["o2"],
            )
            self.assertEqual(
                [a["operation"]["operationId"] for a in page["right"]["predecessors"]],
                ["o2"],
            )
            self.assertEqual(second[2], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[3], (400, b'{"error":"invalid_request"}\n'))
            # A shape 404 is answered by the generic route fallthrough
            # without a trailing newline, like every other unknown route.
            self.assertEqual(second[4], (404, b'{"error":"not_found"}'))
            # The read-only queries created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
