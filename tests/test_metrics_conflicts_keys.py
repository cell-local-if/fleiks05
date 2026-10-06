"""Tests for the read-only per-key conflict report endpoint::

    GET /v1/metrics/conflicts/keys

It returns a paginated report of the keys under conflict pressure —
those with at least one pair of candidates that disagree on the value —
computed from a single snapshot under the shared commit lock. Each page
is exactly::

    {"entries": [...], "hasMore": bool, "nextCursor": int,
     "summary": {"conflictKeys": int, "conflictPairs": int,
                 "returnedKeys": int}}

and each entry is exactly::

    {"key": str, "candidates": int, "candidatePairs": int,
     "conflictPairs": int, "distinctValues": int}

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot, ordering, and
recovery semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

ENTRY_FIELDS = {"key", "candidates", "candidatePairs", "conflictPairs", "distinctValues"}
RESPONSE_FIELDS = {"entries", "hasMore", "nextCursor", "summary"}
SUMMARY_FIELDS = {"conflictKeys", "conflictPairs", "returnedKeys"}

EMPTY_REPORT = {
    "entries": [],
    "hasMore": False,
    "nextCursor": 0,
    "summary": {"conflictKeys": 0, "conflictPairs": 0, "returnedKeys": 0},
}

TOKEN = "s3cret-token_123"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica: str, operation_id: str) -> dict:
    return {"replicaId": replica, "operationId": operation_id}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


def report(store: StateStore, after: int = 0, limit: int = 100) -> dict:
    entries, next_cursor, has_more, summary = store.get_conflict_key_metrics(
        after, limit
    )
    return {
        "entries": entries,
        "hasMore": has_more,
        "nextCursor": next_cursor,
        "summary": summary,
    }


class ConflictKeyReportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store_reports_an_empty_page(self) -> None:
        self.assertEqual(report(self.store), EMPTY_REPORT)

    def test_same_value_candidates_do_not_qualify(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        self.assertEqual(report(self.store), EMPTY_REPORT)

    def test_single_conflict_key_reports_its_counts(self) -> None:
        # Candidates (r1,o1)=a, (r2,o2)=b, (r3,o3)=a: three unordered
        # pairs, only the two a-b pairs are conflict pairs.
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "a", {"r3": 1}))
        self.assertEqual(
            report(self.store),
            {
                "entries": [
                    {
                        "key": "k",
                        "candidates": 3,
                        "candidatePairs": 3,
                        "conflictPairs": 2,
                        "distinctValues": 2,
                    }
                ],
                "hasMore": False,
                "nextCursor": 1,
                "summary": {"conflictKeys": 1, "conflictPairs": 2, "returnedKeys": 1},
            },
        )

    def test_entry_fields_are_exactly_the_five_contract_fields(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        entries, _, _, _ = self.store.get_conflict_key_metrics(0, 100)
        self.assertEqual(len(entries), 1)
        self.assertEqual(set(entries[0]), ENTRY_FIELDS)
        for name, value in entries[0].items():
            if name == "key":
                self.assertIs(type(value), str)
            else:
                self.assertIs(type(value), int, f"{name} must be an int")
                self.assertGreaterEqual(value, 0)

    def test_pairs_are_counted_once_in_identity_order(self) -> None:
        # Applied in reverse identity order; the pair enumeration sorts by
        # (replicaId, operationId) first, so the counts are unaffected.
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        entries, _, _, _ = self.store.get_conflict_key_metrics(0, 100)
        self.assertEqual(entries[0]["candidatePairs"], 1)
        self.assertEqual(entries[0]["conflictPairs"], 1)

    def test_keys_sort_by_pressure_then_candidates_then_key(self) -> None:
        # "bulk" and "busy" tie at 3 conflict pairs; "bulk" has 4
        # candidates (values x, x, x, y) against "busy"'s 3 (a, b, c), so
        # the candidate-count tiebreak pages "bulk" first.
        for replica, op_id, value, tick in (
            ("r1", "o1", "x", 1),
            ("r2", "o2", "x", 1),
            ("r3", "o3", "x", 1),
            ("r4", "o4", "y", 1),
        ):
            self.store.apply_operation(
                replica, operation(op_id, "bulk", value, {replica: tick})
            )
        for replica, op_id, value in (
            ("r1", "o5", "a"),
            ("r2", "o6", "b"),
            ("r3", "o7", "c"),
        ):
            self.store.apply_operation(
                replica, operation(op_id, "busy", value, {replica: 2})
            )
        # "wide" has values x, y, x (2 conflict pairs, 3 candidates) and
        # "alpha" 1 conflict pair with 2 candidates, so pressure orders
        # them after the two 3-pair keys.
        self.store.apply_operation("r1", operation("o8", "wide", "x", {"r1": 3}))
        self.store.apply_operation("r2", operation("o9", "wide", "y", {"r2": 3}))
        self.store.apply_operation("r3", operation("o10", "wide", "x", {"r3": 3}))
        self.store.apply_operation("r1", operation("o11", "alpha", "p", {"r1": 4}))
        self.store.apply_operation("r2", operation("o12", "alpha", "q", {"r2": 4}))
        # "quiet" never qualifies: its candidates agree on the value.
        self.store.apply_operation("r1", operation("o13", "quiet", "z", {"r1": 5}))
        self.store.apply_operation("r2", operation("o14", "quiet", "z", {"r2": 5}))
        entries, next_cursor, has_more, summary = self.store.get_conflict_key_metrics(
            0, 100
        )
        self.assertEqual(
            [entry["key"] for entry in entries], ["bulk", "busy", "wide", "alpha"]
        )
        self.assertEqual(
            [(entry["conflictPairs"], entry["candidates"]) for entry in entries],
            [(3, 4), (3, 3), (2, 3), (1, 2)],
        )
        self.assertEqual((next_cursor, has_more), (4, False))
        self.assertEqual(
            summary, {"conflictKeys": 4, "conflictPairs": 9, "returnedKeys": 4}
        )

    def test_key_tiebreak_uses_unicode_code_point_order(self) -> None:
        # Equal pressure and candidate counts: "Z" (U+005A) pages before
        # "a" (U+0061), and "é" (U+00E9) after both.
        for key in ("é", "a", "Z"):
            self.store.apply_operation(
                "r1", operation(f"o1-{key}", key, "v1", {"r1": 1})
            )
            self.store.apply_operation(
                "r2", operation(f"o2-{key}", key, "v2", {"r2": 1})
            )
        entries, _, _, _ = self.store.get_conflict_key_metrics(0, 100)
        self.assertEqual([entry["key"] for entry in entries], ["Z", "a", "é"])

    def test_pagination_walks_the_report_in_order(self) -> None:
        for index in range(5):
            key = f"k{index}"
            self.store.apply_operation(
                "r1", operation(f"a{index}", key, "x", {"r1": index + 1})
            )
            self.store.apply_operation(
                "r2", operation(f"b{index}", key, "y", {"r2": index + 1})
            )
        seen: list[str] = []
        after = 0
        pages = 0
        while True:
            entries, next_cursor, has_more, summary = (
                self.store.get_conflict_key_metrics(after, 2)
            )
            seen.extend(entry["key"] for entry in entries)
            self.assertEqual(next_cursor, after + len(entries))
            self.assertEqual(summary["conflictKeys"], 5)
            self.assertEqual(summary["conflictPairs"], 5)
            self.assertEqual(summary["returnedKeys"], len(entries))
            after = next_cursor
            pages += 1
            if not has_more:
                break
        self.assertEqual(pages, 3)
        self.assertEqual(seen, ["k0", "k1", "k2", "k3", "k4"])
        self.assertEqual(after, 5)

    def test_after_equal_to_total_returns_an_empty_page(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        entries, next_cursor, has_more, summary = self.store.get_conflict_key_metrics(
            1, 100
        )
        self.assertEqual(entries, [])
        self.assertEqual((next_cursor, has_more), (1, False))
        self.assertEqual(
            summary, {"conflictKeys": 1, "conflictPairs": 1, "returnedKeys": 0}
        )

    def test_after_past_the_total_is_rejected(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        with self.assertRaises(ValueError):
            self.store.get_conflict_key_metrics(2, 100)
        # The empty report has zero qualifying keys, so any cursor past 0
        # is out of range.
        with self.assertRaises(ValueError):
            StateStore().get_conflict_key_metrics(1, 100)

    def test_resolution_removes_the_key_from_the_report(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.assertEqual(report(self.store)["summary"]["conflictKeys"], 1)
        status, error = self.store.apply_resolution(
            "k",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.assertEqual((status, error), (201, None))
        self.assertEqual(report(self.store), EMPTY_REPORT)

    def test_read_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before = self.store.get_sync_operations(0, 100)[0]
        first = report(self.store)
        second = report(self.store)
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], before)
        # The existing metrics are untouched by the new read.
        self.assertEqual(self.store.get_metrics()["conflictKeys"], 1)
        self.assertEqual(self.store.get_conflict_metrics()["conflictPairs"], 1)


class ConflictKeyReportRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_report_and_pages_match_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "v1", {"r3": 1}))
        store.apply_operation("r4", operation("o4", "other", "x", {"r4": 1}))
        store.apply_operation("r5", operation("o5", "other", "y", {"r5": 1}))
        # A stale write that adds no candidate.
        store.apply_operation("r1", operation("o6", "k", "old", {"r1": 0}))
        before_full = report(store)
        before_paged = report(store, 1, 1)
        self.assertEqual([e["key"] for e in before_full["entries"]], ["k", "other"])

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(report(recovered), before_full)
        self.assertEqual(report(recovered, 1, 1), before_paged)
        del recovered
        reloaded = StateStore(data_file=self.data_file)
        self.assertEqual(report(reloaded), before_full)
        self.assertEqual(report(reloaded, 1, 1), before_paged)


class ConflictKeyReportHttpServerTests(unittest.TestCase):
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

    def raw_request(
        self, method: str, path: str, body: object = None
    ) -> tuple[int, dict, bytes, list[tuple[str, str]]]:
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
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        headers = response.getheaders()
        conn.close()
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def keys(self, path: str = "/v1/metrics/conflicts/keys") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_report_is_an_empty_page(self) -> None:
        status, payload = self.keys()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_REPORT)

    def test_response_shape_and_headers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload, raw, headers = self.raw_request(
            "GET", "/v1/metrics/conflicts/keys"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), RESPONSE_FIELDS)
        self.assertEqual(set(payload["summary"]), SUMMARY_FIELDS)
        self.assertEqual(len(payload["entries"]), 1)
        self.assertEqual(set(payload["entries"][0]), ENTRY_FIELDS)
        self.assertIs(type(payload["hasMore"]), bool)
        self.assertIs(type(payload["nextCursor"]), int)
        for name, value in payload["summary"].items():
            self.assertIs(type(value), int, f"{name} must be an int")
            self.assertGreaterEqual(value, 0)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_counts_reflect_writes_and_repairs(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = self.keys()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "entries": [
                    {
                        "key": "k",
                        "candidates": 2,
                        "candidatePairs": 1,
                        "conflictPairs": 1,
                        "distinctValues": 2,
                    }
                ],
                "hasMore": False,
                "nextCursor": 1,
                "summary": {
                    "conflictKeys": 1,
                    "conflictPairs": 1,
                    "returnedKeys": 1,
                },
            },
        )
        status, _ = self.request(
            "POST",
            "/v1/states/k/resolve",
            resolution(
                "r3",
                "fix-1",
                "k",
                "merged",
                {"r1": 1, "r2": 1, "r3": 1},
                [candidate("r1", "o1"), candidate("r2", "o2")],
            ),
        )
        self.assertEqual(status, 201)
        status, payload = self.keys()
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_REPORT)

    def test_default_page_matches_explicit_defaults(self) -> None:
        for index in range(3):
            key = f"k{index}"
            self.post_operation("r1", operation(f"a{index}", key, "x", {"r1": 1}))
            self.post_operation("r2", operation(f"b{index}", key, "y", {"r2": 1}))
        _, implicit = self.keys()
        _, explicit = self.keys("/v1/metrics/conflicts/keys?after=0&limit=100")
        self.assertEqual(implicit, explicit)
        self.assertEqual(len(implicit["entries"]), 3)
        self.assertEqual(implicit["hasMore"], False)
        self.assertEqual(implicit["nextCursor"], 3)

    def test_pagination_over_http(self) -> None:
        for index in range(3):
            key = f"k{index}"
            self.post_operation("r1", operation(f"a{index}", key, "x", {"r1": 1}))
            self.post_operation("r2", operation(f"b{index}", key, "y", {"r2": 1}))
        status, first = self.keys("/v1/metrics/conflicts/keys?limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["key"] for e in first["entries"]], ["k0", "k1"])
        self.assertEqual(first["hasMore"], True)
        self.assertEqual(first["nextCursor"], 2)
        self.assertEqual(
            first["summary"],
            {"conflictKeys": 3, "conflictPairs": 3, "returnedKeys": 2},
        )
        status, second = self.keys("/v1/metrics/conflicts/keys?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["key"] for e in second["entries"]], ["k2"])
        self.assertEqual(second["hasMore"], False)
        self.assertEqual(second["nextCursor"], 3)
        self.assertEqual(
            second["summary"],
            {"conflictKeys": 3, "conflictPairs": 3, "returnedKeys": 1},
        )
        # A cursor exactly at the total is a valid empty page.
        status, third = self.keys("/v1/metrics/conflicts/keys?after=3")
        self.assertEqual(status, 200)
        self.assertEqual(third["entries"], [])
        self.assertEqual(third["hasMore"], False)
        self.assertEqual(third["nextCursor"], 3)
        self.assertEqual(
            third["summary"],
            {"conflictKeys": 3, "conflictPairs": 3, "returnedKeys": 0},
        )

    def test_bad_query_parameters_are_400(self) -> None:
        for path in (
            "/v1/metrics/conflicts/keys?x=1",
            "/v1/metrics/conflicts/keys?keys=1",
            "/v1/metrics/conflicts/keys?after=1&after=2",
            "/v1/metrics/conflicts/keys?limit=1&limit=2",
            "/v1/metrics/conflicts/keys?after=",
            "/v1/metrics/conflicts/keys?limit=",
            "/v1/metrics/conflicts/keys?after",
            "/v1/metrics/conflicts/keys?=1",
            "/v1/metrics/conflicts/keys?after=-1",
            "/v1/metrics/conflicts/keys?limit=-5",
            "/v1/metrics/conflicts/keys?after=+1",
            "/v1/metrics/conflicts/keys?after=1.0",
            "/v1/metrics/conflicts/keys?after=%D9%A1",  # non-ASCII decimal digit
            "/v1/metrics/conflicts/keys?limit=0",
            "/v1/metrics/conflicts/keys?limit=101",
            "/v1/metrics/conflicts/keys?after=0&limit=1&x=2",
        ):
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_after_past_the_total_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        # One qualifying key: a cursor of 1 is the valid empty page, any
        # larger cursor is out of range.
        status, payload = self.keys("/v1/metrics/conflicts/keys?after=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["hasMore"], False)
        status, payload = self.keys("/v1/metrics/conflicts/keys?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_report_rejects_any_cursor_past_zero(self) -> None:
        status, payload = self.keys("/v1/metrics/conflicts/keys?after=1&limit=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.keys("/v1/metrics/conflicts/keys?")
        self.assertEqual(status, 200)

    def test_wrong_path_shape_is_404(self) -> None:
        for path in (
            "/v1/metrics/conflicts/keys/extra",
            "/v1/metrics/conflicts/keys/",
            "/v1/metrics/conflicts/key",
            "/metrics/conflicts/keys",
            "/v1/conflicts/keys",
        ):
            status, payload = self.keys(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_wrong_path_shape_is_404_even_with_bad_query(self) -> None:
        # Route matching runs before query validation.
        status, payload = self.keys("/v1/metrics/conflicts/keys/extra?after=-1")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload = self.keys("/v1/metrics/conflicts/keys/?limit=0")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_non_get_methods_are_404(self) -> None:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            status, payload, _, _ = self.raw_request(
                method, "/v1/metrics/conflicts/keys"
            )
            self.assertEqual(status, 404, method)
            self.assertEqual(payload, {"error": "not_found"}, method)

    def test_read_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, first = self.keys()
        for _ in range(3):
            status, payload = self.keys()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        # The accepted-operation log and the existing metrics are untouched.
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 2)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["conflictKeys"], 1)
        self.assertEqual(metrics["candidateVersions"], 2)
        status, conflicts = self.request("GET", "/v1/metrics/conflicts")
        self.assertEqual(status, 200)
        self.assertEqual(conflicts["conflictPairs"], 1)

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                entries, next_cursor, has_more, summary = (
                    self.server.store.get_conflict_key_metrics(0, 100)
                )
                if summary["conflictKeys"] != len(entries) and not has_more:
                    violations.append("conflictKeys != final page length")
                if summary["returnedKeys"] != len(entries):
                    violations.append("returnedKeys != page length")
                if next_cursor != len(entries):
                    violations.append("nextCursor != skipped + returned")
                total_pairs = sum(entry["conflictPairs"] for entry in entries)
                if summary["conflictPairs"] != total_pairs and not has_more:
                    violations.append("summary conflictPairs != entry sum")
                for entry in entries:
                    if entry["conflictPairs"] > entry["candidatePairs"]:
                        violations.append("conflictPairs > candidatePairs")
                    if entry["distinctValues"] > entry["candidates"]:
                        violations.append("distinctValues > candidates")
                    if entry["distinctValues"] < 2:
                        violations.append("non-qualifying key reported")
                ordered = sorted(
                    entries,
                    key=lambda entry: (
                        -entry["conflictPairs"],
                        -entry["candidates"],
                        entry["key"],
                    ),
                )
                if ordered != entries:
                    violations.append("entries out of order")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                replica = f"r{index}"
                self.post_operation(
                    replica,
                    operation(f"op-{index}", "shared", f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.keys()
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["entries"]), 1)
        entry = payload["entries"][0]
        self.assertEqual(entry["key"], "shared")
        self.assertEqual(entry["candidates"], 40)
        # 40 concurrent distinct-value candidates: C(40, 2) pairs, all conflicts.
        self.assertEqual(entry["candidatePairs"], 780)
        self.assertEqual(entry["conflictPairs"], 780)
        self.assertEqual(entry["distinctValues"], 40)
        self.assertEqual(
            payload["summary"],
            {"conflictKeys": 1, "conflictPairs": 780, "returnedKeys": 1},
        )


class ConflictKeyReportAuthTests(unittest.TestCase):
    """Authentication and scope enforcement for GET /v1/metrics/conflicts/keys."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token=TOKEN
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def raw_get(self, path: str, headers: list[tuple[str, str]]) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        conn.close()
        return response.status, payload

    def test_missing_authorization_is_401(self) -> None:
        status, payload = self.raw_get("/v1/metrics/conflicts/keys", [])
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_wrong_token_is_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys", [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_malformed_authorization_is_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys", [("Authorization", TOKEN)]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_duplicate_authorization_headers_are_401(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys",
            [
                ("Authorization", f"Bearer {TOKEN}"),
                ("Authorization", f"Bearer {TOKEN}"),
            ],
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_token_is_200(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys", [("Authorization", f"Bearer {TOKEN}")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), RESPONSE_FIELDS)

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.raw_get("/health", [])
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


class ConflictKeyReportScopePolicyTests(unittest.TestCase):
    """Scope-policy mode: read or admin scope is required, write is not enough."""

    POLICY = {
        "reader-token": frozenset({"read"}),
        "writer-token": frozenset({"write"}),
        "admin-token": frozenset({"read", "write", "admin"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_scopes=dict(cls.POLICY)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, token: str | None) -> tuple[int, dict]:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/v1/metrics/conflicts/keys", headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        conn.close()
        return response.status, payload

    def test_read_scope_is_200(self) -> None:
        status, payload = self.get("reader-token")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), RESPONSE_FIELDS)

    def test_admin_scope_covers_reads(self) -> None:
        status, _ = self.get("admin-token")
        self.assertEqual(status, 200)

    def test_write_only_token_is_403(self) -> None:
        status, payload = self.get("writer-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_unknown_token_is_401(self) -> None:
        status, payload = self.get("unknown-token")
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})


class PersistentConflictKeyReportHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(
        self, server: SemanticStateServer, method: str, path: str, body: object = None
    ) -> tuple[int, dict]:
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
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
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_report_and_pages_survive_restart(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v1", {"r1": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k", "v2", {"r2": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r3/operations",
            operation("o3", "other", "x", {"r3": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r4/operations",
            operation("o4", "other", "y", {"r4": 1}),
        )
        status, before = self.request(server, "GET", "/v1/metrics/conflicts/keys")
        self.assertEqual(status, 200)
        status, before_page = self.request(
            server, "GET", "/v1/metrics/conflicts/keys?after=1&limit=1"
        )
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "GET", "/v1/metrics/conflicts/keys")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        status, after_page = self.request(
            server, "GET", "/v1/metrics/conflicts/keys?after=1&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(after_page, before_page)

    def test_read_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for _ in range(5):
            status, _ = self.request(server, "GET", "/v1/metrics/conflicts/keys")
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
