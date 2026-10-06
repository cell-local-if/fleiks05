"""Tests for the read-only per-key conflict-pressure report endpoint::

    GET /v1/metrics/conflicts/keys

It returns exactly four outer fields — ``entries``, ``hasMore``,
``nextCursor``, ``summary`` — with each entry carrying exactly ``key``,
``candidates``, ``candidatePairs``, ``conflictPairs`` and
``distinctValues``. A key qualifies only when at least one candidate pair
disagrees on the value; qualifying keys are ordered by conflictPairs
descending, candidates descending, then key Unicode code point ascending.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
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
REPORT_FIELDS = {"entries", "hasMore", "nextCursor", "summary"}
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


def write_conflict_pair(store: StateStore, key: str, index: int = 0) -> None:
    """Give ``key`` one conflict pair from two concurrent candidates."""
    store.apply_operation(
        "r1", operation(f"op-{key}-{index}-a", key, "a", {"r1": index + 1})
    )
    store.apply_operation(
        "r2", operation(f"op-{key}-{index}-b", key, "b", {"r2": index + 1})
    )


class ConflictKeyReportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_store_reports_empty_page(self) -> None:
        self.assertEqual(self.store.get_conflict_key_report(0, 100), EMPTY_REPORT)

    def test_single_candidate_key_does_not_qualify(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(self.store.get_conflict_key_report(0, 100), EMPTY_REPORT)

    def test_same_value_concurrent_candidates_do_not_qualify(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "same", {"r3": 1}))
        self.assertEqual(self.store.get_conflict_key_report(0, 100), EMPTY_REPORT)

    def test_disagreeing_pair_qualifies_with_exact_counts(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        report = self.store.get_conflict_key_report(0, 100)
        self.assertEqual(set(report), REPORT_FIELDS)
        self.assertEqual(set(report["summary"]), SUMMARY_FIELDS)
        self.assertEqual(
            report,
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

    def test_conflict_pairs_count_only_value_mismatches(self) -> None:
        # Candidates a, b, a: three unordered pairs, only the two a-b pairs
        # are conflict pairs; distinct values is 2.
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "k", "a", {"r3": 1}))
        entry = self.store.get_conflict_key_report(0, 100)["entries"][0]
        self.assertEqual(
            entry,
            {
                "key": "k",
                "candidates": 3,
                "candidatePairs": 3,
                "conflictPairs": 2,
                "distinctValues": 2,
            },
        )

    def test_ordering_conflict_pairs_then_candidates_then_key(self) -> None:
        # big1: three distinct values -> 3 conflict pairs, 3 candidates.
        self.store.apply_operation("r1", operation("o1", "big1", "x", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "big1", "y", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "big1", "z", {"r3": 1}))
        # big2: three a's and one b -> 3 conflict pairs, 4 candidates, so it
        # must outrank big1 despite its key.
        self.store.apply_operation("r1", operation("o4", "big2", "a", {"r1": 2}))
        self.store.apply_operation("r2", operation("o5", "big2", "a", {"r2": 2}))
        self.store.apply_operation("r3", operation("o6", "big2", "a", {"r3": 2}))
        self.store.apply_operation("r4", operation("o7", "big2", "b", {"r4": 1}))
        # mid: a, b, a -> 2 conflict pairs.
        self.store.apply_operation("r1", operation("o8", "mid", "a", {"r1": 3}))
        self.store.apply_operation("r2", operation("o9", "mid", "b", {"r2": 3}))
        self.store.apply_operation("r3", operation("o10", "mid", "a", {"r3": 3}))
        # Two one-conflict-pair keys: tie broken by key code point.
        # 'a' (U+0061) sorts before 'é' (U+00E9).
        write_conflict_pair(self.store, "ké")
        write_conflict_pair(self.store, "ka")
        # A same-value key that must never appear.
        self.store.apply_operation("r1", operation("o11", "quiet", "q", {"r1": 4}))
        self.store.apply_operation("r2", operation("o12", "quiet", "q", {"r2": 4}))

        keys = [
            (e["key"], e["conflictPairs"], e["candidates"])
            for e in self.store.get_conflict_key_report(0, 100)["entries"]
        ]
        self.assertEqual(
            keys,
            [
                ("big2", 3, 4),
                ("big1", 3, 3),
                ("mid", 2, 3),
                ("ka", 1, 2),
                ("ké", 1, 2),
            ],
        )

    def test_summary_conflict_pairs_matches_the_overview_total(self) -> None:
        write_conflict_pair(self.store, "k1")
        write_conflict_pair(self.store, "k2", index=1)
        self.store.apply_operation("r3", operation("o5", "same", "x", {"r3": 5}))
        self.store.apply_operation("r4", operation("o6", "same", "x", {"r4": 5}))
        report = self.store.get_conflict_key_report(0, 100)
        overview = self.store.get_conflict_metrics()
        self.assertEqual(report["summary"]["conflictKeys"], overview["conflictKeys"])
        self.assertEqual(
            report["summary"]["conflictPairs"], overview["conflictPairs"]
        )
        self.assertEqual(report["summary"]["conflictKeys"], 2)
        self.assertEqual(report["summary"]["conflictPairs"], 2)

    def test_paging_window_and_summary_covers_the_whole_snapshot(self) -> None:
        for index in range(5):
            write_conflict_pair(self.store, f"k{index}", index=index)
        first = self.store.get_conflict_key_report(0, 2)
        self.assertEqual(len(first["entries"]), 2)
        self.assertEqual(first["nextCursor"], 2)
        self.assertTrue(first["hasMore"])
        self.assertEqual(first["summary"]["conflictKeys"], 5)
        self.assertEqual(first["summary"]["conflictPairs"], 5)
        self.assertEqual(first["summary"]["returnedKeys"], 2)
        second = self.store.get_conflict_key_report(2, 2)
        self.assertEqual(len(second["entries"]), 2)
        self.assertEqual(second["nextCursor"], 4)
        self.assertTrue(second["hasMore"])
        self.assertEqual(second["summary"]["returnedKeys"], 2)
        last = self.store.get_conflict_key_report(4, 2)
        self.assertEqual(len(last["entries"]), 1)
        self.assertEqual(last["nextCursor"], 5)
        self.assertFalse(last["hasMore"])
        self.assertEqual(last["summary"]["returnedKeys"], 1)
        # The page slices concatenate back into the full ordered report.
        full = self.store.get_conflict_key_report(0, 100)
        self.assertEqual(
            first["entries"] + second["entries"] + last["entries"],
            full["entries"],
        )

    def test_after_equal_to_total_is_a_stable_empty_page(self) -> None:
        write_conflict_pair(self.store, "k")
        report = self.store.get_conflict_key_report(1, 100)
        self.assertEqual(report["entries"], [])
        self.assertFalse(report["hasMore"])
        self.assertEqual(report["nextCursor"], 1)
        self.assertEqual(report["summary"]["returnedKeys"], 0)
        self.assertEqual(report["summary"]["conflictKeys"], 1)

    def test_after_past_total_raises_value_error(self) -> None:
        write_conflict_pair(self.store, "k")
        with self.assertRaises(ValueError):
            self.store.get_conflict_key_report(2, 100)
        with self.assertRaises(ValueError):
            StateStore().get_conflict_key_report(1, 100)

    def test_resolution_removes_the_key_from_the_report(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        self.assertEqual(
            self.store.get_conflict_key_report(0, 100)["summary"]["conflictKeys"], 1
        )
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
        self.assertEqual(self.store.get_conflict_key_report(0, 100), EMPTY_REPORT)

    def test_read_is_side_effect_free(self) -> None:
        write_conflict_pair(self.store, "k")
        before = self.store.get_sync_operations(0, 100)[0]
        first = self.store.get_conflict_key_report(0, 100)
        for _ in range(3):
            self.assertEqual(
                self.store.get_conflict_key_report(0, 100), first
            )
        self.assertEqual(self.store.get_sync_operations(0, 100)[0], before)
        # The existing reads are untouched by the new one.
        self.assertEqual(self.store.get_conflict_metrics()["conflictKeys"], 1)
        self.assertEqual(self.store.get_metrics()["conflictKeys"], 1)


class ConflictKeyReportRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_report_matches_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        for index in range(5):
            write_conflict_pair(store, f"k{index}", index=index)
        store.apply_operation("r3", operation("o99", "quiet", "x", {"r3": 9}))
        store.apply_operation("r4", operation("o100", "quiet", "x", {"r4": 9}))
        before_pages = [
            store.get_conflict_key_report(after, 2) for after in (0, 2, 4)
        ]
        before_full = store.get_conflict_key_report(0, 100)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(
            recovered.get_conflict_key_report(0, 100), before_full
        )
        self.assertEqual(
            [recovered.get_conflict_key_report(after, 2) for after in (0, 2, 4)],
            before_pages,
        )


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

    def report(self, path: str = "/v1/metrics/conflicts/keys") -> tuple[int, dict]:
        return self.request("GET", path)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def seed_conflict_keys(self, count: int) -> None:
        for index in range(count):
            key = f"k{index:03d}"
            self.post_operation(
                "r1", operation(f"op-{index}-a", key, "a", {"r1": index + 1})
            )
            self.post_operation(
                "r2", operation(f"op-{index}-b", key, "b", {"r2": index + 1})
            )

    def test_empty_report(self) -> None:
        status, payload, raw, _ = self.raw_request("GET", "/v1/metrics/conflicts/keys")
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_REPORT)
        # The contract fixes the outer field order.
        self.assertTrue(raw.startswith(b'{"entries"'))
        self.assertIn(b'"hasMore"', raw)
        self.assertIn(b'"nextCursor"', raw)
        self.assertTrue(raw.endswith(b'}}'))

    def test_payload_shape_and_field_types(self) -> None:
        self.seed_conflict_keys(2)
        status, payload, raw, headers = self.raw_request(
            "GET", "/v1/metrics/conflicts/keys"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), REPORT_FIELDS)
        self.assertEqual(set(payload["summary"]), SUMMARY_FIELDS)
        self.assertEqual(len(payload["entries"]), 2)
        for entry in payload["entries"]:
            self.assertEqual(set(entry), ENTRY_FIELDS)
            self.assertIs(type(entry["key"]), str)
            for name in ("candidates", "candidatePairs", "conflictPairs", "distinctValues"):
                self.assertIs(type(entry[name]), int, name)
                self.assertGreaterEqual(entry[name], 0)
        for name in ("nextCursor",):
            self.assertIs(type(payload[name]), int)
        self.assertIs(type(payload["hasMore"]), bool)
        for name, value in payload["summary"].items():
            self.assertIs(type(value), int, name)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_defaults_and_paging(self) -> None:
        self.seed_conflict_keys(3)
        status, page0 = self.report("/v1/metrics/conflicts/keys?limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["key"] for e in page0["entries"]], ["k000", "k001"])
        self.assertTrue(page0["hasMore"])
        self.assertEqual(page0["nextCursor"], 2)
        self.assertEqual(page0["summary"], {"conflictKeys": 3, "conflictPairs": 3, "returnedKeys": 2})

        status, page1 = self.report("/v1/metrics/conflicts/keys?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["key"] for e in page1["entries"]], ["k002"])
        self.assertFalse(page1["hasMore"])
        self.assertEqual(page1["nextCursor"], 3)
        self.assertEqual(page1["summary"]["returnedKeys"], 1)

        # Default limit is 100 and default after is 0.
        status, full = self.report("/v1/metrics/conflicts/keys?")
        self.assertEqual(status, 200)
        self.assertEqual(len(full["entries"]), 3)
        self.assertEqual(full["nextCursor"], 3)
        self.assertFalse(full["hasMore"])

    def test_limit_boundaries_1_and_100_are_accepted(self) -> None:
        self.seed_conflict_keys(1)
        for path in (
            "/v1/metrics/conflicts/keys?limit=1",
            "/v1/metrics/conflicts/keys?limit=100",
            "/v1/metrics/conflicts/keys?after=0&limit=100",
        ):
            status, _ = self.report(path)
            self.assertEqual(status, 200, path)

    def test_after_equal_to_total_is_empty_page(self) -> None:
        self.seed_conflict_keys(2)
        status, payload = self.report("/v1/metrics/conflicts/keys?after=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertEqual(payload["summary"]["returnedKeys"], 0)
        self.assertEqual(payload["summary"]["conflictKeys"], 2)

    def test_invalid_query_is_400(self) -> None:
        bad_paths = (
            "/v1/metrics/conflicts/keys?x=1",
            "/v1/metrics/conflicts/keys?after",
            "/v1/metrics/conflicts/keys?after=",
            "/v1/metrics/conflicts/keys?after=-1",
            "/v1/metrics/conflicts/keys?after=1%20",
            "/v1/metrics/conflicts/keys?after=1.5",
            "/v1/metrics/conflicts/keys?after=%2B1",
            "/v1/metrics/conflicts/keys?after=%EF%BC%91",  # fullwidth digit one
            "/v1/metrics/conflicts/keys?limit=0",
            "/v1/metrics/conflicts/keys?limit=101",
            "/v1/metrics/conflicts/keys?limit=-1",
            "/v1/metrics/conflicts/keys?limit=1%20",
            "/v1/metrics/conflicts/keys?limit=abc",
            "/v1/metrics/conflicts/keys?after=1&after=2",
            "/v1/metrics/conflicts/keys?limit=1&limit=2",
            "/v1/metrics/conflicts/keys?after=0&x=1",
            "/v1/metrics/conflicts/keys?=1",
        )
        for path in bad_paths:
            status, payload, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_after_past_total_is_400(self) -> None:
        self.seed_conflict_keys(1)
        status, payload = self.report("/v1/metrics/conflicts/keys?after=2")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        # An empty store rejects any positive after as well.
        self.server.store = type(self.server.store)()
        status, payload = self.report("/v1/metrics/conflicts/keys?after=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_wrong_path_shape_is_404(self) -> None:
        for path in (
            "/v1/metrics/conflicts/keys/extra",
            "/v1/metrics/conflicts/keys/",
            "/v1/metrics/conflicts/key",
            "/v1/metrics/keys",
            "/metrics/conflicts/keys",
            "/v1/conflicts/keys",
        ):
            status, payload = self.report(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_precedes_query_validation(self) -> None:
        status, payload = self.report(
            "/v1/metrics/conflicts/keys/extra?after=not-a-number"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload = self.report("/v1/metrics/conflicts/keys/?after=1")
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
        self.seed_conflict_keys(2)
        _, first = self.report()
        for _ in range(3):
            status, payload = self.report()
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 4)
        status, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["conflictKeys"], 2)

    def test_concurrent_commits_always_observe_a_consistent_snapshot(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                report = self.server.store.get_conflict_key_report(0, 100)
                entries = report["entries"]
                summary = report["summary"]
                if summary["returnedKeys"] != len(entries):
                    violations.append("returnedKeys mismatch")
                if summary["conflictKeys"] < len(entries):
                    violations.append("conflictKeys below page size")
                if report["nextCursor"] != len(entries):
                    violations.append("nextCursor mismatch")
                if report["hasMore"]:
                    violations.append("full page must not report hasMore")
                pairs = sum(e["conflictPairs"] for e in entries)
                if pairs != summary["conflictPairs"]:
                    violations.append("summary conflictPairs disagrees with page")
                for entry in entries:
                    if entry["conflictPairs"] > entry["candidatePairs"]:
                        violations.append("conflictPairs above candidatePairs")
                    if entry["distinctValues"] > entry["candidates"]:
                        violations.append("distinctValues above candidates")
                    if entry["conflictPairs"] == 0:
                        violations.append("non-qualifying key on the report")
                ordered_keys = [
                    (-e["conflictPairs"], -e["candidates"], e["key"])
                    for e in entries
                ]
                if ordered_keys != sorted(ordered_keys):
                    violations.append("entries not in contract order")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                key = f"shared-{index}"
                replica = f"rx{index}"
                self.post_operation(
                    "r1", operation(f"op-{index}-a", key, "a", {"r1": index + 1})
                )
                self.post_operation(
                    replica,
                    operation(f"op-{index}-b", key, f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])
        status, payload = self.report()
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["conflictKeys"], 40)


class ConflictKeyReportAuthTests(unittest.TestCase):
    """Authentication for GET /v1/metrics/conflicts/keys."""

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

    def test_bad_credential_precedes_query_validation(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys?limit=0", [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_token_is_200(self) -> None:
        status, payload = self.raw_get(
            "/v1/metrics/conflicts/keys", [("Authorization", f"Bearer {TOKEN}")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), REPORT_FIELDS)

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

    def get(self, token: str | None, query: str = "") -> tuple[int, dict]:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", f"/v1/metrics/conflicts/keys{query}", headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else None
        conn.close()
        return response.status, payload

    def test_read_scope_is_200(self) -> None:
        status, payload = self.get("reader-token")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), REPORT_FIELDS)

    def test_admin_scope_covers_reads(self) -> None:
        status, _ = self.get("admin-token")
        self.assertEqual(status, 200)

    def test_write_only_token_is_403(self) -> None:
        status, payload = self.get("writer-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})

    def test_missing_scope_precedes_query_validation(self) -> None:
        status, payload = self.get("writer-token", "?limit=0")
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

    def test_report_survives_restart(self) -> None:
        server = self.start_server()
        for index in range(5):
            key = f"k{index:03d}"
            self.request(
                server, "POST", "/v1/replicas/r1/operations",
                operation(f"op-{index}-a", key, "a", {"r1": index + 1}),
            )
            self.request(
                server, "POST", "/v1/replicas/r2/operations",
                operation(f"op-{index}-b", key, "b", {"r2": index + 1}),
            )
        pages_before = []
        for query in ("?limit=2", "?after=2&limit=2", "?after=4&limit=2"):
            status, page = self.request(
                server, "GET", f"/v1/metrics/conflicts/keys{query}"
            )
            self.assertEqual(status, 200)
            pages_before.append(page)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        pages_after = []
        for query in ("?limit=2", "?after=2&limit=2", "?after=4&limit=2"):
            status, page = self.request(
                server, "GET", f"/v1/metrics/conflicts/keys{query}"
            )
            self.assertEqual(status, 200)
            pages_after.append(page)
        self.assertEqual(pages_after, pages_before)

    def test_read_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "a", {"r1": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k", "b", {"r2": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()

        for query in ("", "?limit=1", "?after=1", "?after=0&limit=100"):
            status, _ = self.request(
                server, "GET", f"/v1/metrics/conflicts/keys{query}"
            )
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
