"""Tests for the read-only cross-replica snapshot comparison endpoint::

    POST /v1/replication/compare

The request body carries only the remote replica id and a complete
candidate snapshot keyed by business key; each remote candidate keeps its
value, vector clock, and operation identity and obeys the existing write
constraints (non-boolean, non-negative integer clock components; no
repeated identity; no unknown fields; JSON integers only). The response
groups per-candidate differences by business key into ``shared``,
``localOnly``, ``remoteOnly``, and ``contentConflict``, sorts every group
by business key and identity, marks missing / content-conflict /
clock-cover relations, and summarizes both sides' key and candidate
counts, whether the candidate digests match, and the minimal difference
count. The comparison is strictly read-only and observes one committed
local snapshot.

Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    replication_compare_report,
    parse_replication_compare_payload,
    _verification_digest_input,
)

COMPARE_PATH = "/v1/replication/compare"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(
    replica: str, operation_id: str, value: str, clock: dict
) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica,
        "operationId": operation_id,
    }


def compare_body(replica_id: str, snapshot: dict) -> dict:
    return {"replicaId": replica_id, "snapshot": snapshot}


def candidate_digest(snapshot: dict) -> str:
    return hashlib.sha256(_verification_digest_input(snapshot)).hexdigest()


class ParseComparePayloadTests(unittest.TestCase):
    def parse(self, raw):
        return parse_replication_compare_payload(raw)

    def test_valid_body_is_normalized(self) -> None:
        body = compare_body(
            "remote-r",
            {
                "color": [
                    candidate("r1", "o1", "blue", {"r1": 1}),
                    candidate("r2", "o2", "red", {"r2": 1, "r1": 1}),
                ],
                "shape": [candidate("r3", "o3", "round", {"r3": 7})],
            },
        )
        replica_id, snapshot = self.parse(json.dumps(body))
        self.assertEqual(replica_id, "remote-r")
        self.assertEqual(set(snapshot), {"color", "shape"})
        self.assertEqual(
            snapshot["color"][0], candidate("r1", "o1", "blue", {"r1": 1})
        )
        # Clock component names are fixed in lexicographic order.
        self.assertEqual(
            snapshot["color"][1]["clock"], {"r1": 1, "r2": 1}
        )

    def test_empty_snapshot_is_valid(self) -> None:
        replica_id, snapshot = self.parse(
            json.dumps(compare_body("remote-r", {}))
        )
        self.assertEqual(replica_id, "remote-r")
        self.assertEqual(snapshot, {})

    def test_malformed_and_wrong_shaped_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"[]",
            b"null",
            b'{"replicaId":"r"}',
            b'{"snapshot":{}}',
            b'{"replicaId":"r","snapshot":{},"extra":1}',
            b'{"replicaId":"","snapshot":{}}',
            b'{"replicaId":7,"snapshot":{}}',
            b'{"replicaId":null,"snapshot":{}}',
            b'{"replicaId":"r","snapshot":[]}',
            b'{"replicaId":"r","snapshot":null}',
            b'{"replicaId":"r","snapshot":{"":[]}}',
            b'{"replicaId":"r","snapshot":{"k":[]}}',
            b'{"replicaId":"r","snapshot":{"k":"x"}}',
            b'{"replicaId":"r","snapshot":{"k":[{}]}}',
            b'{"replicaId":"r","snapshot":{"k":[[]]}}',
            b'{"replicaId":"r","snapshot":{"k":[null]}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.parse(raw)

    def test_candidate_fields_and_types_are_enforced(self) -> None:
        base = candidate("r1", "o1", "v", {"r1": 1})
        bad_candidates = [
            {**base, "extra": 1},
            {"value": "v", "clock": {"r1": 1}, "replicaId": "r1"},
            {"value": "", "clock": {"r1": 1}, "replicaId": "r1", "operationId": "o1"},
            {"value": 5, "clock": {"r1": 1}, "replicaId": "r1", "operationId": "o1"},
            {"value": "v", "clock": {"r1": 1}, "replicaId": "", "operationId": "o1"},
            {"value": "v", "clock": {"r1": 1}, "replicaId": 3, "operationId": "o1"},
            {"value": "v", "clock": {"r1": 1}, "replicaId": "r1", "operationId": ""},
            {"value": "v", "clock": {"r1": 1}, "replicaId": "r1", "operationId": 9},
        ]
        for bad in bad_candidates:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.parse(json.dumps(compare_body("r", {"k": [bad]})))

    def test_clocks_follow_the_live_write_constraints(self) -> None:
        for clock in (
            {},
            {"": 1},
            {"r1": -1},
            {"r1": True},
            {"r1": "1"},
            {"r1": None},
            {"r2": 1},  # must contain the candidate's own replicaId
            [],
        ):
            bad = compare_body("r", {"k": [candidate("r1", "o1", "v", clock)]})
            with self.subTest(clock=clock):
                with self.assertRaises(ValueError):
                    self.parse(json.dumps(bad))

    def test_only_json_integers_are_accepted(self) -> None:
        for literal in ("1.0", "-0.0", "1e3", "1E2", "0.5", "NaN", "Infinity", "-Infinity"):
            raw = (
                '{"replicaId":"r","snapshot":{"k":['
                '{"value":"v","clock":{"r1":' + literal + '},"replicaId":"r1","operationId":"o1"}]}}'
            )
            with self.subTest(literal=literal):
                with self.assertRaises(ValueError):
                    self.parse(raw)

    def test_plain_integer_zero_and_negative_zero_integer_pass(self) -> None:
        replica_id, snapshot = self.parse(
            json.dumps(
                compare_body(
                    "r", {"k": [candidate("r1", "o1", "v", {"r1": 0, "r2": 2})]}
                )
            )
        )
        self.assertEqual(snapshot["k"][0]["clock"], {"r1": 0, "r2": 2})

    def test_duplicate_identities_are_rejected_anywhere(self) -> None:
        # Same identity twice inside one key.
        with self.assertRaises(ValueError):
            self.parse(
                json.dumps(
                    compare_body(
                        "r",
                        {
                            "k": [
                                candidate("r1", "o1", "a", {"r1": 1}),
                                candidate("r1", "o1", "b", {"r1": 2}),
                            ]
                        },
                    )
                )
            )
        # The same identity under two different keys is still one identity.
        with self.assertRaises(ValueError):
            self.parse(
                json.dumps(
                    compare_body(
                        "r",
                        {
                            "k": [candidate("r1", "o1", "a", {"r1": 1})],
                            "j": [candidate("r1", "o1", "b", {"r1": 2})],
                        },
                    )
                )
            )

    def test_duplicate_json_fields_are_rejected(self) -> None:
        for raw in (
            b'{"replicaId":"r","replicaId":"s","snapshot":{}}',
            b'{"replicaId":"r","snapshot":{},"snapshot":{}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"a","value":"b",'
            b'"clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v","replicaId":"r1",'
            b'"operationId":"o1","clock":{"r1":1,"r1":2}}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v","replicaId":"r1",'
            b'"operationId":"o1","clock":{}}],"k":[]}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.parse(raw)


def store_snapshot(store: StateStore) -> dict:
    """Return a plain snapshot-shaped copy of the store's current candidates."""
    return {
        key: [
            {
                "value": c["value"],
                "clock": dict(c["clock"]),
                "replicaId": c["replicaId"],
                "operationId": c["operationId"],
            }
            for c in candidates
        ]
        for key, candidates in store._candidates.items()
    }


class ReplicationCompareReportTests(unittest.TestCase):
    def identities(self, entries):
        return [(e["key"], e["identity"]["replicaId"], e["identity"]["operationId"])
                for e in entries]

    def test_both_empty_reports_same_digest_and_no_differences(self) -> None:
        report = replication_compare_report("rr", {}, {})
        self.assertEqual(report["remoteReplicaId"], "rr")
        self.assertEqual(
            report["differences"],
            {"shared": [], "localOnly": [], "remoteOnly": [], "contentConflict": []},
        )
        self.assertEqual(
            report["summary"],
            {
                "localKeys": 0,
                "remoteKeys": 0,
                "localCandidates": 0,
                "remoteCandidates": 0,
                "sameDigest": True,
                "differences": 0,
            },
        )

    def test_one_side_empty_only_reports_that_sides_keys(self) -> None:
        remote = {"color": [candidate("r1", "o1", "blue", {"r1": 1})]}
        report = replication_compare_report("rr", {}, remote)
        self.assertEqual(report["differences"]["shared"], [])
        self.assertEqual(report["differences"]["localOnly"], [])
        self.assertEqual(len(report["differences"]["remoteOnly"]), 1)
        entry = report["differences"]["remoteOnly"][0]
        self.assertEqual(entry["relation"], "missing_local")
        self.assertIsNone(entry["local"])
        self.assertEqual(entry["remote"], remote["color"][0])
        self.assertEqual(report["summary"]["localKeys"], 0)
        self.assertEqual(report["summary"]["remoteKeys"], 1)
        self.assertFalse(report["summary"]["sameDigest"])
        self.assertEqual(report["summary"]["differences"], 1)

        report = replication_compare_report("rr", remote, {})
        self.assertEqual(len(report["differences"]["localOnly"]), 1)
        self.assertEqual(report["differences"]["localOnly"][0]["relation"], "missing_remote")
        self.assertIsNone(report["differences"]["localOnly"][0]["remote"])
        self.assertEqual(report["summary"]["differences"], 1)

    def test_identical_snapshot_is_all_shared(self) -> None:
        snapshot = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
            "shape": [candidate("r3", "o3", "round", {"r3": 1})],
        }
        report = replication_compare_report("rr", snapshot, snapshot)
        self.assertEqual(len(report["differences"]["shared"]), 3)
        for name in ("localOnly", "remoteOnly", "contentConflict"):
            self.assertEqual(report["differences"][name], [])
        self.assertTrue(report["summary"]["sameDigest"])
        self.assertEqual(report["summary"]["differences"], 0)
        for entry in report["differences"]["shared"]:
            self.assertEqual(entry["relation"], "identical")
            self.assertEqual(entry["local"], entry["remote"])

    def test_content_conflict_is_marked(self) -> None:
        local = {"color": [candidate("r1", "o1", "blue", {"r1": 1})]}
        remote = {"color": [candidate("r1", "o1", "red", {"r1": 1})]}
        report = replication_compare_report("rr", local, remote)
        conflicts = report["differences"]["contentConflict"]
        self.assertEqual(len(conflicts), 1)
        entry = conflicts[0]
        self.assertEqual(entry["relation"], "content_conflict")
        self.assertEqual(entry["local"]["value"], "blue")
        self.assertEqual(entry["remote"]["value"], "red")
        self.assertFalse(report["summary"]["sameDigest"])
        self.assertEqual(report["summary"]["differences"], 1)

    def test_same_value_different_clock_marks_coverage(self) -> None:
        local = {"color": [candidate("r1", "o1", "blue", {"r1": 1})]}
        remote_covers = {"color": [candidate("r1", "o1", "blue", {"r1": 2})]}
        report = replication_compare_report("rr", local, remote_covers)
        self.assertEqual(
            [e["relation"] for e in report["differences"]["contentConflict"]],
            ["remote_clock_covers"],
        )
        report = replication_compare_report("rr", remote_covers, local)
        self.assertEqual(
            [e["relation"] for e in report["differences"]["contentConflict"]],
            ["local_clock_covers"],
        )
        divergent_remote = {
            "color": [candidate("r1", "o1", "blue", {"r1": 1, "r2": 1})]
        }
        # Remote also carries an r2 tick local lacks: it covers local.
        report = replication_compare_report("rr", local, divergent_remote)
        self.assertEqual(
            [e["relation"] for e in report["differences"]["contentConflict"]],
            ["remote_clock_covers"],
        )
        # Truly divergent clocks (each side has a component the other lacks).
        local_div = {"color": [candidate("r1", "o1", "blue", {"r1": 1, "r3": 1})]}
        remote_div = {"color": [candidate("r1", "o1", "blue", {"r1": 1, "r2": 1})]}
        report = replication_compare_report("rr", local_div, remote_div)
        self.assertEqual(
            [e["relation"] for e in report["differences"]["contentConflict"]],
            ["clock_divergent"],
        )

    def test_mixed_snapshot_groups_and_counts(self) -> None:
        local = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
            "local-key": [candidate("r1", "o9", "mine", {"r1": 9})],
        }
        remote = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),  # identical
                candidate("r2", "o2", "crimson", {"r2": 2}),  # value + clock differ
            ],
            "remote-key": [candidate("r9", "z9", "theirs", {"r9": 1})],
        }
        report = replication_compare_report("rr", local, remote)
        groups = report["differences"]
        self.assertEqual(self.identities(groups["shared"]),
                         [("color", "r1", "o1")])
        self.assertEqual(self.identities(groups["localOnly"]),
                         [("local-key", "r1", "o9")])
        self.assertEqual(self.identities(groups["remoteOnly"]),
                         [("remote-key", "r9", "z9")])
        self.assertEqual(self.identities(groups["contentConflict"]),
                         [("color", "r2", "o2")])
        summary = report["summary"]
        self.assertEqual(summary["localKeys"], 2)
        self.assertEqual(summary["remoteKeys"], 2)
        self.assertEqual(summary["localCandidates"], 3)
        self.assertEqual(summary["remoteCandidates"], 3)
        self.assertFalse(summary["sameDigest"])
        # Minimal difference count excludes the shared candidate.
        self.assertEqual(summary["differences"], 3)

    def test_groups_are_sorted_by_key_then_identity(self) -> None:
        local = {
            "zeta": [candidate("r5", "o5", "v", {"r5": 1})],
            "alpha": [
                candidate("r2", "o9", "v", {"r2": 1}),
                candidate("r1", "o1", "v", {"r1": 1}),
                candidate("r1", "o0", "v", {"r1": 0, "r2": 1}),
            ],
        }
        report = replication_compare_report("rr", local, {})
        only = report["differences"]["localOnly"]
        self.assertEqual(
            self.identities(only),
            [
                ("alpha", "r1", "o0"),
                ("alpha", "r1", "o1"),
                ("alpha", "r2", "o9"),
                ("zeta", "r5", "o5"),
            ],
        )

    def test_report_is_deterministic_for_the_same_inputs(self) -> None:
        local = {"k": [candidate("r2", "b", "v", {"r2": 1}),
                       candidate("r1", "a", "w", {"r1": 1})]}
        remote = {"k": [candidate("r1", "a", "w", {"r1": 1}),
                        candidate("r9", "c", "x", {"r9": 1})]}
        first = replication_compare_report("rr", local, remote)
        second = replication_compare_report("rr", local, remote)
        self.assertEqual(first, second)

    def test_same_digest_tracks_canonical_candidate_snapshot(self) -> None:
        local = {"k": [candidate("r1", "o1", "v", {"r1": 1, "r2": 1})]}
        # Same candidates, different request ordering: digest still matches.
        remote = {
            "k": [
                candidate(
                    "r1",
                    "o1",
                    "v",
                    {"r2": 1, "r1": 1},
                )
            ]
        }
        report = replication_compare_report("rr", local, remote)
        self.assertTrue(report["summary"]["sameDigest"])
        self.assertEqual(candidate_digest(local), candidate_digest(remote))

    def test_identity_under_another_key_is_a_per_key_difference(self) -> None:
        # An identity local holds under "a" that the remote holds under "b"
        # is missing on each side at its own key, so both directions show.
        local = {"a": [candidate("r1", "o1", "v", {"r1": 1})]}
        remote = {"b": [candidate("r1", "o1", "v", {"r1": 1})]}
        report = replication_compare_report("rr", local, remote)
        groups = report["differences"]
        self.assertEqual(self.identities(groups["localOnly"]),
                         [("a", "r1", "o1")])
        self.assertEqual(self.identities(groups["remoteOnly"]),
                         [("b", "r1", "o1")])
        self.assertEqual(groups["shared"], [])
        self.assertEqual(groups["contentConflict"], [])
        self.assertEqual(report["summary"]["differences"], 2)


class CompareStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "color", "blue", {"r1": 1})
        )
        self.store.apply_operation(
            "r2", operation("o2", "color", "red", {"r2": 1})
        )
        self.store.apply_operation(
            "r1", operation("o3", "shape", "round", {"r1": 2})
        )

    def test_compare_is_read_only_and_imports_nothing(self) -> None:
        self.seed()
        before = self.store.get_replication_snapshot()
        remote = {
            "remote-only": [candidate("r9", "z9", "theirs", {"r9": 1})],
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "crimson", {"r2": 5}),
            ],
        }
        report = self.store.compare_replication("remote-r", remote)
        self.assertEqual(report["summary"]["remoteKeys"], 2)
        # No remote candidate entered the store.
        self.assertEqual(self.store.get_replication_snapshot(), before)
        status, _ = self.store.get_state_explanation("remote-only")
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        metrics = self.store.get_metrics()
        self.assertEqual(metrics["keys"], 2)
        self.assertEqual(metrics["candidateVersions"], 3)

    def test_compare_reads_one_committed_snapshot(self) -> None:
        self.seed()
        remote = store_snapshot(self.store)
        report = self.store.compare_replication("remote-r", remote)
        self.assertTrue(report["summary"]["sameDigest"])
        self.assertEqual(report["summary"]["differences"], 0)

    def test_concurrent_commits_keep_each_report_internally_consistent(
        self,
    ) -> None:
        import threading

        stop = threading.Event()
        violations: list[str] = []

        def reader() -> None:
            remote = {
                "color": [candidate("r1", "o1", "blue", {"r1": 1})]
            }
            while not stop.is_set():
                result = self.store.compare_replication("remote-r", remote)
                summary = result["summary"]
                total_groups = sum(
                    len(g) for g in result["differences"].values()
                )
                # Every candidate is counted exactly once across the groups.
                if total_groups != summary["localCandidates"] + 1:
                    violations.append("group totals disagree with summary")
                if summary["localCandidates"] < summary["localKeys"]:
                    violations.append("candidate count below key count")
                if summary["differences"] < 0:
                    violations.append("negative difference count")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for thread in readers:
            thread.start()
        try:
            for index in range(40):
                self.store.apply_operation(
                    f"r{index}",
                    operation(f"op-{index}", "shared", f"v{index}", {f"r{index}": 1}),
                )
        finally:
            stop.set()
            for thread in readers:
                thread.join(timeout=5)
        self.assertEqual(violations, [])


class CompareHttpServerTests(unittest.TestCase):
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
            conn.request(
                method, path, body=body,
                headers={"Content-Type": "application/json"},
            )
        else:
            conn.request(
                method, path, body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        headers = dict(response.getheaders())
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload, headers, raw

    def compare(self, replica_id: str, snapshot: dict, query: str = ""):
        return self.request(
            "POST", f"{COMPARE_PATH}{query}", compare_body(replica_id, snapshot)
        )

    def compare_raw(self, raw: bytes, path: str = COMPARE_PATH):
        return self.request("POST", path, raw)

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def seed(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        self.post_operation("r1", operation("o3", "shape", "round", {"r1": 2}))

    def test_empty_store_vs_empty_remote(self) -> None:
        status, payload, headers, raw = self.compare("remote-r", {})
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["differences"],
            {"shared": [], "localOnly": [], "remoteOnly": [], "contentConflict": []},
        )
        self.assertEqual(payload["remoteReplicaId"], "remote-r")
        self.assertTrue(payload["summary"]["sameDigest"])
        self.assertEqual(payload["summary"]["differences"], 0)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_success_body_is_compact_ordered_json_with_one_newline(self) -> None:
        self.seed()
        remote = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "crimson", {"r2": 2}),
            ],
        }
        status, payload, _, raw = self.compare("remote-r", remote)
        self.assertEqual(status, 200)
        # Top-level field order is fixed, not alphabetized.
        self.assertEqual(
            list(payload), ["remoteReplicaId", "differences", "summary"]
        )
        self.assertEqual(
            list(payload["differences"]),
            ["shared", "localOnly", "remoteOnly", "contentConflict"],
        )
        self.assertEqual(
            list(payload["summary"]),
            [
                "localKeys",
                "remoteKeys",
                "localCandidates",
                "remoteCandidates",
                "sameDigest",
                "differences",
            ],
        )
        entry = payload["differences"]["contentConflict"][0]
        self.assertEqual(
            list(entry), ["key", "identity", "relation", "local", "remote"]
        )
        self.assertEqual(list(entry["identity"]), ["replicaId", "operationId"])
        self.assertEqual(
            list(entry["local"]), ["value", "clock", "replicaId", "operationId"]
        )
        # Compact JSON in payload field order, exactly one trailing newline.
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")
            + b"\n",
        )

    def test_full_diff_over_http(self) -> None:
        self.seed()
        remote = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "crimson", {"r2": 2}),
            ],
            "size": [candidate("r9", "z9", "big", {"r9": 1})],
        }
        status, payload, _, _ = self.compare("remote-r", remote)
        self.assertEqual(status, 200)
        groups = payload["differences"]
        self.assertEqual(
            [(e["key"], e["identity"]["operationId"]) for e in groups["shared"]],
            [("color", "o1")],
        )
        self.assertEqual(
            [(e["key"], e["identity"]["operationId"]) for e in groups["localOnly"]],
            [("shape", "o3")],
        )
        self.assertEqual(
            [(e["key"], e["identity"]["operationId"]) for e in groups["remoteOnly"]],
            [("size", "z9")],
        )
        conflict = groups["contentConflict"][0]
        self.assertEqual(conflict["key"], "color")
        self.assertEqual(conflict["identity"]["operationId"], "o2")
        self.assertEqual(conflict["relation"], "content_conflict")
        self.assertEqual(conflict["local"]["value"], "red")
        self.assertEqual(conflict["remote"]["value"], "crimson")
        summary = payload["summary"]
        self.assertEqual(summary["localKeys"], 2)
        self.assertEqual(summary["remoteKeys"], 2)
        self.assertEqual(summary["localCandidates"], 3)
        self.assertEqual(summary["remoteCandidates"], 3)
        self.assertFalse(summary["sameDigest"])
        self.assertEqual(summary["differences"], 3)

    def test_missing_side_is_null_in_the_entry(self) -> None:
        self.post_operation(
            "r1", operation("o1", "k", "v", {"r1": 1})
        )
        status, payload, _, _ = self.compare("rr", {})
        self.assertEqual(status, 200)
        entry = payload["differences"]["localOnly"][0]
        self.assertEqual(entry["relation"], "missing_remote")
        self.assertIsNone(entry["remote"])
        self.assertEqual(entry["local"]["operationId"], "o1")

    def test_remote_content_is_not_imported(self) -> None:
        self.seed()
        remote = {
            "new-key": [candidate("r9", "z9", "v", {"r9": 1})],
        }
        status, _, _, _ = self.compare("remote-r", remote)
        self.assertEqual(status, 200)
        status, state, _, _ = self.request("GET", "/v1/states/new-key")
        self.assertEqual(status, 404)
        status, sync, _, _ = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync["operations"]), 3)

    def test_malformed_bodies_are_400(self) -> None:
        for raw in (
            b"",
            b"{not json",
            b"{}",
            b"[]",
            b'{"replicaId":"r"}',
            b'{"snapshot":{}}',
            b'{"replicaId":"r","snapshot":{},"x":1}',
            b'{"replicaId":"","snapshot":{}}',
            b'{"replicaId":"r","snapshot":{"k":[]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":1},"replicaId":"r1","operationId":""}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r2":1},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":true},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":1.0},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":-0.0},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":NaN},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":Infinity},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","snapshot":{"k":['
            b'{"value":"a","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}],'
            b'"j":[{"value":"b","clock":{"r1":2},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"r","replicaId":"s","snapshot":{}}',
        ):
            with self.subTest(raw=raw):
                status, payload, _, response_raw = self.compare_raw(raw)
                self.assertEqual(status, 400, raw)
                self.assertEqual(payload, {"error": "invalid_request"}, raw)
                self.assertTrue(response_raw.endswith(b"\n"))

    def test_any_query_parameter_is_400(self) -> None:
        for query in ("?x=1", "?replicaId=r", "?x=", "?x", "?=1", "?x=1&x=2"):
            with self.subTest(query=query):
                status, payload, _, _ = self.compare("rr", {}, query)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)
        status, _, _, _ = self.compare("rr", {}, "?")
        self.assertEqual(status, 200)

    def test_query_check_precedes_body_check(self) -> None:
        status, payload, _, _ = self.compare_raw(b"{not json", f"{COMPARE_PATH}?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/compare/extra",
            "/v1/replication/compare/",
            "/v1/replication",
            "/v1/replication/comparison",
            "/v2/replication/compare",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.compare_raw(b"{}", path)
                self.assertEqual(status, 404, path)
                self.assertEqual(payload, {"error": "not_found"}, path)

    def test_empty_middle_segment_folds_into_the_route_shape(self) -> None:
        # Like the other flat three-segment POST routes (/v1/sync//operations
        # folds into /v1/sync/operations), the empty middle segment is
        # filtered before shape matching, so the request reaches the compare
        # body validation and is rejected there as 400 — it is not a 404.
        status, payload, _, _ = self.compare_raw(b"{}", "/v1/replication//compare")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_beats_query_and_body(self) -> None:
        for path, raw in (
            ("/v1/replication/compare/extra?x=1", b"{not json"),
            ("/v1/replication/compare/?x=1", b"{not json"),
        ):
            status, payload, _, _ = self.compare_raw(raw, path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_compare_route_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", COMPARE_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class CompareRequestLimitTests(unittest.TestCase):
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

    def post_raw(self, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", COMPARE_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        status, payload = self.post_raw([], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_content_length_is_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "5.0", "1 2"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    [("Content-Length", value)], b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_declared_length_at_the_limit_is_accepted(self) -> None:
        document = compare_body(
            "r",
            {"k": [candidate("r1", "o1", "v", {"r1": 1})]},
        )
        base = json.dumps(document, separators=(",", ":")).encode("utf-8")
        old_length = len(document["snapshot"]["k"][0]["value"])
        pad = MAX_BODY_BYTES - len(base) + old_length
        document["snapshot"]["k"][0]["value"] = "x" * pad
        body = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.assertEqual(len(body), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            [("Content-Length", str(len(body)))], body
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["differences"], 1)

    def test_over_limit_declaration_is_413_before_body_validation(self) -> None:
        status, payload = self.post_raw(
            [("Content-Length", str(MAX_BODY_BYTES + 1))], b"{not json"
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})


class CompareAuthTests(unittest.TestCase):
    def start_server(self, **kwargs) -> SemanticStateServer:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def post(self, server: SemanticStateServer, headers: dict, body: object = None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        if body is None:
            conn.request("POST", COMPARE_PATH, headers=headers)
        else:
            conn.request(
                "POST", COMPARE_PATH, body=json.dumps(body),
                headers={"Content-Type": "application/json", **headers},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        www = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, www

    def test_single_token_mode_enforces_bearer(self) -> None:
        server = self.start_server(auth_token="s3cret")
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "s3cret"},
            {"Authorization": "Bearer s3cret extra"},
        ):
            with self.subTest(headers=headers):
                status, payload, www = self.post(server, headers, compare_body("r", {}))
                self.assertEqual(status, 401, headers)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(www, "Bearer")
        status, payload, _ = self.post(
            server, {"Authorization": "Bearer s3cret"}, compare_body("r", {})
        )
        self.assertEqual(status, 200)
        self.assertIn("summary", payload)

    def test_duplicate_authorization_header_is_401(self) -> None:
        server = self.start_server(auth_token="s3cret")
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.putrequest("POST", COMPARE_PATH)
        conn.putheader("Content-Length", "2")
        conn.putheader("Authorization", "Bearer s3cret")
        conn.putheader("Authorization", "Bearer s3cret")
        conn.endheaders(b"{}")
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(response.read()), {"error": "unauthorized"})
        conn.close()

    def test_scope_policy_gates_read(self) -> None:
        server = self.start_server(
            auth_scopes={
                "t-read": frozenset({"read"}),
                "t-write": frozenset({"write"}),
                "t-admin": frozenset({"read", "write", "admin"}),
            }
        )
        status, _, www = self.post(server, {}, compare_body("r", {}))
        self.assertEqual(status, 401)
        self.assertEqual(www, "Bearer")
        status, payload, www = self.post(
            server, {"Authorization": "Bearer t-write"}, compare_body("r", {})
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(www)
        for token in ("t-read", "t-admin"):
            status, payload, _ = self.post(
                server, {"Authorization": f"Bearer {token}"}, compare_body("r", {})
            )
            self.assertEqual(status, 200, token)
            self.assertIn("summary", payload)

    def test_length_rejections_keep_priority_over_authentication(self) -> None:
        server = self.start_server(auth_token="s3cret")
        for headers, expected in (
            ([("Content-Length", "abc")], 400),
            ([("Content-Length", str(MAX_BODY_BYTES + 1))], 413),
        ):
            conn = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            conn.putrequest("POST", COMPARE_PATH)
            for name, value in headers:
                conn.putheader(name, value)
            conn.endheaders(None)
            response = conn.getresponse()
            self.assertEqual(response.status, expected, headers)
            response.read()
            conn.close()


class ComparePersistenceTests(unittest.TestCase):
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

    def request(self, server: SemanticStateServer, body: dict):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.request(
            "POST", COMPARE_PATH, body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw.decode("utf-8")), raw

    def write_operation(self, server: SemanticStateServer, replica: str, op: dict):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.request(
            "POST", f"/v1/replicas/{replica}/operations",
            body=json.dumps(op), headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    def test_same_inputs_give_same_diff_after_restart(self) -> None:
        server = self.start_server()
        self.write_operation(
            server, "r1", operation("o1", "color", "blue", {"r1": 1})
        )
        self.write_operation(
            server, "r2", operation("o2", "color", "red", {"r2": 1})
        )
        remote = {
            "color": [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "crimson", {"r2": 2}),
            ],
            "size": [candidate("r9", "z9", "big", {"r9": 1})],
        }
        status, before, _ = self.request(server, compare_body("remote-r", remote))
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after, _ = self.request(server, compare_body("remote-r", remote))
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_query_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.write_operation(
            server, "r1", operation("o1", "color", "blue", {"r1": 1})
        )
        remote = {"color": [candidate("r1", "o1", "red", {"r1": 2})]}
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()
        directory = set(Path(self._tmp.name).iterdir())
        for _ in range(5):
            status, _, _ = self.request(server, compare_body("remote-r", remote))
            self.assertEqual(status, 200)
        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        # No temporary files are left behind in the data directory.
        self.assertEqual(set(Path(self._tmp.name).iterdir()), directory)


if __name__ == "__main__":
    unittest.main()
