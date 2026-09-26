"""Tests for the cross-replica candidate comparison endpoint.

The endpoint is::

    POST /v1/replication/compare

with a body of exactly ``{"replicaId": ..., "snapshot": {...}}`` naming a
remote replica and its complete candidate snapshot (business keys mapped
to non-empty candidate arrays, each candidate carrying exactly ``value``,
``clock``, ``replicaId``, and ``operationId`` under the live write
constraints). The endpoint diffs the remote snapshot against the local
current candidates of one committed snapshot — strictly read-only: the
remote content is never imported, and no repair, transaction, sync, or
persistence runs. The report groups the union of business keys, marks
every entry ``shared``/``missing_remote``/``missing_local``/
``conflict``/``clock``, and summarizes both sides' key counts, candidate
counts, digests, and the minimal difference count. The success body is
compact canonical UTF-8 JSON terminated by one newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for diff semantics. Only the
Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
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
    _verification_digest_input,
    load_scope_policy,
    parse_replication_compare_payload,
)

COMPARE_PATH = "/v1/replication/compare"
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def body(replica_id: str = "remote", snapshot: dict | None = None) -> dict:
    return {"replicaId": replica_id, "snapshot": {} if snapshot is None else snapshot}


class ParseReplicationComparePayloadTests(unittest.TestCase):
    """Body validation: exactly ``replicaId`` plus a candidate snapshot."""

    def test_minimal_and_populated_bodies_pass(self) -> None:
        self.assertEqual(parse_replication_compare_payload(body()), ("remote", {}))
        snapshot = {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        replica_id, parsed = parse_replication_compare_payload(body("peer", snapshot))
        self.assertEqual(replica_id, "peer")
        self.assertEqual(parsed, snapshot)

    def test_bytes_and_str_and_mapping_forms(self) -> None:
        text = b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}}'
        expected = ("p", {"k": [candidate("r1", "o1", "v", {"r1": 1})]})
        self.assertEqual(parse_replication_compare_payload(text), expected)
        self.assertEqual(parse_replication_compare_payload(text.decode("utf-8")), expected)
        self.assertEqual(
            parse_replication_compare_payload(json.loads(text.decode("utf-8"))),
            expected,
        )

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(
            parse_replication_compare_payload(b'  { "replicaId": "p", "snapshot": { } }\n'),
            ("p", {}),
        )

    def test_malformed_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}x",
            b"[]",
            b"null",
            b'""',
            b"42",
            b"true",
            b"\xff\xfe{}",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        for raw in (
            b"{}",
            b'{"replicaId":"p"}',
            b'{"snapshot":{}}',
            b'{"replicaId":"p","snapshot":{},"x":1}',
            b'{"replicaId":"p","snapshot":{}}\n{"replicaId":"p","snapshot":{}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_duplicate_fields_are_rejected(self) -> None:
        for raw in (
            b'{"replicaId":"p","replicaId":"q","snapshot":{}}',
            b'{"replicaId":"p","snapshot":{},"snapshot":{}}',
            b'{"replicaId":"p","snapshot":{"k":[{"value":"v","value":"w","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":{"r1":1,"r1":2},"replicaId":"r1","operationId":"o1"}]}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_replica_id_must_be_a_non_empty_string(self) -> None:
        for value in ('""', "1", "true", "null", "[]", "{}"):
            raw = b'{"replicaId":' + value.encode("ascii") + b',"snapshot":{}}'
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_snapshot_must_be_an_object(self) -> None:
        for value in ("[]", "null", "1", '"k"', "true"):
            raw = b'{"replicaId":"p","snapshot":' + value.encode("ascii") + b"}"
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_empty_key_and_empty_candidate_array_are_rejected(self) -> None:
        for raw in (
            b'{"replicaId":"p","snapshot":{"":[]}}',
            b'{"replicaId":"p","snapshot":{"k":[]}}',
            b'{"replicaId":"p","snapshot":{"":[{"value":"v","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_candidate_field_set_is_exact(self) -> None:
        template = (
            b'{"replicaId":"p","snapshot":{"k":[%s]}}'
        )
        full = b'{"value":"v","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}'
        for raw_candidate in (
            b"{}",
            b'{"value":"v","clock":{"r1":1},"replicaId":"r1"}',
            full[:-1] + b',"x":1}',
            b'{"value":"v","clock":{"r1":1},"replicaId":"r1","operationId":"o1","key":"k"}',
        ):
            with self.subTest(raw=raw_candidate):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(template % raw_candidate)

    def test_candidate_scalars_must_be_non_empty_strings(self) -> None:
        for field in ("value", "replicaId", "operationId"):
            for bad in ('""', "1", "true", "null", "[]", "{}"):
                entry = candidate("r1", "o1", "v", {"r1": 1})
                document = body("p", {"k": [entry]})
                document["snapshot"]["k"][0][field] = json.loads(bad)
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        parse_replication_compare_payload(document)

    def test_clock_must_follow_write_constraints(self) -> None:
        for clock in (
            "{}",
            "[]",
            "null",
            "1",
            '{"r1":true}',
            '{"r1":-1}',
            '{"r1":1.0}',
            '{"r1":-0.0}',
            '{"r1":1e2}',
            '{"r1":NaN}',
            '{"r1":Infinity}',
            '{"r1":-Infinity}',
            '{"r1":"1"}',
            '{"":1,"r1":1}',
            '{"r2":1}',
        ):
            raw = (
                b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":'
                + clock.encode("ascii")
                + b',"replicaId":"r1","operationId":"o1"}]}}'
            )
            with self.subTest(clock=clock):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(raw)

    def test_duplicate_identity_is_rejected_within_and_across_keys(self) -> None:
        within = body(
            "p",
            {
                "k": [
                    candidate("r1", "o1", "a", {"r1": 1}),
                    candidate("r1", "o1", "b", {"r1": 2}),
                ]
            },
        )
        across = body(
            "p",
            {
                "k1": [candidate("r1", "o1", "a", {"r1": 1})],
                "k2": [candidate("r1", "o1", "a", {"r1": 1})],
            },
        )
        for document in (within, across):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_compare_payload(document)

    def test_same_operation_id_on_different_replicas_is_allowed(self) -> None:
        document = body(
            "p",
            {
                "k": [
                    candidate("r1", "o1", "a", {"r1": 1}),
                    candidate("r2", "o1", "b", {"r2": 1}),
                ]
            },
        )
        _, parsed = parse_replication_compare_payload(document)
        self.assertEqual(len(parsed["k"]), 2)

    def test_clock_is_normalized_to_a_clean_copy(self) -> None:
        document = body("p", {"k": [candidate("r1", "o1", "v", {"r2": 3, "r1": 1})]})
        _, parsed = parse_replication_compare_payload(document)
        self.assertEqual(parsed["k"][0]["clock"], {"r2": 3, "r1": 1})


class CompareStoreTests(unittest.TestCase):
    """Diff semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def compare(self, snapshot: dict, replica_id: str = "remote") -> dict:
        return self.store.compare_replication_snapshot(replica_id, snapshot)

    def test_both_empty_reports_identical_digests_and_no_differences(self) -> None:
        report = self.compare({})
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["replicaId"], "remote")
        self.assertEqual(report["keys"], [])
        self.assertEqual(
            report["summary"],
            {
                "localKeys": 0,
                "remoteKeys": 0,
                "localCandidates": 0,
                "remoteCandidates": 0,
                "localDigest": EMPTY_DIGEST,
                "remoteDigest": EMPTY_DIGEST,
                "identical": True,
                "differences": 0,
            },
        )

    def test_identical_snapshots_report_only_shared_entries(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k1", "w", {"r2": 1}))
        self.store.apply_operation("r1", operation("o3", "k2", "x", {"r1": 2}))
        snapshot = {
            "k1": [
                candidate("r1", "o1", "v", {"r1": 1}),
                candidate("r2", "o2", "w", {"r2": 1}),
            ],
            "k2": [candidate("r1", "o3", "x", {"r1": 2})],
        }
        report = self.compare(snapshot)
        self.assertTrue(report["summary"]["identical"])
        self.assertEqual(report["summary"]["differences"], 0)
        self.assertEqual(
            report["summary"]["localDigest"], report["summary"]["remoteDigest"]
        )
        kinds = [
            entry["kind"] for group in report["keys"] for entry in group["differences"]
        ]
        self.assertEqual(kinds, ["shared", "shared", "shared"])
        # Local digest matches the verification digest of the same state.
        verification = self.store.get_verification_digest()
        self.assertEqual(report["summary"]["localDigest"], verification["digest"])
        self.assertEqual(report["summary"]["localKeys"], verification["keys"])
        self.assertEqual(
            report["summary"]["localCandidates"], verification["candidateVersions"]
        )

    def test_remote_digest_follows_the_verification_digest_rules(self) -> None:
        snapshot = {"ké": [candidate("r1", "o1", "a\nb→", {"r2": 1, "r1": 2})]}
        report = self.compare(snapshot)
        self.assertEqual(
            report["summary"]["remoteDigest"],
            hashlib.sha256(_verification_digest_input(snapshot)).hexdigest(),
        )

    def test_one_side_empty_reports_only_that_side(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.compare({})
        self.assertFalse(report["summary"]["identical"])
        self.assertEqual(report["summary"]["remoteDigest"], EMPTY_DIGEST)
        self.assertEqual(report["summary"]["differences"], 1)
        self.assertEqual(len(report["keys"]), 1)
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "missing_remote")
        self.assertIsNone(entry["remote"])
        self.assertEqual(entry["local"]["value"], "v")

        empty_store = StateStore()
        report = empty_store.compare_replication_snapshot(
            "remote", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        )
        self.assertEqual(report["summary"]["localDigest"], EMPTY_DIGEST)
        self.assertEqual(report["summary"]["differences"], 1)
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "missing_local")
        self.assertIsNone(entry["local"])
        self.assertEqual(entry["remote"]["value"], "v")

    def test_all_difference_kinds_and_ordering(self) -> None:
        # Local: shared o1, conflict o2, clock o3, missing_remote o4 (key a);
        # key m only local; key z untouched.
        self.store.apply_operation("r1", operation("o1", "a", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "a", "local-val", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "a", "clk", {"r3": 2}))
        self.store.apply_operation("r4", operation("o4", "a", "only-local", {"r4": 1}))
        self.store.apply_operation("r5", operation("o5", "m", "mine", {"r5": 1}))
        snapshot = {
            "a": [
                candidate("r1", "o1", "same", {"r1": 1}),
                candidate("r2", "o2", "remote-val", {"r2": 1}),
                candidate("r3", "o3", "clk", {"r3": 1}),
                candidate("r6", "o6", "only-remote", {"r6": 1}),
            ],
            "z": [candidate("r7", "o7", "theirs", {"r7": 1})],
        }
        report = self.compare(snapshot)
        self.assertEqual([group["key"] for group in report["keys"]], ["a", "m", "z"])
        entries = report["keys"][0]["differences"]
        # Sorted by (replicaId, operationId) within the key.
        self.assertEqual(
            [(e["kind"],) for e in entries],
            [("shared",), ("conflict",), ("clock",), ("missing_remote",), ("missing_local",)],
        )
        self.assertEqual(entries[1]["local"]["value"], "local-val")
        self.assertEqual(entries[1]["remote"]["value"], "remote-val")
        self.assertEqual(entries[2]["local"]["clock"], {"r3": 2})
        self.assertEqual(entries[2]["remote"]["clock"], {"r3": 1})
        self.assertEqual(entries[2]["local"]["value"], entries[2]["remote"]["value"])
        self.assertIsNone(entries[3]["remote"])
        self.assertIsNone(entries[4]["local"])
        self.assertEqual(
            report["keys"][1]["differences"][0]["kind"], "missing_remote"
        )
        self.assertEqual(
            report["keys"][2]["differences"][0]["kind"], "missing_local"
        )
        self.assertEqual(
            report["summary"],
            {
                "localKeys": 2,
                "remoteKeys": 2,
                "localCandidates": 5,
                "remoteCandidates": 5,
                "localDigest": report["summary"]["localDigest"],
                "remoteDigest": report["summary"]["remoteDigest"],
                "identical": False,
                "differences": 6,
            },
        )
        self.assertRegex(report["summary"]["localDigest"], DIGEST_RE)
        self.assertRegex(report["summary"]["remoteDigest"], DIGEST_RE)

    def test_clock_kind_requires_same_value_and_different_clock(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        # Same value, strictly smaller remote clock: a clock-only difference.
        report = self.compare({"k": [candidate("r1", "o1", "v", {"r1": 1})]})
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "clock")
        # Same value, extra component on the remote clock: still clock-only.
        report = self.compare({"k": [candidate("r1", "o1", "v", {"r1": 2, "r9": 1})]})
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "clock")
        # Different value wins over any clock relationship: a conflict.
        report = self.compare({"k": [candidate("r1", "o1", "other", {"r1": 3})]})
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "conflict")

    def test_stale_writes_and_replays_do_not_enter_the_comparison(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        # Stale write: accepted into the log but adds no candidate.
        self.assertIs(
            self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        # Replay and conflict never enter the log.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))
        report = self.compare({"k": [candidate("r1", "o1", "v", {"r1": 2})]})
        self.assertTrue(report["summary"]["identical"])
        self.assertEqual(report["summary"]["localCandidates"], 1)

    def test_comparison_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        first = self.compare(snapshot)
        second = self.compare(snapshot)
        self.assertEqual(first, second)
        # The remote candidate was not imported and nothing else moved.
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "v")
        self.assertEqual(
            self.store.get_verification_digest()["digest"],
            first["summary"]["localDigest"],
        )


class CompareRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_same_state_same_remote_snapshot_same_report_after_restart(self) -> None:
        snapshot = {
            "k1": [candidate("r1", "o1", "v1", {"r1": 1})],
            "k2": [candidate("r9", "o9", "w", {"r9": 1})],
        }
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k1", "old", {"r1": 0}))
        before = store.compare_replication_snapshot("remote", snapshot)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(
            recovered.compare_replication_snapshot("remote", snapshot), before
        )
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).compare_replication_snapshot(
                "remote", snapshot
            ),
            before,
        )


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

    def compare(self, document: dict, path: str = COMPARE_PATH) -> tuple[int, dict]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_both_empty_over_http(self) -> None:
        status, payload = self.compare(body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["replicaId"], "remote")
        self.assertEqual(payload["keys"], [])
        summary = payload["summary"]
        self.assertTrue(summary["identical"])
        self.assertEqual(summary["differences"], 0)
        self.assertEqual(summary["localDigest"], EMPTY_DIGEST)
        self.assertEqual(summary["remoteDigest"], EMPTY_DIGEST)

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request(
            "POST",
            COMPARE_PATH,
            body("remote", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "replicaId", "keys", "summary"})
        self.assertEqual(
            set(payload["summary"]),
            {
                "localKeys",
                "remoteKeys",
                "localCandidates",
                "remoteCandidates",
                "localDigest",
                "remoteDigest",
                "identical",
                "differences",
            },
        )
        for name in ("localDigest", "remoteDigest"):
            self.assertRegex(payload["summary"][name], DIGEST_RE)
        for name in (
            "localKeys",
            "remoteKeys",
            "localCandidates",
            "remoteCandidates",
            "differences",
        ):
            self.assertIs(type(payload["summary"][name]), int, name)
        self.assertIs(type(payload["summary"]["identical"]), bool)
        # Compact canonical JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(
            raw[:-1].decode("utf-8"),
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_full_diff_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        self.post_operation("r1", operation("o3", "k2", "x", {"r1": 2}))
        document = body(
            "peer-b",
            {
                "k1": [
                    candidate("r1", "o1", "v1", {"r1": 1}),
                    candidate("r2", "o2", "CHANGED", {"r2": 1}),
                    candidate("r9", "o9", "w", {"r9": 1}),
                ],
                "k3": [candidate("r5", "o5", "y", {"r5": 1})],
            },
        )
        status, payload = self.compare(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replicaId"], "peer-b")
        self.assertEqual([group["key"] for group in payload["keys"]], ["k1", "k2", "k3"])
        kinds = [
            entry["kind"]
            for group in payload["keys"]
            for entry in group["differences"]
        ]
        self.assertEqual(
            kinds,
            ["shared", "conflict", "missing_local", "missing_remote", "missing_local"],
        )
        summary = payload["summary"]
        self.assertEqual(
            (summary["localKeys"], summary["remoteKeys"]),
            (2, 2),
        )
        self.assertEqual(
            (summary["localCandidates"], summary["remoteCandidates"]),
            (3, 4),
        )
        self.assertFalse(summary["identical"])
        self.assertEqual(summary["differences"], 4)
        _, verification = self.request("GET", "/v1/verification/digest")
        self.assertEqual(summary["localDigest"], verification["digest"])

    def test_any_query_parameter_is_400(self) -> None:
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.compare(body(), COMPARE_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.compare(body(), COMPARE_PATH + "?")
        self.assertEqual(status, 200)

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/compare/extra",
            "/v1/replication",
            "/v1/replication/compare/",
            "/v1/replication/compares",
            "/v1/replication/compare//",
        ):
            status, payload = self.compare(body(), path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_and_body_checks(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/compare/extra?x=1", {"not": "valid"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", COMPARE_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_invalid_bodies_are_400(self) -> None:
        for document in (
            {},
            {"replicaId": "p"},
            {"snapshot": {}},
            {"replicaId": "", "snapshot": {}},
            {"replicaId": "p", "snapshot": {}, "x": 1},
            {"replicaId": "p", "snapshot": []},
            {"replicaId": "p", "snapshot": {"k": []}},
            {"replicaId": "p", "snapshot": {"k": [{}]}},
            {
                "replicaId": "p",
                "snapshot": {"k": [candidate("r1", "o1", "v", {"r1": 1.0})]},
            },
            {
                "replicaId": "p",
                "snapshot": {"k": [candidate("r1", "o1", "v", {"r2": 1})]},
            },
            {
                "replicaId": "p",
                "snapshot": {
                    "k": [
                        candidate("r1", "o1", "a", {"r1": 1}),
                        candidate("r1", "o1", "b", {"r1": 2}),
                    ]
                },
            },
        ):
            status, payload = self.compare(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)

    def test_malformed_json_body_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            COMPARE_PATH,
            body=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_rejected_requests_change_no_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_digest = self.request("GET", "/v1/verification/digest")
        _, before_state = self.request("GET", "/v1/states/k")
        _, before_sync = self.request("GET", "/v1/sync/operations")
        for document in (
            {"replicaId": "p"},
            {"replicaId": "p", "snapshot": {"k": [candidate("r1", "o1", "v", {"r1": -1})]}},
        ):
            status, _ = self.compare(document)
            self.assertEqual(status, 400)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        _, after_state = self.request("GET", "/v1/states/k")
        _, after_sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_sync, after_sync)

    def test_compare_does_not_import_or_mutate(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = body(
            "remote",
            {
                "k": [candidate("r2", "o2", "w", {"r2": 1})],
                "new": [candidate("r3", "o3", "z", {"r3": 1})],
            },
        )
        for _ in range(3):
            status, payload = self.compare(document)
            self.assertEqual(status, 200)
            self.assertEqual(payload["summary"]["differences"], 3)
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 1)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v")
        status, _ = self.request("GET", "/v1/states/new")
        self.assertEqual(status, 404)
        _, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync_payload["operations"]), 1)

    def test_concurrent_commits_observe_consistent_comparisons(self) -> None:
        remote_snapshot = {
            "shared": [candidate(f"r{index}", f"op-{index}", f"v{index}", {f"r{index}": 1}) for index in range(40)]
        }
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                report = self.server.store.compare_replication_snapshot(
                    "remote", remote_snapshot
                )
                summary = report["summary"]
                if not DIGEST_RE.match(summary["localDigest"]):
                    violations.append("malformed local digest")
                if summary["localCandidates"] < summary["localKeys"]:
                    violations.append("localCandidates < localKeys")
                # The local digest must always agree with a digest computed
                # over the candidates the same report diffs against.
                seen = set()
                for group in report["keys"]:
                    identities = set()
                    for entry in group["differences"]:
                        for side in ("local", "remote"):
                            if entry[side] is not None:
                                identities.add(
                                    (entry[side]["replicaId"], entry[side]["operationId"])
                                )
                    if len(identities) != len(group["differences"]):
                        violations.append("identity repeated within a key group")
                    if group["key"] in seen:
                        violations.append("key group repeated")
                    seen.add(group["key"])
                if summary["differences"] < 0:
                    violations.append("negative difference count")

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
        # After all commits the remote snapshot matches the local state.
        status, payload = self.compare(body("remote", remote_snapshot))
        self.assertEqual(status, 200)
        self.assertTrue(payload["summary"]["identical"])
        self.assertEqual(payload["summary"]["differences"], 0)


class CompareHttpRequestLimitTests(unittest.TestCase):
    """The compare route keeps the shared Content-Length contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(("127.0.0.1", 0), RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

        cls.auth_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.auth_thread = threading.Thread(
            target=cls.auth_server.serve_forever, daemon=True
        )
        cls.auth_thread.start()
        cls.auth_port = cls.auth_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.auth_server.shutdown()
        cls.auth_server.server_close()
        cls.thread.join(timeout=5)
        cls.auth_thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()
        self.auth_server.store = type(self.auth_server.store)()

    def post_raw(self, port: int, path: str, headers: list, body: bytes = b""):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", COMPARE_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, COMPARE_PATH, [("Content-Length", value)], b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400(self) -> None:
        status, payload = self.post_raw(
            self.port,
            COMPARE_PATH,
            [("Content-Length", "2"), ("Content-Length", "3")],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        # The content itself is invalid JSON; the declared size wins.
        status, payload = self.post_raw(
            self.port,
            COMPARE_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_absurdly_long_content_length_digits_are_413(self) -> None:
        status, payload = self.post_raw(
            self.port,
            COMPARE_PATH,
            [("Content-Length", "9" * 5000)],
            b"{}",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        # Missing declaration: 400 even without a bearer token.
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", COMPARE_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        # Over-limit declaration: 413, not 401, even with no token.
        status, payload = self.post_raw(
            self.auth_port,
            COMPARE_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_body_at_exact_limit_is_processed_normally(self) -> None:
        template = b'{"replicaId":"","snapshot":{}}'
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"r" + b"x" * (pad - 1)
        body_bytes = (
            b'{"replicaId":"' + name + b'","snapshot":{}}'
        )
        self.assertEqual(len(body_bytes), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            self.port,
            COMPARE_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["identical"], True)

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        body_bytes = b"x" * MAX_BODY_BYTES
        status, payload = self.post_raw(
            self.port,
            COMPARE_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class CompareHttpAuthTests(unittest.TestCase):
    """The compare endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-compare-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]

        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reader": ["read"],
                    "writer": ["write"],
                    "admin": ["read", "write", "admin"],
                },
                handle,
            )
        cls.scope_server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes=dict(load_scope_policy(cls.policy_path)),
        )
        cls.scope_thread = threading.Thread(
            target=cls.scope_server.serve_forever, daemon=True
        )
        cls.scope_thread.start()
        cls.scope_port = cls.scope_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.single_server.shutdown()
        cls.single_server.server_close()
        cls.scope_server.shutdown()
        cls.scope_server.server_close()
        cls.single_thread.join(timeout=5)
        cls.scope_thread.join(timeout=5)
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        self.single_server.store = type(self.single_server.store)()
        self.scope_server.store = type(self.scope_server.store)()

    def request(self, port: int, method: str, path: str, body: object = None,
                auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, challenge

    def seed(self, port: int, token: str) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            port, "POST", "/v1/replicas/r1/operations", op, auth=token
        )
        self.assertEqual(status, 201)

    def test_single_token_mode_requires_bearer_token(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", COMPARE_PATH, body(), auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", COMPARE_PATH, body(), auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_duplicate_bearer_headers_are_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("POST", COMPARE_PATH)
        document = json.dumps(body()).encode("utf-8")
        conn.putheader("Content-Length", str(len(document)))
        conn.putheader("Authorization", "Bearer sekret")
        conn.putheader("Authorization", "Bearer sekret")
        conn.endheaders(document)
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(response.read()), {"error": "unauthorized"})
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "POST", COMPARE_PATH, body(), auth="Bearer writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", COMPARE_PATH, body(), auth=token
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["status"], "ok", token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", COMPARE_PATH + "?x=1", {"nope": {}},
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")

    def test_rejected_auth_reads_and_changes_nothing(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        before, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.request(self.single_port, "POST", COMPARE_PATH, body())
        self.request(
            self.single_port, "POST", COMPARE_PATH, body(), auth="Bearer nope"
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class CompareHttpPersistenceTests(unittest.TestCase):
    """With --data-file the same state and remote snapshot compare identically."""

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
    ):
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

    def test_comparison_survives_restart(self) -> None:
        snapshot = {
            "k1": [candidate("r1", "o1", "v1", {"r1": 1})],
            "k2": [candidate("r9", "o9", "w", {"r9": 1})],
        }
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k1", "v2", {"r2": 1}),
        )
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o3", "k1", "old", {"r1": 0}),
        )
        status, before = self.request(
            server, "POST", COMPARE_PATH, body("remote", snapshot)
        )
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(
            server, "POST", COMPARE_PATH, body("remote", snapshot)
        )
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_comparison_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()
        before_entries = set(os.listdir(self._tmp.name))

        document = body("remote", {"k": [candidate("r2", "o2", "w", {"r2": 1})]})
        for _ in range(5):
            status, _ = self.request(server, "POST", COMPARE_PATH, document)
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        # No temporary file is created alongside the data file.
        self.assertEqual(set(os.listdir(self._tmp.name)), before_entries)


if __name__ == "__main__":
    unittest.main()
