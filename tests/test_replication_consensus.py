"""Tests for the multi-replica convergence-consensus endpoint.

The endpoint is::

    POST /v1/replication/consensus

with a body that is a JSON array of two to one hundred entries, each an
object with exactly ``replicaId`` (a non-empty, non-repeated string) and
``snapshot`` (that replica's complete candidate snapshot under the same
constraints as ``POST /v1/replication/compare``). The local committed
candidate state participates as an additional source (``"local"``). The
endpoint aggregates, per business key and operation identity, every
source's observation and classifies each identity as ``converged``
(identical value and clock everywhere), ``propagable`` (same value, one
clock dominates every other observed clock — the winning version's source
is named and every eliminated clock kept as evidence), or ``conflict``
(equal values with no dominant clock, or differing values — every
observation retained, no value chosen and no clock merged). The summary
is read-only and the success body is compact canonical UTF-8 JSON
terminated by one newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for the aggregation
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import math
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
    LOCAL_CONSENSUS_SOURCE,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_scope_policy,
    parse_replication_consensus_payload,
)

CONSENSUS_PATH = "/v1/replication/consensus"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def entry(replica_id: str, snapshot: dict | None = None) -> dict:
    return {"replicaId": replica_id, "snapshot": {} if snapshot is None else snapshot}


def body(*entries: dict) -> list:
    return list(entries)


class ParseReplicationConsensusPayloadTests(unittest.TestCase):
    """Body validation: an array of 2-100 replicaId/snapshot entries."""

    def test_minimal_body_passes(self) -> None:
        sources = parse_replication_consensus_payload(body(entry("a"), entry("b")))
        self.assertEqual(sources, [("a", {}), ("b", {})])

    def test_bytes_str_and_sequence_forms(self) -> None:
        text = (
            b'[{"replicaId":"a","snapshot":{}},'
            b'{"replicaId":"b","snapshot":{"k":[{"value":"v","clock":{"r1":1},'
            b'"replicaId":"r1","operationId":"o1"}]}}]'
        )
        expected = [
            ("a", {}),
            ("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        ]
        self.assertEqual(parse_replication_consensus_payload(text), expected)
        self.assertEqual(
            parse_replication_consensus_payload(text.decode("utf-8")), expected
        )
        self.assertEqual(parse_replication_consensus_payload(json.loads(text)), expected)

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(
            parse_replication_consensus_payload(b' [ { "replicaId": "a", "snapshot": { } } ,\n {"replicaId":"b","snapshot":{}} ] '),
            [("a", {}), ("b", {})],
        )

    def test_malformed_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"null",
            b'""',
            b"42",
            b"true",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_array_size_bounds(self) -> None:
        one = body(entry("a"))
        zero: list = []
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(one)
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(zero)
        okay = [entry(f"r{i}") for i in range(100)]
        self.assertEqual(len(parse_replication_consensus_payload(okay)), 100)
        too_many = [entry(f"r{i}") for i in range(101)]
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(too_many)

    def test_entry_must_be_an_object_with_exactly_two_fields(self) -> None:
        for first_entry in (
            "null",
            "[]",
            '"a"',
            '{"snapshot":{}}',
            '{"replicaId":"a"}',
            '{"replicaId":"a","snapshot":{},"x":1}',
        ):
            raw = "[" + first_entry + ',{"replicaId":"b","snapshot":{}}]'
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_duplicate_fields_inside_entries_are_rejected(self) -> None:
        for raw in (
            b'[{"replicaId":"a","replicaId":"c","snapshot":{}},{"replicaId":"b","snapshot":{}}]',
            b'[{"replicaId":"a","snapshot":{},"snapshot":{}},{"replicaId":"b","snapshot":{}}]',
            b'[{"replicaId":"a","snapshot":{"k":[{"value":"v","value":"w","clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}},{"replicaId":"b","snapshot":{}}]',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_empty_and_duplicate_replica_ids_are_rejected(self) -> None:
        for documents in (
            body(entry(""), entry("b")),
            body(entry("a"), entry("a")),
            body(entry("a"), entry("b"), entry("a")),
        ):
            with self.subTest(documents=documents):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(documents)

    def test_non_string_replica_id_is_rejected(self) -> None:
        for bad in (1, True, None, [], {}):
            documents = [{"replicaId": bad, "snapshot": {}}, entry("b")]
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(documents)

    def test_local_source_id_is_reserved(self) -> None:
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(
                body(entry(LOCAL_CONSENSUS_SOURCE), entry("b"))
            )

    def test_snapshot_violations_are_rejected(self) -> None:
        base = candidate("r1", "o1", "v", {"r1": 1})
        bad_snapshots = [
            [],
            None,
            {"k": []},
            {"": [base]},
            {"k": [{}]},
            {"k": [dict(base, value="")]},
            {"k": [dict(base, replicaId="")]},
            {"k": [dict(base, operationId="")]},
            {"k": [dict(base, clock={"r2": 1})]},
            {"k": [candidate("r1", "o1", "a", {"r1": 1}), candidate("r1", "o1", "b", {"r1": 2})]},
        ]
        for bad_snapshot in bad_snapshots:
            documents = [
                {"replicaId": "a", "snapshot": bad_snapshot},
                {"replicaId": "b", "snapshot": {}},
            ]
            with self.subTest(bad_snapshot=bad_snapshot):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(documents)

    def test_float_negative_zero_and_non_finite_clocks_are_rejected(self) -> None:
        bad_ticks = (
            ('1.0', lambda: {"r1": 1.0}),
            ('-0.0', lambda: {"r1": -0.0}),
            ('1e2', None),
            ('NaN-text', lambda: {"r1": float("nan")}),
            ('Infinity-text', lambda: {"r1": math.inf}),
        )
        # Python-level floats.
        for label, factory in bad_ticks:
            if factory is None:
                continue
            documents = body(
                entry("a", {"k": [candidate("r1", "o1", "v", factory())]}),
                entry("b"),
            )
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(documents)
        # JSON-token forms.
        for token in ("1.0", "-0.0", "1e2", "NaN", "Infinity", "-Infinity"):
            raw = (
                b'[{"replicaId":"a","snapshot":{"k":[{"value":"v","clock":{"r1":'
                + token.encode("ascii")
                + b'},"replicaId":"r1","operationId":"o1"}]}},{"replicaId":"b","snapshot":{}}]'
            )
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_entries_keep_request_order(self) -> None:
        sources = parse_replication_consensus_payload(
            body(entry("z"), entry("a"), entry("m"))
        )
        self.assertEqual([source_id for source_id, _ in sources], ["z", "a", "m"])


class ConsensusStoreTests(unittest.TestCase):
    """Aggregation semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def consensus(self, *entries_value: dict) -> dict:
        return self.store.replication_consensus_summary(
            parse_replication_consensus_payload(body(*entries_value))
        )

    def identity(self, report: dict, key: str, replica_id: str, operation_id: str) -> dict:
        for group in report["keys"]:
            if group["key"] == key:
                for item in group["identities"]:
                    if (
                        item["replicaId"] == replica_id
                        and item["operationId"] == operation_id
                    ):
                        return item
        raise AssertionError(f"identity {replica_id}/{operation_id} for {key} not found")

    def test_all_sources_empty_is_an_empty_converged_report(self) -> None:
        report = self.consensus(entry("a"), entry("b"))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["sources"], ["local", "a", "b"])
        self.assertEqual(report["keys"], [])
        self.assertEqual(
            report["summary"], {"converged": 0, "propagable": 0, "conflicts": 0}
        )

    def test_identical_content_everywhere_is_converged(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        snap = {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        report = self.consensus(entry("a", snap), entry("b", snap))
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "converged")
        self.assertEqual(item["value"], "v")
        self.assertEqual(item["clock"], {"r1": 1})
        self.assertEqual(item["sources"], ["local", "a", "b"])
        self.assertEqual(
            report["summary"], {"converged": 1, "propagable": 0, "conflicts": 0}
        )

    def test_single_holder_is_converged(self) -> None:
        # An identity held by only one of the sources needs no decision.
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            entry("b"),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "converged")
        self.assertEqual(item["sources"], ["a"])
        self.assertEqual(item["value"], "v")
        self.assertEqual(item["clock"], {"r1": 1})

    def test_single_dominant_clock_same_value_is_propagable(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            entry("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "propagable")
        self.assertEqual(
            item["decision"],
            {"source": "a", "value": "v", "clock": {"r1": 2}},
        )
        # Every eliminated clock is retained, in source (observation) order.
        self.assertEqual(
            item["supersededClocks"],
            [
                {"source": "local", "clock": {"r1": 1}},
                {"source": "b", "clock": {"r1": 1}},
            ],
        )
        self.assertEqual(
            report["summary"], {"converged": 0, "propagable": 1, "conflicts": 0}
        )

    def test_local_version_can_be_the_propagable_winner(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            entry("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "propagable")
        self.assertEqual(item["decision"]["source"], "local")
        self.assertEqual(item["decision"]["clock"], {"r1": 3})
        self.assertEqual(
            item["supersededClocks"],
            [
                {"source": "a", "clock": {"r1": 2}},
                {"source": "b", "clock": {"r1": 1}},
            ],
        )

    def test_dominant_clock_requires_dominating_every_observation(self) -> None:
        # Three clocks: c1 dominates local but is concurrent with c2; c2
        # dominates local but is concurrent with c1. No clock dominates
        # all the others, so same value => semantic repair.
        self.store.apply_operation(
            "r1", operation("o1", "k", "v", {"r1": 1})
        )
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 1, "r2": 1})]}),
            entry("b", {"k": [candidate("r1", "o1", "v", {"r1": 1, "r3": 1})]}),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "conflict")
        self.assertEqual(len(item["observations"]), 3)

    def test_equal_clocks_same_value_are_converged_not_conflict(self) -> None:
        snap = {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        report = self.consensus(entry("a", snap), entry("b", snap))
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "converged")

    def test_equal_value_equal_clock_but_mixed_tie_is_stable(self) -> None:
        # Two observations with the same clock are converged even when a
        # third source lacks the identity entirely.
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            entry("b"),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "converged")
        self.assertEqual(item["sources"], ["a"])

    def test_different_values_always_conflict_even_with_dominant_clock(self) -> None:
        # A content divergence can never be auto-propagated, even when one
        # clock strictly dominates the other.
        self.store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        report = self.consensus(
            entry("a", {"k": [candidate("r1", "o1", "old", {"r1": 1})]}),
            entry("b"),
        )
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "conflict")
        self.assertNotIn("decision", item)
        self.assertNotIn("value", item)
        observations = item["observations"]
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]["source"], "local")
        self.assertEqual(observations[0]["value"], "new")
        self.assertEqual(observations[0]["clock"], {"r1": 2})
        self.assertEqual(observations[1]["source"], "a")
        self.assertEqual(observations[1]["value"], "old")
        self.assertEqual(observations[1]["clock"], {"r1": 1})
        self.assertEqual(
            report["summary"], {"converged": 0, "propagable": 0, "conflicts": 1}
        )

    def test_three_way_value_split_is_conflict_with_all_observations(self) -> None:
        snap = [
            entry("a", {"k": [candidate("r1", "o1", "v1", {"r1": 1})]}),
            entry("b", {"k": [candidate("r1", "o1", "v2", {"r1": 1})]}),
            entry("c", {"k": [candidate("r1", "o1", "v3", {"r1": 1})]}),
        ]
        report = self.consensus(*snap)
        item = self.identity(report, "k", "r1", "o1")
        self.assertEqual(item["status"], "conflict")
        self.assertEqual(
            [(o["source"], o["value"]) for o in item["observations"]],
            [("a", "v1"), ("b", "v2"), ("c", "v3")],
        )

    def test_conflict_retains_identity_and_all_observations(self) -> None:
        report = self.consensus(
            entry("a", {"k": [candidate("r9", "o9", "v1", {"r9": 1, "r1": 2})]}),
            entry("b", {"k": [candidate("r9", "o9", "v2", {"r9": 1, "r2": 2})]}),
        )
        item = self.identity(report, "k", "r9", "o9")
        self.assertEqual(item["replicaId"], "r9")
        self.assertEqual(item["operationId"], "o9")
        self.assertEqual(item["status"], "conflict")
        self.assertEqual(len(item["observations"]), 2)
        for observation in item["observations"]:
            self.assertEqual(set(observation), {"source", "value", "clock"})

    def test_report_is_sorted_by_key_then_identity(self) -> None:
        self.store.apply_operation("r2", operation("o2", "zeta", "z", {"r2": 1}))
        self.store.apply_operation("r1", operation("o1", "alpha", "a", {"r1": 1}))
        report = self.consensus(
            entry("a", {
                "zeta": [
                    candidate("r2", "o2", "z", {"r2": 1}),
                    candidate("r1", "o0", "x", {"r1": 1}),
                ],
                "alpha": [candidate("r1", "o1", "a", {"r1": 1})],
            }),
            entry("b"),
        )
        self.assertEqual([group["key"] for group in report["keys"]], ["alpha", "zeta"])
        zeta = report["keys"][1]
        self.assertEqual(
            [(i["replicaId"], i["operationId"]) for i in zeta["identities"]],
            [("r1", "o0"), ("r2", "o2")],
        )

    def test_counts_cover_every_identity_once(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        report = self.consensus(
            entry("a", {
                "k1": [
                    candidate("r1", "o1", "v", {"r1": 2}),  # propagable winner
                    candidate("r2", "o2", "w", {"r2": 1}),  # only on a: converged
                ],
                "k2": [candidate("r3", "o3", "p", {"r3": 1})],
            }),
            entry("b", {
                "k1": [
                    candidate("r1", "o1", "v", {"r1": 1}),
                ],
                "k2": [candidate("r3", "o3", "q", {"r3": 1})],
            }),
        )
        self.assertEqual(
            report["summary"], {"converged": 1, "propagable": 1, "conflicts": 1}
        )

    def test_query_is_side_effect_free_and_deterministic(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        documents = body(
            entry("a", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
            entry("b"),
        )
        first = self.store.replication_consensus_summary(
            parse_replication_consensus_payload(documents)
        )
        second = self.store.replication_consensus_summary(
            parse_replication_consensus_payload(documents)
        )
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "v")


class ConsensusRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_same_inputs_give_same_report_after_restart(self) -> None:
        documents = body(
            entry("a", {
                "k1": [candidate("r1", "o1", "v1", {"r1": 2})],
                "k2": [candidate("r9", "o9", "w", {"r9": 1})],
            }),
            entry("b", {"k1": [candidate("r1", "o1", "v1", {"r1": 1})]}),
        )
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        before = store.replication_consensus_summary(
            parse_replication_consensus_payload(documents)
        )
        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(
            recovered.replication_consensus_summary(
                parse_replication_consensus_payload(documents)
            ),
            before,
        )


class ConsensusHttpServerTests(unittest.TestCase):
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
        self, method: str, path: str, body_value: object = None
    ) -> tuple[int, object, bytes, list[tuple[str, str]]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body_value is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body_value),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        headers = response.getheaders()
        conn.close()
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body_value: object = None) -> tuple[int, object]:
        status, payload, _, _ = self.raw_request(method, path, body_value)
        return status, payload

    def consensus(self, document: object, path: str = CONSENSUS_PATH) -> tuple[int, object]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def minimal_document(self) -> list:
        return body(entry("a"), entry("b"))

    def test_minimal_request_over_http(self) -> None:
        status, payload = self.consensus(self.minimal_document())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["sources"], ["local", "a", "b"])
        self.assertEqual(payload["keys"], [])
        self.assertEqual(
            payload["summary"], {"converged": 0, "propagable": 0, "conflicts": 0}
        )

    def test_full_report_shape_and_ordering(self) -> None:
        self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "local-val", {"r2": 1}))
        document = body(
            entry("a", {
                "k": [
                    candidate("r1", "o1", "same", {"r1": 1}),
                    candidate("r2", "o2", "remote-val", {"r2": 1}),
                    candidate("r3", "o3", "v", {"r3": 1}),
                ],
            }),
            entry("b", {
                "k": [
                    candidate("r1", "o1", "same", {"r1": 1}),
                    candidate("r2", "o2", "local-val", {"r2": 1}),
                    candidate("r3", "o3", "v", {"r3": 1}),
                ],
            }),
        )
        status, payload = self.consensus(document)
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "sources", "keys", "summary"})
        (group,) = payload["keys"]
        self.assertEqual(group["key"], "k")
        statuses = [(i["replicaId"], i["status"]) for i in group["identities"]]
        self.assertEqual(
            statuses,
            [("r1", "converged"), ("r2", "conflict"), ("r3", "converged")],
        )
        conflict = group["identities"][1]
        self.assertEqual(
            sorted(o["source"] for o in conflict["observations"]),
            ["a", "b", "local"],
        )
        self.assertEqual(
            payload["summary"], {"converged": 2, "propagable": 0, "conflicts": 1}
        )

    def test_propagable_decision_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = body(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 3})]}),
            entry("b", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
        )
        status, payload = self.consensus(document)
        self.assertEqual(status, 200)
        item = payload["keys"][0]["identities"][0]
        self.assertEqual(item["status"], "propagable")
        self.assertEqual(item["decision"]["source"], "a")
        self.assertEqual(item["decision"]["clock"], {"r1": 3})
        self.assertEqual(
            item["supersededClocks"],
            [
                {"source": "local", "clock": {"r1": 1}},
                {"source": "b", "clock": {"r1": 2}},
            ],
        )

    def test_body_is_compact_canonical_json_with_one_newline(self) -> None:
        status, payload, raw, headers = self.raw_request(
            "POST", CONSENSUS_PATH, self.minimal_document()
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(
            raw[:-1].decode("utf-8"),
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_counts_and_clock_ticks_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = body(
            entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            entry("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        _, _, raw, _ = self.raw_request("POST", CONSENSUS_PATH, document)
        # No float, exponent, negative-zero, or non-finite token anywhere:
        # every numeric value is a plain JSON integer.
        self.assertIsNone(
            re.search(rb"[0-9](?:\.[0-9]|[eE][-+]?[0-9])|NaN|Infinity|-0(?![0-9])", raw)
        )

    def test_any_query_parameter_is_400(self) -> None:
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.consensus(self.minimal_document(), CONSENSUS_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.consensus(self.minimal_document(), CONSENSUS_PATH + "?")
        self.assertEqual(status, 200)

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/consensus/extra",
            "/v1/replication",
            "/v1/replication/consensus/",
            "/v1/replication/consensuses",
            "/v1/replication/consensus//",
        ):
            status, payload = self.consensus(self.minimal_document(), path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_and_body_checks(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/consensus/extra?x=1", {"not": "valid"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", CONSENSUS_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_invalid_bodies_are_400(self) -> None:
        valid_snapshot = {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        invalid_documents = [
            {},
            None,
            "x",
            42,
            True,
            [],
            body(entry("a")),
            [entry(f"r{i}") for i in range(101)],
            body(entry("a"), {"replicaId": "", "snapshot": {}}),
            body(entry("a"), entry("a")),
            body(entry("a"), {"replicaId": "b", "snapshot": []}),
            body(entry("a"), {"replicaId": "b", "snapshot": {"k": []}}),
            body(entry("a"), {"replicaId": "b", "snapshot": {"k": [{}]}}),
            body(
                entry("a"),
                {"replicaId": "b", "snapshot": {"k": [candidate("r1", "o1", "v", {"r2": 1})]}},
            ),
            body(
                entry("a"),
                entry("b", {
                    "k": [
                        candidate("r1", "o1", "x", {"r1": 1}),
                        candidate("r1", "o1", "y", {"r1": 2}),
                    ],
                }),
            ),
        ]
        for document in invalid_documents:
            status, payload = self.consensus(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)
        # valid body for sanity
        status, _ = self.consensus(body(entry("a", valid_snapshot), entry("b")))
        self.assertEqual(status, 200)

    def test_float_clocks_are_400(self) -> None:
        for document in (
            body(entry("a", {"k": [candidate("r1", "o1", "v", {"r1": 1.0})]}), entry("b")),
            body(entry("a", {"k": [candidate("r1", "o1", "v", {"r1": -0.0})]}), entry("b")),
        ):
            status, payload = self.consensus(document)
            self.assertEqual(status, 400)
            self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_json_and_non_finite_tokens_are_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        for raw_body in (
            b"{not json",
            b'[{"replicaId":"a","snapshot":{"k":[{"value":"v","clock":{"r1":NaN},"replicaId":"r1","operationId":"o1"}]}},{"replicaId":"b","snapshot":{}}]',
        ):
            conn.request(
                "POST",
                CONSENSUS_PATH,
                body=raw_body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 400, raw_body)
            self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_rejected_requests_change_no_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_digest = self.request("GET", "/v1/verification/digest")
        for document in ([], body(entry("a")), body(entry("a"), entry("a"))):
            status, _ = self.consensus(document)
            self.assertEqual(status, 400)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)

    def test_concurrent_commits_observe_complete_snapshots(self) -> None:
        remote = body(
            entry("a", {"shared": [candidate("r1", "o1", "v", {"r1": 1})]}),
            entry("b"),
        )
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                report = self.server.store.replication_consensus_summary(
                    parse_replication_consensus_payload(remote)
                )
                counts = report["summary"]
                total = counts["converged"] + counts["propagable"] + counts["conflicts"]
                seen: set[tuple[str, str, str]] = set()
                for group in report["keys"]:
                    for item in group["identities"]:
                        marker = (group["key"], item["replicaId"], item["operationId"])
                        if marker in seen:
                            violations.append("identity repeated")
                        seen.add(marker)
                        if item["status"] not in ("converged", "propagable", "conflict"):
                            violations.append("unknown status")
                if total != len(seen):
                    violations.append("counts do not match identities")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(30):
                self.post_operation(
                    f"r{index}",
                    operation(f"op-{index}", f"key-{index % 5}", f"v{index}", {f"r{index}": 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])


class ConsensusHttpRequestLimitTests(unittest.TestCase):
    """The consensus route keeps the shared Content-Length contract."""

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

    def post_raw(self, port: int, path: str, headers: list, body_value: bytes = b""):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", path)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body_value if body_value else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", CONSENSUS_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, CONSENSUS_PATH, [("Content-Length", value)], b"[]"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", CONSENSUS_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        status, payload = self.post_raw(
            self.auth_port,
            CONSENSUS_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_body_at_exact_limit_is_processed_normally(self) -> None:
        template = b'[{"replicaId":"","snapshot":{}},{"replicaId":"b","snapshot":{}}]'
        # A large, still-valid document at exactly the limit: pad the first
        # replica id.
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"r" + b"x" * (pad - 1)
        body_bytes = (
            b'[{"replicaId":"' + name + b'","snapshot":{}},{"replicaId":"b","snapshot":{}}]'
        )
        self.assertEqual(len(body_bytes), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"], {"converged": 0, "propagable": 0, "conflicts": 0})

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        body_bytes = b"x" * MAX_BODY_BYTES
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class ConsensusHttpAuthTests(unittest.TestCase):
    """The consensus endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-consensus-auth-")
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

    def request(self, port: int, method: str, path: str, body_value: object = None,
                auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        if body_value is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body_value), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        challenge = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, challenge

    def document(self) -> list:
        return body(entry("a"), entry("b"))

    def test_single_token_mode_requires_bearer_token(self) -> None:
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", CONSENSUS_PATH, self.document(), auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", CONSENSUS_PATH, self.document(),
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_scope_mode_requires_read_or_admin(self) -> None:
        status, payload, challenge = self.request(
            self.scope_port, "POST", CONSENSUS_PATH, self.document(),
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", CONSENSUS_PATH, self.document(), auth=token
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["status"], "ok", token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        status, payload, challenge = self.request(
            self.scope_port, "POST", CONSENSUS_PATH + "?x=1", ["nope"],
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_403_and_401_do_not_read_body_or_change_state(self) -> None:
        before, _, _ = self.request(
            self.scope_port, "GET", "/v1/metrics", auth="Bearer reader"
        )
        self.request(
            self.scope_port, "POST", CONSENSUS_PATH, self.document(),
            auth="Bearer writer",
        )
        self.request(self.scope_port, "POST", CONSENSUS_PATH, self.document())
        after, _, _ = self.request(
            self.scope_port, "GET", "/v1/metrics", auth="Bearer reader"
        )
        self.assertEqual(before, after)

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")


class ConsensusHttpPersistenceTests(unittest.TestCase):
    """With --data-file the same inputs summarize identically after restart."""

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

    def request(self, server: SemanticStateServer, method: str, path: str,
                body_value: object = None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        if body_value is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body_value),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_consensus_survives_restart(self) -> None:
        document = body(
            entry("a", {"k1": [candidate("r1", "o1", "v1", {"r1": 2})]}),
            entry("b", {"k1": [candidate("r1", "o1", "v1", {"r1": 1})]}),
        )
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 1}),
        )
        status, before = self.request(server, "POST", CONSENSUS_PATH, document)
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(server, "POST", CONSENSUS_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_consensus_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()
        before_entries = set(os.listdir(self._tmp.name))

        document = body(
            entry("a", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
            entry("b"),
        )
        for _ in range(5):
            status, _ = self.request(server, "POST", CONSENSUS_PATH, document)
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(set(os.listdir(self._tmp.name)), before_entries)


if __name__ == "__main__":
    unittest.main()
