"""Tests for the multi-replica convergence decision endpoint.

The endpoint is::

    POST /v1/replication/consensus

with a body that is a JSON array of 2 to 100 entries, each an object
with exactly ``replicaId`` (a non-empty, unique identifier) and
``snapshot`` (the remote's complete candidate snapshot, under exactly
the cross-replica comparison constraints). The endpoint aggregates the
local candidates with every supplied snapshot by business key and
operation identity ``(replicaId, operationId)`` and classifies each
identity:

- ``converged`` — every observation holds identical candidate content;
- ``propagated`` — same value with different clocks where exactly one
  distinct clock dominates all the others; the decision names the
  local version or a remote version and keeps the dominated clocks as
  ``supersededClocks`` evidence;
- ``conflict`` — the values diverge across sources or the equal-value
  clocks are concurrent; every observation is retained and no value is
  selected.

The summary lists the sources (local first, then the remotes in request
order) and the converged/conflict counts. Keys sort lexicographically and
identities by ``(replicaId, operationId)``. The query is strictly
read-only: the snapshots are never imported, and no repair, sync,
checkpoint, or persistence runs. The success body is compact canonical
UTF-8 JSON terminated by one newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for decision semantics. Only
the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    REPLICATION_CONSENSUS_MAX_PEERS,
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


def peer(replica_id: str, snapshot: dict | None = None) -> dict:
    return {"replicaId": replica_id, "snapshot": {} if snapshot is None else snapshot}


def consensus_body(*peers: dict) -> list:
    return list(peers)


class ParseReplicationConsensusPayloadTests(unittest.TestCase):
    """Body validation: an array of 2-100 unique remote snapshots."""

    def test_minimal_and_populated_bodies_pass(self) -> None:
        parsed = parse_replication_consensus_payload(
            json.dumps(consensus_body(peer("a"), peer("b"))).encode()
        )
        self.assertEqual(parsed, [("a", {}), ("b", {})])
        document = consensus_body(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            peer("b", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
        )
        parsed = parse_replication_consensus_payload(json.dumps(document).encode())
        self.assertEqual(parsed[0][0], "a")
        self.assertEqual(parsed[0][1], document[0]["snapshot"])
        self.assertEqual(parsed[1][0], "b")
        self.assertEqual(parsed[1][1], document[1]["snapshot"])

    def test_bytes_str_and_list_forms(self) -> None:
        text = (
            b'[{"replicaId":"a","snapshot":{}},'
            b'{"replicaId":"b","snapshot":{"k":[{"value":"v","clock":{"r1":1},'
            b'"replicaId":"r1","operationId":"o1"}]}}]'
        )
        expected = [("a", {}), ("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]})]
        self.assertEqual(parse_replication_consensus_payload(text), expected)
        self.assertEqual(
            parse_replication_consensus_payload(text.decode("utf-8")), expected
        )
        self.assertEqual(parse_replication_consensus_payload(json.loads(text)), expected)

    def test_request_order_is_preserved(self) -> None:
        document = consensus_body(peer("zeta"), peer("alpha"), peer("mid"))
        parsed = parse_replication_consensus_payload(document)
        self.assertEqual([replica_id for replica_id, _ in parsed], ["zeta", "alpha", "mid"])

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
            b"{}x",
            b"\xff\xfe[]",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_root_must_be_an_array_of_two_to_one_hundred(self) -> None:
        for raw in (b"[]", b"[{}]"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)
        one = consensus_body(peer("only"))
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(one)
        oversized = consensus_body(*(peer(f"p{i}") for i in range(101)))
        self.assertEqual(len(oversized), REPLICATION_CONSENSUS_MAX_PEERS + 1)
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(oversized)
        # Exactly the bounds are accepted structurally.
        two = consensus_body(peer("p0"), peer("p1"))
        self.assertEqual(len(parse_replication_consensus_payload(two)), 2)
        hundred = consensus_body(*(peer(f"p{i}") for i in range(100)))
        self.assertEqual(
            len(parse_replication_consensus_payload(hundred)),
            REPLICATION_CONSENSUS_MAX_PEERS,
        )

    def test_duplicate_and_empty_replica_ids_are_rejected(self) -> None:
        for document in (
            consensus_body(peer("a"), peer("a")),
            consensus_body(peer(""), peer("b")),
            consensus_body(peer("a"), peer("b"), peer("a")),
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(document)

    def test_entry_field_set_is_exact(self) -> None:
        for document in (
            [peer("a"), {}],
            [{"replicaId": "a"}, peer("b")],
            [{"snapshot": {}}, peer("b")],
            [{"replicaId": "a", "snapshot": {}, "x": 1}, peer("b")],
            [{"replicaId": 1, "snapshot": {}}, peer("b")],
            [{"replicaId": None, "snapshot": {}}, peer("b")],
            ["not-an-object", peer("b")],
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(document)

    def test_duplicate_json_fields_anywhere_are_rejected(self) -> None:
        for raw in (
            b'[{"replicaId":"a","replicaId":"z","snapshot":{}},'
            b'{"replicaId":"b","snapshot":{}}]',
            b'[{"replicaId":"a","snapshot":{},"snapshot":{}},'
            b'{"replicaId":"b","snapshot":{}}]',
            b'[{"replicaId":"a","snapshot":{"k":[{"value":"v","value":"w",'
            b'"clock":{"r1":1},"replicaId":"r1","operationId":"o1"}]}},'
            b'{"replicaId":"b","snapshot":{}}]',
            b'[{"replicaId":"a","snapshot":{"k":[{"value":"v",'
            b'"clock":{"r1":1,"r1":2},"replicaId":"r1","operationId":"o1"}]}},'
            b'{"replicaId":"b","snapshot":{}}]',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_snapshot_constraints_match_the_comparison_endpoint(self) -> None:
        def wrap(snapshot):
            return consensus_body(peer("a", snapshot), peer("b"))

        # Empty key / empty candidate array.
        for document in (
            wrap({"": []}),
            wrap({"k": []}),
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(document)
        # Illegal candidate field set or scalars.
        for document in (
            wrap({"k": [{}]}),
            wrap({"k": [{"value": "v", "clock": {"r1": 1}, "replicaId": "r1"}]}),
            wrap({"k": [candidate("", "o1", "v", {"r1": 1})]}),
            wrap({"k": [candidate("r1", "o1", "v", {"r2": 1})]}),
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(document)

    def test_float_negative_zero_and_non_finite_clocks_are_rejected(self) -> None:
        for clock in (
            '{"r1":1.0}',
            '{"r1":-0.0}',
            '{"r1":1e2}',
            '{"r1":NaN}',
            '{"r1":Infinity}',
            '{"r1":-Infinity}',
            '{"r1":-1}',
            '{"r1":true}',
            '{"r1":"1"}',
        ):
            raw = (
                b'[{"replicaId":"a","snapshot":{"k":[{"value":"v","clock":'
                + clock.encode("ascii")
                + b',"replicaId":"r1","operationId":"o1"}]}},'
                b'{"replicaId":"b","snapshot":{}}]'
            )
            with self.subTest(clock=clock):
                with self.assertRaises(ValueError):
                    parse_replication_consensus_payload(raw)

    def test_duplicate_identity_within_one_snapshot_is_rejected(self) -> None:
        document = consensus_body(
            peer(
                "a",
                {
                    "k": [
                        candidate("r1", "o1", "v", {"r1": 1}),
                        candidate("r1", "o1", "w", {"r1": 2}),
                    ]
                },
            ),
            peer("b"),
        )
        with self.assertRaises(ValueError):
            parse_replication_consensus_payload(document)

    def test_same_identity_may_appear_across_different_sources(self) -> None:
        # The same operation identity is exactly what gets aggregated
        # across sources, so it must be legal in different snapshots even
        # though it may not repeat within one snapshot.
        document = consensus_body(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            peer("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        parsed = parse_replication_consensus_payload(document)
        self.assertEqual(len(parsed), 2)


class ConsensusStoreTests(unittest.TestCase):
    """Decision semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def consensus(self, *remote_peers: dict) -> dict:
        peers = parse_replication_consensus_payload(json.dumps(list(remote_peers)).encode())
        return self.store.replication_consensus(peers)

    def test_all_sources_empty_reports_no_keys(self) -> None:
        report = self.consensus(peer("a"), peer("b"))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["sources"], ["local", "a", "b"])
        self.assertEqual(report["keys"], [])
        self.assertEqual(
            report["summary"], {"sources": 3, "converged": 0, "conflicts": 0}
        )

    def test_identical_content_everywhere_is_converged(self) -> None:
        self.store.apply_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        snapshot = {"color": [candidate("r1", "o1", "blue", {"r1": 1})]}
        report = self.consensus(peer("a", snapshot), peer("b", snapshot))
        (group,) = report["keys"]
        (entry,) = group["identities"]
        self.assertEqual(group["key"], "color")
        self.assertEqual(entry["status"], "converged")
        self.assertEqual(entry["identity"], {"replicaId": "r1", "operationId": "o1"})
        self.assertNotIn("decision", entry)
        self.assertNotIn("supersededClocks", entry)
        self.assertEqual(len(entry["observations"]), 3)
        self.assertEqual(
            [obs["source"] for obs in entry["observations"]], ["local", "a", "b"]
        )
        for obs in entry["observations"]:
            self.assertEqual(obs["value"], "blue")
            self.assertEqual(obs["clock"], {"r1": 1})
        self.assertEqual(
            report["summary"], {"sources": 3, "converged": 1, "conflicts": 0}
        )

    def test_single_dominating_clock_propagates_local_version(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            peer("b", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "propagated")
        self.assertEqual(entry["decision"]["source"], "local")
        self.assertEqual(
            entry["decision"]["candidate"],
            candidate("r1", "o1", "v", {"r1": 2}),
        )
        self.assertEqual(entry["supersededClocks"], [{"r1": 1}])
        self.assertEqual(len(entry["observations"]), 3)
        self.assertEqual(report["summary"], {"sources": 3, "converged": 0, "conflicts": 0})

    def test_single_dominating_clock_propagates_a_remote_version(self) -> None:
        # Local lacks the identity; the first remote holding the
        # dominating clock is chosen over a later one.
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            peer("b", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            peer("c", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "propagated")
        self.assertEqual(entry["decision"]["source"], "b")
        self.assertEqual(entry["decision"]["candidate"]["clock"], {"r1": 2})
        self.assertEqual(entry["supersededClocks"], [{"r1": 1}])
        # All three remote observations are retained.
        self.assertEqual(
            sorted(obs["source"] for obs in entry["observations"]),
            ["a", "b", "c"],
        )

    def test_identical_dominating_clock_on_several_remotes_still_propagates(self) -> None:
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            peer("b", {"k": [candidate("r1", "o1", "v", {"r1": 2})]}),
            peer("c", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "propagated")
        self.assertEqual(entry["decision"]["source"], "a")
        # The shared dominated clock is listed once.
        self.assertEqual(entry["supersededClocks"], [{"r1": 1}])

    def test_concurrent_equal_value_clocks_need_semantic_repair(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "k", "v", {"r1": 1, "r2": 1})
        )
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1, "r3": 1})]}),
            peer("b"),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "conflict")
        self.assertNotIn("decision", entry)
        self.assertNotIn("supersededClocks", entry)
        self.assertEqual(len(entry["observations"]), 2)
        self.assertEqual(report["summary"], {"sources": 3, "converged": 0, "conflicts": 1})

    def test_value_divergence_is_a_conflict_even_when_a_clock_dominates(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "a", {"r1": 2}))
        report = self.consensus(
            peer("x", {"k": [candidate("r1", "o1", "b", {"r1": 1})]}),
            peer("y", {"k": [candidate("r1", "o1", "a", {"r1": 2})]}),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "conflict")
        self.assertNotIn("decision", entry)
        self.assertEqual(len(entry["observations"]), 3)
        values = {obs["value"] for obs in entry["observations"]}
        self.assertEqual(values, {"a", "b"})
        self.assertEqual(report["summary"]["conflicts"], 1)
        self.assertEqual(report["summary"]["converged"], 0)

    def test_conflict_retains_every_observation_and_identity(self) -> None:
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "a", {"r1": 1})]}),
            peer("b", {"k": [candidate("r1", "o1", "b", {"r1": 1})]}),
        )
        entry = report["keys"][0]["identities"][0]
        self.assertEqual(entry["status"], "conflict")
        self.assertEqual(
            [obs["source"] for obs in entry["observations"]], ["a", "b"]
        )
        self.assertEqual(entry["identity"], {"replicaId": "r1", "operationId": "o1"})

    def test_keys_and_identities_are_stably_sorted(self) -> None:
        self.store.apply_operation("r9", operation("o9", "zeta", "z", {"r9": 1}))
        self.store.apply_operation("r1", operation("o1", "alpha", "a", {"r1": 1}))
        report = self.consensus(
            peer(
                "a",
                {
                    "mid": [
                        candidate("r5", "o5", "m", {"r5": 1}),
                        candidate("r2", "o2", "m", {"r2": 1}),
                    ],
                    "alpha": [candidate("r1", "o1", "a", {"r1": 1})],
                },
            ),
            peer("b", {"zeta": [candidate("r9", "o9", "z", {"r9": 1})]}),
        )
        self.assertEqual([group["key"] for group in report["keys"]], ["alpha", "mid", "zeta"])
        identities = [
            (entry["identity"]["replicaId"], entry["identity"]["operationId"])
            for entry in report["keys"][1]["identities"]
        ]
        self.assertEqual(identities, [("r2", "o2"), ("r5", "o5")])

    def test_observations_only_include_sources_holding_the_key(self) -> None:
        # A source whose snapshot omits a key (or the identity) holds no
        # observation of it; local alone holding an identity is still
        # converged with itself.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.consensus(
            peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}),
            peer("b", {"other": [candidate("r2", "o2", "w", {"r2": 1})]}),
        )
        groups = {group["key"]: group for group in report["keys"]}
        entry_k = groups["k"]["identities"][0]
        self.assertEqual(entry_k["status"], "converged")
        self.assertEqual(
            [obs["source"] for obs in entry_k["observations"]], ["local", "a"]
        )
        entry_other = groups["other"]["identities"][0]
        self.assertEqual(entry_other["status"], "converged")
        self.assertEqual(
            [obs["source"] for obs in entry_other["observations"]], ["b"]
        )
        self.assertEqual(report["summary"], {"sources": 3, "converged": 2, "conflicts": 0})

    def test_mixed_classifications_and_counts(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "new", {"r2": 2}))
        report = self.consensus(
            peer(
                "a",
                {
                    "k": [
                        candidate("r1", "o1", "same", {"r1": 1}),  # converged
                        candidate("r2", "o2", "new", {"r2": 1}),   # local dominates
                        candidate("r3", "o3", "x", {"r3": 1}),
                        candidate("r4", "o4", "p", {"r4": 1}),
                    ]
                },
            ),
            peer(
                "b",
                {
                    "k": [
                        candidate("r1", "o1", "same", {"r1": 1}),
                        candidate("r2", "o2", "new", {"r2": 2}),
                        candidate("r3", "o3", "y", {"r3": 1}),  # value conflict
                        candidate("r4", "o4", "q", {"r4": 2}),  # value conflict, dom clock
                    ]
                },
            ),
        )
        statuses = {
            (e["identity"]["replicaId"], e["identity"]["operationId"]): e["status"]
            for e in report["keys"][0]["identities"]
        }
        self.assertEqual(
            statuses,
            {
                ("r1", "o1"): "converged",
                ("r2", "o2"): "propagated",
                ("r3", "o3"): "conflict",
                ("r4", "o4"): "conflict",
            },
        )
        propagated = next(
            e
            for e in report["keys"][0]["identities"]
            if e["identity"] == {"replicaId": "r2", "operationId": "o2"}
        )
        self.assertEqual(propagated["decision"]["source"], "local")
        self.assertEqual(propagated["supersededClocks"], [{"r2": 1}])
        self.assertEqual(report["summary"], {"sources": 3, "converged": 1, "conflicts": 2})

    def test_query_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = [
            peer("a", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
            peer("b", {"new": [candidate("r3", "o3", "z", {"r3": 1})]}),
        ]
        peers = parse_replication_consensus_payload(json.dumps(document).encode())
        first = self.store.replication_consensus(peers)
        second = self.store.replication_consensus(peers)
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "v")
        self.assertEqual(self.store.get_state("new")[0], HTTPStatus.NOT_FOUND)


class ConsensusRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_same_inputs_produce_the_same_report_after_restart(self) -> None:
        document = [
            peer("a", {"k": [candidate("r1", "o1", "v1", {"r1": 2})]}),
            peer(
                "b",
                {
                    "k": [candidate("r1", "o1", "v1", {"r1": 1})],
                    "z": [candidate("r9", "o9", "w", {"r9": 1})],
                },
            ),
        ]
        peers = parse_replication_consensus_payload(json.dumps(document).encode())
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before = store.replication_consensus(peers)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.replication_consensus(peers), before)
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).replication_consensus(peers), before
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
        self, method: str, path: str, body: object = None
    ) -> tuple[int, object, bytes, list[tuple[str, str]]]:
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def consensus(self, document: list, path: str = CONSENSUS_PATH) -> tuple[int, object]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_local_and_remote_snapshots_over_http(self) -> None:
        status, payload = self.consensus(consensus_body(peer("a"), peer("b")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["sources"], ["local", "a", "b"])
        self.assertEqual(payload["keys"], [])
        self.assertEqual(
            payload["summary"], {"sources": 3, "converged": 0, "conflicts": 0}
        )

    def test_full_convergence_report_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 2}))
        document = consensus_body(
            peer("a", {"color": [candidate("r1", "o1", "blue", {"r1": 1})]}),
            peer(
                "b",
                {
                    "color": [
                        candidate("r1", "o1", "blue", {"r1": 1}),
                        candidate("r2", "o2", "red", {"r2": 1}),
                    ]
                },
            ),
        )
        status, payload = self.consensus(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["sources"], ["local", "a", "b"])
        (entry1, entry2) = payload["keys"][0]["identities"]
        self.assertEqual(entry1["identity"], {"replicaId": "r1", "operationId": "o1"})
        self.assertEqual(entry1["status"], "propagated")
        self.assertEqual(entry1["decision"]["source"], "local")
        self.assertEqual(entry1["decision"]["candidate"]["clock"], {"r1": 2})
        self.assertEqual(entry1["supersededClocks"], [{"r1": 1}])
        self.assertEqual(entry2["identity"], {"replicaId": "r2", "operationId": "o2"})
        self.assertEqual(entry2["status"], "converged")
        self.assertEqual(
            [obs["source"] for obs in entry2["observations"]], ["b"]
        )
        self.assertEqual(
            payload["summary"], {"sources": 3, "converged": 1, "conflicts": 0}
        )

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
        status, payload, raw, headers = self.raw_request(
            "POST",
            CONSENSUS_PATH,
            consensus_body(peer("a"), peer("b")),
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"status", "sources", "keys", "summary"})
        self.assertEqual(set(payload["summary"]), {"sources", "converged", "conflicts"})
        for name in ("sources", "converged", "conflicts"):
            self.assertIs(type(payload["summary"][name]), int, name)
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

    def test_any_query_parameter_is_400(self) -> None:
        document = consensus_body(peer("a"), peer("b"))
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.consensus(document, CONSENSUS_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.consensus(
            consensus_body(peer("a"), peer("b")), CONSENSUS_PATH + "?"
        )
        self.assertEqual(status, 200)

    def test_path_shape_mismatches_are_404(self) -> None:
        document = consensus_body(peer("a"), peer("b"))
        for path in (
            "/v1/replication/consensus/extra",
            "/v1/replication",
            "/v1/replication/consensus/",
            "/v1/replication/consensuses",
            "/v1/replication/consensus//",
        ):
            status, payload = self.consensus(document, path)
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
        for document in (
            {},
            [],
            [peer("only")],
            consensus_body(*(peer(f"p{i}") for i in range(101))),
            consensus_body(peer("a"), peer("a")),
            consensus_body(peer(""), peer("b")),
            [peer("a"), {}],
            [peer("a"), {"replicaId": "b", "snapshot": {}, "x": 1}],
            consensus_body(peer("a", {"k": []}), peer("b")),
            consensus_body(
                peer("a", {"k": [candidate("r1", "o1", "v", {"r1": 1.0})]}),
                peer("b"),
            ),
            consensus_body(
                peer("a", {"k": [candidate("r1", "o1", "v", {"r2": 1})]}),
                peer("b"),
            ),
            consensus_body(
                peer(
                    "a",
                    {
                        "k": [
                            candidate("r1", "o1", "v", {"r1": 1}),
                            candidate("r1", "o1", "w", {"r1": 2}),
                        ]
                    },
                ),
                peer("b"),
            ),
        ):
            status, payload = self.consensus(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)
        # Snapshot of a valid form (used only to confirm the baseline).
        status, _ = self.consensus(consensus_body(peer("a", valid_snapshot), peer("b")))
        self.assertEqual(status, 200)

    def test_malformed_json_body_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            CONSENSUS_PATH,
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
        _, before_sync = self.request("GET", "/v1/sync/operations")
        for document in (
            [],
            consensus_body(peer("a"), peer("a")),
            consensus_body(
                peer("a", {"k": [candidate("r1", "o1", "v", {"r1": -1})]}),
                peer("b"),
            ),
        ):
            status, _ = self.consensus(document)
            self.assertEqual(status, 400)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        _, after_sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_sync, after_sync)

    def test_consensus_does_not_import_or_mutate(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = consensus_body(
            peer("a", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
            peer("b", {"new": [candidate("r3", "o3", "z", {"r3": 1})]}),
        )
        for _ in range(3):
            status, payload = self.consensus(document)
            self.assertEqual(status, 200)
            identities = [
                entry
                for group in payload["keys"]
                for entry in group["identities"]
            ]
            # o1 local-only, o2 remote-only, o3 remote-only: all converged.
            self.assertTrue(all(entry["status"] == "converged" for entry in identities))
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 1)
        self.assertEqual(self.request("GET", "/v1/states/new")[0], 404)
        _, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync_payload["operations"]), 1)

    def test_concurrent_commits_observe_complete_snapshots(self) -> None:
        snapshots = consensus_body(
            peer(
                "a",
                {
                    "shared": [
                        candidate(f"r{index}", f"op-{index}", f"v{index}", {f"r{index}": 1})
                        for index in range(40)
                    ]
                },
            ),
            peer(
                "b",
                {
                    "shared": [
                        candidate(f"r{index}", f"op-{index}", f"v{index}", {f"r{index}": 1})
                        for index in range(40)
                    ]
                },
            ),
        )
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                peers = parse_replication_consensus_payload(json.dumps(snapshots).encode())
                report = self.server.store.replication_consensus(peers)
                seen_statuses = set()
                for group in report["keys"]:
                    for entry in group["identities"]:
                        if entry["status"] not in {"converged", "propagated", "conflict"}:
                            violations.append("unknown status")
                        seen_statuses.add(entry["status"])
                        sources = [obs["source"] for obs in entry["observations"]]
                        if sources != sorted(sources, key=["local", "a", "b"].index):
                            violations.append("observations out of source order")
                counted = sum(
                    1
                    for group in report["keys"]
                    for entry in group["identities"]
                    if entry["status"] == "converged"
                )
                if counted != report["summary"]["converged"]:
                    violations.append("converged count disagrees with entries")

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
        status, payload = self.consensus(snapshots)
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["conflicts"], 0)
        self.assertEqual(payload["summary"]["converged"], 40)


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

    @property
    def valid_body(self) -> bytes:
        return json.dumps(consensus_body(peer("a"), peer("b"))).encode("utf-8")

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
                    self.port,
                    CONSENSUS_PATH,
                    [("Content-Length", value)],
                    self.valid_body,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", "2"), ("Content-Length", "3")],
            b"[]",
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

    def test_absurdly_long_content_length_digits_are_413(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", "9" * 5000)],
            b"[]",
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
        # A valid two-entry array padded out to exactly the limit by a
        # long replica id: structurally valid, so it answers 200.
        template = b'[{"replicaId":"","snapshot":{}},{"replicaId":"b","snapshot":{}}]'
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"r" + b"x" * (pad - 1)
        body_bytes = (
            b'[{"replicaId":"' + name + b'","snapshot":{}},'
            b'{"replicaId":"b","snapshot":{}}]'
        )
        self.assertEqual(len(body_bytes), MAX_BODY_BYTES)
        status, payload = self.post_raw(
            self.port,
            CONSENSUS_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["sources"], 3)

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

    def document(self) -> list:
        return consensus_body(peer("a"), peer("b"))

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

    def test_duplicate_bearer_headers_are_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.single_port, timeout=5)
        conn.putrequest("POST", CONSENSUS_PATH)
        document = json.dumps(self.document()).encode("utf-8")
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
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CONSENSUS_PATH + "?x=1", {"nope": {}},
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
        self.request(self.single_port, "POST", CONSENSUS_PATH, self.document())
        self.request(
            self.single_port, "POST", CONSENSUS_PATH, self.document(),
            auth="Bearer nope",
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class ConsensusHttpPersistenceTests(unittest.TestCase):
    """With --data-file the same local state and snapshots summarize identically."""

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

    def test_consensus_survives_restart(self) -> None:
        document = consensus_body(
            peer("a", {"k1": [candidate("r1", "o1", "v1", {"r1": 2})]}),
            peer(
                "b",
                {
                    "k1": [candidate("r1", "o1", "v1", {"r1": 1})],
                    "k2": [candidate("r9", "o9", "w", {"r9": 1})],
                },
            ),
        )
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 2}),
        )
        self.request(
            server, "POST", "/v1/replicas/r2/operations",
            operation("o2", "k1", "v2", {"r2": 1}),
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

        document = consensus_body(
            peer("a", {"k": [candidate("r2", "o2", "w", {"r2": 1})]}),
            peer("b"),
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
