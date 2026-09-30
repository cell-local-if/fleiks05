"""Tests for the offline inclusion-proof endpoints::

    GET /v1/audit/proofs/root
    GET /v1/replicas/{replicaId}/operations/{operationId}/proof?treeSize=N&root=H

The first returns the current log prefix's ``algorithm``, ``treeSize``,
``root``, and ``auditHead`` (the audit-chain tail). The second returns one
record's inclusion proof for the prefix of length ``treeSize``: ``record``
(the existing archive record), ``sequence`` (global 1-based position),
``treeSize``, ``root``, ``auditHead``, ``leafDigest``, and ``proof``.

A third party recomputes the root offline from only the record's canonical
audit bytes (:func:`semantic_state_engine.server._audit_record_bytes`, the
same encoding the audit chain uses) and the stable SHA-256 prefix-tree
rules:

- ``leafDigest = SHA256(b"\\x00" + recordBytes)`` (64 lowercase hex);
- internal node ``SHA256(b"\\x01" + left32 + right32)`` over raw 32-byte
  child digests;
- a lone odd node at a level is promoted unchanged (no sibling, no item);
- proof items run leaf-to-root, each ``{"side","digest","position"}``,
  ``side`` "left" when the sibling sits left of the running node and
  "right" otherwise, ``position`` the sibling's 0-based index in its level.

Appending to the log never changes an older prefix's root or proof. Both
endpoints are strictly read-only and the values survive a data-file
restart unchanged.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _audit_record_bytes,
    _audit_tree_inclusion_proof,
    _audit_tree_leaf_digest,
    _audit_tree_node_digest,
    _audit_tree_root,
    _ordered_json_bytes,
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
GENESIS = "0" * 64
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"").hexdigest()

ROOT_FIELDS = ["algorithm", "treeSize", "root", "auditHead"]
PROOF_FIELDS = [
    "record",
    "sequence",
    "treeSize",
    "root",
    "auditHead",
    "leafDigest",
    "proof",
]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def records(count: int, replica: str = "r") -> list[tuple[str, dict]]:
    return [
        (f"{replica}{i}", operation(f"o{i}", f"k{i % 3}", f"v{i}", {f"{replica}{i}": i}))
        for i in range(1, count + 1)
    ]


def build_tree(recs: list[tuple[str, dict]]) -> list[str]:
    return [
        _audit_tree_leaf_digest(_audit_record_bytes(replica_id, op))
        for replica_id, op in recs
    ]


def independent_root(leaves: list[str]) -> str:
    if not leaves:
        return EMPTY_ROOT
    level = list(leaves)
    while len(level) > 1:
        parents = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                parents.append(
                    hashlib.sha256(
                        NODE_PREFIX
                        + bytes.fromhex(level[i])
                        + bytes.fromhex(level[i + 1])
                    ).hexdigest()
                )
            else:
                parents.append(level[i])
        level = parents
    return level[0]


def verify_proof(payload: dict) -> None:
    """Strictly recompute root from the proof response, asserting each item."""
    sequence = payload["sequence"]
    tree_size = payload["treeSize"]
    record = payload["record"]
    replica_id, op = record["replicaId"], record["operation"]

    # 1. Leaf digest from the canonical record bytes.
    record_bytes = _audit_record_bytes(replica_id, op)
    running = hashlib.sha256(LEAF_PREFIX + record_bytes).hexdigest()
    assert running == payload["leafDigest"], "leafDigest mismatch"

    # 2. Walk levels, independently predicting the sibling layout.
    idx = sequence - 1
    level_size = tree_size
    items = payload["proof"]
    consumed = 0
    while level_size > 1:
        if idx % 2 == 1:
            # Left sibling exists.
            expected_side, expected_pos = "left", idx - 1
            do_combine = True
        elif idx + 1 < level_size:
            # Right sibling exists.
            expected_side, expected_pos = "right", idx + 1
            do_combine = True
        else:
            # Lone odd node promoted: no proof item at this level.
            do_combine = False
        if do_combine:
            assert consumed < len(items), "proof ended early"
            item = items[consumed]
            consumed += 1
            assert set(item) == {"side", "digest", "position"}, item
            assert item["side"] == expected_side, (item, expected_side)
            assert item["position"] == expected_pos, (item, expected_pos)
            assert DIGEST_RE.match(item["digest"])
            sibling = bytes.fromhex(item["digest"])
            current = bytes.fromhex(running)
            if expected_side == "left":
                running = hashlib.sha256(NODE_PREFIX + sibling + current).hexdigest()
            else:
                running = hashlib.sha256(NODE_PREFIX + current + sibling).hexdigest()
        idx //= 2
        level_size = (level_size + 1) // 2
    assert consumed == len(items), "extra proof items"

    # 3. The running digest is the root.
    assert running == payload["root"], "recomputed root mismatch"


class TreeRuleTests(unittest.TestCase):
    """Pin the prefix-tree primitives against literals and properties."""

    def test_leaf_digest_literal(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        record_bytes = _audit_record_bytes("r1", op)
        expected = hashlib.sha256(LEAF_PREFIX + record_bytes).hexdigest()
        self.assertEqual(_audit_tree_leaf_digest(record_bytes), expected)
        self.assertRegex(expected, DIGEST_RE)

    def test_node_digest_order_matters(self) -> None:
        a = "a" * 64
        b = "b" * 64
        self.assertEqual(
            _audit_tree_node_digest(a, b),
            hashlib.sha256(NODE_PREFIX + bytes.fromhex(a) + bytes.fromhex(b)).hexdigest(),
        )
        self.assertNotEqual(_audit_tree_node_digest(a, b), _audit_tree_node_digest(b, a))

    def test_empty_root_is_hash_of_nothing(self) -> None:
        self.assertEqual(_audit_tree_root([]), EMPTY_ROOT)

    def test_single_leaf_root_is_the_leaf(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        leaf = _audit_tree_leaf_digest(_audit_record_bytes("r1", op))
        self.assertEqual(_audit_tree_root([leaf]), leaf)

    def test_odd_nodes_are_promoted_unchanged(self) -> None:
        # Three leaves: root = node(node(L0,L1), L2) with L2 promoted once.
        leaves = [("%064x" % (i + 1)) for i in range(3)]
        left_pair = _audit_tree_node_digest(leaves[0], leaves[1])
        expected = _audit_tree_node_digest(left_pair, leaves[2])
        self.assertEqual(_audit_tree_root(leaves), expected)

    def test_proofs_for_all_sizes_and_indices(self) -> None:
        recs = records(12)
        all_leaves = build_tree(recs)
        for size in range(1, len(recs) + 1):
            leaves = all_leaves[:size]
            root = _audit_tree_root(leaves)
            for leaf_index in range(size):
                proof = _audit_tree_inclusion_proof(leaves, leaf_index)
                # Recombine from the proof alone.
                running = leaves[leaf_index]
                index = leaf_index
                level = leaves
                item_i = 0
                while len(level) > 1:
                    if index % 2 == 1 or index + 1 < len(level):
                        item = proof[item_i]
                        item_i += 1
                        sibling = bytes.fromhex(item["digest"])
                        current = bytes.fromhex(running)
                        if item["side"] == "left":
                            running = _audit_tree_node_digest(item["digest"], running)
                            self.assertEqual(item["position"], index - 1)
                        else:
                            running = _audit_tree_node_digest(running, item["digest"])
                            self.assertEqual(item["position"], index + 1)
                    level = (
                        [
                            _audit_tree_node_digest(level[i], level[i + 1])
                            for i in range(0, len(level) - 1, 2)
                        ]
                        + ([level[-1]] if len(level) % 2 else [])
                    )
                    index //= 2
                self.assertEqual(item_i, len(proof))
                self.assertEqual(running, root)


class ProofStoreTests(unittest.TestCase):
    """Snapshot and recovery semantics against StateStore directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def commit_all(self, recs: list[tuple[str, dict]]) -> None:
        for replica_id, op in recs:
            status = self.store.apply_operation(replica_id, op)
            self.assertIn(status, (201, 200))

    def test_empty_root_summary(self) -> None:
        summary = self.store.get_audit_proofs_root()
        self.assertEqual(
            summary,
            {
                "algorithm": "sha256",
                "treeSize": 0,
                "root": EMPTY_ROOT,
                "auditHead": GENESIS,
            },
        )

    def test_root_matches_independent_build_and_chain_head(self) -> None:
        recs = records(6)
        self.commit_all(recs)
        summary = self.store.get_audit_proofs_root()
        self.assertEqual(summary["algorithm"], "sha256")
        self.assertEqual(summary["treeSize"], 6)
        self.assertEqual(summary["root"], independent_root(build_tree(recs)))
        _, _, _, head = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(summary["auditHead"], head)

    def test_proof_shape_and_offline_verification(self) -> None:
        recs = records(7)
        self.commit_all(recs)
        root = self.store.get_audit_proofs_root()["root"]
        for index, (replica_id, op) in enumerate(recs):
            status, payload = self.store.get_operation_proof(
                replica_id, op["operationId"], 7, root
            )
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(list(payload), PROOF_FIELDS)
            self.assertEqual(payload["sequence"], index + 1)
            self.assertEqual(payload["treeSize"], 7)
            self.assertEqual(payload["root"], root)
            self.assertEqual(
                payload["record"], {"replicaId": replica_id, "operation": op}
            )
            verify_proof(payload)

    def test_prefix_proof_and_root_are_stable_under_append(self) -> None:
        recs = records(5)
        self.commit_all(recs)
        prefix_root = independent_root(build_tree(recs[:3]))
        status, before = self.store.get_operation_proof("r1", "o1", 3, prefix_root)
        self.assertIs(status, HTTPStatus.OK)
        # Append more records; the treeSize=3 proof and root do not change.
        self.commit_all(records(4, replica="s"))
        status, after = self.store.get_operation_proof("r1", "o1", 3, prefix_root)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(after, before)
        self.assertEqual(after["root"], prefix_root)

    def test_tree_size_zero_cannot_contain_any_leaf(self) -> None:
        recs = records(1)
        self.commit_all(recs)
        status, payload = self.store.get_operation_proof(
            "r1", "o1", 0, EMPTY_ROOT
        )
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unknown_identity_is_404(self) -> None:
        self.commit_all(records(2))
        root = self.store.get_audit_proofs_root()["root"]
        for replica_id, operation_id in [("r1", "nope"), ("nope", "o1")]:
            status, payload = self.store.get_operation_proof(
                replica_id, operation_id, 2, root
            )
            self.assertIs(status, HTTPStatus.NOT_FOUND)
            self.assertEqual(payload, {"error": "not_found"})

    def test_identity_outside_prefix_is_404_but_known_later(self) -> None:
        recs = records(3)
        self.commit_all(recs)
        # Prefix of length 2: record 3 exists but is outside the prefix.
        prefix2_root = independent_root(build_tree(recs[:2]))
        status, payload = self.store.get_operation_proof(
            "r3", "o3", 2, prefix2_root
        )
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})
        # Against the full prefix the same identity is provable.
        full_root = independent_root(build_tree(recs))
        status, payload = self.store.get_operation_proof("r3", "o3", 3, full_root)
        self.assertIs(status, HTTPStatus.OK)
        verify_proof(payload)

    def test_tree_size_past_log_length_raises(self) -> None:
        self.commit_all(records(1))
        root = self.store.get_audit_proofs_root()["root"]
        with self.assertRaises(ValueError):
            self.store.get_operation_proof("r1", "o1", 2, root)

    def test_root_mismatch_is_conflict(self) -> None:
        self.commit_all(records(3))
        status, payload = self.store.get_operation_proof(
            "r1", "o1", 3, "0" * 64
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "proof_conflict"})

    def test_same_snapshot_for_all_fields(self) -> None:
        # A batch import is one commit; proof fields must agree with the
        # prefix root summary computed over the same prefix length.
        recs = records(4)
        status, _, _ = self.store.import_operations(recs)
        self.assertIs(status, HTTPStatus.CREATED)
        current = self.store.get_audit_proofs_root()
        status, payload = self.store.get_operation_proof(
            "r2", "o2", current["treeSize"], current["root"]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["root"], current["root"])
        self.assertEqual(payload["auditHead"], current["auditHead"])
        self.assertEqual(payload["treeSize"], current["treeSize"])

    def test_read_only(self) -> None:
        self.commit_all(records(3))
        root = self.store.get_audit_proofs_root()
        metrics_before = self.store.get_metrics()
        self.store.get_audit_proofs_root()
        self.store.get_operation_proof("r1", "o1", 3, root["root"])
        self.store.get_operation_proof("r9", "o9", 3, root["root"])
        self.store.get_operation_proof("r1", "o1", 3, "0" * 64)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(self.store.get_audit_proofs_root(), root)

    def test_recovery_preserves_root_and_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            recs = records(5)
            for replica_id, op in recs:
                store.apply_operation(replica_id, op)
            root_summary = store.get_audit_proofs_root()
            proofs = {}
            for replica_id, op in recs:
                status, payload = store.get_operation_proof(
                    replica_id, op["operationId"], 5, root_summary["root"]
                )
                self.assertIs(status, HTTPStatus.OK)
                proofs[(replica_id, op["operationId"])] = payload

            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_audit_proofs_root(), root_summary)
            for key, payload in proofs.items():
                status, recovered_payload = recovered.get_operation_proof(
                    key[0], key[1], 5, root_summary["root"]
                )
                self.assertIs(status, HTTPStatus.OK)
                self.assertEqual(recovered_payload, payload)


class ProofHttpTests(unittest.TestCase):
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
    ) -> tuple[int, dict, bytes]:
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
        conn.close()
        return response.status, payload, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _ = self.raw_request(method, path, body)
        return status, payload

    def seed(self, count: int = 5) -> list[tuple[str, dict]]:
        recs = records(count)
        for replica_id, op in recs:
            status, _ = self.request(
                "POST", f"/v1/replicas/{replica_id}/operations", op
            )
            assert status == 201
        return recs

    def current_root(self) -> dict:
        status, payload, _ = self.raw_request("GET", "/v1/audit/proofs/root")
        assert status == 200
        return payload

    def proof(self, replica: str, op_id: str, query: str) -> tuple[int, dict, bytes]:
        return self.raw_request(
            "GET", f"/v1/replicas/{replica}/operations/{op_id}/proof{query}"
        )

    # ---- /v1/audit/proofs/root ----

    def test_root_empty_shape_and_framing(self) -> None:
        status, payload, raw = self.raw_request("GET", "/v1/audit/proofs/root")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["treeSize"], 0)
        self.assertEqual(payload["root"], EMPTY_ROOT)
        self.assertEqual(payload["auditHead"], GENESIS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        # Field order is the contracted order, not sorted.
        self.assertEqual(raw[:-1], _ordered_json_bytes(payload))

    def test_root_after_writes(self) -> None:
        recs = self.seed(4)
        payload = self.current_root()
        self.assertEqual(payload["treeSize"], 4)
        self.assertEqual(payload["root"], independent_root(build_tree(recs)))
        status, chain, _ = self.raw_request("GET", "/v1/audit/log/chain")
        self.assertEqual(payload["auditHead"], chain["head"])

    def test_root_any_query_parameter_is_400(self) -> None:
        for query in ("?x=1", "?treeSize=1", "?root=" + "a" * 64, "?x", "?x=", "?=1"):
            with self.subTest(query=query):
                status, payload, _ = self.raw_request(
                    "GET", f"/v1/audit/proofs/root{query}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_root_route_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/audit/proofs",
            "/v1/audit/proofs/root/extra",
            "/v1/audit/proofs/root/",
            "/v1/audit/proofs/other",
            "/v1/audit/root",
        ):
            with self.subTest(path=path):
                status, payload, _ = self.raw_request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_root_shape_check_precedes_query_check(self) -> None:
        status, payload, _ = self.raw_request(
            "GET", "/v1/audit/proofs/root/extra?x=1"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # ---- per-operation proof ----

    def test_proof_success_and_offline_verification(self) -> None:
        recs = self.seed(6)
        root = self.current_root()["root"]
        for index, (replica_id, op) in enumerate(recs):
            status, payload, raw = self.proof(
                replica_id, op["operationId"], f"?treeSize=6&root={root}"
            )
            self.assertEqual(status, 200)
            self.assertEqual(list(payload), PROOF_FIELDS)
            self.assertEqual(payload["record"], {"replicaId": replica_id, "operation": op})
            self.assertEqual(payload["sequence"], index + 1)
            self.assertEqual(payload["treeSize"], 6)
            self.assertEqual(payload["root"], root)
            self.assertRegex(payload["leafDigest"], DIGEST_RE)
            for item in payload["proof"]:
                self.assertEqual(set(item), {"side", "digest", "position"})
                self.assertIn(item["side"], ("left", "right"))
                self.assertIs(type(item["position"]), int)
            verify_proof(payload)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\n", raw[:-1])
            self.assertEqual(raw[:-1], _ordered_json_bytes(payload))

    def test_proof_for_a_smaller_prefix(self) -> None:
        self.seed(5)
        # Compute the treeSize=3 root by requesting each prefix directly via
        # the store-independent chain: use the first three records' tree.
        recs = records(5)
        prefix_root = independent_root(build_tree(recs[:3]))
        status, payload, _ = self.proof("r2", "o2", f"?treeSize=3&root={prefix_root}")
        self.assertEqual(status, 200)
        self.assertEqual(payload["treeSize"], 3)
        self.assertEqual(payload["root"], prefix_root)
        verify_proof(payload)

    def test_append_does_not_change_old_prefix_proof(self) -> None:
        self.seed(3)
        recs = records(3)
        prefix_root = independent_root(build_tree(recs))
        status, before, _ = self.proof("r1", "o1", f"?treeSize=3&root={prefix_root}")
        self.assertEqual(status, 200)
        # Append two genuinely new records (distinct identities).
        appended = [
            ("r4", operation("o4", "k", "v4", {"r4": 4})),
            ("r5", operation("o5", "k", "v5", {"r5": 5})),
        ]
        for replica_id, op in appended:
            status, _ = self.request(
                "POST", f"/v1/replicas/{replica_id}/operations", op
            )
            self.assertEqual(status, 201)
        status, after, _ = self.proof("r1", "o1", f"?treeSize=3&root={prefix_root}")
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        # The new full-prefix root differs but still verifies each record.
        full_root = independent_root(build_tree(recs + appended))
        self.assertNotEqual(full_root, prefix_root)
        status, payload, _ = self.proof("r5", "o5", f"?treeSize=5&root={full_root}")
        self.assertEqual(status, 200)
        verify_proof(payload)

    def test_proof_missing_parameters_are_400(self) -> None:
        self.seed(2)
        root = self.current_root()["root"]
        bad = [
            "",
            "?treeSize=2",
            f"?root={root}",
            f"?treeSize=2&root={root}&x=1",
            f"?treeSize=2&treeSize=1&root={root}",
            f"?treeSize=2&root={root}&root={root}",
        ]
        for query in bad:
            with self.subTest(query=query):
                status, payload, _ = self.proof("r1", "o1", query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_proof_bad_treesize_is_400(self) -> None:
        self.seed(2)
        root = self.current_root()["root"]
        for tree_size in ("-1", "0x1", "1.0", "+1", "%201", "abc", "", "%D9%A1"):
            query = f"?treeSize={tree_size}&root={root}"
            with self.subTest(tree_size=tree_size):
                status, payload, _ = self.proof("r1", "o1", query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_proof_treesize_past_log_length_is_400(self) -> None:
        self.seed(2)
        root = self.current_root()["root"]
        status, payload, _ = self.proof("r1", "o1", f"?treeSize=3&root={root}")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_proof_bad_root_is_400(self) -> None:
        self.seed(2)
        for bad_root in ("", "abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65, "0" * 32):
            query = f"?treeSize=2&root={bad_root}"
            with self.subTest(bad_root=bad_root):
                status, payload, _ = self.proof("r1", "o1", query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_unknown_operation_is_404(self) -> None:
        self.seed(2)
        root = self.current_root()["root"]
        for replica, op_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            status, payload, _ = self.proof(
                replica, op_id, f"?treeSize=2&root={root}"
            )
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

    def test_operation_outside_prefix_is_404(self) -> None:
        self.seed(3)
        recs = records(3)
        prefix_root = independent_root(build_tree(recs[:2]))
        status, payload, _ = self.proof(
            "r3", "o3", f"?treeSize=2&root={prefix_root}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_root_mismatch_is_409(self) -> None:
        self.seed(3)
        status, payload, _ = self.proof("r1", "o1", f"?treeSize=3&root={'0' * 64}")
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "proof_conflict"})

    def test_404_boundary_precedes_root_comparison(self) -> None:
        # An out-of-prefix leaf is 404 even when the supplied root is wrong;
        # membership is decided before the root agreement check.
        self.seed(3)
        status, payload, _ = self.proof(
            "r3", "o3", f"?treeSize=2&root={'0' * 64}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_proof_route_shape_mismatches_are_404(self) -> None:
        self.seed(1)
        root = self.current_root()["root"]
        suffix = f"?treeSize=1&root={root}"
        for path in (
            f"/v1/replicas/r1/operations/o1/proof/extra{suffix}",
            f"/v1/replicas/r1/operations/o1/proof/{suffix}",
            f"/v1/replicas//operations/o1/proof{suffix}",
            f"/v1/replicas/r1/proof{suffix}",
        ):
            with self.subTest(path=path):
                status, payload, _ = self.raw_request("GET", path)
                self.assertEqual(status, 404, path)
                self.assertEqual(payload, {"error": "not_found"}, path)

    def test_operations_proof_without_id_hits_archive_route(self) -> None:
        # Five segments with a trailing "proof" identity is not the proof
        # route at all: it is the existing per-operation archive route
        # (operationId "proof"), whose strict no-query contract answers 400
        # before the unknown identity can be resolved to a 404.
        self.seed(1)
        root = self.current_root()["root"]
        status, payload, _ = self.raw_request(
            "GET", f"/v1/replicas/r1/operations/proof?treeSize=1&root={root}"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_proof_shape_check_precedes_query_check(self) -> None:
        status, payload, _ = self.raw_request(
            "GET", "/v1/replicas/r1/operations/o1/proof/extra?treeSize=x"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_proof_percent_decodes_path_segments(self) -> None:
        op = operation("op 1", "k", "v", {"r/1": 1})
        status, _ = self.request("POST", "/v1/replicas/r%2F1/operations", op)
        self.assertEqual(status, 201)
        root = self.current_root()["root"]
        status, payload, _ = self.raw_request(
            "GET",
            f"/v1/replicas/r%2F1/operations/op%201/proof?treeSize=1&root={root}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["record"], {"replicaId": "r/1", "operation": op})
        verify_proof(payload)

    def test_query_is_read_only_over_http(self) -> None:
        self.seed(2)
        root = self.current_root()
        before, _, _ = self.raw_request("GET", "/v1/metrics")
        self.raw_request("GET", "/v1/audit/proofs/root")
        self.proof("r1", "o1", f"?treeSize=2&root={root['root']}")
        self.proof("r9", "o9", f"?treeSize=2&root={root['root']}")
        after, _, _ = self.raw_request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_proof_routes_is_404(self) -> None:
        status, payload, _ = self.raw_request(
            "POST", "/v1/audit/proofs/root", {"x": 1}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload, _ = self.raw_request(
            "POST", "/v1/replicas/r1/operations/o1/proof", {"x": 1}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ProofAuthTests(unittest.TestCase):
    """The proof endpoints authenticate like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret-token"
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path: str, headers: list[tuple[str, str]]) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers=dict(headers))
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_root_requires_token(self) -> None:
        for headers in ([], [("Authorization", "Bearer wrong")]):
            status, payload = self.get("/v1/audit/proofs/root", headers)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})

    def test_root_accepts_valid_token(self) -> None:
        status, payload = self.get(
            "/v1/audit/proofs/root", [("Authorization", "Bearer s3cret-token")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)

    def test_proof_requires_token(self) -> None:
        status, payload = self.get(
            f"/v1/replicas/r1/operations/o1/proof?treeSize=0&root={EMPTY_ROOT}", []
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})


class ProofPersistenceHttpTests(unittest.TestCase):
    """Root and proofs survive a data-file restart unchanged."""

    def test_restart_preserves_root_and_proof(self) -> None:
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
                        results.append(
                            (response.status, json.loads(raw.decode("utf-8")) if raw else None)
                        )
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            op = operation("o1", "k", "v", {"r1": 1})
            self.assertEqual(
                serve_once([("POST", "/v1/replicas/r1/operations", op)])[0][0], 201
            )

            def reads(root: str) -> list:
                return [
                    ("GET", "/v1/audit/proofs/root", None),
                    (
                        "GET",
                        f"/v1/replicas/r1/operations/o1/proof?treeSize=1&root={root}",
                        None,
                    ),
                ]

            first = serve_once(reads("0" * 64))
            root_summary = first[0][1]
            self.assertEqual(first[0][0], 200)
            # First proof call used a placeholder root, so it is a 409; now
            # fetch it with the real root in a second restart round.
            second = serve_once(reads(root_summary["root"]))
            self.assertEqual(second[0], first[0])
            self.assertEqual(second[0][1], root_summary)
            self.assertEqual(second[1][0], 200)
            proof_payload = second[1][1]
            self.assertEqual(proof_payload["sequence"], 1)
            self.assertEqual(proof_payload["treeSize"], 1)
            self.assertEqual(proof_payload["root"], root_summary["root"])
            self.assertEqual(proof_payload["auditHead"], root_summary["auditHead"])
            verify_proof(proof_payload)


if __name__ == "__main__":
    unittest.main()
