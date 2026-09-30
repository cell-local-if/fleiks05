"""Tests for the offline audit-inclusion proofs::

    GET  /v1/audit/proofs/root
    GET  /v1/replicas/{replicaId}/operations/{operationId}/proof?treeSize=N&root=H

The root route returns the current log prefix's ``algorithm``,
``treeSize``, ``root``, and ``auditHead`` and accepts no query
parameters at all. The proof route pins one prefix (``treeSize`` from 0
to the current log length, ``root`` the 64-character lowercase
hexadecimal SHA-256 prefix-tree root the caller expects) and returns the
existing archive ``record``, its 1-based global ``sequence``, the same
``treeSize``/``root``/``auditHead``, the ``leafDigest``, and a
leaf-to-root ``proof`` whose items each carry ``side``, ``digest``, and
``position``.

A third party recomputes everything offline using only the record's
canonical audit bytes and the published, stable prefix-tree rules:

- leaf digest   = SHA256(0x00 || canonical record bytes);
- internal node = SHA256(0x01 || left32 || right32), raw digest bytes;
- each level walks the level below in adjacent left/right pairs, and an
  odd trailing node is promoted to the next level unchanged;
- the verifier starts from the leaf digest and combines each proof
  sibling on the advertised side until it reaches the root.

The tests here carry an *independent* reference implementation of those
rules and confirm the server's root and every leaf's proof against it
for prefixes of every size, that appending never changes an earlier
prefix's root or proof, the strict HTTP contract (404 path shape before
the 400 query check, 404 for unknown or out-of-prefix operations, 409
for a mismatched root), restart stability under ``--data-file``, and
the compact single-newline framing. Only the Python standard library is
used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    AUDIT_PROOF_ALGORITHM,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _audit_record_bytes,
    parse_audit_proof_query,
)

GENESIS = "0" * 64
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

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
PROOF_ITEM_FIELDS = ["side", "digest", "position"]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


# ---------------------------------------------------------------------------
# Independent reference implementation of the published prefix-tree rules.
# ---------------------------------------------------------------------------


def reference_leaf(replica: str, op: dict) -> str:
    return hashlib.sha256(
        LEAF_PREFIX + _audit_record_bytes(replica, op)
    ).hexdigest()


def reference_node(left: str, right: str) -> str:
    return hashlib.sha256(
        NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()


def reference_levels(records: list[tuple[str, dict]]) -> list[list[str]]:
    """All levels of digests; level 0 is the leaves."""
    levels = [[reference_leaf(replica, op) for replica, op in records]]
    if not levels[0]:
        return []
    current = levels[0]
    while len(current) > 1:
        nxt = []
        index = 0
        while index + 1 < len(current):
            nxt.append(reference_node(current[index], current[index + 1]))
            index += 2
        if index < len(current):
            nxt.append(current[index])  # odd-leaf promotion
        levels.append(nxt)
        current = nxt
    return levels


def reference_root(records: list[tuple[str, dict]]) -> str:
    levels = reference_levels(records)
    return levels[-1][0] if levels else GENESIS


def reference_positions(width: int) -> list[list[int]]:
    """Smallest covered leaf index for every node of every level."""
    positions = [list(range(width))]
    if width == 0:
        return []
    current = positions[0]
    while len(current) > 1:
        nxt = []
        index = 0
        while index + 1 < len(current):
            nxt.append(current[index])  # the parent keeps the left child's
            index += 2
        if index < len(current):
            nxt.append(current[index])
        positions.append(nxt)
        current = nxt
    return positions


def reference_proof(
    records: list[tuple[str, dict]], leaf_index: int
) -> tuple[str, list[dict]]:
    """Independently derive ``(leaf_digest, proof)`` for one leaf."""
    levels = reference_levels(records)
    positions = reference_positions(len(records))
    leaf_digest = levels[0][leaf_index]
    proof = []
    index = leaf_index
    for level_number in range(1, len(levels)):
        lower = levels[level_number - 1]
        lower_positions = positions[level_number - 1]
        if index % 2 == 1:
            proof.append(
                {
                    "side": "left",
                    "digest": lower[index - 1],
                    "position": lower_positions[index - 1],
                }
            )
        elif index + 1 < len(lower):
            proof.append(
                {
                    "side": "right",
                    "digest": lower[index + 1],
                    "position": lower_positions[index + 1],
                }
            )
        # An odd trailing node is promoted: no sibling item.
        index //= 2
    return leaf_digest, proof


def reference_chain_head(records: list[tuple[str, dict]]) -> str:
    previous = GENESIS
    for index, (replica, op) in enumerate(records):
        previous = hashlib.sha256(
            previous.encode("ascii")
            + str(index + 1).encode("ascii")
            + _audit_record_bytes(replica, op)
        ).hexdigest()
    return previous


def offline_recompute(leaf_digest: str, proof: list[dict]) -> str:
    """Recompute the root from only a leaf digest and its proof items."""
    running = leaf_digest
    for item in proof:
        assert list(item) == PROOF_ITEM_FIELDS
        assert item["side"] in ("left", "right")
        assert isinstance(item["position"], int) and item["position"] >= 0
        sibling = bytes.fromhex(item["digest"])
        if item["side"] == "left":
            running = hashlib.sha256(
                NODE_PREFIX + sibling + bytes.fromhex(running)
            ).hexdigest()
        else:
            running = hashlib.sha256(
                NODE_PREFIX + bytes.fromhex(running) + sibling
            ).hexdigest()
    return running


def seed_records(count: int) -> list[tuple[str, dict]]:
    """Deterministic multi-replica history of ``count`` records."""
    records = []
    for index in range(1, count + 1):
        replica = f"r{index % 3 + 1}"
        clock = {replica: index}
        records.append(
            (replica, operation(f"o{index}", f"k{index % 4}", f"v{index}", clock))
        )
    return records


class TreeRuleTests(unittest.TestCase):
    """The stable hashing rules, pinned independently of the server."""

    def test_leaf_digest_pins_0x00_prefixed_record_bytes(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        raw = b'\x00{"replicaId":"r1","operation":'
        raw += b'{"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}'
        self.assertEqual(reference_leaf("r1", op), hashlib.sha256(raw).hexdigest())

    def test_node_digest_pins_0x01_prefixed_raw_digest_bytes(self) -> None:
        op_a = operation("o1", "k", "a", {"r1": 1})
        op_b = operation("o2", "k", "b", {"r2": 1})
        left = reference_leaf("r1", op_a)
        right = reference_leaf("r2", op_b)
        expected = hashlib.sha256(
            NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)
        ).hexdigest()
        self.assertEqual(reference_node(left, right), expected)
        # Order is significant.
        self.assertNotEqual(reference_node(left, right), reference_node(right, left))

    def test_odd_tail_is_promoted_not_duplicated(self) -> None:
        records = seed_records(3)
        leaves = reference_levels(records)[0]
        paired = reference_node(leaves[0], leaves[1])
        # The promoted third leaf pairs with the pair digest unchanged.
        self.assertEqual(reference_root(records), reference_node(paired, leaves[2]))
        # And the leaf itself is not re-hashed against itself.
        self.assertNotEqual(
            reference_root(records),
            reference_node(paired, reference_node(leaves[2], leaves[2])),
        )

    def test_power_of_two_and_odd_sizes(self) -> None:
        for count in range(0, 21):
            records = seed_records(count)
            levels = reference_levels(records)
            if count == 0:
                self.assertEqual(levels, [])
                self.assertEqual(reference_root(records), GENESIS)
                continue
            self.assertEqual(len(levels[0]), count)
            self.assertEqual(len(levels[-1]), 1)
            self.assertEqual(
                len(levels[-2]) if len(levels) > 1 else count,
                min(count, 2),
            )


class ProofReferenceTests(unittest.TestCase):
    """Every leaf of every prefix verifies against the reference root."""

    def test_all_leaves_of_all_prefixes(self) -> None:
        for count in range(1, 21):
            records = seed_records(count)
            root = reference_root(records)
            for leaf_index in range(count):
                leaf_digest, proof = reference_proof(records, leaf_index)
                self.assertEqual(
                    offline_recompute(leaf_digest, proof),
                    root,
                    f"count={count} leaf={leaf_index}",
                )

    def test_proof_items_are_leaf_to_root_and_positions_are_starts(self) -> None:
        records = seed_records(7)
        for leaf_index in range(7):
            _, proof = reference_proof(records, leaf_index)
            for item in proof:
                self.assertEqual(list(item), PROOF_ITEM_FIELDS)
            # Each proof step moves strictly upward (item list leaf-first).
            self.assertLessEqual(len(proof), 3)

    def test_single_leaf_proof_is_empty(self) -> None:
        records = seed_records(1)
        leaf, proof = reference_proof(records, 0)
        self.assertEqual(proof, [])
        self.assertEqual(offline_recompute(leaf, proof), reference_root(records))


class QueryParserTests(unittest.TestCase):
    def test_accepts_exactly_tree_size_and_root_once(self) -> None:
        self.assertEqual(
            parse_audit_proof_query(f"treeSize=3&root={'a' * 64}"),
            (3, "a" * 64),
        )
        self.assertEqual(parse_audit_proof_query("treeSize=0&root=" + "f" * 64),
                         (0, "f" * 64))

    def test_rejects_bad_queries(self) -> None:
        bad = [
            "",
            "treeSize=3",
            f"root={'a' * 64}",
            "treeSize=&root=" + "a" * 64,
            f"treeSize=-1&root={'a' * 64}",
            f"treeSize=1.0&root={'a' * 64}",
            f"treeSize=x&root={'a' * 64}",
            f"treeSize=%201&root={'a' * 64}",
            "treeSize=01&root=" + "A" * 64,  # valid int, uppercase root
            f"treeSize=3&root={'a' * 63}",
            f"treeSize=3&root={'g' * 64}",
            f"treeSize=3&root=0x{'a' * 62}",
            f"treeSize=3&treeSize=4&root={'a' * 64}",
            f"treeSize=3&root={'a' * 64}&root={'b' * 64}",
            f"treeSize=3&root={'a' * 64}&unknown=1",
            "x=1",
            "x",
            "=1",
            f"treeSize=3&root=",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(parse_audit_proof_query(query))


class StoreProofTests(unittest.TestCase):
    """Snapshot, identity, and recovery semantics against StateStore."""

    def setUp(self) -> None:
        self.store = StateStore()

    def commit_records(self, records: list[tuple[str, dict]]) -> None:
        for replica, op in records:
            status = self.store.apply_operation(replica, op)
            self.assertIn(status, (200, 201))

    def test_empty_prefix_summary(self) -> None:
        self.assertEqual(
            self.store.get_audit_proofs_root(),
            {
                "algorithm": AUDIT_PROOF_ALGORITHM,
                "treeSize": 0,
                "root": GENESIS,
                "auditHead": GENESIS,
            },
        )

    def test_root_summary_matches_reference_and_chain_head(self) -> None:
        records = seed_records(6)
        self.commit_records(records)
        summary = self.store.get_audit_proofs_root()
        self.assertEqual(summary["algorithm"], "sha256")
        self.assertEqual(summary["treeSize"], 6)
        self.assertEqual(summary["root"], reference_root(records))
        self.assertEqual(summary["auditHead"], reference_chain_head(records))

    def test_successful_proof_body_and_offline_recomputation(self) -> None:
        records = seed_records(5)
        self.commit_records(records)
        summary = self.store.get_audit_proofs_root()
        for leaf_index, (replica, op) in enumerate(records):
            status, payload = self.store.get_operation_audit_proof(
                replica, op["operationId"], summary["treeSize"], summary["root"]
            )
            self.assertEqual(status, 200)
            self.assertEqual(list(payload), PROOF_FIELDS)
            self.assertEqual(
                payload["record"], {"replicaId": replica, "operation": op}
            )
            self.assertEqual(payload["sequence"], leaf_index + 1)
            self.assertEqual(payload["treeSize"], 5)
            self.assertEqual(payload["root"], summary["root"])
            self.assertEqual(payload["auditHead"], summary["auditHead"])
            self.assertEqual(
                payload["leafDigest"], reference_leaf(replica, op)
            )
            self.assertEqual(
                payload["proof"], reference_proof(records, leaf_index)[1]
            )
            self.assertEqual(
                offline_recompute(payload["leafDigest"], payload["proof"]),
                payload["root"],
            )

    def test_single_record_prefix_has_empty_proof(self) -> None:
        records = seed_records(1)
        self.commit_records(records)
        status, payload = self.store.get_operation_audit_proof(
            "r2", "o1", 1, reference_root(records)
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["proof"], [])
        self.assertEqual(payload["leafDigest"], payload["root"])

    def test_proof_for_a_pinned_earlier_prefix(self) -> None:
        records = seed_records(4)
        self.commit_records(records[:2])
        pinned_root = reference_root(records[:2])
        pinned_head = reference_chain_head(records[:2])
        self.commit_records(records[2:])
        # The old treeSize's root, head, and proofs survive the append.
        status, payload = self.store.get_operation_audit_proof(
            "r2", "o1", 2, pinned_root
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["treeSize"], 2)
        self.assertEqual(payload["root"], pinned_root)
        self.assertEqual(payload["auditHead"], pinned_head)
        self.assertEqual(payload["proof"], reference_proof(records[:2], 0)[1])
        self.assertEqual(
            offline_recompute(payload["leafDigest"], payload["proof"]), pinned_root
        )
        # A leaf committed after the prefix is not found inside it.
        status, payload = self.store.get_operation_audit_proof(
            records[3][0], records[3][1]["operationId"], 2, pinned_root
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_error_matrix(self) -> None:
        records = seed_records(3)
        self.commit_records(records)
        summary = self.store.get_audit_proofs_root()
        error_bodies = {
            400: {"error": "invalid_request"},
            404: {"error": "not_found"},
            409: {"error": "proof_conflict"},
        }
        cases = [
            ("rX", "o1", 3, summary["root"], 404),        # unknown replica
            ("r2", "missing", 3, summary["root"], 404),    # unknown operation
            ("r2", "o1", 0, summary["root"], 404),         # empty prefix
            (records[2][0], records[2][1]["operationId"], 2,
             summary["root"], 404),                        # outside prefix
            ("r2", "o1", 4, summary["root"], 400),         # past the log
            ("r2", "o1", 3, "a" * 64, 409),               # root mismatch
        ]
        for replica, op_id, size, root, expected in cases:
            with self.subTest(replica=replica, op=op_id, size=size):
                status, payload = self.store.get_operation_audit_proof(
                    replica, op_id, size, root
                )
                self.assertEqual(status, expected)
                self.assertEqual(payload, error_bodies[expected])

    def test_conflict_takes_one_snapshot_with_consistent_fields(self) -> None:
        records = seed_records(2)
        self.commit_records(records)
        status, payload = self.store.get_operation_audit_proof(
            "r2", "o1", 2, "a" * 64
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "proof_conflict"})

    def test_repairs_and_imports_enter_the_tree_in_commit_order(self) -> None:
        op1 = operation("o1", "k", "a", {"r1": 2, "r2": 1})
        op2 = operation("o2", "k", "b", {"r2": 1})  # stale
        self.assertIn(self.store.apply_operation("r1", op1), (200, 201))
        self.assertIn(self.store.apply_operation("r2", op2), (200, 201))
        imported = [
            {"replicaId": "r5",
             "operation": operation("o5", "other", "x", {"r5": 1})},
        ]
        status, _, _ = self.store.import_operations(
            [("r5", imported[0]["operation"])]
        )
        self.assertEqual(status, 201)
        records = [("r1", op1), ("r2", op2), ("r5", imported[0]["operation"])]
        summary = self.store.get_audit_proofs_root()
        self.assertEqual(summary["root"], reference_root(records))
        status, payload = self.store.get_operation_audit_proof(
            "r5", "o5", 3, summary["root"]
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["sequence"], 3)

    def test_recovery_rebuilds_identical_roots_and_proofs(self) -> None:
        records = seed_records(7)
        with tempfile.TemporaryDirectory() as directory:
            data_file = os.path.join(directory, "state.json")
            store = StateStore(data_file=data_file)
            for replica, op in records:
                self.assertIn(store.apply_operation(replica, op), (200, 201))
            before_root = store.get_audit_proofs_root()
            before_proofs = {}
            for size in range(1, 8):
                root = reference_root(records[:size])
                for replica, op in records[:size]:
                    status, payload = store.get_operation_audit_proof(
                        replica, op["operationId"], size, root
                    )
                    self.assertEqual(status, 200)
                    before_proofs[(size, replica, op["operationId"])] = payload
            # Force the durable file to exist before recovery.
            self.assertTrue(Path(data_file).exists())
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_audit_proofs_root(), before_root)
            for key, payload in before_proofs.items():
                size, replica, op_id = key
                root = payload["root"]
                status, reopened = recovered.get_operation_audit_proof(
                    replica, op_id, size, root
                )
                self.assertEqual(status, 200, key)
                self.assertEqual(reopened, payload, key)

    def test_concurrent_reads_always_verify_during_appends(self) -> None:
        records = seed_records(80)
        stop = threading.Event()
        failures = []

        def reader() -> None:
            while not stop.is_set():
                summary = self.store.get_audit_proofs_root()
                size = summary["treeSize"]
                if size == 0:
                    continue
                replica, op = records[size - 1]
                status, payload = self.store.get_operation_audit_proof(
                    replica, op["operationId"], size, summary["root"]
                )
                if status != 200:
                    failures.append((status, payload))
                    return
                recomputed = offline_recompute(
                    payload["leafDigest"], payload["proof"]
                )
                if recomputed != payload["root"] or payload["treeSize"] != size:
                    failures.append("half-batch snapshot")
                    return

        workers = [threading.Thread(target=reader) for _ in range(4)]
        for worker in workers:
            worker.start()
        for replica, op in records:
            self.assertIn(self.store.apply_operation(replica, op), (200, 201))
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
        self.assertEqual(failures, [])


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
    ) -> tuple[int, object, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        status, payload, _ = self.raw_request(method, path, body)
        return status, payload

    def post_operation(self, replica: str, op: dict) -> None:
        status, _ = self.request(
            "POST", f"/v1/replicas/{replica}/operations", op
        )
        self.assertEqual(status, 201)

    def seed(self, count: int) -> list[tuple[str, dict]]:
        records = seed_records(count)
        for replica, op in records:
            self.post_operation(replica, op)
        return records

    def proof_path(self, replica: str, op_id: str) -> str:
        return f"/v1/replicas/{replica}/operations/{op_id}/proof"

    # -- root route --------------------------------------------------------

    def test_root_on_empty_log(self) -> None:
        status, payload, raw = self.raw_request("GET", "/v1/audit/proofs/root")
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ROOT_FIELDS)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["treeSize"], 0)
        self.assertEqual(payload["root"], GENESIS)
        self.assertEqual(payload["auditHead"], GENESIS)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        # Field order is the contracted order, not the sorted order.
        self.assertEqual(
            raw[:-1],
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )

    def test_root_after_writes_matches_reference(self) -> None:
        records = self.seed(7)
        status, payload = self.request("GET", "/v1/audit/proofs/root")
        self.assertEqual(status, 200)
        self.assertEqual(payload["treeSize"], 7)
        self.assertEqual(payload["root"], reference_root(records))
        self.assertEqual(payload["auditHead"], reference_chain_head(records))

    def test_root_rejects_any_query_parameter(self) -> None:
        for query in [
            "?treeSize=1",
            "?root=" + "a" * 64,
            "?unknown=1",
            "?x",
            "?x=",
            "?=1",
        ]:
            with self.subTest(query=query):
                status, payload = self.request(
                    "GET", "/v1/audit/proofs/root" + query
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_bare_question_mark_is_an_empty_query(self) -> None:
        status, payload = self.request("GET", "/v1/audit/proofs/root?")
        self.assertEqual(status, 200)
        self.assertEqual(payload["treeSize"], 0)

    def test_root_route_shape_mismatches_are_404(self) -> None:
        for path in [
            "/v1/audit/proofs",
            "/v1/audit/proofs/root/extra",
            "/v1/audit/proofs/root/",
            "/v1/audit/proofs/other",
            "/v1/audit/proof/root",
        ]:
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_root_shape_check_precedes_query_check(self) -> None:
        status, payload = self.request(
            "GET", "/v1/audit/proofs/root/extra?treeSize=1"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    # -- proof route -------------------------------------------------------

    def test_proof_success_body_and_offline_recomputation(self) -> None:
        records = self.seed(7)
        _, root_payload, _ = self.raw_request("GET", "/v1/audit/proofs/root")
        root = root_payload["root"]
        for leaf_index, (replica, op) in enumerate(records):
            status, payload, raw = self.raw_request(
                "GET",
                f"{self.proof_path(replica, op['operationId'])}"
                f"?treeSize=7&root={root}",
            )
            self.assertEqual(status, 200)
            self.assertEqual(list(payload), PROOF_FIELDS)
            self.assertEqual(
                payload["record"], {"replicaId": replica, "operation": op}
            )
            self.assertEqual(payload["sequence"], leaf_index + 1)
            self.assertEqual(payload["treeSize"], 7)
            self.assertEqual(payload["root"], root)
            self.assertEqual(payload["auditHead"], root_payload["auditHead"])
            self.assertEqual(payload["leafDigest"], reference_leaf(replica, op))
            self.assertEqual(
                payload["proof"], reference_proof(records, leaf_index)[1]
            )
            self.assertEqual(
                offline_recompute(payload["leafDigest"], payload["proof"]), root
            )
            # Compact ordered JSON with exactly one trailing newline.
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\n", raw[:-1])
            self.assertEqual(
                raw[:-1],
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )

    def test_proof_against_a_prefix_smaller_than_the_log(self) -> None:
        records = self.seed(5)
        prefix = records[:3]
        pinned_root = reference_root(prefix)
        pinned_head = reference_chain_head(prefix)
        status, payload = self.request(
            "GET",
            f"{self.proof_path(records[1][0], records[1][1]['operationId'])}"
            f"?treeSize=3&root={pinned_root}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["treeSize"], 3)
        self.assertEqual(payload["root"], pinned_root)
        self.assertEqual(payload["auditHead"], pinned_head)
        self.assertEqual(payload["sequence"], 2)
        self.assertEqual(payload["proof"], reference_proof(prefix, 1)[1])
        self.assertEqual(
            offline_recompute(payload["leafDigest"], payload["proof"]), pinned_root
        )

    def test_pinned_prefix_survives_later_appends(self) -> None:
        records = self.seed(3)
        pinned_root = reference_root(records)
        pinned_head = reference_chain_head(records)
        # Append a second batch.
        more = seed_records(3)
        more = [
            (replica, {**op, "operationId": op["operationId"] + "b",
                       "clock": {replica: 9 + index}})
            for index, (replica, op) in enumerate(more, start=1)
        ]
        for replica, op in more:
            self.post_operation(replica, op)
        _, current, _ = self.raw_request("GET", "/v1/audit/proofs/root")
        self.assertEqual(current["treeSize"], 6)
        self.assertNotEqual(current["root"], pinned_root)
        # The old treeSize still answers with the exact old root/head/proof.
        for leaf_index, (replica, op) in enumerate(records):
            status, payload = self.request(
                "GET",
                f"{self.proof_path(replica, op['operationId'])}"
                f"?treeSize=3&root={pinned_root}",
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload["root"], pinned_root)
            self.assertEqual(payload["auditHead"], pinned_head)
            self.assertEqual(
                payload["proof"], reference_proof(records, leaf_index)[1]
            )

    def test_tree_size_zero_never_contains_an_operation(self) -> None:
        self.seed(1)
        status, payload = self.request(
            "GET",
            f"{self.proof_path('r2', 'o1')}?treeSize=0&root={GENESIS}",
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unknown_or_out_of_prefix_operation_is_404(self) -> None:
        records = self.seed(3)
        root = reference_root(records)
        for path in [
            f"{self.proof_path('rX', 'o1')}?treeSize=3&root={root}",
            f"{self.proof_path('r2', 'missing')}?treeSize=3&root={root}",
            # o3 is at sequence 3; prefix of size 2 excludes it.
            f"{self.proof_path(records[2][0], records[2][1]['operationId'])}"
            f"?treeSize=2&root={reference_root(records[:2])}",
        ]:
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_tree_size_past_current_log_length_is_400(self) -> None:
        records = self.seed(2)
        root = reference_root(records)
        status, payload = self.request(
            "GET", f"{self.proof_path('r2', 'o1')}?treeSize=3&root={root}"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_root_mismatch_is_409(self) -> None:
        self.seed(2)
        status, payload = self.request(
            "GET",
            f"{self.proof_path('r2', 'o1')}?treeSize=2&root={'a' * 64}",
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "proof_conflict"})

    def test_malformed_queries_are_400(self) -> None:
        records = self.seed(1)
        root = reference_root(records)
        bad_queries = [
            f"?root={root}",
            "?treeSize=1",
            f"?treeSize=&root={root}",
            f"?treeSize=-1&root={root}",
            f"?treeSize=1.0&root={root}",
            f"?treeSize=x&root={root}",
            "?treeSize=01&root=" + "A" * 64,
            "?treeSize=1&root=" + "a" * 63,
            "?treeSize=1&root=" + "g" * 64,
            f"?treeSize=1&root={root.upper()}",
            f"?treeSize=1&treeSize=2&root={root}",
            f"?treeSize=1&root={root}&unknown=1",
            f"?treeSize=1&root={root}&x",
            "?x=1",
            "?=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query[:40]):
                status, payload = self.request(
                    "GET", self.proof_path("r2", "o1") + query
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_proof_route_shape_mismatches_are_404(self) -> None:
        root = "a" * 64
        query = f"?treeSize=1&root={root}"
        for path in [
            f"/v1/replicas/r1/operations/o1/proof/extra{query}",
            f"/v1/replicas/r1/operations/o1/proof/{query}",
            f"/v1/replicas//operations/o1/proof{query}",
            f"/v1/replicas/r1/proof{query}",
        ]:
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_proof_shape_check_precedes_query_check(self) -> None:
        status, payload = self.request(
            "GET", "/v1/replicas/r1/operations/o1/proof/extra?treeSize=x"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_url_encoded_identity_segments(self) -> None:
        # An identity with a reserved slash-encoded character round-trips.
        op = operation("o%2F1", "k", "v", {"r1": 1})
        self.post_operation("r1", op)
        _, root_payload, _ = self.raw_request("GET", "/v1/audit/proofs/root")
        status, payload = self.request(
            "GET",
            "/v1/replicas/r1/operations/o%252F1/proof"
            f"?treeSize=1&root={root_payload['root']}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["record"]["operation"]["operationId"], "o%2F1")


if __name__ == "__main__":
    unittest.main()
