"""Tests for the read-only replication-repair execution audit::

    GET /v1/replication/repairs/executions?after=N&limit=N

It pages the committed ``repairExecutions`` bindings — the same records
``POST /v1/replication/repairs/apply`` commits atomically with the
restored checkpoint and that replay by ``(peerId, ackId)`` appends
nothing — and adds a full-history summary and integrity conclusion.

The tests cover the query parser, the success body (compact ordered
UTF-8 JSON, one trailing newline, explicit Content-Length; executions,
nextCursor, hasMore, then algorithm/digest/executionsCount, then
verification), stable ordering (ascending ``peerId`` then creation
order), full-history digest/count/verification on every page, the stable
empty tail and the past-end 400, the independent verification scan
(duplicate bindings, action order, suggestion boundaries, checkpoint
advancement, binding legality, each marker carrying the 0-based
``executionIndex`` plus ``peerId`` and ``ackId``), idempotent replay not
appending, the HTTP precedence chain (404 path shape first, 401, 403,
400 query), ``--data-file`` recovery, and the strictly read-only
guarantee. Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
from http import HTTPStatus
import json
import os
import shutil
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    REPAIR_ACTIONS,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _repair_executions_digest_input,
    _repair_executions_verification_locked,
    load_scope_policy,
    parse_replication_repair_executions_query,
)

EXECUTIONS_PATH = "/v1/replication/repairs/executions"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
SCOPE_POLICY = {READ_TOKEN: ["read"], WRITE_TOKEN: ["write"]}

OK_VERIFICATION = {
    "status": "ok",
    "duplicateBindings": [],
    "outOfOrderActions": [],
    "invalidSuggestions": [],
    "checkpointRegressions": [],
    "invalidBindings": [],
}


def interval_suggestion(action: str, ack_id: str, start: int, end: int) -> dict:
    return {
        "action": action,
        "ackId": ack_id,
        "location": {"start": start, "end": end},
        "target": {"start": start, "end": end},
    }


def interval_result(action: str, start: int, end: int) -> dict:
    return {"action": action, "boundary": {"start": start, "end": end}}


def binding(
    expected_checkpoint: int = 0,
    cursor: int | None = None,
    *,
    digest: str = "a" * 64,
    suggestions: list[dict] | None = None,
    results: list[dict] | None = None,
) -> dict:
    if suggestions is None:
        suggestions = [interval_suggestion("resend", "ack-1", 0, 1)]
    if results is None:
        results = [interval_result("resend", 0, 1)]
    return {
        "expectedCheckpoint": expected_checkpoint,
        "expectedReceipts": digest,
        "suggestions": suggestions,
        "results": results,
        "cursor": expected_checkpoint if cursor is None else cursor,
    }


def record(peer_id: str, ack_id: str, body: dict) -> dict:
    return {
        "peerId": peer_id,
        "ackId": ack_id,
        "expectedCheckpoint": body["expectedCheckpoint"],
        "expectedReceipts": body["expectedReceipts"],
        "suggestions": body["suggestions"],
        "results": body["results"],
        "cursor": body["cursor"],
    }


class ParseRepairExecutionsQueryTests(unittest.TestCase):
    def test_requires_both_after_and_limit(self) -> None:
        for query in ("", "after=0", "limit=1"):
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_query(query)
                )

    def test_accepts_plain_ascii_decimal_pairs(self) -> None:
        self.assertEqual(
            parse_replication_repair_executions_query("after=0&limit=1"),
            (0, 1),
        )
        self.assertEqual(
            parse_replication_repair_executions_query("after=00&limit=0100"),
            (0, 100),
        )

    def test_rejects_bad_values(self) -> None:
        bad = [
            "after=&limit=1",
            "after=0&limit=",
            "after=-1&limit=1",
            "after=0&limit=-1",
            "after=%200&limit=1",
            "after=0&limit=1%20",
            "after=+0&limit=1",
            "after=1.0&limit=1",
            "after=0&limit=1.0",
            "after=%C2%B2&limit=1",
            "after=0&limit=0",
            "after=0&limit=101",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_query(query)
                )

    def test_rejects_unknown_and_repeated_parameters(self) -> None:
        bad = [
            "after=0&limit=1&x=1",
            "after=0&after=1&limit=1",
            "after=0&limit=1&limit=2",
            "x=0&limit=1",
            "after=0&x=1",
        ]
        for query in bad:
            with self.subTest(query=query):
                self.assertIsNone(
                    parse_replication_repair_executions_query(query)
                )


class RepairExecutionsVerificationTests(unittest.TestCase):
    def history(self, *entries: tuple[str, str, dict]):
        return [((peer, ack), body) for peer, ack, body in entries]

    def test_empty_history_is_intact(self) -> None:
        self.assertEqual(
            _repair_executions_verification_locked([]), OK_VERIFICATION
        )

    def test_well_formed_history_is_ok(self) -> None:
        history = self.history(
            ("peer-a", "e1", binding(0, 2)),
            ("peer-a", "e2", binding(2, 4)),
            ("peer-b", "e1", binding(0, 0)),
        )
        self.assertEqual(
            _repair_executions_verification_locked(history)["status"], "ok"
        )

    def test_marker_index_uses_exported_peer_then_creation_order(self) -> None:
        # Creation order: b/e1 (0), a/e1 (1), a/e0 (2). Exported order
        # groups by peer: a/e1 -> 0, a/e0 -> 1, b/e1 -> 2.
        history = self.history(
            ("peer-b", "e1", binding(0, 1)),
            ("peer-a", "e1", binding(0, 1)),
            ("peer-a", "e0", binding(0, 1, digest="not-hex")),
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        (marker,) = verification["invalidBindings"]
        self.assertEqual(
            marker,
            {
                "executionIndex": 1,
                "peerId": "peer-a",
                "ackId": "e0",
                "expected": None,
                "observed": "not-hex",
            },
        )

    def test_duplicate_binding_marks_the_later_occurrence(self) -> None:
        history = self.history(
            ("peer-a", "e1", binding(0, 1)),
            ("peer-a", "e1", binding(0, 1)),
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["duplicateBindings"],
            [{"executionIndex": 1, "peerId": "peer-a", "ackId": "e1"}],
        )

    def test_out_of_order_actions_are_marked(self) -> None:
        suggestions = [
            interval_suggestion("correct_cursor", "ack-2", 2, 2),
            interval_suggestion("resend", "ack-1", 0, 1),
        ]
        results = [
            interval_result("correct_cursor", 2, 2),
            interval_result("resend", 0, 1),
        ]
        history = self.history(
            ("peer-a", "e1", binding(0, 2, suggestions=suggestions, results=results))
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["outOfOrderActions"],
            [
                {
                    "executionIndex": 0,
                    "peerId": "peer-a",
                    "ackId": "e1",
                    "expected": REPAIR_ACTIONS.index("correct_cursor"),
                    "observed": REPAIR_ACTIONS.index("resend"),
                }
            ],
        )

    def test_in_order_actions_pass(self) -> None:
        suggestions = [
            interval_suggestion("resend", "ack-1", 0, 1),
            interval_suggestion("correct_cursor", "ack-2", 2, 2),
        ]
        results = [
            interval_result("resend", 0, 1),
            interval_result("correct_cursor", 2, 2),
        ]
        history = self.history(
            ("peer-a", "e1", binding(0, 2, suggestions=suggestions, results=results))
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "ok")

    def test_invalid_suggestion_is_marked(self) -> None:
        bad_suggestion = {"action": "resend", "ackId": "ack-1"}
        good_result = interval_result("resend", 0, 1)
        history = self.history(
            (
                "peer-a",
                "e1",
                binding(0, 1, suggestions=[bad_suggestion], results=[good_result]),
            )
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        (marker,) = verification["invalidSuggestions"]
        self.assertEqual(marker["executionIndex"], 0)
        self.assertEqual(marker["peerId"], "peer-a")
        self.assertEqual(marker["ackId"], "e1")
        self.assertIsNone(marker["expected"])
        self.assertEqual(marker["observed"], bad_suggestion)

    def test_result_action_mismatch_is_marked(self) -> None:
        history = self.history(
            (
                "peer-a",
                "e1",
                binding(
                    0,
                    1,
                    suggestions=[interval_suggestion("resend", "ack-1", 0, 1)],
                    results=[interval_result("deduplicate", 0, 1)],
                ),
            )
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(len(verification["invalidSuggestions"]), 1)

    def test_checkpoint_regression_within_a_peer_is_marked(self) -> None:
        history = self.history(
            ("peer-a", "e1", binding(0, 5)),
            ("peer-a", "e2", binding(3, 5)),
        )
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        self.assertEqual(
            verification["checkpointRegressions"],
            [
                {
                    "executionIndex": 1,
                    "peerId": "peer-a",
                    "ackId": "e2",
                    "expected": 5,
                    "observed": 3,
                }
            ],
        )

    def test_cursor_before_expected_checkpoint_is_an_invalid_binding(self) -> None:
        history = self.history(("peer-a", "e1", binding(4, 2)))
        verification = _repair_executions_verification_locked(history)
        self.assertEqual(verification["status"], "broken")
        (marker,) = verification["invalidBindings"]
        self.assertEqual(marker["expected"], 4)
        self.assertEqual(marker["observed"], 2)

    def test_other_peers_checkpoint_chains_are_independent(self) -> None:
        history = self.history(
            ("peer-a", "e1", binding(0, 5)),
            ("peer-b", "e1", binding(0, 1)),
        )
        self.assertEqual(
            _repair_executions_verification_locked(history)["status"], "ok"
        )

    def test_malformed_binding_shapes_are_marked(self) -> None:
        cases = [
            binding(0, 1, digest="A" * 64),
            binding(True, 1),  # type: ignore[arg-type]
            binding(0, -1),
        ]
        for body in cases:
            with self.subTest(body=body):
                history = self.history(("peer-a", "e1", body))
                verification = _repair_executions_verification_locked(history)
                self.assertEqual(verification["status"], "broken")
                self.assertTrue(verification["invalidBindings"])


class RepairExecutionsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def seed(self, *keys: tuple[str, str, dict]) -> None:
        for (peer_id, ack_id), body in keys:
            self.store._repairs[(peer_id, ack_id)] = {
                "expectedCheckpoint": body["expectedCheckpoint"],
                "expectedReceipts": body["expectedReceipts"],
                "suggestions": [dict(s) for s in body["suggestions"]],
                "results": [dict(r) for r in body["results"]],
                "cursor": body["cursor"],
            }

    def test_empty_history(self) -> None:
        status, payload = self.store.get_repair_executions(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            list(payload),
            [
                "executions",
                "nextCursor",
                "hasMore",
                "algorithm",
                "digest",
                "executionsCount",
                "verification",
            ],
        )
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["executionsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        self.assertEqual(payload["verification"], OK_VERIFICATION)

    def test_after_equal_to_count_is_a_stable_empty_page(self) -> None:
        self.seed((("peer-a", "e1"), binding(0, 1)))
        status, payload = self.store.get_repair_executions(1, 10)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 1)
        self.assertEqual(payload["verification"]["status"], "ok")

    def test_after_past_count_raises_value_error(self) -> None:
        self.seed((("peer-a", "e1"), binding(0, 1)))
        with self.assertRaises(ValueError):
            self.store.get_repair_executions(2, 10)

    def test_executions_order_by_peer_then_creation_and_preserve_fields(self) -> None:
        # Creation/insertion order: b/e1, a/e1, a/e0.
        self.seed(
            (("peer-b", "e1"), binding(0, 3)),
            (("peer-a", "e1"), binding(0, 2)),
            (("peer-a", "e0"), binding(2, 4)),
        )
        status, payload = self.store.get_repair_executions(0, 100)
        self.assertIs(status, HTTPStatus.OK)
        page = payload["executions"]
        self.assertEqual([(e["peerId"], e["ackId"]) for e in page], [
            ("peer-a", "e1"),
            ("peer-a", "e0"),
            ("peer-b", "e1"),
        ])
        first = page[0]
        self.assertEqual(
            list(first),
            [
                "peerId",
                "ackId",
                "expectedCheckpoint",
                "expectedReceipts",
                "suggestions",
                "results",
                "cursor",
            ],
        )
        self.assertEqual(first["expectedCheckpoint"], 0)
        self.assertEqual(first["cursor"], 2)
        self.assertEqual(
            list(first["suggestions"][0]),
            ["action", "ackId", "location", "target"],
        )

    def test_digest_follows_creation_order_and_count_covers_all(self) -> None:
        bodies = [
            (("peer-b", "e1"), binding(0, 3)),
            (("peer-a", "e1"), binding(0, 2)),
            (("peer-a", "e0"), binding(2, 4)),
        ]
        self.seed(*bodies)
        history = list(self.store._repairs.items())
        expected_digest = hashlib.sha256(
            _repair_executions_digest_input(history)
        ).hexdigest()
        pages = [
            self.store.get_repair_executions(0, 1)[1],
            self.store.get_repair_executions(1, 1)[1],
            self.store.get_repair_executions(2, 1)[1],
            self.store.get_repair_executions(3, 10)[1],
        ]
        for page in pages:
            self.assertEqual(page["digest"], expected_digest)
            self.assertEqual(page["executionsCount"], 3)
            self.assertEqual(page["verification"]["status"], "ok")
        self.assertEqual(
            [page["nextCursor"] for page in pages], [1, 2, 3, 3]
        )
        self.assertEqual(
            [page["hasMore"] for page in pages], [True, True, False, False]
        )

    def test_empty_history_digest_is_sha256_of_empty_array(self) -> None:
        _, payload = self.store.get_repair_executions(0, 1)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())

    def test_replay_appends_no_record(self) -> None:
        self.seed((("peer-a", "e1"), binding(0, 1)))
        # An identical replay of the same binding leaves the history at one.
        existing = self.store._repairs[("peer-a", "e1")]
        self.store._repairs[("peer-a", "e1")] = dict(existing)
        _, payload = self.store.get_repair_executions(0, 100)
        self.assertEqual(payload["executionsCount"], 1)
        self.assertEqual(len(payload["executions"]), 1)

    def test_query_is_read_only(self) -> None:
        self.seed((("peer-a", "e1"), binding(0, 1)))
        before = self.store.get_repair_executions(0, 100)[1]
        metrics_before = self.store.get_metrics()
        for after, limit in ((0, 1), (1, 1), (1, 10)):
            self.store.get_repair_executions(after, limit)
        after = self.store.get_repair_executions(0, 100)[1]
        self.assertEqual(after, before)
        self.assertEqual(self.store.get_metrics(), metrics_before)
        self.assertEqual(len(self.store._repairs), 1)


class RepairExecutionsRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sestate-repairs-exec-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def test_recovery_reproduces_page_digest_count_and_verification(self) -> None:
        data_file = os.path.join(self.tmpdir, "state.json")

        def build(store: StateStore) -> dict:
            for index in range(4):
                self.assertIs(
                    store.apply_operation(
                        f"r{index}",
                        {
                            "operationId": f"o{index}",
                            "key": "k",
                            "value": f"v{index}",
                            "clock": {f"r{index}": 1},
                        },
                    ),
                    HTTPStatus.CREATED,
                )
            store.save_checkpoint("peer-a", 4)
            store._acks[("peer-a", "ack-1")] = {
                "cursor": 1,
                "operations": [{"replicaId": "r0", "operationId": "o0"}],
            }
            store._acks[("peer-a", "ack-2")] = {
                "cursor": 3,
                "operations": [{"replicaId": "r2", "operationId": "o2"}],
            }
            repairs = store.get_replication_repairs(0, 100)[1]
            item = repairs["suggestions"][0]
            suggestion = {
                "action": item["action"],
                "ackId": item["ackId"],
                "location": item["location"],
                "target": item["target"],
            }
            committed = [
                (ack_id, receipt)
                for (peer, ack_id), receipt in store._acks.items()
                if peer == "peer-a"
            ]
            from semantic_state_engine.server import _repair_receipts_digest

            digest = _repair_receipts_digest("peer-a", committed)
            status, _payload, error = store.apply_replication_repairs(
                "peer-a", "exec-1", 4, digest, [suggestion]
            )
            self.assertIs(status, HTTPStatus.CREATED, error)
            return store.get_repair_executions(0, 100)[1]

        store = StateStore(data_file=data_file)
        before = build(store)
        recovered = StateStore(data_file=data_file)
        after = recovered.get_repair_executions(0, 100)[1]
        self.assertEqual(after["executions"], before["executions"])
        self.assertEqual(after["digest"], before["digest"])
        self.assertEqual(after["executionsCount"], before["executionsCount"])
        self.assertEqual(after["verification"], before["verification"])
        self.assertEqual(after["verification"]["status"], "ok")


class RepairExecutionsHttpFixture(unittest.TestCase):
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
        self.server.store = StateStore()

    def raw_request(self, path: str, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers=dict(headers or {}))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return response.status, payload, raw, response_headers

    def seed_execution(
        self,
        peer_id: str = "peer-a",
        ack_id: str = "exec-1",
        *,
        expected_checkpoint: int = 0,
        cursor: int = 1,
    ) -> None:
        self.server.store._repairs[(peer_id, ack_id)] = binding(
            expected_checkpoint, cursor
        )


class RepairExecutionsHttpTests(RepairExecutionsHttpFixture):
    def test_empty_history_is_compact_ordered_json_with_single_newline(self) -> None:
        status, payload, raw, headers = self.raw_request(
            EXECUTIONS_PATH + "?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            list(payload),
            [
                "executions",
                "nextCursor",
                "hasMore",
                "algorithm",
                "digest",
                "executionsCount",
                "verification",
            ],
        )
        digest = payload["digest"]
        self.assertEqual(
            raw,
            b'{"executions":[],"nextCursor":0,"hasMore":false,'
            b'"algorithm":"sha256","digest":"' + digest.encode()
            + b'","executionsCount":0,"verification":{"status":"ok",'
            b'"duplicateBindings":[],"outOfOrderActions":[],'
            b'"invalidSuggestions":[],"checkpointRegressions":[],'
            b'"invalidBindings":[]}}\n',
        )
        self.assertTrue(raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Length"], str(len(raw)))

    def test_execution_page_field_order_and_full_history_summary(self) -> None:
        self.seed_execution("peer-b", "e2", cursor=3)
        self.seed_execution("peer-a", "e1", cursor=2)
        status, payload, raw, _ = self.raw_request(
            EXECUTIONS_PATH + "?after=0&limit=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["executions"]), 1)
        item = payload["executions"][0]
        self.assertEqual(item["peerId"], "peer-a")
        self.assertEqual(item["ackId"], "e1")
        self.assertEqual(
            list(item),
            [
                "peerId",
                "ackId",
                "expectedCheckpoint",
                "expectedReceipts",
                "suggestions",
                "results",
                "cursor",
            ],
        )
        self.assertTrue(payload["hasMore"])
        # The summary covers the complete history on a partial page.
        self.assertEqual(payload["executionsCount"], 2)
        self.assertEqual(payload["verification"]["status"], "ok")
        self.assertTrue(raw.endswith(b"\n"))

    def test_stable_empty_tail(self) -> None:
        self.seed_execution()
        first = self.raw_request(EXECUTIONS_PATH + "?after=1&limit=10")
        second = self.raw_request(EXECUTIONS_PATH + "?after=1&limit=10")
        self.assertEqual(first[:3], second[:3])
        status, payload, raw, _ = first
        self.assertEqual(status, 200)
        self.assertEqual(payload["executions"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["executionsCount"], 1)
        self.assertTrue(raw.endswith(b"\n"))

    def test_invalid_queries_are_400(self) -> None:
        bad_queries = [
            "",
            "?after=0",
            "?limit=1",
            "?after=&limit=1",
            "?after=0&limit=",
            "?after=-1&limit=1",
            "?after=0&limit=-1",
            "?after=%20&limit=1",
            "?after=0&limit=1%20",
            "?after=1.0&limit=1",
            "?after=0&limit=0",
            "?after=0&limit=101",
            "?after=0&limit=1&x=1",
            "?after=0&after=1&limit=1",
            "?after=0&limit=1&limit=2",
            "?x=0&limit=1",
            "?after=%C2%B9&limit=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload, raw, _ = self.raw_request(EXECUTIONS_PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
                self.assertTrue(raw.endswith(b"\n"))

    def test_after_past_count_is_400(self) -> None:
        self.seed_execution()
        status, payload, raw, _ = self.raw_request(
            EXECUTIONS_PATH + "?after=2&limit=1"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        self.assertTrue(raw.endswith(b"\n"))

    def test_invalid_query_changes_nothing(self) -> None:
        self.seed_execution()
        for query in ("?after=nope&limit=1", "?after=99&limit=1", "?x=1"):
            self.raw_request(EXECUTIONS_PATH + query)
        status, payload, _, _ = self.raw_request(
            EXECUTIONS_PATH + "?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 1)

    def test_path_shape_mismatches_are_404_even_with_bad_query(self) -> None:
        for path in (
            EXECUTIONS_PATH + "/",
            EXECUTIONS_PATH + "/extra",
            "/v1/replication/repairs/executionsx",
            "/v1/replication/repairs/other",
            "/v1/replication/repair/executions",
            "/v1/replication/executions",
        ):
            with self.subTest(path=path):
                status, payload, _, _ = self.raw_request(path + "?bogus=1")
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_non_get_method_is_404(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", EXECUTIONS_PATH + "?after=0&limit=1", body=b"{}")
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        self.assertEqual(response.status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_advice_route_keeps_its_own_contract(self) -> None:
        status, payload, _, _ = self.raw_request(
            "/v1/replication/repairs?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertIn("suggestions", payload)
        self.assertNotIn("executions", payload)


class RepairExecutionsAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-repairs-exec-auth-")
        cls.single_server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret-token"
        )
        cls.single_thread = threading.Thread(
            target=cls.single_server.serve_forever, daemon=True
        )
        cls.single_thread.start()
        cls.single_port = cls.single_server.server_address[1]
        cls.policy_path = os.path.join(cls.tmpdir, "scopes.json")
        with open(cls.policy_path, "w", encoding="utf-8") as handle:
            json.dump(SCOPE_POLICY, handle)
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

    def get(self, port, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", EXECUTIONS_PATH + "?after=0&limit=1", headers=dict(headers or {}))
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, json.loads(raw), response_headers

    def test_single_token_required(self) -> None:
        status, payload, headers = self.get(self.single_port)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        status, payload, _ = self.get(
            self.single_port, [("Authorization", "Bearer s3cret-token")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executions"], [])

    def test_read_scope_suffices_write_scope_does_not(self) -> None:
        status, payload, headers = self.get(
            self.scope_port, [("Authorization", f"Bearer {WRITE_TOKEN}")]
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        status, payload, _ = self.get(
            self.scope_port, [("Authorization", f"Bearer {READ_TOKEN}")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["executionsCount"], 0)

    def test_path_shape_precedes_authentication_for_unknown_routes(self) -> None:
        # Auth runs before route matching on GETs; a missing credential is
        # 401 even on an unknown path, while a good read token on a wrong
        # shape gets 404.
        conn = http.client.HTTPConnection("127.0.0.1", self.scope_port, timeout=5)
        conn.request("GET", EXECUTIONS_PATH + "/extra?after=0&limit=1")
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        conn.close()
        status, payload, _ = self.get(
            self.scope_port,
            [
                ("Authorization", f"Bearer {READ_TOKEN}"),
            ],
        )
        # The exact path is fine (200); check the wrong shape separately.
        self.assertEqual(status, 200)
        conn = http.client.HTTPConnection("127.0.0.1", self.scope_port, timeout=5)
        conn.request(
            "GET",
            EXECUTIONS_PATH + "/extra?after=0&limit=1",
            headers={"Authorization": f"Bearer {READ_TOKEN}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        self.assertEqual(response.status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
