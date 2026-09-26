"""HTTP tests for the read-only batched automatic-resolution preview.

The endpoint is::

    POST /v1/resolve/auto/plan

It accepts the exact same ``{"resolutions":[...]}`` body as the committing
batch route (1-100 entries, distinct keys and identities) and reports, for
each entry in request order, the value and policy a commit *would* select
against one complete committed snapshot — without changing any business
state. Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    load_data_file,
)

PLAN_PATH = "/v1/resolve/auto/plan"
BATCH_PATH = "/v1/resolve/auto/batch"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def plan_entry(
    key: str,
    replica: str = "r3",
    operation_id: str = "fix-1",
    clock: dict | None = None,
    policy: str = "lowest_identity",
) -> dict:
    return {
        "key": key,
        "replicaId": replica,
        "operationId": operation_id,
        "clock": clock if clock is not None else {"r1": 1, "r2": 1, "r3": 1},
        "policy": policy,
    }


def plan_document(*entries: dict) -> dict:
    return {"resolutions": list(entries)}


class HttpServerTestCase(unittest.TestCase):
    """Spin up one in-memory server per class; reset the store per test."""

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

    def request_bytes(
        self, method: str, path: str, body: bytes | None = None, headers: dict | None = None
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(method, path, body=body, headers=merged)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        if body is None:
            status, raw = self.request_bytes(method, path)
        elif isinstance(body, (bytes, str)):
            data = body.encode("utf-8") if isinstance(body, str) else body
            status, raw = self.request_bytes(method, path, data)
        else:
            status, raw = self.request_bytes(
                method, path, json.dumps(body).encode("utf-8")
            )
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_plan(self, body: object) -> tuple[int, object]:
        return self.request("POST", PLAN_PATH, body)

    def post_batch(self, body: object) -> tuple[int, object]:
        return self.request("POST", BATCH_PATH, body)

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_auto(self, key: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/states/{key}/resolve/auto", body)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self) -> tuple[int, object]:
        return self.request("GET", "/v1/sync/operations")

    def get_audit(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_conflict(self, key: str = "k", low: str = "v1", high: str = "v2") -> None:
        """Two concurrent writes with distinct global identities on ``key``."""
        salt = getattr(self, "_seed_salt", 0)
        self._seed_salt = salt + 1
        id1, id2 = f"o1-{key}-{salt}", f"o2-{key}-{salt}"
        self.assertEqual(
            self.post_operation("r1", operation(id1, key, low, {"r1": 1}))[0], 201
        )
        self.assertEqual(
            self.post_operation("r2", operation(id2, key, high, {"r2": 1}))[0], 201
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class PlanHappyPathTests(HttpServerTestCase):
    def test_preview_reports_selected_values_policies_and_counts(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2", low="aaa", high="zzz")
        doc = plan_document(
            plan_entry("k1", "r3", "f1"),
            plan_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "status": "planned",
                "resolutions": [
                    {
                        "key": "k1",
                        "replicaId": "r3",
                        "operationId": "f1",
                        "value": "v1",
                        "policy": "lowest_identity",
                    },
                    {
                        "key": "k2",
                        "replicaId": "r3",
                        "operationId": "f2",
                        "value": "zzz",
                        "policy": "highest_identity",
                    },
                ],
                "accepted": 2,
                "replayed": 0,
            },
        )
        self.assertIsInstance(payload["accepted"], int)
        self.assertIsInstance(payload["replayed"], int)

    def test_status_is_planned_even_when_everything_would_commit(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_plan(plan_document(plan_entry("k")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")

    def test_results_follow_request_order(self) -> None:
        self.seed_conflict("k1", low="a", high="b")
        self.seed_conflict("k2", low="c", high="d")
        doc = plan_document(
            plan_entry("k2", "r9", "zzz-fix", {"r1": 1, "r2": 1, "r9": 1}),
            plan_entry("k1", "r3", "aaa-fix", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual([r["key"] for r in payload["resolutions"]], ["k2", "k1"])

    def test_response_is_compact_json_ending_with_one_newline(self) -> None:
        self.seed_conflict("k")
        body = json.dumps(plan_document(plan_entry("k"))).encode("utf-8")
        status, raw = self.request_bytes("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["status"], "planned")
        # Conflict responses for the route carry the same terminator.
        status, raw = self.request_bytes("POST", PLAN_PATH, b"{oops")
        self.assertEqual(status, 400)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"error": "invalid_request"})

    def test_single_entry_preview(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_plan(plan_document(plan_entry("k")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)

    def test_one_hundred_entries_are_previewed(self) -> None:
        doc_entries = []
        for i in range(100):
            key = f"k{i:03d}"
            pa, pb = f"p{i:03d}a", f"p{i:03d}b"
            self.assertEqual(
                self.post_operation(pa, operation("o1", key, "lo", {pa: 1}))[0], 201
            )
            self.assertEqual(
                self.post_operation(pb, operation("o2", key, "hi", {pb: 1}))[0], 201
            )
            doc_entries.append(
                plan_entry(
                    key,
                    f"r{i:03d}",
                    f"f{i:03d}",
                    {pa: 1, pb: 1, f"r{i:03d}": 1},
                    "highest_identity" if i % 2 else "lowest_identity",
                )
            )
        status, payload = self.post_plan(plan_document(*doc_entries))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 100)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(len(payload["resolutions"]), 100)
        expected_values = ["hi" if i % 2 else "lo" for i in range(100)]
        self.assertEqual([r["value"] for r in payload["resolutions"]], expected_values)

    def test_repeated_previews_are_identical_and_keep_conflict(self) -> None:
        self.seed_conflict("k")
        doc = plan_document(plan_entry("k"))
        status, first = self.post_plan(doc)
        self.assertEqual(status, 200)
        for _ in range(3):
            status, payload = self.post_plan(json.loads(json.dumps(doc)))
            self.assertEqual(status, 200)
            self.assertEqual(payload, first)
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), 2)

    def test_preview_matches_what_the_commit_then_does(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2", low="aaa", high="zzz")
        doc = plan_document(
            plan_entry("k1", "r3", "f1"),
            plan_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, planned = self.post_plan(doc)
        self.assertEqual(status, 200)
        status, committed = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(committed["resolutions"], planned["resolutions"])
        self.assertEqual(committed["accepted"], planned["accepted"])
        self.assertEqual(committed["replayed"], planned["replayed"])


class PlanReadOnlyTests(HttpServerTestCase):
    def test_preview_appends_no_sync_records(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        _, before = self.get_sync()
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        self.assertEqual(self.post_plan(doc)[0], 200)
        _, after = self.get_sync()
        self.assertEqual(after, before)
        self.assertEqual(len(after["operations"]), 4)

    def test_preview_moves_no_metrics(self) -> None:
        self.seed_conflict("k")
        _, before = self.get_metrics()
        self.assertEqual(
            self.post_plan(plan_document(plan_entry("k")))[0], 200
        )
        _, after = self.get_metrics()
        self.assertEqual(after, before)
        self.assertEqual(after["conflictKeys"], 1)
        self.assertEqual(after["resolvedKeys"], 0)

    def test_preview_appends_no_audit_records(self) -> None:
        self.seed_conflict("k")
        _, before = self.get_audit("k")
        self.assertEqual(
            self.post_plan(plan_document(plan_entry("k")))[0], 200
        )
        _, after = self.get_audit("k")
        self.assertEqual(after, before)
        self.assertEqual(
            [e["operation"]["operationId"] for e in after["operations"]],
            [e["operation"]["operationId"] for e in before["operations"]],
        )

    def test_rejected_preview_changes_nothing_either(self) -> None:
        self.seed_conflict("k1")
        _, sync_before = self.get_sync()
        _, metrics_before = self.get_metrics()
        # Missing key: a 409 preview must leave everything exactly as it was.
        status, payload = self.post_plan(
            plan_document(plan_entry("missing", "r3", "f9", {"r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, sync_after = self.get_sync()
        _, metrics_after = self.get_metrics()
        self.assertEqual(sync_after, sync_before)
        self.assertEqual(metrics_after, metrics_before)
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_preview_creates_no_policy_bindings(self) -> None:
        self.seed_conflict("k")
        doc = plan_document(plan_entry("k"))
        self.assertEqual(self.post_plan(doc)[0], 200)
        # A later commit with the *same* identity is a fresh 201, not a
        # replay: the preview never bound (r3, f1) to anything.
        status, payload = self.post_batch(json.loads(json.dumps(doc)))
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)

    def test_preview_identity_does_not_block_a_different_binding(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        # Preview f1 against k1 ...
        status, _ = self.post_plan(
            plan_document(plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 200)
        # ... then a real commit that binds f1 to k2 must not conflict.
        status, payload = self.post_batch(
            plan_document(plan_entry("k2", "r3", "f1", {"r1": 1, "r2": 1, "r3": 9}))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)


class PlanReplayTests(HttpServerTestCase):
    def test_already_bound_identities_are_counted_replayed(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        self.assertEqual(self.post_batch(json.loads(json.dumps(doc)))[0], 201)
        status, payload = self.post_plan(json.loads(json.dumps(doc)))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual(
            [r["value"] for r in payload["resolutions"]], ["v1", "v1"]
        )
        # The preview of replays appends nothing.
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 6)

    def test_mixed_new_and_replayed_entries(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        self.assertEqual(
            self.post_auto(
                "k2",
                {"replicaId": "r3", "operationId": "f2",
                 "clock": {"r1": 1, "r2": 1, "r3": 2},
                 "policy": "lowest_identity"},
            )[0],
            201,
        )
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(
            [(r["key"], r["value"]) for r in payload["resolutions"]],
            [("k1", "v1"), ("k2", "v1")],
        )
        # Nothing committed: k1 is still conflicted.
        _, state = self.get_state("k1")
        self.assertEqual(state["status"], "conflict")

    def test_replay_reports_original_value_after_key_moved_on(self) -> None:
        self.seed_conflict("k")
        doc = plan_document(plan_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        self.assertEqual(self.post_batch(json.loads(json.dumps(doc)))[0], 201)
        # A later concurrent write reopens the conflict on k.
        self.assertEqual(
            self.post_operation(
                "r2", operation("o3", "k", "v3", {"r1": 1, "r2": 2, "r3": 0})
            )[0],
            201,
        )
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 1)
        self.assertEqual(payload["resolutions"][0]["value"], "v1")

    def test_different_policy_under_known_identity_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        good = {"replicaId": "r3", "operationId": "f1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "lowest_identity"}
        self.assertEqual(self.post_auto("k", good)[0], 201)
        status, payload = self.post_plan(
            plan_document(plan_entry("k", "r3", "f1", good["clock"], "highest_identity"))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_identity_bound_to_plain_write_is_operation_conflict(self) -> None:
        self.seed_conflict("k")
        self.assertEqual(
            self.post_operation("r3", operation("f1", "other", "v", {"r3": 1}))[0],
            201,
        )
        status, payload = self.post_plan(
            plan_document(plan_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 2}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_conflict_kind_follows_first_failing_entry(self) -> None:
        self.seed_conflict("k1")
        self.assertEqual(
            self.post_operation("r3", operation("bound", "x", "v", {"r3": 1}))[0],
            201,
        )
        doc = plan_document(
            plan_entry("missing-a", "r3", "fa", {"r3": 2}),
            plan_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

        doc = plan_document(
            plan_entry("k1", "r3", "bound", {"r1": 1, "r2": 1, "r3": 3}),
            plan_entry("missing-a", "r3", "fa", {"r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class PlanConflictTests(HttpServerTestCase):
    def assert_unchanged_conflict(self, key: str = "k", candidates: int = 2) -> None:
        _, state = self.get_state(key)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(len(state["candidates"]), candidates)

    def test_missing_key_is_409(self) -> None:
        status, payload = self.post_plan(
            plan_document(plan_entry("absent", "r3", "f", {"r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_unconflicted_key_is_409(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("o9", "k2", "solo", {"r1": 9}))[0],
            201,
        )
        self.seed_conflict("k1")
        status, payload = self.post_plan(
            plan_document(
                plan_entry("k2", "r3", "f2", {"r1": 10, "r3": 2}),
                plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            )
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_unchanged_conflict("k1")

    def test_same_value_candidates_are_409(self) -> None:
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "same", {"r1": 1}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k", "same", {"r2": 1}))[0],
            201,
        )
        status, payload = self.post_plan(
            plan_document(plan_entry("k", "r3", "f", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})

    def test_clock_not_dominating_is_409(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            # Concurrent with k2's r2 candidate.
            plan_entry("k2", "r3", "f2", {"r1": 1, "r3": 2}),
        )
        status, payload = self.post_plan(doc)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_unchanged_conflict("k1")
        self.assert_unchanged_conflict("k2")

    def test_candidate_set_grew_since_caller_expectation_is_409(self) -> None:
        self.seed_conflict("k")
        # A third concurrent candidate: the caller's two-candidate clock no
        # longer dominates every current candidate.
        self.assertEqual(
            self.post_operation("r4", operation("o4", "k", "v4", {"r4": 1}))[0],
            201,
        )
        status, payload = self.post_plan(
            plan_document(plan_entry("k", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assert_unchanged_conflict(candidates=3)

    def test_conflict_response_is_compact_json_with_newline(self) -> None:
        status, raw = self.request_bytes(
            "POST",
            PLAN_PATH,
            json.dumps(plan_document(plan_entry("absent"))).encode("utf-8"),
        )
        self.assertEqual(status, 409)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"error": "resolution_conflict"})


class PlanValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict("k")
        valid = plan_document(plan_entry("k"))
        bad_bodies = [
            b"{oops",
            [],
            {},
            {"operations": [plan_entry("k")]},
            dict(valid, extra=1),
            plan_document(),
            plan_document(
                *[plan_entry(f"k{i}", "r3", f"f{i}", {"r3": 1}) for i in range(101)]
            ),
            plan_document(dict(plan_entry("k"), extra=1)),
            plan_document({k: v for k, v in plan_entry("k").items() if k != "key"}),
            plan_document(dict(plan_entry("k"), key="")),
            plan_document(dict(plan_entry("k"), replicaId="")),
            plan_document(dict(plan_entry("k"), operationId="")),
            plan_document(dict(plan_entry("k"), clock={"r2": 2})),
            plan_document(dict(plan_entry("k"), clock={"r3": -1})),
            plan_document(dict(plan_entry("k"), policy="middle")),
        ]
        for body in bad_bodies:
            status, payload = self.post_plan(body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_float_clocks_are_400(self) -> None:
        self.seed_conflict("k")
        for tick in (1.5, 1.0, -0.0):
            with self.subTest(tick=tick):
                status, payload = self.post_plan(
                    plan_document(plan_entry("k", clock={"r1": 1, "r2": 1, "r3": tick}))
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        for literal in ("NaN", "Infinity", "-Infinity", "-0.0", "1e3"):
            with self.subTest(literal=literal):
                raw = (
                    b'{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f1",'
                    b'"clock":{"r1":1,"r2":1,"r3":' + literal.encode("ascii")
                    + b'},"policy":"lowest_identity"}]}'
                )
                status, payload = self.post_plan(raw)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_duplicate_keys_and_identities_are_400(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        duplicate_keys = plan_document(
            plan_entry("k1", "r3", "f1"),
            plan_entry("k1", "r4", "f2", {"r1": 1, "r2": 1, "r4": 1}),
        )
        status, payload = self.post_plan(duplicate_keys)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})
        duplicate_identities = plan_document(
            plan_entry("k1", "r3", "same", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry("k2", "r3", "same", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_plan(duplicate_identities)
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class PlanRoutingTests(HttpServerTestCase):
    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        body = plan_document(plan_entry("k"))
        for path in (
            "/v1/resolve/auto/plan/extra",
            "/v1/resolve/auto/plan/extra/two",
            "/v1/resolve/auto/plan/",
            "/v1/resolve/auto",
            "/v1/resolve",
            "/v1/resolve/auto/plans",
            "/v1/resolve/auto/",
            "/v1/nope",
        ):
            status, payload = self.request("POST", path, body)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_404_takes_priority_over_body_and_query(self) -> None:
        # Even a malformed body and an illegal query on the wrong shape are
        # 404, not 400.
        for path in (
            "/v1/resolve/auto/plan/extra?x=1",
            "/v1/resolve/auto/plan/?x=1",
        ):
            status, payload = self.request("POST", path, b"{not json")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_plan_path_is_404(self) -> None:
        status, payload = self.request("GET", PLAN_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class PlanQueryTests(HttpServerTestCase):
    def test_no_query_parameters_are_accepted(self) -> None:
        self.seed_conflict("k")
        # A bare '?' carries an empty query string and is fine.
        status, payload = self.request(
            "POST", PLAN_PATH + "?", plan_document(plan_entry("k"))
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")

    def test_any_query_parameter_is_400(self) -> None:
        self.seed_conflict("k")
        body = plan_document(plan_entry("k"))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?=1", "?x&y=", "?x=%20"):
            with self.subTest(query=query):
                status, payload = self.request("POST", PLAN_PATH + query, body)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)
        # State is untouched and the same request without a query succeeds.
        status, payload = self.request("POST", PLAN_PATH, body)
        self.assertEqual(status, 200)

    def test_bad_query_is_rejected_before_the_body_is_validated(self) -> None:
        # An illegal query is 400 even when the body would also be invalid.
        status, payload = self.request(
            "POST", PLAN_PATH + "?x=1", b"{not json"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class PlanConcurrencyTests(HttpServerTestCase):
    def test_concurrent_commits_only_ever_show_a_complete_snapshot(self) -> None:
        self.seed_conflict("k1")
        self.seed_conflict("k2")
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry("k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}),
        )
        plans: list[tuple[int, object]] = []
        plans_lock = threading.Lock()
        stop = threading.Event()

        def planner() -> None:
            while not stop.is_set():
                status, payload = self.post_plan(json.loads(json.dumps(doc)))
                with plans_lock:
                    plans.append((status, payload))

        threads = [threading.Thread(target=planner) for _ in range(4)]
        for thread in threads:
            thread.start()
        # Let the planners observe the old snapshot, then commit once.
        time.sleep(0.1)
        commit_status, committed = self.post_batch(json.loads(json.dumps(doc)))
        time.sleep(0.1)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(commit_status, 201)
        self.assertEqual(committed["accepted"], 2)
        self.assertGreater(len(plans), 0)
        for status, payload in plans:
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "planned")
            self.assertEqual(len(payload["resolutions"]), 2)
            self.assertEqual(
                [r["value"] for r in payload["resolutions"]], ["v1", "v1"]
            )
            # Atomic snapshot: a plan either sees both entries as new
            # (accepted 2) or both as replays (replayed 2), never a mix.
            self.assertEqual(
                (payload["accepted"], payload["replayed"]),
                (2, 0) if payload["accepted"] == 2 else (0, 2),
            )
        kinds = {(p["accepted"], p["replayed"]) for _, p in plans}
        self.assertIn((2, 0), kinds)
        self.assertIn((0, 2), kinds)
        # Exactly one commit happened.
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 6)
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["resolvedKeys"], 2)
        self.assertEqual(metrics["conflictKeys"], 0)


class PlanRequestLimitTests(unittest.TestCase):
    """The plan route keeps the shared Content-Length contract."""

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

    def post_raw(self, headers: list, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", PLAN_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body if body else None)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_length_is_400(self) -> None:
        for value in ("", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            status, payload = self.post_raw([("Content-Length", value)], b"{}")
            self.assertEqual(status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_over_limit_declaration_is_413(self) -> None:
        status, payload = self.post_raw([("Content-Length", str(MAX_BODY_BYTES + 1))])
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_declared_length_exactly_at_the_limit_is_processed(self) -> None:
        # Seed a conflict on a padded key so a valid document reaches
        # MAX_BODY_BYTES exactly and is processed as a normal preview.
        template = {
            "key": "",
            "replicaId": "r3",
            "operationId": "f1",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "policy": "lowest_identity",
        }
        base = json.dumps(
            {"resolutions": [template]}, separators=(",", ":")
        ).encode("utf-8")
        key_length = MAX_BODY_BYTES - len(base)
        self.assertGreater(key_length, 0)
        key = "k" + "x" * (key_length - 1)
        doc = json.dumps(
            plan_document(plan_entry(key, "r3", "f1", {"r1": 1, "r2": 1, "r3": 1})),
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(len(doc), MAX_BODY_BYTES)
        for replica, op_id, clock in (
            ("r1", "o1", {"r1": 1}),
            ("r2", "o2", {"r2": 1}),
        ):
            body = json.dumps(
                operation(op_id, key, f"v{op_id[-1]}", clock), separators=(",", ":")
            ).encode()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            conn.request(
                "POST",
                f"/v1/replicas/{replica}/operations",
                body=body,
                headers={"Content-Type": "application/json",
                         "Content-Length": str(len(body))},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201, response.read())
            conn.close()
        status, payload = self.post_raw([("Content-Length", str(len(doc)))], doc)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["resolutions"][0]["key"], key)
        self.assertEqual(payload["resolutions"][0]["value"], "v1")
        # The at-limit preview still wrote nothing.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/sync/operations")
        response = conn.getresponse()
        page = json.loads(response.read())
        conn.close()
        self.assertEqual(len(page["operations"]), 2)


class PlanAuthTests(unittest.TestCase):
    def test_unauthorized_plan_is_401_without_reading_the_body(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            PLAN_PATH,
            body=json.dumps(plan_document(plan_entry("k"))),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(response.read()), {"error": "unauthorized"})
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        conn.close()

    def test_malformed_and_duplicate_bearer_headers_are_401(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]
        for headers in (
            [("Authorization", "Bearer")],
            [("Authorization", "Basic c2VjcmV0LXRva2Vu")],
            [("Authorization", "bearer secret-token")],
            [("Authorization", "Bearer wrong-token")],
            [("Authorization", "Bearer secret-token ")],
            [("Authorization", "Bearer secret-token"),
             ("Authorization", "Bearer secret-token")],
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.putrequest("POST", PLAN_PATH)
            conn.putheader("Content-Length", "2")
            for name, value in headers:
                conn.putheader(name, value)
            conn.endheaders(b"{}")
            response = conn.getresponse()
            self.assertEqual(response.status, 401, headers)
            self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
            response.read()
            conn.close()

    def test_authorized_preview_succeeds_and_health_stays_anonymous(self) -> None:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        def call(method: str, path: str, payload: object = None,
                 auth: str | None = "Bearer secret-token"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            headers = {"Content-Type": "application/json"}
            if auth is not None:
                headers["Authorization"] = auth
            if payload is None:
                conn.request(method, path, headers=headers)
            else:
                conn.request(method, path, body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            conn.close()
            return response.status, data

        status, payload = call("GET", "/health", auth=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        for replica, op_id, clock in (
            ("r1", "o1", {"r1": 1}),
            ("r2", "o2", {"r2": 1}),
        ):
            status, _ = call(
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", f"v{op_id[-1]}", clock),
            )
            self.assertEqual(status, 201)
        status, payload = call("POST", PLAN_PATH, plan_document(plan_entry("k")))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["resolutions"][0]["value"], "v1")


class PlanScopePolicyTests(unittest.TestCase):
    """In scope mode the preview needs read (or admin), not write."""

    READ_TOKEN = "reader-token"
    WRITE_TOKEN = "writer-token"
    ADMIN_TOKEN = "admin-token"
    POLICY = {
        READ_TOKEN: frozenset({"read"}),
        WRITE_TOKEN: frozenset({"write"}),
        ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
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

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def call(self, token: str | None, body: object = None) -> tuple[int, dict, dict]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request("POST", PLAN_PATH, headers=headers)
        else:
            conn.request("POST", PLAN_PATH, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def seed_conflict(self) -> None:
        conn_headers = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {self.WRITE_TOKEN}"}
        for replica, op_id, clock in (
            ("r1", "o1", {"r1": 1}),
            ("r2", "o2", {"r2": 1}),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                f"/v1/replicas/{replica}/operations",
                body=json.dumps(operation(op_id, "k", f"v{op_id[-1]}", clock)),
                headers=conn_headers,
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 201)
            response.read()
            conn.close()

    def test_read_and_admin_scopes_preview(self) -> None:
        self.seed_conflict()
        doc = plan_document(plan_entry("k"))
        for token in (self.READ_TOKEN, self.ADMIN_TOKEN):
            with self.subTest(token=token):
                status, payload, _ = self.call(token, json.loads(json.dumps(doc)))
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "planned")
                self.assertEqual(payload["resolutions"][0]["value"], "v1")

    def test_write_scope_is_forbidden_without_challenge(self) -> None:
        status, payload, headers = self.call(
            self.WRITE_TOKEN, plan_document(plan_entry("k"))
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        self.assertNotIn("Www-Authenticate", headers)

    def test_missing_or_bad_token_is_401_with_challenge(self) -> None:
        for token in (None, "unknown-token"):
            with self.subTest(token=token):
                status, payload, headers = self.call(
                    token, plan_document(plan_entry("k"))
                )
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        # A write-only token gets 403 even with an illegal query and body.
        headers = [
            ("Content-Length", "2"),
            ("Authorization", f"Bearer {self.WRITE_TOKEN}"),
        ]
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", PLAN_PATH + "?x=1")
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(b"{}")
        response = conn.getresponse()
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(response.read()), {"error": "forbidden"})
        self.assertIsNone(response.getheader("WWW-Authenticate"))
        conn.close()


class PersistentPlanTestCase(unittest.TestCase):
    """The preview against a --data-file-backed server."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def start_server(self) -> SemanticStateServer:
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=str(self.data_file)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def request(self, server: SemanticStateServer, method: str, path: str,
                body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            conn.request(
                method, path, body=data,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_conflicts(self, server: SemanticStateServer, *keys: str) -> None:
        for key in keys:
            for replica, op_id in (("r1", f"o1-{key}"), ("r2", f"o2-{key}")):
                status, _ = self.request(
                    server,
                    "POST",
                    f"/v1/replicas/{replica}/operations",
                    operation(op_id, key, f"v-{key}-{replica}", {replica: 1}),
                )
                self.assertEqual(status, 201)

    def test_preview_never_touches_the_data_file_or_creates_temps(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        # Settle the directory, then snapshot bytes and mtime.
        before = self.data_file.read_bytes()
        before_mtime = self.data_file.stat().st_mtime_ns
        doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        for _ in range(3):
            status, payload = self.request(server, "POST", PLAN_PATH, doc)
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "planned")
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(self.data_file.stat().st_mtime_ns, before_mtime)
        # No temporary files were created in the data directory.
        self.assertEqual(list(self.tmp.iterdir()), [self.data_file])
        # Only the four seed writes are durable.
        self.assertEqual(len(load_data_file(str(self.data_file))), 4)

    def test_same_state_gives_same_preview_across_restart(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k1", "k2")
        # Commit one repair for real; the other key stays conflicted.
        committed_doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, _ = self.request(server, "POST", BATCH_PATH, committed_doc)
        self.assertEqual(status, 201)
        # A mixed preview: f1 replays, f2 would be created.
        preview_doc = plan_document(
            plan_entry("k1", "r3", "f1", {"r1": 1, "r2": 1, "r3": 1}),
            plan_entry(
                "k2", "r3", "f2", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"
            ),
        )
        status, before_restart = self.request(server, "POST", PLAN_PATH, preview_doc)
        self.assertEqual(status, 200)
        self.assertEqual(
            (before_restart["accepted"], before_restart["replayed"]), (1, 1)
        )

        server.shutdown()
        server.server_close()
        server = self.start_server()
        status, after_restart = self.request(server, "POST", PLAN_PATH, preview_doc)
        self.assertEqual(status, 200)
        self.assertEqual(after_restart, before_restart)
        # The previews on either side of the restart appended nothing.
        self.assertEqual(len(load_data_file(str(self.data_file))), 5)

    def test_rejected_preview_never_touches_the_data_file(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server, "k")
        before = self.data_file.read_bytes()
        status, payload = self.request(
            server, "POST", PLAN_PATH,
            plan_document(plan_entry("missing", "r3", "f9", {"r3": 1})),
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        self.assertEqual(self.data_file.read_bytes(), before)
        self.assertEqual(list(self.tmp.iterdir()), [self.data_file])


if __name__ == "__main__":
    unittest.main()
