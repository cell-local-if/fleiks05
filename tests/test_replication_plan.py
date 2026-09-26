"""Tests for the cross-replica sync-planning endpoint.

The endpoint is::

    POST /v1/replication/plan

with the same body as the comparison endpoint — exactly ``{"replicaId":
..., "snapshot": {...}}`` naming a remote replica and its complete
candidate snapshot under the live write constraints. The endpoint turns
the read-only diff into an executable follow-up plan against one
committed local snapshot — strictly read-only: the remote content is
never imported, and no repair, transaction, sync, or persistence runs.
Each candidate identity that needs convergence is marked ``send_local``
(local-only or locally dominating), ``fetch_remote`` (remote-only or
remotely dominating), or ``semantic_resolution`` (a content conflict or
concurrent clocks — neither version is overwritten). Candidates both
sides hold with the same value and the same clock generate no action.
The success body is compact canonical UTF-8 JSON terminated by one
newline, with exactly ``status``, ``replicaId``, ``keys``, and
``summary``.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for plan semantics. Only the
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
    load_scope_policy,
)

PLAN_PATH = "/v1/replication/plan"
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


class PlanStoreTests(unittest.TestCase):
    """Plan semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def plan(self, snapshot: dict, replica_id: str = "remote") -> dict:
        return self.store.plan_replication_sync(replica_id, snapshot)

    def test_both_empty_reports_identical_with_no_actions(self) -> None:
        report = self.plan({})
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
                "actions": 0,
                "identical": True,
            },
        )

    def test_identical_snapshots_generate_no_actions(self) -> None:
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
        report = self.plan(snapshot)
        self.assertEqual(report["keys"], [])
        self.assertEqual(report["summary"]["actions"], 0)
        self.assertTrue(report["summary"]["identical"])
        self.assertEqual(report["summary"]["localKeys"], 2)
        self.assertEqual(report["summary"]["remoteKeys"], 2)
        self.assertEqual(report["summary"]["localCandidates"], 3)
        self.assertEqual(report["summary"]["remoteCandidates"], 3)

    def test_local_only_and_remote_only_entries(self) -> None:
        self.store.apply_operation("r1", operation("o1", "mine", "v", {"r1": 1}))
        report = self.plan({"theirs": [candidate("r2", "o2", "w", {"r2": 1})]})
        self.assertEqual(
            [(e["key"], e["action"], e["kind"]) for e in report["keys"]],
            [
                ("mine", "send_local", "missing_remote"),
                ("theirs", "fetch_remote", "missing_local"),
            ],
        )
        mine, theirs = report["keys"]
        self.assertEqual(mine["local"]["value"], "v")
        self.assertIsNone(mine["remote"])
        self.assertIsNone(theirs["local"])
        self.assertEqual(theirs["remote"]["value"], "w")
        self.assertEqual(report["summary"]["actions"], 2)
        self.assertFalse(report["summary"]["identical"])

    def test_dominating_clock_sends_or_fetches(self) -> None:
        self.store.apply_operation("r1", operation("o1", "a", "v", {"r1": 2}))
        self.store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        snapshot = {
            # Local clock dominates: send the local version.
            "a": [candidate("r1", "o1", "v", {"r1": 1})],
            # Remote clock dominates: fetch the remote version.
            "b": [candidate("r2", "o2", "w", {"r2": 1, "r9": 3})],
        }
        report = self.plan(snapshot)
        self.assertEqual(
            [(e["key"], e["action"], e["kind"]) for e in report["keys"]],
            [("a", "send_local", "clock"), ("b", "fetch_remote", "clock")],
        )
        self.assertEqual(report["keys"][0]["local"]["clock"], {"r1": 2})
        self.assertEqual(report["keys"][0]["remote"]["clock"], {"r1": 1})
        self.assertEqual(report["keys"][1]["local"]["clock"], {"r2": 1})
        self.assertEqual(report["keys"][1]["remote"]["clock"], {"r2": 1, "r9": 3})

    def test_concurrent_clocks_with_same_value_need_semantic_resolution(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "k", "v", {"r1": 1, "r2": 2})
        )
        # Same identity, same value, concurrent clocks (missing components
        # count as zero): neither side may overwrite the other.
        report = self.plan({"k": [candidate("r1", "o1", "v", {"r1": 2, "r2": 1})]})
        (entry,) = report["keys"]
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["action"], "semantic_resolution")
        self.assertEqual(entry["local"]["clock"], {"r1": 1, "r2": 2})
        self.assertEqual(entry["remote"]["clock"], {"r1": 2, "r2": 1})
        self.assertEqual(report["summary"]["actions"], 1)

    def test_conflicting_values_need_semantic_resolution_and_keep_both(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "local-val", {"r1": 2}))
        report = self.plan({"k": [candidate("r1", "o1", "remote-val", {"r1": 1})]})
        (entry,) = report["keys"]
        self.assertEqual(entry["kind"], "conflict")
        self.assertEqual(entry["action"], "semantic_resolution")
        # Both conflicting versions are preserved for semantic repair.
        self.assertEqual(entry["local"]["value"], "local-val")
        self.assertEqual(entry["remote"]["value"], "remote-val")

    def test_all_actions_and_ordering(self) -> None:
        # Key a: shared o1 (no action), conflict o2, dominating clock o3,
        # local-only o4, remote-only o6. Key m: local-only. Key z:
        # remote-only.
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
        report = self.plan(snapshot)
        self.assertEqual(
            [(e["key"], e["action"], e["kind"]) for e in report["keys"]],
            [
                ("a", "semantic_resolution", "conflict"),
                ("a", "send_local", "clock"),
                ("a", "send_local", "missing_remote"),
                ("a", "fetch_remote", "missing_local"),
                ("m", "send_local", "missing_remote"),
                ("z", "fetch_remote", "missing_local"),
            ],
        )
        self.assertEqual(
            report["summary"],
            {
                "localKeys": 2,
                "remoteKeys": 2,
                "localCandidates": 5,
                "remoteCandidates": 5,
                "actions": 6,
                "identical": False,
            },
        )

    def test_stale_writes_and_replays_do_not_enter_the_plan(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        # Stale write: accepted into the log but adds no candidate.
        self.assertIs(
            self.store.apply_operation("r1", operation("o2", "k", "old", {"r1": 1})),
            HTTPStatus.CREATED,
        )
        # Replay and conflict never enter the log.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        self.store.apply_operation("r1", operation("o1", "k", "tampered", {"r1": 2}))
        report = self.plan({"k": [candidate("r1", "o1", "v", {"r1": 2})]})
        self.assertTrue(report["summary"]["identical"])
        self.assertEqual(report["keys"], [])
        self.assertEqual(report["summary"]["localCandidates"], 1)

    def test_plan_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        first = self.plan(snapshot)
        second = self.plan(snapshot)
        self.assertEqual(first, second)
        # The remote candidate was not imported and nothing else moved.
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "v")

    def test_plan_and_comparison_agree_on_identity_classification(self) -> None:
        self.store.apply_operation("r1", operation("o1", "a", "same", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "a", "local-val", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "b", "clk", {"r3": 2}))
        snapshot = {
            "a": [
                candidate("r1", "o1", "same", {"r1": 1}),
                candidate("r2", "o2", "remote-val", {"r2": 1}),
            ],
            "b": [candidate("r3", "o3", "clk", {"r3": 1})],
        }
        plan = self.plan(snapshot)
        compare = self.store.compare_replication_snapshot("remote", snapshot)
        # Every non-shared comparison entry appears in the plan with the
        # same kind, and the action count matches the difference count.
        diff_kinds = sorted(
            entry["kind"]
            for group in compare["keys"]
            for entry in group["differences"]
            if entry["kind"] != "shared"
        )
        self.assertEqual(sorted(e["kind"] for e in plan["keys"]), diff_kinds)
        self.assertEqual(
            plan["summary"]["actions"], compare["summary"]["differences"]
        )
        self.assertEqual(
            plan["summary"]["identical"], compare["summary"]["identical"]
        )


class PlanRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_same_state_same_remote_snapshot_same_plan_after_restart(self) -> None:
        snapshot = {
            "k1": [candidate("r1", "o1", "v1", {"r1": 1})],
            "k2": [candidate("r9", "o9", "w", {"r9": 1})],
        }
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "k1", "old", {"r1": 0}))
        before = store.plan_replication_sync("remote", snapshot)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.plan_replication_sync("remote", snapshot), before)
        del recovered
        self.assertEqual(
            StateStore(data_file=self.data_file).plan_replication_sync(
                "remote", snapshot
            ),
            before,
        )


class PlanHttpServerTests(unittest.TestCase):
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

    def plan(self, document: dict, path: str = PLAN_PATH) -> tuple[int, dict]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_both_empty_over_http(self) -> None:
        status, payload = self.plan(body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["replicaId"], "remote")
        self.assertEqual(payload["keys"], [])
        self.assertEqual(payload["summary"]["actions"], 0)
        self.assertTrue(payload["summary"]["identical"])

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request(
            "POST",
            PLAN_PATH,
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
                "actions",
                "identical",
            },
        )
        for name in (
            "localKeys",
            "remoteKeys",
            "localCandidates",
            "remoteCandidates",
            "actions",
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

    def test_full_plan_over_http(self) -> None:
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
        status, payload = self.plan(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replicaId"], "peer-b")
        self.assertEqual(
            [(e["key"], e["action"], e["kind"]) for e in payload["keys"]],
            [
                ("k1", "semantic_resolution", "conflict"),
                ("k1", "fetch_remote", "missing_local"),
                ("k2", "send_local", "missing_remote"),
                ("k3", "fetch_remote", "missing_local"),
            ],
        )
        summary = payload["summary"]
        self.assertEqual((summary["localKeys"], summary["remoteKeys"]), (2, 2))
        self.assertEqual(
            (summary["localCandidates"], summary["remoteCandidates"]), (3, 4)
        )
        self.assertEqual(summary["actions"], 4)
        self.assertFalse(summary["identical"])
        # Every action entry keeps both sides (null where absent).
        for entry in payload["keys"]:
            if entry["action"] == "send_local":
                self.assertIsNone(entry["remote"])
            elif entry["action"] == "fetch_remote":
                self.assertIsNone(entry["local"])
            else:
                self.assertEqual(entry["action"], "semantic_resolution")
                self.assertIsNotNone(entry["local"])
                self.assertIsNotNone(entry["remote"])

    def test_any_query_parameter_is_400(self) -> None:
        for suffix in ("?x=1", "?after=0", "?x=", "?x", "?=1", "?x=1&x=2"):
            status, payload = self.plan(body(), PLAN_PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(payload, {"error": "invalid_request"}, suffix)

    def test_empty_query_separator_is_still_200(self) -> None:
        status, _ = self.plan(body(), PLAN_PATH + "?")
        self.assertEqual(status, 200)

    def test_path_shape_mismatches_are_404(self) -> None:
        for path in (
            "/v1/replication/plan/extra",
            "/v1/replication",
            "/v1/replication/plan/",
            "/v1/replication/plans",
            "/v1/replication/plan//",
        ):
            status, payload = self.plan(body(), path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_and_body_checks(self) -> None:
        status, payload, _, _ = self.raw_request(
            "POST", "/v1/replication/plan/extra?x=1", {"not": "valid"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_on_the_route_is_404(self) -> None:
        status, payload = self.request("GET", PLAN_PATH)
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
                "snapshot": {"k": [candidate("r1", "o1", "v", {"r1": -0.0})]},
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
            status, payload = self.plan(document)
            self.assertEqual(status, 400, document)
            self.assertEqual(payload, {"error": "invalid_request"}, document)

    def test_non_finite_and_duplicate_raw_bodies_are_400(self) -> None:
        for raw_body in (
            b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":{"r1":NaN},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":{"r1":Infinity},"replicaId":"r1","operationId":"o1"}]}}',
            b'{"replicaId":"p","snapshot":{"k":[{"value":"v","clock":{"r1":1},"replicaId":"r1","operationId":"o1","operationId":"o2"}]}}',
            b'{"replicaId":"p","replicaId":"q","snapshot":{}}',
        ):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request(
                "POST",
                PLAN_PATH,
                body=raw_body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 400, raw_body)
            self.assertEqual(
                json.loads(response.read()), {"error": "invalid_request"}, raw_body
            )
            conn.close()

    def test_malformed_json_body_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            PLAN_PATH,
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
            status, _ = self.plan(document)
            self.assertEqual(status, 400)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        _, after_state = self.request("GET", "/v1/states/k")
        _, after_sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_sync, after_sync)

    def test_plan_does_not_import_or_mutate(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = body(
            "remote",
            {
                "k": [candidate("r2", "o2", "w", {"r2": 1})],
                "new": [candidate("r3", "o3", "z", {"r3": 1})],
            },
        )
        for _ in range(3):
            status, payload = self.plan(document)
            self.assertEqual(status, 200)
            self.assertEqual(payload["summary"]["actions"], 3)
        _, metrics = self.request("GET", "/v1/metrics")
        self.assertEqual(metrics["acceptedOperations"], 1)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v")
        status, _ = self.request("GET", "/v1/states/new")
        self.assertEqual(status, 404)
        _, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync_payload["operations"]), 1)

    def test_concurrent_commits_observe_consistent_plans(self) -> None:
        remote_snapshot = {
            "shared": [candidate(f"r{index}", f"op-{index}", f"v{index}", {f"r{index}": 1}) for index in range(40)]
        }
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                report = self.server.store.plan_replication_sync(
                    "remote", remote_snapshot
                )
                summary = report["summary"]
                if summary["localCandidates"] < summary["localKeys"]:
                    violations.append("localCandidates < localKeys")
                if summary["actions"] != len(report["keys"]):
                    violations.append("action count disagrees with keys")
                if summary["actions"] < 0:
                    violations.append("negative action count")
                seen = set()
                for entry in report["keys"]:
                    for side in ("local", "remote"):
                        if entry[side] is not None:
                            identity = (
                                entry["key"],
                                entry[side]["replicaId"],
                                entry[side]["operationId"],
                            )
                            if identity in seen:
                                violations.append("identity repeated in plan")
                            seen.add(identity)
                ordered = [
                    (e["key"], e.get("local") or e["remote"]) for e in report["keys"]
                ]
                keys = [key for key, _ in ordered]
                if keys != sorted(keys):
                    violations.append("keys not sorted")

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
        status, payload = self.plan(body("remote", remote_snapshot))
        self.assertEqual(status, 200)
        self.assertTrue(payload["summary"]["identical"])
        self.assertEqual(payload["summary"]["actions"], 0)
        self.assertEqual(payload["keys"], [])


class PlanHttpRequestLimitTests(unittest.TestCase):
    """The plan route keeps the shared Content-Length contract."""

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
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, PLAN_PATH, [("Content-Length", value)], b"{}"
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_conflicting_content_length_headers_are_400(self) -> None:
        status, payload = self.post_raw(
            self.port,
            PLAN_PATH,
            [("Content-Length", "2"), ("Content-Length", "3")],
            b"{}",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        # The content itself is invalid JSON; the declared size wins.
        status, payload = self.post_raw(
            self.port,
            PLAN_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_absurdly_long_content_length_digits_are_413(self) -> None:
        status, payload = self.post_raw(
            self.port,
            PLAN_PATH,
            [("Content-Length", "9" * 5000)],
            b"{}",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        # Missing declaration: 400 even without a bearer token.
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        # Over-limit declaration: 413, not 401, even with no token.
        status, payload = self.post_raw(
            self.auth_port,
            PLAN_PATH,
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
            PLAN_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["identical"], True)

    def test_invalid_body_at_exact_limit_is_400_not_413(self) -> None:
        body_bytes = b"x" * MAX_BODY_BYTES
        status, payload = self.post_raw(
            self.port,
            PLAN_PATH,
            [("Content-Length", str(len(body_bytes)))],
            body_bytes,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})


class PlanHttpAuthTests(unittest.TestCase):
    """The plan endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-plan-auth-")
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
                self.single_port, "POST", PLAN_PATH, body(), auth=auth
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", PLAN_PATH, body(), auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        # A write-only token is 403 without a challenge.
        status, payload, challenge = self.request(
            self.scope_port, "POST", PLAN_PATH, body(), auth="Bearer writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", PLAN_PATH, body(), auth=token
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["status"], "ok", token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", PLAN_PATH + "?x=1", {"nope": {}},
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
        self.request(self.single_port, "POST", PLAN_PATH, body())
        self.request(
            self.single_port, "POST", PLAN_PATH, body(), auth="Bearer nope"
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)


class PlanHttpPersistenceTests(unittest.TestCase):
    """With --data-file the same state and remote snapshot plan identically."""

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

    def test_plan_survives_restart(self) -> None:
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
            server, "POST", PLAN_PATH, body("remote", snapshot)
        )
        self.assertEqual(status, 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, after = self.request(
            server, "POST", PLAN_PATH, body("remote", snapshot)
        )
        self.assertEqual(status, 200)
        self.assertEqual(after, before)

    def test_plan_writes_nothing_to_disk(self) -> None:
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
            status, _ = self.request(server, "POST", PLAN_PATH, document)
            self.assertEqual(status, 200)

        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        # No temporary file is created alongside the data file.
        self.assertEqual(set(os.listdir(self._tmp.name)), before_entries)


class CompareClockDirectionTests(unittest.TestCase):
    """The comparison's same-value clock entries carry ``clockDirection``."""

    def setUp(self) -> None:
        self.store = StateStore()

    def compare_entry(self, remote_clock: dict) -> dict:
        snapshot = {"k": [candidate("r1", "o1", "v", remote_clock)]}
        report = self.store.compare_replication_snapshot("remote", snapshot)
        (entry,) = report["keys"][0]["differences"]
        return entry

    def test_local_dominates_is_L(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        entry = self.compare_entry({"r1": 1})
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "L")

    def test_remote_dominates_is_R(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        entry = self.compare_entry({"r1": 2, "r9": 1})
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "R")

    def test_concurrent_clocks_are_C(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "k", "v", {"r1": 1, "r2": 2})
        )
        entry = self.compare_entry({"r1": 2, "r2": 1})
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "C")

    def test_missing_components_count_as_zero(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # The remote clock adds a component the local clock lacks (zero).
        entry = self.compare_entry({"r1": 1, "r2": 1})
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "R")

    def test_only_clock_entries_carry_the_direction(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k", "local", {"r2": 1}))
        snapshot = {
            "k": [
                candidate("r1", "o1", "v", {"r1": 1}),
                candidate("r2", "o2", "remote", {"r2": 1}),
                candidate("r3", "o3", "w", {"r3": 1}),
            ],
            "other": [candidate("r4", "o4", "x", {"r4": 1})],
        }
        report = self.store.compare_replication_snapshot("remote", snapshot)
        for group in report["keys"]:
            for entry in group["differences"]:
                if entry["kind"] == "clock":
                    self.assertIn(entry["clockDirection"], ("L", "R", "C"))
                else:
                    self.assertNotIn("clockDirection", entry)


if __name__ == "__main__":
    unittest.main()
