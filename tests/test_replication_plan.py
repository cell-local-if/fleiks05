"""Tests for the read-only replica synchronization plan endpoint.

The endpoint is::

    POST /v1/replication/plan

with a body of exactly ``{"replicaId": ..., "snapshot": {...}}`` — the
same remote identifier and complete candidate snapshot accepted by
``POST /v1/replication/compare`` under the same write constraints (floats,
``-0.0``, non-finite values, duplicated identities, and unknown fields are
all rejected). The endpoint never imports the remote snapshot; it builds
an executable follow-up plan from one committed local snapshot:

- a local-only or locally dominating identity is ``send_local``;
- a remote-only or remotely dominating identity is ``fetch_remote``;
- an identity the two sides hold with different values, or whose clocks
  are concurrent, is ``semantic_resolution`` and keeps both candidates;
- an identity held identically (same value and clock) produces no action,
  and when the two states are identical the plan is empty.

The success body is compact canonical UTF-8 JSON terminated by one
newline and carries exactly ``status``, ``replicaId``, ``keys``, and
``summary``; the summary uses JSON integers/booleans only.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for plan semantics. Only the
Python standard library is used.
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
    RequestHandler,
    SemanticStateServer,
    StateStore,
    clock_direction,
    load_scope_policy,
    parse_replication_compare_payload,
)

PLAN_PATH = "/v1/replication/plan"
COMPARE_PATH = "/v1/replication/compare"


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


class ClockDirectionTests(unittest.TestCase):
    def test_directions(self) -> None:
        self.assertEqual(clock_direction({"r1": 2}, {"r1": 1}), "L")
        self.assertEqual(clock_direction({"r1": 1}, {"r1": 2}), "R")
        self.assertEqual(clock_direction({"r1": 1}, {"r2": 1}), "C")
        # Missing components count as zero.
        self.assertEqual(clock_direction({"r1": 1, "r2": 1}, {"r1": 1}), "L")
        self.assertEqual(clock_direction({"r1": 1}, {"r1": 1, "r2": 1}), "R")


class PlanStoreTests(unittest.TestCase):
    """Plan semantics against ``StateStore`` directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def plan(self, snapshot: dict, replica_id: str = "remote") -> dict:
        return self.store.plan_replication_sync(replica_id, snapshot)

    def test_both_empty_is_an_empty_identical_plan(self) -> None:
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

    def test_identical_snapshots_produce_no_actions(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "k1", "w", {"r2": 1}))
        snapshot = {
            "k1": [
                candidate("r1", "o1", "v", {"r1": 1}),
                candidate("r2", "o2", "w", {"r2": 1}),
            ],
        }
        report = self.plan(snapshot)
        self.assertEqual(report["keys"], [])
        self.assertEqual(report["summary"]["actions"], 0)
        self.assertTrue(report["summary"]["identical"])
        self.assertEqual(
            (report["summary"]["localKeys"], report["summary"]["remoteKeys"]),
            (1, 1),
        )
        self.assertEqual(
            (
                report["summary"]["localCandidates"],
                report["summary"]["remoteCandidates"],
            ),
            (2, 2),
        )

    def test_local_only_is_send_local_and_remote_only_is_fetch_remote(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.plan({"k2": [candidate("r2", "o2", "w", {"r2": 1})]})
        by_key = {group["key"]: group["actions"] for group in report["keys"]}
        (send,) = by_key["k"]
        self.assertEqual(send["action"], "send_local")
        self.assertEqual(send["kind"], "missing_remote")
        self.assertIsNone(send["remote"])
        self.assertEqual(send["local"]["value"], "v")
        (fetch,) = by_key["k2"]
        self.assertEqual(fetch["action"], "fetch_remote")
        self.assertEqual(fetch["kind"], "missing_local")
        self.assertIsNone(fetch["local"])
        self.assertEqual(fetch["remote"]["value"], "w")
        self.assertEqual(report["summary"]["actions"], 2)
        self.assertFalse(report["summary"]["identical"])

    def test_same_value_local_dominating_is_send_local(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        report = self.plan({"k": [candidate("r1", "o1", "v", {"r1": 1})]})
        (action,) = report["keys"][0]["actions"]
        self.assertEqual(action["action"], "send_local")
        self.assertEqual(action["kind"], "clock")
        self.assertEqual(action["local"]["clock"], {"r1": 2})
        self.assertEqual(action["remote"]["clock"], {"r1": 1})

    def test_same_value_remote_dominating_is_fetch_remote(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        report = self.plan(
            {"k": [candidate("r1", "o1", "v", {"r1": 2, "r9": 1})]}
        )
        (action,) = report["keys"][0]["actions"]
        self.assertEqual(action["action"], "fetch_remote")
        self.assertEqual(action["kind"], "clock")

    def test_same_value_concurrent_clocks_is_semantic_resolution(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.plan({"k": [candidate("r1", "o1", "v", {"r2": 1})]})
        (action,) = report["keys"][0]["actions"]
        self.assertEqual(action["action"], "semantic_resolution")
        self.assertEqual(action["kind"], "clock")
        # Both concurrent versions are retained for semantic repair.
        self.assertEqual(action["local"]["clock"], {"r1": 1})
        self.assertEqual(action["remote"]["clock"], {"r2": 1})
        self.assertEqual(action["local"]["value"], "v")
        self.assertEqual(action["remote"]["value"], "v")

    def test_different_values_are_semantic_resolution_even_when_dominated(self) -> None:
        # Even though the local clock dominates, a value conflict must not
        # be auto-overwritten by a send.
        self.store.apply_operation(
            "r1", operation("o1", "k", "local-v", {"r1": 2})
        )
        report = self.plan(
            {"k": [candidate("r1", "o1", "remote-v", {"r1": 1})]}
        )
        (action,) = report["keys"][0]["actions"]
        self.assertEqual(action["action"], "semantic_resolution")
        self.assertEqual(action["kind"], "conflict")
        self.assertEqual(action["local"]["value"], "local-v")
        self.assertEqual(action["remote"]["value"], "remote-v")

    def test_different_concurrent_values_are_semantic_resolution(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "k", "local-v", {"r1": 1})
        )
        report = self.plan(
            {"k": [candidate("r1", "o1", "remote-v", {"r2": 1})]}
        )
        (action,) = report["keys"][0]["actions"]
        self.assertEqual(action["action"], "semantic_resolution")
        self.assertEqual(action["kind"], "conflict")

    def test_keys_sorted_and_identities_sorted_and_converged_keys_omitted(self) -> None:
        self.store.apply_operation("r1", operation("o1", "z", "v", {"r1": 1}))
        self.store.apply_operation("r3", operation("o3", "a", "v", {"r3": 1}))
        snapshot = {
            # Key a: o3 converged (omitted as a group), o4 to fetch.
            "a": [
                candidate("r3", "o3", "v", {"r3": 1}),
                candidate("r4", "o4", "x", {"r4": 1}),
            ],
            "m": [candidate("r5", "o5", "y", {"r5": 1})],
            "z": [candidate("r1", "o1", "v", {"r1": 1})],
        }
        report = self.plan(snapshot)
        self.assertEqual([group["key"] for group in report["keys"]], ["a", "m"])
        ids = [
            (entry["local"] or entry["remote"])["operationId"]
            for entry in report["keys"][0]["actions"]
        ]
        self.assertEqual(ids, ["o4"])

    def test_plan_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        snapshot = {"k": [candidate("r2", "o2", "w", {"r2": 1})]}
        first = self.plan(snapshot)
        second = self.plan(snapshot)
        self.assertEqual(first, second)
        # The remote candidate was not imported.
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)
        status, state = self.store.get_state("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(state["value"], "v")

    def test_summary_counts_both_sides(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k1", "v", {"r1": 1}))
        report = self.plan(
            {
                "k2": [candidate("r2", "o2", "w", {"r2": 1})],
                "k3": [candidate("r3", "o3", "x", {"r3": 1})],
            }
        )
        self.assertEqual(
            report["summary"],
            {
                "localKeys": 1,
                "remoteKeys": 2,
                "localCandidates": 1,
                "remoteCandidates": 2,
                "actions": 3,
                "identical": False,
            },
        )


class PlanRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_same_state_same_snapshot_same_plan_after_restart(self) -> None:
        snapshot = {
            "k1": [candidate("r1", "o1", "v1", {"r1": 1})],
            "k2": [candidate("r9", "o9", "w", {"r9": 1})],
        }
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k1", "v2", {"r2": 1}))
        before = store.plan_replication_sync("remote", snapshot)

        del store
        recovered = StateStore(data_file=self.data_file)
        self.assertEqual(recovered.plan_replication_sync("remote", snapshot), before)


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
    ) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, bytes) else body,
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

    def plan(self, document: dict, path: str = PLAN_PATH) -> tuple[int, dict]:
        return self.request("POST", path, document)

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def test_empty_plan_over_http(self) -> None:
        status, payload = self.plan(body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["replicaId"], "remote")
        self.assertEqual(payload["keys"], [])
        summary = payload["summary"]
        self.assertTrue(summary["identical"])
        self.assertEqual(summary["actions"], 0)

    def test_payload_shape_headers_and_trailing_newline(self) -> None:
        status, payload, raw = self.raw_request("POST", PLAN_PATH, body())
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
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        self.assertEqual(
            raw[:-1].decode("utf-8"),
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )

    def test_action_entries_have_exactly_four_fields(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        document = body("rb", {"k": [candidate("r1", "o1", "w", {"r1": 1, "r2": 1})]})
        status, payload = self.plan(document)
        self.assertEqual(status, 200)
        (entry,) = payload["keys"][0]["actions"]
        self.assertEqual(set(entry), {"action", "kind", "local", "remote"})
        self.assertEqual(entry["action"], "semantic_resolution")
        self.assertEqual(entry["kind"], "conflict")

    def test_full_plan_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        document = body(
            "peer-b",
            {
                "k1": [
                    candidate("r1", "o1", "v1", {"r1": 2}),
                    candidate("r9", "o9", "w", {"r9": 1}),
                ],
                "k3": [candidate("r5", "o5", "y", {"r5": 1})],
            },
        )
        status, payload = self.plan(document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replicaId"], "peer-b")
        self.assertEqual([group["key"] for group in payload["keys"]], ["k1", "k3"])
        actions = [
            (entry["action"], entry["kind"])
            for group in payload["keys"]
            for entry in group["actions"]
        ]
        self.assertEqual(
            actions,
            [("fetch_remote", "clock"), ("fetch_remote", "missing_local"),
             ("fetch_remote", "missing_local")],
        )
        summary = payload["summary"]
        self.assertEqual((summary["localKeys"], summary["remoteKeys"]), (1, 2))
        self.assertEqual(
            (summary["localCandidates"], summary["remoteCandidates"]), (1, 3)
        )
        self.assertFalse(summary["identical"])
        self.assertEqual(summary["actions"], 3)

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
        status, payload, _ = self.raw_request(
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

    def test_malformed_json_body_is_400(self) -> None:
        status, payload, _ = self.raw_request("POST", PLAN_PATH, b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_rejected_requests_change_no_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_digest = self.request("GET", "/v1/verification/digest")
        for document in (
            {"replicaId": "p"},
            {
                "replicaId": "p",
                "snapshot": {"k": [candidate("r1", "o1", "v", {"r1": -1})]},
            },
        ):
            status, _ = self.plan(document)
            self.assertEqual(status, 400)
        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_digest = self.request("GET", "/v1/verification/digest")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)

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
        status, _ = self.request("GET", "/v1/states/new")
        self.assertEqual(status, 404)
        _, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync_payload["operations"]), 1)


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

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw(
            self.port,
            PLAN_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", PLAN_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        status, payload = self.post_raw(
            self.auth_port,
            PLAN_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

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

    def test_single_token_mode_requires_bearer_token(self) -> None:
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

    def test_health_stays_anonymous(self) -> None:
        for port in (self.single_port, self.scope_port):
            status, payload, _ = self.request(port, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")


class PlanHttpPersistenceTests(unittest.TestCase):
    """With --data-file the plan is read-only and stable across restarts."""

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
            "k1": [candidate("r1", "o1", "vX", {"r1": 9})],
            "k2": [candidate("r9", "o9", "w", {"r9": 1})],
        }
        server = self.start_server()
        self.request(
            server, "POST", "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 1}),
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
        self.assertEqual(set(os.listdir(self._tmp.name)), before_entries)


class CompareClockDirectionTests(unittest.TestCase):
    """The existing comparison marks same-value/different-clock entries."""

    def setUp(self) -> None:
        self.store = StateStore()

    def test_clock_entries_carry_direction(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 2}))
        report = self.store.compare_replication_snapshot(
            "rb", {"k": [candidate("r1", "o1", "v", {"r1": 1})]}
        )
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "L")

        report = self.store.compare_replication_snapshot(
            "rb", {"k": [candidate("r1", "o1", "v", {"r1": 2, "r9": 1})]}
        )
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["clockDirection"], "R")

        fresh = StateStore()
        fresh.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = fresh.compare_replication_snapshot(
            "rb", {"k": [candidate("r1", "o1", "v", {"r2": 1})]}
        )
        (entry,) = report["keys"][0]["differences"]
        self.assertEqual(entry["kind"], "clock")
        self.assertEqual(entry["clockDirection"], "C")

    def test_non_clock_entries_have_no_direction(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        report = self.store.compare_replication_snapshot(
            "rb",
            {
                "k": [candidate("r1", "o1", "v", {"r1": 1})],
                "k2": [candidate("r2", "o2", "w", {"r2": 1})],
            },
        )
        for group in report["keys"]:
            for entry in group["differences"]:
                self.assertNotIn("clockDirection", entry)


if __name__ == "__main__":
    unittest.main()
