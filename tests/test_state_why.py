"""Tests for the read-only causal state explanation endpoint.

The explanation endpoint is::

    GET /v1/states/{key}/why

It reads one key's current candidate state and reports exactly five
categories of information: the key, the conflict status, the candidates
(identity, value, and clock), the pairwise causal relations (dominates,
concurrent, or overwrites), and the candidate each existing identity
policy would select. The query is strictly read-only, shares the commit
lock with writes/imports/repairs/checkpoints, and answers with compact
UTF-8 JSON terminated by one newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(replica_id: str, operation_id: str, value: str, clock: dict) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def identity(replica_id: str, operation_id: str) -> dict:
    return {"replicaId": replica_id, "operationId": operation_id}


class StateExplanationStoreTests(unittest.TestCase):
    """Store-level semantics of the explanation snapshot."""

    def test_unknown_key_is_404(self) -> None:
        store = StateStore()
        self.assertEqual(
            store.get_state_explanation("nope"),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_other_keys_do_not_leak(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_state_explanation("other"),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_single_candidate_resolved_with_empty_relations(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        only = candidate("r1", "o1", "v", {"r1": 1})
        self.assertEqual(
            payload,
            {
                "key": "k",
                "status": "resolved",
                "candidates": [only],
                "relations": [],
                "suggestion": {"lowest_identity": only, "highest_identity": only},
            },
        )

    def test_resolved_candidates_report_overwrites(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "blue", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "blue", {"r2": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "blue", {"r2": 1}),
            ],
        )
        # The agreed value's unique source relation: each pair covers the
        # same value, so neither side can conflict.
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r2", "o2"),
                    "relation": "overwrites",
                }
            ],
        )

    def test_conflict_explains_each_pairwise_concurrency(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "v3", {"r3": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r2", "o2"),
                    "relation": "concurrent",
                },
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r3", "o3"),
                    "relation": "concurrent",
                },
                {
                    "from": identity("r2", "o2"),
                    "to": identity("r3", "o3"),
                    "relation": "concurrent",
                },
            ],
        )

    def test_mixed_values_classify_each_pair(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "same", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "same", {"r2": 1}))
        store.apply_operation("r3", operation("o3", "k", "other", {"r3": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r2", "o2"),
                    "relation": "overwrites",
                },
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r3", "o3"),
                    "relation": "concurrent",
                },
                {
                    "from": identity("r2", "o2"),
                    "to": identity("r3", "o3"),
                    "relation": "concurrent",
                },
            ],
        )

    def test_candidates_sorted_by_identity(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o1", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o2", "k", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o1", "k", "v0", {"r1": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in payload["candidates"]],
            [("r1", "o1"), ("r1", "o2"), ("r2", "o1")],
        )

    def test_suggestion_matches_both_identity_policies(self) -> None:
        # The suggestion names exactly the candidates the two existing
        # automatic-resolution policies commit for the same conflict.
        for policy, expected in (
            ("lowest_identity", "v1"),
            ("highest_identity", "v2"),
        ):
            with self.subTest(policy=policy):
                store = StateStore()
                store.apply_operation("r2", operation("o1", "k", "v2", {"r2": 1}))
                store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
                status, payload = store.get_state_explanation("k")
                self.assertIs(status, HTTPStatus.OK)
                self.assertEqual(payload["suggestion"][policy]["value"], expected)
                request = {
                    "replicaId": "rx",
                    "operationId": "auto",
                    "clock": {"r1": 1, "r2": 1, "rx": 1},
                    "policy": policy,
                }
                status_code, committed, error = store.apply_auto_resolution("k", request)
                self.assertIs(status_code, HTTPStatus.CREATED)
                self.assertIsNone(error)
                self.assertEqual(committed["value"], expected)

    def test_suggestion_present_when_resolved(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v", {"r2": 1}))
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["suggestion"],
            {
                "lowest_identity": candidate("r1", "o1", "v", {"r1": 1}),
                "highest_identity": candidate("r2", "o2", "v", {"r2": 1}),
            },
        )

    def test_resolution_collapses_explanation_to_single_candidate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        resolution = {
            "replicaId": "r3",
            "operationId": "fix",
            "value": "merged",
            "clock": {"r1": 1, "r2": 1, "r3": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, error = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_state_explanation("k")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(
            payload["candidates"],
            [candidate("r3", "fix", "merged", {"r1": 1, "r2": 1, "r3": 1})],
        )
        self.assertEqual(payload["relations"], [])

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("k")
        before_state = store.get_state("k")
        store.get_state_explanation("k")
        store.get_state_explanation("absent")
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("k"), before_audit)
        self.assertEqual(store.get_state("k"), before_state)

    def test_data_file_restart_preserves_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
            expected = store.get_state_explanation("k")
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_state_explanation("k"), expected)
            self.assertEqual(
                recovered.get_state_explanation("absent"),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpStateWhyTests(unittest.TestCase):
    """HTTP contract against an in-memory server."""

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
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        headers = dict(response.getheaders())
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload, headers, raw

    def post_operation(self, replica: str, op: dict):
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_why(self, key: str, query: str = ""):
        return self.request("GET", f"/v1/states/{key}/why{query}")

    def test_conflict_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 1}))
        status, payload, headers, raw = self.get_why("color")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {"key", "status", "candidates", "relations", "suggestion"})
        self.assertEqual(payload["key"], "color")
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(
            payload["candidates"],
            [
                candidate("r1", "o1", "blue", {"r1": 1}),
                candidate("r2", "o2", "red", {"r2": 1}),
            ],
        )
        self.assertEqual(
            payload["relations"],
            [
                {
                    "from": identity("r1", "o1"),
                    "to": identity("r2", "o2"),
                    "relation": "concurrent",
                }
            ],
        )
        self.assertEqual(
            payload["suggestion"],
            {
                "lowest_identity": candidate("r1", "o1", "blue", {"r1": 1}),
                "highest_identity": candidate("r2", "o2", "red", {"r2": 1}),
            },
        )
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact JSON terminated by exactly one newline, with the declared
        # length covering the terminator.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n",
        )
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_resolved_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 1}))
        status, payload, _, raw = self.get_why("color")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["candidates"], [candidate("r1", "o1", "blue", {"r1": 1})])
        self.assertEqual(payload["relations"], [])
        self.assertTrue(raw.endswith(b"\n"))

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "color", "blue", {"r1": 3}))
        self.post_operation("r2", operation("o2", "color", "red", {"r2": 2}))
        status, _, _, raw = self.get_why("color")
        self.assertEqual(status, 200)
        body = raw.decode("utf-8")
        self.assertNotIn(".", body)
        self.assertNotIn("NaN", body)
        self.assertNotIn("Infinity", body)
        self.assertIn('"r1":3', body)
        self.assertIn('"r2":2', body)

    def test_unknown_key_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.get_why("absent")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("o1", "k/1", "v", {"r1": 1})
        status, _, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_why("k%2F1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k/1")

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/why/extra",
            "/v1/states/k/why/extra/more",
            "/v1/states//why",
            "/v1/why",
            "/v2/states/k/why",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?after=0", "?limit=1"):
            status, payload, _, _ = self.get_why("k", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_route_shape_error_beats_query_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/states/k/why/extra?x=1",
            "/v1/states//why?x=1",
            "/v2/states/k/why?x=1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_rejected_query_reads_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_state, _, _, _ = self.request("GET", "/v1/states/k")
        before_why, _, _, _ = self.get_why("k")
        self.get_why("k", "?x=1")
        self.get_why("absent", "?x=1")
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_state, _, _, _ = self.request("GET", "/v1/states/k")
        after_why, _, _, _ = self.get_why("k")
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_state, after_state)
        self.assertEqual(before_why, after_why)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_why("k")
        self.get_why("absent")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)

    def test_post_to_why_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("POST", "/v1/states/k/why", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class HttpStateWhyAuthTests(unittest.TestCase):
    """With auth enabled the explanation endpoint authenticates like any GET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="sekret"
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

    def request(self, method: str, path: str, body: object = None, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_why_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret"):
            status, payload = self.request("GET", "/v1/states/k/why", auth=auth)
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
        status, payload = self.request("GET", "/v1/states/k/why", auth="Bearer sekret")
        self.assertEqual(status, 200)
        self.assertEqual(payload["key"], "k")
        self.assertEqual(payload["status"], "resolved")

    def test_rejected_auth_changes_nothing(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.request("POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret")
        before, _ = self.request("GET", "/v1/metrics", auth="Bearer sekret")
        self.request("GET", "/v1/states/k/why")
        self.request("GET", "/v1/states/k/why", auth="Bearer nope")
        after, _ = self.request("GET", "/v1/metrics", auth="Bearer sekret")
        self.assertEqual(before, after)


class HttpStateWhyPersistenceTests(unittest.TestCase):
    """The explanation survives a data-file restart unchanged."""

    def test_restart_preserves_explanation(self) -> None:
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
                        results.append((response.status, raw))
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            first = serve_once(
                [
                    (
                        "POST",
                        "/v1/replicas/r1/operations",
                        operation("o1", "k", "v1", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "k", "v2", {"r2": 1}),
                    ),
                    ("GET", "/v1/states/k/why", None),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            second = serve_once(
                [
                    ("GET", "/v1/states/k/why", None),
                    ("GET", "/v1/states/absent/why", None),
                    ("GET", "/v1/states/k/why?x=1", None),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[2][1])
            self.assertEqual(second[1], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))


if __name__ == "__main__":
    unittest.main()
