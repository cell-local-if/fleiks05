"""Tests for the read-only operation provenance query.

The provenance endpoint is::

    GET /v1/replicas/{replicaId}/operations/{operationId}/provenance

It explains one first-accepted operation's immediate semantic effect on
its target key: the archive record, the local origin (an automatic
resolution with its bound policy, or anything else), and the key's
candidate snapshots replayed from the shared commit-order log to just
before and just after the target record. The query is strictly read-only
and shares the commit lock; a later overwrite of the key never changes
the report.

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
    _ordered_json_bytes,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def candidate(value: str, clock: dict, replica_id: str, operation_id: str) -> dict:
    return {
        "value": value,
        "clock": clock,
        "replicaId": replica_id,
        "operationId": operation_id,
    }


def snapshot(status: str, candidates: list) -> dict:
    return {"status": status, "candidates": candidates}


class OperationProvenanceStoreTests(unittest.TestCase):
    """Store-level semantics, both in memory and file backed."""

    def test_plain_write_reports_other_origin(self) -> None:
        store = StateStore()
        op = operation("o1", "k", "v", {"r1": 1})
        store.apply_operation("r1", op)
        status, payload = store.get_operation_provenance("r1", "o1")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "replicaId": "r1",
                "operationId": "o1",
                "operation": op,
                "origin": "other",
                "policy": None,
                "before": snapshot("absent", []),
                "after": snapshot(
                    "resolved", [candidate("v", {"r1": 1}, "r1", "o1")]
                ),
            },
        )

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica_id, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            self.assertEqual(
                store.get_operation_provenance(replica_id, operation_id),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )

    def test_stale_write_leaves_snapshot_unchanged(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "new", {"r1": 2}))
        stale = operation("o2", "k", "old", {"r1": 1})
        store.apply_operation("r1", stale)
        status, payload = store.get_operation_provenance("r1", "o2")
        self.assertIs(status, HTTPStatus.OK)
        expected = snapshot("resolved", [candidate("new", {"r1": 2}, "r1", "o1")])
        self.assertEqual(payload["before"], expected)
        self.assertEqual(payload["after"], expected)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])

    def test_conflicting_write_shows_conflict_after(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        status, payload = store.get_operation_provenance("r2", "o2")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["before"],
            snapshot("resolved", [candidate("v1", {"r1": 1}, "r1", "o1")]),
        )
        self.assertEqual(
            payload["after"],
            snapshot(
                "conflict",
                [
                    candidate("v1", {"r1": 1}, "r1", "o1"),
                    candidate("v2", {"r2": 1}, "r2", "o2"),
                ],
            ),
        )

    def test_candidates_sort_by_identity_and_carry_four_fields(self) -> None:
        store = StateStore()
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = store.get_operation_provenance("r1", "o1")
        self.assertIs(status, HTTPStatus.OK)
        candidates = payload["after"]["candidates"]
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in candidates],
            [("r1", "o1"), ("r2", "o2")],
        )
        for entry in candidates:
            self.assertEqual(
                set(entry), {"value", "clock", "replicaId", "operationId"}
            )

    def test_manual_resolution_reports_other_origin(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        resolution = {
            "replicaId": "r1",
            "operationId": "fix",
            "value": "merged",
            "clock": {"r1": 2, "r2": 1},
            "candidates": [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r2", "operationId": "o2"},
            ],
        }
        status, error = store.apply_resolution("k", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_operation_provenance("r1", "fix")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(
            payload["after"],
            snapshot(
                "resolved",
                [candidate("merged", {"r1": 2, "r2": 1}, "r1", "fix")],
            ),
        )

    def test_auto_resolution_reports_policy_and_dominance(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "highest_value",
        }
        status, committed, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        self.assertEqual(committed["value"], "v2")
        status, payload = store.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["operation"], committed)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "highest_value")
        # The automatic resolution committed from a value conflict, and
        # its clock dominated every prior candidate afterwards.
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(
            payload["after"],
            snapshot(
                "resolved",
                [candidate("v2", {"r3": 1, "r1": 1, "r2": 1}, "r3", "auto")],
            ),
        )

    def test_sync_import_reports_other_origin(self) -> None:
        store = StateStore()
        record = operation("o9", "k", "v", {"r9": 3})
        status, accepted, _ = store.import_operations([("r9", record)])
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 1)
        status, payload = store.get_operation_provenance("r9", "o9")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])

    def test_imported_resolution_carries_no_policy_binding(self) -> None:
        # A record that looks like an automatic resolution but arrived by
        # sync import has no local policy binding: it reports "other".
        source = StateStore()
        source.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        source.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "lowest_identity",
        }
        _, committed, _ = source.apply_auto_resolution("k", request)
        replica = StateStore()
        status, _, _ = replica.import_operations(
            [("r1", operation("o1", "k", "v1", {"r1": 1})),
             ("r2", operation("o2", "k", "v2", {"r2": 1})),
             ("r3", committed)]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        status, payload = replica.get_operation_provenance("r3", "auto")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["origin"], "other")
        self.assertIsNone(payload["policy"])

    def test_later_overwrite_does_not_change_report(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, first = store.get_operation_provenance("r2", "o2")
        # A dominating write and an automatic resolution later: the
        # target's immediate-effect report is frozen at its own commit.
        store.apply_operation("r1", operation("o3", "k", "v3", {"r1": 2}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 2, "r2": 1},
            "policy": "lowest_identity",
        }
        status, _, error = store.apply_auto_resolution("k", request)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        _, second = store.get_operation_provenance("r2", "o2")
        self.assertEqual(first, second)

    def test_only_target_key_records_enter_the_replay(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "other", "x", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "k", "v", {"r1": 2}))
        status, payload = store.get_operation_provenance("r1", "o2")
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["before"], snapshot("absent", []))
        self.assertEqual(
            payload["after"],
            snapshot("resolved", [candidate("v", {"r1": 2}, "r1", "o2")]),
        )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("k")
        store.get_operation_provenance("r1", "o1")
        store.get_operation_provenance("r1", "absent")
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("k"), before_audit)

    def test_data_file_restart_preserves_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
            request = {
                "replicaId": "r3",
                "operationId": "auto",
                "clock": {"r3": 1, "r1": 1, "r2": 1},
                "policy": "lowest_value",
            }
            store.apply_auto_resolution("k", request)
            expected = store.get_operation_provenance("r3", "auto")
            plain = store.get_operation_provenance("r1", "o1")
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_operation_provenance("r3", "auto"), expected)
            self.assertEqual(recovered.get_operation_provenance("r1", "o1"), plain)
            self.assertEqual(
                recovered.get_operation_provenance("r3", "absent"),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )

    def test_data_file_without_policies_section_reports_other(self) -> None:
        # A version:1 file written before policy bindings existed recovers
        # with an empty binding table: every identity reports "other".
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            document = {
                "version": 1,
                "operations": [
                    {
                        "replicaId": "r1",
                        "operation": operation("o1", "k", "v", {"r1": 1}),
                    }
                ],
            }
            Path(data_file).write_text(
                json.dumps(document), encoding="utf-8"
            )
            store = StateStore(data_file=data_file)
            status, payload = store.get_operation_provenance("r1", "o1")
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(payload["origin"], "other")
            self.assertIsNone(payload["policy"])
            self.assertEqual(payload["before"], snapshot("absent", []))
            self.assertEqual(payload["after"]["status"], "resolved")


class HttpOperationProvenanceTests(unittest.TestCase):
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

    def get_provenance(self, replica: str, operation_id: str, query: str = ""):
        return self.request(
            "GET", f"/v1/replicas/{replica}/operations/{operation_id}/provenance{query}"
        )

    def test_accepted_operation_round_trip(self) -> None:
        op = operation("o1", "color", "blue", {"r1": 1})
        status, _, _, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, payload, headers, raw = self.get_provenance("r1", "o1")
        self.assertEqual(status, 200)
        expected = {
            "replicaId": "r1",
            "operationId": "o1",
            "operation": op,
            "origin": "other",
            "policy": None,
            "before": {"status": "absent", "candidates": []},
            "after": {
                "status": "resolved",
                "candidates": [
                    {
                        "value": "blue",
                        "clock": {"r1": 1},
                        "replicaId": "r1",
                        "operationId": "o1",
                    }
                ],
            },
        }
        self.assertEqual(payload, expected)
        # The body is compact ordered JSON in the contracted field order
        # with exactly one trailing newline.
        self.assertEqual(raw, _ordered_json_bytes(expected) + b"\n")
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_auto_resolution_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        request = {
            "replicaId": "r3",
            "operationId": "auto",
            "clock": {"r3": 1, "r1": 1, "r2": 1},
            "policy": "highest_identity",
        }
        status, _, _, _ = self.request("POST", "/v1/states/k/resolve/auto", request)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_provenance("r3", "auto")
        self.assertEqual(status, 200)
        self.assertEqual(payload["origin"], "automatic_resolution")
        self.assertEqual(payload["policy"], "highest_identity")
        self.assertEqual(payload["before"]["status"], "conflict")
        self.assertEqual(payload["operation"]["value"], "v2")
        self.assertEqual(payload["after"]["status"], "resolved")
        self.assertEqual(
            [c["operationId"] for c in payload["after"]["candidates"]], ["auto"]
        )

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for replica, operation_id in [("r1", "nope"), ("nope", "o1"), ("nope", "nope")]:
            status, payload, _, _ = self.get_provenance(replica, operation_id)
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

    def test_path_segments_are_percent_decoded(self) -> None:
        op = operation("op 1", "k", "v", {"r/1": 1})
        status, _, _, _ = self.post_operation("r%2F1", op)
        self.assertEqual(status, 201)
        status, payload, _, _ = self.get_provenance("r%2F1", "op%201")
        self.assertEqual(status, 200)
        self.assertEqual(payload["replicaId"], "r/1")
        self.assertEqual(payload["operationId"], "op 1")

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/replicas/r1/operations/o1/provenance/extra",
            "/v1/replicas/r1/operations",
            "/v1/replicas/r1/provenance",
            "/v1/replicas//operations/o1/provenance",
            "/v1/replicas/r1/operations//provenance",
            "/v1/replicas/r1/operations/o1/provenance/",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_any_query_parameter_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?after=0"):
            status, payload, _, _ = self.get_provenance("r1", "o1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_route_shape_error_beats_query_error(self) -> None:
        # A route that does not match the provenance shape is 404 even
        # when it also carries a query parameter.
        for path in (
            "/v1/replicas/r1/operations/o1/provenance/extra?x=1",
            "/v1/replicas/r1/operations?x=1",
            "/v1/replicas//operations/o1/provenance?x=1",
        ):
            status, payload, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_wrong_method_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for method in ("POST", "PUT", "DELETE"):
            status, payload, _, _ = self.request(
                method, "/v1/replicas/r1/operations/o1/provenance", {}
            )
            self.assertEqual(status, 404, method)
            self.assertEqual(payload, {"error": "not_found"}, method)

    def test_query_is_read_only_over_http(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.get_provenance("r1", "o1")
        self.get_provenance("r1", "absent")
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)


class HttpOperationProvenanceAuthTests(unittest.TestCase):
    """With auth enabled the provenance query authenticates like any GET."""

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
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def test_provenance_requires_bearer_token(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, auth="Bearer sekret"
        )
        self.assertEqual(status, 201)
        status, payload, headers = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        status, payload, _ = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance", auth="Bearer sekret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operation"], op)


class HttpOperationProvenanceScopeTests(unittest.TestCase):
    """In scope-policy mode the read (or admin) scope gates the query."""

    READ_TOKEN = "reader-token"
    WRITE_TOKEN = "writer-token"
    POLICY = {
        READ_TOKEN: frozenset({"read"}),
        WRITE_TOKEN: frozenset({"write"}),
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

    def request(self, method: str, path: str, body: object = None, token: str | None = None):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers

    def test_read_scope_reaches_provenance(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _, _ = self.request(
            "POST", "/v1/replicas/r1/operations", op, token=self.WRITE_TOKEN
        )
        self.assertEqual(status, 201)
        status, payload, _ = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance", token=self.READ_TOKEN
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operation"], op)

    def test_write_only_token_is_forbidden_without_challenge(self) -> None:
        status, payload, headers = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance", token=self.WRITE_TOKEN
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn("WWW-Authenticate", headers)
        self.assertNotIn("Www-Authenticate", headers)

    def test_missing_token_is_unauthorized_with_challenge(self) -> None:
        status, payload, headers = self.request(
            "GET", "/v1/replicas/r1/operations/o1/provenance"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")


class HttpOperationProvenancePersistenceTests(unittest.TestCase):
    """The provenance report survives a data-file restart unchanged."""

    def test_restart_preserves_report(self) -> None:
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
                    ("POST", "/v1/replicas/r1/operations", operation("o1", "k", "v1", {"r1": 1})),
                    ("POST", "/v1/replicas/r2/operations", operation("o2", "k", "v2", {"r2": 1})),
                    (
                        "POST",
                        "/v1/states/k/resolve/auto",
                        {
                            "replicaId": "r3",
                            "operationId": "auto",
                            "clock": {"r3": 1, "r1": 1, "r2": 1},
                            "policy": "lowest_value",
                        },
                    ),
                ]
            )
            self.assertEqual([status for status, _ in first], [201, 201, 201])
            before = serve_once(
                [("GET", "/v1/replicas/r3/operations/auto/provenance", None)]
            )
            self.assertEqual(before[0][0], 200)
            after = serve_once(
                [
                    ("GET", "/v1/replicas/r3/operations/auto/provenance", None),
                    ("GET", "/v1/replicas/r3/operations/absent/provenance", None),
                    ("GET", "/v1/replicas/r3/operations/auto/provenance?x=1", None),
                ]
            )
            # Byte-identical success bodies across the restart.
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[1][0], 404)
            self.assertEqual(json.loads(after[1][1].decode("utf-8")), {"error": "not_found"})
            self.assertEqual(after[2][0], 400)
            self.assertEqual(
                json.loads(after[2][1].decode("utf-8")), {"error": "invalid_request"}
            )
            report = json.loads(after[0][1].decode("utf-8"))
            self.assertEqual(report["origin"], "automatic_resolution")
            self.assertEqual(report["policy"], "lowest_value")


if __name__ == "__main__":
    unittest.main()
