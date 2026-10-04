"""Tests for the read-only multi-operation causal closure endpoint.

The endpoint is::

    POST /v1/causal/closure

with a body of exactly ``{"operations": [...]}`` naming 1 to 100 distinct
operation identities, each an object of exactly ``replicaId`` and
``operationId`` (both non-empty strings). From one committed snapshot it
computes the minimal common causal closure of the requested roots: every
root, plus every first-accepted record committed before at least one root
in the shared accepted log whose clock that root's clock strictly
dominates (missing components count as 0). The report is exactly
``roots`` (the requested identities in request order), ``operations``
(the closure in global commit order, archive shape), ``edges`` (the
direct cover edges inside the closure, identities only, in global commit
order), and ``summary`` (the counts ``roots``, ``operations``, ``edges``,
and ``sharedAncestors`` — the last counting closure records that strictly
precede at least two distinct roots). The whole query is one
committed-snapshot read and is strictly read-only.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
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
    load_scope_policy,
    parse_causal_closure_payload,
)

CLOSURE_PATH = "/v1/causal/closure"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identity(replica_id: str, operation_id: str) -> dict:
    return {"replicaId": replica_id, "operationId": operation_id}


def record(replica_id: str, operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {
        "replicaId": replica_id,
        "operation": operation(operation_id, key, value, clock),
    }


def edge(from_replica: str, from_operation: str, to_replica: str, to_operation: str) -> dict:
    return {
        "from": identity(from_replica, from_operation),
        "to": identity(to_replica, to_operation),
    }


def body(*identities: dict) -> dict:
    return {"operations": list(identities)}


class ParseCausalClosurePayloadTests(unittest.TestCase):
    """Body validation: exactly operations, 1-100 distinct identities."""

    def test_valid_bodies(self) -> None:
        self.assertEqual(
            parse_causal_closure_payload(
                b'{"operations":[{"replicaId":"r1","operationId":"o1"}]}'
            ),
            [("r1", "o1")],
        )
        self.assertEqual(
            parse_causal_closure_payload(
                {"operations": [identity("r1", "o1"), identity("r2", "o2")]}
            ),
            [("r1", "o1"), ("r2", "o2")],
        )
        # Request order is preserved, and whitespace around the document
        # is ordinary JSON whitespace.
        self.assertEqual(
            parse_causal_closure_payload(
                b' { "operations" : [ {"operationId":"o2","replicaId":"r2"},'
                b'{"replicaId":"r1","operationId":"o1"} ] }\n'
            ),
            [("r2", "o2"), ("r1", "o1")],
        )

    def test_list_bounds(self) -> None:
        identities = [identity(f"r{index}", "o") for index in range(100)]
        self.assertEqual(
            len(parse_causal_closure_payload({"operations": identities})), 100
        )
        for operations in ([], identities + [identity("r100", "o")]):
            with self.assertRaises(ValueError):
                parse_causal_closure_payload({"operations": operations})

    def test_duplicate_identities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_causal_closure_payload(
                b'{"operations":[{"replicaId":"r1","operationId":"o1"},'
                b'{"replicaId":"r1","operationId":"o1"}]}'
            )

    def test_identity_fields_are_validated(self) -> None:
        for raw in (
            b'{"operations":[{"replicaId":"r1"}]}',
            b'{"operations":[{"operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1","x":1}]}',
            b'{"operations":[{"replicaId":"","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":""}]}',
            b'{"operations":[{"replicaId":1,"operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":true}]}',
            b'{"operations":[{"replicaId":null,"operationId":"o1"}]}',
            b'{"operations":["r1"]}',
            b'{"operations":[null]}',
        ):
            with self.assertRaises(ValueError, msg=raw):
                parse_causal_closure_payload(raw)

    def test_document_shape_is_validated(self) -> None:
        for raw in (
            b"{}",
            b"[]",
            b"null",
            b'"operations"',
            b"1",
            b'{"operations":{}}',
            b'{"operations":"r1"}',
            b'{"operations":null}',
            b'{"operations":1}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}],"x":1}',
            b'{"x":[{"replicaId":"r1","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}]} {}',
            b"not json",
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}]}\xff',
        ):
            with self.assertRaises(ValueError, msg=raw):
                parse_causal_closure_payload(raw)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        for raw in (
            b'{"operations":[],"operations":[]}',
            b'{"operations":[{"replicaId":"r1","replicaId":"r2","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1","operationId":"o2"}]}',
        ):
            with self.assertRaises(ValueError, msg=raw):
                parse_causal_closure_payload(raw)


class CausalClosureStoreTests(unittest.TestCase):
    """Store-level semantics of the common causal closure snapshot."""

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        self.assertEqual(
            store.get_causal_closure([("r1", "nope")]),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        self.assertEqual(
            store.get_causal_closure([("nope", "o1")]),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )
        # One unknown identity fails the whole request.
        self.assertEqual(
            store.get_causal_closure([("r1", "o1"), ("r1", "nope")]),
            (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
        )

    def test_single_root_without_predecessors(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_causal_closure([("r1", "o1")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["roots"], [identity("r1", "o1")])
        self.assertEqual(
            payload["operations"], [record("r1", "o1", "k", "v", {"r1": 1})]
        )
        self.assertEqual(payload["edges"], [])
        self.assertEqual(
            payload["summary"],
            {"roots": 1, "operations": 1, "edges": 0, "sharedAncestors": 0},
        )

    def test_chain_closure_and_cover_edges(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "c", "x", {"r1": 3}))
        status, payload = store.get_causal_closure([("r1", "o3")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(o["replicaId"], o["operation"]["operationId"]) for o in payload["operations"]],
            [("r1", "o1"), ("r1", "o2"), ("r1", "o3")],
        )
        # o1 -> o3 is covered by o2 and never appears.
        self.assertEqual(
            payload["edges"],
            [edge("r1", "o1", "r1", "o2"), edge("r1", "o2", "r1", "o3")],
        )
        self.assertEqual(
            payload["summary"],
            {"roots": 1, "operations": 3, "edges": 2, "sharedAncestors": 0},
        )

    def test_diamond_closure(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("o3", "c", "x", {"r1": 1, "r3": 1}))
        store.apply_operation(
            "r1", operation("o4", "d", "y", {"r1": 2, "r2": 1, "r3": 1})
        )
        status, payload = store.get_causal_closure([("r1", "o4")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2", "o3", "o4"],
        )
        # o1 -> o4 is covered by both o2 and o3 and never appears.
        self.assertEqual(
            payload["edges"],
            [
                edge("r1", "o1", "r2", "o2"),
                edge("r1", "o1", "r3", "o3"),
                edge("r2", "o2", "r1", "o4"),
                edge("r3", "o3", "r1", "o4"),
            ],
        )
        self.assertEqual(
            payload["summary"],
            {"roots": 1, "operations": 4, "edges": 4, "sharedAncestors": 0},
        )

    def test_shared_predecessor_appears_once(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("base", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("x", "b", "w", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("y", "c", "x", {"r1": 1, "r3": 1}))
        store.apply_operation("r4", operation("z", "d", "y", {"r1": 1, "r4": 1}))
        status, payload = store.get_causal_closure(
            [("r2", "x"), ("r3", "y"), ("r4", "z")]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["roots"],
            [identity("r2", "x"), identity("r3", "y"), identity("r4", "z")],
        )
        # The shared base record enters the closure exactly once, in
        # global commit order.
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["base", "x", "y", "z"],
        )
        self.assertEqual(
            payload["edges"],
            [
                edge("r1", "base", "r2", "x"),
                edge("r1", "base", "r3", "y"),
                edge("r1", "base", "r4", "z"),
            ],
        )
        self.assertEqual(
            payload["summary"],
            {"roots": 3, "operations": 4, "edges": 3, "sharedAncestors": 1},
        )

    def test_concurrent_roots_without_shared_ancestors(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r2": 1}))
        status, payload = store.get_causal_closure([("r2", "o2"), ("r1", "o1")])
        self.assertIs(status, HTTPStatus.OK)
        # Roots keep the request order; operations keep commit order.
        self.assertEqual(
            payload["roots"], [identity("r2", "o2"), identity("r1", "o1")]
        )
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2"],
        )
        self.assertEqual(payload["edges"], [])
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 2, "edges": 0, "sharedAncestors": 0},
        )

    def test_root_can_precede_another_root(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "w", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "c", "x", {"r1": 3}))
        status, payload = store.get_causal_closure([("r1", "o3"), ("r1", "o2")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["roots"], [identity("r1", "o3"), identity("r1", "o2")]
        )
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2", "o3"],
        )
        self.assertEqual(
            payload["edges"],
            [edge("r1", "o1", "r1", "o2"), edge("r1", "o2", "r1", "o3")],
        )
        # o1 strictly precedes two distinct roots (o2 and o3); o2
        # strictly precedes only o3.
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 3, "edges": 2, "sharedAncestors": 1},
        )

    def test_records_committed_after_all_roots_are_excluded(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Committed after the only root and strictly smaller: it is not a
        # predecessor of any root, so it stays out of the closure.
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 0}))
        status, payload = store.get_causal_closure([("r1", "o1")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]], ["o1"]
        )

    def test_stale_writes_and_repairs_participate(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r1": 2, "r2": 1}))
        # Stale on its key (dominated by o2): accepted but adds no
        # candidate; it is still an ordinary committed record.
        store.apply_operation("r3", operation("o3", "k", "z", {"r1": 1, "r2": 1}))
        store.apply_operation("r3", operation("p1", "j", "a", {"r3": 1}))
        store.apply_operation("r4", operation("p2", "j", "b", {"r4": 1}))
        resolution = {
            "replicaId": "r5",
            "operationId": "fix-1",
            "value": "merged",
            "clock": {"r3": 1, "r4": 1, "r5": 1},
            "candidates": [
                {"replicaId": "r3", "operationId": "p1"},
                {"replicaId": "r4", "operationId": "p2"},
            ],
        }
        status, error = store.apply_resolution("j", resolution)
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertIsNone(error)
        status, payload = store.get_causal_closure([("r5", "fix-1")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["p1", "p2", "fix-1"],
        )
        # The stale write is a valid root; o2 was committed before it but
        # its clock is larger, so it is not one of o3's predecessors.
        status, payload = store.get_causal_closure([("r3", "o3")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o3"],
        )

    def test_replays_and_rejected_requests_never_appear(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Identical replay: no new record.
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        # Conflicting identity: rejected, no record.
        store.apply_operation("r1", operation("o1", "a", "other", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        status, payload = store.get_causal_closure([("r2", "o2")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2"],
        )

    def test_query_is_read_only(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        before_metrics = store.get_metrics()
        before_digest = store.get_verification_digest()
        before_audit = store.get_key_audit_digest("a")
        before_state = store.get_state("a")
        before_sync = store.get_sync_operations(0, 100)
        before_archive = store.get_operation("r2", "o2")
        store.get_causal_closure([("r2", "o2")])
        store.get_causal_closure([("absent", "absent")])
        self.assertEqual(store.get_metrics(), before_metrics)
        self.assertEqual(store.get_verification_digest(), before_digest)
        self.assertEqual(store.get_key_audit_digest("a"), before_audit)
        self.assertEqual(store.get_state("a"), before_state)
        self.assertEqual(store.get_sync_operations(0, 100), before_sync)
        self.assertEqual(store.get_operation("r2", "o2"), before_archive)

    def test_data_file_restart_preserves_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = str(Path(tmp) / "state.json")
            store = StateStore(data_file=data_file)
            store.apply_operation("r1", operation("o1", "a", "v", {"r1": 1}))
            store.apply_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
            expected = store.get_causal_closure([("r2", "o2")])
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_causal_closure([("r2", "o2")]), expected)
            self.assertEqual(
                recovered.get_causal_closure([("r2", "absent")]),
                (HTTPStatus.NOT_FOUND, {"error": "not_found"}),
            )


class HttpCausalClosureTests(unittest.TestCase):
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

    def close(self, identities: list[dict], query: str = ""):
        return self.request("POST", f"{CLOSURE_PATH}{query}", body(*identities))

    def seed_diamond(self) -> None:
        self.post_operation("r1", operation("o1", "a", "v", {"r1": 1}))
        self.post_operation("r2", operation("o2", "b", "w", {"r1": 1, "r2": 1}))
        self.post_operation("r3", operation("o3", "c", "x", {"r1": 1, "r3": 1}))

    def test_round_trip(self) -> None:
        self.seed_diamond()
        status, payload, headers, raw = self.close(
            [identity("r2", "o2"), identity("r3", "o3")]
        )
        self.assertEqual(status, 200)
        # Exactly roots, operations, edges, summary, in that order.
        self.assertEqual(list(payload), ["roots", "operations", "edges", "summary"])
        self.assertEqual(
            payload["roots"], [identity("r2", "o2"), identity("r3", "o3")]
        )
        self.assertEqual(
            payload["operations"],
            [
                record("r1", "o1", "a", "v", {"r1": 1}),
                record("r2", "o2", "b", "w", {"r1": 1, "r2": 1}),
                record("r3", "o3", "c", "x", {"r1": 1, "r3": 1}),
            ],
        )
        self.assertEqual(
            payload["edges"],
            [edge("r1", "o1", "r2", "o2"), edge("r1", "o1", "r3", "o3")],
        )
        self.assertEqual(list(payload["summary"]), ["roots", "operations", "edges", "sharedAncestors"])
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 3, "edges": 2, "sharedAncestors": 1},
        )
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        # Compact JSON in the contracted field order terminated by exactly
        # one newline, with the declared length covering the terminator.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw, json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        self.assertLess(raw.index(b'"roots"'), raw.index(b'"operations"'))
        self.assertLess(raw.index(b'"operations"'), raw.index(b'"edges"'))
        self.assertLess(raw.index(b'"edges"'), raw.index(b'"summary"'))
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_same_request_on_same_snapshot_is_byte_identical(self) -> None:
        self.seed_diamond()
        _, _, _, first = self.close([identity("r2", "o2"), identity("r3", "o3")])
        _, _, _, second = self.close([identity("r2", "o2"), identity("r3", "o3")])
        self.assertEqual(first, second)

    def test_non_ascii_strings_are_written_literally(self) -> None:
        self.post_operation("r1", operation("o1", "clé", "bléu", {"r1": 1}))
        status, _, _, raw = self.close([identity("r1", "o1")])
        self.assertEqual(status, 200)
        body_text = raw.decode("utf-8")
        self.assertIn('"clé"', body_text)
        self.assertIn('"bléu"', body_text)
        self.assertNotIn("\\u", body_text)

    def test_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 3}))
        self.post_operation("r2", operation("o2", "a", "x", {"r1": 3, "r2": 2}))
        status, _, _, raw = self.close([identity("r2", "o2")])
        self.assertEqual(status, 200)
        body_text = raw.decode("utf-8")
        self.assertNotIn(".", body_text)
        self.assertNotIn("NaN", body_text)
        self.assertNotIn("Infinity", body_text)
        self.assertIn('"r1":3', body_text)
        self.assertIn('"r2":2', body_text)
        self.assertIn('"roots":1', body_text)

    def test_unknown_identity_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for identities in (
            [identity("r1", "absent")],
            [identity("absent", "o1")],
            [identity("r1", "o1"), identity("absent", "absent")],
        ):
            status, payload, _, raw = self.close(identities)
            self.assertEqual(status, 404, identities)
            self.assertEqual(payload, {"error": "not_found"}, identities)
            self.assertEqual(raw, b'{"error":"not_found"}\n', identities)

    def test_bad_bodies_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for invalid in (
            body(),
            body(*[identity(f"r{index}", "o") for index in range(101)]),
            body(identity("r1", "o1"), identity("r1", "o1")),
            {"operations": [{"replicaId": "r1"}]},
            {"operations": [{"replicaId": "r1", "operationId": "o1", "x": 1}]},
            {"operations": [{"replicaId": "", "operationId": "o1"}]},
            {"operations": [{"replicaId": "r1", "operationId": ""}]},
            {"operations": [{"replicaId": 1, "operationId": "o1"}]},
            {"operations": [{"replicaId": "r1", "operationId": True}]},
            {"operations": ["r1"]},
            {"operations": "r1"},
            {"operations": {}},
            {"operations": None},
            {"operations": [identity("r1", "o1")], "x": 1},
            {},
            [],
        ):
            status, payload, _, _ = self.request("POST", CLOSURE_PATH, invalid)
            self.assertEqual(status, 400, invalid)
            self.assertEqual(payload, {"error": "invalid_request"}, invalid)

    def test_malformed_documents_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for raw_body in (
            b"not json",
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}]} {}',
            b'{"operations":[],"operations":[]}',
            b'{"operations":[{"replicaId":"r1","replicaId":"r2","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1","operationId":"o2"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}]}\xff',
        ):
            status, payload, _, _ = self.request("POST", CLOSURE_PATH, raw_body)
            self.assertEqual(status, 400, raw_body)
            self.assertEqual(payload, {"error": "invalid_request"}, raw_body)

    def test_query_parameters_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for query in ("?x=1", "?x=1&x=2", "?x", "?=1"):
            status, payload, _, _ = self.close([identity("r1", "o1")], query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_missing_and_extra_segments_are_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/closure/extra",
            "/v1/causal/closure/",
            "/v1/causal//closure",
            "/v1/causal",
            "/v2/causal/closure",
        ):
            status, payload, _, _ = self.request(
                "POST", path, body(identity("r1", "o1"))
            )
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_route_shape_error_beats_body_error(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        for path in (
            "/v1/causal/closure/extra",
            "/v1/causal/closure/",
            "/v2/causal/closure",
        ):
            status, payload, _, _ = self.request("POST", path, b"not json")
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_closure_route_is_404(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.request("GET", CLOSURE_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejected_requests_read_and_change_nothing(self) -> None:
        self.seed_diamond()
        before_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        before_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        before_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        _, _, _, before_closure = self.close([identity("r2", "o2")])
        self.close([], "?x=1")
        self.request("POST", CLOSURE_PATH, b"not json")
        self.close([identity("absent", "absent")])
        self.request("POST", f"{CLOSURE_PATH}/", body(identity("r1", "o1")))
        after_metrics, _, _, _ = self.request("GET", "/v1/metrics")
        after_digest, _, _, _ = self.request("GET", "/v1/verification/digest")
        after_archive, _, _, _ = self.request("GET", "/v1/replicas/r2/operations/o2")
        _, _, _, after_closure = self.close([identity("r2", "o2")])
        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_archive, after_archive)
        self.assertEqual(before_closure, after_closure)

    def test_query_is_read_only_over_http(self) -> None:
        self.seed_diamond()
        before, _, _, _ = self.request("GET", "/v1/metrics")
        self.close([identity("r2", "o2"), identity("r3", "o3")])
        self.close([identity("absent", "absent")])
        after, _, _, _ = self.request("GET", "/v1/metrics")
        self.assertEqual(before, after)


class HttpCausalClosureAuthTests(unittest.TestCase):
    """The closure endpoint authenticates as a read endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="sestate-closure-auth-")
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

    def closure_body(self) -> dict:
        return body(identity("r1", "o1"))

    def test_single_token_mode_requires_bearer_token(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        for auth in (None, "Bearer nope", "bearer sekret", "Bearer  sekret", "Bearer"):
            status, payload, challenge = self.request(
                self.single_port, "POST", CLOSURE_PATH, self.closure_body(),
                auth=auth,
            )
            self.assertEqual(status, 401, auth)
            self.assertEqual(payload, {"error": "unauthorized"}, auth)
            self.assertEqual(challenge, "Bearer", auth)
        status, payload, _ = self.request(
            self.single_port, "POST", CLOSURE_PATH, self.closure_body(),
            auth="Bearer sekret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["roots"], [identity("r1", "o1")])

    def test_scope_mode_requires_read_or_admin(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CLOSURE_PATH, self.closure_body(),
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)
        for token in ("Bearer reader", "Bearer admin"):
            status, payload, _ = self.request(
                self.scope_port, "POST", CLOSURE_PATH, self.closure_body(),
                auth=token,
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(payload["roots"], [identity("r1", "o1")], token)

    def test_scope_failure_precedes_query_and_body_validation(self) -> None:
        self.seed(self.scope_port, "Bearer writer")
        status, payload, challenge = self.request(
            self.scope_port, "POST", CLOSURE_PATH + "?x=1", {"nope": {}},
            auth="Bearer writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(challenge)

    def test_rejected_auth_reads_and_changes_nothing(self) -> None:
        self.seed(self.single_port, "Bearer sekret")
        before, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.request(self.single_port, "POST", CLOSURE_PATH, self.closure_body())
        self.request(
            self.single_port, "POST", CLOSURE_PATH, self.closure_body(),
            auth="Bearer nope",
        )
        after, _, _ = self.request(
            self.single_port, "GET", "/v1/metrics", auth="Bearer sekret"
        )
        self.assertEqual(before, after)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.request(self.single_port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class HttpCausalClosureRequestLimitTests(unittest.TestCase):
    """The closure route keeps the shared Content-Length contract."""

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
        conn.putrequest("POST", CLOSURE_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()

    def test_malformed_content_lengths_are_400(self) -> None:
        for value in ("abc", "", "+5", "-5", "5 ", "1 2", "1.0", "²", "5,5"):
            with self.subTest(value=value):
                status, payload = self.post_raw(
                    self.port, CLOSURE_PATH,
                    [("Content-Length", value)], b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_over_limit_declaration_is_413_without_reading_body(self) -> None:
        status, payload = self.post_raw(
            self.port,
            CLOSURE_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_400_and_413_keep_priority_over_authentication(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.auth_port, timeout=10)
        conn.putrequest("POST", CLOSURE_PATH)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read()), {"error": "invalid_request"})
        conn.close()
        status, payload = self.post_raw(
            self.auth_port,
            CLOSURE_PATH,
            [("Content-Length", str(MAX_BODY_BYTES + 1))],
            b"not json",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_body_at_exact_limit_is_processed_normally(self) -> None:
        template = b'{"operations":[{"replicaId":"","operationId":"o"}]}'
        pad = MAX_BODY_BYTES - len(template)
        self.assertGreater(pad, 0)
        name = b"r" + b"x" * (pad - 1)
        body = (
            b'{"operations":[{"replicaId":"' + name + b'","operationId":"o"}]}'
        )
        self.assertEqual(len(body), MAX_BODY_BYTES)
        # The valid at-limit document is processed by the endpoint's
        # normal semantics: an unknown identity is 404, not a rejection.
        status, payload = self.post_raw(
            self.port, CLOSURE_PATH,
            [("Content-Length", str(len(body)))], body,
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_rejections_change_no_state(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST", "/v1/replicas/r1/operations",
            body=json.dumps(operation("o1", "k", "v1", {"r1": 1})),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/states/k")
        response = conn.getresponse()
        before = response.read()
        conn.close()
        for headers, body_bytes in (
            ([], b"{}"),
            ([("Content-Length", "abc")], b"{}"),
            ([("Content-Length", str(MAX_BODY_BYTES + 1))], b"junk"),
        ):
            status, _ = self.post_raw(self.port, CLOSURE_PATH, headers, body_bytes)
            self.assertIn(status, (400, 413))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/states/k")
        response = conn.getresponse()
        self.assertEqual(response.read(), before)
        conn.close()


class HttpCausalClosurePersistenceTests(unittest.TestCase):
    """The closure report survives a data-file restart unchanged."""

    def test_restart_preserves_closure(self) -> None:
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
                    for method, path, request_body in actions:
                        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                        if request_body is None:
                            conn.request(method, path)
                        else:
                            conn.request(
                                method,
                                path,
                                body=json.dumps(request_body),
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
                        operation("o1", "a", "v", {"r1": 1}),
                    ),
                    (
                        "POST",
                        "/v1/replicas/r2/operations",
                        operation("o2", "b", "w", {"r1": 1, "r2": 1}),
                    ),
                    (
                        "POST",
                        CLOSURE_PATH,
                        body(identity("r2", "o2")),
                    ),
                ]
            )
            self.assertEqual(first[0][0], 201)
            self.assertEqual(first[1][0], 201)
            self.assertEqual(first[2][0], 200)
            second = serve_once(
                [
                    ("POST", CLOSURE_PATH, body(identity("r2", "o2"))),
                    ("POST", CLOSURE_PATH, body(identity("r2", "absent"))),
                    ("POST", CLOSURE_PATH, body()),
                ]
            )
            # Same state before and after the restart: identical bytes,
            # including the trailing newline.
            self.assertEqual(second[0][0], 200)
            self.assertEqual(second[0][1], first[2][1])
            self.assertEqual(second[1], (404, b'{"error":"not_found"}\n'))
            self.assertEqual(second[2], (400, b'{"error":"invalid_request"}\n'))
            # The read-only query created no temporary files.
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main()
