"""Tests for the read-only multi-operation causal closure endpoint.

The endpoint is::

    POST /v1/causal/closure

with a body of exactly ``{"operations": [...]}`` naming between 1 and
100 distinct operation identities, each ``{"replicaId", "operationId"}``
with non-empty string values. From one committed snapshot the report
contains every root plus each first-accepted record that is a strict
causal predecessor of at least one root (committed before that root in
the shared accepted log and strictly dominated by the root's clock,
missing components counting as 0). The response is exactly ``roots``
(request order), ``operations`` (the closure in global commit order,
archive shape), ``edges`` (the direct cover edges inside the closure),
and ``summary`` (``roots``/``operations``/``edges``/``sharedAncestors``
counts), as compact UTF-8 JSON with one trailing newline.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
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
        "operation": {
            "operationId": operation_id,
            "key": key,
            "value": value,
            "clock": clock,
        },
    }


def edge(from_replica: str, from_operation: str, to_replica: str, to_operation: str) -> dict:
    return {
        "from": identity(from_replica, from_operation),
        "to": identity(to_replica, to_operation),
    }


class ParseCausalClosurePayloadTests(unittest.TestCase):
    """Body validation: exactly operations + a 1-100 list of identities."""

    def test_minimal_document_passes(self) -> None:
        self.assertEqual(
            parse_causal_closure_payload(
                b'{"operations":[{"replicaId":"r1","operationId":"o1"}]}'
            ),
            [("r1", "o1")],
        )
        self.assertEqual(
            parse_causal_closure_payload(
                {"operations": [{"replicaId": "r1", "operationId": "o1"}]}
            ),
            [("r1", "o1")],
        )

    def test_request_order_is_preserved(self) -> None:
        parsed = parse_causal_closure_payload(
            b'{"operations":['
            b'{"replicaId":"r2","operationId":"o9"},'
            b'{"replicaId":"r1","operationId":"o2"},'
            b'{"replicaId":"r1","operationId":"o1"}]}'
        )
        self.assertEqual(parsed, [("r2", "o9"), ("r1", "o2"), ("r1", "o1")])

    def test_up_to_one_hundred_operations_pass(self) -> None:
        operations = [
            {"replicaId": "r1", "operationId": f"o{i}"} for i in range(100)
        ]
        parsed = parse_causal_closure_payload({"operations": operations})
        self.assertEqual(parsed, [("r1", f"o{i}") for i in range(100)])

    def test_malformed_documents_are_rejected(self) -> None:
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}x",
            b"[]",
            b"null",
            b'""',
            b"42",
            b'{"operations":[]}\n{}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}]}\xff',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload(raw)

    def test_missing_extra_and_unknown_fields_are_rejected(self) -> None:
        for raw in (
            b"{}",
            b'{"operations":[],"x":1}',
            b'{"x":[{"replicaId":"r1","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}],"x":1}',
            b'{"operations":[],"operations":[]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload(raw)

    def test_duplicate_keys_anywhere_are_rejected(self) -> None:
        for raw in (
            b'{"operations":[{"replicaId":"r1","replicaId":"r2","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1","operationId":"o2"}]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload(raw)

    def test_operations_must_be_a_non_empty_list(self) -> None:
        for raw in (
            b'{"operations":[]}',
            b'{"operations":"o1"}',
            b'{"operations":null}',
            b'{"operations":{}}',
            b'{"operations":1}',
            b'{"operations":true}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload(raw)

    def test_over_one_hundred_operations_are_rejected(self) -> None:
        operations = [
            {"replicaId": "r1", "operationId": f"o{i}"} for i in range(101)
        ]
        with self.assertRaises(ValueError):
            parse_causal_closure_payload({"operations": operations})

    def test_each_entry_must_hold_exactly_the_two_identity_fields(self) -> None:
        for operations in (
            [{}],
            [{"replicaId": "r1"}],
            [{"operationId": "o1"}],
            [{"replicaId": "r1", "operationId": "o1", "key": "k"}],
            [["r1", "o1"]],
            ["r1"],
            [1],
            [True],
            [None],
        ):
            with self.subTest(operations=operations):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload({"operations": operations})

    def test_identity_values_must_be_non_empty_strings(self) -> None:
        for entry in (
            {"replicaId": "", "operationId": "o1"},
            {"replicaId": "r1", "operationId": ""},
            {"replicaId": 1, "operationId": "o1"},
            {"replicaId": "r1", "operationId": 1},
            {"replicaId": True, "operationId": "o1"},
            {"replicaId": "r1", "operationId": None},
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload({"operations": [entry]})

    def test_duplicate_identities_are_rejected(self) -> None:
        for raw in (
            b'{"operations":[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r1","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r1","operationId":"o2"},'
            b'{"replicaId":"r1","operationId":"o1"}]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_closure_payload(raw)

    def test_same_operation_id_on_different_replicas_is_distinct(self) -> None:
        parsed = parse_causal_closure_payload(
            b'{"operations":[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r2","operationId":"o1"}]}'
        )
        self.assertEqual(parsed, [("r1", "o1"), ("r2", "o1")])

    def test_json_whitespace_is_allowed(self) -> None:
        self.assertEqual(
            parse_causal_closure_payload(
                b'  { "operations": [ { "replicaId": "r1", "operationId": "o1" } ] }\n'
            ),
            [("r1", "o1")],
        )


class CausalClosureStoreTests(unittest.TestCase):
    """Store-level semantics of the common-closure computation."""

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

    def test_unknown_identity_is_404(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = store.get_causal_closure([("r1", "nope")])
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})
        # One unknown identity fails the whole request.
        status, payload = store.get_causal_closure([("r1", "o1"), ("r2", "o2")])
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_closure_is_the_union_of_strict_predecessors_plus_roots(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "v2", {"r2": 1}))
        store.apply_operation("r1", operation("o3", "c", "v3", {"r1": 2, "r2": 1}))
        store.apply_operation("r2", operation("o4", "d", "v4", {"r2": 2}))
        # o3's predecessors: o1, o2. o4's predecessors: o2. The closure is
        # the union {o1, o2, o3, o4} in global commit order; o2 appears once.
        status, payload = store.get_causal_closure([("r1", "o3"), ("r2", "o4")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["roots"], [identity("r1", "o3"), identity("r2", "o4")]
        )
        self.assertEqual(
            payload["operations"],
            [
                record("r1", "o1", "a", "v1", {"r1": 1}),
                record("r2", "o2", "b", "v2", {"r2": 1}),
                record("r1", "o3", "c", "v3", {"r1": 2, "r2": 1}),
                record("r2", "o4", "d", "v4", {"r2": 2}),
            ],
        )
        self.assertEqual(
            payload["edges"],
            [
                edge("r1", "o1", "r1", "o3"),
                edge("r2", "o2", "r1", "o3"),
                edge("r2", "o2", "r2", "o4"),
            ],
        )
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 4, "edges": 3, "sharedAncestors": 1},
        )

    def test_shared_ancestors_need_two_distinct_roots(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "c", "v3", {"r1": 3}))
        # One root only: every predecessor precedes exactly one root.
        status, payload = store.get_causal_closure([("r1", "o3")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["summary"]["sharedAncestors"], 0)
        # Two roots on the same chain: o1 precedes both o2 and o3, o2
        # precedes only o3 (o2 does not precede itself).
        status, payload = store.get_causal_closure([("r1", "o2"), ("r1", "o3")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["summary"]["sharedAncestors"], 1)

    def test_root_can_be_a_shared_ancestor_of_later_roots(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "c", "v3", {"r1": 3}))
        # o1 is itself a root and strictly precedes the two other roots.
        status, payload = store.get_causal_closure(
            [("r1", "o1"), ("r1", "o2"), ("r1", "o3")]
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["summary"]["roots"], 3)
        self.assertEqual(payload["summary"]["operations"], 3)
        self.assertEqual(payload["summary"]["sharedAncestors"], 1)

    def test_edges_are_transitively_reduced(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "a", "2", {"r1": 2}))
        store.apply_operation("r1", operation("o3", "a", "3", {"r1": 3}))
        status, payload = store.get_causal_closure([("r1", "o3")])
        self.assertIs(status, HTTPStatus.OK)
        # o1 -> o3 is covered by o2 and never appears.
        self.assertEqual(
            payload["edges"],
            [edge("r1", "o1", "r1", "o2"), edge("r1", "o2", "r1", "o3")],
        )
        self.assertEqual(payload["summary"]["edges"], 2)

    def test_concurrent_roots_have_no_edges_and_no_shared_ancestors(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "b", "v2", {"r2": 1}))
        status, payload = store.get_causal_closure([("r1", "o1"), ("r2", "o2")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["edges"], [])
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 2, "edges": 0, "sharedAncestors": 0},
        )

    def test_records_committed_after_a_root_are_not_its_predecessors(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 2}))
        # A stale write committed later with a smaller clock: it enters the
        # log after o1, so it is never o1's predecessor even though o1's
        # clock dominates it.
        store.apply_operation("r2", operation("o2", "a", "old", {"r1": 1}))
        status, payload = store.get_causal_closure([("r1", "o1")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["operations"], [record("r1", "o1", "a", "v1", {"r1": 2})]
        )
        self.assertEqual(payload["edges"], [])
        # Rooting at o2 as well brings both records into the closure, but
        # still no edge: o2 was committed after o1.
        status, payload = store.get_causal_closure([("r1", "o1"), ("r2", "o2")])
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["summary"]["operations"], 2)
        self.assertEqual(payload["edges"], [])

    def test_read_is_byte_stable_and_does_not_move_metrics(self) -> None:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        store.apply_operation("r1", operation("o2", "b", "v2", {"r1": 2}))
        before = store.get_metrics()
        first = store.get_causal_closure([("r1", "o2")])
        second = store.get_causal_closure([("r1", "o2")])
        self.assertEqual(first, second)
        self.assertEqual(store.get_metrics(), before)


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
            conn.request(
                method, path, body=body, headers={"Content-Type": "application/json"}
            )
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

    def closure(self, operations: list, query: str = ""):
        return self.request("POST", f"{CLOSURE_PATH}{query}", {"operations": operations})

    def closure_raw(self, raw: bytes, query: str = ""):
        return self.request("POST", f"{CLOSURE_PATH}{query}", raw)

    def seed(self) -> None:
        self.post_operation("r1", operation("o1", "a", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "b", "v2", {"r2": 1}))
        self.post_operation("r1", operation("o3", "c", "v3", {"r1": 2, "r2": 1}))
        self.post_operation("r2", operation("o4", "d", "v4", {"r2": 2}))

    def test_round_trip_shape_and_encoding(self) -> None:
        self.seed()
        status, payload, headers, raw = self.closure(
            [identity("r1", "o3"), identity("r2", "o4")]
        )
        self.assertEqual(status, 200)
        # Top-level fields are exactly roots, operations, edges, summary,
        # in that order, compact JSON with one trailing newline.
        self.assertEqual(list(payload), ["roots", "operations", "edges", "summary"])
        self.assertEqual(
            list(payload["summary"]),
            ["roots", "operations", "edges", "sharedAncestors"],
        )
        self.assertEqual(
            payload["roots"], [identity("r1", "o3"), identity("r2", "o4")]
        )
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2", "o3", "o4"],
        )
        self.assertEqual(
            payload["edges"],
            [
                edge("r1", "o1", "r1", "o3"),
                edge("r2", "o2", "r1", "o3"),
                edge("r2", "o2", "r2", "o4"),
            ],
        )
        self.assertEqual(
            payload["summary"],
            {"roots": 2, "operations": 4, "edges": 3, "sharedAncestors": 1},
        )
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertEqual(
            raw,
            json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n",
        )
        self.assertLess(raw.index(b'"roots"'), raw.index(b'"operations"'))
        self.assertLess(raw.index(b'"operations"'), raw.index(b'"edges"'))
        self.assertLess(raw.index(b'"edges"'), raw.index(b'"summary"'))
        self.assertEqual(int(headers.get("Content-Length")), len(raw))

    def test_same_request_is_byte_identical(self) -> None:
        self.seed()
        body = {"operations": [identity("r1", "o3"), identity("r2", "o2")]}
        _, _, _, first = self.request("POST", CLOSURE_PATH, body)
        _, _, _, second = self.request("POST", CLOSURE_PATH, body)
        self.assertEqual(first, second)

    def test_roots_keep_request_order(self) -> None:
        self.seed()
        status, payload, _, _ = self.closure(
            [identity("r2", "o4"), identity("r1", "o1"), identity("r1", "o3")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["roots"],
            [identity("r2", "o4"), identity("r1", "o1"), identity("r1", "o3")],
        )
        # The closure itself stays in global commit order.
        self.assertEqual(
            [o["operation"]["operationId"] for o in payload["operations"]],
            ["o1", "o2", "o3", "o4"],
        )

    def test_single_root_without_predecessors(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, _, _ = self.closure([identity("r1", "o1")])
        self.assertEqual(status, 200)
        self.assertEqual(payload["roots"], [identity("r1", "o1")])
        self.assertEqual(len(payload["operations"]), 1)
        self.assertEqual(payload["edges"], [])
        self.assertEqual(
            payload["summary"],
            {"roots": 1, "operations": 1, "edges": 0, "sharedAncestors": 0},
        )

    def test_hundred_identity_request(self) -> None:
        for i in range(100):
            self.post_operation(
                "r1", operation(f"o{i}", "k", f"v{i}", {"r1": i + 1})
            )
        status, payload, _, _ = self.closure(
            [identity("r1", f"o{i}") for i in range(100)]
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["roots"], 100)
        self.assertEqual(payload["summary"]["operations"], 100)
        self.assertEqual(payload["summary"]["edges"], 99)
        self.assertEqual(payload["summary"]["sharedAncestors"], 98)

    def test_unknown_identity_is_404(self) -> None:
        self.seed()
        for operations in (
            [identity("r9", "o9")],
            [identity("r1", "o3"), identity("r9", "o9")],
            [identity("r1", "o9")],
        ):
            with self.subTest(operations=operations):
                status, payload, _, _ = self.closure(operations)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_bodies_are_400(self) -> None:
        self.seed()
        for raw in (
            b"",
            b"   ",
            b"{not json",
            b"{}",
            b"[]",
            b"null",
            b'{"operations":[]}',
            b'{"operations":"o1"}',
            b'{"operations":null}',
            b'{"operations":[null]}',
            b'{"operations":[{}]}',
            b'{"operations":[{"replicaId":"r1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1","x":1}]}',
            b'{"operations":[{"replicaId":"","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":""}]}',
            b'{"operations":[{"replicaId":1,"operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"}],"x":1}',
            b'{"operations":[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r1","operationId":"o1"}]}',
            b'{"operations":[{"replicaId":"r1","replicaId":"r2","operationId":"o1"}]}',
            b'{"operations":[],"operations":[]}',
            b'{"operations":['
            + b",".join(b'{"replicaId":"r1","operationId":"o%d"}' % i for i in range(101))
            + b"]}",
        ):
            with self.subTest(raw=raw):
                status, payload, _, _ = self.closure_raw(raw)
                self.assertEqual(status, 400, raw)
                self.assertEqual(payload, {"error": "invalid_request"}, raw)

    def test_invalid_utf8_is_400(self) -> None:
        status, payload, _, _ = self.closure_raw(
            b'{"operations":[{"replicaId":"r1","operationId":"\xff"}]}'
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_any_query_parameter_is_400(self) -> None:
        self.seed()
        for query in ("?x=1", "?x=1&x=2", "?x=", "?x", "?=1", "?x&y="):
            with self.subTest(query=query):
                status, payload, _, _ = self.closure([identity("r1", "o3")], query)
                self.assertEqual(status, 400, query)
                self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_path_shape_mismatches_are_404_and_precede_body_checks(self) -> None:
        self.seed()
        for path in (
            "/v1/causal/closure/",
            "/v1/causal/closure/extra",
            "/v1/causal",
            "/v1/causal//closure",
            "/v1//causal/closure",
        ):
            with self.subTest(path=path):
                # Even an invalid body and a query parameter stay 404: the
                # route-shape check runs first.
                status, payload, _, _ = self.request(
                    "POST", f"{path}?x=1", b"{not json"
                )
                self.assertEqual(status, 404, path)
                self.assertEqual(payload, {"error": "not_found"}, path)

    def test_get_on_the_path_is_404(self) -> None:
        status, payload, _, _ = self.request("GET", CLOSURE_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_content_length_is_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", CLOSURE_PATH)
        conn.endheaders(b'{"operations":[]}')
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_declared_length_over_the_limit_is_413(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            CLOSURE_PATH,
            body=b"x" * 10,
            headers={"Content-Length": str(1024 * 1024 + 1)},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 413)
        self.assertEqual(payload, {"error": "payload_too_large"})

    def test_error_requests_do_not_change_business_state(self) -> None:
        self.seed()
        _, before, _, _ = self.request("GET", "/v1/states/a")
        for raw in (b"{not json", b'{"operations":[]}', b'{"operations":[{}]}'):
            self.closure_raw(raw)
        self.closure([identity("r9", "o9")])
        _, after, _, _ = self.request("GET", "/v1/states/a")
        self.assertEqual(before, after)


class HttpCausalClosureAuthTests(unittest.TestCase):
    """Scope-policy authentication: read or admin may call, write may not."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            auth_scopes={
                "read-token": frozenset({"read"}),
                "write-token": frozenset({"write"}),
                "admin-token": frozenset({"read", "write", "admin"}),
            },
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

    def request(self, method: str, path: str, body: object = None, token: str = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn.request(
            method, path, body=json.dumps(body) if body is not None else None,
            headers=headers,
        )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_read_and_admin_may_query_write_may_not(self) -> None:
        status, _ = self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
            token="write-token",
        )
        self.assertEqual(status, 201)
        body = {"operations": [identity("r1", "o1")]}
        # No credential at all is 401.
        status, payload = self.request("POST", CLOSURE_PATH, body)
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        # A write-only token is 403.
        status, payload = self.request("POST", CLOSURE_PATH, body, token="write-token")
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        # Read and admin both reach the report.
        for token in ("read-token", "admin-token"):
            with self.subTest(token=token):
                status, payload = self.request("POST", CLOSURE_PATH, body, token=token)
                self.assertEqual(status, 200)
                self.assertEqual(payload["summary"]["roots"], 1)
        # /health stays anonymous.
        status, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)


class HttpCausalClosurePersistenceTests(unittest.TestCase):
    """The same request answers identically across a restart."""

    def test_restart_preserves_the_report(self) -> None:
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
                        conn.request(
                            method,
                            path,
                            body=json.dumps(body),
                            headers={"Content-Type": "application/json"},
                        )
                        response = conn.getresponse()
                        results.append((response.status, response.read()))
                        conn.close()
                    return results
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            query = {
                "operations": [identity("r1", "o3"), identity("r2", "o2")],
            }
            first = serve_once(
                [
                    ("POST", "/v1/replicas/r1/operations",
                     operation("o1", "a", "v1", {"r1": 1})),
                    ("POST", "/v1/replicas/r2/operations",
                     operation("o2", "b", "v2", {"r2": 1})),
                    ("POST", "/v1/replicas/r1/operations",
                     operation("o3", "c", "v3", {"r1": 2, "r2": 1})),
                    ("POST", CLOSURE_PATH, query),
                ]
            )
            self.assertEqual([status for status, _ in first], [201, 201, 201, 200])
            second = serve_once([("POST", CLOSURE_PATH, query)])
            # Identical bytes, including the trailing newline.
            self.assertEqual(second[0], first[3])

    def test_query_never_touches_the_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "state.json"
            server = SemanticStateServer(
                ("127.0.0.1", 0), RequestHandler, data_file=str(data_file)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                def call(method, path, body=None):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request(
                        method,
                        path,
                        body=json.dumps(body) if body is not None else None,
                        headers={"Content-Type": "application/json"},
                    )
                    response = conn.getresponse()
                    result = (response.status, response.read())
                    conn.close()
                    return result

                self.assertEqual(
                    call(
                        "POST", "/v1/replicas/r1/operations",
                        operation("o1", "k", "v", {"r1": 1}),
                    )[0],
                    201,
                )
                before = data_file.read_bytes()
                before_listing = sorted(os.listdir(tmp))
                status, _ = call(
                    "POST", CLOSURE_PATH, {"operations": [identity("r1", "o1")]}
                )
                self.assertEqual(status, 200)
                status, _ = call(
                    "POST", CLOSURE_PATH, {"operations": [identity("r9", "o9")]}
                )
                self.assertEqual(status, 404)
                self.assertEqual(data_file.read_bytes(), before)
                self.assertEqual(sorted(os.listdir(tmp)), before_listing)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
