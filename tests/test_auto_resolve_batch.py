"""HTTP, validation, atomicity, and persistence tests for batch automatic
resolution.

The endpoint is::

    POST /v1/resolve/auto/batch

It accepts 1-100 automatic conflict resolutions in one JSON object, each
carrying its own target key, replica id, operation id, clock, and policy,
processes them in request order, and commits every new resolution as one
indivisible unit (one durable write). Everything here goes through the real
HTTP entry point (``SemanticStateServer`` + a request thread); only the
Python standard library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine import server as server_module
from semantic_state_engine.server import (
    MAX_BODY_BYTES,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_file,
    load_data_file_full,
    parse_auto_resolve_batch,
)

BATCH_PATH = "/v1/resolve/auto/batch"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def entry(
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


def batch(*entries: dict) -> dict:
    return {"resolutions": list(entries)}


class ParseAutoResolveBatchTests(unittest.TestCase):
    def test_valid_batch_is_normalized_in_order(self) -> None:
        payload = parse_auto_resolve_batch(
            json.dumps(
                batch(
                    entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
                    entry("m", "r4", "fix-m", {"r1": 2, "r4": 1}, "highest_identity"),
                )
            )
        )
        self.assertEqual(
            payload,
            [
                {
                    "key": "k",
                    "replicaId": "r3",
                    "operationId": "fix-k",
                    "clock": {"r1": 1, "r2": 1, "r3": 1},
                    "policy": "lowest_identity",
                },
                {
                    "key": "m",
                    "replicaId": "r4",
                    "operationId": "fix-m",
                    "clock": {"r1": 2, "r4": 1},
                    "policy": "highest_identity",
                },
            ],
        )

    def test_rejects_malformed_and_wrong_shapes(self) -> None:
        valid = entry("k")
        bad_bodies = [
            b"{not json",
            [],
            "text",
            {},
            {"extra": 1},
            {"resolutions": []},
            {"resolutions": "nope"},
            {"resolutions": {}},
            {"resolutions": [valid], "extra": 1},
            {"resolutions": [valid] * 101},
            {"resolutions": [b"x"]},
            {"resolutions": [{}]},
            {"resolutions": [["k"]]},
            {"resolutions": [{k: v for k, v in valid.items() if k != "key"}]},
            {"resolutions": [{k: v for k, v in valid.items() if k != "policy"}]},
            {"resolutions": [dict(valid, extra=1)]},
            {"resolutions": [dict(valid, key="")]},
            {"resolutions": [dict(valid, key=7)]},
            {"resolutions": [dict(valid, replicaId="")]},
            {"resolutions": [dict(valid, replicaId=4)]},
            {"resolutions": [dict(valid, operationId="")]},
            {"resolutions": [dict(valid, operationId=4)]},
            {"resolutions": [dict(valid, policy="")]},
            {"resolutions": [dict(valid, policy="lowest")]},
            {"resolutions": [dict(valid, policy="Lowest_Identity")]},
            {"resolutions": [dict(valid, policy=42)]},
            {"resolutions": [dict(valid, clock={})]},
            {"resolutions": [dict(valid, clock={"r1": 1, "r2": 1})]},
            {"resolutions": [dict(valid, clock={"r1": -1, "r2": 1, "r3": 1})]},
            {"resolutions": [dict(valid, clock={"r1": True, "r2": 1, "r3": 1})]},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(body)

    def test_rejects_duplicate_target_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(
                batch(
                    entry("k", "r3", "fix-1"),
                    entry("k", "r4", "fix-2", {"r1": 1, "r2": 1, "r4": 1}),
                )
            )

    def test_rejects_duplicate_identities(self) -> None:
        with self.assertRaises(ValueError):
            parse_auto_resolve_batch(
                batch(
                    entry("k", "r3", "fix-1"),
                    entry("m", "r3", "fix-1", {"r1": 1, "r2": 1, "r3": 2}),
                )
            )

    def test_rejects_float_and_signed_zero_clock_ticks(self) -> None:
        for tick in (1.0, 1.5, -0.0):
            with self.subTest(tick=tick):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(
                        batch(entry("k", "r3", "fix-1", {"r1": tick, "r2": 1, "r3": 1}))
                    )

    def test_rejects_non_finite_json_literals(self) -> None:
        # Python's json module accepts NaN/Infinity by default; the parser
        # must still refuse them wherever they appear.
        raw_documents = [
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":NaN},"policy":"lowest_identity"}]}',
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":Infinity},"policy":"lowest_identity"}]}',
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":-Infinity},"policy":"lowest_identity"}]}',
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":1},"policy":NaN}]}',
            '{"resolutions":NaN}',
        ]
        for raw in raw_documents:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_auto_resolve_batch(raw.encode("utf-8"))

    def test_exactly_one_and_one_hundred_entries_are_accepted(self) -> None:
        parse_auto_resolve_batch(batch(entry("k")))
        entries = [
            entry(f"k{i:03d}", "r3", f"fix-{i:03d}", {"r1": 1, "r2": 1, "r3": i + 1})
            for i in range(100)
        ]
        self.assertEqual(len(parse_auto_resolve_batch(batch(*entries))), 100)


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

    def post_raw(self, path: str, body: bytes, headers: dict | None = None) -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            path,
            body=body,
            headers=headers or {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        if body is None:
            status, raw, _ = self.post_raw_bytes(method, path, None)
        elif isinstance(body, (bytes, str)):
            data = body.encode("utf-8") if isinstance(body, str) else body
            status, raw, _ = self.post_raw_bytes(method, path, data)
        else:
            status, raw, _ = self.post_raw_bytes(
                method, path, json.dumps(body).encode("utf-8")
            )
        payload = json.loads(raw.decode("utf-8")) if raw else None
        return status, payload

    def post_raw_bytes(
        self, method: str, path: str, body: bytes | None
    ) -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, raw, response_headers

    def post_batch(self, document: object) -> tuple[int, object]:
        return self.request("POST", BATCH_PATH, document)

    def post_batch_raw(self, document: dict) -> tuple[int, bytes, dict]:
        return self.post_raw(BATCH_PATH, json.dumps(document).encode("utf-8"))

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def get_state(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/states/{key}")

    def get_sync(self, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/operations{query}")

    def get_audit(self, key: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/audit/keys/{key}/operations")

    def get_archive(self, replica: str, operation_id: str) -> tuple[int, object]:
        return self.request("GET", f"/v1/replicas/{replica}/operations/{operation_id}")

    def get_metrics(self) -> tuple[int, object]:
        return self.request("GET", "/v1/metrics")

    def seed_conflict(self, key: str, value_a: str = "v1", value_b: str = "v2") -> None:
        # Operation identities are global, so the two seed writes derive
        # their ids from the key.
        self.assertEqual(
            self.post_operation("r1", operation(f"o1-{key}", key, value_a, {"r1": 1}))[0],
            201,
        )
        self.assertEqual(
            self.post_operation("r2", operation(f"o2-{key}", key, value_b, {"r2": 1}))[0],
            201,
        )
        status, state = self.get_state(key)
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")


class BatchHappyPathTests(HttpServerTestCase):
    def test_batch_resolves_each_key_by_its_policy(self) -> None:
        self.seed_conflict("k", "v1", "v2")
        self.seed_conflict("m", "w1", "w2")
        document = batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}, "lowest_identity"),
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 201)
        self.assertEqual(
            payload,
            {
                "status": "created",
                "accepted": 2,
                "replayed": 0,
                "results": [
                    {
                        "key": "k",
                        "replicaId": "r3",
                        "operationId": "fix-k",
                        "value": "v1",
                        "policy": "lowest_identity",
                    },
                    {
                        "key": "m",
                        "replicaId": "r3",
                        "operationId": "fix-m",
                        "value": "w2",
                        "policy": "highest_identity",
                    },
                ],
            },
        )
        _, state_k = self.get_state("k")
        self.assertEqual(state_k["status"], "resolved")
        self.assertEqual(state_k["value"], "v1")
        _, state_m = self.get_state("m")
        self.assertEqual(state_m["status"], "resolved")
        self.assertEqual(state_m["value"], "w2")

    def test_response_is_compact_json_terminated_by_a_newline(self) -> None:
        self.seed_conflict("k")
        status, raw, headers = self.post_batch_raw(
            batch(entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw[-2:], b'}\n')
        # Exactly one trailing newline and no compact-separator whitespace.
        body = raw[:-1]
        self.assertFalse(body.endswith(b"\n"))
        self.assertNotIn(b", ", body)
        self.assertNotIn(b": ", body)
        decoded = json.loads(body.decode("utf-8"))
        # Every counter is an integer.
        self.assertIsInstance(decoded["accepted"], int)
        self.assertIsInstance(decoded["replayed"], int)
        self.assertEqual(int(headers["Content-Length"]), len(raw))

    def test_single_entry_batch_is_accepted(self) -> None:
        self.seed_conflict("k")
        status, payload = self.post_batch(
            batch(entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 0)
        self.assertEqual(payload["results"][0]["value"], "v1")

    def test_results_stay_in_request_order_regardless_of_key_order(self) -> None:
        for key in ("a", "b", "c"):
            self.seed_conflict(key, f"{key}-1", f"{key}-2")
        document = batch(
            entry("c", "r9", "fix-c", {"r1": 1, "r2": 1, "r9": 3}, "highest_identity"),
            entry("a", "r8", "fix-a", {"r1": 1, "r2": 1, "r8": 1}),
            entry("b", "r7", "fix-b", {"r1": 1, "r2": 1, "r7": 2}),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 201)
        self.assertEqual([r["key"] for r in payload["results"]], ["c", "a", "b"])
        self.assertEqual(
            [r["value"] for r in payload["results"]], ["c-2", "a-1", "b-1"]
        )

    def test_exactly_100_entries_commit_together(self) -> None:
        for i in range(100):
            self.seed_conflict(f"k{i:03d}")
        entries = [
            entry(
                f"k{i:03d}",
                "r3",
                f"fix-{i:03d}",
                {"r1": 1, "r2": 1, "r3": i + 1},
            )
            for i in range(100)
        ]
        status, payload = self.post_batch(batch(*entries))
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 100)
        self.assertEqual(len(payload["results"]), 100)
        _, metrics = self.get_metrics()
        # 200 seeded writes + 100 repairs.
        self.assertEqual(metrics["acceptedOperations"], 300)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["resolvedKeys"], 100)


class BatchReplayTests(HttpServerTestCase):
    def two_key_batch(self) -> dict:
        return batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}, "lowest_identity"),
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
        )

    def test_all_replay_is_200_and_appends_nothing(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = self.two_key_batch()
        self.assertEqual(self.post_batch(document)[0], 201)
        _, page_before = self.get_sync()
        self.assertEqual(len(page_before["operations"]), 6)

        status, raw, _ = self.post_batch_raw(document)
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual([r["value"] for r in payload["results"]], ["v1", "w2"])
        self.assertTrue(raw.endswith(b"\n"))
        _, page_after = self.get_sync()
        self.assertEqual(page_after["operations"], page_before["operations"])

    def test_mixed_new_and_replayed_entries_is_201(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = self.two_key_batch()
        self.assertEqual(self.post_batch(document)[0], 201)
        self.seed_conflict("n", "a", "b")
        mixed = batch(
            document["resolutions"][0],
            entry("n", "r3", "fix-n", {"r1": 1, "r2": 1, "r3": 9}),
            document["resolutions"][1],
        )
        status, payload = self.post_batch(mixed)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual([r["key"] for r in payload["results"]], ["k", "n", "m"])
        self.assertEqual([r["value"] for r in payload["results"]], ["v1", "a", "w2"])

    def test_replay_after_keys_moved_on_reports_original_values(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = self.two_key_batch()
        self.assertEqual(self.post_batch(document)[0], 201)
        # Key k gets a new conflict after the repair.
        self.assertEqual(
            self.post_operation(
                "r2", operation("o3", "k", "later", {"r1": 1, "r2": 2, "r3": 0})
            )[0],
            201,
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["value"] for r in payload["results"]], ["v1", "w2"]
        )
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1-k", "o2-k", "o1-m", "o2-m", "fix-k", "fix-m", "o3"],
        )

    def test_single_endpoint_and_batch_share_one_identity_space(self) -> None:
        self.seed_conflict("k")
        document = batch(entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}))
        self.assertEqual(self.post_batch(document)[0], 201)
        # The same binding replayed through the single-key endpoint is a 200.
        single_body = {k: v for k, v in document["resolutions"][0].items() if k != "key"}
        status, payload = self.request("POST", "/v1/states/k/resolve/auto", single_body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["value"], "v1")


class BatchValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        valid = entry("k")
        other = entry("m", "r4", "fix-m", {"r1": 1, "r2": 1, "r4": 1})
        bad_bodies = [
            b"{oops",
            [],
            {},
            {"extra": 1},
            {"resolutions": []},
            {"resolutions": [valid], "extra": 1},
            {"resolutions": "nope"},
            {"resolutions": [valid] * 101},
            {"resolutions": [{}]},
            {"resolutions": [dict(valid, extra=1)]},
            {"resolutions": [dict(valid, key="")]},
            {"resolutions": [dict(valid, policy="lowest")]},
            {"resolutions": [dict(valid, clock={"r1": 1, "r2": 1})]},
            {"resolutions": [dict(valid, clock={"r1": -1, "r2": 1, "r3": 1})]},
            {"resolutions": [dict(valid, clock={"r1": True, "r2": 1, "r3": 1})]},
            # Duplicate target key or identity, even with distinct other fields.
            {"resolutions": [valid, dict(other, key="k")]},
            {"resolutions": [valid, dict(other, replicaId="r3", operationId="fix-1")]},
            # Float, fractional, and signed-zero clock ticks are illegal input.
            {"resolutions": [dict(valid, operationId="f2", clock={"r1": 1.0, "r2": 1, "r3": 1})]},
            {"resolutions": [dict(valid, operationId="f3", clock={"r1": -0.0, "r2": 1, "r3": 1})]},
        ]
        for document in bad_bodies:
            with self.subTest(document=document):
                status, payload = self.post_batch(document)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})
        _, state = self.get_state("k")
        self.assertEqual(state["status"], "conflict")

    def test_non_finite_json_literals_are_400(self) -> None:
        raw_documents = [
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":NaN},"policy":"lowest_identity"}]}',
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":Infinity},"policy":"lowest_identity"}]}',
            '{"resolutions":[{"key":"k","replicaId":"r3","operationId":"f",'
            '"clock":{"r1":1,"r2":1,"r3":1},"policy":NaN}]}',
        ]
        for raw in raw_documents:
            with self.subTest(raw=raw):
                status, raw_body, _ = self.post_raw(BATCH_PATH, raw.encode("utf-8"))
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(raw_body), {"error": "invalid_request"})

    def test_extra_path_segments_are_404(self) -> None:
        for path in (
            "/v1/resolve/auto/batch/extra",
            "/v1/resolve/auto",
            "/v1/resolve/auto/other",
            "/v1/resolve/batch",
            "/v1/resolve",
        ):
            status, payload = self.request("POST", path, {"resolutions": []})
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_trailing_slash_matches_the_route_like_the_other_post_endpoints(self) -> None:
        # A trailing empty segment collapses just as it does for the
        # existing routes (e.g. /v1/metrics/), so the request reaches the
        # batch handler and fails its normal body validation.
        status, payload = self.request("POST", "/v1/resolve/auto/batch/", {"resolutions": []})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_get_on_batch_route_is_404(self) -> None:
        status, payload = self.request("GET", BATCH_PATH)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_query_string_does_not_change_handling(self) -> None:
        self.seed_conflict("k")
        status, payload = self.request(
            "POST", f"{BATCH_PATH}?x=1", batch(entry("k", "r3", "fix-k"))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["results"][0]["value"], "v1")


class BatchConflictTests(HttpServerTestCase):
    def test_operation_conflict_leaves_whole_batch_unchanged(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        self.assertEqual(
            self.post_batch(
                batch(entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}))
            )[0],
            201,
        )
        # A new entry for m is placed before the conflicting replay: neither
        # may commit.
        document = batch(
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}, "highest_identity"),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state_m = self.get_state("m")
        self.assertEqual(state_m["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1-k", "o2-k", "o1-m", "o2-m", "fix-k"],
        )
        # The rejected fix-m identity is still free to commit afterwards.
        status, payload = self.post_batch(
            batch(entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"))
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["results"][0]["value"], "w2")

    def test_identity_known_without_policy_binding_conflicts(self) -> None:
        self.seed_conflict("k")
        # A plain write claims the identity first.
        self.assertEqual(
            self.post_operation(
                "r3", operation("fix-k", "k", "v1", {"r1": 1, "r2": 1, "r3": 1})
            )[0],
            201,
        )
        # Rebuild a conflict on a different key so the batch has a valid new
        # entry that must roll back with the conflicting one.
        self.seed_conflict("m", "w1", "w2")
        document = batch(
            entry("m", "r4", "fix-m", {"r1": 1, "r2": 1, "r4": 1}),
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state_m = self.get_state("m")
        self.assertEqual(state_m["status"], "conflict")

    def test_resolution_conflict_for_missing_key_leaves_batch_unchanged(self) -> None:
        self.seed_conflict("k")
        document = batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
            entry("ghost", "r4", "fix-g", {"r4": 1}),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state_k = self.get_state("k")
        self.assertEqual(state_k["status"], "conflict")
        _, page = self.get_sync()
        self.assertEqual(len(page["operations"]), 2)

    def test_resolution_conflict_for_agreed_key_leaves_batch_unchanged(self) -> None:
        self.seed_conflict("k")
        self.post_operation("r9", operation("o9", "m", "only", {"r9": 1}))
        document = batch(
            entry("m", "r3", "fix-m", {"r9": 2, "r3": 1}),
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 2}),
        )
        status, payload = self.post_batch(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "resolution_conflict"})
        _, state_k = self.get_state("k")
        self.assertEqual(state_k["status"], "conflict")

    def test_clock_not_dominating_is_resolution_conflict(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        for policy in ("lowest_identity", "highest_identity"):
            document = batch(
                entry("m", "r4", "fix-m", {"r1": 1, "r2": 1, "r4": 1}, policy),
                entry("k", "r3", "fix-k", {"r1": 1, "r3": 1}, policy),
            )
            status, payload = self.post_batch(document)
            self.assertEqual(status, 409, policy)
            self.assertEqual(payload, {"error": "resolution_conflict"}, policy)
        _, state_k = self.get_state("k")
        self.assertEqual(state_k["status"], "conflict")
        _, state_m = self.get_state("m")
        self.assertEqual(state_m["status"], "conflict")

    def test_failed_batch_appends_no_records_for_any_entry(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
            entry("missing", "r4", "fix-x", {"r4": 1}),
            entry("m", "r5", "fix-m", {"r1": 1, "r2": 1, "r5": 1}),
        )
        self.assertEqual(self.post_batch(document)[0], 409)
        _, page = self.get_sync()
        self.assertEqual(
            [e["operation"]["operationId"] for e in page["operations"]],
            ["o1-k", "o2-k", "o1-m", "o2-m"],
        )
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 4)
        self.assertEqual(metrics["conflictKeys"], 2)
        # Audit streams of both target keys are untouched.
        _, audit_k = self.get_audit("k")
        _, audit_m = self.get_audit("m")
        self.assertEqual(
            [e["operation"]["operationId"] for e in audit_k["operations"]],
            ["o1-k", "o2-k"],
        )
        self.assertEqual(
            [e["operation"]["operationId"] for e in audit_m["operations"]],
            ["o1-m", "o2-m"],
        )


class BatchDownstreamTests(HttpServerTestCase):
    def test_committed_repairs_reach_sync_audit_archive_metrics_and_digests(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
        )
        self.assertEqual(self.post_batch(document)[0], 201)
        _, page = self.get_sync()
        self.assertEqual(
            [(e["replicaId"], e["operation"]["operationId"]) for e in page["operations"]],
            [("r1", "o1-k"), ("r2", "o2-k"), ("r1", "o1-m"), ("r2", "o2-m"),
             ("r3", "fix-k"), ("r3", "fix-m")],
        )
        # The two repairs commit as one contiguous segment.
        _, audit_k = self.get_audit("k")
        self.assertEqual(
            [e["operation"]["operationId"] for e in audit_k["operations"]],
            ["o1-k", "o2-k", "fix-k"],
        )
        self.assertEqual(audit_k["operations"][2]["operation"]["value"], "v1")
        _, audit_m = self.get_audit("m")
        self.assertEqual(audit_m["operations"][2]["operation"]["value"], "w2")
        status, archived = self.get_archive("r3", "fix-m")
        self.assertEqual(status, 200)
        self.assertEqual(archived["operation"]["value"], "w2")
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 6)
        self.assertEqual(metrics["resolvedKeys"], 2)
        self.assertEqual(metrics["conflictKeys"], 0)
        self.assertEqual(metrics["replicas"], 3)
        _, verification = self.request("GET", "/v1/verification/digest")
        self.assertEqual(verification["keys"], 2)
        self.assertEqual(verification["candidateVersions"], 2)
        _, digest_k = self.request("GET", "/v1/audit/keys/k/digest")
        self.assertEqual(digest_k["operations"], 3)

    def test_imported_batch_resolutions_resolve_the_same_conflicts(self) -> None:
        self.seed_conflict("k")
        self.seed_conflict("m", "w1", "w2")
        document = batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
        )
        self.assertEqual(self.post_batch(document)[0], 201)
        _, page = self.get_sync()
        other = StateStore()
        status, accepted, _ = other.import_operations(
            [(e["replicaId"], e["operation"]) for e in page["operations"]]
        )
        self.assertIs(status, HTTPStatus.CREATED)
        self.assertEqual(accepted, 6)
        for key, value in (("k", "v1"), ("m", "w2")):
            status, state = other.get_state(key)
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(state["status"], "resolved")
            self.assertEqual(state["value"], value)

    def test_imported_identities_carry_no_policy_binding(self) -> None:
        self.seed_conflict("k")
        document = batch(entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}))
        self.assertEqual(self.post_batch(document)[0], 201)
        _, page = self.get_sync()
        self.server.store = type(self.server.store)()
        status, _ = self.request("POST", "/v1/sync/operations", {"operations": page["operations"]})
        self.assertEqual(status, 201)
        status, payload = self.post_batch(document)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})


class BatchConcurrencyTests(HttpServerTestCase):
    def test_disjoint_batches_commit_as_serializable_segments(self) -> None:
        for i in range(8):
            self.seed_conflict(f"k{i}")
        documents = [
            batch(
                entry(
                    f"k{i}",
                    "r3",
                    f"fix-{i}",
                    {"r1": 1, "r2": 1, "r3": i + 1},
                    "lowest_identity" if i % 2 == 0 else "highest_identity",
                )
            )
            for i in range(8)
        ]
        results: list[tuple[int, object] | None] = [None] * 8

        def worker(index: int, document: dict) -> None:
            results[index] = self.post_batch(document)

        threads = [
            threading.Thread(target=worker, args=(index, document))
            for index, document in enumerate(documents)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(result is not None for result in results))
        self.assertEqual(sorted(status for status, _ in results), [201] * 8)
        _, metrics = self.get_metrics()
        self.assertEqual(metrics["acceptedOperations"], 24)
        self.assertEqual(metrics["conflictKeys"], 0)


class BatchAuthAndLimitsTests(unittest.TestCase):
    """The batch route keeps the shared Content-Length/auth priority."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="secret-token"
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post_raw(self, headers: list, body: bytes | None) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", BATCH_PATH)
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    def test_unauthorized_request_is_401_without_reading_the_body(self) -> None:
        # A valid declared length but no body sent: the 401 must arrive
        # without the server waiting for body bytes.
        status, raw = self.post_raw([("Content-Length", "64")], None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_unauthorized_with_body_is_401(self) -> None:
        body = json.dumps(batch(entry("k"))).encode("utf-8")
        status, raw = self.post_raw(
            [("Content-Length", str(len(body))), ("Authorization", "Bearer wrong")],
            body,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_malformed_length_is_400_before_auth(self) -> None:
        status, raw = self.post_raw([("Content-Length", "abc")], b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw), {"error": "invalid_request"})

    def test_over_limit_is_413_before_auth_and_body(self) -> None:
        status, raw = self.post_raw(
            [("Content-Length", str(MAX_BODY_BYTES + 1))], b"not json"
        )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(raw), {"error": "payload_too_large"})


class PersistentBatchTestCase(unittest.TestCase):
    """Batch automatic resolution against a data-file-backed server."""

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

    def request(self, server: SemanticStateServer, method: str, path: str, body: object = None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body) if not isinstance(body, (bytes, str)) else body,
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def seed_conflicts(self, server: SemanticStateServer) -> None:
        for replica, op in (
            ("r1", operation("o1", "k", "v1", {"r1": 1})),
            ("r2", operation("o2", "k", "v2", {"r2": 1})),
            ("r1", operation("p1", "m", "w1", {"r1": 1})),
            ("r2", operation("p2", "m", "w2", {"r2": 1})),
        ):
            status, _ = self.request(
                server, "POST", f"/v1/replicas/{replica}/operations", op
            )
            self.assertEqual(status, 201)

    def two_key_batch(self) -> dict:
        return batch(
            entry("k", "r3", "fix-k", {"r1": 1, "r2": 1, "r3": 1}),
            entry("m", "r3", "fix-m", {"r1": 1, "r2": 1, "r3": 2}, "highest_identity"),
        )

    def test_batch_is_one_durable_commit_and_recovers(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server)
        document = self.two_key_batch()
        status, payload = self.request(server, "POST", BATCH_PATH, document)
        self.assertEqual(status, 201)
        self.assertEqual([r["value"] for r in payload["results"]], ["v1", "w2"])
        records = load_data_file(str(self.data_file))
        self.assertEqual(
            [(r, o["key"], o["operationId"], o["value"]) for r, o in records],
            [
                ("r1", "k", "o1", "v1"),
                ("r2", "k", "o2", "v2"),
                ("r1", "m", "p1", "w1"),
                ("r2", "m", "p2", "w2"),
                ("r3", "k", "fix-k", "v1"),
                ("r3", "m", "fix-m", "w2"),
            ],
        )
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(
            policies,
            {("r3", "fix-k"): "lowest_identity", ("r3", "fix-m"): "highest_identity"},
        )

        server.shutdown()
        server.server_close()
        server = self.start_server()
        for key, value in (("k", "v1"), ("m", "w2")):
            status, state = self.request(server, "GET", f"/v1/states/{key}")
            self.assertEqual(status, 200)
            self.assertEqual(state["status"], "resolved")
            self.assertEqual(state["value"], value)
        # A full replay after restart is still 200 and appends nothing.
        status, payload = self.request(server, "POST", BATCH_PATH, document)
        self.assertEqual(status, 200)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual(len(load_data_file(str(self.data_file))), 6)
        # A different binding under a known identity still conflicts.
        tampered = batch(
            dict(document["resolutions"][0], policy="highest_identity"),
        )
        status, payload = self.request(server, "POST", BATCH_PATH, tampered)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})

    def test_mixed_batch_after_restart_counts_replays(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server)
        document = self.two_key_batch()
        self.assertEqual(self.request(server, "POST", BATCH_PATH, document)[0], 201)

        server.shutdown()
        server.server_close()
        server = self.start_server()
        # A new conflict on n plus replays of the two recovered bindings.
        for replica, op in (
            ("r1", operation("q1", "n", "a", {"r1": 5})),
            ("r2", operation("q2", "n", "b", {"r2": 5})),
        ):
            self.assertEqual(
                self.request(server, "POST", f"/v1/replicas/{replica}/operations", op)[0],
                201,
            )
        mixed = batch(
            document["resolutions"][0],
            entry("n", "r3", "fix-n", {"r1": 5, "r2": 5, "r3": 1}),
            document["resolutions"][1],
        )
        status, payload = self.request(server, "POST", BATCH_PATH, mixed)
        self.assertEqual(status, 201)
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["replayed"], 2)
        self.assertEqual([r["key"] for r in payload["results"]], ["k", "n", "m"])

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(
                server, "POST", BATCH_PATH, self.two_key_batch()
            )
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
        self.assertEqual(self.data_file.read_bytes(), before)
        _, _, policies = load_data_file_full(str(self.data_file))
        self.assertEqual(policies, {})
        for key in ("k", "m"):
            _, state = self.request(server, "GET", f"/v1/states/{key}")
            self.assertEqual(state["status"], "conflict")
        _, page = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual(len(page["operations"]), 4)
        # The batch commits cleanly once persistence works again.
        status, payload = self.request(server, "POST", BATCH_PATH, self.two_key_batch())
        self.assertEqual(status, 201)
        self.assertEqual([r["value"] for r in payload["results"]], ["v1", "w2"])
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_pure_replay_needs_no_durable_write(self) -> None:
        server = self.start_server()
        self.seed_conflicts(server)
        document = self.two_key_batch()
        self.assertEqual(self.request(server, "POST", BATCH_PATH, document)[0], 201)
        before = self.data_file.read_bytes()
        with patch.object(
            StateStore, "_persist_locked", side_effect=server_module.PersistenceError("disk gone")
        ):
            status, payload = self.request(server, "POST", BATCH_PATH, document)
            self.assertEqual(status, 200)
            self.assertEqual(payload["replayed"], 2)
        self.assertEqual(self.data_file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
