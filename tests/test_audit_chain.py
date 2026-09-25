"""Tests for the global audit-chain endpoint::

    GET /v1/audit/log/chain?after=N&limit=N

It returns exactly four fields — ``entries`` (one page of chain links in
global commit order), ``nextCursor`` (the resume cursor), ``hasMore``
(whether further links remain), and ``head`` (the digest of the chain's
last link, invariant across pages) — computed from one snapshot under the
shared commit lock. Each link carries ``sequence`` (1-based), ``prevDigest``
(the previous link's digest, 64 ``"0"`` characters for the first link), and
``digest``: the lowercase hexadecimal SHA-256 of the previous digest and the
decimal sequence (both ASCII) concatenated with the single record's
canonical bytes (the per-key audit digest's record encoding). An empty log
reports an empty page and a head of 64 ``"0"`` characters.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is tested,
and against ``StateStore`` directly for snapshot and recovery semantics.
Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _audit_record_bytes,
)

CHAIN_FIELDS = {"entries", "nextCursor", "hasMore", "head"}
ENTRY_FIELDS = {"sequence", "prevDigest", "digest"}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

GENESIS = "0" * 64


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


def resolution(
    replica: str,
    operation_id: str,
    key: str,
    value: str,
    clock: dict,
    candidates: list,
) -> dict:
    return {
        "replicaId": replica,
        "operationId": operation_id,
        "value": value,
        "clock": clock,
        "candidates": candidates,
    }


def expected_chain(records: list[tuple[str, dict]]) -> list[dict]:
    """Compute the full expected chain for (replica, operation) pairs."""
    entries = []
    previous = GENESIS
    for index, (replica, op) in enumerate(records):
        sequence = index + 1
        digest = hashlib.sha256(
            previous.encode("ascii")
            + str(sequence).encode("ascii")
            + _audit_record_bytes(replica, op)
        ).hexdigest()
        entries.append({"sequence": sequence, "prevDigest": previous, "digest": digest})
        previous = digest
    return entries


class ChainLinkTests(unittest.TestCase):
    """The link digest format, pinned against literals."""

    def test_first_link_input_layout(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        expected_input = (
            GENESIS.encode("ascii")
            + b"1"
            + b'{"replicaId":"r1","operation":'
            b'{"operationId":"o1","key":"k","value":"v","clock":{"r1":1}}}'
        )
        entries = expected_chain([("r1", op)])
        self.assertEqual(
            entries[0]["digest"], hashlib.sha256(expected_input).hexdigest()
        )
        self.assertEqual(entries[0]["prevDigest"], GENESIS)
        self.assertEqual(entries[0]["sequence"], 1)

    def test_links_chain_through_previous_digest(self) -> None:
        records = [
            ("r1", operation("o1", "k", "a", {"r1": 1})),
            ("r2", operation("o2", "k", "b", {"r2": 1})),
        ]
        entries = expected_chain(records)
        self.assertEqual(entries[1]["prevDigest"], entries[0]["digest"])
        second_input = (
            entries[0]["digest"].encode("ascii")
            + b"2"
            + _audit_record_bytes("r2", records[1][1])
        )
        self.assertEqual(
            entries[1]["digest"], hashlib.sha256(second_input).hexdigest()
        )

    def test_record_bytes_match_key_audit_encoding(self) -> None:
        op = operation("o1", "clé", "a\nb→", {"z": 1, "a": 2, "r1": 3})
        self.assertEqual(
            _audit_record_bytes("r1", op),
            b'{"replicaId":"r1","operation":'
            b'{"operationId":"o1","key":"cl\xc3\xa9","value":"a\\u000ab\xe2\x86\x92",'
            b'"clock":{"a":2,"r1":3,"z":1}}}',
        )


class ChainStoreTests(unittest.TestCase):
    """Snapshot and paging semantics against StateStore directly."""

    def setUp(self) -> None:
        self.store = StateStore()

    def commit(self, replica: str, op: dict) -> None:
        status = self.store.apply_operation(replica, op)
        self.assertIn(status, (201, 200))

    def test_empty_log_reports_genesis_head(self) -> None:
        entries, cursor, has_more, head = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(entries, [])
        self.assertEqual(cursor, 0)
        self.assertIs(has_more, False)
        self.assertEqual(head, GENESIS)

    def test_after_equal_to_chain_length_is_an_empty_tail(self) -> None:
        self.commit("r1", operation("o1", "k", "v", {"r1": 1}))
        entries, cursor, has_more, head = self.store.get_audit_log_chain(1, 100)
        self.assertEqual(entries, [])
        self.assertEqual(cursor, 1)
        self.assertIs(has_more, False)
        self.assertNotEqual(head, GENESIS)

    def test_after_past_chain_length_raises(self) -> None:
        self.commit("r1", operation("o1", "k", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            self.store.get_audit_log_chain(2, 100)

    def test_pages_tile_the_full_chain_and_head_is_stable(self) -> None:
        records = [
            ("r1", operation(f"o{i}", "k", f"v{i}", {"r1": i}))
            for i in range(1, 8)
        ]
        for replica, op in records:
            self.commit(replica, op)
        full, cursor, has_more, head = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(full, expected_chain(records))
        self.assertEqual(cursor, 7)
        self.assertIs(has_more, False)
        self.assertEqual(head, full[-1]["digest"])

        seen: list[dict] = []
        after = 0
        while True:
            page, cursor, has_more, page_head = self.store.get_audit_log_chain(after, 3)
            self.assertEqual(page_head, head)
            seen.extend(page)
            after = cursor
            if not has_more:
                break
        self.assertEqual(seen, full)

    def test_replays_and_conflicts_never_enter_the_chain(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        self.commit("r1", op)
        before = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(self.store.apply_operation("r1", op), 200)  # replay
        # Conflicting identity: same identity, different content.
        self.assertEqual(
            self.store.apply_operation(
                "r1", operation("o1", "k", "other", {"r1": 1})
            ),
            409,
        )
        self.assertEqual(self.store.get_audit_log_chain(0, 100), before)

    def test_stale_writes_and_repairs_enter_the_chain(self) -> None:
        self.commit("r1", operation("o1", "k", "a", {"r1": 2, "r2": 1}))
        # Stale write: dominated clock, adds no candidate but is committed.
        self.commit("r2", operation("o2", "k", "b", {"r2": 1}))
        # A conflicting concurrent write, then a repair dominating both.
        self.commit("r3", operation("o3", "k", "c", {"r3": 5}))
        status, _ = self.store.apply_resolution(
            "k",
            resolution(
                "r4",
                "o4",
                "k",
                "fixed",
                {"r1": 2, "r2": 1, "r3": 5, "r4": 1},
                [
                    {"replicaId": "r1", "operationId": "o1"},
                    {"replicaId": "r3", "operationId": "o3"},
                ],
            ),
        )
        self.assertEqual(status, 201)
        entries, cursor, has_more, _ = self.store.get_audit_log_chain(0, 100)
        self.assertEqual([entry["sequence"] for entry in entries], [1, 2, 3, 4])
        self.assertEqual(cursor, 4)
        self.assertIs(has_more, False)

    def test_imported_batch_enters_the_chain_as_one_segment(self) -> None:
        records = [
            ("r1", operation("o1", "a", "x", {"r1": 1})),
            ("r2", operation("o2", "b", "y", {"r2": 1})),
        ]
        status, accepted, replayed = self.store.import_operations(records)
        self.assertEqual((status, accepted, replayed), (201, 2, 0))
        entries, _, _, head = self.store.get_audit_log_chain(0, 100)
        self.assertEqual(entries, expected_chain(records))
        self.assertEqual(head, entries[-1]["digest"])

    def test_recovery_rebuilds_the_identical_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = str(Path(directory) / "state.json")
            store = StateStore(data_file=data_file)
            records = [
                ("r1", operation("o1", "a", "x", {"r1": 1})),
                ("r2", operation("o2", "b", "y", {"r2": 1})),
                ("r1", operation("o3", "a", "z", {"r1": 2})),
            ]
            for replica, op in records:
                store.apply_operation(replica, op)
            before = store.get_audit_log_chain(0, 100)
            recovered = StateStore(data_file=data_file)
            self.assertEqual(recovered.get_audit_log_chain(0, 100), before)
            # Paged reads recover identically too.
            self.assertEqual(
                recovered.get_audit_log_chain(1, 1), store.get_audit_log_chain(1, 1)
            )


class ChainHttpTests(unittest.TestCase):
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
        payload = json.loads(raw.decode("utf-8")) if raw else None
        headers = response.getheaders()
        conn.close()
        return response.status, payload, raw, headers

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def chain(self, query: str = "") -> tuple[int, dict]:
        return self.request("GET", f"/v1/audit/log/chain{query}")

    def post_operation(self, replica: str, op: dict) -> tuple[int, dict]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_sync(self, body: object) -> tuple[int, dict]:
        return self.request("POST", "/v1/sync/operations", body)

    def post_resolve(self, key: str, body: object) -> tuple[int, dict]:
        return self.request("POST", f"/v1/states/{key}/resolve", body)

    def seed(self, count: int = 5) -> list[tuple[str, dict]]:
        records = [
            ("r1", operation(f"o{i}", f"k{i}", f"v{i}", {"r1": i}))
            for i in range(1, count + 1)
        ]
        for replica, op in records:
            status, _ = self.post_operation(replica, op)
            assert status == 201
        return records

    def test_empty_chain_shape_and_framing(self) -> None:
        status, payload, raw, headers = self.raw_request("GET", "/v1/audit/log/chain")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), CHAIN_FIELDS)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["head"], GENESIS)
        # Compact canonical JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertEqual(raw[:-1], json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_entries_link_in_commit_order(self) -> None:
        records = self.seed(3)
        status, payload = self.chain()
        self.assertEqual(status, 200)
        expected = expected_chain(records)
        self.assertEqual(payload["entries"], expected)
        self.assertEqual(payload["head"], expected[-1]["digest"])
        self.assertEqual(payload["nextCursor"], 3)
        self.assertIs(payload["hasMore"], False)
        for entry in payload["entries"]:
            self.assertEqual(set(entry), ENTRY_FIELDS)
            self.assertIs(type(entry["sequence"]), int)
            self.assertRegex(entry["prevDigest"], DIGEST_RE)
            self.assertRegex(entry["digest"], DIGEST_RE)

    def test_paging_walks_the_chain_with_stable_head(self) -> None:
        records = self.seed(5)
        expected = expected_chain(records)
        seen: list[dict] = []
        after = 0
        heads = set()
        while True:
            status, payload = self.chain(f"?after={after}&limit=2")
            self.assertEqual(status, 200)
            heads.add(payload["head"])
            seen.extend(payload["entries"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual(seen, expected)
        self.assertEqual(heads, {expected[-1]["digest"]})
        self.assertEqual(after, 5)

    def test_after_equal_to_chain_length_is_an_empty_page(self) -> None:
        self.seed(2)
        status, payload = self.chain("?after=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["nextCursor"], 2)
        self.assertIs(payload["hasMore"], False)
        self.assertRegex(payload["head"], DIGEST_RE)
        self.assertNotEqual(payload["head"], GENESIS)

    def test_after_past_chain_length_is_400(self) -> None:
        self.seed(2)
        status, payload = self.chain("?after=3")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_default_limit_is_100(self) -> None:
        self.seed(3)
        status, payload = self.chain("?after=1")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["entries"]), 2)
        self.assertEqual(payload["entries"][0]["sequence"], 2)

    def test_invalid_query_parameters_are_400(self) -> None:
        self.seed(1)
        bad_queries = [
            "?after=",
            "?after=-1",
            "?after=+1",
            "?after=1.0",
            "?after=%201",
            "?after=%D9%A1",  # non-ASCII digit
            "?after=0&after=1",
            "?limit=0",
            "?limit=101",
            "?limit=",
            "?limit=-5",
            "?limit=1&limit=2",
            "?after=0&unknown=1",
            "?x",
            "?x=",
            "?=1",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, payload = self.chain(query)
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shape_mismatches_are_404(self) -> None:
        bad_paths = [
            "/v1/audit/log",
            "/v1/audit/log/chain/extra",
            "/v1/audit/log/chain/",
            "/v1/audit/chain",
            "/v1/audit/log/unknown",
            "/v1/log/chain",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, payload = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(payload, {"error": "not_found"})

    def test_route_shape_check_precedes_query_check(self) -> None:
        status, payload = self.request("GET", "/v1/audit/log/chain/extra?after=x")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload = self.request("GET", "/v1/audit/log/?after=x")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_stale_writes_repairs_and_imports_enter_the_chain(self) -> None:
        records: list[tuple[str, dict]] = []
        op1 = operation("o1", "k", "a", {"r1": 2, "r2": 1})
        op2 = operation("o2", "k", "b", {"r2": 1})  # stale: dominated by op1
        op3 = operation("o3", "k", "c", {"r3": 5})
        for replica, op in (("r1", op1), ("r2", op2), ("r3", op3)):
            status, _ = self.post_operation(replica, op)
            assert status == 201
            records.append((replica, op))
        repair = resolution(
            "r4",
            "o4",
            "k",
            "fixed",
            {"r1": 2, "r2": 1, "r3": 5, "r4": 1},
            [
                {"replicaId": "r1", "operationId": "o1"},
                {"replicaId": "r3", "operationId": "o3"},
            ],
        )
        status, _ = self.post_resolve("k", repair)
        self.assertEqual(status, 201)
        records.append(
            ("r4", operation("o4", "k", "fixed", {"r1": 2, "r2": 1, "r3": 5, "r4": 1}))
        )
        imported = [
            record("r5", operation("o5", "other", "x", {"r5": 1})),
            record("r6", operation("o6", "other", "y", {"r6": 1})),
        ]
        status, _ = self.post_sync({"operations": imported})
        self.assertEqual(status, 201)
        records.extend(
            [("r5", imported[0]["operation"]), ("r6", imported[1]["operation"])]
        )

        status, payload = self.chain()
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], expected_chain(records))
        self.assertEqual(payload["head"], payload["entries"][-1]["digest"])

    def test_replays_and_rejections_never_enter_the_chain(self) -> None:
        op = operation("o1", "k", "v", {"r1": 1})
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 201)
        status, before = self.chain()
        # Identical replay (200), conflicting identity (409), malformed (400),
        # and a conflicting import batch (409) all leave the chain untouched.
        status, _ = self.post_operation("r1", op)
        self.assertEqual(status, 200)
        status, _ = self.post_operation("r1", operation("o1", "k", "other", {"r1": 1}))
        self.assertEqual(status, 409)
        status, _ = self.post_operation("r1", {"operationId": "bad"})
        self.assertEqual(status, 400)
        status, _ = self.post_sync(
            {"operations": [record("r1", operation("o1", "k", "again", {"r1": 1}))]}
        )
        self.assertEqual(status, 409)
        status, after = self.chain()
        self.assertEqual(after, before)
        self.assertEqual(len(after["entries"]), 1)

    def test_query_is_strictly_read_only(self) -> None:
        self.seed(2)
        status, metrics_before = self.request("GET", "/v1/metrics")
        status, sync_before = self.request("GET", "/v1/sync/operations")
        status, _ = self.chain()
        status, _ = self.chain("?after=1&limit=1")
        status, metrics_after = self.request("GET", "/v1/metrics")
        status, sync_after = self.request("GET", "/v1/sync/operations")
        self.assertEqual(metrics_after, metrics_before)
        self.assertEqual(sync_after, sync_before)

    def test_post_to_chain_route_is_404(self) -> None:
        status, payload = self.request(
            "POST", "/v1/audit/log/chain", {"operations": []}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


class ChainAuthTests(unittest.TestCase):
    """The chain endpoint authenticates like every other non-/health route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        token_path = Path(cls.directory.name) / "token"
        token_path.write_text("s3cret-token", encoding="ascii")
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret-token"
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.directory.cleanup()

    def get(self, path: str, headers: list[tuple[str, str]]) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers=dict(headers))
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def test_missing_token_is_401(self) -> None:
        status, payload = self.get("/v1/audit/log/chain", [])
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_wrong_token_is_401(self) -> None:
        status, payload = self.get(
            "/v1/audit/log/chain", [("Authorization", "Bearer wrong")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_malformed_scheme_is_401(self) -> None:
        status, payload = self.get(
            "/v1/audit/log/chain", [("Authorization", "s3cret-token")]
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "unauthorized"})

    def test_valid_token_is_200(self) -> None:
        status, payload = self.get(
            "/v1/audit/log/chain", [("Authorization", "Bearer s3cret-token")]
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), CHAIN_FIELDS)

    def test_health_stays_anonymous(self) -> None:
        status, payload = self.get("/health", [])
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
