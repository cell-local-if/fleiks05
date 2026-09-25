"""Tests for the sender-side consumption-receipts query.

The receipts endpoint is::

    GET /v1/sync/peers/{peerId}/receipts?after=N&limit=N

It returns one read-only page of the consumption receipts a sending peer
previously committed with ``POST .../acknowledge``, in their commit
(creation) order, together with an integrity summary over the peer's
whole committed receipt set: ``algorithm``, the 64-character lowercase
hexadecimal SHA-256 ``digest`` of the canonical receipt-array bytes, and
``receiptsCount``. The list page and the summary come from one committed
snapshot, the query advances no checkpoint, and it changes neither memory
nor the data file.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) or the ``StateStore``
directly; only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _receipts_digest_input,
    parse_receipts_query,
)

TOKEN = "receipts-token_77"


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def identities(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"replicaId": replica, "operationId": op} for replica, op in pairs]


def canonical_receipts_array(receipts: list[dict]) -> bytes:
    """Independently render the canonical digest input for comparisons."""
    return _receipts_digest_input(receipts)


class ParseReceiptsQueryTests(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(parse_receipts_query(""), (0, 100))

    def test_accepts_valid_paging(self) -> None:
        self.assertEqual(parse_receipts_query("after=2&limit=5"), (2, 5))
        self.assertEqual(parse_receipts_query("after=0"), (0, 100))
        self.assertEqual(parse_receipts_query("limit=1"), (0, 1))
        self.assertEqual(parse_receipts_query("limit=100"), (0, 100))
        # Parameter order is insignificant.
        self.assertEqual(parse_receipts_query("limit=5&after=2"), (2, 5))

    def test_rejects_blank_malformed_and_negative_values(self) -> None:
        for query in (
            "?after=&limit=10",
            "after",
            "after=",
            "limit=",
            "after=-1",
            "after=1.0",
            "after=0x1",
            "after=%2B1",
            "after=1%20",
            "after=%D9%A1",
            "limit=one",
            "limit=-1",
            "limit=1.0",
        ):
            self.assertIsNone(parse_receipts_query(query), query)

    def test_rejects_out_of_range_limit(self) -> None:
        for query in ("limit=0", "limit=101", "after=0&limit=0", "after=0&limit=101"):
            self.assertIsNone(parse_receipts_query(query), query)

    def test_rejects_repeated_and_unknown_parameters(self) -> None:
        for query in (
            "after=1&after=2",
            "limit=1&limit=2",
            "foo=1",
            "after=0&foo=1",
            "=1",
            "after=0&limit=10&foo=1",
        ):
            self.assertIsNone(parse_receipts_query(query), query)


class ReceiptsDigestInputTests(unittest.TestCase):
    def test_empty_set_is_empty_array(self) -> None:
        self.assertEqual(_receipts_digest_input([]), b"[]")
        self.assertEqual(
            hashlib.sha256(b"[]").hexdigest(),
            hashlib.sha256(_receipts_digest_input([])).hexdigest(),
        )

    def test_fixed_field_and_identity_order(self) -> None:
        receipts = [
            {
                "peerId": "p1",
                "ackId": "a1",
                "cursor": 2,
                "operations": identities(("r1", "o1"), ("r2", "o2")),
            },
            {"peerId": "p1", "ackId": "a2", "cursor": 2, "operations": []},
        ]
        self.assertEqual(
            _receipts_digest_input(receipts),
            b'[{"peerId":"p1","ackId":"a1","cursor":2,'
            b'"operations":[{"replicaId":"r1","operationId":"o1"},'
            b'{"replicaId":"r2","operationId":"o2"}]},'
            b'{"peerId":"p1","ackId":"a2","cursor":2,"operations":[]}]',
        )

    def test_creation_order_is_preserved_never_sorted(self) -> None:
        first = {"peerId": "p", "ackId": "z", "cursor": 0, "operations": []}
        second = {"peerId": "p", "ackId": "a", "cursor": 0, "operations": []}
        raw = _receipts_digest_input([first, second]).decode("utf-8")
        self.assertLess(raw.index('"ackId":"z"'), raw.index('"ackId":"a"'))

    def test_identity_order_is_preserved(self) -> None:
        receipt = {
            "peerId": "p",
            "ackId": "a",
            "cursor": 2,
            "operations": identities(("r9", "z"), ("r1", "a")),
        }
        raw = _receipts_digest_input([receipt]).decode("utf-8")
        self.assertLess(raw.index('"operationId":"z"'), raw.index('"operationId":"a"'))

    def test_string_escaping_and_unicode(self) -> None:
        # Quote, backslash, a control character (newline), and a literal
        # non-ASCII code point all appear in peerId, ackId, and the
        # identities.
        weird = 'p"q' + chr(92) + chr(10) + "é"
        receipts = [
            {
                "peerId": weird,
                "ackId": weird,
                "cursor": 1,
                "operations": identities((weird, weird)),
            }
        ]
        raw = _receipts_digest_input(receipts)
        # No whitespace is emitted between encoded bytes; the only escapes
        # are the quote, the backslash, and the lowercase \u00xx control
        # escape, and e-acute is written as raw UTF-8. The weird string
        # appears four times: peerId, ackId, and both identity fields.
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)
        self.assertEqual(raw.count(b"\\u000a"), 4)
        self.assertEqual(raw.count(b'\\"'), 4)
        self.assertEqual(raw.count(b"\\\\"), 4)
        self.assertIn("é".encode("utf-8"), raw)
        fragment = b'p' + b'\\"' + b"q" + b"\\\\" + b"\\u000a" + "é".encode("utf-8")
        self.assertEqual(raw.count(fragment), 4)
        self.assertEqual(
            raw,
            b'[{"peerId":"' + fragment
            + b'","ackId":"' + fragment
            + b'","cursor":1,"operations":[{"replicaId":"' + fragment
            + b'","operationId":"' + fragment + b'"}]}]',
        )

    def test_every_control_character_uses_lowercase_hex(self) -> None:
        receipt = {"peerId": "p", "ackId": "".join(chr(i) for i in range(0x20)),
                   "cursor": 0, "operations": []}
        raw = _receipts_digest_input([receipt]).decode("utf-8")
        for i in range(0x20):
            self.assertIn(f"\\u{i:04x}", raw)


class PeerReceiptsStoreTests(unittest.TestCase):
    """Store-level receipts semantics, in memory."""

    def make_store(self) -> StateStore:
        store = StateStore()
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.apply_operation(
            "r1", operation("o3", "k", "v3", {"r1": 2, "r2": 1})
        )
        return store

    def commit_receipts(self, store: StateStore, peer: str = "p1") -> None:
        store.save_checkpoint(peer, 0)
        store.acknowledge_operations(
            peer, "a1", 2, identities(("r1", "o1"), ("r2", "o2"))
        )
        store.acknowledge_operations(peer, "a2", 3, identities(("r1", "o3")))
        store.acknowledge_operations(peer, "a3", 3, [])

    def test_unregistered_peer_is_not_found(self) -> None:
        store = self.make_store()
        status, payload = store.get_peer_receipts("nobody", 0, 100)
        self.assertIs(status, HTTPStatus.NOT_FOUND)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts_hashes_empty_array(self) -> None:
        store = self.make_store()
        store.save_checkpoint("p1", 0)
        status, payload = store.get_peer_receipts("p1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())

    def test_receipts_keep_commit_order_and_confirmed_content(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        status, payload = store.get_peer_receipts("p1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [(r["ackId"], r["cursor"]) for r in payload["receipts"]],
            [("a1", 2), ("a2", 3), ("a3", 3)],
        )
        self.assertEqual(
            payload["receipts"][0],
            {
                "peerId": "p1",
                "ackId": "a1",
                "cursor": 2,
                "operations": identities(("r1", "o1"), ("r2", "o2")),
            },
        )
        self.assertEqual(
            payload["receipts"][2],
            {"peerId": "p1", "ackId": "a3", "cursor": 3, "operations": []},
        )
        for receipt in payload["receipts"]:
            self.assertEqual(set(receipt), {"peerId", "ackId", "cursor", "operations"})
            for identity in receipt["operations"]:
                self.assertEqual(set(identity), {"replicaId", "operationId"})

    def test_summary_covers_the_whole_set_not_the_page(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        full = store.get_peer_receipts("p1", 0, 100)[1]
        all_receipts = full["receipts"]
        expected_digest = hashlib.sha256(
            canonical_receipts_array(all_receipts)
        ).hexdigest()
        # A one-item page still reports the full-set digest and count.
        status, page = store.get_peer_receipts("p1", 1, 1)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(page["receipts"]), 1)
        self.assertEqual(page["receipts"][0]["ackId"], "a2")
        self.assertEqual(page["nextCursor"], 2)
        self.assertIs(page["hasMore"], True)
        self.assertEqual(page["receiptsCount"], 3)
        self.assertEqual(page["digest"], expected_digest)
        self.assertEqual(full["digest"], expected_digest)

    def test_paging_and_empty_tail(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        status, first = store.get_peer_receipts("p1", 0, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual([r["ackId"] for r in first["receipts"]], ["a1", "a2"])
        self.assertEqual(first["nextCursor"], 2)
        self.assertIs(first["hasMore"], True)
        status, tail = store.get_peer_receipts("p1", 2, 2)
        self.assertEqual([r["ackId"] for r in tail["receipts"]], ["a3"])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertIs(tail["hasMore"], False)
        # after equal to the count is a valid empty page.
        status, empty = store.get_peer_receipts("p1", 3, 2)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(empty["receipts"], [])
        self.assertEqual(empty["nextCursor"], 3)
        self.assertIs(empty["hasMore"], False)
        self.assertEqual(empty["receiptsCount"], 3)

    def test_after_past_count_is_value_error(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        with self.assertRaises(ValueError):
            store.get_peer_receipts("p1", 4, 100)

    def test_receipts_are_isolated_per_peer(self) -> None:
        store = self.make_store()
        self.commit_receipts(store, "p1")
        # Another registered peer confirms a subset; commit interleaving
        # must never leak one peer's receipts into the other peer's list.
        store.save_checkpoint("p2", 0)
        store.acknowledge_operations(
            "p2", "b1", 1, identities(("r1", "o1"))
        )
        _, p1 = store.get_peer_receipts("p1", 0, 100)
        _, p2 = store.get_peer_receipts("p2", 0, 100)
        self.assertEqual([r["ackId"] for r in p1["receipts"]], ["a1", "a2", "a3"])
        self.assertEqual([r["ackId"] for r in p2["receipts"]], ["b1"])
        self.assertTrue(all(r["peerId"] == "p1" for r in p1["receipts"]))
        self.assertTrue(all(r["peerId"] == "p2" for r in p2["receipts"]))
        self.assertNotEqual(p1["digest"], p2["digest"])
        self.assertEqual((p1["receiptsCount"], p2["receiptsCount"]), (3, 1))

    def test_replay_adds_no_receipt(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        before = store.get_peer_receipts("p1", 0, 100)[1]
        store.acknowledge_operations(
            "p1", "a1", 2, identities(("r1", "o1"), ("r2", "o2"))
        )
        after = store.get_peer_receipts("p1", 0, 100)[1]
        self.assertEqual(after, before)

    def test_query_is_read_only(self) -> None:
        store = self.make_store()
        self.commit_receipts(store)
        metrics_before = store.get_metrics()
        checkpoint_before = store.get_checkpoint("p1")
        sync_before = store.get_sync_operations(0, 100)
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("a read must not persist"),
        ):
            for after in (0, 1, 3):
                store.get_peer_receipts("p1", after, 2)
        self.assertEqual(store.get_metrics(), metrics_before)
        self.assertEqual(store.get_checkpoint("p1"), checkpoint_before)
        self.assertEqual(store.get_sync_operations(0, 100), sync_before)

    def test_default_limit_is_one_hundred(self) -> None:
        store = StateStore()
        for i in range(150):
            store.apply_operation(
                "r", operation(f"o{i}", "k", "v", {"r": i + 1})
            )
        store.save_checkpoint("p1", 0)
        # Each receipt covers one record; 150 receipts committed in order.
        for i in range(150):
            store.acknowledge_operations(
                "p1", f"a{i}", i + 1, identities(("r", f"o{i}"))
            )
        _, first = store.get_peer_receipts("p1", 0, 100)
        self.assertEqual(len(first["receipts"]), 100)
        self.assertIs(first["hasMore"], True)
        _, second = store.get_peer_receipts("p1", 100, 100)
        self.assertEqual(len(second["receipts"]), 50)
        self.assertIs(second["hasMore"], False)
        # The digest and count are identical on every page.
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(first["receiptsCount"], 150)


class PersistentPeerReceiptsTests(unittest.TestCase):
    """Receipts read against and recovered from a real data file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_file = self.tmp / "state.json"

    def make_store(self) -> StateStore:
        return StateStore(data_file=str(self.data_file))

    def seed(self, store: StateStore) -> None:
        store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        store.save_checkpoint("p1", 0)
        store.acknowledge_operations(
            "p1", "a1", 2, identities(("r1", "o1"), ("r2", "o2"))
        )
        store.acknowledge_operations("p1", "a2", 2, [])

    def test_restart_preserves_order_paging_count_and_digest(self) -> None:
        store = self.make_store()
        self.seed(store)
        before = store.get_peer_receipts("p1", 0, 1)[1]
        before_full = store.get_peer_receipts("p1", 0, 100)[1]
        del store

        reloaded = self.make_store()
        after_full = reloaded.get_peer_receipts("p1", 0, 100)[1]
        self.assertEqual(after_full, before_full)
        # Page boundaries resume identically.
        after = reloaded.get_peer_receipts("p1", 0, 1)[1]
        self.assertEqual(after, before)
        self.assertEqual(after["nextCursor"], 1)
        self.assertIs(after["hasMore"], True)
        tail = reloaded.get_peer_receipts("p1", 2, 1)[1]
        self.assertEqual(tail["receipts"], [])
        self.assertEqual(tail["digest"], before_full["digest"])
        self.assertEqual(tail["receiptsCount"], 2)

    def test_old_version1_file_without_acks_recovers_empty(self) -> None:
        self.data_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "operations": [
                        {"replicaId": "r1",
                         "operation": operation("o1", "k", "v", {"r1": 1})}
                    ],
                    "checkpoints": {"p1": 1},
                }
            ),
            encoding="utf-8",
        )
        store = self.make_store()
        status, payload = store.get_peer_receipts("p1", 0, 100)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["receipts"], [])
        self.assertEqual(payload["receiptsCount"], 0)
        self.assertEqual(payload["digest"], hashlib.sha256(b"[]").hexdigest())
        # The read-only query did not upgrade or rewrite the old file.
        on_disk = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.assertNotIn("acks", on_disk)

    def test_read_does_not_touch_the_data_file(self) -> None:
        server_store = self.make_store()
        self.seed(server_store)
        before_bytes = self.data_file.read_bytes()
        before_mtime = self.data_file.stat().st_mtime_ns
        with patch.object(
            StateStore,
            "_persist_locked",
            side_effect=AssertionError("a read must not persist"),
        ):
            for query_after in (0, 1, 2):
                status, _ = server_store.get_peer_receipts("p1", query_after, 1)
                self.assertIs(status, HTTPStatus.OK)
            self.assertIs(
                server_store.get_peer_receipts("nobody", 0, 100)[0],
                HTTPStatus.NOT_FOUND,
            )
        self.assertEqual(self.data_file.read_bytes(), before_bytes)
        self.assertEqual(self.data_file.stat().st_mtime_ns, before_mtime)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])


class HttpReceiptsTests(unittest.TestCase):
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

    def request_raw(
        self, method: str, path: str, headers: dict | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        sent = {k.lower(): v for k, v in response.getheaders()}
        conn.close()
        return response.status, sent, raw

    def request(self, method: str, path: str) -> tuple[int, object]:
        status, _, raw = self.request_raw(method, path)
        return status, json.loads(raw.decode("utf-8")) if raw else None

    def post_json(self, path: str, body: object) -> tuple[int, object]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            path,
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw) if raw else None

    def seed_receipts(self, peer: str = "p1") -> list[dict]:
        for replica, op_id in (("r1", "o1"), ("r2", "o2"), ("r1", "o3")):
            clock = (
                {"r1": 2, "r2": 1}
                if op_id == "o3"
                else {replica: 1}
            )
            status, _ = self.post_json(
                f"/v1/replicas/{replica}/operations",
                operation(op_id, "k", op_id, clock),
            )
            self.assertEqual(status, 201)
        self.assertEqual(
            self.post_json(f"/v1/sync/peers/{peer}/checkpoint", {"cursor": 0})[0],
            200,
        )
        acks = [
            ("a1", 2, identities(("r1", "o1"), ("r2", "o2"))),
            ("a2", 3, identities(("r1", "o3"))),
        ]
        for ack_id, cursor, ops in acks:
            status, _ = self.post_json(
                f"/v1/sync/peers/{peer}/acknowledge",
                {"ackId": ack_id, "cursor": cursor, "operations": ops},
            )
            self.assertEqual(status, 201, ack_id)
        return [
            {"peerId": peer, "ackId": ack_id, "cursor": cursor, "operations": ops}
            for ack_id, cursor, ops in acks
        ]

    def receipts(self, peer: str, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/receipts{query}")

    def test_unregistered_peer_is_404(self) -> None:
        self.seed_receipts()
        status, payload = self.receipts("nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_registered_peer_without_receipts(self) -> None:
        self.assertEqual(
            self.post_json("/v1/sync/peers/empty/checkpoint", {"cursor": 0})[0],
            200,
        )
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/empty/receipts"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        digest = hashlib.sha256(b"[]").hexdigest()
        self.assertEqual(
            raw,
            f'{{"algorithm":"sha256","digest":"{digest}","hasMore":false,'
            f'"nextCursor":0,"receipts":[],"receiptsCount":0}}\n'.encode("utf-8"),
        )

    def test_success_body_is_compact_canonical_json_with_one_newline(self) -> None:
        receipts = self.seed_receipts()
        digest = hashlib.sha256(canonical_receipts_array(receipts)).hexdigest()
        status, headers, raw = self.request_raw(
            "GET", "/v1/sync/peers/p1/receipts?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertEqual(int(headers["content-length"]), len(raw))
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        expected = (
            b'{"algorithm":"sha256","digest":"' + digest.encode("ascii")
            + b'","hasMore":false,"nextCursor":2,'
            b'"receipts":[{"ackId":"a1","cursor":2,"operations":['
            b'{"operationId":"o1","replicaId":"r1"},'
            b'{"operationId":"o2","replicaId":"r2"}],"peerId":"p1"},'
            b'{"ackId":"a2","cursor":3,"operations":['
            b'{"operationId":"o3","replicaId":"r1"}],"peerId":"p1"}],'
            b'"receiptsCount":2}\n'
        )
        self.assertEqual(raw, expected)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            set(payload),
            {"receipts", "nextCursor", "hasMore", "algorithm", "digest",
             "receiptsCount"},
        )

    def test_paging_and_empty_tail_over_http(self) -> None:
        receipts = self.seed_receipts()
        full_digest = hashlib.sha256(canonical_receipts_array(receipts)).hexdigest()
        status, first = self.receipts("p1", "?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in first["receipts"]], ["a1"])
        self.assertEqual((first["nextCursor"], first["hasMore"]), (1, True))
        self.assertEqual((first["digest"], first["receiptsCount"]), (full_digest, 2))
        status, second = self.receipts("p1", f"?after={first['nextCursor']}&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in second["receipts"]], ["a2"])
        self.assertEqual((second["nextCursor"], second["hasMore"]), (2, False))
        # after equal to the count is an empty page, summary unchanged.
        status, tail = self.receipts("p1", "?after=2&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(tail["receipts"], [])
        self.assertEqual((tail["nextCursor"], tail["hasMore"]), (2, False))
        self.assertEqual((tail["digest"], tail["receiptsCount"]), (full_digest, 2))

    def test_defaults_apply_with_no_query(self) -> None:
        self.seed_receipts()
        status, payload = self.receipts("p1")
        self.assertEqual(status, 200)
        self.assertEqual([r["ackId"] for r in payload["receipts"]], ["a1", "a2"])
        self.assertEqual(payload["nextCursor"], 2)

    def test_invalid_queries_are_400(self) -> None:
        self.seed_receipts()
        bad_queries = [
            "?after=",
            "?after",
            "?after=-1",
            "?after=1.0",
            "?after=0x1",
            "?after=%2B1",
            "?after=1%20",
            "?after=%D9%A1",
            "?limit=0",
            "?limit=101",
            "?limit=-1",
            "?limit=one",
            "?after=1&after=2",
            "?limit=1&limit=2",
            "?foo=1",
            "?after=0&foo=1",
            "?=1",
            # after past the peer's receipt count (2).
            "?after=3&limit=10",
        ]
        for query in bad_queries:
            status, payload = self.receipts("p1", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)
        # The receipts are still all there.
        self.assertEqual(self.receipts("p1")[1]["receiptsCount"], 2)

    def test_malformed_query_on_unknown_peer_is_400_not_404(self) -> None:
        # Query validation runs before the peer lookup.
        status, payload = self.receipts("nope", "?after=x")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_route_shapes_are_404_before_query_checks(self) -> None:
        self.seed_receipts()
        not_found_paths = [
            "/v1/sync/peers//receipts",
            "/v1/sync/peers//receipts?after=x",
            "/v1/sync/peers//receipts?after=0&limit=10",
            "/v1/sync/peers/p1/receipts/extra",
            "/v1/sync/peers/p1/receipts/extra?after=0",
            "/v1/sync/peers/p1/receipts/",
            "/v1/sync/peers/p1",
            "/v1/sync/peers/p1/not-receipts",
            "/v1/sync/operations/p1/receipts",
            "/v2/sync/peers/p1/receipts",
        ]
        for path in not_found_paths:
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_other_peer_routes_are_not_shadowed(self) -> None:
        self.seed_receipts()
        status, payload = self.request("GET", "/v1/sync/peers/p1/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "p1", "cursor": 3})
        status, payload = self.request(
            "GET", "/v1/sync/peers/p1/operations?after=0&limit=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operations"], [])
        # POST is not published on the receipts route.
        self.assertEqual(self.request("POST", "/v1/sync/peers/p1/receipts")[0], 404)

    def test_peer_id_is_percent_decoded(self) -> None:
        self.seed_receipts("peer%20one")
        status, payload = self.receipts("peer%20one")
        self.assertEqual(status, 200)
        self.assertEqual(payload["receiptsCount"], 2)
        self.assertTrue(
            all(r["peerId"] == "peer one" for r in payload["receipts"])
        )

    def test_get_is_read_only(self) -> None:
        receipts = self.seed_receipts()
        metrics_before = self.request("GET", "/v1/metrics")[1]
        checkpoint_before = self.request("GET", "/v1/sync/peers/p1/checkpoint")[1]
        full_digest = hashlib.sha256(canonical_receipts_array(receipts)).hexdigest()
        for _ in range(3):
            for query in ("", "?after=0&limit=1", "?after=2&limit=1"):
                status, payload = self.receipts("p1", query)
                self.assertEqual(status, 200)
                self.assertEqual(payload["digest"], full_digest)
        self.assertEqual(self.request("GET", "/v1/metrics")[1], metrics_before)
        self.assertEqual(
            self.request("GET", "/v1/sync/peers/p1/checkpoint")[1], checkpoint_before
        )


class AuthenticatedReceiptsHttpTests(unittest.TestCase):
    """Bearer auth over a persistent data file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = Path(self._tmp.name) / "state.json"
        self.server = SemanticStateServer(
            ("127.0.0.1", 0),
            RequestHandler,
            data_file=str(self.data_file),
            auth_token=TOKEN,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def call(self, path: str, auth: str | None = None) -> tuple[int, object, bytes]:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=5
        )
        conn.request("GET", path, headers={"Authorization": auth} if auth else {})
        response = conn.getresponse()
        raw = response.read()
        headers = {k.lower(): v for k, v in response.getheaders()}
        conn.close()
        return (
            response.status,
            json.loads(raw) if raw else None,
            headers.get("www-authenticate", "").encode("utf-8"),
        )

    def test_unauthenticated_is_401(self) -> None:
        for auth in (None, "Bearer wrong", "Bearer", f"Token {TOKEN}",
                     f"Bearer {TOKEN} ") :
            status, payload, challenge = self.call(
                "/v1/sync/peers/p1/receipts", auth
            )
            self.assertEqual(status, 401, repr(auth))
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertEqual(challenge, b"Bearer")
        # Health stays anonymous.
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=5
        )
        conn.request("GET", "/health")
        self.assertEqual(conn.getresponse().status, 200)

    def test_authenticated_unknown_peer_is_404(self) -> None:
        status, payload, _ = self.call(
            "/v1/sync/peers/p1/receipts", f"Bearer {TOKEN}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
