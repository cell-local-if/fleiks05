"""Tests for the read-only complete-candidate export endpoint::

    GET /v1/replication/export

It pages the **complete** current candidate snapshot grouped by business
key. A successful response carries exactly seven fields, in this order:
``snapshot`` (one page of ``{"key","candidates"}`` groups, each candidate
in the comparison entry point's ``value``/``clock``/``replicaId``/
``operationId`` shape, sorted by ``(replicaId, operationId)``),
``nextCursor``, ``hasMore``, ``algorithm`` (always ``"sha256"``),
``digest`` (the 64-char lowercase SHA-256 of the complete snapshot under
the verification-digest rules — identical on every page; the empty
snapshot hashes ``[]``), ``keys`` (the complete key count), and
``candidateVersions`` (the total candidate count across all keys).

Paging trims only whole key groups: a key's complete candidate set is
never split across pages. All three query parameters — ``after``,
``limit``, and ``expectedDigest`` — are required. A digest mismatch is
HTTP 409 ``export_conflict`` with no page content.

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread) where HTTP behavior is
tested, and against ``StateStore`` directly for snapshot and recovery
semantics. Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
    _verification_digest_input,
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
EMPTY_CANDIDATE_DIGEST = hashlib.sha256(b"[]").hexdigest()
FIELD_ORDER = [
    "snapshot",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "keys",
    "candidateVersions",
]


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def full_digest(store: StateStore) -> str:
    return hashlib.sha256(
        _verification_digest_input(store._candidates)
    ).hexdigest()


def export_path(after: int, limit: int, digest: str) -> str:
    return f"/v1/replication/export?after={after}&limit={limit}&expectedDigest={digest}"


class ExportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def test_empty_snapshot_empty_page(self) -> None:
        status, payload = self.store.export_replication_candidates(
            0, 100, EMPTY_CANDIDATE_DIGEST
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["snapshot"], [])
        self.assertEqual(payload["nextCursor"], 0)
        self.assertIs(payload["hasMore"], False)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["digest"], EMPTY_CANDIDATE_DIGEST)
        self.assertEqual(payload["keys"], 0)
        self.assertEqual(payload["candidateVersions"], 0)

    def test_wrong_digest_is_conflict_without_page(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.store.export_replication_candidates(
            0, 100, EMPTY_CANDIDATE_DIGEST
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "export_conflict"})

    def test_conflict_then_re_request_with_current_digest_succeeds(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        current = full_digest(self.store)
        # A stale expectation is rejected without page content, then the
        # caller retries with the current digest (as the snapshot entry
        # point reports it) and gets the page.
        status, _ = self.store.export_replication_candidates(
            0, 100, EMPTY_CANDIDATE_DIGEST
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        status, payload = self.store.export_replication_candidates(0, 100, current)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(len(payload["snapshot"]), 1)
        self.assertEqual(payload["digest"], current)

    def test_page_groups_keys_and_keeps_full_candidate_sets(self) -> None:
        # Two concurrent candidates on "a", one on "b".
        self.store.apply_operation("r1", operation("o1", "a", "x", {"r1": 1}))
        self.store.apply_operation("r2", operation("o2", "a", "y", {"r2": 1}))
        self.store.apply_operation("r3", operation("o3", "b", "z", {"r3": 1}))
        digest = full_digest(self.store)
        status, page_one = self.store.export_replication_candidates(0, 1, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(page_one["keys"], 2)
        self.assertEqual(page_one["candidateVersions"], 3)
        self.assertIs(page_one["hasMore"], True)
        self.assertEqual(page_one["nextCursor"], 1)
        self.assertEqual(
            page_one["snapshot"],
            [
                {
                    "key": "a",
                    "candidates": [
                        {
                            "value": "x",
                            "clock": {"r1": 1},
                            "replicaId": "r1",
                            "operationId": "o1",
                        },
                        {
                            "value": "y",
                            "clock": {"r2": 1},
                            "replicaId": "r2",
                            "operationId": "o2",
                        },
                    ],
                }
            ],
        )
        # The second page carries the complete second group; the digest
        # and both counts stay identical across pages.
        status, page_two = self.store.export_replication_candidates(1, 1, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(page_two["nextCursor"], 2)
        self.assertIs(page_two["hasMore"], False)
        self.assertEqual([g["key"] for g in page_two["snapshot"]], ["b"])
        self.assertEqual(page_two["digest"], digest)
        self.assertEqual(page_two["keys"], 2)
        self.assertEqual(page_two["candidateVersions"], 3)

    def test_after_equal_to_key_count_is_stable_empty_page(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        digest = full_digest(self.store)
        status, payload = self.store.export_replication_candidates(1, 100, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["snapshot"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertIs(payload["hasMore"], False)
        # Re-requesting the same page is stable.
        status, again = self.store.export_replication_candidates(1, 100, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(again, payload)

    def test_after_past_key_count_raises(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        with self.assertRaises(ValueError):
            self.store.export_replication_candidates(
                2, 100, full_digest(self.store)
            )

    def test_stale_digest_conflicts_even_when_cursor_is_past_end(self) -> None:
        # One key exists; the caller's digest is stale and its cursor is
        # past the current count. The stale expectation is the primary
        # fact: answer 409, not the 400 a correct digest would give.
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.store.export_replication_candidates(
            9, 100, EMPTY_CANDIDATE_DIGEST
        )
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "export_conflict"})

    def test_candidate_ordering_and_clock_copy(self) -> None:
        self.store.apply_operation("r2", operation("o9", "k", "v2", {"r2": 1}))
        self.store.apply_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = self.store.export_replication_candidates(
            0, 100, full_digest(self.store)
        )
        self.assertIs(status, HTTPStatus.OK)
        identities = [
            (c["replicaId"], c["operationId"])
            for c in payload["snapshot"][0]["candidates"]
        ]
        self.assertEqual(identities, [("r1", "o1"), ("r2", "o9")])
        # The exported candidates are copies, not the live objects.
        payload["snapshot"][0]["candidates"][0]["clock"]["r1"] = 99
        self.assertEqual(
            self.store._candidates["k"][1]["clock"], {"r1": 1}
        )

    def test_keys_sorted_in_unicode_order(self) -> None:
        for key in ("z", "a", "m", "é"):
            self.store.apply_operation(
                "r1", operation(f"o-{key}", key, "v", {"r1": 1})
            )
        status, payload = self.store.export_replication_candidates(
            0, 100, full_digest(self.store)
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            [g["key"] for g in payload["snapshot"]], ["a", "m", "z", "é"]
        )

    def test_digest_matches_verification_endpoint_rules(self) -> None:
        self.store.apply_operation(
            "r1", operation("o1", "clé", "a\nb→", {"r2": 1, "r1": 2})
        )
        expected = hashlib.sha256(
            _verification_digest_input(self.store._candidates)
        ).hexdigest()
        status, payload = self.store.export_replication_candidates(
            0, 100, expected
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["digest"], expected)
        self.assertEqual(
            payload["digest"], self.store.get_verification_digest()["digest"]
        )

    def test_export_is_side_effect_free(self) -> None:
        self.store.apply_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        digest = full_digest(self.store)
        for _ in range(3):
            status, payload = self.store.export_replication_candidates(
                0, 1, digest
            )
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(payload["digest"], digest)
        self.assertEqual(self.store.get_metrics()["acceptedOperations"], 1)


class ExportRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_file = str(Path(self._tmp.name) / "state.json")

    def test_pages_and_digest_match_after_restart(self) -> None:
        store = StateStore(data_file=self.data_file)
        store.apply_operation("r1", operation("o1", "k1", "v1", {"r1": 1}))
        store.apply_operation("r2", operation("o2", "k2", "v2", {"r2": 1}))
        digest = full_digest(store)
        del store
        recovered = StateStore(data_file=self.data_file)
        seen = []
        after = 0
        while True:
            status, payload = recovered.export_replication_candidates(
                after, 1, digest
            )
            self.assertIs(status, HTTPStatus.OK)
            seen.extend(payload["snapshot"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual([g["key"] for g in seen], ["k1", "k2"])
        self.assertEqual(after, 2)

        status, payload = recovered.export_replication_candidates(
            0, 100, digest
        )
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["digest"], digest)


class ExportHttpServerTests(unittest.TestCase):
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
    ) -> tuple[int, object, bytes, list[tuple[str, str]]]:
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

    def request(self, method: str, path: str) -> tuple[int, object]:
        status, payload, _, _ = self.raw_request(method, path)
        return status, payload

    def post_operation(self, replica: str, op: dict) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            body=json.dumps(op),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        self.assertEqual(response.status, 201, raw)

    def export(self, after: int, limit: int, digest: str) -> tuple[int, object]:
        return self.request("GET", export_path(after, limit, digest))

    def current_digest(self) -> str:
        status, payload = self.request("GET", "/v1/verification/digest")
        assert status == 200
        return payload["digest"]

    def test_empty_store_first_page(self) -> None:
        status, payload = self.export(0, 100, EMPTY_CANDIDATE_DIGEST)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "snapshot": [],
                "nextCursor": 0,
                "hasMore": False,
                "algorithm": "sha256",
                "digest": EMPTY_CANDIDATE_DIGEST,
                "keys": 0,
                "candidateVersions": 0,
            },
        )

    def test_field_order_headers_and_single_trailing_newline(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload, raw, headers = self.raw_request(
            "GET", export_path(0, 100, self.current_digest())
        )
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), FIELD_ORDER)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertRegex(payload["digest"], DIGEST_RE)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        # Field order is fixed, not sorted: snapshot precedes nextCursor.
        text = raw[:-1].decode("utf-8")
        self.assertLess(text.index('"snapshot"'), text.index('"nextCursor"'))
        self.assertLess(text.index('"nextCursor"'), text.index('"hasMore"'))
        self.assertLess(text.index('"hasMore"'), text.index('"algorithm"'))
        self.assertLess(text.index('"algorithm"'), text.index('"digest"'))
        self.assertLess(text.index('"digest"'), text.index('"keys"'))
        self.assertLess(text.index('"keys"'), text.index('"candidateVersions"'))
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_paging_walks_every_group_once(self) -> None:
        for index in range(5):
            self.post_operation(
                f"r{index}", operation(f"o{index}", f"k{index}", "v", {f"r{index}": 1})
            )
        digest = self.current_digest()
        groups = []
        after = 0
        while True:
            status, payload = self.export(after, 2, digest)
            self.assertEqual(status, 200)
            self.assertEqual(payload["digest"], digest)
            self.assertEqual(payload["keys"], 5)
            self.assertEqual(payload["candidateVersions"], 5)
            self.assertLessEqual(len(payload["snapshot"]), 2)
            groups.extend(g["key"] for g in payload["snapshot"])
            after = payload["nextCursor"]
            if not payload["hasMore"]:
                break
        self.assertEqual(groups, [f"k{i}" for i in range(5)])
        self.assertEqual(after, 5)
        # The tail cursor returns a stable empty page.
        status, payload = self.export(5, 2, digest)
        self.assertEqual(status, 200)
        self.assertEqual(payload["snapshot"], [])
        self.assertEqual(payload["nextCursor"], 5)
        self.assertIs(payload["hasMore"], False)

    def test_single_key_group_is_never_split(self) -> None:
        self.post_operation("r1", operation("o1", "k", "a", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "b", {"r2": 1}))
        digest = self.current_digest()
        status, payload = self.export(0, 1, digest)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["snapshot"]), 1)
        self.assertEqual(len(payload["snapshot"][0]["candidates"]), 2)
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 2)

    def test_digest_mismatch_is_409_export_conflict(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.export(0, 100, EMPTY_CANDIDATE_DIGEST)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "export_conflict"})
        # Retrying with the digest the verification endpoint reports works.
        status, payload = self.export(0, 100, self.current_digest())
        self.assertEqual(status, 200)
        self.assertEqual([g["key"] for g in payload["snapshot"]], ["k"])

    def test_conflict_does_not_paginate(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        # A wrong digest conflicts regardless of the page position.
        for after in (0, 1):
            status, payload = self.export(after, 100, EMPTY_CANDIDATE_DIGEST)
            self.assertEqual(status, 409)
            self.assertEqual(payload, {"error": "export_conflict"})

    def test_missing_parameters_are_400(self) -> None:
        for path in (
            "/v1/replication/export",
            "/v1/replication/export?",
            "/v1/replication/export?after=0&limit=100",
            "/v1/replication/export?after=0",
            "/v1/replication/export?limit=100",
            "/v1/replication/export?after=0&limit=100",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_repeated_parameters_are_400(self) -> None:
        digest = EMPTY_CANDIDATE_DIGEST
        for path in (
            f"/v1/replication/export?after=0&after=1&limit=1&expectedDigest={digest}",
            f"/v1/replication/export?after=0&limit=1&limit=2&expectedDigest={digest}",
            f"/v1/replication/export?after=0&limit=1&expectedDigest={digest}&expectedDigest={digest}",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_unknown_parameter_is_400(self) -> None:
        digest = EMPTY_CANDIDATE_DIGEST
        for path in (
            f"/v1/replication/export?after=0&limit=1&expectedDigest={digest}&x=1",
            "/v1/replication/export?after=0&limit=1&expectedDigest=" + digest + "&x",
            "/v1/replication/export?=1",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(payload, {"error": "invalid_request"}, path)

    def test_illegal_after_is_400(self) -> None:
        from urllib.parse import quote

        digest = EMPTY_CANDIDATE_DIGEST
        for after in ("-1", "1.0", " 1", "1 ", "+1", "", "abc", "١"):
            path = (
                f"/v1/replication/export?after={quote(after, safe='')}"
                f"&limit=1&expectedDigest={digest}"
            )
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, repr(after))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(after))

    def test_illegal_or_out_of_range_limit_is_400(self) -> None:
        from urllib.parse import quote

        digest = EMPTY_CANDIDATE_DIGEST
        for limit in ("0", "101", "-1", "1.0", " 1", "1 ", "+5", "", "abc", "١"):
            path = (
                f"/v1/replication/export?after=0"
                f"&limit={quote(limit, safe='')}&expectedDigest={digest}"
            )
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, repr(limit))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(limit))

    def test_limit_boundaries_1_and_100_are_accepted(self) -> None:
        for limit in (1, 100):
            status, payload = self.export(0, limit, EMPTY_CANDIDATE_DIGEST)
            self.assertEqual(status, 200, limit)

    def test_illegal_digest_is_400(self) -> None:
        good = EMPTY_CANDIDATE_DIGEST
        bad_values = (
            good.upper(),
            good[:-1],
            good + "a",
            "z" * 64,
            good[:63] + "g",
            "",
            "abc",
        )
        for value in bad_values:
            path = f"/v1/replication/export?after=0&limit=1&expectedDigest={value}"
            status, payload = self.request("GET", path)
            self.assertEqual(status, 400, value)
            self.assertEqual(payload, {"error": "invalid_request"}, value)

    def test_after_past_key_count_is_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, payload = self.export(2, 100, self.current_digest())
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_path_shape_mismatches_are_404(self) -> None:
        digest = EMPTY_CANDIDATE_DIGEST
        suffix = f"after=0&limit=1&expectedDigest={digest}"
        for path in (
            "/v1/replication/export/extra",
            "/v1/replication",
            "/v1/replication/export/",
            "/v1/replication/exports",
            f"/v1/replication/export/extra?{suffix}",
            f"/v1/replication/export/?{suffix}",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_non_exact_path_precedes_parameter_check(self) -> None:
        # A malformed query on a wrong path shape is still 404.
        for path in (
            "/v1/replication/export/extra",
            "/v1/replication/export/?after=not-a-number",
            "/v1/replication/export/extra?limit=0",
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/replication/export")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_export_does_not_mutate_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        digest = self.current_digest()
        for _ in range(3):
            status, payload = self.export(0, 1, digest)
            self.assertEqual(status, 200)
            self.assertEqual(payload["digest"], digest)
        self.assertEqual(self.current_digest(), digest)
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["value"], "v")

    def test_all_numbers_are_json_integers(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        status, _, raw, _ = self.raw_request(
            "GET", export_path(0, 1, self.current_digest())
        )
        self.assertEqual(status, 200)
        text = raw[:-1].decode("utf-8")
        for forbidden in ("0.0", "-0", "1.0", "e+", "NaN", "Infinity"):
            self.assertNotIn(forbidden, text)

    def test_concurrent_commits_observe_consistent_pages(self) -> None:
        violations: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            # The reader keeps the digest it first observes and re-requests
            # on 409; every accepted page must be internally consistent.
            while not stop.is_set():
                status, verification = self.request("GET", "/v1/verification/digest")
                if status != 200:
                    violations.append("digest endpoint failed")
                    continue
                digest = verification["digest"]
                status, page = self.export(0, 100, digest)
                if status == 409:
                    continue
                if status != 200:
                    violations.append(f"unexpected status {status}")
                    continue
                grouped = sum(len(g["candidates"]) for g in page["snapshot"])
                if page["candidateVersions"] < page["keys"]:
                    violations.append("candidateVersions < keys")
                if page["hasMore"]:
                    violations.append("limit 100 page reported hasMore")
                if grouped > page["candidateVersions"]:
                    violations.append("page holds more candidates than total")
                if page["digest"] != digest:
                    violations.append("page digest disagrees with snapshot")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader_thread in readers:
            reader_thread.start()
        try:
            for index in range(40):
                replica = f"r{index}"
                self.post_operation(
                    replica,
                    operation(f"op-{index}", f"key-{index % 7}", f"v{index}", {replica: 1}),
                )
        finally:
            stop.set()
            for reader_thread in readers:
                reader_thread.join(timeout=5)
        self.assertEqual(violations, [])


class ExportPersistentHttpServerTests(unittest.TestCase):
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

    def test_pages_survive_restart(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k1", "v1", {"r1": 1}),
        )
        self.request(
            server,
            "POST",
            "/v1/replicas/r2/operations",
            operation("o2", "k2", "v2", {"r2": 1}),
        )
        status, digest_payload = self.request(server, "GET", "/v1/verification/digest")
        self.assertEqual(status, 200)
        digest = digest_payload["digest"]
        before_pages = []
        for after in (0, 1, 2):
            status, page = self.request(
                server,
                "GET",
                f"/v1/replication/export?after={after}&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
            before_pages.append(page)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        for after, expected in zip((0, 1, 2), before_pages):
            status, page = self.request(
                server,
                "GET",
                f"/v1/replication/export?after={after}&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
            self.assertEqual(page, expected)

    def test_export_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        status, digest_payload = self.request(server, "GET", "/v1/verification/digest")
        digest = digest_payload["digest"]
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()
        for after in (0, 1):
            status, _ = self.request(
                server,
                "GET",
                f"/v1/replication/export?after={after}&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


class ExportAuthTests(unittest.TestCase):
    """The endpoint authenticates like every other non-/health route."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        token_file = Path(self._tmp.name) / "token"
        token_file.write_text("s3cret", encoding="ascii")
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, auth_token="s3cret"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def raw_get(self, path: str, headers: dict | None = None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=5
        )
        conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        www = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, www

    def test_missing_or_mismatched_token_is_401_with_challenge(self) -> None:
        path = export_path(0, 1, EMPTY_CANDIDATE_DIGEST)
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "s3cret"},
            {"Authorization": "Bearer"},
        ):
            status, payload, www = self.raw_get(path, headers)
            self.assertEqual(status, 401, headers)
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertEqual(www, "Bearer")

    def test_duplicate_authorization_header_is_401(self) -> None:
        import socket

        with socket.create_connection(("127.0.0.1", self.server.server_address[1]), timeout=5) as sock:
            request = (
                f"GET {export_path(0, 1, EMPTY_CANDIDATE_DIGEST)} HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Authorization: Bearer s3cret\r\n"
                "Authorization: Bearer s3cret\r\n"
                "Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode("ascii"))
            reply = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                reply += chunk
        self.assertIn(b" 401 ", reply.split(b"\r\n", 1)[0])
        self.assertIn(b"WWW-Authenticate: Bearer", reply)

    def test_valid_token_reaches_the_endpoint(self) -> None:
        status, payload, www = self.raw_get(
            export_path(0, 1, EMPTY_CANDIDATE_DIGEST),
            {"Authorization": "Bearer s3cret"},
        )
        self.assertEqual(status, 200)
        self.assertIsNone(www)
        self.assertEqual(list(payload), FIELD_ORDER)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _ = self.raw_get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")


class ExportScopePolicyTests(unittest.TestCase):
    """read/admin scopes pass; write-only is 403 without a challenge."""

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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store = type(self.server.store)()

    def get(self, token: str | None):
        import socket

        headers = "Host: localhost\r\nConnection: close\r\n"
        if token is not None:
            headers += f"Authorization: Bearer {token}\r\n"
        with socket.create_connection(
            ("127.0.0.1", self.server.server_address[1]), timeout=5
        ) as sock:
            sock.sendall(
                (
                    f"GET {export_path(0, 1, EMPTY_CANDIDATE_DIGEST)} HTTP/1.1\r\n"
                    + headers
                    + "\r\n"
                ).encode("ascii")
            )
            reply = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                reply += chunk
        status_line = reply.split(b"\r\n", 1)[0].decode("ascii")
        challenge = b"WWW-Authenticate: Bearer" in reply
        return int(status_line.split()[1]), challenge

    def test_scopes(self) -> None:
        status, challenge = self.get(self.READ_TOKEN)
        self.assertEqual((status, challenge), (200, False))
        status, challenge = self.get(self.ADMIN_TOKEN)
        self.assertEqual((status, challenge), (200, False))
        status, challenge = self.get(self.WRITE_TOKEN)
        self.assertEqual((status, challenge), (403, False))
        status, challenge = self.get(None)
        self.assertEqual((status, challenge), (401, True))


if __name__ == "__main__":
    unittest.main()
