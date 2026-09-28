"""Tests for the read-only full candidate-snapshot export endpoint::

    GET /v1/replication/export

It pages the complete current candidate snapshot one business-key group
at a time. A successful response carries exactly seven fields in the
fixed order ``snapshot``, ``nextCursor``, ``hasMore``, ``algorithm``,
``digest``, ``keys``, and ``candidateVersions``. The page trims only
business keys (never the candidates of a single key), each group keeps
the comparison entry's ``{"value","clock","replicaId","operationId"}``
candidate shape sorted by ``(replicaId, operationId)``, and ``digest`` is
the 64-character lowercase SHA-256 of the canonical candidate snapshot
under exactly the verification-digest rules (an empty store hashes
``[]``). A digest expectation that does not match the committed snapshot
is HTTP 409 ``export_conflict``.

The three required query parameters are ``after`` (a 0-based count of
business keys already exported), ``limit`` (an ASCII decimal integer from
1 to 100), and ``expectedDigest`` (exactly 64 lowercase hexadecimal
characters). Anything missing, repeated, unknown, malformed, or
out of range is HTTP 400 ``invalid_request``. Path-shape mismatches are
404 before any query check. Authentication and scopes follow every other
non-``/health`` GET: 401 with a ``Bearer`` challenge, 403 without one for
a token lacking ``read``/``admin``.

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
EXPORT_FIELDS = {
    "snapshot",
    "nextCursor",
    "hasMore",
    "algorithm",
    "digest",
    "keys",
    "candidateVersions",
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def export_digest(store: StateStore) -> str:
    return hashlib.sha256(
        _verification_digest_input(store._candidates)
    ).hexdigest()


class ExportQueryParserTests(unittest.TestCase):
    """The parser enforces the three required parameters."""

    def setUp(self) -> None:
        from semantic_state_engine.server import parse_replication_export_query

        self.parse = parse_replication_export_query

    def test_well_formed(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            self.parse(f"after=0&limit=100&expectedDigest={digest}"),
            (0, 100, digest),
        )
        self.assertEqual(
            self.parse(f"after=7&limit=1&expectedDigest={digest}"),
            (7, 1, digest),
        )

    def test_missing_parameters(self) -> None:
        digest = "a" * 64
        for query in (
            "",
            f"limit=1&expectedDigest={digest}",
            f"after=0&expectedDigest={digest}",
            "after=0&limit=1",
            "after=0",
            "limit=1",
            "expectedDigest=" + digest,
        ):
            self.assertIsNone(self.parse(query), query)

    def test_repeated_or_unknown_parameters(self) -> None:
        digest = "a" * 64
        base = f"after=0&limit=1&expectedDigest={digest}"
        for query in (
            f"{base}&x=1",
            f"{base}&after=1",
            "after=0&after=1&limit=1&expectedDigest=" + digest,
            "after=0&limit=1&limit=2&expectedDigest=" + digest,
            "after=0&limit=1&expectedDigest=" + digest + "&expectedDigest=" + digest,
        ):
            self.assertIsNone(self.parse(query), query)

    def test_after_must_be_non_negative_ascii_decimal(self) -> None:
        digest = "a" * 64
        for after in ("-1", "1.0", "01 ", " 1", "+", "0x1", "１", ""):
            query = f"after={after}&limit=1&expectedDigest={digest}"
            self.assertIsNone(self.parse(query), after)

    def test_limit_bounds_and_shape(self) -> None:
        digest = "a" * 64
        for limit in ("0", "101", "-1", "1.0", " 1", "1 ", "+", "１", ""):
            query = f"after=0&limit={limit}&expectedDigest={digest}"
            self.assertIsNone(self.parse(query), limit)
        for limit in ("1", "50", "100"):
            self.assertEqual(
                self.parse(f"after=0&limit={limit}&expectedDigest={digest}"),
                (0, int(limit), digest),
            )

    def test_expected_digest_must_be_64_lowercase_hex(self) -> None:
        base = "after=0&limit=1&expectedDigest="
        for value in (
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,
            ("a" * 63) + "g",
            "f" * 64 + "0",
        ):
            self.assertIsNone(self.parse(base + value), value)
        # A valid 64-lowercase-hex value is accepted.
        self.assertEqual(
            self.parse(base + "0123456789abcdef" * 4),
            (0, 1, "0123456789abcdef" * 4),
        )


class ExportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StateStore()

    def add(
        self, replica: str, operation_id: str, key: str, value: str, clock: dict
    ) -> None:
        self.store.apply_operation(replica, operation(operation_id, key, value, clock))

    def test_empty_store_is_hash_of_empty_array(self) -> None:
        status, payload = self.store.get_replication_export(
            0, 100, EMPTY_CANDIDATE_DIGEST
        )
        self.assertIs(status, HTTPStatus.OK)
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

    def test_after_equal_to_key_count_is_a_stable_empty_page(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        digest = export_digest(self.store)
        status, payload = self.store.get_replication_export(1, 100, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(payload["snapshot"], [])
        self.assertEqual(payload["nextCursor"], 1)
        self.assertFalse(payload["hasMore"])
        # Counts and digest still cover the complete snapshot.
        self.assertEqual(payload["keys"], 1)
        self.assertEqual(payload["candidateVersions"], 1)
        self.assertEqual(payload["digest"], digest)

    def test_after_past_count_raises(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        with self.assertRaises(ValueError):
            self.store.get_replication_export(2, 100, export_digest(self.store))

    def test_digest_mismatch_is_conflict(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        status, payload = self.store.get_replication_export(0, 100, "0" * 64)
        self.assertIs(status, HTTPStatus.CONFLICT)
        self.assertEqual(payload, {"error": "export_conflict"})

    def test_groups_keep_full_candidates_and_identity_order(self) -> None:
        self.add("r1", "op-1", "color", "blue", {"r1": 1})
        self.add("r2", "op-2", "color", "red", {"r2": 1})
        self.add("r2", "op-9", "size", "large", {"r2": 1})
        digest = export_digest(self.store)
        status, payload = self.store.get_replication_export(0, 100, digest)
        self.assertIs(status, HTTPStatus.OK)
        self.assertEqual(
            payload["snapshot"],
            [
                {
                    "key": "color",
                    "candidates": [
                        {
                            "value": "blue",
                            "clock": {"r1": 1},
                            "replicaId": "r1",
                            "operationId": "op-1",
                        },
                        {
                            "value": "red",
                            "clock": {"r2": 1},
                            "replicaId": "r2",
                            "operationId": "op-2",
                        },
                    ],
                },
                {
                    "key": "size",
                    "candidates": [
                        {
                            "value": "large",
                            "clock": {"r2": 1},
                            "replicaId": "r2",
                            "operationId": "op-9",
                        }
                    ],
                },
            ],
        )
        self.assertEqual(payload["keys"], 2)
        self.assertEqual(payload["candidateVersions"], 3)
        self.assertEqual(payload["digest"], digest)

    def test_paging_trims_keys_but_never_a_keys_candidates(self) -> None:
        # Three keys, the middle carrying two candidates.
        self.add("r1", "o1", "a", "1", {"r1": 1})
        self.add("r1", "o2", "b", "2", {"r1": 1})
        self.add("r2", "o3", "b", "3", {"r2": 1})
        self.add("r1", "o4", "c", "4", {"r1": 1})
        digest = export_digest(self.store)

        first_status, first = self.store.get_replication_export(0, 1, digest)
        self.assertIs(first_status, HTTPStatus.OK)
        self.assertEqual([g["key"] for g in first["snapshot"]], ["a"])
        self.assertEqual(first["nextCursor"], 1)
        self.assertTrue(first["hasMore"])

        second_status, second = self.store.get_replication_export(1, 1, digest)
        self.assertIs(second_status, HTTPStatus.OK)
        # The whole "b" group — both candidates — stays on one page.
        self.assertEqual([g["key"] for g in second["snapshot"]], ["b"])
        self.assertEqual(len(second["snapshot"][0]["candidates"]), 2)
        self.assertEqual(second["nextCursor"], 2)
        self.assertTrue(second["hasMore"])

        third_status, third = self.store.get_replication_export(2, 1, digest)
        self.assertIs(third_status, HTTPStatus.OK)
        self.assertEqual([g["key"] for g in third["snapshot"]], ["c"])
        self.assertEqual(third["nextCursor"], 3)
        self.assertFalse(third["hasMore"])

        # Digest and counts are identical on every page.
        for page in (first, second, third):
            self.assertEqual(page["digest"], digest)
            self.assertEqual(page["keys"], 3)
            self.assertEqual(page["candidateVersions"], 4)
            self.assertEqual(page["algorithm"], "sha256")

    def test_digest_matches_verification_digest(self) -> None:
        self.add("r1", "o1", "clé", "a\nb→", {"r2": 1, "r1": 2})
        self.assertEqual(export_digest(self.store), self.store.get_verification_digest()["digest"])

    def test_export_is_side_effect_free(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        digest = export_digest(self.store)
        for _ in range(3):
            status, payload = self.store.get_replication_export(0, 100, digest)
            self.assertIs(status, HTTPStatus.OK)
            self.assertEqual(payload["digest"], digest)
        self.assertEqual(self.store.get_metrics()["keys"], 1)
        self.assertEqual(self.store.get_metrics()["candidateVersions"], 1)


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

    def raw_request(self, method: str, path: str, body: object = None):
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

    def request(self, method: str, path: str, body: object = None):
        status, payload, _, _ = self.raw_request(method, path, body)
        return status, payload

    def add(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            operation(operation_id, key, value, clock),
        )

    def digest(self) -> str:
        return export_digest(self.server.store)

    def export(self, after: int, limit: int, digest: str):
        return self.request(
            "GET",
            f"/v1/replication/export?after={after}&limit={limit}&expectedDigest={digest}",
        )

    def test_empty_store_export(self) -> None:
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

    def test_payload_shape_field_order_and_terminator(self) -> None:
        self.add("r1", "op-1", "color", "blue", {"r1": 1})
        self.add("r2", "op-2", "color", "red", {"r2": 1})
        status, payload, raw, headers = self.raw_request(
            "GET",
            f"/v1/replication/export?after=0&limit=100&expectedDigest={self.digest()}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), EXPORT_FIELDS)
        # The contracted field order is preserved (not key-sorted).
        self.assertEqual(
            list(payload),
            [
                "snapshot",
                "nextCursor",
                "hasMore",
                "algorithm",
                "digest",
                "keys",
                "candidateVersions",
            ],
        )
        for name in ("nextCursor", "keys", "candidateVersions"):
            self.assertIs(type(payload[name]), int)
        self.assertRegex(payload["digest"], DIGEST_RE)
        self.assertEqual(payload["algorithm"], "sha256")
        # Compact ordered JSON terminated by exactly one newline.
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")
        ordered = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.assertEqual(raw[:-1], ordered)
        header_map = {name.lower(): value for name, value in headers}
        self.assertEqual(header_map["content-type"], "application/json; charset=utf-8")
        self.assertEqual(header_map["content-length"], str(len(raw)))

    def test_pages_then_stable_empty_tail(self) -> None:
        self.add("r1", "o1", "a", "1", {"r1": 1})
        self.add("r1", "o2", "b", "2", {"r1": 1})
        self.add("r2", "o3", "b", "3", {"r2": 1})
        self.add("r1", "o4", "c", "4", {"r1": 1})
        digest = self.digest()

        status, first = self.export(0, 1, digest)
        self.assertEqual(status, 200)
        self.assertEqual([g["key"] for g in first["snapshot"]], ["a"])
        self.assertEqual(first["nextCursor"], 1)
        self.assertTrue(first["hasMore"])

        status, second = self.export(first["nextCursor"], 1, digest)
        self.assertEqual(status, 200)
        self.assertEqual([g["key"] for g in second["snapshot"]], ["b"])
        self.assertEqual(len(second["snapshot"][0]["candidates"]), 2)
        self.assertEqual(second["nextCursor"], 2)
        self.assertTrue(second["hasMore"])

        status, third = self.export(second["nextCursor"], 1, digest)
        self.assertEqual(status, 200)
        self.assertEqual([g["key"] for g in third["snapshot"]], ["c"])
        self.assertEqual(third["nextCursor"], 3)
        self.assertFalse(third["hasMore"])

        # after equal to the key count is a stable empty page.
        status, tail = self.export(third["nextCursor"], 1, digest)
        self.assertEqual(status, 200)
        self.assertEqual(tail["snapshot"], [])
        self.assertEqual(tail["nextCursor"], 3)
        self.assertFalse(tail["hasMore"])

    def test_mismatched_digest_is_409_and_keeps_page_absent(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        status, payload, raw, _ = self.raw_request(
            "GET",
            "/v1/replication/export?after=0&limit=100&expectedDigest=" + "0" * 64,
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "export_conflict"})
        # The 409 shares the endpoint's single-newline terminator (like
        # the apply endpoint's apply_conflict).
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotEqual(raw[-2:-1], b"\n")

    def test_after_past_count_is_400(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        status, payload = self.export(2, 100, self.digest())
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_malformed_queries_are_400(self) -> None:
        digest = self.digest()
        queries = (
            "",
            "limit=1&expectedDigest=" + digest,
            "after=0&expectedDigest=" + digest,
            "after=0&limit=1",
            "after=0&limit=0&expectedDigest=" + digest,
            "after=0&limit=101&expectedDigest=" + digest,
            "after=-1&limit=1&expectedDigest=" + digest,
            "after=1.0&limit=1&expectedDigest=" + digest,
            "after=%201&limit=1&expectedDigest=" + digest,
            "after=0&limit=1&expectedDigest=" + digest.upper(),
            "after=0&limit=1&expectedDigest=" + "a" * 63,
            "after=0&limit=1&expectedDigest=" + ("a" * 63) + "g",
            "after=0&limit=1&expectedDigest=" + digest + "&x=1",
            "after=0&after=1&limit=1&expectedDigest=" + digest,
            "after=0&limit=1&limit=2&expectedDigest=" + digest,
            "after=0&limit=1",  # missing expectedDigest
        )
        for query in queries:
            status, payload = self.request(
                "GET", "/v1/replication/export?" + query
            )
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_path_shape_mismatches_are_404(self) -> None:
        suffix = f"?after=0&limit=1&expectedDigest={'a' * 64}"
        for path in (
            "/v1/replication/export/",
            "/v1/replication/export/extra",
            "/v1/replication",
            "/v1/replication/exports",
            "/v1/replication/export/extra" + suffix,
        ):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 404, path)
            self.assertEqual(payload, {"error": "not_found"}, path)

    def test_path_shape_check_precedes_query_check(self) -> None:
        # A wrong path shape together with an invalid query is still 404.
        status, payload = self.request(
            "GET", "/v1/replication/export/extra?after=nope"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_post_is_not_routed(self) -> None:
        status, payload = self.request("POST", "/v1/replication/export", {})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_export_does_not_mutate_state(self) -> None:
        self.add("r1", "o1", "k", "v", {"r1": 1})
        digest = self.digest()
        for _ in range(3):
            status, payload = self.export(0, 100, digest)
            self.assertEqual(status, 200)
            self.assertEqual(payload["digest"], digest)
        status, sync_payload = self.request("GET", "/v1/sync/operations")
        self.assertEqual(status, 200)
        self.assertEqual(len(sync_payload["operations"]), 1)


class PersistentExportHttpServerTests(unittest.TestCase):
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

    def request(self, server, method, path, body=None):
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

    def test_pages_and_digest_survive_restart(self) -> None:
        server = self.start_server()
        for replica, op_id, key, value in (
            ("r1", "o1", "a", "1"),
            ("r1", "o2", "b", "2"),
            ("r2", "o3", "b", "3"),
        ):
            status, _ = self.request(
                server,
                "POST",
                f"/v1/replicas/{replica}/operations",
                operation(op_id, key, value, {replica: 1}),
            )
            self.assertEqual(status, 201)
        status, digest_payload = self.request(
            server, "GET", "/v1/verification/digest"
        )
        self.assertEqual(status, 200)
        digest = digest_payload["digest"]

        pages = []
        cursor = 0
        while True:
            status, page = self.request(
                server,
                "GET",
                f"/v1/replication/export?after={cursor}&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
            pages.append(page)
            cursor = page["nextCursor"]
            if not page["hasMore"]:
                break
        server.shutdown()
        server.server_close()

        server = self.start_server()
        cursor = 0
        restarted_pages = []
        while True:
            status, page = self.request(
                server,
                "GET",
                f"/v1/replication/export?after={cursor}&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
            restarted_pages.append(page)
            cursor = page["nextCursor"]
            if not page["hasMore"]:
                break
        self.assertEqual(restarted_pages, pages)

    def test_export_writes_nothing_to_disk(self) -> None:
        server = self.start_server()
        self.request(
            server,
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
        )
        status, digest_payload = self.request(
            server, "GET", "/v1/verification/digest"
        )
        digest = digest_payload["digest"]
        before_bytes = Path(self.data_file).read_bytes()
        before_stat = Path(self.data_file).stat()
        for _ in range(5):
            status, _ = self.request(
                server,
                "GET",
                f"/v1/replication/export?after=0&limit=1&expectedDigest={digest}",
            )
            self.assertEqual(status, 200)
        self.assertEqual(Path(self.data_file).read_bytes(), before_bytes)
        after_stat = Path(self.data_file).stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)


class ExportAuthTests(unittest.TestCase):
    """Authentication and scope behavior matches the shared contract."""

    def start(self, **kwargs) -> SemanticStateServer:
        server = SemanticStateServer(("127.0.0.1", 0), RequestHandler, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def raw_get(self, server, path, headers=None, duplicate_auth=False):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        if duplicate_auth:
            conn.putrequest("GET", path)
            conn.putheader("Authorization", "Bearer s3cret")
            conn.putheader("Authorization", "Bearer s3cret")
            conn.endheaders()
        else:
            conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        www = response.getheader("WWW-Authenticate")
        conn.close()
        return response.status, payload, www

    def path(self, digest: str = EMPTY_CANDIDATE_DIGEST) -> str:
        return f"/v1/replication/export?after=0&limit=1&expectedDigest={digest}"

    def test_single_token_mode(self) -> None:
        server = self.start(auth_token="s3cret")
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "s3cret"},
        ):
            status, payload, www = self.raw_get(server, self.path(), headers)
            self.assertEqual(status, 401, headers)
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertEqual(www, "Bearer")
        status, payload, www = self.raw_get(
            server, self.path(), {"Authorization": "Bearer s3cret"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), EXPORT_FIELDS)
        # Health stays anonymous.
        self.assertEqual(self.raw_get(server, "/health")[0], 200)

    def test_duplicate_authorization_is_401(self) -> None:
        server = self.start(auth_token="s3cret")
        status, _, www = self.raw_get(server, self.path(), duplicate_auth=True)
        self.assertEqual(status, 401)
        self.assertEqual(www, "Bearer")

    def test_scope_policy_mode(self) -> None:
        server = self.start(
            auth_scopes={
                "reader": frozenset({"read"}),
                "writer": frozenset({"write"}),
                "admin": frozenset({"read", "write", "admin"}),
            }
        )
        path = self.path()
        # Missing credential is 401 with a challenge.
        status, _, www = self.raw_get(server, path)
        self.assertEqual(status, 401)
        self.assertEqual(www, "Bearer")
        # A write-only token is 403 without a challenge.
        status, payload, www = self.raw_get(
            server, path, {"Authorization": "Bearer writer"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertIsNone(www)
        # read and admin tokens reach the endpoint.
        for token in ("reader", "admin"):
            status, payload, _ = self.raw_get(
                server, path, {"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(status, 200, token)
            self.assertEqual(set(payload), EXPORT_FIELDS)


if __name__ == "__main__":
    unittest.main()
