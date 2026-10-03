"""Tests for the read-only full-store export endpoint.

The endpoint is::

    GET /v1/admin/store/export

It answers the complete version-1 recovery document — ``version`` plus
the ``operations``, ``checkpoints``, ``policies``, ``transactions``,
``acks``, ``repairExecutions``, ``policyEvents``, and ``compensations``
sections — as compact UTF-8 JSON with one trailing newline, assembled
from one committed snapshot. The query is strictly read-only: it never
reads or rewrites the data file, creates no temporary file, advances no
cursor, and changes no idempotence decision.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    StateStore,
)

EXPORT_PATH = "/v1/admin/store/export"

READ_TOKEN = "reader-token"
WRITE_TOKEN = "writer-token"
ADMIN_TOKEN = "admin-token"
RW_TOKEN = "read-write-token"
SINGLE_TOKEN = "the-one-token"

POLICY = {
    READ_TOKEN: frozenset({"read"}),
    WRITE_TOKEN: frozenset({"write"}),
    ADMIN_TOKEN: frozenset({"read", "write", "admin"}),
    RW_TOKEN: frozenset({"read", "write"}),
}

EMPTY_DOCUMENT = {
    "version": 1,
    "operations": [],
    "checkpoints": {},
    "policies": [],
    "transactions": [],
    "acks": [],
    "repairExecutions": [],
    "policyEvents": [],
    "compensations": [],
}


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


class ServerTestCase(unittest.TestCase):
    """Base harness: a running server plus a raw HTTP helper."""

    server_kwargs: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.data_file = os.path.join(cls.tmp.name, "state.json")
        kwargs = dict(cls.server_kwargs)
        if kwargs.pop("with_data_file", False):
            kwargs["data_file"] = cls.data_file
        cls.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, **kwargs
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        token: str | None = None,
    ) -> tuple[int, dict, dict, bytes]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is None:
            conn.request(method, path, headers=headers)
        else:
            conn.request(method, path, body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else None
        response_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload, response_headers, raw

    def export(self, token: str | None = None) -> tuple[int, dict, dict, bytes]:
        return self.request("GET", EXPORT_PATH, token=token)

    def post_operation(self, replica: str, op: dict) -> int:
        status, _, _, _ = self.request("POST", f"/v1/replicas/{replica}/operations", op)
        return status


class EmptyStoreExportTests(ServerTestCase):
    """An untouched store exports every section as empty."""

    def test_empty_store_exports_the_empty_document(self) -> None:
        status, payload, headers, raw = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(payload, EMPTY_DOCUMENT)
        # Exactly the nine contracted root keys, nothing internal.
        self.assertEqual(set(payload.keys()), set(EMPTY_DOCUMENT.keys()))

    def test_body_is_compact_json_with_one_trailing_newline(self) -> None:
        status, _, _, raw = self.export()
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw[:-1])
        self.assertNotIn(b" ", raw)
        self.assertEqual(
            raw, json.dumps(EMPTY_DOCUMENT, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        )

    def test_repeated_exports_are_byte_identical(self) -> None:
        first = self.export()[3]
        second = self.export()[3]
        self.assertEqual(first, second)


class PopulatedStoreExportTests(ServerTestCase):
    """A committed history exports every section in its stable order."""

    server_kwargs = {"with_data_file": True}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.expected_identities: list[tuple[str, str]] = []

    def setUp(self) -> None:
        if self.server.store._accepted:  # populated once, shared by the class
            return
        self._populate()

    def _populate(self) -> None:
        # Two concurrent writes on k leave it in conflict.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1})), 201
        )
        self.assertEqual(
            self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1})), 201
        )
        # An automatic resolution commits an operation with a policy binding.
        status, _, _, _ = self.request(
            "POST",
            "/v1/states/k/resolve/auto",
            {
                "replicaId": "r3",
                "operationId": "a1",
                "clock": {"r1": 1, "r2": 1, "r3": 1},
                "policy": "lowest_identity",
            },
        )
        self.assertEqual(status, 201)
        # A seed on k2 gives the transaction a before-candidate to restore.
        self.assertEqual(
            self.post_operation("r2", operation("o3", "k2", "old", {"r2": 2})), 201
        )
        # An atomic transaction binds its entries under the transaction id.
        status, payload, _, _ = self.request(
            "POST",
            "/v1/transactions/apply",
            {
                "transactionId": "tx-1",
                "operations": [
                    {
                        "key": "k2",
                        "replicaId": "r1",
                        "operationId": "t1",
                        "value": "new",
                        "clock": {"r1": 2, "r2": 2},
                        "candidates": [{"replicaId": "r2", "operationId": "o3"}],
                    }
                ],
            },
        )
        self.assertEqual(status, 201, payload)
        # A verifiable compensation of that transaction.
        status, plan, _, _ = self.request("GET", "/v1/transactions/tx-1/compensation")
        self.assertEqual(status, 200)
        self.assertEqual(plan["conclusion"], "reversible")
        status, payload, _, _ = self.request(
            "POST",
            "/v1/transactions/tx-1/compensate",
            {
                "compensationId": "c-1",
                "expectedPlanDigest": plan["expectedPlanDigest"],
                "operations": [item["operation"] for item in plan["keys"]],
            },
        )
        self.assertEqual(status, 201, payload)
        # A registered checkpoint, then a receipt covering the whole log.
        status, _, _, _ = self.request(
            "POST", "/v1/sync/peers/peer-a/checkpoint", {"cursor": 0}
        )
        self.assertEqual(status, 200)
        status, page, _, _ = self.request("GET", "/v1/sync/operations?limit=100")
        self.assertEqual(status, 200)
        identities = [
            {"replicaId": r["replicaId"], "operationId": r["operation"]["operationId"]}
            for r in page["operations"]
        ]
        type(self).expected_identities = [
            (item["replicaId"], item["operationId"]) for item in identities
        ]
        status, payload, _, _ = self.request(
            "POST",
            "/v1/sync/peers/peer-a/acknowledge",
            {"ackId": "ack-1", "cursor": len(identities), "operations": identities},
        )
        self.assertEqual(status, 201, payload)
        # One scope-policy change event, committed straight to the store.
        event = self.server.store.record_policy_reload("a" * 64, 2)
        self.assertEqual(event, {"sequence": 1, "digest": "a" * 64, "tokens": 2})

    def test_export_matches_the_data_file_document(self) -> None:
        status, payload, _, raw = self.export()
        self.assertEqual(status, 200)
        with open(self.data_file, "rb") as handle:
            persisted = json.loads(handle.read().decode("utf-8"))
        self.assertEqual(payload, persisted)
        # The export body is the persisted document plus one newline.
        self.assertEqual(
            raw,
            json.dumps(persisted, separators=(",", ":"), sort_keys=True).encode() + b"\n",
        )

    def test_operations_follow_the_shared_commit_order(self) -> None:
        status, payload, _, _ = self.export()
        self.assertEqual(status, 200)
        identities = [
            (record["replicaId"], record["operation"]["operationId"])
            for record in payload["operations"]
        ]
        self.assertEqual(identities, self.expected_identities)
        for record in payload["operations"]:
            self.assertEqual(set(record.keys()), {"replicaId", "operation"})
            self.assertEqual(
                set(record["operation"].keys()), {"operationId", "key", "value", "clock"}
            )

    def test_local_bindings_are_exported_in_their_sections(self) -> None:
        status, payload, _, _ = self.export()
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["policies"],
            [{"replicaId": "r3", "operationId": "a1", "policy": "lowest_identity"}],
        )
        self.assertEqual(len(payload["transactions"]), 1)
        self.assertEqual(payload["transactions"][0]["transactionId"], "tx-1")
        self.assertEqual(
            set(payload["transactions"][0].keys()), {"transactionId", "operations"}
        )
        self.assertEqual(payload["checkpoints"], {"peer-a": len(self.expected_identities)})
        self.assertEqual(len(payload["acks"]), 1)
        ack = payload["acks"][0]
        self.assertEqual(set(ack.keys()), {"peerId", "ackId", "cursor", "operations"})
        self.assertEqual((ack["peerId"], ack["ackId"]), ("peer-a", "ack-1"))
        self.assertEqual(ack["cursor"], len(self.expected_identities))
        self.assertEqual(
            [(item["replicaId"], item["operationId"]) for item in ack["operations"]],
            self.expected_identities,
        )
        self.assertEqual(
            payload["policyEvents"], [{"sequence": 1, "digest": "a" * 64, "tokens": 2}]
        )
        self.assertEqual(len(payload["compensations"]), 1)
        compensation = payload["compensations"][0]
        self.assertEqual(
            set(compensation.keys()),
            {"compensationId", "transactionId", "expectedPlanDigest", "operations", "status"},
        )
        self.assertEqual(compensation["compensationId"], "c-1")
        self.assertEqual(compensation["transactionId"], "tx-1")
        self.assertEqual(compensation["status"], "committed")
        self.assertEqual(payload["repairExecutions"], [])

    def test_export_does_not_touch_the_data_file_or_leave_temp_files(self) -> None:
        with open(self.data_file, "rb") as handle:
            before = handle.read()
        before_stat = os.stat(self.data_file)
        status, _, _, _ = self.export()
        self.assertEqual(status, 200)
        with open(self.data_file, "rb") as handle:
            after = handle.read()
        self.assertEqual(before, after)
        after_stat = os.stat(self.data_file)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(after_stat.st_ctime_ns, before_stat.st_ctime_ns)
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name != os.path.basename(self.data_file)
        ]
        self.assertEqual(leftovers, [])

    def test_export_is_read_only_for_cursors_and_idempotence(self) -> None:
        first = self.export()[3]
        self.export()
        # A replayed operation is still a replay: the export changed no
        # idempotence decision and appended nothing.
        self.assertEqual(
            self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1})), 200
        )
        third = self.export()[3]
        self.assertEqual(first, third)

    def test_export_is_identical_across_a_restart(self) -> None:
        before = self.export()[3]
        self.assertEqual(before.endswith(b"\n"), True)
        reloaded = StateStore(data_file=self.data_file)
        try:
            self.server.store, previous = reloaded, self.server.store
            after = self.export()[3]
        finally:
            self.server.store = previous
        self.assertEqual(before, after)

    def test_export_serves_as_a_recovery_file_sample(self) -> None:
        raw = self.export()[3]
        sample = os.path.join(self.tmp.name, "sample.json")
        with open(sample, "wb") as handle:
            handle.write(raw)
        # The export (minus its trailing newline) is a valid version-1
        # recovery file and recovers the same committed state.
        recovered = StateStore(data_file=sample)
        try:
            self.assertEqual(
                recovered.get_store_export(), self.server.store.get_store_export()
            )
        finally:
            del recovered
        os.unlink(sample)


class SeededRepairExportTests(unittest.TestCase):
    """A recovered repair-execution binding exports field by field."""

    def test_recovered_repair_execution_is_exported(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data_file = os.path.join(tmp.name, "state.json")
        document = {
            "version": 1,
            "operations": [
                {
                    "replicaId": "r1",
                    "operation": operation("o1", "k", "v", {"r1": 1}),
                }
            ],
            "checkpoints": {"peer-a": 1},
            "repairExecutions": [
                {
                    "peerId": "peer-a",
                    "ackId": "ack-1",
                    "expectedCheckpoint": 0,
                    "expectedReceipts": "b" * 64,
                    "suggestions": [
                        {
                            "action": "correct_cursor",
                            "ackId": "ack-1",
                            "location": {"start": 0, "end": 0},
                            "target": {"start": 1, "end": 1},
                        }
                    ],
                    "results": [
                        {"action": "correct_cursor", "boundary": {"start": 1, "end": 1}}
                    ],
                    "cursor": 1,
                }
            ],
        }
        with open(data_file, "wb") as handle:
            handle.write(json.dumps(document).encode("utf-8"))
        server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=data_file
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        conn.request("GET", EXPORT_PATH)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["repairExecutions"], document["repairExecutions"])
        self.assertEqual(payload["checkpoints"], {"peer-a": 1})
        # Every other section recovers empty and exports empty.
        self.assertEqual(payload["policies"], [])
        self.assertEqual(payload["transactions"], [])
        self.assertEqual(payload["acks"], [])
        self.assertEqual(payload["policyEvents"], [])
        self.assertEqual(payload["compensations"], [])


class StoreExportRouteTests(ServerTestCase):
    """Path-shape and query-string rejections change no state."""

    def assert_not_found(self, path: str) -> None:
        status, payload, _, _ = self.request("GET", path)
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_missing_extra_and_trailing_segments_are_404(self) -> None:
        self.assert_not_found("/v1/admin/store")
        self.assert_not_found("/v1/admin/store/export/extra")
        self.assert_not_found("/v1/admin/store/export/")
        self.assert_not_found("/v1/admin")
        # A query string does not rescue a wrong path shape.
        self.assert_not_found("/v1/admin/store/export/extra?x=1")

    def test_any_query_parameter_is_400(self) -> None:
        for query in ("x=1", "x=1&x=2", "x=", "x", "=1", "after=0", "limit=100"):
            with self.subTest(query=query):
                status, payload, _, _ = self.request("GET", f"{EXPORT_PATH}?{query}")
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "invalid_request"})

    def test_rejections_change_no_state(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v", {"r1": 1}))
        before = self.export()[3]
        self.assert_not_found("/v1/admin/store/export/")
        status, _, _, _ = self.request("GET", f"{EXPORT_PATH}?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(self.export()[3], before)


class StoreExportSingleTokenAuthTests(ServerTestCase):
    """The legacy single token keeps full access; bad credentials are 401."""

    server_kwargs = {"auth_token": SINGLE_TOKEN}

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.export(token=token)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_the_configured_token_exports(self) -> None:
        status, payload, _, _ = self.export(token=SINGLE_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_DOCUMENT)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


class StoreExportScopePolicyAuthTests(ServerTestCase):
    """In scope-policy mode the export requires the admin scope."""

    server_kwargs = {"auth_scopes": dict(POLICY)}

    def test_missing_and_bad_credentials_are_401_with_challenge(self) -> None:
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                status, payload, headers, _ = self.export(token=token)
                self.assertEqual(status, 401)
                self.assertEqual(payload, {"error": "unauthorized"})
                self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_read_and_write_tokens_are_403_without_challenge(self) -> None:
        for token in (READ_TOKEN, WRITE_TOKEN, RW_TOKEN):
            with self.subTest(token=token):
                status, payload, headers, _ = self.export(token=token)
                self.assertEqual(status, 403)
                self.assertEqual(payload, {"error": "forbidden"})
                self.assertNotIn("WWW-Authenticate", headers)
                self.assertNotIn("Www-Authenticate", headers)

    def test_admin_token_exports(self) -> None:
        status, payload, _, _ = self.export(token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload, EMPTY_DOCUMENT)

    def test_forbidden_requests_never_observe_the_snapshot(self) -> None:
        # A committed operation must not leak through a rejected request:
        # the 403 body is the bare error, and a later admin export is the
        # only observation of the state.
        status, _, _, _ = self.request(
            "POST",
            "/v1/replicas/r1/operations",
            operation("o1", "k", "v", {"r1": 1}),
            token=ADMIN_TOKEN,
        )
        self.assertEqual(status, 201)
        status, payload, _, raw = self.export(token=READ_TOKEN)
        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "forbidden"})
        self.assertNotIn(b"o1", raw)
        status, payload, _, _ = self.export(token=ADMIN_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["operations"]), 1)

    def test_health_stays_anonymous(self) -> None:
        status, payload, _, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})


if __name__ == "__main__":
    unittest.main()
