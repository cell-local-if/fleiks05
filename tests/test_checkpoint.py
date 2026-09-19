"""HTTP, concurrency, persistence-failure, and recovery tests for peer checkpoints.

The checkpoint endpoints are::

    POST /v1/sync/peers/{peerId}/checkpoint   {"cursor": N}
    GET  /v1/sync/peers/{peerId}/checkpoint

Everything here goes through the real HTTP entry point
(``SemanticStateServer`` + a request thread); only the Python standard
library is used.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
    load_data_document,
    load_data_file,
)


def operation(operation_id: str, key: str, value: str, clock: dict) -> dict:
    return {"operationId": operation_id, "key": key, "value": value, "clock": clock}


def record(replica: str, op: dict) -> dict:
    return {"replicaId": replica, "operation": op}


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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, object]:
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
        payload = json.loads(raw.decode("utf-8")) if raw else None
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, op: dict) -> tuple[int, object]:
        return self.request("POST", f"/v1/replicas/{replica}/operations", op)

    def post_checkpoint(self, peer: str, body: object) -> tuple[int, object]:
        return self.request("POST", f"/v1/sync/peers/{peer}/checkpoint", body)

    def get_checkpoint(self, peer: str, query: str = "") -> tuple[int, object]:
        return self.request("GET", f"/v1/sync/peers/{peer}/checkpoint{query}")


class CheckpointRegistrationTests(HttpServerTestCase):
    def test_first_registration_is_200_with_zero_cursor_on_empty_log(self) -> None:
        status, payload = self.post_checkpoint("peer-a", {"cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 0})

    def test_register_advance_and_get_round_trip(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k", "v2", {"r1": 2}))

        status, payload = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})

        status, payload = self.get_checkpoint("peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})

        status, payload = self.post_checkpoint("peer-a", {"cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 2})
        status, payload = self.get_checkpoint("peer-a")
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 2})

    def test_cursor_may_equal_the_accepted_log_length(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, _ = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 200)

    def test_same_value_replay_is_200_and_changes_nothing(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.assertEqual(self.post_checkpoint("peer-a", {"cursor": 1})[0], 200)
        status, payload = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        _, current = self.get_checkpoint("peer-a")
        self.assertEqual(current, {"peerId": "peer-a", "cursor": 1})

    def test_regression_is_409_and_stored_cursor_is_kept(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k", "v2", {"r1": 2}))
        self.assertEqual(self.post_checkpoint("peer-a", {"cursor": 2})[0], 200)

        status, payload = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        # The stored checkpoint did not regress.
        _, current = self.get_checkpoint("peer-a")
        self.assertEqual(current, {"peerId": "peer-a", "cursor": 2})

    def test_checkpoints_are_independent_per_peer(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r1", operation("o2", "k", "v2", {"r1": 2}))
        self.assertEqual(self.post_checkpoint("peer-a", {"cursor": 2})[0], 200)
        self.assertEqual(self.post_checkpoint("peer-b", {"cursor": 1})[0], 200)
        _, a = self.get_checkpoint("peer-a")
        _, b = self.get_checkpoint("peer-b")
        self.assertEqual(a, {"peerId": "peer-a", "cursor": 2})
        self.assertEqual(b, {"peerId": "peer-b", "cursor": 1})
        # A regression on one peer does not affect the other.
        self.assertEqual(self.post_checkpoint("peer-b", {"cursor": 0})[0], 409)
        _, a = self.get_checkpoint("peer-a")
        self.assertEqual(a["cursor"], 2)

    def test_peer_id_is_url_decoded(self) -> None:
        status, payload = self.post_checkpoint("peer%20a", {"cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer a", "cursor": 0})
        status, payload = self.get_checkpoint("peer%20a")
        self.assertEqual(payload, {"peerId": "peer a", "cursor": 0})


class CheckpointValidationTests(HttpServerTestCase):
    def test_invalid_bodies_are_400(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        bad_bodies = [
            b"{not json",
            b"",
            [],
            {},
            {"cursor": 0, "extra": 1},
            {"cursor": 0, "peerId": "peer-a"},
            {"Cursor": 0},
            {"cursor": True},
            {"cursor": False},
            {"cursor": "1"},
            {"cursor": 1.0},
            {"cursor": -1},
            {"cursor": None},
            {"cursor": []},
            {"cursor": {}},
        ]
        for body in bad_bodies:
            status, payload = self.post_checkpoint("peer-a", body)
            self.assertEqual(status, 400, repr(body))
            self.assertEqual(payload, {"error": "invalid_request"}, repr(body))
        # Nothing was registered by the rejected bodies.
        status, payload = self.get_checkpoint("peer-a")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_cursor_past_accepted_log_is_400(self) -> None:
        status, payload = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        status, _ = self.post_checkpoint("peer-a", {"cursor": 2})
        self.assertEqual(status, 400)
        status, _ = self.post_checkpoint("peer-a", {"cursor": 1})
        self.assertEqual(status, 200)
        # The bound tracks new commits.
        self.post_operation("r1", operation("o2", "k", "v2", {"r1": 2}))
        status, _ = self.post_checkpoint("peer-a", {"cursor": 2})
        self.assertEqual(status, 200)

    def test_cursor_bound_counts_imported_operations(self) -> None:
        body = {
            "operations": [
                record("r1", operation("o1", "k", "v1", {"r1": 1})),
                record("r2", operation("o2", "k", "v2", {"r2": 1})),
            ]
        }
        status, _ = self.request("POST", "/v1/sync/operations", body)
        self.assertEqual(status, 201)
        status, _ = self.post_checkpoint("peer-a", {"cursor": 2})
        self.assertEqual(status, 200)
        status, _ = self.post_checkpoint("peer-a", {"cursor": 3})
        self.assertEqual(status, 400)

    def test_get_unregistered_peer_is_404(self) -> None:
        status, payload = self.get_checkpoint("nobody")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_with_any_query_parameter_is_400(self) -> None:
        self.post_checkpoint("peer-a", {"cursor": 0})
        for query in ("?cursor=0", "?x=1", "?x", "?x=", "?x=1&x=2"):
            status, payload = self.get_checkpoint("peer-a", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload, {"error": "invalid_request"}, query)

    def test_empty_peer_segment_is_404(self) -> None:
        status, payload = self.request("POST", "/v1/sync/peers//checkpoint", {"cursor": 0})
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        status, payload = self.request("GET", "/v1/sync/peers//checkpoint")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_unknown_checkpoint_routes_are_404(self) -> None:
        for method in ("GET", "POST"):
            for path in (
                "/v1/sync/peers",
                "/v1/sync/peers/peer-a",
                "/v1/sync/peers/peer-a/checkpoint/extra",
                "/v1/sync/peers/peer-a/cursor",
            ):
                body = {"cursor": 0} if method == "POST" else None
                status, payload = self.request(method, path, body)
                self.assertEqual(status, 404, (method, path))
                self.assertEqual(payload, {"error": "not_found"}, (method, path))


class CheckpointIsolationTests(HttpServerTestCase):
    """A checkpoint is not an operation: every existing surface is unchanged."""

    def test_checkpoint_does_not_touch_log_state_audit_or_metrics(self) -> None:
        self.post_operation("r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation("r2", operation("o2", "k", "v2", {"r2": 1}))
        _, before_metrics = self.request("GET", "/v1/metrics")
        _, before_sync = self.request("GET", "/v1/sync/operations")
        _, before_audit = self.request("GET", "/v1/audit/keys/k/operations")
        _, before_state = self.request("GET", "/v1/states/k")

        self.assertEqual(self.post_checkpoint("peer-a", {"cursor": 2})[0], 200)
        self.assertEqual(self.get_checkpoint("peer-a")[0], 200)

        _, after_metrics = self.request("GET", "/v1/metrics")
        _, after_sync = self.request("GET", "/v1/sync/operations")
        _, after_audit = self.request("GET", "/v1/audit/keys/k/operations")
        _, after_state = self.request("GET", "/v1/states/k")
        self.assertEqual(after_metrics, before_metrics)
        self.assertEqual(after_metrics["acceptedOperations"], 2)
        self.assertEqual(after_sync, before_sync)
        self.assertEqual(after_sync["nextCursor"], 2)
        self.assertEqual(after_audit, before_audit)
        self.assertEqual(after_state, before_state)


class CheckpointConcurrencyTests(HttpServerTestCase):
    def test_concurrent_posts_and_writes_keep_one_consistent_commit(self) -> None:
        thread_count = 8
        errors: list[BaseException] = []
        results: list[tuple[int, int]] = []
        results_lock = threading.Lock()

        def worker(index: int) -> None:
            try:
                status, _ = self.post_operation(
                    f"r{index}", operation(f"op-{index}", "k", f"v{index}", {f"r{index}": 1})
                )
                assert status == 201
                # Each worker registers its own peer and races one shared peer.
                status, _ = self.post_checkpoint(f"peer-{index}", {"cursor": index + 1})
                assert status == 200
                status, _ = self.post_checkpoint("shared", {"cursor": index + 1})
                assert status in (200, 409)
                with results_lock:
                    results.append((status, index + 1))
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])

        # The shared peer ends at the largest cursor ever posted; every 200
        # response advanced or replayed, every 409 lost to a bigger cursor.
        _, shared = self.get_checkpoint("shared")
        self.assertEqual(shared, {"peerId": "shared", "cursor": thread_count})
        for status, cursor in results:
            if status == 409:
                self.assertLess(cursor, thread_count)
        # Each private peer holds exactly its own cursor.
        for i in range(thread_count):
            _, mine = self.get_checkpoint(f"peer-{i}")
            self.assertEqual(mine, {"peerId": f"peer-{i}", "cursor": i + 1})
        # The log itself is untouched by all of this.
        _, sync = self.request("GET", "/v1/sync/operations")
        self.assertEqual(len(sync["operations"]), thread_count)


class PersistentCheckpointTestCase(unittest.TestCase):
    """Checkpoints against a data-file-backed server with real HTTP."""

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

    def post_operation(self, server: SemanticStateServer, replica: str, op: dict) -> int:
        return self.request(server, "POST", f"/v1/replicas/{replica}/operations", op)[0]

    def post_checkpoint(self, server: SemanticStateServer, peer: str, cursor: int):
        return self.request(server, "POST", f"/v1/sync/peers/{peer}/checkpoint", {"cursor": cursor})

    def get_checkpoint(self, server: SemanticStateServer, peer: str):
        return self.request(server, "GET", f"/v1/sync/peers/{peer}/checkpoint")


class PersistentCheckpointTests(PersistentCheckpointTestCase):
    def test_checkpoint_is_durable_before_the_200_response(self) -> None:
        server = self.start_server()
        self.post_operation(server, "r1", operation("o1", "k", "v1", {"r1": 1}))
        status, payload = self.post_checkpoint(server, "peer-a", 1)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        # The file already contains the checkpoint next to the operation log.
        records, checkpoints = load_data_document(str(self.data_file))
        self.assertEqual(len(records), 1)
        self.assertEqual(checkpoints, {"peer-a": 1})
        # The operation-only view of the file is unchanged in shape.
        self.assertEqual(len(load_data_file(str(self.data_file))), 1)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_restart_preserves_checkpoints_and_existing_semantics(self) -> None:
        server = self.start_server()
        self.post_operation(server, "r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation(server, "r1", operation("o2", "k", "v2", {"r1": 2}))
        self.assertEqual(self.post_checkpoint(server, "peer-a", 2)[0], 200)
        self.assertEqual(self.post_checkpoint(server, "peer-b", 1)[0], 200)
        server.shutdown()
        server.server_close()

        server = self.start_server()
        status, payload = self.get_checkpoint(server, "peer-a")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 2})
        _, payload = self.get_checkpoint(server, "peer-b")
        self.assertEqual(payload, {"peerId": "peer-b", "cursor": 1})
        # Regression is still rejected after the restart.
        status, payload = self.post_checkpoint(server, "peer-a", 1)
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "checkpoint_conflict"})
        # The accepted log and its semantics survived alongside.
        _, state = self.request(server, "GET", "/v1/states/k")
        self.assertEqual(state["value"], "v2")
        _, sync = self.request(server, "GET", "/v1/sync/operations")
        self.assertEqual([e["operation"]["operationId"] for e in sync["operations"]], ["o1", "o2"])
        # The bound still tracks the recovered log length.
        status, _ = self.post_checkpoint(server, "peer-c", 3)
        self.assertEqual(status, 400)
        status, _ = self.post_checkpoint(server, "peer-c", 2)
        self.assertEqual(status, 200)

    def test_version1_file_without_checkpoints_recovers_with_none(self) -> None:
        document = {
            "version": 1,
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})}
            ],
        }
        self.data_file.write_text(json.dumps(document), encoding="utf-8")
        server = self.start_server()
        status, payload = self.get_checkpoint(server, "peer-a")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})
        # Registering against the recovered log works and rewrites the file
        # in the supplemented format.
        status, _ = self.post_checkpoint(server, "peer-a", 1)
        self.assertEqual(status, 200)
        _, checkpoints = load_data_document(str(self.data_file))
        self.assertEqual(checkpoints, {"peer-a": 1})

    def test_persistence_failure_is_500_and_changes_nothing(self) -> None:
        server = self.start_server()
        self.post_operation(server, "r1", operation("o1", "k", "v1", {"r1": 1}))
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.post_checkpoint(server, "peer-a", 1)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # The failed registration is not visible in memory.
            status, payload = self.get_checkpoint(server, "peer-a")
            self.assertEqual(status, 404)
            self.assertEqual(payload, {"error": "not_found"})

        # The file is byte-identical and the request can be retried.
        self.assertEqual(self.data_file.read_bytes(), before)
        status, payload = self.post_checkpoint(server, "peer-a", 1)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        leftovers = [p.name for p in self.tmp.iterdir() if p.name != self.data_file.name]
        self.assertEqual(leftovers, [])

    def test_failed_advance_keeps_previous_checkpoint_and_file(self) -> None:
        server = self.start_server()
        self.post_operation(server, "r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation(server, "r1", operation("o2", "k", "v2", {"r1": 2}))
        self.assertEqual(self.post_checkpoint(server, "peer-a", 1)[0], 200)
        before = self.data_file.read_bytes()

        with patch.object(
            StateStore, "_persist_locked", side_effect=PersistenceError("disk gone")
        ):
            status, payload = self.post_checkpoint(server, "peer-a", 2)
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            # The previous checkpoint is still the visible one.
            _, current = self.get_checkpoint(server, "peer-a")
            self.assertEqual(current, {"peerId": "peer-a", "cursor": 1})
            # A same-value replay needs no durable write and still succeeds.
            status, _ = self.post_checkpoint(server, "peer-a", 1)
            self.assertEqual(status, 200)

        self.assertEqual(self.data_file.read_bytes(), before)
        _, checkpoints = load_data_document(str(self.data_file))
        self.assertEqual(checkpoints, {"peer-a": 1})
        # The advance commits cleanly once persistence works again.
        status, _ = self.post_checkpoint(server, "peer-a", 2)
        self.assertEqual(status, 200)
        _, checkpoints = load_data_document(str(self.data_file))
        self.assertEqual(checkpoints, {"peer-a": 2})

    def test_conflict_and_replay_do_not_rewrite_the_file(self) -> None:
        server = self.start_server()
        self.post_operation(server, "r1", operation("o1", "k", "v1", {"r1": 1}))
        self.post_operation(server, "r1", operation("o2", "k", "v2", {"r1": 2}))
        self.assertEqual(self.post_checkpoint(server, "peer-a", 2)[0], 200)
        before = self.data_file.read_bytes()

        self.assertEqual(self.post_checkpoint(server, "peer-a", 2)[0], 200)
        self.assertEqual(self.post_checkpoint(server, "peer-a", 1)[0], 409)
        self.assertEqual(self.data_file.read_bytes(), before)


class CheckpointDataFileValidationTests(PersistentCheckpointTestCase):
    def assert_rejected(self, document: dict) -> None:
        self.data_file.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(PersistenceError):
            StateStore(data_file=str(self.data_file))

    def valid_document(self, checkpoints: object) -> dict:
        return {
            "version": 1,
            "operations": [
                {"replicaId": "r1", "operation": operation("o1", "k", "v1", {"r1": 1})}
            ],
            "checkpoints": checkpoints,
        }

    def test_checkpoints_must_be_a_list(self) -> None:
        self.assert_rejected(self.valid_document({}))
        self.assert_rejected(self.valid_document({"peer-a": 1}))

    def test_checkpoint_record_shape_is_enforced(self) -> None:
        self.assert_rejected(self.valid_document([{"peerId": "p"}]))
        self.assert_rejected(self.valid_document([{"cursor": 0}]))
        self.assert_rejected(self.valid_document([{"peerId": "p", "cursor": 0, "x": 1}]))
        self.assert_rejected(self.valid_document([{"peerId": "", "cursor": 0}]))
        self.assert_rejected(self.valid_document([{"peerId": 42, "cursor": 0}]))
        self.assert_rejected(self.valid_document([{"peerId": "p", "cursor": -1}]))
        self.assert_rejected(self.valid_document([{"peerId": "p", "cursor": True}]))
        self.assert_rejected(self.valid_document([{"peerId": "p", "cursor": "1"}]))

    def test_duplicate_peer_checkpoints_are_rejected(self) -> None:
        self.assert_rejected(
            self.valid_document(
                [{"peerId": "p", "cursor": 0}, {"peerId": "p", "cursor": 1}]
            )
        )

    def test_checkpoint_past_the_accepted_log_is_rejected(self) -> None:
        self.assert_rejected(self.valid_document([{"peerId": "p", "cursor": 2}]))

    def test_unknown_root_members_are_still_rejected(self) -> None:
        document = self.valid_document([])
        document["extra"] = 1
        self.assert_rejected(document)

    def test_valid_checkpoints_load(self) -> None:
        document = self.valid_document(
            [{"peerId": "peer-a", "cursor": 1}, {"peerId": "peer-b", "cursor": 0}]
        )
        self.data_file.write_text(json.dumps(document), encoding="utf-8")
        store = StateStore(data_file=str(self.data_file))
        status, payload = store.get_checkpoint("peer-a")
        self.assertEqual(payload, {"peerId": "peer-a", "cursor": 1})
        _, payload = store.get_checkpoint("peer-b")
        self.assertEqual(payload, {"peerId": "peer-b", "cursor": 0})


if __name__ == "__main__":
    unittest.main()
