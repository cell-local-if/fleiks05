import http.client
import json
import os
import tempfile
import threading
import unittest

from semantic_state_engine.server import (
    PersistenceError,
    RequestHandler,
    SemanticStateServer,
    StateStore,
)


class PersistentServerCase(unittest.TestCase):
    """Runs a real server bound to a data file inside a temp directory."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_file = os.path.join(self.tmp.name, "state.json")
        self.server = None

    def start_server(self) -> int:
        self.server = SemanticStateServer(
            ("127.0.0.1", 0), RequestHandler, data_file=self.data_file
        )
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_server)
        return self.server.server_address[1]

    def _stop_server(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def restart_server(self) -> int:
        self._stop_server()
        return self.start_server()

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            {"operationId": operation_id, "key": key, "value": value, "clock": clock},
        )

    def read_file(self) -> str:
        with open(self.data_file, "r", encoding="utf-8") as handle:
            return handle.read()


class PersistenceStartupTests(PersistentServerCase):
    def test_creates_missing_data_file_on_first_start(self) -> None:
        self.port = self.start_server()
        self.assertTrue(os.path.isfile(self.data_file))
        document = json.loads(self.read_file())
        self.assertEqual(document["operations"], [])
        self.assertEqual(document["candidates"], {})

    def test_missing_parent_directory_fails(self) -> None:
        missing = os.path.join(self.tmp.name, "no-such-dir", "state.json")
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=missing)

    def test_directory_as_data_file_fails(self) -> None:
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=self.tmp.name)

    def test_unparseable_file_fails(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=self.data_file)

    def test_wrong_structure_fails(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump({"format": "something-else", "operations": [], "candidates": {}}, handle)
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=self.data_file)

    def test_constraint_violating_operation_fails(self) -> None:
        document = {
            "format": "semantic-state-engine/v1",
            "operations": [
                {
                    "replicaId": "r1",
                    "operation": {
                        "operationId": "op-1",
                        "key": "k",
                        "value": "v",
                        "clock": {"r1": -1},
                    },
                }
            ],
            "candidates": {},
        }
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=self.data_file)

    def test_candidate_without_matching_operation_fails(self) -> None:
        document = {
            "format": "semantic-state-engine/v1",
            "operations": [],
            "candidates": {
                "k": [
                    {"value": "v", "clock": {"r1": 1}, "replicaId": "r1", "operationId": "op-1"}
                ]
            },
        }
        with open(self.data_file, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(PersistenceError):
            SemanticStateServer(("127.0.0.1", 0), RequestHandler, data_file=self.data_file)

    def test_leftover_tmp_file_is_ignored(self) -> None:
        self.port = self.start_server()
        self.post_operation("r1", "op-1", "k", "v", {"r1": 1})
        with open(self.data_file + ".tmp", "w", encoding="utf-8") as handle:
            handle.write("partial garbage")
        self.port = self.restart_server()
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v")


class PersistenceRoundTripTests(PersistentServerCase):
    def test_state_survives_restart(self) -> None:
        self.port = self.start_server()
        # A dominating chain on "color", a stale write, and a concurrent
        # conflict on "shape".
        self.assertEqual(self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})[0], 201)
        self.assertEqual(self.post_operation("r1", "op-2", "color", "red", {"r1": 2})[0], 201)
        self.assertEqual(self.post_operation("r1", "op-3", "color", "stale", {"r1": 1})[0], 201)
        self.post_operation("r2", "op-1", "shape", "square", {"r2": 1})
        self.post_operation("r3", "op-1", "shape", "round", {"r3": 1})
        _, before_color = self.request("GET", "/v1/states/color")
        _, before_shape = self.request("GET", "/v1/states/shape")
        self.assertEqual(before_shape["status"], "conflict")

        self.port = self.restart_server()

        _, after_color = self.request("GET", "/v1/states/color")
        _, after_shape = self.request("GET", "/v1/states/shape")
        self.assertEqual(after_color, before_color)
        self.assertEqual(after_shape, before_shape)
        self.assertEqual(after_color["value"], "red")

        # Replays of already-accepted operations still return 200.
        self.assertEqual(self.post_operation("r1", "op-2", "color", "red", {"r1": 2})[0], 200)
        self.assertEqual(self.post_operation("r1", "op-3", "color", "stale", {"r1": 1})[0], 200)
        # Same identity with different content still conflicts with 409.
        status, payload = self.post_operation("r1", "op-2", "color", "green", {"r1": 2})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        # Dominance still applies to restored candidates.
        self.assertEqual(self.post_operation("r1", "op-4", "color", "gold", {"r1": 3})[0], 201)
        _, state = self.request("GET", "/v1/states/color")
        self.assertEqual(state["value"], "gold")

    def test_every_accepted_write_is_persisted_before_response(self) -> None:
        self.port = self.start_server()
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 201)
        # A stale write produces no candidate but must still be recorded.
        self.assertEqual(self.post_operation("r2", "op-9", "k", "old", {"r2": 0})[0], 201)
        document = json.loads(self.read_file())
        identities = {
            (entry["replicaId"], entry["operation"]["operationId"])
            for entry in document["operations"]
        }
        self.assertEqual(identities, {("r1", "op-1"), ("r2", "op-9")})

    def test_replay_and_conflict_do_not_touch_file(self) -> None:
        self.port = self.start_server()
        self.post_operation("r1", "op-1", "k", "v", {"r1": 1})
        before = self.read_file()
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 200)
        self.assertEqual(self.post_operation("r1", "op-1", "k", "other", {"r1": 1})[0], 409)
        self.assertEqual(self.read_file(), before)

    def test_concurrent_writes_commit_in_order(self) -> None:
        self.port = self.start_server()
        results = []
        threads = [
            threading.Thread(
                target=lambda i=i: results.append(
                    self.post_operation("r1", f"op-{i}", "k", f"v{i}", {"r1": i + 1})[0]
                )
            )
            for i in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results, [201] * 20)
        # The file must reflect a prefix-consistent final state: reloading
        # it yields exactly the in-memory answer.
        _, before = self.request("GET", "/v1/states/k")
        self.port = self.restart_server()
        _, after = self.request("GET", "/v1/states/k")
        self.assertEqual(after, before)
        self.assertEqual(after["clock"], {"r1": 20})


class InMemoryStoreTests(unittest.TestCase):
    def test_store_without_data_file_stays_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            before = set(os.listdir(tmp))
            store = StateStore()
            store.apply_operation(
                "r1", {"operationId": "op-1", "key": "k", "value": "v", "clock": {"r1": 1}}
            )
            self.assertEqual(set(os.listdir(tmp)), before)


if __name__ == "__main__":
    unittest.main()
