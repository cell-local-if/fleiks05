import http.client
import json
import threading
import unittest

from semantic_state_engine.server import (
    RequestHandler,
    SemanticStateServer,
    clock_dominates,
    health_payload,
    parse_operation_payload,
)


class HealthPayloadTests(unittest.TestCase):
    def test_health_payload_is_stable(self) -> None:
        self.assertEqual(
            health_payload(),
            {"service": "semantic-state-engine", "status": "ok"},
        )


class ClockDominatesTests(unittest.TestCase):
    def test_dominates_when_greater_on_one_component(self) -> None:
        self.assertTrue(clock_dominates({"a": 2, "b": 1}, {"a": 1, "b": 1}))

    def test_missing_components_count_as_zero(self) -> None:
        self.assertTrue(clock_dominates({"a": 1}, {}))
        self.assertFalse(clock_dominates({}, {"a": 1}))

    def test_equal_clocks_do_not_dominate(self) -> None:
        self.assertFalse(clock_dominates({"a": 1}, {"a": 1}))
        self.assertFalse(clock_dominates({"a": 1}, {"a": 1, "b": 0}))

    def test_concurrent_clocks_do_not_dominate(self) -> None:
        self.assertFalse(clock_dominates({"a": 1, "b": 2}, {"a": 2, "b": 1}))
        self.assertFalse(clock_dominates({"a": 2, "b": 1}, {"a": 1, "b": 2}))


class ParseOperationPayloadTests(unittest.TestCase):
    def test_valid_payload_is_normalized(self) -> None:
        payload = parse_operation_payload(
            json.dumps(
                {
                    "operationId": "op-1",
                    "key": "color",
                    "value": "blue",
                    "clock": {"r1": 1},
                }
            ),
            "r1",
        )
        self.assertEqual(
            payload,
            {
                "operationId": "op-1",
                "key": "color",
                "value": "blue",
                "clock": {"r1": 1},
            },
        )

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(b"{not json", "r1")

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload("[1, 2]", "r1")

    def test_rejects_empty_strings(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(
                {"operationId": "", "key": "k", "value": "v", "clock": {"r1": 1}}, "r1"
            )

    def test_rejects_negative_clock_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(
                {"operationId": "o", "key": "k", "value": "v", "clock": {"r1": -1}}, "r1"
            )

    def test_rejects_boolean_clock_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(
                {"operationId": "o", "key": "k", "value": "v", "clock": {"r1": True}}, "r1"
            )

    def test_rejects_clock_missing_replica(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(
                {"operationId": "o", "key": "k", "value": "v", "clock": {"r2": 1}}, "r1"
            )


class ServerTests(unittest.TestCase):
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

    def request(self, method: str, path: str, body: object = None) -> tuple[int, dict]:
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
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def post_operation(self, replica: str, operation_id: str, key: str, value: str, clock: dict):
        return self.request(
            "POST",
            f"/v1/replicas/{replica}/operations",
            {"operationId": operation_id, "key": key, "value": value, "clock": clock},
        )

    def test_health_still_works(self) -> None:
        status, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"service": "semantic-state-engine", "status": "ok"})

    def test_unknown_route_is_404(self) -> None:
        status, payload = self.request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_get_missing_key_is_404(self) -> None:
        status, payload = self.request("GET", "/v1/states/absent")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not_found"})

    def test_malformed_json_is_400(self) -> None:
        status, payload = self.request("POST", "/v1/replicas/r1/operations", b"{oops")
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_invalid_field_is_400(self) -> None:
        status, payload = self.post_operation("r1", "op-1", "", "v", {"r1": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_clock_must_contain_replica(self) -> None:
        status, payload = self.post_operation("r1", "op-1", "k", "v", {"r2": 1})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "invalid_request"})

    def test_new_write_is_201_and_resolves(self) -> None:
        status, _ = self.post_operation("r1", "op-1", "color", "blue", {"r1": 1})
        self.assertEqual(status, 201)
        status, payload = self.request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {"key": "color", "value": "blue", "clock": {"r1": 1}, "status": "resolved"},
        )

    def test_replay_is_200_and_adds_no_version(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 201)
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 200)
        status, payload = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "resolved")

    def test_same_identity_different_content_is_409(self) -> None:
        self.assertEqual(self.post_operation("r1", "op-1", "k", "v", {"r1": 1})[0], 201)
        status, payload = self.post_operation("r1", "op-1", "k", "other", {"r1": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload, {"error": "operation_conflict"})
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["value"], "v")

    def test_dominating_write_replaces_candidate(self) -> None:
        self.post_operation("r1", "op-1", "k", "old", {"r1": 1})
        self.assertEqual(self.post_operation("r1", "op-2", "k", "new", {"r1": 2})[0], 201)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state, {"key": "k", "value": "new", "clock": {"r1": 2}, "status": "resolved"})

    def test_stale_write_adds_no_candidate(self) -> None:
        self.post_operation("r1", "op-1", "k", "new", {"r1": 2})
        self.assertEqual(self.post_operation("r1", "op-2", "k", "stale", {"r1": 1})[0], 201)
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(state["status"], "resolved")
        self.assertEqual(state["value"], "new")

    def test_concurrent_writes_conflict_and_sort(self) -> None:
        self.post_operation("r2", "op-2", "k", "v2", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "v1", {"r1": 1})
        status, state = self.request("GET", "/v1/states/k")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(state["key"], "k")
        self.assertEqual(
            state["candidates"],
            [
                {"value": "v1", "clock": {"r1": 1}, "replicaId": "r1", "operationId": "op-1"},
                {"value": "v2", "clock": {"r2": 1}, "replicaId": "r2", "operationId": "op-2"},
            ],
        )

    def test_same_value_concurrent_writes_resolve_to_min_identity(self) -> None:
        self.post_operation("r2", "op-2", "k", "same", {"r2": 1})
        self.post_operation("r1", "op-1", "k", "same", {"r1": 1})
        _, state = self.request("GET", "/v1/states/k")
        self.assertEqual(
            state,
            {"key": "k", "value": "same", "clock": {"r1": 1}, "status": "resolved"},
        )

    def test_keys_are_isolated(self) -> None:
        self.post_operation("r1", "op-1", "a", "va", {"r1": 1})
        self.post_operation("r1", "op-2", "b", "vb", {"r1": 2})
        _, state_a = self.request("GET", "/v1/states/a")
        _, state_b = self.request("GET", "/v1/states/b")
        self.assertEqual(state_a["value"], "va")
        self.assertEqual(state_b["value"], "vb")


if __name__ == "__main__":
    unittest.main()
