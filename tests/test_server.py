import json
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from semantic_state_engine.server import (
    Operation,
    RequestHandler,
    StateStore,
    build_server,
    clock_dominates,
    health_payload,
    merge_candidates,
    parse_operation_payload,
    state_view,
)


class HealthPayloadTests(unittest.TestCase):
    def test_health_payload_is_stable(self) -> None:
        self.assertEqual(
            health_payload(),
            {"service": "semantic-state-engine", "status": "ok"},
        )


class ClockDominatesTests(unittest.TestCase):
    def test_greater_in_one_component_dominates(self) -> None:
        self.assertTrue(clock_dominates({"a": 2, "b": 1}, {"a": 1, "b": 1}))

    def test_equal_clocks_do_not_dominate(self) -> None:
        clock = {"a": 1, "b": 2}
        self.assertFalse(clock_dominates(clock, dict(clock)))

    def test_concurrent_clocks_do_not_dominate(self) -> None:
        left = {"a": 1, "b": 0}
        right = {"a": 0, "b": 1}
        self.assertFalse(clock_dominates(left, right))
        self.assertFalse(clock_dominates(right, left))

    def test_missing_components_count_as_zero(self) -> None:
        self.assertTrue(clock_dominates({"a": 1}, {}))
        self.assertTrue(clock_dominates({"a": 1}, {"b": 0}))
        self.assertFalse(clock_dominates({}, {"a": 1}))

    def test_partial_ordering(self) -> None:
        self.assertTrue(clock_dominates({"a": 2, "b": 3}, {"a": 2, "b": 2}))
        self.assertFalse(clock_dominates({"a": 1, "b": 3}, {"a": 2, "b": 2}))


class ParseOperationPayloadTests(unittest.TestCase):
    def _valid(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "operationId": "op-1",
            "key": "color",
            "value": "blue",
            "clock": {"replica-a": 1},
        }
        payload.update(overrides)
        return payload

    def test_valid_payload(self) -> None:
        operation = parse_operation_payload(
            json.dumps(self._valid()), "replica-a"
        )
        self.assertEqual(operation.replica_id, "replica-a")
        self.assertEqual(operation.operation_id, "op-1")
        self.assertEqual(operation.key, "color")
        self.assertEqual(operation.value, "blue")
        self.assertEqual(operation.clock, {"replica-a": 1})

    def test_clock_may_carry_other_replica_components(self) -> None:
        operation = parse_operation_payload(
            json.dumps(self._valid(clock={"replica-a": 3, "replica-b": 2})),
            "replica-a",
        )
        self.assertEqual(operation.clock, {"replica-a": 3, "replica-b": 2})

    def test_malformed_json_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(b"{not json", "replica-a")

    def test_non_object_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(json.dumps([1, 2]), "replica-a")

    def test_required_string_fields(self) -> None:
        for field, bad in (
            ("operationId", ""),
            ("operationId", 5),
            ("operationId", None),
            ("key", ""),
            ("key", 7),
            ("value", ""),
            ("value", False),
        ):
            with self.subTest(field=field, bad=bad):
                with self.assertRaises(ValueError):
                    parse_operation_payload(
                        json.dumps(self._valid(**{field: bad})), "replica-a"
                    )

    def test_clock_constraints(self) -> None:
        for clock in (
            None,
            {},
            {"replica-a": -1},
            {"replica-a": 1.5},
            {"replica-a": True},
            {"replica-a": "1"},
            {"": 1},
            {1: 1},
        ):
            with self.subTest(clock=clock):
                with self.assertRaises(ValueError):
                    parse_operation_payload(
                        json.dumps(self._valid(clock=clock)), "replica-a"
                    )

    def test_clock_must_contain_replica_component(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(
                json.dumps(self._valid(clock={"replica-b": 1})), "replica-a"
            )

    def test_empty_replica_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_operation_payload(json.dumps(self._valid()), "")


def _op(
    replica_id: str,
    operation_id: str,
    clock: dict[str, int],
    key: str = "color",
    value: str = "blue",
) -> Operation:
    return Operation(replica_id, operation_id, key, value, dict(clock))


class MergeCandidatesTests(unittest.TestCase):
    def test_first_candidate_is_stored(self) -> None:
        candidate = _op("a", "1", {"a": 1})
        survivors, outcome = merge_candidates([], candidate)
        self.assertEqual(outcome, "stored")
        self.assertEqual(survivors, [candidate])

    def test_dominating_candidate_replaces_older(self) -> None:
        older = _op("a", "1", {"a": 1})
        newer = _op("a", "2", {"a": 2})
        survivors, outcome = merge_candidates([older], newer)
        self.assertEqual(outcome, "stored")
        self.assertEqual(survivors, [newer])

    def test_dominated_candidate_is_dropped_without_version(self) -> None:
        newer = _op("a", "2", {"a": 2, "b": 1})
        older = _op("b", "1", {"a": 1, "b": 1})
        survivors, outcome = merge_candidates([newer], older)
        self.assertEqual(outcome, "stored")
        self.assertEqual(survivors, [newer])

    def test_concurrent_different_values_both_survive(self) -> None:
        left = _op("a", "1", {"a": 1}, value="blue")
        right = _op("b", "1", {"b": 1}, value="green")
        survivors, outcome = merge_candidates([left], right)
        self.assertEqual(outcome, "stored")
        self.assertEqual(survivors, [left, right])

    def test_concurrent_same_value_both_survive_for_resolution(self) -> None:
        left = _op("a", "1", {"a": 1}, value="blue")
        right = _op("b", "1", {"b": 1}, value="blue")
        survivors, outcome = merge_candidates([left], right)
        self.assertEqual(outcome, "stored")
        self.assertEqual(survivors, [left, right])

    def test_exact_replay_is_duplicate(self) -> None:
        original = _op("a", "1", {"a": 1}, value="blue")
        replay = _op("a", "1", {"a": 1}, value="blue")
        survivors, outcome = merge_candidates([original], replay)
        self.assertEqual(outcome, "duplicate")
        self.assertEqual(survivors, [original])

    def test_same_identity_different_value_conflicts(self) -> None:
        original = _op("a", "1", {"a": 1}, value="blue")
        retried = _op("a", "1", {"a": 1}, value="red")
        survivors, outcome = merge_candidates([original], retried)
        self.assertEqual(outcome, "conflict")
        self.assertEqual(survivors, [original])

    def test_same_identity_different_clock_conflicts(self) -> None:
        original = _op("a", "1", {"a": 1})
        retried = _op("a", "1", {"a": 2})
        survivors, outcome = merge_candidates([original], retried)
        self.assertEqual(outcome, "conflict")
        self.assertEqual(survivors, [original])


class StateViewTests(unittest.TestCase):
    def test_missing_key_is_not_found(self) -> None:
        self.assertEqual(state_view("color", []), {"key": "color", "error": "not_found"})

    def test_single_candidate_resolved(self) -> None:
        view = state_view("color", [_op("a", "1", {"a": 1}, value="blue")])
        self.assertEqual(
            view,
            {
                "key": "color",
                "value": "blue",
                "clock": {"a": 1},
                "status": "resolved",
            },
        )

    def test_same_value_picks_smallest_replica_then_operation(self) -> None:
        candidates = [
            _op("b", "9", {"b": 1}, value="blue"),
            _op("a", "5", {"a": 1}, value="blue"),
            _op("a", "2", {"a": 0, "b": 1}, value="blue"),
        ]
        view = state_view("color", candidates)
        self.assertEqual(view["status"], "resolved")
        self.assertEqual(view["value"], "blue")
        self.assertEqual(view["clock"], {"a": 0, "b": 1})

    def test_conflict_candidates_sorted(self) -> None:
        candidates = [
            _op("b", "2", {"b": 1}, value="green"),
            _op("a", "9", {"a": 1}, value="blue"),
            _op("a", "1", {"a": 0, "b": 1}, value="red"),
        ]
        view = state_view("color", candidates)
        self.assertEqual(view["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in view["candidates"]],
            [("a", "1"), ("a", "9"), ("b", "2")],
        )
        self.assertEqual(
            [c["value"] for c in view["candidates"]],
            ["red", "blue", "green"],
        )
        self.assertEqual(view["candidates"][1]["clock"], {"a": 1})


class StateStoreConcurrencyTests(unittest.TestCase):
    def test_same_operation_replayed_concurrently_is_single_version(self) -> None:
        store = StateStore()
        operation = _op("a", "1", {"a": 1})
        with ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(lambda _: store.apply(operation), range(64)))
        self.assertEqual(sorted(outcomes).count("stored"), 1)
        self.assertEqual(sorted(outcomes).count("duplicate"), 63)
        view = store.view("color")
        self.assertEqual(view["status"], "resolved")

    def test_parallel_dominating_writes_leave_highest_clock(self) -> None:
        store = StateStore()

        def write(index: int) -> None:
            store.apply(_op("a", f"op-{index}", {"a": index + 1}))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, range(100)))
        view = store.view("color")
        self.assertEqual(view["status"], "resolved")
        self.assertEqual(view["clock"], {"a": 100})

    def test_identity_conflict_across_keys_does_not_mutate(self) -> None:
        store = StateStore()
        self.assertEqual(store.apply(_op("a", "1", {"a": 1}, key="k1")), "stored")
        conflicted = _op("a", "1", {"a": 1}, key="k2", value="red")
        self.assertEqual(store.apply(conflicted), "conflict")
        self.assertEqual(store.view("k2"), {"key": "k2", "error": "not_found"})


class HttpApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = build_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _request(
        self, method: str, path: str, payload: object | None = None
    ) -> tuple[int, dict[str, object]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    @staticmethod
    def _payload(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "operationId": "op-1",
            "key": "color",
            "value": "blue",
            "clock": {"r1": 1},
        }
        payload.update(overrides)
        return payload

    def test_health_still_available(self) -> None:
        status, body = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, health_payload())

    def _post_raw(self, raw: bytes) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/replicas/r1/operations",
            data=raw,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_invalid_json_returns_400(self) -> None:
        status, body = self._post_raw(b"{broken")
        self.assertEqual(status, 400)
        self.assertEqual(body, {"error": "invalid_request"})

    def test_invalid_fields_return_400(self) -> None:
        cases = [
            self._payload(value=""),
            self._payload(operationId=42),
            self._payload(clock={}),
            self._payload(clock={"r2": 1}),
            self._payload(clock={"r1": -1}),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status, body = self._request(
                    "POST", "/v1/replicas/r1/operations", payload
                )
                self.assertEqual(status, 400)
                self.assertEqual(body, {"error": "invalid_request"})

    def test_write_replay_conflict_and_query_flow(self) -> None:
        payload = self._payload()

        status, _ = self._request("POST", "/v1/replicas/r1/operations", payload)
        self.assertEqual(status, 201)

        # Identical replay is idempotent and does not add a version.
        status, body = self._request("POST", "/v1/replicas/r1/operations", payload)
        self.assertEqual(status, 200)

        status, body = self._request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "resolved")
        self.assertEqual(body["value"], "blue")
        self.assertEqual(body["clock"], {"r1": 1})

        # Same identity, different content conflicts without state change.
        conflict = self._payload(value="red")
        status, body = self._request("POST", "/v1/replicas/r1/operations", conflict)
        self.assertEqual(status, 409)
        self.assertEqual(body, {"error": "operation_conflict"})

        status, body = self._request("GET", "/v1/states/color")
        self.assertEqual(body["value"], "blue")

    def test_dominating_write_overwrites(self) -> None:
        first = self._payload(operationId="op-1", clock={"r1": 1}, value="blue")
        second = self._payload(operationId="op-2", clock={"r1": 2}, value="red")
        self.assertEqual(
            self._request("POST", "/v1/replicas/r1/operations", first)[0], 201
        )
        self.assertEqual(
            self._request("POST", "/v1/replicas/r1/operations", second)[0], 201
        )
        status, body = self._request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "key": "color",
            "value": "red",
            "clock": {"r1": 2},
            "status": "resolved",
        })

    def test_concurrent_conflict_is_reported_sorted(self) -> None:
        left = self._payload(
            operationId="op-a", value="blue", clock={"r1": 1}
        )
        right = self._payload(
            operationId="op-b", value="green", clock={"r2": 1}
        )
        self.assertEqual(
            self._request("POST", "/v1/replicas/r1/operations", left)[0], 201
        )
        self.assertEqual(
            self._request("POST", "/v1/replicas/r2/operations", right)[0], 201
        )
        status, body = self._request("GET", "/v1/states/color")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "conflict")
        self.assertEqual(
            [(c["replicaId"], c["operationId"]) for c in body["candidates"]],
            [("r1", "op-a"), ("r2", "op-b")],
        )

    def test_missing_state_returns_404(self) -> None:
        status, body = self._request("GET", "/v1/states/missing")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "not_found"})

    def test_keys_are_isolated(self) -> None:
        self._request(
            "POST",
            "/v1/replicas/r1/operations",
            self._payload(key="shape", operationId="s1", value="round"),
        )
        self._request(
            "POST",
            "/v1/replicas/r1/operations",
            self._payload(key="color", operationId="c1", value="blue"),
        )
        shape_status, shape = self._request("GET", "/v1/states/shape")
        color_status, color = self._request("GET", "/v1/states/color")
        self.assertEqual(shape_status, 200)
        self.assertEqual(shape["value"], "round")
        self.assertEqual(color_status, 200)
        self.assertEqual(color["value"], "blue")

    def test_unknown_routes_404(self) -> None:
        status, body = self._request("GET", "/v1/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "not_found"})
        status, body = self._request("POST", "/v1/replicas/r1/other", self._payload())
        self.assertEqual(status, 404)

    def test_handler_carries_exposed_helpers(self) -> None:
        self.assertTrue(hasattr(RequestHandler, "do_POST"))
        self.assertTrue(callable(clock_dominates))
        self.assertTrue(callable(parse_operation_payload))


if __name__ == "__main__":
    unittest.main()
