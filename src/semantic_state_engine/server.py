"""Minimal HTTP boundary for the semantic state engine."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit


def health_payload() -> dict[str, str]:
    return {"service": "semantic-state-engine", "status": "ok"}


@dataclass(frozen=True)
class Operation:
    """A single candidate write originating from one replica."""

    replica_id: str
    operation_id: str
    key: str
    value: str
    clock: dict[str, int]


def clock_dominates(left: dict[str, int], right: dict[str, int]) -> bool:
    """Return True when ``left`` dominates (happens-after) ``right``.

    A vector clock dominates another when every component of ``left`` is at
    least the corresponding component of ``right`` (missing components count
    as zero) and at least one component is strictly greater. Equal clocks are
    concurrent and therefore do not dominate each other.
    """

    components = left.keys() | right.keys()
    strictly_greater = False
    for component in components:
        a = left.get(component, 0)
        b = right.get(component, 0)
        if a < b:
            return False
        if a > b:
            strictly_greater = True
    return strictly_greater


def parse_operation_payload(raw_body: bytes | str, replica_id: str) -> Operation:
    """Parse and validate an operation request body for ``replica_id``.

    Raises ``ValueError`` when the body is malformed JSON or fails any field
    constraint. On success the returned clock always contains ``replica_id``
    and integer values, with no extra whitespace assumptions.
    """

    if not isinstance(replica_id, str) or replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("malformed JSON body") from exc
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    operation_id = payload.get("operationId")
    key = payload.get("key")
    value = payload.get("value")
    for name, field in (("operationId", operation_id), ("key", key), ("value", value)):
        if not isinstance(field, str) or field == "":
            raise ValueError(f"{name} must be a non-empty string")

    clock = payload.get("clock")
    if not isinstance(clock, dict) or not clock:
        raise ValueError("clock must be a non-empty object")
    parsed_clock: dict[str, int] = {}
    for component, counter in clock.items():
        if not isinstance(component, str) or component == "":
            raise ValueError("clock components must be non-empty strings")
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ValueError("clock values must be non-negative integers")
        parsed_clock[component] = counter
    if replica_id not in parsed_clock:
        raise ValueError("clock must contain the replica's own component")

    return Operation(
        replica_id=replica_id,
        operation_id=operation_id,
        key=key,
        value=value,
        clock=parsed_clock,
    )


def merge_candidates(
    existing: list[Operation], candidate: Operation
) -> tuple[list[Operation], str]:
    """Merge ``candidate`` into the surviving ``existing`` candidates for a key.

    Candidates dominated on all clock components by another are discarded.
    Returns the new surviving list plus an outcome: ``"stored"`` when the
    candidate was added, ``"duplicate"`` when the identical operation was
    replayed, or ``"conflict"`` when the same (replicaId, operationId) was
    already stored with different content. Callers hold the per-key lock.
    """

    # Identity first: re-submissions of the same operation never change state.
    duplicates = [
        other
        for other in existing
        if other.replica_id == candidate.replica_id
        and other.operation_id == candidate.operation_id
    ]
    if duplicates:
        previous = duplicates[0]
        if previous.key == candidate.key and previous.value == candidate.value and (
            previous.clock == candidate.clock
        ):
            return list(existing), "duplicate"
        return list(existing), "conflict"

    survivors: list[Operation] = []
    candidate_dominated = False
    for other in existing:
        if clock_dominates(other.clock, candidate.clock):
            candidate_dominated = True
            survivors.append(other)
            continue
        if clock_dominates(candidate.clock, other.clock):
            # The new candidate supersedes this older version; drop it.
            continue
        survivors.append(other)

    if not candidate_dominated:
        survivors.append(candidate)
    return survivors, "stored"


def state_view(key: str, candidates: list[Operation]) -> dict[str, Any]:
    """Build the GET state response from surviving candidates for ``key``."""

    if not candidates:
        return {"key": key, "error": "not_found"}

    values = {candidate.value for candidate in candidates}
    if len(values) == 1:
        chosen = min(
            candidates,
            key=lambda candidate: (candidate.replica_id, candidate.operation_id),
        )
        return {
            "key": key,
            "value": chosen.value,
            "clock": dict(sorted(chosen.clock.items())),
            "status": "resolved",
        }

    entries = sorted(
        (
            {
                "value": candidate.value,
                "clock": dict(sorted(candidate.clock.items())),
                "replicaId": candidate.replica_id,
                "operationId": candidate.operation_id,
            }
            for candidate in candidates
        ),
        key=lambda entry: (entry["replicaId"], entry["operationId"]),
    )
    return {"key": key, "status": "conflict", "candidates": entries}


class StateStore:
    """Thread-safe container of surviving candidates keyed by state key.

    Writes on different keys proceed in parallel; a small identity lock only
    guards the ``(replicaId, operationId)`` index, while the bulk of a merge
    runs under the per-key guard. Identity is tracked across keys so an
    operationId cannot be reused with different content on another key.
    """

    def __init__(self) -> None:
        self._guards_lock = threading.Lock()
        self._identity_lock = threading.Lock()
        self._key_guards: dict[str, threading.Lock] = {}
        self._states: dict[str, list[Operation]] = {}
        self._identities: dict[tuple[str, str], Operation] = {}

    def _guard_for(self, key: str) -> threading.Lock:
        with self._guards_lock:
            guard = self._key_guards.get(key)
            if guard is None:
                guard = threading.Lock()
                self._key_guards[key] = guard
            return guard

    def apply(self, operation: Operation) -> str:
        """Apply one operation and return the ``merge_candidates`` outcome."""

        identity = (operation.replica_id, operation.operation_id)
        with self._identity_lock:
            previous = self._identities.get(identity)
            if previous is not None:
                if (
                    previous.key == operation.key
                    and previous.value == operation.value
                    and previous.clock == operation.clock
                ):
                    return "duplicate"
                return "conflict"
            # Claim the identity before merging so concurrent submissions of
            # the same operation resolve deterministically.
            self._identities[identity] = operation

        guard = self._guard_for(operation.key)
        with guard:
            existing = self._states.get(operation.key, [])
            survivors, outcome = merge_candidates(existing, operation)
            self._states[operation.key] = survivors
            return outcome

    def view(self, key: str) -> dict[str, Any]:
        guard = self._guard_for(key)
        with guard:
            return state_view(key, self._states.get(key, []))


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticStateEngine/0.1"

    #: Shared store; set by ``build_server`` / ``main``.
    store: StateStore | None = None

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, health_payload())
            return
        if parsed.path.startswith("/v1/states/"):
            key = unquote(parsed.path[len("/v1/states/") :])
            if key == "" or "/" in key:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            view = self._require_store().view(key)
            if view.get("status") is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._json(HTTPStatus.OK, view)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        prefix = "/v1/replicas/"
        suffix = "/operations"
        if not (parsed.path.startswith(prefix) and parsed.path.endswith(suffix)):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        replica_id = unquote(parsed.path[len(prefix) : -len(suffix)])
        if replica_id == "" or "/" in replica_id:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else 0
            if length < 0:
                raise ValueError
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw_body = self.rfile.read(length) if length > 0 else b""

        try:
            operation = parse_operation_payload(raw_body, replica_id)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return

        outcome = self._require_store().apply(operation)
        if outcome == "conflict":
            self._json(HTTPStatus.CONFLICT, {"error": "operation_conflict"})
        elif outcome == "duplicate":
            self._json(HTTPStatus.OK, {"status": "duplicate"})
        else:
            self._json(HTTPStatus.CREATED, {"status": "stored"})

    def _require_store(self) -> StateStore:
        if self.store is None:
            raise RuntimeError("StateStore has not been configured")
        return self.store

    def log_message(self, format: str, *args: object) -> None:
        return


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    store = StateStore()
    handler = type("BoundRequestHandler", (RequestHandler,), {"store": store})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Semantic State Engine HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
