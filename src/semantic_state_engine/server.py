"""HTTP boundary for the semantic state engine."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit


def health_payload() -> dict[str, str]:
    return {"service": "semantic-state-engine", "status": "ok"}


def clock_dominates(clock_a: dict[str, int], clock_b: dict[str, int]) -> bool:
    """Return True when clock_a dominates clock_b.

    Missing components count as 0. A clock dominates another when it is
    greater than or equal on every component and strictly greater on at
    least one (i.e. A >= B and A != B).
    """
    keys = set(clock_a) | set(clock_b)
    strictly_greater = False
    for key in keys:
        a = clock_a.get(key, 0)
        b = clock_b.get(key, 0)
        if a < b:
            return False
        if a > b:
            strictly_greater = True
    return strictly_greater


def parse_operation_payload(raw: bytes | str | dict[str, Any], replica_id: str) -> dict[str, Any]:
    """Parse and validate an operation payload for ``replica_id``.

    Accepts a JSON document (bytes or text) or an already-decoded mapping.
    Returns a normalized operation dict. Raises ValueError when the payload
    is malformed or fails validation.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("body must be UTF-8 JSON") from exc
    if isinstance(raw, str):
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("body must be valid JSON") from exc
    else:
        payload = raw
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    for field in ("operationId", "key", "value"):
        value = payload.get(field)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{field} must be a non-empty string")

    clock = payload.get("clock")
    if not isinstance(clock, dict) or not clock:
        raise ValueError("clock must be a non-empty object")
    for component, tick in clock.items():
        if not isinstance(component, str) or component == "":
            raise ValueError("clock components must be non-empty strings")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("clock values must be non-negative integers")
    if replica_id not in clock:
        raise ValueError("clock must contain the replica id")

    return {
        "operationId": payload["operationId"],
        "key": payload["key"],
        "value": payload["value"],
        "clock": dict(clock),
    }


class StateStore:
    """Concurrency-safe store of versioned candidates per key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._candidates: dict[str, list[dict[str, Any]]] = {}
        self._operations: dict[tuple[str, str], dict[str, Any]] = {}

    def apply_operation(self, replica_id: str, operation: dict[str, Any]) -> HTTPStatus:
        """Apply a validated operation, returning the HTTP status to use."""
        with self._lock:
            identity = (replica_id, operation["operationId"])
            seen = self._operations.get(identity)
            if seen is not None:
                if seen == operation:
                    return HTTPStatus.OK
                return HTTPStatus.CONFLICT
            self._operations[identity] = operation

            key = operation["key"]
            candidates = self._candidates.setdefault(key, [])
            new_candidate = {
                "value": operation["value"],
                "clock": operation["clock"],
                "replicaId": replica_id,
                "operationId": operation["operationId"],
            }
            # A candidate already dominates the incoming clock: the write is
            # stale, so it is recorded but adds no new version.
            if any(clock_dominates(c["clock"], new_candidate["clock"]) for c in candidates):
                return HTTPStatus.CREATED
            survivors = [
                c for c in candidates if not clock_dominates(new_candidate["clock"], c["clock"])
            ]
            survivors.append(new_candidate)
            self._candidates[key] = survivors
            return HTTPStatus.CREATED

    def get_state(self, key: str) -> tuple[HTTPStatus, dict[str, Any]]:
        with self._lock:
            candidates = list(self._candidates.get(key, []))
        if not candidates:
            return HTTPStatus.NOT_FOUND, {"error": "not_found"}
        ordered = sorted(candidates, key=lambda c: (c["replicaId"], c["operationId"]))
        if all(c["value"] == ordered[0]["value"] for c in ordered):
            chosen = ordered[0]
            return HTTPStatus.OK, {
                "key": key,
                "value": chosen["value"],
                "clock": chosen["clock"],
                "status": "resolved",
            }
        return HTTPStatus.OK, {
            "key": key,
            "status": "conflict",
            "candidates": [
                {
                    "value": c["value"],
                    "clock": c["clock"],
                    "replicaId": c["replicaId"],
                    "operationId": c["operationId"],
                }
                for c in ordered
            ],
        }


class SemanticStateServer(ThreadingHTTPServer):
    """Threading HTTP server carrying its own StateStore."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler] = None) -> None:
        super().__init__(server_address, handler_class or RequestHandler)
        self.store = StateStore()


_FALLBACK_STORE = StateStore()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticStateEngine/0.1"

    @property
    def _store(self) -> StateStore:
        return getattr(self.server, "store", _FALLBACK_STORE)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _path_segments(self) -> list[str]:
        path = urlsplit(self.path).path
        return [unquote(segment) for segment in path.split("/") if segment != ""]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(HTTPStatus.OK, health_payload())
            return
        segments = self._path_segments()
        if len(segments) == 3 and segments[0] == "v1" and segments[1] == "states":
            status, payload = self._store.get_state(segments[2])
            self._json(status, payload)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        segments = self._path_segments()
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "replicas"
            and segments[3] == "operations"
        ):
            replica_id = segments[2]
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                operation = parse_operation_payload(raw, replica_id)
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            status = self._store.apply_operation(replica_id, operation)
            if status is HTTPStatus.CONFLICT:
                self._json(status, {"error": "operation_conflict"})
                return
            payload = {
                "status": "ok" if status is HTTPStatus.OK else "created",
                "replicaId": replica_id,
                "operationId": operation["operationId"],
                "key": operation["key"],
            }
            self._json(status, payload)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Semantic State Engine HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = SemanticStateServer((args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
