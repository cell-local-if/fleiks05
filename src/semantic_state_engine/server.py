"""HTTP boundary for the semantic state engine."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
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


class PersistenceError(Exception):
    """Raised when the data file cannot be loaded, validated, or written."""


DATA_FILE_FORMAT = "semantic-state-engine/v1"


def _serialize_state(
    operations: dict[tuple[str, str], dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
) -> str:
    document = {
        "format": DATA_FILE_FORMAT,
        "operations": [
            {"replicaId": replica_id, "operation": operation}
            for (replica_id, _operation_id), operation in operations.items()
        ],
        "candidates": candidates,
    }
    return json.dumps(document, sort_keys=True)


def _validated_stored_operation(entry: Any, context: str) -> tuple[str, dict[str, Any]]:
    """Validate one stored ``{"replicaId", "operation"}`` entry.

    The operation must satisfy exactly the same constraints as a live
    request payload; anything else is a structural error.
    """
    if not isinstance(entry, dict) or set(entry) != {"replicaId", "operation"}:
        raise PersistenceError(f"{context}: expected an object with replicaId and operation")
    replica_id = entry["replicaId"]
    if not isinstance(replica_id, str) or replica_id == "":
        raise PersistenceError(f"{context}: replicaId must be a non-empty string")
    operation = entry["operation"]
    try:
        normalized = parse_operation_payload(operation, replica_id)
    except ValueError as exc:
        raise PersistenceError(f"{context}: invalid stored operation: {exc}") from exc
    if normalized != operation:
        raise PersistenceError(f"{context}: stored operation has unexpected fields")
    return replica_id, operation


def _load_persisted_state(
    path: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Read and strictly validate the data file, returning restored state.

    Any parse failure, structural mismatch, or constraint violation raises
    PersistenceError; nothing is silently dropped or repaired.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise PersistenceError(f"cannot read data file {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"data file {path} is not fully parseable: {exc}") from exc

    if not isinstance(document, dict) or set(document) != {"format", "operations", "candidates"}:
        raise PersistenceError(f"data file {path}: unexpected top-level structure")
    if document["format"] != DATA_FILE_FORMAT:
        raise PersistenceError(f"data file {path}: unsupported format {document['format']!r}")

    raw_operations = document["operations"]
    if not isinstance(raw_operations, list):
        raise PersistenceError(f"data file {path}: operations must be a list")
    operations: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(raw_operations):
        replica_id, operation = _validated_stored_operation(entry, f"operations[{index}]")
        identity = (replica_id, operation["operationId"])
        if identity in operations:
            raise PersistenceError(f"data file {path}: duplicate operation identity {identity}")
        operations[identity] = operation

    raw_candidates = document["candidates"]
    if not isinstance(raw_candidates, dict):
        raise PersistenceError(f"data file {path}: candidates must be an object")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for key, stored_list in raw_candidates.items():
        if not isinstance(key, str) or key == "":
            raise PersistenceError(f"data file {path}: candidate keys must be non-empty strings")
        if not isinstance(stored_list, list) or not stored_list:
            raise PersistenceError(f"data file {path}: candidates for {key!r} must be a non-empty list")
        restored: list[dict[str, Any]] = []
        for index, candidate in enumerate(stored_list):
            context = f"candidates[{key!r}][{index}]"
            if not isinstance(candidate, dict) or set(candidate) != {
                "value",
                "clock",
                "replicaId",
                "operationId",
            }:
                raise PersistenceError(f"{context}: unexpected candidate structure")
            replica_id = candidate["replicaId"]
            if not isinstance(replica_id, str) or replica_id == "":
                raise PersistenceError(f"{context}: replicaId must be a non-empty string")
            # The candidate must satisfy the same input constraints as a
            # live operation and must match a recorded accepted operation.
            operation = {
                "operationId": candidate["operationId"],
                "key": key,
                "value": candidate["value"],
                "clock": candidate["clock"],
            }
            try:
                parse_operation_payload(operation, replica_id)
            except ValueError as exc:
                raise PersistenceError(f"{context}: invalid stored candidate: {exc}") from exc
            if operations.get((replica_id, candidate["operationId"])) != operation:
                raise PersistenceError(
                    f"{context}: candidate has no matching accepted operation"
                )
            restored.append(
                {
                    "value": candidate["value"],
                    "clock": dict(candidate["clock"]),
                    "replicaId": replica_id,
                    "operationId": candidate["operationId"],
                }
            )
        candidates[key] = restored
    return operations, candidates


class DataFilePersister:
    """Atomically persists full store snapshots to a single data file.

    Each save writes a temp file in the same directory, fsyncs it, and
    os.replace()s it over the target, so an interrupted write never leaves
    the data file holding partial content.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._tmp_path = f"{path}.tmp"

    def save(
        self,
        operations: dict[tuple[str, str], dict[str, Any]],
        candidates: dict[str, list[dict[str, Any]]],
    ) -> None:
        data = _serialize_state(operations, candidates)
        try:
            with open(self._tmp_path, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._tmp_path, self._path)
            self._fsync_dir()
        except OSError as exc:
            raise PersistenceError(f"cannot write data file {self._path}: {exc}") from exc

    def _fsync_dir(self) -> None:
        directory = os.path.dirname(os.path.abspath(self._path))
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)


class StateStore:
    """Concurrency-safe store of versioned candidates per key.

    When constructed with a persister, every newly accepted operation is
    durably written (inside the store lock, so the file commits in the same
    order as the in-memory state) before the in-memory state is updated and
    before the caller can respond.
    """

    def __init__(
        self,
        operations: dict[tuple[str, str], dict[str, Any]] | None = None,
        candidates: dict[str, list[dict[str, Any]]] | None = None,
        persister: DataFilePersister | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._operations: dict[tuple[str, str], dict[str, Any]] = operations or {}
        self._candidates: dict[str, list[dict[str, Any]]] = candidates or {}
        self._persister = persister

    @classmethod
    def from_data_file(cls, path: str) -> "StateStore":
        """Open or create a persistent store backed by ``path``.

        An existing file must be a regular file whose content parses and
        validates completely; a missing file is created (its parent
        directory must already exist). Any other situation raises
        PersistenceError and yields no store.
        """
        persister = DataFilePersister(path)
        if os.path.exists(path):
            try:
                mode = os.stat(path).st_mode
            except OSError as exc:
                raise PersistenceError(f"cannot access data file {path}: {exc}") from exc
            if not stat.S_ISREG(mode):
                raise PersistenceError(f"data file {path} is not a regular file")
            operations, candidates = _load_persisted_state(path)
        else:
            parent = os.path.dirname(os.path.abspath(path))
            if not os.path.isdir(parent):
                raise PersistenceError(
                    f"parent directory {parent} of data file {path} does not exist"
                )
            operations, candidates = {}, {}
            # Create the file now; this also proves the target is writable.
            persister.save(operations, candidates)
        return cls(operations=operations, candidates=candidates, persister=persister)

    def apply_operation(self, replica_id: str, operation: dict[str, Any]) -> HTTPStatus:
        """Apply a validated operation, returning the HTTP status to use."""
        with self._lock:
            identity = (replica_id, operation["operationId"])
            seen = self._operations.get(identity)
            if seen is not None:
                # Replays and conflicts never touch memory or the data file.
                if seen == operation:
                    return HTTPStatus.OK
                return HTTPStatus.CONFLICT

            operations = dict(self._operations)
            operations[identity] = operation
            candidates = dict(self._candidates)
            key = operation["key"]
            existing = candidates.get(key, [])
            new_candidate = {
                "value": operation["value"],
                "clock": operation["clock"],
                "replicaId": replica_id,
                "operationId": operation["operationId"],
            }
            # A candidate already dominates the incoming clock: the write is
            # stale, so it is recorded but adds no new version.
            if any(clock_dominates(c["clock"], new_candidate["clock"]) for c in existing):
                if key in candidates:
                    candidates[key] = list(existing)
            else:
                survivors = [
                    c for c in existing if not clock_dominates(new_candidate["clock"], c["clock"])
                ]
                survivors.append(new_candidate)
                candidates[key] = survivors

            if self._persister is not None:
                # Persist first; only commit to memory once the state is
                # durable, so a acknowledged write is never lost and a
                # failed write never becomes visible.
                self._persister.save(operations, candidates)
            self._operations = operations
            self._candidates = candidates
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

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler] = None,
        data_file: str | None = None,
    ) -> None:
        # Build (and validate or create) the store before binding, so a
        # persistence failure yields no servable instance at all.
        store = StateStore.from_data_file(data_file) if data_file is not None else StateStore()
        super().__init__(server_address, handler_class or RequestHandler)
        self.store = store


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
            try:
                status = self._store.apply_operation(replica_id, operation)
            except PersistenceError:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "persistence_error"})
                return
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
    parser.add_argument(
        "--data-file",
        default=None,
        metavar="PATH",
        help="persist accepted operations to PATH and restore them on startup; "
        "omit for pure in-memory operation",
    )
    args = parser.parse_args()
    try:
        server = SemanticStateServer((args.host, args.port), data_file=args.data_file)
    except PersistenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
