"""HTTP boundary for the semantic state engine."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

DATA_FORMAT_VERSION = 1


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
    """Raised when the data file cannot be opened, parsed, or written.

    At startup this means the service must refuse to start; while serving
    it means the current write could not be committed durably.
    """


def _validate_stored_operation(entry: Any) -> tuple[str, dict[str, Any]]:
    """Validate one persisted operation record against the input constraints.

    Returns the ``(replica_id, operation)`` pair. Raises PersistenceError on
    any structural mismatch or constraint violation.
    """
    if not isinstance(entry, dict) or set(entry.keys()) != {"replicaId", "operation"}:
        raise PersistenceError("each operation record must be an object with replicaId and operation")
    replica_id = entry["replicaId"]
    operation_raw = entry["operation"]
    if not isinstance(replica_id, str) or replica_id == "":
        raise PersistenceError("stored replicaId must be a non-empty string")
    if not isinstance(operation_raw, dict) or set(operation_raw.keys()) != {
        "operationId",
        "key",
        "value",
        "clock",
    }:
        raise PersistenceError("stored operation has an unexpected shape")
    try:
        operation = parse_operation_payload(operation_raw, replica_id)
    except ValueError as exc:
        raise PersistenceError(f"stored operation violates input constraints: {exc}") from exc
    return replica_id, operation


def load_data_file(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Read and strictly validate a data file.

    Returns the accepted operations in their original commit order. Raises
    PersistenceError when the file is missing-readable, not UTF-8 JSON, has
    an unexpected structure, or contains records violating the live input
    constraints.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise PersistenceError(f"cannot read data file {path!r}: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PersistenceError("data file is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"data file is not complete, valid JSON: {exc}") from exc

    if not isinstance(document, dict) or set(document.keys()) != {"version", "operations"}:
        raise PersistenceError("data file root must be an object with version and operations")
    version = document["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != DATA_FORMAT_VERSION:
        raise PersistenceError(f"unsupported data file version: {version!r}")
    records_raw = document["operations"]
    if not isinstance(records_raw, list):
        raise PersistenceError("data file operations must be a list")

    records: list[tuple[str, dict[str, Any]]] = []
    identities: set[tuple[str, str]] = set()
    for entry in records_raw:
        replica_id, operation = _validate_stored_operation(entry)
        identity = (replica_id, operation["operationId"])
        if identity in identities:
            raise PersistenceError(f"duplicate accepted operation {identity!r} in data file")
        identities.add(identity)
        records.append((replica_id, operation))
    return records


def ensure_data_file(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Validate the data-file location and return its committed records.

    A missing target file is accepted (its parent directory must exist and
    be writable); an existing target must be a regular, parseable data
    file. Anything else raises PersistenceError.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise PersistenceError(f"data file parent directory does not exist: {parent!r}")
    try:
        exists = os.path.lexists(path)
        if exists:
            mode = os.stat(path).st_mode
    except OSError as exc:
        raise PersistenceError(f"cannot access data file {path!r}: {exc}") from exc
    if exists:
        if not stat.S_ISREG(mode):
            raise PersistenceError(f"data file path is not a regular file: {path!r}")
        return load_data_file(path)
    return []


def _fsync_directory(directory: str) -> None:
    """Best-effort fsync of a directory so a rename is durable."""
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

    When ``data_file`` is given, state is recovered from it at construction
    and every first-accepted operation is persisted atomically (write to a
    temp file, fsync, rename) before the caller observes success. Without a
    data file the store is purely in memory, as before.
    """

    def __init__(self, data_file: str | None = None) -> None:
        self._lock = threading.Lock()
        self._candidates: dict[str, list[dict[str, Any]]] = {}
        self._operations: dict[tuple[str, str], dict[str, Any]] = {}
        self._accepted: list[tuple[str, dict[str, Any]]] = []
        self._data_file: str | None = None
        if data_file is not None:
            path = os.path.abspath(data_file)
            records = ensure_data_file(path)
            with self._lock:
                for replica_id, operation in records:
                    self._commit_locked(replica_id, operation)
                self._data_file = path
                if not records and not os.path.exists(path):
                    # Create the store up front so an unwritable target is a
                    # startup failure rather than a failure of the first write.
                    self._persist_locked()

    @staticmethod
    def _next_candidates(
        candidates: list[dict[str, Any]], replica_id: str, operation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compute the candidate list for a key after a new operation."""
        new_candidate = {
            "value": operation["value"],
            "clock": operation["clock"],
            "replicaId": replica_id,
            "operationId": operation["operationId"],
        }
        # A candidate already dominates the incoming clock: the write is
        # stale, so it is recorded but adds no new version.
        if any(clock_dominates(c["clock"], new_candidate["clock"]) for c in candidates):
            return list(candidates)
        survivors = [
            c for c in candidates if not clock_dominates(new_candidate["clock"], c["clock"])
        ]
        survivors.append(new_candidate)
        return survivors

    def _commit_locked(self, replica_id: str, operation: dict[str, Any]) -> None:
        """Commit an unseen operation to the in-memory state."""
        identity = (replica_id, operation["operationId"])
        self._operations[identity] = operation
        self._accepted.append((replica_id, operation))
        key = operation["key"]
        self._candidates[key] = self._next_candidates(
            self._candidates.get(key, []), replica_id, operation
        )

    def _persist_locked(self) -> None:
        """Atomically write the full accepted-operation log.

        The temp file is fsynced before an atomic rename, and the directory
        is fsynced afterwards, so a crash mid-write leaves either the
        previous complete file or the new complete file, never a mix.
        """
        assert self._data_file is not None
        document = {
            "version": DATA_FORMAT_VERSION,
            "operations": [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted
            ],
        }
        data = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
        directory = os.path.dirname(self._data_file)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".sestate-", suffix=".tmp")
        except OSError as exc:
            raise PersistenceError(f"cannot persist state to {self._data_file!r}: {exc}") from exc
        try:
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self._data_file)
                _fsync_directory(directory)
            except OSError as exc:
                raise PersistenceError(f"cannot persist state to {self._data_file!r}: {exc}") from exc
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def apply_operation(self, replica_id: str, operation: dict[str, Any]) -> HTTPStatus:
        """Apply a validated operation, returning the HTTP status to use.

        First-accepted operations are persisted (when configured) before the
        in-memory commit becomes visible; replays and conflicts touch neither
        the file nor the memory state.
        """
        with self._lock:
            identity = (replica_id, operation["operationId"])
            seen = self._operations.get(identity)
            if seen is not None:
                if seen == operation:
                    return HTTPStatus.OK
                return HTTPStatus.CONFLICT

            next_candidates = self._next_candidates(
                self._candidates.get(operation["key"], []), replica_id, operation
            )
            if self._data_file is not None:
                # Stage the full new log before touching the visible state.
                # The atomic rename is the single commit point; only after it
                # succeeds do memory structures change, so a crash at any
                # instant leaves file and recovered memory on the same commit.
                self._accepted.append((replica_id, operation))
                try:
                    self._persist_locked()
                except BaseException:
                    self._accepted.pop()
                    raise
            else:
                self._accepted.append((replica_id, operation))
            self._operations[identity] = operation
            self._candidates[operation["key"]] = next_candidates
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
        *,
        data_file: str | None = None,
        store: StateStore | None = None,
    ) -> None:
        super().__init__(server_address, handler_class or RequestHandler)
        self.store = store if store is not None else StateStore(data_file=data_file)


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
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Semantic State Engine HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--data-file",
        default=None,
        help=(
            "optional path to a file used to persist accepted operations and "
            "recover them on startup; without it the service stays purely in memory"
        ),
    )
    args = parser.parse_args(argv)

    try:
        store = StateStore(data_file=args.data_file)
    except PersistenceError as exc:
        print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    server = SemanticStateServer((args.host, args.port), store=store)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
