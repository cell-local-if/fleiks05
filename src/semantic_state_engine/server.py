"""HTTP boundary for the semantic state engine."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import sys
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

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


def _validate_clock(clock: Any, replica_id: str) -> dict[str, int]:
    """Validate a vector clock for ``replica_id`` and return a clean copy."""
    if not isinstance(clock, dict) or not clock:
        raise ValueError("clock must be a non-empty object")
    for component, tick in clock.items():
        if not isinstance(component, str) or component == "":
            raise ValueError("clock components must be non-empty strings")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("clock values must be non-negative integers")
    if replica_id not in clock:
        raise ValueError("clock must contain the replica id")
    return dict(clock)


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

    clock = _validate_clock(payload.get("clock"), replica_id)

    return {
        "operationId": payload["operationId"],
        "key": payload["key"],
        "value": payload["value"],
        "clock": dict(clock),
    }


def parse_resolve_payload(raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a conflict-resolution request body.

    The body must be a JSON object with exactly ``replicaId``,
    ``operationId``, ``value``, ``clock``, and ``candidates``. The first
    four follow the live operation constraints (the key comes from the
    request path); ``candidates`` is a non-empty list of distinct
    ``{"replicaId", "operationId"}`` identities. Returns a normalized
    resolution dict. Raises ValueError on any violation.
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
    if not isinstance(payload, dict) or set(payload.keys()) != {
        "replicaId",
        "operationId",
        "value",
        "clock",
        "candidates",
    }:
        raise ValueError(
            "payload must be an object with only replicaId, operationId, value, clock, candidates"
        )

    replica_id = payload["replicaId"]
    if not isinstance(replica_id, str) or replica_id == "":
        raise ValueError("replicaId must be a non-empty string")
    for field in ("operationId", "value"):
        value = payload.get(field)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{field} must be a non-empty string")
    clock = _validate_clock(payload.get("clock"), replica_id)

    candidates_raw = payload["candidates"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates must be a non-empty list")
    candidates: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for entry in candidates_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {"replicaId", "operationId"}:
            raise ValueError("each candidate must have only replicaId and operationId")
        candidate_replica = entry["replicaId"]
        candidate_operation = entry["operationId"]
        if not isinstance(candidate_replica, str) or candidate_replica == "":
            raise ValueError("candidate replicaId must be a non-empty string")
        if not isinstance(candidate_operation, str) or candidate_operation == "":
            raise ValueError("candidate operationId must be a non-empty string")
        identity = (candidate_replica, candidate_operation)
        if identity in identities:
            raise ValueError(f"duplicate candidate {identity!r}")
        identities.add(identity)
        candidates.append({"replicaId": candidate_replica, "operationId": candidate_operation})

    return {
        "replicaId": replica_id,
        "operationId": payload["operationId"],
        "value": payload["value"],
        "clock": clock,
        "candidates": candidates,
    }


SYNC_BATCH_MIN = 1
SYNC_BATCH_MAX = 100
SYNC_DEFAULT_LIMIT = 100


def parse_sync_batch(raw: bytes | str | dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Parse and validate a sync-import batch.

    The body must be a JSON object whose only key is ``operations`` holding
    between 1 and 100 records; each record must contain exactly ``replicaId``
    and ``operation`` and satisfy the live write constraints. Returns the
    records in request order. Raises ValueError on any violation.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("body must be UTF-8 JSON") from exc
    if isinstance(raw, str):
        try:
            document: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("body must be valid JSON") from exc
    else:
        document = raw
    if not isinstance(document, dict) or set(document.keys()) != {"operations"}:
        raise ValueError("body must be an object with only operations")
    records_raw = document["operations"]
    if not isinstance(records_raw, list) or not (SYNC_BATCH_MIN <= len(records_raw) <= SYNC_BATCH_MAX):
        raise ValueError("operations must be a list of 1-100 records")

    records: list[tuple[str, dict[str, Any]]] = []
    for entry in records_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {"replicaId", "operation"}:
            raise ValueError("each record must have only replicaId and operation")
        replica_id = entry["replicaId"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replicaId must be a non-empty string")
        try:
            operation = parse_operation_payload(entry["operation"], replica_id)
        except ValueError as exc:
            raise ValueError(f"invalid operation: {exc}") from exc
        records.append((replica_id, operation))
    return records


def _non_negative_int(token: str) -> int | None:
    """Parse a base-10 non-negative integer token, or return None.

    Only ASCII decimal digits are accepted, which rejects signs, decimals,
    whitespace, blanks, and non-ASCII numerals.
    """
    if not token or any(ch < "0" or ch > "9" for ch in token):
        return None
    return int(token)


MAX_BODY_BYTES = 1_048_576

BEARER_SCHEME = "Bearer "


class AuthTokenError(Exception):
    """Raised when ``--auth-token-file`` cannot provide a valid token.

    At startup this means the service refuses to start exactly as it does
    for a rejected data file: exit code 2 before the listening socket is
    bound.
    """


def load_auth_token(path: str) -> str:
    """Read and validate the bearer token file, returning the token.

    The path must identify a readable regular file whose entire content is
    one non-empty ASCII printable token (bytes 0x21-0x7E) with no
    whitespace and in particular no trailing newline. A missing,
    inaccessible, or non-regular target, an empty file, or any byte outside
    that range raises AuthTokenError. Error messages name only the path and
    the violation, never the token content.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise AuthTokenError(f"cannot access auth token file {path!r}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise AuthTokenError(f"auth token file is not a regular file: {path!r}")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise AuthTokenError(f"cannot read auth token file {path!r}: {exc}") from exc
    if not raw:
        raise AuthTokenError("auth token file must contain a non-empty token")
    if any(byte < 0x21 or byte > 0x7E for byte in raw):
        raise AuthTokenError(
            "auth token must be a single non-empty ASCII printable token "
            "with no whitespace or newline"
        )
    return raw.decode("ascii")


def _content_length_token(token: str) -> int | None:
    """Parse a Content-Length header value, or return None when malformed.

    Only a non-empty run of ASCII decimal digits is accepted, which rejects
    blanks, signs, decimals, whitespace, and non-ASCII numerals. Values with
    more significant digits than the cap are reported as just over it, so
    absurdly long digit strings never reach ``int()`` (which has its own
    conversion digit limit).
    """
    if not token or any(ch < "0" or ch > "9" for ch in token):
        return None
    significant = token.lstrip("0")
    if not significant:
        return 0
    if len(significant) > len(str(MAX_BODY_BYTES)):
        return MAX_BODY_BYTES + 1
    return int(significant)


def declared_body_length(headers: Any) -> int | None:
    """Return the validated Content-Length of a request, or None.

    The header must be present and every occurrence must be a plain ASCII
    decimal integer; multiple occurrences are tolerated only when they all
    declare the same length. A missing, blank, malformed, negative, or
    conflicting declaration returns None, which callers map to HTTP 400.
    """
    values = headers.get_all("Content-Length")
    if not values:
        return None
    declared: set[int] = set()
    for value in values:
        length = _content_length_token(value)
        if length is None:
            return None
        declared.add(length)
    if len(declared) != 1:
        return None
    return declared.pop()


def parse_paging_query(query: str) -> tuple[int, int] | None:
    """Validate an ``after``/``limit`` paging query string.

    Accepts only ``after`` (default 0) and ``limit`` (default 100, 1-100),
    each non-negative ASCII decimal integers with no repeats. Unknown
    parameters, malformed or negative values, and out-of-range limits return
    None. The sync-export and key-audit streams share these rules; bounds on
    ``after`` against the stream length are checked by the store.
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if any(len(values) != 1 for values in parsed.values()):
        return None
    allowed = {"after", "limit"}
    if not set(parsed) <= allowed:
        return None
    after = 0
    limit = SYNC_DEFAULT_LIMIT
    if "after" in parsed:
        after_value = _non_negative_int(parsed["after"][0])
        if after_value is None:
            return None
        after = after_value
    if "limit" in parsed:
        limit_value = _non_negative_int(parsed["limit"][0])
        if limit_value is None or not (1 <= limit_value <= SYNC_BATCH_MAX):
            return None
        limit = limit_value
    return after, limit


def parse_sync_query(query: str) -> tuple[int, int] | None:
    """Validate the sync-export query string.

    Delegates to :func:`parse_paging_query`; kept as a named entry point for
    the sync route.
    """
    return parse_paging_query(query)


def parse_metrics_query(query: str) -> bool:
    """Validate the metrics query string, which accepts no parameters.

    Returns True only for an empty query. Any parameter is rejected,
    including repeated names (``x=1&x=2``) and blank names/values
    (``x=``, ``x``, ``=1``); ``keep_blank_values`` ensures the latter are
    seen rather than silently dropped.
    """
    return not parse_qs(query, keep_blank_values=True)


def parse_checkpoint_payload(raw: bytes | str | dict[str, Any]) -> int:
    """Parse and validate a checkpoint body, returning the cursor.

    The body must be a JSON object whose only key is ``cursor`` holding a
    non-negative integer (booleans are rejected, as everywhere else). The
    bound against the accepted-log length is checked by the store, not here,
    because the cursor is only meaningful against a committed snapshot.
    Raises ValueError on any violation.
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
    if not isinstance(payload, dict) or set(payload.keys()) != {"cursor"}:
        raise ValueError("body must be an object with only cursor")
    cursor = payload["cursor"]
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError("cursor must be a non-negative integer")
    return cursor


def _escape_digest_string(value: str) -> str:
    """Escape a string for the canonical verification-digest input.

    Only the quote, the backslash, and control characters (U+0000-U+001F)
    are escaped — control characters always as ``\\u00XX`` with lowercase
    hex. Every other Unicode code point is written literally, so the
    resulting byte sequence is a single deterministic encoding of the
    string.
    """
    escaped: list[str] = []
    for ch in value:
        if ch == '"' or ch == "\\":
            escaped.append("\\" + ch)
        elif ch < " ":
            escaped.append(f"\\u{ord(ch):04x}")
        else:
            escaped.append(ch)
    return '"' + "".join(escaped) + '"'


def _verification_digest_input(candidates: dict[str, list[dict[str, Any]]]) -> bytes:
    """Serialize the current candidate sets to the canonical digest input.

    The result is a compact UTF-8 JSON array with one ``{"key","candidates"}``
    entry per key, keys in lexicographic (Unicode code point) order. Each
    candidate list is sorted by ``(replicaId, operationId)`` ascending, each
    candidate carries its fields in the fixed order ``value``, ``clock``,
    ``replicaId``, ``operationId``, and the clock's component names are
    sorted lexicographically. No whitespace is emitted anywhere. Only the
    current candidates are covered — never the accepted-operation log,
    stale writes, or checkpoints.
    """
    parts: list[str] = ["["]
    for key_index, key in enumerate(sorted(candidates)):
        if key_index:
            parts.append(",")
        parts.append('{"key":')
        parts.append(_escape_digest_string(key))
        parts.append(',"candidates":[')
        ordered = sorted(
            candidates[key], key=lambda c: (c["replicaId"], c["operationId"])
        )
        for index, candidate in enumerate(ordered):
            if index:
                parts.append(",")
            clock = ",".join(
                f"{_escape_digest_string(name)}:{tick}"
                for name, tick in sorted(candidate["clock"].items())
            )
            parts.append('{"value":')
            parts.append(_escape_digest_string(candidate["value"]))
            parts.append(',"clock":{')
            parts.append(clock)
            parts.append('},"replicaId":')
            parts.append(_escape_digest_string(candidate["replicaId"]))
            parts.append(',"operationId":')
            parts.append(_escape_digest_string(candidate["operationId"]))
            parts.append("}")
        parts.append("]}")
    parts.append("]")
    return "".join(parts).encode("utf-8")


def _key_audit_digest_input(records: list[tuple[str, dict[str, Any]]]) -> bytes:
    """Serialize one key's accepted-operation stream to the digest input.

    The result is a compact UTF-8 JSON array with one entry per accepted
    operation for the key, in global commit order (the caller filters the
    shared accepted-operation log, which already preserves that order).
    Each entry has the fixed shape
    ``{"replicaId":R,"operation":{"operationId":I,"key":K,"value":V,"clock":C}}``
    and the clock's component names are sorted lexicographically (Unicode
    code point order). No whitespace is emitted anywhere, and strings are
    escaped exactly as in :func:`_escape_digest_string` — only the quote,
    the backslash, and U+0000-U+001F control characters. An empty stream
    serializes to ``[]``.
    """
    parts: list[str] = ["["]
    for index, (replica_id, operation) in enumerate(records):
        if index:
            parts.append(",")
        clock = ",".join(
            f"{_escape_digest_string(name)}:{tick}"
            for name, tick in sorted(operation["clock"].items())
        )
        parts.append('{"replicaId":')
        parts.append(_escape_digest_string(replica_id))
        parts.append(',"operation":{"operationId":')
        parts.append(_escape_digest_string(operation["operationId"]))
        parts.append(',"key":')
        parts.append(_escape_digest_string(operation["key"]))
        parts.append(',"value":')
        parts.append(_escape_digest_string(operation["value"]))
        parts.append(',"clock":{')
        parts.append(clock)
        parts.append("}}}")
    parts.append("]")
    return "".join(parts).encode("utf-8")


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


def _validate_stored_checkpoints(
    document: Any, log_length: int
) -> dict[str, int]:
    """Validate the optional ``checkpoints`` section of a data file.

    Returns a clean ``{peerId: cursor}`` mapping. The section is optional
    (a version:1 file written before checkpoints existed simply has none);
    when present it must be an object of non-empty peer ids mapped to
    non-boolean non-negative integers no greater than the accepted-log
    length, since a cursor may never name an unaccepted record.
    """
    if "checkpoints" not in document:
        return {}
    raw = document["checkpoints"]
    if not isinstance(raw, dict):
        raise PersistenceError("data file checkpoints must be an object")
    checkpoints: dict[str, int] = {}
    for peer_id, cursor in raw.items():
        if not isinstance(peer_id, str) or peer_id == "":
            raise PersistenceError("checkpoint peerId must be a non-empty string")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise PersistenceError("checkpoint cursor must be a non-negative integer")
        if cursor > log_length:
            raise PersistenceError(
                f"checkpoint cursor {cursor} for {peer_id!r} is past the accepted log"
            )
        checkpoints[peer_id] = cursor
    return checkpoints


def load_data_file_full(
    path: str,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """Read and strictly validate a data file.

    Returns the accepted operations in their original commit order and the
    persisted ``{peerId: cursor}`` checkpoints (empty for a version:1 file
    written before checkpoints existed). Raises PersistenceError when the
    file is missing-readable, not UTF-8 JSON, has an unexpected structure,
    or contains records or checkpoints violating the live constraints.
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

    if not isinstance(document, dict) or not set(document.keys()) <= {
        "version",
        "operations",
        "checkpoints",
    } or "version" not in document or "operations" not in document:
        raise PersistenceError(
            "data file root must be an object with version and operations "
            "and optionally checkpoints"
        )
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
    checkpoints = _validate_stored_checkpoints(document, len(records))
    return records, checkpoints


def load_data_file(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Read and strictly validate a data file, returning its operations.

    Thin wrapper over :func:`load_data_file_full` for callers that only
    need the accepted-operation log; persisted checkpoints are validated
    the same way but not returned.
    """
    records, _ = load_data_file_full(path)
    return records


def ensure_data_file(
    path: str,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """Validate the data-file location and return its committed state.

    A missing target file is accepted (its parent directory must exist and
    be writable); an existing target must be a regular, parseable data
    file. Returns ``(records, checkpoints)``. Anything else raises
    PersistenceError.
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
        return load_data_file_full(path)
    return [], {}


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


def _fsync_directory_required(directory: str) -> None:
    """fsync a directory or raise PersistenceError; used by the startup probe."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise PersistenceError(f"cannot open directory for fsync {directory!r}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise PersistenceError(f"cannot fsync directory {directory!r}: {exc}") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


# Probe names used by the startup atomic-commit preflight. They live only in
# the data file's parent directory, carry an exclusive prefix, the owning
# process id, and a random component claimed via mkstemp; they are always
# removed before preflight returns.
_PREFLIGHT_PAYLOAD = b"semantic-state-engine preflight probe\n"
_PREFLIGHT_SUFFIXES = (".src.tmp", ".dst.tmp")


def _preflight_name_prefix(data_path_abs: str) -> str:
    return f".sestate-preflight-{os.path.basename(data_path_abs)}."


def _preflight_probe_paths(directory: str, scan_prefix: str) -> tuple[int, str, str]:
    """Create the source probe and derive the target probe path.

    mkstemp claims an exclusive random name (O_EXCL) so concurrent startups
    using the same directory never collide or reap one another's probes. The
    generated name keeps the shared scan prefix followed by the owning pid,
    so a later startup can attribute leftovers:
    ``<scan_prefix><pid>.<random>.src.tmp``.
    """
    fd, source = tempfile.mkstemp(
        dir=directory,
        prefix=f"{scan_prefix}{os.getpid()}.",
        suffix=".src.tmp",
    )
    target = source[: -len(".src.tmp")] + ".dst.tmp"
    return fd, source, target


def _preflight_owner_is_dead(name: str, prefix: str) -> bool:
    """Return True for probes left by a process that no longer exists.

    Probe names end in ``<pid>.<random>.(src|dst).tmp`` after the shared
    prefix. Entries we cannot attribute (unknown shape, a live pid, a pid we
    cannot signal) are never treated as reclaimable.
    """
    remainder = name[len(prefix) :]
    pid_token, _, rest = remainder.partition(".")
    if not pid_token.isdigit() or "." not in rest:
        return False
    if not rest.endswith(_PREFLIGHT_SUFFIXES):
        return False
    pid = int(pid_token)
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # The process exists; the probe may still be live.
        return False
    except OSError:
        return False
    return False


def _cleanup_stale_probes(directory: str, prefix: str) -> None:
    """Remove probes abandoned by earlier failed attempts of this data path."""
    try:
        names = os.listdir(directory)
    except OSError as exc:
        raise PersistenceError(f"cannot inspect directory {directory!r}: {exc}") from exc
    for name in names:
        if not (name.startswith(prefix) and name.endswith(".tmp")):
            continue
        if not _preflight_owner_is_dead(name, prefix):
            continue
        try:
            os.unlink(os.path.join(directory, name))
        except FileNotFoundError:
            pass
        except OSError:
            # A foreign-owned or locked leftover is outside this preflight's
            # scope (it cannot block our uniquely named probes); leave it.
            pass


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        # Best effort while unwinding a failed preflight; the original
        # failure is what must reach the caller.
        pass


def preflight_data_file_directory(path: str) -> None:
    """Probe the parent directory for the atomic-commit capability.

    Verifies, entirely with two probe files in the parent directory, that
    the service can exclusively create a named file, write and fsync it, and
    atomically replace it over another name (``os.replace``) followed by a
    directory fsync. Both probes are removed afterwards, together with probes
    abandoned by an earlier failed attempt whose owning process is gone.

    The target data file itself is never opened for writing, truncated,
    renamed, or replaced. Any failure raises PersistenceError so the caller
    can refuse to start before it begins listening.
    """
    data_path = os.path.abspath(path)
    directory = os.path.dirname(data_path)
    if not os.path.isdir(directory):
        raise PersistenceError(f"data file parent directory does not exist: {directory!r}")

    prefix = _preflight_name_prefix(data_path)
    _cleanup_stale_probes(directory, prefix)

    source = target = ""
    try:
        try:
            fd, source, target = _preflight_probe_paths(directory, prefix)
        except OSError as exc:
            raise PersistenceError(
                f"cannot create preflight probe in {directory!r}: {exc}"
            ) from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_PREFLIGHT_PAYLOAD)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_quiet(source)
            raise
        try:
            os.replace(source, target)
        except BaseException:
            _unlink_quiet(source)
            _unlink_quiet(target)
            raise
        try:
            _fsync_directory_required(directory)
            with open(target, "rb") as handle:
                committed = handle.read()
            if committed != _PREFLIGHT_PAYLOAD:
                raise PersistenceError(
                    f"atomic replace probe in {directory!r} did not preserve its contents"
                )
        finally:
            _unlink_quiet(target)
            _fsync_directory(directory)
    except PersistenceError:
        raise
    except OSError as exc:
        raise PersistenceError(
            f"startup atomic-commit preflight failed in {directory!r}: {exc}"
        ) from exc


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
        self._checkpoints: dict[str, int] = {}
        self._data_file: str | None = None
        if data_file is not None:
            path = os.path.abspath(data_file)
            # Probe directory-level atomic commit first; an existing data
            # file is never touched by the probe and is opened only for
            # reading afterwards.
            preflight_data_file_directory(path)
            records, checkpoints = ensure_data_file(path)
            with self._lock:
                for replica_id, operation in records:
                    self._commit_locked(replica_id, operation)
                self._checkpoints = dict(checkpoints)
                self._data_file = path
                if not records and not os.path.exists(path):
                    # The target is missing and the preflight proved the
                    # directory supports atomic commits; create the store up
                    # front so its first state is a durable empty log.
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
        document: dict[str, Any] = {
            "version": DATA_FORMAT_VERSION,
            "operations": [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted
            ],
            "checkpoints": dict(self._checkpoints),
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

    def apply_resolution(self, key: str, resolution: dict[str, Any]) -> tuple[HTTPStatus, str | None]:
        """Apply a validated conflict resolution for ``key``.

        A resolution commits as an ordinary operation in the shared commit
        order: because its clock dominates every current candidate, the
        normal candidate semantics atomically clear the dominated candidates
        and leave the resolution value as the only version. The operation
        therefore flows through sync export/import and the data file exactly
        like a local write.

        Returns ``(status, error)``: 201/200 with ``error=None`` on
        commit/replay, or 409 with ``"operation_conflict"`` (known identity,
        different content) or ``"resolution_conflict"`` (missing key, key not
        in conflict, or candidate-set mismatch). Raises ValueError when a
        candidate identity is unknown or the clock does not dominate every
        candidate; raises PersistenceError when the durable commit fails, in
        which case memory, the identity index, and the file are unchanged.
        """
        replica_id = resolution["replicaId"]
        operation = {
            "operationId": resolution["operationId"],
            "key": key,
            "value": resolution["value"],
            "clock": resolution["clock"],
        }
        with self._lock:
            identity = (replica_id, operation["operationId"])
            seen = self._operations.get(identity)
            if seen is not None:
                if seen == operation:
                    return HTTPStatus.OK, None
                return HTTPStatus.CONFLICT, "operation_conflict"

            # Every listed candidate must name a known operation, and the
            # resolution clock must dominate each candidate's clock.
            for candidate in resolution["candidates"]:
                candidate_identity = (candidate["replicaId"], candidate["operationId"])
                known = self._operations.get(candidate_identity)
                if known is None:
                    raise ValueError(f"unknown candidate identity {candidate_identity!r}")
                if not clock_dominates(operation["clock"], known["clock"]):
                    raise ValueError("clock does not dominate every candidate")

            current = self._candidates.get(key, [])
            if not current:
                return HTTPStatus.CONFLICT, "resolution_conflict"
            if all(c["value"] == current[0]["value"] for c in current):
                return HTTPStatus.CONFLICT, "resolution_conflict"
            current_identities = {(c["replicaId"], c["operationId"]) for c in current}
            requested_identities = {
                (c["replicaId"], c["operationId"]) for c in resolution["candidates"]
            }
            if current_identities != requested_identities:
                return HTTPStatus.CONFLICT, "resolution_conflict"

            next_candidates = self._next_candidates(current, replica_id, operation)
            if self._data_file is not None:
                # Same commit discipline as local writes: the atomic rename
                # is the single commit point, and memory moves only after it.
                self._accepted.append((replica_id, operation))
                try:
                    self._persist_locked()
                except BaseException:
                    self._accepted.pop()
                    raise
            else:
                self._accepted.append((replica_id, operation))
            self._operations[identity] = operation
            self._candidates[key] = next_candidates
            return HTTPStatus.CREATED, None

    def get_sync_operations(
        self, after: int, limit: int
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """Return one page of the accepted-operation log from a single snapshot.

        ``after`` is the number of records already skipped (a 0-based cursor)
        and ``limit`` the page size. The slice, the returned cursor, and
        ``has_more`` are all computed against the same snapshot under the
        commit lock, so pages interleave cleanly with concurrent commits.
        Returns ``(records, next_cursor, has_more)`` where ``next_cursor`` is
        the number of records skipped after this page.
        """
        with self._lock:
            total = len(self._accepted)
            if after > total:
                raise ValueError("after is past the end of the operation log")
            page = [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted[after : after + limit]
            ]
        next_cursor = after + len(page)
        return page, next_cursor, next_cursor < total

    def get_key_operations(
        self, key: str, after: int, limit: int
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """Return one page of one key's accepted operations from one snapshot.

        The audit stream is the shared accepted-operation log filtered to
        records whose ``operation.key`` equals ``key``, kept in global commit
        order. It therefore includes stale writes and resolutions accepted
        via ``/resolve`` (both are ordinary committed operations) and excludes
        other keys, replays, conflicts, and uncommitted requests.

        ``after`` is the number of this key's records already skipped (a
        per-key 0-based cursor) and ``limit`` the page size. The filtered
        list, the slice, the returned cursor, and ``has_more`` are all
        computed against the same snapshot under the commit lock, so pages
        interleave cleanly with concurrent commits and never observe half an
        import batch. Returns ``(records, next_cursor, has_more)`` where
        ``next_cursor`` is the number of the key's records skipped after this
        page. Raises ValueError when ``after`` is past the key's record count.
        """
        with self._lock:
            key_log = [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted
                if operation["key"] == key
            ]
            total = len(key_log)
            if after > total:
                raise ValueError("after is past the end of the key's audit log")
            page = key_log[after : after + limit]
        next_cursor = after + len(page)
        return page, next_cursor, next_cursor < total

    def get_metrics(self) -> dict[str, int]:
        """Return read-only engine counters from a single locked snapshot.

        All six counters are computed together under the same commit lock
        used by local writes, sync imports, and resolutions, so they always
        describe one commit: a read can never observe half an import batch
        or a partially applied repair. The snapshot mutates neither memory
        nor the data file.

        ``acceptedOperations`` counts first-accepted operations in the
        shared log (ordinary writes, stale writes that add no candidate, and
        conflict repairs) and therefore excludes identical replays,
        conflicting/invalid requests, and operations whose durable commit
        failed. ``keys`` counts keys that currently hold at least one
        candidate and ``candidateVersions`` the candidates across them; a
        key contributes to exactly one of ``conflictKeys`` (its candidates
        disagree on the value) or ``resolvedKeys`` (all agree), so those two
        always sum to ``keys``. ``replicas`` is the number of distinct
        ``replicaId`` values in the accepted log (a repair counts under its
        initiating replica). With ``--data-file`` the counters are rebuilt
        identically during recovery, so they match the pre-restart values.
        """
        with self._lock:
            accepted = len(self._accepted)
            replicas = {replica_id for replica_id, _ in self._accepted}
            keys = 0
            candidate_versions = 0
            conflict_keys = 0
            for candidates in self._candidates.values():
                keys += 1
                candidate_versions += len(candidates)
                first_value = candidates[0]["value"]
                if any(c["value"] != first_value for c in candidates):
                    conflict_keys += 1
            resolved_keys = keys - conflict_keys
            return {
                "acceptedOperations": accepted,
                "keys": keys,
                "candidateVersions": candidate_versions,
                "conflictKeys": conflict_keys,
                "resolvedKeys": resolved_keys,
                "replicas": len(replicas),
            }

    def get_verification_digest(self) -> dict[str, Any]:
        """Return the read-only replica-convergence digest from one snapshot.

        The digest input and both counters are computed together under the
        same commit lock used by local writes, sync imports, and repairs, so
        the response always describes a single commit: a read can never
        observe half an import batch or a partially applied repair. The
        snapshot mutates neither memory nor the data file.

        The digest covers only the current candidate sets — never the
        accepted-operation log, stale writes that added no candidate, or
        checkpoints — serialized by :func:`_verification_digest_input` and
        hashed with SHA-256. ``keys`` counts keys holding at least one
        candidate and ``candidateVersions`` the candidates across them.
        With ``--data-file`` the candidate state is rebuilt identically
        during recovery, so the same state yields the same response before
        and after a restart.
        """
        with self._lock:
            keys = len(self._candidates)
            candidate_versions = sum(len(c) for c in self._candidates.values())
            digest_input = _verification_digest_input(self._candidates)
        return {
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "keys": keys,
            "candidateVersions": candidate_versions,
        }

    def get_key_audit_digest(self, key: str) -> dict[str, Any]:
        """Return one key's audit-integrity digest from a single snapshot.

        The key's filtered accepted-operation stream and the digest input
        are computed together under the same commit lock used by local
        writes, sync imports, and repairs, so ``operations`` and ``digest``
        always describe the same commit: a read can never observe half an
        import batch or a partially applied repair. The snapshot mutates
        neither memory, the data file, logs, nor checkpoints.

        The digest covers the key's whole audit stream in global commit
        order — including stale writes that added no candidate and accepted
        conflict repairs — serialized by :func:`_key_audit_digest_input` and
        hashed with SHA-256. Identical replays, conflicting or malformed
        requests, operations for other keys, and operations whose durable
        commit failed never enter the log and so never influence the
        digest. A key with no history yields the hash of ``[]`` with
        ``operations`` 0. With ``--data-file`` the log is rebuilt
        identically during recovery, so the same history yields the same
        digest before and after a restart.
        """
        with self._lock:
            key_records = [
                (replica_id, operation)
                for replica_id, operation in self._accepted
                if operation["key"] == key
            ]
            operations = len(key_records)
            digest_input = _key_audit_digest_input(key_records)
        return {
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "operations": operations,
        }

    def import_operations(
        self, records: list[tuple[str, dict[str, Any]]]
    ) -> tuple[HTTPStatus, int, int]:
        """Import already-validated records sequentially as one atomic batch.

        Records share the single commit order with local writes: the whole
        batch is processed under one lock hold, so concurrent readers never
        observe half a batch. Unknown identities are accepted with the same
        write semantics as local requests; known identities with identical
        content are replays, while a known identity with different content is
        a conflict and changes nothing (memory, identity index, and data file
        all stay as they were). With a data file, all new operations are
        persisted in one atomic commit before memory becomes visible.

        Returns ``(status, accepted, replayed)``: 201 when at least one new
        operation was committed, 200 when every record was a replay, 409 when
        any record conflicts (the batch is untouched).
        """
        with self._lock:
            # Dry-run against the committed identities plus this batch. This
            # both validates the whole batch and fixes its commit segment
            # before anything is persisted or made visible.
            staged = dict(self._operations)
            new_records: list[tuple[str, dict[str, Any]]] = []
            accepted = 0
            replayed = 0
            for replica_id, operation in records:
                identity = (replica_id, operation["operationId"])
                seen = staged.get(identity)
                if seen is None:
                    staged[identity] = operation
                    new_records.append((replica_id, operation))
                    accepted += 1
                elif seen == operation:
                    replayed += 1
                else:
                    return HTTPStatus.CONFLICT, 0, 0

            if not new_records:
                return HTTPStatus.OK, accepted, replayed

            # Commit the new operations together. The atomic rename is the
            # single durable commit point; memory and the identity index move
            # only after it succeeds, so a failed durable commit leaves
            # everything exactly as it was before the batch.
            if self._data_file is not None:
                previous_length = len(self._accepted)
                self._accepted.extend(new_records)
                try:
                    self._persist_locked()
                except BaseException:
                    del self._accepted[previous_length:]
                    raise
            else:
                self._accepted.extend(new_records)
            for replica_id, operation in new_records:
                self._operations[(replica_id, operation["operationId"])] = operation
                key = operation["key"]
                self._candidates[key] = self._next_candidates(
                    self._candidates.get(key, []), replica_id, operation
                )
            return HTTPStatus.CREATED, accepted, replayed

    def save_checkpoint(self, peer_id: str, cursor: int) -> tuple[HTTPStatus, str | None]:
        """Persist a sender-side replication checkpoint for ``peer_id``.

        A checkpoint is not an operation: it never touches the accepted
        log, the identity index, the candidate state, sync export, the
        per-key audit, or the metrics counters. It shares their commit
        lock, however, so validating the cursor against the accepted-log
        length, persisting, and making the new cursor visible are one
        indivisible commit: the cursor can never name a record that is not
        durably accepted, and a concurrent reader sees either the old or
        the new checkpoint, never a half state.

        Returns ``(status, error)``: 200 with ``error=None`` for a first
        registration, an equal-value replay, or an advance; 409 with
        ``"checkpoint_conflict"`` when a larger cursor is already stored
        (the stored value never moves backwards). Raises ValueError when
        ``cursor`` is past the accepted-log length of the validation
        snapshot; raises PersistenceError when the durable commit fails, in
        which case memory and the file are unchanged and the request can be
        retried.
        """
        with self._lock:
            # Validate against the same committed snapshot the write will
            # use, so the cursor can never be committed past an unaccepted
            # or not-yet-durable record.
            if cursor > len(self._accepted):
                raise ValueError("cursor is past the end of the accepted log")
            current = self._checkpoints.get(peer_id)
            if current is not None and current > cursor:
                return HTTPStatus.CONFLICT, "checkpoint_conflict"
            if current == cursor:
                # An equal-value replay is idempotent and needs no durable
                # rewrite. (A first registration at cursor 0 is not a
                # replay: current is None, so it falls through and is
                # durably committed like any other registration.)
                return HTTPStatus.OK, None
            if self._data_file is not None:
                # Same commit discipline as the operation paths: stage the
                # new mapping, make the atomic rename the single commit
                # point, and move the visible state only after it succeeds.
                self._checkpoints[peer_id] = cursor
                try:
                    self._persist_locked()
                except BaseException:
                    if current is None:
                        del self._checkpoints[peer_id]
                    else:
                        self._checkpoints[peer_id] = current
                    raise
            else:
                self._checkpoints[peer_id] = cursor
            return HTTPStatus.OK, None

    def get_checkpoint(self, peer_id: str) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return the registered checkpoint, or 404 when ``peer_id`` is unknown.

        Read under the shared commit lock so the response never observes a
        checkpoint commit halfway through its durable update.
        """
        with self._lock:
            cursor = self._checkpoints.get(peer_id)
        if cursor is None:
            return HTTPStatus.NOT_FOUND, {"error": "not_found"}
        return HTTPStatus.OK, {"peerId": peer_id, "cursor": cursor}

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
        auth_token: str | None = None,
    ) -> None:
        # Build (and thus preflight/recover) the store before binding and
        # listening, so a rejected data file fails startup before any port is
        # open rather than surfacing on the first accepted write.
        resolved_store = store if store is not None else StateStore(data_file=data_file)
        super().__init__(server_address, handler_class or RequestHandler)
        self.store = resolved_store
        # None leaves the service anonymous, as before; a non-None token
        # gates every route except GET /health.
        self.auth_token = auth_token


_FALLBACK_STORE = StateStore()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticStateEngine/0.1"

    @property
    def _store(self) -> StateStore:
        return getattr(self.server, "store", _FALLBACK_STORE)

    def _authorized(self) -> bool:
        """Authenticate a request against the optional bearer token.

        With no configured token every request is anonymous. With a token,
        exactly one ``Authorization`` header is required whose value is
        precisely ``Bearer `` followed by the token; the comparison uses the
        standard library's constant-time comparison. A missing, repeated,
        malformed, or non-matching header fails.
        """
        token = getattr(self.server, "auth_token", None)
        if token is None:
            return True
        values = self.headers.get_all("Authorization")
        if values is None or len(values) != 1:
            return False
        # Compare raw header bytes (headers are latin-1 on the wire) against
        # the ASCII expected value; constant-time and safe for any header.
        presented = values[0].encode("latin-1", errors="replace")
        expected = (BEARER_SCHEME + token).encode("ascii")
        return hmac.compare_digest(presented, expected)

    def _unauthorized(self) -> None:
        """Answer 401 without touching the store, files, or the request body."""
        self.close_connection = True
        body = json.dumps({"error": "unauthorized"}, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    def _checkpoint_route(self) -> tuple[bool, str]:
        """Match ``/v1/sync/peers/{peerId}/checkpoint`` on the raw path.

        Returns ``(matched, peer_id)``. Unlike :meth:`_path_segments`, the
        empty segment of ``/v1/sync/peers//checkpoint`` is preserved: the
        path still matches the route shape (so it is not a generic 404) but
        yields an empty ``peer_id``, letting the caller reject it as an
        invalid request. Any other segment count falls through to 404, so
        extra segments such as ``.../checkpoint/extra`` never match.
        """
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 6:
            return False, ""
        if parts[1:4] != ["v1", "sync", "peers"] or parts[5] != "checkpoint":
            return False, ""
        return True, unquote(parts[4])

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(HTTPStatus.OK, health_payload())
            return
        if not self._authorized():
            self._unauthorized()
            return
        segments = self._path_segments()
        if len(segments) == 2 and segments[0] == "v1" and segments[1] == "metrics":
            self._handle_metrics_get()
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "verification"
            and segments[2] == "digest"
        ):
            self._handle_verification_digest_get()
            return
        if len(segments) == 3 and segments[0] == "v1" and segments[1] == "states":
            status, payload = self._store.get_state(segments[2])
            self._json(status, payload)
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "sync"
            and segments[2] == "operations"
        ):
            self._handle_sync_get()
            return
        if (
            len(segments) == 5
            and segments[0] == "v1"
            and segments[1] == "audit"
            and segments[2] == "keys"
            and segments[4] == "operations"
        ):
            self._handle_audit_get(segments[3])
            return
        if (
            len(segments) == 5
            and segments[0] == "v1"
            and segments[1] == "audit"
            and segments[2] == "keys"
            and segments[4] == "digest"
        ):
            self._handle_audit_digest_get(segments[3])
            return
        matched, checkpoint_peer = self._checkpoint_route()
        if matched:
            self._handle_checkpoint_get(checkpoint_peer)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _check_content_length(self) -> int | None:
        """Validate Content-Length under the shared POST size contract.

        Content-Length is validated before authentication and before
        anything else: a missing, malformed, or conflicting declaration is
        answered with HTTP 400 and a declared length over
        ``MAX_BODY_BYTES`` with HTTP 413 — both before a single body byte is
        read, so an over-limit declaration is rejected on its declared size
        alone, however invalid the content would have been. Returns the
        declared length, or None when the error response has already been
        sent. Either rejection closes the connection because the unread body
        can no longer be framed.
        """
        length = declared_body_length(self.headers)
        if length is None:
            self.close_connection = True
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return None
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "payload_too_large"})
            return None
        return length

    def _read_bounded_body(self) -> bytes | None:
        """Validate Content-Length, authenticate, then read the POST body.

        Content-Length's 400/413 take precedence over authentication, so the
        size contract is identical whether or not a token is configured. A
        request with a legal length that fails authentication is answered
        with HTTP 401 without reading a single body byte; the connection is
        closed because the unread body can no longer be framed. Only after
        authentication are exactly the declared bytes read. Returns the body,
        or None when an error response has already been sent.
        """
        length = self._check_content_length()
        if length is None:
            return None
        if not self._authorized():
            self._unauthorized()
            return None
        return self.rfile.read(length) if length > 0 else b""

    def _handle_checkpoint_get(self, peer_id: str) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        if peer_id == "":
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        status, payload = self._store.get_checkpoint(peer_id)
        self._json(status, payload)

    def _handle_checkpoint_post(self, peer_id: str) -> None:
        if peer_id == "":
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            cursor = parse_checkpoint_payload(raw)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, error = self._store.save_checkpoint(peer_id, cursor)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        except PersistenceError:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json(status, {"error": error})
            return
        self._json(status, {"peerId": peer_id, "cursor": cursor})

    def _handle_metrics_get(self) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(HTTPStatus.OK, self._store.get_metrics())

    def _handle_verification_digest_get(self) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(HTTPStatus.OK, self._store.get_verification_digest())

    def _handle_sync_get(self) -> None:
        params = parse_sync_query(urlsplit(self.path).query)
        if params is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        after, limit = params
        try:
            page, next_cursor, has_more = self._store.get_sync_operations(after, limit)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(
            HTTPStatus.OK,
            {"operations": page, "nextCursor": next_cursor, "hasMore": has_more},
        )

    def _handle_audit_get(self, key: str) -> None:
        params = parse_paging_query(urlsplit(self.path).query)
        if params is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        after, limit = params
        try:
            page, next_cursor, has_more = self._store.get_key_operations(key, after, limit)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(
            HTTPStatus.OK,
            {"operations": page, "nextCursor": next_cursor, "hasMore": has_more},
        )

    def _handle_audit_digest_get(self, key: str) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        # Every path key is a valid audit subject: a key with no accepted
        # history hashes the empty stream and reports operations 0.
        self._json(HTTPStatus.OK, self._store.get_key_audit_digest(key))

    def _handle_sync_post(self) -> None:
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            records = parse_sync_batch(raw)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, accepted, replayed = self._store.import_operations(records)
        except PersistenceError:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json(status, {"error": "operation_conflict"})
            return
        self._json(
            status,
            {
                "status": "created" if status is HTTPStatus.CREATED else "ok",
                "accepted": accepted,
                "replayed": replayed,
            },
        )

    def _handle_resolve_post(self, key: str) -> None:
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            resolution = parse_resolve_payload(raw)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, error = self._store.apply_resolution(key, resolution)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        except PersistenceError:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json(status, {"error": error})
            return
        self._json(
            status,
            {
                "status": "created" if status is HTTPStatus.CREATED else "ok",
                "key": key,
                "replicaId": resolution["replicaId"],
                "operationId": resolution["operationId"],
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        # With a token configured, the four POST endpoints keep Content-
        # Length's 400/413 ahead of authentication, and authentication ahead
        # of route matching (so unknown routes cannot be reached unauthenticated).
        # The length check is header-only and reads no body; applying it here,
        # before the route match, is what lets an existing endpoint's size
        # error take precedence without matching the route first. With no
        # token configured this step is skipped, preserving the prior
        # route-first ordering exactly.
        if getattr(self.server, "auth_token", None) is not None:
            if self._check_content_length() is None:
                return
            if not self._authorized():
                self._unauthorized()
                return
        matched, checkpoint_peer = self._checkpoint_route()
        if matched:
            self._handle_checkpoint_post(checkpoint_peer)
            return
        segments = self._path_segments()
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "replicas"
            and segments[3] == "operations"
        ):
            replica_id = segments[2]
            raw = self._read_bounded_body()
            if raw is None:
                return
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
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "resolve"
        ):
            self._handle_resolve_post(segments[2])
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "sync"
            and segments[2] == "operations"
        ):
            self._handle_sync_post()
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
    parser.add_argument(
        "--auth-token-file",
        default=None,
        help=(
            "optional path to a readable regular file holding exactly one "
            "non-empty ASCII printable bearer token (no whitespace or trailing "
            "newline); when given, every route except GET /health requires "
            "'Authorization: Bearer <token>' and an invalid file refuses startup"
        ),
    )
    args = parser.parse_args(argv)

    auth_token: str | None = None
    if args.auth_token_file is not None:
        try:
            auth_token = load_auth_token(args.auth_token_file)
        except AuthTokenError as exc:
            print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        store = StateStore(data_file=args.data_file)
    except PersistenceError as exc:
        print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    server = SemanticStateServer(
        (args.host, args.port), store=store, auth_token=auth_token
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
