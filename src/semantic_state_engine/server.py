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
from typing import Any, Callable
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


AUTO_RESOLVE_POLICIES = ("lowest_identity", "highest_identity")


def parse_auto_resolve_payload(raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate an automatic conflict-resolution request body.

    The body must be a JSON object with exactly ``replicaId``,
    ``operationId``, ``clock``, and ``policy``. The first three follow the
    live resolution constraints (the key and the chosen value come from the
    server: the value is taken from a current candidate selected by
    ``policy``); ``policy`` must be one of ``"lowest_identity"`` (candidate
    with the smallest ``(replicaId, operationId)``) or ``"highest_identity"``
    (candidate with the largest). Returns a normalized request dict. Raises
    ValueError on any violation.
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
        "clock",
        "policy",
    }:
        raise ValueError(
            "payload must be an object with only replicaId, operationId, clock, policy"
        )

    replica_id = payload["replicaId"]
    if not isinstance(replica_id, str) or replica_id == "":
        raise ValueError("replicaId must be a non-empty string")
    operation_id = payload["operationId"]
    if not isinstance(operation_id, str) or operation_id == "":
        raise ValueError("operationId must be a non-empty string")
    clock = _validate_clock(payload.get("clock"), replica_id)
    policy = payload["policy"]
    if policy not in AUTO_RESOLVE_POLICIES:
        raise ValueError("policy must be 'lowest_identity' or 'highest_identity'")

    return {
        "replicaId": replica_id,
        "operationId": operation_id,
        "clock": clock,
        "policy": policy,
    }


AUTO_RESOLVE_BATCH_MIN = 1
AUTO_RESOLVE_BATCH_MAX = 100


def parse_auto_resolve_batch(raw: bytes | str | dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and validate a batch of automatic conflict-resolution requests.

    The body must be a JSON object whose only key is ``resolutions`` holding
    between 1 and 100 entries in request order. Each entry must contain
    exactly ``key``, ``replicaId``, ``operationId``, ``clock``, and
    ``policy``: the four fields of a single automatic resolution plus the
    target key (there is no path key for the batch route). The constraints
    match :func:`parse_auto_resolve_payload`, with two extra batch-level
    rules: no two entries may name the same key or the same
    ``(replicaId, operationId)`` identity. Returns the normalized entries in
    request order. Raises ValueError on any violation.
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
    if not isinstance(document, dict) or set(document.keys()) != {"resolutions"}:
        raise ValueError("body must be an object with only resolutions")
    entries_raw = document["resolutions"]
    if not isinstance(entries_raw, list) or not (
        AUTO_RESOLVE_BATCH_MIN <= len(entries_raw) <= AUTO_RESOLVE_BATCH_MAX
    ):
        raise ValueError("resolutions must be a list of 1-100 entries")

    entries: list[dict[str, Any]] = []
    keys: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for entry in entries_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "key",
            "replicaId",
            "operationId",
            "clock",
            "policy",
        }:
            raise ValueError(
                "each resolution must have only key, replicaId, operationId, clock, policy"
            )
        key = entry["key"]
        if not isinstance(key, str) or key == "":
            raise ValueError("key must be a non-empty string")
        replica_id = entry["replicaId"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replicaId must be a non-empty string")
        operation_id = entry["operationId"]
        if not isinstance(operation_id, str) or operation_id == "":
            raise ValueError("operationId must be a non-empty string")
        clock = _validate_clock(entry.get("clock"), replica_id)
        policy = entry["policy"]
        if policy not in AUTO_RESOLVE_POLICIES:
            raise ValueError("policy must be 'lowest_identity' or 'highest_identity'")
        if key in keys:
            raise ValueError(f"duplicate key {key!r} in batch")
        identity = (replica_id, operation_id)
        if identity in identities:
            raise ValueError(f"duplicate identity {identity!r} in batch")
        keys.add(key)
        identities.add(identity)
        entries.append(
            {
                "key": key,
                "replicaId": replica_id,
                "operationId": operation_id,
                "clock": clock,
                "policy": policy,
            }
        )
    return entries


TRANSACTION_MIN_OPERATIONS = 1
TRANSACTION_MAX_OPERATIONS = 100


def _parse_transaction_entries(entries_raw: Any) -> list[dict[str, Any]]:
    """Validate the operation entries of a transaction, in request order.

    Each entry must contain exactly ``key``, ``replicaId``, ``operationId``,
    ``value``, ``clock``, and ``candidates``: the fields of an ordinary
    write (with the initiating replica carried on the entry, as in a sync
    record) plus the expected pre-commit candidate identity set for the
    key. ``candidates`` may be empty (the key is expected to hold no
    current candidates); when non-empty, each element is a distinct
    ``{"replicaId", "operationId"}`` identity. The normalized entries sort
    each candidate set by ``(replicaId, operationId)`` so two requests that
    name the same set in a different order are the same transaction
    content. No two entries may name the same key or the same
    ``(replicaId, operationId)`` identity. Raises ValueError on any
    violation.
    """
    if not isinstance(entries_raw, list) or not (
        TRANSACTION_MIN_OPERATIONS <= len(entries_raw) <= TRANSACTION_MAX_OPERATIONS
    ):
        raise ValueError("operations must be a list of 1-100 entries")

    entries: list[dict[str, Any]] = []
    keys: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for entry in entries_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "key",
            "replicaId",
            "operationId",
            "value",
            "clock",
            "candidates",
        }:
            raise ValueError(
                "each operation must have only key, replicaId, operationId, "
                "value, clock, candidates"
            )
        key = entry["key"]
        if not isinstance(key, str) or key == "":
            raise ValueError("key must be a non-empty string")
        replica_id = entry["replicaId"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replicaId must be a non-empty string")
        operation_id = entry["operationId"]
        if not isinstance(operation_id, str) or operation_id == "":
            raise ValueError("operationId must be a non-empty string")
        value = entry["value"]
        if not isinstance(value, str) or value == "":
            raise ValueError("value must be a non-empty string")
        clock = _validate_clock(entry.get("clock"), replica_id)

        candidates_raw = entry["candidates"]
        if not isinstance(candidates_raw, list):
            raise ValueError("candidates must be a list")
        candidates: list[dict[str, str]] = []
        candidate_identities: set[tuple[str, str]] = set()
        for candidate in candidates_raw:
            if not isinstance(candidate, dict) or set(candidate.keys()) != {
                "replicaId",
                "operationId",
            }:
                raise ValueError("each candidate must have only replicaId and operationId")
            candidate_replica = candidate["replicaId"]
            candidate_operation = candidate["operationId"]
            if not isinstance(candidate_replica, str) or candidate_replica == "":
                raise ValueError("candidate replicaId must be a non-empty string")
            if not isinstance(candidate_operation, str) or candidate_operation == "":
                raise ValueError("candidate operationId must be a non-empty string")
            candidate_identity = (candidate_replica, candidate_operation)
            if candidate_identity in candidate_identities:
                raise ValueError(f"duplicate candidate {candidate_identity!r}")
            candidate_identities.add(candidate_identity)
            candidates.append(
                {"replicaId": candidate_replica, "operationId": candidate_operation}
            )
        candidates.sort(key=lambda c: (c["replicaId"], c["operationId"]))

        if key in keys:
            raise ValueError(f"duplicate key {key!r} in transaction")
        identity = (replica_id, operation_id)
        if identity in identities:
            raise ValueError(f"duplicate identity {identity!r} in transaction")
        keys.add(key)
        identities.add(identity)
        entries.append(
            {
                "key": key,
                "replicaId": replica_id,
                "operationId": operation_id,
                "value": value,
                "clock": clock,
                "candidates": candidates,
            }
        )
    return entries


def parse_transaction_apply(raw: bytes | str | dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Parse and validate an atomic multi-key transaction body.

    The body must be a JSON object with exactly ``transactionId`` (a
    non-empty string) and ``operations`` (1-100 entries, see
    :func:`_parse_transaction_entries`). Returns the transaction id and the
    normalized entries in request order. Raises ValueError on any
    violation.
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
    if not isinstance(document, dict) or set(document.keys()) != {
        "transactionId",
        "operations",
    }:
        raise ValueError("body must be an object with only transactionId and operations")
    transaction_id = document["transactionId"]
    if not isinstance(transaction_id, str) or transaction_id == "":
        raise ValueError("transactionId must be a non-empty string")
    return transaction_id, _parse_transaction_entries(document["operations"])


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
    whitespace, blanks, and non-ASCII numerals. A run of digits longer than
    the interpreter's integer-string conversion limit (Python 3.11+) raises
    ``ValueError`` from ``int``; that is a malformed value like any other
    and is reported as None rather than escaping into a dropped connection.
    """
    if not token or any(ch < "0" or ch > "9" for ch in token):
        return None
    try:
        return int(token)
    except ValueError:
        return None


MAX_BODY_BYTES = 1_048_576


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


def parse_state_at_query(query: str) -> int | None:
    """Validate the history-query query string, returning the cursor.

    Exactly one parameter is accepted: ``cursor``, which is required and
    must appear exactly once with a non-negative ASCII decimal integer
    value (signs, decimals, whitespace, blanks, and non-ASCII numerals
    rejected). A missing, repeated, blank, or malformed ``cursor`` and any
    unknown parameter return None. The bound against the accepted-log
    length is checked by the store against the committed snapshot, not
    here, because the cursor is only meaningful against one.
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if set(parsed) != {"cursor"}:
        return None
    values = parsed["cursor"]
    if len(values) != 1:
        return None
    return _non_negative_int(values[0])


def parse_audit_log_verify_query(
    query: str,
) -> tuple[int, int, str, int] | None:
    """Validate the global audit-chain verification query string.

    All four parameters are **required**, each appearing exactly once:
    the paging parameters ``after`` (a non-negative ASCII decimal
    integer — there is no default) and ``limit`` (between 1 and 100),
    plus the two external expectations ``head`` (exactly 64 lowercase
    hexadecimal characters, the chain-tail digest the caller expects)
    and ``count`` (a non-negative ASCII decimal integer, the total chain
    length the caller expects). A missing, repeated, blank-named, or
    unknown parameter, a blank or malformed ``head`` (uppercase,
    non-hex, or the wrong length all rejected), a blank, signed, or
    non-ASCII-decimal ``count``, and a blank, signed, non-ASCII-decimal,
    or out-of-range ``after``/``limit`` return None. The bound on
    ``after`` against the chain length is checked by the store against
    the committed snapshot (``after`` equal to the chain length is a
    valid empty page).
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if any(len(values) != 1 for values in parsed.values()):
        return None
    if set(parsed) != {"after", "limit", "head", "count"}:
        return None
    head = parsed["head"][0]
    if not _is_sha256_hex64(head):
        return None
    count = _non_negative_int(parsed["count"][0])
    if count is None:
        return None
    after = _non_negative_int(parsed["after"][0])
    if after is None:
        return None
    limit = _non_negative_int(parsed["limit"][0])
    if limit is None or not (1 <= limit <= SYNC_BATCH_MAX):
        return None
    return after, limit, head, count


def parse_peer_pickup_query(query: str) -> tuple[int, int] | None:
    """Validate the peer-progress pickup query string.

    Unlike :func:`parse_paging_query`, both ``after`` and ``limit`` are
    required parameters: a request missing either one (including a bare
    request with no query string) is rejected, as are repeated names,
    unknown parameters, blank, negative, or non-ASCII-decimal values,
    and a ``limit`` outside ``1-100``. The bound on ``after`` against
    the peer's unconsumed record count is checked by the store against
    the committed snapshot.
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if any(len(values) != 1 for values in parsed.values()):
        return None
    if set(parsed) != {"after", "limit"}:
        return None
    after_value = _non_negative_int(parsed["after"][0])
    if after_value is None:
        return None
    limit_value = _non_negative_int(parsed["limit"][0])
    if limit_value is None or not (1 <= limit_value <= SYNC_BATCH_MAX):
        return None
    return after_value, limit_value


def parse_peer_receipts_query(query: str) -> tuple[int, int] | None:
    """Validate the peer-receipts query string.

    Delegates to :func:`parse_peer_pickup_query` — both ``after`` and
    ``limit`` are required, with the same rejection rules; kept as a named
    entry point for the receipts route. The bound on ``after`` against the
    peer's committed receipt count is checked by the store against the
    committed snapshot.
    """
    return parse_peer_pickup_query(query)


def parse_peer_receipts_audit_query(query: str) -> tuple[int, int] | None:
    """Validate the peer-receipts chain-audit query string.

    Shares the receipts route's contract exactly: both ``after`` and
    ``limit`` are required non-repeated ASCII decimal integers,
    ``after`` non-negative, ``limit`` between 1 and 100, and any unknown
    parameter is rejected. Kept as a named entry point for the receipts
    audit route; the bound on ``after`` against the peer's committed
    receipt count is checked by the store against the committed snapshot.
    """
    return parse_peer_pickup_query(query)


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _valid_percent_escapes(raw: str) -> bool:
    """Return True when every ``%`` in ``raw`` starts a two-hex-digit escape.

    Anything else — a trailing ``%``, a single hex digit, or a non-hex
    escape like ``%GG`` — is an illegal encoding rather than a literal
    character, so the caller rejects it instead of silently treating the
    percent sign as data.
    """
    index = 0
    while index < len(raw):
        if raw[index] == "%":
            if (
                index + 2 >= len(raw)
                or raw[index + 1] not in _HEX_DIGITS
                or raw[index + 2] not in _HEX_DIGITS
            ):
                return False
            index += 3
        else:
            index += 1
    return True


def parse_replication_status_query(query: str) -> str | None:
    """Validate the replication-status query string, returning the peer id.

    Requires exactly one parameter, ``peerId``, appearing exactly once
    with a non-empty percent-decoded value — the same percent-decoding
    and non-empty rules the replication routes apply to their ``peerId``
    path segment, here applied to the query value. A missing or repeated
    parameter, an unknown parameter, a blank name or value, and an
    illegal encoding — a malformed percent escape or an escape sequence
    that is not valid UTF-8 — return None.
    """
    if not _valid_percent_escapes(query):
        return None
    try:
        parsed = parse_qs(query, keep_blank_values=True, errors="strict")
    except UnicodeDecodeError:
        return None
    if any(len(values) != 1 for values in parsed.values()):
        return None
    if set(parsed) != {"peerId"}:
        return None
    peer_id = parsed["peerId"][0]
    if peer_id == "":
        return None
    return peer_id


def parse_scope_policy_audit_query(query: str) -> tuple[int, int] | None:
    """Validate the scope-policy change-audit query string.

    Both ``after`` and ``limit`` are required non-repeated ASCII decimal
    integers: ``after`` is a 0-based resume cursor into the reload
    history and ``limit`` must be between 1 and 100. A missing or
    repeated name, a blank, signed, whitespace-bearing, decimal-point,
    or non-ASCII-decimal value, and any unknown parameter are rejected
    exactly as on the other audited paging routes. The bound on ``after``
    against the committed event count is checked by the store against
    the committed snapshot (``after`` equal to the count is a valid
    empty page).
    """
    return parse_peer_pickup_query(query)


def parse_scope_policy_audit_verify_query(query: str) -> tuple[int, int] | None:
    """Validate the scope-policy audit-verification query string.

    Shares the change-audit route's contract exactly: both ``after`` and
    ``limit`` are required non-repeated ASCII decimal integers, ``after``
    non-negative (the number of successful events already skipped,
    starting at ``0``), ``limit`` between 1 and 100, and any unknown
    parameter is rejected. Kept as a named entry point for the verify
    route; the bound on ``after`` against the committed event count is
    checked by the store against the committed snapshot (``after`` equal
    to the count is a valid stable empty page).
    """
    return parse_peer_pickup_query(query)


CAUSAL_COMPARE_IDENTITY_PARAMS = (
    "leftReplicaId",
    "leftOperationId",
    "rightReplicaId",
    "rightOperationId",
)


def parse_causal_compare_query(
    query: str,
) -> tuple[str, str, str, str, int, int] | None:
    """Validate the causal-comparison query string.

    Requires exactly one occurrence of each identity parameter —
    ``leftReplicaId``, ``leftOperationId``, ``rightReplicaId``,
    ``rightOperationId`` — each with a non-empty percent-decoded value
    (percent decoding is applied by ``parse_qs``), plus the shared paging
    parameters ``after`` (default 0) and ``limit`` (default 100, 1-100)
    under the same rules as :func:`parse_paging_query`. Unknown parameters,
    repeated names (identities included), blank identity values, malformed
    or negative paging values, and out-of-range limits return None. The
    bound on ``after`` against the predecessor counts is checked by the
    store against the committed snapshot.
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if any(len(values) != 1 for values in parsed.values()):
        return None
    required = set(CAUSAL_COMPARE_IDENTITY_PARAMS)
    if not required <= set(parsed):
        return None
    if not set(parsed) <= required | {"after", "limit"}:
        return None
    identities = tuple(parsed[name][0] for name in CAUSAL_COMPARE_IDENTITY_PARAMS)
    if any(identity == "" for identity in identities):
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
    return (*identities, after, limit)


CAUSAL_DESCENDANTS_IDENTITY_PARAMS = ("replicaId", "operationId")


def parse_causal_descendants_query(query: str) -> tuple[str, str, int, int] | None:
    """Validate the causal-descendants query string.

    Requires exactly one occurrence of each identity parameter —
    ``replicaId`` and ``operationId`` — each with a non-empty
    percent-decoded value (percent decoding is applied by ``parse_qs``),
    plus the shared paging parameters ``after`` (default 0) and ``limit``
    (default 100, 1-100) under the same rules as
    :func:`parse_paging_query`. Unknown parameters, repeated names
    (identities included), blank identity values, malformed or negative
    paging values, and out-of-range limits return None. The bound on
    ``after`` against the descendant count is checked by the store against
    the committed snapshot.
    """
    parsed = parse_qs(query, keep_blank_values=True)
    if any(len(values) != 1 for values in parsed.values()):
        return None
    required = set(CAUSAL_DESCENDANTS_IDENTITY_PARAMS)
    if not required <= set(parsed):
        return None
    if not set(parsed) <= required | {"after", "limit"}:
        return None
    replica_id, operation_id = (parsed[name][0] for name in CAUSAL_DESCENDANTS_IDENTITY_PARAMS)
    if replica_id == "" or operation_id == "":
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
    return replica_id, operation_id, after, limit


def parse_causal_frontier_query(query: str) -> tuple[int, int] | None:
    """Validate the causal-frontier query string.

    Both ``after`` and ``limit`` are **required** (there are no defaults,
    unlike the single-operation causal chains): each must appear exactly
    once as a non-negative ASCII decimal integer, with ``limit`` between
    1 and 100. A missing or repeated name, an unknown parameter, a blank,
    signed, whitespace-bearing, decimal-point, or non-ASCII-decimal
    value, and an out-of-range ``limit`` return None. The bound on
    ``after`` against the frontier size is checked by the store against
    the committed snapshot (``after`` equal to the frontier size is a
    valid stable empty page).
    """
    return parse_peer_pickup_query(query)


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


def parse_empty_object_payload(raw: bytes | str | dict[str, Any]) -> None:
    """Validate a request body that must be exactly the empty object.

    The body must be a complete JSON document whose parsed value is an
    object with no fields: only ``{}`` (with optional JSON whitespace)
    passes. Malformed JSON, a non-object document, ``null``, and any
    object carrying a field — known or unknown — raise ValueError.
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
    if not isinstance(payload, dict) or payload:
        raise ValueError("body must be an empty JSON object")


def _validate_boundary_clock(clock: Any) -> dict[str, int]:
    """Validate a causal-at boundary clock and return a clean copy.

    Unlike an operation clock, a boundary clock may be empty (the causal
    origin). When non-empty, every component name must be a non-empty
    string and every value a non-boolean, non-negative JSON integer;
    booleans, floats (including ``1.0`` and ``-0.0``), and non-finite
    values are rejected. Raises ValueError on any violation.
    """
    if not isinstance(clock, dict):
        raise ValueError("clock must be a JSON object")
    for component, tick in clock.items():
        if not isinstance(component, str) or component == "":
            raise ValueError("clock components must be non-empty strings")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("clock values must be non-negative integers")
    return dict(clock)


def parse_causal_at_payload(raw: bytes | str | dict[str, Any]) -> dict[str, int]:
    """Parse and validate a causal-slice boundary body.

    The body must be a complete JSON object whose only key is ``clock``;
    unknown fields, a non-object document, or a duplicated key anywhere
    in the document raise ValueError. ``clock`` is a boundary vector
    clock (see :func:`_validate_boundary_clock`): it may be empty (the
    causal origin), and otherwise maps replica ids to non-boolean
    non-negative integers. Returns the normalized clock.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("body must be UTF-8 JSON") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        # Already-decoded mappings come from store-level callers, which
        # bypass JSON and therefore the duplicate-key hook.
        if not isinstance(raw, dict) or set(raw.keys()) != {"clock"}:
            raise ValueError("body must be an object with only clock")
        return _validate_boundary_clock(raw.get("clock"))

    def reject_duplicate_keys(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
        document: dict[Any, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("duplicate field in body")
            document[key] = value
        return document

    try:
        payload: Any = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("body must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload.keys()) != {"clock"}:
        raise ValueError("body must be an object with only clock")
    return _validate_boundary_clock(payload.get("clock"))


def parse_replication_compare_payload(
    raw: bytes | str | dict[str, Any],
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Parse and validate a cross-replica candidate-comparison body.

    The body must be a JSON object with exactly ``replicaId`` and
    ``snapshot``: the remote replica's identifier (a non-empty string) and
    its complete candidate snapshot — an object mapping each business key
    to a non-empty array of candidates. Each candidate must contain
    exactly ``value``, ``clock``, ``replicaId``, and ``operationId`` and
    satisfy the live write constraints: the value and both identity
    components are non-empty strings, and the clock follows
    :func:`_validate_clock` (a non-empty object whose component values are
    non-boolean, non-negative JSON integers — floats such as ``1.0`` and
    ``-0.0`` and non-finite values are rejected — and which contains the
    candidate's own replica id). A duplicated field anywhere in the
    document, an unknown field, an empty key or candidate array, or an
    operation identity ``(replicaId, operationId)`` appearing more than
    once across the whole snapshot raises ValueError. Returns the remote
    replica id and the normalized snapshot, ready for
    :func:`_verification_digest_input`. The snapshot is only validated —
    it is never imported into local state.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("body must be UTF-8 JSON") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        # Already-decoded mappings come from store-level callers, which
        # bypass JSON and therefore the duplicate-key hook.
        text = None
    if text is None:
        payload = raw
    else:
        def reject_duplicate_keys(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
            document: dict[Any, Any] = {}
            for key, value in pairs:
                if key in document:
                    raise ValueError("duplicate field in body")
                document[key] = value
            return document

        try:
            payload = json.loads(text, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ValueError("body must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload.keys()) != {"replicaId", "snapshot"}:
        raise ValueError("body must be an object with only replicaId and snapshot")
    replica_id = payload["replicaId"]
    if not isinstance(replica_id, str) or replica_id == "":
        raise ValueError("replicaId must be a non-empty string")
    snapshot_raw = payload["snapshot"]
    if not isinstance(snapshot_raw, dict):
        raise ValueError("snapshot must be an object mapping keys to candidate arrays")

    snapshot: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, str]] = set()
    for key, candidates_raw in snapshot_raw.items():
        if not isinstance(key, str) or key == "":
            raise ValueError("snapshot keys must be non-empty strings")
        if not isinstance(candidates_raw, list) or not candidates_raw:
            raise ValueError("each snapshot entry must be a non-empty candidate array")
        candidates: list[dict[str, Any]] = []
        for entry in candidates_raw:
            if not isinstance(entry, dict) or set(entry.keys()) != {
                "value",
                "clock",
                "replicaId",
                "operationId",
            }:
                raise ValueError(
                    "each candidate must have only value, clock, replicaId, operationId"
                )
            value = entry["value"]
            if not isinstance(value, str) or value == "":
                raise ValueError("value must be a non-empty string")
            candidate_replica = entry["replicaId"]
            if not isinstance(candidate_replica, str) or candidate_replica == "":
                raise ValueError("replicaId must be a non-empty string")
            operation_id = entry["operationId"]
            if not isinstance(operation_id, str) or operation_id == "":
                raise ValueError("operationId must be a non-empty string")
            clock = _validate_clock(entry["clock"], candidate_replica)
            identity = (candidate_replica, operation_id)
            if identity in identities:
                raise ValueError("duplicate candidate identity in snapshot")
            identities.add(identity)
            candidates.append(
                {
                    "value": value,
                    "clock": clock,
                    "replicaId": candidate_replica,
                    "operationId": operation_id,
                }
            )
        snapshot[key] = candidates
    return replica_id, snapshot


ACK_MAX_OPERATIONS = 100


def _parse_ack_operations(raw: Any) -> list[dict[str, str]]:
    """Validate the identity list of an acknowledge request.

    ``operations`` records, in order, the identities the peer consumed: a
    list of at most ``ACK_MAX_OPERATIONS`` entries, each an object with
    exactly ``replicaId`` and ``operationId`` (both non-empty strings) and
    no repeated identity. An empty list acknowledges the empty segment at
    the peer's checkpoint. Raises ValueError on any violation.
    """
    if not isinstance(raw, list) or len(raw) > ACK_MAX_OPERATIONS:
        raise ValueError("operations must be a list of at most 100 identities")
    operations: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for entry in raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {"replicaId", "operationId"}:
            raise ValueError("each operation must have only replicaId and operationId")
        replica_id = entry["replicaId"]
        operation_id = entry["operationId"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replicaId must be a non-empty string")
        if not isinstance(operation_id, str) or operation_id == "":
            raise ValueError("operationId must be a non-empty string")
        identity = (replica_id, operation_id)
        if identity in identities:
            raise ValueError(f"duplicate identity {identity!r}")
        identities.add(identity)
        operations.append({"replicaId": replica_id, "operationId": operation_id})
    return operations


def parse_acknowledge_payload(raw: bytes | str | dict[str, Any]) -> tuple[str, int, list[dict[str, str]]]:
    """Parse and validate a consumption-acknowledgement body.

    The body must be a JSON object with exactly ``ackId`` (a non-empty
    string), ``cursor`` (a non-boolean non-negative integer), and
    ``operations`` (the ordered identities the peer consumed, see
    :func:`_parse_ack_operations`). Returns the normalized
    ``(ack_id, cursor, operations)`` triple. The segment check against the
    peer's checkpoint and the accepted log is left to the store, because it
    is only meaningful against a committed snapshot. Raises ValueError on
    any violation.
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
        "ackId",
        "cursor",
        "operations",
    }:
        raise ValueError("body must be an object with only ackId, cursor, operations")
    ack_id = payload["ackId"]
    if not isinstance(ack_id, str) or ack_id == "":
        raise ValueError("ackId must be a non-empty string")
    cursor = payload["cursor"]
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError("cursor must be a non-negative integer")
    operations = _parse_ack_operations(payload["operations"])
    return ack_id, cursor, operations


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


def _emit_compact_json(value: Any, *, sort_keys: bool) -> bytes:
    """Serialize a payload to compact UTF-8 JSON with strict escaping.

    Arrays are emitted in order and no insignificant whitespace appears
    anywhere. Strings escape only the quote, the backslash, and control
    characters (U+0000-U+001F, always as ``\\u00XX`` with lowercase hex) —
    every other code point is written literally. Integers are emitted as
    plain JSON integers (booleans as ``true``/``false``); no float,
    negative zero, or non-finite value can appear because the store only
    ever holds validated integers. Object keys are sorted lexicographically
    (Unicode code point order) when ``sort_keys`` is true, and kept in
    their insertion order otherwise.
    """

    def emit(item: Any, parts: list[str]) -> None:
        if isinstance(item, dict):
            parts.append("{")
            names = sorted(item) if sort_keys else list(item)
            for index, name in enumerate(names):
                if index:
                    parts.append(",")
                parts.append(_escape_digest_string(name))
                parts.append(":")
                emit(item[name], parts)
            parts.append("}")
        elif isinstance(item, (list, tuple)):
            parts.append("[")
            for index, element in enumerate(item):
                if index:
                    parts.append(",")
                emit(element, parts)
            parts.append("]")
        elif isinstance(item, str):
            parts.append(_escape_digest_string(item))
        elif item is True:
            parts.append("true")
        elif item is False:
            parts.append("false")
        elif item is None:
            parts.append("null")
        elif isinstance(item, int):
            parts.append(str(item))
        else:  # pragma: no cover - payloads never carry other types
            raise TypeError(f"cannot serialize {type(item)!r} canonically")

    parts: list[str] = []
    emit(value, parts)
    return "".join(parts).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize a response payload to canonical compact UTF-8 JSON.

    Objects are emitted with their keys sorted lexicographically (Unicode
    code point order); otherwise identical to :func:`_emit_compact_json`.
    """
    return _emit_compact_json(value, sort_keys=True)


def _ordered_json_bytes(value: Any) -> bytes:
    """Serialize a response payload to compact UTF-8 JSON in field order.

    Same encoding rules as :func:`_canonical_json_bytes` — no insignificant
    whitespace, strings escaping only the quote, the backslash, and control
    characters, and every number a plain JSON integer — but object fields
    keep their insertion order instead of being sorted, for endpoints whose
    contract fixes the field order of the response.
    """
    return _emit_compact_json(value, sort_keys=False)


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


def _audit_record_bytes(replica_id: str, operation: dict[str, Any]) -> bytes:
    """Serialize one accepted-operation record to its canonical bytes.

    The record has the fixed shape
    ``{"replicaId":R,"operation":{"operationId":I,"key":K,"value":V,"clock":C}}``
    and the clock's component names are sorted lexicographically (Unicode
    code point order). No whitespace is emitted anywhere, and strings are
    escaped exactly as in :func:`_escape_digest_string` — only the quote,
    the backslash, and U+0000-U+001F control characters. This is the single
    record encoding shared by the per-key audit digest (one array element)
    and the global audit chain (one link's record bytes).
    """
    clock = ",".join(
        f"{_escape_digest_string(name)}:{tick}"
        for name, tick in sorted(operation["clock"].items())
    )
    parts: list[str] = ['{"replicaId":']
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
    return "".join(parts).encode("utf-8")


def _key_audit_digest_input(records: list[tuple[str, dict[str, Any]]]) -> bytes:
    """Serialize one key's accepted-operation stream to the digest input.

    The result is a compact UTF-8 JSON array with one entry per accepted
    operation for the key, in global commit order (the caller filters the
    shared accepted-operation log, which already preserves that order).
    Each entry is serialized by :func:`_audit_record_bytes`. An empty
    stream serializes to ``[]``.
    """
    parts: list[str] = ["["]
    for index, (replica_id, operation) in enumerate(records):
        if index:
            parts.append(",")
        parts.append(_audit_record_bytes(replica_id, operation).decode("utf-8"))
    parts.append("]")
    return "".join(parts).encode("utf-8")


# The chain link preceding the first accepted operation, and the head of an
# empty chain: 64 lowercase "0" characters.
_AUDIT_CHAIN_GENESIS = "0" * 64


def _audit_chain_link(
    previous_digest: str, sequence: int, replica_id: str, operation: dict[str, Any]
) -> str:
    """Compute one link of the global audit chain.

    The hash input is the concatenation of the previous link's digest and
    the link's decimal sequence number (both ASCII-encoded) with the single
    record's canonical bytes from :func:`_audit_record_bytes`; the result is
    the 64-character lowercase hexadecimal SHA-256 of exactly those bytes.
    """
    hash_input = (
        previous_digest.encode("ascii")
        + str(sequence).encode("ascii")
        + _audit_record_bytes(replica_id, operation)
    )
    return hashlib.sha256(hash_input).hexdigest()


def _audit_log_verification_locked(
    accepted: list[tuple[str, dict[str, Any]]],
    entries: list[dict[str, Any]],
    expected_head: str,
    expected_count: int,
) -> dict[str, Any]:
    """Independently verify the whole global audit chain in one pass.

    ``accepted`` is the complete shared accepted-operation log in global
    commit order (the raw source of truth) and ``entries`` is the full
    materialized chain of claimed links — one per log position, each
    ``{"sequence", "prevDigest", "digest"}`` (exactly what the chain query
    computes for the whole log). The scan never pages: it independently
    re-walks the raw log, recomputing every link with
    :func:`_audit_chain_link`, and validates each claimed link against that
    recomputation:

    - **sequence continuity** — claimed sequences must form the continuous
      range ``1..N`` with no missing, duplicate, or out-of-range claim;
    - **predecessor closure** (``prevDigest``) — the first link closes
      against the 64-zero genesis and every later link against the previous
      link's recomputed digest;
    - **digest recomputation** — each claimed ``digest`` must equal the
      independently recomputed value;
    - **chain-tail agreement** — the recomputed last-link digest (the
      ``head``, 64 zeros for an empty log) must equal the external
      ``head``, and the full length ``N`` must equal the external
      ``count``.

    It returns the conclusion, always over the complete log::

        {"status": "ok" | "broken",
         "missingSequences": [...], "duplicateSequences": [...],
         "outOfRangeSequences": [...], "brokenLinks": [...],
         "digestMismatches": [...],
         "headMismatches": [...], "countMismatches": [...]}

    Every marker keeps the 0-based chain-link position (``linkIndex``),
    the link's 1-based ``sequence``, and the observed value:

    - ``missingSequences``: a position ``S`` in ``1..N`` no link claims —
      ``{"linkIndex": S - 1, "sequence": S}``.
    - ``duplicateSequences``: a link whose claimed sequence an earlier link
      already claimed (the later occurrence only) —
      ``{"linkIndex": I, "sequence": S}``.
    - ``outOfRangeSequences``: a link whose ``sequence`` is not an integer
      in ``1..N`` — ``{"linkIndex": I, "sequence": S}``.
    - ``brokenLinks``: the predecessor-closure failure (断链) — a link
      whose claimed ``prevDigest`` is not the predecessor link's
      recomputed digest (the genesis for the first link) —
      ``{"linkIndex": I, "sequence": S, "expected": D, "observed": D}``
      with the recomputed predecessor first and the claimed one second.
    - ``digestMismatches``: the digest-recomputation failure — a link whose
      claimed ``digest`` is not the value independently recomputed from the
      record's canonical bytes and the running predecessor —
      ``{"linkIndex": I, "sequence": S, "expected": D, "observed": D}``
      with the recomputed digest first and the claimed one second.
    - ``headMismatches``: at most one marker
      ``{"expected": H, "observed": H}`` (external first, recomputed tail
      second); empty on agreement.
    - ``countMismatches``: at most one marker
      ``{"expected": C, "observed": N}`` (external first, actual length
      second); empty on agreement.

    ``status`` is ``"ok"`` exactly when the internal chain is intact (the
    first five lists empty) **and** both external expectations match;
    otherwise ``"broken"``. The live store materializes ``entries`` from
    ``accepted`` in the same committed snapshot, so a healthy committed
    history is internally intact by construction and verifies ``"ok"``
    exactly when ``head`` and ``count`` match; the scan exists to expose a
    damaged chain. An empty log is intact with a 64-zero tail and verifies
    ``"ok"`` for the genesis head and count ``0``.
    """
    missing: list[dict[str, Any]] = []
    duplicate: list[dict[str, Any]] = []
    out_of_range: list[dict[str, Any]] = []
    broken_links: list[dict[str, Any]] = []
    digest_mismatches: list[dict[str, Any]] = []
    total = len(accepted)
    seen: set[int] = set()
    recomputed_previous = _AUDIT_CHAIN_GENESIS
    for index in range(max(total, len(entries))):
        claimed = entries[index] if index < len(entries) else None
        raw_sequence = claimed.get("sequence") if isinstance(claimed, dict) else None
        valid_sequence = isinstance(raw_sequence, int) and not isinstance(
            raw_sequence, bool
        )
        if not valid_sequence or raw_sequence < 1 or raw_sequence > total:
            out_of_range.append({"linkIndex": index, "sequence": raw_sequence})
        elif raw_sequence in seen:
            # Only the later occurrence is a duplicate; the first keeps its
            # claim on the sequence position.
            duplicate.append({"linkIndex": index, "sequence": raw_sequence})
        else:
            seen.add(raw_sequence)
        if index < total:
            replica_id, operation = accepted[index]
            recomputed = _audit_chain_link(
                recomputed_previous, index + 1, replica_id, operation
            )
            claimed_previous = (
                claimed.get("prevDigest") if isinstance(claimed, dict) else None
            )
            claimed_digest = (
                claimed.get("digest") if isinstance(claimed, dict) else None
            )
            if claimed_previous != recomputed_previous:
                # Broken predecessor closure: the link does not close off
                # the predecessor's recomputed digest. The marker carries
                # the recomputed predecessor first and the claimed one
                # second.
                broken_links.append(
                    {
                        "linkIndex": index,
                        "sequence": raw_sequence,
                        "expected": recomputed_previous,
                        "observed": claimed_previous,
                    }
                )
            if claimed_digest != recomputed:
                # Digest recomputation mismatch: the claimed digest is not
                # the SHA-256 recomputed from the record and the running
                # predecessor. The marker carries the recomputed digest
                # first and the claimed one second.
                digest_mismatches.append(
                    {
                        "linkIndex": index,
                        "sequence": raw_sequence,
                        "expected": recomputed,
                        "observed": claimed_digest,
                    }
                )
            recomputed_previous = recomputed
    for expected_sequence in range(1, total + 1):
        if expected_sequence not in seen:
            missing.append(
                {"linkIndex": expected_sequence - 1, "sequence": expected_sequence}
            )
    head = recomputed_previous
    head_mismatches: list[dict[str, Any]] = []
    if head != expected_head:
        head_mismatches.append({"expected": expected_head, "observed": head})
    count_mismatches: list[dict[str, Any]] = []
    if total != expected_count:
        count_mismatches.append({"expected": expected_count, "observed": total})
    broken = bool(
        missing
        or duplicate
        or out_of_range
        or broken_links
        or digest_mismatches
        or head_mismatches
        or count_mismatches
    )
    return {
        "status": "broken" if broken else "ok",
        "missingSequences": missing,
        "duplicateSequences": duplicate,
        "outOfRangeSequences": out_of_range,
        "brokenLinks": broken_links,
        "digestMismatches": digest_mismatches,
        "headMismatches": head_mismatches,
        "countMismatches": count_mismatches,
    }


def _replication_snapshot_input(
    candidate_digest: str, log_cursor: int, checkpoints: dict[str, int]
) -> bytes:
    """Serialize the replication-snapshot summary to the canonical digest input.

    The result is a compact UTF-8 JSON array of exactly three elements, in
    this fixed order: the candidate digest (a 64-character lowercase
    hexadecimal string, computed exactly as for the verification digest),
    the accepted-log cursor (a JSON integer — the number of first-accepted
    operations, i.e. the sync-export resume cursor at the tail of the log),
    and the checkpoint mapping of sender-side progress with peer ids sorted
    lexicographically (Unicode code point order); an empty mapping is kept
    as ``{}``. No whitespace is emitted anywhere, and strings are escaped
    exactly as in :func:`_escape_digest_string` — only the quote, the
    backslash, and U+0000-U+001F control characters.
    """
    parts: list[str] = ["["]
    parts.append(_escape_digest_string(candidate_digest))
    parts.append(",")
    parts.append(str(log_cursor))
    parts.append(",{")
    for index, peer_id in enumerate(sorted(checkpoints)):
        if index:
            parts.append(",")
        parts.append(_escape_digest_string(peer_id))
        parts.append(":")
        parts.append(str(checkpoints[peer_id]))
    parts.append("}]")
    return "".join(parts).encode("utf-8")


def _receipts_digest_input(
    peer_id: str, receipts: list[tuple[str, dict[str, Any]]]
) -> bytes:
    """Serialize one peer's committed receipts to the canonical digest input.

    The result is a compact UTF-8 JSON array with one entry per committed
    consumption receipt of the peer, in commit (creation) order. Each entry
    carries its fields in the fixed order ``peerId``, ``ackId``, ``cursor``,
    ``operations``; the operations keep their confirmation order and each
    identity carries its fields in the fixed order ``replicaId``,
    ``operationId``. No whitespace is emitted anywhere, numbers are plain
    JSON integers, and strings are escaped exactly as in
    :func:`_escape_digest_string` — only the quote, the backslash, and
    U+0000-U+001F control characters. An empty receipt set serializes to
    ``[]``.
    """
    parts: list[str] = ["["]
    for receipt_index, (ack_id, receipt) in enumerate(receipts):
        if receipt_index:
            parts.append(",")
        parts.append('{"peerId":')
        parts.append(_escape_digest_string(peer_id))
        parts.append(',"ackId":')
        parts.append(_escape_digest_string(ack_id))
        parts.append(',"cursor":')
        parts.append(str(receipt["cursor"]))
        parts.append(',"operations":[')
        for index, identity in enumerate(receipt["operations"]):
            if index:
                parts.append(",")
            parts.append('{"replicaId":')
            parts.append(_escape_digest_string(identity["replicaId"]))
            parts.append(',"operationId":')
            parts.append(_escape_digest_string(identity["operationId"]))
            parts.append("}")
        parts.append("]}")
    parts.append("]")
    return "".join(parts).encode("utf-8")


def _policy_events_digest_input(events: list[dict[str, Any]]) -> bytes:
    """Serialize the policy-change event history to the canonical digest input.

    The result is a compact UTF-8 JSON array with one entry per successful
    scope-policy hot reload, in the order the reloads committed. Each entry
    carries its fields in the fixed order ``sequence``, ``digest``,
    ``tokens``: the event's 1-based history position, the 64-character
    lowercase hexadecimal SHA-256 of the reloaded policy file's raw UTF-8
    bytes, and the new policy's entry count. No whitespace is emitted
    anywhere, numbers are plain JSON integers, and strings are escaped
    exactly as in :func:`_escape_digest_string`. An empty history
    serializes to ``[]``.
    """
    parts: list[str] = ["["]
    for index, event in enumerate(events):
        if index:
            parts.append(",")
        parts.append('{"sequence":')
        parts.append(str(event["sequence"]))
        parts.append(',"digest":')
        parts.append(_escape_digest_string(event["digest"]))
        parts.append(',"tokens":')
        parts.append(str(event["tokens"]))
        parts.append("}")
    parts.append("]")
    return "".join(parts).encode("utf-8")


def _policy_events_verification_locked(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the complete scope-policy change history in one pass.

    ``events`` is the complete history in commit order (a snapshot taken
    under the commit lock). The conclusion is always computed over the
    complete history, never the current page::

        {"status": "ok" | "broken",
         "missingSequences": [...], "duplicateSequences": [...],
         "outOfRangeSequences": [...], "digestMismatches": [...]}

    Every anomaly is marked with the event's 0-based ``eventsIndex`` in
    the complete history and its claimed 1-based ``sequence``:

    - ``missingSequences``: a position ``S`` in ``1..len(events)`` that no
      event claims. The marker points at the index where that sequence is
      missing: ``{"eventsIndex": S - 1, "sequence": S}``.
    - ``duplicateSequences``: an event whose claimed sequence was already
      claimed by an earlier event — ``{"eventsIndex": I, "sequence": S}``
      for the repeated occurrence only.
    - ``outOfRangeSequences``: an event whose ``sequence`` is not an
      integer in ``1..len(events)`` (a non-integer, a boolean, zero, a
      negative value, or a value past the event count) —
      ``{"eventsIndex": I, "sequence": S}``.
    - ``digestMismatches``: an event whose recorded ``digest`` is not
      exactly 64 lowercase hexadecimal characters, i.e. it does not match
      the SHA-256 digest shape recorded by every successful reload —
      ``{"eventsIndex": I, "sequence": S, "expected": null,
      "observed": D}``; ``expected`` is null because the true digest is a
      hash of the policy file's raw bytes, which the history never
      retains, so only the recorded digest's shape can be checked.

    ``status`` is ``"ok"`` exactly when all four lists are empty: the
    sequences form the continuous range ``1..N`` and every recorded
    digest has the SHA-256 shape. An empty history is complete and
    intact — ``"ok"`` with four empty lists. Live history is appended one
    verified event at a time, so the conclusion is ``"ok"`` by
    construction; the scan exists to detect a damaged history (and
    paging, the digest, and ``eventsCount`` never influence it — they are
    all derived from the same complete snapshot).
    """
    missing: list[dict[str, Any]] = []
    duplicate: list[dict[str, Any]] = []
    out_of_range: list[dict[str, Any]] = []
    digest_mismatches: list[dict[str, Any]] = []
    count = len(events)
    # First index at which each in-range claimed sequence was seen.
    seen: set[int] = set()
    for events_index, event in enumerate(events):
        raw_sequence = event.get("sequence") if isinstance(event, dict) else None
        valid_sequence = (
            isinstance(raw_sequence, int)
            and not isinstance(raw_sequence, bool)
        )
        if not valid_sequence or raw_sequence < 1 or raw_sequence > count:
            out_of_range.append(
                {"eventsIndex": events_index, "sequence": raw_sequence}
            )
        elif raw_sequence in seen:
            # Only the later occurrence is a duplicate; the first keeps
            # its claim on the sequence position.
            duplicate.append(
                {"eventsIndex": events_index, "sequence": raw_sequence}
            )
        else:
            seen.add(raw_sequence)
        recorded_digest = event.get("digest") if isinstance(event, dict) else None
        if not _is_sha256_hex64(recorded_digest):
            digest_mismatches.append(
                {
                    "eventsIndex": events_index,
                    "sequence": raw_sequence,
                    "expected": None,
                    "observed": recorded_digest,
                }
            )
    for expected_sequence in range(1, count + 1):
        if expected_sequence not in seen:
            missing.append(
                {
                    "eventsIndex": expected_sequence - 1,
                    "sequence": expected_sequence,
                }
            )
    broken = bool(missing or duplicate or out_of_range or digest_mismatches)
    return {
        "status": "broken" if broken else "ok",
        "missingSequences": missing,
        "duplicateSequences": duplicate,
        "outOfRangeSequences": out_of_range,
        "digestMismatches": digest_mismatches,
    }


def _receipt_chain_audit_locked(
    committed: list[tuple[str, dict[str, Any]]],
    accepted: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Audit one peer's committed receipts as one confirmation chain.

    ``committed`` is the peer's receipts in creation order —
    ``[(ackId, {"cursor", "operations"})]`` — and ``accepted`` is the
    shared accepted-operation log in global commit order
    (``[(replicaId, operation)]``). Returns the chain-integrity
    conclusion, always computed over the peer's whole receipt history,
    never the current page::

        {"status": "ok" | "broken",
         "coverage": {"start": N, "end": N},
         "gaps": [...], "overlaps": [...],
         "identityMismatches": [...], "cursorRegressions": [...]}

    Each receipt ``i`` confirms the half-open log segment
    ``[start_i, cursor_i)`` where ``start_i`` is derived from the receipt's
    confirmation cursor and its operation count
    (``cursor_i - len(operations_i)``) — the first receipt's start comes
    from its own count and cursor, and every later segment must begin
    exactly where the previous one ended. The four anomaly lists are
    independent and each entry retains where it occurred:

    - ``gaps``: a receipt starts past the previous receipt's end, so the
      accepted records between ``from`` and ``to`` are confirmed by no
      receipt.
    - ``overlaps``: a receipt starts before the previous receipt's end,
      so the segment between ``from`` and ``to`` is confirmed twice.
    - ``cursorRegressions``: a receipt's confirmation cursor does not
      advance past the previous receipt's cursor (``from``/``to`` are the
      previous and the regressed cursors).
    - ``identityMismatches``: one confirmed position names a different
      ``(replicaId, operationId)`` than the accepted record at that log
      position (or names a position outside the current log, in which
      case ``expected`` is null); the entry carries the absolute log
      ``position`` and both ``expected`` and ``observed`` identities.

    An empty confirmation segment is a legal receipt on its own and
    produces no anomaly. An empty receipt set is a complete,
    anomaly-free empty coverage — ``{"start": 0, "end": 0}`` with
    ``status`` ``"ok"``; otherwise the coverage runs from the first
    receipt's derived start to the last receipt's confirmation cursor.
    ``status`` is ``"ok"`` exactly when all four anomaly lists are empty,
    i.e. the receipts form one seamless, non-overlapping confirmation
    chain whose identities match the accepted log position by position.
    """
    gaps: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    cursor_regressions: list[dict[str, Any]] = []
    if not committed:
        return {
            "status": "ok",
            "coverage": {"start": 0, "end": 0},
            "gaps": gaps,
            "overlaps": overlaps,
            "identityMismatches": identity_mismatches,
            "cursorRegressions": cursor_regressions,
        }
    log_length = len(accepted)
    coverage_start = committed[0][1]["cursor"] - len(committed[0][1]["operations"])
    previous_cursor: int | None = None
    for receipt_index, (ack_id, receipt) in enumerate(committed):
        cursor = receipt["cursor"]
        operations = receipt["operations"]
        start = cursor - len(operations)
        if previous_cursor is not None:
            if start > previous_cursor:
                gaps.append(
                    {
                        "receiptIndex": receipt_index,
                        "ackId": ack_id,
                        "from": previous_cursor,
                        "to": start,
                    }
                )
            elif start < previous_cursor:
                overlaps.append(
                    {
                        "receiptIndex": receipt_index,
                        "ackId": ack_id,
                        "from": previous_cursor,
                        "to": start,
                    }
                )
            if cursor < previous_cursor:
                cursor_regressions.append(
                    {
                        "receiptIndex": receipt_index,
                        "ackId": ack_id,
                        "from": previous_cursor,
                        "to": cursor,
                    }
                )
        for offset, observed in enumerate(operations):
            position = start + offset
            if position < 0 or position >= log_length:
                identity_mismatches.append(
                    {
                        "receiptIndex": receipt_index,
                        "ackId": ack_id,
                        "position": position,
                        "expected": None,
                        "observed": dict(observed),
                    }
                )
                continue
            replica_id, accepted_operation = accepted[position]
            if (
                observed["replicaId"] != replica_id
                or observed["operationId"] != accepted_operation["operationId"]
            ):
                identity_mismatches.append(
                    {
                        "receiptIndex": receipt_index,
                        "ackId": ack_id,
                        "position": position,
                        "expected": {
                            "replicaId": replica_id,
                            "operationId": accepted_operation["operationId"],
                        },
                        "observed": dict(observed),
                    }
                )
        previous_cursor = cursor
    coverage_end = committed[-1][1]["cursor"]
    broken = bool(gaps or overlaps or identity_mismatches or cursor_regressions)
    return {
        "status": "broken" if broken else "ok",
        "coverage": {"start": coverage_start, "end": coverage_end},
        "gaps": gaps,
        "overlaps": overlaps,
        "identityMismatches": identity_mismatches,
        "cursorRegressions": cursor_regressions,
    }


class PersistenceError(Exception):
    """Raised when the data file cannot be opened, parsed, or written.

    At startup this means the service must refuse to start; while serving
    it means the current write could not be committed durably.
    """


class AuthTokenError(Exception):
    """Raised when the auth token file cannot be used.

    Always a startup failure: the service refuses to start before it begins
    listening, exactly like a rejected data file. The token itself is never
    included in the error message.
    """


def load_auth_token(path: str) -> str:
    """Read and strictly validate the bearer token file.

    The target must be a readable regular file whose entire content is one
    non-empty ASCII printable token (bytes 0x21-0x7E): no whitespace, no
    newlines, nothing before or after the token. Raises AuthTokenError on a
    missing, unreadable, or non-regular target and on any format violation;
    the error never echoes the file's content.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise AuthTokenError(f"cannot access auth token file {path!r}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise AuthTokenError(f"auth token path is not a regular file: {path!r}")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise AuthTokenError(f"cannot read auth token file {path!r}: {exc}") from exc
    if not raw or any(byte < 0x21 or byte > 0x7E for byte in raw):
        raise AuthTokenError(
            "auth token file must contain exactly one non-empty ASCII "
            "printable token without whitespace or newlines"
        )
    return raw.decode("ascii")


SCOPE_READ = "read"
SCOPE_WRITE = "write"
SCOPE_ADMIN = "admin"
ALLOWED_SCOPES = (SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN)


class ScopePolicyError(Exception):
    """Raised when the scope policy file cannot be used.

    Always a startup failure: the service refuses to start before it begins
    listening, exactly like a rejected token or data file. Neither the
    configured tokens nor their scopes are ever included in the error
    message.
    """


class ScopePolicyReloadError(Exception):
    """Raised when a runtime scope-policy reload cannot be completed.

    ``kind`` classifies the failure for the HTTP boundary: ``"unavailable"``
    means the configured file is missing, unreadable, not a regular file, or
    could not be read (HTTP 503 ``policy_unavailable``); ``"conflict"``
    means the file was readable but its content failed the same UTF-8 JSON
    and token/scope validation as startup (HTTP 409 ``policy_conflict``).
    Either way the live policy stays in force and the error never echoes the
    file's tokens or contents.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def read_scope_policy_bytes(path: str) -> bytes:
    """Read the raw bytes of the scope policy file at ``path``.

    The target must be a readable regular file. A missing, unreadable, or
    non-regular target (for example a directory) and any read failure raise
    ScopePolicyError; the error never echoes the file's content. The bytes
    themselves are neither decoded nor validated here, so a reload can tell
    an unreadable file (503) apart from an invalid document (409).
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise ScopePolicyError(f"cannot access scope policy file {path!r}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ScopePolicyError(f"scope policy path is not a regular file: {path!r}")
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ScopePolicyError(f"cannot read scope policy file {path!r}: {exc}") from exc


def parse_scope_policy(raw: bytes) -> dict[str, frozenset[str]]:
    """Validate raw scope-policy bytes and return the token-to-scopes mapping.

    The bytes must encode one UTF-8 JSON object: every key is a non-empty
    ASCII printable token (bytes 0x21-0x7E, no whitespace) and every value
    is a non-empty array containing only the scopes ``"read"``,
    ``"write"``, and ``"admin"`` without repetition. Malformed or
    incomplete JSON, a non-object document, a duplicate token key, an
    illegal token, or an unknown/empty/duplicated scope value raises
    ScopePolicyError; the error never echoes the file's tokens or contents.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScopePolicyError("scope policy file must be UTF-8 JSON") from exc

    def reject_duplicate_keys(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
        document: dict[Any, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("duplicate token key in scope policy")
            document[key] = value
        return document

    try:
        document = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ScopePolicyError("scope policy file must be one complete, valid JSON document") from exc
    except ValueError as exc:
        raise ScopePolicyError("scope policy must not repeat a token key") from exc
    if not isinstance(document, dict):
        raise ScopePolicyError("scope policy must be a JSON object mapping tokens to scope arrays")

    policy: dict[str, frozenset[str]] = {}
    for token, scope_list in document.items():
        if not isinstance(token, str) or not token or any(
            ord(char) < 0x21 or ord(char) > 0x7E for char in token
        ):
            raise ScopePolicyError(
                "scope policy tokens must be non-empty ASCII printable strings without whitespace"
            )
        if not isinstance(scope_list, list) or not scope_list:
            raise ScopePolicyError("each scope policy value must be a non-empty array")
        scopes: set[str] = set()
        for scope in scope_list:
            if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
                raise ScopePolicyError("each scope must be one of read, write, admin")
            if scope in scopes:
                raise ScopePolicyError("scope arrays must not repeat a scope")
            scopes.add(scope)
        policy[token] = frozenset(scopes)
    return policy


def load_scope_policy(path: str) -> dict[str, frozenset[str]]:
    """Read and strictly validate the token-to-scopes policy file.

    The target must be a readable regular UTF-8 JSON file whose whole
    content is one JSON object: every key is a non-empty ASCII printable
    token (bytes 0x21-0x7E, no whitespace) and every value is a non-empty
    array containing only the scopes ``"read"``, ``"write"``, and
    ``"admin"`` without repetition. A missing, unreadable, or non-regular
    target, malformed or incomplete JSON, a non-object document, a
    duplicate token key, an illegal token, or an unknown/empty/duplicated
    scope value raises ScopePolicyError; the error never echoes the file's
    tokens or contents.
    """
    return parse_scope_policy(read_scope_policy_bytes(path))


class ScopePolicyManager:
    """Thread-safe holder of the live token-to-scopes policy.

    The manager owns the immutable policy mapping and the path of the file
    given at startup. Authentication takes a snapshot of the mapping under
    the manager's lock; a reload re-reads and validates that same configured
    file and only then swaps the mapping, so the replacement is one atomic
    commit: concurrent reloads run strictly one after another as complete
    read-validate-swap units, and every request authenticates against
    either the whole old policy or the whole new one, never a partial
    state. A failed reload leaves the live mapping untouched.
    """

    def __init__(
        self, path: str | None, policy: dict[str, frozenset[str]]
    ) -> None:
        # The path comes from the startup configuration; reloads may only
        # ever re-read this exact file, never a request-supplied path. It is
        # None only for servers assembled directly without a policy file, in
        # which case a reload reports the policy as unavailable.
        self._path = os.path.abspath(path) if path is not None else None
        self._lock = threading.Lock()
        self._policy = dict(policy)

    @property
    def path(self) -> str | None:
        return self._path

    def snapshot(self) -> dict[str, frozenset[str]]:
        """Return a point-in-time copy of the live policy mapping."""
        with self._lock:
            return dict(self._policy)

    def reload(
        self,
        recorder: "Callable[[str, int], Any] | None" = None,
    ) -> tuple[str, int]:
        """Atomically reload the policy from the startup-configured file.

        Re-reads the same file the service started with (the request may
        not name another path), validates it under exactly the startup
        constraints, and swaps the live mapping in one commit serialized
        against concurrent reloads and authentication snapshots. On
        success returns ``(policy_digest, tokens)`` where
        ``policy_digest`` is the 64-character lowercase hexadecimal
        SHA-256 of the file's raw UTF-8 bytes and ``tokens`` is the
        non-negative number of token entries. A missing, unreadable,
        non-regular, or otherwise unreadable file raises
        ScopePolicyReloadError(kind="unavailable"); readable-but-invalid
        content raises ScopePolicyReloadError(kind="conflict") and leaves
        the live policy complete and in force. No temporary file is
        created.

        When ``recorder`` is given it is called with
        ``(policy_digest, tokens)`` *before* the live mapping is swapped,
        still under this manager's lock, so the durable audit commit and
        the policy swap are one commit serialized against other reloads.
        Any exception the recorder raises (for example a failed durable
        write) propagates unchanged and prevents the swap: the old policy
        stays fully in force and the failed reload records no event.
        """
        path = self._path
        if path is None:
            raise ScopePolicyReloadError(
                "unavailable", "no scope policy file was configured at startup"
            )
        # Hold the lock across read, validation, the durable audit record,
        # and the swap, so each reload is one complete commit serialized
        # against other reloads and against authentication snapshots:
        # while a reload is reading, validating, or recording, every
        # request still sees the old policy, and once it releases every
        # request sees the new one.
        with self._lock:
            try:
                raw = read_scope_policy_bytes(path)
            except ScopePolicyError as exc:
                raise ScopePolicyReloadError("unavailable", str(exc)) from exc
            try:
                policy = parse_scope_policy(raw)
            except ScopePolicyError as exc:
                raise ScopePolicyReloadError("conflict", str(exc)) from exc
            digest = hashlib.sha256(raw).hexdigest()
            entries = len(policy)
            if recorder is not None:
                # Durably commit the change event before swapping the live
                # boundary: if this raises, the line below never runs, so
                # the old policy and the old history both survive.
                recorder(digest, entries)
            self._policy = policy
        return digest, entries


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


def _validate_stored_policies(
    document: Any, identities: set[tuple[str, str]]
) -> dict[tuple[str, str], str]:
    """Validate the optional ``policies`` section of a data file.

    Returns a clean ``{(replicaId, operationId): policy}`` mapping. The
    section is optional (a version:1 file written before automatic
    resolutions recorded their policy simply has none); when present it must
    be a list of ``{"replicaId", "operationId", "policy"}`` records with a
    known policy string, one per identity, and every identity must name an
    accepted operation, since a policy binding is committed atomically
    together with its operation.
    """
    if "policies" not in document:
        return {}
    raw = document["policies"]
    if not isinstance(raw, list):
        raise PersistenceError("data file policies must be a list")
    policies: dict[tuple[str, str], str] = {}
    for entry in raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "replicaId",
            "operationId",
            "policy",
        }:
            raise PersistenceError(
                "each policy record must be an object with replicaId, operationId, policy"
            )
        replica_id = entry["replicaId"]
        operation_id = entry["operationId"]
        policy = entry["policy"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise PersistenceError("policy replicaId must be a non-empty string")
        if not isinstance(operation_id, str) or operation_id == "":
            raise PersistenceError("policy operationId must be a non-empty string")
        if policy not in AUTO_RESOLVE_POLICIES:
            raise PersistenceError(f"unknown stored policy: {policy!r}")
        identity = (replica_id, operation_id)
        if identity in policies:
            raise PersistenceError(f"duplicate policy binding {identity!r} in data file")
        if identity not in identities:
            raise PersistenceError(
                f"policy binding {identity!r} names no accepted operation"
            )
        policies[identity] = policy
    return policies


def _validate_stored_transactions(
    document: Any, identities: set[tuple[str, str]]
) -> dict[str, list[dict[str, Any]]]:
    """Validate the optional ``transactions`` section of a data file.

    Returns a clean ``{transactionId: entries}`` mapping. The section is
    optional (a version:1 file written before transactions existed simply
    has none); when present it must be a list of
    ``{"transactionId", "operations"}`` records with a non-empty distinct
    transaction id each, the entries must satisfy the live transaction
    constraints, and every entry identity must name an accepted operation,
    since a transaction binding is committed atomically together with its
    operations.
    """
    if "transactions" not in document:
        return {}
    raw = document["transactions"]
    if not isinstance(raw, list):
        raise PersistenceError("data file transactions must be a list")
    transactions: dict[str, list[dict[str, Any]]] = {}
    for record in raw:
        if not isinstance(record, dict) or set(record.keys()) != {
            "transactionId",
            "operations",
        }:
            raise PersistenceError(
                "each transaction record must be an object with transactionId and operations"
            )
        transaction_id = record["transactionId"]
        if not isinstance(transaction_id, str) or transaction_id == "":
            raise PersistenceError("transaction id must be a non-empty string")
        if transaction_id in transactions:
            raise PersistenceError(f"duplicate transaction {transaction_id!r} in data file")
        try:
            entries = _parse_transaction_entries(record["operations"])
        except ValueError as exc:
            raise PersistenceError(
                f"stored transaction violates input constraints: {exc}"
            ) from exc
        for entry in entries:
            identity = (entry["replicaId"], entry["operationId"])
            if identity not in identities:
                raise PersistenceError(
                    f"transaction {transaction_id!r} names no accepted operation "
                    f"for identity {identity!r}"
                )
        transactions[transaction_id] = entries
    return transactions


def _validate_stored_acks(
    document: Any,
    checkpoints: dict[str, int],
    records: list[tuple[str, dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate the optional ``acks`` section of a data file.

    Returns a clean ``{(peerId, ackId): {"cursor", "operations"}}``
    mapping. The section is optional (a version:1 file written before
    consumption receipts existed simply has none, and recovers with no
    receipt bindings); when present it must be a list of
    ``{"peerId", "ackId", "cursor", "operations"}`` records satisfying the
    live acknowledge constraints, one per ``(peerId, ackId)`` pair. Because
    a receipt is committed atomically with the checkpoint advance it
    caused, every binding must name a registered peer whose checkpoint has
    reached at least the binding's cursor, and the covered segment must
    match the accepted log exactly — anything else is a corrupt file.
    """
    if "acks" not in document:
        return {}
    raw = document["acks"]
    if not isinstance(raw, list):
        raise PersistenceError("data file acks must be a list")
    acks: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "peerId",
            "ackId",
            "cursor",
            "operations",
        }:
            raise PersistenceError(
                "each ack record must be an object with peerId, ackId, cursor, operations"
            )
        peer_id = entry["peerId"]
        ack_id = entry["ackId"]
        cursor = entry["cursor"]
        if not isinstance(peer_id, str) or peer_id == "":
            raise PersistenceError("ack peerId must be a non-empty string")
        if not isinstance(ack_id, str) or ack_id == "":
            raise PersistenceError("ackId must be a non-empty string")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise PersistenceError("ack cursor must be a non-negative integer")
        try:
            operations = _parse_ack_operations(entry["operations"])
        except ValueError as exc:
            raise PersistenceError(
                f"stored ack violates input constraints: {exc}"
            ) from exc
        key = (peer_id, ack_id)
        if key in acks:
            raise PersistenceError(f"duplicate ack binding {key!r} in data file")
        registered = checkpoints.get(peer_id)
        if registered is None or registered < cursor:
            raise PersistenceError(
                f"ack binding {key!r} names a peer whose checkpoint has not reached it"
            )
        start = cursor - len(operations)
        if start < 0 or cursor > len(records):
            raise PersistenceError(
                f"ack binding {key!r} covers a segment outside the accepted log"
            )
        for (replica_id, operation), identity in zip(records[start:cursor], operations):
            if (
                replica_id != identity["replicaId"]
                or operation["operationId"] != identity["operationId"]
            ):
                raise PersistenceError(
                    f"ack binding {key!r} does not match the accepted log segment"
                )
        acks[key] = {"cursor": cursor, "operations": operations}
    return acks


def _is_sha256_hex64(value: Any) -> bool:
    """Return True for exactly 64 lowercase hexadecimal characters."""
    return isinstance(value, str) and len(value) == 64 and all(
        ("0" <= ch <= "9") or ("a" <= ch <= "f") for ch in value
    )


def _validate_stored_policy_events(document: Any) -> list[dict[str, Any]]:
    """Validate the optional ``policyEvents`` section of a data file.

    Returns a clean ordered list of policy-change events, each
    ``{"sequence", "digest", "tokens"}``. The section is optional (a
    version:1 file written before scope-policy change auditing existed
    simply has none, and recovers with an empty history); when present it
    must be a list whose entries carry exactly those three keys:
    ``sequence`` is a non-boolean positive integer starting at 1 and
    increasing without gaps, ``digest`` is 64 lowercase hexadecimal
    characters (the SHA-256 recorded at reload time), and ``tokens`` is a
    non-boolean non-negative integer (the reloaded policy's entry count).
    Anything else is a corrupt file.
    """
    if "policyEvents" not in document:
        return []
    raw = document["policyEvents"]
    if not isinstance(raw, list):
        raise PersistenceError("data file policyEvents must be a list")
    events: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "sequence",
            "digest",
            "tokens",
        }:
            raise PersistenceError(
                "each policy event must be an object with sequence, digest, tokens"
            )
        sequence = entry["sequence"]
        tokens = entry["tokens"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise PersistenceError("policy event sequence must be a positive integer")
        if sequence != len(events) + 1:
            raise PersistenceError(
                "policy event sequences must start at 1 and continue without gaps"
            )
        if not _is_sha256_hex64(entry["digest"]):
            raise PersistenceError("policy event digest must be 64 lowercase hex characters")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise PersistenceError("policy event tokens must be a non-negative integer")
        events.append(
            {"sequence": sequence, "digest": entry["digest"], "tokens": tokens}
        )
    return events


def _load_data_file_complete(
    path: str,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, int],
    dict[tuple[str, str], str],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Read and strictly validate a data file, returning every section.

    Returns the accepted operations in their original commit order, the
    persisted ``{peerId: cursor}`` checkpoints, the persisted
    ``{(replicaId, operationId): policy}`` automatic-resolution policy
    bindings, the persisted ``{transactionId: entries}`` transaction
    bindings, the persisted ``{(peerId, ackId): receipt}`` consumption
    receipts, and the persisted scope-policy change events (each empty for
    a version:1 file written before that section existed). Raises
    PersistenceError when the file is missing-readable, not UTF-8 JSON,
    has an unexpected structure, or contains records, checkpoints, policy
    bindings, transaction bindings, receipts, or policy events violating
    the live constraints.
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
        "policies",
        "transactions",
        "acks",
        "policyEvents",
    } or "version" not in document or "operations" not in document:
        raise PersistenceError(
            "data file root must be an object with version and operations "
            "and optionally checkpoints, policies, transactions, acks, and "
            "policyEvents"
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
    policies = _validate_stored_policies(document, identities)
    transactions = _validate_stored_transactions(document, identities)
    acks = _validate_stored_acks(document, checkpoints, records)
    policy_events = _validate_stored_policy_events(document)
    return records, checkpoints, policies, transactions, acks, policy_events


def load_data_file_full(
    path: str,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int], dict[tuple[str, str], str]]:
    """Read and strictly validate a data file.

    Returns the accepted operations in their original commit order, the
    persisted ``{peerId: cursor}`` checkpoints, and the persisted
    ``{(replicaId, operationId): policy}`` automatic-resolution policy
    bindings (both empty for a version:1 file written before those sections
    existed). Persisted transaction bindings, consumption receipts, and
    scope-policy change events are validated the same way but not returned
    here. Raises PersistenceError when the file is missing-readable, not
    UTF-8 JSON, has an unexpected structure, or contains records,
    checkpoints, or policy bindings violating the live constraints.
    """
    records, checkpoints, policies, _, _, _ = _load_data_file_complete(path)
    return records, checkpoints, policies


def load_data_file_transactions(path: str) -> dict[str, list[dict[str, Any]]]:
    """Read and strictly validate a data file, returning its transactions.

    Thin wrapper over :func:`_load_data_file_complete` for callers that
    only need the persisted ``{transactionId: entries}`` bindings; every
    other section is validated the same way but not returned.
    """
    _, _, _, transactions, _, _ = _load_data_file_complete(path)
    return transactions


def load_data_file_acks(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Read and strictly validate a data file, returning its receipts.

    Thin wrapper over :func:`_load_data_file_complete` for callers that
    only need the persisted ``{(peerId, ackId): receipt}`` consumption
    receipts; every other section is validated the same way but not
    returned.
    """
    _, _, _, _, acks, _ = _load_data_file_complete(path)
    return acks


def load_data_file(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Read and strictly validate a data file, returning its operations.

    Thin wrapper over :func:`load_data_file_full` for callers that only
    need the accepted-operation log; persisted checkpoints and policy
    bindings are validated the same way but not returned.
    """
    records, _, _ = load_data_file_full(path)
    return records


def load_data_file_policy_events(path: str) -> list[dict[str, Any]]:
    """Read and strictly validate a data file, returning its policy events.

    Thin wrapper over :func:`_load_data_file_complete` for callers that
    only need the persisted scope-policy change history; every other
    section is validated the same way but not returned.
    """
    _, _, _, _, _, policy_events = _load_data_file_complete(path)
    return policy_events


def ensure_data_file(
    path: str,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, int],
    dict[tuple[str, str], str],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate the data-file location and return its committed state.

    A missing target file is accepted (its parent directory must exist and
    be writable); an existing target must be a regular, parseable data
    file. Returns ``(records, checkpoints, policies, transactions, acks,
    policy_events)``. Anything else raises PersistenceError.
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
        return _load_data_file_complete(path)
    return [], {}, {}, {}, {}, []


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
        # Policy bindings of accepted automatic resolutions, keyed by the
        # operation identity. An identity absent here (a plain write, a
        # manual resolution, or an imported operation) carries no policy.
        self._policies: dict[tuple[str, str], str] = {}
        # Transaction bindings of accepted atomic batches, keyed by the
        # transaction id and holding the normalized entry list. The binding
        # is local to this replica: it is persisted with its operations but
        # never exported by sync.
        self._transactions: dict[str, list[dict[str, Any]]] = {}
        # Consumption receipts of accepted acknowledgements, keyed by
        # ``(peerId, ackId)`` and holding the bound ``{"cursor",
        # "operations"}`` content. A receipt is not an operation: it never
        # touches the accepted log, and it is committed atomically with the
        # checkpoint advance it caused.
        self._acks: dict[tuple[str, str], dict[str, Any]] = {}
        # Scope-policy change history: one event per successful hot reload,
        # in commit order. Each event is ``{"sequence", "digest",
        # "tokens"}``: a 1-based continuous position, the SHA-256 of the
        # reloaded policy file's raw UTF-8 bytes, and the new policy's entry
        # count. Like the policy itself, the history rides in the data file
        # but is otherwise unrelated to business state; a file written
        # before this section existed recovers with an empty history.
        self._policy_events: list[dict[str, Any]] = []
        self._data_file: str | None = None
        if data_file is not None:
            path = os.path.abspath(data_file)
            # Probe directory-level atomic commit first; an existing data
            # file is never touched by the probe and is opened only for
            # reading afterwards.
            preflight_data_file_directory(path)
            (
                records,
                checkpoints,
                policies,
                transactions,
                acks,
                policy_events,
            ) = ensure_data_file(path)
            with self._lock:
                for replica_id, operation in records:
                    self._commit_locked(replica_id, operation)
                self._checkpoints = dict(checkpoints)
                self._policies = dict(policies)
                self._transactions = dict(transactions)
                self._acks = dict(acks)
                self._policy_events = [dict(event) for event in policy_events]
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
            # Policy bindings ride along in commit order, so each binding is
            # durable in the same atomic commit as its operation.
            "policies": [
                {
                    "replicaId": replica_id,
                    "operationId": operation["operationId"],
                    "policy": self._policies[(replica_id, operation["operationId"])],
                }
                for replica_id, operation in self._accepted
                if (replica_id, operation["operationId"]) in self._policies
            ],
            # Transaction bindings ride along in commit order, so each
            # binding is durable in the same atomic commit as its
            # operations. The binding itself is local: it is never part of
            # the exported sync records.
            "transactions": [
                {"transactionId": transaction_id, "operations": entries}
                for transaction_id, entries in self._transactions.items()
            ],
            # Consumption receipts ride along in commit order, so each
            # binding is durable in the same atomic commit as the
            # checkpoint advance it caused. A receipt is not an operation:
            # it is never part of the accepted log or the exported sync
            # records.
            "acks": [
                {
                    "peerId": peer_id,
                    "ackId": ack_id,
                    "cursor": receipt["cursor"],
                    "operations": receipt["operations"],
                }
                for (peer_id, ack_id), receipt in self._acks.items()
            ],
            # Scope-policy change events ride along in the order their
            # reloads committed, so every successful hot reload and its
            # event become durable in one atomic commit together.
            "policyEvents": [dict(event) for event in self._policy_events],
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

    def apply_auto_resolution(
        self, key: str, request: dict[str, Any]
    ) -> tuple[HTTPStatus, dict[str, Any] | None, str | None]:
        """Apply a validated automatic resolution for ``key``.

        Unlike :meth:`apply_resolution`, neither the value nor the candidate
        set come from the request: the key must currently hold candidates
        with at least two distinct values, and the resolution value is taken
        deterministically from a current candidate selected by the request
        policy — the smallest ``(replicaId, operationId)`` for
        ``lowest_identity``, the largest for ``highest_identity``. The
        request clock must dominate every current candidate. The resolution
        then commits exactly like a manual resolution — one ordinary
        operation in the shared commit order, so it flows through sync
        export/import, the audit streams, the metrics, and the data file
        identically.

        The identity is bound to the key, the clock, and the policy: a
        replay with the same binding is answered from the committed
        operation, and any difference — including a different policy under
        an otherwise identical request — is an operation conflict. The
        binding is persisted atomically with the operation.

        Returns ``(status, operation, error)``: 201/200 carry the committed
        (or previously seen) operation and ``error=None``, or 409 carries
        ``"operation_conflict"`` (known identity, different binding) or
        ``"resolution_conflict"`` (an unseen identity for a missing key or a
        key not currently in value conflict). Raises ValueError when the
        clock does not dominate every candidate; raises PersistenceError when
        the durable commit fails, in which case memory, the identity index,
        the policy bindings, and the file are unchanged.
        """
        replica_id = request["replicaId"]
        operation = {
            "operationId": request["operationId"],
            "key": key,
            "value": "",
            "clock": request["clock"],
        }
        with self._lock:
            identity = (replica_id, operation["operationId"])
            seen = self._operations.get(identity)

            # Identity replay semantics share the write/manual-resolution
            # rule and take precedence over the live conflict precondition.
            # The request carries no value of its own (the policy derives
            # it server-side), so the identity binding is key plus clock
            # plus policy: a replay of the same binding is answered from
            # the committed operation and reports the value that was
            # originally chosen, however the candidate set has moved since.
            # A different key, clock, or policy under the same identity is
            # an operation conflict; an identity committed without a policy
            # binding (a plain write, a manual resolution, or an imported
            # operation) never matches a policy-carrying request.
            if seen is not None:
                if (
                    seen["key"] == key
                    and seen["clock"] == operation["clock"]
                    and self._policies.get(identity) == request["policy"]
                ):
                    return HTTPStatus.OK, seen, None
                return HTTPStatus.CONFLICT, None, "operation_conflict"

            current = self._candidates.get(key, [])
            if not current or all(c["value"] == current[0]["value"] for c in current):
                return HTTPStatus.CONFLICT, None, "resolution_conflict"

            # Deterministic policy: the candidate with the smallest or
            # largest (replicaId, operationId) supplies the resolution value.
            if request["policy"] == "highest_identity":
                chosen = max(current, key=lambda c: (c["replicaId"], c["operationId"]))
            else:
                chosen = min(current, key=lambda c: (c["replicaId"], c["operationId"]))
            operation["value"] = chosen["value"]

            if not all(clock_dominates(operation["clock"], c["clock"]) for c in current):
                raise ValueError("clock does not dominate every candidate")

            next_candidates = self._next_candidates(current, replica_id, operation)
            if self._data_file is not None:
                # Same commit discipline as manual resolutions and writes:
                # the atomic rename is the single commit point, and memory
                # moves only after it. The policy binding is staged together
                # with the operation so both are durable in one commit.
                self._accepted.append((replica_id, operation))
                self._policies[identity] = request["policy"]
                try:
                    self._persist_locked()
                except BaseException:
                    self._accepted.pop()
                    del self._policies[identity]
                    raise
            else:
                self._accepted.append((replica_id, operation))
                self._policies[identity] = request["policy"]
            self._operations[identity] = operation
            self._candidates[key] = next_candidates
            return HTTPStatus.CREATED, operation, None

    def apply_auto_resolutions(
        self, requests: list[dict[str, Any]]
    ) -> tuple[HTTPStatus, list[dict[str, Any]], int, int, str | None]:
        """Apply validated automatic resolutions sequentially as one batch.

        Each entry carries the same fields as a single automatic resolution
        plus its target ``key``; the parser already guarantees 1-100 entries
        with distinct keys and distinct identities. Entries are processed in
        request order against a staged view of the store, so the semantics
        are exactly those of :meth:`apply_auto_resolution` applied in
        sequence:

        - a known identity with the same binding (key, clock, and policy) is
          a replay, reporting the originally chosen value;
        - a known identity with a different binding is an operation
          conflict;
        - an unseen identity for a missing key, a key not currently in value
          conflict, a key whose candidates moved, or a request clock that
          does not dominate every current candidate is a resolution
          conflict (a structurally *illegal* clock is rejected earlier by
          the parser as an invalid request, distinct from a legal clock
          that simply fails to dominate the live candidates);

        The whole batch is validated (dry-run) before anything is persisted
        or made visible: any conflict or invalid clock rejects the entire
        batch unchanged, however far validation got. All new operations and
        their policy bindings are then committed together in one atomic
        commit, exactly like a sync-import batch.

        Returns ``(status, results, accepted, replayed, error)``: 201 when
        at least one entry was newly committed or 200 when every entry was a
        replay, with ``results`` in request order (each carrying ``key``,
        ``replicaId``, ``operationId``, the selected ``value``, and
        ``policy``); or 409 with ``"operation_conflict"`` /
        ``"resolution_conflict"`` and an unchanged store. Raises
        PersistenceError when the durable commit fails, in which case
        memory, the identity index, the policy bindings, and the file are
        unchanged.
        """
        with self._lock:
            # Staged copies keep the whole dry run off the visible state:
            # the batch is fixed in its entirety before the single commit.
            staged_operations = dict(self._operations)
            staged_policies = dict(self._policies)
            staged_candidates = {
                key: list(candidates) for key, candidates in self._candidates.items()
            }
            new_records: list[tuple[str, dict[str, Any]]] = []
            new_policies: dict[tuple[str, str], str] = {}
            results: list[dict[str, Any]] = []
            accepted = 0
            replayed = 0

            for entry in requests:
                replica_id = entry["replicaId"]
                key = entry["key"]
                identity = (replica_id, entry["operationId"])
                operation = {
                    "operationId": entry["operationId"],
                    "key": key,
                    "value": "",
                    "clock": entry["clock"],
                }
                seen = staged_operations.get(identity)
                if seen is not None:
                    # Same replay binding as the single-request path: key,
                    # clock, and policy must all match the committed
                    # operation (which must carry a policy binding).
                    if (
                        seen["key"] == key
                        and seen["clock"] == operation["clock"]
                        and staged_policies.get(identity) == entry["policy"]
                    ):
                        replayed += 1
                        results.append(
                            {
                                "key": key,
                                "replicaId": replica_id,
                                "operationId": entry["operationId"],
                                "value": seen["value"],
                                "policy": entry["policy"],
                            }
                        )
                        continue
                    return HTTPStatus.CONFLICT, [], 0, 0, "operation_conflict"

                current = staged_candidates.get(key, [])
                if not current or all(c["value"] == current[0]["value"] for c in current):
                    return HTTPStatus.CONFLICT, [], 0, 0, "resolution_conflict"

                if entry["policy"] == "highest_identity":
                    chosen = max(current, key=lambda c: (c["replicaId"], c["operationId"]))
                else:
                    chosen = min(current, key=lambda c: (c["replicaId"], c["operationId"]))
                operation["value"] = chosen["value"]

                # A legal clock that nevertheless fails to dominate the live
                # candidates is a failed resolution precondition (409), not
                # a malformed request: structurally invalid clocks were
                # rejected by the parser before the store was ever reached.
                if not all(clock_dominates(operation["clock"], c["clock"]) for c in current):
                    return HTTPStatus.CONFLICT, [], 0, 0, "resolution_conflict"

                staged_candidates[key] = self._next_candidates(
                    current, replica_id, operation
                )
                staged_operations[identity] = operation
                staged_policies[identity] = entry["policy"]
                new_records.append((replica_id, operation))
                new_policies[identity] = entry["policy"]
                accepted += 1
                results.append(
                    {
                        "key": key,
                        "replicaId": replica_id,
                        "operationId": entry["operationId"],
                        "value": operation["value"],
                        "policy": entry["policy"],
                    }
                )

            if not new_records:
                return HTTPStatus.OK, results, accepted, replayed, None

            # Commit every new operation and binding together. The atomic
            # rename is the single durable commit point; memory, the identity
            # index, and the policy bindings move only after it succeeds, so
            # a failed durable commit leaves everything exactly as before.
            if self._data_file is not None:
                previous_length = len(self._accepted)
                self._accepted.extend(new_records)
                self._policies.update(new_policies)
                try:
                    self._persist_locked()
                except BaseException:
                    del self._accepted[previous_length:]
                    for identity in new_policies:
                        del self._policies[identity]
                    raise
            else:
                self._accepted.extend(new_records)
                self._policies.update(new_policies)
            for replica_id, operation in new_records:
                self._operations[(replica_id, operation["operationId"])] = operation
                op_key = operation["key"]
                self._candidates[op_key] = self._next_candidates(
                    self._candidates.get(op_key, []), replica_id, operation
                )
            return HTTPStatus.CREATED, results, accepted, replayed, None

    def plan_auto_resolutions(
        self, requests: list[dict[str, Any]]
    ) -> tuple[HTTPStatus, list[dict[str, Any]], int, int, str | None]:
        """Preview a batch of automatic resolutions without committing.

        Read-only dry run of :meth:`apply_auto_resolutions` against one
        complete committed snapshot. The parser already guarantees 1-100
        entries with distinct keys and distinct identities. The entries are
        processed in request order against staged copies of the store, using
        exactly the same per-entry rules as the committing batch:

        - a known identity with the same binding (key, clock, and policy) is
          counted as a replay, reporting the originally chosen value;
        - a known identity with a different binding is an operation
          conflict;
        - an unseen identity for a missing key, a key not currently in value
          conflict, or a request clock that does not dominate every current
          candidate is a resolution conflict.

        Nothing is written: no candidates, accepted log, policy bindings,
        checkpoints, audit state, or data file change, and no temporary file
        is created. Because the dry run runs entirely under the commit lock
        on staged copies, a concurrent commit can only move the whole store
        from one complete snapshot to another — a returned plan is fixed by
        the snapshot observed and never changes afterwards.

        Returns ``(status, results, accepted, replayed, error)`` exactly
        shaped like :meth:`apply_auto_resolutions`, except success always
        carries HTTP 200 (the caller adds the ``"planned"`` status string):
        ``results`` in request order (each carrying ``key``, ``replicaId``,
        ``operationId``, the selected ``value``, and ``policy``), with
        ``accepted`` counting entries that a commit would newly create and
        ``replayed`` the same-binding entries; or 409 with
        ``"operation_conflict"`` / ``"resolution_conflict"`` and no results.
        """
        with self._lock:
            # Staged copies keep the entire preview off the visible state;
            # the same discipline as the committing batch, but nothing from
            # the staging is ever written back or persisted.
            staged_operations = dict(self._operations)
            staged_policies = dict(self._policies)
            staged_candidates = {
                key: list(candidates) for key, candidates in self._candidates.items()
            }
            results: list[dict[str, Any]] = []
            accepted = 0
            replayed = 0

            for entry in requests:
                replica_id = entry["replicaId"]
                key = entry["key"]
                identity = (replica_id, entry["operationId"])
                operation = {
                    "operationId": entry["operationId"],
                    "key": key,
                    "value": "",
                    "clock": entry["clock"],
                }
                seen = staged_operations.get(identity)
                if seen is not None:
                    # Same replay binding as the committing paths: key,
                    # clock, and policy must all match the committed
                    # operation (which must carry a policy binding).
                    if (
                        seen["key"] == key
                        and seen["clock"] == operation["clock"]
                        and staged_policies.get(identity) == entry["policy"]
                    ):
                        replayed += 1
                        results.append(
                            {
                                "key": key,
                                "replicaId": replica_id,
                                "operationId": entry["operationId"],
                                "value": seen["value"],
                                "policy": entry["policy"],
                            }
                        )
                        continue
                    return HTTPStatus.CONFLICT, [], 0, 0, "operation_conflict"

                current = staged_candidates.get(key, [])
                if not current or all(c["value"] == current[0]["value"] for c in current):
                    return HTTPStatus.CONFLICT, [], 0, 0, "resolution_conflict"

                if entry["policy"] == "highest_identity":
                    chosen = max(current, key=lambda c: (c["replicaId"], c["operationId"]))
                else:
                    chosen = min(current, key=lambda c: (c["replicaId"], c["operationId"]))
                operation["value"] = chosen["value"]

                # A legal clock that nevertheless fails to dominate the live
                # candidates is a failed resolution precondition (409), as
                # in the committing batch; structurally invalid clocks were
                # rejected by the parser before the store was reached.
                if not all(clock_dominates(operation["clock"], c["clock"]) for c in current):
                    return HTTPStatus.CONFLICT, [], 0, 0, "resolution_conflict"

                staged_candidates[key] = self._next_candidates(
                    current, replica_id, operation
                )
                staged_operations[identity] = operation
                staged_policies[identity] = entry["policy"]
                accepted += 1
                results.append(
                    {
                        "key": key,
                        "replicaId": replica_id,
                        "operationId": entry["operationId"],
                        "value": operation["value"],
                        "policy": entry["policy"],
                    }
                )

            return HTTPStatus.OK, results, accepted, replayed, None

    def apply_transaction(
        self, transaction_id: str, entries: list[dict[str, Any]]
    ) -> tuple[HTTPStatus, list[dict[str, Any]], int, int, str | None]:
        """Apply a validated atomic multi-key transaction as one commit.

        The parser already guarantees 1-100 entries with distinct keys,
        distinct identities, and per-entry candidate sets of distinct
        identities. The transaction id is bound to the exact normalized
        entry list: a replay of the same id with identical entries is
        answered from the committed state without re-checking anything, and
        the same id with different entries is an operation conflict.

        Otherwise the entries are processed in request order against a
        staged view of the store:

        - a known ``(replicaId, operationId)`` with identical operation
          content is a replay and is answered from the committed operation
          without any state check;
        - a known identity with different content is an operation
          conflict;
        - a new identity commits only when its expected candidate set
          exactly matches the key's current candidate identities (an empty
          set expects no candidates) and its clock strictly dominates every
          one of those candidates.

        Any failure rejects the whole transaction unchanged. All new
        operations and the transaction binding are then committed together
        in one atomic commit, exactly like a sync-import batch: the
        operations enter the shared accepted log in request order and flow
        through sync export, the audit streams, the metrics, the
        verification digest, and the per-operation archive identically.

        Returns ``(status, results, accepted, replayed, error)``: 201 when
        at least one entry was newly committed or 200 when every entry was
        a replay (including a replay of the whole transaction id), with
        ``results`` in request order (each carrying ``key``, ``replicaId``,
        ``operationId``, and the committed ``value``); or 409 with
        ``"operation_conflict"`` / ``"transaction_conflict"`` and an
        unchanged store. Raises ValueError when an entry clock does not
        dominate its expected candidates (a malformed request, not a state
        conflict); raises PersistenceError when the durable commit fails,
        in which case memory, the identity index, the transaction bindings,
        and the file are unchanged and the request can be retried.
        """
        with self._lock:
            binding = self._transactions.get(transaction_id)
            if binding is not None:
                if binding == entries:
                    # Identical replay: answer from the committed operations
                    # without inspecting the current candidate state.
                    replay_results = []
                    for entry in entries:
                        seen = self._operations[
                            (entry["replicaId"], entry["operationId"])
                        ]
                        replay_results.append(
                            {
                                "key": entry["key"],
                                "replicaId": entry["replicaId"],
                                "operationId": entry["operationId"],
                                "value": seen["value"],
                            }
                        )
                    return HTTPStatus.OK, replay_results, 0, len(entries), None
                return HTTPStatus.CONFLICT, [], 0, 0, "operation_conflict"

            # Staged copies keep the whole dry run off the visible state:
            # the transaction is fixed in its entirety before the commit.
            staged_operations = dict(self._operations)
            staged_candidates = {
                key: list(candidates) for key, candidates in self._candidates.items()
            }
            new_records: list[tuple[str, dict[str, Any]]] = []
            results: list[dict[str, Any]] = []
            accepted = 0
            replayed = 0

            for entry in entries:
                replica_id = entry["replicaId"]
                key = entry["key"]
                identity = (replica_id, entry["operationId"])
                operation = {
                    "operationId": entry["operationId"],
                    "key": key,
                    "value": entry["value"],
                    "clock": entry["clock"],
                }
                seen = staged_operations.get(identity)
                if seen is not None:
                    if seen == operation:
                        replayed += 1
                        results.append(
                            {
                                "key": key,
                                "replicaId": replica_id,
                                "operationId": entry["operationId"],
                                "value": seen["value"],
                            }
                        )
                        continue
                    return HTTPStatus.CONFLICT, [], 0, 0, "operation_conflict"

                current = staged_candidates.get(key, [])
                current_identities = {
                    (c["replicaId"], c["operationId"]) for c in current
                }
                expected_identities = {
                    (c["replicaId"], c["operationId"]) for c in entry["candidates"]
                }
                if current_identities != expected_identities:
                    return HTTPStatus.CONFLICT, [], 0, 0, "transaction_conflict"
                # A legal clock that fails to dominate the expected
                # candidates is a malformed transaction (400), not a state
                # conflict; the dry run has touched nothing.
                if not all(
                    clock_dominates(operation["clock"], c["clock"]) for c in current
                ):
                    raise ValueError("clock does not dominate every expected candidate")

                staged_candidates[key] = self._next_candidates(
                    current, replica_id, operation
                )
                staged_operations[identity] = operation
                new_records.append((replica_id, operation))
                accepted += 1
                results.append(
                    {
                        "key": key,
                        "replicaId": replica_id,
                        "operationId": entry["operationId"],
                        "value": operation["value"],
                    }
                )

            status = HTTPStatus.CREATED if new_records else HTTPStatus.OK
            # The binding is stored with deep-copied entries so later
            # caller-side mutation can never rewrite a committed binding.
            binding_entries = [
                {
                    **entry,
                    "clock": dict(entry["clock"]),
                    "candidates": [dict(c) for c in entry["candidates"]],
                }
                for entry in entries
            ]
            # Commit the new operations and the binding together. The
            # atomic rename is the single durable commit point; memory, the
            # identity index, and the bindings move only after it succeeds,
            # so a failed durable commit leaves everything exactly as
            # before. Even a pure-replay transaction persists its (new)
            # binding, so restart replay/conflict decisions are identical.
            if self._data_file is not None:
                previous_length = len(self._accepted)
                self._accepted.extend(new_records)
                self._transactions[transaction_id] = binding_entries
                try:
                    self._persist_locked()
                except BaseException:
                    del self._accepted[previous_length:]
                    del self._transactions[transaction_id]
                    raise
            else:
                self._accepted.extend(new_records)
                self._transactions[transaction_id] = binding_entries
            for replica_id, operation in new_records:
                self._operations[(replica_id, operation["operationId"])] = operation
                op_key = operation["key"]
                self._candidates[op_key] = self._next_candidates(
                    self._candidates.get(op_key, []), replica_id, operation
                )
            return status, results, accepted, replayed, None

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

    def get_replication_snapshot(self) -> dict[str, Any]:
        """Return the read-only replication-snapshot verification summary.

        The candidate state, the accepted-log position, and the whole
        checkpoint mapping are read together under the same commit lock used
        by local writes, sync imports, repairs, and checkpoint commits, so
        the response always describes a single commit: a read can never
        observe half an import batch, a partially applied repair, or a
        checkpoint commit halfway through its durable update — only the old
        or the new complete state. The snapshot mutates neither memory nor
        the data file and creates no temporary file.

        The result carries exactly seven fields:

        - ``status``: the verification conclusion, always ``"ok"`` — the
          summary is assembled atomically from one commit, so it is
          internally consistent by construction.
        - ``candidateDigest``: the 64-character lowercase hexadecimal
          SHA-256 of the canonical candidate snapshot, following exactly
          the verification-digest rules (it covers only the current
          candidate sets).
        - ``snapshotDigest``: the 64-character lowercase hexadecimal
          SHA-256 of the canonical bytes produced by
          :func:`_replication_snapshot_input` from the candidate digest,
          the log cursor, and the checkpoint mapping, in that order.
        - ``logCursor``: the number of first-accepted operations in the
          shared log — the sync-export resume cursor at the tail of the
          log.
        - ``keys`` and ``candidateVersions``: the same counts reported by
          :meth:`get_metrics` and :meth:`get_verification_digest`.
        - ``checkpoints``: the full ``{peerId: cursor}`` mapping of
          sender-side replication progress (empty when none is
          registered).

        With ``--data-file`` the log and the checkpoints are rebuilt
        identically during recovery, so the same state yields the same
        verification result before and after a restart.
        """
        with self._lock:
            keys = len(self._candidates)
            candidate_versions = sum(len(c) for c in self._candidates.values())
            candidate_input = _verification_digest_input(self._candidates)
            log_cursor = len(self._accepted)
            checkpoints = dict(self._checkpoints)
        candidate_digest = hashlib.sha256(candidate_input).hexdigest()
        snapshot_input = _replication_snapshot_input(
            candidate_digest, log_cursor, checkpoints
        )
        return {
            "status": "ok",
            "candidateDigest": candidate_digest,
            "snapshotDigest": hashlib.sha256(snapshot_input).hexdigest(),
            "logCursor": log_cursor,
            "keys": keys,
            "candidateVersions": candidate_versions,
            "checkpoints": checkpoints,
        }

    def compare_replication_snapshot(
        self, replica_id: str, snapshot: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Diff the current candidates against a remote replica's snapshot.

        The local candidate state is copied under the same commit lock used
        by local writes, sync imports, repairs, and checkpoint commits, so
        the comparison always describes a single commit: it can never
        observe half an import batch, a partially applied repair, or a
        half-committed recovery — only the old or the new complete state.
        The remote snapshot (already normalized by
        :func:`parse_replication_compare_payload`) is only read: nothing is
        imported, and no repair, transaction, sync, checkpoint, or
        persistence runs. The read mutates neither memory nor the data
        file and creates no temporary file, so with ``--data-file`` the
        same local state and the same remote snapshot yield the same
        report before and after a restart.

        Both sides are digested with exactly the verification-digest rules
        (:func:`_verification_digest_input`), so ``identical`` is true
        precisely when the two candidate states are the same — in
        particular when both are empty. The result groups the union of
        business keys, each key's entries sorted by ``(replicaId,
        operationId)`` and carrying both sides' candidate (``null`` on the
        side that lacks the identity) under one ``kind`` mark:

        - ``"shared"``: both sides hold the identity with the same value
          and the same clock — not a difference.
        - ``"missing_remote"``: only the local side holds the identity.
        - ``"missing_local"``: only the remote side holds the identity.
        - ``"conflict"``: both sides hold the identity but the values
          differ (a content conflict).
        - ``"clock"``: both sides hold the identity with the same value
          but different clocks (one side's clock covers the other's).

        The summary reports both sides' key counts and candidate counts,
        both digests, whether they are identical, and ``differences`` —
        the number of non-shared entries, the minimal candidate-level
        difference count a follow-up sync must reconcile.
        """
        with self._lock:
            local = {
                key: [
                    {
                        "value": candidate["value"],
                        "clock": dict(candidate["clock"]),
                        "replicaId": candidate["replicaId"],
                        "operationId": candidate["operationId"],
                    }
                    for candidate in candidates
                ]
                for key, candidates in self._candidates.items()
            }
        local_digest = hashlib.sha256(_verification_digest_input(local)).hexdigest()
        remote_digest = hashlib.sha256(_verification_digest_input(snapshot)).hexdigest()

        groups: list[dict[str, Any]] = []
        differences = 0
        for key in sorted(set(local) | set(snapshot)):
            local_by_id = {
                (c["replicaId"], c["operationId"]): c for c in local.get(key, [])
            }
            remote_by_id = {
                (c["replicaId"], c["operationId"]): c for c in snapshot.get(key, [])
            }
            entries: list[dict[str, Any]] = []
            for identity in sorted(set(local_by_id) | set(remote_by_id)):
                local_candidate = local_by_id.get(identity)
                remote_candidate = remote_by_id.get(identity)
                if local_candidate is None:
                    kind = "missing_local"
                elif remote_candidate is None:
                    kind = "missing_remote"
                elif local_candidate["value"] != remote_candidate["value"]:
                    kind = "conflict"
                elif local_candidate["clock"] != remote_candidate["clock"]:
                    kind = "clock"
                else:
                    kind = "shared"
                if kind != "shared":
                    differences += 1
                entries.append(
                    {
                        "kind": kind,
                        "local": local_candidate,
                        "remote": remote_candidate,
                    }
                )
            groups.append({"key": key, "differences": entries})
        return {
            "status": "ok",
            "replicaId": replica_id,
            "keys": groups,
            "summary": {
                "localKeys": len(local),
                "remoteKeys": len(snapshot),
                "localCandidates": sum(len(c) for c in local.values()),
                "remoteCandidates": sum(len(c) for c in snapshot.values()),
                "localDigest": local_digest,
                "remoteDigest": remote_digest,
                "identical": local_digest == remote_digest,
                "differences": differences,
            },
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

    def get_audit_log_chain(
        self, after: int, limit: int
    ) -> tuple[list[dict[str, Any]], int, bool, str]:
        """Return one page of the global audit chain from a single snapshot.

        The chain covers the whole shared accepted-operation log in global
        commit order — ordinary writes, stale writes that added no
        candidate, accepted conflict repairs, and sync-imported records
        alike; identical replays, conflicting or malformed requests,
        uncommitted requests, and rejected batches never enter the log and
        so never enter the chain. Each link carries ``sequence`` (the
        1-based position in the log), ``prevDigest`` (the previous link's
        digest, or 64 ``"0"`` characters for the first link), and
        ``digest`` (computed by :func:`_audit_chain_link`).

        The page slice, the returned cursor, ``has_more``, and the chain
        ``head`` (the digest of the last link, or 64 ``"0"`` characters
        when the log is empty) are all computed against the same snapshot
        under the commit lock, so they always describe one commit even
        while commits are in flight; the head never varies with paging.
        The snapshot mutates neither memory nor the data file and creates
        no temporary file. Returns ``(entries, next_cursor, has_more,
        head)`` where ``next_cursor`` is the number of links skipped after
        this page. Raises ValueError when ``after`` is past the chain
        length. With ``--data-file`` the log is rebuilt identically during
        recovery, so the same history yields the same chain, cursors, and
        head before and after a restart.
        """
        with self._lock:
            total = len(self._accepted)
            if after > total:
                raise ValueError("after is past the end of the audit chain")
            page_end = min(after + limit, total)
            previous = _AUDIT_CHAIN_GENESIS
            entries: list[dict[str, Any]] = []
            for index, (replica_id, operation) in enumerate(self._accepted):
                sequence = index + 1
                digest = _audit_chain_link(previous, sequence, replica_id, operation)
                if after <= index < page_end:
                    entries.append(
                        {
                            "sequence": sequence,
                            "prevDigest": previous,
                            "digest": digest,
                        }
                    )
                previous = digest
            head = previous
        next_cursor = after + len(entries)
        return entries, next_cursor, next_cursor < total, head

    def get_audit_log_verify(
        self, after: int, limit: int, expected_head: str, expected_count: int
    ) -> dict[str, Any]:
        """Return one chain page plus an independent whole-chain verification.

        This is the read-only integrity-verification companion to
        :meth:`get_audit_log_chain`. The ``entries`` page, ``nextCursor``,
        ``hasMore``, and the chain-tail ``head`` are produced exactly as for
        the plain chain query — same paging, same link shape, same genesis —
        and the only addition is ``verification``, the independent
        conclusion produced by :func:`_audit_log_verification_locked`,
        passed the external ``expected_head``/``expected_count``.

        Paging trims only the returned ``entries`` page: the ``head``, the
        full length used for the count comparison, and the ``verification``
        conclusion always cover the complete log. The page slice, cursors,
        head, full materialized links, and verification are all computed
        from one snapshot under the commit lock, so a concurrent commit is
        observed only as the whole old or the whole new history — the page,
        the expectations comparison, and the conclusion can never disagree
        across commits. The scan strictly re-walks the log and recomputes
        every link rather than trusting the materialized page; it mutates
        neither memory nor the data file and creates no temporary file.

        Returns a report with exactly five fields: ``entries``,
        ``nextCursor``, ``hasMore``, ``head`` (identical in meaning to
        :meth:`get_audit_log_chain`), and ``verification`` (whose ``status``
        is ``"ok"`` exactly when the internal chain is intact and both
        external expectations match, otherwise ``"broken"``). An ``after``
        equal to the chain length is a valid stable empty page whose
        verification still covers the complete log. Raises ValueError when
        ``after`` is past the chain length of the snapshot. With
        ``--data-file`` the log is rebuilt identically during recovery, so
        the same history yields the same page, head, and verification
        before and after a restart.
        """
        with self._lock:
            total = len(self._accepted)
            if after > total:
                raise ValueError("after is past the end of the audit chain")
            page_end = min(after + limit, total)
            previous = _AUDIT_CHAIN_GENESIS
            all_entries: list[dict[str, Any]] = []
            page: list[dict[str, Any]] = []
            for index, (replica_id, operation) in enumerate(self._accepted):
                sequence = index + 1
                digest = _audit_chain_link(previous, sequence, replica_id, operation)
                link = {
                    "sequence": sequence,
                    "prevDigest": previous,
                    "digest": digest,
                }
                all_entries.append(link)
                if after <= index < page_end:
                    page.append(dict(link))
                previous = digest
            head = previous
            verification = _audit_log_verification_locked(
                self._accepted, all_entries, expected_head, expected_count
            )
        next_cursor = after + len(page)
        return {
            "entries": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "head": head,
            "verification": verification,
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

    def acknowledge_operations(
        self, peer_id: str, ack_id: str, cursor: int, operations: list[dict[str, str]]
    ) -> tuple[HTTPStatus, str | None]:
        """Commit a verifiable consumption receipt for ``peer_id``.

        A receipt is not an operation: it never touches the accepted log,
        the identity index, the candidate state, sync export, the per-key
        audit, or the metrics counters. It shares their commit lock,
        however, so validating the segment against the accepted log,
        advancing the peer's checkpoint, and recording the
        ``(peerId, ackId)`` binding are one indivisible commit: a
        concurrent reader sees either the old or the new checkpoint and
        binding, never a half state.

        The receipt must exactly cover the contiguous accepted records from
        the peer's registered checkpoint up to (but not including)
        ``cursor``: ``operations[i]`` must name the identity of the
        ``checkpoint + i``-th accepted record, and the checkpoint plus the
        segment length must equal ``cursor``.

        Returns ``(status, error)``: 201 with ``error=None`` when the
        receipt is newly committed (the checkpoint advanced to ``cursor``),
        or 200 with ``error=None`` when the same ``(peerId, ackId)`` is
        replayed with identical content (answered from the committed
        binding without re-checking the current state, and appending
        nothing). 404 with ``"not_found"`` when the peer never registered a
        checkpoint; 409 with ``"operation_conflict"`` when the
        ``(peerId, ackId)`` binding exists with different content; 409 with
        ``"checkpoint_conflict"`` when ``cursor`` is below the peer's
        current checkpoint; 409 with ``"ack_conflict"`` when the segment
        does not exactly match the accepted log. Raises PersistenceError
        when the durable commit fails, in which case memory, the
        checkpoint, the bindings, and the file are unchanged and the
        request can be retried.
        """
        with self._lock:
            current = self._checkpoints.get(peer_id)
            if current is None:
                return HTTPStatus.NOT_FOUND, "not_found"
            binding = self._acks.get((peer_id, ack_id))
            if binding is not None:
                if binding["cursor"] == cursor and binding["operations"] == operations:
                    # Identical replay: answer from the committed binding,
                    # however the checkpoint has moved since.
                    return HTTPStatus.OK, None
                return HTTPStatus.CONFLICT, "operation_conflict"
            if cursor < current:
                return HTTPStatus.CONFLICT, "checkpoint_conflict"
            # The covered segment is accepted[current:cursor]; it must
            # exist in the log and match the acknowledged identities
            # exactly, in order. The cursor must agree with the segment
            # length, so a receipt can never advance the checkpoint past
            # the accepted log.
            if cursor - current != len(operations):
                return HTTPStatus.CONFLICT, "ack_conflict"
            segment = self._accepted[current:cursor]
            if len(segment) != len(operations) or any(
                replica_id != identity["replicaId"]
                or operation["operationId"] != identity["operationId"]
                for (replica_id, operation), identity in zip(segment, operations)
            ):
                return HTTPStatus.CONFLICT, "ack_conflict"
            receipt = {
                "cursor": cursor,
                "operations": [dict(identity) for identity in operations],
            }
            if self._data_file is not None:
                # Same commit discipline as checkpoints: stage the advanced
                # cursor and the new binding, make the atomic rename the
                # single commit point, and move the visible state only
                # after it succeeds.
                self._checkpoints[peer_id] = cursor
                self._acks[(peer_id, ack_id)] = receipt
                try:
                    self._persist_locked()
                except BaseException:
                    self._checkpoints[peer_id] = current
                    del self._acks[(peer_id, ack_id)]
                    raise
            else:
                self._checkpoints[peer_id] = cursor
                self._acks[(peer_id, ack_id)] = receipt
            return HTTPStatus.CREATED, None

    def get_operation(
        self, replica_id: str, operation_id: str
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one accepted operation by identity from a single snapshot.

        The lookup runs under the same commit lock used by local writes,
        sync imports, repairs, and checkpoints, so the response always
        describes a committed state: a read can never observe half an import
        batch or a partially applied repair. The snapshot mutates neither
        memory, the data file, candidates, metrics, audits, checkpoints,
        nor logs.

        Every first-accepted operation is addressable by its
        ``(replicaId, operationId)`` identity — ordinary writes, stale
        writes that added no candidate, manual and automatic resolutions,
        and sync-imported records alike. Identical replays add no record,
        and conflicting, invalid, or undurably-committed requests never
        enter the identity index, so they stay 404. Returns
        ``(404, {"error": "not_found"})`` for an unknown identity and
        ``(200, {"replicaId", "operation"})`` otherwise, where ``operation``
        carries exactly ``operationId``, ``key``, ``value``, and ``clock``
        with their committed values. With ``--data-file`` the identity
        index is rebuilt identically during recovery, so the same identity
        yields the same response before and after a restart.
        """
        with self._lock:
            operation = self._operations.get((replica_id, operation_id))
            if operation is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            record = {
                "operationId": operation["operationId"],
                "key": operation["key"],
                "value": operation["value"],
                "clock": dict(operation["clock"]),
            }
        return HTTPStatus.OK, {"replicaId": replica_id, "operation": record}

    @staticmethod
    def _strict_predecessors_locked(
        accepted: list[tuple[str, dict[str, Any]]],
        source_replica_id: str,
        source_operation_id: str,
        source_clock: dict[str, int],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Collect one source operation's strict causal predecessors.

        Scans the shared accepted log (already in global commit order) up to
        but excluding the source record itself, keeping every earlier record
        whose clock the source clock strictly dominates. Missing clock
        components count as 0. The source identity is unique in the log, so
        the scan stops at its record; records committed after the source are
        never predecessors even when their clocks are smaller.
        """
        predecessors: list[tuple[str, dict[str, Any]]] = []
        for accepted_replica, accepted_operation in accepted:
            if (
                accepted_replica == source_replica_id
                and accepted_operation["operationId"] == source_operation_id
            ):
                break
            if clock_dominates(source_clock, accepted_operation["clock"]):
                predecessors.append((accepted_replica, accepted_operation))
        return predecessors

    @staticmethod
    def _ancestor_entries_locked(
        predecessors: list[tuple[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Render strict predecessors in the ancestor-entry response shape.

        Each entry preserves the archive record content and adds
        ``relation``: ``"direct"`` when no other strict predecessor's clock
        dominates the record's clock, ``"transitive"`` otherwise. Entries
        keep the shared log's global commit order.
        """
        ancestors: list[dict[str, Any]] = []
        for index, (pred_replica, pred_operation) in enumerate(predecessors):
            dominated = any(
                other_index != index
                and clock_dominates(other_operation["clock"], pred_operation["clock"])
                for other_index, (_, other_operation) in enumerate(predecessors)
            )
            ancestors.append(
                {
                    "replicaId": pred_replica,
                    "operation": {
                        "operationId": pred_operation["operationId"],
                        "key": pred_operation["key"],
                        "value": pred_operation["value"],
                        "clock": dict(pred_operation["clock"]),
                    },
                    "relation": "transitive" if dominated else "direct",
                }
            )
        return ancestors

    @staticmethod
    def _source_record_locked(
        replica_id: str, operation: dict[str, Any]
    ) -> dict[str, Any]:
        """Render one accepted operation in the per-operation archive shape."""
        return {
            "replicaId": replica_id,
            "operation": {
                "operationId": operation["operationId"],
                "key": operation["key"],
                "value": operation["value"],
                "clock": dict(operation["clock"]),
            },
        }

    def get_causal_ancestors(
        self, replica_id: str, operation_id: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of an operation's strict causal predecessors.

        The source operation, the predecessor list, and the paging
        boundaries are all read under the same commit lock used by local
        writes, sync imports, repairs, and checkpoint commits, so the
        response always describes a single commit: a read can never observe
        half an import batch or a partially applied repair. The snapshot
        mutates neither memory, the data file, candidates, metrics, audits,
        checkpoints, nor logs.

        A strict predecessor is a first-accepted record committed **before**
        the source operation in the shared log whose clock is strictly
        smaller than the source operation's clock — the source clock
        dominates it (missing components count as 0, and domination already
        requires the clocks to differ). The source operation itself never
        appears. Stale writes and accepted repairs are ordinary committed
        records and participate like any other; identical replays,
        conflicting or invalid requests, and uncommitted writes never enter
        the log, so they can never appear.

        Each strict predecessor is classified under ``relation``:
        ``"direct"`` when no other strict predecessor's clock dominates its
        own, ``"transitive"`` otherwise.

        Returns ``(404, {"error": "not_found"})`` when the identity was
        never first-accepted. Otherwise returns ``(200, report)`` with
        exactly four fields: ``operation`` (the source record in the
        per-operation archive shape ``{"replicaId", "operation"}``),
        ``ancestors`` (one page of predecessor records in the shared log's
        global commit order, each preserving the archive record content
        plus the ``relation`` field), ``cursor`` (the number of
        predecessors skipped after this page — feed it back as the next
        ``after``), and ``more`` (whether further predecessors remain).
        Raises ValueError when ``after`` is past the predecessor count of
        the snapshot. With ``--data-file`` the log is rebuilt identically
        during recovery, so the same state yields the same source, the same
        relations, and the same pages before and after a restart.
        """
        with self._lock:
            source = self._operations.get((replica_id, operation_id))
            if source is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            predecessors = self._strict_predecessors_locked(
                self._accepted, replica_id, operation_id, source["clock"]
            )
            ancestors = self._ancestor_entries_locked(predecessors)
            total = len(ancestors)
            if after > total:
                raise ValueError("after is past the end of the ancestor list")
            page = ancestors[after : after + limit]
            source_record = self._source_record_locked(replica_id, source)
        cursor = after + len(page)
        return HTTPStatus.OK, {
            "operation": source_record,
            "ancestors": page,
            "cursor": cursor,
            "more": cursor < total,
        }

    def get_causal_comparison(
        self,
        left_replica_id: str,
        left_operation_id: str,
        right_replica_id: str,
        right_operation_id: str,
        after: int,
        limit: int,
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Compare the strict causal predecessor slices of two operations.

        The two source records, their complete predecessor sets, the
        relation, the difference counts, and the paging boundaries are all
        read under the same commit lock used by local writes, sync imports,
        repairs, and checkpoint commits, so the response always describes a
        single commit: a read can never observe half an import batch or a
        partially applied repair. The snapshot mutates neither memory, the
        data file, candidates, metrics, audits, checkpoints, nor logs.

        Each side follows :meth:`get_causal_ancestors` exactly: its
        predecessors are the accepted records committed before that side's
        source in the shared log whose clocks its source clock strictly
        dominates, classified ``"direct"``/``"transitive"`` against that
        side's own predecessor set, and rendered with the archive entry
        shape plus a per-side ``cursor``/``more`` pair. The single
        ``after`` is the number of predecessors skipped on *both* sides —
        both pages start at the same offset — and ``limit`` bounds each
        page. Paging only trims the two pages: the relation and the
        difference counts are always computed from the complete, unpaged
        predecessor sets. A page whose side has no record at the offset
        (including an ``after`` at or past that side's predecessor count)
        is an empty array, and its ``more`` is False.

        Predecessor identity for the difference is the accepted record
        identity ``(replicaId, operationId)``, de-duplicated per side;
        ``difference`` reports ``shared``, ``leftOnly``, and ``rightOnly``
        counts over those identity sets. ``relation`` is
        ``"left_dominates_right"`` when the left source clock dominates the
        right source clock, ``"right_dominates_left"`` for the reverse, and
        ``"concurrent"`` when neither dominates the other (equal clocks
        included).

        Returns ``(404, {"error": "not_found"})`` when either identity was
        never first-accepted. Otherwise returns ``(200, report)``. Raises
        ValueError when ``after`` is past the larger of the two
        predecessor counts of the snapshot. With ``--data-file`` the log is
        rebuilt identically during recovery, so the same state yields the
        same sides, relation, difference counts, and pages before and after
        a restart.
        """
        with self._lock:
            left_source = self._operations.get((left_replica_id, left_operation_id))
            right_source = self._operations.get((right_replica_id, right_operation_id))
            if left_source is None or right_source is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            left_predecessors = self._strict_predecessors_locked(
                self._accepted,
                left_replica_id,
                left_operation_id,
                left_source["clock"],
            )
            right_predecessors = self._strict_predecessors_locked(
                self._accepted,
                right_replica_id,
                right_operation_id,
                right_source["clock"],
            )
            left_ancestors = self._ancestor_entries_locked(left_predecessors)
            right_ancestors = self._ancestor_entries_locked(right_predecessors)
            left_total = len(left_ancestors)
            right_total = len(right_ancestors)
            if after > max(left_total, right_total):
                raise ValueError("after is past the end of the predecessor lists")
            left_page = left_ancestors[after : after + limit]
            right_page = right_ancestors[after : after + limit]
            left_identities = {
                (pred_replica, pred_operation["operationId"])
                for pred_replica, pred_operation in left_predecessors
            }
            right_identities = {
                (pred_replica, pred_operation["operationId"])
                for pred_replica, pred_operation in right_predecessors
            }
            shared = len(left_identities & right_identities)
            left_only = len(left_identities - right_identities)
            right_only = len(right_identities - left_identities)
            if clock_dominates(left_source["clock"], right_source["clock"]):
                relation = "left_dominates_right"
            elif clock_dominates(right_source["clock"], left_source["clock"]):
                relation = "right_dominates_left"
            else:
                relation = "concurrent"
            left_record = self._source_record_locked(left_replica_id, left_source)
            right_record = self._source_record_locked(right_replica_id, right_source)
        left_cursor = after + len(left_page)
        right_cursor = after + len(right_page)
        return HTTPStatus.OK, {
            "left": {
                "operation": left_record,
                "predecessors": left_page,
                "cursor": left_cursor,
                "more": left_cursor < left_total,
            },
            "right": {
                "operation": right_record,
                "predecessors": right_page,
                "cursor": right_cursor,
                "more": right_cursor < right_total,
            },
            "relation": relation,
            "difference": {
                "shared": shared,
                "leftOnly": left_only,
                "rightOnly": right_only,
            },
        }

    def get_causal_diff(
        self,
        left_replica_id: str,
        left_operation_id: str,
        right_replica_id: str,
        right_operation_id: str,
        after: int,
        limit: int,
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return the grouped strict-predecessor diff of two operations.

        Like :meth:`get_causal_comparison`, both source records and their
        complete strict predecessor sets are read under one commit lock, so
        the response always describes a single commit and is strictly
        read-only.

        The de-duplicated predecessor identities are partitioned into three
        evidence groups, each kept in the shared log's global commit order:
        ``shared`` (present on both sides), ``left_only`` (only the left
        source dominates the record), and ``right_only``. Each entry keeps
        the comparison endpoint's predecessor record shape
        (``replicaId``/``operation`` plus a ``relation`` field); the
        ``direct``/``transitive`` relation is computed once against the
        merged evidence union rather than per side, so every identity has a
        single stable classification.

        ``explanation`` is the compressed causal explanation: one entry per
        one-sided predecessor that no other one-sided predecessor on the
        same side dominates — the minimal boundary that still differentiates
        the two sources. Shared evidence and same-side one-sided evidence
        that is itself covered by another one-sided predecessor never
        appear. Each entry carries exactly ``from`` (the boundary
        identity), ``to`` (that side's source identity), and ``side``
        (``"left"`` or ``"right"``); the entries keep global commit order.

        Paging runs over one stable merge of the three groups — shared,
        then left-only, then right-only — and the current window is
        partitioned back into the three groups. ``difference`` counts and
        the explanation are always computed from the complete, unpaged
        predecessor sets. ``after`` equal to the merged predecessor count is
        a valid empty tail.

        Returns ``(404, {"error": "not_found"})`` when either identity was
        never first-accepted. Otherwise returns ``(200, report)``. Raises
        ValueError when ``after`` is past the merged predecessor count of
        the snapshot. With ``--data-file`` the log is rebuilt identically
        during recovery, so the same state yields the same groups,
        difference counts, explanation, and pages before and after a
        restart.
        """
        with self._lock:
            left_source = self._operations.get((left_replica_id, left_operation_id))
            right_source = self._operations.get((right_replica_id, right_operation_id))
            if left_source is None or right_source is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            left_predecessors = self._strict_predecessors_locked(
                self._accepted,
                left_replica_id,
                left_operation_id,
                left_source["clock"],
            )
            right_predecessors = self._strict_predecessors_locked(
                self._accepted,
                right_replica_id,
                right_operation_id,
                right_source["clock"],
            )
            left_identities = {
                (pred_replica, pred_operation["operationId"])
                for pred_replica, pred_operation in left_predecessors
            }
            right_identities = {
                (pred_replica, pred_operation["operationId"])
                for pred_replica, pred_operation in right_predecessors
            }
            # Merge the two sides in the shared log's global commit order.
            # Each predecessor list is a commit-ordered subsequence of the
            # accepted log, so a scan of the log reproduces their identity
            # union in exactly that order.
            union: list[tuple[str, dict[str, Any]]] = [
                (accepted_replica, accepted_operation)
                for accepted_replica, accepted_operation in self._accepted
                if (
                    accepted_replica,
                    accepted_operation["operationId"],
                )
                in left_identities
                or (
                    accepted_replica,
                    accepted_operation["operationId"],
                )
                in right_identities
            ]
            # Classify direct/transitive once against the merged evidence
            # pool, giving every identity one stable relation value.
            union_entries = self._ancestor_entries_locked(union)
            shared: list[dict[str, Any]] = []
            left_only: list[dict[str, Any]] = []
            right_only: list[dict[str, Any]] = []
            left_only_records: list[tuple[str, dict[str, Any]]] = []
            right_only_records: list[tuple[str, dict[str, Any]]] = []
            for record, entry in zip(union, union_entries):
                identity = (record[0], record[1]["operationId"])
                in_left = identity in left_identities
                in_right = identity in right_identities
                if in_left and in_right:
                    shared.append(entry)
                elif in_left:
                    left_only.append(entry)
                    left_only_records.append(record)
                else:
                    right_only.append(entry)
                    right_only_records.append(record)

            # Compressed minimal-boundary explanation. A one-sided
            # predecessor is located only when no other one-sided
            # predecessor on the same side dominates it; walking the union
            # keeps the explanation itself in global commit order.
            explanation: list[dict[str, Any]] = []
            for pred_replica, pred_operation in union:
                identity = (pred_replica, pred_operation["operationId"])
                in_left = identity in left_identities
                in_right = identity in right_identities
                if in_left and in_right:
                    continue
                if in_left:
                    side = "left"
                    group_records = left_only_records
                    target = (left_replica_id, left_operation_id)
                else:
                    side = "right"
                    group_records = right_only_records
                    target = (right_replica_id, right_operation_id)
                covered = any(
                    (other_replica, other_operation["operationId"]) != identity
                    and clock_dominates(other_operation["clock"], pred_operation["clock"])
                    for other_replica, other_operation in group_records
                )
                if not covered:
                    explanation.append(
                        {
                            "from": {
                                "replicaId": pred_replica,
                                "operationId": pred_operation["operationId"],
                            },
                            "to": {
                                "replicaId": target[0],
                                "operationId": target[1],
                            },
                            "side": side,
                        }
                    )

            merged = [*shared, *left_only, *right_only]
            total = len(merged)
            if after > total:
                raise ValueError("after is past the end of the merged predecessor list")
            window = merged[after : after + limit]
            shared_page: list[dict[str, Any]] = []
            left_page: list[dict[str, Any]] = []
            right_page: list[dict[str, Any]] = []
            for entry in window:
                identity = (entry["replicaId"], entry["operation"]["operationId"])
                if identity in left_identities and identity in right_identities:
                    shared_page.append(entry)
                elif identity in left_identities:
                    left_page.append(entry)
                else:
                    right_page.append(entry)
            left_record = self._source_record_locked(left_replica_id, left_source)
            right_record = self._source_record_locked(right_replica_id, right_source)
        cursor = after + len(window)
        return HTTPStatus.OK, {
            "left": {"operation": left_record},
            "right": {"operation": right_record},
            "difference": {
                "shared": len(shared),
                "leftOnly": len(left_only),
                "rightOnly": len(right_only),
            },
            "shared": shared_page,
            "leftOnly": left_page,
            "rightOnly": right_page,
            "explanation": explanation,
            "cursor": cursor,
            "more": cursor < total,
        }

    @staticmethod
    def _strict_descendants_locked(
        accepted: list[tuple[str, dict[str, Any]]],
        source_replica_id: str,
        source_operation_id: str,
        source_clock: dict[str, int],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Collect one source operation's strict causal descendants.

        Scans the shared accepted log (already in global commit order)
        starting just after the source record itself, keeping every later
        record whose clock strictly dominates the source clock. Missing
        clock components count as 0. The source identity is unique in the
        log, so everything before its record is skipped; records committed
        before the source are never descendants even when their clocks are
        larger.
        """
        descendants: list[tuple[str, dict[str, Any]]] = []
        seen_source = False
        for accepted_replica, accepted_operation in accepted:
            if not seen_source:
                if (
                    accepted_replica == source_replica_id
                    and accepted_operation["operationId"] == source_operation_id
                ):
                    seen_source = True
                continue
            if clock_dominates(accepted_operation["clock"], source_clock):
                descendants.append((accepted_replica, accepted_operation))
        return descendants

    def get_causal_descendants(
        self, replica_id: str, operation_id: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of an operation's strict causal descendants.

        The source operation, the descendant list, the relation
        classification, and the paging boundaries are all read under the
        same commit lock used by local writes, sync imports, repairs, and
        checkpoint commits, so the response always describes a single
        commit: a read can never observe half an import batch or a
        partially applied repair. The snapshot mutates neither memory, the
        data file, candidates, metrics, audits, checkpoints, nor logs.

        A strict descendant is a first-accepted record committed **after**
        the source operation in the shared log whose clock strictly
        dominates the source operation's clock (missing components count
        as 0, and domination already requires the clocks to differ). The
        source operation itself never appears. Stale writes that added no
        candidate and accepted repairs are ordinary committed records and
        participate like any other; identical replays, conflicting or
        invalid requests, and uncommitted writes never enter the log, so
        they can never appear.

        Each strict descendant is classified under ``relation``:
        ``"direct"`` when no other strict descendant's clock dominates its
        own, ``"transitive"`` otherwise. The classification is computed
        once against the complete, unpaged descendant set, so paging never
        changes a record's relation.

        Returns ``(404, {"error": "not_found"})`` when the identity was
        never first-accepted. Otherwise returns ``(200, report)`` with
        exactly four fields: ``operation`` (the source record in the
        per-operation archive shape ``{"replicaId", "operation"}``),
        ``descendants`` (one page of descendant records in the shared
        log's global commit order, each preserving the archive record
        content plus the ``relation`` field), ``cursor`` (the number of
        descendants skipped after this page — feed it back as the next
        ``after``), and ``more`` (whether further descendants remain).
        Raises ValueError when ``after`` is past the descendant count of
        the snapshot. With ``--data-file`` the log is rebuilt identically
        during recovery, so the same state yields the same source, the
        same relations, and the same pages before and after a restart.
        """
        with self._lock:
            source = self._operations.get((replica_id, operation_id))
            if source is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            descendants = self._strict_descendants_locked(
                self._accepted, replica_id, operation_id, source["clock"]
            )
            entries = self._ancestor_entries_locked(descendants)
            total = len(entries)
            if after > total:
                raise ValueError("after is past the end of the descendant list")
            page = entries[after : after + limit]
            source_record = self._source_record_locked(replica_id, source)
        cursor = after + len(page)
        return HTTPStatus.OK, {
            "operation": source_record,
            "descendants": page,
            "cursor": cursor,
            "more": cursor < total,
        }

    def get_causal_frontier(
        self, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of the causal frontier from a single snapshot.

        The frontier is the maximal set of first-accepted operations: the
        accepted records whose clock no **other** accepted record's clock
        strictly dominates (missing components count as 0, and domination
        already requires the clocks to differ). Two records whose clocks
        are equal, or that dominate each other in neither direction, both
        stay on the frontier; the operation's source kind is irrelevant —
        ordinary writes, stale writes that added no candidate, sync
        imports, and accepted manual or automatic repairs all participate
        as ordinary committed records, while identical replays, rejected
        (400/409) requests, uncommitted requests, and requests whose
        durable commit failed never enter the log and so never appear.

        The frontier, the page slice, the resume cursor, the remaining
        flag, the complete count, and the digest are all computed under
        the same commit lock used by local writes, sync imports, repairs,
        and checkpoint commits, so the response always describes a single
        commit: a read can never observe half an import batch or a
        partially applied repair. The snapshot mutates neither memory,
        the data file, candidates, metrics, audits, checkpoints, nor
        logs.

        Returns ``(200, report)`` with exactly six fields: ``operations``
        (one page of frontier records in the shared log's global commit
        order, each in the per-operation archive shape
        ``{"replicaId", "operation"}``), ``nextCursor`` (the number of
        frontier records skipped after this page — feed it back as the
        next ``after``), ``hasMore`` (whether further frontier records
        remain), ``algorithm`` (``"sha256"``), ``digest`` (the
        64-character lowercase hexadecimal SHA-256 of the complete
        frontier serialized by :func:`_key_audit_digest_input` — the
        audit chain's canonical record bytes, one compact array element
        per frontier record in frontier order, an empty frontier hashing
        ``[]``), and ``frontierCount`` (the complete frontier size, never
        the page length). An ``after`` equal to the frontier size is a
        valid stable empty page. Raises ValueError when ``after`` is past
        the frontier size of the snapshot. With ``--data-file`` the log
        is rebuilt identically during recovery, so the same state yields
        the same frontier, the same digest, and the same pages before and
        after a restart.
        """
        with self._lock:
            accepted = self._accepted
            frontier: list[tuple[str, dict[str, Any]]] = []
            for index, (replica_id, operation) in enumerate(accepted):
                dominated = any(
                    other_index != index
                    and clock_dominates(other_operation["clock"], operation["clock"])
                    for other_index, (_, other_operation) in enumerate(accepted)
                )
                if not dominated:
                    frontier.append((replica_id, operation))
            total = len(frontier)
            if after > total:
                raise ValueError("after is past the end of the causal frontier")
            digest_input = _key_audit_digest_input(frontier)
            page = [
                self._source_record_locked(replica_id, operation)
                for replica_id, operation in frontier[after : after + limit]
            ]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "operations": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "frontierCount": total,
        }

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

    def get_peer_operations(
        self, peer_id: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of a registered peer's unconsumed operations.

        The pickup stream is the shared accepted-operation log starting
        after the peer's registered checkpoint cursor — the accepted
        operations the peer has not yet consumed — in global commit
        order. The peer only selects the progress anchor: every accepted
        record past the checkpoint is returned whatever its ``replicaId``
        (each item keeps the committed sync record's own replica
        identity), exactly like the sync-export tail beginning at the
        checkpoint. It therefore reads only the shared *accepted* log:
        stale writes (which add no candidate), sync-imported records,
        and manual and automatic repairs are ordinary accepted records
        and appear like any other, while identical replays, rejected
        (400/409) requests, uncommitted requests, and requests whose
        durable commit failed never enter the log and so never appear.

        ``after`` is the number of unconsumed records already skipped
        relative to the checkpoint (a 0-based resume cursor, not an
        absolute log position) and ``limit`` the page size. The
        registered cursor, the slice, the returned cursor, and
        ``has_more`` are all computed against the same snapshot under
        the commit lock, so pages interleave cleanly with concurrent
        commits and never observe half an import batch or a checkpoint
        commit halfway through its durable update. The query never
        advances or writes the checkpoint and mutates neither memory
        nor the data file.

        Returns ``(404, {"error": "not_found"})`` when the peer has never
        registered a checkpoint. Otherwise returns
        ``(200, {"operations", "nextCursor", "hasMore"})`` where
        ``operations`` preserves the committed record shape
        ``{"replicaId", "operation"}`` and ``nextCursor`` is the number
        of unconsumed records skipped after this page (relative to the
        checkpoint) — feed it back as the next ``after``. An ``after``
        equal to the unconsumed record count is a valid empty tail.
        Raises ValueError when ``after`` is past that count of the
        snapshot. With ``--data-file`` the checkpoints and the log are
        rebuilt identically during recovery, so the same recovered state
        yields the same pages before and after a restart; a recovered
        cursor past the log length is a corrupt file and makes startup
        fail rather than reaching here.
        """
        with self._lock:
            start = self._checkpoints.get(peer_id)
            if start is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            total = len(self._accepted) - start
            if after > total:
                raise ValueError("after is past the end of the peer's unconsumed log")
            page = [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted[
                    start + after : start + after + limit
                ]
            ]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "operations": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
        }

    def get_peer_receipts(
        self, peer_id: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of a registered peer's consumption receipts.

        The receipt stream is the peer's committed ``(peerId, ackId)``
        bindings in commit (creation) order — the order the
        acknowledgements were accepted. Each item preserves the receipt's
        confirmation-time ``cursor`` and the confirmed identities in their
        confirmation order, each identity carrying exactly ``replicaId``
        and ``operationId``.

        ``after`` is the number of the peer's receipts already skipped (a
        0-based resume cursor) and ``limit`` the page size. The registered
        checkpoint, the receipt list, the page slice, the returned cursor,
        ``hasMore``, and the integrity summary are all computed against
        the same snapshot under the commit lock, so the page and the
        summary always describe a single commit even while
        acknowledgements are being committed. The query never advances or
        writes the checkpoint, records no receipt, and mutates neither
        memory nor the data file.

        The summary covers the peer's **whole** committed receipt set,
        never just the page: ``digest`` is the 64-character lowercase
        hexadecimal SHA-256 of the canonical bytes produced by
        :func:`_receipts_digest_input` (an empty set hashes ``[]``), and
        ``receiptsCount`` counts committed receipts, not page items.

        Returns ``(404, {"error": "not_found"})`` when the peer has never
        registered a checkpoint. Otherwise returns ``(200, report)`` with
        exactly six fields: ``receipts`` (the page), ``nextCursor`` (the
        number of receipts skipped after this page — feed it back as the
        next ``after``), ``hasMore`` (whether further receipts remain),
        ``algorithm`` (``"sha256"``), ``digest``, and ``receiptsCount``.
        An ``after`` equal to the receipt count is a valid empty tail.
        Raises ValueError when ``after`` is past the receipt count of the
        snapshot. With ``--data-file`` the receipts are rebuilt
        identically during recovery (a file written before receipts
        existed recovers with an empty set), so the same state yields the
        same order, pages, and summary before and after a restart.
        """
        with self._lock:
            if peer_id not in self._checkpoints:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            committed = [
                (ack_id, receipt)
                for (receipt_peer, ack_id), receipt in self._acks.items()
                if receipt_peer == peer_id
            ]
            total = len(committed)
            if after > total:
                raise ValueError("after is past the end of the peer's receipts")
            digest_input = _receipts_digest_input(peer_id, committed)
            page = [
                {
                    "peerId": peer_id,
                    "ackId": ack_id,
                    "cursor": receipt["cursor"],
                    "operations": [dict(identity) for identity in receipt["operations"]],
                }
                for ack_id, receipt in committed[after : after + limit]
            ]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "receipts": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "receiptsCount": total,
        }

    def get_peer_receipts_audit(
        self, peer_id: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of a registered peer's receipts with a chain audit.

        This is the sender-side audit view over the peer's whole
        confirmation chain: it pages the same receipt stream as
        :meth:`get_peer_receipts` — the peer's committed receipts in
        creation order, under the same paging rules — but adds an
        ``audit`` conclusion covering the segment of the shared accepted
        log from the earliest receipt's derived start through the last
        confirmation cursor.

        As with the plain receipts query, ``after`` is the number of the
        peer's receipts already skipped (a 0-based resume cursor) and
        ``limit`` the page size. Paging trims only the exported receipt
        page: the ``audit`` conclusion, the ``digest``, and
        ``receiptsCount`` are always computed from the peer's complete
        receipt history and the full accepted log on one snapshot under
        the commit lock, so every page carries the same summary and
        conclusion even while acknowledgements are being committed. The
        query never advances or writes the checkpoint, records no
        receipt, and mutates neither memory nor the data file.

        Returns ``(404, {"error": "not_found"})`` when the peer has never
        registered a checkpoint. Otherwise returns ``(200, report)`` with
        exactly seven fields: the six of :meth:`get_peer_receipts`
        (``receipts``, ``nextCursor``, ``hasMore``, ``algorithm``,
        ``digest``, ``receiptsCount`` — the digest follows the same
        canonical encoding over **all** of the peer's receipts, an empty
        set hashing ``[]``) plus ``audit``, whose structure is produced by
        :func:`_receipt_chain_audit_locked`. An ``after`` equal to the
        receipt count is a valid stable empty page whose audit still
        covers the complete history; an empty receipt set reports a
        complete, anomaly-free empty coverage (``{"start": 0, "end":
        0}``, ``"ok"``). Raises ValueError when ``after`` is past the
        receipt count of the snapshot.
        """
        with self._lock:
            if peer_id not in self._checkpoints:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            committed = [
                (ack_id, receipt)
                for (receipt_peer, ack_id), receipt in self._acks.items()
                if receipt_peer == peer_id
            ]
            total = len(committed)
            if after > total:
                raise ValueError("after is past the end of the peer's receipts")
            digest_input = _receipts_digest_input(peer_id, committed)
            audit = _receipt_chain_audit_locked(committed, self._accepted)
            page = [
                {
                    "peerId": peer_id,
                    "ackId": ack_id,
                    "cursor": receipt["cursor"],
                    "operations": [dict(identity) for identity in receipt["operations"]],
                }
                for ack_id, receipt in committed[after : after + limit]
            ]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "receipts": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "receiptsCount": total,
            "audit": audit,
        }

    def get_replication_status(
        self, peer_id: str
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one registered peer's sender-side delivery status.

        The peer's registered checkpoint cursor, the count of accepted
        records the peer has not yet consumed, the peer's committed
        receipt count, and the receipt chain-audit conclusion are all
        read together under the same commit lock used by local writes,
        sync imports, repairs, checkpoint commits, and acknowledgement
        commits, so the progress, the counts, and the conclusion always
        describe a single commit even while commits are in flight. The
        query is strictly read-only: it never advances or writes the
        checkpoint, records no receipt, and mutates neither memory nor
        the data file.

        Returns ``(404, {"error": "not_found"})`` when the peer has never
        registered a checkpoint. Otherwise returns ``(200, status)``
        with exactly five fields, in this order:

        - ``peer``: the decoded peer id the request selected.
        - ``pos``: the peer's registered checkpoint cursor — the number
          of accepted records the peer has consumed.
        - ``left``: the number of accepted records past the checkpoint
          the peer has not yet consumed.
        - ``acks``: the number of the peer's committed receipts.
        - ``chain``: the chain-audit conclusion produced by
          :func:`_receipt_chain_audit_locked` over the peer's whole
          committed receipt set — the status, the coverage interval, and
          the gap, overlap, identity-mismatch, and cursor-regression
          lists. An empty receipt set reports a complete, anomaly-free
          empty coverage (``{"start": 0, "end": 0}``) with status
          ``"ok"``.

        With ``--data-file`` the log, the checkpoints, and the receipts
        are rebuilt identically during recovery, so the same state
        yields the same status before and after a restart.
        """
        with self._lock:
            cursor = self._checkpoints.get(peer_id)
            if cursor is None:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            committed = [
                (ack_id, receipt)
                for (receipt_peer, ack_id), receipt in self._acks.items()
                if receipt_peer == peer_id
            ]
            left = len(self._accepted) - cursor
            chain = _receipt_chain_audit_locked(committed, self._accepted)
        return HTTPStatus.OK, {
            "peer": peer_id,
            "pos": cursor,
            "left": left,
            "acks": len(committed),
            "chain": chain,
        }

    def record_policy_reload(self, digest: str, tokens: int) -> dict[str, Any]:
        """Commit one successful scope-policy hot reload to the audit history.

        Appends an event ``{"sequence", "digest", "tokens"}`` — the next
        1-based position, the 64-character lowercase SHA-256 of the
        reloaded policy file's raw UTF-8 bytes, and the new policy's entry
        count — in the reload's commit order. With ``--data-file`` the
        event is made durable by the same atomic commit protocol as
        business state *before* it becomes visible: a durable failure
        raises PersistenceError and leaves the in-memory history exactly
        as it was, so a caller can answer HTTP 500 while both the old
        policy and the old history stay in force. Returns the committed
        event.
        """
        with self._lock:
            event = {
                "sequence": len(self._policy_events) + 1,
                "digest": digest,
                "tokens": tokens,
            }
            self._policy_events.append(event)
            if self._data_file is not None:
                try:
                    self._persist_locked()
                except BaseException:
                    self._policy_events.pop()
                    raise
            return dict(event)

    def get_policy_events(
        self, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one page of the scope-policy change history.

        The stream is the history of successful hot reloads in their
        commit order — one event per reload that atomically replaced the
        live policy; failed reloads (409/503) and rejected requests never
        enter it. ``after`` is the number of events already skipped (a
        0-based resume cursor) and ``limit`` the page size. The page
        slice, ``nextCursor``, ``hasMore``, the full-history digest, and
        the count are all computed against the same snapshot under the
        commit lock, so a concurrent reload is observed only as the whole
        old or the whole new history. The query is strictly read-only.

        Returns ``(200, report)`` with exactly six fields: ``events`` (the
        page, each item carrying exactly ``sequence``, ``digest``, and
        ``tokens``), ``nextCursor`` (the number of events skipped after
        this page — feed it back as the next ``after``), ``hasMore``,
        ``algorithm`` (``"sha256"``), ``digest`` (the SHA-256 of the
        canonical compact JSON array produced by
        :func:`_policy_events_digest_input` over the **whole** history, an
        empty history hashing ``[]``), and ``eventsCount`` (the full
        history length, not the page length). An ``after`` equal to the
        event count is a valid empty tail. Raises ValueError when
        ``after`` is past the event count of the snapshot. With
        ``--data-file`` the history is rebuilt identically during
        recovery (a file written before policy events existed recovers
        with an empty set).
        """
        with self._lock:
            total = len(self._policy_events)
            if after > total:
                raise ValueError("after is past the end of the policy event history")
            digest_input = _policy_events_digest_input(self._policy_events)
            page = [dict(event) for event in self._policy_events[after : after + limit]]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "events": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "eventsCount": total,
        }

    def get_policy_events_verify(
        self, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return one policy-history page plus an independent verification.

        This is the integrity-verification companion to
        :meth:`get_policy_events`. The event page, the resume cursor, the
        remaining-event flag, the full-history digest, the algorithm, and
        the complete event count are produced exactly as for the plain
        change-audit query; the only addition is ``verification``, the
        independent integrity conclusion produced by
        :func:`_policy_events_verification_locked`.

        ``after`` is the number of successful events already skipped (a
        0-based resume cursor starting at ``0``) and ``limit`` the page
        size. Paging trims only the exported ``events`` page: the digest,
        ``eventsCount``, and the ``verification`` conclusion always cover
        the complete history, an empty history hashing ``[]`` and
        verifying as intact. The page slice, cursors, digest, count, and
        conclusion are computed from one snapshot under the commit lock,
        so a concurrent hot reload is observed only as the whole old or
        the whole new history; the query is strictly read-only and
        creates no temporary file.

        Returns ``(200, report)`` with exactly seven fields: ``events``,
        ``nextCursor``, ``hasMore``, ``algorithm``, ``digest``,
        ``eventsCount`` (identical in meaning to
        :meth:`get_policy_events`), and ``verification``. An ``after``
        equal to the event count is a valid stable empty page whose
        verification still covers the complete history. Raises ValueError
        when ``after`` is past the event count of the snapshot. With
        ``--data-file`` the history is rebuilt identically during
        recovery, so the same events, digest, count, and verification
        conclusion are reported before and after a restart.
        """
        with self._lock:
            total = len(self._policy_events)
            if after > total:
                raise ValueError("after is past the end of the policy event history")
            digest_input = _policy_events_digest_input(self._policy_events)
            verification = _policy_events_verification_locked(self._policy_events)
            page = [dict(event) for event in self._policy_events[after : after + limit]]
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "events": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
            "algorithm": "sha256",
            "digest": hashlib.sha256(digest_input).hexdigest(),
            "eventsCount": total,
            "verification": verification,
        }

    def get_state_explanation(self, key: str) -> tuple[HTTPStatus, dict[str, Any]]:
        """Explain one key's current candidate state from a single snapshot.

        The candidate set is copied under the same commit lock used by local
        writes, sync imports, repairs, and checkpoint commits, so the
        response always describes one commit: a read can never observe half
        an import batch or a partially applied repair. The snapshot mutates
        neither memory, the data file, logs, nor checkpoints, and the
        explanation creates no repair operation, log record, or checkpoint.

        Returns ``(404, {"error": "not_found"})`` when the key holds no
        current candidates — whether it never appeared or its history leaves
        no current candidate. Otherwise returns ``(200, explanation)`` with
        exactly five categories of information:

        - ``key``: the requested key.
        - ``status``: ``"resolved"`` when every current candidate agrees on
          the value, ``"conflict"`` otherwise — the same classification as
          :meth:`get_state`.
        - ``candidates``: the current candidates in the existing query
          order (sorted by ``(replicaId, operationId)`` ascending), each
          carrying exactly ``value``, ``clock``, ``replicaId``, and
          ``operationId``.
        - ``relations``: one entry per unordered pair of current
          candidates, enumerated in candidate order. Each entry names the
          pair's endpoints as ``{"replicaId", "operationId"}`` identities
          under ``from``/``to`` and classifies the pair under ``relation``:
          ``"dominates"`` when one candidate's clock dominates the other's
          (``from`` dominates ``to``; current candidates never dominate
          each other, so the kind completes the vocabulary without being
          emitted), ``"overwrites"`` when the two candidates hold the same
          value (either covers the other, so the pair cannot conflict —
          for a resolved key these entries report the agreed value's
          unique source relation), and ``"concurrent"`` when the values
          differ and neither clock dominates the other, which is exactly
          why the pair does not dominate each other. A key with a single
          candidate yields an empty list.
        - ``suggestion``: ``{"lowest_identity": C, "highest_identity": C}``
          reporting which current candidate each of the two existing
          automatic-resolution policies would select — the smallest and
          largest ``(replicaId, operationId)`` — in the same shape as the
          ``candidates`` entries.

        With ``--data-file`` the candidate state is rebuilt identically
        during recovery, so the same state yields the same relations,
        sources, and policy suggestions before and after a restart.
        """
        with self._lock:
            ordered = sorted(
                self._candidates.get(key, []),
                key=lambda c: (c["replicaId"], c["operationId"]),
            )
            candidates = [
                {
                    "value": c["value"],
                    "clock": dict(c["clock"]),
                    "replicaId": c["replicaId"],
                    "operationId": c["operationId"],
                }
                for c in ordered
            ]
        if not candidates:
            return HTTPStatus.NOT_FOUND, {"error": "not_found"}

        relations: list[dict[str, Any]] = []
        for index, first in enumerate(candidates):
            first_identity = {
                "replicaId": first["replicaId"],
                "operationId": first["operationId"],
            }
            for second in candidates[index + 1 :]:
                second_identity = {
                    "replicaId": second["replicaId"],
                    "operationId": second["operationId"],
                }
                # Fixed classification: an equal value always reports
                # "overwrites" (either side covers the other, so the pair
                # cannot conflict); a dominating clock reports "dominates";
                # only different values under mutually non-dominating clocks
                # report "concurrent". In a mixed conflict a pair with
                # different values and non-dominating clocks is therefore
                # never misreported as "overwrites" or "dominates".
                if first["value"] == second["value"]:
                    relations.append(
                        {
                            "from": first_identity,
                            "to": second_identity,
                            "relation": "overwrites",
                        }
                    )
                elif clock_dominates(first["clock"], second["clock"]):
                    relations.append(
                        {
                            "from": first_identity,
                            "to": second_identity,
                            "relation": "dominates",
                        }
                    )
                elif clock_dominates(second["clock"], first["clock"]):
                    relations.append(
                        {
                            "from": second_identity,
                            "to": first_identity,
                            "relation": "dominates",
                        }
                    )
                else:
                    relations.append(
                        {
                            "from": first_identity,
                            "to": second_identity,
                            "relation": "concurrent",
                        }
                    )

        agreed = all(c["value"] == candidates[0]["value"] for c in candidates)
        return HTTPStatus.OK, {
            "key": key,
            "status": "resolved" if agreed else "conflict",
            "candidates": candidates,
            "relations": relations,
            "suggestion": {
                "lowest_identity": candidates[0],
                "highest_identity": candidates[-1],
            },
        }

    def get_state_impact(
        self, key: str, after: int, limit: int
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Report the cross-key causal impact on one key from one snapshot.

        The candidate set and the accepted-operation log are read under the
        same commit lock used by local writes, sync imports, repairs, and
        checkpoint commits, so the impact list, the status, and the paging
        boundaries always describe a single commit: a read can never observe
        half an import batch or a partially applied repair. The snapshot
        mutates neither memory, the data file, logs, nor checkpoints, and
        the query creates no operation, record, or file.

        Returns ``(404, {"error": "not_found"})`` when the key holds no
        current candidates. Otherwise returns ``(200, report)`` with exactly
        six categories of information:

        - ``key``: the requested key.
        - ``status``: ``"resolved"`` when every current candidate agrees on
          the value, ``"conflict"`` otherwise — the same classification as
          :meth:`get_state`.
        - ``candidates``: the current candidates the judgement is based on,
          in the existing query order (sorted by ``(replicaId, operationId)``
          ascending), each carrying exactly ``value``, ``clock``,
          ``replicaId``, and ``operationId``.
        - ``impacts``: one page of the accepted operations on *other* keys
          whose clocks dominate at least one of those candidates (i.e. the
          operations causally later than the key's current state), kept in
          the shared log's global commit order. Each entry preserves the
          committed record shape ``{"replicaId", "operation"}`` with the
          operation's identity, key, value, and clock. The key's own
          operations never appear, and identical replays, rejected requests,
          and uncommitted writes never enter the accepted log, so they can
          never appear either.
        - ``nextCursor``: the number of impact records skipped after this
          page — feed it back as the next ``after``.
        - ``hasMore``: whether further impact records remain.

        ``after`` is the number of impact records already skipped (a 0-based
        resume cursor) and ``limit`` the page size. Raises ValueError when
        ``after`` is past the impact record count of the snapshot. With
        ``--data-file`` the candidate state and the log are rebuilt
        identically during recovery, so the same state yields the same
        impact report before and after a restart.
        """
        with self._lock:
            ordered = sorted(
                self._candidates.get(key, []),
                key=lambda c: (c["replicaId"], c["operationId"]),
            )
            if not ordered:
                return HTTPStatus.NOT_FOUND, {"error": "not_found"}
            candidates = [
                {
                    "value": c["value"],
                    "clock": dict(c["clock"]),
                    "replicaId": c["replicaId"],
                    "operationId": c["operationId"],
                }
                for c in ordered
            ]
            basis_clocks = [c["clock"] for c in ordered]
            impacts = [
                {"replicaId": replica_id, "operation": operation}
                for replica_id, operation in self._accepted
                if operation["key"] != key
                and any(
                    clock_dominates(operation["clock"], basis) for basis in basis_clocks
                )
            ]
            total = len(impacts)
            if after > total:
                raise ValueError("after is past the end of the impact list")
            page = impacts[after : after + limit]
        agreed = all(c["value"] == candidates[0]["value"] for c in candidates)
        next_cursor = after + len(page)
        return HTTPStatus.OK, {
            "key": key,
            "status": "resolved" if agreed else "conflict",
            "candidates": candidates,
            "impacts": page,
            "nextCursor": next_cursor,
            "hasMore": next_cursor < total,
        }

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

    def get_state_at(self, key: str, cursor: int) -> tuple[HTTPStatus, dict[str, Any]]:
        """Replay the accepted log up to ``cursor`` and report one key's state.

        ``cursor`` is the number of accepted-log records replayed from the
        empty state, so 0 is always the empty state and the log length is
        exactly the current state. The replay covers every first-accepted
        record in commit order — ordinary writes, stale writes whose clock
        was already dominated, accepted conflict repairs, and sync-imported
        records — and nothing else, because identical replays, conflicting
        or malformed requests, uncommitted requests, and failed durable
        commits never enter the log. The whole replay runs against one
        committed snapshot under the commit lock, so a concurrent commit
        is either fully below or fully above the answered position; the
        read mutates neither memory, the data file, logs, nor checkpoints
        and creates no files.

        Returns ``(404, {"error": "not_found"})`` when the key holds no
        candidate at that position — whether it never appeared or appears
        only later in the log. Otherwise returns ``(200, report)`` with
        exactly four fields: ``cursor`` (the replayed position as a JSON
        integer), ``key``, ``status`` (``"resolved"`` when every candidate
        at that position agrees on the value, ``"conflict"`` otherwise —
        the same classification as :meth:`get_state`), and ``candidates``
        (always an array, even when resolved: every candidate still
        present at that position in the existing query order, sorted by
        ``(replicaId, operationId)`` ascending, each carrying exactly
        ``value``, ``clock``, ``replicaId``, and ``operationId``). The
        first candidate is therefore the same value and clock the current
        state query would choose for the same candidate set.

        Raises ValueError when ``cursor`` is past the accepted-log length.
        With ``--data-file`` the log is recovered identically on restart,
        so the same cursor answers the same report before and after.
        """
        with self._lock:
            total = len(self._accepted)
            if cursor > total:
                raise ValueError("cursor is past the end of the operation log")
            candidates: list[dict[str, Any]] = []
            for replica_id, operation in self._accepted[:cursor]:
                if operation["key"] == key:
                    candidates = self._next_candidates(candidates, replica_id, operation)
            ordered = sorted(candidates, key=lambda c: (c["replicaId"], c["operationId"]))
            present = [
                {
                    "value": c["value"],
                    "clock": dict(c["clock"]),
                    "replicaId": c["replicaId"],
                    "operationId": c["operationId"],
                }
                for c in ordered
            ]
        if not present:
            return HTTPStatus.NOT_FOUND, {"error": "not_found"}
        status = "resolved" if all(c["value"] == present[0]["value"] for c in present) else "conflict"
        return HTTPStatus.OK, {
            "cursor": cursor,
            "key": key,
            "status": status,
            "candidates": present,
        }

    def get_state_causal_at(
        self, key: str, boundary: dict[str, int]
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        """Replay the accepted log up to a vector-clock boundary.

        Unlike :meth:`get_state_at`, the slice is selected causally rather
        than by a global log position: starting from the empty state, only
        first-accepted records whose clock is componentwise no greater
        than ``boundary`` (missing components count as 0) are replayed in
        global commit order. The empty clock is the causal origin: only
        operations already covered by it can apply. Every first-accepted
        record participates in exactly the same way as a position replay
        — ordinary writes, stale writes whose clock was already dominated
        (they still add no candidate), sync-imported records, and accepted
        repairs — because identical replays, conflicting or malformed
        requests, uncommitted requests, and failed durable commits never
        enter the log. Candidate add/delete keeps the existing
        vector-clock domination semantics via
        :meth:`_next_candidates`.

        The whole replay runs against one committed snapshot under the
        commit lock, and the read mutates neither memory, the data file,
        logs, nor checkpoints and creates no files.

        Returns ``(404, {"error": "not_found"})`` when the key holds no
        candidate within the boundary — even when the key appears only
        past it. Otherwise returns ``(200, report)`` with exactly four
        fields: ``clock`` (the requested boundary, echoed back), ``key``,
        ``status`` (``"resolved"`` when every replayed candidate agrees
        on the value, ``"conflict"`` otherwise), and ``candidates``
        (always an array, sorted by ``(replicaId, operationId)``
        ascending, each carrying exactly ``value``, ``clock``,
        ``replicaId``, and ``operationId``).
        """
        with self._lock:
            candidates: list[dict[str, Any]] = []
            for replica_id, operation in self._accepted:
                if operation["key"] != key:
                    continue
                clock = operation["clock"]
                if any(tick > boundary.get(component, 0) for component, tick in clock.items()):
                    continue
                candidates = self._next_candidates(candidates, replica_id, operation)
            ordered = sorted(candidates, key=lambda c: (c["replicaId"], c["operationId"]))
            present = [
                {
                    "value": c["value"],
                    "clock": dict(c["clock"]),
                    "replicaId": c["replicaId"],
                    "operationId": c["operationId"],
                }
                for c in ordered
            ]
        if not present:
            return HTTPStatus.NOT_FOUND, {"error": "not_found"}
        status = "resolved" if all(c["value"] == present[0]["value"] for c in present) else "conflict"
        return HTTPStatus.OK, {
            "clock": dict(boundary),
            "key": key,
            "status": status,
            "candidates": present,
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
        auth_scopes: dict[str, frozenset[str]] | None = None,
        scope_policy_file: str | None = None,
    ) -> None:
        # Build (and thus preflight/recover) the store before binding and
        # listening, so a rejected data file fails startup before any port is
        # open rather than surfacing on the first accepted write.
        resolved_store = store if store is not None else StateStore(data_file=data_file)
        super().__init__(server_address, handler_class or RequestHandler)
        self.store = resolved_store
        # The bearer token clients must present, or None when authentication
        # is disabled. It is never written to the data file or any log.
        self.auth_token = auth_token
        # The scope policy: configured tokens mapped to their allowed scopes.
        # Exactly one of auth_token and auth_scopes is set; both are None only
        # when authentication is disabled. Like the single token, the policy
        # is never written to the data file or any log.
        self.auth_scopes = auth_scopes
        # The manager owns the live policy and its startup-configured file so
        # the admin reload endpoint can atomically swap the mapping without a
        # restart. It exists only in scope-policy mode; None means the
        # endpoint is not published (single-token mode and anonymous mode
        # answer it with 404).
        self.scope_policy = (
            ScopePolicyManager(scope_policy_file, auth_scopes)
            if auth_scopes is not None
            else None
        )


_FALLBACK_STORE = StateStore()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticStateEngine/0.1"

    @property
    def _store(self) -> StateStore:
        return getattr(self.server, "store", _FALLBACK_STORE)

    def _json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json_newline(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Respond with compact JSON terminated by one newline.

        Same compact encoding as :meth:`_json`, with a single trailing
        ``\\n`` included in both the body and its declared length; used by
        the batch endpoint whose contract requires the line terminator.
        """
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json_ordered(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Respond with compact JSON in the payload's field order.

        Same compact encoding as :meth:`_json` but object fields keep their
        insertion order rather than being sorted, for endpoints whose
        contract fixes the response field order.
        """
        body = _ordered_json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self) -> tuple[bool, frozenset[str] | None]:
        """Authenticate the request's bearer credential when auth is enabled.

        Returns ``(True, scopes)`` when the request may proceed: with
        authentication disabled the scopes are ``None`` (every request is
        allowed); with the single-token configuration they are all three
        scopes (preserving the legacy token's unrestricted access); with a
        scope policy they are the configured token's scope set. Otherwise
        sends HTTP 401 with ``{"error": "unauthorized"}`` and the
        ``WWW-Authenticate: Bearer`` challenge and returns ``(False, None)``.

        The request must carry exactly one ``Authorization`` header whose
        value is exactly ``Bearer `` (one space) followed by one configured
        token: a missing, duplicated, or malformed header and any token
        mismatch are 401. Token comparison uses the standard library's
        constant-time primitive (the presented token is compared against
        each configured token), the request body is never read here, and
        neither memory nor the data file is touched.
        """
        token = getattr(self.server, "auth_token", None)
        manager = getattr(self.server, "scope_policy", None)
        if token is None and manager is None:
            return True, None
        values = self.headers.get_all("Authorization")
        if values is not None and len(values) == 1:
            presented = values[0]
            if presented.startswith("Bearer "):
                credential = presented[len("Bearer "):].encode("utf-8")
                if token is not None:
                    # Legacy single-token mode: the one token keeps its full,
                    # unrestricted access — the configured scope boundary is
                    # only enforced in the scope-policy mode.
                    if hmac.compare_digest(credential, token.encode("ascii")):
                        return True, frozenset(ALLOWED_SCOPES)
                else:
                    # Take one point-in-time copy of the live policy so the
                    # whole authentication runs against a single committed
                    # revision even if a reload swaps the mapping meanwhile:
                    # a request is authorized entirely by the policy in force
                    # when it authenticated, never by a later revision.
                    policy = manager.snapshot()
                    for configured_token, granted_scopes in policy.items():
                        if hmac.compare_digest(credential, configured_token.encode("ascii")):
                            return True, granted_scopes
        self.close_connection = True
        self._json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "unauthorized"},
            {"WWW-Authenticate": "Bearer"},
        )
        return False, None

    def _require_scope(self, required: str) -> bool:
        """Enforce authentication and the ``required`` scope for this request.

        Returns True when the request may proceed (authentication disabled,
        the legacy single token, or a policy token carrying ``required`` or
        the ``admin`` scope). An unauthenticated request gets the 401
        response from :meth:`_authenticate`; an authenticated token lacking
        the scope gets HTTP 403 with ``{"error": "forbidden"}`` and **no**
        ``WWW-Authenticate`` challenge. Either rejection closes the
        connection, never reads the request body, and runs before route
        matching, query parsing, the commit lock, and any state access.
        """
        authenticated, scopes = self._authenticate()
        if not authenticated:
            return False
        if scopes is None or required in scopes or SCOPE_ADMIN in scopes:
            return True
        self.close_connection = True
        self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        return False

    def _path_segments(self) -> list[str]:
        path = urlsplit(self.path).path
        segments = [unquote(segment) for segment in path.split("/") if segment != ""]
        if path != "/" and path.endswith("/"):
            # A trailing slash on a published path is a missing/extra
            # segment boundary, not an alias for the bare path: keep an
            # empty final segment so no route shape matches and the
            # request falls through to 404 instead of being served.
            segments.append("")
        return segments

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

    def _peer_operations_route(self) -> tuple[bool, str]:
        """Match ``/v1/sync/peers/{peerId}/operations`` on the raw path.

        Returns ``(matched, peer_id)``. As with :meth:`_checkpoint_route`,
        the empty segment of ``/v1/sync/peers//operations`` is preserved
        so the shape still matches and yields an empty ``peer_id``; the
        pickup contract treats an empty peer id exactly like a shape
        failure (404), so the handler rejects it before any query check.
        Any other segment count — missing segments, extra segments such
        as ``.../operations/extra``, or a trailing slash — falls through
        to the generic 404.
        """
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 6:
            return False, ""
        if parts[1:4] != ["v1", "sync", "peers"] or parts[5] != "operations":
            return False, ""
        return True, unquote(parts[4])

    def _acknowledge_route(self) -> tuple[bool, str]:
        """Match ``/v1/sync/peers/{peerId}/acknowledge`` on the raw path.

        Returns ``(matched, peer_id)``. As with :meth:`_checkpoint_route`,
        the empty segment of ``/v1/sync/peers//acknowledge`` is preserved
        so the shape still matches and yields an empty ``peer_id``; the
        acknowledge contract treats an empty peer id exactly like a shape
        failure (404), so the handler rejects it before any query check.
        Any other segment count — missing segments, extra segments such as
        ``.../acknowledge/extra``, or a trailing slash — falls through to
        the generic 404.
        """
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 6:
            return False, ""
        if parts[1:4] != ["v1", "sync", "peers"] or parts[5] != "acknowledge":
            return False, ""
        return True, unquote(parts[4])

    def _receipts_route(self) -> tuple[bool, str]:
        """Match ``/v1/sync/peers/{peerId}/receipts`` on the raw path.

        Returns ``(matched, peer_id)``. As with :meth:`_checkpoint_route`,
        the empty segment of ``/v1/sync/peers//receipts`` is preserved so
        the shape still matches and yields an empty ``peer_id``; the
        receipts contract treats an empty peer id exactly like a shape
        failure (404), so the handler rejects it before any query check.
        Any other segment count — missing segments, extra segments such as
        ``.../receipts/extra``, or a trailing slash — falls through to the
        generic 404.
        """
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 6:
            return False, ""
        if parts[1:4] != ["v1", "sync", "peers"] or parts[5] != "receipts":
            return False, ""
        return True, unquote(parts[4])

    def _receipts_audit_route(self) -> tuple[bool, str]:
        """Match ``/v1/sync/peers/{peerId}/receipts/audit`` on the raw path.

        Returns ``(matched, peer_id)``. As with :meth:`_receipts_route`,
        the empty segment of ``/v1/sync/peers//receipts/audit`` is
        preserved so the shape still matches and yields an empty
        ``peer_id``; the audit contract treats an empty peer id exactly
        like a shape failure (404), so the handler rejects it before any
        query check. Any other segment count — missing segments, extra
        segments such as ``.../receipts/audit/extra``, or a trailing
        slash — falls through to the generic 404.
        """
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 7:
            return False, ""
        if parts[1:4] != ["v1", "sync", "peers"] or parts[5:7] != [
            "receipts",
            "audit",
        ]:
            return False, ""
        return True, unquote(parts[4])

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            # The health probe stays anonymous even when auth is enabled.
            self._json(HTTPStatus.OK, health_payload())
            return
        segments = self._path_segments()
        is_scope_policy_audit_verify_get = (
            len(segments) == 5
            and segments[0] == "v1"
            and segments[1] == "admin"
            and segments[2] == "scope-policy"
            and segments[3] == "audit"
            and segments[4] == "verify"
        )
        is_scope_policy_audit_get = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "admin"
            and segments[2] == "scope-policy"
            and segments[3] == "audit"
        )
        if is_scope_policy_audit_verify_get or is_scope_policy_audit_get:
            # Both change-audit entries are admin-gated, like the reload
            # endpoint, and exist only in scope-policy mode. Authentication
            # still runs before the mode gate, so a missing or bad
            # credential is 401 in every mode, a valid token without the
            # admin scope is 403 without a challenge, and single-token plus
            # anonymous modes answer 404 exactly like an unpublished route.
            # Path-shape mismatches never reach here, so they still fall
            # through to the generic 404 below.
            if not self._require_scope(SCOPE_ADMIN):
                return
            if getattr(self.server, "scope_policy", None) is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if is_scope_policy_audit_verify_get:
                self._handle_scope_policy_audit_verify_get()
            else:
                self._handle_scope_policy_audit_get()
            return
        # Every other route — known or unknown — authenticates and checks
        # the read scope before route matching, query parsing, or any state
        # access; the admin scope covers reads as well.
        if not self._require_scope(SCOPE_READ):
            return
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
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "replication"
            and segments[2] == "snapshot"
        ):
            self._handle_replication_snapshot_get()
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "replication"
            and segments[2] == "status"
        ):
            self._handle_replication_status_get()
            return
        if len(segments) == 3 and segments[0] == "v1" and segments[1] == "states":
            status, payload = self._store.get_state(segments[2])
            self._json(status, payload)
            return
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "at"
        ):
            self._handle_state_at_get(segments[2])
            return
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "why"
        ):
            self._handle_state_why_get(segments[2])
            return
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "impact"
        ):
            self._handle_state_impact_get(segments[2])
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
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "audit"
            and segments[2] == "log"
            and segments[3] == "chain"
        ):
            self._handle_audit_chain_get()
            return
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "audit"
            and segments[2] == "log"
            and segments[3] == "verify"
        ):
            self._handle_audit_log_verify_get()
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
        if (
            len(segments) == 5
            and segments[0] == "v1"
            and segments[1] == "replicas"
            and segments[3] == "operations"
        ):
            self._handle_operation_archive_get(segments[2], segments[4])
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "causal"
            and segments[2] == "compare"
        ):
            self._handle_causal_compare_get()
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "causal"
            and segments[2] == "diff"
        ):
            self._handle_causal_diff_get()
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "causal"
            and segments[2] == "descendants"
        ):
            self._handle_causal_descendants_get()
            return
        if (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "causal"
            and segments[2] == "frontier"
        ):
            self._handle_causal_frontier_get()
            return
        if (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "causal"
            and segments[2] not in ("compare", "diff", "descendants", "frontier")
        ):
            self._handle_causal_get(segments[2], segments[3])
            return
        matched, checkpoint_peer = self._checkpoint_route()
        if matched:
            self._handle_checkpoint_get(checkpoint_peer)
            return
        matched, pickup_peer = self._peer_operations_route()
        if matched:
            self._handle_peer_operations_get(pickup_peer)
            return
        matched, receipts_peer = self._receipts_route()
        if matched:
            self._handle_peer_receipts_get(receipts_peer)
            return
        matched, receipts_audit_peer = self._receipts_audit_route()
        if matched:
            self._handle_peer_receipts_audit_get(receipts_audit_peer)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _validate_declared_length(self) -> int | None:
        """Validate the declared Content-Length, sending 400/413 on rejection.

        A missing, malformed, or conflicting declaration is answered with
        HTTP 400 and a declared length over ``MAX_BODY_BYTES`` with HTTP 413.
        Both rejections close the connection because the unread body can no
        longer be framed. Returns the validated length, or None when the
        error response has already been sent.
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
        """Read the request body under the shared POST size contract.

        Content-Length is validated before anything else (see
        :meth:`_validate_declared_length`): an over-limit declaration is
        rejected on its declared size alone, before a single body byte is
        read and however invalid the content would have been. Only when the
        declared length is within the limit are exactly that many bytes
        read. Returns the body, or None when the error response has already
        been sent.
        """
        length = self._validate_declared_length()
        if length is None:
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

    def _handle_peer_operations_get(self, peer_id: str) -> None:
        # Route-shape matching ran first in do_GET (missing/extra segments
        # and trailing slashes never reach here); an empty peer id is a
        # shape failure and stays 404 even when the query is malformed.
        if peer_id == "":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        params = parse_peer_pickup_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_peer_operations(peer_id, after, limit)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_peer_receipts_get(self, peer_id: str) -> None:
        # Route-shape matching ran first in do_GET (missing/extra segments
        # and trailing slashes never reach here); an empty peer id is a
        # shape failure and stays 404 even when the query is malformed.
        if peer_id == "":
            self._json_canonical_newline(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        params = parse_peer_receipts_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_peer_receipts(peer_id, after, limit)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        # The receipts contract fixes the response field order (receipts,
        # nextCursor, hasMore, algorithm, digest, receiptsCount; peerId,
        # ackId, cursor, operations per receipt; replicaId, operationId per
        # identity), so the body is emitted in the payload's field order
        # rather than sorted.
        self._json_ordered_newline(status, payload)

    def _handle_peer_receipts_audit_get(self, peer_id: str) -> None:
        # Route-shape matching ran first in do_GET (missing/extra segments
        # and trailing slashes never reach here); an empty peer id is a
        # shape failure and stays 404 even when the query is malformed.
        if peer_id == "":
            self._json_canonical_newline(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        params = parse_peer_receipts_audit_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_peer_receipts_audit(peer_id, after, limit)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        # As with the plain receipts route the response keeps the payload's
        # contracted field order: receipts, nextCursor, hasMore, algorithm,
        # digest, receiptsCount, audit (with coverage and the four anomaly
        # lists in the order the chain audit builds them).
        self._json_ordered_newline(status, payload)

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

    def _handle_acknowledge_post(self, peer_id: str) -> None:
        # Route-shape matching ran first in do_POST (missing/extra segments
        # and trailing slashes never reach here); an empty peer id is a
        # shape failure and stays 404 even when the query is malformed.
        if peer_id == "":
            self._json_canonical_newline(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        # The route accepts no query parameters; any parameter — repeated,
        # blank-named, or blank-valued — is an invalid request.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            ack_id, cursor, operations = parse_acknowledge_payload(raw)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        try:
            status, error = self._store.acknowledge_operations(
                peer_id, ack_id, cursor, operations
            )
        except PersistenceError:
            self._json_canonical_newline(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"}
            )
            return
        if status is HTTPStatus.NOT_FOUND:
            self._json_canonical_newline(status, {"error": "not_found"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json_canonical_newline(status, {"error": error})
            return
        self._json_canonical_newline(
            status,
            {
                "status": "created" if status is HTTPStatus.CREATED else "ok",
                "peerId": peer_id,
                "ackId": ack_id,
                "cursor": cursor,
            },
        )

    def _handle_metrics_get(self) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(HTTPStatus.OK, self._store.get_metrics())

    def _handle_state_at_get(self, key: str) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query check), so a malformed query is rejected here without any
        # state being read or changed. The history report follows the
        # compact-single-line contract: compact UTF-8 JSON, one trailing
        # newline, numbers only as JSON integers.
        cursor = parse_state_at_query(urlsplit(self.path).query)
        if cursor is None:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, payload = self._store.get_state_at(key, cursor)
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json_newline(status, payload)

    def _handle_state_causal_at_post(self, key: str) -> None:
        # The declared-length check (400/413) and the read-scope check ran
        # in do_POST before this handler, neither reading the body;
        # route-shape mismatches — missing, extra, or a trailing slash —
        # fall through to the generic 404 before any of those. The route
        # accepts no query parameters, and that check precedes the body
        # check. The report follows the compact-single-line contract:
        # compact UTF-8 JSON, one trailing newline, numbers only as JSON
        # integers, and the query is strictly read-only.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            boundary = parse_causal_at_payload(raw)
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        status, payload = self._store.get_state_causal_at(key, boundary)
        self._json_newline(status, payload)

    def _handle_state_why_get(self, key: str) -> None:
        # The route-shape check in do_GET already ran, so a query parameter
        # is rejected here without any state being read or changed. The
        # explanation body follows the compact-single-line contract: one
        # trailing newline, numbers only as JSON integers.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        status, payload = self._store.get_state_explanation(key)
        self._json_newline(status, payload)

    def _json_canonical_newline(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Respond with canonical compact JSON terminated by one newline.

        The body is serialized by :func:`_canonical_json_bytes`: sorted
        object keys, no insignificant whitespace, strings escaping only the
        quote, the backslash, and control characters, and every number a
        plain JSON integer. A single trailing ``\\n`` is included in both
        the body and its declared length.
        """
        body = _canonical_json_bytes(payload) + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_ordered_newline(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Respond with compact JSON in the payload's field order, newline-terminated.

        The body is serialized by :func:`_ordered_json_bytes`: object fields
        keep their insertion order (the endpoint's contracted field order),
        with the same escaping, integer-only numbers, and single trailing
        ``\\n`` as :meth:`_json_canonical_newline`.
        """
        body = _ordered_json_bytes(payload) + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_state_impact_get(self, key: str) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there), so a
        # malformed query is rejected here without any state being read or
        # changed. The response follows the compact-single-line contract:
        # canonical JSON, one trailing newline, numbers only as integers.
        params = parse_paging_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_state_impact(key, after, limit)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_verification_digest_get(self) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json(HTTPStatus.OK, self._store.get_verification_digest())

    def _handle_replication_snapshot_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there), so a
        # query parameter is rejected here without any state being read.
        # The success body follows the compact-single-line contract:
        # canonical JSON, one trailing newline, counts and the cursor only
        # as JSON integers.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._json_canonical_newline(
            HTTPStatus.OK, self._store.get_replication_snapshot()
        )

    def _handle_replication_status_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before
        # any query check), so a malformed query is rejected here without
        # any state being read or changed. The success body fixes the
        # field order (peer, pos, left, acks, chain) and follows the
        # compact-single-line contract: one trailing newline, numbers
        # only as JSON integers.
        peer_id = parse_replication_status_query(urlsplit(self.path).query)
        if peer_id is None:
            self._json_ordered_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        status, payload = self._store.get_replication_status(peer_id)
        self._json_ordered_newline(status, payload)

    def _handle_replication_compare_post(self) -> None:
        # The declared-length check (400/413) and the read-scope check ran
        # in do_POST before this handler, neither reading the body;
        # route-shape mismatches — missing, extra, or a trailing slash —
        # fall through to the generic 404 before any of those. The route
        # accepts no query parameters, and that check precedes the body
        # check. The comparison is strictly read-only: the remote snapshot
        # is validated and diffed against one committed local snapshot but
        # never imported, and no repair, transaction, sync, or persistence
        # runs. The report follows the compact-single-line contract:
        # canonical UTF-8 JSON, one trailing newline, counts only as JSON
        # integers.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            replica_id, snapshot = parse_replication_compare_payload(raw)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(
            HTTPStatus.OK,
            self._store.compare_replication_snapshot(replica_id, snapshot),
        )

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

    def _handle_audit_chain_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query check), so a malformed query is rejected here without any
        # state being read or changed. The response follows the
        # compact-single-line contract: canonical JSON, one trailing
        # newline, counts only as JSON integers.
        params = parse_paging_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            entries, next_cursor, has_more, head = self._store.get_audit_log_chain(
                after, limit
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(
            HTTPStatus.OK,
            {
                "entries": entries,
                "nextCursor": next_cursor,
                "hasMore": has_more,
                "head": head,
            },
        )

    def _handle_audit_log_verify_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query check), so a malformed query is rejected here without any
        # state being read or changed. Besides the chain query's
        # after/limit paging, the request requires two external
        # expectations, ``head`` (64 lowercase hex chars) and ``count`` (a
        # non-negative ASCII decimal integer); any missing, repeated,
        # unknown, blank, or malformed value is rejected. The response
        # follows the chain query's compact-single-line contract exactly,
        # adding only the ``verification`` conclusion.
        params = parse_audit_log_verify_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit, expected_head, expected_count = params
        try:
            payload = self._store.get_audit_log_verify(
                after, limit, expected_head, expected_count
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(HTTPStatus.OK, payload)

    def _handle_audit_digest_get(self, key: str) -> None:
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        # Every path key is a valid audit subject: a key with no accepted
        # history hashes the empty stream and reports operations 0.
        self._json(HTTPStatus.OK, self._store.get_key_audit_digest(key))

    def _handle_operation_archive_get(self, replica_id: str, operation_id: str) -> None:
        # The route accepts no query parameters; any parameter — repeated,
        # blank-named, or blank-valued — is an invalid request. Route-shape
        # mismatches (missing, empty, or extra segments) never reach this
        # handler: they fall through to the generic 404 first.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        status, payload = self._store.get_operation(replica_id, operation_id)
        self._json(status, payload)

    def _handle_causal_get(self, replica_id: str, operation_id: str) -> None:
        # The route-shape check in do_GET already ran (missing, empty, or
        # extra segments — including a trailing slash — are 404 there), so
        # a malformed query is rejected here without any state being read
        # or changed. The response follows the compact-single-line
        # contract: canonical JSON, one trailing newline, numbers only as
        # integers.
        params = parse_paging_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_causal_ancestors(
                replica_id, operation_id, after, limit
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_causal_compare_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query or identity check), so a malformed query is rejected here
        # without any state being read or changed. The response follows the
        # compact-single-line contract: canonical JSON, one trailing newline,
        # numbers only as integers.
        params = parse_causal_compare_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        (
            left_replica_id,
            left_operation_id,
            right_replica_id,
            right_operation_id,
            after,
            limit,
        ) = params
        try:
            status, payload = self._store.get_causal_comparison(
                left_replica_id,
                left_operation_id,
                right_replica_id,
                right_operation_id,
                after,
                limit,
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_causal_diff_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query or identity check), so a malformed query is rejected here
        # without any state being read or changed. The diff carries the same
        # four identity parameters plus after/limit as the comparison route,
        # and its response follows the same compact-single-line contract:
        # canonical JSON, one trailing newline, counts and cursors only as
        # integers.
        params = parse_causal_compare_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        (
            left_replica_id,
            left_operation_id,
            right_replica_id,
            right_operation_id,
            after,
            limit,
        ) = params
        try:
            status, payload = self._store.get_causal_diff(
                left_replica_id,
                left_operation_id,
                right_replica_id,
                right_operation_id,
                after,
                limit,
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_causal_descendants_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query or identity check), so a malformed query is rejected here
        # without any state being read or changed. The descendants route
        # carries the two identity parameters plus after/limit, and its
        # response follows the same compact-single-line contract as the
        # other causal queries: canonical JSON, one trailing newline,
        # numbers only as integers.
        params = parse_causal_descendants_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        replica_id, operation_id, after, limit = params
        try:
            status, payload = self._store.get_causal_descendants(
                replica_id, operation_id, after, limit
            )
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

    def _handle_causal_frontier_get(self) -> None:
        # The route-shape check in do_GET already ran (missing or extra
        # segments — including a trailing slash — are 404 there, before any
        # query check), so a malformed query is rejected here without any
        # state being read or changed. The frontier route carries only the
        # required after/limit paging pair, and its response follows the
        # same compact-single-line contract as the other causal queries:
        # canonical JSON, one trailing newline, numbers only as integers.
        params = parse_causal_frontier_query(urlsplit(self.path).query)
        if params is None:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_causal_frontier(after, limit)
        except ValueError:
            self._json_canonical_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        self._json_canonical_newline(status, payload)

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

    def _handle_auto_resolve_post(self, key: str) -> None:
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            request = parse_auto_resolve_payload(raw)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, operation, error = self._store.apply_auto_resolution(key, request)
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
                "replicaId": request["replicaId"],
                "operationId": request["operationId"],
                "value": operation["value"],
                "policy": request["policy"],
            },
        )

    def _handle_auto_resolve_batch_post(self) -> None:
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            entries = parse_auto_resolve_batch(raw)
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, results, accepted, replayed, error = self._store.apply_auto_resolutions(
                entries
            )
        except PersistenceError:
            self._json_newline(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json_newline(status, {"error": error})
            return
        self._json_newline(
            status,
            {
                "status": "created" if status is HTTPStatus.CREATED else "ok",
                "resolutions": results,
                "accepted": accepted,
                "replayed": replayed,
            },
        )

    def _handle_auto_resolve_plan_post(self) -> None:
        # Read-only preview of the committing batch: the response shows, per
        # entry, the value and policy a commit would select against the
        # snapshot observed now, without changing any business state. The
        # route-shape check in do_POST already ran (missing or extra
        # segments — including a trailing slash — are 404 there), so a query
        # parameter is rejected here without the body being read.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            entries = parse_auto_resolve_batch(raw)
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        # The preview cannot persist anything, so it has no durable-failure
        # path: success carries the planned results, a conflict carries the
        # same error body as the committing batch.
        status, results, accepted, replayed, error = self._store.plan_auto_resolutions(
            entries
        )
        if status is HTTPStatus.CONFLICT:
            self._json_newline(status, {"error": error})
            return
        self._json_newline(
            status,
            {
                "status": "planned",
                "resolutions": results,
                "accepted": accepted,
                "replayed": replayed,
            },
        )

    def _handle_transaction_apply_post(self) -> None:
        # The route-shape check in do_POST already ran (missing or extra
        # segments — including a trailing slash — are 404 there), so a
        # query parameter is rejected here without any state being read or
        # changed. The response follows the batch contract: compact JSON,
        # one trailing newline, counts only as JSON integers.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            transaction_id, entries = parse_transaction_apply(raw)
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        try:
            status, results, accepted, replayed, error = self._store.apply_transaction(
                transaction_id, entries
            )
        except ValueError:
            self._json_newline(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        except PersistenceError:
            self._json_newline(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        if status is HTTPStatus.CONFLICT:
            self._json_newline(status, {"error": error})
            return
        self._json_newline(
            status,
            {
                "status": "created" if status is HTTPStatus.CREATED else "ok",
                "transactionId": transaction_id,
                "operations": results,
                "accepted": accepted,
                "replayed": replayed,
            },
        )

    def _handle_scope_policy_reload_post(self) -> None:
        # The declared-length check (400/413), authentication, the admin
        # scope, and the scope-mode gate (404) all ran in do_POST, none of
        # them reading the body; route-shape mismatches — missing, extra, or
        # a trailing slash — fall through to the generic 404 before any of
        # those. The route accepts no query parameters, and that check
        # precedes the body check even on a correct route.
        if not parse_metrics_query(urlsplit(self.path).query):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        raw = self._read_bounded_body()
        if raw is None:
            return
        try:
            parse_empty_object_payload(raw)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        manager = getattr(self.server, "scope_policy", None)
        if manager is None:
            # Defensive: the mode gate already ran in do_POST. This keeps a
            # missing manager a not-found rather than a server error.
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        # The recorder durably commits the change event before the manager
        # swaps the live policy, so a durable failure aborts the whole
        # reload: the old policy stays in force and the failed reload
        # leaves no event behind.
        recorder = self._store.record_policy_reload
        try:
            digest, tokens = manager.reload(recorder)
        except ScopePolicyReloadError as exc:
            if exc.kind == "unavailable":
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE, {"error": "policy_unavailable"}
                )
            else:
                self._json(HTTPStatus.CONFLICT, {"error": "policy_conflict"})
            return
        except PersistenceError:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})
            return
        # Field order is part of the contract: status, policyDigest, tokens.
        self._json_ordered(
            HTTPStatus.OK,
            {"status": "reloaded", "policyDigest": digest, "tokens": tokens},
        )

    def _handle_scope_policy_audit_get(self) -> None:
        # Authentication, the admin scope, and the scope-mode gate all ran
        # in do_GET; route-shape mismatches — missing, extra, or a trailing
        # slash — fall through to the generic 404 before this handler runs.
        # The query accepts only the required after/limit pair.
        params = parse_scope_policy_audit_query(urlsplit(self.path).query)
        if params is None:
            self._json_ordered_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_policy_events(after, limit)
        except ValueError:
            self._json_ordered_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        # The response keeps the payload's contracted field order: events,
        # nextCursor, hasMore, algorithm, digest, eventsCount (and per event
        # sequence, digest, tokens), terminated by one newline.
        self._json_ordered_newline(status, payload)

    def _handle_scope_policy_audit_verify_get(self) -> None:
        # Authentication, the admin scope, and the scope-mode gate all ran
        # in do_GET; route-shape mismatches — missing, extra, or a trailing
        # slash — fall through to the generic 404 before this handler runs.
        # The query accepts only the required after/limit pair, exactly
        # like the plain change-audit route.
        params = parse_scope_policy_audit_verify_query(urlsplit(self.path).query)
        if params is None:
            self._json_ordered_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        after, limit = params
        try:
            status, payload = self._store.get_policy_events_verify(after, limit)
        except ValueError:
            self._json_ordered_newline(
                HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
            )
            return
        # The response keeps the payload's contracted field order: events,
        # nextCursor, hasMore, algorithm, digest, eventsCount,
        # verification (status then the four anomaly lists), terminated by
        # one newline.
        self._json_ordered_newline(status, payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        matched, checkpoint_peer = self._checkpoint_route()
        matched_ack, acknowledge_peer = self._acknowledge_route()
        segments = self._path_segments()
        is_operation_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "replicas"
            and segments[3] == "operations"
        )
        is_resolve_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "resolve"
        )
        is_auto_resolve_post = (
            len(segments) == 5
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "resolve"
            and segments[4] == "auto"
        )
        is_sync_post = (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "sync"
            and segments[2] == "operations"
        )
        is_auto_resolve_batch_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "resolve"
            and segments[2] == "auto"
            and segments[3] == "batch"
        )
        is_auto_resolve_plan_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "resolve"
            and segments[2] == "auto"
            and segments[3] == "plan"
        )
        is_transaction_apply_post = (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "transactions"
            and segments[2] == "apply"
        )
        is_scope_policy_reload_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "admin"
            and segments[2] == "scope-policy"
            and segments[3] == "reload"
        )
        is_causal_at_post = (
            len(segments) == 4
            and segments[0] == "v1"
            and segments[1] == "states"
            and segments[3] == "causal-at"
        )
        is_replication_compare_post = (
            len(segments) == 3
            and segments[0] == "v1"
            and segments[1] == "replication"
            and segments[2] == "compare"
        )
        if (
            matched
            or matched_ack
            or is_operation_post
            or is_resolve_post
            or is_auto_resolve_post
            or is_sync_post
            or is_auto_resolve_batch_post
            or is_auto_resolve_plan_post
            or is_transaction_apply_post
            or is_scope_policy_reload_post
            or is_causal_at_post
            or is_replication_compare_post
        ):
            # On the POST endpoints the Content-Length contract keeps its
            # priority: a 400/413 is answered before authentication. The
            # write-scope check then runs before the body is read, the
            # commit lock is taken, or any state or data file is touched; a
            # rejected request leaves the body unread and the connection
            # closed.
            if self._validate_declared_length() is None:
                return
            if is_scope_policy_reload_post:
                # The reload endpoint shares the length-first priority but
                # is admin-gated rather than write-gated. Authentication
                # still runs before the mode gate, so a missing or bad
                # credential is 401 in every mode; the endpoint then exists
                # only in scope-policy mode — single-token and anonymous
                # modes answer 404, exactly like an unpublished route.
                if not self._require_scope(SCOPE_ADMIN):
                    return
                if getattr(self.server, "scope_policy", None) is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
            elif is_auto_resolve_plan_post:
                # The preview changes no state, so it is gated like the read
                # endpoints: a read or admin scope suffices.
                if not self._require_scope(SCOPE_READ):
                    return
            elif is_causal_at_post:
                # The causal slice is strictly read-only, so it is gated
                # like every other read: a read or admin scope suffices.
                if not self._require_scope(SCOPE_READ):
                    return
            elif is_replication_compare_post:
                # The cross-replica comparison is strictly read-only — the
                # remote snapshot is diffed, never imported — so it is
                # gated like every other read: a read or admin scope
                # suffices.
                if not self._require_scope(SCOPE_READ):
                    return
            elif not self._require_scope(SCOPE_WRITE):
                return
        else:
            # Every other route authenticates and checks the write scope
            # before anything else (so a failed check never becomes a 404).
            if not self._require_scope(SCOPE_WRITE):
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if matched:
            self._handle_checkpoint_post(checkpoint_peer)
            return
        if matched_ack:
            self._handle_acknowledge_post(acknowledge_peer)
            return
        if is_operation_post:
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
        if is_resolve_post:
            self._handle_resolve_post(segments[2])
            return
        if is_auto_resolve_post:
            self._handle_auto_resolve_post(segments[2])
            return
        if is_auto_resolve_batch_post:
            self._handle_auto_resolve_batch_post()
            return
        if is_auto_resolve_plan_post:
            self._handle_auto_resolve_plan_post()
            return
        if is_transaction_apply_post:
            self._handle_transaction_apply_post()
            return
        if is_scope_policy_reload_post:
            self._handle_scope_policy_reload_post()
            return
        if is_causal_at_post:
            self._handle_state_causal_at_post(segments[2])
            return
        if is_replication_compare_post:
            self._handle_replication_compare_post()
            return
        self._handle_sync_post()

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
            "optional path to a readable regular file containing the bearer "
            "token clients must present; without it every endpoint stays "
            "anonymous except that /health always is; mutually exclusive "
            "with --scope-policy-file"
        ),
    )
    parser.add_argument(
        "--scope-policy-file",
        default=None,
        help=(
            "optional path to a readable regular UTF-8 JSON file mapping "
            "tokens to non-empty scope arrays of read/write/admin; enables "
            "scope-policy authentication and is mutually exclusive with "
            "--auth-token-file"
        ),
    )
    args = parser.parse_args(argv)

    if args.auth_token_file is not None and args.scope_policy_file is not None:
        # The two authentication configurations are mutually exclusive;
        # fail before any file is read or any port is bound, and never echo
        # either configuration's contents.
        print(
            "semantic-state-engine: startup failed: --auth-token-file and "
            "--scope-policy-file are mutually exclusive",
            file=sys.stderr,
        )
        raise SystemExit(2)

    auth_token = None
    auth_scopes = None
    if args.auth_token_file is not None:
        # Read and validate the token before binding any port; a rejected
        # file fails startup exactly like a rejected data file, and the
        # token is never printed.
        try:
            auth_token = load_auth_token(args.auth_token_file)
        except AuthTokenError as exc:
            print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    elif args.scope_policy_file is not None:
        # Read and validate the whole policy before binding any port; a
        # rejected file fails startup exactly like a rejected token file,
        # and neither tokens nor scopes are ever printed.
        try:
            auth_scopes = load_scope_policy(args.scope_policy_file)
        except ScopePolicyError as exc:
            print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        store = StateStore(data_file=args.data_file)
    except PersistenceError as exc:
        print(f"semantic-state-engine: startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    server = SemanticStateServer(
        (args.host, args.port),
        store=store,
        auth_token=auth_token,
        auth_scopes=auth_scopes,
        scope_policy_file=args.scope_policy_file,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
