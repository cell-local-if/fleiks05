# Semantic State Engine

Semantic State Engine is a Python backend for building a distributed state system that can detect, explain, and repair semantic conflicts between independently updated replicas.

The current baseline is a small, runnable service boundary implemented with the Python standard library and requires Python 3.11 or newer.

## Current API

Run the service:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --host 127.0.0.1 --port 8080
```

`GET /health` returns HTTP 200 and a JSON object:

```json
{"service":"semantic-state-engine","status":"ok"}
```

### Submit an operation

```
POST /v1/replicas/{replicaId}/operations
```

The JSON body must contain non-empty string `operationId`, `key`, and `value`,
plus a non-empty `clock` object mapping non-empty strings to non-negative
integers that includes the path's `replicaId`:

```json
{"operationId":"op-1","key":"color","value":"blue","clock":{"r1":1,"r2":0}}
```

Malformed JSON or invalid fields return `400 {"error":"invalid_request"}`.

Candidates per key are merged with vector-clock semantics (missing clock
components count as zero; clock A dominates B when `A >= B` on every component
with at least one strict inequality):

- `201` — the write was stored; versions it dominates are removed.
- `200` — an identical submission (same path replica, `operationId`, key,
  value, and clock) was replayed; no new version is added.
- `409 {"error":"operation_conflict"}` — the same `(replicaId, operationId)`
  already exists with different content; state is left unchanged.

Concurrent writes with different values whose clocks do not dominate one
another are all retained. Writes on different keys are isolated.

### Read a key's state

```
GET /v1/states/{key}
```

- No surviving versions: `404 {"error":"not_found"}`.
- All surviving candidates share one value: `200` with `status:"resolved"`,
  the shared `value`, and the `clock` of the candidate chosen by the smallest
  `(replicaId, operationId)` pair.
- Otherwise: `200` with `status:"conflict"` and a `candidates` array
  (`value`, `clock`, `replicaId`, `operationId`) sorted by
  `(replicaId, operationId)` ascending.

`clock_dominates(left, right)` and `parse_operation_payload(raw_body, replica_id)`
are exposed from `semantic_state_engine.server` for direct use and testing.

Unknown routes return HTTP 404 with `{"error":"not_found"}`. Responses use UTF-8 JSON and include an explicit content length.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
