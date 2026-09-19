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

Unknown routes return HTTP 404 with `{"error":"not_found"}`. Responses use UTF-8 JSON and include an explicit content length.

### Writing operations

`POST /v1/replicas/{replicaId}/operations` accepts a JSON object:

- `operationId`, `key`, `value`: non-empty strings.
- `clock`: an object mapping non-empty replica ids to non-negative integers; it must contain the path's `replicaId`.

Malformed JSON or invalid fields return HTTP 400 with `{"error":"invalid_request"}`.

Candidates are stored per key with vector-clock semantics (missing components count as 0; A dominates B when A ≥ B on every component and A ≠ B). A write deletes the candidates its clock dominates; concurrent candidates with different values are kept. A write whose clock is already dominated is recorded but adds no version.

- New operation: HTTP 201.
- Same `replicaId` + `operationId` + identical content replayed: HTTP 200, no new version.
- Same `replicaId` + `operationId` with different content: HTTP 409 with `{"error":"operation_conflict"}`; state is unchanged.

### Reading state

`GET /v1/states/{key}`:

- No versions for the key: HTTP 404 with `{"error":"not_found"}`.
- All candidates agree on the value: HTTP 200 with `{"key","value","clock","status":"resolved"}`; the chosen clock comes from the candidate with the lexicographically smallest `(replicaId, operationId)`.
- Otherwise: HTTP 200 with `{"key","status":"conflict","candidates":[...]}` where each candidate carries `value`, `clock`, `replicaId`, `operationId`, sorted by `(replicaId, operationId)` ascending.

Keys are isolated from each other, and reads reflect the latest writes.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
