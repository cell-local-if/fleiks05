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

### Local persistence and recovery (optional)

The service is purely in memory by default. Pass `--data-file PATH` to persist every accepted operation to a file and recover it on startup:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --data-file ./var/state.json
```

- If `PATH` does not exist, it is created on startup (the parent directory must already exist and be writable). The file is created immediately, so an unwritable target fails startup rather than the first write.
- If `PATH` exists it must be a regular, readable file in the service's JSON data format. A parent directory that does not exist, an inaccessible path, or a non-regular target (for example a directory or a named pipe) makes startup fail with exit code 2 and no serving instance.
- On startup the file must parse completely and match the required structure; every stored candidate/operation record must satisfy the same input constraints as live requests. Any corruption, truncation, structural mismatch, or duplicate/illegal record makes the service refuse to start — state is never silently dropped or "repaired" by guessing.
- Recovery replays the accepted operations in their original commit order, so candidate ordering, vector-clock domination, stale-write handling, replay `200`, and content-conflict `409` are identical to a process that never restarted.
- Every first-accepted valid operation (the requests that return `201`, including stale writes that add no candidate) is flushed to disk and atomically committed (`write temp file → fsync → rename → fsync directory`) before the response is sent. Identical replays (`200`) append no record; conflicting requests (`409`) change neither memory nor the file. The atomic rename ensures a crash or interrupted write never leaves a partially updated file: the previous or the new complete state survives, never a mix.
- Concurrent writes share one commit order between memory and disk. If a durable write fails at runtime (for example the disk becomes unavailable), the request fails with HTTP 500 `{"error":"internal_error"}` and leaves prior state untouched; the request can be retried.

The data file is a single UTF-8 JSON document, e.g.:

```json
{"version":1,"operations":[{"replicaId":"r1","operation":{"clock":{"r1":1},"key":"color","operationId":"op-1","value":"blue"}}]}
```

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
