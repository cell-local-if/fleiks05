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

- Before anything is recovered or served, a startup **atomic-commit preflight** verifies only the capabilities observable in the parent directory. It creates two exclusively named probe files in that directory: a small payload is written to the source probe, flushed and `fsync`ed; the source is then atomically replaced (`os.replace`) onto a separate target probe path; the directory is `fsync`ed; the target contents are verified; and both probes are removed (the directory is `fsync`ed again). Probes abandoned by an earlier failed start whose owning process is gone are reclaimed first. The data file itself is never modified, truncated, reordered, replaced, or opened for writing by the preflight — a successful preflight leaves its bytes and the recovered in-memory state unchanged.
- If `PATH` does not exist, it is created after the preflight passes (the parent directory must already exist). The file is created immediately as an empty, valid store, so an unwritable target fails startup rather than the first write.
- If `PATH` exists it must be a regular, readable file in the service's JSON data format. It is opened only for reading. A parent directory that does not exist, an inaccessible path, or a non-regular target (for example a directory or a named pipe) makes startup fail with exit code 2 and no serving instance.
- Any preflight failure (cannot create or write a probe, cannot `fsync` the probe or the directory, cannot atomically replace it) makes the service refuse to start with exit code 2 **before it begins listening** — the first valid write never turns into a 500. Probes are cleaned up on every failure path.
- The preflight guarantees only these directory-level capabilities. It does not predict a lock or ACL that is specific to the existing target file, nor changes in the environment after startup (for example the disk becoming unavailable). If a durable write fails at runtime, the request fails with HTTP 500 `{"error":"internal_error"}` and leaves memory, the operation identity, and the data file exactly as they were before that request; the request can be retried.
- On startup the file must parse completely and match the required structure; every stored candidate/operation record must satisfy the same input constraints as live requests. Any corruption, truncation, structural mismatch, or duplicate/illegal record makes the service refuse to start — state is never silently dropped or "repaired" by guessing.
- Recovery replays the accepted operations in their original commit order, so candidate ordering, vector-clock domination, stale-write handling, replay `200`, and content-conflict `409` are identical to a process that never restarted.
- Every first-accepted valid operation (the requests that return `201`, including stale writes that add no candidate) is flushed to disk and atomically committed (`write temp file → fsync → rename → fsync directory`) before the response is sent. Identical replays (`200`) append no record; conflicting requests (`409`) change neither memory nor the file. The atomic rename ensures a crash or interrupted write never leaves a partially updated file: the previous or the new complete state survives, never a mix.
- Concurrent writes share one commit order between memory and disk.

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
