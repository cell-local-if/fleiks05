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

### Request body limits

All four POST interfaces — `POST /v1/replicas/{replicaId}/operations`, `POST /v1/sync/operations`, `POST /v1/states/{key}/resolve`, and `POST /v1/sync/peers/{peerId}/checkpoint` — share one request-body size contract, enforced before anything else touches the request:

- Every POST must carry a `Content-Length` header holding a single ASCII decimal integer (multiple `Content-Length` headers are tolerated only when every value is identical). A missing, empty, signed, whitespace-padded, non-ASCII, or otherwise malformed value — or multiple conflicting values — returns HTTP 400 with `{"error":"invalid_request"}` and is never treated as an empty body.
- The declared length must not exceed **1,048,576 bytes** (1 MiB) of raw UTF-8 body. A larger declared length returns HTTP 413 with `{"error":"payload_too_large"}` before any body byte is read — ahead of JSON parsing, the commit lock, and any change to memory or the data file — even when the body itself would have been invalid.
- A body declared at exactly the limit is processed normally: the service reads exactly the declared number of bytes and then applies the endpoint's usual JSON and field validation, status codes, batch atomicity, idempotency, and persistence semantics.
- A rejected request (400/413) adds no operation, candidate, checkpoint, or audit record, and leaves no temporary persistence file behind.

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
- On startup the file must parse completely and match the required structure; every stored candidate/operation record must satisfy the same input constraints as live requests, and every stored checkpoint must be a non-empty peer id mapped to a non-boolean non-negative integer no greater than the recovered log length. Any corruption, truncation, structural mismatch, unknown top-level key, or duplicate/illegal record makes the service refuse to start — state is never silently dropped or "repaired" by guessing.
- Recovery replays the accepted operations in their original commit order, so candidate ordering, vector-clock domination, stale-write handling, replay `200`, and content-conflict `409` are identical to a process that never restarted.
- Every first-accepted valid operation (the requests that return `201`, including stale writes that add no candidate) is flushed to disk and atomically committed (`write temp file → fsync → rename → fsync directory`) before the response is sent. Identical replays (`200`) append no record; conflicting requests (`409`) change neither memory nor the file. The atomic rename ensures a crash or interrupted write never leaves a partially updated file: the previous or the new complete state survives, never a mix.
- Registered sync checkpoints live in the same file and are created or advanced in the same atomic commit before the checkpoint `200`; a checkpoint-only commit writes the unchanged operation log together with the new mapping. Checkpoint validation, persistence, and visibility are one commit under the same lock as writes, imports, and repairs.
- Concurrent writes share one commit order between memory and disk.

The data file is a single UTF-8 JSON document, e.g.:

```json
{"checkpoints":{"peer-a":2},"operations":[{"replicaId":"r1","operation":{"clock":{"r1":1},"key":"color","operationId":"op-1","value":"blue"}}],"version":1}
```

The `checkpoints` section is optional and holds sender-side replication cursors (see below); a file written before checkpoints existed contains only `version` and `operations`, and recovers with no registered checkpoints. `version` stays `1`: the supplemented format is backward compatible, and an old file is upgraded on disk the first time a checkpoint (or any other new commit) is persisted.

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

### Read-only metrics

`GET /v1/metrics` returns HTTP 200 with a UTF-8 JSON object containing exactly six non-negative integer counters:

```json
{"acceptedOperations":4,"candidateVersions":3,"conflictKeys":1,"keys":2,"replicas":3,"resolvedKeys":1}
```

- `acceptedOperations`: the total number of first-accepted operations in the shared log — ordinary writes, stale writes that add no candidate, and conflict repairs — excluding identical replays (`200`), conflicting or invalid requests (`409`/`400`), and operations whose durable commit failed.
- `keys`: the number of keys that currently hold at least one candidate.
- `candidateVersions`: the total number of current candidates across those keys.
- `conflictKeys`: keys whose candidates do not all agree on a value.
- `resolvedKeys`: all other keys; `conflictKeys + resolvedKeys` always equals `keys`.
- `replicas`: the number of distinct `replicaId` values in the accepted-operation log; a repair counts under its initiating replica.

The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`. Extra path segments (for example `/v1/metrics/extra`) return HTTP 404 with `{"error":"not_found"}`.

All six counters are computed from one snapshot under the same commit lock used by local writes, sync-import batches, and repairs, so they always describe a single commit: a read can never observe half a batch or counters that disagree with each other. The request is strictly read-only — it modifies neither memory nor the data file or sync log — and its response uses the same explicit `Content-Length` contract as the other endpoints.

With `--data-file`, the counters are rebuilt from the recovered log on startup, so after a restart they are identical to those reported just before the restart.

### Replica-convergence verification digest

`GET /v1/verification/digest` returns HTTP 200 with a UTF-8 JSON object containing exactly four fields:

```json
{"algorithm":"sha256","candidateVersions":3,"digest":"<64 lowercase hex chars>","keys":2}
```

- `algorithm` is always `"sha256"`.
- `digest` is the 64-character lowercase hexadecimal SHA-256 of the canonical candidate snapshot described below.
- `keys` and `candidateVersions` are the same non-negative counts reported by `GET /v1/metrics`: keys currently holding at least one candidate, and the total number of current candidates across them.

The digest covers **only the current candidate sets** — never the accepted-operation log, stale writes that added no candidate, or sync checkpoints. Two replicas holding the same candidates therefore report the same digest no matter how their logs, checkpoints, or operation histories differ, which is what makes it usable as a convergence check.

The digest input is a compact UTF-8 JSON array with one entry per key:

- Entries are `{"key":K,"candidates":C}`, sorted by key in lexicographic (Unicode code point) order.
- `C` is sorted by `(replicaId, operationId)` ascending; each candidate carries its fields in the fixed order `{"value":V,"clock":D,"replicaId":R,"operationId":O}`, and `D`'s component names are sorted lexicographically.
- No whitespace appears anywhere. Strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex); every other Unicode code point is written literally.

The SHA-256 is computed over exactly those bytes; for an empty store the input is `[]`.

The digest input and both counters are computed from one snapshot under the same commit lock used by local writes, sync-import batches, and repairs, so the response always describes a single commit and never observes half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file — and its response uses the same explicit `Content-Length` contract as the other endpoints.

The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`. Extra path segments (for example `/v1/verification/digest/extra`) return HTTP 404 with `{"error":"not_found"}`.

With `--data-file`, the candidate state is rebuilt from the recovered log on startup, so the same state yields the identical response before and after a restart.

### Resolving conflicts

`POST /v1/states/{key}/resolve` repairs a key that is currently in conflict. The body is a JSON object with exactly these keys:

```json
{"replicaId":"r3","operationId":"fix-1","value":"merged","clock":{"r1":1,"r2":1,"r3":1},"candidates":[{"replicaId":"r1","operationId":"op-1"},{"replicaId":"r2","operationId":"op-2"}]}
```

- `replicaId`, `operationId`, `value`, `clock` follow the same constraints as a local write (the key comes from the path; the clock must contain `replicaId`).
- `candidates` is a non-empty list of distinct `{"replicaId","operationId"}` identities naming the conflicting candidates being resolved.

A resolution commits only when the key is currently in conflict, the listed set is exactly the key's current candidate set, and `clock` dominates every listed candidate. The resolution is then accepted atomically as one operation in the shared commit order: the dominated candidates are cleared and the resolution value becomes the only version, so `GET /v1/states/{key}` reports `resolved` with that value. Because a resolution is an ordinary accepted operation, it is exported by `GET /v1/sync/operations`, imported by `POST /v1/sync/operations` (resolving the same conflict on replicas that hold it), persisted to `--data-file`, and recovered on restart, interleaving with local writes and import batches in exactly one commit order.

- Success: HTTP 201 with `{"status":"created","key","replicaId","operationId"}`.
- A malformed body, an unknown or duplicated candidate identity, or a clock that does not dominate every candidate: HTTP 400 with `{"error":"invalid_request"}`; nothing changes.
- The key does not exist, is not in conflict, the candidate set does not match the current candidates, or a concurrent write changed the set: HTTP 409 with `{"error":"resolution_conflict"}`; nothing changes.
- The same `(replicaId, operationId)` replayed with identical content: HTTP 200 with `"status":"ok"` and no new log record; with different content: HTTP 409 with `{"error":"operation_conflict"}`; nothing changes.
- With `--data-file`, the resolution is committed durably before the 201 response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory, the identity index, and the file exactly as they were (the request can be retried).

### Incremental sync between replicas

Two additional endpoints stream the accepted-operation log between replicas. The existing endpoints, payloads, and status codes are unchanged; a sync record is simply the path `replicaId` paired with an otherwise ordinary `operation`.

#### Exporting operations

`GET /v1/sync/operations?after=N&limit=N` returns one page of the log in commit order:

- `after` is the number of records already skipped (a 0-based resume cursor); it defaults to `0`. `after=N` returns the records committed after the first `N`, and `after` equal to the current log length is a valid empty tail.
- `limit` defaults to `100` and must be between `1` and `100`.
- A successful HTTP 200 response is `{"operations":[...],"nextCursor":N,"hasMore":bool}`. Each operation is `{"replicaId","operation"}` in the same shape as the data file, ordered as committed (including stale writes that added no candidate). `nextCursor` is the number of records skipped after this page — feed it back as the next `after` — and `hasMore` reports whether records remain.
- The page is sliced from a single snapshot under the commit lock, so the records, `nextCursor`, and `hasMore` always agree even while writes are committing concurrently.
- A negative or malformed `after`/`limit`, a limit outside `1-100`, an `after` past the end of the log, or any unknown/repeated query parameter returns HTTP 400 with `{"error":"invalid_request"}`.

#### Importing operations

`POST /v1/sync/operations` accepts an object with a single key:

```json
{"operations":[{"replicaId":"r1","operation":{"operationId":"op-1","key":"color","value":"blue","clock":{"r1":1}}}]}
```

- `operations` must contain between 1 and 100 records. Each record has exactly `replicaId` (a non-empty string) and `operation`; the operation obeys the same constraints as a local write, and its clock must contain that record's `replicaId`. Any other shape returns HTTP 400 with `{"error":"invalid_request"}`.
- Records are imported in order. An unknown `(replicaId, operationId)` is accepted exactly like a local write (dominated/stale writes are recorded but add no candidate); a known identity with identical content is a replay; a known identity with different content is a conflict.
- HTTP 201 with `{"status":"created","accepted":A,"replayed":R}` when the batch contains at least one new operation, or HTTP 200 with `"status":"ok"` when every record was a replay. `accepted`/`replayed` count records of each kind.
- A conflicting record returns HTTP 409 with `{"error":"operation_conflict"}` and the **entire batch is unchanged**: earlier records in the same request are not partially committed, and memory, the identity index, and the data file stay as they were before the request.

Imports share the single commit order with local `POST /v1/replicas/...` writes: an import batch is processed as one indivisible unit, so a concurrent state read never sees half a batch, and records from local writes and imports interleave in exactly one global order in the export.

With `--data-file`, all new operations of a batch are written to the file in one atomic commit before the success response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory, the identity index, and the file exactly as they were before the batch (the request can be retried). Pure-replay batches need no write and still succeed. After a restart the export order, cursor resume, replay `200`, and conflict `409` are identical to a process that never restarted.

### Sender-side consumption checkpoints

`POST /v1/sync/peers/{peerId}/checkpoint` lets a sending peer persist how far it has consumed the accepted-operation log, and `GET` returns the registered object:

```json
POST /v1/sync/peers/peer-a/checkpoint
{"cursor": 2}
```

- `{peerId}` must be a non-empty path segment (`/v1/sync/peers//checkpoint` is HTTP 400); percent-encoded segments are decoded.
- The body must be a JSON object whose only key is `cursor` holding a non-boolean, non-negative integer: exactly `{"cursor":N}`. Malformed JSON, a non-object body, extra keys, a missing/negative/boolean/string/float cursor all return HTTP 400 with `{"error":"invalid_request"}`.
- `N` must not exceed the length of the accepted log at validation time, so a cursor can never point at an unaccepted record; otherwise HTTP 400 with `{"error":"invalid_request"}`.
- First registration, an equal-value replay, and an advance all return HTTP 200 with `{"peerId","cursor"}`.
- When a strictly larger cursor is already registered for the peer, the request returns HTTP 409 with `{"error":"checkpoint_conflict"}` and the stored cursor never moves backwards.
- `GET /v1/sync/peers/{peerId}/checkpoint` returns HTTP 200 with `{"peerId","cursor"}`, or HTTP 404 with `{"error":"not_found"}` when the peer has never registered. The GET takes no query parameters: any parameter (including a blank name/value or a repeated name) returns HTTP 400 with `{"error":"invalid_request"}`.
- Extra path segments (for example `/v1/sync/peers/{peerId}/checkpoint/extra`, or a missing `peerId`/`checkpoint` segment) return HTTP 404 with `{"error":"not_found"}` for both methods.

A checkpoint is not an operation. It changes neither the accepted-operation log nor sync export, the per-key audit, candidate state, or any of the six metrics counters; it is not exported by `GET /v1/sync/operations` and does not appear in audit streams. Checkpoints do share the commit lock with local writes, import batches, and conflict repairs, however: validating the cursor, persisting it, and making it visible are one indivisible commit. The bound is checked against the same committed snapshot that is persisted, a concurrent reader always sees either the old or the new checkpoint and can never observe half an import batch alongside a moved cursor, and a cursor never names a record that is not durably accepted.

With `--data-file`, a new or advanced checkpoint is written to the data file in the same atomic commit protocol (`write temp file → fsync → rename → fsync directory`) before the HTTP 200. A durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves both memory and the file unchanged — a new registration is absent and an advanced cursor keeps its old value — so the request is safely retryable; an equal-value replay needs no write and still succeeds during the fault. A version:1 file written before checkpoints existed (without the `checkpoints` section) recovers with no registered checkpoints and otherwise unchanged semantics; after the format is supplemented on disk, a restart preserves both the checkpoints and every existing behavior. Without `--data-file` checkpoints live only in memory, exactly like the rest of the state.

### Per-key operation audit

`GET /v1/audit/keys/{key}/operations?after=N&limit=N` returns the history of **accepted operations for one key**, reusing the same commit order, record shape (`{"replicaId","operation"}`), and paging rules as sync export — the stream is simply the shared accepted-operation log filtered to records whose `operation.key` equals the path key.

The stream contains every first-accepted operation for the key, including:

- stale writes whose clock was already dominated and therefore added no candidate, and
- conflict repairs accepted through `POST /v1/states/{key}/resolve`.

It never contains operations for other keys, identical replays (`200`), conflicting or malformed requests (`409`/`400`), or uncommitted requests.

- `after` is the number of this key's records already skipped (a per-key 0-based resume cursor); it defaults to `0`. It counts only records for the path key — operations for other keys do not consume cursor positions. `after=N` returns the key's records committed after the first `N` of *that key's* records, and `after` equal to the key's current record count is a valid empty tail (a key with no history therefore accepts only `after=0`).
- `limit` defaults to `100` and must be between `1` and `100`.
- A successful HTTP 200 response is `{"operations":[...],"nextCursor":N,"hasMore":bool}`, identical in shape to sync export; `nextCursor` is the number of the key's records skipped after this page — feed it back as the next `after`. A key with no history returns HTTP 200 with an empty page.
- The filtered list, the page slice, `nextCursor`, and `hasMore` are computed from a single snapshot under the same commit lock used by local writes, sync imports, and resolutions, so the three values always agree even while commits are in flight. An import batch commits as one indivisible segment of the global order: its records for the key appear consecutively in the audit stream, and a read can never observe half a batch.
- A negative, blank, or non-ASCII-decimal `after`/`limit` (signs, decimals, whitespace, and non-ASCII numerals are all rejected), a repeated or unknown query parameter, a `limit` outside `1-100`, or an `after` greater than the key's current record count returns HTTP 400 with `{"error":"invalid_request"}`. Unknown route shapes (for example `/v1/audit/keys/{key}/operations/extra`) return HTTP 404.

With `--data-file`, the audit reads exactly the same durable log that sync export and recovery use: after a restart the per-key order, page boundaries, cursor resume, stale-write records, and accepted repair records are identical to a process that never restarted. A durable commit failure leaves no audit record (the operation neither reaches memory nor the file), and a conflicting import batch is rejected as a whole and likewise leaves no audit trace.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
