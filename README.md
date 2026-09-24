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

Unknown routes return HTTP 404 with `{"error":"not_found"}`. Responses use UTF-8 JSON and include an explicit content length. A trailing slash on any published path is treated as a missing/extra path-segment boundary and returns HTTP 404 with `{"error":"not_found"}` — it is never served as an alias for the bare path.

### Request body limits

All six POST endpoints (`POST /v1/replicas/{replicaId}/operations`, `POST /v1/sync/operations`, `POST /v1/states/{key}/resolve`, `POST /v1/states/{key}/resolve/auto`, `POST /v1/resolve/auto/batch`, `POST /v1/sync/peers/{peerId}/checkpoint`) share one body-size contract:

- The request body is limited to **1,048,576 raw UTF-8 bytes** (1 MiB). A body whose declared length is exactly the limit is processed by the normal endpoint semantics.
- `Content-Length` is required and validated before anything else. It must be a plain ASCII decimal integer: a missing header, an empty value, a sign, whitespace, a negative number, non-ASCII digits, or multiple headers declaring conflicting lengths all return HTTP 400 with `{"error":"invalid_request"}` — the request is never treated as having an empty body. (Multiple headers are accepted only when every occurrence declares the same length.)
- A declared length over the limit returns HTTP 413 with `{"error":"payload_too_large"}` **before the body is read**, before JSON parsing, and before the commit lock or any memory/data-file state is touched — an over-limit declaration is rejected on its size alone, even when the content would also have been invalid.
- When the declared length is within the limit, exactly that many bytes are read and the endpoint's existing JSON and field validation, status codes, batch atomicity, idempotency, and persistence-failure semantics apply unchanged.
- A rejected request (400 or 413) adds no operation, candidate, checkpoint, or audit record and creates no temporary persistence file; memory and the data file are exactly as they were before the request.
- When bearer-token authentication is enabled (see below), these 400/413 rejections keep their priority: they are answered before the 401 authentication check, and only a request with a valid declared length can reach the authentication check at all.

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
{"checkpoints":{"peer-a":2},"operations":[{"replicaId":"r1","operation":{"clock":{"r1":1},"key":"color","operationId":"op-1","value":"blue"}}],"policies":[{"operationId":"fix-1","policy":"lowest_identity","replicaId":"r3"}],"version":1}
```

The `checkpoints` section is optional and holds sender-side replication cursors (see below); a file written before checkpoints existed contains only `version` and `operations`, and recovers with no registered checkpoints. The `policies` section is likewise optional and holds the automatic-resolution policy bindings (see below): one `{"replicaId","operationId","policy"}` record per accepted automatic resolution, committed atomically with its operation. `version` stays `1`: the supplemented format is backward compatible, and an old file is upgraded on disk the first time a checkpoint (or any other new commit) is persisted.

### Optional bearer-token authentication

The service is anonymous by default: without authentication options every documented behavior above is unchanged. Pass `--auth-token-file PATH` to require a bearer token on every endpoint except the health probe:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --auth-token-file ./var/token
```

- The token file is read and validated **before the service begins listening**. It must be a readable regular file whose entire content is exactly one non-empty ASCII printable token — no whitespace, no newlines, nothing before or after it. A missing, unreadable, or non-regular target (for example a directory) and any format violation make startup fail with exit code 2, exactly like a rejected data file: no port is bound and the token is never printed.
- `GET /health` stays anonymous. Every other route — known or unknown, GET or POST — requires the request to carry **exactly one** `Authorization` header whose value is exactly `Bearer ` (one space) followed by the token. A missing, duplicated, or malformed header and any token mismatch return HTTP 401 with `{"error":"unauthorized"}` and a `WWW-Authenticate: Bearer` response header — before route matching, query parsing, the commit lock, any state read, any data-file access, and any POST body read. The comparison uses the standard library's constant-time primitive.
- A rejected request changes nothing: it creates no temporary file and leaves memory, logs, checkpoints, audit streams, and the data file exactly as they were; the token is never leaked in responses or logs.
- The six POST endpoints keep their Content-Length priority: a missing/malformed declaration still returns 400 and an over-limit declaration still returns 413 **before** authentication is checked. When the declared length is valid but the request is unauthorized, the 401 is sent **without reading the body** and the connection is closed.
- Once a request is authenticated, every existing behavior — success codes, 400/404/409/500, paging, digests, idempotency, concurrency, and recovery — is exactly as documented for the anonymous service.
- The authentication configuration is never written to the data file: a `--data-file` restart recovers only operations and checkpoints, and the token is supplied again (or not) via the command line on each start.

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

### Explaining a key's candidate state

`GET /v1/states/{key}/why` returns a read-only causal explanation of one key's current candidate state. A key with no current candidates — one that never appeared, or one whose history leaves no current candidate — returns HTTP 404 with `{"error":"not_found"}`.

A successful HTTP 200 response is a compact UTF-8 JSON object with exactly five fields, terminated by a single newline:

```json
{"candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"},{"clock":{"r2":1},"operationId":"op-2","replicaId":"r2","value":"red"}],"key":"color","relations":[{"from":{"operationId":"op-1","replicaId":"r1"},"relation":"concurrent","to":{"operationId":"op-2","replicaId":"r2"}}],"status":"conflict","suggestion":{"highest_identity":{"clock":{"r2":1},"operationId":"op-2","replicaId":"r2","value":"red"},"lowest_identity":{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}}}
```

- `key`: the requested key (the path segment is percent-decoded like every route).
- `status`: `"resolved"` when every current candidate agrees on the value, `"conflict"` otherwise — the same classification as `GET /v1/states/{key}`.
- `candidates`: the current candidates in the same order as the conflict view of `GET /v1/states/{key}` — sorted by `(replicaId, operationId)` ascending — each carrying exactly `value`, `clock`, `replicaId`, and `operationId`.
- `relations`: one entry per unordered pair of current candidates, enumerated in candidate order. Each entry names the pair's endpoints as `{"replicaId","operationId"}` identities under `from`/`to` and classifies the pair under `relation`:
  - `"overwrites"` when the two candidates hold the same value: either one covers the other, so the pair cannot conflict. For a resolved key these entries report the agreed value's unique source relation. The same-value rule takes precedence over the clock comparison.
  - `"dominates"` when the values differ and one candidate's clock dominates the other's (`from` is the dominating candidate). Current candidates never dominate one another, so this kind completes the vocabulary without being emitted by the present store.
  - `"concurrent"` when the values differ and neither clock dominates the other — exactly why the pair does not dominate each other. In a mixed conflict (some pairs sharing a value, some not) a different-valued, mutually non-dominating pair is therefore never misreported as `overwrites` or `dominates`.
  A key with a single candidate has an empty relation set, still expressed as an array (`[]`).
- `suggestion`: `{"lowest_identity":C,"highest_identity":C}` reporting which current candidate each of the two existing automatic-resolution policies would select — the smallest and largest `(replicaId, operationId)` — in the same shape as the `candidates` entries. The suggestion is purely informational: the endpoint creates no repair operation, log record, or checkpoint.

Every number in the response is a JSON integer (the only numbers are vector-clock ticks); no float, negative zero, or non-finite value can appear.

The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state. A missing, empty, or extra path segment (for example `/v1/states/{key}/why/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.

The candidate set, the relations, and the suggestion are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit and never observes half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file and creates no file. With `--data-file`, the candidate state is rebuilt identically during recovery, so the same state yields the same relations, sources, and policy suggestions before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route.

### Cross-key causal impact

`GET /v1/states/{key}/impact?after=N&limit=N` returns a read-only report of the operations on **other keys** that are causally later than one key's current state. A key with no current candidates — one that never appeared, or one whose history leaves no current candidate — returns HTTP 404 with `{"error":"not_found"}`.

The query reads only the target key's current candidates and the shared accepted-operation log: an accepted operation on a different key is an impact when its clock **dominates** at least one of the target key's current candidates (missing components count as 0, exactly as in the write semantics). The target key's own operations never appear, and identical replays (`200`), conflicting or malformed requests (`409`/`400`), and uncommitted writes never enter the accepted log, so they can never appear either.

A successful HTTP 200 response is a compact UTF-8 JSON object with exactly six fields, terminated by a single newline:

```json
{"candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}],"hasMore":false,"impacts":[{"operation":{"clock":{"r1":1,"r2":1},"key":"size","operationId":"op-9","value":"large"},"replicaId":"r2"}],"key":"color","nextCursor":1,"status":"resolved"}
```

- `key`: the requested key (the path segment is percent-decoded like every route).
- `status`: `"resolved"` when every current candidate agrees on the value, `"conflict"` otherwise — the same classification as `GET /v1/states/{key}`.
- `candidates`: the basis candidates the judgement is made from — the target key's current candidates in the same order as the conflict view of `GET /v1/states/{key}` (sorted by `(replicaId, operationId)` ascending), each carrying exactly `value`, `clock`, `replicaId`, and `operationId`.
- `impacts`: one page of the impacting operations in the shared log's global commit order. Each entry preserves the committed record shape `{"replicaId":R,"operation":{"operationId","key","value","clock"}}` — identity, key, value, and clock exactly as committed.
- `nextCursor`: the number of impact records skipped after this page — feed it back as the next `after`.
- `hasMore`: whether further impact records remain.

Paging follows the sync-export rules: `after` is the number of impact records already skipped (a 0-based resume cursor) and defaults to `0`; `limit` defaults to `100` and must be between `1` and `100`. A negative, blank, or non-ASCII-decimal `after`/`limit`, a repeated or unknown query parameter, a `limit` outside `1-100`, or an `after` past the impact record count returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state. A missing, empty, or extra path segment (for example `/v1/states/{key}/impact/extra` or `/v1/states/{key}/impact/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

The candidate set, the impact list, the status, and the paging boundaries are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit and never observes half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the candidate state and the log are rebuilt identically during recovery, so the same state yields the same impact report before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route.

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

### Deterministic automatic conflict resolution

`POST /v1/states/{key}/resolve/auto` resolves a conflict without the caller naming a value or candidate set. The body is a JSON object with exactly these keys:

```json
{"replicaId":"r3","operationId":"auto-fix-1","clock":{"r1":1,"r2":1,"r3":1},"policy":"lowest_identity"}
```

- `replicaId`, `operationId`, `clock` follow the same constraints as a manual resolution (the key comes from the path; the clock must contain `replicaId`).
- `policy` must be the literal string `"lowest_identity"` or `"highest_identity"`. Any other shape, field, or value is HTTP 400 `{"error":"invalid_request"}`.
- There is no `value` and no `candidates` list: both are determined by the server from the key's current candidates.

The request commits only when the key currently holds different value candidates. The resolution value is then chosen deterministically by the policy: `"lowest_identity"` takes the value of the current candidate with the lexicographically smallest `(replicaId, operationId)`, `"highest_identity"` the largest (ties on value do not matter — the extreme identity is unique), and the request clock must dominate **every** current candidate. The resolution is accepted atomically as one operation in the shared commit order, exactly like a manual resolution: the dominated candidates are cleared and the chosen value becomes the only version. Because it is an ordinary accepted operation, it is exported by `GET /v1/sync/operations` (as the chosen `value` together with the request's `replicaId`/`operationId`/`clock`), imported by `POST /v1/sync/operations`, appears in the key's audit stream and audit digest, counts in every metrics counter and in the verification digest, is persisted to `--data-file`, and is recovered on restart.

- Success: HTTP 201 with `{"status":"created","key","replicaId","operationId","value","policy"}`, where `value` is the chosen candidate's value and `policy` echoes the request's policy.
- A malformed body, an unknown `policy`, or a clock that is invalid or does not dominate every current candidate: HTTP 400 with `{"error":"invalid_request"}`; nothing changes.
- The key does not exist, its candidates all already agree on one value, or the candidate set moved between validation and commit: HTTP 409 with `{"error":"resolution_conflict"}`; nothing changes.
- The identity is bound to the key, the clock, **and the policy**: the same `(replicaId, operationId)` replayed with the same binding is HTTP 200 with `{"status":"ok","key","replicaId","operationId","value","policy"}` and appends no log record (the originally chosen value is reported back); a known identity with a different key, clock, or policy — even when the value it would choose is the same — is HTTP 409 with `{"error":"operation_conflict"}`. Identity replay is answered from the committed operation, so replaying after the key has moved on neither re-resolves nor appends.
- Automatic and manual resolutions share one identity space with ordinary writes: a `(replicaId, operationId)` committed without a policy binding (a plain write, a manual resolution, or an operation imported via sync) never matches a policy-carrying request and conflicts by the same rules. The policy binding is local to the resolving replica — it is persisted to `--data-file` but is not part of the exported sync record, so an importing replica holds the operation without the binding.
- With `--data-file`, the operation and its policy binding are persisted together (write temp file → fsync → rename → fsync directory) before the 201 response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory, the identity index, the policy bindings, and the file exactly as they were (the request can be retried). After a restart the chosen value, replay `200`, and conflict `409` are identical to a process that never restarted.

### Batched automatic conflict resolution

`POST /v1/resolve/auto/batch` applies between 1 and 100 automatic resolutions in one request, each naming its own target key. The body is a JSON object with exactly one key:

```json
{"resolutions":[{"key":"color","replicaId":"r3","operationId":"auto-fix-1","clock":{"r1":1,"r2":1,"r3":1},"policy":"lowest_identity"}]}
```

- `resolutions` must contain between 1 and 100 entries, kept in request order. Each entry has exactly `key`, `replicaId`, `operationId`, `clock`, and `policy`: the same four fields as the single-key automatic resolution plus its target `key` (the route carries no path segment). Every field obeys the single-key constraints: non-empty strings, a clock of non-boolean non-negative integer components containing the entry's `replicaId`, and `policy` equal to `"lowest_identity"` or `"highest_identity"`.
- Every numeric value the request carries must be an integer: JSON floats (including `1.0`), negative zero (`-0.0`), and the non-finite tokens `NaN`/`Infinity`/`-Infinity` are all rejected as malformed input.
- No two entries may name the same `key`, and no two may carry the same `(replicaId, operationId)` identity, even across different keys.
- An empty batch, more than 100 entries, a duplicate key or identity, malformed JSON, an unknown field at the root or on an entry, or a structurally illegal clock all return HTTP 400 with `{"error":"invalid_request"}`; nothing changes.
- Extra path segments (for example `/v1/resolve/auto/batch/extra`) return HTTP 404 with `{"error":"not_found"}`.

Entries are processed **in request order**, each with exactly the single-key semantics: the key must currently hold different-valued candidates, the policy selects the value of the candidate with the smallest or largest `(replicaId, operationId)`, and the entry clock must dominate every current candidate of its key at that position in the sequence. Any single failure rejects the **whole batch**: earlier entries in the same request are not partially committed, and memory, the identity index, policy bindings, and the data file stay exactly as they were before the request.

- No conflict (a missing key, candidates that already agree), a candidate set that moved, or a legal clock that does **not** dominate the current candidates returns HTTP 409 with `{"error":"resolution_conflict"}` (the clock itself must still be structurally valid — an illegal clock is the `400 invalid_request` above).
- A known `(replicaId, operationId)` with a different binding (different key, clock, or policy, or an identity committed without a policy binding) returns HTTP 409 with `{"error":"operation_conflict"}`. The same identity with the same binding is answered from the committed operation, in whatever position it occurs.
- HTTP 201 with `{"status":"created","resolutions":[...],"accepted":A,"replayed":R}` when at least one entry newly commits; HTTP 200 with `"status":"ok"` when **every** entry is a replay. `accepted`/`replayed` are integer counts. `resolutions` has one result per entry, in request order, each carrying `key`, `replicaId`, `operationId`, the selected string `value`, and `policy`. A pure-replay batch appends no log records.
- All new operations commit together once, so batch repairs share the single global commit order: they are exported by `GET /v1/sync/operations`, appear in per-key audit streams and audit digests, move every metrics counter and the verification digest, are addressable in the per-operation archive, are persisted to `--data-file` (with their policy bindings) in one atomic commit, and recover identically after a restart. The policy binding stays local to the resolving replica: an importing replica holds the operation without the binding, so replaying the batch entry there is an operation conflict.
- The response body is compact JSON with no insignificant whitespace and ends with exactly one newline; every numeric field is a JSON integer.
- With `--data-file`, the whole batch (operations and bindings together) is persisted before the 201 response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory and the file exactly as they were — the batch can be retried. After a restart the results, replay `200`, and conflict `409` are identical to a process that never restarted.

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
- conflict repairs accepted through `POST /v1/states/{key}/resolve` or `POST /v1/states/{key}/resolve/auto`.

It never contains operations for other keys, identical replays (`200`), conflicting or malformed requests (`409`/`400`), or uncommitted requests.

- `after` is the number of this key's records already skipped (a per-key 0-based resume cursor); it defaults to `0`. It counts only records for the path key — operations for other keys do not consume cursor positions. `after=N` returns the key's records committed after the first `N` of *that key's* records, and `after` equal to the key's current record count is a valid empty tail (a key with no history therefore accepts only `after=0`).
- `limit` defaults to `100` and must be between `1` and `100`.
- A successful HTTP 200 response is `{"operations":[...],"nextCursor":N,"hasMore":bool}`, identical in shape to sync export; `nextCursor` is the number of the key's records skipped after this page — feed it back as the next `after`. A key with no history returns HTTP 200 with an empty page.
- The filtered list, the page slice, `nextCursor`, and `hasMore` are computed from a single snapshot under the same commit lock used by local writes, sync imports, and resolutions, so the three values always agree even while commits are in flight. An import batch commits as one indivisible segment of the global order: its records for the key appear consecutively in the audit stream, and a read can never observe half a batch.
- A negative, blank, or non-ASCII-decimal `after`/`limit` (signs, decimals, whitespace, and non-ASCII numerals are all rejected), a repeated or unknown query parameter, a `limit` outside `1-100`, or an `after` greater than the key's current record count returns HTTP 400 with `{"error":"invalid_request"}`. Unknown route shapes (for example `/v1/audit/keys/{key}/operations/extra`) return HTTP 404.

With `--data-file`, the audit reads exactly the same durable log that sync export and recovery use: after a restart the per-key order, page boundaries, cursor resume, stale-write records, and accepted repair records are identical to a process that never restarted. A durable commit failure leaves no audit record (the operation neither reaches memory nor the file), and a conflicting import batch is rejected as a whole and likewise leaves no audit trace.

### Per-key audit-integrity digest

`GET /v1/audit/keys/{key}/digest` returns an integrity summary over one key's audit stream. It always returns HTTP 200 — even for a key that has never been written — with a UTF-8 JSON object containing exactly three fields:

```json
{"algorithm":"sha256","digest":"<64 lowercase hex chars>","operations":4}
```

- `algorithm` is always `"sha256"`.
- `digest` is the 64-character lowercase hexadecimal SHA-256 of the canonical audit-stream bytes described below.
- `operations` is the number of accepted operations for the key — the length of the stream returned by `GET /v1/audit/keys/{key}/operations` (counting every page).

The digest covers the key's **entire audit stream** in global commit order: every first-accepted operation for the key, including stale writes whose clock was already dominated (and which therefore added no candidate) and conflict repairs accepted through `POST /v1/states/{key}/resolve` or its deterministic variant `POST /v1/states/{key}/resolve/auto`. It never covers operations for other keys, identical replays (`200`), conflicting or malformed requests (`409`/`400`), or operations whose durable commit failed. A key with no history hashes the empty stream.

The hash input is a compact UTF-8 JSON array with one element per accepted operation for the key, in global commit order:

- Each element has the fixed shape `{"replicaId":R,"operation":{"operationId":I,"key":K,"value":V,"clock":C}}` — fields are never reordered, and elements are kept in commit order (never sorted).
- The clock `C`'s component names are sorted lexicographically (Unicode code point order).
- No whitespace appears anywhere. Strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex); every other Unicode code point is written literally.

The SHA-256 is computed over exactly those bytes; for a key with no history the input is `[]`.

The filtered stream, the `operations` count, and the hashed bytes are all produced from one snapshot under the same commit lock used by local writes, sync-import batches, and repairs, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair, and `operations` always agrees with the hashed records. The request is strictly read-only — it modifies neither memory, logs, checkpoints, nor the data file (no temp file is created) — and its response uses the same explicit `Content-Length` contract as the other endpoints.

The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment (for example `/v1/audit/keys//digest` or `/v1/audit/keys/{key}/digest/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check, so a missing segment together with a query parameter is still 404. A percent-encoded key segment is decoded just as for the audit stream.

With `--data-file`, the stream is rebuilt from the recovered log on startup, so the same recovery history yields the identical digest and `operations` count before and after a restart. Persistence failures and rejected (conflicting) batches never enter the log and therefore cannot influence the digest, either before or after a restart.

### Per-operation archive query

`GET /v1/replicas/{replicaId}/operations/{operationId}` locates one **first-accepted operation** by its `(replicaId, operationId)` identity. Both path segments are percent-decoded like every other route and must be non-empty after decoding.

- An accepted identity returns HTTP 200 with a UTF-8 JSON object containing exactly two fields: `{"replicaId":R,"operation":{...}}`, where `operation` carries exactly `operationId`, `key`, `value`, and `clock` with their committed values.
- Every accepted record is addressable: ordinary writes, stale writes whose clock was already dominated, manual and automatic conflict repairs, and sync-imported records all appear under the identity they were committed with.
- An identity that was never first-accepted returns HTTP 404 with `{"error":"not_found"}`. Identical replays add no record, and conflicting (`409`), invalid (`400`), or undurably-committed requests never enter the archive, so they stay 404.
- The lookup runs under the same commit lock used by local writes, sync imports, repairs, and checkpoints, so the response always describes a committed snapshot. The request is strictly read-only — it changes no metrics, candidates, audit streams, checkpoints, logs, or the data file — and its response uses the same explicit `Content-Length` contract as the other endpoints.
- The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`. A missing, empty, or extra path segment (for example `/v1/replicas//operations/{operationId}` or `/v1/replicas/{replicaId}/operations/{operationId}/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check, so a malformed path together with a query parameter is still 404.

With `--data-file`, the identity index is rebuilt from the recovered log on startup, so successful results, the 404 boundary, and error statuses are identical before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

### Per-operation causal ancestor chain

`GET /v1/causal/{replicaId}/{operationId}?after=N&limit=N` returns a read-only page of one operation's **strict causal predecessors**. Both path segments are percent-decoded like every other route and must be non-empty after decoding; an identity that was never first-accepted returns HTTP 404 with `{"error":"not_found"}`, and the route-shape check takes precedence over every other check.

A strict predecessor is a first-accepted record committed **before** the source operation in the shared log whose clock is strictly smaller than the source operation's clock — the source clock dominates it (missing components count as 0, and domination already requires the clocks to differ). The source operation itself never appears. Stale writes and accepted conflict repairs are ordinary committed records and participate like any other; identical replays (`200`), conflicting or malformed requests (`409`/`400`), and uncommitted writes never enter the log, so they can never appear.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"ancestors":[{"operation":{"clock":{"r1":1},"key":"color","operationId":"op-1","value":"blue"},"relation":"direct","replicaId":"r1"}],"cursor":1,"more":false,"operation":{"operation":{"clock":{"r1":1,"r2":1},"key":"size","operationId":"op-2","value":"large"},"replicaId":"r2"}}
```

- `operation`: the source record in the per-operation archive shape `{"replicaId":R,"operation":{"operationId","key","value","clock"}}`.
- `ancestors`: one page of the strict predecessors in the shared log's global commit order. Each entry preserves the archive record content and adds `relation`: `"direct"` when no other strict predecessor's clock dominates the record's clock, `"transitive"` otherwise.
- `cursor`: the number of predecessors skipped after this page — feed it back as the next `after`.
- `more`: whether further predecessors remain.

Paging follows the sync-export rules: `after` is the number of predecessors already skipped (a 0-based resume cursor) and defaults to `0`; `limit` defaults to `100` and must be between `1` and `100`. A legal identity with no predecessors — or an `after` equal to the predecessor count — returns HTTP 200 with an empty `ancestors` list. A negative, blank, or non-ASCII-decimal `after`/`limit`, a repeated or unknown query parameter, a `limit` outside `1-100`, or an `after` past the predecessor count returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state. A missing, empty, or extra path segment (for example `/v1/causal/{replicaId}` or `/v1/causal/{replicaId}/{operationId}/`) returns HTTP 404 with `{"error":"not_found"}`.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

The source record, the predecessor list, the relation classification, and the paging boundaries are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovered state yields the same source, the same predecessor relations, and the same pages before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route.

### Two-operation causal slice comparison

`GET /v1/causal/compare?leftReplicaId=R&leftOperationId=O&rightReplicaId=R&rightOperationId=O&after=N&limit=N` compares the strict causal predecessor slices of two first-accepted operations in one read-only report. Unlike the single-operation chain, both identities are carried as **query parameters** (percent-decoded like every route value); all four are required and must be non-empty. An identity that was never first-accepted on either side returns HTTP 404 with `{"error":"not_found"}`, and the route-shape check takes precedence over every other check.

Each side's strict predecessors follow the single-operation chain rules exactly: first-accepted records committed **before** that side's source in the shared log whose clocks its source clock strictly dominates, with the source itself never appearing. Stale writes and accepted conflict repairs participate like any other committed record; identical replays (`200`), conflicting or malformed requests (`409`/`400`), and uncommitted writes never enter the log and so never appear.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"difference":{"leftOnly":0,"rightOnly":2,"shared":2},"left":{"cursor":2,"more":false,"operation":{"operation":{"clock":{"r1":2,"r2":1},"key":"d","operationId":"o4","value":"y"},"replicaId":"r1"},"predecessors":[{"operation":{"clock":{"r1":1},"key":"a","operationId":"o1","value":"v"},"relation":"transitive","replicaId":"r1"},{"operation":{"clock":{"r1":1,"r2":1},"key":"b","operationId":"o2","value":"w"},"relation":"direct","replicaId":"r2"}]},"relation":"right_dominates_left","right":{"cursor":4,"more":false,"operation":{"operation":{"clock":{"r1":2,"r2":2,"r3":1},"key":"e","operationId":"o5","value":"z"},"replicaId":"r2"},"predecessors":[]}}
```

- `left` and `right` each carry exactly four fields:
  - `operation`: the side's source record in the per-operation archive shape `{"replicaId":R,"operation":{"operationId","key","value","clock"}}`.
  - `predecessors`: one page of that side's strict predecessors in the shared log's global commit order. Each entry preserves the archive record content and adds `relation`: `"direct"` when no other strict predecessor **of that side** dominates the record's clock, `"transitive"` otherwise — the same classification as the single-operation chain, computed independently per side.
  - `cursor`: the number of that side's predecessors skipped after this page — feed it back as the next `after`.
  - `more`: whether further predecessors remain on that side.
- `relation`: the ordering of the two **source clocks** — `"left_dominates_right"` when the left source clock dominates the right source clock, `"right_dominates_left"` for the reverse, and `"concurrent"` when neither dominates the other (including equal clocks; missing components count as 0).
- `difference`: `{"shared":S,"leftOnly":L,"rightOnly":R}`, three JSON integer counts over the two sides' predecessor **identity sets**. An identity is the accepted record `(replicaId, operationId)`, de-duplicated per side before counting: `shared` names identities present on both sides, `leftOnly`/`rightOnly` the rest.

Paging follows the sync-export rules with one cursor shared by both sides: `after` (default `0`) is the number of predecessors skipped **on both sides together** — both pages start at the same offset — and `limit` (default `100`, `1`-`100`) bounds each page independently. A side shorter than the offset returns an empty array for that page (so an `after` at or past one side's count is still valid while the other side continues), and `cursor`/`more` are reported per side. Paging only trims the two predecessor pages: **the relation and the difference counts are always computed from the complete, unpaged predecessor sets.** An `after` past the larger of the two predecessor counts returns HTTP 400 with `{"error":"invalid_request"}`. A missing, unknown, or repeated parameter (any of the six names), a blank or missing identity value, a negative or non-ASCII-decimal `after`/`limit`, signs, decimals, whitespace, or a `limit` outside `1-100` likewise return HTTP 400. A missing, empty, or extra path segment (for example `/v1/causal/compare/`, `/v1/causal/compare/extra`, or `/v1/causal`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and identity checks.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

The two source records, both complete predecessor sets, the relation, the difference counts, and the paging boundaries are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory, candidates, logs, checkpoints, audit streams, metrics, nor the data file, and it creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovered state yields the same two sides, relation, difference counts, and pages before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

### Two-operation causal difference with minimal explanation

`GET /v1/causal/diff?leftReplicaId=R&leftOperationId=O&rightReplicaId=R&rightOperationId=O&after=N&limit=N` returns a read-only report that makes the causal difference between two first-accepted operations directly visible. It reuses the comparison route's four identity query parameters and its strict predecessor rules: all four identities are required and must be non-empty; each side's predecessors are the first-accepted records committed **before** that side's source in the shared log whose clocks its source clock strictly dominates, with the source itself never appearing. Stale writes and accepted conflict repairs participate like any other committed record; identical replays (`200`), conflicting or malformed requests (`409`/`400`), and uncommitted writes never enter the log and so never appear. An identity that was never first-accepted on either side returns HTTP 404 with `{"error":"not_found"}`, and the route-shape check takes precedence over every other check.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly nine fields:

```json
{"cursor":4,"difference":{"leftOnly":0,"rightOnly":2,"shared":2},"explanation":[{"from":{"operationId":"o3","replicaId":"r3"},"side":"right","to":{"operationId":"o5","replicaId":"r2"}},{"from":{"operationId":"o4","replicaId":"r1"},"side":"right","to":{"operationId":"o5","replicaId":"r2"}}],"left":{"operation":{"operation":{"clock":{"r1":2,"r2":1},"key":"d","operationId":"o4","value":"y"},"replicaId":"r1"}},"leftOnly":[],"more":false,"right":{"operation":{"operation":{"clock":{"r1":2,"r2":2,"r3":1},"key":"e","operationId":"o5","value":"z"},"replicaId":"r2"}},"rightOnly":[{"operation":{"clock":{"r3":1},"key":"c","operationId":"o3","value":"x"},"relation":"direct","replicaId":"r3"},{"operation":{"clock":{"r1":2,"r2":1},"key":"d","operationId":"o4","value":"y"},"relation":"direct","replicaId":"r1"}],"shared":[{"operation":{"clock":{"r1":1},"key":"a","operationId":"o1","value":"v"},"relation":"transitive","replicaId":"r1"},{"operation":{"clock":{"r1":1,"r2":1},"key":"b","operationId":"o2","value":"w"},"relation":"transitive","replicaId":"r2"}]}
```

- `left` and `right`: the two sides' source operations, each as `{"operation": ...}` in the per-operation archive shape `{"replicaId":R,"operation":{"operationId","key","value","clock"}}`.
- `shared`, `leftOnly`, `rightOnly`: the three predecessor evidence groups — identities present on both sides, only on the left, and only on the right. Each group is ordered by the shared log's global commit order, and each entry keeps the comparison endpoint's predecessor record shape: the archive record content plus `relation` (`"direct"`/`"transitive"`), classified once against the merged evidence pool of the three groups so each identity has one stable classification.
- `difference`: `{"shared":S,"leftOnly":L,"rightOnly":R}` — the same three integer counts as the comparison endpoint, over the complete de-duplicated predecessor identity sets, never the current page.
- `explanation`: the compressed causal explanation. It locates only the minimal one-sided boundary: a `leftOnly` or `rightOnly` predecessor that no other one-sided predecessor on the same side dominates. Shared evidence, and one-sided evidence that is itself covered by another one-sided record on that side, never appear, so the list is the smallest set that still differentiates the two sources. Entries keep global commit order and each carries exactly three fields: `from` (the boundary identity `{"replicaId","operationId"}`), `to` (that side's source identity), and `side` (`"left"` or `"right"`).
- `cursor`: the number of merged predecessors skipped after this page — feed it back as the next `after`.
- `more`: whether further merged predecessors remain.

Paging runs over one stable merge of the three groups — `shared`, then `leftOnly`, then `rightOnly`, each in global commit order: `after` (default `0`) skips that many records of the merged sequence, and `limit` (default `100`, `1`-`100`) bounds the current page. The window is then partitioned back into the `shared`/`leftOnly`/`rightOnly` arrays, so resuming with the returned `cursor` continues exactly where the previous page ended. The `difference` counts and the minimal `explanation` are always computed from the complete, unpaged predecessor sets. A missing, unknown, repeated, blank, or malformed parameter (any of the six names), a negative or non-ASCII-decimal `after`/`limit`, signs, decimals, whitespace, a `limit` outside `1-100`, or an `after` past the merged predecessor count returns HTTP 400 with `{"error":"invalid_request"}`; an `after` exactly equal to the merged count is a valid empty page. A missing, empty, or extra path segment (for example `/v1/causal/diff/`, `/v1/causal/diff/extra`, or `/v1/causal`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and identity checks.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

Both source records, the complete predecessor sets, the three groups, the difference counts, the explanation, and the paging boundaries are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair, and a concurrent commit is seen only as a complete old or new snapshot. The request is strictly read-only — it modifies neither memory, candidates, logs, checkpoints, audit streams, metrics, nor the data file, and it creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovered state yields the same sources, groups, difference counts, explanation, and pages before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route, returning HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` response header on failure (and `/health` stays anonymous).

### Replication-snapshot consistency verification

`GET /v1/replication/snapshot` returns a read-only consistency summary over one committed snapshot: the current candidate state, the accepted-log position, and the whole checkpoint mapping are read together. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly seven fields:

```json
{"candidateDigest":"<64 lowercase hex chars>","candidateVersions":3,"checkpoints":{"peer-a":2},"keys":2,"logCursor":4,"snapshotDigest":"<64 lowercase hex chars>","status":"ok"}
```

- `status`: the verification conclusion, always `"ok"` — the summary is assembled atomically from one commit, so it is internally consistent by construction.
- `candidateDigest`: the 64-character lowercase hexadecimal SHA-256 of the canonical candidate snapshot, following exactly the rules of `GET /v1/verification/digest` (it covers only the current candidate sets).
- `snapshotDigest`: the 64-character lowercase hexadecimal SHA-256 of the canonical snapshot bytes described below.
- `logCursor`: the number of first-accepted operations in the shared log — the sync-export resume cursor at the tail of the log.
- `keys` and `candidateVersions`: the same counts reported by `GET /v1/metrics` and `GET /v1/verification/digest`.
- `checkpoints`: the full `{peerId: cursor}` mapping of sender-side replication progress; an empty mapping is reported as `{}`.

The snapshot-digest input is a compact UTF-8 JSON array of exactly three elements, in this fixed order: the candidate digest (as a hex string), the log cursor (a JSON integer), and the checkpoint mapping with peer ids sorted lexicographically (an empty mapping is kept as `{}`). No whitespace appears anywhere, and strings escape exactly as in the verification-digest rules — only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex). The SHA-256 is computed over exactly those bytes. Both digests are 64-character lowercase hexadecimal strings; the counts and the cursor appear only as JSON integers.

The candidate state, the log cursor, and the checkpoint mapping are read from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read observes either the old or the new complete state, never half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the log and the checkpoints are rebuilt identically during recovery, so the same state yields the same verification result before and after a restart.

The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}` without reading any state. A missing or extra path segment, an unknown route, or a trailing slash (for example `/v1/replication/snapshot/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
