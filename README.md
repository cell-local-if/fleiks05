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

All fourteen POST endpoints (`POST /v1/replicas/{replicaId}/operations`, `POST /v1/sync/operations`, `POST /v1/states/{key}/resolve`, `POST /v1/states/{key}/resolve/auto`, `POST /v1/resolve/auto/batch`, `POST /v1/resolve/auto/plan`, `POST /v1/sync/peers/{peerId}/checkpoint`, `POST /v1/sync/peers/{peerId}/acknowledge`, `POST /v1/transactions/apply`, `POST /v1/replication/apply`, and the read-only `POST /v1/states/{key}/causal-at`, `POST /v1/replication/compare`, `POST /v1/replication/plan`, and `POST /v1/replication/consensus`) share one body-size contract:

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

The `checkpoints` section is optional and holds sender-side replication cursors (see below); a file written before checkpoints existed contains only `version` and `operations`, and recovers with no registered checkpoints. The `policies` section is likewise optional and holds the automatic-resolution policy bindings (see below): one `{"replicaId","operationId","policy"}` record per accepted automatic resolution, committed atomically with its operation. The `transactions` section is likewise optional and holds the atomic-transaction bindings (see below): one `{"transactionId","operations"}` record per accepted transaction, committed atomically with its operations. The `acks` section is likewise optional and holds the consumption receipts (see below): one `{"peerId","ackId","cursor","operations"}` record per accepted acknowledgement, committed atomically with the checkpoint advance it caused. The `policyEvents` section is likewise optional and holds the scope-policy change history (see below): one `{"sequence","digest","tokens"}` record per successful scope-policy hot reload, committed atomically with the policy replacement; a file written before policy auditing existed recovers with an empty history. `version` stays `1`: the supplemented format is backward compatible, and an old file is upgraded on disk the first time a checkpoint (or any other new commit) is persisted.

### Optional bearer-token authentication

The service is anonymous by default: without authentication options every documented behavior above is unchanged. There are two mutually exclusive ways to enable authentication — at most one of `--auth-token-file` and `--scope-policy-file` may be passed; supplying both makes startup fail with exit code 2 before any port is bound.

#### Single token mode: `--auth-token-file PATH`

Pass `--auth-token-file PATH` to require one bearer token on every endpoint except the health probe. The single token is unrestricted — it authorizes every documented GET and POST exactly as before scopes existed:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --auth-token-file ./var/token
```

#### Scope policy mode: `--scope-policy-file PATH`

Pass `--scope-policy-file PATH` to require a bearer token that also carries an authorization scope. The file must be a readable **regular** UTF-8 JSON **object** whose keys are tokens and whose values are scope arrays:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --scope-policy-file ./var/scopes.json
```

```json
{
  "reader-token-1": ["read"],
  "writer-token-1": ["write"],
  "admin-token-1": ["read", "write", "admin"]
}
```

- Every key must be a non-empty ASCII printable token — bytes 0x21-0x7E, i.e. no whitespace, quotes, or non-ASCII characters — and no token key may repeat. Every value must be a **non-empty** array whose elements are chosen only from `"read"`, `"write"`, and `"admin"`, with no repetition.
- Scopes authorize HTTP methods: `read` accesses every documented GET and the read-only POSTs — the batch preview `POST /v1/resolve/auto/plan`, the causal-slice query `POST /v1/states/{key}/causal-at`, the cross-replica comparison `POST /v1/replication/compare`, the cross-replica synchronization plan `POST /v1/replication/plan`, and the multi-replica convergence-consensus summary `POST /v1/replication/consensus`; `write` submits the eight state-changing business POST endpoints; and `admin` covers both classes (it implies read and write) plus the scope-policy reload endpoint and the scope-policy change-audit and audit-verification endpoints. Apart from the read-only plan preview, causal-slice query, cross-replica comparison, synchronization plan, and convergence consensus, the state-changing POST endpoints are not reachable with only `read` and the GET endpoints are not reachable with only `write`; neither `read` nor `write` alone reaches the admin-only reload and audit endpoints. `GET /health` stays anonymous in every mode.
- The policy file is read and validated **before the service begins listening**. A missing, unreadable, or non-regular target (for example a directory), a non-UTF-8 or incomplete/invalid JSON document, a non-object root, a duplicate token key, an illegal token, an unknown scope value, an empty value, or a duplicated scope all make startup fail with exit code 2, exactly like a rejected token or data file: no port is bound and neither tokens nor scopes are ever printed.

#### Runtime policy reload: `POST /v1/admin/scope-policy/reload`

In scope policy mode an administrator can atomically replace the live token/scope boundary without a restart. The reload re-reads **only the file supplied with `--scope-policy-file` at startup** — the request never names, and the server never accepts, another path. The endpoint is published **only** in scope policy mode: single-token mode and anonymous mode answer it with `404 {"error":"not_found"}`.

- The request must carry an `admin` token. Authentication failure is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; an authenticated token without the `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge. Neither rejection reads the body.
- The request body must be exactly the empty JSON object `{}` (JSON whitespace around it is allowed). Malformed JSON, a non-object document, or any field — known or unknown — is HTTP 400 with `{"error":"invalid_request"}`.
- The shared Content-Length priority applies: a missing/malformed declaration is 400 and an over-limit declaration is 413, both **before** authentication. The path must be exactly `/v1/admin/scope-policy/reload` — a missing segment, extra segment, or trailing slash is `404 {"error":"not_found"}`, decided before any query or body check. Any query parameter is 400 `invalid_request`, and that check precedes the body check even on the correct route.
- The configured file must remain a readable **regular** UTF-8 JSON object satisfying the same token and scope constraints as at startup. If it is missing, unreadable, non-regular, or cannot be read, the response is HTTP 503 with `{"error":"policy_unavailable"}`. If it is readable but its content is invalid, the response is HTTP 409 with `{"error":"policy_conflict"}` and the **old** mapping stays fully in force.
- On success the response is HTTP 200 with exactly three fields, in this order:
  `{"status":"reloaded","policyDigest":"<64 lowercase hex>","tokens":<non-negative integer>}`. `policyDigest` is the SHA-256 of the policy file's **raw UTF-8 bytes** (the bytes are hashed, never canonicalized) written as 64 lowercase hexadecimal characters, and `tokens` is the number of token entries (an empty-object policy reports 0).
- The replacement is one atomic commit serialized across concurrent reloads: every request observes either the whole old policy or the whole new one, and a request already authenticated continues to execute under the policy revision in force when it authenticated, unaffected by a later reload. Authentication rejections, permission rejections, and failed reloads change no business state and create no temporary file. A restart still recovers the policy from the configured file (never from the data file); health stays anonymous and every existing write/query/idempotency/persistence behavior is unchanged.

#### Auditing scope-policy changes: `GET /v1/admin/scope-policy/audit`

In scope policy mode an administrator can page the history of successful policy replacements made through the reload endpoint. The query is strictly read-only and is published **only** in scope policy mode: single-token mode and anonymous mode answer it with `404 {"error":"not_found"}` (after the single-token authentication check, exactly like the reload endpoint).

- The request must carry an `admin` token. A missing, duplicated, or malformed `Authorization` header, or a token mismatch, is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; an authenticated token without the `admin` scope is HTTP 403 with `{"error":"forbidden"}` and **no** challenge. The scope decision runs before the query is parsed, so a bad query for a non-admin token is still 403.
- The path must be exactly `/v1/admin/scope-policy/audit`. A missing segment, an extra segment, a trailing slash, or any unknown route is `404 {"error":"not_found"}`, decided **before** any query check — a wrong path shape together with an invalid query is still 404.
- The query accepts exactly two parameters, both **required**: `after` and `limit` are ASCII decimal integers; `after` is the 0-based resume cursor into the reload history (it starts at `0`) and `limit` must be between `1` and `100`. A missing parameter, an out-of-range `limit`, a negative value, a blank or whitespace-bearing value, a sign, a decimal point, non-ASCII numerals, a repeated `after`/`limit`, or an unknown parameter is HTTP 400 with `{"error":"invalid_request"}`. An `after` equal to the current history length is a valid stable empty page; an `after` past it is HTTP 400.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly six fields:

```json
{"events":[{"sequence":1,"digest":"<64 lowercase hex chars>","tokens":3}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","eventsCount":1}
```

- `events`: one page of the policy-change history in the order the reloads succeeded. Each item carries exactly three fields: `sequence` (its 1-based, continuous position in the complete history), `digest` (the 64-character lowercase hexadecimal SHA-256 of the reloaded policy file's **raw UTF-8 bytes** — the same digest the reload response reported; the bytes are hashed, never canonicalized), and `tokens` (the number of token entries the reloaded policy carried; an empty-object policy records `0`).
- `nextCursor`: the number of events skipped after this page — feed it back as the next `after`; `hasMore` reports whether further events remain.
- `algorithm` is always `"sha256"`, and `eventsCount` counts the **complete** history, never just the page.
- `digest` summarizes the whole history, so it is identical on every page. The hash input is a compact UTF-8 JSON array with one element per successful reload in commit order, each written with its fields in the fixed order `{"sequence":N,"digest":"<64 lowercase hex>","tokens":N}` — no whitespace anywhere, numbers as plain JSON integers, strings escaping only the quote, the backslash, and U+0000-U+001F control characters. An empty history hashes the empty array `[]`.

Only successful reloads create events: authentication/permission rejections, malformed requests, a `409 policy_conflict`, a `503 policy_unavailable`, and Content-Length `400`/`413` rejections never enter the history. Each successful reload commits the new live policy **and** its event as one atomic step — the durable event write happens before the live mapping swaps — so if the durable commit fails the response is HTTP 500 with `{"error":"internal_error"}` and both the old policy and the old history stay fully in force. The page, cursors, summary, and count are computed from one snapshot under the same serialization as reloads, so a query observes only the complete old or new history.

With `--data-file`, events live in the optional `policyEvents` section described above: a successful reload is made durable in the same atomic commit protocol (`write temp file → fsync → rename → fsync directory`) before its `200`, and the history is rebuilt identically on restart with stable sequence numbers and digests; an old file without the section recovers with an empty history. The policy tokens and scopes themselves are still never written to the data file — an event records only the sequence, the raw-byte digest, and the entry count. The audit query itself is read-only: it creates no temporary file and changes neither memory nor the data file.

#### Verifying scope-policy change-history integrity: `GET /v1/admin/scope-policy/audit/verify`

In scope policy mode an administrator can incrementally export the same successful-reload history as `GET /v1/admin/scope-policy/audit` **and** receive an independent integrity conclusion over it. The query is strictly read-only and, like the change-audit and reload entries, is published **only** in scope policy mode: single-token mode and anonymous mode answer it with `404 {"error":"not_found"}` (after the single-token authentication check, exactly like the other two entries).

- The request must carry an `admin` token. A missing, duplicated, or malformed `Authorization` header, or a token mismatch, is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; an authenticated token without the `admin` scope is HTTP 403 with `{"error":"forbidden"}` and **no** challenge. The scope decision runs before the query is parsed, so a bad query for a non-admin token is still 403.
- The path must be exactly `/v1/admin/scope-policy/audit/verify`. A missing segment, an extra segment, a trailing slash, or any unknown route is `404 {"error":"not_found"}`, decided **before** any query check — a wrong path shape together with an invalid query is still 404, and the plain audit route `/v1/admin/scope-policy/audit` is unchanged.
- The incremental-export parameters `after` and `limit` are both **required**: `after` is the number of successful events already skipped (a 0-based resume cursor that starts at `0`) and `limit` accepts only an ASCII decimal integer between `1` and `100`. A missing, repeated, or unknown parameter, a blank or empty value, a sign, a decimal point, whitespace-bearing or non-ASCII numerals, a `limit` outside `1-100`, or an `after` past the current event count is HTTP 400 with `{"error":"invalid_request"}`. An `after` equal to the current history length is a valid stable empty page. A rejected query changes no state.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly seven fields:

```json
{"events":[{"sequence":1,"digest":"<64 lowercase hex chars>","tokens":3}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","eventsCount":1,"verification":{"status":"ok","missingSequences":[],"duplicateSequences":[],"outOfRangeSequences":[],"digestMismatches":[]}}
```

- `events`, `nextCursor`, `hasMore`, `algorithm`, `digest`, and `eventsCount` are exactly the six fields of the plain change-audit response: the events are exported in the order the reloads succeeded (each carrying `sequence`, `digest`, `tokens`), `nextCursor` is the number of events skipped after this page (feed it back as the next `after`), and `hasMore` reports whether further events remain.
- The summary always covers the **complete** history, never just the page: `algorithm` is `"sha256"`, `eventsCount` counts every successful reload, and `digest` is the SHA-256 of the same canonical compact UTF-8 JSON array the plain audit uses — one `{"sequence":N,"digest":"<64 lowercase hex>","tokens":N}` element per successful reload in commit order with that fixed field order, no whitespace anywhere, plain JSON integers, and strings escaping only the quote, the backslash, and U+0000-U+001F control characters. An empty history hashes the empty array `[]`. Paging trims only the exported `events` page; the digest and the count are identical on every page of one snapshot.
- `verification` carries the independent integrity conclusion over the complete history (also independent of the page), with exactly five fields:
  - `status`: `"ok"` when every anomaly list below is empty — the claimed sequences are exactly the continuous 1-based range `1..eventsCount` with no missing, duplicate, or out-of-range position and every recorded digest has the 64-lowercase-hex SHA-256 shape — otherwise `"broken"`. An empty history is intact: `"ok"`.
  - `missingSequences`: each unclaimed position in `1..eventsCount`, marked with `{"eventsIndex":I,"sequence":S}` — `eventsIndex` is the 0-based history index where the sequence is missing (`S - 1`), and `sequence` is the 1-based missing position.
  - `duplicateSequences`: each later event claiming a sequence an earlier event already claimed, marked with `{"eventsIndex":I,"sequence":S}` (the repeated occurrence only).
  - `outOfRangeSequences`: each event whose `sequence` is not an integer in `1..eventsCount` (zero, negative, past the count, or non-integer), marked with `{"eventsIndex":I,"sequence":S}`.
  - `digestMismatches`: each event whose recorded `digest` is not exactly 64 lowercase hexadecimal characters, marked with `{"eventsIndex":I,"sequence":S,"expected":null,"observed":D}`; `expected` is null because an event retains only the recorded digest, never the policy bytes it was computed from.

  `eventsIndex` is always the anomaly's 0-based position in the complete history and `sequence` its claimed 1-based position. The live history is appended one verified event at a time (each successful reload commits its event before the policy swap), so the conclusion is `"ok"` by construction; the scan independently re-checks the numbering, the digest shapes, and the event count against the actual snapshot.

The page slice, cursor, remaining flag, digest, count, and verification are all computed from one snapshot of the complete history under the same serialization as reloads, so a concurrent hot reload is observed only as the whole old or the whole new history, never a mix. The query is strictly read-only: it changes neither memory nor the data file and creates no temporary file. With `--data-file`, the history is rebuilt identically during recovery, so a restart reports the same event pages, full-history digest, `eventsCount`, and `verification` conclusion; an old file without the `policyEvents` section verifies as an intact empty history. When bearer-token authentication is enabled, `/health` stays anonymous and every other authentication behavior is unchanged.

#### Shared authentication contract

Both enabled modes share one request contract:

- `GET /health` stays anonymous. Every other route — known or unknown, GET or POST — requires the request to carry **exactly one** `Authorization` header whose value is exactly `Bearer ` (one space) followed by a configured token. A missing, duplicated, or malformed header and any token mismatch return HTTP 401 with `{"error":"unauthorized"}` and a `WWW-Authenticate: Bearer` response header — before route matching, query parsing, the commit lock, any state read, any data-file access, and any POST body read. Token comparison uses the standard library's constant-time primitive.
- When scope policy mode is enabled and the token is valid but does not carry the scope the method requires (nor `admin`), the response is HTTP 403 with a body of exactly `{"error":"forbidden"}` and **no** `WWW-Authenticate` header. The scope decision runs before route matching and query validation: an unauthorized request is never downgraded into, nor upgraded past, a `404`, `400`, or `409` business result, and it never enters the commit lock.
- A rejected request changes nothing: it creates no temporary file and leaves memory, logs, checkpoints, audit streams, candidates, and the data file exactly as they were; the token and policy are never leaked in responses or logs.
- The business POST endpoints keep their Content-Length priority: a missing/malformed declaration still returns 400 and an over-limit declaration still returns 413 **before** authentication (and therefore before the scope check). When the declared length is valid but authentication or the scope check fails, the response (401/403) is sent **without reading the body** and the connection is closed. The reload endpoint shares exactly this priority and no-body-reading guarantee.
- Once a request carries a credential with the required scope (or uses single token mode, or authentication is disabled), every existing behavior — success codes, 400/404/409/500, paging, digests, idempotency, concurrency, and recovery — is exactly as documented.
- Neither authentication configuration is ever written to the data file: a `--data-file` restart recovers only operations and the other sections, and the token or scope policy is supplied again (or not) via the command line on each start. The data file format is unchanged and old data files stay compatible.

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

### Reading historical state

`GET /v1/states/{key}/at?cursor=N` returns a read-only report of one key's candidate state **as it was after the first `cursor` records of the shared accepted-operation log**, replayed from the empty state in commit order. The replay covers exactly what the log holds — first-accepted ordinary writes, stale writes whose clock was already dominated, accepted conflict repairs, and sync-imported records — and nothing else: identical replays (`200`), conflicting or malformed requests (`409`/`400`), uncommitted requests, and records whose durable commit failed never enter the log and so never move the replayed state.

The `cursor` parameter is required and must appear exactly once as a non-negative ASCII decimal integer: `cursor=0` replays nothing (the empty state, so every key answers HTTP 404 with `{"error":"not_found"}`), and `cursor` equal to the log length is exactly the current state — the candidates and the classification match `GET /v1/states/{key}`. A missing, repeated, blank, signed, decimal, whitespace-padded, or non-ASCII-digit `cursor`, any unknown parameter, and a `cursor` past the accepted-log length return HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state. A missing, empty, or extra path segment (for example `/v1/states/{key}/at/extra` or `/v1/states/{key}/at/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.

A key with no candidate at the requested position — one that never appeared, or one that appears only later in the log — returns HTTP 404 with `{"error":"not_found"}`. A successful HTTP 200 response is a compact UTF-8 JSON object with exactly four fields, terminated by a single newline:

```json
{"candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"},{"clock":{"r2":1},"operationId":"op-2","replicaId":"r2","value":"red"}],"cursor":2,"key":"color","status":"conflict"}
```

- `cursor`: the replayed position, as a JSON integer.
- `key`: the requested key (the path segment is percent-decoded like every route).
- `status`: `"resolved"` when every candidate at that position agrees on the value, `"conflict"` otherwise — the same classification as `GET /v1/states/{key}`.
- `candidates`: always an array, even when resolved — the historical view does not collapse to the current query's `value`/`clock` shape. Each entry carries exactly `value`, `clock`, `replicaId`, and `operationId`, sorted by `(replicaId, operationId)` ascending, so the first entry is the same value and clock the current-state query would choose for the same candidate set.

Every number in the response is a JSON integer; no float, negative zero, or non-finite value can appear.

The whole replay runs against one committed snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit and never observes half an import batch. The request is strictly read-only — it modifies neither memory nor the data file and creates no file. With `--data-file`, the accepted log is recovered identically during startup, so the same `cursor` yields the same report before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing or bad credential is HTTP 401, and in scope-policy mode a token without the `read` or `admin` scope is HTTP 403.

### Reading historical state at a causal boundary

`POST /v1/states/{key}/causal-at` returns a read-only report of one key's candidate state as seen at a caller-supplied **vector-clock boundary**, instead of a global log position. The request body is a JSON object with exactly one key:

```json
{"clock":{"r1":3,"r2":1}}
```

- `clock` is an object whose component names are replica ids and whose values are non-boolean, non-negative JSON integers. It may be empty: `{"clock":{}}` names the causal origin, the componentwise-zero boundary. A boundary covers an operation when it is componentwise no smaller than that operation's clock — the minimum boundary that covers an operation is that operation's own clock (missing components count as 0, exactly as in the write semantics).
- Malformed JSON, a non-object body, a missing or unknown field, a duplicated field (including a duplicated clock component), a structurally illegal clock (a non-object `clock`, an empty component name, a boolean, negative, float — including `1.0` and `-0.0` — string, or non-finite value such as `NaN`/`Infinity`/`-Infinity`) all return HTTP 400 with `{"error":"invalid_request"}`.
- The route accepts no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check even on the correct route. A missing, empty, or extra path segment (for example `/v1/states/{key}/causal-at/extra` or `/v1/states/{key}/causal-at/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and body checks.

The query starts from the empty state and replays, in global commit order, only the first-accepted records whose clock is componentwise no greater than the given boundary. The replay covers exactly the records the log holds that fall inside the boundary — ordinary writes, stale writes whose clock was already dominated, accepted conflict repairs, and sync-imported records — and nothing else: records past the boundary are skipped even when they sit earlier in the global log than a covered record, and identical replays (`200`), conflicting or malformed requests (`409`/`400`), uncommitted requests, and records whose durable commit failed never enter the log and so never move the replayed state. Candidate addition and deletion keep the existing vector-clock domination semantics (missing components count as 0): a covered write deletes the candidates its clock dominates, and a covered write whose clock is already dominated inside the slice is replayed as a stale write and adds no version.

A key with no candidate inside the boundary — one that never appeared, or one that appears only past the boundary — returns HTTP 404 with `{"error":"not_found"}`, even when the key appears at a later position in the log. A successful HTTP 200 response is a compact UTF-8 JSON object with exactly four fields, terminated by a single newline:

```json
{"candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}],"clock":{"r1":3,"r2":1},"key":"color","status":"resolved"}
```

- `clock`: the requested boundary, echoed back exactly as sent.
- `key`: the requested key (the path segment is percent-decoded like every route).
- `status`: `"resolved"` when every candidate inside the boundary agrees on the value, `"conflict"` otherwise — the same classification as `GET /v1/states/{key}`.
- `candidates`: always an array, even when resolved. Each entry carries exactly `value`, `clock`, `replicaId`, and `operationId`, sorted by `(replicaId, operationId)` ascending, so the first entry is the same value and clock the current-state query would choose for the same candidate set.

Every number in the response is a JSON integer; no float, negative zero, or non-finite value can appear. The compact encoding and terminator are the same as `GET /v1/states/{key}/at` — no insignificant whitespace, keys sorted, one trailing newline.

The replay runs against one committed snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit and never observes half an import batch. The request is strictly read-only — it modifies neither memory nor the data file, creates no temporary file, and changes neither writes, repairs, sync, audit, nor persistence behavior. With `--data-file`, the accepted log is recovered identically during startup, so the same boundary yields the same report before and after a restart. The route shares the common request contract: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered **before** reading the body and before authentication; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge; and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge. `/health` stays anonymous. A `GET` on the path is an unknown route and answers HTTP 404.

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

### Read-only batch preview (plan)

`POST /v1/resolve/auto/plan` shows a caller what a batch automatic resolution *would* do, without committing anything. The request body is exactly the committing batch's body — the same JSON object with one `resolutions` key, 1-100 entries in request order, distinct target keys, distinct `(replicaId, operationId)` identities, and the same per-field constraints (non-empty strings, structurally valid clocks, and `policy` equal to `"lowest_identity"` or `"highest_identity"`). Malformed JSON, an empty or oversized batch, a duplicate key or identity, an unknown field, an unknown policy, an illegal identity or clock, and any float (including `1.0`, `-0.0`, `NaN`, `Infinity`, or `-Infinity`) return HTTP 400 with `{"error":"invalid_request"}` and no results.

The entries are evaluated **in request order against one complete committed snapshot**, using exactly the per-entry rules of the committing batch, on a staged copy of the store:

- An unseen identity whose key currently holds different-valued candidates and whose clock dominates every current candidate contributes the value the policy would select and counts towards `accepted` — the number of entries a commit would newly create.
- A known identity with the same binding (key, clock, and policy) reports the originally chosen value and counts towards `replayed`; a known identity with a different binding (or an identity committed without a policy binding) returns HTTP 409 with `{"error":"operation_conflict"}`.
- A missing key, candidates that already agree on one value (no value conflict), a candidate set that has changed, or a legal clock that does not dominate the current candidates returns HTTP 409 with `{"error":"resolution_conflict"}`. A structurally illegal clock is the `400 invalid_request` above.

Neither the success case nor a rejection changes any business state: the preview writes no candidates, accepted-log records, policy bindings, checkpoints, audit entries, metrics, or data-file bytes and creates no temporary files. In particular, a previewed identity is not bound, so previewing a request and then committing it is still a fresh `201`.

- Success is always HTTP 200 with `{"status":"planned","resolutions":[...],"accepted":A,"replayed":R}`. The top-level `status` is the fixed string `"planned"`; `accepted` counts entries this request would newly create and `replayed` counts same-binding entries. `resolutions` has one result per entry in request order, each carrying `key`, `replicaId`, `operationId`, the selected string `value`, and `policy`. `accepted`/`replayed` are JSON integers.
- The response body is compact UTF-8 JSON with no insignificant whitespace and ends with exactly one newline; the `400`/`409` error responses for the route carry the same terminator.
- The preview observes a single snapshot: a concurrent commit moves the store directly from one complete snapshot to another, so a returned plan is never a mix of two snapshots and a plan already returned does not change because of a later commit. Repeating the same preview against unchanged state yields the identical response; after committing, the same request previews as replays instead of new entries.
- With `--data-file`, the preview reads the same durable state but never writes it: identical state before and after a restart produces the identical preview (including the accepted/replayed split).
- The route accepts no query parameters: an unknown, repeated, blank, or otherwise illegal parameter returns HTTP 400 with `{"error":"invalid_request"}`; that check precedes body validation. A path with a missing or extra segment, a trailing slash (for example `/v1/resolve/auto/plan/`), or any unknown route returns HTTP 404 with `{"error":"not_found"}`, and the route-shape decision takes priority over the body (an unreadable body on a wrong shape is still 404).
- The common request contract applies: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered before authentication and without reading the body; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge, while `/health` stays anonymous. Because the preview is read-only it is gated by the **read** scope in scope-policy mode: a token carrying `read` or `admin` may call it, a token lacking both is HTTP 403 `{"error":"forbidden"}` with no challenge, and an unauthorized request never reads the body.

### Atomic multi-key transactions

`POST /v1/transactions/apply` commits between 1 and 100 conditional writes to **distinct keys** as one atomic transaction. The body is a JSON object with exactly two keys:

```json
{"transactionId":"tx-1","operations":[{"key":"color","replicaId":"r3","operationId":"tx-op-1","value":"blue","clock":{"r1":1,"r3":1},"candidates":[{"replicaId":"r1","operationId":"op-1"}]}]}
```

- `transactionId` is a non-empty string identifying the transaction. `operations` holds 1-100 entries in request order, each with exactly `key`, `replicaId`, `operationId`, `value`, `clock`, and `candidates`: the fields of an ordinary write (with the initiating replica carried on the entry, as in a sync record) plus the expected pre-commit candidate identity set for the key. Every field obeys the ordinary write constraints: non-empty strings and a clock of non-boolean non-negative integer components containing the entry's `replicaId`.
- `candidates` is the expected set of current candidate identities for the key, each a distinct `{"replicaId","operationId"}` object. An empty list expects the key to hold no current candidates; a non-empty list must match the key's current candidate identities exactly. The candidate set is a set: its order in the request is not significant.
- Each entry clock must contain the entry's `replicaId` and must strictly **dominate every candidate of its expected set** (missing components count as 0, exactly as in the write semantics).
- No two entries may name the same `key`, no two may carry the same `(replicaId, operationId)` identity, and no candidate identity may repeat within an entry.
- A malformed body, an invalid `transactionId`, an empty or oversized batch, a duplicate key, identity, or candidate, an unknown field at the root or on an entry, or a structurally illegal clock all return HTTP 400 with `{"error":"invalid_request"}`; nothing changes. A legal clock that does **not** dominate its expected candidates is also HTTP 400 with `{"error":"invalid_request"}` — the whole transaction is unchanged either way.
- Any query parameter returns HTTP 400 with `{"error":"invalid_request"}`; extra path segments (for example `/v1/transactions/apply/extra`) or a trailing slash return HTTP 404 with `{"error":"not_found"}`.

Entries are validated **in request order** against a staged view of the store. A new `(replicaId, operationId)` commits only when its expected candidate set exactly matches the key's current candidate identities at that position in the sequence. Only when **every** entry's expected state matches and every structure is legal do the operations enter the shared accepted log as one accepted batch — all entries commit together in a single atomic commit, so a concurrent reader sees either the old or the new complete state, never half a transaction.

- The transaction id is bound to the exact entry list: the same `transactionId` replayed with identical entries returns HTTP 200 with `"status":"ok"` and appends no log records — the replay is answered from the committed binding without re-checking the current state. The same `transactionId` with different entries returns HTTP 409 with `{"error":"operation_conflict"}`.
- Within a new transaction, an entry whose `(replicaId, operationId)` is already known with identical operation content is a replay (it adds no log record and skips the state check); a known identity with different content returns HTTP 409 with `{"error":"operation_conflict"}` and the whole transaction is unchanged.
- An expected candidate set that does not match the key's current identities — including a set naming identities the key no longer (or never did) hold, or a state change observed at validation time — returns HTTP 409 with `{"error":"transaction_conflict"}` and the whole transaction is unchanged.
- HTTP 201 with `{"status":"created","transactionId":T,"operations":[...],"accepted":A,"replayed":R}` when at least one entry newly commits; HTTP 200 with `"status":"ok"` when **every** entry is a replay. `accepted`/`replayed` are integer counts. `operations` has one result per entry, in request order, each carrying `key`, `replicaId`, `operationId`, and the committed `value`. The response body is compact JSON with no insignificant whitespace and ends with exactly one newline.
- Transaction operations are ordinary accepted operations in the shared global commit order: they are exported by `GET /v1/sync/operations`, imported by `POST /v1/sync/operations`, appear in per-key audit streams and audit digests, move every metrics counter and the verification digest, and are addressable in the per-operation archive and the causal queries. The transaction binding itself is **local to this replica**: it is persisted with its operations but is not part of the exported sync records, so an importing replica holds the operations without the binding.
- With `--data-file`, the whole transaction (operations and binding together) is persisted in one atomic commit before the 201/200 response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory, the identity index, the bindings, and the file exactly as they were — the transaction can be retried. After a restart the create `201`, replay `200`, and conflict `409` decisions are identical to a process that never restarted.
- The endpoint shares the common request contract: Content-Length is validated before authentication (400/413 first), an invalid or missing bearer token returns HTTP 401 `{"error":"unauthorized"}` without reading the body, and `/health` stays anonymous.

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

#### Picking up operations past a peer's checkpoint

`GET /v1/sync/peers/{peerId}/operations?after=N&limit=N` lets a consuming replica fetch the accepted operations it has not yet consumed, anchored at a checkpoint previously registered with `POST /v1/sync/peers/{peerId}/checkpoint`. The peer only selects the progress anchor: the response is the tail of the **shared accepted-operation log** beginning right after that peer's registered cursor, in global commit order — every accepted record past the cursor is returned whatever its `replicaId`, each item keeping the committed sync record's own `{"replicaId","operation"}` identity and content, exactly as `GET /v1/sync/operations` would export from that position.

- `{peerId}` must be a non-empty percent-decoded path segment, following the same path rules as the checkpoint routes.
- `after` is the number of unconsumed records already skipped **relative to the checkpoint** (a 0-based resume cursor, not an absolute log position); it is required and starts the page at `0`. `after=N` skips the first `N` records after the checkpoint, and `after` equal to the current number of unconsumed records is a valid empty page.
- `limit` is required, must be between `1` and `100`, and (like `after`) accepts only ASCII decimal integers. A missing or repeated `after`/`limit`, a blank, negative, or non-ASCII-decimal value, an unknown parameter, a `limit` outside `1-100`, or an `after` past the current unconsumed record count returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state.
- A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly three fields: `{"operations":[...],"nextCursor":N,"hasMore":bool}`. `operations` preserves the committed record shape and global commit order; `nextCursor` is the cumulative number of unconsumed records skipped after this page (relative to the checkpoint) — feed it back as the next `after`; `hasMore` reports whether further records remain.
- The records come only from the shared accepted log, so stale writes (including those that added no candidate), sync-imported records, and manually or automatically resolved repairs are visible like any other accepted record. Identical replays (`200`), rejected requests (`400`/`409`), uncommitted writes, and requests whose durable commit failed never enter the log and never appear.
- The checkpoint cursor, the page slice, `nextCursor`, and `hasMore` are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the four values always describe a single commit even while commits are in flight.
- The pickup is strictly read-only: it neither advances nor writes the checkpoint and changes neither the accepted log, candidates, audit streams, metrics counters, summaries, nor the data file; it creates no temporary file. Repeated GETs return the same page until the peer separately posts an advanced checkpoint.
- A peer that has never registered a checkpoint returns HTTP 404 with `{"error":"not_found"}`; the checkpoint GET on the same peer remains 404 in that case. On startup with `--data-file`, a recovered checkpoint cursor greater than the recovered log length is a corrupt file and makes the service refuse to start with exit code 2 before it begins listening.
- A missing, empty (`/v1/sync/peers//operations`), or extra (`/v1/sync/peers/{peerId}/operations/extra`, a trailing slash, or a missing segment) path shape returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.
- Every number in the response is a JSON integer, and strings use the same escaping as the other endpoints. With `--data-file`, the checkpoints and log are rebuilt identically during recovery, so pickup results, cursor resume, and the error boundaries are identical before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

#### Acknowledging consumed operations

`POST /v1/sync/peers/{peerId}/acknowledge` creates a verifiable consumption receipt: the sending peer confirms, segment by segment, exactly which accepted records it consumed, and its checkpoint advances with the confirmation. The body is a JSON object with exactly three keys:

```json
{"ackId":"ack-1","cursor":2,"operations":[{"replicaId":"r1","operationId":"op-1"},{"replicaId":"r2","operationId":"op-2"}]}
```

- `{peerId}` must be a non-empty percent-decoded path segment, following the same path rules as the checkpoint and pickup routes; an empty segment (`/v1/sync/peers//acknowledge`) is a route-shape failure and returns HTTP 404 with `{"error":"not_found"}`.
- `ackId` is a non-empty string naming the receipt. `cursor` is a non-boolean, non-negative integer. `operations` lists, in order, the identities the peer consumed: each entry is an object with exactly `replicaId` and `operationId`, both non-empty strings, and no identity may repeat. One request confirms at most 100 records.
- Starting from the peer's registered checkpoint, `operations` must exactly cover the contiguous accepted records up to (but not including) `cursor`: `operations[i]` names the identity of the `checkpoint + i`-th accepted record, and the checkpoint plus the segment length equals `cursor`. An empty `operations` list confirms the empty segment at the current checkpoint.

Malformed JSON, a non-object body, a missing or unknown key, a wrong-typed or empty identifier, a repeated identity, any query parameter, or a segment longer than 100 records all return HTTP 400 with `{"error":"invalid_request"}` and change nothing. A missing, empty, or extra path segment (for example `/v1/sync/peers/{peerId}/acknowledge/extra` or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. A peer that has never registered a checkpoint returns HTTP 404 with `{"error":"not_found"}`.

- Success: HTTP 201 with exactly `{"status":"created","peerId","ackId","cursor"}` — a compact JSON object terminated by a single newline. The receipt, the checkpoint advance to `cursor`, and the `(peerId, ackId)` binding commit together.
- The same peer replaying the same `ackId` with identical content returns HTTP 200 with `"status":"ok"` in the same response shape and appends nothing — the replay is answered from the committed binding, however the checkpoint has moved since. The same `(peerId, ackId)` with different content returns HTTP 409 with `{"error":"operation_conflict"}`; nothing changes.
- A `cursor` below the peer's current checkpoint returns HTTP 409 with `{"error":"checkpoint_conflict"}`; nothing changes.
- A segment that does not exactly match the accepted log — a wrong identity, a wrong record count, or a `cursor` past the log end — returns HTTP 409 with `{"error":"ack_conflict"}`; nothing changes.

A receipt is not an operation. It changes neither the accepted-operation log nor sync export, the per-key audit, candidate state, or any of the six metrics counters; it is not exported by `GET /v1/sync/operations` and does not appear in audit streams. The checkpoint advance it carries is visible to the checkpoint GET, the pickup query, and the replication snapshot exactly like a checkpoint POST. Confirmation, checkpoint advancement, and binding are one commit under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so a concurrent reader always sees either the old or the new complete state, never half a receipt.

With `--data-file`, the receipt and the advanced checkpoint are written to the data file in the same atomic commit protocol (`write temp file → fsync → rename → fsync directory`) before the HTTP 201; receipts live in the optional `acks` section, one `{"peerId","ackId","cursor","operations"}` record per accepted receipt. A durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory, the checkpoint, the bindings, and the file exactly as they were — the request is safely retryable. A version:1 file written before receipts existed (without the `acks` section) recovers with no receipt bindings and otherwise unchanged semantics; after a restart the create `201`, replay `200`, and conflict `409` decisions are identical to a process that never restarted. The endpoint shares the common request contract: Content-Length is validated before authentication (400/413 first), an invalid or missing bearer token returns HTTP 401 `{"error":"unauthorized"}` without reading the body, and `/health` stays anonymous.

#### Reading a peer's consumption receipts

`GET /v1/sync/peers/{peerId}/receipts?after=N&limit=N` lets a sending replica review the consumption receipts a peer has committed through `POST /v1/sync/peers/{peerId}/acknowledge`, together with an integrity summary over the peer's whole committed receipt set. The endpoint is strictly read-only.

- `{peerId}` must be a non-empty percent-decoded path segment, following the same path rules as the checkpoint, pickup, and acknowledge routes.
- `after` is the number of the peer's receipts already skipped (a 0-based resume cursor); it is required and starts the page at `0`. `after` equal to the peer's current receipt count is a valid empty page. `limit` is required, must be between `1` and `100`, and (like `after`) accepts only ASCII decimal integers. A missing or repeated `after`/`limit`, a blank, negative, or non-ASCII-decimal value, an unknown parameter, a `limit` outside `1-100`, or an `after` past the current receipt count returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state.
- A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly six fields: `{"receipts":[...],"nextCursor":N,"hasMore":bool,"algorithm":"sha256","digest":"...","receiptsCount":N}`. `receipts` keeps the commit (creation) order of the peer's committed receipts; each item carries exactly `peerId`, `ackId`, the confirmation-time `cursor`, and `operations` — the confirmed identities in their confirmation order, each with exactly `replicaId` and `operationId`. `nextCursor` is the cumulative number of receipts skipped after this page — feed it back as the next `after`; `hasMore` reports whether further receipts remain.
- The summary covers the peer's **whole** committed receipt set, never just the page: `receiptsCount` counts committed receipts (not page items), and `digest` is the 64-character lowercase hexadecimal SHA-256 of the canonical digest input — a compact UTF-8 JSON array of the peer's receipts in creation order, each receipt written with its fields in the fixed order `peerId`, `ackId`, `cursor`, `operations`, each identity written as `replicaId`, `operationId` in confirmation order, numbers as plain JSON integers, strings escaping only the quote, the backslash, and U+0000-U+001F control characters (always as lowercase `\u00xx`, every other code point written literally), and no whitespace anywhere. An empty receipt set hashes the empty array `[]`.
- The page slice, `nextCursor`, `hasMore`, and the summary are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so the list and the summary always describe a single commit even while commits are in flight.
- The query is strictly read-only: it neither advances nor writes the checkpoint, records no receipt, and changes neither the accepted log, candidates, audit streams, metrics counters, summaries, nor the data file; it creates no temporary file. Repeated GETs return the same page and summary until the peer separately commits another acknowledgement.
- A peer that has never registered a checkpoint returns HTTP 404 with `{"error":"not_found"}`. A missing, empty (`/v1/sync/peers//receipts`), or extra (`/v1/sync/peers/{peerId}/receipts/extra`, a trailing slash, or a missing segment) path shape likewise returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.
- Every number in the response is a JSON integer, and strings use the same escaping as the other endpoints. With `--data-file`, the receipts are rebuilt identically during recovery (a file written before receipts existed recovers with an empty set), so the receipt order, page boundaries, cursor resume, and the summary are identical before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

#### Auditing a peer's whole confirmation chain

`GET /v1/sync/peers/{peerId}/receipts/audit?after=N&limit=N` is the sender-side read-only entry point over the peer's **entire confirmation chain**: it pages the same committed receipts as `GET /v1/sync/peers/{peerId}/receipts` (same creation order, same required-parameter paging rules) and additionally reports whether the receipts together form one seamless confirmation of the shared accepted log. The endpoint never advances a checkpoint, records a receipt, or changes candidates, the log, transactions, repairs, or the data file.

- `{peerId}` must be a non-empty percent-decoded path segment, following the same path rules as the checkpoint, pickup, acknowledge, and receipts routes.
- `after` and `limit` are both **required** (a request missing either, including a bare request with no query string, returns HTTP 400): `after` accepts only a non-negative ASCII decimal integer — the number of the peer's receipts already skipped, starting at `0`; `limit` accepts only an ASCII decimal integer between `1` and `100`. The endpoint otherwise follows the existing receipt paging rules exactly: a missing or repeated `after`/`limit`, a blank, negative, signed, decimal-point, whitespace, or non-ASCII-decimal value, an unknown parameter, a `limit` outside `1-100`, or an `after` past the current receipt count returns HTTP 400 with `{"error":"invalid_request"}`, and `after` equal to the receipt count is a valid stable empty page.
- A missing, empty (`/v1/sync/peers//receipts/audit`), multi-segment, or extra (`/v1/sync/peers/{peerId}/receipts/audit/extra`, a trailing slash, or a missing segment) path shape returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. A peer that has never registered a checkpoint likewise returns HTTP 404 with `{"error":"not_found"}`, without changing any visible state. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route — a missing, duplicated, or malformed `Authorization` header or a token mismatch returns HTTP 401 `{"error":"unauthorized"}` (and `/health` stays anonymous).

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly seven fields — the six receipt fields and an additional `audit` object:

```json
{"receipts":[{"peerId":"peer-a","ackId":"ack-1","cursor":2,"operations":[{"replicaId":"r1","operationId":"op-1"},{"replicaId":"r2","operationId":"op-2"}]}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","receiptsCount":1,"audit":{"status":"ok","coverage":{"start":0,"end":2},"gaps":[],"overlaps":[],"identityMismatches":[],"cursorRegressions":[]}}
```

- `receipts`, `nextCursor`, and `hasMore` are exactly the page of the plain receipts query: receipts in creation order, each carrying `peerId`, `ackId`, the confirmation-time `cursor`, and the confirmed `operations` identities in confirmation order; `nextCursor` resumes at the next `after`.
- `algorithm` is always `"sha256"`, `digest` is the 64-character lowercase hexadecimal SHA-256 of the same canonical receipt encoding used by the receipts endpoint, and `receiptsCount` counts the peer's committed receipts. **The digest and count cover the peer's whole receipt history, never the current page** — every page of the same snapshot reports identical summary values, and an empty receipt set hashes the empty array `[]`.
- `audit` is the chain-integrity conclusion over the complete history, also independent of the page:
  - `status`: `"ok"` when every anomaly list below is empty, else `"broken"`.
  - `coverage`: `{"start":S,"end":E}` — the half-open segment of the shared accepted log covered by the peer's receipts, from the earliest receipt's start to the last confirmation cursor. The first receipt's start is derived from its operation count and confirmation cursor (`cursor - len(operations)`); later receipts must begin exactly where the previous one ended, and every later segment's length must continue strictly from the prior end. An empty receipt set reports the complete, anomaly-free empty coverage `{"start":0,"end":0}`.
  - `gaps`: each entry marks a receipt beginning past the previous receipt's end, leaving accepted records unconfirmed — `{"receiptIndex":I,"ackId":A,"from":N,"to":M}` with the two boundary cursors.
  - `overlaps`: each entry marks a receipt beginning before the previous receipt's end, confirming some records twice — same `{receiptIndex,ackId,from,to}` shape.
  - `identityMismatches`: each entry marks one confirmed position whose `(replicaId, operationId)` does not match the accepted record at that absolute log position — `{"receiptIndex":I,"ackId":A,"position":P,"expected":{"replicaId","operationId"}|null,"observed":{"replicaId","operationId"}}`; `expected` is null when the position lies outside the current log.
  - `cursorRegressions`: each entry marks a confirmation cursor that does not advance past the previous receipt's cursor — same `{receiptIndex,ackId,from,to}` shape.

`receiptIndex` is the receipt's 0-based position in creation order, so every anomaly retains where it occurred; a receipt confirming an empty segment (`operations: []`) is legal on its own and raises no anomaly. With `status: "ok"` the receipts form one seamless, non-overlapping, log-consistent chain: every receipt starts exactly where the previous one ended, each confirmed identity matches the corresponding log position, and no cursor moves backwards.

Paging trims only the exported receipt derivation: the `audit` conclusion, `digest`, and `receiptsCount` are always computed from the complete receipt history and the full accepted log. The page slice, cursors, summary, and conclusion are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so a concurrent commit is observed only as a complete old or new snapshot. Every number in the response is a JSON integer, and strings use the same escaping as the other endpoints. The query is strictly read-only. With `--data-file`, receipts and the log are rebuilt identically during recovery (a file written before receipts existed recovers with an empty set), so audit results are identical before and after a restart; the README startup entry and the existing receipt, acknowledgement, and checkpoint behaviors are unchanged.

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

### Global audit-chain query

`GET /v1/audit/log/chain?after=N&limit=N` returns a read-only page of the **global operation chain**: one integrity link per record of the shared accepted-operation log, in global commit order. The chain covers everything the log covers — ordinary writes, stale writes whose clock was already dominated, accepted conflict repairs, and sync-imported records — and nothing else: identical replays (`200`), conflicting or malformed requests (`409`/`400`), uncommitted requests, and rejected batches never enter the log and so never enter the chain.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"entries":[{"sequence":1,"prevDigest":"<64 zeros>","digest":"<64 lowercase hex chars>"}],"nextCursor":1,"hasMore":false,"head":"<64 lowercase hex chars>"}
```

- `entries`: one page of chain links in global commit order. Each entry carries exactly `sequence` (the link's 1-based position in the log), `prevDigest` (the previous link's digest, or 64 `0` characters for the first link), and `digest` (this link's digest).
- `nextCursor`: the number of links skipped after this page — feed it back as the next `after`.
- `hasMore`: whether further links remain.
- `head`: the digest of the chain's last link — the chain-tail summary. It describes the whole chain, never the page, so it is identical on every page; an empty log reports 64 `0` characters.

Each link's `digest` is the 64-character lowercase hexadecimal SHA-256 of the concatenation of three byte strings: the previous link's digest (ASCII), the link's decimal sequence number (ASCII), and the single record's canonical bytes. The record bytes follow exactly the per-key audit digest's record encoding — the fixed shape `{"replicaId":R,"operation":{"operationId":I,"key":K,"value":V,"clock":C}}` with the clock's component names sorted lexicographically, no whitespace anywhere, and strings escaping only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex). Every count in the response is a JSON integer.

Paging follows the sync-export rules: `after` is the number of links already skipped (a 0-based resume cursor) and defaults to `0`; `limit` defaults to `100` and must be between `1` and `100`. An `after` equal to the chain length is a valid empty tail. A negative, blank, or non-ASCII-decimal `after`/`limit` (signs, decimals, whitespace, and non-ASCII numerals are all rejected), a repeated or unknown query parameter, a `limit` outside `1-100`, or an `after` past the chain length returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment (for example `/v1/audit/log`, `/v1/audit/log/chain/extra`, or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check.

The page slice, `nextCursor`, `hasMore`, and `head` are computed from a single snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the four values always agree even while commits are in flight: a read can never observe half an import batch or a partially applied repair. The request is strictly read-only — it changes no metrics, candidates, audit streams, checkpoints, or logs, modifies neither memory nor the data file, and creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same history yields the same record order, chain digests, cursors, and head before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

### Global audit-chain integrity verification

`GET /v1/audit/log/verify?after=N&limit=N&head=H&count=N` is the read-only integrity-verification companion to the global audit-chain query. It is a separate entry point that does not change the chain query in any way; it uses the same read permission and the same accepted-operation log, but all four query parameters are required. The chain links are returned in global commit order and cover exactly what the chain query covers — ordinary writes, stale writes whose clock was already dominated, accepted conflict repairs, and sync-imported records — while identical replays (`200`), rejected requests (`409`/`400`), uncommitted requests, failed batches, and records whose durable commit failed never enter the log and so never enter verification.

Besides the chain query's paging, the request carries two **required external expectations**:

- `head`: exactly 64 lowercase hexadecimal characters — the chain-tail digest the caller expects (the `head` previously returned by the chain query). Uppercase, non-hex, blank, or wrong-length values are rejected.
- `count`: a non-negative ASCII decimal integer — the total chain length the caller expects (the full log length, not the page length). Signs, decimals, whitespace, and non-ASCII numerals are rejected.

`after` and `limit` are both **required** — there are no defaults: a request missing either one is rejected. `after` is a non-negative ASCII decimal integer and `limit` must be between `1` and `100`. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly five fields — the chain query's four fields plus `verification`:

```json
{"entries":[{"sequence":1,"prevDigest":"<64 zeros>","digest":"<64 lowercase hex chars>"}],"nextCursor":1,"hasMore":false,"head":"<64 lowercase hex chars>","verification":{"status":"ok","missingSequences":[],"duplicateSequences":[],"outOfRangeSequences":[],"brokenLinks":[],"digestMismatches":[],"headMismatches":[],"countMismatches":[]}}
```

The `entries` page, `nextCursor`, `hasMore`, and `head` are produced exactly as for the chain query. The `verification` object is an independent scan of the **complete** log — it never pages and never trusts a materialized link, recomputing every link itself — and always covers the whole history even when the page is empty or partial. Its checks are:

- **Sequence continuity** — the claimed links form the continuous 1-based range `1..N` with no missing, duplicate, or out-of-range sequence.
- **Predecessor closure** — the first link closes against the 64-`0` genesis and every later link closes against the previous link's digest.
- **Digest recomputation** — each link digest is recomputed from the record's canonical bytes (the per-key audit digest record encoding) and the predecessor digest.
- **Chain-tail agreement** — the recomputed last-link digest (the `head`, 64 zeros for an empty log) must equal the external `head`, and the full length must equal the external `count`.

Each anomaly list is independent and every entry keeps the chain-link position (the 0-based `linkIndex`), the link's 1-based `sequence`, and the observed value:

- `missingSequences`: `{"linkIndex":I,"sequence":S}` — a position in `1..N` no link claims.
- `duplicateSequences`: `{"linkIndex":I,"sequence":S}` — a sequence an earlier link already claims (the later occurrence only).
- `outOfRangeSequences`: `{"linkIndex":I,"sequence":S}` — a sequence that is not an integer in `1..N`.
- `brokenLinks`: `{"linkIndex":I,"sequence":S,"expected":D,"observed":D}` — a predecessor-closure failure: the claimed `prevDigest` is not the predecessor link's recomputed digest (the genesis for the first link); the recomputed predecessor is first and the claimed one second.
- `digestMismatches`: `{"linkIndex":I,"sequence":S,"expected":D,"observed":D}` — a digest-recomputation failure: the claimed `digest` is not the value independently recomputed from the record's canonical bytes and the running predecessor; the recomputed digest is first and the claimed one second.
- `headMismatches`: at most one `{"expected":H,"observed":H}` — the external `head` first, the recomputed chain tail second.
- `countMismatches`: at most one `{"expected":C,"observed":N}` — the external `count` first, the actual full length second.

`status` is `"ok"` exactly when the internal chain is intact (the first five lists empty) **and** both external expectations match; otherwise it is `"broken"`. An empty log is intact with a 64-zero head and verifies `"ok"` for `head` equal to 64 zeros and `count` `0`.

Paging trims only the `entries` page: the `head`, the count comparison, and the whole `verification` conclusion always cover the complete history on every page, including a stable empty page returned when `after` equals the chain length. A missing, repeated, or unknown parameter (a missing `after` or `limit` included), a blank value, a malformed `head`/`count`/`after`/`limit`, a `limit` outside `1-100`, or an `after` past the chain length returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment (for example `/v1/audit/log`, `/v1/audit/log/verify/extra`, or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check.

The page, the expectation comparison, and the verification conclusion are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so they always describe a single commit even while commits are in flight. The request is strictly read-only — it changes no candidates, operation logs, checkpoints, receipts, transactions, policy audits, metrics, or data files, and creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovery history yields the same page, head, count comparison, and verification before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with a `Bearer` challenge; in scope-policy mode an authenticated token lacking the read scope is HTTP 403 without a challenge; and `/health` stays anonymous.

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

### Per-operation causal descendant chain

`GET /v1/causal/descendants?replicaId=R&operationId=O&after=N&limit=N` returns a read-only page of one operation's **strict causal descendants**. Like the comparison and difference routes, the identity is carried as **query parameters** (percent-decoded like every route value); both are required and must be non-empty. An identity that was never first-accepted returns HTTP 404 with `{"error":"not_found"}`, and the route-shape check takes precedence over every other check.

A strict descendant is a first-accepted record committed **after** the source operation in the shared log whose clock strictly dominates the source operation's clock (missing components count as 0, and domination already requires the clocks to differ). The source operation itself never appears. Stale writes that added no candidate and accepted conflict repairs are ordinary committed records and participate like any other; identical replays (`200`), conflicting or malformed requests (`409`/`400`), and uncommitted writes never enter the log, so they can never appear.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"cursor":1,"descendants":[{"operation":{"clock":{"r1":1,"r2":1},"key":"size","operationId":"op-2","value":"large"},"relation":"direct","replicaId":"r2"}],"more":false,"operation":{"operation":{"clock":{"r1":1},"key":"color","operationId":"op-1","value":"blue"},"replicaId":"r1"}}
```

- `operation`: the source record in the per-operation archive shape `{"replicaId":R,"operation":{"operationId","key","value","clock"}}`.
- `descendants`: one page of the strict descendants in the shared log's global commit order. Each entry preserves the archive record content and adds `relation`: `"direct"` when no other strict descendant's clock dominates the record's clock, `"transitive"` otherwise. The classification is computed once against the complete, unpaged descendant set, so paging never changes a record's relation.
- `cursor`: the number of descendants skipped after this page — feed it back as the next `after`.
- `more`: whether further descendants remain.

Paging follows the sync-export rules: `after` is the number of descendants already skipped (a 0-based resume cursor) and defaults to `0`; `limit` defaults to `100` and must be between `1` and `100`. A legal identity with no descendants — or an `after` equal to the descendant count — returns HTTP 200 with an empty `descendants` list. A missing, unknown, or repeated parameter (any of the four names), a blank identity value, a negative, blank, or non-ASCII-decimal `after`/`limit`, a `limit` outside `1-100`, or an `after` past the descendant count returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state. A missing, empty, or extra path segment (for example `/v1/causal/descendants/` or `/v1/causal/descendants/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and identity checks.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

The source record, the descendant list, the relation classification, and the paging boundaries are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovered state yields the same source, the same descendant relations, and the same pages before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route (and `/health` stays anonymous).

### Causal frontier overview

`GET /v1/causal/frontier?after=N&limit=N` returns a read-only page of the **causal frontier**: the maximal set of first-accepted operations — the accepted records whose clock no *other* accepted record's clock strictly dominates. Domination follows the shared vector-clock rules exactly (missing components count as 0, and domination already requires the clocks to differ), so two records whose clocks are equal, or that dominate each other in neither direction, both stay on the frontier. The operation's source kind is irrelevant: ordinary writes, stale writes that added no candidate, sync-imported records, and accepted manual or automatic repairs all participate as ordinary committed records, while identical replays (`200`), rejected (`400`/`409`) requests, uncommitted requests, and requests whose durable commit failed never enter the log and so never appear.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly six fields:

```json
{"operations":[{"operation":{"clock":{"r1":1,"r2":1},"key":"size","operationId":"op-2","value":"large"},"replicaId":"r2"}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","frontierCount":1}
```

- `operations`: one page of the frontier in the shared log's global commit order. Each entry preserves the per-operation archive record content: `{"replicaId":R,"operation":{"operationId","key","value","clock"}}`.
- `nextCursor`: the number of frontier records skipped after this page — feed it back as the next `after`; `hasMore` reports whether further frontier records remain.
- `algorithm` is always `"sha256"`, and `frontierCount` counts the **complete** frontier, never just the page.
- `digest` summarizes the whole frontier, so it is identical on every page of one snapshot. The hash input is a compact UTF-8 JSON array with one element per frontier record in frontier order, each element written with the audit chain's canonical record encoding — the fixed shape `{"replicaId":R,"operation":{"operationId":I,"key":K,"value":V,"clock":C}}` with the clock's component names sorted lexicographically, no whitespace anywhere, and strings escaping only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex). An empty frontier hashes the empty array `[]`.

Paging requires **both** parameters: `after` (the number of frontier records already skipped, a 0-based resume cursor) and `limit` (between `1` and `100`) must each appear exactly once as an ASCII decimal integer — there are no defaults. A missing or repeated parameter, an unknown parameter, a blank, signed, whitespace-bearing, decimal-point, or non-ASCII-decimal value, a `limit` outside `1-100`, or an `after` past the frontier size returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state; an `after` equal to the frontier size is a valid stable empty page. A missing, empty, or extra path segment (for example `/v1/causal/frontier/` or `/v1/causal/frontier/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check.

Every number in the response is a JSON integer; strings escape only the quote (`\"`), the backslash (`\\`), and control characters U+0000–U+001F (always as `\u00XX` with lowercase hex) — every other Unicode code point is written literally.

The frontier, the page slice, the resume cursor, the remaining flag, the complete count, and the digest are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit: a read can never observe half an import batch or a partially applied repair. The request is strictly read-only — it modifies neither memory, candidates, logs, checkpoints, receipts, nor the data file, and it creates no temporary file. With `--data-file`, the log is rebuilt identically during recovery, so the same recovered state yields the same frontier, the same digest, and the same pages before and after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, or malformed `Authorization` header or a token mismatch is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge, and in scope-policy mode a token without the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge (and `/health` stays anonymous).

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

### Cross-replica candidate comparison

`POST /v1/replication/compare` returns a read-only diff between the local current candidates and a remote replica's complete candidate snapshot, so a caller can locate exactly which candidate versions still need to converge. The request body is a JSON object with exactly two fields:

```json
{"replicaId":"replica-b","snapshot":{"color":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}]}}
```

- `replicaId`: the remote replica's identifier, a non-empty string. It only names the comparison partner — the snapshot itself may hold candidates from any replica.
- `snapshot`: the remote's complete candidate state, an object mapping each business key to a non-empty array of candidates. Each candidate must contain exactly `value`, `clock`, `replicaId`, and `operationId` and satisfy the live write constraints: the value and both identity components are non-empty strings, and the clock is a non-empty object whose component values are non-boolean, non-negative JSON integers (floats such as `1.0` and `-0.0` and non-finite values such as `NaN`/`Infinity`/`-Infinity` are rejected) and which contains the candidate's own replica id. An operation identity `(replicaId, operationId)` may appear at most once across the whole snapshot.

Malformed JSON, a non-object body, a missing or unknown field, a duplicated field anywhere in the document, an empty key or candidate array, a duplicated candidate identity, or a structurally illegal candidate all return HTTP 400 with `{"error":"invalid_request"}` and change nothing. The route accepts no query parameters: any parameter returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing or extra path segment or a trailing slash (for example `/v1/replication/compare/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and body checks.

The remote snapshot is **only compared** — it is never imported into local state, and the request triggers no repair, transaction, sync, checkpoint, or persistence write. The local candidates are read from one committed snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so the response always describes a single commit even while commits are in flight: a read observes either the old or the new complete state, never half an import batch or a partially applied repair. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"keys":[{"differences":[{"kind":"conflict","local":{"clock":{"r1":2},"operationId":"op-1","replicaId":"r1","value":"blue"},"remote":{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"red"}}],"key":"color"}],"replicaId":"replica-b","status":"ok","summary":{"differences":1,"identical":false,"localCandidates":1,"localDigest":"<64 lowercase hex chars>","localKeys":1,"remoteCandidates":1,"remoteDigest":"<64 lowercase hex chars>","remoteKeys":1}}
```

- `status`: always `"ok"`.
- `replicaId`: the requested remote replica id, echoed back.
- `keys`: one entry per business key in the union of both sides, sorted lexicographically. Each entry's `differences` array covers every candidate identity either side holds for the key, sorted by `(replicaId, operationId)`, and each element keeps both sides' candidate — `local` and `remote`, each `{"value","clock","replicaId","operationId"}` or `null` on the side that lacks the identity — under one `kind` mark:
  - `"shared"`: both sides hold the identity with the same value and the same clock — not a difference.
  - `"missing_remote"`: only the local side holds the identity.
  - `"missing_local"`: only the remote side holds the identity.
  - `"conflict"`: both sides hold the identity but the values differ (a content conflict).
  - `"clock"`: both sides hold the identity with the same value but different clocks (one side's clock covers the other's). Such an entry additionally carries `clockDirection`: `"L"` when the local clock dominates the remote clock, `"R"` when the remote clock dominates the local clock, and `"C"` when the two clocks are concurrent — neither dominates the other (missing components count as zero).
- `summary`: the convergence totals — `localKeys`/`remoteKeys` and `localCandidates`/`remoteCandidates` count each side's keys and candidate versions; `localDigest` and `remoteDigest` are the 64-character lowercase hexadecimal SHA-256 of each side's canonical candidate snapshot, following exactly the verification-digest rules; `identical` reports whether the two digests are equal (true precisely when the candidate states are the same — in particular when both are empty, where both digests are the hash of `[]` and `keys` is empty); and `differences` counts the non-shared entries — the minimal candidate-level difference count a follow-up sync must reconcile. When one side is empty, only the other side's keys and candidates appear, each marked missing on the empty side.

Every number in the response is a JSON integer; no float, negative zero, or non-finite value can appear. The compact encoding, escaping, and terminator are the same as `GET /v1/replication/snapshot`.

The comparison is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the candidate state is rebuilt identically during recovery, so the same local state and the same remote snapshot yield the same report before and after a restart. The route shares the common request contract: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered **before** reading the body and before authentication; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge; and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge. `/health` stays anonymous. A `GET` on the path is an unknown route and answers HTTP 404.

### Follow-up replica synchronization plan

`POST /v1/replication/plan` turns the same cross-replica picture into an executable follow-up sync plan. The request body is **exactly the comparison request** — a JSON object with exactly two fields, reusing the comparison's remote identifier and complete candidate snapshot:

```json
{"replicaId":"replica-b","snapshot":{"color":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}]}}
```

- `replicaId` names the synchronization partner (a non-empty string), and `snapshot` is its complete candidate state under **exactly the comparison's write constraints**: an object mapping each business key to a non-empty candidate array, each candidate carrying exactly `value`, `clock`, `replicaId`, and `operationId` with non-empty strings and a non-empty clock of non-boolean, non-negative JSON integers that contains the candidate's own replica id. Floats (including `1.0` and `-0.0`), non-finite values (`NaN`/`Infinity`/`-Infinity`), duplicate fields, duplicate candidate identities across the snapshot, unknown fields, empty keys or arrays, malformed JSON, and non-object documents are all rejected with HTTP 400 `{"error":"invalid_request"}` and change nothing.
- The route accepts no query parameters: any parameter is HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing, extra, or multi segment, a trailing slash (for example `/v1/replication/plan/`), or any unknown route is HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes priority over the query-parameter and body checks (an unreadable body on a wrong shape is still 404). A `GET` on the path is an unknown route and answers HTTP 404.

The remote snapshot is only read — it is never imported, and the request triggers no repair, transaction, sync, checkpoint, or persistence write. The local candidates are read from one committed snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so a concurrent commit is observed only as a whole old or whole new snapshot, never a mix. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"keys":[{"key":"color","actions":[{"action":"fetch_remote","kind":"missing_local","local":null,"remote":{"clock":{"r2":1},"operationId":"op-9","replicaId":"r2","value":"red"}}]}],"replicaId":"replica-b","status":"ok","summary":{"localKeys":1,"remoteKeys":1,"localCandidates":1,"remoteCandidates":1,"actions":1,"identical":false}}
```

- `status`: always `"ok"`.
- `replicaId`: the requested remote replica id, echoed back.
- `keys`: business keys that hold at least one action, sorted lexicographically by key (a key whose identities are all already converged is omitted). Each entry carries exactly `key` and `actions`; the actions cover the key's identity union sorted by `(replicaId, operationId)`, and each action carries exactly four fields:
  - `action`: the executable next step — `"send_local"` or `"fetch_remote"` — or the string `"semantic_resolution"` when neither version may be auto-overwritten.
  - `kind`: one of the comparison marks describing why the action exists — `"missing_remote"`, `"missing_local"`, `"clock"`, or `"conflict"`.
  - `local` and `remote`: the candidate on each side, each `{"value","clock","replicaId","operationId"}`, or `null` on the side that lacks the identity.
- `summary`: JSON values only — `localKeys`/`remoteKeys` and `localCandidates`/`remoteCandidates` count each side's keys and candidate versions, `actions` counts the generated actions, and `identical` is a JSON boolean, true precisely when the two candidate states are identical (following the same canonical-digest equality as the comparison, including both sides empty).

Per identity, the action is decided as follows:

- An identity held only locally (kind `missing_remote`), or held on both sides with the same value whose local clock dominates the remote clock (kind `clock`, local side dominates), is marked `"send_local"`: the local version is the one to propagate.
- An identity held only remotely (kind `missing_local`), or held on both sides with the same value whose remote clock dominates the local clock (kind `clock`, remote side dominates), is marked `"fetch_remote"`: the remote version is the one to pull.
- An identity held on both sides with the same value but concurrent clocks — neither clock dominates the other — is marked `"semantic_resolution"` (kind `clock`); neither version overwrites the other automatically.
- An identity held on both sides with **different values** is always marked `"semantic_resolution"` (kind `conflict`), regardless of the clock relationship — even when one clock dominates the other; both conflicting candidates are retained in `local` and `remote` for the existing semantic-repair flow, and no `send_local`/`fetch_remote` is ever auto-selected for a content conflict.
- An identity the two sides hold with the same value and the same clock is already converged: it generates no action. When the two states are identical overall, `keys` is empty, `actions` is `0`, every summary pair is equal, and `identical` is `true`.

Every number in the response is a JSON integer (the only numbers are key/candidate/action counts and vector-clock ticks); no float, negative zero, or non-finite value can appear. The compact encoding, escaping, and terminator are the same as `GET /v1/replication/snapshot`, and every count is written as a JSON integer.

The plan is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the candidate state is rebuilt identically during recovery, so the same local state and the same remote snapshot yield the same plan before and after a restart. The route shares the common request contract: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered **before** reading the body and before authentication; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge; and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge (a `read`-only token is sufficient). `/health` stays anonymous.

### Multi-replica convergence-consensus summary

`POST /v1/replication/consensus` turns the same cross-replica comparison inputs into one read-only **convergence decision across several remote replicas at once**. The request body is a JSON **array** of between two and one hundred entries, in request order, each naming one remote replica and its complete candidate snapshot:

```json
[{"replicaId":"replica-b","snapshot":{"color":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}]}},{"replicaId":"replica-c","snapshot":{"color":[{"clock":{"r1":2},"operationId":"op-1","replicaId":"r1","value":"blue"}]}}]
```

- Each entry must be an object with exactly `replicaId` and `snapshot`. `replicaId` is a non-empty string, unique within the array; the reserved id `"local"` (the source id under which the local committed state participates) must not be used by any entry.
- Every `snapshot` obeys **exactly the comparison's input constraints** (`parse`/`_parse_remote_snapshot`): an object mapping each business key to a non-empty candidate array, each candidate carrying exactly `value`, `clock`, `replicaId`, and `operationId` with non-empty strings and a non-empty clock of non-boolean, non-negative JSON integers that contains the candidate's own replica id. Floats (including `1.0` and `-0.0`), non-finite values (`NaN`/`Infinity`/`-Infinity`), duplicate fields, duplicate candidate identities within one snapshot, and unknown fields are all rejected.
- The empty array, an array shorter than two or longer than one hundred entries, a non-array document, an empty or duplicated `replicaId`, an unknown or missing field on an entry, malformed JSON, or a structurally illegal snapshot all return HTTP 400 with `{"error":"invalid_request"}`.
- The route accepts no query parameters: any unknown, repeated, blank, or otherwise illegal parameter is HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing, extra, or multi segment, a trailing slash (for example `/v1/replication/consensus/`), or any unknown route is HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes priority over the query-parameter and body checks (an unreadable body on a wrong shape is still 404). A `GET` on the path is an unknown route and answers HTTP 404.

The local current candidates are read from one committed snapshot and participate as the first source, always under the source id `"local"`; the remote entries then follow in request order. The query aggregates, per business key, every source's observations of the same operation identity `(replicaId, operationId)` — an operation the sources all describe, regardless of which business replica authored it. Each observed identity is classified under exactly one status:

- **`"converged"`** — every observation holds exactly the same value and exactly the same clock. An identity a single source alone holds is likewise converged (there is no divergence to reconcile).
- **`"propagable"`** — every observation holds the same value, the clocks differ, and **one observation's clock strictly dominates every other observed clock** (missing components count as 0, exactly as in the write semantics). The result names the version to propagate: its `decision` carries the winning `source` (`"local"` or one remote id), the winning `value`, and the winning `clock`; `supersededClocks` retains every eliminated observation as `{"source","clock"}` evidence, in source order. Nothing is actually propagated — the summary is read-only.
- **`"conflict"`** — anything that cannot be auto-decided: equal values whose clocks do not admit a single dominator (two mutually concurrent clocks, or any clock tie), or the same identity holding **different values** across sources (even when one clock dominates the others). Such an entry retains the identity and **all** observations, each as `{"source","value","clock"}` in source order; the response never picks a value and never merges clocks.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields:

```json
{"keys":[{"identities":[{"decision":{"clock":{"r1":2},"source":"replica-c","value":"blue"},"operationId":"op-1","replicaId":"r1","status":"propagable","supersededClocks":[{"clock":{"r1":1},"source":"local"},{"clock":{"r1":1},"source":"replica-b"}]}],"key":"color"}],"sources":["local","replica-b","replica-c"],"status":"ok","summary":{"conflicts":0,"converged":0,"propagable":1}}
```

- `status`: always `"ok"`.
- `sources`: the source ids the decision was computed over — `"local"` first, followed by the request's remote replica ids in request order.
- `keys`: one entry per business key in the union of all sources, sorted lexicographically (Unicode code point order). Each entry carries exactly `key` and `identities`; the identities cover the key's identity union, sorted by `(replicaId, operationId)` ascending. A converged identity carries `replicaId`, `operationId`, `status`, its agreed `value` and `clock`, and `sources` (the source ids that observed it, in source order). A propagable identity carries `replicaId`, `operationId`, `status`, the `decision` object (`source`, `value`, `clock`), and `supersededClocks`. A conflict identity carries `replicaId`, `operationId`, `status`, and `observations` (one `source`/`value`/`clock` entry per holding source).
- `summary`: the stable totals as JSON integers — `converged`, `propagable`, and `conflicts`, each counting the classified identities across every key exactly once.

Every number in the response is a JSON integer (the only numbers are the three counts and vector-clock ticks); no float, negative zero, or non-finite value can appear. The compact encoding, escaping, and terminator are the same as `GET /v1/replication/snapshot`.

The remote snapshots are only read — they are never imported, and the request triggers no repair, transaction, sync, checkpoint, or persistence write. The local snapshot, the aggregation, the decisions, and the counts are all computed under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so a concurrent commit is observed only as a whole old or whole new snapshot, never a mix; the same local state and the same request body always produce the identical response. The query is strictly read-only — it modifies neither memory nor the data file and creates no temporary file. With `--data-file`, the candidate state is rebuilt identically during recovery, so the same inputs produce the same decision before and after a restart and every existing endpoint's behavior is unchanged. The route shares the common request contract: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered **before** reading the body and before authentication; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge; and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge (a `read`-only token suffices). `/health` stays anonymous.

### Executable replica repair batch

`POST /v1/replication/apply` executes a planned repair batch against the local replica: it is the committing counterpart of the read-only comparison and plan. The request body is a JSON object with exactly four fields:

```json
{"replicaId":"replica-b","expectedLocalDigest":"<64 lowercase hex chars>","snapshot":{"color":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}]},"actions":[{"action":"fetch_remote","key":"color","replicaId":"r2","operationId":"op-9","value":"red","clock":{"r2":1}}]}
```

- `replicaId`: the remote replica's identifier, a non-empty string. As in the comparison, it only names the repair partner — the snapshot may hold candidates from any replica.
- `expectedLocalDigest`: the 64-character lowercase hexadecimal SHA-256 the caller expects the **current committed local candidate snapshot** to have — exactly the `localDigest` the comparison reports (the verification-digest rules over the current candidate sets). If it does not match the committed state, the whole batch is rejected unchanged with HTTP 409 `{"error":"apply_conflict"}`.
- `snapshot`: the remote's complete candidate state under **exactly the comparison's constraints** — an object mapping each business key to a non-empty candidate array, each candidate carrying exactly `value`, `clock`, `replicaId`, and `operationId` with non-empty strings and a non-empty clock of non-boolean, non-negative JSON integers that contains the candidate's own replica id.
- `actions`: the ordered repair batch, 1 to 100 entries, validated and executed **in request order against a staged view** of the store (an earlier action's effect is visible to a later one). Each entry carries `action` — `"send_local"`, `"fetch_remote"`, or `"semantic_resolution"` — plus the candidate identity (`replicaId`, `operationId`), `key`, `value`, and `clock` it acts on; a `"semantic_resolution"` entry additionally carries `candidates`, the non-empty list of distinct `{"replicaId","operationId"}` identities it expects its target key to currently hold, and its `value` is the merged value. No two actions may name the same `(replicaId, operationId)` identity.

Malformed JSON, a non-object body, a missing or unknown field, a duplicated field anywhere in the document, an unknown direction, a structurally illegal entry (including an illegal clock or an illegal expected-candidate set), a duplicated action identity, or an illegal snapshot all return HTTP 400 with `{"error":"invalid_request"}` and change nothing. A semantic-repair clock that is structurally legal but does **not dominate** every expected candidate is likewise HTTP 400 `{"error":"invalid_request"}` — the same malformed-request rule the manual repair flow applies. The route accepts no query parameters: any parameter returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing or extra path segment or a trailing slash (for example `/v1/replication/apply/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and body checks. A `GET` on the path is an unknown route and answers HTTP 404.

Per action, against the staged view:

- A known `(replicaId, operationId)` whose committed operation carries the same key, value, and clock is a **replay**: it is answered from the committed operation without any state check and counts as replayed. A known identity with different content is HTTP 409 `{"error":"operation_conflict"}`.
- `"send_local"` and `"fetch_remote"` follow the synchronization plan's directions. A send whose identity is not a current local candidate (the direction no longer holds or the candidate moved) and a fetch whose remote snapshot no longer holds the candidate exactly as claimed are HTTP 409 `{"error":"apply_conflict"}`. A send changes no local state — the candidate is already committed locally — so it is idempotent by construction; a fetch imports the remote candidate as one ordinary operation.
- `"semantic_resolution"` follows the manual repair semantics: a missing target key, a key no longer in value conflict, or an expected candidate set that does not match the key's current identities is HTTP 409 `{"error":"resolution_conflict"}`. Otherwise the merged value commits as one ordinary operation whose clock dominates every expected candidate.

Any failure rejects the **whole batch unchanged**, however far validation got. When at least one action is newly accepted, every new operation is committed together in one atomic commit — persisted before the caller observes success, exactly like a sync-import batch — and the response is HTTP 201; when every action is a replay, nothing is written and the response is HTTP 200. Both are compact UTF-8 JSON objects terminated by a single newline:

```json
{"accepted":1,"actions":[{"action":"fetch_remote","key":"color","operationId":"op-9","replicaId":"r2","value":"red"}],"replicaId":"replica-b","replayed":0,"status":"created"}
```

- `status`: `"created"` when at least one action was newly committed, `"ok"` when every action was a replay.
- `replicaId`: the requested remote replica id, echoed back.
- `actions`: one result per requested action, in request order, each carrying exactly `action`, `key`, `replicaId`, `operationId`, and the committed `value`.
- `accepted` and `replayed`: how many actions were newly committed and how many were replays.

Every number in the response is a JSON integer; no float, negative zero, or non-finite value can appear. The compact encoding, escaping, and terminator are the same as `GET /v1/replication/snapshot`. The whole batch runs under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so a concurrent commit is observed only as a complete old or new snapshot, never a mix; a rejected batch creates no temporary file, and a failed durable commit returns HTTP 500 `{"error":"internal_error"}` with memory, the identity index, and the data file exactly as before. The route shares the common request contract (length checks before authentication, 401 with a `Bearer` challenge, 403 without one); because the batch commits operations, it requires the `write` or `admin` scope in scope-policy mode. With `--data-file`, the committed operations are rebuilt identically during recovery, so replay and conflict decisions are the same before and after a restart.

### Sender-side replication delivery status

`GET /v1/replication/status?peerId=P` returns a strictly read-only delivery-status summary for one registered sender-side replication peer: the checkpoint progress, the unconsumed count, the receipt count, and the receipt chain-audit conclusion are read together from one committed snapshot. The endpoint creates no receipt, never advances or writes the checkpoint, and changes neither the accepted log, candidates, audit streams, metrics counters, summaries, nor the data file; it creates no temporary file.

The query string carries exactly one parameter, `peerId`, appearing exactly once with a non-empty percent-decoded value — the same percent-decoding and non-empty rules the replication routes apply to their `{peerId}` path segment. A missing or repeated `peerId`, an empty value, an unknown parameter, or an illegal encoding (a malformed percent escape or an escape sequence that is not valid UTF-8) returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly five fields in this order:

```json
{"peer":"peer-a","pos":2,"left":1,"acks":1,"chain":{"status":"ok","coverage":{"start":0,"end":2},"gaps":[],"overlaps":[],"identityMismatches":[],"cursorRegressions":[]}}
```

- `peer`: the decoded peer id the request selected.
- `pos`: the peer's registered checkpoint cursor — the number of accepted records the peer has consumed.
- `left`: the number of accepted records past the checkpoint the peer has not yet consumed.
- `acks`: the number of the peer's committed receipts.
- `chain`: the receipt chain-audit conclusion over the peer's **whole** committed receipt set, exactly as reported by `GET /v1/sync/peers/{peerId}/receipts/audit`: the `status` (`"ok"` exactly when all four anomaly lists are empty), the `coverage` interval `{"start","end"}`, and the `gaps`, `overlaps`, `identityMismatches`, and `cursorRegressions` lists. An empty receipt set reports a complete, anomaly-free empty coverage (`{"start":0,"end":0}`) with status `"ok"`.

Every number in the response is a JSON integer. The checkpoint cursor, the unconsumed count, the receipt count, and the chain conclusion are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so the progress, the counts, and the conclusion always describe a single commit even while commits are in flight.

A peer that has never registered a checkpoint returns HTTP 404 with `{"error":"not_found"}`. A missing or extra path segment (for example `/v1/replication`, `/v1/replication/status/extra`, or a trailing slash) likewise returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with a `Bearer` challenge, in scope-policy mode an authenticated token lacking the read scope is HTTP 403 without a challenge, and `/health` stays anonymous. With `--data-file`, the log, the checkpoints, and the receipts are rebuilt identically during recovery, so the same state yields the same status before and after a restart.

### All-peers replication delivery overview

`GET /v1/replication/status/all?after=N&limit=N` returns a strictly read-only delivery-status overview across **every registered** sender-side replication peer: one page of per-peer details together with the paging cursor, the complete registered count, the progress totals, and the receipt-anomaly counts are all read from one committed snapshot. The endpoint creates no receipt, never advances or writes a checkpoint, and changes neither the accepted log, candidates, audit streams, metrics counters, summaries, nor the data file; it creates no temporary file.

The query string carries exactly two parameters, both **required** and each appearing exactly once: `after` is the number of registered peers already skipped (a 0-based resume cursor that starts at `0`) and `limit` is the page size. Both accept only ASCII decimal digits — a missing or repeated parameter, an unknown parameter, an empty value, a sign, a decimal point, whitespace, or non-ASCII numerals returns HTTP 400 with `{"error":"invalid_request"}`, as does a `limit` outside `1-100`. An `after` equal to the registered peer count is a valid stable empty page; an `after` past it is HTTP 400 with `{"error":"invalid_request"}`. No rejected query reads or changes any state.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly six fields in this order:

```json
{"peers":[{"peer":"peer-a","pos":2,"left":1,"acks":1,"chainStatus":"ok"}],"nextCursor":1,"hasMore":false,"peerCount":1,"totals":{"pos":2,"left":1,"acks":1},"anomalies":{"gaps":0,"overlaps":0,"identityMismatches":0,"cursorRegressions":0}}
```

- `peers`: one page of the registered peers in ascending `peerId` (Unicode code point) order. Each item carries exactly five fields in this order: `peer` (the peer id), `pos` (its registered checkpoint cursor — the number of accepted records it has consumed), `left` (the number of accepted records past the checkpoint it has not yet consumed), `acks` (its committed receipt count), and `chainStatus` (the `status` of the receipt chain-audit conclusion over the peer's **whole** committed receipt set, exactly as `GET /v1/replication/status` reports it — `"ok"` or `"broken"`).
- `nextCursor`: the number of peers skipped after this page — feed it back as the next `after`; `hasMore` reports whether further peers remain.
- `peerCount`: the complete registered peer count, never just the page size.
- `totals`: `pos`, `left`, and `acks` summed over the **complete** registered set, not just the current page.
- `anomalies`: for each of the receipt chain-audit's four anomaly classes — `gaps`, `overlaps`, `identityMismatches`, and `cursorRegressions` — the number of registered peers whose whole-chain audit reports a non-empty list for that class, again over the complete registered set.

Every number in the response is a JSON integer. An empty registered set reports an empty page with an all-zero summary. The page, the cursor, the count, the totals, and the anomaly counts are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so the overview always describes a single commit even while commits are in flight.

A missing or extra path segment (for example `/v1/replication`, `/v1/replication/status/all/extra`, or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with a `Bearer` challenge, in scope-policy mode an authenticated token lacking the read scope is HTTP 403 without a challenge, and `/health` stays anonymous. With `--data-file`, the log, the checkpoints, and the receipts are rebuilt identically during recovery, so the same state yields the same overview before and after a restart.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
