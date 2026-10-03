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

All twenty POST endpoints (`POST /v1/replicas/{replicaId}/operations`, `POST /v1/sync/operations`, `POST /v1/states/{key}/resolve`, `POST /v1/states/{key}/resolve/auto`, `POST /v1/resolve/auto/batch`, `POST /v1/resolve/auto/plan`, `POST /v1/sync/peers/{peerId}/checkpoint`, `POST /v1/sync/peers/{peerId}/acknowledge`, `POST /v1/transactions/apply`, the read-only `POST /v1/transactions/plan`, `POST /v1/transactions/{transactionId}/compensate`, `POST /v1/replication/apply`, the read-only `POST /v1/states/{key}/causal-at`, the read-only cross-key `POST /v1/states/causal-at`, `POST /v1/replication/compare`, `POST /v1/replication/plan`, `POST /v1/replication/consensus`, `POST /v1/replication/repairs/plan`, and `POST /v1/replication/repairs/diagnosis`, and the committing `POST /v1/replication/repairs/apply`) share one body-size contract:

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

The `checkpoints` section is optional and holds sender-side replication cursors (see below); a file written before checkpoints existed contains only `version` and `operations`, and recovers with no registered checkpoints. The `policies` section is likewise optional and holds the automatic-resolution policy bindings (see below): one `{"replicaId","operationId","policy"}` record per accepted automatic resolution, committed atomically with its operation. The `transactions` section is likewise optional and holds the atomic-transaction bindings (see below): one `{"transactionId","operations"}` record per accepted transaction, committed atomically with its operations. The `acks` section is likewise optional and holds the consumption receipts (see below): one `{"peerId","ackId","cursor","operations"}` record per accepted acknowledgement, committed atomically with the checkpoint advance it caused. The `repairExecutions` section is likewise optional and holds the conditional replication-repair executions (see below): one `{"peerId","ackId","expectedCheckpoint","expectedReceipts","suggestions","results","cursor"}` record per accepted repair execution, committed atomically with the checkpoint cursor it restored; like a receipt, a repair is not an operation and never enters the accepted log. The `policyEvents` section is likewise optional and holds the scope-policy change history (see below): one `{"sequence","digest","tokens"}` record per successful scope-policy hot reload, committed atomically with the policy replacement; a file written before policy auditing existed recovers with an empty history. The `compensations` section is likewise optional and holds the verifiable transaction-compensation bindings (see below): one `{"compensationId","transactionId","expectedPlanDigest","operations","status"}` record per accepted compensation, committed atomically with its compensation operations; like a transaction binding, a compensation is local and never enters the exported sync records, and a file written before compensations existed recovers with an empty compensation history. `version` stays `1`: the supplemented format is backward compatible, and an old file is upgraded on disk the first time a checkpoint (or any other new commit) is persisted.

### Optional vector-clock width admission bound: `--max-clock-components N`

The service admits any well-formed vector clock by default: a clock only needs non-empty string component names, non-negative integer ticks, and (for operation clocks) the issuing replica id. Pass `--max-clock-components N` to add a configurable causal-metadata admission bound on top of that baseline:

```bash
PYTHONPATH=src python3 -m semantic_state_engine.server --max-clock-components 8
```

- `N` must be a plain decimal integer between **1 and 1024** inclusive. An invalid value (a sign, whitespace, decimals, non-ASCII digits, zero, or anything above 1024) makes startup fail with exit code 2 before any port is bound, before the authentication configuration or any data file is read, and without creating the target data file. Without the option every documented request, response, persistence, and recovery behavior is unchanged.
- With the bound set, every clock participating in a causal decision may hold **at most N components**: the `operation.clock` of writes and sync imports, the clocks of manual and automatic resolutions, transaction and compensation entry clocks, the remote-snapshot clocks of the replication compare/plan/consensus/apply endpoints, and the caller-supplied boundary clocks of the single-key and cross-key causal-at queries. A request carrying an over-wide clock in any of these positions fails with HTTP 400 and `{"error":"invalid_request"}` before any business state is read: it adds no operation, candidate, audit, metric, checkpoint, receipt, or repair record, creates no temporary file, and leaves the data file untouched. A batch or transaction containing even one over-wide entry is rejected as a whole under the existing atomicity rules.
- Committed replays keep the existing idempotency rules: an identical replay of an accepted `(replicaId, operationId)` is still answered from the committed binding, and only an over-wide clock actually present in the request triggers the new 400 — the rejection never rewrites an existing binding.
- Startup recovery applies the same bound: if any clock stored in the data file — in an operation, an automatic-resolution policy's operation, a transaction, a compensation, a replication-repair record, or a causal-boundary record — exceeds N components, startup fails with exit code 2, the file is left unchanged, and no port is opened. A width-compliant file recovers exactly as it would without the bound: candidates, log order, audit digests, and persistence results are identical.
- The bound constrains only the component count. Tick ranges, clock domination and concurrency classification, key/value constraints, the sync export format, paging summaries, error priorities, the request body limit, and the authentication and scope semantics are all unchanged, and old data files keep recovering when the option is not passed.

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
- Scopes authorize HTTP methods: `read` accesses every documented GET and the read-only POSTs — the batch preview `POST /v1/resolve/auto/plan`, the transaction preflight `POST /v1/transactions/plan`, the causal-slice query `POST /v1/states/{key}/causal-at`, the cross-key causal snapshot `POST /v1/states/causal-at`, the cross-replica comparison `POST /v1/replication/compare`, the cross-replica synchronization plan `POST /v1/replication/plan`, the multi-replica convergence-consensus summary `POST /v1/replication/consensus`, the replication-repair preflight `POST /v1/replication/repairs/plan`, and the replication-repair lifecycle diagnosis `POST /v1/replication/repairs/diagnosis`; `write` submits the nine state-changing business POST endpoints; and `admin` covers both classes (it implies read and write) plus the scope-policy reload endpoint, the scope-policy change-audit and audit-verification endpoints, and the full-store export endpoint. Apart from the read-only batch preview, transaction preflight, causal-slice query, cross-key causal snapshot, cross-replica comparison, synchronization plan, convergence consensus, repair preflight, and lifecycle diagnosis, the state-changing POST endpoints are not reachable with only `read` and the GET endpoints are not reachable with only `write`; neither `read` nor `write` alone reaches the admin-only reload, audit, and store-export endpoints. `GET /health` stays anonymous in every mode.
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

#### Exporting the whole store: `GET /v1/admin/store/export`

An administrator can export the complete committed store as one version-1 document — exactly the document the persistence layer commits to `--data-file` — produced from a single atomic snapshot. Unlike the scope-policy entries above, this entry is published in **every** mode (anonymous, single-token, and scope-policy).

- The shared authentication contract applies: a missing, duplicated, or malformed `Authorization` header, or a token mismatch, is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge. When scope policy mode is enabled the entry requires the `admin` scope: an authenticated token carrying only `read` and/or `write` is HTTP 403 with `{"error":"forbidden"}` and **no** challenge. The scope decision runs before the query is parsed, so a bad query for a non-admin token is still 403, and an unauthorized request never observes a snapshot. `GET /health` stays anonymous in every mode.
- The path must be exactly `/v1/admin/store/export`. A missing segment, an extra segment, or a trailing slash is `404 {"error":"not_found"}`, decided **before** any query check — a wrong path shape together with an illegal query is still 404. The endpoint accepts no query parameters: an unknown, repeated, blank, or empty-named or empty-valued parameter is HTTP 400 with `{"error":"invalid_request"}`. A rejected request changes no state.
- On success the response is HTTP 200 with compact UTF-8 JSON terminated by a single newline: an object with exactly nine fields — `version` (always `1`), `operations`, `checkpoints`, `policies`, `transactions`, `acks`, `repairExecutions`, `policyEvents`, and `compensations`. Empty sections are exported as empty arrays or objects, and no internal field or extra metadata ever appears. `operations` follows the shared commit order and `policyEvents` the hot-reload commit order; the remaining bindings keep the stable order of the version-1 recovery document, and every record keeps its persisted field content. The local bindings (automatic-resolution policies, transactions, compensations, consumption receipts, and repair executions) are exported as the local audit data they are — they never enter the incremental sync operation stream.
- The whole response is generated from one committed snapshot under the same commit lock used by local writes, sync imports, transactions, repairs, checkpoint commits, and policy-reload records, so a concurrent commit appears only as the whole state before it or the whole state after it — never a mix of sections from different commits or half a batch. The query is strictly read-only: within the request it neither reads nor rewrites the data file, creates no temporary file, appends to no log, advances no cursor, and changes neither memory nor any idempotency judgment. Without `--data-file` the in-memory document is exported; with it, the export of the current committed state is byte-identical to the data file's content (plus the trailing newline), so the same recovered state exports identical content before and after a restart and the response doubles as a check sample of the startup recovery format.

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

### Reading several keys at one causal boundary

`POST /v1/states/causal-at` is the read-only cross-key counterpart of the single-key causal slice: it answers several keys against the **same** vector-clock boundary in one committed-snapshot read. It coexists with `POST /v1/states/{key}/causal-at`; neither route's body is accepted on the other. The request body is a JSON object with exactly two keys:

```json
{"clock":{"r1":3,"r2":1},"keys":["color","shape","absent"]}
```

- `clock` follows exactly the single-key boundary rules: an object of replica ids mapped to non-boolean, non-negative JSON integers, possibly empty (the causal origin), with the same componentwise coverage rule (missing components count as 0).
- `keys` is a list of **1 to 100** keys, in the order the caller wants the results; every element must be a distinct, non-empty string. The same list twice is two separate requests; a duplicated element within one request is rejected.
- Malformed JSON, a non-object body, a missing or unknown field, a duplicated field (including a duplicated clock component or a duplicated `keys` element), a structurally illegal clock (the same cases as the single-key route), a `keys` value that is not a list, an empty list, a list of more than 100 elements, or an element that is not a non-empty string all return HTTP 400 with `{"error":"invalid_request"}`.
- The route accepts no query parameters: any parameter — including a repeated name or a blank name/value — returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing, empty, or extra path segment (for example `/v1/states/causal-at/extra` or `/v1/states/causal-at/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter and body checks. A `GET` on the path answers HTTP 404.

The accepted log is replayed **once**, in global commit order: starting from the empty state, every first-accepted record whose clock is componentwise no greater than the boundary is applied through the same candidate add/delete semantics as the single-key replay — ordinary writes, stale writes, accepted repairs, and sync imports — maintaining one candidate set per requested key. Because the whole pass runs under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, every result entry describes one and the same commit: a concurrent commit can only move the entire batch from one complete result to another, never showing one key the old log and another the new.

A key with no candidate inside the boundary does not fail the batch: it is reported as `"absent"` with an empty candidate array. A successful HTTP 200 response is always a compact UTF-8 JSON object with exactly four fields, terminated by a single newline:

```json
{"clock":{"r1":3,"r2":1},"results":[{"key":"color","status":"conflict","candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"},{"clock":{"r2":1},"operationId":"op-2","replicaId":"r2","value":"red"}]},{"key":"shape","status":"resolved","candidates":[{"clock":{"r1":3},"operationId":"op-3","replicaId":"r1","value":"round"}]},{"key":"absent","status":"absent","candidates":[]}],"found":2,"missing":1}
```

- `clock`: the requested boundary, echoed back exactly as sent.
- `results`: one entry per requested key, in request order. Each entry carries exactly `key`, `status`, and `candidates`; `status` is `"resolved"` or `"conflict"` under the same classification as the single-key query when the boundary holds candidates for the key, and `"absent"` with `candidates: []` otherwise. Candidates are always an array, sorted by `(replicaId, operationId)` ascending, each entry carrying exactly `value`, `clock`, `replicaId`, and `operationId`.
- `found`: the number of entries with candidates (`resolved` plus `conflict`).
- `missing`: the number of `absent` entries; `found + missing` always equals the number of requested keys.

Every number in the response is a JSON integer (clock ticks and the two counts); no float, negative zero, or non-finite value can appear. The body is compact UTF-8 JSON with the field order above and one trailing newline. The request is strictly read-only — it mutates neither memory, the data file, logs, audit records, nor metrics, creates no temporary file, and leaves every existing query, write, sync, transaction, repair, audit, and persistence behavior unchanged. With `--data-file`, the accepted log is recovered identically during startup, so the same accepted history and request yield byte-identical responses before and after a restart. The route shares the common request contract: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered **before** reading the body and before authentication; a missing, duplicated, or malformed `Authorization` header or a non-matching bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge; and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge. `/health` stays anonymous.

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
- `suggestion`: `{"lowest_identity":C,"highest_identity":C}` reporting which current candidate each of the two identity-based automatic-resolution policies would select — the smallest and largest `(replicaId, operationId)` — in the same shape as the `candidates` entries. The value policies (`lowest_value`/`highest_value`) do not add keys here; the response shape is unchanged. The suggestion is purely informational: the endpoint creates no repair operation, log record, or checkpoint.

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

### Read-only conflict-pressure metrics

`GET /v1/metrics/conflicts` returns HTTP 200 with a UTF-8 JSON object containing exactly six non-negative integer counters:

```json
{"keys":2,"conflictKeys":1,"candidatePairs":4,"conflictPairs":3,"maxCandidatesInKey":3,"maxDistinctValuesInKey":2}
```

- `keys`: the number of keys that currently hold at least one candidate — the same count reported by `GET /v1/metrics`.
- `conflictKeys`: keys whose candidates do not all agree on a value — the same classification as `GET /v1/metrics`.
- `candidatePairs`: the total number of unordered candidate pairs, counted once per pair within each key (candidates enumerated in ascending `(replicaId, operationId)` order).
- `conflictPairs`: the subset of those pairs whose two candidates disagree on the value; same-value pairs never count as conflicts, even when their clocks are concurrent.
- `maxCandidatesInKey`: the largest candidate count of any single key.
- `maxDistinctValuesInKey`: the largest number of distinct values within any single key.

An empty store reports six zeroes; a single-candidate key contributes only to `keys` (and the two maxima). The endpoint takes no query parameters: any parameter — including a repeated name (`x=1&x=2`) or a blank name/value (`x=`, `x`, `=1`) — returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment or a trailing slash (for example `/v1/metrics/conflicts/`) returns HTTP 404 with `{"error":"not_found"}`, as does any non-GET method.

All six counters are computed from one snapshot under the same commit lock used by local writes, sync-import batches, and repairs, so they always describe a single commit. The request is strictly read-only — it modifies neither memory, the data file, logs, nor audits. With `--data-file`, the recovered candidate state yields the same counters after a restart as just before it.

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
- `policy` must be the literal string `"lowest_identity"`, `"highest_identity"`, `"lowest_value"`, or `"highest_value"`. Any other shape, field, or value (including a non-string `policy`) is HTTP 400 `{"error":"invalid_request"}`.
- There is no `value` and no `candidates` list: both are determined by the server from the key's current candidates.

The request commits only when the key currently holds different value candidates. The resolution value is then chosen deterministically by the policy: `"lowest_identity"` takes the value of the current candidate with the lexicographically smallest `(replicaId, operationId)`, `"highest_identity"` the largest, `"lowest_value"` takes the smallest candidate string value and `"highest_value"` the largest, compared by Unicode code point in ascending order (ties on identity do not matter for the identity policies; a repeated extreme value under a value policy still resolves to that same value and does not change the identity idempotence rules). The request clock must dominate **every** current candidate. The resolution is accepted atomically as one operation in the shared commit order, exactly like a manual resolution: the dominated candidates are cleared and the chosen value becomes the only version. Because it is an ordinary accepted operation, it is exported by `GET /v1/sync/operations` (as the chosen `value` together with the request's `replicaId`/`operationId`/`clock`), imported by `POST /v1/sync/operations`, appears in the key's audit stream and audit digest, counts in every metrics counter and in the verification digest, is persisted to `--data-file`, and is recovered on restart.

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

- `resolutions` must contain between 1 and 100 entries, kept in request order. Each entry has exactly `key`, `replicaId`, `operationId`, `clock`, and `policy`: the same four fields as the single-key automatic resolution plus its target `key` (the route carries no path segment). Every field obeys the single-key constraints: non-empty strings, a clock of non-boolean non-negative integer components containing the entry's `replicaId`, and `policy` equal to `"lowest_identity"`, `"highest_identity"`, `"lowest_value"`, or `"highest_value"`.
- Every numeric value the request carries must be an integer: JSON floats (including `1.0`), negative zero (`-0.0`), and the non-finite tokens `NaN`/`Infinity`/`-Infinity` are all rejected as malformed input.
- No two entries may name the same `key`, and no two may carry the same `(replicaId, operationId)` identity, even across different keys.
- An empty batch, more than 100 entries, a duplicate key or identity, malformed JSON, an unknown field at the root or on an entry, or a structurally illegal clock all return HTTP 400 with `{"error":"invalid_request"}`; nothing changes.
- Extra path segments (for example `/v1/resolve/auto/batch/extra`) return HTTP 404 with `{"error":"not_found"}`.

Entries are processed **in request order**, each with exactly the single-key semantics: the key must currently hold different-valued candidates, the policy selects the value — the candidate with the smallest or largest `(replicaId, operationId)` for the identity policies, or the smallest or largest candidate string value in Unicode code-point ascending order for the value policies — and the entry clock must dominate every current candidate of its key at that position in the sequence. Any single failure rejects the **whole batch**: earlier entries in the same request are not partially committed, and memory, the identity index, policy bindings, and the data file stay exactly as they were before the request.

- No conflict (a missing key, candidates that already agree), a candidate set that moved, or a legal clock that does **not** dominate the current candidates returns HTTP 409 with `{"error":"resolution_conflict"}` (the clock itself must still be structurally valid — an illegal clock is the `400 invalid_request` above).
- A known `(replicaId, operationId)` with a different binding (different key, clock, or policy, or an identity committed without a policy binding) returns HTTP 409 with `{"error":"operation_conflict"}`. The same identity with the same binding is answered from the committed operation, in whatever position it occurs.
- HTTP 201 with `{"status":"created","resolutions":[...],"accepted":A,"replayed":R}` when at least one entry newly commits; HTTP 200 with `"status":"ok"` when **every** entry is a replay. `accepted`/`replayed` are integer counts. `resolutions` has one result per entry, in request order, each carrying `key`, `replicaId`, `operationId`, the selected string `value`, and `policy`. A pure-replay batch appends no log records.
- All new operations commit together once, so batch repairs share the single global commit order: they are exported by `GET /v1/sync/operations`, appear in per-key audit streams and audit digests, move every metrics counter and the verification digest, are addressable in the per-operation archive, are persisted to `--data-file` (with their policy bindings) in one atomic commit, and recover identically after a restart. The policy binding stays local to the resolving replica: an importing replica holds the operation without the binding, so replaying the batch entry there is an operation conflict.
- The response body is compact JSON with no insignificant whitespace and ends with exactly one newline; every numeric field is a JSON integer.
- With `--data-file`, the whole batch (operations and bindings together) is persisted before the 201 response; a durable failure returns HTTP 500 `{"error":"internal_error"}` and leaves memory and the file exactly as they were — the batch can be retried. After a restart the results, replay `200`, and conflict `409` are identical to a process that never restarted.

### Read-only batch preview (plan)

`POST /v1/resolve/auto/plan` shows a caller what a batch automatic resolution *would* do, without committing anything. The request body is exactly the committing batch's body — the same JSON object with one `resolutions` key, 1-100 entries in request order, distinct target keys, distinct `(replicaId, operationId)` identities, and the same per-field constraints (non-empty strings, structurally valid clocks, and `policy` equal to `"lowest_identity"`, `"highest_identity"`, `"lowest_value"`, or `"highest_value"`). Malformed JSON, an empty or oversized batch, a duplicate key or identity, an unknown field, an unknown or non-string policy, an illegal identity or clock, and any float (including `1.0`, `-0.0`, `NaN`, `Infinity`, or `-Infinity`) return HTTP 400 with `{"error":"invalid_request"}` and no results.

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

### Read-only transaction preflight

`POST /v1/transactions/plan` is the strictly read-only preview of `POST /v1/transactions/apply`. The request body is byte-for-byte the same `{"transactionId","operations"}` document with the same 1-100 conditional writes to distinct keys: every field, candidate, and clock validation, the 1 MiB body limit, and every HTTP 400 `{"error":"invalid_request"}` boundary are identical to apply. The preflight reproduces apply's staged judgment against **one complete snapshot** in request order, but it only reports the outcome: it creates no operation, transaction binding, candidate, accepted-log record, audit entry, metric, or any other persisted change, and it creates no temporary file. In particular a previewed transaction id is never bound, so previewing a request and then applying it is still a fresh `201`.

- A new operation is counted as accepted only when its expected candidates exactly equal the key's current candidate identity set **and** its clock strictly dominates every one of those candidates; expectations are checked against the staged state left by earlier entries of the same request, exactly as in apply. An operation whose `(replicaId, operationId)` is already known with identical content counts as replayed and reports the committed value, skipping the state check. A transaction id already bound to identical content is answered as a whole replay from the committed operations (even after the keys moved on); a transaction id or operation identity already known with different content returns HTTP 409 with `{"error":"operation_conflict"}`; a candidate-set mismatch returns HTTP 409 with `{"error":"transaction_conflict"}`; a structurally legal clock that does not dominate its expected candidates returns HTTP 400 with `{"error":"invalid_request"}`. Any failure answers with the error body alone — a failure never returns a partial plan.
- Success is always HTTP 200 with exactly five top-level fields: `{"status":"planned","transactionId":T,"operations":[...],"accepted":A,"replayed":R}`. The top-level `status` is the fixed string `"planned"`; `accepted` counts entries a subsequent apply would newly create and `replayed` counts same-content entries. `operations` has one result per entry in request order, each carrying exactly `key`, `replicaId`, `operationId`, and the value a commit would observe. The accepted/replayed counts and the per-entry results equal the division a subsequent apply of the same request observes against the same snapshot. The body is compact UTF-8 JSON with no insignificant whitespace and ends with exactly one newline; the route's `400`/`409` error responses carry the same terminator.
- The preview observes a single snapshot: a concurrent commit moves the store directly from one complete snapshot to another, so a returned plan is never a mix of two snapshots, and a plan already returned does not change because of a later commit. Repeating the same preview against unchanged state yields the identical response; after applying, the same request previews as replays rather than new entries.
- With `--data-file`, the preview reads the same durable state but never writes it — file bytes and modification time are unchanged, no temporary file is created, and the same history before and after a restart produces the same plan, including the accepted/replayed split.
- The route accepts no query parameters: any parameter (unknown, repeated, or blank) returns HTTP 400 with `{"error":"invalid_request"}`, decided before the body is read. The path must be exactly `/v1/transactions/plan`: a missing or extra segment (for example `/v1/transactions/plan/extra`), a trailing slash, or any other shape returns HTTP 404 with `{"error":"not_found"}`, and the route-shape decision takes priority over the query and body (a `GET` on the path is an unknown route, also `404`).
- The common request contract applies: an illegal `Content-Length` is HTTP 400 `{"error":"invalid_request"}` and a declared length over the body limit is HTTP 413 `{"error":"payload_too_large"}`, both answered before authentication and without reading the body; a missing, duplicated, malformed, or mismatched bearer token is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge, while `/health` stays anonymous. Because the preflight is read-only it follows the common authentication order and is gated by the **read** scope in scope-policy mode: a token carrying `read` or `admin` may call it, and a token lacking both is HTTP 403 `{"error":"forbidden"}` with no challenge. Transaction commit, replication, audit, and all other authentication behavior are unchanged.

### Read-only transaction-ledger audit

`GET /v1/transactions/verify?after=N&limit=N&expectedCount=N&expectedDigest=H` is the read-only audit entry point over the committed atomic transactions: it pages the same transaction bindings that `POST /v1/transactions/apply` persisted and reports an independent integrity conclusion over the complete transaction history. It creates no binding or operation, advances nothing, and changes neither memory nor the data file; transaction commit, replay, conflict, persistence, and recovery behavior are all unchanged.

- `after`, `limit`, `expectedCount`, and `expectedDigest` are all **required**, each appearing exactly once. `after` is a non-negative ASCII decimal integer — the number of committed transactions already skipped, starting at `0`; `limit` is an ASCII decimal integer between `1` and `100`; `expectedCount` is a non-negative ASCII decimal integer naming the full transaction count the caller expects; `expectedDigest` is exactly 64 lowercase hexadecimal characters naming the full-history digest the caller expects. A missing, repeated, unknown, blank, signed, decimal-point, whitespace-bearing, or non-ASCII-numeral parameter, a `limit` outside `1-100`, and an uppercase, non-hex, or wrong-length `expectedDigest` all return HTTP 400 with `{"error":"invalid_request"}`. An `after` equal to the current transaction count is a valid stable empty page; an `after` past it is HTTP 400.
- The path must be exactly `/v1/transactions/verify`. A missing or extra segment, a trailing slash, or any unknown route is `404 {"error":"not_found"}`, decided before any query check. The plain transaction-commit route `POST /v1/transactions/apply` is unchanged, and a `POST` on this path is an unknown route (`404`).
- When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge; `/health` stays anonymous.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly seven fields in this order:

```json
{"transactions":[{"transactionId":"tx-1","operations":[{"key":"color","replicaId":"r3","operationId":"tx-op-1","value":"blue","clock":{"r1":1,"r3":1},"candidates":[{"replicaId":"r1","operationId":"op-1"}]}]}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","transactionsCount":1,"verification":{"status":"ok","duplicateTransactionIds":[],"recordViolations":[],"identityMismatches":[],"batchViolations":[],"digestMismatches":[],"countMismatches":[]}}
```

- `transactions`: one page of the transaction history in creation (commit) order. Each item carries exactly `transactionId` and `operations`; `operations` keeps the transaction's original operation order, and each operation carries exactly `key`, `replicaId`, `operationId`, `value`, `clock`, and `candidates` with their committed values — the same entry shape the transaction committed (the candidate set and the clock components are emitted in lexicographic order).
- `nextCursor`: the number of transactions skipped after this page — feed it back as the next `after`; `hasMore` reports whether further transactions remain. Both describe only the page, so an `after` at or past the count and an empty history both return an empty page.
- `algorithm` is always `"sha256"`. `transactionsCount` counts the **complete** history, never just the page.
- `digest` summarizes the whole history, so it is identical on every page. The hash input is a compact UTF-8 JSON array with one element per committed transaction in creation order, each written with its fields in the fixed order `{"transactionId":T,"operations":[...]}`; each operation is written in the fixed order `{"key":K,"replicaId":R,"operationId":O,"value":V,"clock":C,"candidates":[...]}` with the clock's component names sorted lexicographically and each candidate identity written as `{"replicaId":R,"operationId":O}` in lexicographic `(replicaId, operationId)` order. No whitespace appears anywhere, numbers are plain JSON integers, and strings escape only the quote, the backslash, and U+0000-U+001F control characters. An empty history hashes the empty array `[]`.
- `verification` carries the independent integrity conclusion over the complete history (also independent of the page), with exactly seven fields. The first four lists are internal anomaly scans, and every marker retains the transaction's 0-based `transactionIndex` in creation order and its `transactionId`:
  - `duplicateTransactionIds`: each later record whose `transactionId` an earlier record already claimed — `{"transactionIndex":I,"transactionId":T}` (the repeated occurrence only).
  - `recordViolations`: each record that is otherwise malformed (a non-string or empty id, or an operations list that is not 1-100 entries each with exactly the six well-formed fields: non-empty strings, a legal clock, or a malformed or internally duplicated candidate list) — `{"transactionIndex":I,"transactionId":T}`.
  - `identityMismatches`: each transaction operation whose stored identity content differs from the accepted operation under the same `(replicaId, operationId)` identity (a missing accepted operation included) — `{"transactionIndex":I,"transactionId":T,"operationIndex":J,"replicaId":R,"operationId":O,"expected":{...}|null,"observed":{...}}`; the marker additionally names the 0-based `operationIndex` within the transaction, the `replicaId` and `operationId`, and gives both sides' identity content as `{"operationId","key","value","clock"}` — `expected` being `null` when no accepted operation carries the identity and otherwise the archive content, `observed` the identity content the transaction record names.
  - `batchViolations`: each later operation in one transaction repeating a key or `(replicaId, operationId)` identity an earlier operation of that transaction already used — one marker per repeated occurrence: `{"transactionIndex":I,"transactionId":T,"operationIndex":J,"replicaId":R,"operationId":O}`.
  - `digestMismatches`: at most one `{"expected":D,"observed":D}` — the caller's `expectedDigest` first, the independently recomputed full-history digest second.
  - `countMismatches`: at most one `{"expected":C,"observed":N}` — the caller's `expectedCount` first, the actual full transaction count second.
  - `status`: `"ok"` when all six anomaly lists are empty, otherwise `"broken"`. An empty history is intact: `"ok"` for the empty-array digest and count `0`.

The live history is appended one validated transaction at a time (each successful transaction commits its binding together with its operations), so the conclusion is `"ok"` by construction; the scan independently re-checks the records, identities, and batch uniqueness against the actual snapshot and never trusts a materialized value. Paging trims only the exported `transactions` page — the digest, count, and verification always cover the complete history on every page. The page slice, cursor, remaining flag, digest, count, and conclusion are computed from one snapshot under the same commit lock used by writes, imports, transactions, repairs, and checkpoint/acknowledgement commits, so a concurrent commit is observed only as the whole old or the whole new history. The query is strictly read-only: it changes neither memory nor the data file and creates no temporary file. With `--data-file`, the bindings are rebuilt identically during recovery (a file written before transactions existed recovers with an empty history), so a restart reports the same pages, full-history digest, `transactionsCount`, and `verification` conclusion.

### Verifiable transaction compensation

Two endpoints cooperate to undo one **committed** transaction when it has no causal successor: a strictly read-only plan, followed by an atomic compensation commit. The original transaction commit and verification, ordinary writes, sync, repairs, idempotent replays, audits, and data-file recovery are all unchanged.

#### Planning a compensation

`GET /v1/transactions/{transactionId}/compensation` builds the plan from the pre-transaction log snapshot and the current snapshot. The `transactionId` must name a committed transaction; an unknown id returns HTTP 404 with `{"error":"not_found"}`. Any query parameter returns HTTP 400 with `{"error":"invalid_request"}`; extra path segments or a trailing slash return HTTP 404. The request is strictly read-only — it writes no candidate, log entry, audit record, binding, or data file and creates no temporary file.

A successful HTTP 200 response is:

```json
{"transactionId":"tx-1","conclusion":"reversible","keys":[{"key":"color","beforeCandidates":[{"value":"red","clock":{"r1":1},"replicaId":"r1","operationId":"op-1"}],"currentCandidates":[{"value":"blue","clock":{"r1":1,"r3":1},"replicaId":"r3","operationId":"tx-op-1"}],"operation":{"key":"color","replicaId":"r3","operationId":"compensation:tx-1:r3:tx-op-1","value":"red","clock":{"r1":1,"r3":2}}}],"descendants":[],"algorithm":"sha256","expectedPlanDigest":"<64 lowercase hex chars>"}
```

- `keys` has one entry per transaction-affected key, in the transaction's operation order. Each carries `key`, `beforeCandidates` (the candidates the key held just before the transaction — its committed expected set, identity-sorted), `currentCandidates` (the key's current candidates, identity-sorted), and `operation` (the compensation write that restores the unique before value, or `null`). Each candidate carries exactly `value`, `clock`, `replicaId`, and `operationId`.
- A planned `operation` carries exactly `key`, `replicaId`, `operationId`, `value`, and `clock`: it restores the unique before-transaction candidate's value under a deterministic new identity (`compensation:{transactionId}:{replicaId}:{transactionOperationId}`) and a clock that strictly dominates the corresponding transaction operation clock (the componentwise maximum of the before clock and the transaction clock, with the transaction replica's component advanced one tick).
- `conclusion` is `"reversible"` only when every affected key held **exactly one** before-transaction candidate, the transaction operation is still that key's **sole** current candidate, and **no causal descendant** exists — an accepted record committed after the transaction whose clock strictly dominates any transaction operation clock. Otherwise the conclusion is `"blocked"` and `descendants` lists every blocking descendant in shared-log order, each with `key`, `replicaId`, `operationId`, and `clock`.
- `algorithm` is always `"sha256"` and `expectedPlanDigest` is the 64-character lowercase SHA-256 of the canonical plan input (the transaction id, the per-key before/current evidence and operations, and the descendants, with every clock's components sorted lexicographically). The derived `conclusion` string is not itself hashed (it follows from the evidence), but the blocking descendants are: so a client that presents an older `expectedPlanDigest` after a causal successor appeared holds a stale plan and is rejected. With `--data-file`, the log and bindings are rebuilt identically during recovery, so the same committed history yields the same plan and digest before and after a restart.

#### Committing a compensation

`POST /v1/transactions/{transactionId}/compensate` accepts a JSON object with exactly three keys:

```json
{"compensationId":"comp-1","expectedPlanDigest":"<64 lowercase hex chars>","operations":[{"key":"color","replicaId":"r3","operationId":"compensation:tx-1:r3:tx-op-1","value":"red","clock":{"r1":1,"r3":2}}]}
```

- `compensationId` is a non-empty string; `expectedPlanDigest` must be exactly 64 lowercase hexadecimal characters (the digest of the plan the operations were copied from); `operations` holds 1-100 entries in request order, each with exactly `key`, `replicaId`, `operationId`, `value`, and `clock`, obeying the ordinary write constraints with no repeated key or `(replicaId, operationId)` identity. The entries must copy the plan's compensation operations **field by field** in plan order; every compensation clock is re-checked to dominate the corresponding transaction operation clock.
- A new `compensationId` commits only when the transaction exists, the freshly recomputed plan still hashes to `expectedPlanDigest`, and that plan is still `reversible`. The operations then commit as one atomic transaction (the same single-commit discipline as `POST /v1/transactions/apply`): they enter the shared accepted log in request order and flow through sync export, audits, metrics, and verification like any ordinary operation, while the compensation binding itself is local and is never exported.
- A successful response is HTTP 201 with `{"status":"created","compensationId":...,"transactionId":...,"operations":[...],"finalState":[...]}`; each committed operation carries `key`, `replicaId`, `operationId`, and `value`, and `finalState` summarizes the post-compensation candidate of every affected key.
- An identical replay — the same `compensationId` **and** `expectedPlanDigest` with the same operation content — returns HTTP 200 with `"status":"ok"` and appends no new version, even if the keys have since moved on. The same `compensationId` with different content returns HTTP 409 with `{"error":"operation_conflict"}`.
- A stale or blocked plan — the recomputed digest differs, the plan is `blocked`, or a causal descendant appeared — returns HTTP 409 with `{"error":"compensation_conflict"}` and leaves the store unchanged. An unknown transaction returns HTTP 404 with `{"error":"not_found"}`.
- Malformed JSON, a wrong field set, an invalid id, digest, identity, or clock, a repeated key, a non-matching or mis-ordered operation list, or any other request-body error returns HTTP 400 with `{"error":"invalid_request"}`. A declared Content-Length over the shared body limit returns HTTP 413 with `{"error":"payload_too_large"}`. Any query parameter returns HTTP 400; extra path segments or a trailing slash return HTTP 404.
- After a successful commit the compensation operations are visible once the atomic durable commit lands; with `--data-file` the operations and the `{compensationId, transactionId, expectedPlanDigest, operations, status}` binding ride in the optional `compensations` section and are rebuilt identically on recovery, so restart preserves the plan digest, idempotent replay decisions, and audit verification. A data file written before compensations existed simply has no `compensations` section and continues to recover.

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

### Offline log-prefix inclusion root

`GET /v1/audit/proofs/root` returns a read-only summary of the **current log prefix** under a stable SHA-256 prefix tree, so an external party can later prove that a single record belongs to that prefix. It takes no query parameters. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields in this order:

```json
{"algorithm":"sha256","treeSize":N,"root":"<64 lowercase hex chars>","auditHead":"<64 lowercase hex chars>"}
```

- `algorithm`: the hash algorithm, always `"sha256"`.
- `treeSize`: the current log length — the number of records in the summarized prefix.
- `root`: the prefix-tree root over every accepted record in global commit order (see the tree rules below). For an empty log it is `SHA256` of the empty byte string.
- `auditHead`: the global audit-chain tail (`head`) over exactly the same prefix — the value `GET /v1/audit/log/chain` reports for that prefix, or 64 `0` characters for an empty log.

The prefix-tree rules are fixed so a third party recomputes every value offline from a record's canonical bytes (the same record encoding the per-key audit digest and global audit chain use):

- **Leaf**: `leafDigest = SHA256(0x00 || recordBytes)` — a one-byte `0x00` domain prefix followed by the record's canonical bytes, encoded as 64 lowercase hex characters.
- **Internal node**: `nodeDigest = SHA256(0x01 || left32 || right32)` — a one-byte `0x01` domain prefix followed by the two children's raw 32-byte digests in left-to-right order.
- **Odd promotion**: levels pair nodes left-to-right; a lone node on an odd-length level is promoted to the next level unchanged (it is not rehashed and contributes no sibling).

The summary is computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoints, so `treeSize`, `root`, and `auditHead` always describe a single commit. Appending to the log never changes an older (smaller) prefix's `root` or its proofs — those cover only the prefix records. The request is strictly read-only and creates no temporary file. Any query parameter — including a repeated, blank-named, or blank-valued one — returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment (for example `/v1/audit/proofs`, `/v1/audit/proofs/root/extra`, or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check. With `--data-file` the log rebuilds identically during recovery, so the same history yields the same `treeSize`, `root`, and `auditHead` after a restart; the endpoint authenticates like every other non-`/health` route.

### Per-operation offline inclusion proof

`GET /v1/replicas/{replicaId}/operations/{operationId}/proof?treeSize=N&root=H` returns one record's **offline inclusion proof** for a log prefix. Both path segments are percent-decoded like every other route, and the two query parameters are required, each appearing exactly once:

- `treeSize`: a non-negative ASCII decimal integer selecting the prefix — the first `treeSize` records of the shared accepted log, from `0` through the current log length. Signs, decimals, whitespace, and non-ASCII numerals are rejected; a `treeSize` past the current log length returns HTTP 400 with `{"error":"invalid_request"}`.
- `root`: exactly 64 lowercase hexadecimal characters — the prefix-tree root the caller expects for that `treeSize` (normally obtained from `GET /v1/audit/proofs/root` or a previously pinned prefix). Uppercase, non-hex, blank, or wrong-length values are rejected.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly seven fields in this order:

```json
{"record":{"replicaId":R,"operation":{...}},"sequence":S,"treeSize":N,"root":"<64 hex>","auditHead":"<64 hex>","leafDigest":"<64 hex>","proof":[{"side":"left","digest":"<64 hex>","position":P}]}
```

- `record`: the existing per-operation archive record — `{"replicaId","operation"}` with the operation's exactly four committed fields (`operationId`, `key`, `value`, `clock`).
- `sequence`: the record's global 1-based position in the log.
- `treeSize`, `root`, `auditHead`: the prefix length actually used and that prefix's tree root and audit-chain tail, identical to the summary endpoint computed at `treeSize`.
- `leafDigest`: the record's tree-leaf digest, `SHA256(0x00 || recordBytes)`.
- `proof`: the inclusion items running **leaf-to-root**, one item per level where the tracked node has a sibling (odd-promotion levels contribute none). Each item has exactly `side` (`"left"` when the sibling is the running node's left neighbour — combine as `SHA256(0x01 || sibling32 || running32)`; `"right"` for the reverse order), `digest` (the sibling's 64-character lowercase hex digest), and `position` (the sibling node's 0-based index from the left within that level). Sides and positions are uniquely determined by the leaf index and prefix size, so an offline verifier need not trust them.

A verifier recomputes the root entirely offline: take `SHA256(0x00 || canonicalRecordBytes)` and confirm it equals `leafDigest`, then for each proof item combine the running digest with `digest` in the stated order using `SHA256(0x01 || left32 || right32)`; the final running digest must equal `root`, and `auditHead` pins the same prefix on the existing audit chain. All seven fields come from one snapshot under the commit lock, so they never mix two batches.

Status codes: a missing, repeated, unknown, blank, or malformed parameter (a non-decimal or out-of-range `treeSize`, or a non-64-lowercase-hex `root`) returns HTTP 400 with `{"error":"invalid_request"}`. An identity that was never first-accepted, or whose record lies outside the requested prefix (`sequence > treeSize`), returns HTTP 404 with `{"error":"not_found"}` (decided before the root comparison). When the record is in the prefix but the recomputed prefix root does not equal the supplied `root`, the endpoint returns HTTP 409 with `{"error":"proof_conflict"}`. A missing, empty, or extra path segment (for example `/v1/replicas//operations/{operationId}/proof`, `/v1/replicas/{replicaId}/operations/{operationId}/proof/extra`, or a trailing slash) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query check. The request is strictly read-only and creates no temporary file. With `--data-file` the same log prefix yields the same `sequence`, `treeSize`, `root`, `auditHead`, `leafDigest`, and `proof` after a restart; the endpoint authenticates like every other non-`/health` route.

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

### Read-only complete-candidate export

`GET /v1/replication/export` incrementally exports the **complete** local candidate snapshot — the same current candidates `GET /v1/replication/snapshot` and `POST /v1/replication/compare` summarize — grouped by business key. The query is strictly read-only: it modifies neither memory nor the data file and creates no temporary file. It accepts exactly three required query parameters:

- `after`: the number of business keys already exported, a 0-based resume cursor that starts at `0`. An `after` equal to the current key count is a valid stable empty page; an `after` past it is HTTP 400.
- `limit`: an ASCII decimal integer between `1` and `100` inclusive, bounding the number of **business key groups** per page.
- `expectedDigest`: exactly 64 lowercase hexadecimal characters — the complete candidate digest the caller expects, i.e. the `candidateDigest` of `GET /v1/replication/snapshot` (the verification-digest rules over the current candidate sets).

A missing, repeated, unknown, blank, signed, decimal-point, whitespace-bearing, or non-ASCII-decimal parameter, a `limit` outside `1-100`, or an `expectedDigest` that is not 64 lowercase hexadecimal characters returns HTTP 400 with `{"error":"invalid_request"}`. A missing or extra path segment, an unknown route, or a trailing slash (for example `/v1/replication/export/`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape check takes precedence over the query-parameter check. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly seven fields in this order:

```json
{"snapshot":[{"key":"color","candidates":[{"clock":{"r1":1},"operationId":"op-1","replicaId":"r1","value":"blue"}]}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","keys":1,"candidateVersions":1}
```

- `snapshot`: one page of business-key groups in lexicographic (Unicode code point) key order. Each group carries exactly `key` and `candidates`; the candidate list is the key's **complete** current candidate set in that snapshot — paging trims only whole business keys and never splits the candidates of one key across pages. Each candidate keeps exactly the comparison entry point's shape and ordering: `value`, `clock`, `replicaId`, `operationId`, sorted by `(replicaId, operationId)` ascending.
- `nextCursor`: the number of business keys skipped after this page — feed it back as the next `after`; `hasMore` reports whether further key groups remain.
- `algorithm` is always `"sha256"`.
- `digest` covers the **complete** snapshot under exactly the verification-digest rules (one `{"key","candidates"}` entry per key, fixed candidate/clock field order, compact UTF-8 JSON, minimal string escaping), so it is identical on every page; an empty snapshot hashes the empty array `[]`.
- `keys` and `candidateVersions` count the whole snapshot, never just the page — the same key and candidate totals reported by `GET /v1/metrics` and `GET /v1/verification/digest`. Every count and cursor in the response is a JSON integer; no float, `-0.0`, or non-finite value can appear.

If the committed snapshot's digest does not equal `expectedDigest`, the response is HTTP 409 with `{"error":"export_conflict"}` and carries no page; state is unchanged. The caller keeps the prior page, reads the new digest from `GET /v1/replication/snapshot`, and re-exports from the appropriate cursor. The page groups, cursor, remaining flag, digest, and both counts are all computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, and checkpoint commits, so a read observes only the whole old or whole new snapshot, never half an import batch or a partially applied repair. With `--data-file`, the candidate state is rebuilt identically during recovery, so the same snapshot yields the same pages and digest after a restart. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 `{"error":"unauthorized"}` with the `WWW-Authenticate: Bearer` challenge, and in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 `{"error":"forbidden"}` with no challenge; `/health` stays anonymous. A `POST` on the path is an unknown route and answers HTTP 404.

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

### Read-only replication repair suggestions

`GET /v1/replication/repairs?after=N&limit=N` is the read-only repair-advice entry point over every registered sender-side replication peer. It audits each peer's whole committed confirmation chain against the shared accepted log — the same audit as `GET /v1/replication/status` — and turns every reported gap, overlap, identity mismatch, and cursor regression into one repair suggestion that locates the affected log interval or position, names the suggested action and the target boundary, and is never executed. The endpoint creates no receipt or temporary file, never advances or writes a checkpoint, and changes neither candidates, the accepted log, the audit streams, the metrics, nor the data file.

The query string carries exactly two parameters, both **required** and each appearing exactly once, under the same rules as `GET /v1/replication/status/all`: `after` counts suggestions already skipped (a 0-based resume cursor that starts at `0`) and `limit` is the page size. Both accept only ASCII decimal digits — a missing or repeated parameter, an unknown parameter, an empty value, a sign, a decimal point, whitespace, or non-ASCII numerals returns HTTP 400 with `{"error":"invalid_request"}`, as does a `limit` outside `1-100`. An `after` equal to the suggestion count is a valid stable empty page; an `after` past it is HTTP 400 with `{"error":"invalid_request"}`. No rejected query reads or changes any state.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly seven fields in this order:

```json
{"suggestions":[{"peer":"peer-a","ackId":"ack-2","kind":"gap","action":"resend","location":{"start":1,"end":2},"length":1,"target":{"start":1,"end":2}}],"nextCursor":1,"hasMore":false,"summary":{"peers":1,"receipts":2,"suggestions":1},"anomalies":{"gaps":1,"overlaps":0,"identityMismatches":0,"cursorRegressions":0},"coverage":{"start":0,"end":3},"conclusion":"broken"}
```

- `suggestions`: one page of suggestions in stable order — every registered peer in ascending `peerId` (Unicode code point) order and, within a peer, the confirmation chain in receipt creation order; for one receipt the boundary anomalies (`gap`, then `overlap`, then `cursorRegression`) precede its identity mismatches, which follow their log-position order. Each item carries exactly `peer` (the sending endpoint), `ackId` (the receipt identity), `kind`, `action`, the affected `location`, and the `target` boundary:
  - a `gap` suggests `resend` — the accepted records in the half-open interval `[start,end)` are confirmed by no receipt; `location` and `target` are both `{"start":N,"end":M}` (the previous and the new boundary cursors) and the item additionally carries `length` (`end - start`);
  - an `overlap` suggests `deduplicate` — the records in `[start,end)` are confirmed twice; `location` and `target` are both that interval (the later receipt's start through the previous end, ordered) and the item additionally carries `length`;
  - a `cursorRegression` on an ordinary (non-empty) receipt suggests `resend` — the confirmation cursor moved backwards from the previous boundary, so resending the records in `[start,end)` restores it; `location` and `target` are both that interval and the item additionally carries `length`; a `cursorRegression` reported by a later **empty** receipt (which confirms no record) instead suggests `correct_cursor` over `{"start": cursor, "end": cursor}` — there is nothing to resend, only the cursor to restore — and the item carries `length` `0`; an empty confirmation segment at the **chain head** is a legal anchor and raises no anomaly at all (see the receipt chain-audit rules);
  - an `identityMismatch` suggests `correct_identity` at one log position — `location` and `target` are both `{"position":P}`, and the item additionally reports `expected` (the identity the accepted log holds at that position, or `null` when the position lies outside the current log) and `observed` (the identity the receipt confirms).
- `nextCursor`: the number of suggestions skipped after this page — feed it back as the next `after`; `hasMore` reports whether further suggestions remain.
- `summary`: `peers` (the complete registered peer count), `receipts` (the committed receipt count over every registered peer), and `suggestions` (the complete suggestion count), each a JSON integer.
- `anomalies`: for each of the four chain-audit classes — `gaps`, `overlaps`, `identityMismatches`, `cursorRegressions` — the total number of anomalies of that class across every registered peer's whole chain; each anomaly contributes exactly one suggestion.
- `coverage`: the half-open segment of the shared accepted log spanned by all peers' chains together, from the earliest chain's derived start to the latest confirmation cursor (`{"start":S,"end":E}`); peers without receipts contribute nothing, and when no peer holds any receipt the coverage is the empty interval `{"start":0,"end":0}`.
- `conclusion`: `"ok"` exactly when every registered peer's whole chain is seamless, non-overlapping, identity-consistent, and cursor-monotonic — all four anomaly counts are zero; otherwise `"broken"`, with the suggestions retaining every anomaly's location.

Every number in the response is a JSON integer. An empty registered set returns an empty page and an empty plan: zeroed `summary` and `anomalies`, the empty coverage, and `"ok"`; a set of peers whose chains are all seamless returns the same empty plan with a non-zero receipt count. **Paging trims only `suggestions`**: `summary`, the anomaly counts, `coverage`, and `conclusion` are always computed from all committed receipts and the full accepted log, so every page of one snapshot reports identical summary values. The suggestions, the page slice, the cursors, the summary, the coverage, and the conclusion are computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so a concurrent commit is observed only as a complete old or new snapshot, never a mix; repeated GETs return the identical plan until a separate commit changes the state. The query is strictly read-only — it records no receipt, advances no checkpoint, executes no suggested repair, and creates no temporary file.

A missing or extra path segment (for example `/v1/replication`, `/v1/replication/repairs/extra`, or a trailing slash) or any unknown route returns HTTP 404 with `{"error":"not_found"}`, and the route-shape check takes precedence over the query-parameter check; a non-`GET` method on the path is an unknown route and answers HTTP 404. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with a `Bearer` challenge, in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 without a challenge (a `read`-only token suffices), and `/health` stays anonymous. With `--data-file`, the log, the checkpoints, and the receipts are rebuilt identically during recovery, so the same receipts and log produce the same plan, summary, counts, and locations before and after a restart. This entry point changes neither the README startup entry nor the existing delivery-status, receipt-audit, sync, transaction, write, repair, checkpoint, or persistence behavior.

### Replication-repair execution preflight

`POST /v1/replication/repairs/plan` is the strictly read-only preflight for a conditional replication-repair execution. The body is a JSON object with exactly five keys:

```json
{"peerId":"peer-a","ackId":"exec-1","expectedCheckpoint":4,"expectedReceipts":"<64 lowercase hex chars>","suggestions":[{"action":"resend","ackId":"ack-2","location":{"start":1,"end":2},"target":{"start":1,"end":2}}]}
```

- `peerId` names the targeted sender-side replication peer (a non-empty string); `ackId` names this conditional execution (a non-empty string).
- `expectedCheckpoint` is the non-boolean non-negative integer checkpoint cursor the caller expects the peer to currently hold.
- `expectedReceipts` is the 64-character lowercase hexadecimal SHA-256 the caller expects the peer's whole committed receipt set to digest to — exactly the `digest` reported by `GET /v1/sync/peers/{peerId}/receipts` (an empty set hashes `[]`).
- `suggestions` is the ordered batch, 1 to 100 entries, that a later apply would execute in the fixed processing order. The four actions are `"resend"` (补发), `"deduplicate"` (去重), `"correct_identity"` (纠正身份), and `"correct_cursor"` (纠正游标). A `resend`/`deduplicate`/`correct_cursor` entry carries exactly `action`, `ackId` (the receipt that produced the anomaly), and matching `location`/`target` interval boundaries `{"start","end"}` (non-boolean non-negative integers with `start <= end`); a `correct_identity` entry carries `location`/`target` `{"position":P}` with the same position, plus the `expected` and `observed` `{"replicaId","operationId"}` identities.
- Malformed JSON, a non-object body, a missing or unknown field, a duplicated field, an empty or oversized batch, an unknown action, a structurally illegal boundary, position, or identity, an illegal `expectedCheckpoint`, or an `expectedReceipts` that is not 64 lowercase hexadecimal characters all return HTTP 400 with `{"error":"invalid_request"}` (error bodies contain only the `error` field).
- The route accepts no query parameters: any parameter returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check. A missing, extra, or trailing path segment (for example `/v1/replication/repairs`, `/v1/replication/repairs/plan/`, or `/v1/replication/repairs/plan/extra`) returns HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes priority over the query and body checks. A non-`POST` method on the path answers HTTP 404.

The preflight only checks the suggestions against a **staged view** of one committed snapshot — it never produces a repair. The peer must be registered, its current checkpoint must equal `expectedCheckpoint`, the receipt-set digest must equal `expectedReceipts`, and the suggestions must be in the fixed processing order; a mismatch at that level returns HTTP 409 with `{"error":"apply_conflict"}` and reads/changes nothing further. Each suggestion is then aligned by its `(action, ackId, ordinal)` slot against the advice the read-only `GET /v1/replication/repairs` currently derives for that peer: a slot whose log location or identity has moved reports that item as not executable. A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline:

```json
{"status":"planned","peerId":"peer-a","ackId":"exec-1","results":[{"action":"resend","executable":true,"boundary":{"start":1,"end":2}}]}
```

- `results`: one per requested suggestion in request order, each carrying exactly `action`, `executable` (a JSON boolean — whether the slot still exists in the peer's current advice with the same log location and identity), and `boundary` (the predicted boundary the repair would restore — an interval `{"start","end"}` or a position `{"position":P}` — or `null` when not executable).
- Every number (cursors, positions, starts, ends) is a plain JSON integer; no float, boolean count, negative zero, or non-finite value appears.

The preflight, the anchors, the alignment, and the boundaries are computed under the shared commit lock from one snapshot, so a concurrent commit is observed only as a whole old or new snapshot. It is strictly read-only: it writes no repair record, advances no checkpoint, records no receipt, adds no accepted-log record or candidate, moves no metric, creates no temporary file, and touches no data-file byte; a preflighted execution is therefore still a fresh `201` when first applied. In scope-policy mode the preflight is gated by the **read** scope (or `admin`); the common Content-Length and authentication contract applies unchanged.

### Conditional replication-repair execution

`POST /v1/replication/repairs/apply` is the committing counterpart of the preflight and takes the **exact same body** (`peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, and the ordered `suggestions`), with the same structural 400, path-shape 404, and no-query-parameter rules. The `peerId`, expected checkpoint, expected receipt digest, and ordered suggestions together lock the target.

Under the shared commit lock, a known `(peerId, ackId)` execution is answered from its committed binding first:

- The same peer and `ackId` with the identical `expectedCheckpoint`, `expectedReceipts`, and ordered suggestions replayed returns HTTP 200, adds **no** repair record, and advances the checkpoint no further — the replay is answered from the committed binding however the store has moved since.
- A known `(peerId, ackId)` whose content differs (a different expected anchor or ordered suggestion set) returns HTTP 409 with `{"error":"operation_conflict"}`; all state stays unchanged.

For a new execution every guard must pass against one committed snapshot, with the batch processed **in order** — resend, deduplicate, correct identity, then correct cursor:

- An unregistered peer, an `expectedCheckpoint` or `expectedReceipts` that does not match the committed snapshot, or a suggestion list whose positions do not line up (including a batch not in the fixed processing order) returns HTTP 409 with `{"error":"apply_conflict"}`; the whole batch state is unchanged.
- A suggestion whose `(action, ackId, ordinal)` slot still exists for the same receipt but whose log location or identity no longer matches the committed advice returns HTTP 409 with `{"error":"repair_conflict"}`; there is no half execution.

Only after all guards pass are the repair record and the restored checkpoint committed **together** in one atomic commit (the same `write temp file → fsync → rename → fsync directory` protocol under the shared commit lock as writes, imports, repairs, checkpoints, and acknowledgements), before the caller observes success. Each result restores the boundary its advice named; the checkpoint is advanced to the highest restored interval end that lies ahead of it (never moved backwards). A new execution returns HTTP 201; a pure replay returns HTTP 200. Both are compact UTF-8 JSON objects terminated by a single newline:

```json
{"status":"created","peerId":"peer-a","ackId":"exec-1","results":[{"action":"resend","boundary":{"start":1,"end":2}}],"newExecutions":1,"replayed":0}
```

- `status` is `"created"` for a new execution, `"ok"` for a full replay; `results` gives, in request order, one item per suggestion with exactly `action` and the restored `boundary`; `newExecutions` and `replayed` are JSON integer counts.
- A repair is not an operation: it never enters the accepted log, sync export, the per-key audit, candidate state, or the six metrics counters. With `--data-file` it lives in the optional `repairExecutions` section and, together with the restored checkpoint, is rebuilt identically on restart, so the create `201`, replay `200`, and conflict `409` decisions are identical before and after a restart; an old file without the section recovers with no repair bindings. A durable failure returns HTTP 500 with `{"error":"internal_error"}` and leaves memory, the binding, the checkpoint, and the data file exactly as before, so the request can be retried.
- A structural body problem is HTTP 400 `invalid_request`; an illegal `Content-Length` is HTTP 400 and an over-limit declaration is HTTP 413, both answered before authentication and without reading the body; a bad or missing bearer token is HTTP 401 with a `Bearer` challenge. In scope-policy mode the execution is gated by the **write** or `admin` scope; a token with only `read` answers HTTP 403 with `{"error":"forbidden"}` and no challenge.

The `GET /v1/replication/repairs` advice entry point, the receipt audits, sync, transactions, and recovery behavior are unchanged apart from the corrected empty-receipt classification: an empty confirmation receipt at the chain head is legal and raises nothing, and a later empty receipt at the same cursor reports only a cursor regression (advice `correct_cursor`) — it never produces a gap, overlap, or identity mismatch.

### Replication-repair execution audit

`GET /v1/replication/repairs/executions?after=N&limit=N` is the strictly read-only audit entry point over the committed history of conditional repair executions. It pages one record per execution that committed successfully through `POST /v1/replication/repairs/apply`; an identical replay answered from an existing binding appends no record, so replayed executions never appear twice. The query writes no repair record, advances no checkpoint, records no receipt, creates no temporary file, and changes neither the repair advice, the preflight, executions, receipts, nor persistence behavior.

The query string carries exactly two parameters under the same contract as the repair-advice route: both `after` and `limit` are **required** and appear exactly once; `after` counts executions already skipped (a 0-based resume cursor starting at `0`) and `limit` accepts only ASCII decimal digits from 1 to 100. A missing, repeated, or unknown parameter, an empty value, a sign, a decimal point, whitespace, or non-ASCII numerals returns HTTP 400 with `{"error":"invalid_request"}` without changing any state, as does an `after` past the committed execution count. An `after` equal to the count is a valid stable empty page.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly seven fields in this order:

```json
{"executions":[{"peerId":"peer-a","ackId":"exec-1","expectedCheckpoint":4,"expectedReceipts":"<64 lowercase hex chars>","suggestions":[{"action":"resend","ackId":"ack-2","location":{"start":1,"end":2},"target":{"start":1,"end":2}}],"results":[{"action":"resend","boundary":{"start":1,"end":2}}],"cursor":4}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","executionsCount":1,"verification":{"status":"ok","duplicateBindings":[],"outOfOrderActions":[],"boundaryViolations":[],"checkpointViolations":[],"recordViolations":[]}}
```

- `executions`: the page, ordered by `peerId` in ascending (Unicode code point) order and, within a peer, in creation (first-commit) order. `after` trims only this current list. Each item carries exactly `peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, `suggestions`, `results`, and `cursor` — the execution's `(peerId, ackId)` binding, its expected anchor checkpoint and receipt-set digest, the ordered suggestions it committed (keeping their interval/position boundaries and, for an identity correction, the expected/observed identities), the per-suggestion result evidence in request order, and the checkpoint cursor the execution restored.
- `nextCursor`: the number of executions skipped after this page — feed it back as the next `after`; `hasMore` reports whether further executions remain.
- `algorithm` is `"sha256"`; `digest` is the 64-character lowercase SHA-256 of the canonical compact JSON array of **all** executions in pure creation order — the order the bindings first committed, which is also the order they ride in the data file's `repairExecutions` section — with each record's fields in the fixed order `peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, `suggestions`, `results`, `cursor`; an empty history hashes `[]`.
- `executionsCount` is the full history length, never the page length.
- `verification` is the independent integrity conclusion over the complete history (a snapshot identical to the page, digest, and count): `status` is `"ok"` or `"broken"`, followed by five anomaly lists, each marker retaining the offending execution's 0-based `executionIndex` in creation order plus its `peerId` and `ackId`:
  - `duplicateBindings`: an execution whose `(peerId, ackId)` was already claimed by an earlier execution (the repeated occurrence only);
  - `outOfOrderActions`: an execution whose suggestions are not in the fixed processing order (resend, deduplicate, correct_identity, correct_cursor);
  - `boundaryViolations`: a result whose restored boundary differs from its suggestion's target, additionally naming the 0-based `suggestionIndex` and the `expected`/`observed` boundaries;
  - `checkpointViolations`: an execution whose restored cursor precedes its `expectedCheckpoint`, exceeds the recovered log length, or names a peer whose registered checkpoint has not reached it — the marker gives `expected` (`checkpoint`, the current `registered` checkpoint or `null`, and `logLength`) and `observed` (`cursor`);
  - `recordViolations`: an otherwise malformed stored record (a bad binding, anchor digest, cursor, suggestion, or result shape).

**Paging never changes the summary**: the digest, `executionsCount`, and the `verification` conclusion always cover the whole history from one snapshot under the commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so every page of one snapshot reports identical values, and a concurrent commit is observed only as the complete old or new history. With `--data-file` the history is rebuilt identically during recovery (an old file without the `repairExecutions` section recovers as an empty, intact history), so the page, digest, count, and verification are identical before and after a restart and a recovered binding still replays as `200` without appending.

A missing or extra path segment (for example `/v1/replication/repairs/executions/` or `.../executions/extra`) or any unknown route returns HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes precedence over the query check, while — like every other route — authentication runs before an unknown route's 404, and a non-`GET` method on the path answers HTTP 404. A missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge (a `read`-only token suffices); `/health` stays anonymous. This entry point changes neither the repair advice, preflight, execution, receipt, checkpoint, sync, transaction, write, nor persistence behavior.

### Verifying replication-repair execution history against external expectations

`GET /v1/replication/repairs/executions/verify?after=N&limit=N&expectedDigest=<64 lowercase hex>&expectedCount=N` is the strictly read-only external-verification companion to the execution audit above. It exports the same committed history of conditional repair executions through the same paging and, in addition, independently checks the caller-supplied expectations over the complete history. Like the plain audit it is reachable with the existing execution-audit read permission (in scope-policy mode the `read` or `admin` scope); the query executes no repair, writes no record, advances no checkpoint, records no receipt, creates no temporary file, and changes neither the repair advice, the preflight, executions, receipts, nor persistence behavior.

The query string carries exactly four parameters, each **required** and appearing exactly once:

- `after` and `limit` share the plain audit's paging contract exactly: `after` counts executions already skipped (a 0-based resume cursor starting at `0`) and `limit` accepts only ASCII decimal digits from 1 to 100.
- `expectedDigest` must be exactly 64 lowercase hexadecimal characters — the SHA-256 digest of the complete execution history the caller expects; an uppercase, non-hex, wrong-length, blank, or missing value is rejected.
- `expectedCount` must be a non-negative ASCII decimal integer — the full history length the caller expects — with no sign and no decimal point.

A missing, repeated, duplicated, or unknown parameter, an empty or blank value, a sign, a decimal point, whitespace-bearing or non-ASCII numerals, a `limit` outside `1-100`, or a malformed `expectedDigest` returns HTTP 400 with `{"error":"invalid_request"}` without reading or changing any state, as does an `after` past the committed execution count. An `after` equal to the count is a valid stable empty page.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly the same seven fields in the same order as the plain audit — `executions`, `nextCursor`, `hasMore`, `algorithm`, `digest`, `executionsCount`, `verification` — and each execution carries the same `peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, `suggestions`, `results`, `cursor` record structure:

```json
{"executions":[{"peerId":"peer-a","ackId":"exec-1","expectedCheckpoint":4,"expectedReceipts":"<64 lowercase hex chars>","suggestions":[{"action":"resend","ackId":"ack-2","location":{"start":1,"end":2},"target":{"start":1,"end":2}}],"results":[{"action":"resend","boundary":{"start":1,"end":2}}],"cursor":4}],"nextCursor":1,"hasMore":false,"algorithm":"sha256","digest":"<64 lowercase hex chars>","executionsCount":1,"verification":{"status":"ok","duplicateBindings":[],"outOfOrderActions":[],"boundaryViolations":[],"checkpointViolations":[],"recordViolations":[],"digestMismatches":[],"countMismatches":[]}}
```

- The page ordering and the `after`-to-`limit` paging are identical to the plain audit (ascending `peerId`, creation order within a peer); `after` trims only the current `executions` page.
- The summary still covers **all** first-committed repair executions in pure creation order — the order the bindings first committed and ride in the data file's `repairExecutions` section — with each record hashed in the fixed field order `peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, `suggestions`, `results`, `cursor`; an identical pure replay appends no record, so it never moves the digest or the count. An empty history still hashes the empty array `[]` and reports `executionsCount` 0.
- `verification` independently scans the **complete** history from the same snapshot: it re-runs the five internal anomaly checks exactly as the plain audit does — `duplicateBindings`, `outOfOrderActions`, `boundaryViolations`, `checkpointViolations`, and `recordViolations` keep their original markers and judgement — and adds two external comparisons:
  - `digestMismatches`: at most one marker `{"expected":D,"observed":D}` — the caller's `expectedDigest` first and the SHA-256 independently recomputed over the canonical compact JSON array of the whole creation-order history second (the digest of `[]` for an empty history); empty on agreement.
  - `countMismatches`: at most one marker `{"expected":C,"observed":N}` — the caller's `expectedCount` first and the actual full history length second; empty on agreement.
- `status` is `"ok"` exactly when all seven anomaly lists are empty; any entry in any list makes the verification `"broken"`. Paging never changes the digest, the count, or the conclusion — they are identical on every page of one snapshot, including the stable empty tail.

The page slice, cursor, remaining flag, digest, count, and verification are all computed from one snapshot of the complete history under the commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so a concurrent commit is observed only as the complete old or new history, never a mix. The query is strictly read-only: it changes neither memory nor the data file and creates no temporary file. With `--data-file` the history is rebuilt identically during recovery (an old file without the `repairExecutions` section verifies as an empty, intact history against the empty-array digest and count `0`), so the page, digest, count, and verification conclusion are identical before and after a restart; a recovered binding still replays as `200` without appending.

A missing or extra path segment (for example `/v1/replication/repairs/executions/verify/` or `.../verify/extra`) or any unknown route returns HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes precedence over the query check — a wrong path shape together with an invalid query is still 404, and the plain audit route `/v1/replication/repairs/executions` is unchanged. As on every other route, authentication runs before an unknown route's 404, and a non-`GET` method on the path answers HTTP 404. A missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge (the challenge accompanies only the 401); in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge. The error bodies follow the existing authentication contract and `/health` stays anonymous.

### Cross-stream integrity verification

`GET /v1/integrity/verify` is the strictly read-only entry point that cross-checks the two existing independent audit chains — the atomic-transaction ledger (`GET /v1/transactions/verify`) and the conditional replication-repair execution history (`GET /v1/replication/repairs/executions`) — against the shared accepted-operation log in one committed snapshot. It creates no binding, operation, repair, receipt, or checkpoint, advances nothing, creates no temporary file, and changes neither transaction commit, conditional repair, receipts, sync, writes, nor persistence behavior.

The query takes **no request parameters**: an unknown, repeated, blank, or otherwise present parameter (including a bare `=1` or `x`) returns HTTP 400 with `{"error":"invalid_request"}`. The path must be exactly `/v1/integrity/verify`: a missing segment, an extra segment, a trailing slash, a non-`GET` method, or any unknown route returns HTTP 404 with `{"error":"not_found"}`, decided **before** the parameter check — a wrong path shape together with an illegal query is still 404. When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge (a `read`-only token suffices); `/health` stays anonymous.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly four fields in this order and every count and position a JSON integer:

```json
{"status":"ok","transactions":{"algorithm":"sha256","digest":"<64 lowercase hex chars>","count":2,"anomalies":{"duplicateTransactionIds":[],"recordViolations":[],"identityMismatches":[],"batchViolations":[]}},"repairExecutions":{"algorithm":"sha256","digest":"<64 lowercase hex chars>","count":1,"anomalies":{"duplicateBindings":[],"outOfOrderActions":[],"boundaryViolations":[],"checkpointViolations":[],"recordViolations":[]}},"cross":{"status":"ok","anomalies":{"missingTransactionOperations":[],"unconfirmedRepairCursors":[]}}}
```

- `status` is `"ok"` exactly when the transaction group, the repair group, and the cross group are all intact; any anomaly in any group makes it `"broken"`. It takes only the values `"ok"` and `"broken"`.
- `transactions`: the complete transaction-ledger group, always with the same four fields:
  - `algorithm` is always `"sha256"`.
  - `digest` is the 64-character lowercase SHA-256 in the transaction audit's own canonical encoding — the same full-history digest `GET /v1/transactions/verify` reports, covering the complete creation-order history; an empty history hashes the empty array `[]`.
  - `count` is the full transaction history length.
  - `anomalies` holds the four internal scans of the transaction audit — `duplicateTransactionIds` (a later record reusing a transaction id), `recordViolations` (a malformed stored record), `identityMismatches` (an operation whose stored identity content differs from the accepted operation under the same `(replicaId, operationId)`, a missing accepted operation included), and `batchViolations` (a repeated key or identity within one transaction) — keeping that audit's existing judgement and markers, including each marker's 0-based `transactionIndex`, `transactionId`, and `operationIndex` business location, and `expected`/`observed` content on an identity mismatch.
- `repairExecutions`: the complete repair-execution group with the same four fields:
  - `algorithm` is always `"sha256"`.
  - `digest` is the 64-character lowercase SHA-256 in the repair-execution audit's own canonical encoding — the same full-history digest `GET /v1/replication/repairs/executions` reports over the whole creation-order history; an empty history hashes `[]`.
  - `count` is the full repair-execution history length.
  - `anomalies` holds the five internal scans of the execution audit — `duplicateBindings`, `outOfOrderActions`, `boundaryViolations`, `checkpointViolations`, and `recordViolations` — keeping that audit's existing judgement and markers (0-based `executionIndex`, `peerId`, `ackId`, `suggestionIndex`, and the expected/observed boundaries). A restored cursor below its anchor, past the log length, or not reached by the peer's registered checkpoint stays a group `checkpointViolation`.
- `cross`: the cross-stream conclusion with exactly `status` and `anomalies`. It summarizes **only** inconsistencies between the two local streams and the shared accepted-operation log, never repeating a purely intra-group anomaly:
  - `missingTransactionOperations`: a well-formed transaction operation whose `(replicaId, operationId)` identity the accepted-operation archive carries but the shared accepted-operation log does not place (the transaction ledger binds an identity the shared log never committed). An identity absent from the archive too is already a group `identityMismatch` and is not repeated here. Each marker retains the 0-based `transactionIndex`, the `transactionId`, the 0-based `operationIndex`, and the `replicaId`/`operationId`.
  - `unconfirmedRepairCursors`: a structurally well-formed repair execution whose restored cursor exceeds the shared log length or names a peer whose registered checkpoint has not reached it. Each marker retains the 0-based `executionIndex`, `peerId`, and `ackId`, with `expected` giving the current `registered` checkpoint (`null` for an unknown peer) and `logLength`, and `observed` giving the restored `cursor`. A cursor merely below the execution's own anchor is the group's `checkpointViolations` condition and is not repeated here.
  - `cross.status` is `"ok"` exactly when both cross anomaly lists are empty, otherwise `"broken"`.

An empty transaction history and an empty repair history are both intact: each reports the digest of `[]`, `count` 0, empty anomaly lists, and an `ok` conclusion, and the empty/empty snapshot reports top-level `"ok"`.

The log, the transaction bindings, the repair records, and the checkpoints are all read once from one committed snapshot under the same commit lock used by writes, imports, transactions, repairs, checkpoint commits, and acknowledgement commits, so a concurrent commit is observed only as the complete old or the complete new state, never a mix. The query is strictly read-only: it changes neither memory nor the data file and creates no temporary file. An internal verification failure is HTTP 500 with `{"error":"internal_error"}` and leaves all state exactly as it was. With `--data-file`, both histories and the checkpoints are rebuilt identically during recovery (a file written before either section existed verifies as an empty, intact history), so a restart reports the same group conclusions, digests, counts, and anomaly locations. `/health` stays anonymous and every existing transaction commit, conditional repair, receipt, sync, write, and persistence behavior is unchanged.

### Unified audit root over all persisted audit streams

`GET /v1/integrity/root?expectedDigest=H` is the strictly read-only entry point that binds **every** persisted audit stream under one SHA-256 root: the accepted-operation log (the global audit chain), the atomic-transaction bindings, the conditional replication-repair executions, the per-peer consumption receipts, the scope-policy change events, and the registered sender checkpoint mapping. It creates no binding, operation, repair, receipt, checkpoint, or event, advances nothing, creates no temporary file, and changes neither existing audit behavior nor persistence.

- The query takes exactly one parameter: `expectedDigest` must be present exactly once and hold exactly 64 **lowercase** hexadecimal characters (the SHA-256 root digest the caller expects). A missing, repeated, unknown, blank, or empty parameter, and an uppercase, non-hex, or wrong-length value all return HTTP 400 with `{"error":"invalid_request"}`.
- The path must be exactly `/v1/integrity/root`: a missing segment, an extra segment, a trailing slash, a non-`GET` method, or any unknown route returns HTTP 404 with `{"error":"not_found"}`, decided **before** the query check — a wrong path shape together with an invalid `expectedDigest` is still 404.
- When bearer-token authentication is enabled, the endpoint authenticates like every other non-`/health` route: a missing, duplicated, malformed, or mismatched `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and no challenge; `/health` stays anonymous. The scope decision precedes the query and route checks.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with exactly five fields in this order and every count a JSON integer:

```json
{"algorithm":"sha256","digest":"<64 lowercase hex chars>","evidenceCount":{"streams":6,"records":7,"mappings":2},"streams":[{"name":"acceptedOperations","digest":"<64 lowercase hex chars>","count":3,"status":"ok","anomalies":{...}},{"name":"transactions",...},{"name":"repairExecutions",...},{"name":"receipts",...},{"name":"policyEvents",...},{"name":"checkpoints",...}],"verification":{"status":"ok","rootMismatches":[]}}
```

- `algorithm` is always `"sha256"`, and `digest` is the 64-character lowercase root digest described below.
- `evidenceCount` summarizes the evidence covered: `streams` is always `6`; `records` is the total record count over the five record streams (the accepted log, transactions, repair executions, receipts, and policy events); `mappings` is the number of registered peer-to-cursor checkpoint mappings (the checkpoint stream is a mapping, not a record stream).
- `streams` always covers the same six streams in this fixed order, and each stream object carries exactly `name`, `digest`, `count`, `status`, and `anomalies`:
  1. `acceptedOperations` — the shared accepted-operation log's global audit chain. Its `digest` is the chain-tail `head` reported by `GET /v1/audit/log/chain` (64 zeros for an empty log), `count` is the full log length, and `anomalies` carries the chain verifier's internal locations (`missingSequences`, `duplicateSequences`, `outOfRangeSequences`, `brokenLinks`, `digestMismatches`) in that scan's existing shape.
  2. `transactions` — the complete transaction-ledger history in creation (first-commit) order, with the same full-history `digest`, `count`, and four internal anomaly lists (`duplicateTransactionIds`, `recordViolations`, `identityMismatches`, `batchViolations`) as `GET /v1/transactions/verify`.
  3. `repairExecutions` — the complete repair-execution history in pure creation order, with the same full-history `digest`, `count`, and five internal anomaly lists (`duplicateBindings`, `outOfOrderActions`, `boundaryViolations`, `checkpointViolations`, `recordViolations`) as `GET /v1/replication/repairs/executions`.
  4. `receipts` — every peer's consumption receipts aggregated across senders. Senders are sorted by `peerId` (Unicode code point order); within one sender the receipts keep their first-commit order. The `digest` is the SHA-256 of one canonical compact JSON array concatenating the senders' per-peer receipt arrays in that sender-sorted order using the existing receipt encoding (`{"peerId","ackId","cursor","operations"}`, each identity `{"replicaId","operationId"}` in confirmation order); `count` is the total receipt count; `anomalies` groups the per-peer chain audit's locations (`gaps`, `overlaps`, `identityMismatches`, `cursorRegressions`), each marker additionally carrying its `peerId`. An empty receipt set hashes `[]`.
  5. `policyEvents` — the scope-policy change history in first-commit order, with the same full-history `digest`, `count`, and four anomaly lists (`missingSequences`, `duplicateSequences`, `outOfRangeSequences`, `digestMismatches`) as `GET /v1/admin/scope-policy/audit/verify`.
  6. `checkpoints` — the registered peer-to-cursor mapping with `peerId`s sorted lexicographically. Its `digest` is the SHA-256 of that compact JSON mapping (the same canonical mapping embedded in the replication-snapshot summary; `{}` when empty), `count` is the number of registered peers, `status` is always `"ok"`, and `anomalies` is an empty object.
- The root digest input is a whitespace-free UTF-8 JSON **array** with one element per stream in the fixed order above, each element written `{"name":N,"digest":D,"count":C}` — the stable stream name, that stream's full digest, and its record or mapping count — using plain JSON integers and the standard string escaping (only the quote, the backslash, and U+0000–U+001F control characters). The root `digest` is the SHA-256 of exactly those bytes.
- `verification` independently re-checks all six streams from the same snapshot. Its `status` is `"ok"` exactly when every stream's own integrity status is `ok` **and** the recomputed root equals `expectedDigest`; otherwise it is `"broken"`. On a root mismatch `rootMismatches` carries one marker `{"expected":H,"observed":H}` — the caller's `expectedDigest` first and the independently recomputed root second — and is otherwise empty. A damaged stream makes the verification `broken` even when the presented root digest matches.

All six streams are read once from one committed snapshot under the same commit lock used by writes, imports, transactions, repairs, checkpoint commits, acknowledgement commits, and policy reloads, so a concurrent commit is observed only as the complete old root or the complete new root, never a mix. The query is strictly read-only: it changes neither memory nor the data file and creates no temporary file. An internal failure is HTTP 500 with `{"error":"internal_error"}` and leaves all state exactly as it was. With `--data-file`, the log, bindings, records, receipts, events, and checkpoints are rebuilt identically during recovery (a file written before a section existed treats that stream as empty and intact), so a restart reports the identical stream digests, counts, statuses, anomaly locations, evidence counts, and root digest; `/health` stays anonymous and every existing write, sync, transaction, repair, receipt, checkpoint, audit, and persistence behavior is unchanged.

### Read-only replication-repair lifecycle diagnosis

`POST /v1/replication/repairs/diagnosis` is the strictly read-only entry point that chains one conditional repair execution's suggestion, preflight, execution, and audit stages into evidence. The request body is the exact conditional-execution request the preflight and the committing apply share — a JSON object with exactly `peerId`, `ackId`, `expectedCheckpoint`, `expectedReceipts`, and the ordered `suggestions` (the same structural rules as `POST /v1/replication/repairs/plan`) — and `(peerId, ackId)` locks the one execution under diagnosis. The diagnosis executes no repair: it writes no repair record, advances no checkpoint, records no receipt, adds no accepted-log record or candidate, moves no metric, creates no temporary file, and touches no data-file byte; the candidates, the accepted log, receipts, repair executions, and the data file are exactly as they were before the request.

- The path must be exactly `/v1/replication/repairs/diagnosis`. A missing, extra, or trailing path segment (for example `/v1/replication/repairs`, `/v1/replication/repairs/diagnosis/`, or `/v1/replication/repairs/diagnosis/extra`) answers HTTP 404 with `{"error":"not_found"}`; the route-shape decision takes priority over the query and body checks, so a wrong path shape together with an illegal query or body is still 404. A non-`POST` method on the path (a `GET` included) is an unknown route and answers HTTP 404.
- The route accepts no query parameters: any parameter — including a repeated name or a blank name/value — returns HTTP 400 with `{"error":"invalid_request"}`, and that check precedes the body check even on the correct route.
- A missing, duplicated, conflicting, malformed, or otherwise illegal `Content-Length` declaration returns HTTP 400 with `{"error":"invalid_request"}` **before the body is read and before authentication**; a declared length over 1,048,576 raw UTF-8 bytes returns HTTP 413 with `{"error":"payload_too_large"}` before the body is read, before JSON parsing, and before any state is touched. A body whose declared length is within the limit but is missing, malformed, duplicated-field, structurally illegal, or carries an unknown field returns HTTP 400 with `{"error":"invalid_request"}` (the same parser and shapes as the preflight).
- A missing, duplicated, malformed, or non-matching `Authorization` header is HTTP 401 with `{"error":"unauthorized"}` and the `WWW-Authenticate: Bearer` challenge; in scope-policy mode an authenticated token lacking the `read` or `admin` scope is HTTP 403 with `{"error":"forbidden"}` and **no** challenge (the diagnosis is read-only, so a `read`-only token suffices). An unauthorized request never reads the body. `GET /health` stays anonymous.
- An internal failure is HTTP 500 with `{"error":"internal_error"}`; the diagnosis produces no business state change on any path.

A successful HTTP 200 response is a compact UTF-8 JSON object terminated by a single newline, with an explicit `Content-Length` and exactly five fields in this order:

```json
{"status":"diagnosed","execution":{"position":0,"expectedCheckpoint":4,"expectedReceipts":"<64 lowercase hex chars>","cursor":4,"bindingMatches":true},"links":[{"action":"resend","ackId":"ack-2","location":{"start":1,"end":2},"target":{"start":1,"end":2},"boundary":{"start":1,"end":2},"evidence":{"location":{"start":1,"end":2},"target":{"start":1,"end":2}},"status":"current"}],"summary":{"suggestions":1,"current":1,"superseded":0,"unavailable":0,"executed":1,"anomalies":0},"conclusion":"current"}
```

- `status` is the fixed string `"diagnosed"`.
- `execution`: `null` when no execution is committed for `(peerId, ackId)`. Otherwise it carries exactly five fields: `position` (the execution's 0-based position in the complete creation-order history — the order bindings first committed and ride in the data file's `repairExecutions` section), `expectedCheckpoint` and `expectedReceipts` (the anchor the committed execution locked), `cursor` (the checkpoint cursor it restored), and `bindingMatches` (whether the committed binding exactly equals the presented `expectedCheckpoint`, `expectedReceipts`, and ordered `suggestions`).
- `links`: one link per diagnosed suggestion, in suggestion order. Before the execution commits these are the request's own suggestions; once committed they are the committed binding's suggestions (so the evidence is about the binding that actually executed). Each link carries exactly `action`, `ackId`, `location`, `target`, `boundary`, `evidence`, and `status`:
  - `action`, `ackId`, `location`, and `target` reproduce the suggestion in order.
  - Each suggestion is aligned with the peer's current advice by its `(action, ackId, ordinal)` slot — the same fixed action order (resend, deduplicate, correct_identity, correct_cursor) and within-slot stable ordering the preflight and apply use; suggestions for the same action and receipt consume that slot's current entries from ordinal 0.
  - `status` is `"current"` when the slot still exists with the same log location and identity (一致), `"superseded"` when the slot still exists for the same action/receipt at the same ordinal but its log location or identity has moved (定位变化), and `"unavailable"` when the action/receipt group no longer holds that ordinal (同组缺失). A link is `current` precisely while the corresponding suggestion is still executable.
  - For `current` and `superseded`, `boundary` is the current slot's target boundary (an interval `{"start","end"}` or a position `{"position":P}`) and `evidence` carries the current slot's `location` and `target` and, for an identity correction, its current `expected` and `observed` identities. For `unavailable`, both `boundary` and `evidence` are `null`.
- `summary`: exactly six integer counts — `suggestions` (the diagnosed suggestion count), `current`, `superseded`, and `unavailable` (the three per-link status totals, summing to `suggestions`), `executed` (the committed result count — equal to `suggestions` once the execution is committed and `0` before), and `anomalies` (the total number of integrity anomalies over the **complete** repair-execution history, covering duplicate bindings, out-of-order actions, boundary violations, out-of-bounds cursors, and structurally damaged records — the same five classes the execution audit's `verification` reports).
- `conclusion`:
  - Before commit: `"ready"` when every link is `current` (the batch would stage cleanly now), otherwise `"unexecuted"`.
  - After commit: `"broken"` when the history integrity scan implicates this execution in any of the five anomaly classes; otherwise `"current"` when every link is `current`, and `"superseded"` when any link is `superseded` or `unavailable`. A `bindingMatches: false` lock or a different execution's anomaly does not by itself change an intact execution's conclusion.

The links, the anchor and binding comparison, the history integrity scan, the summary, and the conclusion are all computed from one snapshot under the same commit lock used by local writes, sync imports, repairs, checkpoint commits, and acknowledgement commits, so a concurrent commit is observed only as the whole old or the whole new snapshot — never a mix. Repeated diagnoses against unchanged state return the identical response. With `--data-file` the execution bindings are rebuilt identically during recovery (an old file without the `repairExecutions` section diagnoses every execution as uncommitted), so the position, anchor, cursor, binding match, links, counts, and conclusion are identical before and after a restart. The diagnosis is strictly read-only: it changes neither the repair advice, the preflight, repair executions, receipts, checkpoints, candidates, the accepted log, nor the data file, and creates no temporary file; the existing repair advice, preflight, execution, receipt, checkpoint, sync, transaction, write, and persistence behaviors are unchanged.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
