# Semantic State Engine

Semantic State Engine is a Python backend for building a distributed state system that can detect, explain, and repair semantic conflicts between independently updated replicas.

The repository intentionally begins with a small, runnable service boundary. New capabilities must preserve deterministic behavior, explicit public contracts, auditable state transitions, and safe behavior under retries, reordering, duplication, partial failure, and concurrent updates.

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

## Engineering direction

The long-term product goal is a semantic-conflict-repair distributed state engine. Work should evolve the backend through independently testable features such as durable state models, causality, replication protocols, semantic conflict classification, policy-driven repair, idempotency, audit history, consistency checks, operational APIs, persistence, fault recovery, and observability. Each change must be derived from the repository's current behavior rather than from a pre-generated task list.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
