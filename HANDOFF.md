# HANDOFF.md — QueryArgus integration into QueryPal

**Date:** 2026-05-11  
**Status:** QueryArgus is feature-complete through Weekend 3. Ready to wire into QueryPal.

---

## 1. State of QueryArgus

The full src tree is built and committed on branch `dev`. All three weekend deliverables from `PLAN.md §4` are done:

| Module | Path | What it does |
|--------|------|--------------|
| Models | `src/queryargus/models/` | Pydantic v2: `ArgusConfig`, `AuditReport`, `Finding`, `AgentAction`, `EvaluationResult`, `HistoricalContext`, etc. |
| Tools | `src/queryargus/tools/` | `schema_sample`, `run_query`, `get_stats`, `write_finding` |
| Agent loop | `src/queryargus/agent/loop.py` | `ArgusAgent` — ReAct loop with action/finding/run evaluation gates |
| Evaluation layer | `src/queryargus/agent/evaluation/` | Rules, self-eval, judge, composite strategies; factory + base protocols |
| LLM client | `src/queryargus/llm/` | `LLMClient` Protocol; `GeminiClient` implementation (reads `GEMINI_API_KEY` from env) |
| Azure / Cosmos | `src/queryargus/azure_service.py`, `src/queryargus/models/connection.py` | `CosmosConnection` with three factories: `from_default_credential`, `from_existing_client`, `from_connection_string` |
| Storage | `src/queryargus/storage/postgres.py` | `ReportStore` — raw psycopg2, JSONB `argus_reports`, mirrored relational tables for findings/evals |
| CLI | `src/queryargus/cli/main.py` | `queryargus run --collection X` entry point |

**What is NOT done:**
- Postgres storage is not wired into the QueryPal endpoint (optional for the first integration pass; the agent runs and returns `AuditReport` in-memory just fine)
- Frontend page (`frontend/src/pages/Argus/`) — out of scope for this handoff
- `storage/schema.sql` should be reviewed to confirm it was committed (check `src/queryargus/storage/`)

---

## 2. Key facts for the integration engineer

### Entry point

```python
from queryargus.agent.loop import ArgusAgent
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.llm.gemini import GeminiClient

config = ArgusConfig(max_iterations=20)              # all other fields have safe defaults
llm    = GeminiClient()                              # reads GEMINI_API_KEY from env
agent  = ArgusAgent.with_defaults(config=config, llm=llm)

connection = CosmosConnection.from_existing_client(
    client=existing_mongo_client,                    # QueryPal's MongoClient
    cosmos_account="my-cosmos-account",
    database="my-db",
)

report = agent.run(connection, "my-collection")      # synchronous, blocking
# report is AuditReport — call .model_dump(mode="json") to serialise
```

### `ArgusAgent.run` is synchronous

Both siblings (QueryPal, QueryMCPal) are sync throughout. The agent loop, all tools, and the LLM client are blocking. In a FastAPI handler, wrap with `fastapi.concurrency.run_in_threadpool`.

### `CosmosConnection.from_existing_client`

This is the integration seam. QueryPal already authenticates and holds a `MongoClient` for Cosmos DB. Pass it directly — do **not** create a second client or touch connection strings.

### No new dependencies needed

QueryArgus's `pyproject.toml` declares all its own dependencies. After `pip install -e ./queryargus` in QueryPal's backend venv, nothing else needs adding.

### Evaluation profiles (three presets)

| Profile | Import | Cost | Use when |
|---------|--------|------|----------|
| `PROFILE_FAST` | `from queryargus.models.config import PROFILE_FAST` | free | default for QueryPal endpoint |
| `PROFILE_BALANCED` | same module | ~1× extra LLM call | when quality matters more than speed |
| `PROFILE_THOROUGH` | same module | ~2–3× + OpenAI judge | for scheduled / offline audits |

To use a profile: `ArgusConfig(evaluation=PROFILE_FAST, max_iterations=20)`.

### Observability (optional)

Attach observers to surface per-run cost and structured logs:

```python
from queryargus.observability.cost import CostTracker
from queryargus.observability.logging_observer import StructuredLogObserver

agent = ArgusAgent.with_defaults(
    config=config,
    llm=llm,
    observers=[StructuredLogObserver(), CostTracker()],
)
report = agent.run(connection, "my-collection")
if report.cost:
    print(report.cost.usd_total, [m.model_dump() for m in report.cost.by_model])
```

`StructuredLogObserver` emits one JSON-ready log record per event on the `queryargus.run` logger (configure your own handler/formatter, or attach the shipped `JsonFormatter`). `CostTracker` reads each LLM call's reported `model` + token usage against `DEFAULT_PRICING` and attaches a `RunCost` to the returned report. Observers are optional and have zero impact when omitted.

---

## 3. QueryPal integration steps

### Step A — Add submodule

In the QueryPal repo root:

```bash
git submodule add https://github.com/ChingEnLin/QueryArgus backend/queryargus
git submodule update --init --recursive
pip install -e backend/queryargus          # in QueryPal's active venv
```

Add to `backend/requirements.txt` (or equivalent):
```
# QueryArgus — installed as editable submodule: pip install -e ./queryargus
```

### Step B — Create `backend/routers/argus.py`

Mirror the style of existing QueryPal routers exactly (import ordering, logger pattern, router file shape). Skeleton:

```python
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from queryargus.agent.loop import ArgusAgent
from queryargus.llm.gemini import GeminiClient
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection

logger = logging.getLogger(__name__)
router = APIRouter()


class AuditRequest(BaseModel):
    collection: str
    database: str
    cosmos_account: str
    max_iterations: int = 20


@router.post("/run")
async def run_audit(req: AuditRequest):
    logger.info("argus run start collection=%s database=%s", req.collection, req.database)
    # --- wire to QueryPal's existing MongoClient here ---
    # from <querypal_mongo_service> import get_mongo_client
    # existing_client = get_mongo_client()
    connection = CosmosConnection.from_existing_client(
        client=existing_client,           # <-- fill in
        cosmos_account=req.cosmos_account,
        database=req.database,
    )
    config = ArgusConfig(max_iterations=req.max_iterations)
    llm = GeminiClient()
    agent = ArgusAgent.with_defaults(config=config, llm=llm)

    try:
        report = await run_in_threadpool(agent.run, connection, req.collection)
    except Exception as exc:
        logger.exception("argus run failed collection=%s", req.collection)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    logger.info(
        "argus run done collection=%s findings=%d duration=%.2fs",
        req.collection, len(report.findings), report.duration_seconds,
    )
    return JSONResponse(content=report.model_dump(mode="json"))
```

**The one unknown:** how QueryPal exposes its `MongoClient`. Read `backend/services/mongo_service.py` to find the client accessor before filling in the `existing_client` line. If the client is created per-request (vs. a shared singleton), use whatever pattern the existing routes use.

### Step C — Register the router in `backend/main.py`

```python
from routers.argus import router as argus_router
app.include_router(argus_router, prefix="/argus", tags=["argus"])
```

Adjust the import path to match how other routers are registered.

### Step D — Verify linting/types

Run whatever lint/mypy command QueryPal uses. Fix before committing. No `type: ignore` without a comment.

---

## 4. Caveats and watch-outs

| # | Issue | What to do |
|---|-------|-----------|
| C1 | `AgentAction.action` is a `Literal` — Cosmos Emulator or mongomock will work for smoke-tests but `$sample` behaves differently on Cosmos. Run at least one integration test against a real Cosmos account before declaring done. | See `PLAN.md §5 R4`. |
| C2 | `GeminiClient` reads `GEMINI_API_KEY` from env at construction time. QueryPal's `.env` (or Azure Key Vault secrets) must include this key. | Add to QueryPal's env config. |
| C3 | `ArgusAgent.run` will iterate up to `max_iterations` times, each making at least one LLM call. On a large or complex collection this can take 30–90 seconds. The FastAPI endpoint must not have a short request timeout. | Consider streaming progress or an async task queue for production; for the first pass, a generous timeout is fine. |
| C4 | Storage (`ReportStore`) is not wired in this first pass. Reports are returned to the caller only; they are not persisted. To enable persistence, pass a `psycopg2` connection to `ReportStore` and call `store.save(report)` after `agent.run`. Schema is in `src/queryargus/storage/schema.sql` — apply once with `ReportStore.init_schema()`. | Weekend 3 item, deferrable. |
| C5 | The `GEMINI_API_KEY` env var is already used by QueryPal's `gemini_service.py` (same key name). No duplication needed — QueryArgus reads the same env var. | Confirm the key name matches before deploying. |

---

## 5. What was intentionally deferred

- **Frontend page** (`frontend/src/pages/Argus/`) — out of scope. Add after the API endpoint is stable.
- **Postgres persistence** — `ReportStore` exists and is tested; not wired into the QueryPal endpoint yet.
- **Historical context** (`HistoricalContext`) — `agent.run()` accepts an optional `history` argument. Not plumbed in the first pass; add once persistence is live.
- **`PROFILE_THOROUGH` / judge LLM** — requires an OpenAI key. Not needed for the first integration.

---

## 6. Files to look at in QueryPal before touching anything

1. `backend/services/mongo_service.py` — MongoClient accessor
2. `backend/services/azure_auth.py` — OBO token exchange (FYI; not directly needed for the Argus endpoint)
3. `backend/services/gemini_service.py` — confirm `GEMINI_API_KEY` env var name
4. `backend/main.py` — router registration pattern
5. Any existing router file (e.g. `backend/routers/query.py`) — style reference
