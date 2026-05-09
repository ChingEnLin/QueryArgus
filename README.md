# QueryArgus

> An autonomous data-quality agent for schemaless document databases.
> Investigates. Hypothesises. Reports. No rules required.

QueryArgus connects to a Cosmos DB (MongoDB API) collection, samples documents, infers the implicit schema, and runs a **ReAct planning loop** to discover data-quality issues — null floods, type mismatches, outlier values, inconsistent enumerations — emitting a structured `AuditReport` without you writing a single rule upfront.

It is a **library and CLI first**, with an optional thin FastAPI wrapper. It runs standalone (`queryargus run` in CI or a cron job) or embeds as a Git submodule in [QueryPal](https://github.com/ChingEnLin/QueryPal).

---

## How it works

```
                        ┌─────────────────────────────────┐
                        │          ArgusAgent.run()       │
                        │                                 │
                        │  ┌───────────────────────────┐  │
                        │  │   ReAct Loop (≤20 iters)  │  │
                        │  │                           │  │
  Cosmos DB ──────────► │  │  Planner → AgentAction    │  │
  (MongoDB API)         │  │  Action Gate (evaluate)   │  │
                        │  │  Tool dispatch            │  │
                        │  │  Finding Gate (evaluate)  │  │
                        │  │  Run Gate (evaluate)      │  │
                        │  └───────────────────────────┘  │
                        │                                 │
  Postgres (optional) ◄─│  AuditReport                    │
  HistoricalContext ───►│  (findings, trace, eval audit)  │
                        └─────────────────────────────────┘
```

At each iteration: the LLM **reasons** about the current schema and findings, **picks a tool** (sample schema, run a query, get aggregation stats, or write a finding), the **evaluation layer** gates the action before it executes, and the result feeds back into the next iteration.

---

## Why

MongoDB and Cosmos DB are schemaless by contract — but the applications that read and write to them are not. Every service carries an implicit schema: expected field names, types, value ranges, structural conventions. The database does not enforce any of it.

This creates a class of failure that is silent and cumulative — manual edits that skip required fields, half-run migrations leaving collections in mixed states, bugs that quietly write `null` to required fields for hours.

Existing tools either infer schemas without telling you what's wrong ([Variety](https://github.com/variety/variety)), run assertions you already wrote ([Great Expectations](https://greatexpectations.io/)), or watch infrastructure rather than data semantics (Datadog, Azure Monitor). The gap QueryArgus fills: **autonomously investigate a schemaless collection and produce a structured, evidence-backed audit report.**

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export GEMINI_API_KEY=...
export COSMOS_CONNECTION_STRING=...

# One-off audit
queryargus run --collection users

# With Postgres for cross-run memory and diff
queryargus run \
  --collection users \
  --postgres-url postgresql://user:pass@localhost/argus \
  --eval-profile balanced \
  --output json > report.json

# Inspect schema only (no LLM, no loop)
queryargus inspect --collection users
```

---

## CLI reference

```
queryargus run        Run a full autonomous audit
queryargus inspect    Schema sample only — no LLM
queryargus accounts   List Cosmos DB accounts visible to current credential
queryargus auth-status  Check DefaultAzureCredential readiness
queryargus version    Print installed version
```

**`run` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--collection` | required | Collection to audit |
| `--database` | from connection string | Cosmos DB database name |
| `--cosmos-account` | from connection string | Cosmos DB account name |
| `--eval-profile` | `fast` | `fast` / `balanced` / `thorough` |
| `--eval-action` | — | Override action gate strategy |
| `--eval-finding` | — | Override finding gate strategy |
| `--eval-run` | — | Override run gate strategy |
| `--postgres-url` | — | Enable persistence + cross-run memory |
| `--output` | `text` | `text` / `json` / `silent` |
| `--verbose` | off | Log each iteration to stderr |

---

## Python API

```python
from queryargus.agent.loop import ArgusAgent
from queryargus.models.config import ArgusConfig, PROFILE_BALANCED
from queryargus.models.connection import CosmosConnection
from queryargus.llm.gemini import GeminiClient

config = ArgusConfig(
    sample_size=200,
    max_iterations=20,
    llm_model="gemini-2.5-flash",
    evaluation=PROFILE_BALANCED,
    postgres_url="postgresql://user:pass@localhost/argus",
)

connection = CosmosConnection.from_default_credential(
    account="mycosmosaccount",
    database="prod",
)

agent = ArgusAgent.from_config(config, GeminiClient(model=config.llm_model))
report = agent.run(connection, "users")

print(f"{len(report.findings)} findings, score {report.overall_quality_score:.2f}")
print(report.model_dump_json(indent=2))
```

---

## Development

```bash
pytest                  # run tests
ruff check .            # lint
mypy src                # type check
```

The test suite uses `ScriptedLLMClient` (a deterministic test double) and `mongomock` for isolated unit tests that never touch a real database or LLM.

---

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Component map, data flow, design decisions |
| [Agent Loop](docs/agent-loop.md) | ReAct loop internals, planner, state summarisation, critique injection |
| [Evaluation Layer](docs/evaluation.md) | Three gates, rule lists, self-eval, judge, composite chains |
| [Memory & Persistence](docs/memory.md) | Cross-run memory, HistoricalContext, ReportStore schema, AuditReport diff |
| [Tools Reference](docs/tools.md) | schema_sample, run_query, get_stats, write_finding |

---

## Sibling projects

- [QueryPal](https://github.com/ChingEnLin/QueryPal) — host application; QueryArgus mounts here as a Git submodule.
- [QueryMCPal](https://github.com/ChingEnLin/QueryMCPal) — sibling MCP server for read-only Cosmos DB access.
