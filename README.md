# QueryArgus

> An autonomous data quality agent for schemaless document databases.
> Investigates. Hypothesizes. Reports. No queries required.

QueryArgus connects to a Cosmos DB (MongoDB API) collection, samples documents,
infers the implicit schema, hypothesises about data-quality issues, runs
follow-up queries to confirm or disprove them, and emits a structured
`AuditReport` — without you writing a single rule.

It is a library and a CLI first, with an optional thin FastAPI wrapper. It has
no mandatory dependency on any host application: it can run as a standalone
`queryargus run` invocation in a CI pipeline or a cron job, or be embedded in
another Python service.

## Why

MongoDB and Cosmos DB (MongoDB API) are schemaless by contract — but the
applications that read and write to them are not. Every service carries an
implicit schema: expected field names, types, value ranges, structural
conventions. The database does not enforce any of it.

This creates a class of failure that is silent and cumulative — manual edits
that skip required fields, half-run migrations leaving collections in mixed
states, bugs that quietly write `null` to required fields for hours.

Existing tools either infer schemas without telling you what's wrong
(Variety), run assertions you already wrote (Great Expectations), or watch
infrastructure rather than data semantics (Datadog, Azure Monitor). The gap
QueryArgus fills: **autonomously investigate a schemaless collection, form
hypotheses about what is wrong, and produce a structured audit report.**

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Set credentials (copy from .env.example)
export GEMINI_API_KEY=...
export COSMOS_CONNECTION_STRING=...

# Run an audit
queryargus run --collection users
```

## Development

```bash
pytest
ruff check .
mypy src
```

## Sibling projects

- [QueryPal](https://github.com/ChingEnLin/QueryPal) — host application;
  QueryArgus mounts here as a Git submodule.
- [QueryMCPal](https://github.com/ChingEnLin/QueryMCPal) — sibling MCP server
  for read-only Cosmos DB access.
