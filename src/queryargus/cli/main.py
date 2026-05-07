"""QueryArgus CLI.

Auth model: ``DefaultAzureCredential`` — i.e. the user runs ``az login`` once
and QueryArgus rides their token cache to fetch Cosmos connection strings via
ARM. No connection strings in env vars, no flags exposing credentials.

Commands:

- ``queryargus auth-status`` — verify the credential before doing anything.
- ``queryargus accounts`` — list Cosmos accounts visible to the credential.
- ``queryargus inspect`` — bare schema sample (no LLM, no agent loop).
- ``queryargus run`` — full ReAct audit. Uses Gemini by default; tests inject
  a scripted LLM via ``_build_llm_client``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import typer

from queryargus import __version__
from queryargus.agent.loop import ArgusAgent
from queryargus.azure_service import check_auth, list_cosmos_accounts
from queryargus.llm.client import LLMClient
from queryargus.models.config import ArgusConfig
from queryargus.models.connection import CosmosConnection
from queryargus.models.report import AuditReport
from queryargus.tools.schema_sample import SchemaSampleResult, schema_sample

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="queryargus",
    help="Autonomous data quality agent for Cosmos DB (MongoDB API).",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    if not verbose:
        for noisy in ("azure", "azure.identity", "azure.core", "urllib3", "msal"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_llm_client(config: ArgusConfig) -> LLMClient:
    """Default LLM factory. Tests monkeypatch this to inject ScriptedLLMClient."""
    # Local import — google-genai is heavy; defer until a real run actually needs it.
    from queryargus.llm.gemini import GeminiClient  # noqa: PLC0415

    return GeminiClient(model=config.llm_model)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the installed QueryArgus version."""
    typer.echo(__version__)


@app.command("auth-status")
def auth_status() -> None:
    """Probe DefaultAzureCredential without side effects."""
    status = check_auth()
    if status.get("ok"):
        mode = status.get("mode", "az-login")
        typer.echo(f"ok  ({mode})  signed in as: {status.get('account', 'authenticated')}")
        raise typer.Exit(code=0)
    typer.secho(
        f"not authenticated: {status.get('reason', 'unknown')}",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def accounts(
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List Cosmos DB accounts visible to the current credential."""
    _setup_logging(verbose)
    try:
        accts = list_cosmos_accounts()
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if output == "json":
        typer.echo(json.dumps(accts, indent=2))
        return
    if not accts:
        typer.echo("No Cosmos DB accounts visible. Try `az login` and re-run.")
        return
    typer.echo(f"{'name':<40} {'subscription':<30} {'location':<20} kind")
    for a in accts:
        typer.echo(
            f"{a['name']:<40} {a.get('subscription', '?'):<30} "
            f"{a.get('location', '?'):<20} {a.get('kind', '?')}"
        )


@app.command()
def inspect(
    account: str = typer.Option(..., "--account", "-a", help="Cosmos account name or full ARM resource ID."),
    database: str = typer.Option(..., "--database", "-d"),
    collection: str = typer.Option(..., "--collection", "-c"),
    sample_size: int = typer.Option(200, "--sample-size", "-n", min=1),
    allow_read_write: bool = typer.Option(False, "--allow-read-write"),
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Bare schema sample. No LLM, no agent loop — useful for ad-hoc surveys."""
    _setup_logging(verbose)
    connection = _connect(account=account, database=database, allow_read_write=allow_read_write)
    result = schema_sample(connection, collection, sample_size=sample_size)
    _emit_schema(result, output)


@app.command()
def run(
    account: str = typer.Option(..., "--account", "-a"),
    database: str = typer.Option(..., "--database", "-d"),
    collection: str = typer.Option(..., "--collection", "-c"),
    sample_size: int = typer.Option(200, "--sample-size", "-n", min=1),
    max_iterations: int = typer.Option(20, "--max-iterations", min=1),
    model: str = typer.Option("gemini-2.0-flash-exp", "--model", help="Gemini model name."),
    allow_read_write: bool = typer.Option(False, "--allow-read-write"),
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full ReAct audit loop and emit an AuditReport."""
    _setup_logging(verbose)

    config = ArgusConfig(
        sample_size=sample_size,
        max_iterations=max_iterations,
        llm_model=model,
    )
    connection = _connect(account=account, database=database, allow_read_write=allow_read_write)
    llm = _build_llm_client(config)
    agent = ArgusAgent.with_defaults(config=config, llm=llm)
    report = agent.run(connection=connection, collection=collection)
    _emit_report(report, output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(*, account: str, database: str, allow_read_write: bool) -> CosmosConnection:
    try:
        return CosmosConnection.from_default_credential(
            account=account,
            database=database,
            prefer_read_only=not allow_read_write,
        )
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _emit_schema(result: SchemaSampleResult, output: str) -> None:
    if output == "json":
        typer.echo(result.model_dump_json(indent=2))
        return
    if output != "text":
        typer.secho(f"unknown --output value: {output!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"collection: {result.collection}")
    typer.echo(f"documents sampled: {result.documents_sampled}")
    typer.echo(f"distinct field paths: {len(result.fields)}")
    if result.truncated_paths:
        typer.echo(f"truncated paths (recursion limit): {', '.join(result.truncated_paths)}")
    typer.echo("")
    typer.echo(f"{'field':<40} {'present':>8} {'missing':>8} {'null_rate':>10} {'card':>6}  types")
    for f in result.fields:
        types_str = ",".join(f"{t}={c}" for t, c in sorted(f.types.items()))
        typer.echo(
            f"{_truncate(f.path, 40):<40} {f.present_count:>8} {f.missing_count:>8} "
            f"{f.null_rate:>10.3f} {f.cardinality:>6}{'+' if f.cardinality_capped else ' '} {types_str}"
        )


def _emit_report(report: AuditReport, output: str) -> None:
    if output == "json":
        typer.echo(report.model_dump_json(indent=2))
        return
    if output != "text":
        typer.secho(f"unknown --output value: {output!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"collection: {report.collection}  database: {report.database}  account: {report.cosmos_account}")
    typer.echo(report.summary)
    if report.run_evaluation:
        ev = report.run_evaluation
        typer.echo(f"run gate: {ev.verdict.value} (score={ev.score:.2f}) — {ev.reason}")
    typer.echo("")
    if not report.findings:
        typer.echo("No findings committed.")
    else:
        typer.echo(f"FINDINGS ({len(report.findings)}):")
        for f in sorted(report.findings, key=lambda x: (-_severity_rank(x.severity.value), x.field)):
            typer.echo(
                f"  [{f.severity.value}] {f.field} / {f.category}\n"
                f"      {f.description}\n"
                f"      affected={f.affected_count} ({f.affected_pct:.3f})  evidence: {f.evidence_query}"
            )
    if report.dismissed_findings:
        typer.echo(f"\nDISMISSED ({len(report.dismissed_findings)}):")
        for f in report.dismissed_findings:
            typer.echo(f"  - {f.field} / {f.category}: {f.description[:120]}")


def _severity_rank(s: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(s, 0)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _main() -> Any:  # pragma: no cover — entrypoint
    return app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
