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
import os
import sys
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from queryargus.models.history import HistoricalContext
    from queryargus.storage import ReportStore
    from queryargus.storage.postgres import ReportSummary

from queryargus import __version__
from queryargus.agent.loop import ArgusAgent
from queryargus.azure_service import check_auth, list_cosmos_accounts
from queryargus.llm.client import LLMClient
from queryargus.models.config import (
    PROFILE_BALANCED,
    PROFILE_FAST,
    PROFILE_THOROUGH,
    ArgusConfig,
    EvaluatorConfig,
)
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


def _build_judge_llm_client(model: str) -> LLMClient:
    """Build the judge LLM. Same provider as the agent in v1; different model."""
    from queryargus.llm.gemini import GeminiClient  # noqa: PLC0415

    return GeminiClient(model=model)


_PROFILES: dict[str, EvaluatorConfig] = {
    "fast": PROFILE_FAST,
    "balanced": PROFILE_BALANCED,
    "thorough": PROFILE_THOROUGH,
}


def _resolve_evaluator_config(
    profile: str | None,
    *,
    action: str | None,
    finding: str | None,
    run: str | None,
) -> EvaluatorConfig:
    """Start from a profile (or defaults) and apply per-gate overrides."""
    base = _PROFILES.get(profile, EvaluatorConfig()) if profile else EvaluatorConfig()
    overrides: dict[str, Any] = {}
    if action is not None:
        overrides["action_evaluator"] = action
    if finding is not None:
        overrides["finding_evaluator"] = finding
    if run is not None:
        overrides["run_evaluator"] = run
    return base.model_copy(update=overrides) if overrides else base


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
    model: str = typer.Option("gemini-2.5-flash", "--model", help="Gemini model name."),
    eval_profile: str | None = typer.Option(
        None,
        "--eval-profile",
        help="Evaluation profile: fast | balanced | thorough.",
    ),
    action_evaluator: str | None = typer.Option(
        None, "--action-evaluator",
        help="Override action gate: none | rules.",
    ),
    finding_evaluator: str | None = typer.Option(
        None, "--finding-evaluator",
        help="Override finding gate: none | rules | self | composite.",
    ),
    run_evaluator: str | None = typer.Option(
        None, "--run-evaluator",
        help="Override run gate: none | rules | self | judge | composite.",
    ),
    judge_model: str = typer.Option(
        "gemini-2.5-pro", "--judge-model",
        help="Model used by judge run-evaluator (only consulted when run gate is 'judge' or 'composite').",
    ),
    postgres_url: str | None = typer.Option(
        None, "--postgres-url",
        envvar="POSTGRES_URL",
        help="Persist the report and surface diff against previous run for the same collection.",
    ),
    allow_read_write: bool = typer.Option(False, "--allow-read-write"),
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full ReAct audit loop and emit an AuditReport."""
    _setup_logging(verbose)

    eval_config = _resolve_evaluator_config(
        eval_profile,
        action=action_evaluator,
        finding=finding_evaluator,
        run=run_evaluator,
    )
    config = ArgusConfig(
        sample_size=sample_size,
        max_iterations=max_iterations,
        llm_model=model,
        evaluation=eval_config,
        postgres_url=postgres_url,
    )

    persist_url = _validate_postgres_url(postgres_url, required=False)

    connection = _connect(account=account, database=database, allow_read_write=allow_read_write)
    llm = _build_llm_client(config)
    judge_llm: LLMClient | None = None
    if eval_config.run_evaluator in ("judge", "composite"):
        judge_llm = _build_judge_llm_client(judge_model)

    agent = ArgusAgent.from_config(
        config=config,
        llm=llm,
        agent_model_name=config.llm_model,
        judge_llm=judge_llm,
        judge_model_name=judge_model,
    )

    history = _load_history(persist_url, collection=collection, database=database) if persist_url else None
    report = agent.run(connection=connection, collection=collection, history=history)

    if persist_url:
        report = _persist_and_diff(report, persist_url)

    _emit_report(report, output)


@app.command()
def reports(
    collection: str | None = typer.Option(None, "--collection", "-c"),
    database: str | None = typer.Option(None, "--database", "-d"),
    limit: int = typer.Option(10, "--limit", "-l", min=1),
    postgres_url: str | None = typer.Option(None, "--postgres-url", envvar="POSTGRES_URL"),
    output: str = typer.Option("text", "--output", "-o"),
) -> None:
    """List persisted audit reports."""
    from queryargus.storage import ReportStore  # noqa: PLC0415

    url = _validate_postgres_url(postgres_url, required=True)
    assert url is not None  # validator exits if None
    store = ReportStore(url)
    summaries = _safe_list_reports(store, collection=collection, database=database, limit=limit)
    if output == "json":
        typer.echo(json.dumps([
            {
                "id": str(s.id), "collection": s.collection, "database": s.database,
                "cosmos_account": s.cosmos_account, "run_at": s.run_at.isoformat(),
                "findings_count": s.findings_count,
                "overall_quality_score": s.overall_quality_score,
                "run_eval_verdict": s.run_eval_verdict,
                "total_input_tokens": s.total_input_tokens,
                "total_output_tokens": s.total_output_tokens,
            }
            for s in summaries
        ], indent=2))
        return
    if not summaries:
        typer.echo("No reports found.")
        return
    typer.echo(f"{'run_at':<26} {'collection':<24} {'database':<16} {'findings':>8} {'tokens':>10}  verdict")
    for s in summaries:
        total_tokens = s.total_input_tokens + s.total_output_tokens
        typer.echo(
            f"{s.run_at.isoformat():<26} {s.collection[:24]:<24} {s.database[:16]:<16} "
            f"{s.findings_count:>8} {total_tokens:>10}  {s.run_eval_verdict or '-'}"
        )


@app.command()
def diff(
    collection: str = typer.Option(..., "--collection", "-c"),
    database: str = typer.Option(..., "--database", "-d"),
    postgres_url: str | None = typer.Option(None, "--postgres-url", envvar="POSTGRES_URL"),
    output: str = typer.Option("text", "--output", "-o"),
) -> None:
    """Diff the most recent two runs of (collection, database)."""
    from queryargus.storage import ReportStore  # noqa: PLC0415

    url = _validate_postgres_url(postgres_url, required=True)
    assert url is not None
    store = ReportStore(url)
    summaries = _safe_list_reports(store, collection=collection, database=database, limit=2)
    if len(summaries) < 2:
        typer.secho(
            f"Need at least 2 runs to diff; found {len(summaries)} for "
            f"{collection!r} / {database!r}.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    current = store.get(summaries[0].id)
    previous = store.get(summaries[1].id)
    if current is None or previous is None:
        typer.secho("Failed to load reports.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    diffed = current.diff_against(previous)
    _emit_diff(diffed, previous, output)


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
    if report.total_input_tokens or report.total_output_tokens:
        total = report.total_input_tokens + report.total_output_tokens
        typer.echo(
            f"tokens: input={report.total_input_tokens} output={report.total_output_tokens} total={total}"
        )
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

    if report.previous_run_id is not None:
        typer.echo(
            f"\nDIFF vs previous run {report.previous_run_id}:  "
            f"new={len(report.new_findings)}  "
            f"resolved={len(report.resolved_findings)}  "
            f"regressed_fields={len(report.regressed_fields)}"
        )
        if report.new_findings:
            typer.echo("  NEW:")
            for f in report.new_findings:
                typer.echo(
                    f"    + [{f.severity.value}] {f.field} / {f.category} "
                    f"(pct={f.affected_pct:.3f})"
                )
        if report.resolved_findings:
            typer.echo("  RESOLVED:")
            for f in report.resolved_findings:
                typer.echo(f"    - [{f.severity.value}] {f.field} / {f.category}")
        if report.regressed_fields:
            typer.echo("  REGRESSED FIELDS: " + ", ".join(report.regressed_fields))


def _safe_list_reports(
    store: ReportStore,
    *,
    collection: str | None,
    database: str | None,
    limit: int,
) -> list[ReportSummary]:
    """Wrap ``store.list_reports`` so a connection failure becomes a friendly CLI error."""
    import psycopg2  # noqa: PLC0415

    try:
        return store.list_reports(collection=collection, database=database, limit=limit)
    except psycopg2.OperationalError as exc:
        typer.secho(
            f"Postgres connection failed: {exc}".strip(),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc


def _validate_postgres_url(value: str | None, *, required: bool) -> str | None:
    """Normalise the --postgres-url flag.

    Empty strings (e.g. from ``--postgres-url "$POSTGRES_URL"`` when the env var
    is unset) are treated the same as ``None``. If ``required`` and the URL is
    missing, exit with a clear message instead of falling through to psycopg2's
    Unix-socket default.
    """
    url = (value or "").strip() or None
    if required and url is None:
        typer.secho(
            "--postgres-url is required (or set POSTGRES_URL). "
            "Pass an explicit DSN like postgresql://user:pass@host:5432/db.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return url


def _build_cache_session(*, json_export: str | None) -> Any:
    """Build a cache-lens session around a fresh raw genai.Client.

    cachelens is an optional dependency (the ``cache`` extra) — fail with a
    friendly install hint rather than a traceback.
    """
    try:
        from cache_lens import CacheLens  # noqa: PLC0415
    except ImportError as exc:
        typer.secho(
            "--cache-report requires cachelens: pip install cachelens "
            "(or pip install 'queryargus[cache]').",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        typer.secho(
            "GEMINI_API_KEY is not set; cannot build GeminiClient.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Local import — google-genai is heavy; defer until a real run needs it.
    from google import genai  # noqa: PLC0415

    return CacheLens(genai.Client(api_key=api_key), json_export=json_export)


def _load_history(
    postgres_url: str,
    *,
    collection: str,
    database: str,
    limit: int = 5,
) -> HistoricalContext | None:
    """Load cross-run memory for the agent. Returns None on connection failure."""
    import psycopg2  # noqa: PLC0415

    from queryargus.storage import ReportStore  # noqa: PLC0415

    store = ReportStore(postgres_url)
    try:
        store.init_schema()  # idempotent — covers the first-run-ever case
        return store.load_history(collection=collection, database=database, limit=limit)
    except psycopg2.OperationalError as exc:
        typer.secho(
            f"Could not load historical context (continuing without): {exc}".strip(),
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None


def _persist_and_diff(report: AuditReport, postgres_url: str) -> AuditReport:
    """Persist ``report``; surface diff fields populated from the previous run if any."""
    from queryargus.storage import ReportStore  # noqa: PLC0415

    store = ReportStore(postgres_url)
    store.init_schema()
    previous = store.get_previous(
        collection=report.collection,
        database=report.database,
        before=report.run_at,
    )
    if previous is not None:
        report = report.diff_against(previous)
    store.save(report)
    return report


def _emit_diff(current: AuditReport, previous: AuditReport, output: str) -> None:
    if output == "json":
        typer.echo(current.model_dump_json(indent=2))
        return
    if output != "text":
        typer.secho(f"unknown --output value: {output!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(
        f"diff: {current.collection} / {current.database}  "
        f"current={current.id} (run_at={current.run_at.isoformat()})  "
        f"previous={previous.id} (run_at={previous.run_at.isoformat()})"
    )
    typer.echo(
        f"new={len(current.new_findings)}  "
        f"resolved={len(current.resolved_findings)}  "
        f"regressed_fields={len(current.regressed_fields)}"
    )
    if current.new_findings:
        typer.echo("\nNEW:")
        for f in current.new_findings:
            typer.echo(f"  + [{f.severity.value}] {f.field} / {f.category} ({f.affected_pct:.3f})")
    if current.resolved_findings:
        typer.echo("\nRESOLVED:")
        for f in current.resolved_findings:
            typer.echo(f"  - [{f.severity.value}] {f.field} / {f.category}")
    if current.regressed_fields:
        typer.echo("\nREGRESSED FIELDS: " + ", ".join(current.regressed_fields))


def _severity_rank(s: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(s, 0)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _main() -> Any:  # pragma: no cover — entrypoint
    return app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
