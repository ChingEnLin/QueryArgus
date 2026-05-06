"""QueryArgus CLI.

Auth model: ``DefaultAzureCredential`` — i.e. the user runs ``az login`` once
and QueryArgus rides their token cache to fetch Cosmos connection strings via
ARM. No connection strings in env vars, no flags exposing credentials.

Weekend 1.5 surface:

- ``queryargus auth-status`` — verify the credential before doing anything.
- ``queryargus accounts`` — list Cosmos accounts visible to the credential.
- ``queryargus run --account <name> --database <db> --collection <coll>`` —
  invoke the schema sampler against that collection.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import typer

from queryargus import __version__
from queryargus.azure_service import check_auth, list_cosmos_accounts
from queryargus.models.connection import CosmosConnection
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
    # Quiet noisy SDK loggers unless --verbose is set explicitly.
    if not verbose:
        for noisy in ("azure", "azure.identity", "azure.core", "urllib3", "msal"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


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
    typer.secho(f"not authenticated: {status.get('reason', 'unknown')}", fg=typer.colors.RED, err=True)
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
def run(
    account: str = typer.Option(..., "--account", "-a", help="Cosmos account name or full ARM resource ID."),
    database: str = typer.Option(..., "--database", "-d", help="Database name."),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name to audit."),
    sample_size: int = typer.Option(200, "--sample-size", "-n", min=1),
    allow_read_write: bool = typer.Option(
        False,
        "--allow-read-write",
        help="Use the read-write Cosmos key instead of read-only. "
        "Default is read-only — QueryArgus only ever reads, but the credential is read-only too.",
    ),
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a schema sample against a Cosmos DB collection.

    Authenticates via ``DefaultAzureCredential`` (i.e. ``az login`` token cache).
    """
    _setup_logging(verbose)

    try:
        connection = CosmosConnection.from_default_credential(
            account=account,
            database=database,
            prefer_read_only=not allow_read_write,
        )
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    result = schema_sample(connection, collection, sample_size=sample_size)
    _emit(result, output)


def _emit(result: SchemaSampleResult, output: str) -> None:
    if output == "json":
        typer.echo(result.model_dump_json(indent=2))
        return
    if output == "text":
        _print_text(result)
        return
    typer.secho(
        f"unknown --output value: {output!r} (expected 'text' or 'json')",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=2)


def _print_text(result: SchemaSampleResult) -> None:
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


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _main() -> Any:  # pragma: no cover — entrypoint
    return app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
