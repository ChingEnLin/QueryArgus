"""QueryArgus CLI.

Weekend 1 surface: ``run`` invokes ``schema_sample`` against a real (or emulator)
Cosmos DB collection and prints the per-field statistics. The ReAct planning
loop and evaluation gates land in weekends 2–3.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import typer

from queryargus import __version__
from queryargus.models.connection import CosmosConnection, redact_connection_string
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


@app.command()
def version() -> None:
    """Print the installed QueryArgus version."""
    typer.echo(__version__)


@app.command()
def run(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name to audit."),
    database: str = typer.Option(..., "--database", "-d", help="Database name."),
    cosmos_account: str = typer.Option(
        "local",
        "--cosmos-account",
        help="Logical name for the Cosmos account (used for reporting only).",
    ),
    sample_size: int = typer.Option(200, "--sample-size", "-n", min=1),
    output: str = typer.Option("text", "--output", "-o", help="text | json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a schema sample against a Cosmos DB collection.

    Reads ``COSMOS_CONNECTION_STRING`` from the environment. The full ReAct audit
    loop is not yet wired — this command is the weekend-1 ground truth that the
    schema sampler works end-to-end against a real collection.
    """
    _setup_logging(verbose)

    conn_str = os.environ.get("COSMOS_CONNECTION_STRING")
    if not conn_str:
        typer.secho(
            "COSMOS_CONNECTION_STRING is not set. Copy .env.example to .env and fill it in.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    logger.info("connecting to cosmos: %s", redact_connection_string(conn_str))
    connection = CosmosConnection.from_connection_string(
        conn_str,
        cosmos_account=cosmos_account,
        database=database,
    )

    result = schema_sample(connection, collection, sample_size=sample_size)
    _emit(result, output)


def _emit(result: SchemaSampleResult, output: str) -> None:
    if output == "json":
        typer.echo(result.model_dump_json(indent=2))
        return
    if output == "text":
        _print_text(result)
        return
    typer.secho(f"unknown --output value: {output!r} (expected 'text' or 'json')", fg=typer.colors.RED, err=True)
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
