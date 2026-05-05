"""QueryArgus CLI — placeholder during scaffolding (weekend 0)."""

from __future__ import annotations

import logging

import typer

from queryargus import __version__

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="queryargus",
    help="Autonomous data quality agent for Cosmos DB (MongoDB API).",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed QueryArgus version."""
    typer.echo(__version__)


@app.command()
def run(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name to audit."),
    database: str | None = typer.Option(None, "--database", "-d", help="Database name."),
) -> None:
    """Run an audit against a Cosmos DB collection. (Not yet implemented — scaffolding.)"""
    typer.echo(
        f"queryargus run --collection {collection}"
        + (f" --database {database}" if database else "")
        + "  [scaffolding — agent not yet implemented]"
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
