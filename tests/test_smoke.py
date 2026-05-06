"""Smoke tests: package imports, CLI is wired, version is set."""

from __future__ import annotations

from typer.testing import CliRunner

import queryargus
from queryargus.cli.main import app


def test_package_version_is_set() -> None:
    assert isinstance(queryargus.__version__, str)
    assert queryargus.__version__


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert queryargus.__version__ in result.stdout


def test_cli_run_requires_account_database_collection() -> None:
    """`run` is non-interactive and exits non-zero when required flags are missing."""
    runner = CliRunner()
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
