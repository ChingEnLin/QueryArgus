from __future__ import annotations

import pytest
from pydantic import ValidationError

from queryargus.models.report import AuditReport, ModelCost, RunCost


def _bare_report() -> AuditReport:
    return AuditReport(
        collection="c",
        database="d",
        cosmos_account="a",
        duration_seconds=0.0,
        documents_sampled=0,
        collection_size=0,
    )


def test_model_cost_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        ModelCost(model="x", input_tokens=-1, output_tokens=0, usd=0.0)
    with pytest.raises(ValidationError):
        ModelCost(model="x", input_tokens=0, output_tokens=0, usd=-0.01)


def test_run_cost_defaults_and_pricing_version() -> None:
    rc = RunCost()
    assert rc.usd_total == 0.0
    assert rc.by_model == []
    assert rc.pricing_version == "2026-05"


def test_audit_report_cost_optional_and_roundtrips() -> None:
    report = _bare_report()
    assert report.cost is None
    report.cost = RunCost(
        usd_total=0.0123,
        by_model=[
            ModelCost(
                model="gemini-2.5-flash",
                input_tokens=1000,
                output_tokens=500,
                usd=0.0123,
            )
        ],
    )
    payload = report.model_dump(mode="json")
    restored = AuditReport.model_validate(payload)
    assert restored.cost is not None
    assert restored.cost.usd_total == pytest.approx(0.0123)
    assert restored.cost.by_model[0].model == "gemini-2.5-flash"
