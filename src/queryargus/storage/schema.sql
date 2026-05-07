-- QueryArgus persistent storage schema.
-- Idempotent: safe to apply on every connect via ReportStore.init_schema().
-- See spec §9 for the rationale; small deltas: total_input_tokens /
-- total_output_tokens are added so cost visibility survives across runs.

CREATE TABLE IF NOT EXISTS argus_reports (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection            TEXT NOT NULL,
    database              TEXT NOT NULL,
    cosmos_account        TEXT NOT NULL,
    run_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_secs         DOUBLE PRECISION NOT NULL,
    docs_sampled          INT NOT NULL,
    collection_size       INT NOT NULL,
    summary               TEXT,
    overall_quality_score DOUBLE PRECISION,
    run_eval_verdict      TEXT,
    run_eval_by           TEXT,
    total_input_tokens    INT NOT NULL DEFAULT 0,
    total_output_tokens   INT NOT NULL DEFAULT 0,
    run_trace             JSONB,
    raw_report            JSONB
);

CREATE TABLE IF NOT EXISTS argus_findings (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id          UUID NOT NULL REFERENCES argus_reports(id) ON DELETE CASCADE,
    field              TEXT NOT NULL,
    category           TEXT NOT NULL,
    severity           TEXT NOT NULL,
    description        TEXT NOT NULL,
    hypothesis         TEXT,
    evidence_query     TEXT,
    affected_count     INT NOT NULL,
    affected_pct       DOUBLE PRECISION NOT NULL,
    sample_values      JSONB,
    confirmed          BOOLEAN NOT NULL DEFAULT TRUE,
    evaluation_verdict TEXT,
    evaluation_score   DOUBLE PRECISION,
    evaluated_by       TEXT,
    user_label         TEXT,  -- 'tp' | 'fp' | NULL
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS argus_dismissed_findings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id       UUID NOT NULL REFERENCES argus_reports(id) ON DELETE CASCADE,
    field           TEXT NOT NULL,
    category        TEXT NOT NULL,
    severity        TEXT NOT NULL,
    description     TEXT NOT NULL,
    dismissed_by    TEXT NOT NULL,
    dismiss_verdict TEXT NOT NULL,
    dismiss_reason  TEXT NOT NULL,
    dismiss_score   DOUBLE PRECISION NOT NULL,
    critique        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS argus_evaluation_records (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id    UUID NOT NULL REFERENCES argus_reports(id) ON DELETE CASCADE,
    gate         TEXT NOT NULL,        -- 'action' | 'finding' | 'run'
    evaluated_by TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    score        DOUBLE PRECISION NOT NULL,
    reason       TEXT NOT NULL,
    critique     TEXT,
    target_id    UUID,
    iteration    INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_argus_reports_collection
    ON argus_reports(collection, database, run_at DESC);
CREATE INDEX IF NOT EXISTS idx_argus_findings_report
    ON argus_findings(report_id, severity);
CREATE INDEX IF NOT EXISTS idx_argus_dismissed_report
    ON argus_dismissed_findings(report_id);
CREATE INDEX IF NOT EXISTS idx_argus_eval_records_report
    ON argus_evaluation_records(report_id, gate);
