# Tools Reference

The agent has four tools. The planner chooses one per iteration. Tools are pure functions — they take a `CosmosConnection` and parameters, call pymongo, and return a Pydantic result model. They do not mutate `AgentState` directly; the loop dispatcher updates state after each successful tool call.

---

## Tool Summary

| Tool | Purpose | When used |
|------|---------|-----------|
| `schema_sample` | Sample N documents, infer implicit schema | Always first — required before any other tool |
| `run_query` | Bounded `find()` to confirm a hypothesis | When the planner has a specific filter to test |
| `get_stats` | Aggregation: count, min, max, avg, distinct | When exact statistics are needed without fetching documents |
| `write_finding` | Commit a confirmed finding | After evidence confirms a data-quality issue |

---

## schema_sample

**File:** `tools/schema_sample.py`

Surveys the collection's implicit schema by sampling N documents via `$sample` aggregation.

### Algorithm

```
1. collection.aggregate([{"$sample": {"size": N}}])
2. For each document:
   └── recursive field walk (max depth 20, arrays of objects not descended)
       └── for each field path:
           ├── record type (str, int, float, bool, null, datetime, etc.)
           ├── track present / missing / null counts
           └── collect up to 5 sample values (capped cardinality at 100)
3. Build SchemaSampleResult
```

### Output Model

```python
class FieldStats(BaseModel):
    path:              str              # Dot-notation, e.g. "user.profile.age"
    types:             dict[str, int]   # type_name → doc count
    present_count:     int
    missing_count:     int
    null_count:        int
    null_rate:         float            # (null + missing) / total
    cardinality:       int              # unique value count (capped at 100)
    cardinality_capped: bool
    sample_values:     list[Any]        # Up to 5 representative values

class SchemaSampleResult(BaseModel):
    collection:        str
    documents_sampled: int
    fields:            list[FieldStats]
    truncated_paths:   list[str]        # Paths that hit max depth
```

### What the planner learns

- Which fields exist and how consistently (via `null_rate`)
- Whether a field has mixed types (e.g. `{"str": 180, "int": 20}`)
- Approximate cardinality (is this an enum or a free-form field?)
- Sample values to inform hypothesis generation

---

## run_query

**File:** `tools/run_query.py`

Bounded `find()` — confirms a hypothesis with exact document counts. The agent typically uses this after `schema_sample` to validate a suspected issue.

### Signature

```python
def run_query(
    connection: CosmosConnection,
    collection_name: str,
    filter: dict[str, Any],
    *,
    projection: dict[str, Any] | None = None,
    limit: int = 50,               # Hard-capped at 1000
) -> RunQueryResult
```

### Output Model

```python
class RunQueryResult(BaseModel):
    collection:     str
    filter:         dict
    documents:      list[dict]   # Up to `limit` documents
    matched_count:  int          # Full match count (pre-limit, via count_documents)
    returned_count: int
    truncated:      bool         # True if matched_count > returned_count
```

### Key property: `matched_count`

The planner uses `matched_count` (not `returned_count`) to calculate `affected_pct` for `write_finding`. A limit of 50 documents is enough to inspect samples, but `count_documents()` gives the full scope of the issue without reading every matching document.

---

## get_stats

**File:** `tools/get_stats.py`

Aggregation-based statistics. Use when exact numbers are needed without fetching document bodies.

### Signature

```python
def get_stats(
    connection: CosmosConnection,
    collection_name: str,
    field: str,
    operation: Literal["count", "min", "max", "avg", "distinct"],
    *,
    filter: dict[str, Any] | None = None,
) -> StatsResult
```

### Operations

| Operation | Aggregation pipeline | Use case |
|-----------|---------------------|----------|
| `count` | `$match {field: {$exists: true}}` + `$count` | How many documents have this field at all |
| `min` | `$match` + `$group {_id: null, v: {$min: $field}}` | Minimum value (for outlier detection) |
| `max` | `$max` | Maximum value |
| `avg` | `$avg` | Mean (for numeric fields) |
| `distinct` | `$group {_id: $field}` + `$limit 1000` | Cardinality and unique value list |

### Output Model

```python
class StatsResult(BaseModel):
    collection:  str
    field:       str
    operation:   str
    value:       Any         # int, float, list, etc. depending on operation
    filter_used: dict | None
    query_used:  str         # Human-readable description of the pipeline
```

### `distinct` cardinality cap

`distinct` returns up to 1000 unique values. If the result hits 1000, the planner should treat cardinality as "high/unbounded" and switch strategy (e.g. use `min`/`max`/`avg` instead of listing all values).

---

## write_finding

**File:** `tools/write_finding.py`

Commits a confirmed data-quality finding into `FindingsCollector`. **Idempotent** — calling twice with the same `(field, category)` updates the existing finding in place and returns the same UUID.

### Signature (via ActionInput)

```python
def write_finding(
    collector: FindingsCollector,
    *,
    field: str,
    category: str,
    severity: FindingSeverity,       # "critical" | "high" | "medium" | "low"
    description: str,
    hypothesis: str,
    evidence_query: str,             # The filter that confirmed the finding
    affected_count: int,
    affected_pct: float,             # Fraction [0.0, 1.0], NOT a percentage
    sample_values: list[Any] | None = None,
    confirmed: bool = True,
) -> UUID
```

### Idempotency

```python
key = (field, category)

if key in self._by_key:
    # Update in place — same UUID, updated counts and description
    self._by_key[key] = existing.model_copy(update={...})
    return existing.id

# New finding
finding = Finding(id=uuid4(), ...)
self._by_key[key] = finding
return finding.id
```

This prevents the planner from accumulating duplicate findings across iterations if it revisits the same issue.

### Finding Model

```python
class FindingSeverity(StrEnum):
    CRITICAL = "critical"   # ≥1% affected, unambiguous data corruption
    HIGH     = "high"       # Clear quality issue, meaningful impact
    MEDIUM   = "medium"     # Borderline or partially affecting
    LOW      = "low"        # Minor, cosmetic, or edge case

class Finding(BaseModel):
    id:            UUID
    field:         str              # Dot-notation path, e.g. "orders.payment.method"
    category:      str              # e.g. "null_rate", "type_mismatch", "outlier_value"
    severity:      FindingSeverity
    description:   str
    hypothesis:    str              # Suspected root cause
    evidence_query: str             # Reproducible filter for the evidence
    affected_count: int
    affected_pct:  float            # [0.0, 1.0]
    sample_values: list[Any]
    confirmed:     bool = True
```

### After write_finding: the Finding Gate

The loop does not immediately commit the finding to `AgentState.findings`. It first passes the candidate through the **Finding Gate** evaluators:

```
write_finding tool call
    └── Finding candidate
            │
        Finding Gate
         (rules / self / composite)
            │
         PASS → state.findings.append(finding)
         FAIL → state.dismissed_findings.append(finding)
                 state.last_critique = verdict.critique
```

A dismissed finding is still recorded — it appears in the `AuditReport` and in Postgres, and the `DismissedPattern` is loaded into the next run's `HistoricalContext` so the planner knows not to re-propose it without new evidence.

---

## Tool Constraints Summary

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| `run_query` max limit | 1000 | Prevents quota burn on large Cosmos DB collections |
| `run_query` default limit | 50 | Usually enough for sample inspection |
| `schema_sample` max depth | 20 | Prevents unbounded recursion on deeply nested documents |
| `get_stats` distinct cap | 1000 | Cosmos DB aggregation limit; agent should infer "high cardinality" at cap |
| `schema_sample` cardinality cap | 100 | FieldStats.cardinality is capped; `cardinality_capped=True` flags the limit |
