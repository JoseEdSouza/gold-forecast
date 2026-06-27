# Evaluation and Comparison

## Purpose

Define how the toolkit exposes experiment outcomes so the researcher can compare forecast behavior across horizons and experiment families.

## Actors and Permissions

- Researcher: may inspect metrics and generated artifacts from completed runs.

## User Journeys

1. The researcher completes an experiment run.
2. The toolkit emits metrics and visual artifacts tied to that run.
3. The researcher compares model behavior by horizon and by historical regime.

## Behavior

<a id="cap-eval-001"></a>

### CAP-EVAL-001: Report Comparable Forecast Metrics

**Description:** The toolkit shall report forecast evaluation results on a common basis that supports comparison across experiment families and horizons.

**Priority:** Must

**Rationale:** Comparability is the main research value of the repository.

**Acceptance Criteria:**

1. WHEN an experiment run finishes THEN the toolkit SHALL emit summary metrics for each evaluated horizon.
2. IF multiple experiment families are executed THEN the resulting metrics SHALL be comparable on a shared interpretation basis.

**Verification:** Inspection

**Traceability:** [DOM-EVALUATION-001](../domain.md#dom-evaluation-001), [QUAL-TRACE-001](../quality.md#qual-trace-001)

<a id="cap-eval-002"></a>

### CAP-EVAL-002: Preserve Run Artifacts For Inspection

**Description:** The toolkit shall preserve generated artifacts that let the researcher inspect prediction behavior, residual behavior, and split or series context after a run.

**Priority:** Should

**Rationale:** Visual and tabular artifacts support diagnosis beyond scalar metrics.

**Acceptance Criteria:**

1. WHEN an experiment run completes successfully THEN the toolkit SHALL preserve generated artifacts for later local inspection.
2. IF an artifact relates to a specific horizon or run mode THEN the toolkit SHALL keep that association visible in the artifact set.

**Verification:** Inspection

**Traceability:** [QUAL-TRACE-001](../quality.md#qual-trace-001), [ARCH-ARTIFACT-001](../../realization/architecture.md#arch-artifact-001)

<a id="cap-eval-003"></a>

### CAP-EVAL-003: Support Regime-Aware And Fold-Aware Review

**Description:** The toolkit shall support reviewing results by historical regime or evaluation fold when that context exists in the experiment workflow.

**Priority:** Should

**Rationale:** Aggregate metrics alone can hide unstable behavior across time.

**Acceptance Criteria:**

1. WHEN yearly or fold-based summaries are available THEN the toolkit SHALL expose them as part of the run outputs.
2. IF a workflow uses walk-forward evaluation THEN the toolkit SHALL preserve fold-specific results for inspection.

**Verification:** Inspection

**Traceability:** [DOM-EVALUATION-001](../domain.md#dom-evaluation-001), [CAP-FCST-004](forecasting.md#cap-fcst-004)

## Out of Scope

- Automated report publishing
- External dashboard serving
