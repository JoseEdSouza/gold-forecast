# Forecast Experiment Execution

## Purpose

Define the required behavior for executing local forecasting experiments over prepared gold-market datasets.

## Actors and Permissions

- Researcher: may execute experiment runs and review resulting forecasts.

## User Journeys

1. The researcher selects an implemented experiment family.
2. The toolkit trains and evaluates the experiment over approved data partitions.
3. The toolkit produces point or quantile forecasts for later comparison.

## Behavior

<a id="cap-fcst-001"></a>

### CAP-FCST-001: Support Multi-Horizon Forecasting

**Description:** The toolkit shall support experiment workflows that predict future gold-market behavior for 5, 15, and 30 trading-day horizons from a common anchor date.

**Priority:** Must

**Rationale:** The repository's main comparative workflow is multi-horizon forecasting.

**Acceptance Criteria:**

1. WHEN the researcher runs a multi-horizon experiment THEN the toolkit SHALL emit forecasts for 5, 15, and 30 trading-day horizons from each valid anchor sample.
2. IF a workflow is explicitly baseline-only THEN the toolkit MAY use a different horizon contract as a reference workflow.

**Verification:** Inspection

**Traceability:** [CAP-PREP-002](preparation.md#cap-prep-002), [DOM-HORIZON-001](../domain.md#dom-horizon-001)

<a id="cap-fcst-002"></a>

### CAP-FCST-002: Prevent Temporal Leakage Across Evaluation Splits

**Description:** The toolkit shall separate training, validation, and test observations in chronological order so future information does not leak into earlier evaluation stages.

**Priority:** Must

**Rationale:** Research conclusions are invalid if evaluation uses leaked future context.

**Acceptance Criteria:**

1. WHEN the researcher runs a temporally split experiment THEN the toolkit SHALL maintain chronological train, validation, and test ordering.
2. IF a workflow uses history windows or forecast gaps THEN the toolkit SHALL respect a separation rule that prevents windows from reading future evaluation periods.

**Verification:** Inspection

**Traceability:** [QUAL-LEAKAGE-001](../quality.md#qual-leakage-001), [ARCH-FLOW-001](../../realization/architecture.md#arch-flow-001)

<a id="cap-fcst-003"></a>

### CAP-FCST-003: Support Point Forecast Experiment Runs

**Description:** The toolkit shall support local experiment runs that produce a single forecast value per approved horizon for comparison across model families.

**Priority:** Must

**Rationale:** Point forecasts are the baseline comparison mode across existing experiments.

**Acceptance Criteria:**

1. WHEN the researcher runs a point-forecast experiment THEN the toolkit SHALL produce one predicted value per approved horizon for each evaluated sample.
2. IF multiple point-forecast model families are available THEN the toolkit SHALL allow their outputs to be compared on a common evaluation basis.

**Verification:** Inspection

**Traceability:** [CAP-EVAL-001](evaluation.md#cap-eval-001), [QUAL-TRACE-001](../quality.md#qual-trace-001)

<a id="cap-fcst-004"></a>

### CAP-FCST-004: Support Quantile Forecast Experiment Runs

**Description:** The toolkit shall support a walk-forward forecast mode that produces ordered quantile forecasts for each approved multi-horizon target.

**Priority:** Should

**Rationale:** Interval-oriented outputs improve uncertainty inspection in the existing CNN workflow.

**Acceptance Criteria:**

1. WHEN the researcher runs the quantile forecast workflow THEN the toolkit SHALL produce ordered quantile outputs for each approved horizon.
2. IF the workflow advances through successive evaluation folds THEN the toolkit SHALL preserve the temporal ordering of those folds.

**Verification:** Inspection

**Traceability:** [DOM-FORECAST-001](../domain.md#dom-forecast-001), [CAP-EVAL-003](evaluation.md#cap-eval-003)

## Edge Cases and Failure Behavior

- Baseline reference workflows may use a narrower horizon contract than the main multi-horizon workflows.
- Forecasts for unavailable samples are omitted rather than synthesized.

## Out of Scope

- Live inference endpoints
- Automated trading decisions
