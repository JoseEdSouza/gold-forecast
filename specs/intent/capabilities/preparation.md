# Dataset Preparation

## Purpose

Define the researcher-visible behavior for constructing chronologically valid forecasting datasets from historical gold-market series.

## Actors and Permissions

- Researcher: may prepare or regenerate datasets for local experiment use.

## User Journeys

1. The researcher provides historical gold-market data.
2. The toolkit derives forecast targets and analysis-ready features.
3. The researcher receives a dataset suitable for downstream forecasting experiments.

## Behavior

<a id="cap-prep-001"></a>

### CAP-PREP-001: Preserve Chronological Market Records

**Description:** The toolkit shall normalize input market records into a chronologically ordered gold-price series with a valid date field and numeric closing price.

**Priority:** Must

**Rationale:** Forecast experiments depend on ordered time-series semantics.

**Acceptance Criteria:**

1. WHEN the researcher provides valid historical market records THEN the toolkit SHALL produce records ordered by date.
2. IF rows lack a valid date or closing price THEN the toolkit SHALL exclude those rows from the prepared dataset.

**Verification:** Inspection

**Traceability:** [DOM-DATASET-001](../domain.md#dom-dataset-001), [QUAL-REPRO-001](../quality.md#qual-repro-001)

<a id="cap-prep-002"></a>

### CAP-PREP-002: Generate Multi-Horizon Forecast Targets

**Description:** The toolkit shall generate forward-looking forecast targets for the approved multi-horizon workflow using 5, 15, and 30 trading-day horizons.

**Priority:** Must

**Rationale:** Shared horizons are required for cross-model comparison.

**Acceptance Criteria:**

1. WHEN the researcher prepares a multi-horizon dataset THEN the toolkit SHALL include targets for 5, 15, and 30 trading-day horizons.
2. IF future observations for a horizon are unavailable THEN the toolkit SHALL leave that target unavailable rather than inventing a value.

**Verification:** Inspection

**Traceability:** [DOM-HORIZON-001](../domain.md#dom-horizon-001), [CAP-FCST-001](forecasting.md#cap-fcst-001)

<a id="cap-prep-003"></a>

### CAP-PREP-003: Produce Analysis-Ready Derived Features

**Description:** The toolkit shall derive analysis-ready numeric features from the historical series so non-baseline experiments can reuse a common prepared dataset.

**Priority:** Should

**Rationale:** Reusable derived features support fairer comparison across experiment families.

**Acceptance Criteria:**

1. WHEN the researcher prepares the shared feature dataset THEN the toolkit SHALL emit numeric derived features alongside anchor dates and closing prices.
2. IF a derived feature cannot be computed for early rows because of lookback needs THEN the toolkit SHALL preserve the row chronology and surface the missing feature values rather than backfilling future information.

**Verification:** Inspection

**Traceability:** [DOM-DATASET-001](../domain.md#dom-dataset-001), [QUAL-LEAKAGE-001](../quality.md#qual-leakage-001)

## Edge Cases and Failure Behavior

- Invalid or non-chronological source rows are excluded from the prepared dataset.
- Horizon targets near the end of the series may remain unavailable because future data does not exist yet.

## Out of Scope

- Contracting external data vendors
- Live data ingestion guarantees
