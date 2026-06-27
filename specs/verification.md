# Verification

## Verification Approach

The current approved verification baseline is manual execution and repository inspection. This is intentional: the repository does not yet contain an automated test suite, so this spec records explicit checks and explicit gaps instead of implying nonexistent coverage.

## Requirement-to-Check Matrix

| Requirement ID | Verification Method | Current Check |
| --- | --- | --- |
| [CAP-PREP-001](intent/capabilities/preparation.md#cap-prep-001) | Inspection | Review preparation flow and README-described dataset assumptions for chronological normalization and invalid-row handling. |
| [CAP-PREP-002](intent/capabilities/preparation.md#cap-prep-002) | Inspection | Review prepared-target generation for 5, 15, and 30 trading-day horizons. |
| [CAP-PREP-003](intent/capabilities/preparation.md#cap-prep-003) | Inspection | Review derived-feature preparation behavior for chronology-preserving missing values. |
| [CAP-FCST-001](intent/capabilities/forecasting.md#cap-fcst-001) | Inspection | Review experiment contracts to confirm multi-horizon outputs for the main workflows. |
| [CAP-FCST-002](intent/capabilities/forecasting.md#cap-fcst-002) | Inspection | Review temporal split and window-separation behavior in shared workflow logic and experiment descriptions. |
| [CAP-FCST-003](intent/capabilities/forecasting.md#cap-fcst-003) | Inspection | Review point-forecast experiment outputs and experiment documentation. |
| [CAP-FCST-004](intent/capabilities/forecasting.md#cap-fcst-004) | Inspection | Review walk-forward quantile workflow documentation and persisted fold-oriented outputs. |
| [CAP-EVAL-001](intent/capabilities/evaluation.md#cap-eval-001) | Inspection | Review metrics outputs and experiment descriptions for common comparison semantics. |
| [CAP-EVAL-002](intent/capabilities/evaluation.md#cap-eval-002) | Inspection | Review persisted artifact structure for post-run inspection support. |
| [CAP-EVAL-003](intent/capabilities/evaluation.md#cap-eval-003) | Inspection | Review yearly or fold-based outputs where the workflow exposes them. |
| [QUAL-REPRO-001](intent/quality.md#qual-repro-001) | Inspection | Review declared dependencies and local execution paths in project documentation. |
| [QUAL-LEAKAGE-001](intent/quality.md#qual-leakage-001) | Inspection | Review chronology, gap, and split semantics in shared data workflow logic. |
| [QUAL-TRACE-001](intent/quality.md#qual-trace-001) | Inspection | Review saved output organization for run, horizon, and mode traceability. |

## Known Gaps

- No automated unit, integration, or regression tests currently enforce the preparation, forecasting, or evaluation contracts.
- No automated verification currently checks artifact completeness after experiment execution.
- No automated contract currently validates quantile ordering or fold-by-fold walk-forward consistency.

## Accepted Unverified Assumptions

- Existing checked-in outputs are representative examples of the documented workflows, not formal proofs of current code behavior.
- Manual inspection is an acceptable baseline until a later change explicitly introduces automated verification requirements.
