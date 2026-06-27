# Architecture

## System Context

The repository is a local, script-driven research environment. It organizes forecasting work into data preparation, shared utility logic, experiment families, and persisted outputs. There is no approved hosted runtime or service boundary in the current contract.

## Major Technical Areas

- Preparation flow: transforms historical market data into chronologically ordered forecasting datasets.
- Shared workflow logic: provides reusable data-windowing, split, metric, and plotting behavior across experiments.
- Experiment families: execute baseline, tree-based, recurrent, and convolutional forecasting workflows, including walk-forward quantile forecasting.
- Output storage: preserves metrics, models, tables, and visual artifacts for later inspection.

## Data And Artifact Ownership

- Historical market inputs are the source of truth for preparation.
- Prepared datasets own reusable targets and derived features for downstream experiments.
- Experiment runs own their saved metrics, plots, and serialized model artifacts.

<a id="arch-flow-001"></a>

## ARCH-FLOW-001: Temporal Execution Contract

To satisfy [CAP-FCST-002](../intent/capabilities/forecasting.md#cap-fcst-002) and [QUAL-LEAKAGE-001](../intent/quality.md#qual-leakage-001), experiment workflows use chronological sample construction and temporal split logic with separation rules that prevent history windows from reading future evaluation periods.

<a id="arch-run-001"></a>

## ARCH-RUN-001: Local Script Execution Contract

To satisfy [QUAL-REPRO-001](../intent/quality.md#qual-repro-001), the approved execution model is local script or notebook execution from a checked-in repository state with declared Python dependencies.

<a id="arch-artifact-001"></a>

## ARCH-ARTIFACT-001: Persisted Result Contract

To satisfy [CAP-EVAL-002](../intent/capabilities/evaluation.md#cap-eval-002) and [QUAL-TRACE-001](../intent/quality.md#qual-trace-001), each experiment family persists outputs in a run-associated artifact area that preserves horizon or mode distinctions when those distinctions exist.

## Cross-Cutting Technical Rules

- Shared utilities are the preferred place for reusable temporal split and metric logic.
- Experiment-specific model architecture details may vary without changing intent, as long as they continue to satisfy the approved forecasting and evaluation requirements.
- Quantile forecasting is an approved realization path within the repository, not a separate product.
