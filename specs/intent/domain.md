# Domain

## Glossary and Ubiquitous Language

- Anchor date: the observation date from which a future horizon is predicted.
- Forecast horizon: the trading-day offset between an anchor date and a target outcome.
- Prepared dataset: a chronologically ordered dataset ready for forecasting workflows.
- Experiment run: one complete execution of a model workflow over a prepared dataset.

## Core Concepts

| Concept | Meaning | Detail Spec |
| --- | --- | --- |
| Gold-market dataset | Historical series used as the source of preparation and forecasting workflows | — |
| Forecast horizon | Future target offset used for prediction and comparison | — |
| Forecast output | Point or quantile prediction emitted by an experiment run | — |
| Evaluation summary | Metrics and artifacts used to inspect and compare outcomes | — |

## Entities and Relationships

- A gold-market dataset contains chronologically ordered observations.
- A prepared dataset derives features and targets from that ordered series.
- An experiment run consumes a prepared dataset and produces forecast outputs.
- Evaluation summaries are computed from forecast outputs and realized outcomes.

## Shared States and Lifecycles

- Raw historical observations become prepared datasets after normalization and target derivation.
- Prepared datasets become experiment inputs once enough historical context exists for a valid sample.
- Forecast outputs become evaluation summaries after comparison with realized future outcomes.

## Cross-Cutting Invariants

<a id="dom-dataset-001"></a>

### DOM-DATASET-001: Chronological Dataset Integrity

Prepared datasets preserve chronological ordering and treat missing prerequisite values as unavailable data rather than future-filled data.

<a id="dom-horizon-001"></a>

### DOM-HORIZON-001: Horizon Semantics

A forecast horizon expresses a forward trading-day offset from an anchor date to a future target outcome.

<a id="dom-forecast-001"></a>

### DOM-FORECAST-001: Forecast Output Semantics

Point forecasts represent one expected outcome per horizon, while quantile forecasts represent ordered uncertainty estimates for the same horizon.

<a id="dom-evaluation-001"></a>

### DOM-EVALUATION-001: Evaluation Interpretation

Evaluation summaries compare forecast outputs with realized future outcomes on a shared interpretation basis so experiment families can be compared meaningfully.
