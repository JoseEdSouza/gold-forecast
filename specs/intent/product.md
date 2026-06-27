# Product

## Purpose

`gold-forecast` is a local research toolkit for preparing gold-market time-series datasets, running forecasting experiments, and comparing model behavior across forecast horizons. Its purpose is to support reproducible investigation of price and volatility behavior rather than live prediction serving.

## Users, Actors, and Stakeholders

- Primary actor: single researcher operating the repository locally.
- Secondary stakeholder: future maintainer reviewing results, assumptions, and artifacts.

## Business Goals

- Produce reproducible forecasting experiments over historical gold-market data.
- Compare simple and advanced model families on a common evaluation basis.
- Preserve artifacts and summaries that let the researcher inspect forecast quality by horizon and regime.

## Capability Map

| Capability | Spec | Primary Actors | Status |
| --- | --- | --- | --- |
| Dataset Preparation | [capabilities/preparation.md](capabilities/preparation.md) | Researcher | Active |
| Forecast Experiment Execution | [capabilities/forecasting.md](capabilities/forecasting.md) | Researcher | Active |
| Evaluation and Comparison | [capabilities/evaluation.md](capabilities/evaluation.md) | Researcher | Active |

## Core User Journeys

1. The researcher prepares a chronologically ordered dataset with forecast targets and derived features suitable for experiment runs.
2. The researcher runs one or more forecasting experiments over approved horizons and forecast modes.
3. The researcher reviews metrics and artifacts to compare models, horizons, and yearly behavior.

## Business Concepts

- Gold price series
- Derived feature dataset
- Forecast horizon
- Experiment run
- Point forecast
- Quantile forecast
- Evaluation artifact

## Boundaries and Out of Scope

- Live forecasting services or APIs
- Automated deployment or hosted execution
- External data vendor guarantees, quotas, or SLAs
- Portfolio decision automation or trade execution

## Assumptions, Constraints, and Dependencies

- The repository is operated locally by a technically capable researcher.
- Historical market data is available before running preparation or experiments.
- The initial canonical spec covers existing experiment families, including walk-forward quantile forecasting, without freezing benchmark thresholds.

## Glossary

- Point forecast: a single predicted value per horizon.
- Quantile forecast: an interval-oriented prediction represented by ordered quantiles for a horizon.
- Horizon: the number of trading days between the anchor date and the forecast target date.
