# Spec Index

## Purpose

This spec set is the durable contract for `gold-forecast`. It separates product intent from technical realization so the repository's research behavior can be reviewed without implementation noise, while the implementation structure remains traceable to approved requirements.

## Reader Paths

| Reader Goal | Start Here | Continue To |
| --- | --- | --- |
| Understand project purpose and scope | [intent/product.md](intent/product.md) | [intent/capabilities/index.md](intent/capabilities/index.md), [intent/domain.md](intent/domain.md) |
| Review required research behavior | [intent/capabilities/index.md](intent/capabilities/index.md) | [intent/capabilities/preparation.md](intent/capabilities/preparation.md), [intent/capabilities/forecasting.md](intent/capabilities/forecasting.md), [intent/capabilities/evaluation.md](intent/capabilities/evaluation.md) |
| Review shared terminology and invariants | [intent/domain.md](intent/domain.md) | [intent/quality.md](intent/quality.md), [realization/architecture.md](realization/architecture.md) |
| Understand implementation structure | [realization/architecture.md](realization/architecture.md) | [verification.md](verification.md) |
| Verify a requirement or accepted gap | [verification.md](verification.md) | Linked capability and quality requirements |

## Spec Categories

### Intent Specs

Intent specs describe the repository's purpose, primary actor, user-visible research behavior, shared domain semantics, and measurable quality outcomes. They avoid source paths, libraries, and runtime internals unless those details are externally constraining.

| File | Owns | Does Not Own |
| --- | --- | --- |
| [intent/product.md](intent/product.md) | Purpose, actor, business goals, scope, capability map, assumptions | Module layout, libraries, script names |
| [intent/capabilities/index.md](intent/capabilities/index.md) | Capability catalog and requirement ranges | Detailed requirements |
| [intent/capabilities/preparation.md](intent/capabilities/preparation.md) | Dataset preparation behavior | Feature engineering implementation details |
| [intent/capabilities/forecasting.md](intent/capabilities/forecasting.md) | Forecast experiment behavior, horizons, and forecast modes | Model architecture internals |
| [intent/capabilities/evaluation.md](intent/capabilities/evaluation.md) | Experiment comparison and reporting behavior | Plotting implementation details |
| [intent/domain.md](intent/domain.md) | Shared vocabulary, concepts, and invariants | Storage layout and package ownership |
| [intent/quality.md](intent/quality.md) | Reproducibility, leakage-control, and traceability quality requirements | Specific code mechanisms |

### Realization Specs

Realization specs describe how the approved research behavior is implemented in the repository and link back to the intent requirements they satisfy.

| File | Owns | Links Back To |
| --- | --- | --- |
| [realization/architecture.md](realization/architecture.md) | Repository structure, runtime flow, data/artifact ownership, and technical constraints | Capability, domain, and quality IDs |

### Verification Specs

| File | Owns | Links Back To |
| --- | --- | --- |
| [verification.md](verification.md) | Manual verification matrix, accepted gaps, and requirement traceability | Capability and quality IDs |

## Standards Mapping

| Reference | How This Spec Set Uses It |
| --- | --- |
| [ISO/IEC/IEEE 29148:2018](https://www.iso.org/standard/72089.html) | Stable requirement IDs, acceptance criteria, rationale, and verification-oriented records |
| [ISO/IEC/IEEE 42010:2022](https://www.iso.org/standard/74393.html) | Intent versus realization separation and architecture description concerns |
| [ISO/IEC/IEEE 15288:2023](https://www.iso.org/standard/81702.html) | Lifecycle framing across preparation, execution, evaluation, and verification concerns |

## Traceability Rules

- Intent requirements own required research behavior.
- Realization specs explain implementation structure and link back to affected intent and quality IDs.
- Verification records how each changed requirement is checked or which gap is explicitly accepted.
- The initial spec set accepts manual verification as the current baseline and records missing automation as a known gap.

## Conformance Checklist

- Intent files keep implementation detail out unless it is an externally visible research constraint.
- Realization content links to intent and quality IDs instead of redefining behavior.
- Verification covers each capability and quality requirement in this initial contract.

## Naming Rules

- Capability requirement IDs: `CAP-<CAPABILITY>-NNN`
- Quality requirement IDs: `QUAL-<QUALITY>-NNN`
- Domain concept IDs: `DOM-<CONCEPT>-NNN`
- Architecture contract IDs: `ARCH-<VIEW>-NNN`
