# Data Card

- **Dataset name**: Preference Alignment Lab sample preferences
- **Source**: Repository-provided `data/sample_preferences.jsonl`
- **License/permission**: Educational lab data; use only within the course/repository context.
- **Schema**: JSONL records with `prompt`, `chosen`, `rejected`, and optional `metadata`.
- **Labeling rubric**: `chosen` is more accurate/helpful; `rejected` contains a plausible factual, conceptual, or formatting error.
- **Known biases**: Small English-only dataset focused on introductory machine-learning concepts; no broad demographic or domain coverage.
- **Safety/PII checks**: Pydantic validation, whitespace/case normalization, duplicate-prompt detection, and manual review; no PII was observed.
- **Train/validation/test split method**: 19/5 train-validation split by normalized prompt with deterministic seed 42; no separate test split.
