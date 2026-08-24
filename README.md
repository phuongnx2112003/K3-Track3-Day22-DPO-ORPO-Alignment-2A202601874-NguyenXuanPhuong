# Preference Alignment Lab: DPO \& ORPO Starter

Production-style lab implementation for DPO preference alignment, evaluation, regression checks, tests, and reporting.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

Recommended model for the lab:

Use `distilgpt2` as a deliberately weak, non-instruction-tuned causal language-model baseline so DPO changes are visible on a small dataset.

Prepare the data before fine-tuning:

```bash
pref-lab prepare data/sample_preferences.jsonl --output-dir data/processed --validation-ratio 0.2 --seed 42
```

Optional training dependencies:

```bash
pip install -e '.[dev,train]'
```

## Lab rules

1. Do not rewrite the whole repository.
2. Keep generated model artifacts and secrets out of commits.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `make test` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO TODO | `pytest tests/test_losses.py` |
| 100-115 min | Implement evaluation and report | `make run-eval` and `make run-regression` |
| 115-120 min | One-minute demo | `cat outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [x] Dataset schema validated.
- [x] Train/eval split by prompt, not by row.
- [x] Config committed; generated artifacts ignored.
- [x] Metrics saved as JSON.
- [x] Safety regression prompts run before/after training.
- [x] Data card updated.
