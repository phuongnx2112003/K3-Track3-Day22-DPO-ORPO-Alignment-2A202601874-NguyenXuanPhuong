# Lab Guide

## Task 1: Data loader
The JSONL loader validates records, reports line-numbered errors, rejects duplicate normalized prompts, and performs a prompt-grouped deterministic split.

## Task 1.5: (Optional) Synthetic Data Generation
Use an LLM to expand your dataset. This teaches you how to create high-quality alignment data at scale.
```bash
export OPENAI_API_KEY=your_key
python scripts/generate_data.py --count 10 --domain "python coding"
```

## Task 2: Loss function
The completed implementation provides numerically stable DPO and simplified ORPO-style losses in `src/preference_lab/losses.py`.

## Task 3: Evaluation
Use model-derived response sequence log probabilities with `pref-lab evaluate-model configs/local.yaml`. The evaluator fails if the trained checkpoint is missing; it does not use mock scores.

## Task 4: Report
The completed report is `docs/REPORT.md`; the data card is `docs/DATA_CARD.md`. Run `pref-lab regression configs/local.yaml` and preserve its raw before/after output in `outputs/regression.json`.
