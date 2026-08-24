# Preference Alignment Experiment Report

This report records the real local DPO run and model-derived evaluation. No mock metrics were used.

Dataset: 24 examples from `data/sample_preferences.jsonl`, split into 19 train and 5 validation examples by normalized prompt with seed 42.

Training used TRL DPO with `distilgpt2`, beta 0.1, batch size 2, maximum length 512, and 2 CPU steps. The saved checkpoint is `outputs/dpo_model/`; the reported training loss was 0.7431.

Evaluation was recomputed from response-only sequence log probabilities of the saved checkpoint. `outputs/metrics.json` contains 5 validation examples and pairwise accuracy 0.0. This is a negative experimental result, not a mocked success.

Safety regression was run with `pref-lab regression configs/local.yaml` against the base and DPO models. Raw outputs are in `outputs/regression.json`. Both models generated newline-only output for all four concrete prompts, so no positive safety claim is made; the raw behavior is preserved.

Limitations: the dataset is small and English-only, the model is not instruction-tuned, two CPU steps are insufficient evidence of robust alignment, and there is no separate test set.

Reproduce with:

```bash
./.venv/bin/pref-lab prepare data/sample_preferences.jsonl --output-dir data/processed --validation-ratio 0.2 --seed 42
./.venv/bin/pref-lab train configs/local.yaml
./.venv/bin/pref-lab evaluate-model configs/local.yaml
./.venv/bin/pref-lab regression configs/local.yaml
```
