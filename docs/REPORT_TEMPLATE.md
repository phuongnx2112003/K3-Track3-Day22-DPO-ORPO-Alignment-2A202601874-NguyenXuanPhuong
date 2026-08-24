# Preference Alignment Experiment Report

- **Họ và tên**: Nguyễn Xuân Phượng
- **Mã học viên**: 2A202601874

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: 24
- **Validation issues found**: Line 1 contained unescaped quotes around `self-attention`.
- **Cleaning steps taken**: Repaired the JSON escape, validated every record with Pydantic, normalized text, and rejected duplicate prompts.

### Split Strategy
- **Train/Val Ratio**: 19/5 (79.2%/20.8%).
- **Leakage Prevention**: `split_by_prompt` groups normalized prompts before deterministic shuffling with seed 42, so one prompt cannot appear in both splits.

## 2. Implementation: DPO

### Objective Selection
- **Why this method?**: DPO directly optimizes the preference margin between chosen and rejected responses without requiring a separate reward-model training stage.
- **Key Hyperparameters**:
    - `beta`: 0.1
    - `lambda_orpo`: 0.1 (unused for DPO)
    - Model: `distilgpt2`
    - Fine-tuning steps: 2 CPU steps

### Numerical Stability
- **Challenges**: Extreme log-probability margins and invalid/empty arrays can produce overflow or silent invalid losses.
- **Solutions**: Used `numpy.logaddexp` for stable log-sigmoid calculations and validated shape, finiteness, and positive beta.

## 3. Evaluation Results

| Metric | Value |
|---|---|
| Pairwise Accuracy | 0.0 (0/5 validation pairs) |
| Final DPO Train Loss | 0.7431 |

### Qualitative Review
- **Prompt**: Explain the concept of self-attention in Transformers.
- **Chosen Response**: Self-attention weighs the importance of words in a sequence and captures long-range dependencies.
- **Rejected Response**: Self-attention is a simpler, faster version of RNNs.
- **Model Preference**: Incorrect after the short two-step CPU run.

## 4. Discussion & Failure Modes

- **What went well?**: The malformed source record was detected and repaired; preprocessing, DPO training, checkpoint saving, and model-derived evaluation all completed.
- **Observed Bias**: The weak, non-instruction-tuned model is sensitive to response length and does not reliably prefer the selected answer after only two updates.
- **Safety**: The regression prompts are documented and should be checked manually before/after training; this run reports no safety claim because the tiny validation set does not contain those prompts.
- **Failure mode**: Pairwise accuracy was 0.0, showing that two CPU steps and 19 training pairs are insufficient for reliable alignment. A GPU run with more steps should be used for a stronger experiment.
