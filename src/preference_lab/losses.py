from __future__ import annotations

import numpy as np


def dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray, ref_chosen_logps: np.ndarray, ref_rejected_logps: np.ndarray, beta: float) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    Compare the policy margin against the reference margin with stable arithmetic.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)]
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not arrays[0].size or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("all log-probability arrays must be non-empty and have the same shape")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("log-probability arrays must contain only finite values")
    margin = beta * ((arrays[0] - arrays[1]) - (arrays[2] - arrays[3]))
    return float(np.mean(np.logaddexp(0.0, -margin)))

def orpo_loss(sft_nll: np.ndarray, chosen_logps: np.ndarray, rejected_logps: np.ndarray, lambda_orpo: float) -> float:
    """Compute a simplified ORPO-style objective.

    Combine mean SFT NLL with a stable odds-ratio preference penalty.
    """
    nll, chosen, rejected = [np.asarray(x, dtype=np.float64) for x in (sft_nll, chosen_logps, rejected_logps)]
    if lambda_orpo < 0:
        raise ValueError("lambda_orpo must be non-negative")
    if not nll.size or any(x.shape != nll.shape for x in (chosen, rejected)):
        raise ValueError("ORPO arrays must be non-empty and have the same shape")
    if not all(np.isfinite(x).all() for x in (nll, chosen, rejected)):
        raise ValueError("ORPO inputs must contain only finite values")
    chosen_odds = chosen
    rejected_odds = rejected
    penalty = np.mean(np.logaddexp(0.0, -(chosen_odds - rejected_odds)))
    return float(np.mean(nll) + lambda_orpo * penalty)
