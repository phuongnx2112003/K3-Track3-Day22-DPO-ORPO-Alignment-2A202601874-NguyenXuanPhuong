import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_is_stable() -> None:
    value = dpo_loss(np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1)
    assert value > 0.0
    assert np.isfinite(value)

def test_orpo_loss_is_finite() -> None:
    value = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert value > 1.0
    assert np.isfinite(value)
