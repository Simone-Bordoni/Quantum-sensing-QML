# Minimal callback tests
import pytest
from qsopt import OptimizationCallback, TrainableParameters

def test_callback_init():
    cb = OptimizationCallback()
    assert cb.epoch == 0

def test_callback_call():
    cb = OptimizationCallback()
    p = TrainableParameters()
    p.add_rotation_angles('theta', 1.0)
    cb(p, 0.8, 0.2, 0.6)
    assert cb.epoch == 1
