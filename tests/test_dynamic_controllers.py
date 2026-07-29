import numpy as np

import run_prequential_aci as online


def test_saturated_integrator_zero_and_sign():
    assert online.saturated_integrator(0.0, 10.0) == 0.0
    assert online.saturated_integrator(1.0, 10.0) > 0.0
    assert online.saturated_integrator(-1.0, 10.0) < 0.0


def test_pid_panel_update_is_atomic_and_finite():
    y = np.array([0.0, 2.0, 0.0, 2.0])
    qlo = np.zeros(4)
    qhi = np.ones(4)
    users = np.array([0, 1, 0, 1])
    labels = np.array([0, 0])
    scales = np.ones(2)
    target_index = np.array([10, 10, 11, 11])
    score_sets = {
        "global": np.sort(np.linspace(-0.5, 1.0, 40)),
        "segment": [np.sort(np.linspace(-0.5, 1.0, 40))],
        "user": [np.sort(np.linspace(-0.5, 1.0, 20)), np.sort(np.linspace(-0.5, 1.0, 20))],
    }
    corrections, trace = online.run_batched_pid(
        y, qlo, qhi, users, labels, scales, target_index, score_sets
    )
    assert len(trace) == 2
    for values in corrections.values():
        assert np.isfinite(values).all()
        assert values[0] == values[1]
