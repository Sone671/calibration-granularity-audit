import numpy as np
import pandas as pd

import build_lightgbm_router_full_grid_report as report


def test_sequential_comparators_use_only_completed_months():
    losses = [
        (0.30, 0.20, 0.10),
        (0.10, 0.40, 0.50),
        (0.10, 0.40, 0.50),
    ]
    rows = []
    for order, (global_loss, segment_loss, user_loss) in enumerate(losses):
        month_losses = {
            report.GLOBAL: global_loss,
            report.SEGMENT: segment_loss,
            report.USER: user_loss,
        }
        rows.append(
            {
                "forecaster": "LightGBM",
                "dataset": "Synthetic",
                "configuration": "lightgbm_quantile__c80__h1",
                "coverage": 0.8,
                "horizon_hours": 1.0,
                "window": f"month-{order}",
                "window_order": order,
                "selected_policy": report.USER,
                "oracle_policy": report.conservative_argmin(month_losses),
                "mean_fold_policy": report.USER,
                "latest_fold_policy": report.USER,
                **{f"loss__{policy}": loss for policy, loss in month_losses.items()},
            }
        )
    events = report.add_sequential_choices(pd.DataFrame(rows))
    assert events.previous_window_policy.tolist() == [
        report.GLOBAL,
        report.USER,
        report.GLOBAL,
    ]
    assert events.follow_the_leader_policy.tolist() == [
        report.GLOBAL,
        report.USER,
        report.GLOBAL,
    ]
    assert set(events.best_fixed_policy) == {report.GLOBAL}


def test_synchronized_ci_is_exact_for_constant_contrast():
    rows = []
    for dataset, months in (("A", 3), ("B", 4), ("C", 5)):
        for configuration in ("c1", "c2"):
            for month in range(months):
                rows.append(
                    {
                        "dataset": dataset,
                        "configuration": configuration,
                        "window_order": month,
                        "delta": -0.125,
                    }
                )
    frame = pd.DataFrame(rows)
    for hierarchical in (False, True):
        point, ci = report.synchronized_ci(
            frame,
            "delta",
            block=2,
            reps=250,
            seed=123,
            resample_datasets=hierarchical,
        )
        assert np.isclose(point, -0.125)
        assert np.allclose(ci, [-0.125, -0.125])
