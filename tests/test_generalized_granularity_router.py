import numpy as np

from generalized_granularity_router import GLOBAL, GROUP, USER, select_policy


def metrics(user_gap, group_gap, score=1.0):
    return {
        "macro_user_abs_coverage_gap": user_gap,
        "max_abs_cluster_coverage_gap": group_gap,
        "normalized_interval_score": score,
    }


def test_router_selects_stably_better_user_policy():
    folds = [
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.02, .02)},
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.02, .02)},
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.02, .02)},
    ]
    assert select_policy(folds, user_weight=.5).selected_policy == USER


def test_router_falls_back_when_gain_is_unstable():
    folds = [
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.01, .01)},
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.09, .09)},
        {GLOBAL: metrics(.05, .05), GROUP: metrics(.04, .04), USER: metrics(.01, .01)},
    ]
    assert select_policy(folds, user_weight=.5).selected_policy in {GLOBAL, GROUP}


def test_efficiency_penalty_can_change_the_decision():
    folds = [
        {GLOBAL: metrics(.03, .03, .5), GROUP: metrics(.03, .03, .5), USER: metrics(.02, .02, 2.0)}
        for _ in range(3)
    ]
    assert select_policy(folds, user_weight=.5, efficiency_weight=0.0).selected_policy == USER
    assert select_policy(folds, user_weight=.5, efficiency_weight=.02).selected_policy == GLOBAL

