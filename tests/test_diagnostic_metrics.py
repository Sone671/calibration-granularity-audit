import pandas as pd

import diagnostic_metrics as dm


def example_rows():
    return [
        {"window":"m1","method":"rolling_global_norm","macro_user_abs_coverage_gap":.05,"max_abs_cluster_coverage_gap":.02},
        {"window":"m1","method":"rolling_group_norm","macro_user_abs_coverage_gap":.04,"max_abs_cluster_coverage_gap":.025},
        {"window":"m1","method":"rolling_user_norm","macro_user_abs_coverage_gap":.03,"max_abs_cluster_coverage_gap":.04},
        {"window":"m2","method":"rolling_global_norm","macro_user_abs_coverage_gap":.02,"max_abs_cluster_coverage_gap":.05},
        {"window":"m2","method":"rolling_group_norm","macro_user_abs_coverage_gap":.03,"max_abs_cluster_coverage_gap":.03},
        {"window":"m2","method":"rolling_user_norm","macro_user_abs_coverage_gap":.04,"max_abs_cluster_coverage_gap":.02},
    ]


def test_conflict_rank_reversal_and_material_profile():
    window_rows=example_rows()
    g,rows=dm.granularity_conflict(pd.DataFrame(window_rows))
    assert g["personalization_conflict_windows"]==1 and g["reverse_conflict_windows"]==1
    assert abs(rows[0]["pareto_conflict_margin"]-.02)<1e-12
    assert abs(rows[0]["user_weight_switch_threshold"]-.5)<1e-12
    magnitude,thresholds,_=dm.material_conflict_profile(pd.DataFrame(window_rows),deltas=(0,.015,.025))
    assert magnitude["strict_conflicts"]==1
    assert [row["conflicts"] for row in thresholds]==[1,1,0]
    r,_=dm.rank_reversal(pd.DataFrame(window_rows),lambdas=(1.,))
    assert r["rank_reversals"]>=1


def test_temporal_cancellation_and_oracle_gap():
    window_rows=example_rows()
    users=[{"window":"m1","method":"x","user_index":0,"coverage":.7,"n":100},{"window":"m2","method":"x","user_index":0,"coverage":.9,"n":100}]
    t=dm.temporal_cancellation(pd.DataFrame(users))[0];assert abs(t["mean_user_window_gap"]-.1)<1e-12 and abs(t["pooled_user_gap"])<1e-12
    users90=[{"window":"m1","method":"x","user_index":0,"coverage":.8,"n":100},{"window":"m2","method":"x","user_index":0,"coverage":1.,"n":100}]
    t90=dm.temporal_cancellation(pd.DataFrame(users90),target=.9)[0];assert abs(t90["mean_user_window_gap"]-.1)<1e-12 and abs(t90["pooled_user_gap"])<1e-12
    unequal=[{"window":"m1","method":"x","user_index":0,"coverage":.7,"n":100},{"window":"m2","method":"x","user_index":0,"coverage":.9,"n":300}]
    tu=dm.temporal_cancellation(pd.DataFrame(unequal))[0]
    assert tu["temporal_cancellation_absolute"]>=0 and abs(tu["mean_user_window_gap"]-.1)<1e-12 and abs(tu["pooled_user_gap"]-.05)<1e-12
    assert abs(tu["equal_window_temporal_cancellation_ratio"]-1.)<1e-12
    s90,_=dm.compute_all(window_rows,users90,target=.9);assert abs(s90["temporal_cancellation"][0]["pooled_user_gap"])<1e-12
    rog=dm.routing_oracle_gap(pd.DataFrame(window_rows),lambdas=(1.,));assert rog[0]["routing_oracle_gap"]>0


def test_sign_baseline_and_stratified_conflict_stability():
    alternating=pd.DataFrame([
        {"window":"m1","method":"x","user_index":0,"coverage":.7,"n":100},
        {"window":"m2","method":"x","user_index":0,"coverage":.9,"n":100},
    ])
    sign=dm.temporal_cancellation_sign_baseline(alternating,n_replicates=2000,seed=7)[0]
    assert abs(sign["observed_temporal_cancellation_ratio"]-1.)<1e-12
    assert .4<sign["sign_null_mean"]<.6
    assert sign["excess_over_sign_null"]>.4

    rows=[]
    for user in range(20):
        cluster=user//10
        global_coverage=.75 if user%2==0 else .85
        user_coverage=.82 if cluster==0 else .78
        for method,coverage in (("rolling_global_norm",global_coverage),("rolling_user_norm",user_coverage)):
            rows.append({"user_index":user,"cluster":cluster,"method":method,"coverage":coverage,"n":100})
    stability=dm.stratified_conflict_stability(pd.DataFrame(rows),n_replicates=200,seed=7)
    assert stability["observed_personalization_conflict"]
    assert stability["observed_user_gap_difference_user_minus_global"]<0
    assert stability["observed_segment_gap_difference_user_minus_global"]>0
    assert 0<=stability["bootstrap_strict_conflict_share"]<=1
