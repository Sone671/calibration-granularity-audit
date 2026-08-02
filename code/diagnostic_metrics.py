"""Reusable diagnostics for calibration-granularity conflict under drift."""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

TARGET=.80
POLICIES=("rolling_global_norm","rolling_group_norm","rolling_user_norm")
LAMBDAS=(0.,.25,.5,.75,1.)


def granularity_conflict(window_frame):
    p=window_frame.pivot(index="window",columns="method",values=["macro_user_abs_coverage_gap","max_abs_cluster_coverage_gap"])
    du=p["macro_user_abs_coverage_gap"]["rolling_user_norm"]-p["macro_user_abs_coverage_gap"]["rolling_global_norm"]
    dg=p["max_abs_cluster_coverage_gap"]["rolling_user_norm"]-p["max_abs_cluster_coverage_gap"]["rolling_global_norm"]
    rows=[]
    for window in p.index:
        user_difference=float(du.loc[window]);group_difference=float(dg.loc[window])
        personalization=bool(user_difference<0 and group_difference>0);reverse=bool(user_difference>0 and group_difference<0)
        margin=min(-user_difference,group_difference) if personalization else 0.0
        tradeoff=float(np.hypot(user_difference,group_difference)) if personalization else 0.0
        switch=(group_difference/(group_difference-user_difference)) if personalization else np.nan
        rows.append({
            "window":window,
            "user_gap_difference_user_minus_global":user_difference,
            "group_gap_difference_user_minus_global":group_difference,
            "personalization_conflict":personalization,
            "reverse_conflict":reverse,
            "pareto_conflict_margin":float(margin),
            "pareto_tradeoff_length":tradeoff,
            "user_weight_switch_threshold":float(switch) if np.isfinite(switch) else np.nan,
        })
    return {"windows":len(rows),"personalization_conflict_rate":float(np.mean([r["personalization_conflict"] for r in rows])),"reverse_conflict_rate":float(np.mean([r["reverse_conflict"] for r in rows])),"personalization_conflict_windows":int(sum(r["personalization_conflict"] for r in rows)),"reverse_conflict_windows":int(sum(r["reverse_conflict"] for r in rows))},rows


def material_conflict_profile(window_frame,deltas=(0.0,.0025,.005,.01)):
    """Summarize conflict frequency and magnitude on the coverage-proportion scale.

    The Pareto conflict margin is min(-Delta_user, Delta_group) for a strict
    personalization conflict and zero otherwise. A conflict is material at
    threshold delta exactly when its margin exceeds delta. The switch threshold
    is the user-gap weight at which Global and User have equal scalar loss.
    """
    _,rows=granularity_conflict(window_frame)
    frame=pd.DataFrame(rows)
    strict=frame[frame.personalization_conflict].copy()
    threshold_rows=[]
    for delta in deltas:
        count=int((frame.pareto_conflict_margin>delta).sum())
        threshold_rows.append({
            "delta":float(delta),
            "delta_coverage_points":float(100*delta),
            "conflicts":count,
            "environments":int(len(frame)),
            "conflict_rate":float(count/max(len(frame),1)),
        })
    if strict.empty:
        magnitude={
            "strict_conflicts":0,
            "mean_margin":0.0,
            "median_margin":0.0,
            "q25_margin":0.0,
            "q75_margin":0.0,
            "mean_tradeoff_length":0.0,
            "median_user_weight_switch_threshold":np.nan,
        }
    else:
        magnitude={
            "strict_conflicts":int(len(strict)),
            "mean_margin":float(strict.pareto_conflict_margin.mean()),
            "median_margin":float(strict.pareto_conflict_margin.median()),
            "q25_margin":float(strict.pareto_conflict_margin.quantile(.25)),
            "q75_margin":float(strict.pareto_conflict_margin.quantile(.75)),
            "mean_tradeoff_length":float(strict.pareto_tradeoff_length.mean()),
            "median_user_weight_switch_threshold":float(strict.user_weight_switch_threshold.median()),
        }
    return magnitude,threshold_rows,rows


def scalar_loss(frame,lam):return lam*frame["macro_user_abs_coverage_gap"]+(1-lam)*frame["max_abs_cluster_coverage_gap"]


def rank_reversal(window_frame,policies=POLICIES,lambdas=LAMBDAS):
    rows=[];total=reversals=0
    for lam in lambdas:
        f=window_frame[window_frame.method.isin(policies)].copy();f["loss"]=scalar_loss(f,lam);p=f.pivot(index="window",columns="method",values="loss")
        for a,b in itertools.combinations(policies,2):
            diff=(p[a]-p[b]).to_numpy();valid=0;changed=0
            for i in range(1,len(diff)):
                if diff[i-1]==0 or diff[i]==0:continue
                valid+=1;changed+=int(np.sign(diff[i-1])!=np.sign(diff[i]))
            total+=valid;reversals+=changed;rows.append({"lambda":lam,"policy_a":a,"policy_b":b,"adjacent_comparisons":valid,"rank_reversals":changed,"rank_reversal_rate":changed/max(valid,1)})
    return {"adjacent_comparisons":total,"rank_reversals":reversals,"rank_reversal_rate":reversals/max(total,1)},rows


def temporal_cancellation(user_window_frame,target=TARGET):
    rows=[]
    for method,f in user_window_frame.groupby("method"):
        per_user=[]
        for _,z in f.groupby("user_index",sort=False):
            signed=z.coverage.to_numpy(float)-target
            weights=z.n.to_numpy(float)
            if not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
                raise ValueError("TCI requires finite nonnegative sample weights with positive total mass")
            weighted_abs=float(np.average(np.abs(signed),weights=weights))
            weighted_pooled=float(abs(np.average(signed,weights=weights)))
            equal_abs=float(np.mean(np.abs(signed)))
            equal_pooled=float(abs(np.mean(signed)))
            per_user.append((weighted_abs,weighted_pooled,equal_abs,equal_pooled))
        values=np.asarray(per_user,float)
        mean_window_gap=float(values[:,0].mean());pooled_gap=float(values[:,1].mean());absolute=mean_window_gap-pooled_gap
        equal_mean=float(values[:,2].mean());equal_pooled=float(values[:,3].mean());equal_absolute=equal_mean-equal_pooled
        legacy_mean=float(np.mean(np.abs(f.coverage-target)))
        legacy_absolute=legacy_mean-pooled_gap
        if absolute < -1e-12 or equal_absolute < -1e-12:
            raise RuntimeError("same-weight TCI violated nonnegativity")
        rows.append({
            "method":method,
            "mean_user_window_gap":mean_window_gap,
            "pooled_user_gap":pooled_gap,
            "temporal_cancellation_absolute":max(absolute,0.0),
            "temporal_cancellation_ratio":max(absolute,0.0)/max(mean_window_gap,1e-12),
            "equal_window_mean_user_window_gap":equal_mean,
            "equal_window_pooled_user_gap":equal_pooled,
            "equal_window_temporal_cancellation_absolute":max(equal_absolute,0.0),
            "equal_window_temporal_cancellation_ratio":max(equal_absolute,0.0)/max(equal_mean,1e-12),
            "legacy_hybrid_mean_user_window_gap":legacy_mean,
            "legacy_hybrid_temporal_cancellation_absolute":legacy_absolute,
            "legacy_hybrid_temporal_cancellation_ratio":legacy_absolute/max(legacy_mean,1e-12),
            "weighting":"within-user effective-sample weighting for both terms; users macro-averaged",
        })
    return rows


def temporal_cancellation_sign_baseline(user_window_frame, target=TARGET, n_replicates=2000, seed=20260802):
    """Compare observed TCI with a magnitude-preserving zero-mean sign baseline.

    The null keeps every user--environment absolute coverage error and its
    effective-sample weight fixed, then assigns independent Rademacher signs.
    It is an interpretability baseline, not a coverage-validity test: even
    independent signs produce appreciable cancellation when several windows
    are pooled.  The returned tail share is therefore labeled descriptively
    rather than as a formal p-value.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    rng=np.random.default_rng(seed)
    rows=[]
    for method,f in user_window_frame.groupby("method",sort=False):
        user_errors=[];user_weights=[]
        for _,z in f.groupby("user_index",sort=False):
            z=z.sort_values("window")
            errors=np.abs(z.coverage.to_numpy(float)-target)
            weights=z.n.to_numpy(float)
            if (not np.isfinite(errors).all() or not np.isfinite(weights).all()
                    or np.any(weights<0) or weights.sum()<=0):
                raise ValueError("sign baseline requires finite errors and positive sample weights")
            user_errors.append(errors)
            user_weights.append(weights/weights.sum())
        if not user_errors:
            continue
        lengths={len(x) for x in user_errors}
        if len(lengths)!=1:
            raise ValueError("sign baseline requires the same number of environments per user")
        errors=np.stack(user_errors)
        weights=np.stack(user_weights)
        denominator=float(np.mean(np.sum(weights*errors,axis=1)))
        observed=temporal_cancellation(f,target=target)[0]["temporal_cancellation_ratio"]
        null_tci=[]
        # Chunking keeps memory bounded for a full user-by-window grid.
        chunk_size=min(500,n_replicates)
        for start in range(0,n_replicates,chunk_size):
            size=min(chunk_size,n_replicates-start)
            signs=2*rng.integers(0,2,size=(size,errors.shape[0],errors.shape[1]),dtype=np.int8)-1
            pooled=np.abs(np.sum(signs*errors[None,:,:]*weights[None,:,:],axis=2)).mean(axis=1)
            null_tci.append(1-pooled/max(denominator,1e-12))
        null_tci=np.concatenate(null_tci)
        rows.append({
            "method":method,
            "observed_temporal_cancellation_ratio":float(observed),
            "sign_null_mean":float(null_tci.mean()),
            "sign_null_q025":float(np.quantile(null_tci,.025)),
            "sign_null_q500":float(np.quantile(null_tci,.5)),
            "sign_null_q975":float(np.quantile(null_tci,.975)),
            "excess_over_sign_null":float(observed-null_tci.mean()),
            "sign_null_tail_share_at_least_observed":float(np.mean(null_tci>=observed)),
            "n_users":int(errors.shape[0]),
            "n_environments_per_user":int(errors.shape[1]),
            "n_replicates":int(n_replicates),
            "seed":int(seed),
            "baseline":"independent Rademacher signs; absolute errors and within-user weights fixed",
        })
    return rows


def stratified_conflict_stability(user_window_frame, target=TARGET, delta=.005, n_replicates=1000, seed=20260802):
    """Assess whether a Global-versus-User conflict depends on a few users.

    Users are resampled with replacement within their frozen operational
    segment, while each sampled user's Global and User outcomes move together.
    This captures cross-sectional stability of the two-level comparison.  It
    intentionally does *not* claim a time-series confidence interval because
    timestamp-level hits are not available in the summary panel.
    """
    if n_replicates<=0:
        raise ValueError("n_replicates must be positive")
    if delta<0:
        raise ValueError("delta must be nonnegative")
    policies=("rolling_global_norm","rolling_user_norm")
    required={"user_index","cluster","method","coverage","n"}
    missing=required-set(user_window_frame.columns)
    if missing:
        raise ValueError(f"missing columns for conflict stability: {sorted(missing)}")
    f=user_window_frame[user_window_frame.method.isin(policies)].copy()
    p=f.pivot(index=["user_index","cluster"],columns="method",values=["coverage","n"])
    if not set(policies).issubset(set(p["coverage"].columns)):
        raise ValueError("both Global and User policy rows are required")
    p=p.dropna(subset=pd.MultiIndex.from_product([["coverage"],policies]))
    coverage=p["coverage"].loc[:,policies].to_numpy(float)
    counts=p["n"].loc[:,policies].to_numpy(float)
    if not np.isfinite(coverage).all() or not np.isfinite(counts).all() or np.any(counts<=0):
        raise ValueError("coverage and sample counts must be finite and positive")
    if not np.allclose(counts[:,0],counts[:,1]):
        raise ValueError("paired policies must have the same per-user sample count")
    counts=counts[:,0]
    clusters=p.index.get_level_values("cluster").to_numpy()
    unique_clusters=pd.unique(clusters)
    cluster_indices=[np.flatnonzero(clusters==cluster) for cluster in unique_clusters]

    def two_level_gaps(cov, n):
        user_gap=np.abs(cov-target).mean(axis=0)
        segment_gaps=[]
        for index in cluster_indices:
            segment_coverage=(n[index,None]*cov[index]).sum(axis=0)/n[index].sum()
            segment_gaps.append(np.abs(segment_coverage-target))
        return user_gap,np.max(np.stack(segment_gaps),axis=0)

    observed_user,observed_segment=two_level_gaps(coverage,counts)
    observed_du=float(observed_user[1]-observed_user[0])
    observed_dg=float(observed_segment[1]-observed_segment[0])
    observed_strict=bool(observed_du<0 and observed_dg>0)
    observed_pcm=float(min(-observed_du,observed_dg)) if observed_strict else 0.0

    rng=np.random.default_rng(seed)
    strict_hits=0;material_hits=0;du_samples=[];dg_samples=[]
    chunk_size=min(250,n_replicates)
    for start in range(0,n_replicates,chunk_size):
        size=min(chunk_size,n_replicates-start)
        user_gap_sum=np.zeros((size,2),float)
        segment_gaps=[]
        for index in cluster_indices:
            sampled=index[rng.integers(0,len(index),size=(size,len(index)))]
            sampled_coverage=coverage[sampled]
            sampled_counts=counts[sampled]
            user_gap_sum+=np.abs(sampled_coverage-target).sum(axis=1)
            segment_coverage=(sampled_counts[:,:,None]*sampled_coverage).sum(axis=1)/sampled_counts.sum(axis=1)[:,None]
            segment_gaps.append(np.abs(segment_coverage-target))
        boot_user=user_gap_sum/len(coverage)
        boot_segment=np.max(np.stack(segment_gaps),axis=0)
        du=boot_user[:,1]-boot_user[:,0]
        dg=boot_segment[:,1]-boot_segment[:,0]
        strict=(du<0)&(dg>0)
        material=(du<-delta)&(dg>delta)
        strict_hits+=int(strict.sum());material_hits+=int(material.sum())
        du_samples.append(du);dg_samples.append(dg)
    du_samples=np.concatenate(du_samples);dg_samples=np.concatenate(dg_samples)
    return {
        "observed_user_gap_difference_user_minus_global":observed_du,
        "observed_segment_gap_difference_user_minus_global":observed_dg,
        "observed_personalization_conflict":observed_strict,
        "observed_pareto_conflict_margin":observed_pcm,
        "delta":float(delta),
        "bootstrap_strict_conflict_share":float(strict_hits/n_replicates),
        "bootstrap_material_conflict_share":float(material_hits/n_replicates),
        "bootstrap_du_q025":float(np.quantile(du_samples,.025)),
        "bootstrap_du_q500":float(np.quantile(du_samples,.5)),
        "bootstrap_du_q975":float(np.quantile(du_samples,.975)),
        "bootstrap_dg_q025":float(np.quantile(dg_samples,.025)),
        "bootstrap_dg_q500":float(np.quantile(dg_samples,.5)),
        "bootstrap_dg_q975":float(np.quantile(dg_samples,.975)),
        "n_users":int(len(coverage)),
        "n_segments":int(len(unique_clusters)),
        "n_replicates":int(n_replicates),
        "seed":int(seed),
        "resampling":"paired user resampling within frozen segments; cross-sectional stability screen",
    }


def routing_oracle_gap(window_frame,policies=POLICIES,lambdas=LAMBDAS):
    rows=[]
    for lam in lambdas:
        f=window_frame[window_frame.method.isin(policies)].copy();f["loss"]=scalar_loss(f,lam);p=f.pivot(index="window",columns="method",values="loss");means=p.mean(axis=0);best=means.idxmin();fixed=float(means.min());oracle=float(p.min(axis=1).mean())
        rows.append({"lambda":lam,"best_fixed_policy":best,"best_fixed_mean_loss":fixed,"window_oracle_mean_loss":oracle,"routing_oracle_gap":fixed-oracle,"oracle_policy_changes":int((p.idxmin(axis=1)!=p.idxmin(axis=1).shift()).sum()-1)})
    return rows


def compute_all(window_rows,user_window_rows,target=TARGET):
    wf=pd.DataFrame(window_rows);uf=pd.DataFrame(user_window_rows);gcr,gcr_rows=granularity_conflict(wf);magnitude,material,_=material_conflict_profile(wf);rrr,rrr_rows=rank_reversal(wf);tci=temporal_cancellation(uf,target=target);rog=routing_oracle_gap(wf)
    return {"granularity_conflict":gcr,"conflict_magnitude":magnitude,"material_conflict_profile":material,"rank_reversal":rrr,"temporal_cancellation":tci,"routing_oracle_gap":rog},{"conflict_windows":gcr_rows,"rank_reversal_pairs":rrr_rows}
