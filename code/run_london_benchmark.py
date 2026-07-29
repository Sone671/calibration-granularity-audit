"""London phase-1 multi-window diagnostic benchmark."""

from __future__ import annotations
import argparse,csv,gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;LONDON=ROOT/"probabilistic_load_v5_london_final_confirmation_2026-08-02";EXT=ROOT/"probabilistic_load_hierarchical_rolling_cqr_external_2026-07-30";V3=ROOT/"probabilistic_load_adaptive_safe_shrinkage_v3_2026-07-31";V1=ROOT/"probabilistic_load_group_cqr_go_nogo_2026-07-25"
for p in (HERE,EXT,V3,V1/".deps",V1):sys.path.insert(0,str(p))
import diagnostic_metrics as diag  # noqa:E402
import segmentation_utils as seg  # noqa:E402
import run_external as ext  # noqa:E402
import run_v3 as v3  # noqa:E402
import run_validation as base  # noqa:E402

METHODS=("raw","rolling_global_norm","rolling_group_norm","rolling_user_norm")
def log(x):print(time.strftime("[%H:%M:%S]"),x,flush=True)
def write_csv(path,rows):
    if not rows:return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def evaluate(y,q10,q90,users,groups,scales,corr,n_users,n_groups):
    delta=corr[users]*scales[users];lo,hi=q10-delta,q90+delta;crossed=hi<lo;mid=.5*(lo[crossed]+hi[crossed]);lo[crossed],hi[crossed]=mid,mid;covered=(y>=lo)&(y<=hi)
    un=np.bincount(users,minlength=n_users);uc=np.bincount(users,weights=covered.astype(float),minlength=n_users);gn=np.bincount(groups,minlength=n_groups);gcov=np.bincount(groups,weights=covered.astype(float),minlength=n_groups);user_cov=uc/np.maximum(un,1);group_cov=gcov/np.maximum(gn,1)
    alpha=1.0-ext.TARGET;width=hi-lo;interval_score=width+(2.0/alpha)*(lo-y)*(y<lo)+(2.0/alpha)*(y-hi)*(y>hi)
    metrics={"n":int(len(y)),"picp":float(covered.mean()),"mpiw":float(np.mean(width)),"winkler_interval_score":float(np.mean(interval_score)),"macro_user_abs_coverage_gap":float(np.mean(np.abs(user_cov-ext.TARGET))),"user_coverage_std":float(np.std(user_cov)),"max_abs_cluster_coverage_gap":float(np.max(np.abs(group_cov-ext.TARGET)))}
    return metrics,user_cov,un


def main():
    pa=argparse.ArgumentParser();pa.add_argument("--prepared",type=Path,default=LONDON/"prepared_london");pa.add_argument("--out",type=Path,default=HERE/"london_11window_full");pa.add_argument("--train-rows",type=int,default=600000);pa.add_argument("--num-boost-round",type=int,default=250);pa.add_argument("--threads",type=int,default=4);pa.add_argument("--segments",type=int,default=3);pa.add_argument("--cluster-method",choices=("kmeans","ward"),default="kmeans")
    args=pa.parse_args();args.out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();meta=json.loads((args.prepared/"metadata.json").read_text(encoding="utf-8"));values=np.load(args.prepared/"values.npy").astype(np.float32,copy=False);names=meta["customers"]
    start=pd.Timestamp(meta["start"]).normalize();dt=pd.date_range(start,periods=len(values),freq="30min");train_end=int(dt.searchsorted(start+pd.DateOffset(months=12)));windows=[]
    for off in range(12,23):ws=start+pd.DateOffset(months=off);windows.append((ws.strftime("%Y-%m"),ws,start+pd.DateOffset(months=off+1)))
    tv=values[:train_end];uf=ext.user_features(tv,dt[:train_end]);center,spread=np.median(uf,0),np.std(uf,0);labels=seg.cluster_labels((uf-center)/np.where(spread>1e-9,spread,1),args.segments,args.cluster_method,ext.SEED,base.kmeans);stat=np.column_stack([np.nanmean(tv,0),np.nanstd(tv,0),np.nanquantile(tv,.95,axis=0),np.nanmean(tv<=1e-6,0)]).astype(np.float32);scales=np.maximum.reduce([stat[:,1],.1*stat[:,2],np.full(len(names),1e-3)]).astype(float);prefix=ext.make_prefix(values)
    o,u=ext.sampled_pairs(0,train_end,len(names),args.train_rows,ext.SEED);x,y,_,_=ext.build_rows(values,dt,prefix,o,u,labels,stat);log(f"Training rows: {len(y):,}");params={"verbosity":-1,"learning_rate":.05,"num_leaves":31,"min_data_in_leaf":100,"feature_fraction":.9,"bagging_fraction":.9,"bagging_freq":1,"lambda_l2":1.,"seed":ext.SEED,"num_threads":args.threads,"force_col_wise":True};ds=base.lgb.Dataset(x,label=y,feature_name=ext.FEATURE_NAMES,categorical_feature=[17,18],free_raw_data=False);models={}
    for tau in (.1,.5,.9):log(f"Training tau={tau}");models[tau]=base.lgb.train(dict(params,objective="quantile",alpha=tau,metric="quantile"),ds,num_boost_round=args.num_boost_round)
    del ds,x,y,tv,uf;gc.collect();nu,ng=len(names),args.segments;window_rows=[];user_window_rows=[];correction_rows=[]
    for window,ts,te in windows:
        log(f"Benchmark window {window}");cal0=int(dt.searchsorted(ts-pd.Timedelta(days=56)));test0,test1=int(dt.searchsorted(ts)),int(dt.searchsorted(te));o,u=ext.full_pairs(cal0,test0,nu);x,y,uu,gg=ext.build_rows(values,dt,prefix,o,u,labels,stat);q10,_,q90,_=base.predict_three(models,x);global_q,group_q,user_q,_,user_groups,_=v3.components(np.maximum(q10-y,y-q90),uu,gg,scales,nu,ng);corr={"raw":np.zeros(nu),"rolling_global_norm":np.full(nu,global_q),"rolling_group_norm":group_q[user_groups],"rolling_user_norm":user_q}
        correction_diag={"abs_global_correction":float(abs(global_q)),"user_correction_std":float(np.std(user_q)),"group_correction_range":float(np.ptp(group_q)),"mean_abs_user_global_disagreement":float(np.mean(np.abs(user_q-global_q)))}
        for method in METHODS:correction_rows.append({"window":window,"method":method,"mean_normalized_correction":float(np.mean(corr[method])),"std_normalized_correction":float(np.std(corr[method])),**correction_diag})
        del x,y,uu,gg,q10,q90;gc.collect();o,u=ext.full_pairs(test0,test1,nu);x,y,uu,gg=ext.build_rows(values,dt,prefix,o,u,labels,stat);q10,_,q90,_=base.predict_three(models,x)
        for method in METHODS:
            metrics,user_cov,user_n=evaluate(y,q10,q90,uu,gg,scales,corr[method],nu,ng);window_rows.append({"window":window,"method":method,**metrics})
            for j in range(nu):user_window_rows.append({"window":window,"method":method,"user_index":j,"customer":names[j],"cluster":int(labels[j]),"coverage":float(user_cov[j]),"n":int(user_n[j]),"coverage_gap":float(abs(user_cov[j]-ext.TARGET))})
        del x,y,uu,gg,q10,q90;gc.collect()
    summary,details=diag.compute_all(window_rows,user_window_rows);summary["data"]={"dataset":"London","users":nu,"segments":ng,"cluster_method":args.cluster_method,"cluster_sizes":np.bincount(labels,minlength=ng).tolist(),"windows":[x[0] for x in windows],"wall_seconds":time.perf_counter()-started}
    write_csv(args.out/"window_metrics.csv",window_rows);write_csv(args.out/"per_user_window_metrics.csv",user_window_rows);write_csv(args.out/"corrections.csv",correction_rows);write_csv(args.out/"conflict_windows.csv",details["conflict_windows"]);write_csv(args.out/"rank_reversal_pairs.csv",details["rank_reversal_pairs"]);write_csv(args.out/"temporal_cancellation.csv",summary["temporal_cancellation"]);write_csv(args.out/"routing_oracle_gap.csv",summary["routing_oracle_gap"]);(args.out/"diagnostic_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
