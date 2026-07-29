"""Audited two-pass preparation of official partitioned Low Carbon London data."""

from __future__ import annotations
import argparse,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def log(x):print(time.strftime("[%H:%M:%S]"),x,flush=True)
def month_code(ts):return ts.year*12+ts.month-1
def code_ts(code):return pd.Timestamp(year=code//12,month=code%12+1,day=1)
def expected(code):return int(pd.Period(code_ts(code),freq="M").days_in_month*48)


def monthly_scan(files,chunksize):
    stats=defaultdict(lambda:[0,0,0.,0.,0]);mn=None;mx=None;rows=invalid=0
    for fi,path in enumerate(files,1):
        for chunk in pd.read_csv(path,usecols=[0,2,3],chunksize=chunksize,low_memory=False):
            chunk.columns=["customer","datetime","value"];chunk["customer"]=chunk.customer.astype(str).str.strip()
            dt=pd.to_datetime(chunk.datetime,format="%Y-%m-%d %H:%M:%S.%f",errors="raise");value=pd.to_numeric(chunk.value.astype(str).str.strip(),errors="coerce").to_numpy(float)
            valid=np.isfinite(value)&(value>=0);month=(dt.dt.year.to_numpy()*12+dt.dt.month.to_numpy()-1).astype(np.int32)
            frame=pd.DataFrame({"customer":chunk.customer,"month":month,"row_n":1,"valid_n":valid.astype(np.int32),"sum_v":np.where(valid,value,0.),"sum_sq":np.where(valid,value*value,0.),"nonzero_n":(valid&(value>1e-6)).astype(np.int32)})
            grouped=frame.groupby(["customer","month"],sort=False)[["row_n","valid_n","sum_v","sum_sq","nonzero_n"]].sum()
            for (c,m),r in grouped.iterrows():
                z=stats[(c,int(m))];z[0]+=int(r.row_n);z[1]+=int(r.valid_n);z[2]+=float(r.sum_v);z[3]+=float(r.sum_sq);z[4]+=int(r.nonzero_n)
            mn=dt.min() if mn is None else min(mn,dt.min());mx=dt.max() if mx is None else max(mx,dt.max());rows+=len(chunk);invalid+=int((~valid).sum())
        if fi%20==0:log(f"Pass 1 files={fi}/{len(files)}, rows={rows:,}, customer-months={len(stats):,}")
    return stats,mn,mx,rows,invalid


def complete_bounds(mn,mx):
    first=mn.to_period("M").to_timestamp();last=mx.to_period("M").to_timestamp()
    if mn>first:first+=pd.offsets.MonthBegin(1)
    if mx<last+pd.offsets.MonthBegin(1)-pd.Timedelta(minutes=30):last-=pd.offsets.MonthBegin(1)
    return month_code(first),month_code(last)


def choose(stats,mn,mx):
    first,last=complete_bounds(mn,mx);customers=sorted({c for c,_ in stats});best=None
    for start in range(first,last-22+1):
        limits={m:math.ceil(.95*expected(m)) for m in range(start,start+23)}
        valid=[c for c in customers if all(stats.get((c,m),[0,0])[1]>=limits[m] for m in range(start,start+23))]
        candidate=(len(valid),-start,start,valid)
        if best is None or candidate[:2]>best[:2]:best=candidate
    if best is None:raise RuntimeError("LONDON_BLOCKED_SPAN")
    return best[2],best[3],first,last


def training_filter(stats,start,candidates):
    keep=[];detail={}
    for c in candidates:
        n=sum(stats[(c,m)][1] for m in range(start,start+18));nz=sum(stats[(c,m)][4] for m in range(start,start+18));s=sum(stats[(c,m)][2] for m in range(start,start+18));ss=sum(stats[(c,m)][3] for m in range(start,start+18))
        mean=s/max(n,1);var=max(ss/max(n,1)-mean*mean,0.);frac=nz/max(n,1);detail[c]={"train_valid":n,"train_mean":mean,"train_std":math.sqrt(var),"train_nonzero_fraction":frac}
        if frac>=.2 and var>1e-18:keep.append(c)
    return keep,detail


def interpolate(values):
    out=values.copy();filled=0
    for j in range(out.shape[1]):
        x=out[:,j];miss=~np.isfinite(x);changes=np.diff(np.r_[False,miss,False].astype(np.int8));starts=np.flatnonzero(changes==1);stops=np.flatnonzero(changes==-1)
        for a,b in zip(starts,stops):
            if b-a<=2 and a>0 and b<len(x) and np.isfinite(x[a-1]) and np.isfinite(x[b]):x[a:b]=np.linspace(x[a-1],x[b],b-a+2)[1:-1];filled+=b-a
    return out,int(filled)


def build(files,start,customers,chunksize):
    start_ts=code_ts(start);end_ts=code_ts(start+23);dt=pd.date_range(start_ts,end_ts,freq="30min",inclusive="left");mapping={c:i for i,c in enumerate(customers)}
    sums=np.zeros((len(dt),len(customers)),np.float32);counts=np.zeros((len(dt),len(customers)),np.uint8);seen=used=outside=0
    for fi,path in enumerate(files,1):
        for chunk in pd.read_csv(path,usecols=[0,2,3],chunksize=chunksize,low_memory=False):
            chunk.columns=["customer","datetime","value"];chunk.customer=chunk.customer.astype(str).str.strip();seen+=len(chunk);keep=chunk.customer.isin(mapping)
            if not keep.any():continue
            sub=chunk.loc[keep];ts=pd.to_datetime(sub.datetime,format="%Y-%m-%d %H:%M:%S.%f",errors="raise");val=pd.to_numeric(sub.value.astype(str).str.strip(),errors="coerce").to_numpy(float)
            inrange=(ts>=start_ts)&(ts<end_ts);valid=np.isfinite(val)&(val>=0)&inrange.to_numpy();outside+=int((~inrange).sum());sub=sub.loc[valid];ts=ts.loc[valid];val=val[valid]
            pos=((ts-start_ts)/pd.Timedelta(minutes=30)).astype(np.int64).to_numpy();cols=sub.customer.map(mapping).to_numpy(np.int64);np.add.at(sums,(pos,cols),val.astype(np.float32));np.add.at(counts,(pos,cols),1);used+=len(val)
        if fi%20==0:log(f"Pass 2 files={fi}/{len(files)}, rows_seen={seen:,}, selected_rows={used:,}")
    duplicate=int(np.sum(counts>1));matrix=np.where(counts>0,sums,np.nan).astype(np.float32);missing=int(np.sum(~np.isfinite(matrix)));matrix,filled=interpolate(matrix)
    return dt,matrix,{"rows_seen":seen,"selected_rows_used":used,"duplicate_customer_time_cells":duplicate,"selected_rows_outside_window":outside,"raw_missing_cells":missing,"short_gap_cells_imputed":filled,"post_imputation_missing_fraction":float(np.mean(~np.isfinite(matrix)))}


def main():
    pa=argparse.ArgumentParser();pa.add_argument("--raw-dir",type=Path,default=HERE/"raw_london");pa.add_argument("--out",type=Path,default=HERE/"prepared_london");pa.add_argument("--chunksize",type=int,default=1_000_000);args=pa.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    files=sorted(args.raw_dir.rglob("*.csv"));started=time.perf_counter();log(f"Scanning {len(files)} partition files")
    stats,mn,mx,rows,invalid=monthly_scan(files,args.chunksize);start,continuous,first,last=choose(stats,mn,mx);eligible,detail=training_filter(stats,start,continuous)
    if len(eligible)<100:raise RuntimeError(f"LONDON_BLOCKED_USERS: {len(eligible)}")
    selected=sorted(eligible,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:500];log(f"Span {code_ts(start).date()} to {code_ts(start+23).date()}, continuous={len(continuous)}, eligible={len(eligible)}, selected={len(selected)}")
    dt,matrix,matrix_audit=build(files,start,selected,args.chunksize);audit={"schema":{"customer":"LCLid","datetime":"DateTime","load":"KWH/hh (per half hour)"},"partition_files":len(files),"source_rows":rows,"source_invalid_values":invalid,"source_datetime_min":str(mn),"source_datetime_max":str(mx),
        "first_complete_month":str(code_ts(first).date()),"last_complete_month":str(code_ts(last).date()),"chosen_start_month":str(code_ts(start).date()),"chosen_end_exclusive":str(code_ts(start+23).date()),"continuous_23m_users":len(continuous),"training_eligible_users":len(eligible),"selected_users":len(selected),"matrix_shape":list(matrix.shape),"matrix":matrix_audit,
        "selected_training":{"min_nonzero_fraction":min(detail[c]["train_nonzero_fraction"] for c in selected),"min_std":min(detail[c]["train_std"] for c in selected)},"wall_seconds":time.perf_counter()-started}
    np.save(args.out/"values.npy",matrix,allow_pickle=False);(args.out/"metadata.json").write_text(json.dumps({"start":str(dt[0]),"end":str(dt[-1]),"frequency":"30min","customers":selected},indent=2),encoding="utf-8");(args.out/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":main()
