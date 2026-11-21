import numpy as np
import pandas as pd
from scipy.optimize import nnls

def fit_fare(train):
    values=train[['trip_seconds','trip_miles','fare']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    total=len(values)
    valid=np.isfinite(values).all(axis=1)&(values>=0).all(axis=1)
    values=values[valid]
    if len(values)<3:raise ValueError('At least three complete nonnegative training records required')
    t,d,y=values.T
    design=np.column_stack([np.ones(len(y)),t/60,d])
    beta,_=nnls(design,y)
    residual=y-design@beta
    return {'intercept_usd':float(beta[0]),'per_minute_usd':float(beta[1]),'per_mile_usd':float(beta[2]),'train_n':len(y),'train_rows_total':total,'train_rows_excluded':total-len(y),'train_mae_usd':float(np.abs(residual).mean()),'train_r2':float(1-residual@residual/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else None}

def draw_noise(n,seed):return np.random.RandomState(seed).normal(size=n)

def alter(frame,arm,sd,z,coef,fare_view):
    if arm not in ['time','distance'] or fare_view not in ['cents','public_2p5'] or sd<0:raise ValueError('Invalid coupled scenario')
    z=np.asarray(z,dtype=float)
    if z.shape!=(len(frame),) or not np.isfinite(z).all():raise ValueError('Invalid error vector')
    col='trip_seconds' if arm=='time' else 'trip_miles'
    source=frame[col].to_numpy(float);fare=frame.fare.to_numpy(float)
    if not np.isfinite(source).all() or (source<0).any() or not np.isfinite(fare).all() or (fare<0).any():raise ValueError('Invalid source quantities')
    changed=np.maximum(0,source*(1+sd*z))
    changed=np.rint(changed) if arm=='time' else np.round(changed,2)
    slope=coef['per_minute_usd']/60 if arm=='time' else coef['per_mile_usd']
    if slope<0:raise ValueError('Negative fare slope')
    continuous=np.maximum(0,fare+slope*(changed-source))
    cost=np.round(continuous,2) if fare_view=='cents' else 2.5*np.rint(continuous/2.5)
    out=frame.copy();out[col]=changed;out['fare']=cost
    return out

def raw_view(original_numeric,original_raw,changed):
    out=original_raw.copy()
    for col in ['fare','trip_seconds','trip_miles']:
        mask=changed[col].to_numpy(float)!=original_numeric[col].to_numpy(float)
        out.loc[mask,col]=changed.loc[mask,col].map(lambda v:str(float(v)))
    return out
