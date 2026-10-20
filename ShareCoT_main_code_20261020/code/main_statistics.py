"""Main-comparison metrics, paired tests, and three-run aggregation.

Statistical functions are extracted unchanged from the formal experiment.
No predictions or measured results are embedded in this module.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def macro_f1_rows(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    scores = []
    for label in [0, 1]:
        tp = ((y == label) & (pred == label)).sum(axis=-1)
        fp = ((y != label) & (pred == label)).sum(axis=-1)
        fn = ((y == label) & (pred != label)).sum(axis=-1)
        denominator = 2 * tp + fp + fn
        scores.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
    return (scores[0] + scores[1]) / 2

def align_to_reference(reference, other):
    if reference.sample_id.duplicated().any() or other.sample_id.duplicated().any() or set(reference.sample_id) != set(other.sample_id):
        raise ValueError('Paired comparison requires identical unique sample IDs')
    aligned = other.set_index('sample_id').loc[reference.sample_id]
    if not np.array_equal(reference.y_true.to_numpy(), aligned.y_true.to_numpy()):
        raise ValueError('Paired labels differ')
    return aligned.y_pred.to_numpy(dtype=np.int8)

def paired_summary(y,a,b,resamples=5000,seed=44):
    y,a,b=[np.asarray(v,dtype=int) for v in (y,a,b)]
    if y.ndim!=1 or len(y)==0 or y.shape!=a.shape or y.shape!=b.shape or any(not np.isin(v,[0,1]).all() for v in (y,a,b)):
        raise ValueError('Paired labels/predictions must be aligned binary vectors')
    base=float(macro_f1_rows(y,a));score=float(macro_f1_rows(y,b));delta=base-score
    rng=np.random.RandomState(seed);ar=[];br=[]
    for start in range(0,resamples,100):
        indices=rng.randint(0,len(y),(min(100,resamples-start),len(y)))
        ar.extend(macro_f1_rows(y[indices],a[indices]));br.extend(macro_f1_rows(y[indices],b[indices]))
    rng=np.random.RandomState(seed);exceed=0
    for start in range(0,resamples,100):
        swap=rng.rand(min(100,resamples-start),len(y))<.5
        difference=macro_f1_rows(y,np.where(swap,b,a))-macro_f1_rows(y,np.where(swap,a,b))
        exceed+=int((np.abs(difference)>=abs(delta)-1e-12).sum())
    return {'sharecot_macro_f1':base,'method_macro_f1':score,'sharecot_minus_method':delta,
        'sharecot_macro_f1_95_ci':np.percentile(ar,[2.5,97.5]).tolist(),
        'method_macro_f1_95_ci':np.percentile(br,[2.5,97.5]).tolist(),
        'paired_difference_95_ci':np.percentile(np.asarray(ar)-br,[2.5,97.5]).tolist(),
        'paired_permutation_two_sided_p':(exceed+1)/(resamples+1)}

def read_predictions(path):
    frame = pd.read_csv(path, dtype={'sample_id': str})
    if not {'sample_id', 'y_true', 'y_pred'} <= set(frame):
        raise ValueError('Require sample_id, y_true, y_pred columns')
    if frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError('Require nonempty unique sample IDs')
    if not frame.y_true.isin([0,1]).all() or not frame.y_pred.isin([-1,0,1]).all():
        raise ValueError('Labels must be binary; -1 is allowed only for invalid API predictions')
    return frame

def point_metrics(frame):
    y, pred = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    result = {'accuracy':float(accuracy_score(y,pred)),
              'macro_precision':float(precision_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
              'macro_recall':float(recall_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
              'macro_f1':float(macro_f1_rows(y,pred))}
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pair=sub.add_parser('paired');pair.add_argument('--reference',required=True);pair.add_argument('--comparator',required=True)
    pair.add_argument('--resamples',type=int,default=5000);pair.add_argument('--seed',type=int,default=44)
    model=sub.add_parser('model');model.add_argument('--predictions',required=True)
    model.add_argument('--resamples',type=int,default=1000);model.add_argument('--seed',type=int,default=44)
    multi=sub.add_parser('multiseed');multi.add_argument('--predictions',nargs=3,required=True)
    args=parser.parse_args(argv)
    if hasattr(args,'resamples') and args.resamples<1:parser.error('resamples must be positive')
    if args.command=='paired':
        a=read_predictions(args.reference);b=read_predictions(args.comparator)
        result=paired_summary(a.y_true,a.y_pred,align_to_reference(a,b),args.resamples,args.seed)
    elif args.command=='model':
        a=read_predictions(args.predictions);result=point_metrics(a)
        rng=np.random.RandomState(args.seed);values=[];y=a.y_true.to_numpy();pred=a.y_pred.to_numpy()
        for start in range(0,args.resamples,100):
            ids=rng.randint(0,len(a),(min(100,args.resamples-start),len(a)))
            values.extend(macro_f1_rows(y[ids],pred[ids]))
        result['macro_f1_percentile_95_ci']=np.percentile(values,[2.5,97.5]).tolist()
        result['confusion_matrix_rows_true_columns_pred']=[[int(((y==i)&(pred==j)).sum()) for j in [0,1]] for i in [0,1]]
        result['invalid_predictions']=int((pred==-1).sum())
        result['n']=len(a)
    else:
        if len(set(str(Path(p).resolve()) for p in args.predictions))!=3:parser.error('Three distinct prediction files are required')
        frames=[read_predictions(p) for p in args.predictions]
        for f in frames[1:]:align_to_reference(frames[0],f)
        rows=[point_metrics(f) for f in frames]
        result={'runs':3,'sample_sd_ddof':1,'metrics':{k:{'mean':float(np.mean([r[k] for r in rows])), 'sample_sd':float(np.std([r[k] for r in rows],ddof=1))} for k in rows[0]}}
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
