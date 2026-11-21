from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import TARGET, write_json
from supplement_suite import perturb, noise_frames, align_to_reference, verify_noise_identity

NOISE_COLS=['fare','trip_miles','trip_seconds','additional_charges']
LEVELS=[.05,.10,.20]
REPEATS=5
METHODS=['adaptive','direct_ft','transformer','xgboost']
METRICS=['accuracy','macro_precision','macro_recall','macro_f1','roc_auc','negative_recall','positive_recall']

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def cached_probabilities(cached,frame,probability_col='prob'):
    reference=pd.DataFrame({'sample_id':frame.sample_id,'y_true':frame[TARGET]})
    align_to_reference(reference,cached)
    aligned=cached.set_index('sample_id').loc[frame.sample_id]
    prob=aligned[probability_col].to_numpy(float)
    if (not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any()
        or not np.array_equal(aligned.y_pred.to_numpy(),(prob>=.5).astype(int))):
        raise ValueError('Cached probability/classes invalid or differ')
    return prob

def load_inference(result,method):
    if method in ('adaptive','direct_ft','sharecot'):
        from evaluate_suite import load_qwen,classify
        model,tok,dev=load_qwen(result)
        return (model,tok),lambda frame: classify(model,tok,dev,frame,batch=8)
    from baselines_text_suite import load_model
    predictor=load_model(result,device='cpu',**({'batch_size':64} if method=='transformer' else {}))
    def inference(frame):
        if method=='transformer':
            return np.concatenate([predictor(frame.iloc[start:start+64]) for start in range(0,len(frame),64)])
        return np.asarray(predictor(frame))
    return predictor,inference

def validate_formal_sources(suite):
    suite=Path(suite); main=suite/'experiments/02_main'; registry=read_json(suite/'formal_dataset.json')
    if registry.get('dataset_role')!='formal_primary_dataset_user_adopted' or registry.get('historical_holdout_reuse') is not True:
        raise ValueError('Formal adoption registry required')
    for method in METHODS:
        result=main/'results'/method
        config=read_json(result/'run_config.json'); metrics=read_json(result/'metrics.json')
        if method in ('adaptive','direct_ft'):
            if method=='adaptive' and result.resolve()!=(main/'results/sharecot').resolve(): raise ValueError('Adaptive alias is not formal ShareCoT')
            if metrics.get('trained_steps')!=5000 or config.get('seed')!=44: raise ValueError('Formal Qwen full seed44 run required')
        else:
            if metrics.get('status')!='complete' or config.get('seed')!=44: raise ValueError('Formal completed baseline seed44 run required')
    return registry

def full_metrics(y,prob):
    from baselines_suite import compute_metrics
    values=compute_metrics(y,prob)
    return {key:values[key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')} | {
        'roc_auc':values['auc_roc'],'negative_recall':values['per_class']['0']['recall'],
        'positive_recall':values['per_class']['1']['recall']}

def persist_input(path,frame):
    content=frame.to_csv(index=False,float_format='%.17g').encode('utf-8')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content: raise ValueError('Persisted noisy input changed: '+str(path))
    else:
        temp=path.with_suffix('.csv.tmp'); temp.write_bytes(content); temp.replace(path)

def noise(suite):
    import torch
    suite=Path(suite).resolve(); main=suite/'experiments/02_main'
    registry=validate_formal_sources(suite)
    numeric,raw=noise_frames(main)
    if len(numeric)!=1000: raise ValueError('Formal clean test must contain exactly 1000 rows')
    out=suite/'experiments/13_estimation_noise'; out.mkdir(parents=True,exist_ok=True)
    config={'schema':'formal_reviewer_relative_error_sd_v1',
        'formal_dataset_id':registry['formal_dataset_id'],
        'historical_holdout_reuse':True,
        'methods':METHODS,'columns':NOISE_COLS,'levels':LEVELS,'repeats':REPEATS,'noise_seed_base':4400,
        'training_seed':44,'threshold':.5,
        'batch_sizes':{'adaptive':8,'direct_ft':8,'transformer':64,'xgboost':'full frame'},
        'input_serialization':'Qwen: original pandas numeric parser; Transformer: original lexical CSV and units; XGBoost: original numeric preprocessor. Perturbed floating-point quantities use full precision, which can change numeric tokenization; sensitivity combines numeric error and its text representation.',
        'distribution':'Independent multiplicative Gaussian relative error, clipped at zero, seeds4400..4404 reused for every SD and method; zero source remains zero.',
        'interpretation':'Assumed relative-error SD sensitivity scenarios, not calibrated booking estimates or external deployment validation; fixed selected classifiers only, no SHAP updates or retraining.'}
    verify_noise_identity(out,config)
    prediction_dir=out/'predictions'; prediction_dir.mkdir(exist_ok=True)
    rows=[]; scenarios=[(0.,0)]+[(sd,repeat) for sd in LEVELS for repeat in range(REPEATS)]
    for method in METHODS:
        frame=raw if method=='transformer' else numeric
        result=main/'results'/method
        saved=pd.read_csv(result/'predictions.csv',dtype={'sample_id':str})
        col='prob' if method in ('adaptive','direct_ft') else 'prob_1'
        expected=cached_probabilities(saved,frame,col)
        model,inference=load_inference(result,method)
        clean=np.asarray(inference(frame),dtype=float)
        if clean.shape!=(1000,) or not np.allclose(clean,expected,atol=1e-5,rtol=0) or not np.array_equal(clean>=.5,expected>=.5):
            raise ValueError('Full1000 selected model clean inference drift: '+method)
        for sd,repeat in scenarios:
            name=f'level_{sd:.2f}_repeat_{repeat}'
            altered=perturb(frame,sd,4400+repeat)
            common=out/'inputs'/(name+'.csv')
            persist_input(common,perturb(raw,sd,4400+repeat))
            native_inputs=out/'native_inputs'/(method+'_'+name+'.csv')
            persist_input(native_inputs,altered)
            path=prediction_dir/(method+'_'+name+'.csv')
            if path.exists():
                prob=cached_probabilities(pd.read_csv(path,dtype={'sample_id':str}),frame)
            else:
                prob=clean if sd==0 else np.asarray(inference(altered),dtype=float)
                if prob.shape!=(1000,) or not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any(): raise ValueError('Invalid noisy probabilities')
                temp=path.with_suffix('.csv.tmp')
                pd.DataFrame({'sample_id':frame.sample_id,'y_true':frame[TARGET],'y_pred':(prob>=.5).astype(int),'prob':prob}).to_csv(temp,index=False,float_format='%.17g')
                temp.replace(path)
            if sd==0 and (not np.allclose(prob,clean,atol=1e-5,rtol=0) or not np.array_equal(prob>=.5,clean>=.5)): raise ValueError('Clean cached condition differs from selected model')
            row={'method':method,'relative_error_sd':sd,'repeat':repeat,'noise_seed':4400+repeat,
                **full_metrics(frame[TARGET],prob)}
            rows.append(row); write_json(out/'progress.json',{'processed_conditions':len(rows),'total_conditions':64,'method':method,'level':sd})
            print(method,name,row['macro_f1'],flush=True)
        del model,inference; gc.collect(); torch.cuda.empty_cache()
    table=pd.DataFrame(rows); table.to_csv(out/'conditions.csv',index=False)
    groups=[]
    for (method,sd),group in table.groupby(['method','relative_error_sd']):
        row={'method':method,'relative_error_sd':float(sd),'repeats':len(group)}
        for key in METRICS:
            values=group[key]
            row[key+'_mean']=float(values.mean()); row[key+'_sample_sd']=float(values.std(ddof=1)) if len(group)>1 else None
        groups.append(row)
    pd.DataFrame(groups).to_csv(out/'sensitivity.csv',index=False)
    write_json(out/'summary.json',{'status':'complete','config':config,'conditions':rows,'sensitivity':groups,'complete':True,
        'repeat_variation':'Five independent assumed input-error realizations; sample SD uses ddof1, clean SD not applicable (JSON null). Not training-seed uncertainty.'})
    return rows

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--suite',required=True,type=Path)
    args=parser.parse_args(argv); noise(args.suite)

if __name__=='__main__': main()
