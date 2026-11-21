import argparse,concurrent.futures,datetime,fcntl,gc,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import TARGET,write_json
from supplement_suite import noise_frames
from reviewer_noise import full_metrics,METRICS
from coupled_core import fit_fare,draw_noise,alter,raw_view

METHODS=['logistic_regression','random_forest','xgboost','logit_boost','gradient_boosting','adaboost','transformer','textcnn','bert','adaptive']
SMALL=[m for m in METHODS if m!='adaptive']
SCENARIOS=[{'name':'original','arm':'original','fare_view':'original','sd':0.,'repeat':0}]
for arm in ['time','distance']:
    for view in ['cents','public_2p5']:
        for sd,rep in [(0.,0)]+[(sd,rep) for sd in [.05,.10,.20] for rep in range(5)]:
            SCENARIOS.append({'name':f'{arm}_{view}_sd{sd:.2f}_rep{rep}','arm':arm,'fare_view':view,'sd':sd,'repeat':rep})
PY = sys.executable

def read(p):return json.loads(Path(p).read_text())
def frame_bytes(frame):return frame.to_csv(index=False,float_format='%.17g').encode()
def save_frame(path,frame):
    data=frame_bytes(frame);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Persisted frame differs '+str(path))
    else:path.write_bytes(data)
def validate_prediction(frame,pred):
    if len(pred)!=len(frame) or pred.sample_id.duplicated().any() or pred.sample_id.tolist()!=frame.sample_id.tolist():raise ValueError('Prediction membership/order differs')
    if not np.array_equal(pred.y_true,frame[TARGET]):raise ValueError('Prediction labels differ')
    p=pred['prob' if 'prob' in pred else 'prob_1'].to_numpy(float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.array_equal(pred.y_pred,(p>=.5).astype(int)):raise ValueError('Invalid probability/class predictions')
    return p
def scenario_frames(numeric,raw,scenario,coef):
    if scenario['arm']=='original':return numeric.copy(),raw.copy()
    changed=alter(numeric,scenario['arm'],scenario['sd'],draw_noise(len(numeric),4400+scenario['repeat']),coef,scenario['fare_view'])
    unchanged=set(numeric.columns)-{'fare','trip_seconds' if scenario['arm']=='time' else 'trip_miles'}
    pd.testing.assert_frame_equal(changed[sorted(unchanged)],numeric[sorted(unchanged)])
    return changed,raw_view(numeric,raw,changed)

def prepare():
    main=SUITE/'experiments/02_main';train_path=main/'data/train.csv'
    train=pd.read_csv(train_path,usecols=['trip_seconds','trip_miles','fare'])
    coef=fit_fare(train);coef.update(fit_split='train_only',fit_columns=['trip_seconds','trip_miles','fare'],model='nonnegative least squares with nonnegative intercept and slopes',interpretation='Descriptive historical association, not a platform tariff; original fare residual retained by anchored increments')
    target=ROOT/'fare_fit.json'
    if target.exists():assert read(target)==coef
    else:write_json(target,coef)
    numeric,raw=noise_frames(main)
    assert len(numeric)==1000
    sources={}
    for m in METHODS:
        p = ADABOOST_RESULT if m == 'adaboost' and ADABOOST_RESULT is not None else main / 'results' / m
        cfg=read(p/'run_config.json')
        assert cfg['seed']==44,m
        validate_prediction(numeric,pd.read_csv(p/'predictions.csv',dtype={'sample_id':str}))
        sources[m]={'path':str(p)}
    stats=[]
    for sc in SCENARIOS:
        num,lex=scenario_frames(numeric,raw,sc,coef)
        save_frame(ROOT/'inputs/numeric'/f"{sc['name']}.csv",num)
        save_frame(ROOT/'inputs/raw'/f"{sc['name']}.csv",lex)
        values={**sc,'fare_changed_fraction':float((num.fare!=numeric.fare).mean()),'fare_mean_abs_change_usd':float(np.abs(num.fare-numeric.fare).mean()),'fare_max_abs_change_usd':float(np.abs(num.fare-numeric.fare).max())}
        for col in ['trip_seconds','trip_miles']:
            x=numeric[col].to_numpy(float);delta=num[col].to_numpy(float)-x;valid=x>0
            values[col+'_mean_abs_relative_error']=float(np.mean(np.abs(delta[valid]/x[valid])));values[col+'_max_abs_relative_error']=float(np.max(np.abs(delta[valid]/x[valid])))
        stats.append(values)
    pd.DataFrame(stats).to_csv(ROOT/'input_diagnostics.csv',index=False)
    cfg={'schema':'one_at_a_time_coupled_fare_v1','methods':METHODS,'excluded':['direct_ft','single_cot','api_baselines'],'sources':sources,'scenarios':SCENARIOS,'noise_seed_base':4400,'paired_standard_normal_draws':'same 1000 draws for both arms and fare views at each repeat, scaled by each SD','duration_precision_seconds':1,'altered_distance_precision_miles':.01,'fare_views':['nearest USD0.01','nearest USD2.50 public-data representation'],'rounding':'numpy nearest, ties to even','training_seed':44,'classification_threshold':.5,'original_label_policy':'fixed observed labels under hypothetical input-estimation error; not a behavioral intervention','coefficients_fit_without_labels':True,'model_selection':'existing main checkpoints; no retraining or retuning','precision_control':'Each arm/view has a zero-SD precision-processing control, in addition to original unmodified inputs. Quantization is applied before cost increments; price is anchored at each original recorded fare.','limitations':'No observed booking estimates; no calibrated real error distribution; no true platform tariff; upfront-price products can behave differently; rounding can suppress price changes, reported in diagnostics.'}
    path=ROOT/'frozen_protocol.json'
    if path.exists():assert read(path)==cfg
    else:write_json(path,cfg)
    return cfg

def update(method,stage,**kw):
    write_json(ROOT/'results'/method/'progress.json',{'method':method,'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})

def gpu_for_share():
    state = read(DHRD_STATUS) if DHRD_STATUS is not None else {'stage': 'complete', 'seeds': {}}
    seeds=list(state.get('seeds',{}).values())
    pending=any(x.get('stage')=='queued' for x in seeds)
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    for line in lines:
        g,free,uuid=[x.strip() for x in line.split(',')]
        assigned=[x for x in seeds if str(x.get('gpu'))==g]
        safe=(state['stage']=='complete' or (not pending and all(x.get('stage') in ['complete','failed'] for x in assigned)))
        if safe and uuid not in apps and float(free)>20000:return g
    return None

def worker(method):
    import torch
    torch.set_num_threads(8)
    cfg=read(ROOT/'frozen_protocol.json');source=cfg['sources'][method]
    out=ROOT/'results'/method;out.mkdir(parents=True,exist_ok=True);coef=read(ROOT/'fare_fit.json')
    device='cpu'
    if method=='adaptive':
        while True:
            gpu=gpu_for_share()
            if gpu is not None:device='cuda:'+gpu;break
            update(method,'queued_for_idle_gpu',reason='Waiting for GPU free of DHRD jobs, including its queued seed50',pid=os.getpid());time.sleep(30)
    elif method=='textcnn':device='cuda:0'
    update(method,'loading',device=device,pid=os.getpid())
    p=Path(source['path'])
    if method=='adaptive':
        from evaluate_suite import load_qwen,classify
        model,tok,dev=load_qwen(p,device=device);inference=lambda f:classify(model,tok,dev,f,batch=8)
    elif method in ['textcnn','transformer','bert']:
        from baselines_text_suite import load_model
        if method=='textcnn':torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
        inference=load_model(p,device=device)
    else:
        import joblib
        from baselines_suite import prepare_features
        prep=joblib.load(p/'preprocessor.joblib');model=joblib.load(p/'model.joblib')
        if hasattr(model,'n_jobs'):model.n_jobs=4
        inference=lambda f:model.predict_proba(prep.transform(prepare_features(f)))[:,1]
    numeric,raw=noise_frames(SUITE/'experiments/02_main');is_text=method in ['textcnn','transformer','bert'];frame=raw if is_text else numeric
    expected=validate_prediction(frame,pd.read_csv(p/'predictions.csv',dtype={'sample_id':str}))
    update(method,'checking_original_predictions',device=device)
    clean=np.asarray(inference(frame),dtype=float)
    if not np.allclose(clean,expected,atol=1e-5,rtol=0) or not np.array_equal(clean>=.5,expected>=.5):raise ValueError('Original clean predictions do not reproduce: '+method)
    write_json(out/'original_reload_verification.json',{'status':'verified','n':1000,'max_probability_error':float(np.abs(clean-expected).max()),'identical_classes':True,'device':device})
    cache={};rows=[];start=time.time()
    for sc in SCENARIOS:
        num,lex=scenario_frames(numeric,raw,sc,coef);f=lex if is_text else num
        input_bytes = frame_bytes(f)
        kind = 'raw' if is_text else 'numeric'
        if (ROOT/'inputs'/kind/f"{sc['name']}.csv").read_bytes() != input_bytes:
            raise ValueError('Prepared scenario input differs')
        path=out/'predictions'/f"{sc['name']}.csv"
        if path.exists():
            prob=validate_prediction(frame,pd.read_csv(path,dtype={'sample_id':str}))
        else:
            prob=clean if sc['name']=='original' else cache[input_bytes] if input_bytes in cache else np.asarray(inference(f),dtype=float)
            pred=pd.DataFrame({'sample_id':frame.sample_id,'y_true':frame[TARGET],'y_pred':(prob>=.5).astype(int),'prob':prob});validate_prediction(frame,pred)
            path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.csv.tmp');pred.to_csv(temp,index=False,float_format='%.17g');temp.replace(path)
        cache[input_bytes]=prob
        row={'method':method,**sc,**full_metrics(frame[TARGET],prob),'device':device};rows.append(row)
        pd.DataFrame(rows).to_csv(out/'conditions.csv',index=False)
        update(method,'evaluating',processed_conditions=len(rows),total_conditions=len(SCENARIOS),last=sc['name'],macro_f1=row['macro_f1'],device=device,elapsed_seconds=time.time()-start)
        print(method,sc['name'],row['macro_f1'],flush=True)
    for row in rows:
        prob=validate_prediction(frame,pd.read_csv(out/'predictions'/f"{row['name']}.csv",dtype={'sample_id':str}))
        for key,value in full_metrics(frame[TARGET],prob).items():assert abs(value-row[key])<1e-12
    assert len(rows)==65 and len({x['name'] for x in rows})==65
    write_json(out/'verification.json',{'status':'verified','conditions':65,'n':1000,'all_metrics_recomputed':True,'device':device})
    update(method,'complete',processed_conditions=65,total_conditions=65,device=device)

def aggregate():
    rows=[];complete=[]
    for m in METHODS:
        p=ROOT/'results'/m/'verification.json'
        if p.exists() and read(p)['status']=='verified':
            part=pd.read_csv(p.parent/'conditions.csv');assert len(part)==65;rows.extend(part.to_dict('records'));complete.append(m)
    if not rows:return
    table=pd.DataFrame(rows);table.to_csv(ROOT/'conditions.csv',index=False);groups=[]
    for (m,arm,view,sd),part in table.groupby(['method','arm','fare_view','sd']):
        row={'method':m,'arm':arm,'fare_view':view,'relative_error_sd':float(sd),'repeats':len(part)}
        assert set(part['repeat'])==({0} if sd==0 else set(range(5)))
        for key in METRICS:row[key+'_mean']=float(part[key].mean());row[key+'_sample_sd']=float(part[key].std(ddof=1)) if len(part)>1 else None
        groups.append(row)
    pd.DataFrame(groups).to_csv(ROOT/'sensitivity.csv',index=False)
    write_json(ROOT/'summary.json',{'status':'complete' if len(complete)==10 else 'partial','completed_models':complete,'pending_models':[m for m in METHODS if m not in complete],'conditions':len(rows),'sensitivity':groups,'uncertainty':'Sample SD across five input-error draws, not training seeds'})

def controller():
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare();write_json(ROOT/'controller_status.json',{'stage':'running','pid':os.getpid(),'methods':METHODS,'conditions_per_model':65})
    def job(m):
        if (ROOT/'results'/m/'verification.json').exists():return m,0
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        if m not in ['textcnn','adaptive']:env['CUDA_VISIBLE_DEVICES']=''
        (ROOT/'logs').mkdir(exist_ok=True)
        command = [PY, '-u', str(Path(__file__)), '--suite', str(SUITE),
                   '--output-dir', str(ROOT), '--method', m]
        if DHRD_STATUS is not None:
            command.extend(['--dhrd-status', str(DHRD_STATUS)])
        with (ROOT / 'logs' / f'{m}.log').open('a') as log:
            rc = subprocess.call(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
        if rc:update(m,'failed',returncode=rc)
        return m,rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outer:
        share=outer.submit(job,'adaptive')
        order=['bert','textcnn','transformer']+[m for m in SMALL if m not in ['bert','textcnn','transformer']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(job,m) for m in order]
            results=[]
            for f in concurrent.futures.as_completed(futures):results.append(f.result());aggregate()
        results.append(share.result());aggregate()
    failed=[m for m,rc in results if rc]
    if failed:raise RuntimeError('Failed models: '+str(failed))
    assert read(ROOT/'summary.json')['status']=='complete'
    write_json(ROOT/'controller_status.json',{'stage':'complete','models':10,'conditions':650,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--adaboost-result', type=Path)
    p.add_argument('--dhrd-status', type=Path)
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    SUITE = a.suite.resolve()
    ROOT = a.output_dir.resolve()
    ADABOOST_RESULT = a.adaboost_result.resolve() if a.adaboost_result is not None else None
    DHRD_STATUS = a.dhrd_status.resolve() if a.dhrd_status is not None else None
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if a.method:worker(a.method)
        elif a.prepare_only:prepare()
        else:controller()
    except Exception as exc:
        if a.method:update(a.method,'failed',error=str(exc))
        else:write_json(ROOT/'controller_status.json',{'stage':'failed','error':str(exc)})
        raise
