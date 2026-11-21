import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from protocol import FEATURE_COLS,TARGET,feature_text,build_prompt,rationale_is_clean,write_json

def flatten_metrics(payload):
    metric=payload.get('test',payload);result={k:metric.get(k) for k in ['accuracy','macro_precision','macro_recall','macro_f1','positive_precision','positive_recall','positive_f1','negative_precision','negative_recall','negative_f1']}
    result['roc_auc']=metric.get('roc_auc',metric.get('auc_roc'))
    for cls,prefix in [('0','negative'),('1','positive')]:
        for key in ['precision','recall','f1']:
            per=metric.get('per_class',{}).get(cls,{})
            if key in per:result[prefix+'_'+key]=per[key]
            elif key=='f1' and 'f1-score' in per:result[prefix+'_'+key]=per['f1-score']
    return result

def align_predictions(a,b):
    if a.sample_id.duplicated().any() or b.sample_id.duplicated().any() or set(a.sample_id)!=set(b.sample_id):raise ValueError('Paired evaluation requires the same unique sample IDs')
    other=b.set_index('sample_id').loc[a.sample_id]
    if not np.array_equal(a.y_true.to_numpy(),other.y_true.to_numpy()):raise ValueError('Paired labels differ')
    return a.y_true.to_numpy(),a.y_pred.to_numpy(),other.y_pred.to_numpy()

def read_metrics(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def selected_adapter(result):
    result=Path(result)
    return Path(read_metrics(result/'reuse.json')['model']) if (result/'reuse.json').exists() else result/'model'

def load_qwen(result,device='cuda:0',classification=True):
    import torch
    from torch import nn
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    config=read_metrics(Path(result)/'run_config.json');base_path=config.get('model_dir',config.get('base_model_dir'))
    adapter=selected_adapter(result)
    model=AutoModelForCausalLM.from_pretrained(base_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
    if classification:model.classification_head=nn.Linear(model.config.hidden_size,1,bias=False)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True);tokenizer.padding_side='right'
    return model,tokenizer,device

def classify(model,tokenizer,device,frame,batch=8):
    import torch
    probs=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(frame),batch):
            prompts=[build_prompt(dict(row),tokenizer) for _,row in frame.iloc[start:start+batch].iterrows()]
            inputs=tokenizer(prompts,padding=True,truncation=True,max_length=512,return_tensors='pt',add_special_tokens=False);inputs={k:v.to(device) for k,v in inputs.items()}
            base=model.base_model.model
            states=base.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
            indices=inputs['attention_mask'].sum(-1)-1
            logits=base.classification_head(states[torch.arange(len(prompts),device=device),indices]).squeeze(-1)
            probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return np.asarray(probs)

def generate_reason(model,tokenizer,device,row,single_cot=False):
    import torch
    if single_cot:
        from train_singlecot import structured_prompt
        prompt=structured_prompt(row,tokenizer)
    else:prompt=build_prompt(dict(row),tokenizer)
    inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False);inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():ids=model.generate(**inputs,do_sample=False,max_new_tokens=256,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
    return tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)

def verify_main_inference(model,tokenizer,device,suite):
    main=Path(suite)/'experiments/02_main';test=pd.read_csv(main/'data/test.csv').iloc[:32];saved=pd.read_csv(main/'results/adaptive/predictions.csv').set_index('sample_id');prob=classify(model,tokenizer,device,test)
    if not np.allclose(prob,saved.loc[test.sample_id,'prob'].to_numpy(),atol=1e-5):raise ValueError('Reloaded adaptive inference differs from selected main checkpoint')

def main_result_dirs(suite):
    root=Path(suite)/'experiments/02_main/results'
    return [(name,root/name) for name in ['adaptive','direct_ft','single_cot'] if (root/name/'metrics.json').exists()]

def neural_baseline_predictor(result,name,device='cuda:0'):
    if name not in ('textcnn', 'transformer', 'bert'):
        raise ValueError('Unknown text neural baseline: ' + name)
    from baselines_text_suite import load_model
    predictor=load_model(result,device=device)
    return predictor.model,predictor


def efficiency(suite):
    import torch,gc
    suite=Path(suite);out=suite/'experiments/03_efficiency';out.mkdir(parents=True,exist_ok=True);main=suite/'experiments/02_main';sample=pd.read_csv(main/'data/test.csv').iloc[:10];rows=[]
    required=['adaptive','direct_ft','single_cot','logistic_regression','random_forest','xgboost','logit_boost','gradient_boosting','textcnn','transformer','bert']
    missing=[name for name in required if not (main/'results'/name/'metrics.json').is_file()]
    if missing:raise ValueError('Efficiency comparison missing completed main methods: '+str(missing))
    for name,result in main_result_dirs(suite):
        model,tok,dev=load_qwen(result,classification=name!='single_cot');torch.cuda.reset_peak_memory_stats()
        if name=='adaptive':verify_main_inference(model,tok,dev,suite)
        for _,row in sample.iloc[:2].iterrows():
            if name=='single_cot':generate_reason(model,tok,dev,row,single_cot=True)
            else:classify(model,tok,dev,pd.DataFrame([row]),batch=1)
        times=[]
        for repeat in range(3):
            for _,row in sample.iterrows():
                torch.cuda.synchronize();start=time.perf_counter()
                if name=='single_cot':generate_reason(model,tok,dev,row,single_cot=True)
                else:classify(model,tok,dev,pd.DataFrame([row]),batch=1)
                torch.cuda.synchronize();times.append(time.perf_counter()-start)
        entry={'method':name,'mode':'autoregressive JSON generation' if name=='single_cot' else 'classification head, no generation','batch_size':1,'repetitions':3,'sample_ids':sample.sample_id.tolist(),'seconds':times,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),'mean_seconds':float(np.mean(times)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'includes':'row prompt formatting, tokenization, H2D transfer, forward/generation and sigmoid/decoding; model load excluded','hardware':torch.cuda.get_device_name(),'max_new_tokens':256 if name=='single_cot' else 0,'warmup_queries':2}
        write_json(out/(name+'.json'),entry);rows.append(entry)
        if name=='adaptive':
            gen_times=[]
            for _,row in sample.iterrows():
                torch.cuda.synchronize();start=time.perf_counter();generate_reason(model,tok,dev,row);torch.cuda.synchronize();gen_times.append(time.perf_counter()-start)
            write_json(out/'adaptive_reason_generation.json',{'seconds':gen_times,'sample_ids':sample.sample_id.tolist(),'max_new_tokens':256,'batch_size':1,'median_seconds':float(np.median(gen_times))})
        del model;gc.collect();torch.cuda.empty_cache()
    for name in ['textcnn','transformer','bert']:
        sample=pd.read_csv(main/'data/test.csv',dtype=str,keep_default_na=False).iloc[:10]
        result=main/'results'/name;net,predict=neural_baseline_predictor(result,name);torch.cuda.reset_peak_memory_stats()
        saved=pd.read_csv(result/'predictions.csv').set_index('sample_id')
        if not np.allclose(predict(sample),saved.loc[sample.sample_id,'prob_1'],atol=1e-5):raise ValueError('Neural baseline reload predictions differ: '+name)
        for _,row in sample.iloc[:2].iterrows():predict(pd.DataFrame([row]))
        times=[]
        for _ in range(3):
            for _,row in sample.iterrows():
                torch.cuda.synchronize();start=time.perf_counter();predict(pd.DataFrame([row]));torch.cuda.synchronize();times.append(time.perf_counter()-start)
        entry={'method':name,'device':'same GPU as Qwen methods','seconds':times,'batch_size':1,'warmup_queries':2,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'includes':'row feature conversion/preprocessing or BERT tokenization, transfer, model inference and probability; model load excluded'};write_json(out/(name+'.json'),entry);rows.append(entry)
        del net,predict;gc.collect();torch.cuda.empty_cache()
    import joblib
    from baselines_suite import prepare_features
    for name in ['logistic_regression','random_forest','xgboost','logit_boost','gradient_boosting']:
        result=main/'results'/name
        if not (result/'model.joblib').exists():continue
        model=joblib.load(result/'model.joblib');prep=joblib.load(result/'preprocessor.joblib');times=[]
        for _ in range(3):
            for _,row in sample.iterrows():
                start=time.perf_counter();model.predict_proba(prep.transform(prepare_features(pd.DataFrame([row]))));times.append(time.perf_counter()-start)
        entry={'method':name,'device':'cpu','seconds':times,'batch_size':1,'median_seconds':float(np.median(times)),'includes':'DataFrame feature conversion, preprocessing and predict_proba; model load excluded'};write_json(out/(name+'.json'),entry);rows.append(entry)
    write_json(out/'summary.json',{'results':rows,'training_api_cost':'provider billing required; not estimated as zero','timing_is_observed':True})

def architecture(suite):
    suite=Path(suite);out=suite/'experiments/09_architecture';out.mkdir(parents=True,exist_ok=True)
    results={name:flatten_metrics(read_metrics(root/'metrics.json')) for name,root in main_result_dirs(suite)}
    if set(results)!={'adaptive','direct_ft','single_cot'}:raise ValueError('Architecture comparison missing method')
    write_json(out/'summary.json',{'seed':44,'results':results,'models_reused':True,'reasoning_weighting':'adaptive'})

def cases(suite):
    import torch,gc
    suite=Path(suite);main=suite/'experiments/02_main';out=suite/'experiments/11_cases';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(main/'data/test.csv');pred=pd.read_csv(main/'results/adaptive/predictions.csv');membership=frame.merge(pred[['sample_id','y_pred']],on='sample_id',validate='one_to_one')
    parts=[];empty=[]
    for truth,prediction,quadrant in [(1,1,'TP'),(0,0,'TN'),(1,0,'FN'),(0,1,'FP')]:
        available=membership[(membership[TARGET]==truth)&(membership.y_pred==prediction)]
        if not len(available):empty.append(quadrant);continue
        parts.append(available.sample(n=min(25,len(available)),random_state=44))
    selected=pd.concat(parts)
    remaining=membership[~membership.sample_id.isin(selected.sample_id)]
    if len(selected)<100:selected=pd.concat([selected,remaining.sample(n=min(100-len(selected),len(remaining)),random_state=44)])
    model,tok,dev=load_qwen(main/'results/adaptive');verify_main_inference(model,tok,dev,suite);records=[]
    cache=out/'cache';cache.mkdir(exist_ok=True);mapping=pred.set_index('sample_id')
    for _,row in selected.iterrows():
        path=cache/(str(row.sample_id).replace(':','_')+'.json')
        if path.exists():record=read_metrics(path)
        else:
            raw=generate_reason(model,tok,dev,row);prediction=int(mapping.loc[row.sample_id,'y_pred']);truth=int(row[TARGET])
            record={'sample_id':row.sample_id,'observed_label':truth,'classification_head_pred':prediction,'classification_probability':float(mapping.loc[row.sample_id,'prob']),'confusion_quadrant':('TP' if truth else 'TN') if truth==prediction else ('FN' if truth else 'FP'),'raw_student_reason':raw,'no_forbidden_content':rationale_is_clean(raw),'features':{key:str(row[key]) for key in FEATURE_COLS},'human_reviewed':False,'rationale_vs_classification_agreement':'requires manual labeling; no final generated class is forced'};write_json(path,record)
        records.append(record)
    pd.DataFrame(records).to_csv(out/'student_rationales.csv',index=False)
    pd.DataFrame([{**r,'reviewer1_fact_score':'','reviewer2_fact_score':'','reviewer1_attribution_score':'','reviewer2_attribution_score':'','reviewer1_label_consistency':'','reviewer2_label_consistency':''} for r in records]).to_csv(out/'human_annotation_sheet.csv',index=False)
    write_json(out/'summary.json',{'seed':44,'n':len(records),'sampling':'up to25 held-out instances from each available confusion quadrant, then fixed random fill to100; error-enriched descriptive audit, not a prevalence estimate','empty_quadrants':empty,'forbidden_content_count':sum(not r['no_forbidden_content'] for r in records),'human_reviewed':0,'case_quadrants':{q:sum(r['confusion_quadrant']==q for r in records) for q in ['TP','TN','FP','FN']},'human_agreement':'pending actual independent annotation; not fabricated'})
    del model;gc.collect();torch.cuda.empty_cache()

def review_controls(suite):
    import torch,gc
    suite=Path(suite);out=suite/'experiments/12_review_controls';out.mkdir(parents=True,exist_ok=True);main=suite/'experiments/02_main';a=pd.read_csv(main/'results/adaptive/predictions.csv');comparisons=[]
    for name,path in [('direct_ft',main/'results/direct_ft/predictions.csv'),('shuffled_reason',suite/'experiments/12_shuffle/results/sharecot/predictions.csv')]:
        b=pd.read_csv(path);y,pa,pb=align_predictions(a,b);rng=np.random.RandomState(44);differences=[]
        for _ in range(5000):
            ids=rng.randint(0,len(y),len(y));differences.append(f1_score(y[ids],pa[ids],labels=[0,1],average='macro',zero_division=0)-f1_score(y[ids],pb[ids],labels=[0,1],average='macro',zero_division=0))
        observed=f1_score(y,pa,average='macro')-f1_score(y,pb,average='macro');permuted=[]
        for _ in range(5000):
            swap=rng.rand(len(y))<.5;qa=np.where(swap,pb,pa);qb=np.where(swap,pa,pb);permuted.append(f1_score(y,qa,labels=[0,1],average='macro',zero_division=0)-f1_score(y,qb,labels=[0,1],average='macro',zero_division=0))
        comparisons.append({'reference':'adaptive','control':name,'macro_f1_difference':float(observed),'paired_bootstrap_95_ci':np.percentile(differences,[2.5,97.5]).tolist(),'paired_permutation_two_sided_p':float((1+np.sum(np.abs(permuted)>=abs(observed)))/(1+len(permuted))),'n':len(y),'resamples':5000,'training_seed':44,'interpretation':'conditional fixed-test uncertainty for one selected training seed; exploratory, not training-seed uncertainty; no multiplicity adjustment'})
    frame=pd.read_csv(main/'data/test.csv').sample(n=100,random_state=44);train=pd.read_csv(main/'data/train.csv');model,tok,dev=load_qwen(main/'results/adaptive');verify_main_inference(model,tok,dev,suite);original=classify(model,tok,dev,frame);sensitivity=[]
    for key in FEATURE_COLS:
        changed=frame.copy()
        if key=='trip_start_timestamp':value=train[key].mode().iloc[0]
        elif key=='dropoff_poi_category':value=train[key].mode().iloc[0]
        else:value=float(pd.to_numeric(train[key]).median())
        changed[key]=value;new=classify(model,tok,dev,changed)
        sensitivity.extend({'sample_id':sid,'feature':key,'replacement':str(value),'original_probability':float(before),'replacement_probability':float(after),'probability_difference':float(after-before)} for sid,before,after in zip(frame.sample_id,original,new))
    pd.DataFrame(sensitivity).to_csv(out/'feature_replacement_sensitivity.csv',index=False)
    write_json(out/'summary.json',{'status':'complete','paired_comparisons':comparisons,'intervention':'replace one feature at a time with training median/mode; an input-sensitivity diagnostic, not proof of causal or rationale faithfulness','n_intervention':len(frame),'booking_time_validity':'not established by removing tips alone','human_rationale_agreement':'requires actual annotation of 11_cases/human_annotation_sheet.csv'})
    del model;gc.collect();torch.cuda.empty_cache()

def size_efficiency(suite):
    import torch,gc
    suite=Path(suite);rows=[];state=read_metrics(suite/'status.json')
    for size in ['1.5B','3B']:
        root=suite/'experiments'/('07_size_'+size);result=root/'results/sharecot'
        if not (result/'metrics.json').is_file():continue
        sample=pd.read_csv(root/'data/test.csv').iloc[:10];model,tok,dev=load_qwen(result);saved=pd.read_csv(result/'predictions.csv').set_index('sample_id')
        if not np.allclose(classify(model,tok,dev,sample),saved.loc[sample.sample_id,'prob'],atol=1e-5):raise ValueError('Size model reload prediction mismatch: '+size)
        torch.cuda.reset_peak_memory_stats();classify(model,tok,dev,sample.iloc[:2]);times=[]
        for _ in range(3):
            for _,row in sample.iterrows():
                torch.cuda.synchronize();start=time.perf_counter();classify(model,tok,dev,pd.DataFrame([row]),batch=1);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        job=state['runs'].get('07_size_'+size+'_train',{});elapsed=None
        if job.get('started_at') and job.get('ended_at'):
            import datetime
            elapsed=(datetime.datetime.fromisoformat(job['ended_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
        row={'size':size,'seed':44,'weighting':'adaptive','inference_seconds':times,'median_seconds':float(np.median(times)),'batch_size':1,'warmup_queries':2,'train_load_and_eval_seconds':elapsed,'training_time_definition':'queue process wallclock includes base hashing/loading, all training, validation and final test','peak_allocated_bytes':torch.cuda.max_memory_allocated()};write_json(root/'efficiency.json',row);rows.append(row);del model;gc.collect();torch.cuda.empty_cache()
    write_json(suite/'experiments/07_size_efficiency.json',{'results':rows,'7B_reference':'03_efficiency/adaptive.json; reused seed44 checkpoint; original queue training wallclock unavailable in this suite'})

def collect(suite):
    suite=Path(suite);rows=[]
    size_efficiency(suite)
    for path in sorted((suite/'experiments').glob('*/results/*/metrics.json')):
        metric=read_metrics(path);api=path.parent.name.startswith('api_');m=metric.get('test',metric)
        rows.append({'experiment':path.parents[2].name,'method':path.parent.name,'seed':None if api else 44,'demo_sampling_seed':44 if api else None,**flatten_metrics(metric),'invalid_count':m.get('invalid_count'),'transport_errors':m.get('transport_errors'),'score_policy':m.get('score_policy',m.get('score_convention','continuous classification probability')),'probability_metrics_comparable':m.get('probability_metrics_comparable',not api),'source_metrics':str(path),'model_checkpoint':str(selected_adapter(path.parent)) if (path.parent/'model').exists() or (path.parent/'reuse.json').exists() else ''})
    result=suite/'results';result.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(result/'all_metrics.csv',index=False)
    state=read_metrics(suite/'status.json');expected=read_metrics(suite/'queue.json')['jobs'];unfinished=[{'id':j['id'],'status':state['runs'].get(j['id'],{}).get('status','not_run')} for j in expected if j['id']!='99_collect' and state['runs'].get(j['id'],{}).get('status')!='complete']
    write_json(result/'summary.json',{'n_result_rows':len(rows),'seed':44,'main_weighting':'adaptive','source':'selected checkpoints; per-run validation selection','training_seed_uncertainty':'not estimated from one training seed','unfinished_or_failed_jobs':unfinished,'all_preceding_jobs_complete':not unfinished})
    text=['# No-tip seed44 experiment results','', 'Primary ShareCoT uses adaptive weighting; every new student/baseline run uses seed44. Existing main adaptive and Direct-FT models are reused with provenance.','', '| Experiment | Method | Accuracy | Macro-F1 |','|---|---|---:|---:|']
    for row in rows:text.append(f"| {row['experiment']} | {row['method']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    text.extend(['','Full metrics: results/all_metrics.csv. Detailed configs, predictions and validation history remain inside each experiment.','Removing tips does not establish booking-time validity of realized duration/distance/charges. Human rationale annotations remain pending until actually performed.'])
    (suite/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8');print('Collected',len(rows),'completed result rows',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--mode',choices=['efficiency','architecture','cases','review_controls','collect'],required=True);args=parser.parse_args();globals()[args.mode](args.suite)
