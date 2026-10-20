"""No-tip API baselines, resumable raw responses and training-only four-shot examples."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from protocol import FEATURE_COLS,TARGET,feature_text,rationale_is_clean,write_json,sha256,configure_semantic_dictionary

MODELS=['gpt-5.4','gemini-3.1-pro-preview','deepseek-v3.2','minimax-m2.5','qwen3.5-plus','glm-5','kimi-k2.5']
FOUR_SHOT={'gpt-5.4','gemini-3.1-pro-preview','deepseek-v3.2','qwen3.5-plus'}
SYSTEM='依据给定行程信息预测是否授权拼车。0=未授权，1=授权。只使用输入信息，不推断个人历史或心理属性。输出JSON对象，包含简短reason和整数pred。'

def parsed_label(text):
    try:
        start,end=text.find('{'),text.rfind('}')
        obj=json.loads(text[start:end+1]);pred=obj.get('pred')
        return pred if type(pred) is int and pred in [0,1] else None
    except (ValueError,AttributeError):return None

def validate_demo_membership(demo_ids,train_ids,heldout_ids):
    if not set(demo_ids)<=set(train_ids) or set(demo_ids)&set(heldout_ids):raise ValueError('Four-shot examples must be training members only')

def input_prompt(row):return feature_text(dict(row))

def client_for_suite():
    import os
    import httpx
    from openai import OpenAI
    key = os.environ.get('API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('Set API_KEY or OPENAI_API_KEY in the environment')
    options = {'timeout': 90}
    if os.environ.get('API_HTTP_PROXY'):
        options['proxy'] = os.environ['API_HTTP_PROXY']
    client = OpenAI(api_key=key, base_url=os.environ.get('API_BASE_URL', 'https://aihubmix.com/v1'),
                    http_client=httpx.Client(**options), max_retries=0)
    return client, key


def generated_metrics(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    report=classification_report(y,pred,labels=[0,1],output_dict=True,zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_precision':float(report['macro avg']['precision']),'macro_recall':float(report['macro avg']['recall']),'macro_f1':float(report['macro avg']['f1-score']),'per_class':{str(k):report[str(k)] for k in [0,1]},'confusion_matrix_valid_predictions_only':confusion_matrix(y,pred,labels=[0,1]).tolist(),'invalid_count':int((pred==-1).sum()),'n':len(y),'invalid_policy':'prediction -1; counts as incorrect and false negative for its observed class, never oracle-filled','roc_auc':None,'score_policy':'generated hard class; no continuous class probability available'}

def run(root,workers=6,models=None,shots_filter=None):
    root=Path(root)
    serialization_metadata=configure_semantic_dictionary(root/'data')
    selected=MODELS if models is None else list(models)
    test=pd.read_csv(root/'data/test.csv',dtype={'sample_id':str})
    if test.empty or test.sample_id.isna().any() or test.sample_id.duplicated().any():
        raise ValueError('Test records require nonempty unique sample IDs')
    if not test[TARGET].isin([0,1]).all():
        raise ValueError('Test labels must be binary')
    reasons={};demo=None
    if shots_filter!=0 and any(model in FOUR_SHOT for model in selected):
        train=pd.read_csv(root/'data/train.csv',dtype={'sample_id':str})
        if train.sample_id.isna().any() or train.sample_id.duplicated().any():
            raise ValueError('Training records require nonempty unique sample IDs')
        if set(train.sample_id)&set(test.sample_id):
            raise ValueError('Training and test sample IDs overlap')
        if not train[TARGET].isin([0,1]).all():
            raise ValueError('Training labels must be binary')
        for line in (root/'data/rationales.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            record=json.loads(line)
            if record.get('accepted') is True and rationale_is_clean(record.get('reason')):
                sid=str(record['sample_id'])
                if sid in reasons or sid not in set(train.sample_id):
                    raise ValueError('Accepted rationale IDs must be unique training members')
                reasons[sid]=record['reason']
        candidates=train[train.sample_id.isin(reasons)]
        if any((candidates[TARGET]==label).sum()<2 for label in [0,1]):
            raise ValueError('Four-shot evaluation requires two accepted training rationales per class')
        demo=pd.concat([candidates[candidates[TARGET]==label].sample(n=2,random_state=44) for label in [0,1]])
        validate_demo_membership(demo.sample_id,train.sample_id,test.sample_id)
    client,key=client_for_suite();summary=[];failures=[]
    for model in selected:
        conditions = ([0,4] if model in FOUR_SHOT else [0])
        if shots_filter is not None:
            conditions = [s for s in conditions if s == shots_filter]
        for shots in conditions:
            name=model.replace('/','_')+f'_{shots}shot';out=root/'results'/('api_'+name);cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
            identity={'model':model,'shots':shots,'test_sha256':sha256(root/'data/test.csv'),'train_sha256':sha256(root/'data/train.csv') if shots else None,'system':SYSTEM,'features':FEATURE_COLS,'code_sha256':sha256(__file__),'demo_ids':demo.sample_id.tolist() if shots else [],'temperature':0,'max_completion_tokens':512,'transport':'environment-configured OpenAI-compatible endpoint',
                      'protocol_sha256':sha256(Path(__file__).with_name('protocol.py')),
                      'endpoint_sha256':hashlib.sha256(os.environ.get('API_BASE_URL','https://aihubmix.com/v1').encode()).hexdigest(),
                      **serialization_metadata}
            identity['demonstration_corpus_sha256']=sha256(root/'data/rationales.jsonl') if shots else None
            sig=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            old=out/'run_config.json'
            if old.exists() and json.loads(old.read_text(encoding='utf-8'))!=identity:raise ValueError('API configuration changed; use a new root')
            write_json(old,identity)
            prefix=[{'role':'system','content':SYSTEM}]
            if shots:
                for _,row in demo.iterrows():prefix.extend([{'role':'user','content':input_prompt(row)},{'role':'assistant','content':json.dumps({'reason':reasons[row.sample_id],'pred':int(row[TARGET])},ensure_ascii=False)}])
            def predict(row):
                path=cache/(hashlib.sha256(str(row.sample_id).encode()).hexdigest()+'.json')
                previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
                if previous and previous.get('signature')!=sig:raise ValueError('API cache signature changed')
                if previous and previous.get('status')=='received':return previous
                record={'sample_id':str(row.sample_id),'signature':sig,'status':'error','attempts':(previous or {}).get('attempts',[])}
                options={'extra_body':{'enable_thinking':False}} if model=='qwen3.5-plus' else {}
                for attempt in range(3):
                    started=time.time()
                    try:
                        response=client.chat.completions.create(model=model,messages=prefix+[{'role':'user','content':input_prompt(row)}],temperature=0,max_completion_tokens=512,**options)
                        raw=response.choices[0].message.content or ''
                        item={'status':'received','raw':raw,'request_id':response.id,'actual_model':response.model,'usage':response.usage.model_dump() if response.usage else None,'seconds':time.time()-started}
                        record.update(item,label=parsed_label(raw));record['attempts'].append(item);write_json(path,record);return record
                    except Exception as exc:
                        record['attempts'].append({'status':'error','error':str(exc).replace(key,'[redacted]')[:300],'seconds':time.time()-started});write_json(path,record)
                        if attempt<2:time.sleep(2**attempt)
                return record
            first=predict(test.iloc[0])
            if first['status']!='received':
                failures.append(name);write_json(out/'failure.json',{'reason':'model/API canary failed','sample_id':first['sample_id']});continue
            records=[first]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(predict,row) for _,row in test.iloc[1:].iterrows()]
                for future in concurrent.futures.as_completed(futures):
                    records.append(future.result())
                    if len(records)%20==0:write_json(out/'progress.json',{'processed':len(records),'total':len(test),'errors':sum(r['status']!='received' for r in records)})
            records={r['sample_id']:r for r in records};rows=[]
            for _,row in test.iterrows():
                record=records[str(row.sample_id)];rows.append({'sample_id':row.sample_id,'y_true':int(row[TARGET]),'y_pred':record.get('label') if record.get('label') in [0,1] else -1,'raw':record.get('raw',''),'status':record['status']})
            pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
            metrics=generated_metrics([r['y_true'] for r in rows],[r['y_pred'] for r in rows]);metrics.update(method=name,temperature=0,transport_errors=sum(r['status']!='received' for r in records.values()))
            write_json(out/'metrics.json',metrics);summary.append(metrics)
            if metrics['transport_errors']:failures.append(name)
            print(name,metrics['macro_f1'],'invalid',metrics['invalid_count'],flush=True)
    write_json(root/'api_summary.json',{'results':summary,'failed_models':failures,'complete':not failures})
    return 0 if not failures else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--shots', type=int, choices=[0, 4], default=None)
    args = parser.parse_args()
    selected = MODELS if args.models is None else args.models
    if args.shots == 4:
        selected = [m for m in selected if m in FOUR_SHOT]
        if not selected:
            parser.error('No four-shot models selected')
    raise SystemExit(run(args.root, args.workers, selected, args.shots))
