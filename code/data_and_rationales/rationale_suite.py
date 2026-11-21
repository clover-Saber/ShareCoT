from base64 import urlsafe_b64encode
import argparse
import concurrent.futures
import json
import shutil
import sys
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import FEATURE_COLS,TARGET,feature_text,rationale_is_clean,write_json
from prepare_variant import cot_only_membership

TEACHER='gpt-5.4';JUDGES=['deepseek-v3.2','qwen3.5-plus']
TEACHER_SYSTEM='你为行程级拼车授权分类生成简短的事后归因解释。只能依据输入字段、给定标签和可用的归因模型 SHAP（若未提供归因，则只用行程字段和标签）。标签0表示未授权，标签1表示授权。SHAP正值提高授权评分，负值降低授权评分，属于模型归因而不是因果效应。用“正向归因/负向归因/模型评分”等表述，禁止用“导致/主要因为”做因果解释，禁止推断个人心理属性。选取2至4个主要贡献特征即可，无须罗列所有字段；不要自由翻译POI类别，若提及则保留输入原文。不得提及小费、打赏、价格敏感或体验敏感；数据不包含这些信息。若归因模型预测概率所支持的类别与给定标签不同，要承认模型与观察标签的分歧，不得篡改归因方向。中文理由60至110字。仅输出JSON对象，字段reason和pred，其中pred必须等于给定标签。费用单位为美元；不得擅自替换为其他货币单位。'
JUDGE_SYSTEM='审核行程分类的事后归因解释。仅验证所给字段、标签和SHAP的一致性，不验证心理真实性。标签0=未授权，标签1=授权；正SHAP提高授权评分，负SHAP降低授权评分。负SHAP与标签0并不矛盾。每个样本允许同时出现正负归因，不要求每个特征方向均与标签一致，也不要求列举全部特征。检查是否使用不存在的信息、捏造事实、反转SHAP方向、做因果或个人心理断言。解释不能包含小费、打赏、价格敏感或体验敏感。SHAP与标签可以不一致，但解释必须如实承认。评分规则：事实和方向正确且无额外断言为8至10分；有明确错误为0至7分。无明确错误则不要捏造错误。score必须是0至10的数值，不能使用负分；pass为布尔值；feedback不超过40个汉字。8分及以上且无违规才通过。例如合法通过格式：{"pass":true,"score":9,"feedback":"归因方向与输入一致"}；合法拒绝格式：{"pass":false,"score":3,"feedback":"明确指出哪项事实错误"}。若未提供SHAP，仅核验行程字段、观察标签和解释，不要求计算不存在的归因。'

def teacher_user(record,style):
    text=feature_text(record['features'])+'\n给定观察标签：'+str(record['label'])
    if style!='no_shap':
        text+='\n归因模型授权预测概率：'+str(round(record['xgb_probability'],6))+'\nSHAP基准log-odds：'+str(round(record['shap_base_log_odds'],6))+'\nSHAP归因：'+json.dumps(record['shap'],ensure_ascii=False)
    return text

def raw_teacher_records(records):
    result=[]
    for record in records:
        teacher=record.get('stages',{}).get('teacher',{});verdict=teacher.get('verdict',{})
        valid=teacher.get('status')=='valid' and type(verdict.get('pred')) is int and verdict.get('pred')==record['label'] and rationale_is_clean(verdict.get('reason'))
        result.append({**record,'accepted':valid,'reason':verdict.get('reason','') if valid else '', 'control':'teacher schema/label/no-tip guard retained; posterior judge filtering removed'})
    return result

def generation_identity(root,suite,style,count,cot_only):
    suite = Path(suite)
    ids=json.loads((suite/'experiments/02_main/data/shap_metadata.json').read_text(encoding='utf-8'))['selected_sample_ids'] if cot_only else None
    return {'style':style,'cot_only':cot_only,'requested_count':count,'matched_candidate_ids':ids,'sampling_seed':44,'features':FEATURE_COLS,'teacher':TEACHER,'judges':JUDGES,'teacher_system':TEACHER_SYSTEM,'judge_system':JUDGE_SYSTEM}

def attributions(root,frame,selected,style):
    from xgboost import XGBClassifier,DMatrix
    x=frame[FEATURE_COLS].copy();cats=sorted(x.dropoff_poi_category.dropna().astype(str).unique());mapping={c:i for i,c in enumerate(cats)}
    x['dropoff_poi_category']=x.dropoff_poi_category.astype(str).map(mapping).fillna(-1);x['trip_start_timestamp']=pd.to_datetime(x.trip_start_timestamp).astype('int64')/1e9
    x=x.astype(float);medians=x.median();x=x.fillna(medians)
    if x.isna().any().any():raise ValueError('Entirely missing selected feature')
    if style=='no_shap':
        contrib=np.zeros((len(selected),len(FEATURE_COLS)+1));prob=np.full(len(selected),.5)
    elif style=='logit_boost':
        sys.path.insert(0,str(root.parent.parent/'.analysis_deps'))
        import shap,joblib
        from baselines_suite import LogitBoostClassifier
        model=LogitBoostClassifier(n_estimators=200,max_depth=3,learning_rate=.08,random_state=44).fit(x.to_numpy(),frame[TARGET].to_numpy())
        contrib=np.zeros((len(selected),len(FEATURE_COLS)+1));subset=x.loc[selected.index].to_numpy()
        for tree in model.estimators_:
            explainer=shap.TreeExplainer(tree);contrib[:,:-1]+=model.learning_rate*np.asarray(explainer.shap_values(subset));contrib[:,-1]+=model.learning_rate*float(np.asarray(explainer.expected_value).reshape(-1)[0])
        prob=model.predict_proba(subset)[:,1];joblib.dump(model,root/'data/shap_model.joblib')
        predicted_logit=np.log(np.clip(prob,1e-12,1-1e-12)/(1-np.clip(prob,1e-12,1-1e-12)))
        if not np.allclose(contrib.sum(1),predicted_logit,atol=1e-4):raise ValueError('LogitBoost SHAP additivity check failed')
    else:
        model=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=44,tree_method='hist',n_jobs=8).fit(x,frame[TARGET]);model.save_model(root/'data/shap_model.json')
        contrib=model.get_booster().predict(DMatrix(x.loc[selected.index]),pred_contribs=True);prob=model.predict_proba(x.loc[selected.index])[:,1]
    write_json(root/'data/shap_metadata.json',{'fit_split':'train_only','seed':44,'style':style,'category_mapping':mapping,'medians':medians.to_dict(),'features':FEATURE_COLS,'selected_sample_ids':selected.sample_id.tolist(),'count':len(selected)})
    records=[]
    for i,(_,row) in enumerate(selected.iterrows()):
        hints=[{'feature':FEATURE_COLS[j],'value':str(row[FEATURE_COLS[j]]),'shap_log_odds':float(contrib[i,j])} for j in np.argsort(-np.abs(contrib[i,:-1]))]
        records.append({'sample_id':str(row.sample_id),'label':int(row[TARGET]),'features':{k:str(row[k]) for k in FEATURE_COLS},'shap':hints if style!='no_shap' else [],'shap_base_log_odds':float(contrib[i,-1]),'xgb_probability':float(prob[i])})
    return records

def finish(root,records,cot_only,frame,source,identity):
    path=root/'data/rationales.jsonl';path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sorted(records,key=lambda r:r['sample_id'])),encoding='utf-8')
    if cot_only:(root/'data/rationales_all.jsonl').write_text(path.read_text(encoding='utf-8'),encoding='utf-8')
    accepted=sum(bool(r.get('accepted')) for r in records);errors=sum(bool(r.get('error')) for r in records)
    if not accepted or errors:raise ValueError('Corpus empty or transport errors remain')
    if cot_only:
        selected=cot_only_membership(frame,records);selected.to_csv(root/'data/train.csv',index=False)
        accepted_records=[r for r in records if r.get('accepted')]
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in accepted_records),encoding='utf-8')
        manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
        manifest['training_protocol']='CoT-only: accepted rationale members only; validation and test retained'
        manifest['splits']['train']={'rows':len(selected),'labels':{str(k):int(v) for k,v in selected[TARGET].value_counts().items()}}
        write_json(root/'data/training_manifest.json',manifest)
    else:selected=frame
    write_json(root/'data/rationale_summary.json',{'selected':len(records),'accepted':accepted,'errors':errors,'source':source,'generation_identity':identity,'teacher':TEACHER,'judges':JUDGES,'human_reviewed':0,'cot_only':cot_only,'training_rows':len(selected),'complete':True,'limitations':['automated consistency audit does not establish psychological truth','CoT-only accepted counts may differ across variants']})

def run(root,suite,style,count=1000,workers=6,cot_only=False):
    root,suite=Path(root),Path(suite);data=root/'data';data.mkdir(parents=True,exist_ok=True)
    if cot_only and not (data/'train_all.csv').exists():shutil.copy2(data/'train.csv',data/'train_all.csv')
    identity=generation_identity(root,suite,style,count,cot_only)
    if (data/'rationale_summary.json').exists():
        summary=json.loads((data/'rationale_summary.json').read_text(encoding='utf-8'))
        if summary.get('complete'):
            if summary.get('generation_identity')!=identity:raise ValueError('Completed corpus provenance changed')
            print('Completed corpus retained',root.name);return
    frame=pd.read_csv(data/'train_all.csv' if cot_only else data/'train.csv',dtype={'sample_id':str})
    if cot_only and style=='no_judge':
        from prepare_variant import main as prepare
        prepare(suite,'08_cot_xgboost')
        default_root=suite/'experiments/08_cot_xgboost'
        run(default_root,suite,'xgboost',count,workers,True)
        source=default_root/'data/rationales_all.jsonl';records=raw_teacher_records([json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()])
        identity=generation_identity(root,suite,style,count,cot_only)
        finish(root,records,True,frame,{'path':str(source),'variant':style,'policy':'same fresh teacher traces as XGBoost-CoT; posterior judge acceptance removed, schema/label/no-tip guard retained'},identity);return
    from generate_rationales import request_stage,curate_stages
    from types import SimpleNamespace
    from openai import APIError
    from api_suite import client_for_suite
    client,key=client_for_suite()
    if cot_only:
        ids=json.loads((suite/'experiments/02_main/data/shap_metadata.json').read_text(encoding='utf-8'))['selected_sample_ids'];selected=frame[frame.sample_id.isin(ids)].copy()
        if len(selected)!=len(ids):raise ValueError('Matched CoT construction candidates missing')
    else:selected=frame.sample(n=min(count,len(frame)),random_state=44).sort_index()
    records=attributions(root,frame,selected,style)
    cache=data/'rationale_cache';cache.mkdir(exist_ok=True);ledger=root/'logs/api_calls.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True);lock=threading.Lock()
    def log(entry):
        with lock:
            with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    actual_response=threading.local()
    def create(**kwargs):
        actual_response.model=None
        response=client.chat.completions.create(**kwargs);actual_response.model=response.model;return response
    tagged_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def tagged_log(entry):
        entry['actual_model']=getattr(actual_response,'model',None);log(entry)
    def call(model,system,prompt,limit,sid,role):
        result=request_stage(tagged_client,model,system,prompt,limit,sid,role,tagged_log,key,(APIError,))
        result['actual_model']=next((r.get('actual_model') for r in reversed(result['attempts']) if r.get('success')),None)
        return result
    def process(record):
        path=cache/(urlsafe_b64encode(record['sample_id'].encode()).decode()+'.json')
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if previous and previous.get('curation_complete'):return previous
        result=previous or {**record,'teacher_model':TEACHER,'judge_models':JUDGES,'accepted':False,'human_reviewed':False,'reason':''}
        return curate_stages(result,TEACHER_SYSTEM,JUDGE_SYSTEM,teacher_user(record,style),call,lambda r:write_json(path,r))
    first=process(records[0])
    if first.get('error'):raise RuntimeError('Corpus API canary failed')
    results=[first]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(process,record) for record in records[1:]]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results)%20==0:
                status={'processed':len(results),'total':len(records),'accepted':sum(bool(r.get('accepted')) for r in results),'errors':sum(bool(r.get('error')) for r in results)};write_json(data/'rationale_progress.json',status);print(status,flush=True)
    finish(root,results,cot_only,frame,{'variant':style},identity)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--suite',required=True);parser.add_argument('--style',choices=['xgboost','no_shap','no_judge','logit_boost'],required=True);parser.add_argument('--count',type=int,default=1000);parser.add_argument('--workers',type=int,default=6);parser.add_argument('--cot-only',action='store_true');args=parser.parse_args();run(args.root,args.suite,args.style,args.count,args.workers,args.cot_only)
