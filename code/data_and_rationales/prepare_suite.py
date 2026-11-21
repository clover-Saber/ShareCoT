import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from protocol import FEATURE_COLS, TARGET, rationale_is_clean, write_json

def stable_key(row):
    values=[]
    for key in FEATURE_COLS:
        value=row[key]
        if pd.isna(value): value='__missing__'
        elif key=='trip_start_timestamp': value=pd.Timestamp(value).isoformat()
        elif key!='dropoff_poi_category': value=float(value).hex()
        else: value=str(value)
        values.append(value)
    return tuple(values)

def validate_splits(frames):
    sets={}
    for split,frame in frames.items():
        if not set(FEATURE_COLS+[TARGET]).issubset(frame): raise ValueError('Missing allowed columns')
        if not set(frame[TARGET].astype(int).unique()).issubset({0,1}): raise ValueError('Invalid label')
        keys=frame.apply(stable_key,axis=1)
        if keys.duplicated().any(): raise ValueError(f'{split} duplicate feature rows')
        sets[split]=set(keys)
    overlaps={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlaps.values()): raise ValueError(f'Cross-split overlap: {overlaps}')
    return overlaps

def remap_clean_rationales(original_train,new_train,corpus):
    source={str(row.sample_id):(stable_key(row),int(row[TARGET])) for _,row in original_train.iterrows()}
    dest={stable_key(row):(str(row.sample_id),int(row[TARGET])) for _,row in new_train.iterrows()}
    result=[]
    for record in corpus:
        sid=record['sample_id']
        if sid not in source: raise ValueError('Rationale not from original training membership')
        key,label=source[sid]
        if int(record['label'])!=label: raise ValueError('Rationale label differs')
        if record.get('accepted') and not rationale_is_clean(record.get('reason')): raise ValueError('Rationale contains forbidden content')
        if key not in dest: continue
        new_id,new_label=dest[key]
        if new_label!=label: raise ValueError('Remapped label differs')
        result.append({**record,'source_sample_id':sid,'sample_id':new_id,'remapping_policy':'exact nine-feature membership; new training only'})
    return result

def write_membership(root,frames,sources,policy):
    data=root/'data';data.mkdir(parents=True,exist_ok=True)
    overlaps=validate_splits(frames)
    meta={'features':FEATURE_COLS,'target':TARGET,'student_seed':44,'split_policy':policy,'sources':sources,'splits':{},'cross_split_exact_feature_overlap':overlaps,'limitations':['realized distance, duration and charges remain; no passenger-history claim']}
    for split,frame in frames.items():
        selected=frame[FEATURE_COLS+[TARGET]].copy()
        selected.insert(0,'sample_id',[f'{root.name}:{split}:{i:06d}' for i,(_,row) in enumerate(frame.iterrows())])
        selected.to_csv(data/f'{split}.csv',index=False)
        meta['splits'][split]={'rows':len(selected),'labels':{str(k):int(v) for k,v in selected[TARGET].value_counts().items()}}
    write_json(data/'manifest.json',meta)
    return {split:pd.read_csv(data/f'{split}.csv') for split in frames}

def main(suite,source,pilot,multiseed):
    suite,source,pilot,multiseed=map(Path,[suite,source,pilot,multiseed])
    mainroot=suite/'experiments/02_main'
    (mainroot/'data').mkdir(parents=True,exist_ok=True)
    for name in ['train.csv','val.csv','test.csv','manifest.json','rationales.jsonl','rationale_summary.json','shap_model.json','shap_metadata.json']:
        shutil.copy2(pilot/'data'/name,mainroot/'data'/name)
    for method,old_name in [('adaptive','sharecot'),('direct_ft','direct_ft')]:
        old=multiseed/'runs'/method/'seed_44'/'results'/old_name
        dest=mainroot/'results'/method;dest.mkdir(parents=True,exist_ok=True)
        for name in ['metrics.json','predictions.csv','run_config.json','validation_history.json','initialization.json','encoding_report.json']:
            if (old/name).exists(): shutil.copy2(old/name,dest/name)
        cfg=json.loads((dest/'run_config.json').read_text(encoding='utf-8'))
        if cfg['seed']!=44: raise ValueError('Main reuse provenance mismatch')
        metric=json.loads((dest/'metrics.json').read_text(encoding='utf-8'))
        if metric.get('smoke_run'):raise ValueError('Main reuse corpus/test provenance mismatch')
        if cfg.get('features')!=FEATURE_COLS or cfg.get('target')!=TARGET or cfg.get('weighting_mode')!=('adaptive' if method=='adaptive' else 'fixed_3'):raise ValueError('Main reuse configuration mismatch')
        write_json(dest/'reuse.json',{'source':str(old),'model':str(old/'model'),'student_seed':44,'retrained':False})
    original_train=pd.read_csv(mainroot/'data/train.csv')
    corpus=[json.loads(x) for x in (mainroot/'data/rationales.jsonl').read_text(encoding='utf-8').splitlines()]
    specs=[('04_scale_50k','补充实验-增加数据规模','_50k'),('04_scale_100k','补充实验-增加数据规模100k','_100k'),('05_time','补充实验-按时间划分数据集','_time')]
    for name,folder,suffix in specs:
        paths={split:source/folder/f'chicago_{split}{suffix}.csv' for split in ['train','val','test']}
        frames={k:pd.read_csv(v) for k,v in paths.items()}
        if name=='05_time':
            limits={k:(str(pd.to_datetime(v.trip_start_timestamp).min()),str(pd.to_datetime(v.trip_start_timestamp).max())) for k,v in frames.items()}
            if limits['train'][1]>limits['val'][0] or limits['val'][1]>limits['test'][0]: raise ValueError('Temporal boundary overlap')
        root=suite/'experiments'/name
        cleaned=write_membership(root,frames,{k:{'path':str(v)} for k,v in paths.items()},'preserve existing chronological split' if name=='05_time' else 'preserve existing enlarged dataset split; original resampling seed 42')
        if name.startswith('04_'):
            remapped=remap_clean_rationales(original_train,cleaned['train'],corpus)
            (root/'data/rationales_remapping_reference.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in remapped),encoding='utf-8')
            write_json(root/'data/rationale_summary.json',{'source':str(mainroot/'data/rationales.jsonl'),'selected_reference':len(remapped),'accepted_reference':sum(bool(x.get('accepted')) for x in remapped),'complete':False,'policy':'-'})
    config=json.loads((source/'补充实验-不同区域划分/region_split_config.json').read_text(encoding='utf-8'))['config']
    for region in ['center','suburb']:
        paths={split:source/'补充实验-不同区域划分/data'/f'chicago_{split}_{region}.csv' for split in ['train','val','test']}
        frames={k:pd.read_csv(v) for k,v in paths.items()}
        root=suite/'experiments'/f'06_region_{region}'
        write_membership(root,frames,{k:{'path':str(v)} for k,v in paths.items()},config)
    write_json(suite/'data_preparation.json',{'main_reused_seed':44,'features':FEATURE_COLS,'scale_rationale_policy':'-','complete':True})
    print(json.dumps(json.loads((suite/'data_preparation.json').read_text(encoding='utf-8')),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--source',required=True);parser.add_argument('--pilot',required=True);parser.add_argument('--multiseed',required=True)
    args=parser.parse_args();main(args.suite,args.source,args.pilot,args.multiseed)
