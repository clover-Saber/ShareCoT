import argparse,pathlib,json,shutil,datetime
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from protocol import FEATURE_COLS,TARGET,rationale_is_clean
parser = argparse.ArgumentParser()
parser.add_argument('--suite', required=True, type=pathlib.Path)
parser.add_argument('--source', required=True, type=pathlib.Path)
args = parser.parse_args()
p = args.suite.resolve()
source = args.source.resolve()
root=p/'experiments/04_scale/50k';root.mkdir(parents=True,exist_ok=True)
data=root/'data';stage=root/'data.preparing'
assert not data.exists() and not stage.exists(),'Existing data requires reconciliation'
main_data=p/'experiments/02_main/data'
main={s:pd.read_csv(main_data/(s+'.csv'),float_precision='round_trip') for s in ['train','val','test']}
raw=pd.read_csv(source,float_precision='round_trip')
assert len(raw)==50000 and len(main['train'])==8000
def keys(df):
 frame=df[FEATURE_COLS].copy()
 for col in FEATURE_COLS:
  if col=='trip_start_timestamp':frame[col]=pd.to_datetime(frame[col]).dt.strftime('%Y-%m-%dT%H:%M:%S')
  elif col=='dropoff_poi_category':frame[col]=frame[col].astype(str)
  else:frame[col]=frame[col].map(lambda v:'__missing__' if pd.isna(v) else format(float(v),'.12g'))
 return frame.apply(lambda row:tuple(row.tolist()),axis=1)
mk={s:keys(f) for s,f in main.items()};rk=keys(raw)
owner={};main_rows={}
for split,frame in main.items():
 assert mk[split].nunique()==len(frame)
 for key,(_,row) in zip(mk[split],frame.iterrows()):
  assert key not in owner
  owner[key]=split;main_rows[key]=row
assert len(owner)==10000 and set(owner)<=set(rk)
consumed=set();extras=[];aliases={'train':[],'val':[],'test':[]}
for index,(_,row) in enumerate(raw.iterrows()):
 key=rk.iloc[index]
 if key in owner:
  original=main_rows[key]
  assert int(row[TARGET])==int(original[TARGET])
  for col in FEATURE_COLS:
   if col=='trip_start_timestamp':assert pd.Timestamp(row[col])==pd.Timestamp(original[col])
   elif col=='dropoff_poi_category':assert str(row[col])==str(original[col])
   else:assert np.isclose(float(row[col]),float(original[col]),rtol=1e-11,atol=1e-10,equal_nan=True),(key,col)
  if key not in consumed:consumed.add(key);continue
  target=aliases[owner[key]]
 else:target=extras
 record={col:row[col] for col in FEATURE_COLS+[TARGET]}
 record['sample_id']='scale50k:source:'+str(index).zfill(6)
 target.append(record)
assert len(consumed)==10000 and len(extras)==39973
assert {k:len(v) for k,v in aliases.items()}=={'train':22,'val':1,'test':4}
pool=pd.DataFrame(extras)
pool_keys=keys(pool);assert pool_keys.nunique()==len(pool) and not set(pool_keys)&set(owner)
wanted={'train':40000,'val':5000,'test':5000}
new_sizes={s:wanted[s]-len(main[s])-len(aliases[s]) for s in main}
new_train,new_hold=train_test_split(pool,train_size=new_sizes['train'],random_state=44,stratify=pool[TARGET])
new_val,new_test=train_test_split(new_hold,train_size=new_sizes['val'],test_size=new_sizes['test'],random_state=44,stratify=new_hold[TARGET])
new={'train':new_train,'val':new_val,'test':new_test}
frames={}
for split in main:
 frames[split]=pd.concat([main[split][['sample_id',*FEATURE_COLS,TARGET]],pd.DataFrame(aliases[split]),new[split]],ignore_index=True)[['sample_id',*FEATURE_COLS,TARGET]]
 assert len(frames[split])==wanted[split]
 assert not frames[split].sample_id.duplicated().any()
 assert set(main[split].sample_id)<=set(frames[split].sample_id)
 assert set(frames[split][TARGET].astype(int))=={0,1}
sets={s:set(keys(frame)) for s,frame in frames.items()}
for a,b in [('train','val'),('train','test'),('val','test')]:
 assert not sets[a]&sets[b],a+'_'+b
 assert not set(frames[a].sample_id)&set(frames[b].sample_id)
corpus=[json.loads(line) for line in (main_data/'rationales.jsonl').read_text(encoding='utf-8').splitlines()]
labels=dict(zip(frames['train'].sample_id,frames['train'][TARGET].astype(int)))
accepted=[r for r in corpus if r.get('accepted')]
assert len(corpus)==1000 and len(accepted)==946
assert all(r['sample_id'] in labels and int(r['label'])==labels[r['sample_id']] for r in corpus)
assert all(rationale_is_clean(r['reason']) for r in accepted)
assert set(mk['train'])<=sets['train'] and not set(mk['train'])&(sets['val']|sets['test'])
stage.mkdir()
for split,frame in frames.items():frame.to_csv(stage/(split+'.csv'),index=False)
for name in ['rationales.jsonl','shap_model.json','shap_metadata.json']:
 shutil.copy2(main_data/name,stage/name)
roundtrip={s:pd.read_csv(stage/(s+'.csv'),float_precision='round_trip') for s in main}
for s in main:assert keys(roundtrip[s]).tolist()==keys(frames[s]).tolist()
manifest={'schema_version':1,'features':FEATURE_COLS,'target':TARGET,'student_seed':44,'split_policy':'Nested expansion retaining adopted main train8000/validation1000/test1000 roles; new label-only rows stratified with seed44; feature-equivalent aliases assigned to same role as their main member.','source_path':str(source),'source_rows':50000,'main_memberships_preserved':True,'additional_label_only_training_rows':32000,'total_train_label_only_rows':39054,'accepted_rationales':946,'new_cot_generated':False,'attribution_teacher_training_membership':'Frozen original8000 retained entirely in enlarged training. Source teacher seed42; student seed44.','feature_identity_grouping':'All nine features: timestamp seconds, POI string, numeric12significant-digit candidate keys with main linkage numerical/timestamp/label validation. This is feature equivalence, not passenger identity.','feature_equivalent_extra_rows_by_role':{s:len(v) for s,v in aliases.items()},'cross_split_feature_group_overlap':{'train_val':0,'train_test':0,'val_test':0},'splits':{s:{'rows':len(frame),'unique_feature_groups':len(sets[s]),'labels':{str(k):int(v) for k,v in frame[TARGET].value_counts().items()}} for s,frame in frames.items()},'historical_holdout_reuse':True,'current_trip_tip_removed':True,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
(stage/'rationale_summary.json').write_text(json.dumps({'complete':True,'accepted':946,'train_rows':40000,'label_only_rows':39054,'source_corpus':str(main_data/'rationales.jsonl'),'corpus_byte_identical':True,'new_generation':False,'all_training_only':True},indent=2))
assert (stage/'rationales.jsonl').read_bytes() == (main_data/'rationales.jsonl').read_bytes()
stage.rename(data)
(root/'data_preparation_verified.json').write_text(json.dumps({'verified':True,'all_main_memberships_preserved':True,'all946_reasons_preserved':True,'frozen_teacher_train_fully_inside_scale_train':True,'feature_group_cross_split_overlap':0,'train_val_test_rows':[40000,5000,5000]},indent=2))
print('PREPARED',str(root));print(json.dumps(manifest,indent=2))
