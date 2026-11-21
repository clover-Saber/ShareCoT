import os,sys,json,time,subprocess,csv,random
from pathlib import Path
S = Path(os.environ['SHARECOT_SUITE']).resolve()
OUT = Path(os.environ['SHARECOT_BENCHMARK_OUTPUT_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
UUID = os.environ['SHARECOT_GPU_UUID']
os.environ['CUDA_VISIBLE_DEVICES']=UUID;os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
def save(name,data):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(data,indent=2),encoding='utf-8');q.replace(p)
if len(sys.argv)==1:
 for method in ['dhrd','single_cot']:
  save('progress.json',dict(status='running',method=method))
  subprocess.run([sys.executable,'-u',__file__,method],check=True)
 save('progress.json',dict(status='complete'));sys.exit()
method=sys.argv[1]
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert UUID not in active,active
import numpy as np,pandas as pd,torch,resource
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
sys.path.insert(0,str(S/'code'))
from protocol import FEATURE_COLS,build_prompt
from formal_efficiency_seed44 import environment,summarize
old=json.loads((S/'experiments/03_efficiency/summary.json').read_text())['results'][0]
assert old['physical_gpu0_uuid']==UUID
selected=old['sample_ids'];assert len(selected)==30
frame=pd.read_csv(S/'experiments/02_main/data/test.csv',dtype={'sample_id':str})
assert selected==random.Random(44).sample(frame.sample_id.tolist(),30)
inputs=frame[['sample_id',*FEATURE_COLS]].set_index('sample_id',drop=False).loc[selected]
torch.set_num_threads(36);torch.set_num_interop_threads(36)
if method=='single_cot':
 result = Path(os.environ['SHARECOT_SINGLECOT_RESULT_DIR']).resolve()
 if code_dir := os.environ.get('SHARECOT_SINGLECOT_CODE_DIR'):
  sys.path.insert(0, str(Path(code_dir).resolve()))
 from train_singlecot import structured_prompt,parse_prediction
else:
 result = Path(os.environ['SHARECOT_DHRD_RESULT_DIR']).resolve()
 if code_dir := os.environ.get('SHARECOT_DHRD_CODE_DIR'):
  sys.path.insert(0, str(Path(code_dir).resolve()))
 from dhrd_model import DHRDStyleQwen,initialize_classifier
cfg=json.loads((result/'run_config.json').read_text());assert cfg['seed']==44
adapter=result/'model'
if method=='single_cot':
 base=AutoModelForCausalLM.from_pretrained(cfg['model_dir'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
else:
 base=DHRDStyleQwen.from_pretrained(cfg['model_dir'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
 initialize_classifier(base,44)
model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False).to('cuda:0').eval()
tok=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True);tok.padding_side='right'
assert str(torch.cuda.get_device_properties(0).uuid).lower()==UUID.removeprefix('GPU-').lower()
assert not getattr(model,'is_loaded_in_4bit',False) and not getattr(model,'is_loaded_in_8bit',False)
saved=pd.read_csv(result/'predictions.csv',dtype={'sample_id':str}).set_index('sample_id')
assert set(saved.index)==set(frame.sample_id)
assert np.array_equal(saved.loc[frame.sample_id].y_true,frame.shared_trip_authorized)
def predict(sid):
 one=inputs.loc[[sid],FEATURE_COLS]
 if method=='single_cot':
  prompt=structured_prompt(one.iloc[0].to_dict(),tok)
  batch=tok(prompt,add_special_tokens=False,return_tensors='pt');batch={k:v.to('cuda:0') for k,v in batch.items()}
  assert batch['input_ids'].shape[1]<=512
  ids=model.generate(**batch,max_new_tokens=256,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
  generated=ids[0,batch['input_ids'].shape[1]:]
  text=tok.decode(generated,skip_special_tokens=True)
  label=parse_prediction(text)
  return {'label':label,'text':text,'ids':generated,'input_tokens':batch['input_ids'].shape[1]}
 prompts=[build_prompt(dict(row),tok) for _,row in one.iterrows()]
 batch=tok(prompts,padding=True,truncation=True,max_length=512,return_tensors='pt',add_special_tokens=False);batch={k:v.to('cuda:0') for k,v in batch.items()}
 with torch.autocast('cuda',dtype=torch.bfloat16):
  output=model(**batch,cls_token_index=batch['attention_mask'].sum(-1)-1,compute_lm=False)
  prob=torch.sigmoid(output['class_logits'].float()).cpu().tolist()[0]
 return {'label':int(prob>=.5),'prob':prob,'input_tokens':batch['input_ids'].shape[1]}
print(method,'loaded',flush=True)
raw=[];details=[]
torch.cuda.empty_cache()
with torch.inference_mode():
 for sid in selected[:5]:predict(sid)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0)
 for repeat in range(3):
  for i,sid in enumerate(selected):
   torch.cuda.synchronize();start=time.perf_counter();value=predict(sid);torch.cuda.synchronize();seconds=time.perf_counter()-start
   entry=dict(method=method,sample_id=sid,repeat=repeat,seconds=seconds,label=value['label'],input_tokens=value['input_tokens'])
   if method=='single_cot':
    ids=value.pop('ids').tolist();entry.update(output_tokens=len(ids),non_eos_tokens=sum(t!=tok.eos_token_id for t in ids),ends_with_eos=bool(ids and ids[-1]==tok.eos_token_id),valid_final_label=value['label'] in [0,1])
    details.append(dict(sample_id=sid,repeat=repeat,raw_generation=value['text'],label=value['label'],saved_label=int(saved.loc[sid,'y_pred']),exact_saved_text=value['text']==saved.loc[sid,'raw_generation']))
   else:details.append(dict(sample_id=sid,repeat=repeat,prob=value['prob'],saved_prob=float(saved.loc[sid,'prob']),label=value['label'],saved_label=int(saved.loc[sid,'y_pred'])))
   raw.append(entry)
   save(method+'_progress.json',dict(completed=len(raw),total=90,last_seconds=seconds))
   if (i+1)%10==0:print(method,'measured',len(raw),'/90',flush=True)
peak=torch.cuda.max_memory_allocated(0);reserved=torch.cuda.max_memory_reserved(0)
result_summary=dict(method=method,checkpoint=str(result),sample_ids=selected,batch_size=1,warmup=5,repeats=3,peak_gpu_allocated_bytes=peak,peak_gpu_allocated_GiB=peak/2**30,peak_gpu_reserved_GiB=reserved/2**30,environment=environment(torch,True),quantization='none',precision='BF16 backbone; saved adapter/head dtypes',timing_includes='row DataFrame selection, native feature-only prompt, tokenization, H2D, inference, probability conversion or full reason+label generation, decoding and final-label parsing; CUDA sync before/after',timing_excludes='model loading, disk/network IO; output token counting after timer',**summarize([x['seconds'] for x in raw]))
result_summary['label_mismatches_vs_saved_on_30']=sum(x['label']!=x['saved_label'] for x in details[:30])
result_summary['input_tokens']={k:float(v) for k,v in [('min',min(x['input_tokens'] for x in raw)),('max',max(x['input_tokens'] for x in raw)),('mean',np.mean([x['input_tokens'] for x in raw]))]}
if method=='single_cot':
 n=[x['output_tokens'] for x in raw];result_summary['output_tokens']=dict(min=min(n),max=max(n),mean=float(np.mean(n)),p95=float(np.percentile(n,95)),includes_eos=True)
 result_summary['valid_final_labels']=sum(x['valid_final_label'] for x in raw);result_summary['ended_with_eos']=sum(x['ends_with_eos'] for x in raw);result_summary['exact_saved_text_matches']=sum(x['exact_saved_text'] for x in details[:30])
else:result_summary['max_abs_prob_diff_vs_saved']=max(abs(x['prob']-x['saved_prob']) for x in details)
save(method+'_summary.json',result_summary);save(method+'_outputs.json',details)
with (OUT/(method+'_raw.csv')).open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
print(json.dumps({k:v for k,v in result_summary.items() if k in ['method','mean_seconds','p95_seconds','sequential_batch1_requests_per_second','peak_gpu_allocated_GiB','output_tokens','valid_final_labels']}),flush=True)

