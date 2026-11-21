import argparse,json,shutil
from pathlib import Path
import numpy as np
from protocol import write_json,rationale_is_clean,TARGET

def shuffle_records(records,seed=44):
    accepted=[i for i,r in enumerate(records) if r.get('accepted')]
    if len(accepted)<2:raise ValueError('Need at least two accepted explanations to shuffle')
    order=np.random.RandomState(seed).permutation(accepted).tolist()
    mapping={order[i]:order[(i+1)%len(order)] for i in range(len(order))}
    return [{**record,'reason':records[mapping[i]]['reason'],'rationale_source_sample_id':records[mapping[i]]['sample_id'],'control':'random cyclic derangement of accepted explanations; classification labels retained'} if i in mapping else dict(record) for i,record in enumerate(records)]

def cot_only_membership(frame,records):
    labels=dict(zip(frame.sample_id,frame[TARGET].astype(int)))
    accepted=[]
    for record in records:
        if record['sample_id'] not in labels:raise ValueError('CoT outside training membership')
        if int(record['label'])!=labels[record['sample_id']]:raise ValueError('CoT label differs')
        if record.get('accepted'):
            if not rationale_is_clean(record.get('reason')):raise ValueError('Forbidden CoT content')
            accepted.append(record['sample_id'])
    return frame[frame.sample_id.isin(accepted)].copy()

def main(suite,variant):
    suite=Path(suite);source=suite/'experiments/02_main/data';data=suite/'experiments'/variant/'data';data.mkdir(parents=True,exist_ok=True)
    if (data/'variant.json').exists():
        print('Prepared variant retained',variant);return
    for name in ['train.csv','val.csv','test.csv','manifest.json','rationales.jsonl','shap_metadata.json','shap_model.json']:
        shutil.copy2(source/name,data/name)
    if variant.startswith('08_'):
        manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        manifest['manifest_role']='Immutable preparation membership before CoT-only filtering; final training membership is recorded in training_manifest.json, rationale_summary.json and model run_config.json'
        write_json(data/'manifest.json',manifest)
    if variant=='12_shuffle':
        records=[json.loads(x) for x in (data/'rationales.jsonl').read_text(encoding='utf-8').splitlines()]
        shuffled=shuffle_records(records)
        (data/'rationales.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in shuffled),encoding='utf-8')
    write_json(data/'variant.json',{'variant':variant,'seed':44,'primary_weighting':'adaptive','cot_only':variant.startswith('08_')})
    print('Prepared variant',variant,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--variant',required=True);args=parser.parse_args();main(args.suite,args.variant)
