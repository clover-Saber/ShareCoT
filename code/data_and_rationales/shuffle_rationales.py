import argparse,json
from pathlib import Path
import numpy as np

def shuffle_records(records, seed=44):
    accepted = [i for i,r in enumerate(records) if r.get('accepted') is True]
    if len(accepted) < 2: raise ValueError('At least two accepted rationale records required')
    order = np.random.RandomState(seed).permutation(accepted).tolist()
    donors = {order[i]:order[(i+1)%len(order)] for i in range(len(order))}
    return [{**r, 'reason':records[donors[i]]['reason'],
             'rationale_source_sample_id':records[donors[i]]['sample_id'],
             'control':'seed44 random cyclic derangement; recipient classification label retained',
             'judge_verdict_scope':'original feature/rationale pair only; shuffled pair not reviewed'}
            if i in donors else dict(r) for i,r in enumerate(records)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError('Use a new output file')
    rows=[json.loads(l) for l in Path(args.input).read_text(encoding='utf-8').splitlines() if l.strip()]
    if len({str(r['sample_id']) for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    shuffled=shuffle_records(rows,seed=44)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in shuffled),encoding='utf-8')
if __name__=='__main__':main()
