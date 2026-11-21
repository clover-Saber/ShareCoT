from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from analyze_preaudit import agreement

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    result={}
    for row in rows:
        key=row['item_id']
        if key in result: raise ValueError('Duplicate item: '+key)
        if not row['reviewer_id'].strip(): raise ValueError('Blank reviewer_id: '+key)
        for col in ('fact_score','grounding_score','coherence_score'):
            if row[col] not in ('0','1','2','3'): raise ValueError('Missing/invalid '+col+': '+key)
            row[col]=int(row[col])
        if row['conclusion'] not in ('accept','reject','unclear','contradictory'): raise ValueError('Missing/invalid conclusion: '+key)
        result[key]=row
    return result

if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('Usage: python summarize_human.py reviewer1.csv reviewer2.csv')
    a,b=map(read,sys.argv[1:])
    if set(a)!=set(b): raise ValueError('The two reviewers must annotate identical item IDs.')
    result={'n':len(a),'status':'computed_from_supplied_human_forms; verify qualifications separately','agreement':{}}
    for field in ('fact_score','grounding_score','coherence_score','conclusion'):
        result['agreement'][field]=agreement([a[k][field] for k in sorted(a)],[b[k][field] for k in sorted(a)],ordinal=field!='conclusion')
    Path('human_agreement_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
