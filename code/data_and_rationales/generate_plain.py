from base64 import urlsafe_b64encode
import argparse,concurrent.futures,datetime,json,re,threading,time
from pathlib import Path

SYSTEM='''Given the passage, question, and candidate answer, write a short explanation (2–5 sentences) that justifies the gold label. Let's think step by step.
Rules: (1) Do not include "Yes/No", "True/False", or synonyms in the explanation; (2) End the explanation with [END OF REASONING]; (3) The explanation should be a short paragraph. On a new line, output Answer: Yes or Answer: No.
Use the following output format:
Reasoning:
<short explanation> [END OF REASONING]
Answer: <Yes|No>'''

def build_messages(item):
    if item['label'] not in (0,1):raise ValueError('Invalid label')
    user=('Task: This is ridesharing/carpooling authorization classification, not textual entailment. Authorization means agreeing to share the vehicle ride with other passengers; it does not mean permission to share personal data or trip records. The gold label is the observed authorization outcome. Explain how the recorded features may support this outcome; do not treat absence of an explicitly stated outcome in the feature list as evidence for either class.\n'
          +'Passage:\nRecorded trip features: '+json.dumps(item['features'],ensure_ascii=False)
          +'\nUnits: fare and additional charges in USD; trip distance in miles; duration in seconds; city-center distances in km; dewpoint in degrees Celsius. Missing values are represented by null or nan.'
          +'\nQuestion: Did the passenger authorize sharing the vehicle ride with other passengers?'
          +'\nCandidate answer: The passenger authorized a shared vehicle ride with other passengers.'
          +'\nGold label: '+('Yes' if item['label']==1 else 'No'))
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]

def request_messages(item,attempts):
    messages=build_messages(item)
    if any(x.get('error') in {'sentence_count','answer_format','answer_label_mismatch','missing_or_duplicate_sentinel','empty_reason'} for x in attempts):
        messages.append({'role':'user','content':'Formatting reminder: Write 2–5 sentences in the explanation. End the explanation with [END OF REASONING], then put Answer: Yes or Answer: No on a separate line, matching the supplied gold label. Provide only the requested explanation and answer.'})
    return messages

def parse_response(raw,label):
    if raw.count('[END OF REASONING]')!=1:return {'valid':False,'error':'missing_or_duplicate_sentinel'}
    reason,answer=raw.split('[END OF REASONING]')
    reason=re.sub(r'^\s*Reasoning\s*:\s*','',reason,flags=re.I).strip()
    match=re.fullmatch(r'\s*Answer\s*:\s*(Yes|No)\s*[.]?\s*',answer,re.I)
    if not match:return {'valid':False,'error':'answer_format'}
    pred=int(match.group(1).lower()=='yes')
    if pred!=label:return {'valid':False,'error':'answer_label_mismatch'}
    label_words=re.findall(r'\b(yes|no|true|false)\b',reason.lower())
    sentences=[x for x in re.split(r'(?<=[.!?])\s+',reason) if x.strip()]
    if not 2<=len(sentences)<=5:return {'valid':False,'error':'sentence_count','sentence_count':len(sentences)}
    if not reason:return {'valid':False,'error':'empty_reason'}
    return {'valid':True,'reason':reason,'pred':pred,'sentence_count':len(sentences),'label_words':label_words}

def validate_complete(items,records):
    expected={x['sample_id']:x['label'] for x in items}
    actual={x['sample_id']:x for x in records}
    if len(actual)!=len(records) or set(actual)!=set(expected):raise ValueError('Missing, duplicate, or extra IDs')
    for sid,label in expected.items():
        r=actual[sid]
        if r['label']!=label or r.get('accepted') is not True or not r.get('reason','').strip():raise ValueError('Incomplete or mismatched record '+sid)

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def run(root,workers=8,limit=None):
    root=Path(root);out=root/'generation';cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'frozen_protocol.json').read_text())
    items=[json.loads(x) for x in (root/'data/teacher_items.jsonl').read_text().splitlines()]
    assert len(items)==946 and len({x['sample_id'] for x in items})==946
    from api_suite import client_for_suite
    client,key=client_for_suite();lock=threading.Lock();stop=threading.Event()
    started=time.time();counts={'calls_this_invocation':0}
    selected=items[:limit] if limit is not None else items
    def report(stage):
        records=[json.loads(p.read_text()) for p in cache.glob('*.json')]
        write(out/'status.json',{'stage':stage,'target':946,'valid':sum(x.get('accepted',False) for x in records),'cached':len(records),'calls_this_invocation':counts['calls_this_invocation'],'workers':workers,'elapsed_seconds':time.time()-started,'teacher':protocol['teacher_model'],'human_reviewed':0,'no_semantic_filter':True})
    def process(item):
        sid=item['sample_id'];path=cache/(urlsafe_b64encode(sid.encode()).decode()+'.json')
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous.get('accepted'):return previous
        attempts=list(previous.get('attempts',[]))
        for attempt in range(len(attempts),protocol['max_requests_per_item']):
            if stop.is_set():break
            entry={'sample_id':sid,'attempt':attempt+1,'requested_model':protocol['teacher_model'],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            t0=time.time();value={}
            try:
                messages=request_messages(item,attempts)
                entry['format_reminder']=len(messages)>2
                response=client.chat.completions.create(model=protocol['teacher_model'],messages=messages,temperature=protocol['temperature'],max_completion_tokens=protocol['max_completion_tokens'])
                raw=response.choices[0].message.content or ''
                entry.update(raw=raw,actual_model=response.model,request_id=response.id,usage=response.usage.model_dump() if response.usage else None,finish_reason=response.choices[0].finish_reason)
                value=parse_response(raw,item['label']);entry.update(valid=value['valid'],validation=value)
                if not value['valid']:entry['error']=value['error']
            except Exception as exc:
                message=str(exc).replace(key,'[redacted]')[:700]
                entry.update(valid=False,error=message)
                if any(x in message.lower() for x in ['insufficient_quota','insufficient_user_quota','insufficient balance','invalid_api_key','model_not_found']):stop.set()
            entry['seconds']=time.time()-t0;attempts.append(entry)
            record={'sample_id':sid,'label':item['label'],'accepted':entry['valid'],'attempts':attempts,'human_reviewed':False,'review_policy':'format_and_gold_answer_only; no semantic judge'}
            if entry['valid']:record.update(reason=value['reason'],pred=value['pred'],teacher_model=entry['actual_model'],label_words=value['label_words'],selected_attempt=attempt+1)
            write(path,record)
            with lock:
                counts['calls_this_invocation']+=1
                with (out/'api_calls.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(entry,ensure_ascii=False)+'\n')
                report('generating')
            if entry['valid']:return record
            if not stop.is_set():time.sleep(min(2**attempt,8))
        return json.loads(path.read_text()) if path.exists() else {'sample_id':sid,'accepted':False}
    try:
        report('generating')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool: results=list(pool.map(process,selected))
        if limit is not None:
            report('preview_complete');return all(x.get('accepted') for x in results)
        validate_complete(items,results)
        target=root/'data/rationales.jsonl';temp=target.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8');temp.replace(target)
        write(out/'completion.json',{'status':'complete','n':len(results),'teacher_requested':protocol['teacher_model'],'teacher_returned':sorted({x['teacher_model'] for x in results}),'validation':'format and answer only','no_attribution_inputs':True})
        report('complete');return True
    except Exception:
        report('incomplete_requires_retry');raise
    finally:client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);args=p.parse_args()
    raise SystemExit(0 if run(args.root,args.workers,args.limit) else 1)
