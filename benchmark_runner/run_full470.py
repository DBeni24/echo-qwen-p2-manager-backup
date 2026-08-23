from __future__ import annotations
import argparse,json,random,re,shutil,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scorers.scoring import score_case

def loadj(p): return json.loads(p.read_text(encoding='utf-8'))
def loadjl(p):
    with p.open('r',encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def lms(args,timeout=600):
    p=subprocess.run(['lms']+args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    return p.returncode,(p.stdout or '')+(p.stderr or '')

def loaded_info(identifier):
    rc,out=lms(['ps','--json'])
    if rc!=0: return None,out
    try:
        arr=json.loads(out)
        for x in arr:
            if x.get('identifier')==identifier: return x,out
    except Exception: pass
    return None,out

def ensure_context(rt,target,events):
    info,_=loaded_info(rt['model_identifier'])
    if info and int(info.get('contextLength') or 0)==target:
        return True,'already_loaded'
    # Estimate first for larger bands
    rc,est=lms(['load','--estimate-only',rt['model_key'],'--gpu','max','--context-length',str(target)],timeout=600)
    events.append({'event':'estimate_context','context_length':target,'exit_code':rc,'output':est[-4000:]})
    if rc!=0: return False,'estimate_failed'
    lms(['unload','--all'],timeout=300)
    rc,out=lms(['load',rt['model_key'],'--gpu','max','--context-length',str(target),'--identifier',rt['model_identifier']],timeout=900)
    events.append({'event':'load_context','context_length':target,'exit_code':rc,'output':out[-4000:]})
    if rc!=0: return False,'load_failed'
    info,_=loaded_info(rt['model_identifier'])
    ok=bool(info and int(info.get('contextLength') or 0)==target)
    return ok,'loaded' if ok else 'verification_failed'

def rag_text(pack):
    blocks=[]
    for r in pack.get('records',[]):
        fields=[f"[{r['record_id']}]",f"status: {r['status']}",f"version: {r.get('version',1)}",f"provenance: {r.get('provenance','unknown')}"]
        if r.get('timestamp'): fields.append('timestamp: '+r['timestamp'])
        if r.get('conflict_group'): fields.append('conflict_group: '+r['conflict_group'])
        fields.append('content: '+r['content']); blocks.append('\n'.join(fields))
    return '\n\n'.join(blocks)

def filler_record(i):
    roles=['Small Worker','Manager','Observer','Build Worker','Index Worker']; states=['idle','busy','draining','unknown']
    return (f"REC-{i:05d}: node-{i%997:03d}; role={roles[i%len(roles)]}; state={states[(i*3)%len(states)]}; "
            f"service_port={7000+(i*37)%1900}; version={1+i%7}; lifecycle={'active' if i%11 else 'superseded'}; "
            f"owner=team-{i%13}; synthetic project metadata; unrelated_to_target=true.")

def make_long_context(case):
    s=case['context_spec']; target=int(s['target_chars']); mode=s['mode']; n=int(case['case_id'].split('-')[1]); lines=[]
    i=0
    while len('\n'.join(lines)) < target-2500:
        lines.append(filler_record(i+n*1000)); i+=1
    if mode in ('needle','many_distractors'):
        needle=f"NEEDLE-{n}: target-{n} canonical value = VALUE-{n}."
        pos=s.get('needle_position','middle'); idx={'start':2,'quarter':len(lines)//4,'middle':len(lines)//2,'three_quarter':3*len(lines)//4,'end':len(lines)-2}.get(pos,len(lines)//2)
        lines.insert(max(0,idx),needle)
    elif mode=='lifecycle':
        lines.insert(len(lines)//3,f"OLD-{n}: target lifecycle value = OLD-{n}; status=superseded.")
        lines.insert(2*len(lines)//3,f"ACTIVE-{n}: target lifecycle value = ACTIVE-{n}; status=active.")
    elif mode=='synthesis':
        lines.insert(len(lines)//5,f"SYN-A-{n}: synthesis part A = SYNTH; status=active.")
        lines.insert(len(lines)//2,f"SYN-B-{n}: synthesis part B = {n}; status=active.")
        lines.insert(4*len(lines)//5,f"SYN-C-{n}: synthesis rule = concatenate A + '-' + B; synthesis_result = SYNTH-{n}; status=active.")
    elif mode=='missing':
        lines.append('NOTE: A gamma-worker GPU típusa szerepelhet más rekordokban, de hőmérséklet mező nincs ebben a benchmark contextben.')
    elif mode=='conversation':
        band=s['band']; transcript=[
          'TURN-01 user: Válasszunk ALPHA vagy BETA tervet.',
          'TURN-02 assistant: Első körben ALPHA tűnik jobbnak.',
          'TURN-08 user: Új mérés alapján ALPHA regressziót okoz.',
          'TURN-14 assistant: Akkor BETA legyen a candidate.',
          f'TURN-20 user: Rendben, a végső elfogadott döntési token legyen DECISION-{band}.',
          f'TURN-21 assistant: Elfogadva: DECISION-{band}. A korábbi ALPHA/BETA ötletek közül ez a végső döntés.'
        ]
        step=max(1,len(lines)//len(transcript))
        for k,t in enumerate(transcript): lines.insert(min(len(lines),k*step+1),t)
    text='\n'.join(lines)
    if len(text)<target:
        text += '\n' + ('PAD: synthetic benchmark context metadata. '*((target-len(text))//42+1))
    return text[:target]

def final_msg(resp): return '\n'.join(x.get('content','') for x in resp.get('output',[]) if x.get('type')=='message').strip()
def reasoning_msg(resp): return '\n'.join(x.get('content','') for x in resp.get('output',[]) if x.get('type')=='reasoning').strip()

def make_request(case,rt,packs):
    parts=[]
    if case.get('rag_pack_id'):
        parts.append('=== RAG/KONTEXTUS ===\n'+rag_text(packs[case['rag_pack_id']])+'\n=== RAG/KONTEXTUS VÉGE ===')
    if case.get('context_spec'):
        parts.append('=== HOSSZÚ KONTEXTUS ===\n'+make_long_context(case)+'\n=== HOSSZÚ KONTEXTUS VÉGE ===')
    parts.append('=== FELADAT ===\n'+case['prompt'])
    return {'model':rt['model_identifier'],'input':'\n\n'.join(parts),'system_prompt':case['system_prompt'],'temperature':rt['temperature'],
      'reasoning':case['reasoning_mode'],'max_output_tokens':case['max_output_tokens'],'context_length':case['context_length'],'store':False}

def median(xs): return round(statistics.median(xs),4) if xs else None

def reports(run_dir,results,cases):
    status={}; cats={}; vis={}; critical=[]; review=[]; perf={}; ctx={}
    cmap={c['case_id']:c for c in cases}
    for r in results:
        st=r['status']; status[st]=status.get(st,0)+1
        cat=r['category']; cats.setdefault(cat,{}); cats[cat][st]=cats[cat].get(st,0)+1
        vv=r.get('visibility','?'); vis.setdefault(vv,{}); vis[vv][st]=vis[vv].get(st,0)+1
        det=(r.get('score') or {}).get('details') or {}
        if det.get('critical_violation'): critical.append({'case_id':r['case_id'],'violation':det['critical_violation'],'answer':r.get('final_answer','')})
        if st in ('REVIEW_REQUIRED','OUTPUT_BUDGET_HIT'): review.append({'case_id':r['case_id'],'category':cat,'word_count':det.get('word_count'),'status':st,'answer':r.get('final_answer','')})
        s=r.get('stats') or {}; perf.setdefault(cat,{'tps':[],'ttft':[]})
        if s.get('tokens_per_second') is not None: perf[cat]['tps'].append(float(s['tokens_per_second']))
        if s.get('time_to_first_token_seconds') is not None: perf[cat]['ttft'].append(float(s['time_to_first_token_seconds']))
        if cat=='CTX':
            band=str(cmap[r['case_id']]['context_length']); ctx.setdefault(band,{'cases':0,'pass':0,'input_tokens':[],'ttft':[]})
            ctx[band]['cases']+=1; ctx[band]['pass'] += 1 if st=='PASS' else 0
            if s.get('input_tokens') is not None: ctx[band]['input_tokens'].append(int(s['input_tokens']))
            if s.get('time_to_first_token_seconds') is not None: ctx[band]['ttft'].append(float(s['time_to_first_token_seconds']))
    summary={'status_counts':status,'category_counts':cats,'visibility_counts':vis,'critical_violations':critical,
      'performance':{k:{'median_tps':median(v['tps']),'median_ttft':median(v['ttft'])} for k,v in perf.items()},
      'context_bands':{k:{'cases':v['cases'],'pass':v['pass'],'pass_rate':round(v['pass']/v['cases'],4) if v['cases'] else None,
        'median_input_tokens':median(v['input_tokens']),'median_ttft':median(v['ttft'])} for k,v in ctx.items()}}
    (run_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (run_dir/'manual_review.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for x in review: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    # preliminary gap map from automated fails only
    fails=[r for r in results if r['status']=='FAIL']
    bysub={}
    for r in fails: bysub.setdefault(r.get('subcategory') or 'unknown',[]).append(r['case_id'])
    gap=['# Preliminary Training Gap Map','', '> Csak automatikus scorer eredmények. A HU/PERS/RSN manual review még nincs beolvasztva.','']
    for sub,ids in sorted(bysub.items(),key=lambda x:-len(x[1])): gap.append(f'- **{sub}**: {len(ids)} fail – '+', '.join(ids))
    if critical:
        gap += ['','## Critical violations','']+[f"- **{x['case_id']}** `{x['violation']}`" for x in critical]
    (run_dir/'TRAINING_GAP_PRELIM.md').write_text('\n'.join(gap)+'\n',encoding='utf-8')
    md=['# Echo Qwen Baseline Text470 – Report','',f'Run: `{run_dir.name}`','', '## Status','', '| Status | Count |','|---|---:|']
    for k,v in sorted(status.items()): md.append(f'| {k} | {v} |')
    md += ['','## Kategóriák','','| Category | PASS | FAIL | Review | Budget | Runtime/Other |','|---|---:|---:|---:|---:|---:|']
    for cat in sorted(cats):
        d=cats[cat]; other=sum(v for k,v in d.items() if k not in ('PASS','FAIL','REVIEW_REQUIRED','OUTPUT_BUDGET_HIT'))
        md.append(f"| {cat} | {d.get('PASS',0)} | {d.get('FAIL',0)} | {d.get('REVIEW_REQUIRED',0)} | {d.get('OUTPUT_BUDGET_HIT',0)} | {other} |")
    md += ['','## Critical violations','']
    md += [f"- **{x['case_id']}** `{x['violation']}`" for x in critical] if critical else ['- Nincs automatikusan detektált critical violation.']
    md += ['','## Long-context','', '| Context | Pass | Cases | Median input tokens | Median TTFT |','|---:|---:|---:|---:|---:|']
    for k,v in sorted(summary['context_bands'].items(),key=lambda x:int(x[0])): md.append(f"| {k} | {v['pass']} | {v['cases']} | {v['median_input_tokens']} | {v['median_ttft']} |")
    (run_dir/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--new-run',action='store_true'); ap.add_argument('--only-category'); ap.add_argument('--limit',type=int)
    a=ap.parse_args(); rt=loadj(ROOT/'config/runtime.json'); cases=loadjl(ROOT/'cases/full470.jsonl'); packs={x['pack_id']:x for x in loadjl(ROOT/'rag_packs/all_rag_packs.jsonl')}
    if a.only_category: cases=[c for c in cases if c['category']==a.only_category]
    if a.limit: cases=cases[:a.limit]
    statep=ROOT/'runs'/'current_run.json'
    if statep.exists() and not a.new_run:
        state=loadj(statep); run_dir=Path(state['run_dir'])
    else:
        run_dir=ROOT/'runs'/('full470_'+datetime.now().strftime('%Y%m%d_%H%M%S')); run_dir.mkdir(parents=True,exist_ok=False)
        state={'run_dir':str(run_dir),'started_at':datetime.now().isoformat(),'complete':False}; statep.write_text(json.dumps(state,indent=2),encoding='utf-8')
    respath=run_dir/'results.jsonl'; done={}
    if respath.exists():
        for r in loadjl(respath): done[r['case_id']]=r
    events=[]; url=rt['api_base'].rstrip('/')+rt['endpoint']; print('Run:',run_dir); print('Remaining:',sum(c['case_id'] not in done for c in cases))
    # Snapshot the exact suite/runtime used into the run artifact for auditability.
    if not (run_dir/'cases_snapshot.jsonl').exists():
        shutil.copy2(ROOT/'cases/full470.jsonl', run_dir/'cases_snapshot.jsonl')
        shutil.copy2(ROOT/'rag_packs/all_rag_packs.jsonl', run_dir/'rag_packs_snapshot.jsonl')
        shutil.copy2(ROOT/'cases/suite_manifest.json', run_dir/'suite_manifest.json')
        shutil.copy2(ROOT/'holdout/holdout_manifest.json', run_dir/'holdout_manifest.json')
        shutil.copy2(ROOT/'config/runtime.json', run_dir/'runtime_config.json')
    with httpx.Client(timeout=httpx.Timeout(rt['request_timeout_seconds'],connect=10)) as cli, respath.open('a',encoding='utf-8',newline='\n') as out:
        current_ctx=None
        failed_contexts=set()
        for idx,c in enumerate(cases,1):
            if c['case_id'] in done: continue
            target=int(c['context_length'])
            if target in failed_contexts:
                r={'case_id':c['case_id'],'category':c['category'],'subcategory':c['subcategory'],'visibility':c['visibility'],'severity':c['severity'],'status':'RESOURCE_ERROR','error':'context_band_unavailable','request_meta':{'context_length':target}}
                out.write(json.dumps(r,ensure_ascii=False)+'\n'); out.flush(); done[c['case_id']]=r; print(c['case_id'],'RESOURCE_ERROR context_band_unavailable'); continue
            if current_ctx!=target:
                ok,msg=ensure_context(rt,target,events); current_ctx=target if ok else None
                if not ok:
                    failed_contexts.add(target)
                    r={'case_id':c['case_id'],'category':c['category'],'subcategory':c['subcategory'],'visibility':c['visibility'],'severity':c['severity'],'status':'RESOURCE_ERROR','error':msg,'request_meta':{'context_length':target}}
                    out.write(json.dumps(r,ensure_ascii=False)+'\n'); out.flush(); done[c['case_id']]=r; print(c['case_id'],'RESOURCE_ERROR',msg); continue
            rq=make_request(c,rt,packs); print(f"[{idx:03d}/{len(cases):03d}] {c['case_id']} {c['category']} ctx={target} ... ",end='',flush=True); t=time.time()
            r={'case_id':c['case_id'],'category':c['category'],'subcategory':c['subcategory'],'visibility':c['visibility'],'severity':c['severity'],'request_meta':{'context_length':target,'max_output_tokens':c['max_output_tokens'],'reasoning':c['reasoning_mode']}}
            try:
                z=cli.post(url,json=rq); r['http_status']=z.status_code; r['elapsed_wall_seconds']=time.time()-t
                if z.status_code!=200:
                    txt=z.text[:5000]; r['error']=txt; r['status']='CONTEXT_TIMEOUT' if 'timeout' in txt.lower() else 'RESOURCE_ERROR' if 'context' in txt.lower() or 'memory' in txt.lower() else 'RUNTIME_ERROR'
                else:
                    raw=z.json(); final=final_msg(raw); r['final_answer']=final; r['reasoning_text']=reasoning_msg(raw); r['stats']=raw.get('stats',{}); r['model_instance_id']=raw.get('model_instance_id')
                    if not final: r['status']='MODEL_FAILURE'; r['score']={'status':'MODEL_FAILURE','reasons':['empty_final_answer'],'details':{}}
                    elif r['stats'].get('total_output_tokens',0)>=c['max_output_tokens']:
                        r['status']='OUTPUT_BUDGET_HIT'; r['score']={'status':'OUTPUT_BUDGET_HIT','reasons':['max_output_tokens_reached'],'details':{'word_count':len(final.split())}}
                    else: r['score']=score_case(c,final); r['status']=r['score']['status']
            except httpx.TimeoutException as e: r['status']='CONTEXT_TIMEOUT'; r['error']=str(e)
            except Exception as e: r['status']='RUNTIME_ERROR'; r['error']=type(e).__name__+': '+str(e)
            print(r['status']); out.write(json.dumps(r,ensure_ascii=False)+'\n'); out.flush(); done[c['case_id']]=r
    # restore 8K
    ensure_context(rt,int(rt['restore_context_after_run']),events)
    results=[done[c['case_id']] for c in cases if c['case_id'] in done]
    (run_dir/'runtime_events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    reports(run_dir,results,cases)
    if len(results)==len(cases):
        state['complete']=True; state['completed_at']=datetime.now().isoformat(); statep.write_text(json.dumps(state,indent=2),encoding='utf-8')
        up=ROOT/'UPLOAD_ME_ECHO_FULL470'; zp=shutil.make_archive(str(up),'zip',root_dir=run_dir); print('UPLOAD ZIP:',zp)
    else: print('INCOMPLETE - rerun the same script to resume.')
if __name__=='__main__': main()
