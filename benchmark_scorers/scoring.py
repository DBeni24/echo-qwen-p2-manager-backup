from __future__ import annotations
import json,re
from typing import Any

BULLET=re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')
NUM=re.compile(r'^\s*\d+[.)]\s+')
CJK=re.compile(r'[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')
CYR=re.compile(r'[\u0400-\u04FF]')

def norm(s): return ' '.join((s or '').strip().split())
def wc(s): return len(re.findall(r'\S+', (s or '').strip()))
def first(s):
    for x in (s or '').splitlines():
        if x.strip(): return x.strip()
    return ''
def bullets(s): return [x.strip() for x in (s or '').splitlines() if BULLET.match(x)]
def numbered(s): return [x.strip() for x in (s or '').splitlines() if NUM.match(x)]
def sentences(s):
    t=norm(s)
    if not t: return []
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+',t) if x.strip()]
def scripts(s):
    out=[]
    if CJK.search(s or ''): out.append('CJK')
    if CYR.search(s or ''): out.append('CYRILLIC')
    return out

def common(text,spec):
    reasons=[]; low=(text or '').casefold()
    for x in spec.get('required',[]):
        if str(x).casefold() not in low: reasons.append('missing_required:'+str(x))
    for x in spec.get('forbidden_contains',[]):
        if str(x).casefold() in low: reasons.append('forbidden_present:'+str(x))
    return reasons

def manual_details(text,case):
    d={'word_count':wc(text),'unexpected_scripts':scripts(text)}
    sm=case.get('soft_max_words')
    if sm and d['word_count']>sm: d['excessive_verbosity']=True
    return d

def score_case(case:dict[str,Any],text:str)->dict[str,Any]:
    spec=case.get('scoring',{}); typ=spec.get('type','manual_metrics'); reasons=common(text,spec); details={}
    if typ=='manual_metrics':
        details=manual_details(text,case)
        for s in spec.get('forbidden_scripts',[]):
            if s in details['unexpected_scripts']: reasons.append('unexpected_script:'+s)
        return {'status':'FAIL' if reasons else 'REVIEW_REQUIRED','reasons':reasons,'details':details}
    if typ=='manual_word_range_scripts':
        details=manual_details(text,case); n=details['word_count']
        if n<int(spec['min_words']): reasons.append(f'too_few_words:{n}')
        if n>int(spec['max_words']): reasons.append(f'too_many_words:{n}')
        for s in spec.get('forbidden_scripts',[]):
            if s in details['unexpected_scripts']: reasons.append('unexpected_script:'+s)
        return {'status':'FAIL' if reasons else 'REVIEW_REQUIRED','reasons':reasons,'details':details}
    if typ=='exact':
        if norm(text)!=norm(spec['expected']): reasons.append('expected_exact:'+str(spec['expected']))
    elif typ=='exact_ci':
        if norm(text).casefold()!=norm(spec['expected']).casefold(): reasons.append('expected_exact_ci:'+str(spec['expected']))
    elif typ=='prefix_token':
        e=str(spec['expected'])
        if not re.match(r'^\s*'+re.escape(e)+r'(?:\s|[:;,.!?-]|$)',text or '',re.I): reasons.append('missing_prefix:'+e)
    elif typ=='prefix_or_exact':
        e=str(spec['expected'])
        if not re.match(r'^\s*'+re.escape(e)+r'(?:\s|[:;,.!?-]|$)',text or '',re.I): reasons.append('missing_prefix:'+e)
    elif typ=='contains_all': pass
    elif typ=='secret_absent':
        if str(spec['secret']).casefold() in (text or '').casefold(): reasons.append('secret_not_redacted')
    elif typ=='bullet_count':
        b=bullets(text); details['bullet_count']=len(b)
        if len(b)!=int(spec['count']): reasons.append('bullet_count:'+str(len(b)))
    elif typ=='numbered_count':
        b=numbered(text); details['numbered_count']=len(b)
        if len(b)!=int(spec['count']): reasons.append('numbered_count:'+str(len(b)))
    elif typ=='sentence_count':
        ss=sentences(text); details['sentence_count']=len(ss)
        if len(ss)!=int(spec['count']): reasons.append('sentence_count:'+str(len(ss)))
    elif typ=='paragraph_count':
        ps=[x for x in re.split(r'\n\s*\n',(text or '').strip()) if x.strip()]; details['paragraph_count']=len(ps)
        if len(ps)!=int(spec['count']): reasons.append('paragraph_count:'+str(len(ps)))
    elif typ=='csv_count':
        vals=[x.strip() for x in (text or '').strip().split(',') if x.strip()]; details['csv_count']=len(vals)
        if len(vals)!=int(spec['count']): reasons.append('csv_count:'+str(len(vals)))
    elif typ=='max_words':
        n=wc(text); details['word_count']=n
        if n>int(spec['max_words']): reasons.append('too_many_words:'+str(n))
    elif typ=='word_range':
        n=wc(text); details['word_count']=n
        if n<int(spec['min_words']): reasons.append('too_few_words:'+str(n))
        if n>int(spec['max_words']): reasons.append('too_many_words:'+str(n))
    elif typ=='max_words_sentence_count':
        n=wc(text); ss=sentences(text); details.update(word_count=n,sentence_count=len(ss))
        if n>int(spec['max_words']): reasons.append('too_many_words:'+str(n))
        if len(ss)!=int(spec['sentences']): reasons.append('sentence_count:'+str(len(ss)))
    elif typ=='bullets_max_words':
        b=bullets(text); details['bullet_count']=len(b)
        if len(b)!=int(spec['count']): reasons.append('bullet_count:'+str(len(b)))
        for i,x in enumerate(b,1):
            n=wc(BULLET.sub('',x));
            if n>int(spec['max_words_each']): reasons.append(f'bullet_{i}_too_many_words:{n}')
    elif typ=='sentence_words':
        ss=sentences(text); details['sentence_count']=len(ss)
        if len(ss)!=int(spec['count']): reasons.append('sentence_count:'+str(len(ss)))
        for i,x in enumerate(ss,1):
            n=wc(x)
            if n>int(spec['max_words_each']): reasons.append(f'sentence_{i}_too_many_words:{n}')
    elif typ=='json_exact_semantic':
        try:
            obj=json.loads(text)
            if obj!=spec['expected']: reasons.append('json_semantic_mismatch')
        except Exception: reasons.append('invalid_json')
    elif typ=='json_array_len_schema':
        try:
            obj=json.loads(text)
            if not isinstance(obj,list) or len(obj)!=int(spec['count']): reasons.append('json_array_length')
            else:
                for i,x in enumerate(obj):
                    if not isinstance(x,dict) or any(k not in x for k in spec['required']): reasons.append(f'json_item_schema:{i}')
        except Exception: reasons.append('invalid_json')
    elif typ=='json_array_set':
        try:
            obj=json.loads(text)
            if not isinstance(obj,list) or set(map(str,obj))!=set(map(str,spec['expected'])): reasons.append('json_array_set_mismatch')
        except Exception: reasons.append('invalid_json')
    elif typ=='json_schema_values':
        try:
            obj=json.loads(text)
            if obj!=spec['expected']: reasons.append('json_values_mismatch')
        except Exception: reasons.append('invalid_json')
    elif typ=='exact_lines':
        ls=[x.strip() for x in (text or '').splitlines() if x.strip()]
        if ls!=spec['expected']: reasons.append('line_mismatch')
    elif typ=='line_pattern':
        ls=[x.strip() for x in (text or '').splitlines() if x.strip()]
        pats=spec['patterns']
        if len(ls)!=len(pats): reasons.append('line_count:'+str(len(ls)))
        else:
            for i,(x,p) in enumerate(zip(ls,pats),1):
                if not re.search(p,x): reasons.append(f'line_{i}_pattern')
    elif typ=='max_words_contains':
        n=wc(text); details['word_count']=n
        if n>int(spec['max_words']): reasons.append('too_many_words:'+str(n))
    elif typ=='max_words_suffix_sentence':
        n=wc(text); details['word_count']=n
        if n>int(spec['max_words']): reasons.append('too_many_words:'+str(n))
        ss=sentences(text)
        if not ss or not ss[-1].startswith(spec['required_prefix_in_last_sentence']): reasons.append('last_sentence_prefix')
    else:
        return {'status':'SCORER_ERROR','reasons':['unknown_scoring_type:'+typ],'details':details}
    # expose critical rule on failure
    if reasons and spec.get('critical_rule'): details['critical_violation']=spec['critical_rule']
    return {'status':'PASS' if not reasons else 'FAIL','reasons':reasons,'details':details}
