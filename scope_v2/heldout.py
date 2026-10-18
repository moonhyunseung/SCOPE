"""Freeze a new source population, disjoint from the historical 512 source units."""
import argparse,collections,gzip,json,re
from pathlib import Path
from scope.runner import atomic,canonical,digest
from scope.population_inputs import shingles,near_duplicate
from scope_v2.diagnose import INPUT_HASH

MODELS=[
 dict(key='qwen1_5b',id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
 dict(key='pythia1b',id='EleutherAI/pythia-1b',revision='f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2'),
 dict(key='opt1_3b',id='facebook/opt-1.3b',revision='3f5c25d0bc631cb57ac65913f76e22c2dfb61d62'),
 dict(key='qwen7b',id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28'),
 dict(key='qwen14b',id='Qwen/Qwen2.5-14B-Instruct',revision='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8')]
SEED=20261019
def rank(s):return digest((str(SEED)+':'+s).encode())

def build(old_path,out,per_domain=64):
    from transformers import AutoTokenizer,AutoConfig
    import pyarrow.parquet as pq
    if out.exists():raise RuntimeError('immutable held-out manifest exists')
    if digest(old_path.read_bytes())!=INPUT_HASH:raise RuntimeError('historical input drift')
    old=json.loads(old_path.read_text());excluded={(d['domain'],d['source_unit']) for d in old['documents']}
    known_hashes={d['normalized_text_sha256'] for d in old['documents']}
    previous=[shingles(d['window_tokens']) for d in old['documents']]
    tokenizers={};model_specs=[];unsupported=[]
    for m in MODELS:
        try:
            tok=AutoTokenizer.from_pretrained(m['id'],revision=m['revision'],local_files_only=True,trust_remote_code=False)
            cfg=AutoConfig.from_pretrained(m['id'],revision=m['revision'],local_files_only=True,trust_remote_code=False)
            tokenizers[m['key']]=tok
            heads=getattr(cfg,'num_attention_heads',None);kv=getattr(cfg,'num_key_value_heads',heads)
            model_specs.append(dict(m,model_type=cfg.model_type,num_attention_heads=heads,num_key_value_heads=kv,
                max_position_embeddings=getattr(cfg,'max_position_embeddings',None),attention_backend='eager'))
        except Exception as e:unsupported.append(dict(m,status='UNSUPPORTED_LOCAL_ASSET',reason=type(e).__name__))
    if 'qwen1_5b' not in tokenizers:raise RuntimeError('canonical tokenizer unavailable')
    groups={};sources=[]
    for source in old['sources']:
        path=Path(source['path'])
        if not path.is_file() or digest(path.read_bytes())!=source['sha256']:raise RuntimeError('frozen source asset drift')
        sources.append({k:v for k,v in source.items() if k!='path'})
        if source['domain']=='prose':
            g={};title=None;parts=[]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['text']):
                for text in batch.column(0).to_pylist():
                    match=re.match(r'^\s*=\s+([^=]+?)\s+=\s*$',text)
                    if match:
                        if title and parts:g.setdefault(title,[]).append(dict(key=title,text=''.join(parts),license=source['license']))
                        title=match.group(1).strip();parts=[text]
                    elif title:parts.append(text)
            groups['prose']=g  # Exclude final potentially truncated article.
        else:
            g=collections.defaultdict(list);allowed={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','unlicense','cc0-1.0'}
            with gzip.open(path,'rt') as f:
                for line in f:
                    r=json.loads(line);lic=str(r.get('license','')).lower();text=r['content']
                    if lic not in allowed or not 5000<=len(text)<=500000:continue
                    repo=r['repo_name'];unit=repo.rsplit('/',1)[-1].lower()
                    if ('code',unit) in excluded:continue
                    g[unit].append(dict(key=repo+':'+r['path'],repo=repo,path=r['path'],text=text,license=lic))
            groups['code']=g
    docs=[];counts={}
    for domain,g in groups.items():
        selected=[];count=collections.Counter(candidate_source_units=len(g))
        for unit in sorted(g,key=rank):
            if (domain,unit) in excluded:count['historical_unit_excluded']+=1;continue
            for item in sorted(g[unit],key=lambda x:rank(x['key'])):
                text=item['text'];norm_hash=digest(' '.join(text.split()).encode())
                if norm_hash in known_hashes:count['duplicate_content']+=1;continue
                if not 5000<=len(text)<=500000:continue
                start=int(rank(item['key']+':chars'),16)%max(1,len(text)-24000+1)
                segment=text[start:start+24000];windows={};eligible=True
                for key,tok in tokenizers.items():
                    ids=tok.encode(segment,add_special_tokens=False)
                    if len(ids)<2057:eligible=False;break
                    window=ids[-2057:];windows[key]=dict(tokens=window,sha256=digest(canonical(window).encode()))
                if not eligible:count['short_tokens']+=1;continue
                sh=shingles(windows['qwen1_5b']['tokens'])
                if near_duplicate(sh,previous):count['near_duplicate_window']+=1;continue
                i=len(selected)
                selected.append(dict(document_id=digest((domain+':'+unit).encode())[:20],domain=domain,source_unit=unit,
                    source_key=item['key'],source_repo=item.get('repo'),source_path=item.get('path'),license=item['license'],
                    source_text_sha256=digest(text.encode()),normalized_text_sha256=norm_hash,
                    segment_start_char=start,segment_sha256=digest(segment.encode()),windows=windows,
                    split='screening' if i<8 else 'confirmatory'))
                previous.append(sh);known_hashes.add(norm_hash);break
            if len(selected)==per_domain:break
        if len(selected)!=per_domain:raise RuntimeError(f'insufficient new {domain} sources: {len(selected)}/{per_domain}')
        docs+=selected;counts[domain]=dict(count,selected=len(selected));print('SELECTED',domain,len(selected),flush=True)
    docs=[x for pair in zip([d for d in docs if d['domain']=='prose'],[d for d in docs if d['domain']=='code']) for x in pair]
    payload=dict(protocol='heldout-native-factorial-v2',seed=SEED,models=model_specs,unsupported_models=unsupported,
        historical_manifest_sha256=INPUT_HASH,excluded_source_units=len(excluded),source_units=len(docs),counts=counts,sources=sources,
        documents=docs,scope='New source units disjoint by article title/conservative repository-name family; residual topic/project lineage dependence remains possible. Bounded licensed dataset shards, not a representative global population.',
        cross_model_pairing='Same raw character segment and source endpoint; each pinned tokenizer takes its own final 2057 natural tokens. Cross-vocabulary target tokenization differs; contrasts are within model.',
        confirmation_length=1536,
        primary='Confirmatory Pythia and OPT: mean of 8 paired step errors, K-only minus V-only at BF16 L1536 INT4 g32, separately prose/code and isolated/recurrent. Eight prespecified contrasts. L1536 plus 8 decode positions stays within their 2048-position limits. Other architectures/parameter probes exploratory.',
        uncertainty='Source-block bootstrap; steps/heads/layers/lengths/dtypes/treatments/models are repeated measures. Independent source n is not multiplied by models.')
    atomic(out,payload);out.with_suffix('.sha256').write_text(digest(out.read_bytes())+'  '+out.name+'\n')
    index=dict(payload,documents=[dict(d,windows={k:dict(sha256=v['sha256'],token_count=len(v['tokens'])) for k,v in d['windows'].items()}) for d in docs])
    atomic(out.parent/'public_index.json',index);print('FROZEN',len(docs),digest(out.read_bytes()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--historical',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--per-domain',type=int,default=64);a=p.parse_args();build(a.historical,a.out,a.per_domain)
