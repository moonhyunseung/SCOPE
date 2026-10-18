"""Freeze independent source units and natural matched context windows before GPU outcomes."""
import collections,gzip,hashlib,json,re,sys
from pathlib import Path
from scope.runner import ROOT,atomic,canonical,digest
SEED=20261018
SOURCES=[dict(domain='prose',repo='Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',file='wikitext-103-raw-v1/train-00000-of-00002.parquet',license='CC-BY-SA-3.0/GFDL'),dict(domain='code',repo='codeparrot/codeparrot-clean-valid',revision='4db92d2ec0c1b4c41eeb439cfae16854511d9dcd',file='file-000000000054.json.gz')]
def rank(x):return digest((str(SEED)+':'+x).encode())
def window(ids,key):
    if len(ids)<2057:raise ValueError('short source')
    start=int(rank(key+':window'),16)%(len(ids)-2057+1)
    return ids[start:start+2057],start
def shingles(ids):return set(zip(*(ids[i:] for i in range(5))))
def near_duplicate(a,previous):
    for b in previous:
        intersect=len(a&b)
        if intersect/max(1,len(a)+len(b)-intersect)>=.8:return True
    return False
def build():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    out=ROOT/'data/population_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('frozen manifest already exists; refuse overwrite')
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=True)
    candidates={};provenance=[]
    for source in SOURCES:
        print('DOWNLOAD',source,flush=True)
        path=Path(hf_hub_download(source['repo'],source['file'],repo_type='dataset',revision=source['revision']))
        provenance.append(dict(source,path=str(path),sha256=digest(path.read_bytes()),bytes=path.stat().st_size))
        if source['domain']=='prose':
            groups={};title=None;parts=[];row_start=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['text']):
                for text in batch.column(0).to_pylist():
                    match=re.match(r'^\s*=\s+([^=]+?)\s+=\s*$',text)
                    if match:
                        if title and parts:groups.setdefault(title,[]).append(dict(key=title,text=''.join(parts),license=source['license']))
                        title=match.group(1).strip();parts=[text]
                    elif title:parts.append(text)
            # Last article in a bounded shard may be truncated: deliberately exclude it.
            candidates['prose']=groups
        else:
            groups=collections.defaultdict(list)
            allowed={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','unlicense','cc0-1.0'}
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    row=json.loads(line);text=row['content'];license=str(row.get('license','')).lower()
                    if license not in allowed or not 5000<=len(text)<=500000:continue
                    repo=row['repo_name'];family=repo.rsplit('/',1)[-1].lower()
                    groups[family].append(dict(key=repo+':'+row['path'],repo=repo,path=row['path'],text=text,license=license))
            candidates['code']=groups
        print('CANDIDATE_UNITS',source['domain'],len(candidates[source['domain']]),flush=True)
    docs=[];counts={};seen_hashes=set()
    for domain,groups in candidates.items():
        selected=[];previous=[];count=collections.Counter(candidate_units=len(groups))
        for unit in sorted(groups,key=rank):
            chosen=None
            for item in sorted(groups[unit],key=lambda x:rank(x['key'])):
                text=item['text'];text_hash=digest(' '.join(text.split()).encode())
                if text_hash in seen_hashes:count['duplicate_content']+=1;continue
                if not 5000<=len(text)<=500000:count['character_filter']+=1;continue
                ids=tok.encode(text,add_special_tokens=False)
                if len(ids)<2057:count['short_tokens']+=1;continue
                ids,start=window(ids,item['key']);sh=shingles(ids)
                if near_duplicate(sh,previous):count['near_duplicate_window']+=1;continue
                chosen=dict(domain=domain,source_unit=unit,document_id=digest((domain+':'+unit).encode())[:20],source_key=item['key'],source_text_sha256=digest(text.encode()),normalized_text_sha256=text_hash,window_start_token=start,window_tokens=ids,window_sha256=digest(canonical(ids).encode()),license=item['license'],source_repo=item.get('repo'),source_path=item.get('path'))
                previous.append(sh);seen_hashes.add(text_hash);break
            count['units_examined']+=1
            if chosen:
                i=len(selected);chosen.update(split='exploratory' if i%2==0 else 'confirmatory',dtype_audit=i<32,retain_raw=i==0,variant_order=i%2)
                selected.append(chosen)
                if len(selected)%32==0:print('SELECTED',domain,len(selected),dict(count),flush=True)
                if len(selected)==256:break
        if len(selected)!=256:raise RuntimeError(f'insufficient eligible {domain} units: {len(selected)}; do not repeat sources')
        counts[domain]=dict(count,selected=len(selected));docs.extend(selected)
    # Interleave domains so resource truncation cannot collect only one domain.
    docs=[d for pair in zip([d for d in docs if d['domain']=='prose'],[d for d in docs if d['domain']=='code']) for d in pair]
    payload=dict(protocol='population-inputs-v1',seed=SEED,sources=provenance,counts=counts,documents=docs,sampling='Hash-rank source units before outcomes; one natural 2057-token window per article/project-name family. Bounded dataset shards, not a representative sample of all text/code. Exact content and >=0.8 token-5gram Jaccard window duplicates excluded. Conservative repo basename grouping does not prove all upstream lineages independent.',unit='article or conservative repository-name family',length_pairing='last 512 vs 2048 preceding identical natural 9-token continuation',tokenizer_revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    atomic(out/'manifest.json',payload);(out/'manifest.sha256').write_text(digest((out/'manifest.json').read_bytes())+'  manifest.json\n')
    print('FROZEN',len(docs),counts,digest((out/'manifest.json').read_bytes()),flush=True)
if __name__=='__main__':build()
