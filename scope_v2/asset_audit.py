"""Read-only byte audit of already cached pinned model weights; never download weights."""
import argparse,collections,hashlib,json,struct
from pathlib import Path
from scope.runner import atomic,digest,now

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()

def audit(manifest,out):
    from huggingface_hub import hf_hub_download
    if out.exists():raise RuntimeError('immutable model-asset audit exists')
    data=json.loads(manifest.read_text());results=[]
    for m in data['models']:
        # The full repository snapshot can lack README/LICENSE despite complete weights.
        # Resolve one required cached file, then verify the actual weight/index files below.
        root=Path(hf_hub_download(m['id'],'config.json',revision=m['revision'],local_files_only=True)).parent
        files=sorted(root.glob('*.safetensors')) or sorted(root.glob('pytorch_model*.bin'))
        if not files:raise RuntimeError('BLOCKED cached weights missing '+m['key'])
        for index in root.glob('*.index.json'):
            obj=json.loads(index.read_text())
            if 'weight_map' in obj:
                expected=set(obj['weight_map'].values())
                if not expected.issubset({p.name for p in files}):raise RuntimeError('BLOCKED incomplete checkpoint '+m['key'])
        rows=[];dtypes=collections.Counter()
        for p in files:
            if p.suffix=='.safetensors':
                with p.open('rb') as f:
                    length=struct.unpack('<Q',f.read(8))[0]
                    if length>64*2**20:raise RuntimeError('invalid checkpoint header')
                    header=json.loads(f.read(length))
                dtypes.update(v['dtype'] for k,v in header.items() if k!='__metadata__')
            rows.append(dict(name=p.name,bytes=p.stat().st_size,sha256=file_hash(p)))
        result=dict(model=m['key'],id=m['id'],revision=m['revision'],files=rows,checkpoint_tensor_dtype_counts=dict(dtypes))
        results.append(result);print('AUDITED',m['key'],sum(x['bytes'] for x in rows),flush=True)
        atomic(out.with_suffix('.partial.json'),dict(status='IN_PROGRESS',manifest_sha256=digest(manifest.read_bytes()),models=results))
    atomic(out,dict(status='VERIFIED_CACHED_BYTES',manifest_sha256=digest(manifest.read_bytes()),models=results,
        created=now(),scope='Pinned cached checkpoint files; source tensors retained on server. Runtime dtype conversions remain separately declared.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.manifest,a.out)
