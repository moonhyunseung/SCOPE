"""Minimal same-input projection-shape reproducer from a private diagnostic tensor."""
import argparse,json
from pathlib import Path
from scope_v2.diagnose import MODEL, REV, projection_probe

def run(raw_file,name,dtype,rows):
    import torch
    from transformers import AutoModelForCausalLM
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    # Model revision is fixed; BF16-vs-FP32 runs load from the same checkpoint.
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,
        trust_remote_code=False,dtype=getattr(torch,dtype),attn_implementation='eager').eval().to('cuda')
    raw=torch.load(raw_file,map_location='cpu',weights_only=True)
    module=getattr(model.model.layers[0].self_attn,name)
    with torch.inference_mode():
        r=projection_probe(module,raw['input'].to(device='cuda',dtype=getattr(torch,dtype)).unsqueeze(1),rows)
    print(json.dumps({k:v for k,v in r.items() if not isinstance(v,torch.Tensor)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw_file',type=Path);p.add_argument('--name',choices=['q_proj','k_proj','v_proj'],default='q_proj')
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16');p.add_argument('--rows',type=int,default=513);a=p.parse_args()
    run(a.raw_file,a.name,a.dtype,a.rows)
