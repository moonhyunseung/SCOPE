"""Bounded first-stage classification from sealed public diagnostic and mechanism blocks."""
import argparse,collections,csv,json
from pathlib import Path
from scope.runner import atomic,digest,now
def csv_write(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def classify(diagnostic,mechanism,out):
    if out.exists():raise RuntimeError('immutable classification exists')
    for d in [diagnostic,mechanism]:
        m=json.loads((d/'manifest.json').read_text())
        if not all(digest((d/n).read_bytes())==v['sha256'] for n,v in m['files'].items()):raise RuntimeError('input block hash mismatch')
    ds=json.loads((diagnostic/'summary.json').read_text());ms=json.loads((mechanism/'summary.json').read_text())
    dt={(r['case_id'],r['dtype']):r['token_sha256'] for r in ds['conditions']}
    if not all(dt[(r['case_id'],r['dtype'])]==r['token_sha256'] for r in ms['results']):raise RuntimeError('cross-block tokens differ')
    detail=json.loads((mechanism/'details.json').read_text())
    index={(r['case_id'],r['dtype'],r['step'],r['operation'],r['reduced_precision_reduction']):r for r in detail}
    first=list(csv.DictReader((diagnostic/'first_divergence.csv').open()))
    mapping={'L00.mlp_down':'mlp_down','L00.scores':'QK','L00.attention_out':'PV'};rows=[]
    for x in first:
        if x['dtype']!='bfloat16' or x['contrast']!='cached_full':continue
        op=mapping.get(x['first_nonzero']);r=index.get((x['case_id'],'bfloat16',int(x['step']),op,True))
        replay=(r.get('one_vs_native') or r.get('native_replay')) if r else None
        valid=bool(r and r.get('upstream_input_exact',True) and replay['max_abs']==0 and r['one_vs_many']['max_abs']>0)
        rows.append(dict(case_id=x['case_id'],step=x['step'],first_stage=x['first_nonzero'],operation=op,
            classification='MATCHED_INPUT_SHAPE_EFFECT_DEMONSTRATED' if valid else 'UNRESOLVED',
            shape_relative_l2=r['one_vs_many']['relative_l2'] if r else None,
            native_proxy_exact=bool(replay and replay['max_abs']==0)))
    changes=0
    for r in detail:
        if r['reduced_precision_reduction']:
            b=index[(r['case_id'],r['dtype'],r['step'],r['operation'],False)]
            changes+=r['one_vs_many']!=b['one_vs_many']
    out.mkdir(parents=True);csv_write(out/'metrics.csv',rows)
    summary=dict(status='VERIFIED_BLOCK',traces=len(rows),conditions=len({r['case_id'] for r in rows}),source_units=ds['independent_source_units'],
        classified=sum(r['classification']!='UNRESOLVED' for r in rows),unresolved_first_stage=sum(r['classification']=='UNRESOLVED' for r in rows),
        first_stage_counts=dict(collections.Counter(r['first_stage'] for r in rows)),
        reduction_mode_comparison_pairs=len(detail)//2,recorded_shape_error_metrics_changed=changes,
        original_failed_conditions_directly_sampled=8,original_failed_conditions_not_directly_sampled=445,
        diagnostic_manifest_sha256=digest((diagnostic/'manifest.json').read_bytes()),mechanism_manifest_sha256=digest((mechanism/'manifest.json').read_bytes()),
        scope='First exact nonzero stage at prespecified steps 0,3,7 in 16 outcome-stratified BF16 conditions. '
              'Does not exhaustively attribute final-logit error, classify all historical failures, identify the internal GEMM algorithm, '
              'or prove bitwise equality across reduction modes. FP32 first-stage norm/projection differences are bounded separately by passing final fidelity.',
        publication='PENDING')
    atomic(out/'summary.json',summary)
    (out/'report.md').write_text('# Bounded native BF16 mechanism classification\n\n'+json.dumps(summary,indent=2)+'\n')
    files={p.name:dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}
    atomic(out/'manifest.json',dict(files=files,sealed=now()));(out/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('diagnostic',type=Path);p.add_argument('mechanism',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    print(json.dumps(classify(a.diagnostic,a.mechanism,a.out),indent=2))
