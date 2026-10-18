"""Seal a compact immutable diagnostic block without publishing private tensors/paths."""
import argparse,collections,csv,json,shutil,sqlite3,zipfile
from pathlib import Path
from scope.runner import atomic,digest,canonical,now
from scope_v2.diagnose import ROOT,MODEL,REV,verify,csv_write,seal

def export(folder,label,expected):
    folder=folder.resolve()
    import numpy as np
    campaign=json.loads((folder/'campaign.json').read_text());c=campaign['config']
    db=sqlite3.connect(folder/'registry.sqlite');records=[];stages=[];distribution=[];first=[];probes=[]
    for ident,status,attempt in db.execute('SELECT id,status,attempt FROM jobs ORDER BY id'):
        if status not in ['COLLECTED','FAIL']:continue
        dest=folder/ident/attempt;r=verify(dest);records.append(r)
        if status=='COLLECTED':
            with (dest/'stages.csv').open() as f:stages.extend(csv.DictReader(f))
            distribution+=json.loads((dest/'attention_distributions.json').read_text())
            first+=json.loads((dest/'first_divergence.json').read_text())
            probes += [dict(case_id=r['case_id'],dtype=r['dtype'],**p) for p in json.loads((dest/'projection_controls.json').read_text())]
    if len(records)!=expected:raise RuntimeError(f'block denominator: expected {expected}, got {len(records)}')
    out=folder/'blocks'/label
    if out.exists():raise RuntimeError('immutable block already exists')
    out.mkdir(parents=True)
    good=[r for r in records if r['status']=='COLLECTED'];cases={x['case_id']:x for x in c['cases']}
    parity=[]
    for r in good:
        case=cases[r['case_id']]
        if r['dtype']=='bfloat16':
            delta=max(abs(a['relative_l2']-b) for a,b in zip(r['steps'],case['historical_relative_l2']))
            parity.append(dict(case_id=r['case_id'],max_metric_difference=delta,
                historical_fidelity=case['historical_fidelity'],new_fidelity=r['dense_fidelity'],
                label_matches=case['historical_fidelity']==r['dense_fidelity']))
    csv_write(out/'historical_parity.csv',parity)
    csv_write(out/'first_divergence.csv',first);atomic(out/'projection_controls.json',probes)
    rows=[x for r in good for x in r['steps']];csv_write(out/'metrics.csv',rows)
    # All source-level layer outputs retained; no head/step pseudoreplication.
    layer_rows=[r for r in stages if r['stage'].endswith('.layer_out') or r['stage'] in ['embedding','final_norm','logits']]
    csv_write(out/'layerwise.csv',layer_rows)
    groups=collections.defaultdict(list)
    for r in stages:groups[(r['dtype'],r['contrast'],r['stage'])].append(float(r['relative_l2']))
    quantiles=[]
    for key,values in sorted(groups.items()):
        quantiles.append(dict(dtype=key[0],contrast=key[1],stage=key[2],repeated_measure_rows=len(values),
            q0=float(np.min(values)),q25=float(np.quantile(values,.25)),median=float(np.median(values)),
            q75=float(np.quantile(values,.75)),q100=float(np.max(values))))
    csv_write(out/'stage_error_distributions.csv',quantiles)
    # Compact statistical records, no model activations or raw projection vectors.
    with (out/'attention_statistics.jsonl').open('w') as f:
        for r in distribution:f.write(canonical(r)+'\n')
    public_plan=dict(c,cases=[{k:v for k,v in x.items() if k!='tokens'}|dict(token_sha256=x['tokens']['sha256']) for x in c['cases']])
    public_plan['raw_input_availability']='Frozen exact token IDs remain in the server plan and historical condition tokens.json; regenerate with prepare against full private baseline artifacts.'
    atomic(out/'plan_index.json',public_plan)
    atomic(out/'runtime.json',{k:v for k,v in campaign.items() if k not in ['config','created']})
    earliest=collections.Counter((x['dtype']+':'+str(x['first_nonzero'])) for x in first if x['contrast']=='cached_full')
    summaries=[]
    for dtype in ['bfloat16','float32']:
        a=[x['relative_l2'] for x in rows if x['dtype']==dtype]
        if a:summaries.append(dict(dtype=dtype,steps=len(a),minimum=float(min(a)),median=float(np.median(a)),maximum=float(max(a)),
            first_nonzero_counts={k:v for k,v in earliest.items() if k.startswith(dtype+':')}))
    summary=dict(status='VERIFIED_BLOCK',block=label,baseline=c['baseline'],model=MODEL,revision=REV,
        planned_campaign_conditions=len(c['cases'])*2,block_conditions=len(records),failed=sum(r['status']!='COLLECTED' for r in records),
        independent_source_units=len({r['document_id'] for r in good}),conditions=[{k:v for k,v in r.items() if k!='steps'} for r in records],
        dense_counts=dict(collections.Counter(r['dtype']+'_'+r['dense_fidelity'] for r in good)),
        instrumentation_and_explicit_controls_checked=sum(r['control_checks'] for r in good),
        control_failures=sum(not(r['controls_exact'] and r['append_ok'] and r['structural_exact']) for r in good),
        historical_parity=parity,first_divergence_counts=dict(earliest),error_distributions=summaries,
        stage_rows=len(stages),layerwise_rows=len(layer_rows),attention_statistical_rows=len(distribution),
        projection_shape_sensitivity_conditions=sum(r['projection_shape_sensitivity_demonstrated'] for r in good),
        exhaustive_cause_unresolved=sum(r['unresolved_exhaustive_cause'] for r in good),
        scope='Outcome-stratified Qwen mechanism diagnostic. Does not explain every historical failure. No perf/capacity/model-quality claim.',
        gate_A='BOUNDED_MODEL_SPECIFIC_DIAGNOSIS' if good and all(r['controls_exact'] and r['structural_exact'] and r['append_ok'] for r in good) else 'FAIL',
        code_sha256=campaign['code_sha256'],publication='PENDING')
    if summary['control_failures']:raise RuntimeError('block controls failed')
    seal(out,summary)
    db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(label,'PENDING',digest((out/'manifest.json').read_bytes()),now()));db.commit();db.close()
    archive=ROOT.parent/('SCOPE_DIAG_'+label+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
        for d in ['scope_v2','tests_v2']:
            for p in (ROOT/d).glob('*.py'):z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(block=label,archive=archive.name,sha256=digest(archive.read_bytes()),summary=summary)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--label',required=True);p.add_argument('--expected',type=int,required=True)
    a=p.parse_args();export(a.folder,a.label,a.expected)
