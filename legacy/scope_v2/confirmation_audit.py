"""Additive ACL-R1 confirmation audit; never changes the frozen GPU campaign.

CPU-only raw-step/source denominator audit and jointly paired model contrasts.
"""
import argparse,collections,csv,itertools,json,math,sqlite3,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,now,source_hash
from scope.population_stats import bootstrap
from scope_v2.replication import ident,jobs_for,verify as verify_input
ROOT=Path(__file__).resolve().parents[1]

def write_csv(path,rows):
    keys=sorted(set().union(*(set(r) for r in rows))) if rows else ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def paired_models(left,right,expected,repeats=20000):
    """Paired difference resampling == one shared source draw across both models."""
    if set(left)!=set(expected) or set(right)!=set(expected):raise ValueError('incomplete paired source population')
    ids=sorted(expected);delta=[left[i]-right[i] for i in ids]
    return dict(**bootstrap(delta,confidence=.95,repeats=repeats,seed=20261019),
        inference='exploratory paired-model contrast; same source indices in both models; no new primary claim')

def seal(folder,result):
    folder.mkdir(parents=True,exist_ok=True);atomic(folder/'summary.json',result)
    (folder/'report.md').write_text('# Additive confirmation audit, ACL-R1\n\n'+json.dumps(result,indent=2)+'\n\nPrimary endpoints, populations and thresholds are unchanged. Native cached intervention is not original full-reference quality parity. All head/token/step/model repetitions share source units. Observed tail counts are not estimates of extremely rare population events.\n')
    if not (folder/'metrics.csv').exists():write_csv(folder/'metrics.csv',[])
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in folder.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(folder/'manifest.json',dict(files=files,audit_source_sha256=digest(Path(__file__).read_bytes()),sealed=now()))
    (folder/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))
    verify_input(folder)

def analyze(run,manifest,out):
    import numpy as np
    data=json.loads(manifest.read_text());campaign=json.loads((run/'campaign.json').read_text())
    assert digest(manifest.read_bytes())==campaign['manifest_sha256']
    assert source_hash()==campaign['historical_source_sha256']
    jobs=[s for s in jobs_for(data) if s['split']=='confirmatory'];assert len(jobs)==560
    expected={domain:{d['document_id'] for d in data['documents'] if d['split']=='confirmatory' and d['domain']==domain} for domain in ['prose','code']}
    assert all(len(v)==56 for v in expected.values())
    db=sqlite3.connect(run/'registry.sqlite',timeout=30);registry={r[0]:r[1:] for r in db.execute('SELECT id,spec,status,attempt FROM jobs')};db.close()
    tails=[];denominators=[];inputs={};effect=collections.defaultdict(dict);terminal=collections.Counter();means=[]
    for spec in jobs:
        k=ident(spec);row=registry.get(k);status=row[1] if row else 'MISSING_REGISTRY';terminal[status]+=1
        common=dict(model=spec['model'],domain=spec['domain'],document_id=spec['document_id'])
        entry=dict(common,status=status,dense_fidelity=None,dense_max_step_relative_l2=None)
        if status!='COLLECTED':denominators.append(entry);continue
        assert json.loads(row[0])==spec
        point=run/k/row[2];r=verify_input(point);meta=json.loads((point/'manifest.json').read_text())
        assert meta['code_sha256']==campaign['source_sha256']
        assert r['status']=='COLLECTED' and r['noop_exact']==32 and r['packing_exact']==16 and r['first_step_history_exact']==6 and r['snapshots_unchanged']
        inputs[k]=digest((point/'manifest.json').read_bytes())
        raw=list(csv.DictReader((point/'metrics.csv').open()));assert len(raw)==96
        groups=collections.defaultdict(list)
        for x in raw:groups[x['regime'],x['variant']].append(x)
        assert len(groups)==12 and all(sorted(int(x['step']) for x in g)==list(range(8)) for g in groups.values())
        dense={int(x['step']):float(x['dense_relative_l2']) for x in raw}
        assert all(float(x['dense_relative_l2'])==dense[int(x['step'])] for x in raw)
        entry.update(dense_fidelity=r['dense_fidelity'],dense_max_step_relative_l2=max(dense.values()),dense_steps_over_025=sum(v>.025 for v in dense.values()))
        assert (r['dense_fidelity']=='PASS')==all(v<=.025 for v in dense.values())
        denominators.append(entry)
        per={}
        for (regime,variant),group in groups.items():
            values=[float(x['relative_l2']) for x in group]
            assert all(math.isfinite(float(x[m])) for x in group for m in ['relative_l2','kl','nll_delta','top1_agreement'])
            avg={m:float(np.mean([float(x[m]) for x in group])) for m in ['relative_l2','kl','nll_delta','top1_agreement']}
            per[regime,variant]=avg['relative_l2'];means.append(dict(common,regime=regime,variant=variant,**avg))
            tails.append(dict(common,regime=regime,variant=variant,steps=8,mean_step_relative_l2=avg['relative_l2'],max_step_relative_l2=max(values),min_step_relative_l2=min(values),
                observed_steps_error_over_025=sum(v>.025 for v in values),observed_source_error_over_025=any(v>.025 for v in values),dense_fidelity=r['dense_fidelity']))
        for regime in ['isolated','recurrent']:
            effect[spec['model'],spec['domain'],regime][spec['document_id']]=per[regime,'K1V0']-per[regime,'K0V1']
    primary=[];cross=[];missing=[]
    for model,domain,regime in itertools.product(['pythia1b','opt1_3b'],['prose','code'],['isolated','recurrent']):
        values=effect[model,domain,regime];ids=expected[domain];base=dict(model=model,domain=domain,regime=regime,planned_n=56,observed_n=len(values),prespecified_primary=True)
        if set(values)!=ids:
            primary.append(dict(base,status='NON_ESTIMABLE_MISSING_INPUTS',missing_source_ids=sorted(ids-set(values))));continue
        result=bootstrap([values[i] for i in sorted(ids)],confidence=.99375,repeats=20000,seed=20261019)
        sealed=run/'blocks'/('confirmatory-'+model);s=verify_input(sealed)
        reference=next(x for x in json.loads((sealed/'contrasts.json').read_text()) if x['domain']==domain and x['regime']==regime and x['endpoint']=='relative_l2')
        assert reference['prespecified_primary'] and reference['n']==56 and reference['confidence']==.99375
        for a,b in zip([result['mean']]+result['ci'],[reference['mean']]+reference['ci']):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)
        primary.append(dict(base,status='REPRODUCED_FROZEN_PRIMARY',**result))
    models=[m['key'] for m in data['models']]
    for left,right in itertools.combinations(models,2):
        for domain,regime in itertools.product(['prose','code'],['isolated','recurrent']):
            a,b=effect[left,domain,regime],effect[right,domain,regime];ids=expected[domain]
            base=dict(left_model=left,right_model=right,domain=domain,regime=regime,planned_n=56,contrast='(K-only minus V-only)_left minus (K-only minus V-only)_right')
            if set(a)!=ids or set(b)!=ids:cross.append(dict(base,status='NON_ESTIMABLE_MISSING_INPUTS'));continue
            cross.append(dict(base,status='EXPLORATORY',**paired_models(a,b,ids)))
    tail_summary=[]
    for model,domain,regime,variant in itertools.product(models,['prose','code'],['isolated','recurrent'],['native','K0V0','K1V0','K0V1','K1V1','packed11']):
        rows=[r for r in tails if (r['model'],r['domain'],r['regime'],r['variant'])==(model,domain,regime,variant)]
        x=[r['max_step_relative_l2'] for r in rows]
        tail_summary.append(dict(model=model,domain=domain,regime=regime,variant=variant,planned_sources=56,observed_sources=len(rows),
            sources_with_step_error_over_025=sum(r['observed_source_error_over_025'] for r in rows),steps_over_025=sum(r['observed_steps_error_over_025'] for r in rows),observed_steps=len(rows)*8,
            source_maximum_error_max=max(x) if x else None,source_maximum_error_median=float(np.median(x)) if x else None,source_maximum_error_q95=float(np.quantile(x,.95)) if x else None))
    out.mkdir(parents=True,exist_ok=False);write_csv(out/'metrics.csv',means);write_csv(out/'source_tails.csv',tails);write_csv(out/'denominators.csv',denominators)
    atomic(out/'primary_recheck.json',primary);atomic(out/'cross_model_paired.json',cross);atomic(out/'tail_summary.json',tail_summary);atomic(out/'input_manifests.json',inputs)
    result=dict(status='VERIFIED_ADDITIVE_AUDIT' if terminal=={'COLLECTED':560} else 'AUDITED_WITH_FAILURES',upstream_run=run.name,
        input_manifest_sha256=campaign['manifest_sha256'],upstream_source_sha256=campaign['source_sha256'],planned_conditions=560,terminal_counts=dict(terminal),
        planned_independent_source_units=112,planned_independent_sources_per_domain=56,
        observed_source_units_by_model={m:len({x['document_id'] for x in denominators if x['model']==m and x['status']=='COLLECTED'}) for m in models},
        models_are_repeated_measures=True,primary_contrasts=len(primary),
        primary_reproduced=sum(x['status']=='REPRODUCED_FROZEN_PRIMARY' for x in primary),exploratory_cross_model_contrasts=len(cross),
        source_tail_rows=len(tails),source_mean_rows=len(means),dense_failure_conditions=sum(x['dense_fidelity']=='FAIL' for x in denominators),
        observed_threshold_scope='The .025 intervention error threshold in tail tables is descriptive against native cached logits; it is NOT an execution Gate D result or a model-quality pass/fail gate.',
        uncertainty='Frozen eight primaries: 20k resamples,99.375% intervals. Cross-model contrasts: jointly paired source draws,20k resamples,exploratory95% intervals; no familywise claim. No Q99/rare-population precision claim.',
        cross_model_scope='Differences between within-model normalized logit perturbation effects across distinct vocabularies/tokenizers are exploratory, not model-quality rankings.',publication='PENDING')
    seal(out,result);return result

def run_when_ready(run,manifest):
    identity=dict(upstream_run=run.name,manifest_sha256=digest(manifest.read_bytes()),audit_source_sha256=digest(Path(__file__).read_bytes()))
    folder=ROOT/'results/confirmation_audit_v2'/digest(canonical(identity).encode())[:16];folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'campaign.json',identity);out=folder/'blocks/confirmation'
    if (out/'manifest.json').exists():verify_input(out);return
    deadline=time.monotonic()+86400
    while True:
        state=json.loads((run/'status.json').read_text())
        if state['status']=='COMPLETE_all':break
        if state['status']=='STOPPED_WITH_FAILURE' or time.monotonic()>deadline:
            atomic(folder/'status.json',dict(status='BLOCKED_UPSTREAM',upstream_status=state['status'],updated=now()));return
        atomic(folder/'status.json',dict(status='WAITING_FOR_FROZEN_CAMPAIGN',upstream_status=state['status'],updated=now()));time.sleep(30)
    try:
        r=analyze(run,manifest,out)
    except Exception:
        (folder/'audit.private.log').write_text(traceback.format_exc())
        r=dict(status='FAIL',reason='additive audit validation failed; partial evidence retained and original campaign untouched',upstream_run=run.name,publication='PENDING')
        out.mkdir(parents=True,exist_ok=True);seal(out,r)
    db=sqlite3.connect(folder/'registry.sqlite');db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)')
    db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',('confirmation','PENDING',digest((out/'manifest.json').read_bytes()),now()));db.commit();db.close()
    with zipfile.ZipFile(ROOT.parent/'SCOPE_CONFIRMATION_AUDIT_V2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.relative_to(ROOT).as_posix())
    atomic(folder/'status.json',dict(status='FAIL' if r['status']=='FAIL' else 'COMPLETE',audit_status=r['status'],updated=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();run_when_ready(a.run,a.manifest)
