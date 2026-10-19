"""Document-block statistics; token/head/step rows never count as independent samples."""
import collections,csv,json,math
from pathlib import Path
from scope.runner import atomic,canonical,digest,now
METRICS=['relative_l2','kl','top1_agreement','nll_delta']
def bootstrap(values,confidence=.95,repeats=5000,seed=20261018):
    import numpy as np
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(n=0,mean=None,ci=None)
    result=dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,median=float(np.median(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()))
    if len(x)<2:return dict(result,ci=None)
    rng=np.random.default_rng(seed);draws=x[rng.integers(0,len(x),size=(repeats,len(x)))].mean(1);a=(1-confidence)/2
    return dict(result,ci=[float(v) for v in np.quantile(draws,[a,1-a])],confidence=confidence,repeats=repeats,unit='document/project family; mean of eight correlated decode steps within each unit')
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        fields=sorted(set().union(*(set(r) for r in rows))) if rows else ['status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def paired(docs,left,right,metric):
    a={r['document_id']:r[metric] for r in docs if left(r)};b={r['document_id']:r[metric] for r in docs if right(r)}
    keys=sorted(set(a)&set(b));return keys,[a[k]-b[k] for k in keys]
def analyze(folder,records,complete=False):
    import numpy as np
    out=folder/'analysis';out.mkdir(exist_ok=True);docs=[];factorial=[];failures=[]
    for record in records:
        spec=record['spec'];r=record['result'];dest=folder/record['id']/record['attempt']
        if r['status']!='DATA_COLLECTED':failures.append(dict(spec,status=r['status']));continue
        groups=collections.defaultdict(list)
        for line in (dest/'trajectory.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['kind']=='controlled_treatment':groups[(row['regime'],row['variant'])].append(row)
        for (regime,variant),rows in groups.items():
            if len(rows)!=8:raise RuntimeError('incomplete treatment silently entered analysis')
            metrics=dict(relative_l2=np.mean([x['vs_native_cached']['relative_l2'] for x in rows]),kl=np.mean([x['vs_native_cached']['kl_reference_to_output'] for x in rows]),top1_agreement=np.mean([x['vs_native_cached']['top1_agreement'] for x in rows]),nll_delta=np.mean([x['nll_delta_vs_native'] for x in rows]))
            if not all(math.isfinite(float(v)) for v in metrics.values()):failures.append(dict(spec,status='NONFINITE_METRIC',variant=variant,regime=regime));continue
            docs.append(dict(spec,regime=regime,variant=variant,**{k:float(v) for k,v in metrics.items()},decode_steps=8,dense_fidelity=r.get('dense_fidelity')))
        contrasts=collections.defaultdict(list)
        for line in (dest/'contrasts.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['kind']=='KV_factorial':contrasts[row['regime']].append(row['interaction']['relative_l2'])
        for regime,values in contrasts.items():factorial.append(dict(spec,regime=regime,interaction_relative_l2=float(np.mean(values)),steps=len(values)))
    write_csv(out/'document_metrics.csv',docs);write_csv(out/'document_factorial.csv',factorial)
    distributions=[];comparisons=[];primary=[]
    for domain in ['prose','code']:
        for dtype in ['bfloat16','float32']:
            for split in ['all','exploratory','confirmatory']:
                pool=[r for r in docs if r['category']==domain and r['dtype']==dtype and (split=='all' or r['split']==split)]
                for length in [512,2048]:
                    for regime in ['isolated','recurrent']:
                        common=dict(category=domain,dtype=dtype,split=split,length=length,regime=regime)
                        for variant in ['K10','V01','KV11','packedKV']:
                            values=[r['relative_l2'] for r in pool if r['length']==length and r['regime']==regime and r['variant']==variant]
                            distributions.append(dict(common,variant=variant,metric='mean_step_relative_l2',**bootstrap(values)))
                        for metric in METRICS:
                            ids,delta=paired(pool,lambda r:r['length']==length and r['regime']==regime and r['variant']=='K10',lambda r:r['length']==length and r['regime']==regime and r['variant']=='V01',metric)
                            item=dict(common,contrast='K10 minus V01',metric=metric,**bootstrap(delta));comparisons.append(item)
                            if dtype=='bfloat16' and split=='confirmatory' and length==2048 and metric=='relative_l2':primary.append(dict(common,contrast='K10 minus V01',metric=metric,**bootstrap(delta,confidence=.9875),prespecified=True,planned_n=128,document_ids=ids))
                for metric in METRICS:
                    for regime in ['isolated','recurrent']:
                        ids,delta=paired(pool,lambda r:r['length']==2048 and r['regime']==regime and r['variant']=='KV11',lambda r:r['length']==512 and r['regime']==regime and r['variant']=='KV11',metric)
                        comparisons.append(dict(category=domain,dtype=dtype,split=split,regime=regime,contrast='KV11 L2048 minus L512',metric=metric,**bootstrap(delta)))
                    for length in [512,2048]:
                        ids,delta=paired(pool,lambda r:r['length']==length and r['regime']=='recurrent' and r['variant']=='KV11',lambda r:r['length']==length and r['regime']=='isolated' and r['variant']=='KV11',metric)
                        comparisons.append(dict(category=domain,dtype=dtype,split=split,length=length,contrast='KV11 recurrent minus isolated',metric=metric,**bootstrap(delta)))
    atomic(out/'distributions.json',distributions);atomic(out/'paired_contrasts.json',comparisons);atomic(out/'primary_confirmatory.json',primary)
    counts=collections.Counter((r['spec']['category'],r['spec']['dtype'],r['spec']['length'],r['spec']['split'],r['result']['status']) for r in records)
    valid_ids={r['document_id'] for r in docs if r['dtype']=='bfloat16'}
    summary=dict(updated=now(),complete=complete,observed_valid_primary_source_units=len(valid_ids),planned_source_units=json.loads((folder/'campaign.json').read_text())['planned_independent_units'],attempted_conditions=len(records),failed_conditions=len({(r['document_id'],r['length'],r['dtype']) for r in failures}),condition_counts=[dict(category=k[0],dtype=k[1],length=k[2],split=k[3],status=k[4],count=v) for k,v in counts.items()],failures=failures,primary=primary,independence='Article/project-name families are analysis blocks. Steps, heads, layers, lengths, dtypes and treatment rows are repeated measures. Residual source lineage/topic dependence is possible.',scope='Frozen bounded public-source sampling frame and one model; no overall model quality, performance or resident-memory claim. Dense cached/full fidelity failures reported separately, not filtered from valid intervention/no-op pairs.',multiplicity='Four prespecified confirmatory mean K-minus-V relative-L2 contrasts; 98.75% pointwise percentile document-bootstrap intervals, nominal Bonferroni 95% family coverage. All other intervals descriptive and exploratory. No power guarantee or reliable extreme-tail claim.',missingness='Every planned condition remains in plan.json. Failures retained and excluded only from numeric estimates; bias from failed conditions remains possible. Preliminary intervals are not final conclusions.')
    atomic(out/'SUMMARY.json',summary)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for di,domain in enumerate(['prose','code']):
            for ri,regime in enumerate(['isolated','recurrent']):
                ax=axes[di,ri]
                for variant in ['K10','V01','KV11']:
                    values=sorted(r['relative_l2'] for r in docs if r['category']==domain and r['dtype']=='bfloat16' and r['length']==2048 and r['regime']==regime and r['variant']==variant)
                    if values:ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{variant}, n={len(values)}')
                ax.set(title=f'{domain} / {regime} / L2048',xlabel='Mean decode-step relative L2 per source unit',ylabel='Empirical cumulative probability');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Complete attempted campaign' if complete else 'PRELIMINARY - collection incomplete');fig.savefig(out/'relative_l2_ecdf.png',dpi=160);fig.savefig(out/'relative_l2_ecdf.pdf');plt.close(fig)
    except ImportError:atomic(out/'plot_status.json',dict(status='MATPLOTLIB_UNAVAILABLE'))
    (out/'REPORT.md').write_text('# Independent-source SCOPE campaign\n\n'+('Collection finished.' if complete else 'PRELIMINARY: collection is incomplete; no final inferential conclusion.')+'\n\n'+json.dumps(summary,indent=2)+'\n\nSee document_metrics.csv for the independent-unit distributions, paired_contrasts.json for paired effects, and primary_confirmatory.json for the four prespecified contrasts. Q95 is descriptive; q99 is not estimated. Continuation NLL uses only eight natural next-token targets per document and is not a full-corpus perplexity benchmark.\n')
    files={f.name:digest(f.read_bytes()) for f in out.iterdir() if f.is_file() and f.name!='SHA256SUMS.txt'};(out/'SHA256SUMS.txt').write_text(''.join(v+'  '+k+'\n' for k,v in sorted(files.items())))
