import collections,csv,hashlib,json,pathlib,statistics,time
ROOT=pathlib.Path('/home/inugpu/workspace/scope_h200_20261018_9EfrwM/SCOPE_ablation_v1/results/ablation/e5998af63b11ad5d')
def rr(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def stats(v):
 v=[x for x in v if isinstance(x,(int,float))]
 return {'n':len(v),'min':min(v) if v else None,'median':statistics.median(v) if v else None,'max':max(v) if v else None,'mean':statistics.mean(v) if v else None}
assert json.loads((ROOT/'status.json').read_text())['status']=='COMPLETE'
records=json.loads((ROOT/'registry.json').read_text())
rows=[];effects=[];packing=[];first=[];controls=[];conditions=[]
for r in records:
 d=ROOT/r['id']/r['attempt'];common=dict(condition=r['id'],**r['spec']);control=json.loads((d/'controls.json').read_text());controls.append(dict(common,**control));conditions.append(dict(common,**r['result']))
 for x in rr(d/'trajectory.jsonl'):
  if x['kind']=='controlled_treatment':rows.append(dict(common,variant=x['variant'],regime=x['regime'],step=x['step'],relative_l2=x['vs_native_cached']['relative_l2'],kl=x['vs_native_cached'].get('kl_reference_to_output'),top1_agreement=x['vs_native_cached'].get('top1_agreement'),nll_delta=x['nll_delta_vs_native']))
 for x in rr(d/'contrasts.jsonl'):
  if x['kind']=='packing':packing.append(x['metrics']['relative_l2'])
  if x['kind']=='history_effect' and x['step']==0:first.append(x['first_step_exact'])
  if x['kind']=='KV_factorial':effects.append(dict(common,regime=x['regime'],step=x['step'],K_main=x['K_main']['relative_l2'],V_main=x['V_main']['relative_l2'],combined=x['combined']['relative_l2'],interaction=x['interaction']['relative_l2'],valid=x['valid_noop_controls']))
def csvwrite(name,data):
 fields=sorted(set().union(*(set(x) for x in data)))
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
csvwrite('treatment_numeric.csv',rows);csvwrite('factorial_numeric.csv',effects)
by_group=collections.defaultdict(list)
for x in rows:by_group[(x['dtype'],x['variant'],x['regime'])].append(x)
groups=[]
for key,values in sorted(by_group.items()):groups.append(dict(dtype=key[0],variant=key[1],regime=key[2],observations=len(values),relative_l2=stats([x['relative_l2'] for x in values]),kl=stats([x['kl'] for x in values]),nll_delta=stats([x['nll_delta'] for x in values]),top1_agreement=stats([x['top1_agreement'] for x in values])))
summary=json.loads((ROOT/'SUMMARY.json').read_text())
out=dict(run_id=ROOT.name,source_sha=json.loads((ROOT/'campaign.json').read_text())['identity']['git_sha'],processed=len(records),status_counts=dict(collections.Counter(x['result']['status'] for x in records)),all_noop_controls_pass=all(c['noop_exact'] for c in controls),all_native_snapshots_unchanged=all(c['native_snapshots_unchanged'] for c in controls),all_first_step_history_controls_exact=all(first),first_step_history_checks=len(first),packing_added_relative_l2=stats(packing),trajectory_rows=sum(x['result'].get('trajectory_rows',0) for x in records),treatment_rows=len(rows),factorial_rows=len(effects),cross_dtype_identity=summary['cross_dtype_input_identity'],artifact_integrity=summary['artifact_integrity'],groups=groups,conditions=conditions,limits='One model, two repeated authored input families, four context populations. Dependent observations; no significance or general language-model-quality claim. Native prefill and dense caches retained; no timing/capacity claim.')
(ROOT/'ENGINEERING_FINDINGS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
report='# Controlled ablation findings\n\n'+json.dumps({k:v for k,v in out.items() if k not in ['groups','conditions']},indent=2)+'\n\nPer-treatment numerical summaries are in ENGINEERING_FINDINGS.json. Exact paired rows are in treatment_numeric.csv and factorial_numeric.csv. Raw paired logits remain in per-condition server directories.\n'
(ROOT/'ENGINEERING_FINDINGS.md').write_text(report)
files=['ENGINEERING_FINDINGS.json','ENGINEERING_FINDINGS.md','treatment_numeric.csv','factorial_numeric.csv']
(ROOT/'FINDINGS_SHA256SUMS.txt').write_text(''.join(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in files))
print(json.dumps({k:v for k,v in out.items() if k not in ['groups','conditions']},indent=2))

import zipfile
base=ROOT.parents[3]/('SCOPE_ABLATION_'+ROOT.name+'.zip')
assert base.exists(),str(base)
expected=base.with_suffix('.zip.sha256').read_text().split()[0]
assert hashlib.sha256(base.read_bytes()).hexdigest()==expected
final=base.parent/('SCOPE_CONTROLLED_ABLATION_'+ROOT.name+'_FINAL.zip')
with zipfile.ZipFile(base) as src,zipfile.ZipFile(final,'x',zipfile.ZIP_DEFLATED) as dst:
 assert src.testzip() is None
 for member in src.infolist():dst.writestr(member,src.read(member.filename))
 for name in files+['FINDINGS_SHA256SUMS.txt']:dst.write(ROOT/name,'findings/'+name)
 dst.write(pathlib.Path(__file__),'findings/analysis_source.py')
sha=hashlib.sha256(final.read_bytes()).hexdigest();final.with_suffix('.zip.sha256').write_text(sha+'  '+final.name+'\n')
print('FINAL_PACKAGE',str(final),'SHA256',sha,'BYTES',final.stat().st_size)
