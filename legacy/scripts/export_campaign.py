import argparse,csv,hashlib,json,pathlib,sqlite3,statistics,subprocess,time,zipfile
ap=argparse.ArgumentParser();ap.add_argument('campaign');ap.add_argument('--wait',action='store_true');a=ap.parse_args();p=pathlib.Path(a.campaign).resolve();repo=p.parents[2];task=repo.parent
while a.wait:
 h=p/'heartbeat.json'
 if h.exists() and json.loads(h.read_text()).get('state')=='STOPPED':break
 time.sleep(15)
meta=json.loads((p/'campaign.json').read_text());db=sqlite3.connect(p/'registry.sqlite');jobs=[dict(zip(['id','spec','status','publication','attempts'],r)) for r in db.execute('select id,spec,status,publication,attempts from jobs order by rowid')];counts={};pub={}
for j in jobs:counts[j['status']]=counts.get(j['status'],0)+1;pub[j['publication']]=pub.get(j['publication'],0)+1
assert not any(j['status'] in ['PLANNED','RUNNING'] for j in jobs),'campaign is not terminal'
dest=task/'deliverables'/p.name;dest.mkdir(parents=True,exist_ok=True);capture_rows=[];operator_rows=[];failed=[]
for j in jobs:
 f=p/j['id']/'output.json';d=json.loads(f.read_text()) if f.exists() else {};spec=json.loads(j['spec'])
 if j['status']!='PASS':failed.append({'id':j['id'],'status':j['status'],'reason':d.get('detail','predecessor did not pass'),'planned_condition':spec})
 if j['id'].startswith('P1-') and j['status']=='PASS':
  prov=d['provenance'];ms=d['metrics'];base=d.get('model_decode_baseline',[])
  capture_rows.append({'id':j['id'],'model':prov['model_id'],'architecture':json.dumps(prov['architecture_observed']),'context':prov['context'],'category':spec['params']['category'],'head_step_observations':len(ms),'max_native_relative_l2':max(r['native_relative_l2'] for r in ms),'median_fp32_diagnostic_relative_l2':statistics.median(r.get('fp32_diagnostic_vs_native_relative_l2',r['native_relative_l2']) for r in ms),'max_fp32_diagnostic_relative_l2':max(r.get('fp32_diagnostic_vs_native_relative_l2',r['native_relative_l2']) for r in ms),'median_quantized_head_relative_l2':statistics.median(r['quantized_relative_l2'] for r in ms),'full_model_decode_median_ms':statistics.median(r['median_decode_ms'] for r in base) if base else None,'cache_tensor_storage_bytes':max(r['cache_tensor_storage_bytes'] for r in base) if base else None})
 if j['id'].startswith('P4-') and d:
  metrics=d.get('metrics',[])
  for variant in sorted(set(r['variant'] for r in metrics)):
   rows=[r for r in metrics if r['variant']==variant];good=[r for r in rows if r['status']=='PASS']
   row={'id':j['id'],'variant':variant,'status':'PASS' if good else 'FAIL','timing_windows':len(good),'median_ms':statistics.median(r['median_ms'] for r in good) if good else None,'execution_relative_l2':max(r.get('execution_relative_l2',0) for r in rows)}
   if good:row.update({'storage_allocation_bytes':good[0]['storage_allocation_bytes'],'transient_peak_delta_bytes':max(r['transient_peak_delta_bytes'] for r in good),'representation_relative_l2':good[0].get('representation_relative_l2')})
   operator_rows.append(row)
summary={'run_id':p.name,'code_sha':meta['code_sha'],'source_hash':meta['source_hash'],'planned_execution_blocks':len(jobs),'planned_model_conditions':sum(j['id'].startswith('P1-') for j in jobs),'execution_status_counts':counts,'publication_status_counts':pub,'real_model_conditions_passed':len(capture_rows),'operator_variant_pass_count':sum(r['status']=='PASS' for r in operator_rows),'operator_variant_fail_count':sum(r['status']=='FAIL' for r in operator_rows),'capture_rows':capture_rows,'operator_rows':operator_rows,'failed_or_unsupported':failed,'raw_tensor_location':str(p),'limitations':['No full-model compressed-cache integration or end-model quality evaluation','No measured maximum resident-KV/decode frontier','No real MQA, hybrid/MLA, custom Triton, FlashAttention or vLLM execution','Fixed repeated public prose/code population; no confirmatory held-out replication','P7 is artifact integrity only; block PASS does not mean every screened variant passed']}
(dest/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
for name,rows in [('capture_metrics.csv',capture_rows),('operator_metrics.csv',operator_rows)]:
 with (dest/name).open('w',newline='') as f:
  fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
lines=['# H200 campaign '+p.name,'','Frozen code: '+meta['code_sha'],'','Execution blocks: '+json.dumps(counts),'','Publication: '+json.dumps(pub),'','Real-model conditions passed: '+str(len(capture_rows)),'','## Actual full-model dense baselines','','| Model | L | Input | Median decode ms | Native replay max relative L2 |','|---|---:|---|---:|---:|']
for r in capture_rows:lines.append('| '+r['model']+' | '+str(r['context'])+' | '+r['category']+' | '+str(round(r['full_model_decode_median_ms'],4) if r['full_model_decode_median_ms'] is not None else None)+' | '+str(r['max_native_relative_l2'])+' |')
lines+=['','Full-model values are uninstrumented eager BF16 forward plus greedy token choice, excluding prefill; median of three six-token timing windows. Exact contexts grow from the initial length. These are dense-cache baselines.','','## Operator screening','','| Condition | Variant | Status | Median ms | Execution relative L2 |','|---|---|---|---:|---:|']
for r in operator_rows:lines.append('| '+r['id']+' | '+r['variant']+' | '+r['status']+' | '+str(round(r['median_ms'],5) if r['median_ms'] is not None else None)+' | '+str(round(r['execution_relative_l2'],6))+' |')
lines+=['','Operator medians summarize three CUDA-event timing windows. Failed variants were not timed. Storage bytes count actual K/V tensor storage including scales, rather than a maximum resident-capacity experiment.','','## Limits','']+['- '+s for s in summary['limitations']]
lines+=['','## Reproduction','','Use the frozen source commit and exact configuration in campaign.json with the recorded CUDA environment. On a dedicated H200:','','```bash','python -m scope.runner preflight --config configs/h200_campaign.yaml','python -m scope.runner plan --config configs/h200_campaign.yaml','python -m scope.runner run --config configs/h200_campaign.yaml --resume --publish-each-block','python -m scope.runner reproduce --run-id '+p.name,'```','','The reproduce command validates hashes. Independent GPU replication requires a fresh checkout and a fresh result store; do not copy previous campaign results into it. Publication requires existing server Git authentication. Raw tensors/logs remain at '+str(p)+'.']
(dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
archive=dest/'source_at_frozen_commit.zip';subprocess.run(['git','archive','--format=zip',meta['code_sha'],'-o',str(archive)],cwd=repo,check=True)
portable={'campaign':meta,'jobs':jobs};(dest/'REGISTRY.json').write_text(json.dumps(portable,indent=2)+'\n')
allow={'campaign.json','status.json','manifest.json','summary.json','metrics.csv','report.md','SHA256SUMS.txt','output.json','state_statistics.json','publication.json'}
package=task/('scope_evidence_'+p.name+'.zip')
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(dest.iterdir()):
  if f.is_file():z.write(f,f.name)
 for f in sorted(p.rglob('*')):
  if f.is_file() and (f.name in allow or ('repeat' in f.name and f.suffix=='.json')):z.write(f,'results/'+str(f.relative_to(repo/'results')))
 for f in (repo/'results'/'preflight').glob('*.json'):z.write(f,'results/preflight/'+f.name)
 z.write(__file__,'export_campaign.py')
checksum=hashlib.sha256(package.read_bytes()).hexdigest();(package.with_suffix('.sha256')).write_text(checksum+'  '+package.name+'\n');print(json.dumps({'package':str(package),'bytes':package.stat().st_size,'sha256':checksum,'counts':counts,'publication':pub}),flush=True)
