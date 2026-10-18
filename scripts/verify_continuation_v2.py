"""Check sealed compact diagnostic blocks; no GPU or private tensors required."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
blocks=list((root/'results/continuation_v2').glob('*/blocks/*/manifest.json'))
assert blocks,'no diagnostic blocks'
for manifest in blocks:
    folder=manifest.parent;m=json.loads(manifest.read_text())
    for name,meta in m['files'].items():
        p=(folder/name).resolve();assert p.parent==folder.resolve()
        assert p.stat().st_size==meta['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],p
    s=json.loads((folder/'summary.json').read_text());n=s['block_conditions']-s['failed']
    assert len(s['conditions'])==s['block_conditions']
    assert sum(s['dense_counts'].values())==n
    assert s['control_failures']==0
    assert sum(1 for _ in csv.DictReader((folder/'metrics.csv').open()))==n*8
    assert sum(1 for _ in csv.DictReader((folder/'first_divergence.csv').open()))==n*12
    assert sum(1 for _ in csv.DictReader((folder/'layerwise.csv').open()))==s['layerwise_rows']
    assert len([json.loads(x) for x in (folder/'attention_statistics.jsonl').read_text().splitlines()])==s['attention_statistical_rows']
    assert all(x['label_matches'] for x in s['historical_parity'])
mechanism=list((root/'results/mechanism_v2').glob('*/blocks/*/manifest.json'))
for manifest in mechanism:
    folder=manifest.parent;m=json.loads(manifest.read_text())
    for name,meta in m['files'].items():
        p=(folder/name).resolve();assert p.parent==folder.resolve()
        assert p.stat().st_size==meta['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],p
    s=json.loads((folder/'summary.json').read_text())
    rows=list(csv.DictReader((folder/'metrics.csv').open()))
    if 'expected_rows' in s:
        assert len(rows)==s['rows']==s['expected_rows']==s['collected']*54
        assert len(json.loads((folder/'details.json').read_text()))==s['rows']
        assert len(s['results'])==s['conditions']==s['collected']+s['failed']
    elif 'traces' in s:
        assert len(rows)==s['traces']==s['classified']+s['unresolved_first_stage']
    elif 'operations' in s:
        assert len(rows)==s['operations']==len(s['results'])==3
        assert all(r['upstream_exact'] and r['one_vs_native']['max_abs']==0 and r['many_vs_native_full']['max_abs']==0 and r['one_vs_many']['max_abs']>0 for r in s['results'])
replication=list((root/'results/replication_v2').glob('*/blocks/*/manifest.json'))
for manifest in replication:
    folder=manifest.parent;m=json.loads(manifest.read_text())
    for name,meta in m['files'].items():
        p=(folder/name).resolve();assert p.parent==folder.resolve()
        assert p.stat().st_size==meta['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],p
    s=json.loads((folder/'summary.json').read_text());collected=[r for r in s['conditions'] if r['status']=='COLLECTED']
    assert len(s['conditions'])==s['planned_conditions']==sum(s['counts'].values())
    assert len(collected)==s['counts'].get('COLLECTED',0)
    assert sum(s['dense_counts'].values())==len(collected)
    assert s['source_units']==len({r['spec']['document_id'] for r in collected})
    assert not s['missing']
    assert all(c['noop_exact']==32 and c['packing_exact']==16 and c['first_step_history_exact']==6 and c['snapshots_unchanged'] for c in collected)
    assert len(list(csv.DictReader((folder/'metrics.csv').open())))==s['document_metric_rows']==len(collected)*12
    assert len(list(csv.DictReader((folder/'factorial.csv').open())))==s['factorial_document_rows']==len(collected)*2
print(json.dumps(dict(status='PASS',diagnostic_blocks=len(blocks),mechanism_blocks=len(mechanism),replication_blocks=len(replication),scope='compact artifact hashes/schema/denominators; not GPU replication')))
