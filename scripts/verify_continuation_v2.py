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
print(json.dumps(dict(status='PASS',sealed_blocks=len(blocks),scope='compact artifact hashes/schema/denominators; not GPU replication')))
