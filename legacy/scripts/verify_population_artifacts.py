"""Verify the public population package with the Python standard library only."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/population/46f7c2fa18e3d9c7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(inputs=None):
    manifest = json.loads((RUN / 'PUBLICATION_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()), name
        assert sha(path) == expected, 'hash mismatch: ' + name
    paths = sorted(list((ROOT / 'scope').glob('*.py')) + list((ROOT / 'tests').glob('*.py')))
    content_hash = hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode() + b'\0' + p.read_bytes() for p in paths)).hexdigest()
    audit = json.loads((RUN / 'FINAL_INTEGRITY.json').read_text())
    assert content_hash == audit['source_hash']
    sources = json.loads((RUN / 'input_sources.json').read_text())['documents']
    units = {d['document_id']: d for d in sources}
    assert len(units) == len(sources) == 512
    assert collections.Counter(d['domain'] for d in sources) == {'prose': 256, 'code': 256}
    assert collections.Counter((d['domain'], d['split']) for d in sources) == {
        (d, s): 128 for d in ['prose', 'code'] for s in ['exploratory', 'confirmatory']}
    conditions = list(csv.DictReader((RUN / 'condition_index.csv').open(newline='')))
    assert len(conditions) == 1152
    index = {(r['document_id'], r['dtype'], int(r['length'])): r for r in conditions}
    assert len(index) == len(conditions)
    for r in conditions:
        assert r['status'] == 'DATA_COLLECTED'
        assert all(r[k] == 'True' for k in ['noop_exact', 'cache_unchanged', 'history_firststep_exact'])
        assert int(r['completed_variants']) == 12 and int(r['trajectory_rows']) == 104
    assert collections.Counter((r['dtype'], r['dense_fidelity']) for r in conditions) == {
        ('bfloat16', 'PASS'): 571, ('bfloat16', 'FAIL'): 453, ('float32', 'PASS'): 128}
    assert len({r['document_id'] for r in conditions if r['dtype'] == 'float32'}) == 64
    for (doc, dtype, length), r in index.items():
        if dtype == 'float32':
            assert r['token_sha256'] == index[(doc, 'bfloat16', length)]['token_sha256']
    metrics = list(csv.DictReader((RUN / 'analysis/document_metrics.csv').open(newline='')))
    assert len(metrics) == 13824
    keyed = {}
    for r in metrics:
        key = (r['document_id'], r['dtype'], int(r['length']), r['regime'], r['variant'])
        assert key not in keyed
        keyed[key] = r
        assert int(r['decode_steps']) == 8
        assert all(math.isfinite(float(r[m])) for m in ['relative_l2', 'kl', 'top1_agreement', 'nll_delta'])
        if r['variant'] in ['passthrough', 'replay00']:
            assert float(r['relative_l2']) == 0 and float(r['nll_delta']) == 0
    for key, r in keyed.items():
        if key[-1] == 'packedKV':
            baseline = keyed[(*key[:-1], 'KV11')]
            assert all(r[m] == baseline[m] for m in ['relative_l2', 'kl', 'top1_agreement', 'nll_delta'])
    primary = json.loads((RUN / 'analysis/primary_confirmatory.json').read_text())
    assert len(primary) == 4
    for r in primary:
        ids = r['document_ids']
        assert len(set(ids)) == r['n'] == 128 and r['confidence'] == .9875 and r['repeats'] == 5000
        assert all(units[d]['domain'] == r['category'] and units[d]['split'] == 'confirmatory' for d in ids)
        delta = [float(keyed[(d, 'bfloat16', 2048, r['regime'], 'K10')]['relative_l2']) -
                 float(keyed[(d, 'bfloat16', 2048, r['regime'], 'V01')]['relative_l2']) for d in ids]
        assert math.isclose(sum(delta) / len(delta), r['mean'], rel_tol=1e-12)
        assert r['ci'][0] <= r['mean'] <= r['ci'][1]
    if inputs:
        regenerated = json.loads(Path(inputs).read_text())
        assert len(regenerated['documents']) == 512
        for d in regenerated['documents']:
            ref = units[d['document_id']]
            assert all(d[k] == ref[k] for k in ['source_key', 'source_text_sha256', 'window_start_token', 'window_sha256', 'split', 'dtype_audit'])
            actual = hashlib.sha256(json.dumps(d['window_tokens'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert actual == ref['window_sha256']
    print(json.dumps(dict(status='PASS',hashed_files=len(manifest['files']),conditions=1152,source_units=512,
                          fp32_source_units=64,document_metric_rows=13824,primary_contrasts=4,
                          verification='public artifact integrity and consistency; not GPU replication')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', help='Optional regenerated frozen input manifest to compare by source/window identity')
    verify(parser.parse_args().inputs)
