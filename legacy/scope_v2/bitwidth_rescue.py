"""Additive INT8 bitwidth-only rescue. Previous INT4 source/results are immutable."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sqlite3, subprocess, sys, time, traceback, zipfile
from pathlib import Path
from scope.runner import atomic, canonical, digest, gpu_gate, now
from scope_v2 import acl_r2 as ref
from scope_v2 import role_fidelity as previous

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/bitwidth_rescue_v4.json'
ROLES = {'R4': (False, True), 'R5': (True, False), 'R6': (True, True)}
SOURCE_FILES = ['scope_v2/bitwidth_rescue.py', 'configs/bitwidth_rescue_v4.json', 'scope_v2/role_fidelity.py', 'scope/experiments.py'] + [n for n in previous.SOURCE_FILES if n != 'scope_v2/role_fidelity.py']


def code_hash():
    return digest(b''.join(n.encode() + b'\0' + (ROOT / n).read_bytes() for n in SOURCE_FILES))


def byte_equal(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


def role_value(x, selected):
    """Independent direct formula; production packing is only a check, not a reference."""
    return independent_quant8(x)[0] if selected else x


def checked_value(x, selected):
    from scope.engineering import quantize, unpack
    y = role_value(x, selected)
    if selected:
        p, s, _ = quantize(x, 8, 32)
        oracle = unpack(p, s, 8, 32, x.dtype)
        assert byte_equal(y, oracle), 'independent role quantization vs integer packing mismatch'
    else:
        assert byte_equal(y, x), 'unselected BF16 role modified'
    return y


def mixed_cache(states, role, controls, device='cuda'):
    """Persistent dense representation reference, NOT a packed-memory experiment."""
    from transformers.cache_utils import Cache, DynamicLayer
    select_k, select_v = ROLES[role]

    class MixedLayer(DynamicLayer):
        def update(self, k, v, *a, **kw):
            before = int(self.get_seq_length())
            old_k, old_v = self.keys.clone(), self.values.clone()
            qk, qv = checked_value(k, select_k), checked_value(v, select_v)
            result = super().update(qk, qv, *a, **kw)
            after = int(self.get_seq_length())
            assert before == 512 + self.append_index and k.shape[-2] == 1 and after == before + 1, 'append position/length drift'
            assert byte_equal(self.keys[..., :before, :], old_k) and byte_equal(self.values[..., :before, :], old_v), 'persistent prefix mutation'
            assert byte_equal(self.keys[..., before:after, :], qk) and byte_equal(self.values[..., before:after, :], qv), 'mixed append mismatch'
            controls.append(dict(layer=self.index, append_index=self.append_index, before=before, after=after,
                                 selected_k=select_k, selected_v=select_v, prefix_exact=True,
                                 append_exact=True, independent_integer_oracle_exact=True))
            self.append_index += 1
            return result

    layers = []
    for index, (k, v) in enumerate(states):
        layer = MixedLayer()
        layer.keys = checked_value(k, select_k).to(device).clone()
        layer.values = checked_value(v, select_v).to(device).clone()
        assert byte_equal(layer.keys.cpu(), role_value(k, select_k).cpu())
        assert byte_equal(layer.values.cpu(), role_value(v, select_v).cpu())
        layer.dtype = k.dtype
        layer.device = layer.keys.device
        layer.is_initialized = True
        layer.index, layer.append_index = index, 0
        layers.append(layer)
    return Cache(layers=layers)


def independent_quant8(x):
    import torch
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale = torch.maximum(grouped.abs().amax(-1), torch.tensor(1e-12, device=x.device)) / 127
    codes = torch.clamp(torch.round(grouped / scale[..., None]), -127, 127).to(torch.int8)
    decoded = (codes.float() * scale[..., None]).reshape_as(x).to(x.dtype)
    return decoded, codes.reshape_as(x), scale


def decision(cached_max, full_max, baseline_max, gate=.025):
    a, b, diverged = cached_max <= gate, full_max <= gate, baseline_max > gate
    label = 'REPRESENTATION_CACHED_FAIL' if not a else 'FULL_REFERENCE_FAIL' if not b else \
            'BASELINE_DIVERGENCE_UNRESOLVED' if diverged else 'LOCAL_SCREEN_PASS'
    return dict(cached_gate='PASS' if a else 'FAIL', full_gate='PASS' if b else 'FAIL',
                baseline_status='BASELINE_DIVERGENCE' if diverged else 'BASELINE_SCREEN_PASS', local_candidate_status=label)


def historical():
    dest = ROOT / 'results/role_fidelity_v3/6c2c9a058382dcb0/blocks/role'
    summary = ref.verify(dest)
    assert digest((dest/'manifest.json').read_bytes()) == '995619e8fa3c7283ad8bebf0b8e97071ee36d59a37ee40f12125707c839742da'
    assert summary['status'] == 'COLLECTED' and summary['all_controls_exact']
    parent = json.loads((ROOT/'data/role_fidelity_v3_preregistered.json').read_text())
    assert parent['source_sha256'] == previous.code_hash(), 'previous source fingerprint drift'
    return parent, summary, json.loads((dest/'replay_controls.json').read_text()), json.loads((dest/'source_summary.json').read_text())


def prepare(manifest, prepared):
    c = json.loads(CONFIG.read_text()); parent, summary, controls, source_rows = historical()
    assert c['roles'] == {k: list(v) for k, v in ROLES.items()}
    assert digest(manifest.read_bytes()) == c['input_manifest_sha256']
    data = json.loads(manifest.read_text())
    m = next(x for x in data['models'] if x['key'] == 'qwen1_5b')
    assert (m['id'], m['revision']) == (c['model'], c['revision'])
    inputs = []
    for x in parent['inputs']:
        d = next(d for d in data['documents'] if d['document_id'] == x['document_id'])
        assert d['split'] == 'screening' and ref.window(d)[2] == x['token_window_sha256']
        inputs.append(x)
    assert [x['document_id'] for x in inputs] == c['sources']
    frozen = dict(contract=c, source_sha256=code_hash(), source_files={n: digest((ROOT/n).read_bytes()) for n in SOURCE_FILES},
                  inputs=inputs, prior_role_manifest_sha256='995619e8fa3c7283ad8bebf0b8e97071ee36d59a37ee40f12125707c839742da',
                  prior_role_preregistration_sha256=digest((ROOT/'data/role_fidelity_v3_preregistered.json').read_bytes()),
                  prior_role_source_sha256=parent['source_sha256'])
    if prepared.exists(): assert json.loads(prepared.read_text()) == frozen, 'frozen preregistration drift'
    else: atomic(prepared, frozen)
    print(json.dumps(dict(run=digest(canonical(frozen).encode())[:16], source_sha256=code_hash(),
                         preregistration_sha256=digest(prepared.read_bytes()), conditions=6)))


def resource_check():
    previous.resource_check()


def setup(manifest, frozen):
    assert frozen['source_sha256'] == code_hash(), 'new source fingerprint drift'
    parent, old_summary, _, _ = historical()
    assert parent['inputs'] == frozen['inputs']
    model, docs, runtime = previous.setup(manifest, parent)
    for k in ['python','torch','transformers','cuda','gpu','tf32','deterministic','bf16_reduced_precision_reduction']:
        assert runtime[k] == old_summary['runtime'][k], 'reference runtime drift: '+k
    return model, docs, runtime


def role_decode(model, states, teacher, role):
    import torch
    controls = []
    cache = mixed_cache(states, role, controls)
    initial_hash = ref.state_hash(ref.cache_states(cache))
    logits, hashes = [], []
    for step in range(4):
        out = model(input_ids=torch.tensor([[teacher[step]]], device='cuda'), past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        logits.append(out.logits[:, -1].cpu())
        hashes.append(ref.state_hash(ref.cache_states(cache)))
        del out
        resource_check()
    assert len(controls) == 28 * 4, 'incomplete append controls'
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return logits, hashes, dict(initial_cache_sha256=initial_hash, append_controls=controls)


def run_screen(model, docs, frozen, dest):
    import torch
    parent, old_summary, old_controls, old_sources = historical()
    rows, summaries, replays = [], [], []
    for doc in docs:
        source, domain = doc['document_id'], doc['domain']; prefix, teacher, _ = ref.window(doc)
        saved = saved_controls = None
        baseline = next(x['max_full_relative_l2'] for x in old_sources if x['document_id'] == source and x['representation'] == 'R0')
        for replay in range(2):
            resource_check(); states = ref.prefill(model, prefix)
            prefill_hash = ref.state_hash(states)
            old = next(x for x in old_controls if x['document_id'] == source and x['replay'] == replay)
            assert prefill_hash == old['prefill_sha256'], 'historical prefill reference drift'
            full, _ = ref.decode_path(model, states, prefix, teacher, 'original_full')
            native, nh = ref.decode_path(model, states, prefix, teacher, 'native_cached')
            current = {'original_full': dict(logits=[ref.thash(x) for x in full]),
                       'native_cached': dict(logits=[ref.thash(x) for x in native], cache=nh)}
            assert current['original_full'] == old['paths']['original_full'], 'historical full logits drift'
            assert current['native_cached'] == old['paths']['native_cached'] == old['paths']['R0'], 'historical native R0 reference drift'
            # Recheck natural-reference metrics, not only hashes, before immutable result reuse.
            baseline_rows = [ref.logits_comparison(native[i], full[i], teacher[i+1]) for i in range(4)]
            assert max(x['relative_l2'] for x in baseline_rows) == baseline, 'historical baseline metric drift'
            role_controls = {}
            for role in ROLES:
                outputs, hashes, controls = role_decode(model, states, teacher, role)
                current[role] = dict(logits=[ref.thash(x) for x in outputs], cache=hashes)
                role_controls[role] = controls
                for reference, values in [('native_cached', native), ('original_full', full)]:
                    for step in range(4):
                        m = ref.logits_comparison(outputs[step], values[step], teacher[step+1])
                        m['natural_next_token_nll_delta'] = m['natural_next_token_nll_output'] - m['natural_next_token_nll_reference']
                        rows.append(dict(document_id=source, domain=domain, representation=role, replay=replay, step=step,
                                         reference=reference, gate='PASS' if m['relative_l2'] <= .025 else 'FAIL', **m))
            if replay: assert current == saved and role_controls == saved_controls, 'INT8 exact replay drift'
            saved, saved_controls = current, role_controls
            replays.append(dict(document_id=source, replay=replay, prefill_sha256=prefill_hash, paths=current,
                role_controls=role_controls, historical_native_R0_exact=True, historical_original_full_exact=True,
                historical_baseline_metric_exact=True, historical_runtime_exact=True))
            del states; gc.collect(); torch.cuda.empty_cache()
        for role in ROLES:
            subset = [x for x in rows if x['document_id'] == source and x['representation'] == role]
            cm = max(x['relative_l2'] for x in subset if x['reference'] == 'native_cached')
            fm = max(x['relative_l2'] for x in subset if x['reference'] == 'original_full')
            endpoints = {}
            for reference in frozen['contract']['references']:
                rr = [x for x in subset if x['reference'] == reference]
                endpoints[reference] = dict(max_relative_l2=max(x['relative_l2'] for x in rr),
                    max_kl_reference_to_output=max(x['kl_reference_to_output'] for x in rr),
                    top1_disagreement_steps=sum(x['top1_agreement'] == 0 for x in rr if x['replay'] == 0),
                    nll_delta_min=min(x['natural_next_token_nll_delta'] for x in rr),
                    nll_delta_max=max(x['natural_next_token_nll_delta'] for x in rr),
                    max_abs_nll_delta=max(abs(x['natural_next_token_nll_delta']) for x in rr))
            summaries.append(dict(document_id=source, domain=domain, representation=role, max_cached_relative_l2=cm,
                max_full_relative_l2=fm, baseline_max_relative_l2=baseline, **decision(cm, fm, baseline), endpoints=endpoints))
        atomic(dest.parent.parent/'source_progress.json', dict(completed_source=source, completed_conditions=len(summaries), updated=now()))
    assert len(rows) == 96 and len(summaries) == 6 and len(replays) == 4
    eligibility = {role: all(x['cached_gate'] == 'PASS' for x in summaries if x['representation'] == role) and
                    all(x['full_gate'] == 'PASS' for x in summaries if x['representation'] == role and x['domain'] == 'code') for role in ROLES}
    ref.write_csv(dest/'bitwidth_comparisons.csv', rows); atomic(dest/'source_summary.json', summaries)
    atomic(dest/'replay_controls.json', replays)
    return dict(status='COLLECTED', independent_source_n=2, source_representation_conditions=6, steps=4, replays=2,
        metric_rows=96, all_controls_exact=True, prior_R0_R3_immutable=True, followup_review_eligibility=eligibility,
        stop_uniform_family=all(any(x['cached_gate']=='FAIL' for x in summaries if x['representation']==role) for role in ROLES),
        eligibility_is_not_system_adoption=True, original_failures_preserved=True, scope=frozen['contract']['scope'])


def seal(dest, result, frozen):
    dest.mkdir(parents=True, exist_ok=True)
    atomic(dest/'summary.json', dict(result, source_sha256=frozen['source_sha256'], publication='PENDING'))
    atomic(dest/'preregistered_contract.json', frozen)
    (dest/'report.md').write_text('# INT8 bitwidth rescue v4\n\n' + json.dumps(result, indent=2) +
        '\n\nEvery source/role, both full and cached references, and all failures retained. '
        'Steps/replays are repeated observations, no CI. Dense mixed-cache reference is not a memory-saving measurement. '
        'No token IDs, raw logits or activations are included. Subsequent research remains deferred pending advisor review.\n')
    files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes())) for p in dest.iterdir()
             if p.is_file() and p.name not in ['manifest.json', 'SHA256SUMS.txt']}
    atomic(dest/'manifest.json', dict(files=files, source_sha256=frozen['source_sha256'], sealed=now()))
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n, v in sorted(files.items())))


def worker(manifest, prepared, dest):
    frozen = json.loads(prepared.read_text())
    result = dict(status='RUNNING')
    dest.mkdir(parents=True, exist_ok=True)
    try:
        model, docs, runtime = setup(manifest, frozen)
        import torch
        with torch.inference_mode():
            result.update(run_screen(model, docs, frozen, dest))
        result['runtime'] = runtime
    except Exception as exc:
        result.update(status='CONTROL_FAIL' if isinstance(exc, AssertionError) else 'FAIL', error_type=type(exc).__name__)
        (dest.parent.parent/'failure.private.log').write_text(traceback.format_exc())
    seal(dest, result, frozen)
    if result['status'] in ['CONTROL_FAIL', 'FAIL']:
        raise RuntimeError('sealed role screen failure; no automatic repair')


def run(manifest, prepared):
    import fcntl
    frozen = json.loads(prepared.read_text())
    assert frozen['source_sha256'] == code_hash()
    identity = digest(canonical(frozen).encode())[:16]
    folder = ROOT/'results/bitwidth_rescue_v4'/identity
    dest = folder/'blocks/rescue'
    lock = (ROOT.parent/'engineering_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if (dest/'manifest.json').exists():
            ref.verify(dest)
            return  # Immutable success/failure, no duplicate execution.
        folder.mkdir(parents=True, exist_ok=True)
        atomic(folder/'campaign.json', dict(run=identity, source_sha256=code_hash(),
               prepared_sha256=digest(prepared.read_bytes()), budget_seconds=1200))
        atomic(folder/'status.json', dict(status='RUNNING', updated=now()))
        with (folder/'worker.private.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'scope_v2.bitwidth_rescue', 'worker', '--manifest',
                str(manifest.resolve()), '--prepared', str(prepared.resolve()), '--dest', str(dest.resolve())],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                rc = 124
        if not (dest/'manifest.json').exists():
            seal(dest, dict(status='TIMEOUT' if rc == 124 else 'FAIL', reason='own bounded worker ended without accepted result'), frozen)
        result = ref.verify(dest)
        db = sqlite3.connect(folder/'registry.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)')
        db.execute('INSERT OR IGNORE INTO publications VALUES(?,?,?,?)', ('rescue', 'PENDING', digest((dest/'manifest.json').read_bytes()), now()))
        db.commit(); db.close()
        atomic(folder/'status.json', dict(status='COMPLETE' if rc == 0 else 'STOPPED_WITH_FAILURE', result=result['status'], updated=now()))
        with zipfile.ZipFile(ROOT.parent/f'SCOPE_BITWIDTH_RESCUE_{identity}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in dest.iterdir():
                z.write(p, p.relative_to(ROOT).as_posix())
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run', 'worker'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--dest', type=Path)
    a = p.parse_args()
    if a.mode == 'prepare': prepare(a.manifest, a.prepared)
    elif a.mode == 'run': run(a.manifest, a.prepared)
    else: worker(a.manifest, a.prepared, a.dest)
