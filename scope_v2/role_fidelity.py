"""Additive K/V-role-only fidelity screen. No LLM calls or optimized kernels."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sqlite3, subprocess, sys, time, traceback, zipfile
from pathlib import Path
from scope.runner import atomic, canonical, digest, gpu_gate, now
from scope_v2 import acl_r2 as ref

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/role_fidelity_v3.json'
ROLES = {'R0': (False, False), 'R1': (True, False), 'R2': (False, True), 'R3': (True, True)}
SOURCE_FILES = ['scope_v2/role_fidelity.py', 'configs/role_fidelity_v3.json', 'scope/runner.py',
                'scope_v2/acl_r2.py', 'scope_v2/persistent.py', 'scope_v2/packed_attention.py',
                'scope_v2/resident.py', 'scope/engineering.py']


def code_hash():
    return digest(b''.join(n.encode() + b'\0' + (ROOT / n).read_bytes() for n in SOURCE_FILES))


def byte_equal(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


def role_value(x, selected):
    """Independent direct formula; production packing is only a check, not a reference."""
    return ref.independent_quant(x)[0] if selected else x


def checked_value(x, selected):
    from scope.engineering import quantize, unpack
    y = role_value(x, selected)
    if selected:
        p, s, _ = quantize(x, 4, 32)
        oracle = unpack(p, s, 4, 32, x.dtype)
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


def decision(cached_max, full_max, baseline_max, gate=.025):
    a, b, diverged = cached_max <= gate, full_max <= gate, baseline_max > gate
    return dict(cached_gate='PASS' if a else 'FAIL', full_gate='PASS' if b else 'FAIL',
                baseline_status='BASELINE_DIVERGENCE' if diverged else 'BASELINE_SCREEN_PASS',
                local_candidate_status='FAIL' if not (a and b) else
                    'BASELINE_DIVERGENCE_UNRESOLVED' if diverged else 'LOCAL_SCREEN_PASS')


def prepare(manifest, prepared):
    c = json.loads(CONFIG.read_text())
    assert c['roles'] == {k: list(v) for k, v in ROLES.items()}
    assert digest(manifest.read_bytes()) == c['input_manifest_sha256']
    data = json.loads(manifest.read_text())
    model = next(x for x in data['models'] if x['key'] == 'qwen1_5b')
    assert (model['id'], model['revision']) == (c['model'], c['revision'])
    prior_path = ROOT / 'data/acl_r2_preregistered_repair1.json'
    prior = json.loads(prior_path.read_text())
    inputs = []
    for source in c['sources']:
        d = next(x for x in data['documents'] if x['document_id'] == source)
        assert d['split'] == 'screening'
        h = ref.window(d)[2]
        previous = next(x for x in prior['inputs'] if x['document_id'] == source)
        assert h == previous['token_window_sha256']
        inputs.append(dict(document_id=source, domain=d['domain'], token_window_sha256=h))
    parent = ROOT / 'results/quality_gate_v2/d599462d38ba9322/blocks/quality'
    assert ref.verify(parent)['status'] == 'REPRESENTATION_FIDELITY_FAIL'
    frozen = dict(contract=c, source_sha256=code_hash(),
                  source_files={n: digest((ROOT/n).read_bytes()) for n in SOURCE_FILES}, inputs=inputs,
                  prior_quality_manifest_sha256=digest((parent/'manifest.json').read_bytes()),
                  prior_preregistration_sha256=digest(prior_path.read_bytes()))
    if prepared.exists():
        assert json.loads(prepared.read_text()) == frozen, 'frozen preregistration drift'
    else:
        atomic(prepared, frozen)
    print(json.dumps(dict(run=digest(canonical(frozen).encode())[:16], source_sha256=code_hash(),
                          preregistration_sha256=digest(prepared.read_bytes()), conditions=8)))


def resource_check():
    import torch
    free, _ = torch.cuda.mem_get_info()
    assert free >= 44 * 2**30, 'physical reserve below44GiB'
    assert __import__('shutil').disk_usage(ROOT).free >= 24 * 2**30, 'disk reserve below24GiB'


def setup(manifest, frozen):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TRITON_CACHE_DIR'] = str(ROOT.parent/'triton_cache_continuation_v2')
    import torch, transformers, platform
    from transformers import AutoModelForCausalLM
    c = frozen['contract']
    assert frozen['source_sha256'] == code_hash()
    assert digest(manifest.read_bytes()) == c['input_manifest_sha256']
    gpu_gate(dict(reserve_hbm_gib=44, minimum_free_hbm_gib=8, reserve_disk_gib=24))
    torch.set_num_threads(4)
    torch.manual_seed(20261019)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    free, total = torch.cuda.mem_get_info()
    cap = min(96 * 2**30, free - 44 * 2**30)
    assert cap >= 8 * 2**30, 'insufficient dedicated allocation headroom'
    torch.cuda.set_per_process_memory_fraction(cap / total)
    model = AutoModelForCausalLM.from_pretrained(c['model'], revision=c['revision'], local_files_only=True,
             trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').eval().cuda()
    assert model.config.num_hidden_layers == 28 and model.config.num_attention_heads == 12
    data = json.loads(manifest.read_text())
    docs = [next(d for d in data['documents'] if d['document_id'] == x['document_id']) for x in frozen['inputs']]
    assert all(ref.window(d)[2] == x['token_window_sha256'] for d, x in zip(docs, frozen['inputs']))
    resource_check()
    return model, docs, dict(python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        tf32=False, deterministic=True, allocator_limit_bytes=cap, physical_reserve_gib=44,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


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
    rows, summaries, replays = [], [], []
    historical = ROOT/'results/quality_gate_v2/d599462d38ba9322/blocks/quality/replay_controls.json'
    old_controls = json.loads(historical.read_text())
    for doc in docs:
        source, domain = doc['document_id'], doc['domain']
        prefix, teacher, _ = ref.window(doc)
        saved, saved_prefill, saved_controls = None, None, None
        for replay in range(2):
            resource_check()
            states = ref.prefill(model, prefix)
            prefill_hash = ref.state_hash(states)
            if replay:
                assert prefill_hash == saved_prefill, 'prefill replay drift'
            full, _ = ref.decode_path(model, states, prefix, teacher, 'original_full')
            native, nh = ref.decode_path(model, states, prefix, teacher, 'native_cached')
            current = {'original_full': dict(logits=[ref.thash(x) for x in full]),
                       'native_cached': dict(logits=[ref.thash(x) for x in native], cache=nh)}
            role_controls = {}
            for role in ROLES:
                outputs, hashes, controls = role_decode(model, states, teacher, role)
                role_controls[role] = controls
                current[role] = dict(logits=[ref.thash(x) for x in outputs], cache=hashes)
                if role == 'R0':
                    assert current[role] == current['native_cached'], 'mixed R0 differs from unmodified native reference'
                if source == frozen['contract']['sources'][0] and role == 'R3':
                    assert current[role]['logits'] == old_controls[0]['logit_sha256']['independent_dense_quantized'], 'historical R3 logits changed'
                    assert hashes == old_controls[0]['cache_sha256']['independent_dense_quantized'], 'historical R3 caches changed'
                for reference, values in [('native_cached', native), ('original_full', full)]:
                    for step in range(4):
                        m = ref.logits_comparison(outputs[step], values[step], teacher[step+1])
                        m['natural_next_token_nll_delta'] = m['natural_next_token_nll_output'] - m['natural_next_token_nll_reference']
                        rows.append(dict(document_id=source, domain=domain, representation=role, replay=replay,
                                         step=step, reference=reference, gate='PASS' if m['relative_l2'] <= .025 else 'FAIL', **m))
            if replay:
                assert current == saved and role_controls == saved_controls, 'exact logits/cache/controls replay drift'
            saved, saved_prefill, saved_controls = current, prefill_hash, role_controls
            replays.append(dict(document_id=source, replay=replay, prefill_sha256=prefill_hash, paths=current,
                                role_controls=role_controls, native_R0_exact=True,
                                historical_R3_exact=True if source == frozen['contract']['sources'][0] else None))
            del states
            gc.collect()
            torch.cuda.empty_cache()
        baseline = max(x['relative_l2'] for x in rows if x['document_id'] == source and x['representation'] == 'R0' and x['reference'] == 'original_full')
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
            summaries.append(dict(document_id=source, domain=domain, representation=role,
                max_cached_relative_l2=cm, max_full_relative_l2=fm, baseline_max_relative_l2=baseline,
                **decision(cm, fm, baseline), endpoints=endpoints))
        # Source checkpoint is private until the whole immutable block is sealed.
        atomic(dest.parent.parent/'source_progress.json', dict(completed_source=source, completed_conditions=len(summaries), updated=now()))
    assert len(rows) == 128 and len(summaries) == 8 and len(replays) == 4
    ref.write_csv(dest/'role_comparisons.csv', rows)
    atomic(dest/'source_summary.json', summaries)
    atomic(dest/'replay_controls.json', replays)
    return dict(status='COLLECTED', independent_source_n=2, source_representation_conditions=8,
        steps=4, replays=2, metric_rows=128, all_controls_exact=True,
        candidate_roles={role: 'LOCAL_SCREEN_PASS' if all(x['local_candidate_status'] == 'LOCAL_SCREEN_PASS' for x in summaries if x['representation'] == role)
                         else 'NOT_ACCEPTED' for role in ['R1', 'R2', 'R3']},
        original_failures_preserved=True, scope=frozen['contract']['scope'])


def seal(dest, result, frozen):
    dest.mkdir(parents=True, exist_ok=True)
    atomic(dest/'summary.json', dict(result, source_sha256=frozen['source_sha256'], publication='PENDING'))
    atomic(dest/'preregistered_contract.json', frozen)
    (dest/'report.md').write_text('# K/V role fidelity screen v3\n\n' + json.dumps(result, indent=2) +
        '\n\nEvery source/role, both full and cached references, and all failures retained. '
        'Steps/replays are repeated observations, no CI. Dense mixed-cache reference is not a memory-saving measurement. '
        'No token IDs, raw logits or activations are included. INT8 remains deferred.\n')
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
    folder = ROOT/'results/role_fidelity_v3'/identity
    dest = folder/'blocks/role'
    lock = (ROOT.parent/'engineering_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if (dest/'manifest.json').exists():
            ref.verify(dest)
            return  # Immutable success/failure, no duplicate execution.
        folder.mkdir(parents=True, exist_ok=True)
        atomic(folder/'campaign.json', dict(run=identity, source_sha256=code_hash(),
               prepared_sha256=digest(prepared.read_bytes()), budget_seconds=1500))
        atomic(folder/'status.json', dict(status='RUNNING', updated=now()))
        with (folder/'worker.private.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'scope_v2.role_fidelity', 'worker', '--manifest',
                str(manifest.resolve()), '--prepared', str(prepared.resolve()), '--dest', str(dest.resolve())],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=1500)
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
        db.execute('INSERT OR IGNORE INTO publications VALUES(?,?,?,?)', ('role', 'PENDING', digest((dest/'manifest.json').read_bytes()), now()))
        db.commit(); db.close()
        atomic(folder/'status.json', dict(status='COMPLETE' if rc == 0 else 'STOPPED_WITH_FAILURE', result=result['status'], updated=now()))
        with zipfile.ZipFile(ROOT.parent/f'SCOPE_ROLE_FIDELITY_{identity}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
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
