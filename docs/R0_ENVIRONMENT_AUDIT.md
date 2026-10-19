# SCOPE R0-A — Environment Audit and Baseline Preparation

**Status:** environment audit complete; benchmark proposal only; no vLLM benchmark or inference was run.  
**Audit time:** 2026-10-19 11:17 UTC  
**Active repository reference inspected:** `moonhyunseung/SCOPE` `main` at `34ee992bcbbf8143b0f70c43476da9b4abadc616`.

## Scope and repository identity

The active SCOPE project now targets evidence-led attention-serving efficiency. The previous quantized-KV study remains immutable under `legacy/`; it is not evidence for the new engine's serving performance. The server's currently open workspace is the older continuation-v2 research checkout, not a checkout of current `main`. It was inspected through its existing authenticated code-server terminal without changing files, services, containers, credentials, or GPU allocations.

The current main's `legacy/` root tree is `d79ce6d0b4251b51d8b1fd8d099d89b55ba1329c`, identical to the tree of preserved commit `7b3a1ca7b8c890b9da5b12a7ab5ad98193eb874f`. This verifies archive identity at the Git tree level. Historical files were not modified.

## Observed environment

All values below are point-in-time observations from the attached server workspace. They do not certify a production-serving environment.

| Item | Observation | Qualification |
|---|---|---|
| GPU | 1 × NVIDIA H200, compute capability 9.0 | Read with `nvidia-smi` and PyTorch device properties |
| GPU memory | `nvidia-smi`: 143,771 MiB total; PyTorch: 150,109,880,320 bytes total | These totals differ by about 615 MiB; resolve before setting a serving allocator limit |
| Driver | 580.126.09 | `nvidia-smi` |
| GPU snapshot | 0% utilization and no CUDA compute applications listed | Snapshot only; does not establish a future reservation or exclusive lock |
| System Python | Python 3.12.3; PyTorch and vLLM not importable | `python` command was unavailable; `python3` exists |
| Existing research virtual environment | Python 3.12.3, PyTorch 2.14.1+cu130, CUDA runtime 13.0, CUDA available, one GPU | This belongs to the older continuation-v2 checkout, not a validated current-main serving environment |
| CUDA compiler | `nvcc` 12.8.61 | Compiler/runtime combination has not been validated for vLLM |
| Attention packages in the existing research environment | vLLM absent; FlashAttention absent; Triton 3.8.0 present | No attention backend was selected or exercised |
| Container tooling | Docker CLI unavailable in the inspected shell | Other container/orchestration setups are UNKNOWN |
| Existing serving process | UNKNOWN | No service was started, stopped, or reconfigured |
| GPU/serving lock registry | UNKNOWN | The workspace does not expose a verified server-wide reservation/lock state |

### Existing checkpoint candidate

The local Hugging Face cache contains `Qwen/Qwen2.5-1.5B-Instruct` at snapshot revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, including model configuration, tokenizer files, and a `model.safetensors` file of 3,087,467,144 bytes. This is an available candidate, not a claim that it loads under vLLM. The full weight hash, tokenizer/model compatibility under a pinned vLLM release, and cache completeness for serving have not been validated. Other cached candidates were observed, but no need exists to expand the proposal beyond this smallest candidate.

## Smallest proposed baseline

The proposal uses the cached Qwen checkpoint above on the single observed H200, with BF16 model/KV dtype if the pinned vLLM release supports that exact combination. The two shapes below are taken from the repository's planning template; they are **not** approved or executed workloads:

| Scenario | Prompt tokens | Generated tokens | Concurrency | Prefix cache |
|---|---:|---:|---:|---|
| Long context, low concurrency | 8,192 | 128 | 1 | Off |
| Short context, higher concurrency | 512 | 128 | 16 | Off |

The fixed request-source corpus/hash, tokenizer revision, sampling policy, repetition count, and SLO are UNKNOWN and must be chosen and frozen in scientific review before any run. Do not substitute legacy prompts or token windows.

### Backend verification and measurement plan

Pin an exact vLLM commit/version before setup. At startup, retain the version-specific engine configuration and startup record that identifies the selected attention backend, and query the running worker's effective attention implementation through that release's supported introspection path. Confirm both agree with the intended supported backend. If the selected backend cannot be observed unambiguously, stop without collecting benchmark results. The exact introspection API and compatible vLLM revision remain UNKNOWN because vLLM is not installed.

For each approved scenario, record TTFT, per-request ITL/TPOT distributions (median, P95, P99 with sample counts), end-to-end latency, output tokens/s, requests/s, and goodput at a preregistered SLO. Attribute prefill, decode attention, projections, KV append, scheduler/launch, and whole-model GPU time separately where the pinned profiler supports it. Report profiler/tool overhead and keep instrumented attribution separate from uninstrumented serving timings.

Track persistent KV bytes, page/block padding, vLLM allocator active/allocated/reserved bytes, peak transient workspace, and sampled physical HBM. Reconcile the vLLM memory budget against both memory totals above and retain at least 44 GiB of physical HBM as a proposed safety reserve; a 96 GiB allocator ceiling is only a planning bound from the earlier dedicated-H200 work and must be recalculated from the live device before approval. No allocation setting was applied in this audit.

### Correctness, quality, limits, and budget

- Freeze model and tokenizer revisions, exact request-source hash, seeds/sampling, output-length policy, cache policy, and all thresholds before execution. Keep source/request units separate from repeated requests and repetitions.
- For the unmodified baseline, validate prompt token counts, request completion/error counts, generated token counts, finite outputs, and repeatability under the frozen deterministic setting. Any later approximation needs a separately preregistered behavior/quality contract; historical `0.025` remains unchanged and is not a universal serving-quality threshold.
- The proposal's maximum budget is one H200, at most 30 minutes of GPU wall time across both scenarios, at most 96 GiB allocator use, and at least 44 GiB physical reserve. These are proposed limits, not a resource reservation or run authorization.
- Main limitations: no vLLM installation/version, no selected backend, no validated current-main runtime, no frozen request source/hash or SLO, unknown server-wide lock state, and unresolved GPU memory-total discrepancy. Consequently, a compatible backend or an end-to-end baseline result cannot yet be claimed.

## Validation performed

- Current repository files and research contract were read from the exact current `main` commit named above.
- The five root CPU contract tests passed (`python -m unittest`, 5 tests) against source files fetched from that commit in an isolated local test snapshot; all three source files' Git blob SHA-1 values matched the commit. No package was installed.
- The archived root and current `legacy/` subtree have the same Git tree SHA (`d79ce6d0b4251b51d8b1fd8d099d89b55ba1329c`).
- JSON validation and report privacy checks are recorded with the corresponding R0-A change. No historical evidence, model weights, tokens, activations, credentials, process identifiers, or private server paths are included.
- No vLLM benchmark, inference, package installation, model download, service/container action, or GPU memory allocation was performed.

## Next decision required

R0-A is complete. Scientific review must select the pinned vLLM revision, fixed request source and quality/SLO contract, resolve the runtime/backend and memory-accounting prerequisites, and explicitly approve R0-B before any inference or GPU benchmark begins. Stop here until that review.
