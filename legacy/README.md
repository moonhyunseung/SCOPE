# SCOPE

**State-Conditioned Operator Profiling & Execution**

> Observe the numerical and structural states of real attention before designing semantics-preserving representation transformations and hardware-specific execution paths.

SCOPE is an **upstream research framework** for OFD. It is not an INT4-only project, an OFD benchmark archive, or a claim that any particular GPU kernel is faster.

**Latest completed collection (2026-10-18):** 512 natural source units, 1,024 BF16 conditions and a paired 128-condition FP32 audit on H200. All 1,152 conditions were collected and artifact/control checks passed. The separate dense cached/full fidelity threshold failed in 453 BF16 conditions; every FP32 audit condition passed. See [the measured results and limits](docs/POPULATION_RESULTS_20261018.md), [document-level data](results/population/46f7c2fa18e3d9c7/analysis/document_metrics.csv), and [ECDF](results/population/46f7c2fa18e3d9c7/analysis/relative_l2_ecdf.png).

## Research question

Given a defined autoregressive state-transition contract, which transformations of inputs, intermediate states, and persistent representations allow **equivalent model behavior** at lower measured cost on a particular GPU?

The research order is:

1. **P0 — Reference and capture:** fix a reproducible decode-attention reference, its inputs/outputs, and the observation schema.
2. **P1 — State distribution:** observe values and joint structure at `Q, K, V, Z=QK^T/√d, P=softmax(Z), W=P·S_V, Y=PV` across model/layer/head/context/workload.
3. **P2 — Transformation search:** generate testable candidates from observed patterns; permit packed-direct, tile-local unpacking, or temporary dense materialization rather than forbidding any one path.
4. **P3 — Hardware lowering:** evaluate H200-specific data movement, operand preparation, MMA, reduction, numerics, latency, and peak HBM. Establish a resident-KV/decode frontier.

**P0 smoke tests use synthetic tensors** to validate instrumentation. Synthetic results do not establish any empirical claim about a real model's attention distribution. Real-model pilot capture, controlled ablations, and independent-source numerical collection now have separate runners and evidence; the full resident-KV/decode frontier remains open.

## Core contracts

- **Behavior before shape:** matching tensor shapes or input/output sets is not the same as preserving a state transition. Verify values and observable future behavior under the declared model contract.
- **Snapshot vs trajectory:** equivalent attention outputs at one step do not imply equivalence after cache append or future state updates.
- **Representation vs execution:** distinguish quantization/model-quality loss from execution error against the *same* decoded representation.
- **No prohibited intermediate representation by fiat:** global reconstruction and tile-local expansion are competing execution designs. Judge them by full latency and peak allocated HBM under an identical contract.
- **No benchmark claims from traced runs:** instrumentation and profiling change runtime. Collect distributions in trace mode; collect performance in an uninstrumented replay mode.
- **No result without provenance:** record model/revision, dataset or prompt source, RNG seed, cache position, dtype, attention contract, backend, GPU, software versions, and trace overhead.

See [research contract](docs/RESEARCH_CONTRACT.md), [observation schema](docs/OBSERVATION_SCHEMA.md), [experiment plan](docs/EXPERIMENT_PLAN.md), and [hypothesis registry](docs/HYPOTHESES.md).

## Minimal reproducible smoke test

The synthetic reference requires Python 3.10+ and PyTorch. The complete research/test environment was exercised on Linux with Python 3.12 and Transformers 5.19.0; tiny-model tests run on CPU without downloading checkpoints.

```bash
python -m pip install -e ".[test]"
python scripts/p0_smoke.py --device cpu --length 128 --output results/p0_smoke.json
pytest -q
```

On a CUDA machine:

```bash
python scripts/p0_smoke.py --device cuda --length 2048 --output results/p0_cuda.json
```

The smoke test generates **synthetic** grouped-query decode attention, quantizes K/V to illustrative groupwise signed INT4 codes/scales, computes a decoded-quantized reference, captures `Q,K,V,Z,P,W,Y` statistics, and checks output against PyTorch SDPA. It is **not** an OFD model-reproduction result; quantizer details are illustrative and must be replaced by the target model's actual quantization contract.

## Repository map

```text
docs/
  RESEARCH_CONTRACT.md   Definitions, interfaces, equivalence, optimization
  OBSERVATION_SCHEMA.md  Trace fields, joint statistics, provenance, overhead
  EXPERIMENT_PLAN.md     P0 -> P3 gates and baseline requirements
  HYPOTHESES.md         Falsifiable transformation hypotheses
scope/
  reference.py           Decoded-quantized attention + illustrative INT4 quantizer
  observations.py        Bounded-memory tensor and joint-stat summaries
  runner.py              Bounded H200 pilot, gates and artifact verification
  engineering.py         Exploratory real-model numerical collection
  ablation.py            Controlled cache-fork and K/V factorial ablations
  population_inputs.py   Pinned public-source sampling and frozen windows
  population.py          Sequential resumable independent-source collection
  population_stats.py    Document-block bootstrap and distributions
scripts/
  p0_smoke.py            Synthetic reference/instrumentation sanity run
  verify_population_artifacts.py  CPU verification of published results
tests/
  test_reference.py      Semantics and shape tests
```

## Relation to OFD

**SCOPE asks what attention states actually look like, what semantics must be preserved, and why a hardware execution path could win.** OFD investigates concrete packed-KV representations and GPU implementations. An observation or hypothesis in SCOPE does not become an OFD performance result until independently reproduced and measured.

## Status

**Completed bounded numerical experiments; systems frontier not established.** The latest population study uses one pinned Qwen2.5-1.5B-Instruct checkpoint and one natural window from each of 256 WikiText article titles and 256 conservative Python project-name families. Repeated steps, dtypes and context lengths do not increase independent sample count. It retains dense KV and native prefill, so it does not establish a packed-resident implementation, end-to-end speedup, maximum resident capacity, general model-quality gain or new equivalence theorem.

To verify the published data without a GPU, run `python scripts/verify_population_artifacts.py`. For the frozen sampling, execution contract, dependencies and rerun procedure, see [population results and reproduction](docs/POPULATION_RESULTS_20261018.md). The historical pilot evidence is retained alongside the new results.
