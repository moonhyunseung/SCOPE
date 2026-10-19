# SCOPE R0-B0 gate review and corrective preparation

Status: `PREPARATION_ONLY_NOT_EXECUTED`; R0-B1 remains `NOT_AUTHORIZED`.
Review basis: ACL's review of main commit `d2094df29b3176dcf4e15c09cfd25cb9ca359bf0`.

## Frozen source and conditions

- vLLM: v0.32.0, commit `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- RandomDataset source blobs remain the values in `configs/r0_b0_contract.json`.
- Model/tokenizer candidate and revision remain unchanged.
- W1 remains 8,192/128 prompt/output tokens, concurrency 1, 64 measured unique requests, 3 repetitions and 4 warm-ups per repetition.
- W2 remains 512/128, concurrency 16, 128 measured unique requests, 3 repetitions and 16 warm-ups per repetition.
- Seed 20261019, range ratio 0 and prefix length 0 remain fixed. No request substitution is allowed.

## Request identity correction

The pinned RandomDataset source subtracts `num_special_tokens_to_add()` from
the supplied input length, then re-encodes its generated text without adding
special tokens. Therefore the existing compensation
`requested_prompt_len + num_special_tokens_to_add()` is source-consistent and
is retained. It does not establish how a future server tokenizes or reorders
the request.

The generator now defines an ordered measured set (`W1-0..63` / `W2-0..127`)
and a disjoint `WU-<workload>-R<rep>-<index>` warm-up namespace. CPU validators
check exact count, ID/order, unique prompt hashes, prompt/output lengths, and
warm-up separation. A separate parity check requires server-returned request
IDs, order, token-ID hashes and lengths to match the sealed local manifest;
any auto-tokenization or reordering mismatch blocks measurement. Tokenizer
snapshot manifests hash every regular file by relative path, size and SHA-256;
symlinks are rejected. The manifest hash is content-derived, not inferred from
the revision name. Actual requests and tokenizer content remain ungenerated
and all corresponding hashes remain pending.

## Backend identity and resource records

Backend matching no longer uses substrings. The allowlist is grounded in the
pinned `registry.py`, `flash_attn.py` and `triton_attn.py` source blobs. For
each worker and layer, the CPU validator requires the exact enum, backend
class, implementation class and dispatch entrypoint. It also requires the
runtime override state to be false; FlashAttention records must include the
selected FA version, and Triton records must include a dispatch identifier.
The three Git-blob SHA-1 values are
`207c112a384c1e1450362c7c91ac31893fd783bd`,
`6e906bc520c49e90bc6658c4a189b54215e5e28e`, and
`a95b0c1d0e13e44154f9d63950769823b0abc47e`, respectively.
The allowlist is deliberately partial; unlisted, overridden, custom, mixed,
missing or composite identities return `BACKEND_UNVERIFIED`. A passing record
returns `BACKEND_EVIDENCE_CONSISTENT`, which describes field consistency only.
It does not attest that the evidence came from a live server or resolve
Triton's JIT kernel choice. That lower-level kernel identity remains a blocker
until observed by a separately reviewed runtime trace.

Resource-record validation rejects Booleans masquerading as numbers, missing,
negative, non-finite and out-of-bound measurements, and invalid GPU counts.
It checks the frozen limits: one H200, exclusive-lock evidence, physical free
HBM >=44 GiB, workload HBM increase <=88 GiB, allocator <=96 GiB, disk free
>=24 GiB and wall time <=1,800 seconds. A passing record means only
`RECORD_VALID`; it does not mean resources were reserved or protected.

Actual execution would need a separately reviewed trusted provider that
establishes owner-scoped exclusive lock and timestamped live GPU/disk values,
plus a watchdog that stops only the run-owned child process tree on lock loss,
resource breach, worker failure or deadline. No such live monitor or lock is
implemented here. It must never terminate unrelated processes or services.

## R0-B1 decision packet

`configs/r0_b1_preflight_template.json` preserves the frozen W1/W2 dimensions
and contains the result record fields, a proposed closed-loop dispatch policy,
concurrency caps, queue-wait observations, warm-up ordering, backend evidence
requirements, resource-monitor interface, stop conditions and explicit
researcher approval checklist. The dispatch policy is a proposal: dispatch
the next sealed request when a client slot is available and record offered
rate, completion rate and queue wait; it claims no fixed external arrival
rate. Final workload/SLO decisions remain for scientific review.

The CPU readiness validator reports blockers for pending hashes, unverified
server tokenization/backend, unknown lock ownership, absent live monitor,
incomplete dispatch/result schemas, or missing approval. It is a contract
checker only and cannot authorize a benchmark.

## Validation and stop boundary

Tests use synthetic CPU fixtures only. They do not load vLLM, Transformers,
PyTorch, tokenizers, a model, or CUDA; do not generate actual prompts; and do
not inspect GPU resources. No package installation, download, server action,
profiling or inference occurred. R0-B1 remains blocked pending verified input
hashes, target-runtime compatibility, observed backend, trusted lock/monitor,
and separate explicit researcher approval.
