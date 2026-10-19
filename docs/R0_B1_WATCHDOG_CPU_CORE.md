# R0-B1 watchdog core — CPU-only preparation

This is a hardware-agnostic, standard-library-only watchdog decision core. It
does not inspect NVML/CUDA, establish a reservation, start a model server, or
discover/terminate processes. It is **not a ready-to-run watchdog**.

The only implemented termination operation is an injected
`OwnedChildTree.terminate_owned_tree(reason)` interface. A production adapter
must prove that this capability is bound to the child tree started by the
same experiment. No process-name/PID scan or global kill fallback exists.

Each provider observation must come from a trusted live source and include
single-H200/SM90 identity, owner-scoped lock evidence, physical free HBM,
experiment HBM baseline/current, allocator use, disk free bytes, worker/OOM
failure signals, and a monotonic timestamp. The core rejects missing or stale
samples (>5 s) and checks the Issue #4 limits: physical free HBM >=44 GiB,
experiment HBM growth <=88 GiB, allocator <=96 GiB, disk free >=24 GiB, and
wall time <=1800 s. The monitor loop samples once per second.

No trusted resource provider, reservation adapter, child-tree controller, or
deployment integration is included. Therefore this artifact does not change
the R0-B1 readiness state: GPU lock remains UNKNOWN and watchdog remains NOT
READY until those external pieces are independently verified.

Run the CPU-only tests with:

```powershell
python -m unittest discover -s tests -v
```

