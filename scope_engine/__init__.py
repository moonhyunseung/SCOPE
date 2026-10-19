"""SCOPE active-track contracts (not a working attention backend)."""

from .contracts import AttentionShape, WorkloadShape, dense_kv_payload_lower_bound_bytes

__all__ = ["AttentionShape", "WorkloadShape", "dense_kv_payload_lower_bound_bytes"]
