"""Minimal, stdlib-only experiment shape checks.

These helpers cannot model paged KV allocation, allocator overhead or serving cost.
They are NOT a GPU attention implementation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionShape:
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self):
        if min(self.layers, self.query_heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads for standard GQA")
        if self.kv_heads > self.query_heads:
            raise ValueError("kv_heads cannot exceed query_heads")


@dataclass(frozen=True)
class WorkloadShape:
    prompt_tokens: int
    decode_tokens: int
    concurrency: int

    def __post_init__(self):
        if self.prompt_tokens < 0 or self.decode_tokens <= 0 or self.concurrency <= 0:
            raise ValueError("Invalid workload dimensions")


def dense_kv_payload_lower_bound_bytes(
    attention: AttentionShape, resident_tokens_total: int, bytes_per_element: int
) -> int:
    """Logical dense K+V payload only; excludes pages, padding, metadata and workspace.

    resident_tokens_total is the SUM of actual resident tokens across sequences.
    It is not tokens_per_sequence and must not be multiplied by concurrency again.
    """
    if resident_tokens_total < 0 or bytes_per_element <= 0:
        raise ValueError("Invalid resident length or element width")
    return 2 * attention.layers * attention.kv_heads * attention.head_dim * resident_tokens_total * bytes_per_element
