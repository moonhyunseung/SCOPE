import unittest

from scope_engine import AttentionShape, WorkloadShape, dense_kv_payload_lower_bound_bytes


class AttentionContractTests(unittest.TestCase):
    def test_gqa_valid(self):
        a = AttentionShape(layers=28, query_heads=12, kv_heads=2, head_dim=128)
        self.assertEqual(a.kv_heads, 2)

    def test_bad_heads_rejected(self):
        with self.assertRaises(ValueError):
            AttentionShape(layers=4, query_heads=7, kv_heads=2, head_dim=64)

    def test_unsupported_shapes_not_implicitly_accepted(self):
        with self.assertRaises(ValueError):
            AttentionShape(layers=0, query_heads=4, kv_heads=4, head_dim=64)

    def test_workload_dimensions(self):
        self.assertEqual(WorkloadShape(512, 128, 1).decode_tokens, 128)
        with self.assertRaises(ValueError):
            WorkloadShape(512, 0, 1)

    def test_dense_kv_payload_only(self):
        a = AttentionShape(layers=2, query_heads=8, kv_heads=2, head_dim=64)
        self.assertEqual(dense_kv_payload_lower_bound_bytes(a, 100, 2), 102400)
        with self.assertRaises(ValueError):
            dense_kv_payload_lower_bound_bytes(a, -1, 2)


if __name__ == "__main__":
    unittest.main()
