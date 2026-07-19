# Copyright © 2024 Apple Inc.

import gc
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestZeroCopy(mlx_tests.MLXTestCase):
    """Tests for zero-copy construction: mx.array(cpu_buffer, copy=False).

    On unified memory MLX can adopt a page-aligned CPU buffer via Metal's
    newBufferWithBytesNoCopy instead of copying. Correctness must hold whether
    or not the adopt path is taken (it falls back to a copy when the source is
    not page-aligned), so these tests assert values + lifetime, and only assert
    *sharing* for large (page-aligned) buffers where the adopt path is expected.
    """

    def test_copy_false_values_match(self):
        for dt in [np.int32, np.int64, np.float32, np.uint8, np.float16]:
            a = (np.arange(1024) % 7).astype(dt)
            x = mx.array(a, copy=False)
            self.assertTrue(np.array_equal(np.array(x), a), msg=str(dt))

    def test_default_copies(self):
        # Default (copy=True) must be a true copy: mutating the source afterwards
        # must not change the mlx array.
        a = np.arange(1_000_000, dtype=np.int32)
        x = mx.array(a)  # copy=True default
        a[0] = 12345
        mx.eval(x)
        self.assertNotEqual(int(x[0]), 12345)

    def test_copy_false_shares_for_large_aligned(self):
        # Large buffers are page-aligned -> adopt path -> zero-copy view.
        a = np.arange(1_000_000, dtype=np.int32)
        if a.ctypes.data % 16384 != 0:
            self.skipTest("source buffer not page-aligned; adopt path not taken")
        x = mx.array(a, copy=False)
        a[0] = 12345
        mx.eval(x)
        self.assertEqual(int(x[0]), 12345)

    def test_dtype_conversion_falls_back_to_copy(self):
        # copy=False with a converting dtype must still work (falls back to copy).
        a = np.arange(1024, dtype=np.float64)  # float64 -> float32 conversion
        x = mx.array(a, copy=False, dtype=mx.float32)
        self.assertEqual(x.dtype, mx.float32)
        self.assertTrue(np.allclose(np.array(x), a.astype(np.float32)))

    def test_source_lifetime(self):
        # The adopted array must keep the source buffer alive: drop the source
        # reference, force GC, then use the array. Must not crash / corrupt.
        def make():
            a = np.arange(1_000_000, dtype=np.float32) + 0.5
            return mx.array(a, copy=False)

        x = make()
        gc.collect()
        mx.eval(x + 1)
        self.assertAlmostEqual(float(x[10]), 10.5, places=5)


if __name__ == "__main__":
    unittest.main()
