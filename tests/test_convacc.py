import unittest

import numpy as np

from convacc import ConvSpec, GemmSpec, make_plan, pack_weights
from convacc.reference import execute_packed, reference_for_packed


class ConvAccTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _check(self, spec, weights, layout, activation_shape):
        activations = self.rng.integers(-4, 5, size=activation_shape, dtype=np.int8)
        packed = pack_weights(spec, weights, layout)
        actual = execute_packed(packed, activations)
        expected = reference_for_packed(packed, activations, weights, layout)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(packed.to_bytes()), packed.manifest()["packed_weight_bytes"])
        return packed

    def test_dense_conv(self):
        spec = ConvSpec(n=1, hi=4, wi=4, ci=3, ho=4, wo=4, co=5, kh=3, kw=3, ph=1, pw=1)
        weights = self.rng.integers(-3, 4, size=(3, 3, 1, 3, 5), dtype=np.int8)
        packed = self._check(spec, weights, "HWGIO", (1, 4, 4, 3))
        self.assertEqual(packed.plan.tasks[0].k, 27)
        self.assertEqual(packed.tasks[0].weights.shape, (1, 27, 8))
        logical_macs = packed.plan.tasks[0].m * packed.plan.tasks[0].n * packed.plan.tasks[0].k
        self.assertLess(packed.plan.useful_macs, logical_macs)

    def test_dilated_group_conv(self):
        spec = ConvSpec(
            n=1, hi=5, wi=5, ci=4, ho=5, wo=5, co=6, kh=3, kw=3,
            dh=2, dw=2, ph=2, pw=2, groups=2,
        )
        weights = self.rng.integers(-3, 4, size=(3, 3, 2, 2, 3), dtype=np.int8)
        packed = self._check(spec, weights, "HWGIO", (1, 5, 5, 4))
        self.assertEqual(len(packed.tasks), 2)
        self.assertTrue(all(p.task.mode == "dense" for p in packed.tasks))

    def test_channelwise_conv(self):
        spec = ConvSpec(n=1, hi=4, wi=4, ci=3, ho=4, wo=4, co=3, kh=3, kw=3, ph=1, pw=1, groups=3)
        weights = self.rng.integers(-3, 4, size=(3, 3, 3), dtype=np.int8)
        packed = self._check(spec, weights, "HWM", (1, 4, 4, 3))
        self.assertEqual(packed.plan.tasks[0].mode, "channelwise")
        self.assertEqual(packed.plan.tasks[0].k, 9)

    def test_depthwise_multiplier(self):
        spec = ConvSpec(n=1, hi=3, wi=3, ci=2, ho=3, wo=3, co=4, kh=3, kw=3, ph=1, pw=1, groups=2)
        weights = self.rng.integers(-3, 4, size=(3, 3, 4), dtype=np.int8)
        packed = self._check(spec, weights, "HWM", (1, 3, 3, 2))
        self.assertEqual(packed.plan.tasks[0].n, 4)

    def test_deconv_phases(self):
        spec = ConvSpec(
            n=1, hi=3, wi=3, ci=2, ho=5, wo=5, co=4, kh=3, kw=3,
            sh=2, sw=2, ph=1, pw=1, kind="deconv2d",
        )
        weights = self.rng.integers(-3, 4, size=(3, 3, 2, 4), dtype=np.int8)
        packed = self._check(spec, weights, "HWIO", (1, 3, 3, 2))
        self.assertEqual(len(packed.tasks), 4)
        self.assertEqual(sum(t.task.k for t in packed.tasks), 18)

    def test_sparse_deconv_keeps_empty_phases(self):
        # kh=kw=1 with ph=pw=1 and stride 2 means only the rh=rw=0 phase has a
        # tap -- the other three output sub-grids receive no contribution at
        # all. They must still be planned (with k == 0) so their positions get
        # written with `requant(0)`; dropping them left holes in the image.
        spec = ConvSpec(
            n=1, hi=2, wi=2, ci=2, ho=3, wo=3, co=3, kh=1, kw=1,
            sh=2, sw=2, kind="deconv2d",
        )
        weights = self.rng.integers(-3, 4, size=(1, 1, 2, 3), dtype=np.int8)
        packed = self._check(spec, weights, "HWIO", (1, 2, 2, 2))
        self.assertEqual(len(packed.tasks), 4)
        self.assertEqual(packed.tasks[0].task.name, "phase_0_0")
        self.assertEqual(packed.tasks[0].task.k, 2)
        empty = [t for t in packed.tasks if t.task.k == 0]
        self.assertEqual(len(empty), 3)
        # Every (oh, ow) must be owned by exactly one phase.
        covered = set()
        for task in packed.tasks:
            t = task.task
            for row in range(t.mh):
                for col in range(t.mw):
                    covered.add((t.phase_h + row * t.out_step_h,
                                 t.phase_w + col * t.out_step_w))
        self.assertEqual(covered, {(h, w) for h in range(3) for w in range(3)})

    def test_gemm(self):
        spec = GemmSpec(m=5, k=7, n=11)
        weights = self.rng.integers(-3, 4, size=(7, 11), dtype=np.int8)
        packed = self._check(spec, weights, "KN", (5, 7))
        self.assertEqual(packed.tasks[0].weights.shape, (2, 7, 8))
        np.testing.assert_array_equal(packed.tasks[0].weights[1, :, 4:], 0)

    def test_weight_layout_conversions(self):
        spec = ConvSpec(n=1, hi=3, wi=3, ci=2, ho=3, wo=3, co=4, kh=3, kw=3, ph=1, pw=1)
        hwio = self.rng.integers(-3, 4, size=(3, 3, 2, 4), dtype=np.int8)
        oihw = hwio.transpose(3, 0, 1, 2)
        self.assertEqual(pack_weights(spec, hwio, "HWIO").to_bytes(), pack_weights(spec, oihw, "OIHW").to_bytes())

        grouped_spec = ConvSpec(n=1, hi=3, wi=3, ci=4, ho=3, wo=3, co=4, kh=3, kw=3, ph=1, pw=1, groups=2)
        hwgio = self.rng.integers(-3, 4, size=(3, 3, 2, 2, 2), dtype=np.int8)
        goihw = hwgio.transpose(2, 4, 0, 1, 3)
        self.assertEqual(
            pack_weights(grouped_spec, hwgio, "HWGIO").to_bytes(),
            pack_weights(grouped_spec, goihw, "GOIHW").to_bytes(),
        )

    def test_plan_capacity_and_utilization(self):
        spec = GemmSpec(m=16, k=16, n=16)
        plan = make_plan(spec)
        self.assertEqual(plan.cycles, 64)
        self.assertAlmostEqual(plan.utilization, 1.0)
        with self.assertRaises(ValueError):
            ConvSpec(n=4, hi=64, wi=64, ci=32, ho=1, wo=1, co=1, kh=1, kw=1)


if __name__ == "__main__":
    unittest.main()
