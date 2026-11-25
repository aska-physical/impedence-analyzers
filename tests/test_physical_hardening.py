"""Physical & identifiability regression tests for aska-physical/impedence-analyzers.

Covers:
  1. Real single-frequency 50 kHz Cole Jacobian rank <= 2 < 4 (replacing constant np.ones((2, 4)))
     and Fisher singularity failure on 1-frequency sweeps.
  2. Real two-wire 2x150 Ohm contact impedance + CPE spectrum synthesis & fit (replacing hardcoded arithmetic).
  3. Explicit C_stray_pF identification in residual (recovering 45 pF and unbiased R_inf=410 Ohm) and
     extended Fisher rank deficiency proving series R_contact cannot separate from bulk (R_inf, R0).
  4. Exact method-of-images finite-thickness slab kernel K_slab(a, t): recovers half-space 0.268152 m
     at t >> a and 2D logarithmic sheet kernel 2*pi*t/ln(2) = 0.018129 m on a 2 mm laminate (14.79x bias).

Zero bare asserts; runs across -I -B, -I -O -B, and -I -B -W error.
"""

from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verify


class TestImpedencePhysicalHardening(unittest.TestCase):
    def test_real_single_frequency_cole_jacobian_rank_and_singularity(self) -> None:
        J1 = verify.cole_jacobian(np.array([50000.0]), (410.0, 620.0, 50000.0, 0.80))
        self.assertEqual(J1.shape, (2, 4))
        self.assertEqual(int(np.linalg.matrix_rank(J1)), 2)
        # Must not be a constant ones matrix
        self.assertFalse(np.allclose(J1, np.ones((2, 4))))
        with self.assertRaises(verify.CheckFailure):
            verify.fisher_condition_number(np.array([50000.0]), (410.0, 620.0, 50000.0, 0.80))

        # Verify source no longer contains np.ones((2, 4))
        src = (ROOT / "verify.py").read_text(encoding="utf-8")
        self.assertNotIn("np.ones((2, 4))", src)
        self.assertNotIn("(620.0 + 300.0)", src)

    def test_explicit_c_stray_unaliasing_and_extended_fisher_contact_rank(self) -> None:
        freqs_50 = np.geomspace(5000.0, 1000000.0, 50)
        z_shunt = verify.cole_spectrum(freqs_50, 410.0, 620.0, 50000.0, 0.80, Td_ns=2.0, C_stray_pF=45.0)
        fit = verify.fit_cole_with_stray_capacitance(freqs_50, z_shunt)
        self.assertAlmostEqual(fit["C_stray_pF"], 45.0, delta=0.2)
        self.assertAlmostEqual(fit["R_inf_ohm"], 410.0, delta=0.2)
        self.assertAlmostEqual(fit["Td_ns"], 2.0, delta=0.1)
        self.assertTrue(fit["stray_capacitance_flagged"])

        # Extended Fisher matrix: 5-param (Cole + C_stray) full rank; adding series R_contact is rank-deficient
        ext = verify.fisher_extended_identifiability(freqs_50, (410.0, 620.0, 50000.0, 0.80))
        self.assertEqual(ext["cole_plus_c_stray_rank"], 5)
        self.assertEqual(ext["cole_plus_c_stray_and_r_contact_rank"], 5)
        self.assertTrue(ext["series_r_contact_structurally_unidentifiable"])

    def test_two_wire_contact_cpe_spectrum_corrupts_R0(self) -> None:
        freqs_7 = np.array([5000.0, 12000.0, 28000.0, 50000.0, 110000.0, 250000.0, 600000.0])
        z_2w = verify.cole_spectrum(
            freqs_7, 410.0, 620.0, 50000.0, 0.80, Td_ns=4.0, R_contact_ohm=300.0, Q_contact=5e-7, alpha_contact=0.80
        )
        fit_2w = verify.fit_cole_with_stray_capacitance(freqs_7, z_2w)
        rel_err = abs(fit_2w["R0_ohm"] - 620.0) / 620.0
        self.assertGreater(rel_err, 0.40)

    def test_finite_thickness_slab_kernel_limits_and_guards(self) -> None:
        a = 0.025
        k_half = verify.square_tetrapolar_cell_constant(a)
        # Thick limit t = 1.0 m (a/t = 0.025) matches half-space
        k_thick = verify.square_tetrapolar_slab_cell_constant(a, 1.0)
        self.assertAlmostEqual(k_thick, k_half, delta=1e-4)
        # Thin laminate limit t = 2 mm (a/t = 12.5) matches 2D logarithmic sheet 2*pi*t/ln(2)
        k_2mm = verify.square_tetrapolar_slab_cell_constant(a, 0.002)
        k_2d = (2.0 * math.pi * 0.002) / math.log(2.0)
        self.assertAlmostEqual(k_2mm, k_2d, delta=1e-6)
        self.assertGreater(k_half / k_2mm, 14.0)
        # Monotonic decrease with decreasing thickness
        k_10mm = verify.square_tetrapolar_slab_cell_constant(a, 0.010)
        self.assertLess(k_2mm, k_10mm)
        self.assertLess(k_10mm, k_half)
        for bad in (0.0, -0.005, float("nan"), float("inf"), True):
            with self.assertRaises(verify.CheckFailure):
                verify.square_tetrapolar_slab_cell_constant(a, bad)

    def test_zero_bare_asserts_and_zero_builtin_maxmin(self) -> None:
        for rel in ("verify.py", "tests/test_physical_hardening.py"):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)], [])
        v_tree = ast.parse((ROOT / "verify.py").read_text(encoding="utf-8"))
        maxmin = [
            n.lineno
            for n in ast.walk(v_tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("max", "min")
        ]
        self.assertEqual(maxmin, [])


if __name__ == "__main__":
    unittest.main()
