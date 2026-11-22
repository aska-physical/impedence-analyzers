#!/usr/bin/env python3
"""Assertion-free R5 test suite for studies/ndt-rank1-identifiability-audit/ndt_rank1_identifiability_audit.py.

Zero bare asserts (100% -O immune), pure Python 3 stdlib, deterministic across -I/-O/-B/-W error.
Exercises all 3 pillars (1-layer & 2-layer Kelvin patches, Cole 5-parameter in-span vs out-of-span
identifiability split, dock#8 receiving controls) and every fail-closed guard (100% mutation + BoolOp).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from contextlib import redirect_stderr, redirect_stdout
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ndt_rank1_identifiability_audit as mod


class _ListSub(list):
    pass


class TestNDTRank1Identifiability(unittest.TestCase):
    def test_01_primitives_and_laplacian_guards(self) -> None:
        # frac_str & format_exact_decimal exactness, rounding, boundary & fail-closed guards
        self.assertEqual(mod.frac_str(Fraction(3, 4)), "3/4")
        self.assertEqual(mod.format_exact_decimal(Fraction(0, 1), 3), "0.000")
        self.assertEqual(mod.format_exact_decimal(Fraction(-3, 2), 1), "-1.5")
        self.assertEqual(mod.format_exact_decimal(Fraction(-1, 10000), 2), "0.00")
        self.assertEqual(mod.format_exact_decimal(Fraction(5, 2), 0), "3")
        self.assertEqual(mod.format_exact_decimal(Fraction(1, 2), 18), "0.5" + ("0" * 17))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "exact Fraction"):
            mod.frac_str(1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strict int"):
            mod.format_exact_decimal(Fraction(1, 2), True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, ">= 0"):
            mod.format_exact_decimal(Fraction(1, 2), -1)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "<= 18"):
            mod.format_exact_decimal(Fraction(1, 2), 19)

        # Graph & Laplacian solver exactness + fail-closed guards
        tri = [(0, 1, Fraction(1, 1)), (1, 2, Fraction(1, 1)), (0, 2, Fraction(1, 1))]
        phi = mod.solve_grounded_laplacian_exact(3, tri, [Fraction(1, 1), Fraction(0, 1), Fraction(-1, 1)])
        self.assertEqual(phi[0] - phi[2], Fraction(2, 3))

        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, ">= 2"):
            mod.solve_grounded_laplacian_exact(1, tri, [Fraction(0, 1)])
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "<= 64"):
            mod.solve_grounded_laplacian_exact(65, tri, [Fraction(0, 1)] * 65)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "non-empty"):
            mod.solve_grounded_laplacian_exact(3, [], [Fraction(0, 1)] * 3)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "non-empty"):
            mod.solve_grounded_laplacian_exact(3, _ListSub(tri), [Fraction(0, 1)] * 3)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "3-tuple"):
            mod.solve_grounded_laplacian_exact(3, [(0, 1)], [Fraction(0, 1)] * 3)  # type: ignore[list-item]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "3-tuple"):
            mod.solve_grounded_laplacian_exact(3, [_ListSub([0, 1, Fraction(1, 1)])], [Fraction(0, 1)] * 3)  # type: ignore[list-item]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "self-loop"):
            mod.solve_grounded_laplacian_exact(3, [(0, 0, Fraction(1, 1))], [Fraction(0, 1)] * 3)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "> 0"):
            mod.solve_grounded_laplacian_exact(3, [(0, 1, Fraction(0, 1))], [Fraction(0, 1)] * 3)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "length n_nodes"):
            mod.solve_grounded_laplacian_exact(3, tri, [Fraction(0, 1)] * 2)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "length n_nodes"):
            mod.solve_grounded_laplacian_exact(3, tri, _ListSub([Fraction(0, 1)] * 3))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "sum to zero"):
            mod.solve_grounded_laplacian_exact(3, tri, [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)])
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "disconnected"):
            mod.solve_grounded_laplacian_exact(
                3, [(0, 1, Fraction(1, 1))], [Fraction(1, 1), Fraction(-1, 1), Fraction(0, 1)]
            )

        # Partial cut (rem_g > 0) AND full cut + guards on rank1_edge_cut_exact
        part_cut = mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), (0, 2), (0, 2))
        self.assertTrue(part_cut["sherman_morrison_matches_direct_exact"])
        self.assertGreater(Fraction(part_cut["delta_r_eff_sm"]), 0)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "exceed edge conductance"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(2, 1), (0, 2), (0, 2))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "drive_pair"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), (0,), (0, 2))  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "drive_pair"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), _ListSub([0, 2]), (0, 2))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "sense_pair"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), (0, 2), (0,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "sense_pair"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), (0, 2), _ListSub([0, 2]))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "distinct"):
            mod.rank1_edge_cut_exact(3, tri, 0, Fraction(1, 2), (0, 0), (0, 2))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "disconnects graph"):
            mod.rank1_edge_cut_exact(
                3, [(0, 1, Fraction(1, 1)), (1, 2, Fraction(1, 1))], 0, Fraction(1, 1), (0, 2), (0, 2)
            )

    def test_02_cole_span_and_obstruction_guards(self) -> None:
        r_inf, r0 = Fraction(240, 1), Fraction(1850, 1)
        z_d = mod.cole_debye_exact_q(r_inf, r0, Fraction(1, 1))
        self.assertEqual(z_d, (Fraction(1045, 1), Fraction(-805, 1)))
        z_h = mod.cole_half_exact_q(r_inf, r0, Fraction(1, 1))
        self.assertEqual(z_h, (Fraction(240, 1) + Fraction(1610 * 2, 5), Fraction(-1610, 5)))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strictly greater"):
            mod.cole_debye_exact_q(r0, r0, Fraction(1, 1))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strictly greater"):
            mod.cole_half_exact_q(r0, r0, Fraction(1, 1))

        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "zero complex"):
            mod._c_inv((Fraction(0, 1), Fraction(0, 1)))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "2-tuple"):
            mod.apply_parallel_shunt_exact_q((Fraction(1, 1),), Fraction(1, 100))  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "2-tuple"):
            mod.apply_parallel_shunt_exact_q(_ListSub([Fraction(1, 1), Fraction(0, 1)]), Fraction(1, 100))  # type: ignore[arg-type]

        us = [Fraction(1, 2), Fraction(1, 1), Fraction(2, 1)]
        zs = [mod.cole_debye_exact_q(r_inf, r0, u) for u in us]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "equal-length"):
            mod.evaluate_cole_debye_obstruction_exact_q(us[:2], zs[:2], r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "equal-length"):
            mod.evaluate_cole_debye_obstruction_exact_q(us, zs[:2], r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "equal-length"):
            mod.evaluate_cole_debye_obstruction_exact_q(_ListSub(us), zs, r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "equal-length"):
            mod.evaluate_cole_debye_obstruction_exact_q(us, _ListSub(zs), r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "2-tuple"):
            mod.evaluate_cole_debye_obstruction_exact_q(
                us, [zs[0], zs[1], (Fraction(1, 1),)], r_inf, r0, Fraction(1, 20), Fraction(1, 10000)  # type: ignore[list-item]
            )
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "2-tuple"):
            mod.evaluate_cole_debye_obstruction_exact_q(
                us, [zs[0], zs[1], _ListSub(list(zs[2]))], r_inf, r0, Fraction(1, 20), Fraction(1, 10000)  # type: ignore[list-item]
            )
        # Degenerate constant q_k basis
        const_zs = [(Fraction(500, 1), -u) for u in us]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "degenerate"):
            mod.evaluate_cole_debye_obstruction_exact_q(us, const_zs, r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        # Non-positive slope inv_gamma_hat <= 0 (inverted real part vs q_k -> flagged!)
        inv_zs = [(Fraction(100, 1) - (Fraction(100, 1) / u), -Fraction(10, 1) * u * u) for u in us]
        ev_neg = mod.evaluate_cole_debye_obstruction_exact_q(us, inv_zs, r_inf, r0, Fraction(1, 20), Fraction(1, 10000))
        self.assertTrue(ev_neg["residual_gate_flagged"])
        self.assertFalse(ev_neg["gate_passed"])

        # R_0-only shift (R_inf unchanged, R0 shifted > 5% -> kills baseline_shift_flagged Or mutant!)
        zs_r0_only = [mod.cole_debye_exact_q(r_inf, Fraction(2100, 1), u) for u in us]
        ev_r0 = mod.evaluate_cole_debye_obstruction_exact_q(us, zs_r0_only, r_inf, r0, Fraction(5, 100), Fraction(1, 10000))
        self.assertEqual(ev_r0["rel_shift_r_inf_pct"], "0.0000")
        self.assertTrue(ev_r0["baseline_shift_flagged"])
        self.assertFalse(ev_r0["residual_gate_flagged"])

        # R_inf-only shift (R0 unchanged, R_inf shifted > 5% -> kills other side of Or mutant!)
        zs_rinf_only = [mod.cole_debye_exact_q(Fraction(300, 1), r0, u) for u in us]
        ev_ri = mod.evaluate_cole_debye_obstruction_exact_q(us, zs_rinf_only, r_inf, r0, Fraction(5, 100), Fraction(1, 10000))
        self.assertEqual(ev_ri["rel_shift_r0_pct"], "0.0000")
        self.assertTrue(ev_ri["baseline_shift_flagged"])
        self.assertFalse(ev_ri["residual_gate_flagged"])

        # Tiny shunt (nonzero triples > 0, but residual_sq <= huge limit -> kills And mutant!)
        z_tiny_shunt = [mod.apply_parallel_shunt_exact_q(z, Fraction(1, 1000000) * u) for z, u in zip(zs, us)]
        ev_tiny = mod.evaluate_cole_debye_obstruction_exact_q(us, z_tiny_shunt, r_inf, r0, Fraction(5, 100), Fraction(1, 10))
        self.assertGreater(ev_tiny["collinearity_obstruction_triples"], 0)
        self.assertFalse(ev_tiny["residual_gate_flagged"])

    def test_03_full_certificate_zone_b_and_cli(self) -> None:
        cert = mod.build_signed_certificate()
        self.assertTrue(cert["overall_verified"])
        self.assertTrue(cert["pillar1_laplacian_rank1_kelvin_patch"]["pillar_verified"])
        self.assertTrue(cert["pillar2_cole_span_identifiability_split"]["pillar_verified"])
        self.assertTrue(cert["pillar3_dock8_receiving_controls"]["pillar_verified"])

        # Exact payload validation
        verdict = mod.validate_certificate_payload(cert)
        self.assertTrue(verdict["valid"])
        self.assertEqual(verdict["payload_sha256"], cert["payload_sha256"])

        # Strict fail-closed Zone-B rejections (exact message regexes -> 100% mutation kill)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strict dict"):
            mod.canonical_json_bytes("bad")  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "cannot serialize"):
            mod.canonical_json_bytes({"huge": 10 ** 5000})
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strict dict"):
            mod.compute_payload_sha256("bad")  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "JSON object"):
            mod.validate_certificate_payload(["bad"])
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "must be Path"):
            mod.validate_certificate_file("not_a_path")  # type: ignore[arg-type]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "strict str"):
            mod.validate_certificate_payload({1: "bad_key"})
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "forbidden JSON value"):
            mod.validate_certificate_payload({"f": 1.5})
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "forbidden JSON value"):
            mod.validate_certificate_payload({"t": [1, (2,)]})

        # Exact depth boundary: depth 64 allowed, depth 65 rejected
        d64: Any = 1
        for _ in range(64):
            d64 = [d64]
        mod._assert_safe_json_tree(d64)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "nesting depth"):
            mod.validate_certificate_payload({"deep": [d64]})

        for field, bad_val, rx in [
            ("schema_version", "wrong", "unexpected schema_version"),
            ("study_id", "WRONG", "unexpected study_id"),
            ("mode", "wrong", "unexpected mode"),
            ("claim_status", "bench_measured", "unexpected claim_status"),
            ("overall_verified", False, "overall_verified must be True"),
            ("overall_verified", 1, "invalid type"),
            ("payload_sha256", "0" * 63, "64 lowercase hex ASCII"),
            ("payload_sha256", "0" * 65, "64 lowercase hex ASCII"),
            ("payload_sha256", "\u00e9" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "A" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "g" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "0" * 64, "payload_sha256 mismatch"),
        ]:
            tampered = copy.deepcopy(cert)
            tampered[field] = bad_val
            with self.assertRaisesRegex(mod.NDTIdentifiabilityError, rx):
                mod.validate_certificate_payload(tampered)

        missing = copy.deepcopy(cert)
        del missing["pillar1_laplacian_rank1_kelvin_patch"]
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "missing required key"):
            mod.validate_certificate_payload(missing)

        forged = copy.deepcopy(cert)
        forged["pillar1_laplacian_rank1_kelvin_patch"]["mesh_1layer_nodes"] = 99
        forged["payload_sha256"] = mod.compute_payload_sha256(forged)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "diverges from pure-Z/Q oracle"):
            mod.validate_certificate_payload(forged)

        forged_bool = copy.deepcopy(cert)
        forged_bool["pillar1_laplacian_rank1_kelvin_patch"]["pillar_verified"] = 1
        forged_bool["payload_sha256"] = mod.compute_payload_sha256(forged_bool)
        with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "diverges from pure-Z/Q oracle"):
            mod.validate_certificate_payload(forged_bool)

        # CLI standalone (--json & text) + --check (valid, missing, bad UTF-8, bad JSON, floats, NaN, dup keys, 5000-digit int, deep)
        out_j = io.StringIO()
        with redirect_stdout(out_j):
            self.assertEqual(mod.main(["--standalone", "--json"]), 0)
        self.assertEqual(json.loads(out_j.getvalue()), cert)

        out_t = io.StringIO()
        with redirect_stdout(out_t):
            self.assertEqual(mod.main(["--standalone"]), 0)
        self.assertIn("NDT-RANK1-P5 OK", out_t.getvalue())

        with tempfile.TemporaryDirectory(prefix="ndt_r5_") as td:
            os.chmod(td, 0o700)
            p_ok = Path(td) / "ok.json"
            p_ok.write_text(out_j.getvalue(), encoding="utf-8")
            buf_ok = io.StringIO()
            with redirect_stdout(buf_ok):
                self.assertEqual(mod.main(["--check", str(p_ok)]), 0)
            self.assertTrue(json.loads(buf_ok.getvalue())["valid"])

            err_m = io.StringIO()
            with redirect_stderr(err_m):
                self.assertEqual(mod.main(["--check", str(Path(td) / "missing.json")]), 1)
            self.assertIn("CHECK FAILED", err_m.getvalue())

            p_bin = Path(td) / "bad_utf8.json"
            p_bin.write_bytes(b"\xff\xfe\xfd")
            with self.assertRaisesRegex(mod.NDTIdentifiabilityError, "cannot read"):
                mod.validate_certificate_file(p_bin)

            p_badj = Path(td) / "bad.json"
            bad_non_ascii_cert = copy.deepcopy(cert)
            bad_non_ascii_cert["payload_sha256"] = "\u00e9" * 64
            p_non_ascii = Path(td) / "non_ascii_sha.json"
            p_non_ascii.write_text(json.dumps(bad_non_ascii_cert, ensure_ascii=False), encoding="utf-8")
            err_na = io.StringIO()
            with redirect_stderr(err_na):
                self.assertEqual(mod.main(["--check", str(p_non_ascii)]), 1)
            self.assertIn("64 lowercase hex ASCII", err_na.getvalue())

            for bad_txt, rx in (
                ("{broken", "invalid JSON"),
                ('{"x": 1.5}', "forbidden JSON float"),
                ('{"x": NaN}', "forbidden JSON float"),
                ('{"a": 1, "a": 2}', "duplicate JSON key"),
                ('{"big": ' + ("9" * 5000) + "}", "invalid JSON"),
                ('{"d": ' + ("[" * 100) + "1" + ("]" * 100) + "}", "nesting depth"),
            ):
                p_badj.write_text(bad_txt, encoding="utf-8")
                with self.assertRaisesRegex(mod.NDTIdentifiabilityError, rx):
                    mod.validate_certificate_file(p_badj)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNDTRank1Identifiability)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
