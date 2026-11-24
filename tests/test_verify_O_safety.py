#!/usr/bin/env python3
"""Regression tests proving verify.py fails closed under python3 -O and checks Control 1 on 45 pF shunt."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verify


def _test_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _copy_repo(dst: Path) -> Path:
    work = dst / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    return work


class VerifyOptimizedModeTests(unittest.TestCase):
    def test_clean_tree_passes_all_isolated_modes(self) -> None:
        env = _test_env()
        for flags in (["-I", "-B"], ["-I", "-O", "-B"], ["-I", "-B", "-W", "error"], ["-I", "-O", "-B", "-W", "error"]):
            proc = subprocess.run(
                [sys.executable, *flags, "verify.py"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"flags={flags} stderr={proc.stderr}")
            self.assertIn("[PASS]", proc.stdout)
            self.assertIn("(5/5 controls): OK", proc.stdout)

    def test_clean_write_is_idempotent_and_drift_is_rejected(self) -> None:
        env = _test_env()
        with tempfile.TemporaryDirectory() as td:
            work = _copy_repo(Path(td))
            cert_json = work / "results" / "certificate.json"
            original_cert_bytes = cert_json.read_bytes()

            proc_w = subprocess.run(
                [sys.executable, "-I", "-O", "-B", "-W", "error", "verify.py", "--write"],
                cwd=work,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc_w.returncode, 0, msg=proc_w.stderr)
            self.assertEqual(cert_json.read_bytes(), original_cert_bytes)

            tampered = json.loads(original_cert_bytes.decode("utf-8"))
            tampered["square_patch_25mm_cell_constant_m"] = 0.999
            cert_json.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            proc_d = subprocess.run(
                [sys.executable, "-I", "-O", "-B", "verify.py"],
                cwd=work,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc_d.returncode, 1)
            self.assertIn("mismatch", proc_d.stderr)

    def test_hook_mismatch_and_45pf_shunt_tampering_fail_closed_under_standard_and_O_write(self) -> None:
        """Tampering hook_lead_ns=0.0 or C_stray_pF=0.0 must fail closed and leave certificate.json untouched."""
        env = _test_env()
        cases = [
            ("hook_lead_ns_zero", r"hook_lead_ns\s*=\s*12\.0", "hook_lead_ns=0.0"),
            ("shunt_45pF_zero", r"C_stray_pF\s*=\s*45\.0", "C_stray_pF=0.0"),
        ]
        for label, pattern, repl in cases:
            for mode in ([], ["-O"]):
                with self.subTest(case=label, mode=mode), tempfile.TemporaryDirectory() as td:
                    work = _copy_repo(Path(td))
                    verify_py = work / "verify.py"
                    cert_json = work / "results" / "certificate.json"
                    original_cert_bytes = cert_json.read_bytes()

                    src = verify_py.read_text(encoding="utf-8")
                    patched, n_sub = re.subn(pattern, repl, src, count=1)
                    self.assertEqual(n_sub, 1, msg=f"pattern {pattern!r} not found")
                    verify_py.write_text(patched, encoding="utf-8")

                    for extra_args in ([], ["--write"], []):
                        proc = subprocess.run(
                            [sys.executable, "-I", *mode, "-B", "verify.py", *extra_args],
                            cwd=work,
                            env=env,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        self.assertNotEqual(
                            proc.returncode,
                            0,
                            msg=f"case={label} mode={mode} args={extra_args} unexpectedly passed: {proc.stdout}",
                        )
                        self.assertIn("CheckFailure", proc.stderr)

                    self.assertEqual(cert_json.read_bytes(), original_cert_bytes)

    def test_control1_itself_fails_when_45pf_shunt_not_flagged(self) -> None:
        """Control 1 must reject when fit_clin_shunt is unflagged even if fit_clin_corrupt remains flagged."""
        env = _test_env()
        with tempfile.TemporaryDirectory() as td:
            work = _copy_repo(Path(td))
            verify_py = work / "verify.py"
            cert_json = work / "results" / "certificate.json"
            original_cert_bytes = cert_json.read_bytes()

            src = verify_py.read_text(encoding="utf-8")
            patched, n_sub = re.subn(
                r"(k_sq_25mm\s*=\s*square_tetrapolar_cell_constant\(0\.025\)\n)",
                r'\1    fit_clin_shunt["parasitic_artifact_flagged"] = False\n',
                src,
                count=1,
            )
            self.assertEqual(n_sub, 1)
            verify_py.write_text(patched, encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-I", "-O", "-B", "verify.py", "--write"],
                cwd=work,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("CheckFailure", proc.stderr)
            self.assertIn("unguarded_45pF_stray_capacitance_passes_without_negative_Td_flag", proc.stderr)
            self.assertEqual(cert_json.read_bytes(), original_cert_bytes)

    def test_all_guards_and_strict_boundaries_fail_closed_in_process(self) -> None:
        """Exercise every require() guard and exact strict-inequality boundary directly in-process."""
        with self.assertRaises(verify.CheckFailure):
            verify.require(False, "direct require failure")

        # 1. square_tetrapolar_cell_constant guard (exact 1e-14 boundary and >1e-14 mismatch)
        orig_sqrt = verify.math.sqrt
        calls = {"n": 0}

        def _skew_sqrt(x):
            calls["n"] += 1
            if calls["n"] == 3:
                return orig_sqrt(x) + 1e-6
            return orig_sqrt(x)

        with mock.patch.object(verify.math, "sqrt", side_effect=_skew_sqrt):
            with self.assertRaises(verify.CheckFailure):
                verify.square_tetrapolar_cell_constant(0.025)

        # Build clean fits once so every guard test runs in milliseconds without re-fitting LM
        orig_fit = verify.fit_cole_with_artifact_gate
        orig_cond = verify.fisher_condition_number
        clean_fits: list[dict] = []

        def _record_fit(freqs, z):
            res = orig_fit(freqs, z)
            clean_fits.append(dict(res))
            return res

        with mock.patch.object(verify, "fit_cole_with_artifact_gate", side_effect=_record_fit):
            verify.run_verification()
        self.assertEqual(len(clean_fits), 4)

        def _run_with_fit_override(idx: int, updates: dict) -> None:
            seq = [dict(d) for d in clean_fits]
            seq[idx].update(updates)
            it = iter(seq)
            with mock.patch.object(verify, "fit_cole_with_artifact_gate", side_effect=lambda *a, **k: next(it)):
                with mock.patch.object(verify, "fisher_condition_number", side_effect=[14.34, 13.851]):
                    verify.run_verification()

        # 2. clin guarded flagged -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(0, {"parasitic_artifact_flagged": True})
        # 3. clin guarded R_inf error >= 1e-4 -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(0, {"R_inf_ohm": 410.0 + 410.0 * 1.0001e-4})
        # 4. hook corrupt unflagged / Td_ns >= -1.5 -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(1, {"parasitic_artifact_flagged": False, "Td_ns": 2.0})
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(1, {"Td_ns": -1.5})
        # 5. 45 pF shunt unflagged / Td_ns <= 10.0 -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(2, {"parasitic_artifact_flagged": False, "Td_ns": 2.0})
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(2, {"Td_ns": 10.0})
        # 6. NDT guarded flagged -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(3, {"parasitic_artifact_flagged": True})
        # 7. NDT guarded R0 error >= 1e-4 -> fails
        with self.assertRaises(verify.CheckFailure):
            _run_with_fit_override(3, {"R0_ohm": 1850.0 + 1850.0 * 1.0001e-4})

        # 8. Fisher conditioning ratio exact boundary (cond_7 == 1.25 * cond_50) -> fails strict '<'
        seq_ok = [dict(d) for d in clean_fits]
        it_ok = iter(seq_ok)
        with mock.patch.object(verify, "fit_cole_with_artifact_gate", side_effect=lambda *a, **k: next(it_ok)):
            with mock.patch.object(verify, "fisher_condition_number", side_effect=[10.0, 12.5]):
                with self.assertRaises(verify.CheckFailure) as ctx:
                    verify.run_verification()
                self.assertIn("Fisher conditioning", str(ctx.exception))

        # 9. Must-fail controls guard: cell constant equal to linear Wenner -> fails Control 5
        seq_ok2 = [dict(d) for d in clean_fits]
        it_ok2 = iter(seq_ok2)
        with mock.patch.object(verify, "fit_cole_with_artifact_gate", side_effect=lambda *a, **k: next(it_ok2)):
            with mock.patch.object(verify, "fisher_condition_number", side_effect=[14.34, 13.851]):
                with mock.patch.object(verify, "square_tetrapolar_cell_constant", return_value=2.0 * verify.math.pi * 0.025):
                    with self.assertRaises(verify.CheckFailure) as ctx:
                        verify.run_verification()
                    self.assertIn("square_patch_cell_constant_equals_linear_wenner_2pi_a", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
