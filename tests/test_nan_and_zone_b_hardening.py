#!/usr/bin/env python3
"""Zero-bare-assert R5 regression suite for NaN/Inf fail-closed guards and Zone-B canonical cert I/O.

Covers:
  - NaN/Inf/bool/zero-impedance fail-closed guards in cole_spectrum, fit_cole_with_artifact_gate
    (closing the Track S21c TypeError crash on NaN samples), fisher_condition_number, and
    square_tetrapolar_cell_constant
  - All finite_max / finite_clamp / require_finite boundary and error checks
  - Complete Zone-B canonical certificate verification & write hardening (ZB03 5000-digit int,
    ZB06 bool-as-int, ZB07 duplicate keys, ZB08 int-as-float, ZB10 CRLF/pretty drift,
    ZB14 symlink target / ancestor / relative-PWD / Path-subclass / NUL, ZB15 dir & FIFO
    read/write without blocking, ZB16 invalid UTF-8 & truncated JSON, oversized files)
  - Live subprocess checks under -I -O -B proving exit code 1 with zero raw tracebacks
"""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verify


class _PathSub(type(Path("."))):
    pass


def _test_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


class TestImpedenceNanAndZoneBHardening(unittest.TestCase):
    def test_zero_bare_asserts_and_zero_builtin_maxmin_in_verify(self) -> None:
        tree = ast.parse((ROOT / "verify.py").read_text(encoding="utf-8"))
        asserts = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        self.assertEqual(asserts, [])
        maxmin = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("max", "min")
        ]
        self.assertEqual(maxmin, [])
        test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        self.assertEqual([n.lineno for n in ast.walk(test_tree) if isinstance(n, ast.Assert)], [])

    def test_finite_primitives_and_boundaries(self) -> None:
        self.assertEqual(verify.require_finite(3, "x"), 3.0)
        self.assertEqual(verify.require_finite(np.float64(2.5), "x"), 2.5)
        for bad in (True, False, "1.0", None, float("nan"), float("inf"), float("-inf"), 10**400):
            with self.assertRaises(verify.CheckFailure):
                verify.require_finite(bad, "x")

        self.assertEqual(verify.finite_max(1.0, 2.0), 2.0)
        self.assertEqual(verify.finite_max(3.0, 2.0), 3.0)
        with self.assertRaises(verify.CheckFailure):
            verify.finite_max(1.0, float("nan"))
        with self.assertRaises(verify.CheckFailure):
            verify.finite_max(float("nan"), 1.0)

        self.assertEqual(verify.finite_clamp(0.5, 0.2, 0.8), 0.5)
        self.assertEqual(verify.finite_clamp(0.1, 0.2, 0.8), 0.2)
        self.assertEqual(verify.finite_clamp(0.9, 0.2, 0.8), 0.8)
        self.assertEqual(verify.finite_clamp(0.5, 0.5, 0.5), 0.5)
        with self.assertRaises(verify.CheckFailure):
            verify.finite_clamp(0.5, 0.8, 0.2)
        with self.assertRaises(verify.CheckFailure):
            verify.finite_clamp(float("nan"), 0.2, 0.8)

    def test_nan_inf_and_invalid_physics_inputs_fail_closed_with_check_failure(self) -> None:
        freqs = np.array([5000.0, 12000.0, 28000.0, 50000.0, 110000.0, 250000.0, 600000.0])
        z_ok = verify.cole_spectrum(freqs, 410.0, 620.0, 50000.0, 0.80, Td_ns=4.0, C_stray_pF=0.0)

        # Headline Track S21c repro: NaN sample previously crashed with TypeError on NoneType!
        z_nan_re = z_ok.copy()
        z_nan_re[2] = complex(float("nan"), -50.0)
        with self.assertRaises(verify.CheckFailure) as ctx_nr:
            verify.fit_cole_with_artifact_gate(freqs, z_nan_re)
        self.assertIn("finite complex impedances", str(ctx_nr.exception))

        z_nan_im = z_ok.copy()
        z_nan_im[2] = complex(450.0, float("nan"))
        with self.assertRaises(verify.CheckFailure) as ctx_ni:
            verify.fit_cole_with_artifact_gate(freqs, z_nan_im)
        self.assertIn("finite complex impedances", str(ctx_ni.exception))

        z_inf = z_ok.copy()
        z_inf[0] = complex(float("inf"), 0.0)
        with self.assertRaises(verify.CheckFailure) as ctx_inf:
            verify.fit_cole_with_artifact_gate(freqs, z_inf)
        self.assertIn("finite complex impedances", str(ctx_inf.exception))

        z_zero = z_ok.copy()
        z_zero[0] = 0.0 + 0.0j
        with self.assertRaises(verify.CheckFailure) as ctx_z:
            verify.fit_cole_with_artifact_gate(freqs, z_zero)
        self.assertIn("non-zero impedances", str(ctx_z.exception))

        # Array shape / empty / bad type / bool / str / OverflowError / non-positive freq guards (B1..B2)
        for bad_f in (
            np.array([]),
            np.array([[5000.0, 10000.0]]),
            np.array([5000.0, float("nan")]),
            np.array([5000.0, 0.0]),
            np.array([5000.0, -10.0]),
            np.array([True, True]),
            np.array(["5000.0"]),
            np.array([True, 5000.0], dtype=object),
            [True, 5000.0],
            ["5000.0"],
            ["not-a-float"],
            [10**400],
            "5000.0",
            b"5000",
            bytearray(b"5000"),
            12345,
        ):
            with self.assertRaises(verify.CheckFailure):
                verify.cole_spectrum(bad_f, 410.0, 620.0, 50000.0, 0.80)

        # Valid Python list of floats works cleanly via _require_1d_numeric_array
        self.assertEqual(len(verify.cole_spectrum([5000.0, 10000.0], 410.0, 620.0, 50000.0, 0.80)), 2)

        for bad_z in (
            z_ok[:-1],
            z_ok.reshape(1, -1),
            ["bad"] * len(freqs),
            ["410.0"] * len(freqs),
            [True] * len(freqs),
            np.array([True] * len(freqs)),
            [10**400] * len(freqs),
        ):
            with self.assertRaises(verify.CheckFailure):
                verify.fit_cole_with_artifact_gate(freqs, bad_z)

        # cole_spectrum strict bounds: fc > 0, 0 < alpha <= 1, C_stray >= 0, and finite output (B4)
        with self.assertRaises(verify.CheckFailure):
            verify.cole_spectrum(freqs, 410.0, 620.0, 0.0, 0.80)
        with self.assertRaises(verify.CheckFailure):
            verify.cole_spectrum(freqs, 410.0, 620.0, -1.0, 0.80)
        for bad_alpha in (0.0, -0.2, 1.01, 1e308):
            with self.assertRaises(verify.CheckFailure):
                verify.cole_spectrum(freqs, 410.0, 620.0, 50000.0, bad_alpha)
        # Exact alpha == 1.0 and C_stray == 0.0 boundaries pass cleanly
        self.assertEqual(len(verify.cole_spectrum(freqs, 410.0, 620.0, 50000.0, 1.0, C_stray_pF=0.0)), len(freqs))
        with self.assertRaises(verify.CheckFailure):
            verify.cole_spectrum(freqs, 410.0, 620.0, 50000.0, 0.80, C_stray_pF=-0.1)
        # Extreme overflow -> non-finite impedance fails closed cleanly under -W error
        with self.assertRaises(verify.CheckFailure):
            verify.cole_spectrum(freqs, 410.0, 620.0, 1e-308, 0.80)
        with self.assertRaises(verify.CheckFailure):
            verify.cole_spectrum(freqs, -1e308, 1e308, 50000.0, 0.80)

        # Best candidate None guard when spectrum yields non-finite values
        with mock.patch.object(verify, "cole_spectrum", return_value=np.full_like(z_ok, complex(float("nan"), 0.0))):
            with self.assertRaises(verify.CheckFailure):
                verify.fit_cole_with_artifact_gate(freqs, z_ok)

        # Fisher condition number params length/type, positive R_inf/R0, singular & NaN guards (B3)
        for bad_params in ((410.0, 620.0), (410.0, 620.0, 50000.0, 0.80, 1.0), "1234", (0.0, 620.0, 50000.0, 0.80), (410.0, 0.0, 50000.0, 0.80), (float("nan"), 620.0, 50000.0, 0.80)):
            with self.assertRaises(verify.CheckFailure):
                verify.fisher_condition_number(freqs, bad_params)
        with mock.patch.object(verify.np.linalg, "svd", return_value=np.array([10.0, 5.0, 2.0, 0.0])):
            with self.assertRaises(verify.CheckFailure) as ctx_sing:
                verify.fisher_condition_number(freqs, (410.0, 620.0, 50000.0, 0.80))
        self.assertIn("Fisher Jacobian is singular", str(ctx_sing.exception))

        # square_tetrapolar_cell_constant exact > 0 & finite guards
        for bad_a in (0.0, -0.025, float("nan"), float("inf"), True):
            with self.assertRaises(verify.CheckFailure):
                verify.square_tetrapolar_cell_constant(bad_a)

        # artifact_flagged all three OR branches tested independently: Td < -1.5, Td > 10.0, rel_rms > 0.01, and clean
        z_noisy = z_ok.copy()
        z_noisy[3] *= 1.15
        fit_noisy = verify.fit_cole_with_artifact_gate(freqs, z_noisy)
        self.assertTrue(fit_noisy["parasitic_artifact_flagged"])
        self.assertGreater(fit_noisy["rel_rms_residual_pct"], 1.0)
        self.assertGreaterEqual(fit_noisy["Td_ns"], -1.5)
        self.assertLessEqual(fit_noisy["Td_ns"], 10.0)

        fit_clean = verify.fit_cole_with_artifact_gate(freqs, z_ok)
        self.assertFalse(fit_clean["parasitic_artifact_flagged"])

    def test_zone_b_canonical_cert_verification_and_write_guards(self) -> None:
        cert_bytes = (ROOT / "results" / "certificate.json").read_bytes()
        cert_obj = json.loads(cert_bytes.decode("utf-8"))
        self.assertEqual(verify.render_certificate_bytes(cert_obj), cert_bytes)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good = root / "certificate.json"
            good.write_bytes(cert_bytes)

            # Exact valid file & exact max_bytes boundary passes; max_bytes - 1 fails
            ok, reason = verify.verify_canonical_cert_bytes(good, cert_bytes, max_bytes=len(cert_bytes))
            self.assertTrue(ok, reason)
            ok_over, reason_over = verify.verify_canonical_cert_bytes(good, cert_bytes, max_bytes=len(cert_bytes) - 1)
            self.assertFalse(ok_over)
            self.assertIn("exceeds", reason_over)

            # ZB06 bool-as-int (false -> 0, true -> 1)
            zb06 = root / "zb06.json"
            zb06.write_bytes(cert_bytes.replace(b"false", b"0", 1))
            self.assertFalse(verify.verify_canonical_cert_bytes(zb06, cert_bytes)[0])

            # ZB07 duplicate JSON key
            zb07 = root / "zb07.json"
            zb07.write_bytes(cert_bytes.replace(b'"certificate_version": "1.0.0",', b'"certificate_version": "0.0.0",\n  "certificate_version": "1.0.0",', 1))
            self.assertFalse(verify.verify_canonical_cert_bytes(zb07, cert_bytes)[0])

            # ZB08 int-as-float / formatting drift
            zb08 = root / "zb08.json"
            zb08.write_bytes(cert_bytes.replace(b"410.0", b"410.00", 1))
            self.assertFalse(verify.verify_canonical_cert_bytes(zb08, cert_bytes)[0])

            # ZB10 CRLF line endings
            zb10 = root / "zb10.json"
            zb10.write_bytes(cert_bytes.replace(b"\n", b"\r\n"))
            ok_crlf, reason_crlf = verify.verify_canonical_cert_bytes(zb10, cert_bytes)
            self.assertFalse(ok_crlf)
            self.assertIn("CRLF", reason_crlf)

            # ZB03 5000-digit integer / ZB16 invalid UTF-8 & truncated JSON
            zb03 = root / "zb03.json"
            zb03.write_text('{"x": ' + "9" * 5000 + "}\n", encoding="utf-8")
            self.assertFalse(verify.verify_canonical_cert_bytes(zb03, cert_bytes)[0])

            zb16_utf8 = root / "zb16_utf8.json"
            zb16_utf8.write_bytes(b"\xff\xfe\x00\n")
            self.assertFalse(verify.verify_canonical_cert_bytes(zb16_utf8, cert_bytes)[0])

            zb16_trunc = root / "zb16_trunc.json"
            zb16_trunc.write_bytes(b'{"a": ')
            self.assertFalse(verify.verify_canonical_cert_bytes(zb16_trunc, cert_bytes)[0])

            # ZB14 symlink target, symlink ancestor, relative PWD symlink, Path subclass, NUL byte
            sym = root / "sym.json"
            sym.symlink_to(good)
            ok_sym, r_sym = verify.verify_canonical_cert_bytes(sym, cert_bytes)
            self.assertFalse(ok_sym)
            self.assertIn("symlink", r_sym)
            self.assertFalse(verify.write_canonical_cert_bytes(sym, cert_bytes)[0])

            real_dir = root / "real_dir"
            real_dir.mkdir()
            (real_dir / "c.json").write_bytes(cert_bytes)
            sym_dir = root / "sym_dir"
            sym_dir.symlink_to(real_dir)
            self.assertFalse(verify.verify_canonical_cert_bytes(sym_dir / "c.json", cert_bytes)[0])
            self.assertFalse(verify.write_canonical_cert_bytes(sym_dir / "c.json", cert_bytes)[0])

            # Relative path under symlinked PWD
            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(real_dir)
                os.environ["PWD"] = str(sym_dir)
                self.assertTrue(verify._has_symlink_component(Path("c.json")))
                os.environ["PWD"] = str(real_dir)
                self.assertFalse(verify._has_symlink_component(Path("c.json")))
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd

            self.assertFalse(verify.verify_canonical_cert_bytes(_PathSub(good), cert_bytes)[0])
            self.assertFalse(verify.verify_canonical_cert_bytes(Path("a\x00b"), cert_bytes)[0])

            # ZB15 directory & FIFO (non-blocking fail closed on BOTH read and write!)
            d_target = root / "dir_cert"
            d_target.mkdir()
            ok_dir, r_dir = verify.verify_canonical_cert_bytes(d_target, cert_bytes)
            self.assertFalse(ok_dir)
            self.assertIn("regular file", r_dir)
            self.assertFalse(verify.write_canonical_cert_bytes(d_target, cert_bytes)[0])

            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            ok_fifo_r, r_fifo_r = verify.verify_canonical_cert_bytes(fifo, cert_bytes)
            self.assertFalse(ok_fifo_r)
            self.assertIn("regular file", r_fifo_r)
            # Open a reader fd so O_WRONLY|O_NONBLOCK succeeds and exercises the post-open S_ISREG guard!
            rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            try:
                ok_fifo_w, r_fifo_w = verify.write_canonical_cert_bytes(fifo, cert_bytes)
                self.assertFalse(ok_fifo_w)
                self.assertIn("non-regular target", r_fifo_w)
            finally:
                os.close(rfd)

            # Missing file & write to clean new file
            self.assertFalse(verify.verify_canonical_cert_bytes(root / "missing.json", cert_bytes)[0])
            fresh = root / "sub" / "fresh.json"
            self.assertTrue(verify.write_canonical_cert_bytes(fresh, cert_bytes)[0])
            self.assertTrue(verify.verify_canonical_cert_bytes(fresh, cert_bytes)[0])

            # main() --write refused on symlink returns exit code 1 with [FAIL]
            err_buf = io.StringIO()
            with (
                mock.patch.object(verify, "CERT_PATH", sym),
                mock.patch.object(verify, "run_verification", return_value=cert_obj),
                mock.patch.object(sys, "argv", ["verify.py", "--write"]),
                redirect_stderr(err_buf),
            ):
                rc = verify.main()
            self.assertEqual(rc, 1)
            self.assertIn("write refused", err_buf.getvalue())

    def test_live_subprocess_zone_b_matrix_exits_1_without_tracebacks(self) -> None:
        env = _test_env()
        orig_bytes = (ROOT / "results" / "certificate.json").read_bytes()
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "repo"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            cert_path = work / "results" / "certificate.json"

            # 1. Symlink cert (ZB14)
            outside = Path(td) / "outside.json"
            outside.write_bytes(orig_bytes)
            cert_path.unlink()
            cert_path.symlink_to(outside)
            p_sym = subprocess.run([sys.executable, "-I", "-O", "-B", "verify.py"], cwd=work, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(p_sym.returncode, 1)
            self.assertNotIn("Traceback", p_sym.stderr)

            # 2. Directory at cert path (ZB15)
            cert_path.unlink()
            cert_path.mkdir()
            p_dir = subprocess.run([sys.executable, "-I", "-O", "-B", "verify.py"], cwd=work, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(p_dir.returncode, 1)
            self.assertNotIn("Traceback", p_dir.stderr)
            cert_path.rmdir()

            # 3. Raw payloads: ZB03, ZB06, ZB07, ZB10, ZB16_utf8, ZB16_trunc
            payloads = {
                "ZB03_5000digit": ('{"x":' + "9" * 5000 + "}\n").encode("utf-8"),
                "ZB06_bool_int": orig_bytes.replace(b"false", b"0", 1),
                "ZB07_dup_key": orig_bytes.replace(b'"certificate_version": "1.0.0",', b'"certificate_version": "0.0.0",\n  "certificate_version": "1.0.0",', 1),
                "ZB10_crlf": orig_bytes.replace(b"\n", b"\r\n"),
                "ZB16_utf8": b"\xff\xfe\x00\n",
                "ZB16_trunc": b'{"broken": ',
            }
            for name, raw in payloads.items():
                cert_path.write_bytes(raw)
                p = subprocess.run([sys.executable, "-I", "-O", "-B", "verify.py"], cwd=work, env=env, text=True, capture_output=True, check=False)
                self.assertEqual(p.returncode, 1, msg=f"{name} unexpectedly exited {p.returncode}")
                self.assertIn("[FAIL]", p.stderr, msg=f"{name} missing [FAIL]")
                self.assertNotIn("Traceback", p.stderr, msg=f"{name} leaked traceback: {p.stderr}")


if __name__ == "__main__":
    unittest.main()
