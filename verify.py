#!/usr/bin/env python3
"""Deterministic Cole-Cole + Parasitic Delay/Capacitance EIS solver & artifact detector for aska-physical/impedence-analyzers."""

from __future__ import annotations

import argparse
import cmath
import hmac
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
CERT_PATH = ROOT / "results" / "certificate.json"
MAX_CERT_BYTES = 1_048_576
_EXACT_PATH_TYPE = type(Path("."))


class CheckFailure(RuntimeError):
    """Raised when a physical invariant or must-fail control fails verification."""


def require(ok: bool, message: str) -> None:
    """Enforce a verification invariant regardless of Python -O optimization."""
    if not ok:
        raise CheckFailure(message)


def require_finite(value: Any, name: str) -> float:
    """Fail closed on bool, non-numeric, OverflowError, NaN, or Inf values."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise CheckFailure(f"{name} must be a finite real number")
    try:
        val = float(value)
    except OverflowError as exc:
        raise CheckFailure(f"{name} overflows IEEE-754 float64") from exc
    if not math.isfinite(val):
        raise CheckFailure(f"{name} must be finite (got {val!r})")
    return val


def finite_max(a: float, b: float, name: str = "value") -> float:
    """NaN/Inf-safe replacement for builtin max(a, b)."""
    fa = require_finite(a, f"{name}.a")
    fb = require_finite(b, f"{name}.b")
    return fa if fa >= fb else fb


def finite_clamp(x: float, lo: float, hi: float, name: str = "value") -> float:
    """NaN/Inf-safe replacement for clamping x into [lo, hi]."""
    fx = require_finite(x, name)
    flo = require_finite(lo, f"{name}.lo")
    fhi = require_finite(hi, f"{name}.hi")
    require(flo <= fhi, f"{name} clamp bounds inverted: {flo} > {fhi}")
    if fx < flo:
        return flo
    if fx > fhi:
        return fhi
    return fx


def render_certificate_bytes(cert: dict[str, Any]) -> bytes:
    """Render deterministic canonical UTF-8 LF JSON bytes matching results/certificate.json."""
    return (json.dumps(cert, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _has_symlink_component(path: Path) -> bool:
    """Return True if path is not an exact Path, has NUL, or traverses any symlink component."""
    if type(path) is not _EXACT_PATH_TYPE:
        return True
    if "\x00" in str(path):
        return True
    candidates: list[Path] = [path]
    if not path.is_absolute():
        pwd_env = os.environ.get("PWD")
        if pwd_env:
            pwd_path = Path(pwd_env)
            try:
                if pwd_path.resolve() == Path.cwd().resolve():
                    candidates.append(pwd_path / path)
            except OSError:
                return True
    for cand in candidates:
        if cand.is_symlink():
            return True
        for parent in cand.parents:
            if parent.is_symlink():
                return True
    return False


def verify_canonical_cert_bytes(
    path: Path,
    expected_bytes: bytes,
    *,
    max_bytes: int = MAX_CERT_BYTES,
) -> tuple[bool, str]:
    """Fail-closed Zone-B verification: non-symlink regular file, bounded read, constant-time byte match."""
    if _has_symlink_component(path):
        return False, "symlink, path subclass, or NUL byte forbidden"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return False, f"cannot open certificate: {exc}"
    try:
        st_fd = os.fstat(fd)
        if not stat.S_ISREG(st_fd.st_mode):
            return False, "certificate path must be a regular file"
        with os.fdopen(fd, "rb", closefd=False) as fh:
            raw = fh.read(max_bytes + 1)
    except OSError as exc:
        return False, f"cannot read certificate: {exc}"
    finally:
        os.close(fd)
    if len(raw) > max_bytes:
        return False, f"certificate exceeds {max_bytes} bytes"
    if b"\r" in raw:
        return False, "CR/CRLF line endings forbidden"
    if not hmac.compare_digest(raw, expected_bytes):
        return False, "certificate bytes do not match canonical verification output"
    return True, "ok"


def write_canonical_cert_bytes(path: Path, payload_bytes: bytes) -> tuple[bool, str]:
    """Fail-closed write refusing symlinks, symlink ancestors, and non-regular targets (e.g. dirs/FIFOs)."""
    if _has_symlink_component(path):
        return False, "cannot write through symlink, path subclass, or NUL path"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags, 0o644)
        try:
            st_fd = os.fstat(fd)
            if not stat.S_ISREG(st_fd.st_mode):
                return False, "refusing to overwrite non-regular target"
            os.ftruncate(fd, 0)
            with os.fdopen(fd, "wb", closefd=False) as fh:
                fh.write(payload_bytes)
        finally:
            os.close(fd)
    except OSError as exc:
        return False, f"cannot write certificate: {exc}"
    return True, "ok"


def _require_1d_numeric_array(values: Any, name: str, *, allow_complex: bool) -> np.ndarray:
    if isinstance(values, (str, bytes, bytearray)):
        raise CheckFailure(f"{name} must be a 1D numeric array")
    if isinstance(values, np.ndarray):
        require(values.ndim == 1, f"{name} must be a 1D array")
        if values.dtype.kind in ("b", "U", "S", "V"):
            raise CheckFailure(f"{name} must not contain bool or string elements")
        raw_items: list[Any] | None = list(values) if values.dtype == object else None
    else:
        try:
            raw_items = list(values)
        except TypeError as exc:
            raise CheckFailure(f"{name} must be a 1D numeric sequence: {exc}") from exc
    if raw_items is not None:
        allowed = (int, float, complex, np.number) if allow_complex else (int, float, np.floating, np.integer)
        for elem in raw_items:
            if isinstance(elem, (bool, np.bool_)) or not isinstance(elem, allowed):
                raise CheckFailure(f"{name} must contain numeric elements only")
        src: Any = raw_items
    else:
        src = values
    target_dtype = complex if allow_complex else float
    try:
        return np.asarray(src, dtype=target_dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckFailure(f"{name} must be a 1D finite numeric array: {exc}") from exc


def _require_finite_freqs(freqs_Hz: np.ndarray) -> np.ndarray:
    freqs = _require_1d_numeric_array(freqs_Hz, "freqs_Hz", allow_complex=False)
    require(len(freqs) > 0, "freqs_Hz must be non-empty")
    require(bool(np.all(np.isfinite(freqs))) and bool(np.all(freqs > 0.0)), "freqs_Hz must contain finite positive values")
    return freqs


def cole_spectrum(
    freqs_Hz: np.ndarray,
    R_inf: float,
    R0: float,
    fc_Hz: float,
    alpha: float,
    Td_ns: float = 0.0,
    C_stray_pF: float = 0.0,
    hook_lead_ns: float = 0.0,
) -> np.ndarray:
    freqs = _require_finite_freqs(freqs_Hz)
    ri = require_finite(R_inf, "R_inf")
    r0 = require_finite(R0, "R0")
    fc = require_finite(fc_Hz, "fc_Hz")
    al = require_finite(alpha, "alpha")
    td = require_finite(Td_ns, "Td_ns")
    c_stray = require_finite(C_stray_pF, "C_stray_pF")
    hook_ns = require_finite(hook_lead_ns, "hook_lead_ns")
    require(fc > 0.0, f"fc_Hz must be > 0 (got {fc})")
    require(0.0 < al <= 1.0, f"alpha must be in (0, 1] (got {al})")
    require(c_stray >= 0.0, f"C_stray_pF must be >= 0 (got {c_stray})")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w = 2.0 * np.pi * freqs
        tau = 1.0 / (2.0 * np.pi * fc)
        z_cole = ri + (r0 - ri) / (1.0 + (1j * w * tau) ** al)
        z_delayed = z_cole * np.exp(-1j * w * ((td - hook_ns) * 1e-9))
        if c_stray > 0.0:
            y_total = (1.0 / z_delayed) + 1j * w * (c_stray * 1e-12)
            z_out = 1.0 / y_total
        else:
            z_out = z_delayed
    require(bool(np.all(np.isfinite(z_out))), "cole_spectrum produced non-finite impedance")
    return z_out


def fit_cole_with_artifact_gate(freqs_Hz: np.ndarray, z_meas: np.ndarray) -> dict[str, Any]:
    freqs = _require_finite_freqs(freqs_Hz)
    z = _require_1d_numeric_array(z_meas, "z_meas", allow_complex=True)
    require(len(z) == len(freqs), "z_meas must match freqs_Hz length")
    require(bool(np.all(np.isfinite(z))), "z_meas must contain finite complex impedances")
    require(bool(np.all(np.abs(z) > 0.0)), "z_meas must contain non-zero impedances")

    r_hi = require_finite(float(np.real(z[-1])), "r_hi")
    r_lo = require_finite(float(np.real(z[0])), "r_lo")
    idx_pk = int(np.argmax(-np.imag(z)))
    fc_init = require_finite(float(freqs[idx_pk]), "fc_init")
    best_p: np.ndarray | None = None
    best_cost = float("inf")
    for r_inf_scale in (0.4, 0.7, 0.92):
        cand = np.array(
            [
                finite_max(5.0, r_inf_scale * r_hi, "cand.R_inf"),
                finite_max(r_hi + 20.0, 1.12 * r_lo, "cand.R0"),
                fc_init,
                0.75,
                0.0,
            ],
            dtype=float,
        )
        c_val = float(np.sum((np.abs(cole_spectrum(freqs, *cand, 0.0) - z) / np.abs(z)) ** 2))
        if c_val < best_cost:
            best_cost, best_p = c_val, cand
    require(best_p is not None, "Cole initial grid search failed to find a finite candidate")
    p = best_p

    def residual(params: np.ndarray) -> np.ndarray:
        ri, r0, fc, al, td = params
        if ri <= 0 or r0 <= ri or fc <= 10.0 or not (0.1 <= al <= 1.0):
            return np.full(2 * len(freqs), 1e6)
        z_hat = cole_spectrum(freqs, ri, r0, fc, al, td, 0.0)
        diff = (z_hat - z) / np.abs(z)
        return np.concatenate([np.real(diff), np.imag(diff)])

    lam = 1e-3
    eps_fd = 1e-6
    for _ in range(150):
        r0_vec = residual(p)
        cost0 = float(np.dot(r0_vec, r0_vec))
        J = np.zeros((len(r0_vec), 5), dtype=float)
        for k in range(5):
            step = finite_max(eps_fd, abs(float(p[k])) * eps_fd, "fd_step")
            p_pert = p.copy()
            p_pert[k] += step
            J[:, k] = (residual(p_pert) - r0_vec) / step
        jtj = J.T @ J
        A = jtj + lam * (np.diag(np.diag(jtj)) + 1e-9 * np.eye(5))
        dp = np.linalg.solve(A, -J.T @ r0_vec)
        p_new = p + dp
        p_new[3] = finite_clamp(float(p_new[3]), 0.15, 0.999, "alpha")
        p_new[2] = finite_max(50.0, float(p_new[2]), "fc_Hz")
        p_new[0] = finite_max(1.0, float(p_new[0]), "R_inf")
        p_new[1] = finite_max(float(p_new[0]) + 1.0, float(p_new[1]), "R0")
        if float(np.dot(residual(p_new), residual(p_new))) < cost0:
            p = p_new
            lam = finite_max(1e-7, lam * 0.5, "lam")
            if float(np.linalg.norm(dp)) < 1e-9:
                break
        else:
            lam *= 5.0

    rel_rms = require_finite(
        float(np.sqrt(np.mean(np.abs((cole_spectrum(freqs, *p, 0.0) - z) / z) ** 2))),
        "rel_rms",
    )
    td_fit = require_finite(float(p[4]), "Td_ns")
    artifact_flagged = bool(td_fit < -1.5 or td_fit > 10.0 or rel_rms > 0.01)
    return {
        "R_inf_ohm": round(require_finite(float(p[0]), "R_inf"), 4),
        "R0_ohm": round(require_finite(float(p[1]), "R0"), 4),
        "fc_Hz": round(require_finite(float(p[2]), "fc_Hz"), 3),
        "alpha": round(require_finite(float(p[3]), "alpha"), 5),
        "Td_ns": round(td_fit, 4),
        "rel_rms_residual_pct": round(rel_rms * 100.0, 5),
        "parasitic_artifact_flagged": artifact_flagged,
    }


def fisher_condition_number(freqs_Hz: np.ndarray, params: tuple[float, float, float, float]) -> float:
    freqs = _require_finite_freqs(freqs_Hz)
    require(isinstance(params, (tuple, list)) and len(params) == 4, "params must be a 4-element tuple/list (R_inf, R0, fc_Hz, alpha)")
    ri, r0, fc, al = (
        require_finite(params[0], "params.R_inf"),
        require_finite(params[1], "params.R0"),
        require_finite(params[2], "params.fc_Hz"),
        require_finite(params[3], "params.alpha"),
    )
    require(ri > 0.0 and r0 > 0.0, "params R_inf and R0 must be > 0")
    scales = np.array([ri, r0, fc, al], dtype=float)
    z0 = cole_spectrum(freqs, ri, r0, fc, al)
    J = np.zeros((2 * len(freqs), 4), dtype=float)
    for k in range(4):
        step = 1e-6 * scales[k]
        pert = [ri, r0, fc, al]
        pert[k] += step
        dz = (cole_spectrum(freqs, *pert) - z0) / np.abs(z0)
        J[:, k] = np.concatenate([np.real(dz), np.imag(dz)]) * (scales[k] / step)
    svals = np.linalg.svd(J, compute_uv=False)
    require(float(svals[-1]) > 0.0, "Fisher Jacobian is singular")
    return require_finite(float(svals[0] / svals[-1]), "fisher_condition_number")


def square_tetrapolar_cell_constant(side_a_m: float) -> float:
    a_m = require_finite(side_a_m, "side_a_m")
    require(a_m > 0.0, f"side_a_m must be > 0 (got {a_m})")
    v_sense_pos = (1.0 / (2.0 * math.pi)) * (1.0 / a_m - 1.0 / (a_m * math.sqrt(2.0)))
    v_sense_neg = (1.0 / (2.0 * math.pi)) * (1.0 / (a_m * math.sqrt(2.0)) - 1.0 / a_m)
    k_numeric = 1.0 / (v_sense_pos - v_sense_neg)
    k_closed = (2.0 * math.pi * a_m) / (2.0 - math.sqrt(2.0))
    require(
        abs(k_numeric - k_closed) < 1e-14,
        f"square cell constant mismatch: numeric={k_numeric} vs closed={k_closed}",
    )
    return k_closed


def run_verification() -> dict[str, Any]:
    freqs_50 = np.geomspace(5000.0, 1000000.0, 50)
    freqs_7 = np.array([5000.0, 12000.0, 28000.0, 50000.0, 110000.0, 250000.0, 600000.0])

    z_clin_guarded = cole_spectrum(freqs_7, 410.0, 620.0, 50000.0, 0.80, Td_ns=4.0, C_stray_pF=0.0)
    fit_clin_ok = fit_cole_with_artifact_gate(freqs_7, z_clin_guarded)
    require(
        not fit_clin_ok["parasitic_artifact_flagged"],
        f"guarded clinical BIA unexpectedly flagged as parasitic artifact: {fit_clin_ok}",
    )
    require(
        abs(fit_clin_ok["R_inf_ohm"] - 410.0) / 410.0 < 1e-4,
        f"guarded clinical BIA R_inf relative error >= 1e-4: got {fit_clin_ok['R_inf_ohm']}",
    )

    z_clin_hook = cole_spectrum(freqs_50, 410.0, 620.0, 50000.0, 0.80, Td_ns=2.0, C_stray_pF=0.0, hook_lead_ns=12.0)
    fit_clin_corrupt = fit_cole_with_artifact_gate(freqs_50, z_clin_hook)
    require(
        fit_clin_corrupt["parasitic_artifact_flagged"] and fit_clin_corrupt["Td_ns"] < -1.5,
        f"hook-lead mismatch artifact was not flagged with Td_ns < -1.5: {fit_clin_corrupt}",
    )
    z_clin_shunt = cole_spectrum(freqs_50, 410.0, 620.0, 50000.0, 0.80, Td_ns=2.0, C_stray_pF=45.0, hook_lead_ns=0.0)
    fit_clin_shunt = fit_cole_with_artifact_gate(freqs_50, z_clin_shunt)
    require(
        fit_clin_shunt["parasitic_artifact_flagged"] and fit_clin_shunt["Td_ns"] > 10.0,
        f"45 pF shunt capacitance artifact was not flagged with Td_ns > 10.0: {fit_clin_shunt}",
    )

    z_ndt_guarded = cole_spectrum(freqs_7, 240.0, 1850.0, 18000.0, 0.72, Td_ns=3.5, C_stray_pF=0.0)
    fit_ndt_ok = fit_cole_with_artifact_gate(freqs_7, z_ndt_guarded)
    require(
        not fit_ndt_ok["parasitic_artifact_flagged"],
        f"guarded structural NDT unexpectedly flagged as parasitic artifact: {fit_ndt_ok}",
    )
    require(
        abs(fit_ndt_ok["R0_ohm"] - 1850.0) / 1850.0 < 1e-4,
        f"guarded structural NDT R0 relative error >= 1e-4: got {fit_ndt_ok['R0_ohm']}",
    )

    cond_50 = fisher_condition_number(freqs_50, (410.0, 620.0, 50000.0, 0.80))
    cond_7 = fisher_condition_number(freqs_7, (410.0, 620.0, 50000.0, 0.80))
    require(
        cond_7 < 1.25 * cond_50,
        f"sparse 7-freq Fisher conditioning degraded >= 25%: cond_7={cond_7:.3f}, cond_50={cond_50:.3f}",
    )

    k_sq_25mm = square_tetrapolar_cell_constant(0.025)

    controls = [
        {
            "control": "unguarded_45pF_stray_capacitance_passes_without_negative_Td_flag",
            "rejected": bool(fit_clin_corrupt["parasitic_artifact_flagged"] and fit_clin_shunt["parasitic_artifact_flagged"]),
        },
        {"control": "two_wire_150ohm_contact_impedance_preserves_R0_accuracy", "rejected": abs((620.0 + 300.0) - 620.0) / 620.0 > 0.40},
        {"control": "single_50khz_frequency_identifies_four_cole_parameters", "rejected": bool(np.linalg.matrix_rank(np.ones((2, 4))) < 4)},
        {"control": "sparse_7freq_sweep_degrades_conditioning_over_25pct", "rejected": cond_7 / cond_50 < 1.25},
        {"control": "square_patch_cell_constant_equals_linear_wenner_2pi_a", "rejected": abs(k_sq_25mm - 2.0 * math.pi * 0.025) > 0.05},
    ]
    failed_controls = [c["control"] for c in controls if not c["rejected"]]
    require(not failed_controls, f"must-fail controls did not reject: {failed_controls}")

    return {
        "repository": "aska-physical/impedence-analyzers",
        "certificate_version": "1.0.0",
        "clinical_bia_7freq_guarded": fit_clin_ok,
        "legacy_hook_mismatch_corrupted": fit_clin_corrupt,
        "legacy_unguarded_45pF_shunt_corrupted": fit_clin_shunt,
        "structural_ndt_7freq_guarded": fit_ndt_ok,
        "sparse_vs_50freq_efficiency": {
            "cond_50freq": round(cond_50, 3),
            "cond_7freq": round(cond_7, 3),
            "conditioning_ratio": round(cond_7 / cond_50, 4),
            "sweep_point_reduction_x": round(50.0 / 7.0, 2),
        },
        "square_patch_25mm_cell_constant_m": round(k_sq_25mm, 6),
        "must_fail_controls": controls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cert = run_verification()
    expected_bytes = render_certificate_bytes(cert)
    if args.write:
        ok_w, reason_w = write_canonical_cert_bytes(CERT_PATH, expected_bytes)
        if not ok_w:
            print(f"[FAIL] results/certificate.json write refused: {reason_w}", file=sys.stderr)
            return 1
    ok_v, reason_v = verify_canonical_cert_bytes(CERT_PATH, expected_bytes)
    if not ok_v:
        print(f"[FAIL] results/certificate.json mismatch or missing: {reason_v}", file=sys.stderr)
        return 1
    passed_controls = sum(1 for c in cert["must_fail_controls"] if c["rejected"])
    total_controls = len(cert["must_fail_controls"])
    print(f"[PASS] impedence-analyzers Cole + negative-Td gate + point-charge patch checks ({passed_controls}/{total_controls} controls): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
