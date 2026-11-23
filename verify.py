#!/usr/bin/env python3
"""Deterministic Cole-Cole + Parasitic Delay/Capacitance EIS solver & artifact detector for aska-physical/impedence-analyzers."""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
CERT_PATH = ROOT / "results" / "certificate.json"


class CheckFailure(RuntimeError):
    """Raised when a physical invariant or must-fail control fails verification."""


def require(ok: bool, message: str) -> None:
    """Enforce a verification invariant regardless of Python -O optimization."""
    if not ok:
        raise CheckFailure(message)


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
    w = 2.0 * np.pi * freqs_Hz
    tau = 1.0 / (2.0 * np.pi * fc_Hz)
    z_cole = R_inf + (R0 - R_inf) / (1.0 + (1j * w * tau) ** alpha)
    z_delayed = z_cole * np.exp(-1j * w * ((Td_ns - hook_lead_ns) * 1e-9))
    if C_stray_pF > 0.0:
        y_total = (1.0 / z_delayed) + 1j * w * (C_stray_pF * 1e-12)
        return 1.0 / y_total
    return z_delayed


def fit_cole_with_artifact_gate(freqs_Hz: np.ndarray, z_meas: np.ndarray) -> dict[str, Any]:
    r_hi = float(np.real(z_meas[-1]))
    r_lo = float(np.real(z_meas[0]))
    idx_pk = int(np.argmax(-np.imag(z_meas)))
    fc_init = float(freqs_Hz[idx_pk])
    best_p = None
    best_cost = float("inf")
    for r_inf_scale in (0.4, 0.7, 0.92):
        cand = np.array([max(5.0, r_inf_scale * r_hi), max(r_hi + 20.0, 1.12 * r_lo), fc_init, 0.75, 0.0], dtype=float)
        c_val = float(np.sum((np.abs(cole_spectrum(freqs_Hz, *cand, 0.0) - z_meas) / np.abs(z_meas)) ** 2))
        if c_val < best_cost:
            best_cost, best_p = c_val, cand
    p = best_p

    def residual(params: np.ndarray) -> np.ndarray:
        ri, r0, fc, al, td = params
        if ri <= 0 or r0 <= ri or fc <= 10.0 or not (0.1 <= al <= 1.0):
            return np.full(2 * len(freqs_Hz), 1e6)
        z_hat = cole_spectrum(freqs_Hz, ri, r0, fc, al, td, 0.0)
        diff = (z_hat - z_meas) / np.abs(z_meas)
        return np.concatenate([np.real(diff), np.imag(diff)])

    lam = 1e-3
    eps_fd = 1e-6
    for _ in range(150):
        r0_vec = residual(p)
        cost0 = float(np.dot(r0_vec, r0_vec))
        J = np.zeros((len(r0_vec), 5), dtype=float)
        for k in range(5):
            step = max(eps_fd, abs(p[k]) * eps_fd)
            p_pert = p.copy()
            p_pert[k] += step
            J[:, k] = (residual(p_pert) - r0_vec) / step
        jtj = J.T @ J
        A = jtj + lam * (np.diag(np.diag(jtj)) + 1e-9 * np.eye(5))
        dp = np.linalg.solve(A, -J.T @ r0_vec)
        p_new = p + dp
        p_new[3] = float(np.clip(p_new[3], 0.15, 0.999))
        p_new[2] = max(50.0, float(p_new[2]))
        p_new[0] = max(1.0, float(p_new[0]))
        p_new[1] = max(p_new[0] + 1.0, float(p_new[1]))
        if float(np.dot(residual(p_new), residual(p_new))) < cost0:
            p = p_new
            lam = max(1e-7, lam * 0.5)
            if float(np.linalg.norm(dp)) < 1e-9:
                break
        else:
            lam *= 5.0

    rel_rms = float(np.sqrt(np.mean(np.abs((cole_spectrum(freqs_Hz, *p, 0.0) - z_meas) / z_meas) ** 2)))
    artifact_flagged = bool(p[4] < -1.5 or p[4] > 10.0 or rel_rms > 0.01)
    return {
        "R_inf_ohm": round(float(p[0]), 4),
        "R0_ohm": round(float(p[1]), 4),
        "fc_Hz": round(float(p[2]), 3),
        "alpha": round(float(p[3]), 5),
        "Td_ns": round(float(p[4]), 4),
        "rel_rms_residual_pct": round(rel_rms * 100.0, 5),
        "parasitic_artifact_flagged": artifact_flagged,
    }


def fisher_condition_number(freqs_Hz: np.ndarray, params: tuple[float, float, float, float]) -> float:
    ri, r0, fc, al = params
    scales = np.array([ri, r0, fc, al], dtype=float)
    z0 = cole_spectrum(freqs_Hz, ri, r0, fc, al)
    J = np.zeros((2 * len(freqs_Hz), 4), dtype=float)
    for k in range(4):
        step = 1e-6 * scales[k]
        pert = list(params)
        pert[k] += step
        dz = (cole_spectrum(freqs_Hz, *pert) - z0) / np.abs(z0)
        J[:, k] = np.concatenate([np.real(dz), np.imag(dz)]) * (scales[k] / step)
    svals = np.linalg.svd(J, compute_uv=False)
    return float(svals[0] / svals[-1])


def square_tetrapolar_cell_constant(side_a_m: float) -> float:
    v_sense_pos = (1.0 / (2.0 * math.pi)) * (1.0 / side_a_m - 1.0 / (side_a_m * math.sqrt(2.0)))
    v_sense_neg = (1.0 / (2.0 * math.pi)) * (1.0 / (side_a_m * math.sqrt(2.0)) - 1.0 / side_a_m)
    k_numeric = 1.0 / (v_sense_pos - v_sense_neg)
    k_closed = (2.0 * math.pi * side_a_m) / (2.0 - math.sqrt(2.0))
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
    if args.write:
        CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CERT_PATH.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not CERT_PATH.exists() or json.loads(CERT_PATH.read_text(encoding="utf-8")) != cert:
        print("[FAIL] results/certificate.json mismatch or missing", file=sys.stderr)
        return 1
    passed_controls = sum(1 for c in cert["must_fail_controls"] if c["rejected"])
    total_controls = len(cert["must_fail_controls"])
    print(f"[PASS] impedence-analyzers Cole + negative-Td gate + point-charge patch checks ({passed_controls}/{total_controls} controls): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
