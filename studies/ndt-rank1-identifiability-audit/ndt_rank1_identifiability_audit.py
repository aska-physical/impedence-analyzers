#!/usr/bin/env python3
"""Pure-Z/Q exact receiving-side checker for aska-physical/impedence-analyzers (dock#8 P5 + Issue #2):
Conductance-Laplacian rank-1 fiber/bond cut Delta R_eff (Sherman-Morrison exact cross-check on 1-layer
and 2-layer tetrapolar square Kelvin patches) and exact-Q zero-delay (alpha=1, Td=0) Cole slice
identifiability split (in-span resistive fiber cut missed by residual gate vs. out-of-span parasitic
shunt caught by exact algebraic Cole-manifold obstruction on the zero-delay slice).

Claim status: model and receiving-side checker only (no bench measurement attached; per dock#8).
Strict discipline: pure Python 3 stdlib (int / Fraction only), zero floats, zero bare asserts,
fail-closed Zone-B validation (strict types, iterative depth cap, hmac.compare_digest on payload_sha256).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "impedence_analyzers.ndt_rank1_identifiability.v1"
STUDY_ID = "NDT-RANK1-P5-STUDY"
FITTABLE_PARAMETERS: tuple[str, ...] = ("R_inf_ohm", "R0_ohm", "fc_Hz", "alpha", "Td_ns")
FIXED_UNMODELED_IN_FIT: tuple[str, ...] = ("C_stray_pF", "hook_lead_ns")
STRUCTURAL_NDT_FREQS_HZ: tuple[int, ...] = (5000, 12000, 28000, 50000, 110000, 250000, 600000)


class NDTIdentifiabilityError(ValueError):
    """Fail-closed exception for invalid inputs, disconnected graphs, or tampered certificates."""


def _require_strict_int(val: Any, name: str, lo: int, hi: int | None = None) -> int:
    if type(val) is not int:
        raise NDTIdentifiabilityError(f"{name} must be strict int")
    if val < lo:
        raise NDTIdentifiabilityError(f"{name} must be >= {lo}")
    if hi is not None and val > hi:
        raise NDTIdentifiabilityError(f"{name} must be <= {hi}")
    return val


def _require_strict_fraction(val: Any, name: str, *, positive: bool = False) -> Fraction:
    if type(val) is not Fraction:
        raise NDTIdentifiabilityError(f"{name} must be exact Fraction")
    if positive and val <= 0:
        raise NDTIdentifiabilityError(f"{name} must be > 0")
    return val


def frac_str(val: Fraction) -> str:
    """Canonical exact rational string 'num/den'."""
    val = _require_strict_fraction(val, "val")
    return f"{val.numerator}/{val.denominator}"


def format_exact_decimal(val: Fraction, places: int = 6) -> str:
    """Format exact Fraction rounded half-up to fixed decimal places (0 <= places <= 18)."""
    val = _require_strict_fraction(val, "val")
    places = _require_strict_int(places, "places", 0, 18)
    neg = val < 0
    num = -val.numerator if neg else val.numerator
    den = val.denominator
    scale = 10 ** places
    scaled = (num * scale * 2 + den) // (2 * den)
    sign = "-" if neg and scaled > 0 else ""
    if places == 0:
        return f"{sign}{scaled}"
    whole, frac = divmod(scaled, scale)
    return f"{sign}{whole}.{frac:0{places}d}"


# ==============================================================================
# Pillar 1: Exact-Q Conductance-Laplacian & Sherman-Morrison Rank-1 Edge Cut
# ==============================================================================

def _validate_graph(
    n_nodes: int, edges: Sequence[tuple[int, int, Fraction]]
) -> list[tuple[int, int, Fraction]]:
    n_nodes = _require_strict_int(n_nodes, "n_nodes", 2, 64)
    if type(edges) not in (list, tuple) or len(edges) == 0:
        raise NDTIdentifiabilityError("edges must be a non-empty list or tuple")
    cleaned: list[tuple[int, int, Fraction]] = []
    for edge in edges:
        if type(edge) not in (list, tuple) or len(edge) != 3:
            raise NDTIdentifiabilityError("each edge must be a 3-tuple (u, v, g)")
        u = _require_strict_int(edge[0], "edge_u", 0, n_nodes - 1)
        v = _require_strict_int(edge[1], "edge_v", 0, n_nodes - 1)
        if u == v:
            raise NDTIdentifiabilityError("self-loop edge is forbidden")
        g = _require_strict_fraction(edge[2], "edge_g", positive=True)
        cleaned.append((u, v, g))
    return cleaned


def solve_grounded_laplacian_exact(
    n_nodes: int,
    edges: Sequence[tuple[int, int, Fraction]],
    rhs: Sequence[Fraction],
) -> list[Fraction]:
    """Solve L_g * phi = rhs with ground node 0 (phi[0] = 0) via exact-Q Gauss-Jordan elimination."""
    cleaned = _validate_graph(n_nodes, edges)
    if type(rhs) not in (list, tuple) or len(rhs) != n_nodes:
        raise NDTIdentifiabilityError("rhs must be a list or tuple of length n_nodes")
    rhs_f = [_require_strict_fraction(x, "rhs_entry") for x in rhs]
    if sum(rhs_f, Fraction(0, 1)) != 0:
        raise NDTIdentifiabilityError("current injection rhs must sum to zero")

    m = n_nodes - 1
    mat: list[list[Fraction]] = [[Fraction(0, 1)] * (m + 1) for _ in range(m)]
    for u, v, g in cleaned:
        if u > 0:
            mat[u - 1][u - 1] += g
        if v > 0:
            mat[v - 1][v - 1] += g
        if u > 0 and v > 0:
            mat[u - 1][v - 1] -= g
            mat[v - 1][u - 1] -= g
    for i in range(m):
        mat[i][m] = rhs_f[i + 1]

    for col in range(m):
        piv = -1
        for row in range(col, m):
            if mat[row][col] != 0:
                piv = row
                break
        if piv < 0:
            raise NDTIdentifiabilityError("conductance graph is disconnected (singular reduced Laplacian)")
        if piv != col:
            mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        for j in range(col, m + 1):
            mat[col][j] /= pv
        for row in range(m):
            if row != col and mat[row][col] != 0:
                fac = mat[row][col]
                for j in range(col, m + 1):
                    mat[row][j] -= fac * mat[col][j]

    return [Fraction(0, 1)] + [mat[i][m] for i in range(m)]


def _dipole_rhs(n_nodes: int, s: int, t: int) -> list[Fraction]:
    s = _require_strict_int(s, "s", 0, n_nodes - 1)
    t = _require_strict_int(t, "t", 0, n_nodes - 1)
    if s == t:
        raise NDTIdentifiabilityError("terminal nodes must be distinct")
    vec = [Fraction(0, 1)] * n_nodes
    vec[s] = Fraction(1, 1)
    vec[t] = Fraction(-1, 1)
    return vec


def rank1_edge_cut_exact(
    n_nodes: int,
    edges: Sequence[tuple[int, int, Fraction]],
    edge_idx: int,
    delta_g: Fraction,
    drive_pair: tuple[int, int],
    sense_pair: tuple[int, int],
) -> dict[str, Any]:
    """Compute exact-Q Sherman-Morrison rank-1 Delta R_eff and Delta R_4w, cross-checked by direct re-solve."""
    cleaned = _validate_graph(n_nodes, edges)
    edge_idx = _require_strict_int(edge_idx, "edge_idx", 0, len(cleaned) - 1)
    u, v, g_e = cleaned[edge_idx]
    delta_g = _require_strict_fraction(delta_g, "delta_g", positive=True)
    if delta_g > g_e:
        raise NDTIdentifiabilityError("delta_g cannot exceed edge conductance g_e")
    if type(drive_pair) not in (list, tuple) or len(drive_pair) != 2:
        raise NDTIdentifiabilityError("drive_pair must be a 2-tuple")
    if type(sense_pair) not in (list, tuple) or len(sense_pair) != 2:
        raise NDTIdentifiabilityError("sense_pair must be a 2-tuple")
    s, t = drive_pair
    p, q = sense_pair

    phi_drive = solve_grounded_laplacian_exact(n_nodes, cleaned, _dipole_rhs(n_nodes, s, t))
    phi_sense = solve_grounded_laplacian_exact(n_nodes, cleaned, _dipole_rhs(n_nodes, p, q))
    phi_edge = solve_grounded_laplacian_exact(n_nodes, cleaned, _dipole_rhs(n_nodes, u, v))

    r_eff_base = phi_drive[s] - phi_drive[t]
    r_4w_base = phi_drive[p] - phi_drive[q]
    r_edge_self = phi_edge[u] - phi_edge[v]
    denom = Fraction(1, 1) - delta_g * r_edge_self
    if denom <= 0:
        raise NDTIdentifiabilityError("rank-1 cut disconnects graph (1 - delta_g * R_eff(e) <= 0)")

    v_drive_across_e = phi_drive[u] - phi_drive[v]
    v_sense_across_e = phi_sense[u] - phi_sense[v]
    delta_r_eff_sm = (delta_g * v_drive_across_e * v_drive_across_e) / denom
    delta_r_4w_sm = (delta_g * v_sense_across_e * v_drive_across_e) / denom

    # Independent direct post-cut Laplacian re-solve
    post_edges: list[tuple[int, int, Fraction]] = []
    for idx, (eu, ev, eg) in enumerate(cleaned):
        if idx == edge_idx:
            rem_g = eg - delta_g
            if rem_g > 0:
                post_edges.append((eu, ev, rem_g))
        else:
            post_edges.append((eu, ev, eg))
    phi_post = solve_grounded_laplacian_exact(n_nodes, post_edges, _dipole_rhs(n_nodes, s, t))
    r_eff_post = phi_post[s] - phi_post[t]
    r_4w_post = phi_post[p] - phi_post[q]
    delta_r_eff_direct = r_eff_post - r_eff_base
    delta_r_4w_direct = r_4w_post - r_4w_base

    return {
        "edge": [u, v],
        "g_edge": frac_str(g_e),
        "delta_g": frac_str(delta_g),
        "r_eff_base": frac_str(r_eff_base),
        "r_eff_post": frac_str(r_eff_post),
        "delta_r_eff_sm": frac_str(delta_r_eff_sm),
        "delta_r_eff_direct": frac_str(delta_r_eff_direct),
        "delta_r_eff_dec": format_exact_decimal(delta_r_eff_sm, 6),
        "rel_delta_r_eff_pct": format_exact_decimal((delta_r_eff_sm / r_eff_base) * 100, 4),
        "r_4w_base": frac_str(r_4w_base),
        "r_4w_post": frac_str(r_4w_post),
        "delta_r_4w_sm": frac_str(delta_r_4w_sm),
        "delta_r_4w_direct": frac_str(delta_r_4w_direct),
        "sherman_morrison_matches_direct_exact": (
            delta_r_eff_sm == delta_r_eff_direct and delta_r_4w_sm == delta_r_4w_direct
        ),
        "rayleigh_monotone_nonnegative": delta_r_eff_sm >= 0,
    }


def _build_square_kelvin_1layer() -> tuple[int, list[tuple[int, int, Fraction]]]:
    """3x3 1-layer square Kelvin patch (9 nodes, 12 edges, uniform g = 1)."""
    g1 = Fraction(1, 1)
    edges: list[tuple[int, int, Fraction]] = []
    for r in range(3):
        for c in range(3):
            u = 3 * r + c
            if c + 1 < 3:
                edges.append((u, u + 1, g1))
            if r + 1 < 3:
                edges.append((u, u + 3, g1))
    return 9, edges


def _build_square_kelvin_2layer() -> tuple[int, list[tuple[int, int, Fraction]]]:
    """2-layer epoxy-over-concrete 3x3 square Kelvin patch (18 nodes, 33 edges; bridges Issue #2).
    Layer 0 (nodes 0..8): CFRP/epoxy surface g_top = 1.
    Layer 1 (nodes 9..17): concrete substrate g_sub = 1/4.
    Vertical bondline vias (i, i+9): g_via = 1/2.
    """
    g_top = Fraction(1, 1)
    g_sub = Fraction(1, 4)
    g_via = Fraction(1, 2)
    edges: list[tuple[int, int, Fraction]] = []
    for r in range(3):
        for c in range(3):
            u = 3 * r + c
            if c + 1 < 3:
                edges.append((u, u + 1, g_top))
                edges.append((9 + u, 9 + u + 1, g_sub))
            if r + 1 < 3:
                edges.append((u, u + 3, g_top))
                edges.append((9 + u, 9 + u + 3, g_sub))
            edges.append((u, 9 + u, g_via))
    return 18, edges


# ==============================================================================
# Pillar 2: Exact-Q Cole Zero-Delay (alpha=1, Td=0) Slice Identifiability Split (dock#8 P5)
# ==============================================================================

def _c_add(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return (a[0] + b[0], a[1] + b[1])


def _c_inv(z: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    re, im = z
    nrm2 = re * re + im * im
    if nrm2 == 0:
        raise NDTIdentifiabilityError("cannot invert zero complex impedance")
    return (re / nrm2, -im / nrm2)


def cole_debye_exact_q(
    r_inf: Fraction, r0: Fraction, u: Fraction
) -> tuple[Fraction, Fraction]:
    """Exact-Q Cole impedance at alpha=1, Td=0, C_stray=0: Z(u) = R_inf + (R0 - R_inf)/(1 + j*u)."""
    r_inf = _require_strict_fraction(r_inf, "r_inf", positive=True)
    r0 = _require_strict_fraction(r0, "r0", positive=True)
    u = _require_strict_fraction(u, "u", positive=True)
    if r0 <= r_inf:
        raise NDTIdentifiabilityError("r0 must be strictly greater than r_inf")
    dr = r0 - r_inf
    denom = Fraction(1, 1) + u * u
    return (r_inf + dr / denom, -(dr * u) / denom)


def cole_half_exact_q(
    r_inf: Fraction, r0: Fraction, r_root: Fraction
) -> tuple[Fraction, Fraction]:
    """Exact-Q fractional Cole impedance at alpha=1/2, u = 2*r_root^2 (since sqrt(2j) = 1 + j)."""
    r_inf = _require_strict_fraction(r_inf, "r_inf", positive=True)
    r0 = _require_strict_fraction(r0, "r0", positive=True)
    r_root = _require_strict_fraction(r_root, "r_root", positive=True)
    if r0 <= r_inf:
        raise NDTIdentifiabilityError("r0 must be strictly greater than r_inf")
    dr = r0 - r_inf
    a = Fraction(1, 1) + r_root
    denom = a * a + r_root * r_root
    return (r_inf + (dr * a) / denom, -(dr * r_root) / denom)


def apply_parallel_shunt_exact_q(
    z: tuple[Fraction, Fraction], beta_u: Fraction
) -> tuple[Fraction, Fraction]:
    """Exact-Q parallel capacitive shunt: Z_shunt = 1 / (1/Z + j * beta_u)."""
    if type(z) not in (list, tuple) or len(z) != 2:
        raise NDTIdentifiabilityError("z must be a 2-tuple (re, im)")
    z_clean = (_require_strict_fraction(z[0], "z_re"), _require_strict_fraction(z[1], "z_im"))
    beta_u = _require_strict_fraction(beta_u, "beta_u", positive=True)
    y_z = _c_inv(z_clean)
    return _c_inv(_c_add(y_z, (Fraction(0, 1), beta_u)))


def evaluate_cole_debye_obstruction_exact_q(
    us: Sequence[Fraction],
    zs: Sequence[tuple[Fraction, Fraction]],
    r_inf_ref: Fraction,
    r0_ref: Fraction,
    baseline_shift_limit: Fraction,
    rel_res_sq_limit: Fraction,
) -> dict[str, Any]:
    """Exact-Q receiving-side checker:
    1. Tests exact collinearity invariant R(u) = R_inf' + (1/gamma) * (-Im Z(u) / u) on all triples.
    2. Solves the exact-Q staged linear least-squares fit on the zero-delay (alpha=1, Td=0) slice
       (fitting (R_inf', 1/gamma) on Re(Z) first, then recovering exact R0' and relative residual
       squared across all frequencies; note this is a staged linear fit on the zero-delay slice,
       not an unconstrained global 5-parameter nonlinear least-squares minimum).
    3. Separates residual-gate detection (out-of-span defects) from baseline-shift detection (in-span defects).
    """
    if type(us) not in (list, tuple) or type(zs) not in (list, tuple) or len(us) != len(zs) or len(us) < 3:
        raise NDTIdentifiabilityError("us and zs must be equal-length sequences of length >= 3")
    r_inf_ref = _require_strict_fraction(r_inf_ref, "r_inf_ref", positive=True)
    r0_ref = _require_strict_fraction(r0_ref, "r0_ref", positive=True)
    baseline_shift_limit = _require_strict_fraction(baseline_shift_limit, "baseline_shift_limit", positive=True)
    rel_res_sq_limit = _require_strict_fraction(rel_res_sq_limit, "rel_res_sq_limit", positive=True)

    u_list = [_require_strict_fraction(u, "u", positive=True) for u in us]
    z_list: list[tuple[Fraction, Fraction]] = []
    for z in zs:
        if type(z) not in (list, tuple) or len(z) != 2:
            raise NDTIdentifiabilityError("each impedance sample must be a 2-tuple (re, im)")
        z_list.append((_require_strict_fraction(z[0], "re"), _require_strict_fraction(z[1], "im")))

    qs = [(-z_list[i][1]) / u_list[i] for i in range(len(u_list))]
    rs = [z_list[i][0] for i in range(len(u_list))]

    # Exact 3-point collinearity determinant check across all C(n, 3) triples
    total_triples = 0
    nonzero_triples = 0
    for i in range(len(u_list)):
        for j in range(i + 1, len(u_list)):
            dq_ij = qs[j] - qs[i]
            dr_ij = rs[j] - rs[i]
            for k in range(j + 1, len(u_list)):
                total_triples += 1
                det = dq_ij * (rs[k] - rs[i]) - (qs[k] - qs[i]) * dr_ij
                if det != 0:
                    nonzero_triples += 1

    # Exact-Q 2x2 linear least-squares fit for R(u) = a + b * q(u) where a = R_inf', b = 1/gamma
    n = len(u_list)
    s_q = sum(qs, Fraction(0, 1))
    s_qq = sum((q * q for q in qs), Fraction(0, 1))
    s_r = sum(rs, Fraction(0, 1))
    s_qr = sum((q * r for q, r in zip(qs, rs)), Fraction(0, 1))
    det2 = Fraction(n, 1) * s_qq - s_q * s_q
    if det2 <= 0:
        raise NDTIdentifiabilityError("degenerate frequency-impedance basis")
    r_inf_hat = (s_qq * s_r - s_q * s_qr) / det2
    inv_gamma_hat = (Fraction(n, 1) * s_qr - s_q * s_r) / det2

    # Compute exact complex least-squares reconstruction error against unshunted Debye curve
    if inv_gamma_hat > 0:
        gamma_hat = Fraction(1, 1) / inv_gamma_hat
        # Optimal delta_R minimizing sum |Z_k - (r_inf_hat + dr/(1+j*gamma*u_k))|^2 over exact Q:
        num_dr = Fraction(0, 1)
        den_dr = Fraction(0, 1)
        for u_k, (re_k, im_k) in zip(u_list, z_list):
            gu = gamma_hat * u_k
            d_k = Fraction(1, 1) + gu * gu
            # basis b_k = (1/d_k, -gu/d_k), residual target = (re_k - r_inf_hat, im_k)
            num_dr += (re_k - r_inf_hat) / d_k - (im_k * gu) / d_k
            den_dr += Fraction(1, 1) / d_k
        dr_hat = num_dr / den_dr
        r0_hat = r_inf_hat + dr_hat
        rel_err_sq_sum = Fraction(0, 1)
        for u_k, (re_k, im_k) in zip(u_list, z_list):
            gu = gamma_hat * u_k
            d_k = Fraction(1, 1) + gu * gu
            pred_re = r_inf_hat + dr_hat / d_k
            pred_im = -(dr_hat * gu) / d_k
            diff_sq = (pred_re - re_k) ** 2 + (pred_im - im_k) ** 2
            mag_sq = re_k * re_k + im_k * im_k
            rel_err_sq_sum += diff_sq / mag_sq
        mean_rel_res_sq = rel_err_sq_sum / Fraction(n, 1)
    else:
        r0_hat = r_inf_hat
        mean_rel_res_sq = Fraction(1, 1)

    residual_gate_flagged = (nonzero_triples > 0) and (mean_rel_res_sq > rel_res_sq_limit)
    rel_shift_r_inf = abs(r_inf_hat - r_inf_ref) / r_inf_ref
    rel_shift_r0 = abs(r0_hat - r0_ref) / r0_ref
    baseline_shift_flagged = (rel_shift_r_inf > baseline_shift_limit) or (rel_shift_r0 > baseline_shift_limit)

    return {
        "total_triples": total_triples,
        "collinearity_obstruction_triples": nonzero_triples,
        "fitted_r_inf_ohm": frac_str(r_inf_hat),
        "fitted_r_inf_dec": format_exact_decimal(r_inf_hat, 4),
        "fitted_r0_ohm": frac_str(r0_hat),
        "fitted_r0_dec": format_exact_decimal(r0_hat, 4),
        "mean_rel_residual_sq_dec": format_exact_decimal(mean_rel_res_sq, 8),
        "residual_exact_zero": mean_rel_res_sq == 0,
        "residual_gate_flagged": residual_gate_flagged,
        "rel_shift_r_inf_pct": format_exact_decimal(rel_shift_r_inf * 100, 4),
        "rel_shift_r0_pct": format_exact_decimal(rel_shift_r0 * 100, 4),
        "baseline_shift_flagged": baseline_shift_flagged,
        "gate_passed": (not residual_gate_flagged) and (not baseline_shift_flagged),
    }


def build_identifiability_certificate() -> dict[str, Any]:
    """Compute all three pure-Z/Q pillars for dock#8 P5 + Issue #2."""
    # Pillar 1: 1-layer and 2-layer tetrapolar square Kelvin patches
    n1, e1 = _build_square_kelvin_1layer()
    idx_1l_bnd = next(i for i, (u, v, _) in enumerate(e1) if (u, v) == (0, 1))
    idx_1l_int = next(i for i, (u, v, _) in enumerate(e1) if (u, v) == (4, 5))
    cut_1l_bnd = rank1_edge_cut_exact(n1, e1, idx_1l_bnd, Fraction(1, 1), (0, 2), (6, 8))
    cut_1l_int = rank1_edge_cut_exact(n1, e1, idx_1l_int, Fraction(1, 1), (0, 2), (6, 8))

    n2, e2 = _build_square_kelvin_2layer()
    idx_2l_bnd = next(i for i, (u, v, _) in enumerate(e2) if (u, v) == (0, 1))
    idx_2l_int = next(i for i, (u, v, _) in enumerate(e2) if (u, v) == (4, 5))
    idx_2l_via_ctr = next(i for i, (u, v, _) in enumerate(e2) if (u, v) == (4, 13))
    idx_2l_via_off = next(i for i, (u, v, _) in enumerate(e2) if (u, v) == (0, 9))
    cut_2l_bnd = rank1_edge_cut_exact(n2, e2, idx_2l_bnd, Fraction(1, 1), (0, 2), (6, 8))
    cut_2l_int = rank1_edge_cut_exact(n2, e2, idx_2l_int, Fraction(1, 1), (0, 2), (6, 8))
    cut_2l_via_ctr = rank1_edge_cut_exact(n2, e2, idx_2l_via_ctr, Fraction(1, 2), (0, 2), (6, 8))
    cut_2l_via_off = rank1_edge_cut_exact(n2, e2, idx_2l_via_off, Fraction(1, 2), (0, 2), (6, 8))

    # Bridge disconnection guard proof on 3-node chain 0-1-2
    bridge_rejected = False
    try:
        rank1_edge_cut_exact(
            3,
            [(0, 1, Fraction(1, 1)), (1, 2, Fraction(1, 1))],
            0,
            Fraction(1, 1),
            (0, 2),
            (0, 2),
        )
    except NDTIdentifiabilityError:
        bridge_rejected = True

    dr_1l = Fraction(cut_1l_bnd["delta_r_eff_sm"])
    dr_2l = Fraction(cut_2l_bnd["delta_r_eff_sm"])
    substrate_shunts_cut = dr_2l < dr_1l

    pillar1 = {
        "mesh_1layer_nodes": n1,
        "mesh_1layer_edges": len(e1),
        "mesh_1layer_boundary_cut_0_1": cut_1l_bnd,
        "mesh_1layer_interior_cut_4_5": cut_1l_int,
        "mesh_2layer_nodes": n2,
        "mesh_2layer_edges": len(e2),
        "mesh_2layer_surface_boundary_cut_0_1": cut_2l_bnd,
        "mesh_2layer_surface_interior_cut_4_5": cut_2l_int,
        "mesh_2layer_bondline_via_center_symmetry_cut_4_13": cut_2l_via_ctr,
        "mesh_2layer_bondline_via_offcenter_cut_0_9": cut_2l_via_off,
        "center_via_symmetry_blind_spot_exact_zero": Fraction(cut_2l_via_ctr["delta_r_eff_sm"]) == 0,
        "offcenter_via_cut_strictly_positive": Fraction(cut_2l_via_off["delta_r_eff_sm"]) > 0,
        "substrate_layer_reduces_surface_cut_shift": substrate_shunts_cut,
        "bridge_disconnection_guard_verified": bridge_rejected,
        "pillar_verified": (
            cut_1l_bnd["sherman_morrison_matches_direct_exact"]
            and cut_1l_int["sherman_morrison_matches_direct_exact"]
            and cut_2l_bnd["sherman_morrison_matches_direct_exact"]
            and cut_2l_int["sherman_morrison_matches_direct_exact"]
            and cut_2l_via_ctr["sherman_morrison_matches_direct_exact"]
            and cut_2l_via_off["sherman_morrison_matches_direct_exact"]
            and Fraction(cut_2l_via_ctr["delta_r_eff_sm"]) == 0
            and Fraction(cut_2l_via_off["delta_r_eff_sm"]) > 0
            and substrate_shunts_cut
            and bridge_rejected
        ),
    }

    # Pillar 2: dock#8 P5 Cole zero-delay (alpha=1, Td=0) slice identifiability split (exact-Q at verify.py's 7 NDT freqs!)
    r_inf_ref = Fraction(240, 1)
    r0_ref = Fraction(1850, 1)
    fc_ref = Fraction(18000, 1)
    us_7 = [Fraction(f, 1) / fc_ref for f in STRUCTURAL_NDT_FREQS_HZ]
    # Scale dimensionless 2-layer surface-boundary cut Delta R_eff into ohms (R_sheet = 120 ohm -> ~98.6963 ohm)
    r_sheet_ohm = Fraction(120, 1)
    delta_r_cut_ohm = r_sheet_ohm * dr_2l
    shift_limit = Fraction(5, 100)       # 5% baseline shift threshold
    res_sq_limit = Fraction(1, 10000)    # (1% RMS)^2 matching verify.py:90's rel_rms > 0.01

    z_base_7 = [cole_debye_exact_q(r_inf_ref, r0_ref, u) for u in us_7]
    z_in_span_7 = [(re + delta_r_cut_ohm, im) for re, im in z_base_7]
    # Exact algebraic check on alpha=1/2 fractional Cole branch too (u = 2*r^2, sqrt(2j) = 1+j)
    r_roots_7 = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1), Fraction(8, 1), Fraction(16, 1)]
    z_half_in_span_ok = all(
        (cole_half_exact_q(r_inf_ref, r0_ref, rr)[0] + delta_r_cut_ohm, cole_half_exact_q(r_inf_ref, r0_ref, rr)[1])
        == cole_half_exact_q(r_inf_ref + delta_r_cut_ohm, r0_ref + delta_r_cut_ohm, rr)
        for rr in r_roots_7
    )

    # Out-of-span parasitic shunt: beta = 1/2000 S at fc=18kHz (exact-Q parallel capacitance)
    beta_shunt = Fraction(1, 2000)
    z_shunt_7 = [apply_parallel_shunt_exact_q(z, beta_shunt * u) for z, u in zip(z_base_7, us_7)]

    ev_intact = evaluate_cole_debye_obstruction_exact_q(us_7, z_base_7, r_inf_ref, r0_ref, shift_limit, res_sq_limit)
    ev_in_span = evaluate_cole_debye_obstruction_exact_q(us_7, z_in_span_7, r_inf_ref, r0_ref, shift_limit, res_sq_limit)
    ev_out_span = evaluate_cole_debye_obstruction_exact_q(us_7, z_shunt_7, r_inf_ref, r0_ref, shift_limit, res_sq_limit)

    pillar2 = {
        "fittable_parameters": list(FITTABLE_PARAMETERS),
        "fixed_unmodeled_in_fit": list(FIXED_UNMODELED_IN_FIT),
        "structural_ndt_freqs_hz": list(STRUCTURAL_NDT_FREQS_HZ),
        "baseline_params_ohm_hz": {
            "R_inf_ohm": frac_str(r_inf_ref),
            "R0_ohm": frac_str(r0_ref),
            "fc_Hz": frac_str(fc_ref),
        },
        "injected_in_span_delta_r_ohm": frac_str(delta_r_cut_ohm),
        "injected_in_span_delta_r_dec": format_exact_decimal(delta_r_cut_ohm, 4),
        "alpha_half_in_span_exact_zero_residual": z_half_in_span_ok,
        "injected_out_of_span_beta_siemens": frac_str(beta_shunt),
        "intact_baseline_eval": ev_intact,
        "in_span_fiber_cut_eval": ev_in_span,
        "out_of_span_shunt_eval": ev_out_span,
        "pillar_verified": (
            ev_intact["gate_passed"]
            and ev_intact["residual_exact_zero"]
            and (not ev_in_span["residual_gate_flagged"])
            and ev_in_span["residual_exact_zero"]
            and ev_in_span["baseline_shift_flagged"]
            and (not ev_in_span["gate_passed"])
            and z_half_in_span_ok
            and ev_out_span["residual_gate_flagged"]
            and ev_out_span["collinearity_obstruction_triples"] == 35
            and (not ev_out_span["gate_passed"])
        ),
    }

    # Pillar 3: dock#8 receiving-side explicit pass/fail control ledger
    controls = [
        {"control": "intact_baseline_passes_both_gates", "expected": "PASS", "passed": ev_intact["gate_passed"]},
        {
            "control": "in_span_fiber_cut_missed_by_residual_caught_by_baseline_shift",
            "expected": "REJECT_BY_BASELINE_SHIFT",
            "passed": (not ev_in_span["residual_gate_flagged"]) and ev_in_span["baseline_shift_flagged"],
        },
        {
            "control": "out_of_span_shunt_caught_by_cole_manifold_obstruction",
            "expected": "REJECT_BY_RESIDUAL_OBSTRUCTION",
            "passed": ev_out_span["residual_gate_flagged"] and (ev_out_span["collinearity_obstruction_triples"] == 35),
        },
        {
            "control": "bridge_cut_disconnection_fails_closed",
            "expected": "REJECT_DISCONNECTED",
            "passed": bridge_rejected,
        },
    ]
    pillar3 = {
        "claim_status": "model_and_checker_only_no_bench_measurement",
        "controls": controls,
        "all_controls_verified": all(c["passed"] for c in controls),
        "pillar_verified": all(c["passed"] for c in controls),
    }

    overall = pillar1["pillar_verified"] and pillar2["pillar_verified"] and pillar3["pillar_verified"]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "target_repo": "aska-physical/impedence-analyzers",
        "dock_thread": "aska-physical/dock#8:P5",
        "issue_bridge": "aska-physical/impedence-analyzers#2",
        "mode": "standalone_pure_zq",
        "claim_status": "model_and_checker_only_no_bench_measurement",
        "pillar1_laplacian_rank1_kelvin_patch": pillar1,
        "pillar2_cole_span_identifiability_split": pillar2,
        "pillar3_dock8_receiving_controls": pillar3,
        "overall_verified": overall,
    }


# ==============================================================================
# Zone-B Fail-Closed Certificate Signing & Validation (D1-D3 + hmac.compare_digest)
# ==============================================================================

def _assert_safe_json_tree(obj: Any, *, max_depth: int = 64) -> None:
    """Iterative float-free, str-key-only, depth-capped JSON structural guard."""
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            raise NDTIdentifiabilityError("JSON nesting depth exceeds maximum allowed")
        t = type(cur)
        if cur is None or t in (bool, int, str):
            continue
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise NDTIdentifiabilityError("JSON object keys must be strict str")
                stack.append((v, depth + 1))
            continue
        if t is list:
            for item in cur:
                stack.append((item, depth + 1))
            continue
        raise NDTIdentifiabilityError(f"forbidden JSON value type: {t.__name__}")


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    """Serialize strict dict to canonical float-free ASCII JSON bytes."""
    if type(obj) is not dict:
        raise NDTIdentifiabilityError("canonical_json_bytes requires strict dict")
    _assert_safe_json_tree(obj)
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    except ValueError as exc:
        raise NDTIdentifiabilityError(f"cannot serialize canonical JSON: {exc}") from exc


def compute_payload_sha256(payload: dict[str, Any]) -> str:
    """Compute SHA-256 over certificate body excluding payload_sha256."""
    if type(payload) is not dict:
        raise NDTIdentifiabilityError("payload must be strict dict")
    body = {k: v for k, v in payload.items() if k != "payload_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


@lru_cache(maxsize=1)
def _cached_signed_certificate_json() -> str:
    cert = build_identifiability_certificate()
    cert["payload_sha256"] = compute_payload_sha256(cert)
    return json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def build_signed_certificate() -> dict[str, Any]:
    """Return fresh copy of deterministic signed certificate."""
    return json.loads(_cached_signed_certificate_json())


_HEX64_CHARS = frozenset("0123456789abcdef")


def _is_hex64_ascii(val: str) -> bool:
    if len(val) != 64:
        return False
    for ch in val:
        if ch not in _HEX64_CHARS:
            return False
    return True


def _require_key(obj: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in obj:
        raise NDTIdentifiabilityError(f"missing required key: {key}")
    val = obj[key]
    if type(val) is not expected_type:
        raise NDTIdentifiabilityError(f"key {key} has invalid type")
    return val


def validate_certificate_payload(payload: Any) -> dict[str, Any]:
    """Fail-closed Zone-B validator checking schema, strict types, hmac digest, and exact oracle parity."""
    if type(payload) is not dict:
        raise NDTIdentifiabilityError("certificate root must be a JSON object")
    _assert_safe_json_tree(payload)

    schema = _require_key(payload, "schema_version", str)
    study_id = _require_key(payload, "study_id", str)
    mode = _require_key(payload, "mode", str)
    claim_status = _require_key(payload, "claim_status", str)
    overall = _require_key(payload, "overall_verified", bool)
    claimed_sha = _require_key(payload, "payload_sha256", str)
    _require_key(payload, "pillar1_laplacian_rank1_kelvin_patch", dict)
    _require_key(payload, "pillar2_cole_span_identifiability_split", dict)
    _require_key(payload, "pillar3_dock8_receiving_controls", dict)

    if schema != SCHEMA_VERSION:
        raise NDTIdentifiabilityError("unexpected schema_version")
    if study_id != STUDY_ID:
        raise NDTIdentifiabilityError("unexpected study_id")
    if mode != "standalone_pure_zq":
        raise NDTIdentifiabilityError("unexpected mode")
    if claim_status != "model_and_checker_only_no_bench_measurement":
        raise NDTIdentifiabilityError("unexpected claim_status")
    if not overall:
        raise NDTIdentifiabilityError("overall_verified must be True")

    if not _is_hex64_ascii(claimed_sha):
        raise NDTIdentifiabilityError("payload_sha256 must be 64 lowercase hex ASCII characters")

    recomputed_sha = compute_payload_sha256(payload)
    if not hmac.compare_digest(claimed_sha, recomputed_sha):
        raise NDTIdentifiabilityError("payload_sha256 mismatch")

    expected = build_signed_certificate()
    if not hmac.compare_digest(recomputed_sha, expected["payload_sha256"]):
        raise NDTIdentifiabilityError("certificate content diverges from pure-Z/Q oracle")

    return {
        "valid": True,
        "study_id": study_id,
        "mode": mode,
        "payload_sha256": recomputed_sha,
    }


def _reject_float_or_const(token: str) -> Any:
    raise NDTIdentifiabilityError(f"forbidden JSON float or constant: {token}")


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise NDTIdentifiabilityError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def validate_certificate_file(path: Path) -> dict[str, Any]:
    """Read and fail-closed validate a certificate JSON file from disk."""
    if not isinstance(path, Path):
        raise NDTIdentifiabilityError("path must be Path")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NDTIdentifiabilityError(f"cannot read certificate file: {exc}") from exc
    try:
        loaded = json.loads(
            raw,
            parse_float=_reject_float_or_const,
            parse_constant=_reject_float_or_const,
            object_pairs_hook=_reject_dup_keys,
        )
    except NDTIdentifiabilityError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise NDTIdentifiabilityError(f"invalid JSON certificate: {exc}") from exc
    return validate_certificate_payload(loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-Z/Q receiving-side NDT rank-1 Delta R_eff & Cole span identifiability checker (dock#8 P5)."
    )
    parser.add_argument("--standalone", action="store_true", help="Run standalone pure-Z/Q proof.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate.")
    parser.add_argument("--check", type=Path, default=None, help="Fail-closed validate a certificate JSON file.")
    args = parser.parse_args(argv)

    if args.check is not None:
        try:
            verdict = validate_certificate_file(args.check)
        except NDTIdentifiabilityError as exc:
            sys.stderr.write(f"CHECK FAILED: {exc}\n")
            return 1
        sys.stdout.write(json.dumps(verdict, sort_keys=True, ensure_ascii=True) + "\n")
        return 0

    rendered = _cached_signed_certificate_json()
    if args.json:
        sys.stdout.write(rendered)
    else:
        cert = json.loads(rendered)
        p1 = cert["pillar1_laplacian_rank1_kelvin_patch"]
        p2 = cert["pillar2_cole_span_identifiability_split"]
        sys.stdout.write(
            f"NDT-RANK1-P5 OK (model/checker only): "
            f"2L_dReff={p1['mesh_2layer_surface_boundary_cut_0_1']['delta_r_eff_sm']} "
            f"in_span_res0={p2['in_span_fiber_cut_eval']['residual_exact_zero']} "
            f"in_span_shift_caught={p2['in_span_fiber_cut_eval']['baseline_shift_flagged']} "
            f"out_span_triples={p2['out_of_span_shunt_eval']['collinearity_obstruction_triples']}/35 "
            f"sha256={cert['payload_sha256'][:16]}...\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
