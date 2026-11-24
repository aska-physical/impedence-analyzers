# NDT Rank-1 Conductance-Laplacian $\Delta R_{\text{eff}}$ & Cole 5-Parameter Span Identifiability Study (`dock#8` Item `P5` + Issue `#2`)

**Claim Status:** `model` and receiving-side `checker` only — **no bench measurement is attached** (per `aska-physical/dock#8`).  
**Target Repository:** `aska-physical/impedence-analyzers` (strictly add-only under `studies/ndt-rank1-identifiability-audit/`; zero edits to `verify.py`)  
**Upstream Threads:** `aska-physical/dock#8` (Thread `P5`), `aska-physical/impedence-analyzers#2` (2-layer epoxy-over-concrete square Kelvin patch)

---

## 1. Exact Scope (`dock#8` Thread `P5`)

> *"`P5`: On the impedance checker, name the parameters a fit can move. Show one injected defect inside that span that the residual misses, and one outside it that the residual catches."*

In `verify.py:40-99` (`fit_cole_with_artifact_gate`), the Levenberg–Marquardt solver moves **5 free parameters** while holding `C_stray_pF = 0.0` (`verify.py:58`):

- **Fittable parameters (`5`):** `R_inf_ohm`, `R0_ohm`, `fc_Hz`, `alpha`, `Td_ns`
- **Fixed / unmodeled in fit (`2`):** `C_stray_pF`, `hook_lead_ns`

The residual + delay gate (`verify.py:90`) flags an artifact iff `Td_ns < -1.5`, `Td_ns > 10.0`, or `rel_rms > 0.01` (`rel_rms_sq > 1/10000`). Because `R_inf_ohm` and `R0_ohm` are free fit parameters without an intact-baseline reference check, a pure resistive fiber/bond cut shifts $Z(\omega) \mapsto Z(\omega) + \Delta R_{\text{eff}}$ **strictly inside the Cole span** with **exact zero Cole residual (`0/1`)** (holding on both $\alpha = 1$ and $\alpha = 1/2$ with $T_d = 0$, and across the full 5-parameter family by shifting $R_\infty$ and $R_0$ identically). Conversely, on the **zero-delay ($\alpha = 1$, $T_d = 0$) unshunted Cole slice** (`ndt_rank1_identifiability_audit.py:289`), an unmodeled parallel bondline/cable shunt capacitance $Y_p(\omega) = j\omega C_{\text{stray}}$ lies **strictly outside the unshunted zero-delay Cole manifold** and triggers an exact algebraic collinearity obstruction (`35/35` triples nonzero). Note on full 5-parameter scope: when the delay parameter $T_d$ is also freed in a nonlinear numerical fit, the phase-delay term can absorb most of the shunt phase and reduce the numerical residual to $\approx 0.003\%$; the exact algebraic obstruction certified here applies strictly to the zero-delay ($\alpha = 1, T_d = 0$) slice.

---

## 2. Pure-$\mathbb{Z}/\mathbb{Q}$ Exact Model & Checker Results (Zero Floats, Zero Bare Asserts)

All calculations in `ndt_rank1_identifiability_audit.py` use pure Python 3 stdlib (`int` and `fractions.Fraction` only; zero NumPy, zero floats, zero bare `assert`s; deterministic across `-I -B`, `-I -O -B`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -I -B`).

### Pillar 1 — Exact-$\mathbb{Q}$ Conductance-Laplacian Rank-1 Fiber/Bond Cut $\Delta R_{\text{eff}}$ (1-Layer vs. 2-Layer Square Kelvin Patch, Issue `#2`)

For a connected conductance mesh $G = (V, E, g)$ with reduced Laplacian $L_g$ and edge $e = (u, v)$ losing conductance $\Delta g_e \in (0, g_e]$ (provided $1 - \Delta g_e\, b_e^\top L_g^+ b_e > 0$), the Sherman–Morrison rank-1 shift

$$\Delta R_{\text{eff}}(s,t; e, \Delta g_e) = \frac{\Delta g_e \bigl((e_s - e_t)^\top L_g^+ b_e\bigr)^2}{1 - \Delta g_e\, b_e^\top L_g^+ b_e} \in \mathbb{Q}_{\ge 0}$$

and tetrapolar 4-wire transfer shift $\Delta R_{4\text{w}}(s,t; p,q; e, \Delta g_e)$ match independent direct post-cut Laplacian re-solves bit-for-bit (`sherman_morrison_matches_direct_exact: true`) across drive pair `(0, 2)` and sense pair `(6, 8)`:

| Patch Model (`g_top=1`, `g_sub=1/4`, `g_via=1/2`) | Cut Edge `e` (`delta_g`) | $R_{\text{eff,base}}$ | Exact $\Delta R_{\text{eff}}$ (`dec`, `%` shift) | Exact $\Delta R_{4\text{w}}$ | Model Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 1-layer `3×3` (`9` nodes, `12` edges) | Boundary `(0, 1)` (`1`) | `5/4` | `75/56` (`1.339286`, `+107.1429%`) | `15/56` | Isolated surface baseline |
| 1-layer `3×3` (`9` nodes, `12` edges) | Interior `(4, 5)` (`1`) | `5/4` | `3/20` (`0.150000`, `+12.0000%`) | `3/20` | `8.93×` smaller than boundary cut |
| 2-layer epoxy-over-concrete (`18` nodes, `33` edges) | Surface boundary `(0, 1)` (`1`) | `895/819` | `149791675/182124306` (`0.822469`, `+75.2628%`) | `27949955/182124306` | Concrete substrate shunts current (`0.822 < 1.339`) |
| 2-layer epoxy-over-concrete (`18` nodes, `33` edges) | Surface interior `(4, 5)` (`1`) | `895/819` | `202521/2192008` (`0.092391`, `+8.4545%`) | `202521/2192008` | Interior sensitivity damped by substrate |
| 2-layer epoxy-over-concrete (`18` nodes, `33` edges) | Center via `(4, 13)` (`1/2`) | `895/819` | **`0/1` (`0.000000`, `0.0000%`)** | **`0/1`** | Exact mirror-symmetry blind spot ($\phi_4 = \phi_{13}$; requires $90^\circ$ rotated excitation) |
| 2-layer epoxy-over-concrete (`18` nodes, `33` edges) | Off-center via `(0, 9)` (`1/2`) | `895/819` | `1687675/24404562` (`0.069154`, `+6.3282%`) | `35530/12202281` | Detectable off-axis bondline delamination |

A 3-node bridge-cut negative control (`0-1-2`, cutting `(0, 1)` with $\Delta g = 1$) triggers $1 - \Delta g\, b_e^\top L_g^+ b_e = 0$ and fails closed cleanly.

### Pillar 2 — Exact-$\mathbb{Q}$ Zero-Delay Cole Slice Identifiability Split (`dock#8` `P5`)

Evaluated over exact $\mathbb{Q}$ on the **zero-delay ($\alpha = 1$, $T_d = 0$) unshunted Cole slice** (`cole_debye_exact_q`, line `289`) at `verify.py:128`'s 7 structural NDT frequencies `[5, 12, 28, 50, 110, 250, 600]` kHz and `verify.py:142`'s baseline $(R_\infty, R_0, f_c) = (240\,\Omega,\, 1850\,\Omega,\, 18\,\text{kHz})$ (baseline-shift threshold `5%`, residual threshold `rel_rms_sq > 1/10000` matching `verify.py:90`'s `1%` RMS gate). The exact-$\mathbb{Q}$ `mean_rel_res_sq` column reports the **two-stage linear fit** (`ndt_rank1_identifiability_audit.py:374`: fitting $(R_\infty', 1/\gamma)$ on $\operatorname{Re}(Z)$ via the exact $2\times 2$ normal equations first, then recovering $R_0'$), not an unconstrained nonlinear global least-squares minimum (independent unconstrained numerical fits reach $\approx 0.0396$ on the zero-delay slice and $\approx 0.00389$ with all 5 parameters free):

| Case | Defect Injected | Exact Collinearity Obstruction (`C(7,3) = 35` triples, $T_d = 0$ slice) | Exact-$\mathbb{Q}$ Staged Linear-Fit `mean_rel_res_sq` | Residual Gate | Baseline Shift Gate (`R_inf` / `R0`) | Combined Checker Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Intact baseline** | None | `0 / 35` | `0.00000000` (`0/1`) | Pass | `0.0000%` / `0.0000%` (Pass) | **`PASS`** |
| **In-span fiber cut** | $+\Delta R_{\text{eff}} = 120 \cdot \frac{149791675}{182124306} \approx +98.6963\,\Omega$ | **`0 / 35`** (also exact `0/1` on $\alpha = 1/2$ at $u = 2r^2$) | **`0.00000000` (`0/1`)** | **MISSES** (`False`) | **`+41.1235%` / `+5.3349%` (CATCHES)** | **`REJECT` (`baseline_shift`)** |
| **Out-of-span parasitic shunt** | Parallel $Y_p(u) = j \beta u$, $\beta = 1/2000\,\text{S}$ | **`35 / 35` strictly nonzero** | **`0.09396878`** ($\approx 30.65\%$ RMS $\gg 1\%$, staged fit; unconstrained slice $\approx 0.0396$) | **CATCHES** (`True`) | `-81.1736%` / `-0.3300%` | **`REJECT` (`cole_manifold_obstruction`)** |

---

## 3. Reproduce & Verify (Works Under `-I -O -B`)

```bash
# 1. Run standalone pure-Z/Q proof & print canonical certificate
python3 -I -B studies/ndt-rank1-identifiability-audit/ndt_rank1_identifiability_audit.py --standalone --json

# 2. Fail-closed Zone-B verification of the committed certificate (works under -O)
python3 -I -O -B studies/ndt-rank1-identifiability-audit/ndt_rank1_identifiability_audit.py \
  --check studies/ndt-rank1-identifiability-audit/ndt_rank1_identifiability_certificate.json

# 3. Run assertion-free R5 test suite (zero bare asserts; 100% mutation + BoolOp kill)
python3 -I -O -B -W error studies/ndt-rank1-identifiability-audit/test_ndt_rank1_identifiability_r5.py
```
