"""
nsm_traj.py
===========

Parameterized thermodynamic trajectories for r-process nucleosynthesis in
neutron-star-merger ejecta, with tools to calibrate them against Lagrangian
tracer particles from GRHD/GRMHD simulations and to emit WinNet-readable
trajectory files.

Design philosophy
-----------------
The nucleosynthesis is controlled by the state of the fluid element *at the
moment NSE freezes out* (T0 ~ 6-10 GK) plus the rate at which the density
subsequently drops.  Everything earlier is erased by NSE.  The model therefore
carries exactly three free parameters per trajectory,

    theta = (Ye0, s0, tau)

evaluated at T = T0, plus two optional structural parameters:

    n     : deceleration/geometry index of the density law (n = 3 -> spherical
            homologous; n -> inf -> pure exponential)
    t_hom : time at which the profile is forced onto rho ~ t^-3

and, if neutrino irradiation is switched on, (L_nue, L_nuebar, E_nue, E_nuebar)
plus the radius history r(t).

References for the functional forms
-----------------------------------
LR15   Lippuner & Roberts 2015, ApJ 815, 82   (arXiv:1508.03133)  -- Eq. (1)
K12    Korobkin et al. 2012, MNRAS 426, 1940  (arXiv:1206.2379)   -- Sec. 3.1
R18    Radice et al. 2018, ApJ 869, 130       (arXiv:1809.11161)  -- tau def.
KAR25  Kuske, Arcones & Reichert 2025, ApJ    (arXiv:2506.00092)  -- Eqs. (1),(7)
HWQ97  Hoffman, Woosley & Qian 1997, ApJ 482, 951                 -- s^3/tau
QW96   Qian & Woosley 1996, ApJ 471, 331                          -- nu rates
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares, brentq

# ----------------------------------------------------------------------------
# physical constants (cgs unless stated)
# ----------------------------------------------------------------------------
C_LIGHT = 2.99792458e10          # cm / s
K_B = 1.380649e-16               # erg / K
MEV = 1.602176634e-6             # erg
N_A = 6.02214076e23              # 1 / g
A_RAD = 7.5657e-15               # erg cm^-3 K^-4
M_U = 1.66053907e-24             # g
GK = 1.0e9                       # K
SIGMA_0 = 1.76e-44               # cm^2
G_A = 1.2723
DELTA_NP = 1.2933                # MeV, m_n - m_p
M_E_C2 = 0.51100                 # MeV


# ============================================================================
# 1. DENSITY PARAMETERIZATIONS
# ============================================================================

def rho_lr15(t, rho0, tau):
    """Lippuner & Roberts (2015) Eq. (1); also KAR25 Eq. (1).

    rho = rho0 exp(-t/tau)          t <= 3 tau
    rho = rho0 (3 tau / (e t))^3     t >= 3 tau

    C^1 continuous at t = 3 tau by construction.  tau is the *e-folding time of
    the density at t = 0*, i.e. tau = -(d ln rho / dt)^-1 |_{t=0}.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    early = t <= 3.0 * tau
    out[early] = rho0 * np.exp(-t[early] / tau)
    late = ~early
    out[late] = rho0 * (3.0 * tau / (np.e * t[late])) ** 3
    return out


def rho_powerlaw(t, rho0, tau, n=3.0):
    """Single-branch generalized homologous law.

        rho(t) = rho0 [1 + t/(n tau)]^{-n}

    Properties
    ----------
    * -(d ln rho/dt)|_0 = 1/tau  for every n (same tau convention as LR15).
    * n = 3  : exact for a uniform sphere in free expansion, r(t) = r0(1 + t/3tau),
               v = r0/(3 tau) = const.  Asymptotes to rho ~ t^-3.
    * n = 2  : cylindrical (relevant for equatorial tidal tails before they
               become spherical).
    * n -> inf : recovers rho0 exp(-t/tau), i.e. the early LR15 branch.

    Unlike LR15 this has no kink and no e^3 ~ 20 amplitude offset between the
    early and late branches, which matters if you extrapolate to seconds.
    """
    t = np.asarray(t, dtype=float)
    return rho0 * (1.0 + t / (n * tau)) ** (-n)


def rho_freefall_homologous(t, rho_fin, t_fin):
    """Korobkin et al. (2012) extrapolation: rho = rho_fin (t/t_fin)^-3.

    This is the standard *extrapolation* used past the end of the hydro run,
    not a full-trajectory parameterization.  It is only valid once the fluid
    element has reached terminal velocity (see `homology_diagnostic`).
    """
    t = np.asarray(t, dtype=float)
    return rho_fin * (t / t_fin) ** (-3.0)


def rho_two_stage(t, rho0, tau, tau2, t_break, n=3.0):
    """Piecewise two-timescale profile for shocked / re-accelerated elements.

    Fast initial decompression (tau) followed by a slower phase (tau2) after
    t_break -- e.g. an element that is shocked, decompresses, then coasts in a
    slowly expanding tail, or a disk outflow that stalls.  Continuity in rho is
    enforced; the derivative is intentionally discontinuous, since the physical
    situation it models (a shock) is too.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    a = t <= t_break
    out[a] = rho_powerlaw(t[a], rho0, tau, n)
    rho_b = rho_powerlaw(np.array([t_break]), rho0, tau, n)[0]
    out[~a] = rho_powerlaw(t[~a] - t_break, rho_b, tau2, n)
    return out


DENSITY_MODELS = {
    "lr15": lambda t, p: rho_lr15(t, p["rho0"], p["tau"]),
    "powerlaw": lambda t, p: rho_powerlaw(t, p["rho0"], p["tau"], p.get("n", 3.0)),
    "two_stage": lambda t, p: rho_two_stage(
        t, p["rho0"], p["tau"], p["tau2"], p["t_break"], p.get("n", 3.0)),
}


# ============================================================================
# 2. INITIAL DENSITY FROM (s0, Ye0, T0)
# ============================================================================

def entropy_approx(rho, T, Ye, Abar):
    """Approximate specific entropy [k_B / baryon]: ideal ions + radiation.

    Ions: Sackur-Tetrode, in the form quoted by KAR25 Eqs. (2)-(3).
    Radiation: s_rad = 4 a T^3 / (3 rho) in erg/(g K) -> convert to k_B/baryon.

    NOTE
    ----
    This is a *diagnostic / bootstrapping* approximation only.  It omits
    electron-positron pairs (important for s > 30 k_B/baryon at T ~ 7 GK),
    nuclear internal partition functions, and Coulomb corrections.  For
    production work take rho0 from the same NSE + EOS solver the network uses
    (Timmes & Arnett 1999 in WinNet; Timmes & Swesty 2000 in SkyNet), so that
    the trajectory and the network are thermodynamically consistent.  Use this
    function only to bracket the root before handing off.
    """
    eta = np.log(N_A * (6.62607015e-27) ** 3 / (2 * np.pi * M_U) ** 1.5
                 * rho / ((K_B * T) ** 1.5 * Abar ** 2.5))
    s_ion = (1.0 / Abar) * (2.5 - eta)                       # k_B / baryon
    s_rad = (4.0 * A_RAD * T ** 3 / (3.0 * rho)) / (K_B * N_A)  # k_B / baryon
    return s_ion + s_rad


def rho0_from_entropy(s0, Ye0, T0=7.0 * GK, Abar=None,
                      bracket=(1e2, 1e14)):
    """Invert s(rho, T0, Ye0) = s0 for rho0.

    `Abar` is the NSE-averaged mass number (including free nucleons), i.e.
    Abar = 1 / sum_i Y_i.  If None, a crude Ye-dependent guess is used.
    Replace with a real NSE call for anything quantitative -- the mapping
    (s0, Ye0) -> rho0 is the single largest systematic in this whole scheme
    (KAR25 Fig. 2; rho0 spans 7e5 to 1.4e12 g/cm^3 in LR15).
    """
    if Abar is None:
        # crude: neutron-dominated at low Ye, alpha/iron-group at high Ye
        Abar = 1.0 + 3.0 * np.clip(Ye0 / 0.5, 0, 1) ** 2 * (1.0 + 10.0 * Ye0)
    f = lambda lr: entropy_approx(10.0 ** lr, T0, Ye0, Abar) - s0
    lo, hi = np.log10(bracket[0]), np.log10(bracket[1])
    if f(lo) * f(hi) > 0:
        raise ValueError("rho0 not bracketed; widen `bracket` or fix Abar")
    return 10.0 ** brentq(f, lo, hi, xtol=1e-8)


# ============================================================================
# 3. KINEMATICS AND CONSISTENCY DIAGNOSTICS
# ============================================================================

def radius_from_rho(t, rho, r0, n=3.0):
    """Radius history consistent with baryon-number conservation for a
    self-similar element: rho r^n = const  =>  r(t) = r0 (rho0/rho)^{1/n}.

    Needed for neutrino irradiation (flux ~ 1/r^2) and for the causality check.
    """
    rho = np.asarray(rho, dtype=float)
    return r0 * (rho[0] / rho) ** (1.0 / n)


def homology_diagnostic(t, rho, r0, n=3.0, r_measured=None):
    """Return (v(t), dlnv/dlnt, t_hom) where t_hom is the time after which
    |d ln v / d ln t| < 0.05, i.e. the onset of genuine homology.

    Physical check: free expansion (rho ~ t^-3) is only legitimate once the
    specific enthalpy excess has been converted to kinetic energy,
        (h - 1) c^2  <<  v^2 / 2 .
    For s ~ 10-30 k_B/baryon and T ~ 3 GK the thermal energy is ~ few MeV per
    baryon while v = 0.2 c gives ~ 19 MeV per baryon, so homology is reached
    within a few ms -- but this must be verified per trajectory, not assumed.

    IMPORTANT: pass `r_measured` = the tracer's actual r(t) when testing a
    *simulation* trajectory.  If you let the radius be reconstructed from rho
    via baryon conservation, the test is circular: the reconstruction already
    assumes the self-similar law whose validity you are trying to check.  On a
    parameterized trajectory this function therefore only verifies internal
    consistency and the causality bound, not homology itself.
    """
    r = np.asarray(r_measured, float) if r_measured is not None \
        else radius_from_rho(t, rho, r0, n)
    v = np.gradient(r, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnv = np.gradient(np.log(np.abs(v) + 1e-300), np.log(t + 1e-300))
    idx = np.where(np.abs(dlnv) < 0.05)[0]
    t_hom = t[idx[0]] if len(idx) else np.nan
    return v, dlnv, t_hom


def causality_ok(v, v_max=0.9 * C_LIGHT):
    """Hard filter: reject trajectories whose implied velocity is superluminal
    or unphysically fast.  Extrapolating the LR15 law to late times without
    this check produces v > c (LR15 note this explicitly in their Sec. 3.1)."""
    return np.all(np.abs(v) < v_max)


def tau_from_bulk(r_ext, v_ext, rho_ext, rho0):
    """Radice et al. (2018) style tau, derived rather than fitted.

    Extrapolate the tracer homologously from the extraction sphere,
        rho(t) = rho_ext (t / t_ext)^-3,   t_ext = r_ext / v_ext,
    ask when it reaches the NSE density rho0 = rho0(s, Ye, T0),
        t_* = t_ext (rho_ext / rho0)^{1/3},
    and identify tau with the local e-folding time there, tau = t_*/3:

        tau = (r_ext / (3 v_ext)) (rho_ext / rho0)^{1/3}.

    This is the cheapest possible calibration: it needs only the four numbers
    (r_ext, v_ext, rho_ext, rho0) that every published outflow catalogue
    reports, so it can be applied to Radice et al. (2018)-style released
    histograms without access to full tracer time series.
    """
    return (r_ext / (3.0 * v_ext)) * (rho_ext / rho0) ** (1.0 / 3.0)


# ============================================================================
# 4. CALIBRATION AGAINST TRACER DATA
# ============================================================================

def fit_tau(t_trace, rho_trace, model="lr15", t0=None, n_fixed=3.0,
            fit_n=False, weights=None, rho_floor=1e2):
    """Fit a density parameterization to one tracer, in log-density space.

    Fitting in log rho is essential: the density drops by 8-12 decades and a
    linear-space L2 loss would be dominated entirely by the first few points.

    Parameters
    ----------
    t_trace, rho_trace : arrays, tracer time [s] and rest-mass density [g/cm^3]
    t0                 : time origin.  If None, the *last* time the tracer is
                         above T0 should be passed in; the trajectory is
                         re-zeroed there so that tau has the LR15 meaning.
    fit_n              : if True, also fit the deceleration index n (only for
                         model='powerlaw').

    Returns
    -------
    dict with keys rho0, tau, (n), plus 'rms_dex' = RMS residual in dex and
    'cost'.  Use rms_dex as the primary goodness-of-fit statistic; < 0.1 dex
    is an excellent match, > 0.3 dex usually signals a shock (non-monotonic
    rho) that no smooth parameterization can capture (KAR25 Fig. 11).
    """
    t = np.asarray(t_trace, float)
    rho = np.asarray(rho_trace, float)
    if t0 is not None:
        m = t >= t0
        t, rho = t[m] - t0, rho[m]
    good = rho > rho_floor
    t, rho = t[good], rho[good]
    y = np.log10(rho)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)[good]

    rho0_guess = rho[0]
    # tau guess from the initial logarithmic derivative
    if len(t) > 3:
        slope = np.polyfit(t[:max(3, len(t) // 10)],
                           np.log(rho[:max(3, len(t) // 10)]), 1)[0]
        tau_guess = -1.0 / slope if slope < 0 else 1e-3
    else:
        tau_guess = 1e-3

    def resid(p):
        d = {"rho0": 10.0 ** p[0], "tau": np.exp(p[1])}
        if model == "powerlaw":
            d["n"] = np.exp(p[2]) if fit_n else n_fixed
        return w * (np.log10(DENSITY_MODELS[model](t, d)) - y)

    p0 = [np.log10(rho0_guess), np.log(max(tau_guess, 1e-6))]
    if model == "powerlaw" and fit_n:
        p0.append(np.log(n_fixed))
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.2)

    out = {"rho0": 10.0 ** sol.x[0], "tau": float(np.exp(sol.x[1]))}
    if model == "powerlaw":
        out["n"] = float(np.exp(sol.x[2])) if fit_n else n_fixed
    out["rms_dex"] = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    out["cost"] = float(sol.cost)
    out["n_points"] = int(len(t))
    return out


def state_at_T0(t_trace, T_trace, s_trace, ye_trace, rho_trace, T0=7.0 * GK):
    """Interpolate (t0, rho0, s0, Ye0) at the *last* downward crossing of T0.

    'Last' matters: shock-reheated elements cross T0 several times, and the
    physically relevant NSE exit is the final one (KAR25 Sec. 4).  Returns
    None if the tracer never gets above T0 (a genuinely cold tidal element --
    for those, NSE was never re-established and the initial composition must
    come from the cold NS, not from NSE; see K12 discussion).
    """
    T = np.asarray(T_trace, float)
    t = np.asarray(t_trace, float)
    above = T > T0
    if not above.any():
        return None
    i = np.where(above)[0][-1]
    if i + 1 >= len(T):
        return None
    f = (T[i] - T0) / (T[i] - T[i + 1])
    lin = lambda a: a[i] + f * (a[i + 1] - a[i])
    return {"t0": float(lin(t)), "rho0": float(np.exp(lin(np.log(rho_trace)))),
            "s0": float(lin(np.asarray(s_trace, float))),
            "Ye0": float(lin(np.asarray(ye_trace, float))),
            "n_crossings": int(np.sum(np.diff(above.astype(int)) != 0))}


def abundance_distance(X1, X2, A_max=250, floor=1e-7):
    """KAR25 Eq. (7): mean |log X| difference over mass number.

        d_ij = (1/A_max) sum_A |log X_i(A) - log X_j(A)|

    X1, X2 are arrays of mass fractions indexed by A (1..A_max).  Only bins
    where both exceed `floor` are counted.  Benchmarks from KAR25: the mean
    tracer-vs-model distance over ~20 hydro models is d = 0.23 (+0.15 -0.18),
    i.e. a typical agreement of a quarter dex per mass bin.  Treat d < 0.3 as
    a successful reproduction, d > 0.5 as a failure needing investigation.
    """
    X1 = np.asarray(X1, float)[:A_max]
    X2 = np.asarray(X2, float)[:A_max]
    m = (X1 > floor) & (X2 > floor)
    if not m.any():
        return np.inf
    return float(np.mean(np.abs(np.log10(X1[m]) - np.log10(X2[m])))) 


def mass_weighted_histogram(values, masses, bins):
    """Mass-weighted 1D histogram -- the correct way to compare a tracer
    ensemble to a parameterized grid.  Individual trajectories are not
    comparable one-to-one; only the mass-weighted *distributions* of
    (Ye0, s0, tau) and of the final Y(A) are."""
    h, edges = np.histogram(values, bins=bins, weights=masses)
    return h / np.sum(masses), edges


# ============================================================================
# 5. NEUTRINO IRRADIATION (optional)
# ============================================================================

def sigma_nue_n(E_nu_MeV):
    """nu_e + n -> p + e^-  cross section, allowed approximation.
        sigma = (1 + 3 g_A^2)/4 * sigma_0 * ((E + Delta)/m_e c^2)^2
    ~ 9.6e-44 cm^2 (E/MeV)^2 for E >> Delta."""
    return 0.25 * (1.0 + 3.0 * G_A ** 2) * SIGMA_0 * \
        ((E_nu_MeV + DELTA_NP) / M_E_C2) ** 2


def sigma_nuebar_p(E_nu_MeV):
    """nubar_e + p -> n + e^+ (threshold Delta + m_e ignored above ~2 MeV)."""
    return 0.25 * (1.0 + 3.0 * G_A ** 2) * SIGMA_0 * \
        (np.maximum(E_nu_MeV - DELTA_NP, 0.0) / M_E_C2) ** 2


def nu_capture_rates(r_cm, L_nue, L_nuebar, E_nue, E_nuebar,
                     eps_ratio=1.2, R_nu=3.0e6):
    """Per-nucleon neutrino capture rates [1/s].

        lambda = (L_nu / (4 pi r^2 <E_nu>)) * <sigma> * Phi_geo

    L in erg/s, <E> in MeV.  `eps_ratio` = <E^2>/<E>^2 for the assumed spectrum
    (1.2 for a Fermi-Dirac with eta ~ 3); the capture cross sections scale as
    E^2, so <sigma> ~ sigma(<E>) * eps_ratio.  Phi_geo is the flux-dilution
    correction for a finite neutrinosphere of radius R_nu.

    Typical BNS remnant values within the first ~10 ms (Cusinato et al. 2022;
    Perego et al. 2017; Rosswog & Liebendoerfer 2003):
        L_nue ~ 0.5-2e53, L_nuebar ~ 1-4e53 erg/s  (L_nuebar / L_nue ~ 2-3)
        <E_nue> ~ 8-12 MeV, <E_nuebar> ~ 13-19 MeV, <E_nux> ~ 18-25 MeV
    """
    r = np.asarray(r_cm, float)
    phi = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - (R_nu / np.maximum(r, R_nu)) ** 2, 0.0)))
    phi = np.maximum(phi, 1e-12) * 2.0  # normalized so phi -> (R_nu/2r)^2 far away
    flux_nue = L_nue / (4 * np.pi * r ** 2 * E_nue * MEV)
    flux_nueb = L_nuebar / (4 * np.pi * r ** 2 * E_nuebar * MEV)
    lam_nue = flux_nue * sigma_nue_n(E_nue) * eps_ratio
    lam_nueb = flux_nueb * sigma_nuebar_p(E_nuebar) * eps_ratio
    return lam_nue, lam_nueb


def ye_equilibrium(lam_nue, lam_nuebar):
    """Weak (neutrino-only) equilibrium electron fraction,
        Ye_eq = lambda_nue / (lambda_nue + lambda_nuebar),
    which is *independent of radius* because both rates scale as 1/r^2.  This
    is why an irradiated outflow forgets its initial Ye and why Ye0 cannot be
    varied independently of (L_nue/L_nuebar, <E_nue>, <E_nuebar>) once the
    exposure integral exceeds unity.
    """
    return lam_nue / (lam_nue + lam_nuebar)


def nu_exposure(t, lam_nue, lam_nuebar):
    """Dimensionless exposure X = int (lam_nue + lam_nuebar) dt.
    X << 1  -> Ye frozen at its hydro value (tidal ejecta: valid).
    X >~ 1  -> Ye driven to Ye_eq (neutrino-driven wind: valid).
    0.1 < X < 10 is the awkward regime where Ye0 is genuinely a free parameter
    only in the sense of 'unknown', not 'independent'."""
    return np.trapezoid(np.asarray(lam_nue) + np.asarray(lam_nuebar), t)


# ============================================================================
# 6. PHYSICAL-ADMISSIBILITY FILTERS FOR THE PARAMETER SPACE
# ============================================================================

@dataclass
class ParameterBox:
    """Prior ranges for (Ye0, s0, tau) by ejecta component.

    Defaults are the mass-weighted ranges reported in the literature; see
    KAR25 Table 3 for the tracer-derived means/1-sigma per hydro model.
    """
    ye: tuple = (0.01, 0.50)
    s: tuple = (1.0, 290.0)          # k_B / baryon
    tau: tuple = (0.25e-3, 600e-3)   # s

    @staticmethod
    def tidal():
        return ParameterBox(ye=(0.02, 0.10), s=(0.5, 5.0), tau=(0.2e-3, 2e-3))

    @staticmethod
    def shocked_dynamical():
        return ParameterBox(ye=(0.05, 0.45), s=(5.0, 150.0), tau=(0.5e-3, 10e-3))

    @staticmethod
    def nu_driven_wind():
        return ParameterBox(ye=(0.25, 0.45), s=(10.0, 30.0), tau=(5e-3, 30e-3))

    @staticmethod
    def disk_wind():
        return ParameterBox(ye=(0.10, 0.40), s=(8.0, 40.0), tau=(20e-3, 200e-3))


def neutron_to_seed_proxy(s0, tau, Ye0):
    """Scaling proxy for Y_n / Y_seed (HWQ97).

    Seed nuclei are assembled by the alpha-process, whose rate is dominated by
    the (effectively) three-body bottleneck, so d Y_seed/dt ~ rho^2.  At fixed
    temperature and in the radiation-dominated regime rho ~ T^3 / s, hence

        Y_seed ~ rho^2 tau ~ tau / s^3   =>   Y_n / Y_seed ~ (1 - 2 Ye) s^3 / tau.

    Use this only to *order* the parameter space and to define iso-nucleosynthesis
    surfaces; it fails badly in the ion-dominated corner (s < 30 k_B/baryon and
    Ye < 0.35), where NSE already contains heavy nuclei and Y_seed is set by the
    EOS rather than by the alpha-process (KAR25 Fig. 8).
    """
    return np.maximum(1.0 - 2.0 * np.asarray(Ye0, float), 1e-6) * \
        np.asarray(s0, float) ** 3 / np.asarray(tau, float)


def admissible(Ye0, s0, tau, rho0, r0, v_term_max=0.8 * C_LIGHT,
               rho0_max=1e12, tau_min=0.2e-3):
    """Boolean admissibility mask encoding the hard physical constraints.

    1. Causality / energetics: v = r0 / (3 tau) must be sub-luminal and
       consistent with the Bernoulli-unbound velocities seen in GRHD
       (v_inf <~ 0.3 c for the bulk, tail to 0.6-0.8 c).
    2. EOS validity: rho0 <~ 1e12 g/cm^3, above which the Timmes EOS omits
       nuclear contributions and WinNet's heating treatment is switched off
       anyway (KAR25 Sec. 2.2).
    3. Numerical-resolution floor on tau: GRHD tracer sets do not resolve
       tau < ~0.5 ms (Radice et al. 2016 explicitly report no material below
       this), so smaller tau is an extrapolation, not a calibration.
    """
    v = r0 / (3.0 * np.asarray(tau, float))
    ok = (v < v_term_max)
    ok &= (np.asarray(rho0, float) < rho0_max)
    ok &= (np.asarray(tau, float) > tau_min)
    ok &= (np.asarray(Ye0, float) > 0.0) & (np.asarray(Ye0, float) < 0.5)
    ok &= (np.asarray(s0, float) > 0.0)
    return ok


def sample_correlated(n, mean_log, cov_log, seed=0):
    """Draw (Ye0, log10 s0, log10 tau) from a multivariate lognormal fitted to
    the tracer ensemble.  Fit `mean_log`/`cov_log` with np.cov on the tracer
    values -- this preserves the s-Ye correlation (shock-heated material is
    both hotter and more weak-processed; Radice et al. 2016 Fig. 10) and the
    tau-s anticorrelation, which a naive independent grid destroys.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean_log, cov_log, size=n)


# ============================================================================
# 7. WINNET OUTPUT
# ============================================================================

def write_winnet_trajectory(path, t, T9, rho, Ye=None, r_cm=None,
                            L_nue=None, L_nuebar=None,
                            E_nue=None, E_nuebar=None, header_note=""):
    """Write a whitespace-separated trajectory file.

    Columns:  time[s]  radius[cm]  temperature[GK]  density[g/cm^3]  [Ye]
              [Lnue Lnuebar Enue Enuebar]

    WinNet's `trajectory_format` runtime parameter lets you declare exactly
    this column layout, e.g.

        trajectory_format = "time:s, radius:cm, temp:GK, dens:g/cm3, ye"

    Check the column-name tokens against your WinNet version's
    `parameter.md` / `read_trajectory` routine before running -- the accepted
    unit strings have changed between releases.

    If you use WinNet's new parametric mode (KAR25) you do NOT need this file
    at all: you supply (Ye0, s0, T0) and a rho(t) function directly, and let
    the EOS produce T(t) self-consistently with nuclear heating.  Writing a
    fixed T(t) here means the network cannot feed heating back into the
    temperature, which suppresses the well-known ~1 GK plateau (K12 Fig. 7).
    """
    t = np.asarray(t, float)
    ncol = 4 + (Ye is not None) + 4 * (L_nue is not None)
    cols = [t, np.asarray(r_cm if r_cm is not None else np.zeros_like(t), float),
            np.asarray(T9, float), np.asarray(rho, float)]
    names = ["time[s]", "radius[cm]", "T[GK]", "rho[g/cm3]"]
    if Ye is not None:
        cols.append(np.asarray(Ye, float) * np.ones_like(t)); names.append("Ye")
    if L_nue is not None:
        for a, nm in ((L_nue, "Lnue[erg/s]"), (L_nuebar, "Lnuebar[erg/s]"),
                      (E_nue, "Enue[MeV]"), (E_nuebar, "Enuebar[MeV]")):
            cols.append(np.asarray(a, float) * np.ones_like(t)); names.append(nm)
    M = np.column_stack(cols)
    hdr = f"{header_note}\n" + "  ".join(names)
    np.savetxt(path, M, header=hdr, fmt="%18.10e")
    return path


# ============================================================================
# 8. END-TO-END DRIVER
# ============================================================================

@dataclass
class Trajectory:
    Ye0: float
    s0: float
    tau: float
    rho0: float
    n: float = 3.0
    model: str = "lr15"
    r0: float = 1.0e7          # cm, radius at NSE exit
    t: np.ndarray = field(default=None, repr=False)
    rho: np.ndarray = field(default=None, repr=False)

    def build(self, t_end=100.0, npts=4000, t_start=1e-6):
        self.t = np.geomspace(t_start, t_end, npts)
        p = {"rho0": self.rho0, "tau": self.tau, "n": self.n}
        self.rho = DENSITY_MODELS[self.model](self.t, p)
        return self

    def kinematics(self):
        v, dlnv, t_hom = homology_diagnostic(self.t, self.rho, self.r0, self.n)
        return {"v_terminal_c": float(v[-1] / C_LIGHT), "t_hom_s": t_hom,
                "causal": bool(causality_ok(v))}

    def nu_history(self, L_nue, L_nuebar, E_nue, E_nuebar):
        r = radius_from_rho(self.t, self.rho, self.r0, self.n)
        ln, lnb = nu_capture_rates(r, L_nue, L_nuebar, E_nue, E_nuebar)
        return {"r": r, "lam_nue": ln, "lam_nuebar": lnb,
                "Ye_eq": float(ye_equilibrium(ln[0], lnb[0])),
                "exposure": float(nu_exposure(self.t, ln, lnb))}


def build_grid(box: ParameterBox, n_ye=20, n_s=20, n_tau=8, T0=7.0 * GK,
               r0=1.0e7, apply_filters=True):
    """Generate a filtered (Ye0, s0, tau, rho0) grid ready for the network."""
    ye = np.linspace(*box.ye, n_ye)
    s = np.geomspace(*box.s, n_s)
    tau = np.geomspace(*box.tau, n_tau)
    Y, S, TAU = np.meshgrid(ye, s, tau, indexing="ij")
    R0 = np.empty_like(Y)
    for idx in np.ndindex(Y.shape):
        try:
            R0[idx] = rho0_from_entropy(S[idx], Y[idx], T0)
        except ValueError:
            R0[idx] = np.nan
    mask = np.isfinite(R0)
    if apply_filters:
        mask &= admissible(Y, S, TAU, R0, r0)
    return {"Ye0": Y[mask], "s0": S[mask], "tau": TAU[mask], "rho0": R0[mask],
            "n_total": Y.size, "n_kept": int(mask.sum())}


if __name__ == "__main__":
    # --- smoke test / worked example -------------------------------------
    print("== single trajectory, shocked dynamical ejecta ==")
    rho0 = rho0_from_entropy(s0=20.0, Ye0=0.20, T0=7.0 * GK)
    tr = Trajectory(Ye0=0.20, s0=20.0, tau=4.0e-3, rho0=rho0).build()
    print(f"  rho0        = {rho0:.3e} g/cm^3")
    print(f"  kinematics  = {tr.kinematics()}")
    print(f"  nu          = { {k: v for k, v in tr.nu_history(1e53, 2.5e53, 10.0, 15.0).items() if not hasattr(v,'__len__')} }")

    print("\n== recover tau from a synthetic 'tracer' ==")
    t_syn = np.geomspace(1e-5, 0.05, 300)
    rho_syn = rho_lr15(t_syn, 1e9, 3.0e-3) * (1 + 0.05 * np.random.default_rng(1).normal(size=t_syn.size))
    fit = fit_tau(t_syn, rho_syn, model="lr15")
    print(f"  injected tau = 3.000e-03 s   recovered = {fit['tau']:.3e} s"
          f"   rms = {fit['rms_dex']:.3f} dex")

    print("\n== derived-tau shortcut (Radice-style) ==")
    print(f"  tau = {tau_from_bulk(4.43e7, 0.2*C_LIGHT, 1e6, 1e9)*1e3:.2f} ms")

    print("\n== filtered grid, shocked-dynamical box ==")
    g = build_grid(ParameterBox.shocked_dynamical(), n_ye=12, n_s=12, n_tau=6)
    print(f"  kept {g['n_kept']} / {g['n_total']} grid points")

    write_winnet_trajectory("example_traj.dat", tr.t,
                            np.full_like(tr.t, 7.0), tr.rho, Ye=0.20,
                            r_cm=radius_from_rho(tr.t, tr.rho, tr.r0),
                            header_note="Ye0=0.20 s0=20 kB/bar tau=4 ms (LR15 profile)")
    print("\n  wrote example_traj.dat")