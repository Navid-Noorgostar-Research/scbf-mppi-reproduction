"""Assembled controller variants for experiments V9-V12, and the config -> controller factory.

The three mixins are orthogonal and compose in a fixed order:

    HeadingCostMixin          changes the running cost              (V9)
    CurrentObserverMixin      changes what the controller believes  (V11)
    AdaptiveTemperatureMixin  changes how the samples are weighted  (V12)

With every option at its default each class is exactly the shipped controller: `tests/test_regression.py`
covers the shipped classes, and `selftest_ext.py` checks that the extended classes with default options
reproduce the shipped ones bit for bit as well.

The output filter (V10) is not a mixin: it wraps a finished controller, which is the whole point of the
architecture being compared.
"""
import numpy as np

from .. import solgenia as sg
from .. import harbour as hb
from ..controllers import VesselMPPI, VesselSCBFMPPI, VesselDetMPPI
from ..experiments import DEFAULT
from .costs import HeadingCostMixin
from .observer import CurrentObserverMixin
from .adaptive import AdaptiveTemperatureMixin
from .filters import CBFQPFilter, FilteredController


class MPPIExt(HeadingCostMixin, CurrentObserverMixin, AdaptiveTemperatureMixin, VesselMPPI):
    pass


class SCBFExt(HeadingCostMixin, CurrentObserverMixin, AdaptiveTemperatureMixin, VesselSCBFMPPI):
    pass


class DetExt(HeadingCostMixin, AdaptiveTemperatureMixin, VesselDetMPPI):
    pass


# ------------------------------------------------------------------------------------------------
def make_harbour_ext(cfg):
    a1 = cfg.get("alpha1", DEFAULT["alpha1"])
    a2 = cfg.get("alpha2", DEFAULT["alpha2"])
    return hb.harbour_crossing(a1, a2) if cfg.get("scenario", "static") == "crossing" else hb.harbour_static(a1, a2)


def make_controller_ext(cfg, seed):
    """Build a controller from an extended config dict.  Unknown keys are ignored so a config can be
    reused across experiments."""
    H = make_harbour_ext(cfg)
    kind = cfg["kind"]
    common = dict(K=cfg.get("K", DEFAULT["K"]), T=cfg.get("T", DEFAULT["T"]),
                  lam=cfg.get("lam", DEFAULT["lam"]), s0=tuple(cfg.get("s0", DEFAULT["s0"])), seed=seed)
    ext = dict(w_psi=cfg.get("w_psi", 0.0), w_astern=cfg.get("w_astern", 0.0),
               obs_mode=cfg.get("obs_mode", "off"), obs_gain=cfg.get("obs_gain", 0.2),
               oracle=cfg.get("oracle", False), true_current=tuple(cfg.get("current", (0.0, 0.0))),
               ess_target=cfg.get("ess_target", None))

    base_kind = kind[:-7] if kind.endswith("_filter") else kind
    if base_kind == "mppi":
        c = MPPIExt(H, **common, **ext)
    elif base_kind == "scbf":
        c = SCBFExt(H, delta=cfg.get("delta", DEFAULT["delta"]), form=cfg.get("form", "std"),
                    mode=cfg.get("mode", "hocbf"), alpha=cfg.get("alpha"),
                    is_correction=cfg.get("is_correction", False), **common, **ext)
    elif base_kind == "det":
        common.pop("K")
        e = {k: v for k, v in ext.items() if k not in ("obs_mode", "obs_gain", "oracle", "true_current")}
        c = DetExt(H, iters=cfg.get("iters", 4), shrink=cfg.get("shrink", 0.5), M=cfg.get("M", 125),
                   **common, **e)
    else:
        raise ValueError(kind)

    if kind.endswith("_filter"):
        f = cfg.get("filter", {}) or {}
        filt = CBFQPFilter(H, s0=tuple(cfg.get("s0", DEFAULT["s0"])),
                           delta=f.get("delta", None),
                           disturbance=f.get("disturbance", cfg.get("disturbance", DEFAULT["disturbance"])),
                           s_F=tuple(cfg.get("s_F", DEFAULT["s_F"])),
                           tau_c=cfg.get("tau_c", DEFAULT["tau_c"]),
                           rho=f.get("rho", 1.0e4))
        c = FilteredController(c, filt)
    return c, H


# ------------------------------------------------------------------------------------------------
def barrier_belief(controller):
    """The water current the controller's BARRIER rows are evaluated with (0 for every shipped controller)."""
    inner = getattr(controller, "inner", controller)
    fn = getattr(inner, "_barrier_current", None)
    return tuple(fn()) if fn is not None else (0.0, 0.0)


def observer_history(controller):
    inner = getattr(controller, "inner", controller)
    obs = getattr(inner, "observer", None)
    return np.asarray(obs.history, float) if obs is not None and obs.history else None


class CertificateAudit:
    """Counts, at the plant, how often the controller's own barrier certificate was FALSE.

    At each control instant the applied input tau = B u is checked against the second-order row
    a tau >= b evaluated two ways: with the current the controller believes in, and with the current the
    water actually has.  A "false certificate" is an instant where the controller's row is satisfied and
    the true row is not -- the controller believed it was safe and it was not.  This is the quantity the
    nominal delta = 0.003 has nothing to say about, because delta bounds the effect of the modelled noise,
    not of an unmodelled drift.

    `near_h` restricts a second count to instants where the vessel is within `near_h` metres of a circle;
    far from every obstacle both rows hold trivially and including those instants only dilutes the number.

    IMPORTANT on how to read these counts.  The false-certificate rate is a joint probability, and it is
    only comparable between controllers that operate at the same distance from the obstacles.  A controller
    that is so conservative it never comes near a circle scores zero, not because its certificate is good
    but because it never issues one that could be wrong.  Two further quantities are therefore recorded, and
    they are the ones to compare across configurations because they are properties of the MODEL ERROR and
    not of the operating point:

        hdot_err   |n . (c_believed - c_true)|, the error the unknown current puts into the barrier
                   derivative at the closest obstacle, in m/s.  Compare it with alpha1 h, the other term
                   of psi1: at alpha1 = 0.1 an error of 0.5 m/s is the whole of alpha1 h at h = 5 m.
        margin_err |margin_believed - margin_true| for the second-order row, divided by the norm of the
                   row's coefficients in sampling-std units, i.e. how many standard deviations of thrust
                   the controller is wrong by when it decides whether the row is active.
    """

    def __init__(self, harbour, true_current, near_h=20.0, s0=(350.0, 350.0, 120.0)):
        self.H = harbour
        self.c_true = tuple(float(v) for v in true_current)
        self.near_h = float(near_h)
        self.s0 = np.asarray(s0, float)
        self.n = 0
        self.n_false = 0
        self.n_near = 0
        self.n_false_near = 0
        self.n_bel_ok = 0
        self.n_true_viol = 0
        self.n_bel_viol = 0
        self.c_err = []
        self.hdot_err = []
        self.margin_err = []
        self.psi1_abs = []

    def __call__(self, k, t, x, u, controller):
        c_bel = barrier_belief(controller)
        uc = sg.clip_inputs(np.asarray(u, float)[None], np.asarray(x, float)[None])[0]
        tau = sg.B_ALLOC @ uc
        a_b, b_b, h, _ = self.H.rows(x[None], t, current=(c_bel if (c_bel[0] or c_bel[1]) else None))
        a_t, b_t, _, _ = self.H.rows(x[None], t, current=(self.c_true if (self.c_true[0] or self.c_true[1]) else None))
        m_bel = a_b[0] @ tau - b_b[0]
        m_true = a_t[0] @ tau - b_t[0]
        ok_bel = bool((m_bel >= -1e-9).all())
        ok_true = bool((m_true >= -1e-9).all())
        near = bool(h.min() < self.near_h)
        self.n += 1
        self.n_bel_viol += int(not ok_bel)
        self.n_true_viol += int(not ok_true)
        self.n_bel_ok += int(ok_bel)
        self.n_false += int(ok_bel and not ok_true)
        if near:
            self.n_near += 1
            self.n_false_near += int(ok_bel and not ok_true)
        dc = np.array([c_bel[0] - self.c_true[0], c_bel[1] - self.c_true[1]])
        self.c_err.append(float(np.hypot(dc[0], dc[1])))
        # model error in the barrier derivative at the CLOSEST obstacle: hdot is wrong by exactly n . dc
        j = int(np.argmin(h[0]))
        o = self.H.obs[j]
        rel = np.asarray(x, float)[:2] - o.center(t)
        nrm = rel / max(np.linalg.norm(rel), 1e-9)
        self.hdot_err.append(float(abs(nrm @ dc)))
        self.psi1_abs.append(float(abs(psi1[0][j])))
        # the same error expressed as thrust: how many sampling stds of input the margin is wrong by
        g = np.maximum(np.linalg.norm((a_t[0] @ sg.B_ALLOC) * self.s0[None, :], axis=1), 1e-12)
        self.margin_err.append(float(np.abs((m_bel - m_true) / g).max()))

    def summary(self):
        bel_ok = max(self.n_bel_ok, 1)
        return {
            "cert_instants": self.n,
            "cert_false_frac": self.n_false / max(self.n, 1),
            "cert_false_frac_near": self.n_false_near / max(self.n_near, 1),
            # conditional: of the instants where the controller's own row said safe, how many were not
            "cert_false_given_claimed": self.n_false / bel_ok,
            "cert_near_instants": self.n_near,
            "cert_true_violation_frac": self.n_true_viol / max(self.n, 1),
            "cert_belief_violation_frac": self.n_bel_viol / max(self.n, 1),
            # operating-point-free model error -- the numbers to compare across configurations
            "hdot_err_mean": float(np.mean(self.hdot_err)) if self.hdot_err else None,
            "hdot_err_max": float(np.max(self.hdot_err)) if self.hdot_err else None,
            "psi1_abs_mean": float(np.mean(self.psi1_abs)) if self.psi1_abs else None,
            "margin_err_std_mean": float(np.mean(self.margin_err)) if self.margin_err else None,
            "margin_err_std_max": float(np.max(self.margin_err)) if self.margin_err else None,
            "current_err_mean": float(np.mean(self.c_err)) if self.c_err else None,
            "current_err_final": float(self.c_err[-1]) if self.c_err else None,
        }
