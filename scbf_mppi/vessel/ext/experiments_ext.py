"""Experiments V9-V12: the questions the critique opens, answered with numbers.

    python -m scbf_mppi.vessel.ext.experiments_ext --exp all --runs 30 --procs 12
    python -m scbf_mppi.vessel.ext.experiments_ext --exp V10 --runs 30

  V9   does the missing heading term change any conclusion?          results/vessel_V9_heading.json
  V10  barrier in the sampler or barrier on the output?              results/vessel_V10_filter.json
  V11  what does an unmodelled current do to the certificate?        results/vessel_V11_observer.json
  V12  can a temperature rescue the corrected importance weights?    results/vessel_V12_ess.json

Nothing here re-runs or overwrites a shipped result file.  The regime is the one of the shipped vessel
experiments (see vessel/experiments.py): Solgenia model unchanged, u = (X_AT, Y_AT, F_BT) in newtons,
control interval 1 s, horizon 15 s, K = 500, Sigma0 = diag(350^2, 350^2, 120^2) N^2, lambda = 300 m^2,
environmental force s_F = (120 N, 300 N, 400 N m) with a 5 s correlation time, alpha1 = 0.1, alpha2 = 0.4,
delta = 0.003.

Two scenarios are used throughout, so that every question is asked both of a static field under an
unmodelled drift and of a moving obstacle:
    A  "gust + 0.5 m/s current"  static harbour, coloured (OU) force gust, 0.5 m/s current the
                                 controller does not know about               (the V3 condition)
    B  "crossing ferry"          the group's Experiment-III ferry, R = 25 m at 3 m/s, white force noise
                                 (the V4 condition)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

from .. import solgenia as sg
from ..experiments import DEFAULT, RES, save, load
from ..simulate import run_episode, summarize
from .controllers_ext import make_controller_ext, CertificateAudit, observer_history

# the two scenarios, as keyword blocks that go straight into a config
SCEN_A = dict(scenario="static", disturbance="ou", current=(0.0, -0.5))
SCEN_B = dict(scenario="crossing", disturbance="white", current=(0.0, 0.0))
SCEN = {"A": ("gust + 0.5 m/s current", SCEN_A), "B": ("crossing ferry", SCEN_B)}

DROP = ("traj", "ctrl", "hmin_series", "times", "dist", "ess_series", "_extra_series", "samples")


# ------------------------------------------------------------------------------------------------
def one_run_ext(args):
    cfg, seed = args
    c, H = make_controller_ext(cfg, seed)
    cur = tuple(cfg.get("current", (0.0, 0.0)))
    audit = CertificateAudit(H, cur, near_h=cfg.get("near_h", 20.0)) if cfg.get("audit", False) else None
    extra = tuple(cfg.get("extra_keys", ()))
    r = run_episode(c, H, seed=seed,
                    disturbance=cfg.get("disturbance", DEFAULT["disturbance"]),
                    s_F=np.asarray(cfg.get("s_F", DEFAULT["s_F"]), float) * cfg.get("s_F_mult", 1.0),
                    tau_c=cfg.get("tau_c", DEFAULT["tau_c"]),
                    current=cur,
                    max_time=cfg.get("max_time", DEFAULT["max_time"]),
                    extra_keys=extra, step_hook=audit)
    keep = {k: v for k, v in r.items() if k not in DROP}
    keep["traj_xy"] = r["traj"][:, :2].tolist() if seed < 3 else None
    keep["surge_series"] = r["traj"][1:, 3].tolist() if seed < 2 else None
    keep["hmin_series"] = r["hmin_series"].tolist() if seed == 0 else None
    keep["mean_thrust"] = float(np.linalg.norm(r["ctrl"][:, :2], axis=1).mean()) if len(r["ctrl"]) else None
    if audit is not None:
        keep.update(audit.summary())
    hist = observer_history(c)
    if hist is not None and len(hist):
        err = np.linalg.norm(hist - np.asarray(cur, float)[None, :], axis=1)
        keep["obs_err_final"] = float(err[-1])
        keep["obs_err_mean"] = float(err.mean())
        # first instant after which the estimate stays within 0.1 m/s of the truth
        good = err < 0.1
        keep["obs_settle_s"] = float(np.argmax(np.cumprod(good[::-1])[::-1] > 0)) if good.any() else None
        keep["obs_c_hat_final"] = [float(v) for v in hist[-1]]
        keep["obs_hist"] = hist.tolist() if seed == 0 else None
    return keep


def run_configs_ext(configs, runs, procs=8):
    jobs = [(cfg, s) for cfg in configs for s in range(runs)]
    t = time.time()
    with Pool(procs) as p:
        outs = p.map(one_run_ext, jobs, chunksize=1)
    print(f"  {len(jobs)} episodes in {time.time()-t:.0f}s", file=sys.stderr, flush=True)
    res = {}
    for (cfg, s), o in zip(jobs, outs):
        res.setdefault(cfg["name"], {"cfg": _jsonable(cfg), "runs": []})["runs"].append(o)
    for name in res:
        res[name]["summary"] = summarize(res[name]["runs"], keys=SUMMARY_KEYS)
    return res


SUMMARY_KEYS = ("collision_rate", "collision_rate_ctrl", "ttf", "min_h", "ess_mean", "ess_median",
                "ess_obstacle_mean", "activation_frac_mean", "var_ratio_mean", "sat_active_mean",
                "infeasible_frac_mean", "clip_frac_mean", "effort", "path_length", "mean_speed",
                "unsafe_sample_frac_mean", "multi_violation_frac_mean",
                "astern_frac", "astern_frac_strong", "min_surge",
                "cert_false_frac", "cert_false_frac_near", "cert_false_given_claimed",
                "cert_true_violation_frac", "cert_belief_violation_frac",
                "hdot_err_mean", "hdot_err_max", "psi1_abs_mean",
                "margin_err_std_mean", "margin_err_std_max", "current_err_mean", "current_err_final",
                "obs_err_final", "obs_err_mean", "obs_settle_s",
                "filter_active_mean", "filter_dU_mean", "filter_slack_mean", "filter_tighten_max_mean",
                "filter_failed_mean", "lam_eff_mean", "dU_norm_mean", "ess_target_met_mean")


def _jsonable(cfg):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items() if k != "extra_keys"}


def _touched(res, name):
    """Fraction of seeds whose closest approach went negative."""
    rs = res[name]["runs"]
    return sum(1 for r in rs if r["collided"]), len(rs)


# ------------------------------------------------------------------------------------------------
def V9(runs, procs):
    """Heading-cost ablation.  Does the astern behaviour change any conclusion?

    Three cost variants against three controllers in both scenarios, plus a weight sweep so the answer
    does not rest on one number.  The metric that matters first is astern_frac: if the term does not
    actually stop the vessel going backwards, the ablation says nothing.
    """
    base = [("MPPI", dict(kind="mppi")),
            ("SCBF-MPPI", dict(kind="scbf")),
            ("SCBF-MPPI + IS", dict(kind="scbf", is_correction=True))]
    variants = [("no heading term", dict()),
                ("heading to goal w=400", dict(w_psi=400.0)),
                ("astern penalty w=2000", dict(w_astern=2000.0))]
    cfgs = []
    for sk, (slabel, sk_kw) in SCEN.items():
        for cname, ckw in base:
            for vname, vkw in variants:
                cfgs.append(dict(name=f"{sk} {slabel} | {cname} | {vname}", **ckw, **sk_kw, **vkw))
    # weight sweep, scenario A only
    for cname, ckw in base:
        for w in (100.0, 1600.0):
            cfgs.append(dict(name=f"A gust + 0.5 m/s current | {cname} | heading to goal w={int(w)}",
                             **ckw, **SCEN_A, w_psi=w))
    res = run_configs_ext(cfgs, runs, procs)
    res["_note"] = ("The heading term is w_psi (1 - cos(psi - bearing to goal)), gated off inside the goal "
                    "radius; the astern term is w_astern max(0, -u_surge)^2.  Neither changes the obstacle "
                    "penalty, the barrier, or the sampling covariance.")
    save("vessel_V9_heading", res)
    return res


def V10(runs, procs):
    """Barrier in the sampler versus barrier on the output.

    Six architectures, same MPPI, same barrier, same seeds.  The filter variants pay one convex solve per
    control interval; the sampler variants pay K x T = 7500 per interval.
    """
    arch = [
        ("MPPI (no barrier)", dict(kind="mppi")),
        ("SCBF-MPPI (barrier in the sampler)", dict(kind="scbf")),
        ("SCBF-MPPI + IS", dict(kind="scbf", is_correction=True)),
        ("MPPI + CBF-QP filter", dict(kind="mppi_filter", filter=dict(delta=None))),
        ("MPPI + chance CBF-QP filter", dict(kind="mppi_filter", filter=dict(delta=0.003))),
        ("SCBF-MPPI + CBF-QP filter", dict(kind="scbf_filter", filter=dict(delta=None))),
    ]
    ek = ("filter_active", "filter_dU", "filter_slack", "filter_tighten_max", "filter_failed")
    cfgs = []
    for sk, (slabel, sk_kw) in SCEN.items():
        for aname, akw in arch:
            cfgs.append(dict(name=f"{sk} {slabel} | {aname}", **akw, **sk_kw, extra_keys=ek))
    res = run_configs_ext(cfgs, runs, procs)
    res["_timing"] = V10_timing()
    res["_note"] = ("The filter solves min ||u - u_mppi||^2_W + rho 1's subject to the same second-order rows, "
                    "the 700 N azimuth disk and the speed-dependent bow cap, with W = diag(1/s0^2).  The chance "
                    "variant tightens each row by z sqrt(a Sigma_d a^T) with a in tau coordinates; Sigma_d is "
                    "diag(s_F^2) for the coloured gust and diag((s_F sqrt(2 tau_c / h))^2) for white force "
                    "noise, a factor sqrt(10) larger at tau_c = 5 s, h = 1 s.")
    save("vessel_V10_filter", res)
    return res


def V10_timing():
    """Median wall-clock per control cycle for each architecture, measured alone on this machine."""
    out = {}
    for label, cfg in [("MPPI K=500", dict(kind="mppi", **SCEN_B)),
                       ("SCBF-MPPI (sampler)", dict(kind="scbf", **SCEN_B)),
                       ("SCBF-MPPI + IS", dict(kind="scbf", is_correction=True, **SCEN_B)),
                       ("MPPI + CBF-QP filter", dict(kind="mppi_filter", filter=dict(delta=None), **SCEN_B)),
                       ("MPPI + chance CBF-QP filter", dict(kind="mppi_filter", filter=dict(delta=0.003), **SCEN_B))]:
        c, H = make_controller_ext(dict(name="t", **cfg), 0)
        x = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
        ts = []
        for k in range(25):
            t0 = time.time()
            u = c.plan(x)
            dt_ = time.time() - t0
            if k >= 5:
                ts.append(dt_)
            x = sg.step(x[None], u[None], c.dt)[0]
        out[label] = float(np.median(ts))
    out["control interval s"] = sg.DT_CTRL
    return out


def V11(runs, procs):
    """An unmodelled current versus the chance constraint's nominal delta, and what an observer buys.

    Every configuration audits the plant: at each control instant the applied input is checked against the
    second-order row computed with the controller's belief about the current and with the truth.  A "false
    certificate" is an instant where the belief said safe and the truth said unsafe.  delta = 0.003 has
    nothing to say about that number, which is the point.
    """
    ek = ()
    cfgs = []
    for cname, ckw in [("SCBF-MPPI", dict(kind="scbf")), ("MPPI", dict(kind="mppi"))]:
        for oname, okw in [("no estimate", dict(obs_mode="off")),
                           ("estimate in the rollouts", dict(obs_mode="rollouts")),
                           ("estimate in the barrier", dict(obs_mode="barrier")),
                           ("estimate in both", dict(obs_mode="both")),
                           ("true current known (oracle)", dict(obs_mode="both", oracle=True))]:
            if cname == "MPPI" and oname == "estimate in the barrier":
                continue                                   # MPPI has no barrier rows to correct
            cfgs.append(dict(name=f"{cname} | {oname}", **ckw, **SCEN_A, **okw, audit=True, extra_keys=ek))
    for g in (0.05, 0.5):
        cfgs.append(dict(name=f"SCBF-MPPI | estimate in both, gain {g}", kind="scbf", **SCEN_A,
                         obs_mode="both", obs_gain=g, audit=True, extra_keys=ek))
    # a stronger current, to show the effect scales with the mismatch rather than with the tuning
    strong = dict(SCEN_A); strong["current"] = (0.0, -1.0)
    for oname, okw in [("no estimate", dict(obs_mode="off")), ("estimate in both", dict(obs_mode="both"))]:
        cfgs.append(dict(name=f"SCBF-MPPI | 1.0 m/s current | {oname}", kind="scbf", **strong, **okw,
                         audit=True, extra_keys=ek))
    res = run_configs_ext(cfgs, runs, procs)
    res["_note"] = (
        "cert_false_frac counts control instants where the row evaluated with the controller's believed "
        "current was satisfied and the row evaluated with the true current was not.  It is a JOINT "
        "probability and is only comparable between controllers that operate at the same distance from the "
        "obstacles: a controller so conservative that it never approaches a circle scores zero because it "
        "never issues a certificate that could be wrong, not because its certificate is good.  The numbers "
        "to compare across configurations are hdot_err_mean -- the error |n . (c_believed - c_true)| that "
        "the unknown current puts into the barrier derivative, against psi1_abs_mean for scale -- and "
        "margin_err_std_mean, the same error expressed as sampling standard deviations of thrust.  Those "
        "are properties of the model error alone.  The observer is a trapezoidal kinematic residual with "
        "gain 0.2 unless stated.")
    save("vessel_V11_observer", res)
    return res


def V12(runs, procs):
    """Can an on-line temperature rescue the corrected importance weights?

    The honest reading requires three numbers together: the effective sample size, the norm of the control
    update, and the fraction of episodes that reach the goal.  ESS that rises while the update shrinks and
    the goal is missed is not a rescue.
    """
    ek = ("lam_eff", "dU_norm", "ess_target_met")
    rows = [("SCBF-MPPI + IS, fixed lambda = 300", dict(kind="scbf", is_correction=True))]
    for tgt in (50, 100, 250):
        rows.append((f"SCBF-MPPI + IS, ESS target {tgt}",
                     dict(kind="scbf", is_correction=True, ess_target=float(tgt))))
    rows += [("SCBF-MPPI (no IS), fixed lambda = 300", dict(kind="scbf")),
             ("SCBF-MPPI (no IS), ESS target 100", dict(kind="scbf", ess_target=100.0)),
             ("MPPI, fixed lambda = 300", dict(kind="mppi")),
             ("MPPI, ESS target 100", dict(kind="mppi", ess_target=100.0))]
    cfgs = []
    for sk, (slabel, sk_kw) in SCEN.items():
        for rname, rkw in rows:
            cfgs.append(dict(name=f"{sk} {slabel} | {rname}", **rkw, **sk_kw, extra_keys=ek))
    res = run_configs_ext(cfgs, runs, procs)
    res["_note"] = ("lambda is retuned each cycle so that ESS(lambda) hits the target, using "
                    "log w(lambda) = -J/lambda - C + log q with J the state cost and C the lambda-free part "
                    "of the control term, so the temperature stays consistent in the cost and in the weights. "
                    "ess_target_met_mean is the fraction of cycles where the target was reachable at all: "
                    "as lambda grows the weights tend to softmax(log q - C), whose ESS is a ceiling the "
                    "temperature cannot raise.")
    save("vessel_V12_ess", res)
    return res


# ------------------------------------------------------------------------------------------------
def summary_text():
    """A plain-text digest of whichever of V9-V12 exist, written beside the JSON."""
    lines = []
    for tag, fname, keys in [
        ("V9  heading-cost ablation", "vessel_V9_heading",
         ("astern_frac", "astern_frac_strong", "collision_rate", "ttf", "min_h", "path_length")),
        ("V10 sampler vs output filter", "vessel_V10_filter",
         ("collision_rate", "ttf", "min_h", "mean_speed", "path_length", "filter_active_mean", "filter_dU_mean")),
        ("V11 unmodelled current", "vessel_V11_observer",
         ("cert_false_frac", "cert_false_frac_near", "collision_rate", "min_h", "obs_err_final", "obs_settle_s")),
        ("V12 ESS-targeted temperature", "vessel_V12_ess",
         ("ess_median", "ess_mean", "dU_norm_mean", "lam_eff_mean", "ttf", "collision_rate")),
    ]:
        path = os.path.join(RES, fname + ".json")
        if not os.path.exists(path):
            continue
        res = load(fname)
        lines.append(f"== {tag}")
        for name, blk in res.items():
            if name.startswith("_"):
                continue
            s = blk["summary"]
            touched = sum(1 for r in blk["runs"] if r["collided"])
            n = len(blk["runs"])
            bits = [f"n={n}", f"touched {touched}/{n}", f"reached {s['reached_frac']:.2f}"]
            for k in keys:
                if k in s:
                    bits.append(f"{k} {s[k]['mean']:.4g}")
            lines.append(f"  {name:<66s} " + "  ".join(bits))
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "vessel_ext_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    todo = a.exp.split(",") if a.exp != "all" else ["V9", "V10", "V11", "V12"]
    for e in todo:
        t = time.time()
        print(f"== {e}", file=sys.stderr, flush=True)
        if e == "V9":
            V9(a.runs, a.procs)
        elif e == "V10":
            V10(a.runs, a.procs)
        elif e == "V11":
            V11(a.runs, a.procs)
        elif e == "V12":
            V12(a.runs, a.procs)
        elif e == "summary":
            print(summary_text())
        print(f"   {e} done in {time.time()-t:.0f}s", file=sys.stderr, flush=True)
    if a.exp != "summary":
        summary_text()
