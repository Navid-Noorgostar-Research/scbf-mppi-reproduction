"""Experiments E1..E12.  Usage:
    python -m scbf_mppi.experiments --exp all            # everything (about 20-30 min on two cores)
    python -m scbf_mppi.experiments --exp E1 --runs 10
Results go to results/*.json, figures to figures/*.png (paths relative to the package root).
"""
import argparse, json, os, time, sys
import platform
import numpy as np
from multiprocessing import Pool
from .env import Corridor
from .mppi import MPPI, SCBFMPPI, DeterministicMPPI
from .simulate import run_episode, summarize
from . import scbf as scbf_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# calibrated regime (see README): the paper does not report Sigma0; sigma = I3 as stated is incompatible with Table I
DEFAULT = dict(sigma_v=1.0, sigma_om=1.5, sigma_env=0.1, T=20, lam=1.0, delta=0.003)

def make_controller(cfg, seed):
    env = Corridor(printed=cfg.get("printed", False))
    kind = cfg["kind"]
    common = dict(K=cfg.get("K", 500), T=cfg.get("T", DEFAULT["T"]), lam=cfg.get("lam", DEFAULT["lam"]),
                  sigma_v=cfg.get("sigma_v", DEFAULT["sigma_v"]), sigma_om=cfg.get("sigma_om", DEFAULT["sigma_om"]),
                  seed=seed, u_max=(None if cfg.get("u_max") is None else np.asarray(cfg.get("u_max"), float)))
    if kind == "mppi":
        c = MPPI(env, **common)
    elif kind == "scbf":
        c = SCBFMPPI(env, sigma_env=cfg.get("sigma_env", DEFAULT["sigma_env"]), delta=cfg.get("delta", DEFAULT["delta"]),
                     form=cfg.get("form", "variance"), mode=cfg.get("mode", "both"), alpha=cfg.get("alpha"),
                     is_correction=cfg.get("is_correction", False), **common)
    elif kind == "det":
        common.pop("K")
        c = DeterministicMPPI(env, iters=cfg.get("iters", 4), shrink=cfg.get("shrink", 0.5), M=cfg.get("M", 125), **common)
    else:
        raise ValueError(kind)
    return c, env

def one_run(args):
    cfg, seed = args
    c, env = make_controller(cfg, seed)
    r = run_episode(c, env, sigma=cfg.get("sigma_env", DEFAULT["sigma_env"]), seed=seed,
                    max_steps=cfg.get("max_steps", 250), bridge=cfg.get("bridge", True))
    # keep light: drop big arrays except what E5 needs
    keep = {k: v for k, v in r.items() if k not in ("traj", "ctrl", "hmin_series", "ess_series")}
    keep["v_mean"] = float(np.mean(r["ctrl"][:, 0])) if len(r["ctrl"]) else None
    keep["v_abs_mean"] = float(np.mean(np.abs(r["ctrl"][:, 0]))) if len(r["ctrl"]) else None
    # speed vs distance-to-wall bins for E5
    hm = r["hmin_series"][:-1]; v = np.abs(r["ctrl"][:, 0]); om = np.abs(r["ctrl"][:, 1])
    bins = [-1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 2.0]
    idx = np.digitize(hm, bins) - 1
    keep["v_by_h"] = [float(v[idx == i].mean()) if (idx == i).any() else None for i in range(len(bins) - 1)]
    keep["om_by_h"] = [float(om[idx == i].mean()) if (idx == i).any() else None for i in range(len(bins) - 1)]
    keep["h_bins"] = bins
    keep["traj_xy"] = r["traj"][:, :2].tolist() if seed == 0 else None
    return keep

def run_configs(configs, runs, procs=2):
    jobs = [(cfg, s) for cfg in configs for s in range(runs)]
    t = time.time()
    with Pool(procs) as p:
        outs = p.map(one_run, jobs)
    print(f"  {len(jobs)} episodes in {time.time()-t:.0f}s", file=sys.stderr)
    res = {}
    for (cfg, s), o in zip(jobs, outs):
        res.setdefault(cfg["name"], {"cfg": cfg, "runs": []})["runs"].append(o)
    for name in res:
        res[name]["summary"] = summarize(res[name]["runs"])
    return res

def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        txt = json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        f.write(txt.replace("Infinity", "\"inf\"").replace("NaN", "null"))

def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)

# ---------------------------------------------------------------------------------------------
def E1(runs):
    """Table I / Figure 1 reproduction, with the two constraint encodings."""
    cfgs = [dict(name=f"MPPI K={K}", kind="mppi", K=K) for K in (200, 500, 1000, 4000, 5000)]
    cfgs += [dict(name=f"SCBF-var K={K}", kind="scbf", form="variance", K=K) for K in (200, 300, 400, 500)]
    cfgs += [dict(name=f"SCBF-std K={K}", kind="scbf", form="std", K=K) for K in (200, 500)]
    res = run_configs(cfgs, runs); save("E1_table1", res); return res

def E2(runs):
    """Collision rate against the process-noise level; the paper states sigma = I."""
    cfgs = []
    for sg in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        cfgs.append(dict(name=f"MPPI sigma={sg}", kind="mppi", K=500, sigma_env=sg))
        cfgs.append(dict(name=f"SCBF-var sigma={sg}", kind="scbf", form="variance", K=500, sigma_env=sg))
    res = run_configs(cfgs, runs)
    # free-diffusion reference: a point at the corridor centre with no control, 130 steps
    rng = np.random.default_rng(0); env = Corridor()
    ref = {}
    for sg in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        X = np.zeros((2000, 3)); X[:, 1] = 0.5
        out = np.zeros(2000, bool)
        for k in range(130):
            X = X + sg * np.sqrt(0.05) * rng.standard_normal(X.shape)
            out |= ~env.inside(X)
        ref[str(sg)] = float(out.mean())
    res["_free_diffusion_exit_by_130_steps"] = ref
    save("E2_sigma_sweep", res); return res

def E3():
    """Wall-clock: the literal SDP (8) per sample vs the closed-form vectorised solve vs an MPPI cycle."""
    import cvxpy as cp
    rng = np.random.default_rng(0); env = Corridor()
    z = scbf_mod.quantile(DEFAULT["delta"]); s0, som = DEFAULT["sigma_v"], DEFAULT["sigma_om"]
    P0 = np.diag([s0, som]); mu0 = np.zeros(2)
    # realistic instances from states along the corridor
    X = np.stack([rng.uniform(0, 4, 300), np.zeros(300), rng.uniform(-1, 1, 300)], -1)
    X[:, 1] = env.w(X[:, 0]) + rng.uniform(0.05, 0.95, 300)
    c, b = env.scbf_rows(X, DEFAULT["sigma_env"]); ubar = rng.uniform(0, 1.5, 300)
    times = {"sdp_literal_clarabel": [], "socp_std_clarabel": [], "socp_var_clarabel": [], "sdp_literal_solver_only": []}
    n_ok = 0
    for i in range(60):
        A = [[c[i, 0], 0.0], [c[i, 1], 0.0]]; bb = [b[i, 0], b[i, 1]]; ub = np.array([ubar[i], 0.0])
        t = time.perf_counter(); st, *_ = scbf_mod.solve_sdp_literal(ub, mu0, P0, A, bb, alpha=z); times["sdp_literal_clarabel"].append(time.perf_counter() - t)
        times["sdp_literal_solver_only"].append(scbf_mod.LAST_SOLVE_TIME or 0.0)
        t = time.perf_counter(); scbf_mod.solve_exact_cvxpy(ub, mu0, P0, A, bb, form="std", z=z); times["socp_std_clarabel"].append(time.perf_counter() - t)
        t = time.perf_counter(); scbf_mod.solve_exact_cvxpy(ub, mu0, P0, A, bb, form="variance", alpha=z); times["socp_var_clarabel"].append(time.perf_counter() - t)
        n_ok += st in ("optimal", "optimal_inaccurate")
    # vectorised closed form, batch of K = 500 (one timestep of Algorithm 1)
    K = 500
    Xb = np.repeat(X[:1], K, 0); cb, bbb = env.scbf_rows(Xb, DEFAULT["sigma_env"])
    t = time.perf_counter()
    for _ in range(20):
        scbf_mod.solve_rows(np.full(K, 0.8), cb, bbb, s0, z=z, form="std")
    t_vec = (time.perf_counter() - t) / 20
    # MPPI and SCBF-MPPI cycle times
    c5000 = MPPI(env, K=5000, seed=0, sigma_v=s0, sigma_om=som)
    t = time.perf_counter(); [c5000.plan(np.array([0.0, 0.5, 0.0])) for _ in range(10)]; t_mppi5000 = (time.perf_counter() - t) / 10
    c500 = MPPI(env, K=500, seed=0, sigma_v=s0, sigma_om=som)
    t = time.perf_counter(); [c500.plan(np.array([0.0, 0.5, 0.0])) for _ in range(10)]; t_mppi500 = (time.perf_counter() - t) / 10
    s500 = SCBFMPPI(env, K=500, seed=0, form="variance", sigma_env=DEFAULT["sigma_env"], sigma_v=s0, sigma_om=som)
    t = time.perf_counter(); [s500.plan(np.array([0.0, 0.5, 0.0])) for _ in range(5)]; t_scbf500 = (time.perf_counter() - t) / 5
    med = {k: float(np.median(v)) for k, v in times.items()}
    out = {"per_problem_median_s": med, "n_sdp_ok": n_ok,
           "closed_form_batch500_s": t_vec,
           "algorithm1_cycle_K500_T20_with_literal_SDP_s": 500 * 20 * med["sdp_literal_clarabel"],
           "algorithm1_cycle_K500_T20_solver_only_s": 500 * 20 * med["sdp_literal_solver_only"],
           "algorithm1_cycle_K200_T20_with_literal_SDP_s": 200 * 20 * med["sdp_literal_clarabel"],
           "algorithm1_cycle_K500_T20_with_SOCP_s": 500 * 20 * med["socp_std_clarabel"],
           "scbf_mppi_cycle_K500_closed_form_s": t_scbf500,
           "mppi_cycle_K500_s": t_mppi500, "mppi_cycle_K5000_s": t_mppi5000,
           "machine": platform.machine(), "cpu_count": os.cpu_count(), "note": "single core, cvxpy + Clarabel, numpy vectorised rollouts"}
    save("E3_wallclock", out); return out

def E4():
    """Theorem 2: delivered probability under the printed (variance) constraint as a function of projected std."""
    z = scbf_mod.quantile(DEFAULT["delta"])
    from scipy.stats import norm
    s = np.linspace(0.02, 2.0, 200)
    # printed constraint tight:  A mu - b = alpha * s^2  ->  Pr(Au >= b) = Phi((A mu - b)/s) = Phi(alpha s)
    pr_var = norm.cdf(z * s)
    pr_std = np.full_like(s, 1 - DEFAULT["delta"])
    ex = {"Amu": 1.0, "ASigmaAT": 0.03, "b": 0.85, "z": 3.0}
    sproj = np.sqrt(ex["ASigmaAT"])
    ex["printed_lhs"] = ex["Amu"] - ex["z"] * ex["ASigmaAT"]; ex["correct_lhs"] = ex["Amu"] - ex["z"] * sproj
    ex["actual_pr"] = float(norm.cdf((ex["Amu"] - ex["b"]) / sproj))
    out = {"s_proj": s.tolist(), "pr_printed": pr_var.tolist(), "pr_correct": pr_std.tolist(), "z": z, "worked_example": ex}
    save("E4_theorem2", out); return out

def E6():
    """Sample-count arithmetic from measured quantities (Table II cannot be reproduced without them)."""
    env = Corridor(); s0, som = DEFAULT["sigma_v"], DEFAULT["sigma_om"]
    out = {}
    for name, ctor in [("MPPI", lambda: MPPI(env, K=500, seed=0, sigma_v=s0, sigma_om=som)),
                       ("SCBF-var", lambda: SCBFMPPI(env, K=500, seed=0, form="variance", sigma_env=DEFAULT["sigma_env"], sigma_v=s0, sigma_om=som)),
                       ("SCBF-std", lambda: SCBFMPPI(env, K=500, seed=0, form="std", sigma_env=DEFAULT["sigma_env"], sigma_v=s0, sigma_om=som))]:
        c = ctor(); x = np.array([0.0, 0.5, 0.0]); rng = np.random.default_rng(5)
        E1_sum, E1_mean, varv, varo, ES = [], [], [], [], []
        for k in range(50):
            u = c.plan(x); w_unnorm = np.exp(-c.last["S"] / c.lam)
            E1_sum.append(float(w_unnorm.sum())); E1_mean.append(float(w_unnorm.mean())); ES.append(float(c.last["S"].mean()))
            x = x + np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]]) * 0.05 + DEFAULT["sigma_env"] * np.sqrt(0.05) * rng.standard_normal(3)
        # realised perturbation variance: use the sampler once more
        eps, _, _ = c.sample_eps(x)
        varv.append(float(eps[:, :, 0].var())); varo.append(float(eps[:, :, 1].var()))
        e1s, e1m = np.mean(E1_sum), np.mean(E1_mean)
        eps1, eps2, rho2 = 0.05, 0.1, 0.1
        def N2(var, E1):
            d = E1 - eps1
            return float(4 * var / (rho2 * eps2 ** 2 * d ** 2)) if d > 0 else float("inf")
        out[name] = {"E1_as_sum_K500": e1s, "E1_as_mean": e1m, "Var_dv": np.mean(varv), "Var_dom": np.mean(varo),
                     "E[S]": float(np.mean(ES)),
                     "N2_with_sum_and_Var_dv": N2(np.mean(varv), e1s), "N2_with_mean_and_Var_dv": N2(np.mean(varv), e1m),
                     "N2_with_sum_and_Var_dom": N2(np.mean(varo), e1s), "N2_with_mean_and_Var_dom": N2(np.mean(varo), e1m),
                     "exp_2ES_over_lambda": float(np.exp(min(2 * np.mean(ES), 700)))}
    out["N1_paper_constant"] = float(-np.log(0.05 / 2) / 0.05 ** 2)
    out["N1_standard_hoeffding"] = float(np.log(2 / 0.05) / (2 * 0.05 ** 2))
    out["paper_table_II"] = {"MPPI": {"N1": 1476, "N2": 2973}, "SCBF-MPPI": {"N1": 1476, "N2": 584}}
    save("E6_samplecount", out); return out

def E8(runs):
    """Mechanism separation at an equal rollout budget of 500 per cycle."""
    cfgs = [dict(name="MPPI K=500", kind="mppi", K=500),
            dict(name="MPPI K=500, Sigma x0.25 (beta=0.5)", kind="mppi", K=500, sigma_v=0.5, sigma_om=0.75),
            dict(name="Deterministic MPPI 4x125 (nu=0.5)", kind="det", iters=4, shrink=0.5, M=125),
            dict(name="SCBF-var K=500 (both)", kind="scbf", form="variance", K=500),
            dict(name="SCBF-var K=500 mean-shift only", kind="scbf", form="variance", K=500, mode="mean_only"),
            dict(name="SCBF-var K=500 variance only", kind="scbf", form="variance", K=500, mode="var_only"),
            dict(name="SCBF-var K=500 + IS correction", kind="scbf", form="variance", K=500, is_correction=True),
            dict(name="SCBF-std K=500 (both)", kind="scbf", form="std", K=500)]
    res = run_configs(cfgs, runs); save("E8_mechanisms", res); return res

def E9():
    """Clark's Theorem 3 counterexample: dx = dW, h = x. Exact P(hit 0 by T | x0) = 2 Phi(-x0/sqrt(T))."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    Ts = [1, 5, 20, 50, 100, 200, 500]
    x0 = 1.0; dt = 0.01; n = 20000
    mc = []
    for T in Ts:
        X = np.full(n, x0); hit = np.zeros(n, bool)
        for _ in range(int(T / dt)):
            X += np.sqrt(dt) * rng.standard_normal(n); hit |= X <= 0
        mc.append(float(hit.mean()))
    exact = [float(2 * norm.cdf(-x0 / np.sqrt(T))) for T in Ts]
    out = {"T": Ts, "mc_hit_frac": mc, "exact": exact, "x0": x0, "dt": dt, "n_paths": n}
    save("E9_clark", out); return out

def E11(runs):
    """The correct constraint at different confidence levels: the exploration-safety trade-off."""
    cfgs = [dict(name=f"SCBF-std delta={d}", kind="scbf", form="std", K=500, delta=d) for d in (0.003, 0.05, 0.2, 0.5)]
    cfgs += [dict(name="SCBF-var alpha=3", kind="scbf", form="variance", K=500, alpha=3.0),
             dict(name="SCBF-var alpha=z (2.748)", kind="scbf", form="variance", K=500)]
    cfgs += [dict(name="SCBF-std delta=0.003, |v|<=2", kind="scbf", form="std", K=500, u_max=np.array([2.0, 4.0]).tolist())]
    res = run_configs(cfgs, runs); save("E11_delta_sweep", res); return res

def E12(runs):
    """The printed barriers (sin x) put the goal outside the safe set."""
    env = Corridor(printed=True)
    h1, h2 = env.h(np.array([[4.0, 0.5, 0.0]]))
    cfgs = [dict(name="SCBF-var K=500, printed barriers", kind="scbf", form="variance", K=500, printed=True),
            dict(name="MPPI K=500, printed barriers", kind="mppi", K=500, printed=True)]
    res = run_configs(cfgs, runs); res["_goal_h_printed"] = {"h1": float(h1[0]), "h2": float(h2[0])}
    save("E12_printed_barriers", res); return res


def E13(runs):
    """Sensitivity of the deterministic-MPPI baseline to its schedule (fairness check for E8)."""
    cfgs = [dict(name="Det. MPPI 4x125, nu=0.5", kind="det", iters=4, shrink=0.5, M=125),
            dict(name="Det. MPPI 4x125, nu=0.7", kind="det", iters=4, shrink=0.7, M=125),
            dict(name="Det. MPPI 2x250, nu=0.5", kind="det", iters=2, shrink=0.5, M=250),
            dict(name="Det. MPPI 8x63, nu=0.8", kind="det", iters=8, shrink=0.8, M=63),
            dict(name="Det. MPPI 4x500, nu=0.5 (4x budget)", kind="det", iters=4, shrink=0.5, M=500)]
    res = run_configs(cfgs, runs); save("E13_det_sensitivity", res); return res

def E14(runs):
    """Input bounds |v| <= 2, |omega| <= 4 (the paper has none): does the runaway disappear, and what remains?"""
    ub = [2.0, 4.0]
    cfgs = [dict(name="MPPI K=500, |v|<=2", kind="mppi", K=500, u_max=ub),
            dict(name="SCBF-var K=500, |v|<=2", kind="scbf", form="variance", K=500, u_max=ub),
            dict(name="SCBF-var K=500 + IS, |v|<=2", kind="scbf", form="variance", K=500, u_max=ub, is_correction=True)]
    res = run_configs(cfgs, runs); save("E14_bounded", res); return res

# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="all"); ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    global run_configs
    _rc = run_configs
    def run_configs(configs, runs, procs=a.procs):
        return _rc(configs, runs, procs)
    todo = a.exp.split(",") if a.exp != "all" else ["E1", "E2", "E3", "E4", "E6", "E8", "E9", "E11", "E12", "E13", "E14"]
    for e in todo:
        t = time.time(); print(f"== {e}", file=sys.stderr)
        if e == "E1": E1(a.runs)
        elif e == "E2": E2(max(5, a.runs // 2))
        elif e == "E3": E3()
        elif e == "E4": E4()
        elif e == "E6": E6()
        elif e == "E8": E8(a.runs)
        elif e == "E9": E9()
        elif e == "E11": E11(a.runs)
        elif e == "E12": E12(max(5, a.runs // 2))
        elif e == "E13": E13(a.runs)
        elif e == "E14": E14(a.runs)
        print(f"   done in {time.time()-t:.0f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
