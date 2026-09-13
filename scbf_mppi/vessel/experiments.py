"""Vessel experiments V0..V7.  Usage:
    python -m scbf_mppi.vessel.experiments --exp all --runs 30
    python -m scbf_mppi.vessel.experiments --exp V2,V4 --runs 30 --procs 2
Results go to results/vessel_V*.json (paths relative to the package root).

Regime (see the README, "On the water"): the group's Solgenia model unchanged; u = (X_AT, Y_AT, F_BT) [N];
control interval 1 s, horizon 15 s, K = 500; Sigma0 = diag(350^2, 350^2, 120^2) N^2; lambda = 300 m^2;
environmental force s_F = (120 N, 300 N, 400 N m) with 5 s correlation time (white-noise equivalent
sigma_F = s_F sqrt(10)); barrier gains alpha1 = 0.1 s^-1, alpha2 = 0.4 s^-1; delta = 0.003 as in the paper.
"""
import argparse, json, os, time, sys
import numpy as np
from multiprocessing import Pool
from . import solgenia as sg, harbour as hb
from .controllers import VesselMPPI, VesselSCBFMPPI, VesselDetMPPI
from .simulate import run_episode, summarize
from . import solver_nd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

DEFAULT = dict(K=500, T=15, lam=300.0, s0=(350.0, 350.0, 120.0), s_F=(120.0, 300.0, 400.0), tau_c=5.0,
               alpha1=0.1, alpha2=0.4, delta=0.003, disturbance="white", max_time=150.0)


def make_harbour(cfg):
    a1 = cfg.get("alpha1", DEFAULT["alpha1"]); a2 = cfg.get("alpha2", DEFAULT["alpha2"])
    return hb.harbour_crossing(a1, a2) if cfg.get("scenario", "static") == "crossing" else hb.harbour_static(a1, a2)


def make_controller(cfg, seed):
    H = make_harbour(cfg)
    kind = cfg["kind"]
    common = dict(K=cfg.get("K", DEFAULT["K"]), T=cfg.get("T", DEFAULT["T"]), lam=cfg.get("lam", DEFAULT["lam"]),
                  s0=tuple(cfg.get("s0", DEFAULT["s0"])), seed=seed)
    if kind == "mppi":
        c = VesselMPPI(H, **common)
    elif kind == "scbf":
        c = VesselSCBFMPPI(H, delta=cfg.get("delta", DEFAULT["delta"]), form=cfg.get("form", "std"),
                           mode=cfg.get("mode", "hocbf"), alpha=cfg.get("alpha"),
                           is_correction=cfg.get("is_correction", False), **common)
    elif kind == "det":
        common.pop("K")
        c = VesselDetMPPI(H, iters=cfg.get("iters", 4), shrink=cfg.get("shrink", 0.5), M=cfg.get("M", 125), **common)
    else:
        raise ValueError(kind)
    return c, H


def one_run(args):
    cfg, seed = args
    c, H = make_controller(cfg, seed)
    r = run_episode(c, H, seed=seed, disturbance=cfg.get("disturbance", DEFAULT["disturbance"]),
                    s_F=np.asarray(cfg.get("s_F", DEFAULT["s_F"]), float) * cfg.get("s_F_mult", 1.0),
                    tau_c=cfg.get("tau_c", DEFAULT["tau_c"]), current=tuple(cfg.get("current", (0.0, 0.0))),
                    max_time=cfg.get("max_time", DEFAULT["max_time"]))
    keep = {k: v for k, v in r.items() if k not in ("traj", "ctrl", "hmin_series", "times", "dist", "ess_series")}
    keep["traj_xy"] = r["traj"][:, :2].tolist() if seed < 3 else None
    keep["speed_series"] = np.abs(r["traj"][1:, 3]).tolist() if seed == 0 else None
    keep["hmin_series"] = r["hmin_series"].tolist() if seed == 0 else None
    keep["mean_thrust"] = float(np.linalg.norm(r["ctrl"][:, :2], axis=1).mean()) if len(r["ctrl"]) else None
    return keep


def _rc(configs, runs, procs=2):
    jobs = [(cfg, s) for cfg in configs for s in range(runs)]
    t = time.time()
    with Pool(procs) as p:
        outs = p.map(one_run, jobs, chunksize=1)
    print(f"  {len(jobs)} episodes in {time.time()-t:.0f}s", file=sys.stderr)
    res = {}
    for (cfg, s), o in zip(jobs, outs):
        res.setdefault(cfg["name"], {"cfg": cfg, "runs": []})["runs"].append(o)
    for name in res:
        res[name]["summary"] = summarize(res[name]["runs"])
    return res


run_configs = _rc


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        txt = json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        f.write(txt.replace("Infinity", "\"inf\"").replace("NaN", "null"))


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
def V0():
    """Solver fidelity on REAL per-sample instances taken from one SCBF-MPPI cycle in the crossing scenario, split by
    the number of simultaneously active rows: closed form (one row, exact) / joint candidate solve (several rows)
    vs the exact SOCP optimum (cvxpy + Clarabel).  Also times the literal SDP (8) (PSD blocks, cvxpy wall time and
    solver-only time) for V5."""
    H = hb.harbour_crossing(); c = VesselSCBFMPPI(H, seed=0, **{k: DEFAULT[k] for k in ("K", "T", "lam", "s0")})
    x = hb.X0.copy(); rng = np.random.default_rng(0)
    for k in range(30):
        u = c.plan(x); x = sg.step(x[None], u[None], c.dt)[0]
    xi = rng.standard_normal((c.K, c.T, 3)); K, T = c.K, c.T
    X = np.broadcast_to(x, (K, 6)).copy(); inst = []
    for t in range(T):
        A, b, _, _ = H.rows(X, c.t + c.dt * t); A = A @ sg.B_ALLOC
        ubar = np.broadcast_to(c.U[t], (K, 3))
        res = solver_nd.solve_rows_nd(ubar, A, b, c.s0, z=c.z, alpha=c.alpha, form="std")
        w0 = A * c.s0[None, None, :]; marg0 = (A * ubar[:, None, :]).sum(-1) - b - c.z * np.linalg.norm(w0, axis=-1)
        n_act = (marg0 < 0).sum(-1)
        for k in range(K):
            if res["active"][k]:
                inst.append((ubar[k].copy(), A[k].copy(), b[k].copy(), int(n_act[k])))
        e = res["mu"] + np.einsum("kij,kj->ki", res["Pfac"], xi[:, t])
        X = sg.step(X, ubar + e, c.dt)
    rng2 = np.random.default_rng(1)
    out = {"n_active_instances_in_cycle": len(inst), "K": K, "T": T, "problems_per_cycle": K * T, "by_active_rows": {}}
    t_socp = []; t_sdp = []; t_sdp_solver = []
    for m in sorted(set(i[3] for i in inst)):
        pool = [i for i in inst if i[3] == m]
        pick = [pool[j] for j in rng2.choice(len(pool), size=min(150, len(pool)), replace=False)]
        gaps = []; feas = 0
        for ubar, A, b, _ in pick:
            res = solver_nd.solve_rows_nd(ubar[None], A[None], b[None], c.s0, z=c.z, alpha=c.alpha, form="std")
            t0 = time.time()
            st, mu, P, val = solver_nd.solve_exact_cvxpy(ubar, np.diag(c.s0), [A[j] for j in range(A.shape[0])], list(b), form="std", z=c.z)
            t_socp.append(time.time() - t0)
            if st not in ("optimal", "optimal_inaccurate"):
                continue
            mine = np.abs(res["mu"][0]).sum() + np.linalg.norm(res["Pfac"][0] - np.diag(c.s0))
            gaps.append((mine, val))
            Sig = res["Pfac"][0] @ res["Pfac"][0].T
            feas += int(all(A[j] @ (ubar + res["mu"][0]) - c.z * np.sqrt(max(A[j] @ Sig @ A[j], 0.0)) >= b[j] - 1e-7 * (1 + abs(b[j])) for j in range(A.shape[0])))
        g = np.array(gaps); rel = (g[:, 0] - g[:, 1]) / np.maximum(g[:, 1], 1e-9)
        out["by_active_rows"][str(m)] = {"n_in_cycle": len(pool), "n_compared": int(len(g)), "feasible_all_rows": feas,
                                          "mean_rel_gap": float(rel.mean()), "median_rel_gap": float(np.median(rel)),
                                          "p90_rel_gap": float(np.percentile(rel, 90)), "max_rel_gap": float(rel.max())}
    # literal SDP (8) timing on 40 single-row instances
    for ubar, A, b, _ in [i for i in inst if i[3] == 1][:40]:
        t0 = time.time()
        solver_nd.solve_sdp_literal(ubar, np.diag(c.s0), [A[j] for j in range(A.shape[0])], list(b), alpha=c.alpha)
        t_sdp.append(time.time() - t0)
        if solver_nd.LAST_SOLVE_TIME is not None:
            t_sdp_solver.append(solver_nd.LAST_SOLVE_TIME)
    out["socp_cvxpy_wall_s_per_problem"] = float(np.median(t_socp))
    out["sdp_literal_cvxpy_wall_s_per_problem"] = float(np.median(t_sdp))
    out["sdp_literal_solver_only_s_per_problem"] = float(np.median(t_sdp_solver)) if t_sdp_solver else None
    save("vessel_V0_solver", out); return out


def V1(runs):
    """As printed (relative degree 1): L_g h = 0 for a force input -> Algorithm 1 cannot act; identical to MPPI."""
    cfgs = [dict(name="MPPI", kind="mppi"),
            dict(name="SCBF as printed (rel. degree 1)", kind="scbf", mode="printed_rd1", form="variance")]
    res = run_configs(cfgs, runs)
    # trajectories identical?
    dmax = 0.0
    for r0, r1 in zip(res["MPPI"]["runs"], res["SCBF as printed (rel. degree 1)"]["runs"]):
        if r0["traj_xy"] is not None and r1["traj_xy"] is not None:
            a = np.array(r0["traj_xy"]); b = np.array(r1["traj_xy"]); m = min(len(a), len(b))
            dmax = max(dmax, float(np.abs(a[:m] - b[:m]).max()))
        dmax = max(dmax, abs(r0["collision_rate"] - r1["collision_rate"]), abs(r0["ttf"] - r1["ttf"]))
    res["_max_abs_difference"] = dmax
    save("vessel_V1_printed", res); return res


def V2(runs):
    """Harbour transit, white force noise: the method with a second-order barrier against the baselines."""
    cfgs = [dict(name="MPPI K=500", kind="mppi"),
            dict(name="MPPI K=2000", kind="mppi", K=2000),
            dict(name="SCBF-MPPI (2nd-order barrier)", kind="scbf"),
            dict(name="SCBF-MPPI + IS correction", kind="scbf", is_correction=True),
            dict(name="SCBF-MPPI, variance form as printed", kind="scbf", form="variance"),
            dict(name="Deterministic MPPI 4x125", kind="det")]
    res = run_configs(cfgs, runs); save("vessel_V2_harbour", res); return res


def V3(runs):
    """Disturbance model: white (the paper's assumption), coloured gust (same low-frequency intensity),
    coloured gust + 0.5 m/s cross-current the controller does not know."""
    cfgs = []
    for dname, dist, cur in [("white", "white", (0.0, 0.0)), ("gust (OU, 5 s)", "ou", (0.0, 0.0)),
                             ("gust + 0.5 m/s current", "ou", (0.0, -0.5))]:
        cfgs.append(dict(name=f"MPPI | {dname}", kind="mppi", disturbance=dist, current=cur))
        cfgs.append(dict(name=f"SCBF-MPPI | {dname}", kind="scbf", disturbance=dist, current=cur))
        cfgs.append(dict(name=f"SCBF-MPPI + IS | {dname}", kind="scbf", is_correction=True, disturbance=dist, current=cur))
    res = run_configs(cfgs, runs); save("vessel_V3_disturbance", res); return res


def V4(runs):
    """Crossing ferry (the group's Experiment-III obstacle: R = 25 m, 3 m/s)."""
    cfgs = [dict(name="MPPI", kind="mppi", scenario="crossing"),
            dict(name="SCBF-MPPI (2nd-order barrier)", kind="scbf", scenario="crossing"),
            dict(name="SCBF-MPPI + IS correction", kind="scbf", is_correction=True, scenario="crossing"),
            dict(name="Deterministic MPPI 4x125", kind="det", scenario="crossing")]
    res = run_configs(cfgs, runs); save("vessel_V4_crossing", res); return res


def V5():
    """Wall-clock per control cycle on this machine (median of 20 cycles after warm-up), and the cost of Algorithm 1
    with the SDP (8) exactly as printed (PSD blocks): cvxpy wall time and solver-only time per problem × K·T."""
    out = {}
    H = hb.harbour_crossing()
    for name, mk in [("MPPI K=500", lambda: VesselMPPI(H, seed=0)),
                     ("MPPI K=2000", lambda: VesselMPPI(H, K=2000, seed=0)),
                     ("SCBF-MPPI closed-form per-sample solve", lambda: VesselSCBFMPPI(H, seed=0)),
                     ("SCBF-MPPI + IS", lambda: VesselSCBFMPPI(H, is_correction=True, seed=0)),
                     ("Deterministic MPPI 4x125", lambda: VesselDetMPPI(H, seed=0))]:
        c = mk(); x = hb.X0.copy(); ts = []
        for k in range(25):
            t0 = time.time(); u = c.plan(x); dt_ = time.time() - t0
            if k >= 5: ts.append(dt_)
            x = sg.step(x[None], u[None], c.dt)[0]
        out[name] = float(np.median(ts))
    v0 = load("vessel_V0_solver") if os.path.exists(os.path.join(RES, "vessel_V0_solver.json")) else V0()
    out["literal SDP (8) per problem, cvxpy wall s"] = v0["sdp_literal_cvxpy_wall_s_per_problem"]
    out["literal SDP (8) per problem, solver-only s"] = v0["sdp_literal_solver_only_s_per_problem"]
    out["problems per cycle (K x T)"] = v0["problems_per_cycle"]
    out["Algorithm 1 cycle with literal SDP, cvxpy wall s"] = v0["sdp_literal_cvxpy_wall_s_per_problem"] * v0["problems_per_cycle"]
    out["Algorithm 1 cycle with literal SDP, solver-only s"] = (v0["sdp_literal_solver_only_s_per_problem"] or 0.0) * v0["problems_per_cycle"]
    out["control interval s"] = sg.DT_CTRL
    save("vessel_V5_wallclock", out); return out


def V6(runs):
    """Barrier gains: how conservative the second-order barrier is."""
    cfgs = [dict(name=f"SCBF-MPPI a1={a1} a2={a2}", kind="scbf", alpha1=a1, alpha2=a2)
            for a1, a2 in [(0.05, 0.2), (0.1, 0.4), (0.2, 0.8), (0.1, 1.0)]]
    res = run_configs(cfgs, runs); save("vessel_V6_gains", res); return res


def V7(runs):
    """Confidence level delta of the chance constraint."""
    cfgs = [dict(name=f"SCBF-MPPI delta={d}", kind="scbf", delta=d) for d in (0.003, 0.05, 0.2)]
    res = run_configs(cfgs, runs); save("vessel_V7_delta", res); return res


def V8(runs):
    """Deterministic MPPI (the group's method): temperature and annealing-schedule sensitivity at 4 x 125 rollouts."""
    cfgs = [dict(name=f"Det. MPPI 4x125 lam0={lam} nu={nu}", kind="det", lam=lam, shrink=nu)
            for lam in (300.0, 1000.0, 3000.0) for nu in (0.5, 0.7)]
    cfgs.append(dict(name="Det. MPPI 2x250 lam0=1000 nu=0.5", kind="det", lam=1000.0, shrink=0.5, iters=2, M=250))
    res = run_configs(cfgs, runs); save("vessel_V8_det_sweep", res); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="all"); ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    _rc0 = _rc
    def run_configs(configs, runs, procs=a.procs):
        return _rc0(configs, runs, procs)
    todo = a.exp.split(",") if a.exp != "all" else ["V0", "V1", "V5", "V2", "V4", "V3", "V6", "V7", "V8"]
    for e in todo:
        t = time.time(); print(f"== {e}", file=sys.stderr)
        if e == "V0": V0()
        elif e == "V1": V1(min(10, a.runs))
        elif e == "V2": V2(a.runs)
        elif e == "V3": V3(a.runs)
        elif e == "V4": V4(a.runs)
        elif e == "V5": V5()
        elif e == "V6": V6(max(10, a.runs * 2 // 3))
        elif e == "V7": V7(max(10, a.runs * 2 // 3))
        elif e == "V8": V8(max(10, a.runs * 2 // 3))
        print(f"   {e} done in {time.time()-t:.0f}s", file=sys.stderr)
