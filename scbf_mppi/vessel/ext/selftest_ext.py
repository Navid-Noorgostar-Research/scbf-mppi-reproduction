"""Checks for the V9-V12 additions.  Run:  python -m scbf_mppi.vessel.ext.selftest_ext

Every check either passes or prints the number that failed.  The first two are the important ones: they
prove the extended controllers with default options are the shipped controllers, so that any difference in
V9-V12 comes from the option being studied and not from the refactor that made the options possible.
"""
import numpy as np

from .. import solgenia as sg
from .. import harbour as hb
from ..controllers import VesselMPPI, VesselSCBFMPPI
from .controllers_ext import MPPIExt, SCBFExt, make_controller_ext
from .costs import HeadingCostMixin
from .observer import CurrentObserver
from .adaptive import _ess_from_logw
from .filters import CBFQPFilter

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def _closed_loop(ctrl, H, n=25, current=(0.0, 0.0)):
    x = hb.X0.copy()
    ctrl.t = 0.0
    xs = [x.copy()]
    for k in range(n):
        u = ctrl.plan(x)
        x = sg.step(x[None], u[None], ctrl.dt, current=current)[0]
        xs.append(x.copy())
    return np.array(xs)


def test_defaults_match_shipped():
    H1, H2 = hb.harbour_static(), hb.harbour_static()
    a = _closed_loop(VesselMPPI(H1, seed=3), H1)
    b = _closed_loop(MPPIExt(H2, seed=3), H2)
    check("MPPIExt with default options == VesselMPPI", np.array_equal(a, b),
          f"max |diff| = {np.abs(a - b).max():.3e}")

    H1, H2 = hb.harbour_crossing(), hb.harbour_crossing()
    a = _closed_loop(VesselSCBFMPPI(H1, seed=3), H1)
    b = _closed_loop(SCBFExt(H2, seed=3), H2)
    check("SCBFExt with default options == VesselSCBFMPPI", np.array_equal(a, b),
          f"max |diff| = {np.abs(a - b).max():.3e}")

    H1, H2 = hb.harbour_static(), hb.harbour_static()
    a = _closed_loop(VesselSCBFMPPI(H1, is_correction=True, seed=1), H1)
    b = _closed_loop(SCBFExt(H2, is_correction=True, seed=1), H2)
    check("SCBFExt + IS with default options == shipped + IS", np.array_equal(a, b),
          f"max |diff| = {np.abs(a - b).max():.3e}")


def test_rows_with_current():
    """The second-order row under a KNOWN current, against finite differences of the true dynamics.

    psi1 = hdot + alpha1 h  and the row a tau >= b encodes  d(psi1)/dt + alpha2 psi1 >= 0, so
    d(psi1)/dt must equal  a tau - b - alpha2 psi1.
    """
    H = hb.harbour_static()
    rng = np.random.default_rng(0)
    c = np.array([0.35, -0.5])
    worst_h = 0.0
    worst_p = 0.0
    for _ in range(12):
        x = np.array([rng.uniform(20, 90), rng.uniform(-20, 20), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-2, 2.5), rng.uniform(-0.6, 0.6), rng.uniform(-0.25, 0.25)])
        u = np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-150, 150)])
        u = sg.clip_inputs(u[None], x[None])[0]
        tau = sg.B_ALLOC @ u
        dt = 1e-5
        a, b, h0, p0 = H.rows(x[None], 0.0, current=tuple(c))
        x1 = sg.step(x[None], u[None], dt, current=tuple(c), n_sub=1)[0]
        _, _, h1, p1 = H.rows(x1[None], dt, current=tuple(c))
        hdot_fd = (h1[0] - h0[0]) / dt
        hdot_an = p0[0] - H.alpha1 * h0[0]
        worst_h = max(worst_h, float(np.abs(hdot_fd - hdot_an).max()))
        p1dot_fd = (p1[0] - p0[0]) / dt
        p1dot_an = a[0] @ tau - b[0] - H.alpha2 * p0[0]
        worst_p = max(worst_p, float(np.abs(p1dot_fd - p1dot_an).max()))
    check("hdot with a known current matches finite differences", worst_h < 1e-3, f"max err {worst_h:.2e}")
    check("d(psi1)/dt with a known current matches the row", worst_p < 5e-2, f"max err {worst_p:.2e}")

    # and the default path is untouched
    H2 = hb.harbour_static()
    x = np.array([50.0, 3.0, 0.2, 1.7, 0.1, 0.02])
    r0 = H2.rows(x[None], 4.0)
    r1 = H2.rows(x[None], 4.0, current=None)
    r2 = H2.rows(x[None], 4.0, current=(0.0, 0.0))
    same = all(np.array_equal(p, q) and np.array_equal(p, r) for p, q, r in zip(r0, r1, r2))
    check("rows(current=None) == rows() == rows(current=(0,0))", same)


def test_heading_cost():
    H = hb.harbour_static()
    c0 = MPPIExt(H, seed=0)
    c1 = MPPIExt(H, seed=0, w_psi=400.0)
    # a state 60 m short of the goal, pointing at it versus pointing away
    toward = np.array([[[60.0, 0.0, 0.0, 1.5, 0.0, 0.0]]])
    away = np.array([[[60.0, 0.0, np.pi, 1.5, 0.0, 0.0]]])
    t = np.array([1.0])
    q0t, q0a = c0.q(toward, t)[0, 0], c0.q(away, t)[0, 0]
    q1t, q1a = c1.q(toward, t)[0, 0], c1.q(away, t)[0, 0]
    check("without the term, heading is free", abs(q0t - q0a) < 1e-9, f"{q0t:.1f} vs {q0a:.1f}")
    check("with the term, pointing away costs 2 w_psi more", abs((q1a - q1t) - 800.0) < 1e-6,
          f"difference {q1a - q1t:.1f}")
    inside = np.array([[[118.0, 0.0, np.pi, 0.5, 0.0, 0.0]]])
    check("the term is gated off inside the goal radius",
          abs(c1.q(inside, t)[0, 0] - c0.q(inside, t)[0, 0]) < 1e-9)
    ast = MPPIExt(H, seed=0, w_astern=2000.0)
    back = np.array([[[60.0, 0.0, 0.0, -1.0, 0.0, 0.0]]])
    fwd = np.array([[[60.0, 0.0, 0.0, 1.0, 0.0, 0.0]]])
    check("the astern term penalises negative surge only",
          abs((ast.q(back, t)[0, 0] - c0.q(back, t)[0, 0]) - 2000.0) < 1e-6
          and abs(ast.q(fwd, t)[0, 0] - c0.q(fwd, t)[0, 0]) < 1e-9)


def test_observer():
    """The estimator must find a constant current through a coloured force gust."""
    from ..simulate import S_F_DEFAULT
    rng = np.random.default_rng(4)
    c_true = np.array([0.0, -0.5])
    obs = CurrentObserver(gain=0.2, dt=1.0)
    x = hb.X0.copy()
    tau_d = S_F_DEFAULT * rng.standard_normal(3)
    errs = []
    for k in range(80):
        obs.update(x)
        u = np.array([400.0, 60.0 * np.sin(k / 7.0), 0.0])
        for _ in range(4):
            x = sg.step(x[None], u[None], 0.25, tau_d=tau_d[None], current=tuple(c_true), n_sub=1)[0]
            tau_d = (tau_d * np.exp(-0.25 / 5.0)
                     + S_F_DEFAULT * np.sqrt(1 - np.exp(-0.5 / 5.0)) * rng.standard_normal(3))
        errs.append(np.linalg.norm(obs.c - c_true))
    check("the current observer converges through a 5 s gust", errs[-1] < 0.08,
          f"final error {errs[-1]:.3f} m/s, after 20 s {errs[19]:.3f}")
    check("the observer is not fooled into a large bias", max(errs[30:]) < 0.2,
          f"worst late error {max(errs[30:]):.3f}")


def test_filter():
    """After filtering, every row must hold (unless the slack had to be used), and the input must be legal."""
    H = hb.harbour_crossing()
    f_nom = CBFQPFilter(H, disturbance="ou")
    f_cc = CBFQPFilter(H, delta=0.003, disturbance="ou")
    f_wh = CBFQPFilter(H, delta=0.003, disturbance="white")
    rng = np.random.default_rng(2)
    worst_cone = 0.0
    worst_bow = 0.0
    for _ in range(120):
        x = np.array([rng.uniform(30, 95), rng.uniform(-25, 25), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-2, 2.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2)])
        u_ref = np.array([rng.uniform(-700, 700), rng.uniform(-700, 700), rng.uniform(-200, 200)])
        u, info = f_nom(x, rng.uniform(0, 40), u_ref)
        worst_cone = max(worst_cone, float(np.hypot(u[0], u[1]) - sg.F_AT_MAX))
        worst_bow = max(worst_bow, float(abs(u[2]) - sg.bow_cap(x[3])))
    check("the filter never leaves the 700 N azimuth disk", worst_cone < 1e-6, f"worst excess {worst_cone:.2e} N")
    check("the filter respects the bow-thruster cap", worst_bow < 1e-6, f"worst excess {worst_bow:.2e} N")

    # row satisfaction over many random active instances, re-checked from the raw rows
    for nm, f in (("nominal", f_nom), ("chance", f_cc)):
        rng = np.random.default_rng(5)
        n_act = 0
        n_slack = 0
        worst = 0.0
        for _ in range(200):
            x = np.array([rng.uniform(35, 95), rng.uniform(-25, 25), rng.uniform(-np.pi, np.pi),
                          rng.uniform(-2, 2.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2)])
            t = rng.uniform(0, 40)
            u_ref = sg.clip_inputs(np.array([rng.uniform(-700, 700), rng.uniform(-700, 700),
                                             rng.uniform(-200, 200)])[None], x[None])[0]
            u, info = f(x, t, u_ref)
            if not info["filter_active"]:
                continue
            n_act += 1
            if info["filter_slack"] > 1e-6:
                n_slack += 1
                continue
            a, b, _, _ = H.rows(x[None], t)
            tight = f.z * np.linalg.norm(a[0] * f.sig_d[None, :], axis=-1) if f.z else np.zeros(b.shape[1])
            g = np.maximum(np.linalg.norm((a[0] @ sg.B_ALLOC) * f.s0[None, :], axis=1), 1e-12)
            m = (a[0] @ (sg.B_ALLOC @ u) - b[0] - tight) / g        # margin in sampling stds
            worst = min(worst, float(m.min()))
        check(f"the {nm} filter satisfies every row it reports feasible", worst > -1e-5,
              f"{n_act} active, {n_slack} needed slack, worst margin {worst:.2e} std")
    ratio = f_wh.sig_d / f_cc.sig_d
    check("white-noise tightening is sqrt(2 tau_c / h) times the gust tightening",
          np.allclose(ratio, np.sqrt(10.0)), f"ratio {ratio}")


def test_adaptive():
    """ESS(lambda) has a ceiling set by the importance weights; the search must find it and say so."""
    rng = np.random.default_rng(7)
    K = 500
    J = rng.gamma(2.0, 900.0, K)
    C = rng.normal(0, 1.0, K)
    logq = rng.normal(0, 3.0, K)                      # a deliberately heavy importance weight
    ess = [_ess_from_logw(-(J - J.min()) / l - C + logq) for l in np.geomspace(1.0, 3e6, 40)]
    check("ESS rises with lambda and saturates below K",
          ess[0] < ess[-1] < K, f"ESS(cold) {ess[0]:.1f}  ESS(hot) {ess[-1]:.1f}  K {K}")

    H = hb.harbour_static()
    c = SCBFExt(H, is_correction=True, ess_target=100.0, seed=0)
    x = hb.X0.copy()
    got = []
    for k in range(12):
        u = c.plan(x)
        got.append((c.last["ess"], c.last["lam_eff"], c.last["ess_target_met"]))
        x = sg.step(x[None], u[None], c.dt)[0]
    met = [g for g in got if g[2]]
    check("when the target is reachable the achieved ESS is at the target",
          all(abs(g[0] - 100.0) < 2.0 for g in met) if met else True,
          f"{len(met)}/{len(got)} cycles reachable; ESS {[round(g[0]) for g in got]}")
    check("the fixed-temperature path is unchanged by the mixin",
          np.array_equal(_closed_loop(SCBFExt(hb.harbour_static(), is_correction=True, seed=0), H, 8),
                         _closed_loop(VesselSCBFMPPI(hb.harbour_static(), is_correction=True, seed=0), H, 8)))


def test_why_astern():
    """The claims about WHY the boats go astern, turned into measurements.

    Two of these were first written down from a plausible argument and then found to be wrong, which is the
    reason they are a test now rather than a sentence in a document:

      * "the planar model is invariant under turning the heading 180 deg and reversing surge and sway" is
        FALSE in general -- the terms that are even in (u, v), such as X_rr r^2, break it.  It is exactly
        true only when sway and yaw rate are zero, i.e. in straight-line motion.
      * "turning from dead astern takes about 16 s" was an estimate, not a measurement.  The turn alone takes
        about 11 s at full lateral thrust, and rebuilding headway takes 6 to 7 s more.
    """
    rng = np.random.default_rng(0)

    # (a) straight-line motion: reversing surge reverses the surge drag exactly, same magnitude
    worst_pure = 0.0
    for u in np.linspace(-3.0, 3.0, 41):
        f = sg.f_nu(np.array([[u, 0.0, 0.0]]))[0]
        fm = sg.f_nu(np.array([[-u, 0.0, 0.0]]))[0]
        worst_pure = max(worst_pure, float(np.abs(fm - np.array([-f[0], -f[1], f[2]])).max()))
    check("in straight-line motion the model cannot tell ahead from astern", worst_pure < 1e-12,
          f"worst residual {worst_pure:.1e} m/s^2")

    # (b) but the full 3-DOF model is NOT invariant once the boat is turning
    worst_gen = 0.0
    for _ in range(400):
        nu = np.array([rng.uniform(-2.5, 2.5), rng.uniform(-0.8, 0.8), rng.uniform(-0.4, 0.4)])
        f = sg.f_nu(nu[None])[0]
        fm = sg.f_nu(np.array([[-nu[0], -nu[1], nu[2]]]))[0]
        worst_gen = max(worst_gen, float(np.abs(fm - np.array([-f[0], -f[1], f[2]])).max()))
    check("the symmetry is NOT exact once sway or yaw rate is non-zero", worst_gen > 1e-3,
          f"worst residual {worst_gen:.3f} m/s^2 -- do not claim exact invariance")

    # (c) steady top speed is the same ahead and astern
    def top_speed(sign):
        x = np.zeros(6); U = np.array([sign * sg.F_AT_MAX, 0.0, 0.0])
        for _ in range(4000):
            x = sg.step(x[None], U[None], 0.05, n_sub=1)[0]
        return abs(x[3])
    va, vb = top_speed(+1), top_speed(-1)
    check("steady top speed is the same ahead and astern", abs(va - vb) < 1e-6,
          f"{va:.2f} m/s ahead, {vb:.2f} m/s astern")

    # (d) what the manoeuvre the heading term asks for actually costs
    def turn_time(u0):
        x = np.array([0.0, 0.0, 0.0, u0, 0.0, 0.0]); U = np.array([0.0, -sg.F_AT_MAX, 0.0])
        t = 0.0; turned = 0.0; prev = x[2]
        while t < 60.0:
            x = sg.step(x[None], U[None], 0.05, n_sub=1)[0]
            d = (x[2] - prev + np.pi) % (2 * np.pi) - np.pi
            turned += abs(d); prev = x[2]; t += 0.05
            if turned >= np.pi:
                return t
        return float("inf")

    def recover_time(u0):
        x = np.array([0.0, 0.0, 0.0, u0, 0.0, 0.0]); U = np.array([sg.F_AT_MAX, 0.0, 0.0])
        t = 0.0
        while t < 60.0 and x[3] < 0.0:
            x = sg.step(x[None], U[None], 0.05, n_sub=1)[0]; t += 0.05
        return t

    tt, rt = turn_time(-1.9), recover_time(-1.9)
    check("a 180 deg turn plus rebuilding headway exceeds the 15 s horizon", tt + rt > 15.0,
          f"turn {tt:.1f} s at full lateral thrust, headway {rt:.1f} s more, total {tt+rt:.1f} s")
    check("the turn alone is about 11 s, not 16", 9.0 < tt < 13.0, f"{tt:.1f} s")


def test_factory():
    for cfg in [dict(name="x", kind="mppi"), dict(name="x", kind="scbf"),
                dict(name="x", kind="mppi_filter", filter=dict(delta=None)),
                dict(name="x", kind="scbf_filter", filter=dict(delta=0.003)),
                dict(name="x", kind="det"),
                dict(name="x", kind="scbf", obs_mode="both", current=(0.0, -0.5)),
                dict(name="x", kind="scbf", ess_target=100.0, is_correction=True)]:
        c, H = make_controller_ext(cfg, 0)
        x = hb.X0.copy()
        c.t = 0.0
        u = c.plan(x)
        assert np.all(np.isfinite(u)), cfg
    check("every config in the factory plans a finite input", True)


if __name__ == "__main__":
    print("V9-V12 additions — selftest")
    for fn in (test_defaults_match_shipped, test_rows_with_current, test_heading_cost,
               test_why_astern, test_observer, test_filter, test_adaptive, test_factory):
        print(f"\n{fn.__name__}")
        fn()
    n_bad = sum(1 for v in OK if not v)
    print(f"\n{len(OK) - n_bad} of {len(OK)} checks passed")
    raise SystemExit(1 if n_bad else 0)
