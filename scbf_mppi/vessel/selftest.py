"""Self-test of the vessel transfer:  python -m scbf_mppi.vessel.selftest

  1. Model: the vectorised Solgenia dynamics reproduce the group's MATLAB reference (ShipModel.m + PropModel.m,
     RK4_step.m, MinimumExample.m input) — the reference is re-implemented scalar-wise here, straight from the .m files.
  2. Barrier rows: d(psi1)/dt along the deterministic flow equals  a tau − b − alpha2 psi1  (central differences,
     random states, random inputs, static and moving obstacles).
  3. Relative degree: L_g h = 0 for every force input (the printed rows have a == 0 identically).
  4. Itô term: (1/2) tr(sigma sigma^T Hess psi1) is zero for velocity noise; nonzero (reported) for position noise.
  5. Per-sample solver: closed form vs the exact conic optimum (cvxpy + Clarabel), single row exact, two rows gap.
  6. Delivered probability of the row constraint by Monte Carlo on the solved proposal: std form 1 − delta,
     variance form Phi(alpha s / sqrt(...)) < 1 − delta (Theorem 2 error, in the vessel's input space).
Exit code 0 when all pass."""
import sys, time
import numpy as np
from scipy.stats import norm
from . import solgenia as sg, harbour as hb, solver_nd


def check_model():
    """Scalar re-implementation of the MATLAB files, then compare with the vectorised step()."""
    p = dict(xg=0.0, m=3100.0, J=21179.0, X_du=-155.42, Y_dv=-1070.0, N_dv=-3328.0, Y_dr=-1008.0,
             X_u=-84.01, Y_v=-795.58, N_v=-958.4, N_r=-5319.88, Y_r=-896.11, X_uu=-46.73, Y_vv=0.0, N_vv=-234.94,
             Y_rr=-149.38, N_rr=0.0, X_rr=-312.21, X_vr=434.41, Y_uv=395.0, Y_ur=-368.2, N_ur=392.76, N_uv=-138.5,
             Y_vr=70.39, Y_rv=24.09, N_vr=-85.67, N_rv=14.09)
    def ship(x, tau):
        u, v, r = x[3], x[4], x[5]
        MRB = np.array([[p["m"], 0, 0], [0, p["m"], p["m"] * p["xg"]], [0, p["m"] * p["xg"], p["J"]]])
        MA = -np.array([[p["X_du"], 0, 0], [0, p["Y_dv"], p["Y_dr"]], [0, p["N_dv"], 0]])
        CRB = np.array([[0, -p["m"] * r, -p["m"] * p["xg"] * r], [p["m"] * r, 0, 0], [p["m"] * p["xg"] * r, 0, 0]])
        N = np.array([-p["X_u"] * u - p["X_uu"] * abs(u) * u + p["X_vr"] * v * r + p["X_rr"] * r * r,
                      -p["Y_v"] * v - p["Y_r"] * r - p["Y_vv"] * abs(v) * v + p["Y_ur"] * u * r + p["Y_uv"] * u * v - p["Y_rr"] * abs(r) * r - p["Y_vr"] * abs(v) * r - p["Y_rv"] * abs(r) * v,
                      -p["N_r"] * r - p["N_v"] * v - p["N_rr"] * abs(r) * r + p["N_uv"] * u * v + p["N_ur"] * u * r - p["N_vv"] * abs(v) * v - p["N_vr"] * abs(v) * r - p["N_rv"] * abs(r) * v])
        dnu = np.linalg.solve(MRB + MA, tau - CRB @ x[3:6] - N)
        c, s = np.cos(x[2]), np.sin(x[2])
        deta = np.array([c * u - s * v, s * u + c * v, r])
        return np.concatenate([deta, dnu])
    def prop(x, nAT, al, nBT):
        F_AT = 0.63 * nAT * abs(nAT) - 0.0 * (x[3] * np.cos(al) + (x[4] - x[5] * 2.9) * np.sin(al)) * abs(nAT)
        F_BT = 0.055 * nBT * abs(nBT) * np.exp(-0.62 * x[3] ** 2)
        return np.array([F_AT * np.cos(al), F_AT * np.sin(al) + F_BT, F_BT * 3.7 - F_AT * np.sin(al) * 2.9]), F_AT, al, F_BT
    def rk4(x, tau, h):
        k1 = ship(x, tau); k2 = ship(x + h / 2 * k1, tau); k3 = ship(x + h / 2 * k2, tau); k4 = ship(x + h * k3, tau)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # (a) drift-level: our drift(X, U) against the MATLAB f(x, u) = ShipModel(x, PropModel(x, u)) at random states,
    #     with u = (X_AT, Y_AT, F_BT) chosen to reproduce PropModel's tau exactly at that state
    rng = np.random.default_rng(0); err_d = 0.0
    for _ in range(200):
        x = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-np.pi, np.pi), rng.uniform(-1, 3), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)])
        nAT, al, nBT = rng.uniform(-33, 33), rng.uniform(-np.pi, np.pi), rng.uniform(-66, 66)
        tau, F_AT, _, F_BT = prop(x, nAT, al, nBT)
        U = np.array([F_AT * np.cos(al), F_AT * np.sin(al), F_BT])
        err_d = max(err_d, np.abs(ship(x, tau) - sg.drift(x[None], U[None])[0]).max())
    # (b) trajectory-level: MinimumExample.m (x0, u = (1500/60 Hz, pi/16, −2000/60 Hz), h = 0.1, N = 100) with MATLAB
    #     semantics (PropModel re-evaluated at every RK4 stage) against our force-input abstraction (thrust held over
    #     the step, bow thrust capped at the step's initial speed) — the difference is the abstraction, not the port
    x_ref = np.array([0.0, 10.0, 15 / 8 * np.pi, 1.0, 0.0, 0.0]); x_our = x_ref.copy()
    nAT, al, nBT = 1500 / 60, np.pi / 16, -2000 / 60
    def f_matlab(x):
        return ship(x, prop(x, nAT, al, nBT)[0])
    for _ in range(100):
        h = 0.1
        k1 = f_matlab(x_ref); k2 = f_matlab(x_ref + h / 2 * k1); k3 = f_matlab(x_ref + h / 2 * k2); k4 = f_matlab(x_ref + h * k3)
        x_ref = x_ref + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, F_ATo, _, F_BTo = prop(x_our, nAT, al, nBT)
        x_our = sg.step(x_our[None], np.array([[F_ATo * np.cos(al), F_ATo * np.sin(al), F_BTo]]), 0.1, n_sub=1)[0]
    x_ref[2] = (x_ref[2] + np.pi) % (2 * np.pi) - np.pi
    err_t = np.abs(x_ref[:2] - x_our[:2]).max()
    ok = err_d < 1e-9 and err_t < 0.02
    print(f"[1] model vs MATLAB: drift agrees to {err_d:.1e} at 200 random states (port exact); 10 s of MinimumExample.m with "
          f"thrust held per step differs by {1000 * err_t:.1f} mm (the force-input abstraction) -> {'ok' if ok else 'FAIL'}")
    return ok


def check_rows(n=200, seed=0, eps=1e-4):
    rng = np.random.default_rng(seed)
    H = hb.harbour_crossing()
    X = np.column_stack([rng.uniform(-20, 140, n), rng.uniform(-40, 40, n), rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-1, 3, n), rng.uniform(-1, 1, n), rng.uniform(-0.3, 0.3, n)])
    U = np.column_stack([rng.uniform(-600, 600, n), rng.uniform(-600, 600, n), rng.uniform(-200, 200, n)])
    t = 12.0
    A, b, h, psi1 = H.rows(X, t)
    # deterministic flow with constant tau = B U (no clipping inside: call drift directly)
    tau = U @ sg.B_ALLOC.T
    def psi1_at(Xs, ts):
        return H.rows(Xs, ts)[3]
    dX = sg.drift(X, U)
    Xp = X + eps * dX; Xm = X - eps * dX
    fd = (psi1_at(Xp, t + eps) - psi1_at(Xm, t - eps)) / (2 * eps)            # (n,J) d psi1/dt
    analytic = np.einsum("kji,ki->kj", A, tau) - b - H.alpha2 * psi1
    err = np.abs(fd - analytic).max(); scale = np.abs(fd).max()
    ok = err < 1e-5 * max(1.0, scale)
    print(f"[2] barrier rows: max |d psi1/dt (finite diff) − (a tau − b − alpha2 psi1)| = {err:.2e} on scale {scale:.1f} -> {'ok' if ok else 'FAIL'}")
    return ok


def check_relative_degree(n=100, seed=1):
    rng = np.random.default_rng(seed)
    H = hb.harbour_static()
    X = np.column_stack([rng.uniform(-20, 140, n), rng.uniform(-40, 40, n), rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-1, 3, n), rng.uniform(-1, 1, n), rng.uniform(-0.3, 0.3, n)])
    A, b = H.rows_printed(X, 0.0, alpha=2.748)
    # L_g h by finite differences: h depends on position only, so dh/dnu = 0 and (dh/dnu) M^{-1} B = 0
    eps = 1e-6; lgh = np.zeros((n, len(H.obs), 3))
    for i in range(3):
        Xp = X.copy(); Xp[:, 3 + i] += eps
        lgh[:, :, i] = (H.h_all(Xp, 0.0) - H.h_all(X, 0.0)) / eps
    lgh = lgh @ sg.G_U
    ok = np.abs(A).max() == 0.0 and np.abs(lgh).max() < 1e-9
    print(f"[3] relative degree: max |L_g h| (finite diff) = {np.abs(lgh).max():.1e}; printed rows have a = 0 -> {'ok' if ok else 'FAIL'}")
    return ok


def check_ito(n=50, seed=2):
    rng = np.random.default_rng(seed)
    H = hb.harbour_static()
    X = np.column_stack([rng.uniform(-20, 140, n), rng.uniform(-40, 40, n), rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-1, 3, n), rng.uniform(-1, 1, n), rng.uniform(-0.3, 0.3, n)])
    # velocity noise: Hessian of psi1 in (u, v, r) by central differences
    eps = 1e-3; hess_tr = np.zeros((n, len(H.obs)))
    for i in range(3, 6):
        Xp = X.copy(); Xm = X.copy(); Xp[:, i] += eps; Xm[:, i] -= eps
        hess_tr += (H.rows(Xp, 0.0)[3] - 2 * H.rows(X, 0.0)[3] + H.rows(Xm, 0.0)[3]) / eps ** 2
    ito_v = 0.5 * hess_tr
    ito_p = H.ito_term(X, 0.0, sigma_pos=1.0)
    ok = np.abs(ito_v).max() < 1e-6 and np.abs(ito_p).max() > 1e-3
    print(f"[4] Itô term: velocity noise -> max |(1/2) tr Hess_nu psi1| = {np.abs(ito_v).max():.1e} (zero); "
          f"position noise sigma = 1 m/sqrt(s) -> up to {np.abs(ito_p).max():.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def check_solver():
    t = time.time()
    o1 = solver_nd.validate_against_cvxpy(120, rows=1, form="std", z=2.748)
    o2 = solver_nd.validate_against_cvxpy(80, rows=2, form="std", z=2.748, seed=3)
    ok = o1["max_rel_gap"] < 5e-3 and o2["n_multi_violation"] == 0 and o2["mean_rel_gap"] < 0.05
    print(f"[5] per-sample solver vs cvxpy: 1 row: n={o1['n']} active={o1['n_active']} max rel gap {o1['max_rel_gap']:.1e} (exact up to the grid); "
          f"2 rows: mean rel gap {o2['mean_rel_gap']:.3f}, max {o2['max_rel_gap']:.3f} (joint grid solve), residual violations {o2['n_multi_violation']}/{o2['n']} ({time.time()-t:.1f}s) -> {'ok' if ok else 'FAIL'}")
    return ok


def check_delivered(n=200_000, seed=4):
    """Row active but not collapsed: 0 < slack < z ||w||, so the optimum keeps s = slack/z > 0 and no mean shift."""
    rng = np.random.default_rng(seed)
    s0 = np.array([500.0, 500.0, 150.0]); z = 2.748
    a = np.array([[[1.2e-3, 0.8e-3, 1e-4]]]); ubar = np.zeros((1, 3))
    wn = np.linalg.norm(a[0, 0] * s0)
    b = np.array([[-0.5 * z * wn]])                                  # slack = 0.5 z ||w||  (active, not collapsed)
    out = {}; s_used = {}
    for form in ("std", "variance"):
        res = solver_nd.solve_rows_nd(ubar, a, b, s0, z=z, alpha=z, form=form, n_s=401)
        xi = rng.standard_normal((n, 3))
        e = res["mu"][0] + xi @ res["Pfac"][0].T
        out[form] = float(np.mean((e * a[0, 0]).sum(-1) >= b[0, 0] - 1e-9))
        s_used[form] = float(res["s"][0])
    expect_var = norm.cdf((-b[0, 0] + (a[0, 0] * res["mu"][0]).sum()) / max(s_used["variance"], 1e-12))
    ok = abs(out["std"] - 0.997) < 3e-3 and abs(out["variance"] - expect_var) < 3e-3
    print(f"[6] delivered Pr(a u >= b): std form {out['std']:.4f} (claimed 0.997, s = {s_used['std']:.3f} of {wn:.3f}); "
          f"variance form as printed {out['variance']:.3f} (= Phi(margin/s) = {expect_var:.3f}, s = {s_used['variance']:.3f}) -> {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check_model(), check_rows(), check_relative_degree(), check_ito(), check_solver(), check_delivered()]
    print("all passed" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
