"""The research vessel Solgenia (ISD, HTWG Konstanz) — 3-DOF manoeuvring model, parameters and actuator
geometry copied verbatim from the group's public repository

    https://github.com/hhomb/Solgenia   (Model/ParametersSolgenia.m, Model/ShipModel.m, Model/PropModel.m)

which accompanies Homburger, Wirtensohn, Hoher, Baur, Griesser, Diehl, Reuter, "Solgenia — A Test Vessel Toward
Energy-Efficient Autonomous Water Taxi Applications", Ocean Engineering (2025), arXiv:2502.01207, Table A.5.
(The .m file rounds four of the Table A.5 values: Y_dv −1070 vs −1069.97, N_dv −3328 vs −3328.05, Y_dr −1008 vs
−1008.02, Y_rr −149.38 vs −149.37, J_comb 21179 vs 21178.96, Y_uv 395 vs 395.03, Y_ur −368.2 vs −368.27; the
repository values are used because that is what the group simulates with.)

State  x = (x, y, psi, u, v, r):  ENU position [m], yaw [rad], body-fixed surge, sway [m/s], yaw rate [rad/s].
Kinetics (ShipModel.m):  M dnu/dt = tau − C_RB(nu) nu − N(nu),   deta/dt = R(psi) nu + (current, 0).
Generalised force from the two thrusters (PropModel.m):
    tau = [F_AT cos(alpha),  F_AT sin(alpha) + F_BT,  F_BT L_BT − F_AT sin(alpha) L_AT]^T
with the azimuth thruster (stern, L_AT = 2.9 m) and the bow thruster (L_BT = 3.7 m).

Control input used here:  u = (X_AT, Y_AT, F_BT)  — the azimuth thrust resolved in the body frame plus the bow
thrust, so that tau = B u is LINEAR in u (the paper's method needs input-affine dynamics; a thruster's n|n| law and
azimuth angle are not, so we sample forces and allocate).  Because d_AT = 0 in the identified model, the azimuth
thrust is exactly F_AT = c_AT n|n| with |n| <= 2000/60 Hz, i.e. (X_AT, Y_AT) lies in the disk of radius
F_AT,max = c_AT (2000/60)^2 = 700.0 N; the bow thrust is F_BT = c_BT n|n| exp(−d_BT u^2) with |n| <= 4000/60 Hz,
i.e. |F_BT| <= 244.4 N · exp(−0.62 u^2) (ineffective at speed — the group deactivates it in transit).
Neglected: the actuator time constants (T_Az = 0.2 s, T_alpha = 0.1 s, T_Bow = 0.3 s) and rate limits.
"""
import numpy as np

# ---- ParametersSolgenia.m (verbatim) ---------------------------------------------------------------
XG = 0.0                 # antenna-to-CG offset
MASS = 3100.0            # displacement [kg]
J_COMB = 21179.0         # yaw inertia incl. added inertia [kg m^2]
X_du, Y_dv, N_dv, Y_dr = -155.42, -1070.0, -3328.0, -1008.0        # added mass
X_u, Y_v, N_v, N_r, Y_r = -84.01, -795.58, -958.4, -5319.88, -896.11   # linear damping
X_uu, Y_vv, N_vv, Y_rr, N_rr = -46.73, 0.0, -234.94, -149.38, 0.0       # nonlinear damping
X_rr, X_vr, Y_uv, Y_ur, N_ur, N_uv = -312.21, 434.41, 395.0, -368.2, 392.76, -138.5
Y_vr, Y_rv, N_vr, N_rv = 70.39, 24.09, -85.67, 14.09
C_AT, D_AT, C_BT, D_BT = 0.63, 0.0, 0.055, 0.62                          # thrusters (n in Hz)
L_AT, L_BT = 2.9, 3.7                                                   # lever arms [m]
N_AT_MAX, N_BT_MAX = 2000.0 / 60.0, 4000.0 / 60.0                       # [Hz]
F_AT_MAX = C_AT * N_AT_MAX ** 2                                         # 700.0 N
F_BT_MAX0 = C_BT * N_BT_MAX ** 2                                        # 244.4 N at u = 0
LOA, BEAM = 8.5, 2.2                                                    # hull [m] (Section 3.1)

M_RB = np.array([[MASS, 0.0, 0.0], [0.0, MASS, MASS * XG], [0.0, MASS * XG, J_COMB]])
M_A = -np.array([[X_du, 0.0, 0.0], [0.0, Y_dv, Y_dr], [0.0, N_dv, 0.0]])
M = M_RB + M_A
M_INV = np.linalg.inv(M)
B_ALLOC = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, -L_AT, L_BT]])   # tau = B u,  u = (X_AT, Y_AT, F_BT)
G_U = M_INV @ B_ALLOC                                                       # d(nu)/dt = ... + G_U u
DT = 0.25                                                                   # integration step [s] (the group's Table 3 sampling time)
DT_CTRL = 1.0                                                               # control / sampling interval of the MPPI controllers [s]


def bow_cap(u_surge):
    """Speed-dependent bow-thruster capacity |F_BT| <= 244.4 exp(−0.62 u^2)."""
    return F_BT_MAX0 * np.exp(-D_BT * u_surge ** 2)


def clip_inputs(U, X):
    """Project u = (X_AT, Y_AT, F_BT) onto the reachable set at state X: azimuth thrust onto the 700 N disk,
    bow thrust onto its speed-dependent interval.  U: (..., 3), X: (..., 6)."""
    U = np.array(U, float, copy=True)
    mag = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    scale = np.where(mag > F_AT_MAX, F_AT_MAX / np.maximum(mag, 1e-12), 1.0)
    U[..., 0] *= scale; U[..., 1] *= scale
    cap = bow_cap(X[..., 3])
    U[..., 2] = np.clip(U[..., 2], -cap, cap)
    return U


def coriolis_and_damping(nu):
    """C_RB(nu) nu + N(nu) — the bracket of ShipModel.m, vectorised. nu: (..., 3) -> (..., 3)."""
    u, v, r = nu[..., 0], nu[..., 1], nu[..., 2]
    crb = np.stack([-MASS * r * v - MASS * XG * r * r, MASS * r * u, MASS * XG * r * u], axis=-1)
    n1 = -X_u * u - X_uu * np.abs(u) * u + X_vr * v * r + X_rr * r * r
    n2 = (-Y_v * v - Y_r * r - Y_vv * np.abs(v) * v + Y_ur * u * r + Y_uv * u * v - Y_rr * np.abs(r) * r
          - Y_vr * np.abs(v) * r - Y_rv * np.abs(r) * v)
    n3 = (-N_r * r - N_v * v - N_rr * np.abs(r) * r + N_uv * u * v + N_ur * u * r - N_vv * np.abs(v) * v
          - N_vr * np.abs(v) * r - N_rv * np.abs(r) * v)
    return crb + np.stack([n1, n2, n3], axis=-1)


def f_nu(nu):
    """Drift of the velocity states without input: −M^{-1}(C_RB nu + N(nu))."""
    return -(coriolis_and_damping(nu) @ M_INV.T)


def rot2(psi):
    c, s = np.cos(psi), np.sin(psi)
    return c, s


def drift(X, U, tau_d=None, current=(0.0, 0.0)):
    """dx/dt for state X (...,6) and input U (...,3) (already clipped), optional disturbance force tau_d (...,3)
    and a constant water current (m/s, ENU)."""
    psi, nu = X[..., 2], X[..., 3:6]
    c, s = np.cos(psi), np.sin(psi)
    u, v, r = nu[..., 0], nu[..., 1], nu[..., 2]
    deta = np.stack([c * u - s * v + current[0], s * u + c * v + current[1], r], axis=-1)
    tau = U @ B_ALLOC.T
    if tau_d is not None:
        tau = tau + tau_d
    dnu = (tau - coriolis_and_damping(nu)) @ M_INV.T
    return np.concatenate([deta, dnu], axis=-1)


def step(X, U, dt=DT_CTRL, tau_d=None, current=(0.0, 0.0), sigma_F=None, rng=None, n_sub=None, return_sub=False):
    """One control step of length dt, integrated in n_sub RK4 sub-steps of DT = 0.25 s (the group integrates
    their model with RK4, RK4_step.m).  If sigma_F is given, an additive Euler–Maruyama force-noise increment
    M^{-1} sigma_F sqrt(h) xi is added to the velocity states after every sub-step (additive noise: Milstein =
    Euler–Maruyama, so the scheme is strong order 1)."""
    if n_sub is None:
        n_sub = max(1, int(round(dt / DT)))
    h = dt / n_sub
    U = clip_inputs(U, X)
    Xn = X; subs = []
    for _ in range(n_sub):
        k1 = drift(Xn, U, tau_d, current)
        k2 = drift(Xn + 0.5 * h * k1, U, tau_d, current)
        k3 = drift(Xn + 0.5 * h * k2, U, tau_d, current)
        k4 = drift(Xn + h * k3, U, tau_d, current)
        Xn = Xn + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if sigma_F is not None and rng is not None:
            xi = rng.standard_normal(X.shape[:-1] + (3,))
            Xn = Xn.copy()
            Xn[..., 3:6] += (np.asarray(sigma_F) * xi * np.sqrt(h)) @ M_INV.T
        if return_sub:
            subs.append(Xn.copy())
    Xn = Xn.copy()
    Xn[..., 2] = (Xn[..., 2] + np.pi) % (2 * np.pi) - np.pi
    if return_sub:
        return Xn, subs
    return Xn


def hull_polygon(X, loa=LOA, beam=BEAM):
    """Outline of the hull for drawing (5 points), body frame -> ENU."""
    x, y, psi = X[0], X[1], X[2]
    pts = np.array([[loa / 2, 0.0], [loa / 4, beam / 2], [-loa / 2, beam / 2], [-loa / 2, -beam / 2], [loa / 4, -beam / 2]])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])
