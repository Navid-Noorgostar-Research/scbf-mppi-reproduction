"""Unicycle with additive Brownian noise (Section V-A):
    dx = v cos th dt + s dW1,   dy = v sin th dt + s dW2,   dth = omega dt + s dW3,
sigma = s * I3 (the paper: 'sigma is the identity matrix'), Euler-Maruyama with dt = 0.05 s.
"""
import numpy as np

DT = 0.05

def f_drift(X, U):
    """X: (K,3), U: (K,2) -> (K,3) drift g(x)u (f = 0)."""
    v, om = U[..., 0], U[..., 1]
    th = X[..., 2]
    return np.stack([v * np.cos(th), v * np.sin(th), om], -1)

def step(X, U, dt=DT, sigma=0.0, rng=None):
    """One Euler-Maruyama step. sigma = 0 gives the nominal (rollout) model."""
    Xn = X + f_drift(X, U) * dt
    if sigma > 0:
        Xn = Xn + sigma * np.sqrt(dt) * rng.standard_normal(X.shape)
    return Xn

def rollout(x0, U, dt=DT, sigma=0.0, rng=None):
    """x0: (3,) or (K,3); U: (K,T,2) -> states (K,T+1,3)."""
    K, T, _ = U.shape
    X = np.broadcast_to(np.asarray(x0, float), (K, 3)).copy()
    out = np.empty((K, T + 1, 3)); out[:, 0] = X
    for t in range(T):
        X = step(X, U[:, t], dt, sigma, rng)
        out[:, t + 1] = X
    return out
