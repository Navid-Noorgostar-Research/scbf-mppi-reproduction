"""Corridor environment of Section V of the paper.

Safe set (Section V-B):  C = {(x, y) : w(x) < y < w(x) + alpha},  alpha = 1,
with w(x) = sin(pi x / 2) as the set is *defined*, or w(x) = sin(x) as the barrier
functions are *printed* (h1 = y - sin x, h2 = sin x + alpha - y).  The two are not the
same curve; under the printed one the goal (4, 0.5) lies outside C (h2 = -0.257).
`Corridor(printed=False)` uses the set as defined; `printed=True` uses the printed barriers.

Barrier functions:      h1 = y - w(x)           (lower wall)
                        h2 = w(x) + alpha - y   (upper wall)
Unicycle drift is all in g(x)u, so L_f h = 0.  With g = [[cos th, 0], [sin th, 0], [0, 1]]:
    L_g h1 = [ sin th - w'(x) cos th ,  0 ],     L_g h2 = -L_g h1
The angular-velocity column is identically zero: steering has relative degree 2 w.r.t. both
barriers; only forward speed acts at first order.
Ito term with sigma = s * I3:  0.5 * s^2 * Tr(H) = 0.5 * s^2 * (-w''(x)) for h1, +0.5 s^2 w''(x) for h2.
"""
import numpy as np

class Corridor:
    def __init__(self, alpha=1.0, printed=False):
        self.alpha = float(alpha)
        self.printed = bool(printed)

    # wall profile and derivatives ------------------------------------------------------
    def w(self, x):
        return np.sin(x) if self.printed else np.sin(np.pi * x / 2.0)

    def dw(self, x):
        return np.cos(x) if self.printed else (np.pi / 2.0) * np.cos(np.pi * x / 2.0)

    def d2w(self, x):
        return -np.sin(x) if self.printed else -(np.pi / 2.0) ** 2 * np.sin(np.pi * x / 2.0)

    # barrier values -------------------------------------------------------------------
    def h(self, X):
        """X: (..., 3) -> (h1, h2) each (...)"""
        x, y = X[..., 0], X[..., 1]
        wx = self.w(x)
        return y - wx, wx + self.alpha - y

    def inside(self, X):
        h1, h2 = self.h(X)
        return (h1 > 0) & (h2 > 0)

    def min_h(self, X):
        h1, h2 = self.h(X)
        return np.minimum(h1, h2)

    # first-order SCBF data -------------------------------------------------------------
    def scbf_rows(self, X, sigma):
        """Return (c, b) for the two constraints  c_k * v + 0 * omega >= b_k, where
        c_k = L_g h_k (v-column), b_k = -h_k - L_f h_k - 0.5 Tr(sigma^T H_k sigma).
        X: (K, 3); returns c: (K, 2), b: (K, 2).  L_f h = 0 for the unicycle."""
        x, y, th = X[..., 0], X[..., 1], X[..., 2]
        h1, h2 = self.h(X)
        c1 = np.sin(th) - self.dw(x) * np.cos(th)
        c2 = -c1
        ito1 = 0.5 * sigma ** 2 * (-self.d2w(x))   # 0.5 s^2 Tr(H1), H1 = diag(-w'', 0, 0)
        ito2 = -ito1
        b1 = -h1 - ito1
        b2 = -h2 - ito2
        return np.stack([c1, c2], -1), np.stack([b1, b2], -1)

    # geometry helpers for plotting ----------------------------------------------------
    def walls(self, x0=-0.2, x1=4.3, n=400):
        xs = np.linspace(x0, x1, n)
        return xs, self.w(xs), self.w(xs) + self.alpha
