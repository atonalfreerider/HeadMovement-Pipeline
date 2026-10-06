"""Penalised quintic smoothing spline with a cutoff frequency, on the real (non-uniform) timestamps.

f minimises  sum_i w_i (y_i - f(t_i))^2 + lam * int f'''(t)^2 dt  over quintic splines with a knot at every
sample. For uniform spacing h this is a zero-phase low-pass with gain 1 / (1 + lam * h * w^6), so
lam = 1 / (h (2 pi f_c)^6) puts the -6 dB point at f_c (6th-order roll-off, Butterworth-like but on the true
times, with analytic derivatives up to the 4th). Quadratics are reproduced exactly (null space of the
penalty), so constant gravity-like accelerations are not biased at the ends.

The fit is linear in y: Smoother.operator(nu, x) returns the (len(x), N) matrix mapping samples to the nu-th
derivative at times x, which makes noise propagation and residual bootstraps exact and cheap.
"""
from __future__ import annotations

import math
from functools import cached_property

import numpy as np

K = 5          # quintic
M_PEN = 3      # penalise the 3rd derivative


class Smoother:
    def __init__(self, t: np.ndarray, cutoff_hz: float = 5.0, weights: np.ndarray | None = None):
        from scipy.interpolate import BSpline
        t = np.asarray(t, float)
        if len(t) < K + 2 or not np.all(np.diff(t) > 0):
            raise ValueError("need >= 7 strictly increasing timestamps")
        self.t, self.cutoff_hz = t, float(cutoff_hz)
        self.knots = np.concatenate([[t[0]] * (K + 1), t[1:-1], [t[-1]] * (K + 1)])
        self.n_basis = len(self.knots) - K - 1
        self._basis = BSpline(self.knots, np.eye(self.n_basis), K, extrapolate=False)
        w = np.ones(len(t)) if weights is None else np.clip(np.asarray(weights, float), 1e-3, None)
        w = w / w.mean()
        h = float(np.median(np.diff(t)))
        self.lam = 1.0 / (h * (2 * math.pi * self.cutoff_hz) ** (2 * M_PEN))
        B = self._eval(t, 0)
        # Gauss-Legendre (exact: f''' of a quintic is quadratic, the integrand quartic)
        gx, gw = np.polynomial.legendre.leggauss(3)
        a, b = t[:-1, None], t[1:, None]
        xq = (0.5 * (b - a) * gx + 0.5 * (a + b)).ravel()
        wq = (0.5 * (b - a) * gw).ravel()
        D = self._eval(xq, M_PEN)
        A = B.T @ (w[:, None] * B) + self.lam * (D.T @ (wq[:, None] * D))
        self._hat = np.linalg.solve(A, B.T * w)  # (n_basis, N): samples -> spline coefficients

    def _eval(self, x: np.ndarray, nu: int) -> np.ndarray:
        x = np.clip(np.asarray(x, float), self.t[0], self.t[-1])
        bs = self._basis if nu == 0 else self._basis.derivative(nu)
        return np.nan_to_num(bs(x))

    def operator(self, nu: int = 0, x: np.ndarray | None = None) -> np.ndarray:
        """(len(x), N) linear map from samples to the nu-th derivative at x (default: the sample times)."""
        if x is None:
            return self._ops[nu]
        return self._eval(x, nu) @ self._hat

    @cached_property
    def _ops(self) -> dict:
        return {nu: self._eval(self.t, nu) @ self._hat for nu in range(4)}

    def __call__(self, y: np.ndarray, nu: int = 0, x: np.ndarray | None = None) -> np.ndarray:
        """Smooth / differentiate y (N, ...) along axis 0."""
        y = np.asarray(y, float)
        op = self.operator(nu, x)
        return (op @ y.reshape(len(y), -1)).reshape((op.shape[0],) + y.shape[1:])

    def gain(self, f_hz: np.ndarray) -> np.ndarray:
        """Nominal (uniform-spacing) magnitude response."""
        h = float(np.median(np.diff(self.t)))
        return 1.0 / (1.0 + self.lam * h * (2 * math.pi * np.asarray(f_hz, float)) ** (2 * M_PEN))

    def noise_gain(self, nu: int) -> np.ndarray:
        """Per-sample SD of the nu-th derivative for unit white noise on the samples, (N,)."""
        return np.sqrt((self.operator(nu) ** 2).sum(1))
