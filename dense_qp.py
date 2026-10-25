"""Small dense convex QP active-set solver using the existing NumPy runtime."""
from types import SimpleNamespace

import numpy as np


class DenseQP:
    def __init__(self):
        self.active = []

    def solve(self, p, q, a, lower, upper):
        equality = np.isfinite(lower) & np.isfinite(upper) & (np.abs(upper - lower) < 1e-9)
        e, b = a[equality], upper[equality]
        upper_rows = ~equality & np.isfinite(upper)
        lower_rows = ~equality & np.isfinite(lower)
        g = np.vstack([a[upper_rows], -a[lower_rows]])
        h = np.r_[upper[upper_rows], -lower[lower_rows]]
        scale = np.maximum(np.linalg.norm(g, axis=1), 1e-8)
        g, h = g / scale[:, None], h / scale
        es = np.maximum(np.linalg.norm(e, axis=1), 1e-8)
        e, b = e / es[:, None], b / es
        active = [i for i in self.active if i < len(g)]
        seen = set()
        x = None
        for iteration in range(100):
            signature = tuple(active)
            if signature in seen:
                break
            seen.add(signature)
            c = np.vstack([e, g[active]])
            d = np.r_[b, h[active]]
            kkt = np.block([[p, c.T], [c, np.zeros((len(c), len(c)))]])
            rhs = np.r_[-q, d]
            try:
                result = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                result = np.linalg.lstsq(kkt, rhs, rcond=1e-10)[0]
            x = result[:len(q)]
            dual = result[len(q) + len(e):]
            if len(dual) and np.min(dual) < -1e-5:
                active.pop(int(np.argmin(dual)))
                continue
            violation = g @ x - h
            if np.max(violation) <= 2e-5 and (not len(e) or np.max(np.abs(e @ x - b)) <= 2e-5):
                self.active = active
                return SimpleNamespace(x=x, info=SimpleNamespace(status="solved (NumPy active set)", status_val=1))
            index = int(np.argmax(violation))
            if index in active:
                break
            active.append(index)
        self.active = []
        return SimpleNamespace(x=None, info=SimpleNamespace(status="active set did not converge", status_val=7))
