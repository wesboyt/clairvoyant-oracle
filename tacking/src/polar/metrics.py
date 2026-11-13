"""The polar metric (theory.md S3): distances to the poles, their pole-pole distances, the distance to the hull of the
poles computed from those distances only (Lemma 3), restricted gaps, and step angles.
All distances are Euclidean in sequence form over the non-empty sequences."""
import numpy as np
from scipy.optimize import minimize


def dists(x, poles):
    return np.array([np.linalg.norm(x[1:] - p[1:]) for p in poles])


def pole_gram(poles):
    m = len(poles)
    D2 = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            D2[a, b] = np.sum((poles[a][1:] - poles[b][1:]) ** 2)
    return D2


def hull_dist_from_distances(d, D2):
    """dist(x, conv P) from the pole distances d_i = |x - p_i| and the pole-pole squared distances D2 only:
    dist^2 = min_{lambda in simplex} lambda^T G lambda,  G_ij = (d_i^2 + d_j^2 - D2_ij) / 2.  -> (dist, lambda)"""
    m = len(d)
    if m == 1:
        return float(d[0]), np.ones(1)
    if m == 2:
        D = D2[0, 1]
        if D <= 1e-30:
            return float(min(d)), np.array([1.0, 0.0])
        lam = np.clip((d[1] ** 2 - d[0] ** 2 + D) / (2 * D), 0.0, 1.0)
        v = lam * d[0] ** 2 + (1 - lam) * d[1] ** 2 - lam * (1 - lam) * D
        return float(np.sqrt(max(v, 0.0))), np.array([lam, 1 - lam])
    G = 0.5 * (d[:, None] ** 2 + d[None, :] ** 2 - D2)
    r = minimize(lambda l: l @ G @ l, np.ones(m) / m, jac=lambda l: 2 * G @ l, method='SLSQP',
                 bounds=[(0, 1)] * m, constraints=[dict(type='eq', fun=lambda l: l.sum() - 1)],
                 options=dict(ftol=1e-14, maxiter=200))
    return float(np.sqrt(max(r.fun, 0.0))), r.x


def ellipse_excess(d, D2):
    """E = d_1 + d_2 - |p_1 - p_2| (two poles): zero iff x is on the segment."""
    return float(d[0] + d[1] - np.sqrt(D2[0, 1]))


def restricted_gaps(game, xs, poles_per_player, v):
    """g_k = sum_i [u_i(p_i^k, x_-i) - u_i(x_i, p_-i^k)]-style gap against pole k; for two players
    g_k = (u(p0_k, y) - v) + (v - u(x, p1_k)) >= 0; for one player g = V_opt - V(x) (the same for every pole).
    Equals -d/dt (1/2)|z - p_k|^2 under continuous projected gradient play (theory.md Prop. 1)."""
    if game.players == 1:
        return np.array([v - game.value(xs)] * len(poles_per_player[0]))
    out = []
    for p0, p1 in zip(poles_per_player[0], poles_per_player[1]):
        out.append((game.value([p0, xs[1]]) - v) + (v - game.value([xs[0], p1])))
    return np.array(out)


def cos_angle(step, x, p):
    a = step[1:]
    b = p[1:] - x[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return np.nan
    return float(a @ b / (na * nb))
