"""Bundle adjustment: refine all camera poses and 3D points together.

Minimises the total reprojection error

    sum over observations (i, j) of || x_ij - pi(K, R_i, t_i, X_j) ||^2

with Levenberg-Marquardt, written out in full:

* **Analytic Jacobians.** Each residual depends on one camera (6 parameters)
  and one point (3), so each observation contributes a 2x6 block A_ij and a
  2x3 block B_ij. Rotations are updated multiplicatively, R <- exp([w]x) R, so
  d(R X)/dw = -[R X]x at w = 0.
* **Sparse normal equations and the Schur complement.** J^T J has the block
  form [[U, W], [W^T, V]] where V is block-diagonal (one 3x3 block per point).
  Eliminating the points gives the reduced camera system
  (U - W V^-1 W^T) dc = g_c - W V^-1 g_p, which is only 6m x 6m for m cameras;
  each point update then follows independently.
* **Gauge.** Moving or rotating the whole scene, or scaling it, does not change
  any reprojection error. The first camera is held fixed and the damping keeps
  the remaining scale direction from drifting.

Derivations in docs/math/pnp_and_bundle_adjustment.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mvg.camera import rodrigues
from mvg.homogeneous import skew


@dataclass
class BAResult:
    R: np.ndarray  # (m, 3, 3)
    t: np.ndarray  # (m, 3)
    X: np.ndarray  # (n, 3)
    rms_before: float
    rms_after: float
    history: list[float] = field(default_factory=list)


def _residuals_and_jacobians(K, R, t, X, cam_idx, pt_idx, obs):
    Xc = np.einsum("nij,nj->ni", R[cam_idx], X[pt_idx]) + t[cam_idx]  # camera-frame points
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]
    u = fx * x / z + s * y / z + cx
    v = fy * y / z + cy
    r = np.column_stack([u, v]) - obs
    # d(u, v) / d(camera-frame point)
    dproj = np.zeros((len(obs), 2, 3))
    dproj[:, 0, 0] = fx / z
    dproj[:, 0, 1] = s / z
    dproj[:, 0, 2] = -(fx * x + s * y) / z**2
    dproj[:, 1, 1] = fy / z
    dproj[:, 1, 2] = -fy * y / z**2
    RX = Xc - t[cam_idx]
    d_rot = -np.stack([skew(p) for p in RX])  # d(R X)/dw = -[R X]x
    A = np.concatenate([dproj @ d_rot, dproj], axis=2)  # (n_obs, 2, 6): rotation, translation
    B = dproj @ R[cam_idx]  # (n_obs, 2, 3)
    return r, A, B


def bundle_adjust(K, R0, t0, X0, cam_idx, pt_idx, obs, fixed_cameras=(0,), max_iterations: int = 50,
                  lam: float = 1e-3, tol: float = 1e-10) -> BAResult:
    """``cam_idx``, ``pt_idx``: per-observation camera and point indices; ``obs``: (n_obs, 2) pixels."""
    R, t, X = R0.copy(), t0.copy(), X0.copy()
    m, n = len(R), len(X)
    free = np.array([i not in set(fixed_cameras) for i in range(m)])
    cam_slot = np.cumsum(free) - 1  # position of each free camera in the reduced system
    mf = int(free.sum())

    def cost_of(R_, t_, X_):
        r_, _, _ = _residuals_and_jacobians(K, R_, t_, X_, cam_idx, pt_idx, obs)
        return 0.5 * (r_**2).sum()

    r, A, Bm = _residuals_and_jacobians(K, R, t, X, cam_idx, pt_idx, obs)
    cost = 0.5 * (r**2).sum()
    rms_before = float(np.sqrt((r**2).sum() / len(obs)))
    history = [cost]
    for _ in range(max_iterations):
        # accumulate the blocks of J^T J and J^T r
        U = np.zeros((mf, 6, 6))
        V = np.zeros((n, 3, 3))
        W = np.zeros((mf, n, 6, 3))
        gc = np.zeros((mf, 6))
        gp = np.zeros((n, 3))
        AtA = np.einsum("kai,kaj->kij", A, A)
        BtB = np.einsum("kai,kaj->kij", Bm, Bm)
        AtB = np.einsum("kai,kaj->kij", A, Bm)
        Atr = np.einsum("kai,ka->ki", A, r)
        Btr = np.einsum("kai,ka->ki", Bm, r)
        np.add.at(V, pt_idx, BtB)
        np.add.at(gp, pt_idx, Btr)
        f = free[cam_idx]
        np.add.at(U, cam_slot[cam_idx[f]], AtA[f])
        np.add.at(gc, cam_slot[cam_idx[f]], Atr[f])
        np.add.at(W, (cam_slot[cam_idx[f]], pt_idx[f]), AtB[f])
        if max(np.abs(gc).max(initial=0), np.abs(gp).max()) < tol:
            break
        while True:
            # Marquardt damping on the diagonal blocks
            Ud = U + lam * np.einsum("kii->ki", U)[:, :, None] * np.eye(6)
            Vd = V + lam * np.einsum("kii->ki", V)[:, :, None] * np.eye(3)
            Vinv = np.linalg.inv(Vd)
            # Schur complement: S = U - W V^-1 W^T, rhs = -gc + W V^-1 gp
            WVinv = np.einsum("cpij,pjk->cpik", W, Vinv)
            S = np.zeros((mf * 6, mf * 6))
            for a in range(mf):
                for b in range(mf):
                    blk = -np.einsum("pij,pkj->ik", WVinv[a], W[b])
                    if a == b:
                        blk += Ud[a]
                    S[6 * a : 6 * a + 6, 6 * b : 6 * b + 6] = blk
            rhs = -gc.ravel() + np.einsum("cpij,pj->ci", WVinv, gp).ravel()
            dc = np.linalg.solve(S, rhs).reshape(mf, 6) if mf else np.zeros((0, 6))
            # back-substitute the points: V dp = -gp - W^T dc
            dp = -np.einsum("pij,pj->pi", Vinv, gp + np.einsum("cpij,ci->pj", W, dc))
            R_new, t_new = R.copy(), t.copy()
            for i in np.flatnonzero(free):
                d = dc[cam_slot[i]]
                R_new[i] = rodrigues(d[:3]) @ R[i]
                t_new[i] = t[i] + d[3:]
            X_new = X + dp
            new_cost = cost_of(R_new, t_new, X_new)
            if new_cost < cost:
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e10:
                new_cost = cost
                break
        if new_cost >= cost:
            break
        improvement = cost - new_cost
        R, t, X, cost = R_new, t_new, X_new, new_cost
        history.append(cost)
        r, A, Bm = _residuals_and_jacobians(K, R, t, X, cam_idx, pt_idx, obs)
        if improvement < tol * cost:
            break
    rms_after = float(np.sqrt(2 * cost / len(obs)))
    return BAResult(R, t, X, rms_before, rms_after, history)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising sum ||dst - (s R src + t)||^2 (Umeyama, 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (a**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    return s, R, mu_d - s * R @ mu_s
