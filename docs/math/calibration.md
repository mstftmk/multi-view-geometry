# Camera calibration from a plane (Zhang's method)

This note derives what [`src/mvg/calibration.py`](../../src/mvg/calibration.py) computes. The claims marked *(tested)* are checked in [`tests/test_calibration_two_view.py`](../../tests/test_calibration_two_view.py).

## 1. The model

A board point $(X, Y, 0)$ reaches pixel $\mathbf x$ through

$$
\mathbf x \sim K [r_1\ r_2\ r_3\ \mathbf t] \begin{pmatrix} X \\ Y \\ 0 \\ 1 \end{pmatrix}
= \underbrace{K [r_1\ r_2\ \mathbf t]}_{H} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix},
\qquad
K = \begin{pmatrix} \alpha & \gamma & u_0 \\ 0 & \beta & v_0 \\ 0 & 0 & 1 \end{pmatrix} .
$$

So each view of the board gives a homography $H = [h_1\ h_2\ h_3] = \lambda K [r_1\ r_2\ \mathbf t]$, estimated with the DLT from [`homography.md`](homography.md). Before refinement, lens distortion is ignored.

## 2. Two constraints per view

Since $r_1 = \lambda^{-1} K^{-1} h_1$ and $r_2 = \lambda^{-1} K^{-1} h_2$ are orthonormal:

$$
r_1^\top r_2 = 0 \;\Longrightarrow\; h_1^\top K^{-\top} K^{-1} h_2 = 0,
\qquad
\lVert r_1 \rVert = \lVert r_2 \rVert \;\Longrightarrow\; h_1^\top K^{-\top} K^{-1} h_1 = h_2^\top K^{-\top} K^{-1} h_2
$$

*(tested)*. Write $B = K^{-\top} K^{-1}$. It is symmetric, so it has 6 distinct entries, $\mathbf b = (B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33})^\top$. Each $h_i^\top B h_j$ is linear in $\mathbf b$, $h_i^\top B h_j = \mathbf v_{ij}^\top \mathbf b$, with

$$
\mathbf v_{ij} = \left(h_{i1}h_{j1},\ h_{i1}h_{j2} + h_{i2}h_{j1},\ h_{i2}h_{j2},\ h_{i3}h_{j1} + h_{i1}h_{j3},\ h_{i3}h_{j2} + h_{i2}h_{j3},\ h_{i3}h_{j3}\right)^\top .
$$

Each view adds two rows, $\mathbf v_{12}^\top$ and $(\mathbf v_{11} - \mathbf v_{22})^\top$, to a system $V \mathbf b = 0$. $\mathbf b$ has 5 degrees of freedom (it is defined up to scale), so **three views** determine it, or two if the skew $\gamma$ is assumed to be 0. As with the DLT, $\mathbf b$ is the right singular vector of $V$ with the smallest singular value.

$B$ is $K^{-\top} K^{-1}$ only up to scale, and it is the image of the absolute conic $\omega$ from [`lines_and_stitching.md`](lines_and_stitching.md): the board's two orthogonal axes play the role of two orthogonal vanishing directions.

## 3. From B to K

Expanding $B = \lambda K^{-\top} K^{-1}$ and solving entry by entry gives (Zhang, 2000, appendix B):

$$
\begin{aligned}
v_0 &= \frac{B_{12} B_{13} - B_{11} B_{23}}{B_{11} B_{22} - B_{12}^2}, &
\lambda &= B_{33} - \frac{B_{13}^2 + v_0 (B_{12} B_{13} - B_{11} B_{23})}{B_{11}}, \\
\alpha &= \sqrt{\lambda / B_{11}}, &
\beta &= \sqrt{\frac{\lambda B_{11}}{B_{11} B_{22} - B_{12}^2}}, \\
\gamma &= -\frac{B_{12}\, \alpha^2 \beta}{\lambda}, &
u_0 &= \frac{\gamma v_0}{\beta} - \frac{B_{13}\, \alpha^2}{\lambda} .
\end{aligned}
$$

$B$ must be positive definite, so if the SVD returns $-\mathbf b$, the sign is flipped first. Without noise and distortion this recovers $K$ exactly *(tested)*.

## 4. Pose of each view

With $K$ known, $K^{-1} H = \lambda [r_1\ r_2\ \mathbf t]$, so

$$
\lambda = \frac{1}{\lVert K^{-1} h_1 \rVert}, \quad r_1 = \lambda K^{-1} h_1, \quad r_2 = \lambda K^{-1} h_2, \quad r_3 = r_1 \times r_2, \quad \mathbf t = \lambda K^{-1} h_3 .
$$

The sign of $\lambda$ is chosen so the board is in front of the camera ($t_z > 0$). With noise, $[r_1\ r_2\ r_3]$ is not exactly a rotation. The nearest rotation in the Frobenius norm is $U V^\top$, from its SVD $U \Sigma V^\top$ *(tested)*.

## 5. Refinement with lens distortion

The closed form minimises algebraic quantities and ignores distortion. The final step minimises the **reprojection error**

$$
\sum_{\text{views } i} \sum_{\text{points } j} \left\lVert \mathbf x_{ij} - \pi(K, k_1, k_2, R_i, \mathbf t_i, \mathbf X_j) \right\rVert^2 ,
$$

where $\pi$ projects to normalised coordinates $\mathbf x_n = (X_c / Z_c, Y_c / Z_c)$, applies radial distortion $\mathbf x_d = \mathbf x_n (1 + k_1 r^2 + k_2 r^4)$ with $r = \lVert \mathbf x_n \rVert$, and maps to pixels with $K$. The parameters are $f_x, f_y, u_0, v_0, k_1, k_2$ and, for each view, a rotation (axis-angle, 3 numbers) and a translation. They are refined with the Levenberg-Marquardt solver from [`homography.md`](homography.md), starting from the closed form with $k_1 = k_2 = 0$.

## 6. Results

Synthetic test: 10 views of a $9 \times 6$ board (3 cm squares), a camera with $f_x = 820$, $f_y = 810$, $(u_0, v_0) = (330, 245)$, distortion $k_1 = -0.2$, $k_2 = 0.05$, and 0.3 px Gaussian noise on every corner.

| | $f_x$ | $f_y$ | $u_0$ | $v_0$ | $k_1$ | RMS reprojection error |
|---|---:|---:|---:|---:|---:|---:|
| Truth | 820 | 810 | 330 | 245 | −0.200 | (noise: 0.42 px) |
| Closed form (no distortion) | | | | | | 3.37 px |
| After refinement (ours) | 818.29 | 808.41 | 330.73 | 244.55 | −0.194 | 0.405 px |
| OpenCV `calibrateCamera` | 818.29 | 808.41 | 330.73 | 244.55 | −0.194 | 0.405 px |

The refinement brings the error from 3.37 px to the noise level, and the result agrees with OpenCV to within 0.05 px in $K$ *(tested)*.

## References

- Z. Zhang, "A flexible new technique for camera calibration", *IEEE TPAMI*, 2000.
- R. Hartley and A. Zisserman, *Multiple View Geometry*, 2nd ed., 2004: section 8.8.
