# Homography: derivation and estimation

This note derives what [`src/mvg/homography.py`](../../src/mvg/homography.py) and [`src/mvg/ransac.py`](../../src/mvg/ransac.py) compute. The claims marked *(tested)* are checked in [`tests/test_math_claims.py`](../../tests/test_math_claims.py).

## 1. Where homographies come from

A point $X = (X, Y, Z, 1)^\top$ projects to $x \sim K [R \mid t] X$, where $\sim$ means "equal up to a non-zero scale".

**A plane.** Put the world frame on the plane, so $Z = 0$. Writing $R = [r_1\ r_2\ r_3]$, the third column drops out:

$$
x \sim K [r_1\ r_2\ r_3\ t] \begin{pmatrix} X \\ Y \\ 0 \\ 1 \end{pmatrix} = \underbrace{K [r_1\ r_2\ t]}_{H} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}.
$$

So points on a plane and their images are related by an invertible $3 \times 3$ matrix $H$ *(tested)*. Two images of the same plane are then related by $H_2 H_1^{-1}$, also a homography.

**A rotating camera.** If the camera only rotates about its centre, $x_1 \sim K X_c$ and $x_2 \sim K R X_c$ for the same ray $X_c$, so

$$
x_2 \sim K R K^{-1} x_1 ,
$$

whatever the depth of the scene *(tested)*. This is why panoramas can be stitched with homographies: the scene does not need to be planar if the camera does not translate.

$H$ has 9 entries but is defined only up to scale, so it has **8 degrees of freedom**.

## 2. The direct linear transform (DLT)

For a match $x \leftrightarrow x'$ we want $x' \sim H x$. "Parallel up to scale" is the same as a zero cross product:

$$
x' \times H x = 0 .
$$

Write the rows of $H$ as $h_1^\top, h_2^\top, h_3^\top$ and $x' = (u, v, 1)^\top$. Then $Hx = (h_1^\top x,\ h_2^\top x,\ h_3^\top x)^\top$ and the cross product gives three equations that are **linear in the entries of $H$**:

$$
\begin{pmatrix}
0^\top & -x^\top & v\, x^\top \\
x^\top & 0^\top & -u\, x^\top \\
-v\, x^\top & u\, x^\top & 0^\top
\end{pmatrix}
\begin{pmatrix} h_1 \\ h_2 \\ h_3 \end{pmatrix} = 0 .
$$

The third row is a combination of the first two ($u$ times the first plus $v$ times the second), so each match gives **2 independent equations**. Stacking $n$ matches gives $A h = 0$ with $A$ of size $2n \times 9$. The code builds these rows in `_dlt_rows` (with the second row negated, which changes nothing).

**Four matches** give 8 equations for 8 unknowns (9 entries minus scale), so $A$ has a one-dimensional null space and $H$ is unique, unless three of the four points are collinear. In that case the equations do not pin down a non-singular $H$, and `dlt_homography` rejects the sample by checking the area of every triangle of points (the determinant of three homogeneous points is zero exactly when they are collinear).

**More than four matches** with noise: $Ah = 0$ has no exact solution, and $h = 0$ is a useless one. Fix the scale with $\lVert h \rVert = 1$ and minimise

$$
\min_{\lVert h \rVert = 1} \lVert A h \rVert^2 .
$$

With the SVD $A = U \Sigma V^\top$, $\lVert A h \rVert = \lVert \Sigma V^\top h \rVert$. Put $y = V^\top h$, which also has unit norm because $V$ is orthogonal. Then $\lVert A h \rVert^2 = \sum_i \sigma_i^2 y_i^2$, which is smallest when all the weight is on the smallest singular value: $y = e_9$, so

$$
h = v_9 ,
$$

the **last column of $V$** (the last row of `Vt` in NumPy). Equivalently, $h$ is the eigenvector of $A^\top A$ with the smallest eigenvalue.

## 3. Why normalise the points first

$\lVert A h \rVert$ is an *algebraic* error: it has no meaning in pixels, and its value depends on the coordinate system. With pixel coordinates, the entries of $A$ range from $1$ to about $u x \approx 640 \times 480 \approx 3 \times 10^5$. The matrix is then badly conditioned, and small noise moves the solution a lot.

Hartley's fix (1997) is to change coordinates in each image before the DLT:

1. translate the points so their centroid is at the origin;
2. scale them so their mean distance from the origin is $\sqrt 2$.

Call these similarity transforms $T$ and $T'$ (`normalizing_transform`). With $\hat x = T x$ and $\hat x' = T' x'$, run the DLT on $\hat x \leftrightarrow \hat x'$ to get $\hat H$. Since $\hat x' \sim \hat H \hat x$,

$$
T' x' \sim \hat H\, T x \quad\Longrightarrow\quad x' \sim \underbrace{T'^{-1} \hat H\, T}_{H}\, x .
$$

So the **denormalised** homography is $H = T'^{-1} \hat H T$.

Measured here (200 random homographies, 20 matches each, 1 px Gaussian noise, 640 × 480 images):

| | Condition number of $A$ (median) | Mean transfer error on new points |
|---|---:|---:|
| Normalised DLT | 3.9 | 0.62 px |
| Pixel coordinates | 945,000 | 4.79 px |

Without normalisation the median error is only a little worse (0.70 px), but some cases fail badly, which pushes the mean up.

## 4. From algebraic to geometric error

The DLT minimises $\lVert A h \rVert$, but what we care about is the distance in pixels. The **transfer error** of a match is $d(x', H x)$, where $d$ is the Euclidean distance between inhomogeneous points. Since the noise is in both images, the code uses the **symmetric transfer error**:

$$
E(H) = \sum_i d(x_i', H x_i)^2 + d(x_i, H^{-1} x_i')^2 .
$$

This is not linear in $H$, so it is minimised iteratively, starting from the DLT result. The parameters are the 8 entries of $H$ with $H_{33}$ fixed to 1 (fine unless the true $H_{33}$ is near 0, which happens only for extreme views).

**The Jacobian of one projected point.** With $(a, b, c)^\top = H x$, the predicted pixel is $\hat u = a / c$, $\hat v = b / c$, and

$$
\frac{\partial \hat u}{\partial h_1} = \frac{x^\top}{c}, \qquad
\frac{\partial \hat u}{\partial h_2} = 0^\top, \qquad
\frac{\partial \hat u}{\partial h_3} = -\frac{a}{c^2}\, x^\top ,
$$

and the same for $\hat v$ with $b$ and $h_2$. The code uses numerical differences, which the tests check against.

**Levenberg-Marquardt.** With residual vector $r(p)$ and Jacobian $J$, Gauss-Newton solves $J^\top J\, \delta = -J^\top r$. Levenberg-Marquardt damps this:

$$
\left(J^\top J + \lambda\, \mathrm{diag}(J^\top J)\right) \delta = -J^\top r .
$$

A large $\lambda$ gives a short step along the scaled gradient, which is safe far from the minimum. A small $\lambda$ gives the Gauss-Newton step, which converges fast near it. After a step that lowers the cost, $\lambda$ is divided by 10; after one that does not, the step is rejected and $\lambda$ is multiplied by 10. So the cost never goes up *(tested)*.

## 5. RANSAC with wrong matches

Least squares assumes every match is roughly right, but feature matching produces outright wrong matches. RANSAC repeats: draw a minimal sample (4 matches), fit $H$ with the DLT, and count the matches within a threshold $t$ of it.

**How many samples.** If a fraction $w$ of the matches are correct, one sample of $s$ matches is all-correct with probability $w^s$. The chance that $N$ samples all contain a wrong match is $(1 - w^s)^N$. Setting this to $1 - p$ for confidence $p$ gives

$$
N = \frac{\log(1 - p)}{\log(1 - w^s)} .
$$

| Correct matches $w$ | $N$ for a homography ($s = 4$), $p = 0.99$ | $p = 0.999$ | For comparison, $s = 8$ (fundamental matrix), $p = 0.99$ |
|---:|---:|---:|---:|
| 90% | 5 | 7 | 9 |
| 70% | 17 | 26 | 78 |
| 50% | 72 | 108 | 1,177 |
| 30% | 567 | 850 | 70,188 |
| 20% | 2,876 | 4,314 | 1,798,893 |

$w$ is unknown at the start, so the code re-estimates it from the best model found so far and stops when the number of samples drawn exceeds $N$ (adaptive RANSAC). This is also why minimal samples matter: $N$ grows like $w^{-s}$.

**Choosing the threshold.** If each coordinate has Gaussian noise with standard deviation $\sigma$, the squared transfer error divided by $\sigma^2$ follows a $\chi^2$ distribution with 2 degrees of freedom. Keeping 95% of correct matches needs $t^2 = 5.99\, \sigma^2$, so $t \approx 2.45\, \sigma$.

**MSAC scoring.** Plain RANSAC scores a model by its number of inliers. MSAC (Torr and Zisserman, 2000) minimises $\sum_i \min(e_i^2, t^2)$ instead: inliers count by how well they fit, outliers by a fixed penalty. Among models with the same inlier count, this prefers the tighter one. After the best sample, $H$ is refitted on all inliers with the DLT, then refined with Levenberg-Marquardt.

## References

- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004: chapter 2 (projective plane), chapter 4 (homography estimation, DLT, normalisation, RANSAC), appendix 6 (Levenberg-Marquardt).
- R. Hartley, "In defense of the eight-point algorithm", *IEEE TPAMI*, 1997.
- M. Fischler and R. Bolles, "Random sample consensus", *Communications of the ACM*, 1981.
- P. Torr and A. Zisserman, "MLESAC: A new robust estimator with application to estimating image geometry", *CVIU*, 2000.
