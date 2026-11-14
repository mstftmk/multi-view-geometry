# Two views: epipolar geometry, relative pose and triangulation

This note derives what [`src/mvg/two_view.py`](../../src/mvg/two_view.py) computes. The claims marked *(tested)* are checked in [`tests/test_calibration_two_view.py`](../../tests/test_calibration_two_view.py).

## 1. The epipolar constraint

Put camera 1 at the origin, $P_1 = K_1 [I \mid 0]$, and camera 2 at $P_2 = K_2 [R \mid \mathbf t]$. In normalised coordinates $\hat{\mathbf x}_i = K_i^{-1} \mathbf x_i$, a 3D point $\mathbf X$ is seen along the rays $\hat{\mathbf x}_1 \sim \mathbf X$ and $\hat{\mathbf x}_2 \sim R \mathbf X + \mathbf t$.

The vectors $\hat{\mathbf x}_2$, $\mathbf t$ and $R \hat{\mathbf x}_1$ all lie in one plane, the **epipolar plane** through both camera centres and $\mathbf X$. So the triple product vanishes:

$$
\hat{\mathbf x}_2^\top \left(\mathbf t \times R \hat{\mathbf x}_1\right) = 0
\quad\Longleftrightarrow\quad
\hat{\mathbf x}_2^\top \underbrace{[\mathbf t]_\times R}_{E}\, \hat{\mathbf x}_1 = 0 .
$$

$E$ is the **essential matrix**. In pixels, $\mathbf x_2^\top F \mathbf x_1 = 0$ with the **fundamental matrix**

$$
F = K_2^{-\top} E\, K_1^{-1} = K_2^{-\top} [\mathbf t]_\times R\, K_1^{-1}
$$

*(tested)*.

**Reading the constraint.** $\mathbf l_2 = F \mathbf x_1$ is a line in image 2, the **epipolar line**, and the match of $\mathbf x_1$ must lie on it. This turns a 2D search for matches into a 1D one. All epipolar lines meet at the **epipole**, the image of the other camera's centre, with $F \mathbf e_1 = 0$. That is why $F$ has rank 2. [`docs/epipolar_temple.png`](../epipolar_temple.png) shows real matches lying on their epipolar lines.

**Degrees of freedom.** $F$ has 9 entries, minus 1 for scale and 1 for $\det F = 0$: 7. $E$ has 5: 3 for the rotation and 2 for the direction of $\mathbf t$. The scale of $\mathbf t$ cannot be recovered from two images alone, because moving the cameras twice as far apart over a scene twice as large gives the same images.

## 2. The normalised 8-point algorithm

$\mathbf x_2^\top F \mathbf x_1 = 0$ is linear in the 9 entries of $F$:

$$
(u_2 u_1,\ u_2 v_1,\ u_2,\ v_2 u_1,\ v_2 v_1,\ v_2,\ u_1,\ v_1,\ 1) \cdot \mathbf f = 0 .
$$

With 8 or more matches, $\mathbf f$ is the smallest right singular vector of the stacked rows, after Hartley normalisation of both point sets, for the reasons given in [`homography.md`](homography.md). The result generally has rank 3, so the nearest rank-2 matrix is taken: with $F = U \operatorname{diag}(s_1, s_2, s_3) V^\top$, set $s_3 = 0$ *(tested)*. Then denormalise, $F = T_2^\top \hat F T_1$, since $\mathbf x_2^\top T_2^\top \hat F T_1 \mathbf x_1 = \hat{\mathbf x}_2^\top \hat F \hat{\mathbf x}_1$.

## 3. Measuring the error: Sampson distance

$\mathbf x_2^\top F \mathbf x_1$ is an algebraic residual with no unit. The geometric error would need the corrected points, which are themselves unknown. The **Sampson error** linearises the constraint around the measured points:

$$
d_S = \frac{\left|\mathbf x_2^\top F \mathbf x_1\right|}{\sqrt{(F \mathbf x_1)_1^2 + (F \mathbf x_1)_2^2 + (F^\top \mathbf x_2)_1^2 + (F^\top \mathbf x_2)_2^2}} .
$$

It is measured in pixels, and it spreads the correction over both images. When only image 2 is off, it equals the distance to the epipolar line divided by about $\sqrt 2$ *(tested)*. RANSAC uses it to classify matches (with 8-point samples: see the iteration table in [`homography.md`](homography.md)), and the pose refinement below minimises it.

## 4. From E to R and t

A matrix is essential exactly when its singular values are $(\sigma, \sigma, 0)$ *(tested)*. So after $E = K_2^\top F K_1$, the code replaces both non-zero singular values by their mean.

**Four solutions.** With $E = U \operatorname{diag}(1, 1, 0) V^\top$ (signs chosen so $\det U = \det V = 1$) and

$$
W = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix},
$$

the rotations are $R = U W V^\top$ or $U W^\top V^\top$, and the translation is $\mathbf t = \pm \mathbf u_3$, the last column of $U$ (it spans the left null space of $E$, since $\mathbf t^\top [\mathbf t]_\times = 0$). That gives four combinations, and the true pose is one of them *(tested)*.

**Cheirality.** Only one combination puts the triangulated points **in front of both cameras** (positive depth in each). The other three describe the camera turned around, or the scene behind it. The code triangulates all matches for each candidate and keeps the one with the most points in front *(tested)*.

## 5. Triangulation

For a match $(\mathbf x_1, \mathbf x_2)$, $\mathbf x_i \times P_i \mathbf X = 0$. Each view gives two independent linear equations (as in the DLT):

$$
\begin{pmatrix} u_1 \mathbf p_1^{3\top} - \mathbf p_1^{1\top} \\ v_1 \mathbf p_1^{3\top} - \mathbf p_1^{2\top} \\ u_2 \mathbf p_2^{3\top} - \mathbf p_2^{1\top} \\ v_2 \mathbf p_2^{3\top} - \mathbf p_2^{2\top} \end{pmatrix} \mathbf X = 0,
$$

where $\mathbf p_i^{k\top}$ is row $k$ of $P_i$. $\mathbf X$ is the smallest right singular vector. Without noise this is exact *(tested)*; with noise it minimises an algebraic error, and the multi-view refinement minimises the reprojection error instead.

## 6. Refining the pose

The 8-point estimate minimises an algebraic error. So the code refines the pose with Levenberg-Marquardt on the Sampson error, over exactly the 5 degrees of freedom of $E$:

- **Rotation:** $R \leftarrow \operatorname{Rodrigues}(\boldsymbol\omega)\, R$, with 3 parameters $\boldsymbol\omega$ around the current estimate.
- **Translation direction:** $\mathbf t \leftarrow (\mathbf t + a\, \mathbf b_1 + b\, \mathbf b_2) / \lVert\cdot\rVert$, where $\mathbf b_1, \mathbf b_2$ span the tangent plane of the unit sphere at the current $\mathbf t$. This gives 2 parameters and keeps $\lVert \mathbf t \rVert = 1$.

## 7. Results on the Middlebury temple

The [templeSparseRing](https://vision.middlebury.edu/mview/data/) gives 16 calibrated views of a plaster temple. For each of the 16 neighbouring pairs on the ring, the code uses our SIFT features and the ratio test, then compares three methods on the same matches:

- our normalised 8-point + RANSAC;
- the same plus the Levenberg-Marquardt refinement;
- OpenCV's 5-point RANSAC (`findEssentialMat` + `recoverPose`), with the same 1 px threshold.

Errors are against the calibrated relative pose: the rotation angle of $R_{\text{est}}^\top R_{\text{true}}$, and the angle between translation directions.

On the **10 pairs taken 23° apart** (median 44 RANSAC inliers):

| Method | Median rotation error | Median translation-direction error | Pairs with translation error < 5° |
|---|---:|---:|---:|
| 8-point | 2.0° | 10.3° | 1 / 10 |
| 8-point + LM | **1.5°** | **1.1°** | **10 / 10** |
| OpenCV 5-point | 4.7° | 2.9° | 8 / 10 |

On the other **6 pairs**, the views are 54° to 165° apart (some images in this set are rotated by 180°, which the dataset notes). They leave at most 14 matches after RANSAC, and all three methods fail. The full table is in [`results/two_view.csv`](../../results/two_view.csv).

The temple has little texture on a black background, so there are few matches (30 to 111 per pair), and 10 pairs is a small sample. The clearest result is the effect of the refinement: the translation direction from the algebraic 8-point solution is off by 10° in the median, and the 5-DoF refinement on the Sampson error brings it to 1°.

## References

- H. Longuet-Higgins, "A computer algorithm for reconstructing a scene from two projections", *Nature*, 1981 (the essential matrix).
- R. Hartley, "In defense of the eight-point algorithm", *IEEE TPAMI*, 1997.
- R. Hartley and A. Zisserman, *Multiple View Geometry*, 2nd ed., 2004: chapters 9 (epipolar geometry), 11 (computing F, Sampson error) and 12 (triangulation).
- D. Nistér, "An efficient solution to the five-point relative pose problem", *IEEE TPAMI*, 2004 (OpenCV's method).
- S. Seitz et al., "A comparison and evaluation of multi-view stereo reconstruction algorithms", CVPR 2006 (the temple data).
