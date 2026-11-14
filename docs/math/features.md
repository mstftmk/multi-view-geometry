# Features: detectors, descriptors and matching

This note derives what [`src/mvg/features/`](../../src/mvg/features) computes. The claims marked *(tested)* are checked in [`tests/test_math_claims.py`](../../tests/test_math_claims.py) or [`tests/test_features.py`](../../tests/test_features.py).

## 1. Corners: the structure tensor

Shift a window $w$ by $(u, v)$ and measure how much it changes:

$$
E(u, v) = \sum_{x, y} w(x, y)\, \big(I(x + u, y + v) - I(x, y)\big)^2 .
$$

A first-order Taylor expansion, $I(x + u, y + v) \approx I + I_x u + I_y v$, gives a quadratic form:

$$
E(u, v) \approx \begin{pmatrix} u & v \end{pmatrix} M \begin{pmatrix} u \\ v \end{pmatrix},
\qquad
M = \sum w \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix} .
$$

$M$ is the **structure tensor**. Its eigenvalues $\lambda_1 \ge \lambda_2$ say how fast $E$ grows along the two principal directions:

| $\lambda_1$ | $\lambda_2$ | Region |
|---|---|---|
| small | small | flat: shifting changes nothing |
| large | small | edge: only shifts across the edge change the window |
| large | large | corner: every shift changes the window |

**Harris** avoids the eigen-decomposition with $R = \det M - k\,(\operatorname{tr} M)^2 = \lambda_1 \lambda_2 - k (\lambda_1 + \lambda_2)^2$, $k \approx 0.04$. **Shi-Tomasi** uses $\lambda_2$ directly *(tested)*. In the code, $I_x, I_y$ are derivative-of-Gaussian filters (`sigma_d`) and $w$ is a Gaussian (`sigma_i`).

**FAST** is a segment test instead of a derivative: a pixel is a corner if at least 9 contiguous pixels of the 16 on a radius-3 circle are all brighter, or all darker, than it by a threshold. On a straight edge only 7 of the 16 differ, so edges are rejected *(tested)*.

## 2. Scale: Gaussian scale space and the difference of Gaussians

A corner detector works at one scale. To find the same structure when the image is zoomed, search over scales too. Blurring with Gaussians of growing $\sigma$ gives the **scale space** $L(x, y, \sigma) = G_\sigma * I$. The Gaussian is the natural choice: blurring with it never creates new structure as $\sigma$ grows.

**Scale-normalised Laplacian.** Derivatives get weaker as the blur grows, so they are multiplied by $\sigma$ per derivative order: $\sigma^2 \nabla^2 L$. Its extrema over $(x, y, \sigma)$ mark blobs and their size.

*Why the extremum gives the blob's size.* Take a Gaussian blob of standard deviation $s$. Blurring it with $G_\sigma$ gives a Gaussian of variance $s^2 + \sigma^2$, whose Laplacian at the centre is $-\frac{1}{\pi (s^2 + \sigma^2)^2}$ (up to the blob's amplitude). The normalised response is proportional to

$$
f(\sigma) = \frac{\sigma^2}{(s^2 + \sigma^2)^2},
\qquad
f'(\sigma) = 0 \;\Longleftrightarrow\; s^2 + \sigma^2 = 2 \sigma^2 \;\Longleftrightarrow\; \sigma = s .
$$

So the normalised Laplacian peaks exactly at the blob's scale *(tested)*.

**Difference of Gaussians.** The Gaussian satisfies the heat equation $\partial G / \partial \sigma = \sigma \nabla^2 G$. A finite difference between neighbouring scales $\sigma$ and $k \sigma$ gives

$$
G_{k\sigma} - G_\sigma \approx (k - 1)\, \sigma^2 \nabla^2 G_\sigma ,
$$

the scale-normalised Laplacian, up to the constant $k - 1$, at the cost of one subtraction. SIFT uses $s = 3$ layers per octave, so $k = 2^{1/3}$.

*Where the DoG peaks.* The DoG response at the centre of the same blob, between $\sigma$ and $k\sigma$, is proportional to

$$
g(\sigma) = \frac{\sigma^2}{(s^2 + \sigma^2)(s^2 + k^2 \sigma^2)} .
$$

With $u = \sigma^2$, setting the derivative to zero gives $s^4 = k^2 u^2$, so

$$
\sigma = \frac{s}{\sqrt{k}} .
$$

The DoG labels a blob with the *lower* of the two scales it compares, so the detected scale is $s / \sqrt k \approx 0.89\, s$ for $k = 2^{1/3}$. Our detector finds exactly this on synthetic blobs *(tested)*.

**Sub-pixel refinement.** Around a discrete extremum, expand the DoG in $\mathbf{x} = (x, y, \sigma)$:

$$
D(\mathbf x + \boldsymbol\delta) \approx D + \mathbf g^\top \boldsymbol\delta + \tfrac12 \boldsymbol\delta^\top \mathbf H \boldsymbol\delta
\quad\Longrightarrow\quad
\boldsymbol\delta = -\mathbf H^{-1} \mathbf g ,\qquad
D(\mathbf x + \boldsymbol\delta) = D + \tfrac12 \mathbf g^\top \boldsymbol\delta .
$$

If any component of $\boldsymbol\delta$ exceeds 0.5, the extremum is closer to a neighbouring sample, so the fit moves there and repeats. Points with $|D(\mathbf x + \boldsymbol\delta)| < 0.04 / s$ are dropped as low-contrast.

**Edge rejection.** The DoG also responds strongly along edges, where position is poorly defined. With the $2 \times 2$ spatial Hessian of $D$ and its eigenvalues $\alpha = r \beta$,

$$
\frac{(\operatorname{tr} H)^2}{\det H} = \frac{(\alpha + \beta)^2}{\alpha \beta} = \frac{(r + 1)^2}{r} ,
$$

which grows with the ratio $r$ *(tested)*. Keeping only points with $(\operatorname{tr} H)^2 / \det H < (r + 1)^2 / r$ for $r = 10$ removes edge responses without computing eigenvalues.

## 3. Orientation and the SIFT descriptor

**Orientation.** Build a 36-bin histogram of gradient directions around the keypoint, weighted by gradient magnitude and a Gaussian of $1.5\sigma$. The highest peak, refined with a parabola through three bins, is the keypoint's orientation. Every other peak above 80% of the highest creates a second keypoint with that orientation.

**Descriptor.** Take a window in the keypoint's own frame (rotated by its orientation and scaled by its $\sigma$), split it into $4 \times 4$ cells, and build an 8-bin histogram of gradient directions in each: $4 \times 4 \times 8 = 128$ numbers. Each gradient is spread over the two nearest cells in each direction and the two nearest orientation bins (trilinear interpolation), so the descriptor does not jump when the keypoint moves slightly.

**Why it ignores lighting.** If the lighting changes as $I' = a I + b$:

- $b$ disappears in the gradients: $\nabla I' = a \nabla I$;
- $a$ scales every gradient magnitude equally, and normalising the descriptor to unit length removes it.

So the descriptor is unchanged by affine intensity changes *(tested)*. Non-linear changes, such as camera saturation, mostly affect the size of large gradients, not their direction. So entries are clipped at 0.2 and the vector is normalised again.

## 4. SURF: box filters on an integral image

**Integral image.** $II(y, x) = \sum_{y' < y,\ x' < x} I(y', x')$. Any box sum then costs 4 look-ups, whatever its size:

$$
\sum_{y_0 \le y < y_1,\ x_0 \le x < x_1} I = II(y_1, x_1) - II(y_0, x_1) - II(y_1, x_0) + II(y_0, x_0)
$$

*(tested)*. So instead of shrinking the image, SURF grows its filters: sizes 9, 15, 21, 27 in the first octave, and so on.

**Hessian detector.** The second derivatives of a Gaussian, $L_{xx}, L_{yy}, L_{xy}$, are replaced by box filters $D_{xx}, D_{yy}, D_{xy}$. Each lobe is a box with weight $+1$ or $-2$, and responses are divided by the filter area. Blobs are maxima of

$$
\det \mathcal H \approx D_{xx} D_{yy} - (0.9\, D_{xy})^2 ,
$$

where 0.9 balances the box approximation of $L_{xy}$ against that of $L_{xx}$ and $L_{yy}$. Bay et al. equate a filter of size $L = 9$ with $\sigma = 1.2$, so the code reports $\sigma = 1.2 L / 9$.

*Measured here:* on Gaussian blobs, the normalised Gaussian Hessian determinant peaks at the blob's $\sigma$, as section 2 predicts. The box-filter determinant peaks at a smaller filter, where $1.2 L / 9$ is about 0.6 to 0.8 of the blob's $\sigma$ *(tested)*. So $1.2 L / 9$ underestimates the scale of Gaussian blobs. This does not hurt matching, which only needs the scale to be consistent between images, but the reported scales should not be read as Gaussian $\sigma$.

**Descriptor.** Haar wavelets (box differences) give $d_x, d_y$ at $5 \times 5$ sample points in each of $4 \times 4$ sub-regions of a $20\sigma$ window, rotated into the keypoint frame. Each sub-region stores $\left(\sum d_x, \sum d_y, \sum |d_x|, \sum |d_y|\right)$: 64 numbers. The absolute sums tell apart regions whose responses cancel in the plain sums, such as a stripe pattern.

## 5. ORB-style binary features

**Intensity centroid.** The patch moments are $m_{pq} = \sum_{x, y} x^p y^q I(x, y)$ over a disc. The centroid is $(m_{10} / m_{00},\ m_{01} / m_{00})$, and the orientation is the direction from the centre to it: $\theta = \operatorname{atan2}(m_{01}, m_{10})$ *(tested)*.

**BRIEF.** Choose 256 pairs of points $(p_i, q_i)$ once, at random. Bit $i$ is $1$ if $I(p_i) < I(q_i)$ in a smoothed patch. Rotating the pairs by $\theta$ first makes the descriptor rotation-invariant ("steered BRIEF"). Each comparison only depends on the order of two intensities, so any increasing change of brightness leaves the bits unchanged.

**Hamming distance.** The number of differing bits, $\operatorname{popcount}(a \oplus b)$ *(tested)*. A 256-bit descriptor takes 32 bytes, against 512 bytes for 128 floats, and comparing two takes a few XOR and popcount instructions.

## 6. Matching

**Ratio test.** For each descriptor in image 1, find its nearest and second-nearest neighbours in image 2, at distances $d_1 \le d_2$, and accept only if $d_1 < 0.8\, d_2$. A correct match usually has one clearly closest candidate. A point with no true partner, or one on a repeated texture, tends to have several similar ones. Lowe reports that this threshold removes about 90% of false matches while losing under 5% of correct ones.

**Mutual check.** Keep $(i, j)$ only if $j$ is $i$'s nearest neighbour and $i$ is $j$'s.

## 7. Results on the Oxford sequences

Every method keeps at most 2,000 keypoints and uses the same ratio test. "Correct" means the true homography maps the match to within 3 px. The homography error is the mean corner displacement of the homography estimated from the matches (RANSAC and LM from [`homography.md`](homography.md)) against the true one.

| Pair | Method | Correct / matches | Precision | Homography error | Time per image |
|---|---|---:|---:|---:|---:|
| graf 1→2 | SIFT (ours) | 431 / 488 | 0.88 | 0.8 px | 1.2 s |
| | SIFT (OpenCV) | 773 / 889 | 0.87 | 1.1 px | 0.06 s |
| | SURF (ours) | 510 / 693 | 0.74 | 2.5 px | 1.5 s |
| | ORB-style (ours) | 896 / 985 | 0.91 | 1.0 px | 0.6 s |
| | ORB (OpenCV) | 796 / 864 | 0.92 | 0.9 px | 0.02 s |
| | AKAZE (OpenCV) | 832 / 918 | 0.91 | 0.6 px | 0.04 s |
| boat 1→4 | SIFT (ours) | 201 / 242 | 0.83 | 1.2 px | 1.0 s |
| | SIFT (OpenCV) | 301 / 365 | 0.82 | 1.9 px | 0.06 s |
| | SURF (ours) | 275 / 359 | 0.77 | 1.7 px | 1.5 s |
| | ORB-style (ours) | 298 / 368 | 0.81 | 2.8 px | 0.6 s |
| | ORB (OpenCV) | 387 / 438 | 0.88 | 2.7 px | 0.02 s |
| | AKAZE (OpenCV) | 420 / 459 | 0.92 | 1.9 px | 0.06 s |

The full table for all pairs is in [`results/features.csv`](../../results/features.csv). In short:

- **Precision** of our SIFT and ORB-style features is on par with OpenCV's.
- **Our SIFT finds fewer keypoints** (about 1,400 against OpenCV's 2,000 on these images), so it gives fewer correct matches. OpenCV doubles the image before the first octave and we do not.
- **Speed:** ours are pure NumPy and Python, 10 to 25 times slower.
- **Large viewpoint change:** on graf 1→5 and 1→6 (about 50° and 60°) every method fails: none of them is invariant to the affine distortion such a change causes.

## References

- C. Harris and M. Stephens, "A combined corner and edge detector", 1988.
- J. Shi and C. Tomasi, "Good features to track", CVPR 1994.
- E. Rosten and T. Drummond, "Machine learning for high-speed corner detection", ECCV 2006 (FAST).
- T. Lindeberg, "Feature detection with automatic scale selection", IJCV 1998.
- D. Lowe, "Distinctive image features from scale-invariant keypoints", IJCV 2004 (SIFT).
- H. Bay, T. Tuytelaars and L. Van Gool, "SURF: Speeded up robust features", ECCV 2006.
- M. Calonder et al., "BRIEF: Binary robust independent elementary features", ECCV 2010.
- E. Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011.
- K. Mikolajczyk et al., "A comparison of affine region detectors", IJCV 2005 (the Oxford sequences and repeatability).
