"""Draw docs/two_view_diagram.png: two cameras seeing one 3D point (the epipolar plane, seen face-on)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"


def image_plane(C, X, dist=1.1, half=0.75):
    """A short image-plane segment in front of camera C, facing the scene point X."""
    d = (X - C) / np.linalg.norm(X - C)
    n = np.array([-d[1], d[0]])
    centre = C + dist * d
    return centre - half * n, centre + half * n, d


def hit(C, X, p0, p1):
    """Where the ray from C to X crosses the line p0-p1."""
    A = np.column_stack([X - C, p0 - p1])
    s, _ = np.linalg.solve(A, p0 - C)
    return C + s * (X - C)


def main():
    C1, C2, X = np.array([0.0, 0.0]), np.array([5.0, 0.4]), np.array([2.2, 4.2])
    X_far = C1 + 1.5 * (X - C1)  # a farther point on the same ray of camera 1
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # epipolar plane (here: the drawing plane) shaded through C1, C2, X
    ax.fill([C1[0], C2[0], X[0]], [C1[1], C2[1], X[1]], color=AQUA, alpha=0.08, zorder=0)
    for C, name in ((C1, "Camera 1"), (C2, "Camera 2")):
        p0, p1, _ = image_plane(C, X)
        ax.plot(*zip(p0, p1), color=INK, lw=3, solid_capstyle="butt")
        ax.scatter(*C, s=60, color=INK, zorder=3)
        ax.annotate(name, C, xytext=(0, -18), textcoords="offset points", ha="center", fontsize=10)
    # rays
    ax.plot(*zip(C1, X), color=BLUE, lw=1.6)
    ax.plot(*zip(C2, X), color=ORANGE, lw=1.6)
    ax.plot(*zip(X, X_far), color=BLUE, lw=1.0, ls="--")
    ax.plot(*zip(C1, C2), color=MUTED, lw=1.0, ls=":")
    ax.annotate("baseline", (C1 + C2) / 2, xytext=(0, -14), textcoords="offset points", ha="center",
                fontsize=9, color=MUTED)
    # image points
    p0, p1, _ = image_plane(C1, X)
    x1 = hit(C1, X, p0, p1)
    q0, q1, _ = image_plane(C2, X)
    x2 = hit(C2, X, q0, q1)
    x2_far = hit(C2, X_far, q0, q1)
    ax.plot(*zip(C2, X_far), color=ORANGE, lw=0.8, ls="--", alpha=0.6)
    ax.scatter(*x1, s=45, color=BLUE, zorder=4)
    ax.scatter(*x2, s=45, color=ORANGE, zorder=4)
    ax.scatter(*x2_far, s=30, facecolor="white", edgecolor=ORANGE, zorder=4)
    ax.annotate("$x_1$", x1, xytext=(-22, 6), textcoords="offset points", fontsize=12, color=BLUE)
    ax.annotate("$x_2$", x2, xytext=(8, 2), textcoords="offset points", fontsize=12, color=ORANGE)
    ax.scatter(*X, s=80, color=INK, zorder=4)
    ax.annotate("3D point $X$", X, xytext=(10, 4), textcoords="offset points", fontsize=11)
    ax.scatter(*X_far, s=40, facecolor="white", edgecolor=INK, zorder=4)
    ax.annotate("same pixel $x_1$,\ndifferent depth", X_far, xytext=(10, -6), textcoords="offset points",
                fontsize=9, color=MUTED)
    note = "as the depth along ray 1 changes,\nthe point slides along the\nepipolar line in image 2"
    ax.annotate(note, x2_far, xytext=(30, 30),
                textcoords="offset points", fontsize=9, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 8.2)
    ax.set_ylim(-1.0, 7.0)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/two_view_diagram.png", dpi=130, facecolor="white")
    print("wrote docs/two_view_diagram.png")


if __name__ == "__main__":
    main()
