"""Draw docs/features_oxford.png from results/features.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4df"
# one colour per algorithm; ours solid, OpenCV dashed
COLOR = {"SIFT": "#2a78d6", "SURF": "#1baf7a", "ORB": "#eb6834", "AKAZE": "#6250d6"}


def family(method: str) -> str:
    return next(k for k in COLOR if method.startswith(k))


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results/features.csv").open()))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    titles = {"graf": "graf: viewpoint change (about 20° to 60°)", "boat": "boat: rotation and zoom"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, seq in zip(axes, ("graf", "boat")):
        for m in methods:
            sel = [r for r in rows if r["sequence"] == seq and r["method"] == m]
            x = [int(r["pair"].split("-")[1]) for r in sel]
            y = [max(int(r["correct"]), 0.8) for r in sel]  # log axis: draw 0 just below 1
            ax.plot(x, y, color=COLOR[family(m)], lw=2, marker="o", ms=4,
                    ls="-" if "ours" in m else "--", label=m)
        ax.set_yscale("log")
        ax.set_xticks(range(2, 7), [f"1→{k}" for k in range(2, 7)])
        ax.set_xlabel("Image pair (harder to the right)", color=MUTED)
        ax.set_title(titles[seq], loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("Correct matches (error < 3 px)", color=MUTED)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/features_oxford.png", dpi=150, facecolor="white")
    print("wrote docs/features_oxford.png")


if __name__ == "__main__":
    main()
