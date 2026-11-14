"""Compare our SIFT, SURF and ORB-style features with OpenCV's SIFT, ORB and AKAZE.

Uses the Oxford "graf" (viewpoint) and "boat" (rotation + zoom) sequences:
image 1 is matched to images 2..6, which get progressively harder. Every
method keeps at most 2,000 keypoints and uses the same ratio test (0.8).

    MVG_DATA=data python scripts/compare_features.py --out results/features.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from mvg.datasets import load_oxford
from mvg.features import orb, sift, surf
from mvg.features.evaluation import evaluate_pair
from mvg.features.matching import match

N = 2000


def ours_sift(img):
    kp, d = sift.detect_and_describe(img, N)
    return kp.xy, d, "l2"


def ours_surf(img):
    kp, d = surf.detect_and_describe(img, N)
    return kp.xy, d, "l2"


def ours_orb(img):
    kp, d = orb.detect_and_describe(img, N)
    return kp.xy, d, "hamming"


def opencv(factory, metric):
    def run(img):
        u8 = (img * 255).round().astype(np.uint8)
        kps, d = factory().detectAndCompute(u8, None)
        order = np.argsort([-k.response for k in kps])[:N]
        xy = np.array([kps[i].pt for i in order]).reshape(-1, 2)
        return xy, (d[order] if d is not None else np.zeros((0, 1))), metric

    return run


def methods():
    import cv2

    return {
        "SIFT (ours)": ours_sift,
        "SURF (ours)": ours_surf,
        "ORB-style (ours)": ours_orb,
        "SIFT (OpenCV)": opencv(lambda: cv2.SIFT_create(nfeatures=N), "l2"),
        "ORB (OpenCV)": opencv(lambda: cv2.ORB_create(nfeatures=N), "hamming"),
        "AKAZE (OpenCV)": opencv(lambda: cv2.AKAZE_create(), "hamming"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/features.csv"))
    args = ap.parse_args()
    rows = []
    for seq_name in ("graf", "boat"):
        seq = load_oxford(seq_name)
        for name, fn in methods().items():
            t = time.perf_counter()
            feats = [fn(img) for img in seq.images]
            seconds = (time.perf_counter() - t) / len(seq.images)
            xy1, d1, metric = feats[0]
            for k in range(2, 7):
                xy2, d2, _ = feats[k - 1]
                m = match(d1, d2, ratio=0.8, metric=metric)
                r = evaluate_pair(xy1, xy2, m, seq.homographies[k - 2], seq.images[0].shape, seq.images[k - 1].shape)
                rows.append({"sequence": seq_name, "method": name, "pair": f"1-{k}", "keypoints": len(xy1),
                             "matches": r.matches, "correct": r.correct, "precision": round(r.precision, 3),
                             "repeatability": round(r.repeatability, 3),
                             "homography_error_px": (round(r.homography_error, 2)
                                                     if np.isfinite(r.homography_error) else ""),
                             "seconds_per_image": round(seconds, 3)})
                print(f"{seq_name} {name:18s} 1-{k}: {r.correct:4d}/{r.matches:4d} correct "
                      f"({r.precision:.2f}), rep {r.repeatability:.2f}, H err {r.homography_error:.1f} px")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
