"""The Usage example in README.md runs and recovers the homography."""

import re
from pathlib import Path

import numpy as np

from mvg.homography import apply_homography


def test_readme_usage_example_runs():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    code = re.search(r"## Usage.*?```python\n(.*?)```", readme, re.S).group(1)
    scope: dict = {}
    exec(compile(code, "README.md", "exec"), scope)
    H, H_true = scope["H"], scope["H_true"]
    assert scope["inliers"].sum() >= 100
    # compare where the two matrices send the image corners
    corners = np.array([[0, 0], [319, 0], [319, 239], [0, 239]], float)
    assert np.abs(apply_homography(H, corners) - apply_homography(H_true, corners)).max() < 1.0
