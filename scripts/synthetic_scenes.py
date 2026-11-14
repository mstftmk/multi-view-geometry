"""Synthetic scenes shared by the figure scripts."""

from __future__ import annotations

import numpy as np

from mvg.camera import project, projection_matrix


def room_segments(rng, K, R, t, noise=0.3, per_axis=12):
    """2D image segments of 3D segments parallel to the x, y and z axes, with axis labels."""
    P = projection_matrix(K, R, t)
    segs, labels = [], []
    for axis in range(3):
        for _ in range(per_axis):
            a = rng.uniform(-2, 2, size=3)
            b = a.copy()
            b[axis] += rng.uniform(1.0, 2.0)
            segs.append(project(P, np.array([a, b])) + rng.normal(scale=noise, size=(2, 2)))
            labels.append(axis)
    return np.array(segs), np.array(labels)
