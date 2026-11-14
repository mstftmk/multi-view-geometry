"""Download the evaluation datasets (about 15 MB). The data is not part of this repository.

    python scripts/download_data.py --out data

* Oxford affine-covariant regions benchmark (K. Mikolajczyk et al., "A comparison of
  affine region detectors", IJCV 2005): image sequences "graf" (viewpoint change)
  and "boat" (rotation and zoom), each with ground-truth homographies.
  https://www.robots.ox.ac.uk/~vgg/research/affine/
* Middlebury multi-view stereo, "templeSparseRing" (S. Seitz et al., "A comparison
  and evaluation of multi-view stereo reconstruction algorithms", CVPR 2006):
  16 views with calibrated camera matrices.
  https://vision.middlebury.edu/mview/data/
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "graf": "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz",
    "boat": "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/boat.tar.gz",
    "templeSparseRing": "https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip",
}


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract without letting entries escape ``dest`` (no absolute paths or '..')."""
    dest = dest.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (dest / name).resolve().is_relative_to(dest):
                    raise ValueError(f"unsafe path in archive: {name}")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not (dest / m.name).resolve().is_relative_to(dest) or m.issym() or m.islnk():
                    raise ValueError(f"unsafe entry in archive: {m.name}")
            t.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    for name, url in SOURCES.items():
        target = args.out / name
        if target.exists() and any(target.iterdir()):
            print(f"{name}: already present")
            continue
        target.mkdir(parents=True, exist_ok=True)
        archive = args.out / Path(url).name
        print(f"{name}: downloading {url}")
        # some servers refuse Python's default user agent
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, archive.open("wb") as f:
            f.write(resp.read())
        safe_extract(archive, target)
        archive.unlink()
        print(f"{name}: {sum(1 for _ in target.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
