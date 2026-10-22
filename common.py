from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ["blast", "bacterial_leaf_blight", "brown_spot", "normal"]
CLASS_FR = {
    "blast": "Pyriculariose",
    "bacterial_leaf_blight": "BLB",
    "brown_spot": "Brunissure",
    "normal": "Saine",
}
IMG_SIZE = 224
SEEDS = [0, 1, 2]
SCALES = [1, 1.5, 2, 3, 4, 6, 8]  # facteurs de réduction de la résolution (E3)
REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[tuple[Path, str]]:
    """Retourne (chemin, classe) pour un dossier organisé en <folder>/<classe>/<image>."""
    items = []
    for cdir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                items.append((f, cdir.name))
    return items


def load_square(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible : {path}")
    h, w = img.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    img = img[y0 : y0 + m, x0 : x0 + m]
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def degrade(img224: np.ndarray, s: float) -> np.ndarray:
    if s == 1:
        return img224
    small = max(1, round(IMG_SIZE / s))
    low = cv2.resize(img224, (small, small), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
