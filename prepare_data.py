from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from common import CLASSES, CLASS_FR, REPORTS_DIR

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def phash_int(path: Path) -> int:
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"))  # 64 bits
    return int(str(h), 16)


def group_near_duplicates(hashes: np.ndarray, max_dist: int) -> np.ndarray:
    n = len(hashes)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = hashes.astype(np.uint64)
    for i in range(n - 1):
        x = np.bitwise_xor(h[i], h[i + 1 :])
        # popcount vectorisé sur 64 bits
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= max_dist)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(n)])


def split_groups(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = random.Random(seed)
    assignment = {}
    for _, cdf in df.groupby("label"):
        groups = list(cdf.groupby("group").size().items())
        rng.shuffle(groups)
        total = sum(n for _, n in groups)
        targets = {k: v * total for k, v in SPLITS.items()}
        filled = {k: 0 for k in SPLITS}
        for g, n in groups:
            # on remplit le sous-ensemble le plus en retard sur sa cible
            k = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / targets[s])
            assignment[g] = k
            filled[k] += n
    return df["group"].map(assignment)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paddy-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/paddy4"))
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = pd.read_csv(args.paddy_dir / "train.csv")
    raw_counts = meta["label"].value_counts()
    df = meta[meta["label"].isin(CLASSES)].copy()
    df["path"] = [args.paddy_dir / "train_images" / l / i for l, i in zip(df["label"], df["image_id"])]
    missing = [p for p in df["path"] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images introuvables, ex. {missing[0]}")

    print(f"{len(df)} images retenues sur {len(meta)} ; calcul des pHash…")
    df["phash"] = [phash_int(p) for p in df["path"]]

    df["group"] = ""
    for label, idx in df.groupby("label").groups.items():
        sub = df.loc[idx]
        roots = group_near_duplicates(sub["phash"].to_numpy(np.uint64), args.max_hamming)
        df.loc[idx, "group"] = [f"{label}_{r}" for r in roots]
    df["split"] = split_groups(df, args.seed)

    # Copie au format Ultralytics
    if args.out.exists():
        shutil.rmtree(args.out)
    for row in df.itertuples():
        dst = args.out / row.split / row.label / row.image_id
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.path, dst)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["path"]).to_csv(REPORTS_DIR / "splits.csv", index=False)

    # --- EDA (Tableau 5.1 et compléments) ---
    gsize = df.groupby("group").size()
    dup_groups = gsize[gsize > 1]
    table = pd.DataFrame({
        "Paddy Doctor (brut)": raw_counts.reindex(CLASSES),
        "Groupes de quasi-doublons (>1 image)": df[df["group"].isin(dup_groups.index)].groupby("label")["group"].nunique().reindex(CLASSES).fillna(0).astype(int),
        "Images dans ces groupes": df[df["group"].isin(dup_groups.index)].groupby("label").size().reindex(CLASSES).fillna(0).astype(int),
        **{s.capitalize(): df[df["split"] == s].groupby("label").size().reindex(CLASSES).fillna(0).astype(int) for s in SPLITS},
    })
    table.index = [CLASS_FR[c] for c in table.index]
    table.to_csv(REPORTS_DIR / "eda_effectifs.csv")
    pd.crosstab(df["label"], df["variety"]).to_csv(REPORTS_DIR / "eda_varietes.csv")
    df.groupby("label")["age"].describe().to_csv(REPORTS_DIR / "eda_age.csv")

    print("\nTableau 5.1 — effectifs\n")
    print(table.to_markdown())
    print(f"\nFichiers écrits dans {REPORTS_DIR}/ et images copiées dans {args.out}/")


if __name__ == "__main__":
    main()
