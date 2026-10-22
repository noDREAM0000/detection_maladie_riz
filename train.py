from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import IMG_SIZE, SEEDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/paddy4"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--device", default=None, help="ex. 0 pour le GPU, cpu sinon")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # Suivi MLflow intégré d'Ultralytics (actif si mlflow est installé)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{Path('mlruns').resolve()}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "memoire-a1-entrainement")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow récent : autorise le stockage ./mlruns

    from ultralytics import YOLO

    for seed in args.seeds:
        os.environ["MLFLOW_RUN"] = f"yolo11n-cls_seed{seed}"
        model = YOLO("yolo11n-cls.pt")  # poids pré-entraînés ImageNet
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=IMG_SIZE,
            batch=args.batch,
            seed=seed,
            deterministic=True,
            device=args.device,
            workers=args.workers,
            project=str(Path("runs/cls").resolve()),  # chemin absolu : sinon Ultralytics ajoute runs/classify/
            name=f"seed{seed}",
            exist_ok=True,
            plots=True,
        )
        print(f"Graine {seed} terminée : runs/cls/seed{seed}/weights/best.pt")


if __name__ == "__main__":
    main()
