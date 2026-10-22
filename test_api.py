from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api import app
from common import list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/paddy4"))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    client = TestClient(app)

    # 1. Tests fonctionnels
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["statut"] == "ok"
    r = client.post("/predict", files={"file": ("x.txt", b"pas une image", "text/plain")})
    assert r.status_code == 400, "un fichier non image doit être refusé"
    print("Tests fonctionnels : OK (/health, refus d'un fichier non image)")

    # 2. Temps de réponse et exactitude sur un échantillon du test
    items = list_images(args.data / "test")
    random.Random(0).shuffle(items)
    items = items[: args.n]
    client.post("/predict", files={"file": ("w.jpg", items[0][0].read_bytes(), "image/jpeg")})  # préchauffage
    total, infer, ok, unsure = [], [], 0, 0
    for path, label in items:
        t0 = time.perf_counter()
        r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
        total.append((time.perf_counter() - t0) * 1000)
        body = r.json()
        infer.append(body["temps_inference_ms"])
        ok += body["classe"] == {"blast": "Pyriculariose", "bacterial_leaf_blight": "BLB",
                                 "brown_spot": "Brunissure", "normal": "Saine"}[label]
        unsure += body["incertaine"]
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    print(f"\n{len(items)} requêtes")
    print(f"Temps de réponse total (ms) : médiane {statistics.median(total):.1f}, 95e centile {q(total, .95):.1f}")
    print(f"Temps d'inférence seul (ms) : médiane {statistics.median(infer):.1f}")
    print(f"Exactitude sur l'échantillon : {ok / len(items):.3f}")
    print(f"Prédictions marquées incertaines : {unsure} ({unsure / len(items):.1%})")


if __name__ == "__main__":
    main()
