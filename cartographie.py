from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_FR, IMG_SIZE

# Couleurs BGR de la carte (saine = vert, maladies = chaud, incertaine = gris)
COULEURS = {
    "Saine": (80, 170, 60),
    "Pyriculariose": (40, 40, 220),
    "BLB": (0, 160, 255),
    "Brunissure": (40, 90, 150),
    "Incertaine": (150, 150, 150),
    "Bactériose (CBB)": (0, 200, 255),
    "Striure brune (CBSD)": (30, 60, 120),
    "Acarien vert (CGM)": (200, 120, 0),
    "Mosaïque (CMD)": (180, 40, 180),
}
MALADIES = ["Pyriculariose", "BLB", "Brunissure", "Bactériose (CBB)", "Striure brune (CBSD)",
            "Acarien vert (CGM)", "Mosaïque (CMD)"]


def analyser(model, img: np.ndarray, tuiles_par_cote: int = 4, conf_min: float = 0.6,
             lat: float | None = None, lon: float | None = None, largeur_sol_m: float = 2.0) -> dict:
    """Classe chaque tuile ; retourne les résultats, la carte (BGR) et le GeoJSON éventuel."""
    h, w = img.shape[:2]
    pas = min(h, w) // tuiles_par_cote          # taille d'une tuile carrée en pixels
    nx, ny = w // pas, h // pas
    crops, positions = [], []
    for j in range(ny):
        for i in range(nx):
            crop = img[j * pas:(j + 1) * pas, i * pas:(i + 1) * pas]
            crops.append(cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
            positions.append((i, j))
    res = model.predict(crops, imgsz=IMG_SIZE, verbose=False)

    tuiles = []
    for (i, j), r in zip(positions, res):
        probs = r.probs.data.cpu().numpy()
        k = int(probs.argmax())
        nom = CLASS_FR.get(model.names[k], model.names[k])
        conf = float(probs[k])
        tuiles.append({"i": i, "j": j, "classe": nom if conf >= conf_min else "Incertaine",
                       "classe_brute": nom, "confiance": round(conf, 4)})

    # Statistiques de surface (chaque tuile a la même surface)
    n = len(tuiles)
    presentes = set(t["classe"] for t in tuiles) | {"Saine", "Incertaine"}
    surfaces = {c: round(100 * sum(t["classe"] == c for t in tuiles) / n, 1) for c in COULEURS if c in presentes}
    suspect = round(sum(surfaces.get(m, 0) for m in MALADIES), 1)

    # Carte de chaleur : tuiles colorées semi-transparentes + grille
    carte = img.copy()
    calque = img.copy()
    for t in tuiles:
        x0, y0 = t["i"] * pas, t["j"] * pas
        cv2.rectangle(calque, (x0, y0), (x0 + pas, y0 + pas), COULEURS[t["classe"]], -1)
    carte = cv2.addWeighted(calque, 0.45, carte, 0.55, 0)
    for t in tuiles:
        x0, y0 = t["i"] * pas, t["j"] * pas
        cv2.rectangle(carte, (x0, y0), (x0 + pas, y0 + pas), (255, 255, 255), 1)
        etiquette = f"{t['classe'][:4]} {t['confiance']:.2f}"  # putText : pas d'accents
        cv2.putText(carte, etiquette, (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, pas / 400), (255, 255, 255), 1, cv2.LINE_AA)

    # GeoJSON approximatif : l'image est supposée couvrir largeur_sol_m mètres au sol
    geojson = None
    if lat is not None and lon is not None:
        m_par_px = largeur_sol_m / w
        dlat = lambda m: m / 111_320
        dlon = lambda m: m / (111_320 * math.cos(math.radians(lat)))
        feats = []
        for t in tuiles:
            # origine = centre de l'image ; y vers le sud
            x0 = (t["i"] * pas - w / 2) * m_par_px
            y0 = (t["j"] * pas - h / 2) * m_par_px
            s = pas * m_par_px
            coins = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
            ring = [[round(lon + dlon(x), 8), round(lat - dlat(y), 8)] for x, y in coins]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"classe": t["classe"], "confiance": t["confiance"]}})
        geojson = {"type": "FeatureCollection", "features": feats}

    return {"tuiles": tuiles, "grille": [nx, ny], "taille_tuile_px": pas,
            "surfaces_pourcent": surfaces, "surface_suspecte_pourcent": suspect,
            "carte": carte, "geojson": geojson}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--weights", type=Path, default=Path("runs/cls/seed0/weights/best.pt"))
    ap.add_argument("--tuiles", type=int, default=4, help="nombre de tuiles sur le petit côté")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    img = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    out = analyser(model, img, args.tuiles, lat=args.lat, lon=args.lon)
    sortie = args.image.with_name(args.image.stem + "_carte.png")
    cv2.imencode(".png", out["carte"])[1].tofile(str(sortie))
    if out["geojson"]:
        args.image.with_name(args.image.stem + ".geojson").write_text(json.dumps(out["geojson"]), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("carte", "geojson", "tuiles")}, ensure_ascii=False, indent=2))
    print(f"Carte : {sortie}")


if __name__ == "__main__":
    main()
