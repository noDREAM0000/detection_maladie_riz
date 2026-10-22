from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO

from cartographie import analyser
from common import CLASS_FR, IMG_SIZE

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "runs/cls/seed0/weights/best.pt"))
CONF_MIN = float(os.environ.get("CONF_MIN", "0.6"))  # sous ce seuil : prédiction « incertaine » (BF3)
DATA_DIR = Path(os.environ.get("DATA_DIR", "analyses"))
DATA_DIR.mkdir(exist_ok=True)
DB = DATA_DIR / "historique.db"
AVERTISSEMENT = "Suspicion de symptôme, à confirmer par un technicien ou un agronome."

app = FastAPI(title="Détection des maladies du riz", version="0.2.0",
              description="Classe des images de feuilles de riz (pyriculariose, BLB, brunissure, saine) "
                          "et cartographie les zones suspectes d'une image de parcelle.")
model = YOLO(str(MODEL_PATH))
MODEL_VERSION = f"{MODEL_PATH.parent.parent.name}/{MODEL_PATH.name}"
MULTI_PATH = Path(os.environ.get("MULTI_PATH", "modeles/multi_cultures.pt"))


class _Probs:
    def __init__(self, p): self.data = p
class _Res:
    def __init__(self, p): self.probs = _Probs(p)


class TeteManioc:
    """Tronc riz gelé + tête manioc (voir manioc.py) ; même interface que YOLO.predict."""
    def __init__(self, path: Path):
        import torch
        from manioc import CASSAVA_FR, backbone
        self.torch = torch
        d = torch.load(path)
        self.embed, _ = backbone(d["tronc"], "cpu")
        t = d["tetes"]["manioc"]
        self.W, self.b = t["W"], t["b"]
        self.names = {i: CASSAVA_FR[c] for i, c in enumerate(t["classes"])}

    def predict(self, imgs, **_):
        imgs = imgs if isinstance(imgs, list) else [imgs]
        x = self.torch.from_numpy(self.embed(imgs))
        p = self.torch.softmax(x @ self.W.T + self.b, 1)
        return [_Res(r) for r in p]


MODELES = {"riz": model}
if MULTI_PATH.exists():
    MODELES["manioc"] = TeteManioc(MULTI_PATH)


def choisir(culture: str):
    if culture not in MODELES:
        raise HTTPException(400, f"Culture inconnue ou tête non installée : {culture}. Disponibles : {list(MODELES)}")
    return MODELES[culture]


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, parcelle TEXT, source TEXT,
        lat REAL, lon REAL, surface_suspecte REAL, classe_dominante TEXT,
        surfaces TEXT, version_modele TEXT, duree_ms REAL)""")
    return con


def lire_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Fichier illisible : envoyez une image JPEG ou PNG.")
    return img


def to_square_224(img: np.ndarray) -> np.ndarray:
    """Même prétraitement que l'évaluation : recadrage carré centré puis 224 px."""
    h, w = img.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    return cv2.resize(img[y0:y0 + m, x0:x0 + m], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


@app.get("/health")
def health() -> dict:
    return {"statut": "ok", "version_modele": MODEL_VERSION, "cultures": list(MODELES), "seuil_confiance": CONF_MIN}


@app.post("/predict")
async def predict(file: UploadFile = File(...), culture: str = Form("riz")) -> dict:
    """Classe une image entière (une feuille) ; culture = riz ou manioc."""
    m = choisir(culture)
    img = lire_image(await file.read())
    t0 = time.perf_counter()
    r = m.predict(to_square_224(img), imgsz=IMG_SIZE, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    probs = r.probs.data.cpu().numpy()
    top = int(probs.argmax())
    conf = float(probs[top])
    return {
        "culture": culture,
        "classe": CLASS_FR.get(m.names[top], m.names[top]),
        "confiance": round(conf, 4),
        "incertaine": conf < CONF_MIN,
        "probabilites": {CLASS_FR.get(m.names[i], m.names[i]): round(float(p), 4) for i, p in enumerate(probs)},
        "version_modele": MODEL_VERSION,
        "temps_inference_ms": round(ms, 1),
        "avertissement": AVERTISSEMENT,
    }


@app.post("/analyse")
async def analyse(file: UploadFile = File(...), parcelle: str = Form("Parcelle demo"),
                  source: str = Form("web"), tuiles: int = Form(4),
                  lat: float | None = Form(None), lon: float | None = Form(None),
                  culture: str = Form("riz")) -> dict:
    """Découpe l'image en tuiles, classe chacune, produit la carte et l'enregistre dans l'historique."""
    img = lire_image(await file.read())
    t0 = time.perf_counter()
    out = analyser(choisir(culture), img, tuiles_par_cote=max(1, min(tuiles, 12)), conf_min=CONF_MIN, lat=lat, lon=lon)
    ms = (time.perf_counter() - t0) * 1000
    surfaces = out["surfaces_pourcent"]
    dominante = max(surfaces, key=surfaces.get)
    con = db()
    cur = con.execute(
        "INSERT INTO analyses (date, parcelle, source, lat, lon, surface_suspecte, classe_dominante, surfaces, version_modele, duree_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), f"{parcelle} ({culture})", source, lat, lon,
         out["surface_suspecte_pourcent"], dominante, json.dumps(surfaces), MODEL_VERSION, round(ms, 1)))
    con.commit()
    aid = cur.lastrowid
    con.close()
    cv2.imwrite(str(DATA_DIR / f"{aid}_image.jpg"), img)
    cv2.imwrite(str(DATA_DIR / f"{aid}_carte.png"), out["carte"])
    if out["geojson"]:
        (DATA_DIR / f"{aid}.geojson").write_text(json.dumps(out["geojson"]), encoding="utf-8")
    return {"id": aid, "parcelle": parcelle, "surface_suspecte_pourcent": out["surface_suspecte_pourcent"],
            "surfaces_pourcent": surfaces, "grille": out["grille"], "tuiles": out["tuiles"],
            "carte_url": f"/analyses/{aid}/carte.png", "geojson_url": f"/analyses/{aid}/geojson" if out["geojson"] else None,
            "version_modele": MODEL_VERSION, "duree_ms": round(ms, 1), "avertissement": AVERTISSEMENT}


@app.get("/analyses")
def historique(limite: int = 50) -> list[dict]:
    con = db()
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limite,))]
    con.close()
    for r in rows:
        r["surfaces"] = json.loads(r["surfaces"])
    return rows


@app.get("/analyses/compte")
def compte() -> dict:
    con = db()
    n = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    con.close()
    return {"total": n}


@app.get("/analyses/{aid}/carte.png")
def carte(aid: int):
    p = DATA_DIR / f"{aid}_carte.png"
    if not p.exists():
        raise HTTPException(404, "Analyse inconnue")
    return FileResponse(p, media_type="image/png")


@app.get("/analyses/{aid}/geojson")
def geojson(aid: int):
    p = DATA_DIR / f"{aid}.geojson"
    if not p.exists():
        raise HTTPException(404, "Pas de position GPS pour cette analyse")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/", response_class=HTMLResponse)
def tableau_de_bord() -> str:
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
