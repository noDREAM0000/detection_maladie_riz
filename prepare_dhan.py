import shutil, sys
from pathlib import Path

src = Path(sys.argv[1])                      
dst = Path("data/dhan")
backgrounds = {"field": "champ", "white": "fond_blanc"}
diseases = {"blast": "blast", "brown": "brown_spot", "browon": "brown_spot"}   # "Browon" : faute dans le jeu d'origine  # mot-clé -> nom de classe Paddy Doctor
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for bg_dir in [d for d in src.iterdir() if d.is_dir()]:
    bg = next((v for k, v in backgrounds.items() if k in bg_dir.name.lower()), None)
    if not bg:
        continue
    for cls_dir in [d for d in bg_dir.iterdir() if d.is_dir()]:
        print(f"{bg_dir.name} / {cls_dir.name}", end="")
        cls = next((v for k, v in diseases.items() if k in cls_dir.name.lower()), None)
        if not cls:
            print("  -> ignoré")
            continue
        out = dst / bg / cls
        out.mkdir(parents=True, exist_ok=True)
        files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in exts]
        for f in files:
            shutil.copy2(f, out / f"{cls_dir.name}_{f.name}")
        print(f"  -> {out} ({len(files)} images)")