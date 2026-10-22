from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from common import CLASS_FR, CLASSES, RESULTS_DIR, SCALES, SEEDS, degrade, list_images, load_square

H1_F1 = 0.90       # H1 : F1-macro moyen >= 0,90
H2_DROP = 0.15     # H2 : rappel externe inférieur d'au moins 15 points
H3_DROP = 0.10     # H3 : perte de plus de 10 points de F1-macro
N_BOOT = 1000


def predict(model, imgs: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """Retourne l'indice (dans CLASSES) de la classe prédite pour chaque image BGR 224x224."""
    names = model.names  # {indice_modèle: nom}
    to_ours = {i: CLASSES.index(n) for i, n in names.items()}
    out = []
    for k in range(0, len(imgs), batch):
        res = model.predict(imgs[k : k + batch], imgsz=224, verbose=False)
        out.extend(to_ours[int(r.probs.top1)] for r in res)
    return np.array(out)


def bootstrap_f1(y, p, rng) -> tuple[float, float]:
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y[idx], p[idx], average="macro", labels=range(len(CLASSES)), zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class_table(y, p) -> pd.DataFrame:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(CLASSES)), zero_division=0)
    return pd.DataFrame({"precision": pr, "rappel": rc, "f1": f1, "effectif": sup}, index=[CLASS_FR[c] for c in CLASSES])


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    labels = [CLASS_FR[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/paddy4"))
    ap.add_argument("--external", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=Path("runs/cls"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    from ultralytics import YOLO

    # Suivi MLflow : facultatif, l'évaluation ne doit jamais échouer à cause de lui.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow récent : autorise le stockage ./mlruns
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{Path('mlruns').resolve()}"))
        mlflow.set_experiment("memoire-a1-evaluation")
    except Exception as exc:  # module absent ou stockage refusé
        print(f"MLflow désactivé : {exc}")
        mlflow = None

    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    # Chargement une seule fois des images (déjà recadrées en 224)
    test_items = list_images(args.data / "test")
    y_test = np.array([CLASSES.index(c) for _, c in test_items])
    x_test = [load_square(p) for p, _ in test_items]
    ext_sets = {}
    if args.external and args.external.exists():
        for bg in sorted(d for d in args.external.iterdir() if d.is_dir()):
            items = [(p, c) for p, c in list_images(bg) if c in CLASSES]
            ext_sets[bg.name] = (np.array([CLASSES.index(c) for _, c in items]), [load_square(p) for p, _ in items])

    e1_rows, e1_class, e2_rows, e3_rows = [], [], [], []
    for seed in args.seeds:
        model = YOLO(str(args.runs / f"seed{seed}" / "weights" / "best.pt"))
        run = mlflow.start_run(run_name=f"eval_seed{seed}") if mlflow else None

        # ---- E1 : test interne
        p = predict(model, x_test)
        f1m = f1_score(y_test, p, average="macro", labels=range(len(CLASSES)), zero_division=0)
        lo, hi = bootstrap_f1(y_test, p, rng)
        acc = float((p == y_test).mean())
        e1_rows.append({"graine": seed, "f1_macro": f1m, "ic95_bas": lo, "ic95_haut": hi, "exactitude": acc})
        pc = per_class_table(y_test, p)
        pc["graine"] = seed
        e1_class.append(pc)
        cm = confusion_matrix(y_test, p, labels=range(len(CLASSES)))
        pd.DataFrame(cm, index=pc.index, columns=pc.index).to_csv(RESULTS_DIR / f"e1_confusion_seed{seed}.csv")
        plot_confusion(cm, RESULTS_DIR / f"e1_confusion_seed{seed}.png", f"Test interne — graine {seed}")
        internal_recall = dict(zip(CLASSES, pc["rappel"]))

        # ---- E2 : jeu externe (rappel par maladie + répartition des erreurs)
        for bg, (y_ext, x_ext) in ext_sets.items():
            pe = predict(model, x_ext)
            for k, c in enumerate(CLASSES):
                mask = y_ext == k
                if not mask.any():
                    continue
                dist = np.bincount(pe[mask], minlength=len(CLASSES)) / mask.sum()
                e2_rows.append({
                    "graine": seed, "arriere_plan": bg, "classe": CLASS_FR[c], "effectif": int(mask.sum()),
                    "rappel_externe": float(dist[k]), "rappel_interne": float(internal_recall[c]),
                    **{f"predit_{CLASS_FR[cc]}": float(dist[j]) for j, cc in enumerate(CLASSES)},
                })

        # ---- E3 : résolution
        for s in SCALES:
            ps = p if s == 1 else predict(model, [degrade(im, s) for im in x_test])
            row = {"graine": seed, "facteur_s": s, "resolution_px": round(224 / s),
                   "f1_macro": f1_score(y_test, ps, average="macro", labels=range(len(CLASSES)), zero_division=0)}
            _, rc, _, _ = precision_recall_fscore_support(y_test, ps, labels=range(len(CLASSES)), zero_division=0)
            row.update({f"rappel_{CLASS_FR[c]}": float(r) for c, r in zip(CLASSES, rc)})
            e3_rows.append(row)

        if run:
            mlflow.log_params({"graine": seed, "poids": f"seed{seed}/best.pt"})
            mlflow.log_metrics({"e1_f1_macro": f1m, "e1_exactitude": acc})
            for r in [r for r in e3_rows if r["graine"] == seed]:
                mlflow.log_metric("e3_f1_macro", r["f1_macro"], step=int(r["facteur_s"] * 10))
            mlflow.log_artifact(str(RESULTS_DIR / f"e1_confusion_seed{seed}.png"))
            mlflow.end_run()

    # ---------- Agrégation sur les graines
    e1 = pd.DataFrame(e1_rows)
    e1.to_csv(RESULTS_DIR / "e1_global.csv", index=False)
    e1c = pd.concat(e1_class)
    e1c.to_csv(RESULTS_DIR / "e1_par_classe.csv")
    e1c_agg = e1c.groupby(level=0)[["precision", "rappel", "f1"]].agg(["mean", "std"]).reindex([CLASS_FR[c] for c in CLASSES])
    e1c_agg.columns = [f"{m} ({'moyenne' if s == 'mean' else 'écart-type'})" for m, s in e1c_agg.columns]
    e3 = pd.DataFrame(e3_rows)
    e3.to_csv(RESULTS_DIR / "e3_resolution.csv", index=False)
    e3_agg = e3.groupby("facteur_s").mean(numeric_only=True).drop(columns="graine")

    lines = ["# Résultats — générés automatiquement par evaluate.py\n"]
    lines.append("## E1 — Test interne\n")
    lines.append(f"F1-macro : {e1.f1_macro.mean():.3f} ± {e1.f1_macro.std():.3f} (moyenne ± écart-type sur {len(e1)} graines)  ")
    lines.append(f"Exactitude : {e1.exactitude.mean():.3f} ± {e1.exactitude.std():.3f}\n")
    lines.append(e1.round(3).to_markdown(index=False) + "\n")
    lines.append(e1c_agg.round(3).to_markdown() + "\n")

    # Décisions
    h1 = e1.f1_macro.mean() >= H1_F1
    lines.append(f"**H1** (F1-macro moyen ≥ {H1_F1}) : {'VALIDÉE' if h1 else 'REJETÉE'}\n")

    if e2_rows:
        e2 = pd.DataFrame(e2_rows)
        e2.to_csv(RESULTS_DIR / "e2_externe.csv", index=False)
        e2_agg = e2.groupby(["arriere_plan", "classe"]).mean(numeric_only=True).drop(columns="graine")
        e2_agg["ecart_points"] = (e2_agg.rappel_interne - e2_agg.rappel_externe) * 100
        lines.append("## E2 — Jeu externe\n")
        lines.append(e2_agg.round(3).to_markdown() + "\n")
        h2 = bool((e2_agg.ecart_points >= H2_DROP * 100).any())
        lines.append(f"**H2** (chute ≥ {H2_DROP*100:.0f} points pour au moins une maladie) : {'VALIDÉE' if h2 else 'REJETÉE'}\n")

    lines.append("## E3 — Résolution\n")
    lines.append(e3_agg.round(3).to_markdown() + "\n")
    base = e3_agg.loc[1, "f1_macro"]
    below = e3_agg.index[(base - e3_agg.f1_macro) > H3_DROP]
    s_star = float(below.min()) if len(below) else None
    per_class_star = {}
    for c in CLASSES:
        col = f"rappel_{CLASS_FR[c]}"
        b = e3_agg.index[(e3_agg.loc[1, col] - e3_agg[col]) > H3_DROP]
        per_class_star[CLASS_FR[c]] = float(b.min()) if len(b) else None
    lines.append(f"Facteur seuil global s* : {s_star}  ")
    lines.append("Facteurs seuils par classe (perte de rappel > 10 points) : " +
                 ", ".join(f"{k} = {v}" for k, v in per_class_star.items()) + "\n")
    # H3 porte sur les MALADIES : la classe saine est exclue de la comparaison des seuils
    disease_stars = {k: v for k, v in per_class_star.items() if k != CLASS_FR["normal"]}
    distinct = len(set(disease_stars.values())) > 1
    # H3 a deux volets : (a) un seuil existe ; (b) il diffère selon la maladie
    if s_star is not None and distinct:
        h3 = "VALIDÉE"
    elif s_star is not None:
        h3 = "PARTIELLEMENT VALIDÉE (seuil confirmé, mais identique pour les trois maladies)"
    else:
        h3 = "REJETÉE (aucun seuil de perte > 10 points)"
    lines.append(f"**H3** (seuil existant et différent selon la maladie) : {h3}\n")
    (RESULTS_DIR / "resume.md").write_text("\n".join(lines), encoding="utf-8")

    # Figure E3
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = e3_agg["resolution_px"].round().astype(int)
    sd = e3.groupby("facteur_s")["f1_macro"].std().reindex(e3_agg.index).fillna(0)
    ax.errorbar(x, e3_agg.f1_macro, yerr=sd, color="black", lw=2.2, marker="o", capsize=3, label="F1-macro")
    for c in CLASSES:
        ax.plot(x, e3_agg[f"rappel_{CLASS_FR[c]}"], marker=".", lw=1.2, label=f"Rappel — {CLASS_FR[c]}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.invert_xaxis()
    ax.set_xlabel("Résolution effective (pixels de côté) — vers la droite : caméra plus éloignée")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "e3_resolution.png", dpi=200)
    plt.close(fig)

    print((RESULTS_DIR / "resume.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
