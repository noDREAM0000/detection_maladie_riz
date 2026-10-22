# Mémoire M1 I2AD — Détection des maladies du riz par YOLO11n-cls

Code des expériences du chapitre 5 : réentraînement de YOLO11n-cls sur 4 classes
(pyriculariose, BLB, brunissure, saine), test externe et effet de la résolution.

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate   
pip install -r requirements.txt
```
Avec un GPU NVIDIA, installez d'abord PyTorch avec CUDA depuis https://pytorch.org.

## 2. Données

**Paddy Doctor (entraînement).** Sur Kaggle, rejoignez la compétition
*Paddy Doctor: Paddy Disease Classification*, puis téléchargez les données
(`kaggle competitions download -c paddy-disease-classification`) et décompressez-les.
Le dossier doit contenir `train.csv` et `train_images/<label>/<image_id>.jpg`.

**Dhan-Shomadhan (test externe).** Téléchargez le jeu depuis Mendeley Data
(https://doi.org/10.17632/znsxdctwtt.1) et rangez les images ainsi :

```
data/dhan/champ/blast/…        data/dhan/fond_blanc/blast/…
data/dhan/champ/brown_spot/…   data/dhan/fond_blanc/brown_spot/…
```
Les noms `blast` et `brown_spot` doivent être exactement ceux-là. Les autres
maladies du jeu (leaf scald, tungro, sheath blight) ne sont pas utilisées.

## 3. Lancer les expériences

```bash
python prepare_data.py --paddy-dir chemin/paddy-disease-classification   # pHash + découpage 70/15/15 par groupe
python train.py --data data/paddy4 --device 0                             # 3 graines ; --device cpu sans GPU
python evaluate.py --data data/paddy4 --external data/dhan                # E1, E2, E3 + décision H1-H3
mlflow ui --backend-store-uri mlruns                                      # consulter les runs
```

## 4. Ce que produit chaque script

| Script | Sorties | Dans le mémoire |
| --- | --- | --- |
| `prepare_data.py` | `reports/splits.csv`, `reports/eda_*.csv` | Tableau 5.1, section 5.3 |
| `train.py` | `runs/cls/seed{0,1,2}/` (courbes, best.pt) | Chapitre 6 |
| `evaluate.py` | `results/resume.md`, `e1_*`, `e2_externe.csv`, `e3_resolution.csv/.png` | Chapitre 7 |

`results/resume.md` contient les tableaux prêts à copier et le verdict de chaque
hypothèse, calculé avec les seuils du Tableau 5.3 (H1 : F1-macro ≥ 0,90 ;
H2 : chute ≥ 15 points ; H3 : perte > 10 points). Ne modifiez pas ces seuils
après avoir vu les résultats.

## 5. Durée indicative

YOLO11n-cls est petit (1,6 M de paramètres) : sur un GPU grand public, un
entraînement de 50 époques sur ~5 000 images prend de l'ordre de quelques
dizaines de minutes. Sur CPU, comptez plusieurs heures par graine : utilisez
alors Colab (Exécution → Modifier le type d'exécution → GPU).

## 6. Prototype d'API (section 3.4)

```bash
pip install -r requirements.txt
python test_api.py --data data/paddy4 --n 100     # tests + temps de réponse (à reporter dans le mémoire)
uvicorn api:app --port 8000                        # puis ouvrir http://localhost:8000/docs
docker build -t riz-api . && docker run -p 8000:8000 riz-api   # optionnel, nécessite Docker Desktop
```

## 7. Démonstrateur matériel : ESP32-S3 + OV2640 → Raspberry Pi 5

**Sur le Raspberry Pi 5** (Raspberry Pi OS 64 bits) :
```bash
sudo apt install -y python3-venv libgl1
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt onnx onnxruntime onnxslim
# copier depuis le PC : runs/cls/seed0/weights/best.pt et data/paddy4/test/
python benchmark_pi.py --n 100                 # temps d'inférence sur le Pi (pour le mémoire)
uvicorn api:app --host 0.0.0.0 --port 8000     # l'API, accessible depuis le réseau
hostname -I                                     # adresse IP à mettre dans le sketch ESP32
```

**Sur l'ESP32-S3** : ouvrir `esp32/capture_envoi/capture_envoi.ino` dans l'Arduino IDE
et suivre les 5 étapes en tête du fichier. Le moniteur série (115200 bauds) affiche
la réponse de l'API pour chaque photo. L'ESP32 et le Pi doivent être sur le même Wi-Fi.
