from __future__ import annotations

import argparse
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from common import CLASSES, list_images, load_square


def bench(fn, imgs):
    fn(imgs[0])  # préchauffage
    times, preds = [], []
    for im in imgs:
        t0 = time.perf_counter()
        preds.append(fn(im))
        times.append((time.perf_counter() - t0) * 1000)
    return times, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/cls/seed0/weights/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/paddy4"))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    from ultralytics import YOLO

    items = list_images(args.data / "test")
    random.Random(0).shuffle(items)
    items = items[: args.n]
    imgs = [load_square(p) for p, _ in items]
    truth = [CLASSES.index(c) for _, c in items]

    model = YOLO(str(args.weights))
    names = model.names
    to_ours = {i: CLASSES.index(n) for i, n in names.items()}

    results = {}
    # 1. PyTorch sur CPU
    results["PyTorch (.pt)"] = bench(
        lambda im: to_ours[int(model.predict(im, imgsz=224, device="cpu", verbose=False)[0].probs.top1)], imgs)

    # 2. ONNX (FP32), puis ONNX quantifié INT8
    import onnxruntime as ort
    onnx_path = Path(model.export(format="onnx", imgsz=224, simplify=True))

    def make_ort(path: Path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

        def run(im):
            x = im[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # BGR -> RGB, NCHW
            return to_ours[int(sess.run(None, {inp: x})[0].argmax())]
        return run

    results["ONNX FP32"] = bench(make_ort(onnx_path), imgs)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
        results["ONNX INT8 (dynamique)"] = bench(make_ort(q_path), imgs)
    except Exception as exc:  # la quantification peut échouer selon les versions
        print(f"Quantification INT8 ignorée : {exc}")

    print(f"\nMachine : {platform.machine()} — {platform.platform()}")
    print(f"{len(imgs)} images du test interne, prétraitement exclu\n")
    print("| Format | Médiane (ms) | 95e centile (ms) | Images/s | Exactitude |")
    print("| --- | --- | --- | --- | --- |")
    for name, (t, p) in results.items():
        med = statistics.median(t)
        p95 = sorted(t)[int(0.95 * (len(t) - 1))]
        acc = float(np.mean([a == b for a, b in zip(p, truth)]))
        print(f"| {name} | {med:.1f} | {p95:.1f} | {1000 / med:.1f} | {acc:.3f} |")


if __name__ == "__main__":
    main()
