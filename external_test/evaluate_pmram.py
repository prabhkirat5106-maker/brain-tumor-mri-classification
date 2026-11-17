"""External test: the ResNet18 on PMRAM, a brain-MRI dataset from Bangladeshi hospitals.

    python external_test/evaluate_pmram.py --pmram "PMRAM/Raw" --kaggle "brain-tumor-mri-dataset"

PMRAM: https://data.mendeley.com/datasets/m7w55sw88b/1 (CC BY 4.0), "Raw Data.zip".
Needs demo/weights/resnet18_best.pt (saved by the notebook).

Many PMRAM images turn out to be the same scans as images in the Kaggle training data,
re-cropped or re-sharpened, so exact MD5 hashing misses them. Steps:

1. drop exact duplicates inside PMRAM (MD5)
2. compare every PMRAM image with every Kaggle image using a 256-bit difference hash
   (16x16 dHash). Inspecting pairs by distance showed: < 40 = same scan, >= 50 = different scan.
3. score the model separately on "same scan" and "independent" images (distance >= 50)

Writes external_test/results.json and figures/external_confusion_matrix.png.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import model  # noqa: E402  (the demo's model.py: same ResNet18 + preprocessing as the notebook)

FOLDERS = {"512Glioma": "glioma", "512Meningioma": "meningioma", "512Normal": "notumor", "512Pituitary": "pituitary"}
SAME, INDEPENDENT = 40, 50
BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def dhash256(path):
    g = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmram", required=True, type=Path, help="extracted Raw folder with the 512* class folders")
    ap.add_argument("--kaggle", required=True, type=Path, help="Kaggle dataset folder used for training")
    args = ap.parse_args()

    files, seen = [], set()
    for p in sorted(args.pmram.rglob("*.jpg")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            files.append((p, FOLDERS[p.parent.name]))
    print(f"PMRAM: {len(seen)} unique of {sum(1 for _ in args.pmram.rglob('*.jpg'))} images")

    kaggle = np.stack([dhash256(p) for p in args.kaggle.rglob("*") if p.is_file()])
    dist = np.array([BITS[np.bitwise_xor(kaggle, dhash256(p))].sum(1).min() for p, _ in files])

    y, pred, conf = [], [], []
    with torch.no_grad():
        for p, label in files:
            x = model.eval_transform(Image.open(p).convert("RGB")).unsqueeze(0).to(model.device)
            prob = model.model(x).softmax(1)[0].cpu().numpy()
            y.append(model.CLASSES.index(label))
            pred.append(int(prob.argmax()))
            conf.append(float(prob.max()))
    y, pred, conf = np.array(y), np.array(pred), np.array(conf)

    def score(mask):
        cm = np.zeros((4, 4), int)
        for t, q in zip(y[mask], pred[mask]):
            cm[t, q] += 1
        wrong = (y != pred) & mask
        return {
            "n": int(mask.sum()), "accuracy": round(float((y == pred)[mask].mean()), 4), "confusion": cm.tolist(),
            "recall": {c: (round(cm[k, k] / cm[k].sum(), 4) if cm[k].sum() else None) for k, c in enumerate(model.CLASSES)},
            "wrong": int(wrong.sum()), "wrong_over_90pct_confident": int((conf[wrong] > 0.9).sum()),
        }

    res = {"same_scan_as_kaggle": score(dist < SAME), "independent": score(dist >= INDEPENDENT),
           "between_cutoffs_excluded": int(((dist >= SAME) & (dist < INDEPENDENT)).sum())}
    (ROOT / "external_test" / "results.json").write_text(json.dumps(res, indent=1))
    for k in ("same_scan_as_kaggle", "independent"):
        r = res[k]
        print(f"{k}: {r['n']} images, accuracy {r['accuracy']:.1%}, recall {r['recall']}, "
              f"wrong {r['wrong']} ({r['wrong_over_90pct_confident']} at >90% confidence)")

    # confusion matrix for the independent images, styled like the notebook's
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(res["independent"]["confusion"])
    rows = [k for k in range(4) if cm[k].sum()]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.imshow(cm[rows], cmap="Blues")
    for i, k in enumerate(rows):
        for j in range(4):
            ax.text(j, i, cm[k, j], ha="center", va="center", color="white" if cm[k, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), model.CLASSES)
    ax.set_yticks(range(len(rows)), [model.CLASSES[k] for k in rows])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"ResNet18 on PMRAM, independent images (n={res['independent']['n']})")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "external_confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
