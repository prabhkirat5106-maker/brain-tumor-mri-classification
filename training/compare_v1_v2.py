"""Compare v1 and v2 on the Kaggle test set and the independent PMRAM images, with calibration.

    python training/compare_v1_v2.py --kaggle <dataset folder> --pmram <PMRAM Raw folder>

- v1: notebook model, stretch-resize to 224 (demo/weights/resnet18_best.pt)
- v2: head crop + stronger augmentation (demo/weights/resnet18_v2.pt)
- temperature scaling: one scalar T per model, fitted on the Kaggle *validation* set only,
  then probabilities = softmax(logits / T). It never changes which class wins, only how sure.
- PMRAM independent set: same definition as external_test/evaluate_pmram.py
  (unique by MD5, 256-bit dHash distance >= 50 from every Kaggle image)

Writes training/v2_results.json and prints a summary.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent / "demo"))
from head_crop import crop_to_head  # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
FOLDERS = {"512Glioma": "glioma", "512Meningioma": "meningioma", "512Normal": "notumor", "512Pituitary": "pituitary"}
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dhash256(path):
    g = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def load(name):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 4)
    net.load_state_dict(torch.load(ROOT / "demo" / "weights" / name, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def logits_for(net, paths, crop):
    out = []
    for i in range(0, len(paths), 64):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
        x = torch.stack([TF(crop_to_head(im) if crop else im) for im in imgs]).to(device)
        out.append(net(x).cpu())
    return torch.cat(out)


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def metrics(logits, y, t=1.0):
    p = (logits / t).softmax(1)
    conf, pred = p.max(1)
    right = pred == y
    bins = torch.linspace(0, 1, 16)
    ece = sum((right[(conf > lo) & (conf <= hi)].float().mean() - conf[(conf > lo) & (conf <= hi)].mean()).abs()
              * ((conf > lo) & (conf <= hi)).float().mean()
              for lo, hi in zip(bins[:-1], bins[1:]) if ((conf > lo) & (conf <= hi)).any())
    rec = {c: round(float(right[y == k].float().mean()), 4) for k, c in enumerate(CLASSES) if (y == k).any()}
    return {"n": len(y), "accuracy": round(float(right.float().mean()), 4), "recall": rec,
            "ece": round(float(ece), 4), "nll": round(float(nn.functional.cross_entropy(logits / t, y)), 4),
            "wrong": int((~right).sum()), "wrong_over_90pct": int(((~right) & (conf > 0.9)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True, type=Path)
    ap.add_argument("--pmram", required=True, type=Path)
    a = ap.parse_args()

    split = json.loads((HERE / "split.json").read_text())
    sets = {}
    for name in ("val", "test"):
        sets[name] = ([a.kaggle / p for p, _ in split[name]], torch.tensor([CLASSES.index(l) for _, l in split[name]]))

    seen, pm = set(), []
    for p in sorted(a.pmram.rglob("*.jpg")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            pm.append((p, FOLDERS[p.parent.name]))
    kag = np.stack([dhash256(p) for p in a.kaggle.rglob("*") if p.is_file()])
    indep = [(p, l) for p, l in pm if BITS[np.bitwise_xor(kag, dhash256(p))].sum(1).min() >= 50]
    sets["pmram"] = ([p for p, _ in indep], torch.tensor([CLASSES.index(l) for _, l in indep]))
    print({k: len(v[1]) for k, v in sets.items()}, flush=True)

    res = {}
    for name, weights, crop in [("v1", "resnet18_best.pt", False), ("v2", "resnet18_v2.pt", True)]:
        net = load(weights)
        lg = {k: logits_for(net, paths, crop) for k, (paths, _) in sets.items()}
        t = fit_temperature(lg["val"], sets["val"][1])
        res[name] = {"temperature": round(t, 4)}
        for k in ("test", "pmram"):
            res[name][k] = {"raw": metrics(lg[k], sets[k][1]), "calibrated": metrics(lg[k], sets[k][1], t)}
        print(name, json.dumps(res[name], indent=1), flush=True)
    (HERE / "v2_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
