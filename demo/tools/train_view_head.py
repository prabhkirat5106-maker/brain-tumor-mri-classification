"""Train the scan-view and sagittal-facing heads used to place findings on the 3D brain.

    python tools/train_view_head.py "C:/path/to/brain-tumor-mri-dataset"

Both are linear models on the trained ResNet18's 512-d avgpool features, fitted to the
80 hand-labelled test-split scans in tools/view_labels.json:

- view (axial / coronal / sagittal): ResNet features + 3 silhouette features
  (left-right symmetry, tissue at the bottom edge, height/width). Leave-one-out: 75/80 (v2 features; 76/80 on v1).
- facing (sagittal only): features(image) - features(mirrored image), so mirroring a scan
  exactly flips the answer. Trained on both orientations of each scan. Leave-one-out: 21/22 (v2; 20/22 on v1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402

root = Path(sys.argv[1])
labels = json.loads((Path(__file__).parent / "view_labels.json").read_text())["labels"]


def gray_of(im):
    return np.asarray(im.convert("RGB").resize((224, 224)).convert("L"), np.float32) / 255.0


def fit_softmax(X, y, wd=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W = torch.zeros(Z.shape[1], int(y.max()) + 1, requires_grad=True)
    b = torch.zeros(int(y.max()) + 1, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W + b, y) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"mu": mu, "sd": sd, "W": W.detach(), "b": b.detach()}


def fit_facing(D, t, wd=1e-2):
    sd = D.std(0) + 1e-6
    w = torch.zeros(D.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        Z = torch.cat([D / sd, -D / sd])
        tt = torch.cat([t, -t])
        loss = torch.nn.functional.softplus(-tt * (Z @ w)).mean() + wd * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"sd": sd, "w": w.detach()}


Xv, yv, Df, tf = [], [], [], []
for row in labels:
    im = Image.open(root / row["file"]).convert("RGB")
    feat = model.backbone_features(im)
    Xv.append(model.view_inputs(feat, model.geometry_features(gray_of(im))))
    yv.append("ACS".index(row["view"]))
    if row["view"] == "S":
        Df.append(feat - model.backbone_features(ImageOps.mirror(im)))
        tf.append(1.0 if row["facing"] == "right" else -1.0)

Xv, yv = torch.stack(Xv), torch.tensor(yv)
Df, tf = torch.stack(Df), torch.tensor(tf)

# leave-one-out check, then fit on everything
ok = sum(int(((Xv[i] - h["mu"]) / h["sd"] @ h["W"] + h["b"]).argmax()) == int(yv[i])
         for i in range(len(yv))
         for h in [fit_softmax(Xv[torch.arange(len(yv)) != i], yv[torch.arange(len(yv)) != i])])
okf = sum((float((Df[i] / h["sd"]) @ h["w"]) > 0) == (tf[i] > 0)
          for i in range(len(tf))
          for h in [fit_facing(Df[torch.arange(len(tf)) != i], tf[torch.arange(len(tf)) != i])])
print(f"view LOO {ok}/{len(yv)}   facing LOO {okf}/{len(tf)}")

torch.save({"view": fit_softmax(Xv, yv), "facing": fit_facing(Df, tf)}, model.VIEW_HEAD)
print("saved", model.VIEW_HEAD)
