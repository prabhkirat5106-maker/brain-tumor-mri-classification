"""Export everything the browser version (web/) needs to run without Python.

    python tools/export_web.py

Writes:
  web/model/neurolens.onnx  ResNet18 with three outputs: logits, layer4 activations, pooled features
  web/model/heads.json      fc weights (for CAM) + the view / facing heads

Grad-CAM on layer4 of a ResNet equals the plain class activation map: the logit is
fc(mean(A)), so the gradient w.r.t. A is fc.weight[c] / 49 at every position. The
browser computes ReLU(sum_k fc.weight[c, k] * A_k) directly, with no gradients needed.
"""
import json
import sys
from pathlib import Path

import torch

torch.cuda.is_available = lambda: False  # export on CPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402

OUT = model.HERE / "web" / "model"
OUT.mkdir(parents=True, exist_ok=True)
net = model.model.cpu().eval()


class WithFeatures(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        n = self.n
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        acts = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        pooled = torch.flatten(n.avgpool(acts), 1)
        return n.fc(pooled), acts, pooled


torch.onnx.export(WithFeatures(net), torch.randn(1, 3, 224, 224), OUT / "neurolens.onnx",
                  input_names=["x"], output_names=["logits", "acts", "pooled"],
                  opset_version=17, dynamo=False)

heads = torch.load(model.VIEW_HEAD, map_location="cpu")
r = lambda t: [round(float(v), 6) for v in t.flatten()]  # noqa: E731
json.dump({
    "classes": model.CLASSES,
    "fc_weight": r(net.fc.weight.detach()),          # [4 * 512]
    "view": {"mu": r(heads["view"]["mu"]), "sd": r(heads["view"]["sd"]),
             "W": r(heads["view"]["W"]), "b": r(heads["view"]["b"]),   # W: [536 * 3] row-major
             "views": model.VIEWS},
    "facing": {"sd": r(heads["facing"]["sd"]), "w": r(heads["facing"]["w"])},
}, open(OUT / "heads.json", "w"))
print("wrote", *(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in OUT.iterdir()))
