"""Train ResNet18 v2: same recipe as the notebook, plus head crop and stronger augmentation.

    python training/train_v2.py "C:/path/to/brain-tumor-mri-dataset"

Unchanged from v1: split (training/split.json, from make_split.py), ImageNet ResNet18 with a
4-class head, Adam lr 1e-4, batch 32, 10 epochs, keep the epoch with the lowest validation loss.

Changed:
- every image is cropped to the head and padded to a square first (head_crop.py)
- training augmentation adds random zoom/framing, brightness/contrast, blur and sharpening

Saves demo/weights/resnet18_v2.pt and training/v2_history.json.
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "demo"))
from head_crop import crop_to_head, head_box  # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2.5, p=0.3),
    transforms.ToTensor(),
    NORM,
])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


class MRIDataset(Dataset):
    def __init__(self, root, items, boxes, tf):
        self.root, self.items, self.boxes, self.tf = root, items, boxes, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = crop_to_head(Image.open(self.root / path), self.boxes[path])
        return self.tf(img), CLASSES.index(label)


def main():
    root = Path(sys.argv[1])
    split = json.loads((HERE / "split.json").read_text())
    t0 = time.time()
    boxes = {p: head_box(Image.open(root / p).convert("RGB")) for s in split.values() for p, _ in s}
    print(f"head boxes for {len(boxes)} images in {time.time() - t0:.0f}s", flush=True)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = MRIDataset(root, split["train"], boxes, train_tf)
    val_ds = MRIDataset(root, split["val"], boxes, eval_tf)
    kw = dict(batch_size=32, num_workers=4, persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(512, 4)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state, history = float("inf"), None, []
    for epoch in range(1, 11):
        net.train()
        run = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
        net.eval()
        vl, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = net(x)
                vl += loss_fn(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
        tr, vl, va = run / len(train_ds), vl / len(val_ds), correct / len(val_ds)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl, "val_acc": va})
        print(f"epoch {epoch}: train loss {tr:.3f} | val loss {vl:.3f} | val acc {va:.1%} | {time.time() - t0:.0f}s",
              flush=True)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch

    out = HERE.parent / "demo" / "weights" / "resnet18_v2.pt"
    torch.save(best_state, out)
    (HERE / "v2_history.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_loss": best,
                                                       "history": history}, indent=1))
    print(f"best epoch {best_epoch}, val loss {best:.3f} -> {out}")


if __name__ == "__main__":
    main()
