"""Model loading, prediction, Grad-CAM, scan-view detection and 3D placement.

Mirrors the notebook: ResNet18 with a 4-way head, 224x224 resize, ImageNet
normalization, Grad-CAM on layer4[-1]. Grad-CAM is implemented with hooks so
the demo doesn't need the `grad-cam` package.

On top of that, two tiny linear heads (weights/view_head.pt, trained by
tools/train_view_head.py) read the same backbone features to guess the slice
orientation (axial / coronal / sagittal) and, for sagittal slices, which way
the face points. That is what lets us project the 2D Grad-CAM hotspot onto a
3D template brain.
"""
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

HERE = Path(__file__).parent
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # sorted, same as notebook
VIEWS = ["axial", "coronal", "sagittal"]
WEIGHTS = HERE / "weights" / "resnet18_best.pt"
VIEW_HEAD = HERE / "weights" / "view_head.pt"
BRAIN_MESH = HERE / "web" / "brain.bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(CLASSES))
    net.load_state_dict(torch.load(WEIGHTS, map_location=device))
    return net.to(device).eval()


model = load_model()


class GradCAM:
    def __init__(self, net, layer):
        self.net = net
        self.acts = None
        self.grads = None
        self.pooled = None
        layer.register_forward_hook(self._save)
        net.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "pooled", o.flatten(1).detach()))

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # plain no_grad forward passes (e.g. feature extraction) skip this
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        self.net.zero_grad()
        logits = self.net(x)
        probs = logits.softmax(dim=1)[0]
        if target is None:
            target = int(probs.argmax())
        logits[0, target].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # [1, 1, 7, 7]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return probs.detach().cpu().numpy(), target, cam.detach().cpu().numpy()


cam_engine = GradCAM(model, model.layer4[-1])


def backbone_features(image: Image.Image) -> torch.Tensor:
    """512-d avgpool features of the trained ResNet (no gradients)."""
    with torch.no_grad():
        model(eval_transform(image.convert("RGB")).unsqueeze(0).to(device))
    return cam_engine.pooled[0].cpu()


# ---------------------------------------------------------------- scan geometry
def head_mask(gray):
    return gray > max(0.08, float(np.percentile(gray, 60)) * 0.5)


def geometry_features(gray):
    """[left-right symmetry, tissue touching bottom edge, height/width] of the head silhouette."""
    m = head_mask(gray)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return np.array([0.5, 0.0, 1.0], np.float32)
    cx = int(round(xs.mean()))
    half = max(min(cx, gray.shape[1] - cx) - 1, 4)
    left = gray[:, cx - half:cx]
    right = gray[:, cx + 1:cx + 1 + half][:, ::-1]
    sym = np.corrcoef(left.ravel(), right.ravel())[0, 1]
    bottom = m[-20:].mean()
    hw = (ys.max() - ys.min()) / max(xs.max() - xs.min(), 1)
    return np.array([np.nan_to_num(sym), bottom, hw], np.float32)


def view_inputs(feat, geo):
    return torch.cat([feat, torch.tensor(geo).repeat(8)])  # geometry repeated so it isn't drowned out


_heads = torch.load(VIEW_HEAD, map_location="cpu") if VIEW_HEAD.exists() else None


def detect_view(image, gray, feat):
    """Returns (view name, confidence, facing 'left'/'right' or None)."""
    if _heads is None:
        return "axial", 0.0, None
    h = _heads["view"]
    z = (view_inputs(feat, geometry_features(gray)) - h["mu"]) / h["sd"]
    p = (z @ h["W"] + h["b"]).softmax(0)
    view = VIEWS[int(p.argmax())]
    facing = None
    if view == "sagittal":
        facing = detect_facing(image, feat)
    return view, float(p.max()), facing


def detect_facing(image, feat):
    h = _heads["facing"]
    d = feat - backbone_features(ImageOps.mirror(image.convert("RGB")))  # mirror flips the sign
    return "right" if float((d / h["sd"]) @ h["w"]) > 0 else "left"


# ------------------------------------------------------------ 3D placement
def _load_mesh():
    b = BRAIN_MESH.read_bytes()
    nv, _ = struct.unpack("<II", b[:8])
    return np.frombuffer(b[8:8 + nv * 12], np.float32).reshape(-1, 3)


MESH = _load_mesh()                       # fsaverage5 pial, RAS mm (x+ right, y+ anterior, z+ up)
CENTER = np.array([0.0, -18.0, 15.0])
PITUITARY = np.array([0.0, 2.0, -34.0])   # sella turcica, approximate template coordinates

# Template head extents (mm) used to convert image pixels to template coordinates.
HEAD_WIDTH = 150.0     # ear to ear, outer scalp
HEAD_LENGTH = 205.0    # back of head to nose tip
SCALP_TOP = 88.0       # z of the top of the scalp
SCALP_BACK = -112.0    # y of the back of the head
SCALP_FRONT = 92.0     # y of the front of the head (nose tip)


def _surface_radius(direction):
    """Distance from CENTER to the cortex along a unit direction (max over a narrow cone)."""
    rel = MESH - CENTER
    r = np.linalg.norm(rel, axis=1)
    cos = rel @ direction / (r + 1e-6)
    cone = cos > np.cos(np.radians(9))
    return float(r[cone].max()) if cone.any() else float(r.max())


def _column_center(axis, known):
    """Middle of the brain along `axis`, through the point fixed by `known` {axis: value}."""
    sel = np.ones(len(MESH), bool)
    for a, val in known.items():
        sel &= np.abs(MESH[:, a] - val) < 7
    if sel.sum() < 3:
        return float(CENTER[axis])
    col = MESH[sel, axis]
    return float((col.min() + col.max()) / 2)


def region_name(p, view):
    x, y, z = p
    side = "" if view == "sagittal" else ("Right " if x > 0 else "Left ")
    if np.linalg.norm(p - PITUITARY) < 14:
        return "Sellar region (pituitary gland)"
    if abs(x) < 22 and -45 < y < 10 and -15 < z < 30:
        return f"{side}deep brain (thalamus / basal ganglia)".strip().capitalize()
    if y < -68:
        lobe = "occipital lobe"
    elif z < -8 and abs(x) > 28 and y > -50:
        lobe = "temporal lobe"
    elif y > -12:
        lobe = "frontal lobe"
    else:
        lobe = "parietal lobe"
    name = f"{side}{lobe}"
    return name[0].upper() + name[1:]


class SliceFrame:
    """Maps image pixels (224x224) to template coordinates for one slice orientation.

    Two coordinates come from the pixel position; the third ("depth", the axis the slice
    is cut along) has to be supplied.
    """

    DEPTH_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

    def __init__(self, gray, view, facing):
        self.view, self.facing = view, facing
        ys, xs = np.nonzero(head_mask(gray))
        if len(xs) < 50:
            ys, xs = np.array([20, 204]), np.array([20, 204])
        self.x0, self.x1, self.y0 = xs.min(), xs.max(), ys.min()
        span = max(self.x1 - self.x0, 1)
        self.mm = (HEAD_LENGTH if view == "sagittal" else HEAD_WIDTH) / span

    @property
    def depth_axis(self):
        return self.DEPTH_AXIS[self.view]

    def __call__(self, px, py, depth):
        mm = self.mm
        if self.view == "sagittal":
            along = (px - self.x0) * mm if self.facing == "right" else (self.x1 - px) * mm  # from the back
            return np.array([depth, SCALP_BACK + along, SCALP_TOP - (py - self.y0) * mm])
        # radiological convention: the patient's left is on the image's right
        x = (self.x0 + self.x1) / 2 * mm - px * mm
        if self.view == "axial":  # anterior at the top of the image
            return np.array([x, SCALP_FRONT - 8 - (py - self.y0) * mm, depth])
        return np.array([x, depth, SCALP_TOP - (py - self.y0) * mm])  # coronal: superior at the top

    def plane(self, depth):
        """Template coordinates of the image corners TL, TR, BL, BR."""
        return [self(px, py, depth).round(1).tolist() for py in (0, 224) for px in (0, 224)]


def hotspot(cam):
    hot = cam >= 0.6
    w = np.where(hot, cam, 0)
    if not hot.any():
        cy, cx = np.unravel_index(cam.argmax(), cam.shape)
        return float(cx), float(cy), hot
    cy = float((w * np.arange(224)[:, None]).sum() / w.sum())
    cx = float((w * np.arange(224)[None, :]).sum() / w.sum())
    return cx, cy, hot


def locate(cam, gray, view, facing, cls):
    """Project the Grad-CAM hotspot onto the template brain.

    Returns position (RAS mm), radius (mm), region name and the slice plane corners.
    One 2D slice only fixes two of the three coordinates, so the third comes from a
    prior (middle of the brain along that axis), and the class adds anatomy: pituitary
    tumors sit at the sella, meningiomas on the brain surface, gliomas inside the tissue.
    """
    frame = SliceFrame(gray, view, facing)
    cx, cy, hot = hotspot(cam)
    a = frame.depth_axis

    if cls == "pituitary":
        return PITUITARY.tolist(), 9.0, region_name(PITUITARY, view), frame.plane(PITUITARY[a])

    p = frame(cx, cy, np.nan)
    known = {i: p[i] for i in range(3) if i != a}
    depth = _column_center(a, known)
    if view == "sagittal":
        # a side view can't tell left from right: show gliomas inside one hemisphere, and
        # meningiomas near the midline (parasagittal / skull-base ones are what sagittal slices show)
        depth += 12 if cls == "meningioma" else 28
    p[a] = depth

    d = p - CENTER
    dist = np.linalg.norm(d) + 1e-6
    surf = _surface_radius(d / dist)
    if cls == "meningioma":
        p = CENTER + d / dist * surf * 1.0          # meninges: on the surface
    elif dist > surf * 0.78:
        p = CENTER + d / dist * surf * 0.72          # glioma: keep it inside the tissue
    mm = frame.mm
    d = p - CENTER
    dist = np.linalg.norm(d) + 1e-6
    surf = _surface_radius(d / dist)
    if cls == "meningioma":
        p = CENTER + d / dist * surf * 1.0          # meninges: on the surface
    elif dist > surf * 0.78:
        p = CENTER + d / dist * surf * 0.72          # glioma: keep it inside the tissue

    area_px = hot.sum()
    # Grad-CAM blobs are much wider than the tumor (7x7 resolution), so shrink them
    radius = float(np.clip(np.sqrt(area_px / np.pi) * mm * 0.28, 7, 16))
    # the slice plane passes through the tumor, so the scan and the marker line up
    return p.round(1).tolist(), round(radius, 1), region_name(p, view), frame.plane(p[a])


def analyze(image: Image.Image, target: int | None = None, view_override: str | None = None):
    """Everything the UI needs for one scan."""
    rgb = image.convert("RGB")
    x = eval_transform(rgb).unsqueeze(0).to(device)
    probs, tgt, cam = cam_engine(x, target)
    feat = cam_engine.pooled[0].cpu()
    gray = np.asarray(rgb.resize((224, 224)).convert("L"), dtype=np.float32) / 255.0

    if view_override:
        view, view_conf = view_override.split(":")[0], 1.0
        facing = view_override.split(":")[1] if ":" in view_override else None
    else:
        view, view_conf, facing = detect_view(rgb, gray, feat)

    top = CLASSES[int(np.argmax(probs))]
    place_cls = CLASSES[tgt]
    loc = None
    if place_cls != "notumor":
        pos, radius, region, plane = locate(cam, gray, view, facing, place_cls)
        loc = {"pos": pos, "radius": radius, "region": region, "plane": plane}
    else:
        frame = SliceFrame(gray, view, facing)
        loc_plane = frame.plane(CENTER[frame.depth_axis] + (25 if view == "sagittal" else 0))
        loc = {"plane": loc_plane}
    return {
        "probs": probs, "target": tgt, "cam": cam, "gray": gray, "top": top,
        "view": view, "view_conf": view_conf, "facing": facing, "loc": loc,
    }
