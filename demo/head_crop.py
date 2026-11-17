"""Crop an MRI slice to the head and pad it to a square (v2 preprocessing).

v1 stretched every image to 224x224, so the model saw each source's framing (margins, zoom,
aspect ratio) and could use it as a clue. v2 finds the head first:

1. grayscale thumbnail (longest side 256)
2. bright pixels: > max(0.08, 0.5 * 60th percentile)
3. keep rows/columns where at least 2% of the pixels are bright (drops thin text and rulers)
4. square box around them, 8% larger than the longer side, centred; areas outside the image
   are padded black

The same steps are implemented in the browser (web/app.js, headBox).
"""
import numpy as np
from PIL import Image

THUMB = 256


def head_box(img: Image.Image):
    """Square crop box (left, top, side) in original pixel coordinates."""
    w, h = img.size
    scale = THUMB / max(w, h)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    g = np.asarray(img.convert("L").resize((tw, th), Image.BILINEAR), np.float32) / 255
    thr = max(0.08, float(np.percentile(g, 60)) * 0.5)
    m = g > thr
    rows = np.nonzero(m.sum(1) >= 0.02 * tw)[0]
    cols = np.nonzero(m.sum(0) >= 0.02 * th)[0]
    if len(rows) < 8 or len(cols) < 8:  # nothing head-like found: keep the whole image
        side = max(w, h)
        return (w - side) / 2, (h - side) / 2, side
    y0, y1 = rows[0] / scale, (rows[-1] + 1) / scale
    x0, x1 = cols[0] / scale, (cols[-1] + 1) / scale
    side = max(x1 - x0, y1 - y0) * 1.08
    return (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side


def crop_to_head(img: Image.Image, box=None) -> Image.Image:
    img = img.convert("RGB")
    left, top, side = box if box is not None else head_box(img)
    s = max(1, round(side))
    out = Image.new("RGB", (s, s))
    out.paste(img, (round(-left), round(-top)))
    return out
