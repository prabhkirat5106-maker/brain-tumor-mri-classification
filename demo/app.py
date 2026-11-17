"""NeuroLens: Gradio demo for the ResNet18 brain-MRI classifier with a 3D brain view."""
import base64
import io
import time
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from model import BRAIN_MESH, CLASSES, analyze

HERE = Path(__file__).parent
gr.set_static_paths([HERE / "web"])
MESH_URL = "/gradio_api/file=" + quote(str(BRAIN_MESH.resolve()))

VIEW_CHOICES = {
    "Auto-detect": None,
    "Axial (top-down)": "axial",
    "Coronal (front)": "coronal",
    "Sagittal · face left": "sagittal:left",
    "Sagittal · face right": "sagittal:right",
}
VIEW_LABEL = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}

PRETTY = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}
ACCENT = {"glioma": "#ff5d8f", "meningioma": "#ffb547", "notumor": "#3ee6a8", "pituitary": "#8b7bff"}
BLURB = {
    "glioma": "Tumor arising from glial cells, usually within the brain tissue itself.",
    "meningioma": "Tumor of the meninges, the membranes wrapping the brain; often near the skull edge.",
    "notumor": "No tumor pattern detected in this scan.",
    "pituitary": "Tumor of the pituitary gland at the base of the brain, behind the eyes.",
}

# heat colours: deep violet -> magenta -> orange -> pale yellow
HEAT_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
HEAT_RGB = np.array([
    [12, 7, 50], [110, 20, 140], [225, 50, 110], [255, 150, 40], [255, 250, 190],
], dtype=np.float32)
HEAT_SCALE = [[float(s), f"rgb({r:.0f},{g:.0f},{b:.0f})"] for s, (r, g, b) in zip(HEAT_STOPS, HEAT_RGB)]


def heat_rgb(cam):
    return np.stack([np.interp(cam, HEAT_STOPS, HEAT_RGB[:, c]) for c in range(3)], axis=-1)


def overlay_2d(gray, cam, alpha):
    base = np.repeat(gray[..., None] * 255, 3, axis=-1)
    a = (alpha * np.clip(cam * 1.4, 0, 1))[..., None]  # weak activations stay see-through
    out = base * (1 - a) + heat_rgb(cam) * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((448, 448), Image.BICUBIC)


def downsample(arr, n):
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR), np.float32) / 255


def figure_3d(gray, cam, height, mode):
    n = 112
    g = downsample(gray, n)
    c = downsample(cam, n)
    xs = np.arange(n)
    H = height * n * 0.6
    traces = []

    if mode == "Embossed scan":
        # the scan itself is pushed up by the heat. A mesh lets every vertex carry its
        # own RGB, so MRI grey blends smoothly into heat colour as activation rises.
        z = (c ** 1.5) * H
        w = np.clip((c - 0.25) / 0.5, 0, 1)[..., None] * 0.9
        rgb = (g[..., None] * 240) * (1 - w) + heat_rgb(c) * w
        yy, xx = np.meshgrid(xs, xs, indexing="ij")
        idx = np.arange(n * n).reshape(n, n)
        a, b, cc, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        traces.append(go.Mesh3d(
            # plain lists: int64 typed arrays don't survive the trip to Gradio's plotly.js
            x=xx.ravel().tolist(), y=yy.ravel().tolist(), z=z.ravel().round(3).tolist(),
            i=np.concatenate([a, b]).tolist(), j=np.concatenate([b, d]).tolist(),
            k=np.concatenate([cc, cc]).tolist(),
            vertexcolor=[f"rgb({r:.0f},{gg:.0f},{bb:.0f})" for r, gg, bb in rgb.reshape(-1, 3)],
            hoverinfo="skip", flatshading=False,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=60, y=-120, z=200),
        ))
    else:
        traces.append(go.Surface(
            x=np.arange(224) * n / 224, y=np.arange(224) * n / 224, z=np.zeros_like(gray),
            surfacecolor=gray, colorscale=[[0, "rgb(0,0,0)"], [1, "rgb(240,240,248)"]], cmin=0, cmax=1,
            showscale=False, hoverinfo="skip", lighting=dict(ambient=1, diffuse=0, specular=0),
        ))
        z = (c ** 1.6) * H + 1.0
        z[c < 0.35] = np.nan  # cut away cold regions so the scan shows through
        traces.append(go.Surface(
            x=xs, y=xs, z=z, surfacecolor=c, cmin=0.35, cmax=1, colorscale=HEAT_SCALE,
            opacity=0.85, showscale=False,
            hovertemplate="activation %{surfacecolor:.2f}<extra></extra>",
            lighting=dict(ambient=0.5, diffuse=0.9, specular=0.6, roughness=0.35, fresnel=0.4),
            lightposition=dict(x=60, y=-120, z=200),
        ))

    axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)
    fig = go.Figure(traces)
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=axis, yaxis=dict(axis, autorange="reversed"), zaxis=dict(axis, range=[0, n * 0.62]),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=-0.55, y=1.05, z=0.72), center=dict(x=0, y=0, z=-0.12), up=dict(x=0, y=0, z=1)),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="orbit",
    )
    return fig


def view_text(r, manual):
    name = VIEW_LABEL[r["view"]] + " slice"
    if r["facing"]:
        name += f" (face {r['facing']})"
    if manual:
        return name + " · set manually"
    if r["view_conf"] < 0.7:
        return name + f" · low-confidence guess ({r['view_conf']*100:.0f}%), set it in Scan view"
    return name + f" · auto-detected {r['view_conf']*100:.0f}%"


def result_card(probs, target, vtext=""):
    top = int(np.argmax(probs))
    name = CLASSES[top]
    rows = ""
    for i in np.argsort(-probs):
        cls = CLASSES[i]
        rows += f"""
        <div class="nl-row">
          <span class="nl-row-name">{PRETTY[cls]}</span>
          <div class="nl-bar"><div class="nl-fill" style="--w:{probs[i]*100:.1f}%;--c:{ACCENT[cls]}"></div></div>
          <span class="nl-row-pct">{probs[i]*100:.1f}%</span>
        </div>"""
    explain = "" if target == top else (
        f'<div class="nl-note">Heatmap is showing evidence for <b>{PRETTY[CLASSES[target]]}</b>, not the top prediction.</div>'
    )
    return f"""
    <div class="nl-card" style="--accent:{ACCENT[name]}">
      <div class="nl-kicker">Model prediction</div>
      <div class="nl-head">
        <div class="nl-label">{PRETTY[name]}</div>
        <div class="nl-ringwrap"><div class="nl-ring" style="--p:{probs[top]*100:.1f}"></div><b>{probs[top]*100:.0f}%</b></div>
      </div>
      <div class="nl-blurb">{BLURB[name]}</div>
      <div class="nl-rows">{rows}</div>
      {explain}
      <div class="nl-view">{vtext}</div>
    </div>"""


EMPTY_CARD = """
<div class="nl-card nl-empty">
  <div class="nl-kicker">Model prediction</div>
  <div class="nl-label" style="opacity:.35">Waiting for a scan</div>
  <div class="nl-blurb">Upload an MRI slice or pick an example to see what the network sees.</div>
</div>"""


def png_data_url(img, size=256):
    buf = io.BytesIO()
    img.resize((size, size), Image.BILINEAR).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def brain_payload(r, vtext):
    """What the Three.js viewer needs: where to put the tumor and the slice."""
    probs, tgt = r["probs"], r["target"]
    top = r["top"]
    shown = CLASSES[tgt]
    loc = r["loc"]
    base = {
        "nonce": time.time(), "view": r["view"], "view_conf": r["view_conf"], "view_text": vtext,
        "plane": loc["plane"], "slice": png_data_url(overlay_2d(r["gray"], r["cam"], 0.55)),
        "top_label": PRETTY[top],
    }
    if "pos" not in loc:
        clear = (f"No tumor detected · {probs[CLASSES.index('notumor')]*100:.1f}%" if top == "notumor"
                 else "Showing 'No tumor' evidence: nothing to mark")
        return {**base, "state": "clear", "clear_text": clear}
    return {
        **base, "state": "tumor", "label": PRETTY[shown], "color": ACCENT[shown], "conf": float(probs[tgt]),
        "is_top": shown == top, "pos": loc["pos"], "radius": loc["radius"], "region": loc["region"],
        "blurb": BLURB[shown],
    }


def run(image, explain_for, view_choice, opacity, height, mode):
    if image is None:
        return EMPTY_CARD, None, None, {"state": "empty", "nonce": time.time()}
    target = None if explain_for == "Top prediction" else CLASSES[[PRETTY[c] for c in CLASSES].index(explain_for)]
    target_idx = None if target is None else CLASSES.index(target)
    override = VIEW_CHOICES[view_choice]
    r = analyze(image, target_idx, override)
    vtext = view_text(r, override is not None)
    return (
        result_card(r["probs"], r["target"], vtext),
        figure_3d(r["gray"], r["cam"], height, mode),
        overlay_2d(r["gray"], r["cam"], opacity),
        brain_payload(r, vtext),
    )


CSS = "\n".join((HERE / p).read_text(encoding="utf-8")
                for p in ["style.css", "web/neurolens.css", "web/brain3d.css"])

theme = gr.themes.Base(
    primary_hue="violet", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#06070f", body_background_fill_dark="#06070f",
    body_text_color="#e7e9f5", body_text_color_dark="#e7e9f5",
    body_text_color_subdued="#8c90ad", body_text_color_subdued_dark="#8c90ad",
    block_background_fill="rgba(20,22,40,0.55)", block_background_fill_dark="rgba(20,22,40,0.55)",
    block_border_color="rgba(140,130,255,0.14)", block_border_color_dark="rgba(140,130,255,0.14)",
    block_label_background_fill="rgba(20,22,40,0.9)", block_label_background_fill_dark="rgba(20,22,40,0.9)",
    block_label_text_color="#b9bce0", block_label_text_color_dark="#b9bce0",
    block_title_text_color="#b9bce0", block_title_text_color_dark="#b9bce0",
    input_background_fill="rgba(10,12,25,0.8)", input_background_fill_dark="rgba(10,12,25,0.8)",
    border_color_primary="rgba(140,130,255,0.22)", border_color_primary_dark="rgba(140,130,255,0.22)",
    button_primary_background_fill="linear-gradient(120deg,#7b5cff,#e0458f)",
    button_primary_background_fill_dark="linear-gradient(120deg,#7b5cff,#e0458f)",
    button_primary_background_fill_hover="linear-gradient(120deg,#8d72ff,#f05aa0)",
    button_primary_background_fill_hover_dark="linear-gradient(120deg,#8d72ff,#f05aa0)",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    slider_color="#9b7bff", slider_color_dark="#9b7bff",
    color_accent_soft="rgba(123,92,255,0.2)", color_accent_soft_dark="rgba(123,92,255,0.2)",
    checkbox_label_background_fill="rgba(10,12,25,0.8)", checkbox_label_background_fill_dark="rgba(10,12,25,0.8)",
    checkbox_label_background_fill_selected="rgba(123,92,255,0.25)",
    checkbox_label_background_fill_selected_dark="rgba(123,92,255,0.25)",
    block_radius="18px", container_radius="18px", panel_background_fill="transparent",
    panel_background_fill_dark="transparent",
)

HERO = """
<div class="nl-hero">
  <div class="nl-badge"><span class="nl-dot"></span> ResNet18 · Grad-CAM · 99.0% test accuracy</div>
  <h1>Neuro<span>Lens</span></h1>
  <p>Drop in a brain MRI slice. The network classifies it into one of four classes, then maps
  <em>where</em> it looked onto a 3D brain you can spin, zoom and hover.</p>
</div>
"""

BRAIN_HTML = f'<div id="nb-root" class="nb-root" data-mesh="{MESH_URL}"></div>'
HEAD = '<script type="module">\n' + (HERE / "web" / "brain3d.js").read_text(encoding="utf-8") + "\n</script>"
TO_VIEWER = "(p) => { window.NeuroBrain && window.NeuroBrain.update(p); }"

FOOTER = """
<div class="nl-foot">
  Trained on 4,708 de-duplicated scans from the Kaggle brain-tumor MRI set, evaluated on 1,009 held-out scans.
  The red area is Grad-CAM evidence from <code>layer4</code>, projected from one 2D slice onto a template brain
  (FreeSurfer fsaverage), so its 3D position is an estimate, not a tumor segmentation.
  <b>Research demo, not a medical device.</b> Don't use it for diagnosis.
</div>
"""

with gr.Blocks(title="NeuroLens · Brain MRI explainer") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            image = gr.Image(type="pil", label="MRI scan", height=330, sources=["upload", "clipboard"],
                             elem_id="nl-upload")
            examples = sorted((HERE / "web" / "examples").glob("*"))
            gr.Examples([[str(p)] for p in examples], inputs=image, label="Try a held-out test scan",
                        examples_per_page=8)
            card = gr.HTML(EMPTY_CARD)
        with gr.Column(scale=7, min_width=420):
            with gr.Tabs():
                with gr.Tab("3D brain"):
                    gr.HTML(BRAIN_HTML, padding=False)
                with gr.Tab("Heat terrain"):
                    plot = gr.Plot(label="Drag to orbit · scroll to zoom", elem_id="nl-plot")
                with gr.Tab("2D overlay"):
                    flat = gr.Image(label="Grad-CAM overlay", height=560, interactive=False)
            with gr.Group(elem_classes="nl-controls"):
                with gr.Row():
                    explain_for = gr.Radio(["Top prediction"] + [PRETTY[c] for c in CLASSES],
                                           value="Top prediction", label="Explain evidence for", scale=3)
                    view_choice = gr.Radio(list(VIEW_CHOICES), value="Auto-detect", label="Scan view", scale=3)
                with gr.Row():
                    mode = gr.Radio(["Heat terrain", "Embossed scan"], value="Heat terrain",
                                    label="Terrain style", scale=2)
                    height = gr.Slider(0.2, 1.0, value=0.55, step=0.05, label="Peak height", scale=2)
                    opacity = gr.Slider(0, 1, value=0.65, step=0.05, label="2D overlay strength", scale=2)
            payload = gr.JSON(visible=False)
    gr.HTML(FOOTER)

    inputs = [image, explain_for, view_choice, opacity, height, mode]
    outputs = [card, plot, flat, payload]
    image.change(run, inputs, outputs)
    for ctl in (explain_for, view_choice, mode):
        ctl.change(run, inputs, outputs)
    for ctl in (height, opacity):
        ctl.release(run, inputs, outputs)
    payload.change(None, payload, None, js=TO_VIEWER)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS, head=HEAD)
