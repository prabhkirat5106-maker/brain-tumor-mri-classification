# NeuroLens: interactive demo

**Live: [huggingface.co/spaces/prab12/neurolens](https://huggingface.co/spaces/prab12/neurolens)**
(full screen: [prab12-neurolens.static.hf.space](https://prab12-neurolens.static.hf.space))

Upload a brain MRI slice and the fine-tuned ResNet18 from the notebook classifies it, then shows where it looked:

- the predicted class with per-class probabilities
- a **3D brain** you can rotate and zoom. The area the model flagged glows red. Hover or tap it for the tumor
  type, confidence, approximate location and rough size; hovering elsewhere names the brain region.
  Buttons: X-ray (see-through brain), Slice (your scan floating in its own plane), Spin, Reset.
- a 2D Grad-CAM overlay, and "Explain evidence for" any class, not only the top prediction

There are two versions of the same pipeline:

| | `web/` (deployed) | `app.py` (Gradio) |
|---|---|---|
| Runs | entirely in the browser (onnxruntime-web + three.js) | Python server (PyTorch) |
| Hosting | free static Hugging Face Space | needs Hugging Face PRO for a Gradio Space |
| Extra | scans never leave the visitor's device | "heat terrain" Grad-CAM height map tab |

## Heatmap: Grad-CAM in the browser

The browser can't take gradients, but it doesn't need to. In a ResNet the logit is
`fc(mean over positions of A)`, so the gradient of class *c* with respect to the `layer4` activations `A` is
`fc.weight[c] / 49` at every position. Grad-CAM's channel weights are therefore just the fc weights, and Grad-CAM
equals the class activation map `ReLU(Σₖ fc.weight[c, k] · Aₖ)`. On the test examples the two differ by under 1e-6.

## How the 3D placement works

One 2D slice fixes only two of the three coordinates, so the 3D position is an **estimate on a template brain**:

1. **Slice orientation.** Two linear heads on the ResNet's own 512-d pooled features (`weights/view_head.pt`):
   - axial / coronal / sagittal, from the features plus 3 silhouette features (left-right symmetry, tissue at the
     bottom edge, aspect ratio): **76/80** correct with leave-one-out on the 80 test scans hand-labelled for the
     notebook's view audit
   - for sagittal slices, which way the face points, from features(image) − features(mirrored image), so mirroring
     flips the answer by construction: **20/22** with leave-one-out

   Low-confidence guesses are flagged in the UI, and **Scan view** overrides them.
2. **In-plane position.** The heatmap hotspot (weighted centroid of activations ≥ 0.6) is converted from pixels to
   millimetres using the head outline and standard head dimensions, in radiological convention (the patient's left
   is on the image's right).
3. **The missing coordinate** is the middle of the brain along that axis. Then anatomy is applied per class:
   pituitary tumors go to the sella, meningiomas to the brain surface, gliomas inside the tissue. A sagittal slice
   can't show left vs right, so the hemisphere there is arbitrary.

The cortex is the FreeSurfer **fsaverage5** pial surface (from nilearn's bundled data), shaded by sulcal depth. The
cerebellum and brainstem are stylised shapes, since that surface only covers the cortex.

## Run it locally

The trained weights aren't in the repo (`*.pt` is git-ignored). Run the notebook, which saves
`resnet18_best.pt`, and copy that file to `demo/weights/`.

```
cd demo
pip install gradio==6.32.0 -r requirements.txt
```

**Gradio app:** `python app.py`, then open http://localhost:7860

**Browser version:**

```
python tools/export_web.py   # writes web/model/neurolens.onnx + heads.json
python tools/serve_web.py    # serves web/ at http://localhost:8000
```

**Deploy:** `python deploy_space.py` uploads `web/` as a free static Space (private until you switch it).
`--public` makes a new Space public, and `--gradio` deploys the Gradio app instead (needs PRO).

## Files

```
app.py, model.py, style.css   Gradio app; model.py also has view detection and 3D placement
web/                          browser version (index.html, app.js, styles.css) and shared assets:
  brain3d.js, brain3d.css     the Three.js viewer, used by both versions
  neurolens.css               shared look (hero, prediction card, footer)
  brain.bin                   fsaverage5 cortex mesh + sulcal depth, packed as raw arrays
  model/heads.json            fc weights for the heatmap + view/facing heads, exported for the browser
  examples/                   2 scans per class from the held-out test split
  README.md                   Hugging Face Space card
tools/
  export_web.py               PyTorch -> ONNX + heads.json for the browser
  train_view_head.py          trains weights/view_head.pt from tools/view_labels.json
  view_labels.json            the 80 view labels (+ 22 sagittal facing labels)
  serve_web.py                local static server with correct MIME types
deploy_space.py               Hugging Face deploy
```

## Limitations

- The heatmap is Grad-CAM at 7×7 resolution: regions that raised the class score, not a tumor segmentation.
- The 3D position is projected from one 2D slice onto a template brain. Treat it as "roughly here".
- Everything in the notebook's Limitations section applies here too.

**Research demo, not a medical device. Don't use it for diagnosis.**
