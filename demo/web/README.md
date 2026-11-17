---
title: NeuroLens
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: static
app_file: index.html
pinned: false
short_description: Brain MRI tumor classifier with an interactive 3D brain
---

# NeuroLens: brain MRI classifier demo

A ResNet18 fine-tuned to classify brain MRI slices as **glioma**, **meningioma**, **pituitary** tumor or **no tumor**,
with the model's evidence mapped onto an interactive 3D brain.

**Everything runs in your browser** (onnxruntime-web + three.js). Uploaded scans never leave your device.

- Upload, paste or drop an MRI slice, or pick one of 8 held-out test scans
- Predicted class with per-class probabilities
- **3D brain**: the area the model flagged glows red. Hover or tap it for type, confidence, approximate location
  and size. Drag or swipe to rotate, scroll or pinch to zoom. X-ray, Slice, Spin and Reset buttons.
- 2D Grad-CAM overlay, and "Explain evidence for" any class

## Model

| | |
|---|---|
| Architecture | ResNet18 (ImageNet weights), new 4-way head |
| Data | Kaggle `tombackert/brain-tumor-mri-data`, 7,153 images, 6,726 after removing exact duplicates |
| Split | 70 / 15 / 15 stratified (4,708 / 1,009 / 1,009), de-duplicated before splitting |
| Test accuracy | **99.0%** (999 / 1,009), macro F1 0.990 |

The heatmap is the class activation map from `layer4`. For a ResNet (global average pool + linear head) it
equals Grad-CAM exactly, so the browser computes it without gradients.

## How the 3D placement works

One 2D slice fixes only two of the three coordinates, so the position is an **estimate on a template brain**
(FreeSurfer fsaverage5 cortex; the cerebellum and brainstem are stylised shapes):

1. Linear heads on the ResNet's features detect the slice orientation (axial / coronal / sagittal: 76/80 correct
   with leave-one-out on hand-labelled test scans) and which way a sagittal slice faces (20/22). Low-confidence
   guesses are flagged, and **Scan view** overrides them.
2. The heatmap's hotspot is converted from pixels to millimetres using the head outline.
3. The missing coordinate comes from the middle of the brain along that axis, plus anatomy per class: pituitary
   tumors at the sella, meningiomas on the surface, gliomas inside the tissue.

**Research demo, not a medical device. Don't use it for diagnosis.**
