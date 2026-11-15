# Brain Tumor MRI Classification (in progress)

A convolutional neural network in PyTorch that classifies brain MRI scans into 4 classes: glioma, meningioma, pituitary tumor, or no tumor.

**Dataset:** [Brain Tumor MRI Data on Kaggle](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data) (7,153 images), split 70 / 15 / 15 into train, validation and test sets, stratified by class. The images are not included in this repo.

## Status

| Phase | Status |
|---|---|
| 1. Data pipeline (Dataset class, augmentation, normalization, loaders) | Done |
| 2. Baseline CNN trained from scratch | Done: about 94% validation accuracy |
| 3. Fine-tune a pretrained ResNet18, evaluate on the test set once | Next |
| 4. Grad-CAM, final write-up | Planned |

The test set has not been used yet. It will be evaluated once, on the final model only.

## Known limitations (to be expanded)

- Every image is resized to a 224×224 square, so non-square scans get slightly distorted.
- The dataset mixes scan views (side, top-down, front). If views are unevenly spread across classes, the model could learn the view instead of the tumor. This will be checked with Grad-CAM.

This is a learning project, not a medical tool.
