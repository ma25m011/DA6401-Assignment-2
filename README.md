# DA6401 Assignment 2 — Multi-Task Visual Perception Pipeline

**Student:** Dibyendu Sarkar (ma25m011)
**Course:** DA6401 — Deep Learning, IIT Madras

---

## Links

- **W&B Report:** https://api.wandb.ai/links/ma25m011-ii/lhl0phb4
- **GitHub:** https://github.com/ma25m011/DA6401-Assignment-2

---

## Overview

A multi-task visual perception pipeline on the Oxford-IIIT Pet dataset using a VGG11 backbone built from scratch. Four tasks:

1. **Classification** — 37-class breed classification (Macro F1: 0.436)
2. **Object Localization** — Head bounding box regression (AP@0.5: 0.972, mIoU: 0.804)
3. **Semantic Segmentation** — U-Net pixel-level trimap (Dice: 0.826, Pixel Acc: 0.897)
4. **Unified Multi-Task Pipeline** — Single model, single forward pass, all three outputs

---

## Repository Structure

```
da6401_assignment_2/
├── models/
│   ├── vgg11.py          # VGG11Encoder + VGG11 alias
│   ├── layers.py         # CustomDropout
│   ├── classification.py # VGG11Classifier
│   ├── localization.py   # VGG11Localizer
│   ├── segmentation.py   # VGG11UNet
│   └── multitask.py      # MultiTaskPerceptionModel
├── losses/
│   └── iou_loss.py       # IoULoss (mean/sum reductions)
├── data/
│   └── pets_dataset.py   # OxfordIIITPetDataset
├── checkpoints/
│   ├── classifier.pth
│   ├── localizer.pth
│   └── unet.pth
├── multitask.py          # Top-level re-export for autograder
├── train.py              # Training entrypoint
├── inference.py          # Evaluation entrypoint
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Dataset should be at `../oxford-iiit-pet/` relative to this directory.

---

## Training

```bash
# Task 1: Classification
python train.py --task classifier --epochs 60

# Task 2: Localization
python train.py --task localizer --epochs 60

# Task 3: Segmentation
python train.py --task segmentation --epochs 30

# Ablation experiments (for W&B report)
python train.py --task ablation_bn --epochs 30
python train.py --task ablation_dropout --epochs 30
python train.py --task ablation_seg --epochs 30
```

---

## Evaluation

```bash
# Full evaluation (all tasks)
python inference.py

# Pipeline showcase on novel images
python inference.py --images path/img1.jpg path/img2.jpg path/img3.jpg
```

---

## Key Implementation Notes

- VGG11 built from scratch with `torch.nn` — no pretrained models
- Input images must be normalized (ImageNet mean/std)
- Input size fixed at 224×224
- Localization output: `[x_center, y_center, width, height]` in pixel space
- IoULoss range: [0, 1], supports `mean` and `sum` reductions
- Upsampling in U-Net uses Transposed Convolutions (no bilinear interpolation)
- Checkpoints loaded via relative paths in `multitask.py`
- Only packages used: torch, numpy, matplotlib, pillow, albumentations, wandb, scikit-learn
