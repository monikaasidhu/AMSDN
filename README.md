# AMSDN: Adaptive Multi-Scale Defense Network

A research-grade PyTorch implementation of a unified framework for defending against patch and sparse adversarial attacks.

## 🎯 Overview

AMSDN combines multiple defense mechanisms in a single, end-to-end trainable architecture:

1. **Multi-Scale Feature Extraction** (ConvNeXt-Tiny + FPN)
2. **Adaptive Attention** (Spatial + Channel + Multi-Scale Pyramid)
3. **Selective Purification** (Feature-space denoising)
4. **Prediction Consistency Verification**
5. **Self-Supervised Robustness Training** (SSRT)
6. **Randomized Smoothing Certification**

## 📊 Architecture

```
Input Image (3×32×32)
    ↓
┌─────────────────────────────────────┐
│ Stage 1: ConvNeXt-Tiny + FPN        │
│ Output: Multi-scale features        │
│ [P2, P3, P4, P5] @ 256 channels     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: Adaptive Attention         │
│ • Spatial Attention                 │
│ • Channel Attention                 │
│ • Multi-Scale Pyramid Attention     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 3: Selective Purification     │
│ • Anomaly Detection                 │
│ • Feature Denoising                 │
│ • Selective Fusion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 4: Prediction Consistency     │
│ (Optional, expensive)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Classification Head                 │
│ Output: Logits (10 classes)         │
│         Anomaly Score               │
│         Detection Decision          │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended)

pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Self-Supervised Pretraining 
python training/pretrain_ssrt.py

# 2. Adversarial Training 
python training/adversarial_train.py

# 3. Multi-Attack Fine-tuning 
python training/finetune_attacks.py

# 4. Evaluation 
python evaluation/evaluate.py

# 5. Certification 
python evaluation/certification.py
```

### Google Colab

Open `notebooks/AMSDN_Colab.ipynb` in Google Colab for a complete interactive tutorial.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/AMSDN/blob/main/notebooks/AMSDN_Colab.ipynb)

## 📁 Repository Structure

```
AMSDN/
│
├── data/
│   └── cifar10.py                 # CIFAR-10 data loading
│
├── models/
│   ├── backbone/
│   │   └── convnext_fpn.py        # ConvNeXt + FPN
│   ├── attention/
│   │   └── adaptive_attention.py  # Multi-scale attention
│   ├── purification/
│   │   └── selective_purifier.py  # Adversarial purification
│   └── amsdn.py                   # Main AMSDN model
│
├── training/
│   ├── pretrain_ssrt.py           # Self-supervised pretraining
│   ├── adversarial_train.py       # Adversarial training
│   └── finetune_attacks.py        # Multi-attack fine-tuning
│
├── attacks/
│   ├── patch_attacks.py           # Patch attacks (AdvPatch, BPDA)
│   └── pixel_attacks.py           # Sparse pixel attacks
│
├── evaluation/
│   ├── evaluate.py                # Comprehensive evaluation
│   └── certification.py           # Randomized smoothing
│
├── utils/
│   └── helpers.py                 # Visualization & utilities
│
├── notebooks/
│   └── AMSDN_Colab.ipynb          # Interactive Colab notebook
│
├── requirements.txt
└── README.md
```

## 🔬 Implemented Attacks

- **PGD** (Projected Gradient Descent): ε=8/255, 16/255
- **C&W** (Carlini-Wagner): L2 attack
- **Patch Attacks**: Localized perturbations (4×4, 8×8 pixels)
- **Pixel Attacks**: Sparse perturbations (5, 10 pixels)
- **Adaptive BPDA**: Gradient obfuscation circumvention

## 🙏 Acknowledgments

- ConvNeXt architecture from [timm](https://github.com/rwightman/pytorch-image-models)
- Inspired by adversarial defense research from Cohen et al., Brown et al., and others
- Built with PyTorch

## 🎓 Related Work

- **Randomized Smoothing:** Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- **Adversarial Patch:** Brown et al., "Adversarial Patch" (NIPS 2017 Workshop)
- **PGD Attack:** Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
- **FPN:** Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017)
- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s" (CVPR 2022)

