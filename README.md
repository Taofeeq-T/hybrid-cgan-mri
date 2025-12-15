# Hybrid-cGAN MRI Generator

A PyTorch implementation of a **Hybrid CNN–Vision Transformer Conditional GAN (Hybrid-cGAN)** for synthetic brain MRI generation.

This model combines:

- A **conditional transposed-convolution Generator**
- A **Hybrid Discriminator** that fuses:
  - ✔ CNN local texture features  
  - ✔ Vision Transformer (ViT-Base/16) global context features
- Projection discriminator with **AC-GAN auxiliary classification head**
- Progressive transformer unfreezing
- Spectral normalization (Discriminator)
- Batch normalization (Generator)
- R1 regularization and label smoothing

The system produces high-quality **class-conditional brain MRI images** and supports **training, testing, and sample generation** from a single script.

```bash
python playground.py
```

---

## ⚡ Features

- Conditional GAN (4 MRI classes)
- Hybrid CNN–ViT discriminator
- Spectral normalization (D)
- Label smoothing
- R1 gradient penalty
- Progressive ViT unfreezing
- FID / KID evaluation (via `torchmetrics`)
- MS-SSIM diversity scoring
- Modular, research-friendly file structure

---

## 📂 Project Structure

```
hybrid-cgan-mri/
├── playground.py        # Main training & evaluation script
├── models/              # Generator and discriminator definitions
├── training/            # Trainer, losses, and metrics
├── utils/               # Dataset loading and utilities
├── data/                # Place dataset here
├── ckpts/               # Saved checkpoints
└── results/             # Generated samples and evaluation outputs
```

## 🔗 Pretrained Model

The **best-performing model checkpoint** from our experiments (selected by lowest FID on the validation set) is available here:

👉 **Best Hybrid-cGAN checkpoint:** [https://drive.google.com/file/d/1KWOHqGw4CR_vw3KaRAc2PAvLT5pIrDb8/view?usp=sharing]

This checkpoint can be used directly for evaluation or sample generation.


---

## 📥 Dataset Setup

Download the **Brain Tumor MRI Dataset** (Nickparvar, 2021) from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Notes
- Only **axial slices** were used in our experiments.

### Directory layout

```
data/brain_mri/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

No additional configuration is required — class labels and preprocessing are detected automatically.

---

## 🚀 Running Training

```bash
python playground.py
```

This script performs:

- Dataset loading
- Model initialization
- Full training with progressive ViT unfreezing
- Best checkpoint saving → `ckpts/best_model.pt`
- Test-set evaluation
- Sample image generation → `results/sample_grid.png`

---

## 🧪 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- PyTorch
- torchvision
- timm
- torchmetrics
- numpy
- matplotlib
- Pillow

---

## 🔑 License

This project is released under the **MIT License**.

---

## ✨ Author

Developed by **Taofeeq Togunwa**  
Hybrid-cGAN architecture for medical image synthesis research.
