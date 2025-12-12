# Dataset Instructions

This folder contains the dataset used for training and evaluating the **Hybrid-cGAN MRI Generator**.

---

## 📥 Dataset Source

Download the **Brain Tumor MRI Dataset** (Nickparvar, 2021) from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## 🧠 Preprocessing Notes

- Only **axial slices** were used in our experiments.
- No additional preprocessing is required beyond placing the files in the correct directory structure.

---

## 📂 Expected Directory Structure

After downloading and extracting the dataset, organize it as follows:

```
data/brain_mri/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Each subdirectory should contain the corresponding MRI image files for that class.

The training code will automatically:
- Detect class labels from folder names
- Apply resizing and normalization
- Load images for conditional GAN training

---

