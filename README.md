# 🧐 3D Lung Nodule Classification using LIDC-IDRI 🩻

This project is a 3D convolutional neural network pipeline for binary classification (cancer vs. non-cancer) using the LIDC-IDRI dataset. It includes:

* 📦 DICOM volume preprocessing
* 🧼 On-the-fly normalization & resizing
* 🧪 Balanced train/val/test splitting
* 📊 Real-time performance metrics: Accuracy, Precision, Recall, F1-Score
* ⚙️ Configurable PyTorch training loop with early stopping

---

## 📁 Project Structure

```
.
├── data/
│   └── LIDC_classes_dcm/     # DICOM files organized by class
├── dataset/
│   ├── lidc_dataset.py       # Custom PyTorch Dataset for 3D DICOMs
├── training/
│   ├── train.py              # Model training loop
│   └── model.py              # CNN architecture (e.g. 3D DenseNet)
├── utils/
│   ├── transforms.py         # Resize, normalization, preprocessing
│   └── helpers.py            # Loader functions, metrics, etc.
├── logs/
│   └── saved_model.pth       # Saved best model with metrics
├── README.md
└── requirements.txt
```

---

## 📝 Description

This project aims to classify lung nodules from CT scans into cancerous or non-cancerous using 3D deep learning. Volumes are processed from raw DICOM files, normalized (HU \[-1000, 400]), resized to uniform shape, and fed into a CNN model using on-the-fly loading and augmentation.

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/your-username/lung-nodule-3d-cnn.git
cd lung-nodule-3d-cnn
```

### 2. Install dependencies

```bash
conda create -n lidc_env python=3.12
conda activate lidc_env
pip install -r requirements.txt
```

### 3. Download and organize the LIDC dataset

* Download the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* Use the provided script to organize by class into `data/LIDC_classes_dcm/`

---

## 🏋️‍♀️ Training

```bash
python train.py
```

Customize parameters like:

* `EPOCHS`, `BATCH_SIZE`, `PATIENCE`, `INPUT_SHAPE`
* Model architecture in `model.py`

---

## 🧠 Model Architecture

This project uses a 3D CNN (e.g., 3D DenseNet) with:

* Trilinear resizing
* Single-channel input (grayscale volume)
* Binary classification output with sigmoid activation

---

## 📊 Metrics & Logging

* Accuracy, Precision, Recall, F1
* Confusion matrix logged per epoch
* Best model is saved via early stopping

---

## 🧪 Example Visualization (Grad-CAM planned)

> Heatmap visualizations of model attention on volumetric scans (coming soon)

---

## 🧼 Preprocessing Steps

* Normalize voxel values to \[0, 1] using HU clipping (-1000, 400)
* Resize volumes to (C=1, 96, 96, D) using `Resize` from MONAI
* On-the-fly augmentation can be integrated later with MONAI transforms

---

## 🧑‍💻 Author

**Viviane Alves de Oliveira**
Python Developer | AI Enthusiast | Medical AI Researcher
📧 [LinkedIn](https://www.linkedin.com/in/vivianealvesoliveira)

---

## 📄 License

MIT License
