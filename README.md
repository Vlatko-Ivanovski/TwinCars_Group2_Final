# 🚗 TwinCar — Fine-Grained Car Classification (ResNet-50)

**Final Machine Learning Project – Group 2**  
**Authors:** Vlatko Ivanovski, Emilijan Panpur, Filip Blazevski  
**Academy:** Brainster Data Science Academy – Machine Learning Module (2025)

**Framework:** PyTorch  
**Model:** ResNet-50 (Transfer Learning)  
**Dataset:** Stanford Cars (196 classes)

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Project Organization](#-project-organization)
3. [Script Overview](#-script-overview)
4. [Model & Training](#-model--training)
5. [Training Curves](#-training-curves)
6. [Explainability — Grad-CAM](#-explainability--grad-cam)
7. [Custom Image Predictions](#-custom-image-predictions)
8. [How to Run the Project](#-how-to-run-the-project)
9. [Notes](#-notes)

---
## 📌 Overview

**TwinCar** is a deep-learning project for automatic recognition of:

✅ Car make  
✅ Car model  
✅ Production year  

The model is trained on the **Stanford Cars (196 classes)** dataset using a **ResNet-50 convolutional neural network with transfer learning**.

The project demonstrates a complete Machine Learning pipeline:

- Dataset loading & preprocessing  
- CNN model training (ResNet-50)  
- Model evaluation and visualization  
- Grad-CAM explainability  
- Custom image prediction  
- Model export (.pth)  
- Reproducible project structure  

---

## 📁 Project Organization

```text
TwinCars_Group2_Final/
│
├── data/
│   ├── external/               
│   │   ├── test_4276.jpg
│   │   ├── test_4692.jpg
│   │   ├── test_6502.jpg
│   │   └── test_6714.jpg
│   │
│   ├── hf_cache/              
│   └── classes.txt
│
├── models/
│   └── resnet50_twin_cars.pth
│
├── notebooks/
│   └── 1.0-FB-initial-experiments.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── gradcam_example_1.png
│   │   └── gradcam_example_2.png
│   │
│   └── predictions_custom_images.csv
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── make_classes.py
│   └── create_classes_from_hf.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧠 Script overview

| Script | Description |
|------|------|
| `train.py` | Trains the ResNet-50 model |
| `predict.py` | Makes predictions on images in `data/external/` |
| `make_classes.py` | Creates `classes.txt` from `.mat` metadata |
| `create_classes_from_hf.py` | Creates `classes.txt` using HuggingFace dataset |

---

## 📊 Model & Training

- **Architecture:** ResNet-50  
- **Pretrained on:** ImageNet  
- **Classes:** 196 (make + model + year)  
- **Loss:** Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 20 (trained on Google Colab with GPU)  

Saved model:

```text
models/resnet50_twin_cars.pth
```

---

## 📈 Training Curves

<p align="center">
  <img src="reports/figures/loss_curve.png" width="45%">
  <img src="reports/figures/accuracy_curve.png" width="45%">
</p>

These plots show model convergence and learning stability throughout training.

---

## 🔍 Explainability — Grad-CAM

Grad-CAM visualizations highlight which regions of the image the ResNet-50 model uses to make its predictions.

<p align="center">
  <img src="reports/figures/gradcam_example_1.png" width="45%">
  <img src="reports/figures/gradcam_example_2.png" width="45%">
</p>

Model focuses primarily on:

- Car body silhouette  
- Headlights and tail lights  
- Front grill  
- Roof and trunk shape  

This confirms that the model is learning meaningful car-specific features.

---

## 🧪 Custom Image Predictions

The following four real images were tested using the trained ResNet-50 model:

<p align="center">
  <img src="data/external/test_4276.jpg" width="24%">
  <img src="data/external/test_4692.jpg" width="24%">
  <img src="data/external/test_6502.jpg" width="24%">
  <img src="data/external/test_6714.jpg" width="24%">
</p>

For every image, the system predicts:

✅ Car make  
✅ Car model  
✅ Production year  
✅ Confidence score  

Results are saved in:

```text
reports/predictions_custom_images.csv
```

Example results:

| Image | Make | Model | Year | Confidence |
|------|------|------|------|------|
| test_4276.jpg | Ferrari | 458 Italia Convertible | 2012 | 0.97 |
| test_4692.jpg | Mitsubishi | Lancer Sedan | 2008 | 0.94 |
| test_6502.jpg | BMW | 3 Series Sedan | 2011 | 0.91 |
| test_6714.jpg | Audi | A4 Sedan | 2013 | 0.95 |

---

## ▶️ How to Run the Project

### 1. Create and activate environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Predict on images

```bash
python src/predict.py \
  --images data/external \
  --model models/resnet50_twin_cars.pth \
  --classes data/classes.txt
```

Results will be saved to:

```text
reports/predictions_custom_images.csv
```

---

## ✅ Notes

- `hf_cache` is ignored in `.gitignore`
- Trained on Google Colab (GPU)
- Reproducible project structure
- Scalable to more epochs or other architectures


---

## 🚀 Project

**TwinCar — Intelligent Car Recognition with Deep Learning (ResNet-50)**


