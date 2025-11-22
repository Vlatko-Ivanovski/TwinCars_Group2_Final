# 🚗 TwinCar — Fine-Grained Car Classification (ResNet-18)

**Final Machine Learning Project – Group 2**  
**Authors:** Emilijan Panpur, Filip Blazevski, Vlatko Ivanovski  
**Academy:** Brainster Data Science Academy – Machine Learning Module (2025)

**Framework:** PyTorch  
**Model:** ResNet-18 (Transfer Learning)  
**Dataset:** Stanford Cars (196 classes)

---

## 📌 Overview

**TwinCar** is a deep-learning project for automatic recognition of:

✅ Car make  
✅ Car model  
✅ Production year  

The model is trained on the **Stanford Cars (196 classes)** dataset using **ResNet-18 + Transfer Learning**.

The project demonstrates a complete Machine Learning pipeline:

- Dataset loading & preprocessing  
- CNN model training (ResNet-18)  
- Evaluation with metrics & visualizations  
- Grad-CAM explainability  
- Custom image prediction  
- Model export (.pt and .onnx)  
- Reproducible project structure  

---

## 📁 Project Organization

TwinCars_Group2_Final/
│
├── data/
│   ├── external/              # Custom images for prediction
│   │   ├── test_4276.jpg
│   │   ├── test_4692.jpg
│   │   ├── test_6502.jpg
│   │   └── test_6714.jpg
│   │
│   ├── hf_cache/              # Cached Stanford Cars dataset (ignored in Git)
│   └── classes.txt
│
├── models/
│   ├── stanford_cars_resnet18_head_subset.pt
│   ├── stanford_cars_resnet18_head_subset.onnx
│   └── stanford_cars_resnet18_head_subset.onnx.data
│
├── notebooks/
│   └── 1.0-FB-initial-experiments.ipynb
│
├── reports/
│   ├── predictions_custom_images.csv
│   └── figures/
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── gradcam_example_1.png
│       └── gradcam_example_2.png
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md


---

## 📊 Model & Training

- **Architecture:** ResNet-18 (pretrained on ImageNet)
- **Classes:** 196 (make + model + year)
- **Fine-tuning:** Only the classification head
- **Loss:** Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** **3** (subset training – demonstration purposes)

Saved model formats:

models/
├── stanford_cars_resnet18_head_subset.pt
├── stanford_cars_resnet18_head_subset.onnx
└── stanford_cars_resnet18_head_subset.onnx.data

yaml
Copy code

---

## 📈 Evaluation & Visualizations

All evaluation files are stored in:

reports/figures/

yaml
Copy code

Contains:

- `loss_curve.png` – Training loss progression  
- `accuracy_curve.png` – Training accuracy progression  
- `gradcam_example_1.png`
- `gradcam_example_2.png`

These graphs visually confirm correct training behavior.

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM is used to visualize which parts of the image influence predictions.

Examples are saved in:

reports/figures/

yaml
Copy code

The model mainly focuses on:

- Car body shape  
- Headlights  
- Grille  
- Overall silhouette  

This confirms that the model learned **relevant vehicle features**, not background noise.

---

## 🔮 Custom Image Prediction

You can test your own images.

### 1️⃣ Place images here:

data/external/

shell
Copy code

### 2️⃣ Run prediction:

python src/predict.py

shell
Copy code

### 3️⃣ Output file:

reports/predictions_custom_images.csv

vbnet
Copy code

Example structure:

| image_path | pred_label | confidence | pred_make | pred_model | pred_year |
|----------|------------|-----------|---------|----------|---------|

Example:

test_4276.jpg Ferrari 458 Italia Convertible 0.032
test_4692.jpg Mitsubishi Lancer Sedan 0.033

yaml
Copy code

---

## ▶️ How to Run the Project

### 1. Create & activate virtual environment

python -m venv venv
venv\Scripts\activate

shell
Copy code

### 2. Install requirements

pip install -r requirements.txt

shell
Copy code

### 3. Run notebook (recommended)

jupyter notebook

makefile
Copy code

Open:

notebooks/1.0-FB-initial-experiments.ipynb

shell
Copy code

### 4. Or run through scripts

python src/train.py
python src/predict.py

yaml
Copy code

---

## ✅ Notes

- `hf_cache` folder is **ignored in Git**
- Models are saved in `.pt` and `.onnx` formats
- Structure is fully reproducible
- Designed for demonstration + academic submission
- Easily extendable for more epochs or larger architectures

---

**🚗 TwinCar — Brainster Machine Learning Final Project 2025**
