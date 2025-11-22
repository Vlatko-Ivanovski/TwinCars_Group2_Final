🚗 TwinCar – Fine-Grained Car Classification (ResNet-18)

Final Machine Learning Project – Group 2

Authors:
- Emilijan Panpur
- Filip Blazevski
- Vlatko Ivanovski

Academy: Brainster Data Science Academy  
Module: Machine Learning (2025)

Framework: PyTorch  
Model: ResNet-18 (Transfer Learning)  
Dataset: Stanford Cars (196 classes)

---------------------------------------------------

📌 Overview

TwinCar is a deep learning project for automatic recognition of:

✅ Car make  
✅ Car model  
✅ Production year  

The model is trained on the Stanford Cars dataset (196 classes) using transfer learning with ResNet-18.

This project demonstrates a complete Machine Learning pipeline:

- Dataset loading & preprocessing
- CNN model training
- Evaluation with metrics & visualizations
- Grad-CAM explainability
- Custom image prediction
- Model export (.pt & .onnx)
- Proper project structure

---------------------------------------------------

📁 Project Structure

TwinCars_Group2_Final/
│
├── data/
│   ├── external/           # Custom images for prediction
│   ├── hf_cache/           # Cached Stanford Cars dataset (ignored in Git)
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
├── requirements.txt
├── .gitignore
└── README.md

---------------------------------------------------

📊 Model Evaluation

Evaluation includes:

- Training & validation loss
- Training & validation accuracy
- Top-1 and Top-3 accuracy
- Confusion distribution (visual)
- Grad-CAM visualization

Visual outputs are saved in:

reports/figures/

---------------------------------------------------

🔍 Grad-CAM Explainability

Grad-CAM heatmaps show where the model focuses when predicting.

The model mainly attends to:

- Car body shape
- Headlights
- Front grille
- Overall silhouette

Saved examples:

reports/figures/gradcam_example_1.png  
reports/figures/gradcam_example_2.png

---------------------------------------------------

🖼️ Custom Image Prediction

1. Place your images in:

Use ResNet-50 / EfficientNet

Add Web App (Streamlit / HF Spaces)
