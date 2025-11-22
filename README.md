##🚗TwinCar — Fine-Grained Car Classification (ResNet-18)

Final Machine Learning Project – **Group 2**  
**Authors:** Emilijan Panpur, Filip Blazevski, Vlatko Ivanovski  
**Academy:** Brainster Data Science Academy – Machine Learning Module (2025)  

Framework: **PyTorch**  
Model: **ResNet-18 (Transfer Learning)**  
Dataset: **Stanford Cars (196 classes)**

---

## 📌 Overview

**TwinCar** is a deep learning project for automatic recognition of:

✅ Car make  
✅ Car model  
✅ Production year  

The model is trained on the **Stanford Cars dataset (196 fine-grained classes)** using **ResNet-18 + transfer learning**.

The project demonstrates a complete Machine Learning pipeline:

- Dataset loading & preprocessing  
- CNN model training (ResNet-18)  
- Evaluation with metrics & visualizations  
- Grad-CAM explainability  
- Custom image prediction  
- Model export (`.pt` and `.onnx`)  
- Reproducible project structure  

---

## 📁 Project Organization
winCars_Group2_Final/
│
├── data/
│ ├── external/ # Custom images for prediction
│ │ ├── test_4276.jpg
│ │ ├── test_4692.jpg
│ │ ├── test_6502.jpg
│ │ └── test_6714.jpg
│ │
│ ├── hf_cache/ # Cached Stanford Cars dataset (gitignored)
│ ├── raw/ # (optional)
│ ├── processed/ # (optional)
│ └── classes.txt
│
├── models/
│ ├── stanford_cars_resnet18_head_subset.pt
│ ├── stanford_cars_resnet18_head_subset.onnx
│ └── stanford_cars_resnet18_head_subset.onnx.data
│
├── notebooks/
│ └── 1.0-FB-initial-experiments.ipynb
│
├── reports/
│ ├── predictions_custom_images.csv
│ └── figures/
│ ├── loss_curve.png
│ ├── accuracy_curve.png
│ ├── gradcam_example_1.png
│ └── gradcam_example_2.png
│
├── src/
│ ├── train.py
│ └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
