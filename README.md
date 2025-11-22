## 🚗TwinCar — Fine-Grained Car Classification (ResNet-18)

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
```text
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


---

 ## 📊 Model & Results

- **Architecture:** ResNet-18 (pretrained on ImageNet)
- **Classes:** 196 (make + model + year)
- **Fine-tuning:** Only the classification head
- **Loss:** Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 3 (subset training – demonstration purpose)

Saved model files:

```text
models/
├── stanford_cars_resnet18_head_subset.pt
├── stanford_cars_resnet18_head_subset.onnx
└── stanford_cars_resnet18_head_subset.onnx.data



---

📈 Evaluation & Visualizations

reports/figures/

loss_curve.png

accuracy_curve.png

gradcam_example_1.png

gradcam_example_2.png

These show the training progression and model performance over time.

🔍 Explainability (Grad-CAM)

Grad-CAM visualizations highlight which parts of the image the model focuses on during prediction.

Examples are available in:

reports/figures/

The model mainly focuses on:

Car body shape

Headlights

Grille

Overall silhouette

This confirms the model learned relevant car features, not background noise.
🔮 Custom Image Prediction

You can test your own car images.

Place images here:

data/external/


Run prediction using:

python src/predict.py


Output file:

reports/predictions_custom_images.csv


Columns inside the CSV:

image_path

pred_label

confidence

pred_make

pred_model

pred_year

Example:

image_path	pred_label	confidence
test_4276.jpg	Ferrari 458 Italia Convertible	0.032
test_4692.jpg	Mitsubishi Lancer Sedan	0.033
▶️ How to Run the Project
1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

2. Install requirements
pip install -r requirements.txt

3. Run the notebook
jupyter notebook


Open:

notebooks/1.0-FB-initial-experiments.ipynb

OR run through scripts:
python src/train.py
python src/predict.py
