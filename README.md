🚗 TwinCar – Fine-Grained Car Classification (ResNet-18)

Final Machine Learning Project – Group 2

Authors:

Emilijan Panpur

Filip Blazevski

Vlatko Ivanovski

Academy: Brainster Data Science Academy — Machine Learning Module (2025)
Framework: PyTorch
Model: ResNet-18 (Transfer Learning)
Dataset: Stanford Cars (196 classes)

📌 Overview

TwinCar is a deep learning project for automatic recognition of:

✅ Car make

✅ Car model

✅ Car production year

The model is trained on the Stanford Cars dataset (196 classes) using ResNet-18 with transfer learning.

This project demonstrates a complete Machine Learning pipeline:

Dataset loading & preprocessing

CNN model training

Evaluation with metrics & visualizations

Grad-CAM explainability

Custom image prediction

Model export (.pt and .onnx)

Proper project structuring

📁 Project Structure
TwinCars_Group2_Final
│
├── data/
│   ├── external/              # Custom images for prediction
│   ├── hf_cache/               # Cached Stanford Cars dataset (Git ignored)
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
└── README.md

🧠 Model Architecture

Backbone: ResNet-18 (pretrained on ImageNet)

Final layer adapted for 196 car classes

Loss: CrossEntropyLoss

Optimizer: Adam

Trained for: 3 epochs (demonstration purpose)

✅ Even though 3 epochs is low, it is acceptable for demo/academic project when combined with transfer learning.

📊 Training Visualization

Saved in:
reports/figures/

loss_curve.png

accuracy_curve.png

These plots show the training dynamics.

🔍 Explainability – Grad-CAM

Grad-CAM visualizations are generated and saved in:

reports/figures/
│
├── gradcam_example_1.png
└── gradcam_example_2.png


They demonstrate that the model focuses mainly on:

Car body shape

Headlights

Front grille

Overall vehicle silhouette

✅ This confirms meaningful learning, not background bias.

🔮 Custom Image Prediction
Place your images here:
data/external/


Supported formats:

.jpg

.jpeg

.png

Run prediction

From script:

python src/predict.py


Or from notebook:

notebooks/1.0-FB-initial-experiments.ipynb

Output file
reports/predictions_custom_images.csv


Contains:

image_path

pred_label

confidence

▶️ How to Run the Project
1. Create & activate environment
python -m venv venv
venv\Scripts\activate

2. Install requirements
pip install -r requirements.txt

3. Run notebook
jupyter notebook


Open:

notebooks/1.0-FB-initial-experiments.ipynb


OR run scripts:

python src/train.py
python src/predict.py

✅ Final Notes

The project is complete and functional

Structure follows ML best practices

Suitable for GitHub presentation

Ready for Brainster final submission

If needed in the future:

Increase epochs (10–30)

Use ResNet-50 / EfficientNet

Add Web App (Streamlit / HF Spaces)
