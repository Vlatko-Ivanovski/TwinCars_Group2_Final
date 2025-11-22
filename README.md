🚗 TwinCar – Fine-Grained Car Classification (ResNet-18)

Final Machine Learning Project – Group 2
Authors: Emilijan Panpur, Filip Blazevski, Vlatko Ivanovski
Framework: PyTorch
Backbone: ResNet-18 (Transfer Learning)
Dataset: Stanford Cars (196 classes)
Academy: Brainster Data Science Academy – 2025

📌 Overview

TwinCar is a deep-learning project for automatic recognition of car make, model and year from images, built using PyTorch and ResNet-18 with transfer learning.

The main goal of the project is to demonstrate a full Machine Learning pipeline, including:

Dataset loading & preprocessing

CNN model training

Model evaluation & metrics

Custom image prediction

Model explainability with Grad-CAM

Saving trained model (.pt and .onnx)

Reproducible structured project

The model is trained on the Stanford Cars dataset, which contains 196 fine-grained classes of vehicles.

📂 Project Structure
TwinCars_Group2_Final/
│
├── data/
│   ├── external/              # Custom car images for prediction
│   ├── hf_cache/               # Cached Stanford Cars dataset
│   ├── raw/
│   ├── processed/
│   └── classes.txt
│
├── notebooks/
│   └── 1.0-FB-initial-experiments.ipynb
│
├── models/
│   ├── stanford_cars_resnet18_head_subset.pt
│   ├── stanford_cars_resnet18_head_subset.onnx
│   └── stanford_cars_resnet18_head_subset.onnx.data
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

🗂 Dataset

Stanford Cars 196 – via Hugging Face
https://huggingface.co/datasets/tanganke/stanford_cars

196 car classes

Each image labeled with Make + Model + Year

Stored locally in:

data/hf_cache/


For speed, a smaller subset was used:

Split	Images
Train	~1500
Validation	~300
Test	Full test set
🧠 Model Architecture

Backbone: ResNet-18 (pretrained on ImageNet)

Last layer: Fully connected → 196 classes

Frozen feature extractor

Loss: Cross-Entropy

Optimizer: AdamW

Epochs: 3

The model uses transfer learning, training only the final classifier layer.

📊 Example Results (Subset)
Metric	Value
Train Accuracy	~0.80
Validation Accuracy	~0.55
Macro F1	~0.54
Top-3 Accuracy	~0.72

Note: The problem is extremely fine-grained (196 similar classes), therefore even ~55% validation accuracy is a solid result for a 3-epoch fine-tuning baseline.

🔍 Explainability (Grad-CAM)

Grad-CAM heatmaps show where the model focuses when predicting a class.
Examples are saved in:

reports/figures/


These visualizations show the model concentrates on:

Car body

Headlights

Grille

Overall silhouette

This confirms meaningful learning, not background bias.

🔮 Custom Image Prediction

Put images in:

data/external/


Then run prediction:

From notebook
OR

python src/predict.py


Results are saved here:

reports/predictions_custom_images.csv


Columns:

image_path

pred_label

confidence

pred_make

pred_model

pred_year

▶️ How to Run
1. Activate environment
python -m venv venv
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run notebook
jupyter notebook


Open:

notebooks/1.0-FB-initial-experiments.ipynb


OR use scripts:

python src/train.py
python src/predict.py
