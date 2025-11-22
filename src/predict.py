import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd


IMG_SIZE = 160


def load_class_names(classes_path: Path):
    """Load class names from a text file (one per line)."""
    with classes_path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()
                 if line.strip() and not line.startswith("#")]
    return names


def build_model(num_classes: int, ckpt_path: Path, device: torch.device):
    """Build ResNet-18 classifier and load weights."""
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def parse_label(label_str: str):
    """
    Parse a full label string into make, model, year.

    Example:
        "BMW X3 SUV 2012" -> ("BMW", "X3 SUV", "2012")
    """
    parts = label_str.split()
    year = parts[-1] if parts and parts[-1].isdigit() else None
    make = parts[0] if parts else ""
    if year:
        model = " ".join(parts[1:-1])
    else:
        model = " ".join(parts[1:])
    return make, model, year


def predict_folder(
    images_dir: Path,
    model: nn.Module,
    class_names,
    device: torch.device,
    output_csv: Path,
):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    records = []

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            idx = pred_idx.item()
            confidence = conf.item()
            label_str = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            make, model_name, year = parse_label(label_str)

            records.append({
                "image_path": str(img_path),
                "pred_label": label_str,
                "pred_make": make,
                "pred_model": model_name,
                "pred_year": year,
                "confidence": confidence,
            })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(df)
    print(f"\nSaved predictions to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch prediction for car images")
    parser.add_argument("--images", type=str, required=True,
                        help="Folder with input images (jpg/png)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pt model")
    parser.add_argument("--classes", type=str, required=True,
                        help="Path to classes.txt file (one label per line)")
    parser.add_argument("--output", type=str, default="reports/predictions_custom_images.csv",
                        help="Path to output CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    images_dir = Path(args.images)
    ckpt_path = Path(args.model)
    classes_path = Path(args.classes)
    output_csv = Path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    class_names = load_class_names(classes_path)
    print(f"Loaded {len(class_names)} class names from {classes_path}")

    model = build_model(len(class_names), ckpt_path, device)

    predict_folder(images_dir, model, class_names, device, output_csv)
