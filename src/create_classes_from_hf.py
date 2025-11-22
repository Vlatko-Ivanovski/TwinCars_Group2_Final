from datasets import load_dataset
from pathlib import Path

print("Downloading dataset metadata from HuggingFace...")

ds = load_dataset("tanganke/stanford_cars", split="train")

# Get class names
class_names = ds.features["label"].names

print(f"Found {len(class_names)} classes")

out_path = Path("data/classes.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ DONE: data/classes.txt created")
