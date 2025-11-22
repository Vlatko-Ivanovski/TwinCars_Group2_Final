import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 160
BATCH_SIZE = 32
VAL_RATIO = 0.2
MAX_TRAIN_SAMPLES = 1500
MAX_VAL_SAMPLES = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_DEFAULT = 10


class CarsDataset(Dataset):
    """Simple wrapper: Hugging Face dataset -> PyTorch Dataset"""
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        label = item["label"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def build_dataloaders(cache_dir: Path):
    """Load Stanford Cars from Hugging Face and build train/val loaders."""
    print("Loading Stanford Cars dataset from Hugging Face...")
    hf_ds = load_dataset("tanganke/stanford_cars", cache_dir=str(cache_dir))

    train_hf = hf_ds["train"]
    test_hf = hf_ds["test"]

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds_full = CarsDataset(train_hf, train_tf)
    test_ds = CarsDataset(test_hf, eval_tf)

    # Train/val split
    val_size = int(len(train_ds_full) * VAL_RATIO)
    train_size = len(train_ds_full) - val_size

    train_ds, val_ds = random_split(
        train_ds_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Reduce subsets for faster training
    if len(train_ds) > MAX_TRAIN_SAMPLES:
        idx_train = torch.randperm(len(train_ds))[:MAX_TRAIN_SAMPLES]
        train_ds = Subset(train_ds, idx_train)
        print(f"Using reduced train subset of size: {len(train_ds)}")

    if len(val_ds) > MAX_VAL_SAMPLES:
        idx_val = torch.randperm(len(val_ds))[:MAX_VAL_SAMPLES]
        val_ds = Subset(val_ds, idx_val)
        print(f"Using reduced val subset of size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(hf_ds["train"].features["label"].names)
    print("Number of classes:", num_classes)

    return train_loader, val_loader, num_classes


def build_resnet18_classifier(num_classes: int, fine_tune: bool = False):
    """ResNet-18 classifier with optional fine-tuning."""
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def run_one_epoch(model, loader, optimizer, criterion, device, train_mode: bool = True):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    cache_dir: Path,
    output_path: Path,
    epochs: int = EPOCHS_DEFAULT,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    fine_tune: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, num_classes = build_dataloaders(cache_dir=cache_dir)
    model = build_resnet18_classifier(num_classes, fine_tune=fine_tune).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, optimizer, criterion, device, train_mode=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, optimizer, criterion, device, train_mode=False
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  -> New best model saved to {output_path} (val_acc={val_acc:.4f})")

    print("Training complete. Best val_acc:", best_val_acc)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Stanford Cars")
    parser.add_argument("--cache_dir", type=str, default="data/hf_cache",
                        help="Cache directory for Hugging Face dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--fine_tune", action="store_true",
                        help="If set, fine-tune the whole backbone")
    parser.add_argument("--output", type=str,
                        default="models/stanford_cars_resnet18_head_subset.pt",
                        help="Path to save the best model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train(
        cache_dir=cache_dir,
        output_path=output_path,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fine_tune=args.fine_tune,
    )
