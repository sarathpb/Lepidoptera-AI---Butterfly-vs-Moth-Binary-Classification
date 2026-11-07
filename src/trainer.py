"""
Training pipeline for Butterfly vs Moth Classification.
Supports:
1. PyTorch CNN Training with early stopping and learning rate scheduling
2. Classical ML Morphological Classifier Training (Random Forest)
3. Performance evaluation and metrics persistence (loss, accuracy, confusion matrix)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

from src.config import (
    DATA_DIR, MODELS_DIR, IMG_SIZE, CLASSES,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_SEED
)
from src.features import extract_feature_vector
from src.models import MorphologicalClassifier, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from src.models import ButterflyMothCNN

    class LepidopteraDataset(Dataset):
        def __init__(self, root_dir: Path, split: str = "train", transform=None):
            self.image_paths = []
            self.labels = []
            self.transform = transform
            
            split_dir = Path(root_dir) / split
            for class_idx, class_name in enumerate(CLASSES):
                class_folder = split_dir / class_name
                if class_folder.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for p in class_folder.glob(ext):
                            self.image_paths.append(p)
                            self.labels.append(class_idx)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            label = self.labels[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, str(path)


def train_classical_model(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    """
    Trains the classical Morphological Classifier using engineered biological feature vectors.
    """
    print("\n[ML Training] Extracting handcrafted morphological features...")
    splits = ["train", "val"]
    data = {"train": ([], []), "val": ([], [])}

    for split in splits:
        split_dir = data_dir / split
        for class_idx, class_name in enumerate(CLASSES):
            class_folder = split_dir / class_name
            if not class_folder.exists():
                continue
            for img_path in list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    feat = extract_feature_vector(img)
                    data[split][0].append(feat)
                    data[split][1].append(class_idx)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

    X_train = np.array(data["train"][0], dtype=np.float32)
    y_train = np.array(data["train"][1], dtype=np.int64)
    X_val = np.array(data["val"][0], dtype=np.float32)
    y_val = np.array(data["val"][1], dtype=np.int64)

    print(f"Extracted features: Train shape={X_train.shape}, Val shape={X_val.shape}")

    classifier = MorphologicalClassifier()
    start_time = time.time()
    classifier.fit(X_train, y_train)
    duration = time.time() - start_time

    # Evaluate
    val_preds = classifier.predict(X_val)
    val_probs = classifier.predict_proba(X_val)
    val_acc = float(np.mean(val_preds == y_val))

    # Confusion Matrix
    cm = [[0, 0], [0, 0]]
    for true_l, pred_l in zip(y_val, val_preds):
        cm[true_l][pred_l] += 1

    model_path = MODELS_DIR / "morphological_classifier.pkl"
    classifier.save(model_path)
    print(f"[ML Training Complete] Validation Accuracy: {val_acc*100:.2f}% (Saved to {model_path})")

    results = {
        "model_type": "RandomForest_Morphological",
        "val_accuracy": round(val_acc, 4),
        "duration_sec": round(duration, 2),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "confusion_matrix": cm,
    }
    return results


def train_cnn_model(data_dir: Path = DATA_DIR, epochs: int = EPOCHS) -> Dict[str, Any]:
    """
    Trains the ButterflyMothCNN deep learning architecture using PyTorch.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Skipping CNN training.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Deep Learning Training] Initializing PyTorch ButterflyMothCNN on device: {device}")

    # Data Transforms with biological data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LepidopteraDataset(data_dir, split="train", transform=train_transform)
    val_dataset = LepidopteraDataset(data_dir, split="val", transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError(f"No training data found in {data_dir / 'train'}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ButterflyMothCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / "butterfly_moth_cnn.pt"

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_labels.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / (total_val + 1e-8)
        epoch_val_acc = correct_val / (total_val + 1e-8)
        scheduler.step(epoch_val_acc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.1f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_weights_path)

    # Compute Final Confusion Matrix
    cm = [[0, 0], [0, 0]]
    for tl, pl in zip(all_val_labels, all_val_preds):
        cm[tl][pl] += 1

    report = {
        "model_type": "PyTorch_ButterflyMothCNN",
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs_trained": epochs,
        "history": history,
        "confusion_matrix": cm,
        "device": str(device)
    }

    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Deep learning training report saved to {report_path}")

    return report


def run_full_training(epochs: int = 12):
    """
    Executes end-to-end training pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ml_results = train_classical_model()
    cnn_results = train_cnn_model(epochs=epochs) if TORCH_AVAILABLE else {}

    summary = {
        "classical_rf": ml_results,
        "cnn": cnn_results,
        "classes": CLASSES
    }
    with open(MODELS_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nPipeline training complete. Summary saved to saved_models/pipeline_summary.json")
    return summary


if __name__ == "__main__":
    run_full_training()
