"""
Model definitions for Butterfly vs Moth Binary Classification.
Includes:
1. PyTorch ButterflyMothCNN (Lightweight Deep Convolutional Neural Network with CAM hook)
2. Classical Morphological Ensemble Classifier (Scikit-Learn Random Forest / Gradient Boosting)
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


if TORCH_AVAILABLE:
    class ButterflyMothCNN(nn.Module):
        """
        Deep Convolutional Neural Network specifically designed for Lepidoptera classification.
        Architecture includes 4 convolutional blocks with batch normalization, leaky relu,
        spatial dropout for regularization, global average pooling, and activation hooks for Class Activation Maps (CAM).
        """
        def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
            super().__init__()
            
            # Block 1: Low-level edges, scales, and fine antennae features (224x224 -> 112x112)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            # Block 2: Wing patterns, spots, and color transitions (112x112 -> 56x56)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            # Block 3: Morphological wing shapes, body fur, and venation (56x56 -> 28x28)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            
            # Block 4: High-level semantic abstractions (28x28 -> 14x14)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool4 = nn.MaxPool2d(2, 2)
            
            # Global pooling & Regularized Classification Head
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(256, 64)
            self.fc2 = nn.Linear(64, num_classes)
            
            # Hook variables for Grad-CAM / feature saliency
            self.gradients = None
            self.activations = None

        def activations_hook(self, grad):
            self.gradients = grad

        def forward(self, x, return_cam: bool = False):
            # Conv block 1
            x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
            # Conv block 2
            x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
            # Conv block 3
            x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
            # Conv block 4
            x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
            
            if return_cam:
                self.activations = x
                if x.requires_grad:
                    x.register_hook(self.activations_hook)
            
            x_pooled = self.pool4(x)
            x_flat = self.global_pool(x_pooled).view(x_pooled.size(0), -1)
            x_dense = F.relu(self.fc1(self.dropout(x_flat)))
            logits = self.fc2(x_dense)
            return logits

        def generate_cam(self, class_idx: int) -> np.ndarray:
            """
            Computes Class Activation Map (CAM) heatmap for the target class.
            """
            if self.activations is None:
                return np.zeros((224, 224), dtype=np.float32)
            
            if self.gradients is not None:
                # Grad-CAM: weight by pooled gradients
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            else:
                # Direct feature activation magnitude
                cam = torch.mean(self.activations, dim=1, keepdim=True)
                
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam_np = cam.squeeze().detach().cpu().numpy()
            
            # Normalize between 0 and 1
            norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            return norm
else:
    class ButterflyMothCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ButterflyMothCNN.")


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.preprocessing.data':
            try:
                import sklearn.preprocessing.data
            except ImportError:
                module = 'sklearn.preprocessing._data'
        elif module == 'sklearn.preprocessing._data':
            try:
                import sklearn.preprocessing._data
            except ImportError:
                module = 'sklearn.preprocessing.data'
        return super().find_class(module, name)

class MorphologicalClassifier:
    """
    Classical Machine Learning classifier using Random Forest and Gradient Boosting
    trained on handcrafted morphological and photometric Lepidoptera features.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model, "is_fitted": self.is_fitted}, f)

    def load(self, filepath: Path):
        with open(filepath, "rb") as f:
            unpickler = _CompatUnpickler(f)
            data = unpickler.load()
            self.scaler = data["scaler"]
            self.model = data["model"]
            self.is_fitted = data["is_fitted"]

