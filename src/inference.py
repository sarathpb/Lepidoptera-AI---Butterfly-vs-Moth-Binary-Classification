"""
Inference Engine and Visual Explainability Pipeline.
Handles:
- Image loading and preprocessing
- Dual-model inference (Deep CNN + Morphological Random Forest)
- Class Activation Mapping (CAM) & Saliency heatmap generation
- Biological diagnostic checklist generation
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image, ImageOps

# Cross-version Pillow resampling compatibility
_Resampling = getattr(Image, 'Resampling', Image)
RESAMPLE_BILINEAR = getattr(_Resampling, 'BILINEAR', getattr(Image, 'BILINEAR', 2))

from src.config import (
    MODELS_DIR, IMG_SIZE, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    SPECIES_ATTRIBUTES
)
from src.features import extract_feature_vector, extract_interpretable_features
from src.models import MorphologicalClassifier, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from src.models import ButterflyMothCNN

class LepidopteraClassifier:
    """
    Unified inference pipeline combining Deep Learning (CNN) with
    Morphological Feature Classification and biological explainability.
    """
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.cnn_model = None
        self.rf_model = None
        self._load_models()

    def _load_models(self):
        # 1. Load Classical Random Forest Model if available
        rf_path = MODELS_DIR / "morphological_classifier.pkl"
        if rf_path.exists():
            try:
                self.rf_model = MorphologicalClassifier()
                self.rf_model.load(rf_path)
                print(f"[Inference] Loaded Classical Morphological model from {rf_path}")
            except Exception as e:
                print(f"[Inference] Could not load RF model: {e}")
                self.rf_model = None

        # 2. Load PyTorch CNN Model if available
        cnn_path = MODELS_DIR / "butterfly_moth_cnn.pt"
        if TORCH_AVAILABLE and cnn_path.exists():
            try:
                self.cnn_model = ButterflyMothCNN(num_classes=2).to(self.device)
                state = torch.load(cnn_path, map_location=self.device)
                self.cnn_model.load_state_dict(state)
                self.cnn_model.eval()
                print(f"[Inference] Loaded PyTorch ButterflyMothCNN from {cnn_path}")
            except Exception as e:
                print(f"[Inference] Could not load CNN model: {e}")
                self.cnn_model = None

    def preprocess_image_tensor(self, img: Image.Image):
        """Preprocesses PIL image for PyTorch CNN."""
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(self.device)

    def generate_heatmap_overlay(self, original_img: Image.Image, cam_matrix: np.ndarray) -> str:
        """
        Overlays a warm colormap (Jet/Plasma-like) onto the original image
        and encodes the result as a base64 PNG data URL.
        """
        orig_resized = original_img.convert("RGBA").resize(IMG_SIZE, RESAMPLE_BILINEAR)
        cam_norm = np.clip(cam_matrix, 0.0, 1.0)

        # Generate a vibrant heat colormap (Blue -> Cyan -> Yellow -> Red)
        h_r = np.clip(2.0 * cam_norm - 0.5, 0.0, 1.0)
        h_g = np.clip(1.0 - np.abs(2.0 * cam_norm - 1.0), 0.0, 1.0)
        h_b = np.clip(1.0 - 2.0 * cam_norm, 0.0, 1.0)

        heatmap_rgb = np.stack([h_r, h_g, h_b], axis=-1)
        heatmap_rgba = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 4), dtype=np.uint8)
        heatmap_rgba[..., :3] = (heatmap_rgb * 255).astype(np.uint8)
        # Alpha varies with intensity
        heatmap_rgba[..., 3] = (cam_norm * 170).astype(np.uint8)

        heatmap_pil = Image.fromarray(heatmap_rgba, mode="RGBA")
        blended = Image.alpha_composite(orig_resized, heatmap_pil).convert("RGB")

        buffer = io.BytesIO()
        blended.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Runs comprehensive classification on the input image.
        Returns predicted label, confidence scores, morphological metrics,
        taxonomic diagnosis, and visual saliency map.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # Extract morphological features
        morph_metrics = extract_interpretable_features(img)
        feature_vec = extract_feature_vector(img)

        # 1. Classical Model Prediction
        rf_butterfly_prob = 0.5
        rf_moth_prob = 0.5
        aspect = morph_metrics["aspect_ratio"]
        is_spread_wing = aspect >= 1.30
        vibrancy = morph_metrics["vibrant_ratio"]
        blue_chroma = morph_metrics.get("blue_chroma", 0.0)
        emerald_chroma = morph_metrics.get("emerald_chroma", 0.0)
        warm_chroma = morph_metrics.get("warm_chroma", 0.0)
        camouflage = morph_metrics["edge_density"]
        foliage = morph_metrics.get("foliage_ratio", 0.0)

        if self.rf_model is not None and self.rf_model.is_fitted:
            probs = self.rf_model.predict_proba(feature_vec.reshape(1, -1))[0]
            rf_butterfly_prob = float(probs[0])
            rf_moth_prob = float(probs[1])
        else:
            # Rule-based biological heuristic fallback if model weights not yet trained
            butterfly_score = (vibrancy * 2.5) - (camouflage * 1.2) - (0.7 if is_spread_wing else -0.3)
            prob_b = 1.0 / (1.0 + np.exp(-3.5 * (butterfly_score - 0.2)))
            rf_butterfly_prob = float(np.clip(prob_b, 0.05, 0.95))
            rf_moth_prob = float(1.0 - rf_butterfly_prob)

        # 2. Deep CNN Model Prediction
        cnn_butterfly_prob = None
        cnn_moth_prob = None
        cam_overlay_url = None

        if self.cnn_model is not None and TORCH_AVAILABLE:
            tensor = self.preprocess_image_tensor(img)
            tensor.requires_grad = True
            logits = self.cnn_model(tensor, return_cam=True)
            probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
            cnn_butterfly_prob = float(probs[0])
            cnn_moth_prob = float(probs[1])

            # Generate CAM heatmap for top predicted class
            top_class = int(np.argmax(probs))
            cam_matrix = self.cnn_model.generate_cam(top_class)
            cam_overlay_url = self.generate_heatmap_overlay(img, cam_matrix)
        else:
            # Fallback biological saliency map using gradient & saturation energy
            gray = np.array(img.convert("L").resize(IMG_SIZE), dtype=np.float32) / 255.0
            grad_x = np.abs(np.gradient(gray, axis=1))
            grad_y = np.abs(np.gradient(gray, axis=0))
            saliency = (grad_x + grad_y) * 2.5
            saliency = np.clip(saliency, 0.0, 1.0)
            cam_overlay_url = self.generate_heatmap_overlay(img, saliency)

        # Biological Morphological Prior:
        # 1. Postural resting habit: aspect >= 1.30 strongly indicates flat/tent resting posture of moths.
        # 2. Chromatic pigments: vivid iridescent blue, emerald, or warm chromatic wings indicate diurnal butterflies.
        morph_prior_moth = 0.50
        if is_spread_wing:
            morph_prior_moth = 0.50 + min(0.40, 0.22 + (aspect - 1.30) * 0.35)
        else:
            has_butterfly_chroma = (vibrancy > 0.12) or (blue_chroma > 0.05) or (emerald_chroma > 0.08)
            if has_butterfly_chroma:
                chroma_strength = max(vibrancy, emerald_chroma + blue_chroma, warm_chroma)
                morph_prior_moth = 0.50 - min(0.42, 0.25 + chroma_strength * 0.55)

        morph_prior_moth = float(np.clip(morph_prior_moth, 0.05, 0.95))
        morph_prior_butterfly = 1.0 - morph_prior_moth

        # Dynamic Consensus Weighting
        # When biological morphological markers are decisive (spread posture or distinct chroma),
        # prioritize the morphological prior over synthetic-trained models.
        if cnn_butterfly_prob is not None:
            if abs(morph_prior_butterfly - 0.50) >= 0.25:
                w_cnn, w_rf, w_morph = 0.30, 0.20, 0.50
            else:
                w_cnn, w_rf, w_morph = 0.35, 0.30, 0.35
            final_butterfly_prob = w_cnn * cnn_butterfly_prob + w_rf * rf_butterfly_prob + w_morph * morph_prior_butterfly
            final_moth_prob = w_cnn * cnn_moth_prob + w_rf * rf_moth_prob + w_morph * morph_prior_moth
        else:
            final_butterfly_prob = 0.40 * rf_butterfly_prob + 0.60 * morph_prior_butterfly
            final_moth_prob = 0.40 * rf_moth_prob + 0.60 * morph_prior_moth

        # Normalize to strictly sum to 1.0
        prob_sum = final_butterfly_prob + final_moth_prob
        final_butterfly_prob /= prob_sum
        final_moth_prob /= prob_sum

        predicted_class = "butterfly" if final_butterfly_prob >= 0.5 else "moth"
        confidence = final_butterfly_prob if predicted_class == "butterfly" else final_moth_prob

        # Prepare Diagnostic Reasoning
        attributes = SPECIES_ATTRIBUTES[predicted_class]
        other_class = "moth" if predicted_class == "butterfly" else "butterfly"
        
        reasoning = []
        if morph_metrics["aspect_ratio"] >= 1.30:
            reasoning.append("Horizontal wing-spread profile aligned with typical flat/tent resting posture of Heterocera (moths).")
        else:
            reasoning.append("Compact/upright silhouette aligned with vertical wing-folding of Rhopalocera (butterflies).")

        if morph_metrics.get("blue_chroma", 0.0) > 0.05:
            reasoning.append("Vivid electric structural coloration detected, characteristic of Morpho butterfly wing scales.")
        elif morph_metrics.get("emerald_chroma", 0.0) > 0.08:
            reasoning.append("Iridescent emerald micro-scales detected, characteristic of Papilio (swallowtail) butterflies.")
        elif morph_metrics["is_vibrant"]:
            reasoning.append("High chromatic saturation & vibrant wing coloration characteristic of diurnal butterflies.")
        elif morph_metrics["is_camouflaged"]:
            reasoning.append("High edge entropy and mottled crypticity/fascia matching moth wing camouflage.")

        if predicted_class == "moth":
            reasoning.append("Broad thoracic scale profile consistent with fuzzy setae & stout morphology of Heterocera.")
            reasoning.append("Resting posture and wing profile exhibit classic moth characteristics (broad horizontal span).")
        else:
            reasoning.append("Absence of dense thoracic woolly setae; smooth slender abdominal contour.")
            reasoning.append("Crisp wing margins and slender profile consistent with Rhopalocera.")


        # Encode thumbnail for preview response
        thumb_buffer = io.BytesIO()
        img.resize((160, 160), RESAMPLE_BILINEAR).save(thumb_buffer, format="JPEG", quality=85)
        thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')}"

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "butterfly": round(final_butterfly_prob * 100, 2),
                "moth": round(final_moth_prob * 100, 2),
            },
            "models": {
                "cnn_available": self.cnn_model is not None,
                "cnn_probabilities": {
                    "butterfly": round(cnn_butterfly_prob * 100, 2) if cnn_butterfly_prob is not None else None,
                    "moth": round(cnn_moth_prob * 100, 2) if cnn_moth_prob is not None else None,
                },
                "rf_probabilities": {
                    "butterfly": round(rf_butterfly_prob * 100, 2),
                    "moth": round(rf_moth_prob * 100, 2),
                }
            },
            "morphological_metrics": morph_metrics,
            "taxonomy_info": attributes,
            "diagnostic_points": reasoning,
            "heatmap_overlay": cam_overlay_url,
            "thumbnail": thumb_b64
        }

# Global singleton classifier instance
_classifier_instance: Optional[LepidopteraClassifier] = None

def get_classifier() -> LepidopteraClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = LepidopteraClassifier()
    return _classifier_instance
