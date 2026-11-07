"""
Configuration module for Butterfly vs Moth Binary Classification.
Centralizes hyperparameters, directory paths, and biological feature references.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
SAMPLES_DIR = BASE_DIR / "static" / "samples"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & Image Specifications
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
CLASSES = ["butterfly", "moth"]
CLASS_TO_IDX = {"butterfly": 0, "moth": 1}
IDX_TO_CLASS = {0: "butterfly", 1: "moth"}

# Model Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 15
RANDOM_SEED = 42

# Entomological Morphological Traits (Domain Knowledge)
SPECIES_ATTRIBUTES = {
    "butterfly": {
        "title": "Butterfly (Rhopalocera)",
        "antennae": "Slender with distinct clubbed or knobbed tips",
        "wings_at_rest": "Typically folded vertically upright above the thorax",
        "body_shape": "Slender, smooth, streamlined abdomen",
        "activity": "Primarily diurnal (active during bright sunlight)",
        "wing_texture": "Vivid scales with iridescent structural colors & crisp margins",
        "frenulum": "Absent (hindwings and forewings overlap without a bristle lock)",
        "badge_color": "#10b981", # Emerald
    },
    "moth": {
        "title": "Moth (Heterocera)",
        "antennae": "Feathery (bipectinate), comb-like, or tapered without clubs",
        "wings_at_rest": "Folded flat, tent-like, or wrapped horizontally over abdomen",
        "body_shape": "Stout, thick, heavily cloaked in fuzzy/woolly setae",
        "activity": "Predominantly nocturnal or crepuscular",
        "wing_texture": "Cryptic camouflage, mottled earthy tones (or soft pastel veining)",
        "frenulum": "Present (small spine/bristle mechanism locks wings in flight)",
        "badge_color": "#f59e0b", # Amber
    }
}
