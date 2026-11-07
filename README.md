# 🦋 Lepidoptera AI — Butterfly vs Moth Binary Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Validation Accuracy](https://img.shields.io/badge/Val%20Accuracy-100%25-10B981?style=flat-square)](https://github.com/sarathpb/Lepidoptera-AI---Butterfly-vs-Moth-Binary-Classification)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM%20Enabled-8B5CF6?style=flat-square)](https://github.com/sarathpb/Lepidoptera-AI---Butterfly-vs-Moth-Binary-Classification)

An end-to-end, production-grade Computer Vision and Machine Learning system for distinguishing **Butterflies** (*Rhopalocera*) from **Moths** (*Heterocera*). Built with clean object-oriented architecture, dual-model consensus (Deep Convolutional Neural Network + Handcrafted Morphological Random Forest), Grad-CAM visual explainability, and an interactive modern glassmorphic web application.

---

## 📸 Interface Preview

![Lepidoptera AI Application Interface](Screenshots/Screenshot%202026-10-19%20203433.png)
*Figure 1: Modern obsidian glassmorphic dashboard featuring instant specimen classification, circular confidence gauge, and side-by-side Grad-CAM heatmaps.*

![Biological Diagnostics and Analysis](Screenshots/Screenshot%202026-10-19%20203855.png)
*Figure 2: Real-time morphological diagnostics, probability distribution breakdown, and entomological reasoning.*

---

## 🧠 Binary Classification Model Details

### 1. Problem Formulation
Binary classification of Lepidoptera order into two distinct suborders:
$$\mathcal{Y} \in \{0, 1\} \quad \text{where } 0 \to \text{Butterfly (\textit{Rhopalocera})}, \; 1 \to \text{Moth (\textit{Heterocera})}$$

Distinguishing butterflies from moths is a non-trivial computer vision task due to high intra-class variance and subtle inter-class convergent traits (e.g., day-flying moths exhibiting bright coloration, or dull-colored butterflies mimicking dry leaves). To achieve high robustness, this project employs a **Dual-Engine Consensus Model** that blends deep spatial representations with biologically grounded feature engineering and Bayesian priors.

```
Input Image (224x224x3)
  ├──► [Engine A: PyTorch ButterflyMothCNN] ─────────► P_cnn (Softmax Logits) ──┐
  │         └──► Grad-CAM Attention Heatmap Hook                                │
  │                                                                             ├──► Dynamic Consensus Fusion ──► Final Classification
  ├──► [Feature Extractor (104-D Vector)] ──► [Engine B: Random Forest] ───────► P_rf  │    & Confidence Score
  │                                                                             │
  └──► [Morphological Interpretable Metrics] ──► Biological Bayesian Prior ─────┘
            (Aspect Ratio, Vibrancy Index, Camouflage Density, Entropy)
```

---

### 2. Deep Learning Engine: PyTorch `ButterflyMothCNN`

The deep convolutional model is a tailored 4-stage hierarchical feature extractor with batch normalization, leaky activations, spatial regularization, and integrated gradient hooks for explainability.

#### Layer-by-Layer Architecture Specification:

| Stage / Block | Layer Specification | Output Dimensions | Biological Feature Abstraction |
| :--- | :--- | :--- | :--- |
| **Input** | RGB Image Tensor (Normalized) | $3 \times 224 \times 224$ | Raw pixel grid |
| **Conv Block 1** | $\text{Conv2D}(3 \to 32, k=3, p=1)$<br>$\text{BatchNorm2d}(32)$<br>$\text{LeakyReLU}(\alpha=0.1)$<br>$\text{MaxPool2d}(2 \times 2)$ | $32 \times 112 \times 112$ | Low-level micro-textures, edge boundaries, antenna filaments |
| **Conv Block 2** | $\text{Conv2D}(32 \to 64, k=3, p=1)$<br>$\text{BatchNorm2d}(64)$<br>$\text{LeakyReLU}(\alpha=0.1)$<br>$\text{MaxPool2d}(2 \times 2)$ | $64 \times 56 \times 56$ | Wing venation patterns, submarginal bands, eye-spots (*ocelli*) |
| **Conv Block 3** | $\text{Conv2D}(64 \to 128, k=3, p=1)$<br>$\text{BatchNorm2d}(128)$<br>$\text{LeakyReLU}(\alpha=0.1)$<br>$\text{MaxPool2d}(2 \times 2)$ | $128 \times 28 \times 28$ | Wing contours, thoracic scale profiles, abdominal silhouette |
| **Conv Block 4** | $\text{Conv2D}(128 \to 256, k=3, p=1)$<br>$\text{BatchNorm2d}(256)$<br>$\text{LeakyReLU}(\alpha=0.1)$<br>*(Grad-CAM hook registered here)* | $256 \times 28 \times 28$ | High-level semantic abstractions, body-to-wing spatial posture |
| **Pooling & Head** | $\text{MaxPool2d}(2 \times 2)$<br>$\text{AdaptiveAvgPool2d}((1, 1))$<br>$\text{Dropout}(p=0.30)$<br>$\text{Linear}(256 \to 64) + \text{ReLU}$<br>$\text{Linear}(64 \to 2)$ | $2 \times 1$ (Logits) | Class probability logits for [Butterfly, Moth] |

#### Hyperparameters & Optimization:
- **Loss Function**: Cross-Entropy Loss with L2 weight decay ($10^{-4}$)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, learning rate $\eta = 10^{-3}$)
- **LR Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=3, monitored on validation accuracy)
- **Data Augmentation**: Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Color Jitter (Brightness/Contrast/Saturation $\pm 0.15$)

---

### 3. Classical Machine Learning Engine: Morphological `RandomForestClassifier`

A parallel Scikit-Learn ensemble trained on a 104-dimensional biological and photometric feature vector:

1. **Color Histograms (32 bins)**:
   - 16-bin Hue distribution in $[0^\circ, 360^\circ]$
   - 8-bin Saturation distribution in $[0, 1]$
   - 8-bin Value/Brightness distribution in $[0, 1]$
2. **Statistical Color Moments (12 features)**:
   - Channel-wise mean and standard deviation across RGB and HSV spaces
3. **Texture & Edge Density (8 bins)**:
   - Sobel/Laplacian spatial gradient magnitude distribution detecting cryptic bark/lichen textures
4. **Spatial Block Color Grid (48 features)**:
   - $4 \times 4$ spatial block grid capturing localized color composition across thorax and wing quadrants
5. **Entomological Diagnostic Indicators (7 features)**:
   - **Aspect Ratio**: Resting posture profile ($\ge 1.30$ indicates flat/tent-like moth wings; $< 1.30$ indicates upright folded butterfly wings).
   - **Vibrancy Ratio**: Fraction of high-saturation pixels excluding background foliage.
   - **Camouflage Edge Density**: Frequency of high-contrast micro-edges reflecting bark mimicry.
   - **Chromatic Entropy**: Shannon entropy over hue distributions.
   - **Foliage Suppression Ratio**: Normalization factor that eliminates background plant hues.

**Hyperparameters**:
- 100 decision trees (`n_estimators=100`)
- Maximum tree depth: 8 (`max_depth=8`)
- Class weighting: `class_weight="balanced"`
- Standard feature scaling: `StandardScaler`

---

### 4. Dynamic Consensus & Bayesian Prior Fusion

The final output is computed through a dynamic consensus weighting scheme:

$$P_{\text{final}}(c) = w_{\text{cnn}} P_{\text{cnn}}(c) + w_{\text{rf}} P_{\text{rf}}(c) + w_{\text{morph}} P_{\text{prior}}(c)$$

Where:
- When strong biological morphological markers are detected (e.g. wide spread-wing posture $\ge 1.30$ or iridescent structural blue/emerald coloration), $w_{\text{morph}}$ dynamically increases to $0.50$ to anchor deep neural predictions with domain rules.
- When signals are balanced, weights distribute evenly ($w_{\text{cnn}}=0.35, w_{\text{rf}}=0.30, w_{\text{morph}}=0.35$).

---

### 5. Explainable AI: Grad-CAM Saliency Heatmaps

To eliminate "black-box" behavior, the system calculates Gradient-weighted Class Activation Mapping (Grad-CAM) over the final convolutional layer:

1. Gradients of the target class score $y^c$ with respect to feature map activations $A^k$ are globally pooled:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
2. The attention heatmap is generated via linear combination and ReLU rectification:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. The normalized attention map is interpolated to $224 \times 224$ and blended with the original specimen image using a perceptual Jet/Plasma color palette.

---

## 📊 Empirical Validation & Performance

Evaluation results on the validation split:

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **100.0%** | Both CNN and Random Forest achieved 100% on validation split |
| **Training Accuracy** | **100.0%** | Convergence achieved within 12 epochs |
| **Precision (Butterfly)** | **100.0%** | Zero false positives on validation set |
| **Recall (Butterfly)** | **100.0%** | Zero false negatives on validation set |
| **F1-Score** | **1.000** | Perfect harmonic mean between precision and recall |
| **Final Training Loss** | **0.0901** | Stable cross-entropy convergence |
| **Final Validation Loss** | **0.0823** | Generalizes cleanly without overfitting |

### Confusion Matrix:

```
                  Predicted Butterfly    Predicted Moth
Actual Butterfly          25                   0
Actual Moth                0                  25
```

---

## 🔬 Entomological Grounding: Butterfly vs Moth

| Morphological Trait | Butterfly (*Rhopalocera*) | Moth (*Heterocera*) |
| :--- | :--- | :--- |
| **Antennae** | Slender, filament-like ending in distinct **clubbed or knobbed tips** | **Feathery (bipectinate)**, comb-like, or tapered without clubs |
| **Wings at Rest** | Typically folded **vertically upright** above the thorax | Folded **flat, tent-like**, or wrapped horizontally over abdomen |
| **Body Morphology** | Slender, streamlined, smooth cuticle | **Stout, thick, robust**, heavily cloaked in fuzzy setae (wool) |
| **Circadian Cycle** | **Diurnal** (active during bright sunlight) | Predominantly **nocturnal** or crepuscular |
| **Wing Coupling** | Overlapping wing margins (no frenulum) | **Frenulum present** (bristle hook connecting hindwing & forewing) |
| **Color Strategy** | High chromatic vibrancy, structural iridescence | Cryptic mottled camouflage, bark and lichen mimicry |

---

## 📂 Project Architecture

```
Lepidoptera-AI/
├── app.py                      # Flask Application Server & REST API
├── predict.py                  # CLI Inference Tool with diagnostic output
├── train.py                    # CLI Training Pipeline (CNN + Classical RF)
├── requirements.txt            # Project dependencies
├── run_app.bat                 # One-click Windows application launcher
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── data/                       # Dataset splits (train/val)
│   ├── train/                  # Training set (butterfly / moth)
│   └── val/                    # Validation set (butterfly / moth)
├── saved_models/               # Serialized models and training reports
│   ├── butterfly_moth_cnn.pt   # PyTorch CNN trained weights
│   ├── morphological_classifier.pkl # Scikit-Learn Random Forest model
│   ├── pipeline_summary.json   # Full pipeline summary report
│   └── training_report.json    # Epoch-by-epoch loss and accuracy metrics
├── Screenshots/                # UI and inference screenshots
│   ├── Screenshot 2026-10-19 203433.png
│   └── Screenshot 2026-10-19 203855.png
├── src/                        # Core algorithmic package
│   ├── __init__.py
│   ├── config.py               # Paths, biological constants, hyperparameters
│   ├── dataset_generator.py    # Procedural Lepidoptera specimen synthesizer
│   ├── features.py             # HSV vibrancy, Sobel edge mottling & feature extraction
│   ├── models.py               # PyTorch CNN & Scikit-Learn classifiers
│   ├── trainer.py              # End-to-end training loops & validation tracking
│   └── inference.py            # Unified inference engine & Grad-CAM overlay generator
├── static/
│   ├── css/style.css           # Modern obsidian glassmorphic design system
│   ├── js/app.js               # Client controller (upload, gallery, animations)
│   └── samples/                # Showcase biological specimens for instant testing
└── templates/
    └── index.html              # Modern semantic HTML5 interface
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/sarathpb/Lepidoptera-AI---Butterfly-vs-Moth-Binary-Classification.git
cd Lepidoptera-AI---Butterfly-vs-Moth-Binary-Classification
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

**Features in the Web App:**
- **Drag-and-Drop Uploader**: Test any butterfly or moth image (JPEG, PNG, WEBP).
- **Curated Specimen Gallery**: 8 biological specimens (*Monarch, Blue Morpho, Swallowtail, Emerald Swallowtail, Luna Moth, Peppered Moth, Hawk Moth, Atlas Silk Moth*) for 1-click classification.
- **Grad-CAM Visualization Toggle**: Switch between Original image, Attention Heatmap, and Side-by-Side view.
- **Model Metrics Modal**: View confusion matrix, validation scores, and architectural details directly in the UI.

### 3. Command-Line Inference
Run inference directly from the terminal:
```bash
# Test on a built-in curated sample:
python predict.py --sample monarch_butterfly.jpg
python predict.py --sample luna_moth.jpg

# Test on any external image:
python predict.py --image path/to/specimen.jpg
```

**Sample CLI Output:**
```
============================================================
  LEPIDOPTERA CLASSIFIER - INFERENCE RESULT
============================================================
  File: monarch_butterfly.jpg
  Classification : BUTTERFLY (Confidence: 86.10%)
  Class Probs    : Butterfly: 86.1% | Moth: 13.9%
------------------------------------------------------------
  Biological & Morphological Diagnostic Signals:
   * Color Saturation   : 37.1% (Vibrant Chromatic Ratio: 21.4%)
   * Camouflage Density : 7.2%
   * Aspect Ratio       : 1.00 (Upright / Compact)
   * Color Entropy      : 0.37
------------------------------------------------------------
  Taxonomic Reference (Butterfly (Rhopalocera)):
   * Antennae         : Slender with distinct clubbed or knobbed tips
   * Wings at Rest    : Typically folded vertically upright above the thorax
   * Body Morphology  : Slender, smooth, streamlined abdomen
   * Activity Pattern : Primarily diurnal (active during bright sunlight)
============================================================
```

### 4. Training the Models
To retrain both the CNN and Random Forest models:
```bash
# Train on existing dataset:
python train.py --epochs 12

# Synthesize fresh biological specimens and retrain:
python train.py --generate-data --epochs 12
```

---

## 🌐 REST API Endpoints

The web application exposes a clean REST API:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the main glassmorphic application UI |
| `/api/predict` | `POST` | Accepts `image` file or `sample_name`, returns classification, probabilities, morphological diagnostics, and base64 Grad-CAM overlay |
| `/api/samples` | `GET` | Returns list of curated specimens available in the gallery |
| `/api/metrics` | `GET` | Returns validation metrics, confusion matrix, and training history |

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
