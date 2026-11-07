"""
Feature engineering and morphological analysis module.
Extracts handcrafted biological features and interpretable visual metrics:
- Color saturation and vibrancy (HSV analysis)
- Texture entropy and edge gradients (Sobel/Laplacian proxy)
- Wing-to-body contrast and resting aspect ratio
- Camouflage pattern density
"""

import numpy as np
from PIL import Image, ImageFilter, ImageStat

# Cross-version Pillow resampling compatibility
_Resampling = getattr(Image, 'Resampling', Image)
RESAMPLE_LANCZOS = getattr(_Resampling, 'LANCZOS', getattr(Image, 'LANCZOS', 1))
RESAMPLE_BILINEAR = getattr(_Resampling, 'BILINEAR', getattr(Image, 'BILINEAR', 2))

def rgb_to_hsv_np(rgb_array: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion from RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1).
    """
    arr = rgb_array.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    diff = c_max - c_min
    
    # Hue calculation
    h = np.zeros_like(c_max)
    mask_r = (c_max == r) & (diff > 1e-5)
    mask_g = (c_max == g) & (diff > 1e-5)
    mask_b = (c_max == b) & (diff > 1e-5)
    
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4.0
    h = h * 60.0
    h[h < 0] += 360.0
    
    # Saturation
    s = np.zeros_like(c_max)
    nz = c_max > 1e-5
    s[nz] = diff[nz] / c_max[nz]
    
    # Value
    v = c_max
    
    return np.stack([h, s, v], axis=-1)

def extract_interpretable_features(image: Image.Image) -> dict:
    """
    Extracts biological morphological and photometric indicators from an image.
    Returns a dictionary of raw metrics and qualitative indicators.
    """
    img_rgb = image.convert("RGB").resize((224, 224), RESAMPLE_LANCZOS)
    np_img = np.array(img_rgb)
    hsv = rgb_to_hsv_np(np_img)
    
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    
    # 1. Color Metrics (excluding dark noise pixels where saturation is noisy)
    valid_light = v > 0.15
    mean_saturation = float(np.mean(s[valid_light])) if np.any(valid_light) else float(np.mean(s))
    max_saturation = float(np.percentile(s, 95))
    saturation_std = float(np.std(s))
    mean_brightness = float(np.mean(v))
    
    # Specific biological butterfly chromas (iridescent emerald, electric blue, vivid warm pigments)
    blue_chroma = float(np.mean((h >= 180) & (h <= 260) & (s > 0.38) & (v > 0.22)))
    emerald_chroma = float(np.mean((h >= 135) & (h <= 175) & (s > 0.38) & (v > 0.20)))
    warm_chroma = float(np.mean(((h < 55) | (h > 330)) & (s > 0.38) & (v > 0.22)))
    
    # Wing vibrancy — separating true wing chroma from dull foliage background.
    # Exclude dull green background foliage (hue 65-130 with v < 0.50).
    general_vibrant = (s > 0.38) & (v > 0.20) & ~((h >= 65) & (h <= 130) & (v < 0.50))
    vibrant_pixels_ratio = float(np.mean(general_vibrant))
    
    # Dull foliage background ratio (moderate green, NOT bright vivid green)
    foliage_ratio = float(np.mean((s > 0.20) & (v > 0.15) & (v < 0.50) & (h >= 65) & (h <= 130)))
    
    # 2. Textural Edge Density & Camouflage Mottling
    gray_img = image.convert("L").resize((224, 224), RESAMPLE_BILINEAR)
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32) / 255.0
    edge_density = float(np.mean(edge_arr > 0.15))
    edge_variance = float(np.var(edge_arr))
    
    # 3. Spatial Symmetry & Resting Aspect Profile
    raw_aspect = float(image.width / max(1, image.height))
    
    threshold = np.mean(edge_arr) * 1.2
    y_coords, x_coords = np.where(edge_arr > threshold)
    if len(x_coords) > 20:
        spread_x = np.percentile(x_coords, 90) - np.percentile(x_coords, 10)
        spread_y = np.percentile(y_coords, 90) - np.percentile(y_coords, 10)
        spread_aspect = float(spread_x / (spread_y + 1e-5))
    else:
        spread_aspect = 1.0

    # Distinguish genuine landscape photos and wide spread-wing postures
    if raw_aspect > 1.25:
        aspect_ratio = float(raw_aspect)
    elif spread_aspect > 1.35:
        aspect_ratio = float(spread_aspect)
    else:
        aspect_ratio = 1.0

    # 4. Color entropy (mottled moth wings have high chromatic dispersion in lower saturation)
    hist, _ = np.histogram(hsv[..., 0], bins=18, range=(0, 360), density=True)
    hist = hist[hist > 0]
    hue_entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

    # Qualitative biological synthesis
    is_vibrant = (vibrant_pixels_ratio > 0.12) or (blue_chroma > 0.05) or (emerald_chroma > 0.08)
    is_camouflaged = (edge_density > 0.10) and (mean_saturation < 0.40 or foliage_ratio > 0.15)
    posture_tendency = "Spread-wing (Flat/Tent)" if aspect_ratio >= 1.30 else "Upright / Compact"

    return {
        "mean_saturation": round(mean_saturation, 4),
        "vibrant_ratio": round(vibrant_pixels_ratio, 4),
        "blue_chroma": round(blue_chroma, 4),
        "emerald_chroma": round(emerald_chroma, 4),
        "warm_chroma": round(warm_chroma, 4),
        "foliage_ratio": round(foliage_ratio, 4),
        "mean_brightness": round(mean_brightness, 4),
        "edge_density": round(edge_density, 4),
        "edge_variance": round(edge_variance, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "hue_entropy": round(hue_entropy, 3),
        "is_vibrant": is_vibrant,
        "is_camouflaged": is_camouflaged,
        "posture_tendency": posture_tendency,
    }


def extract_feature_vector(image: Image.Image) -> np.ndarray:
    """
    Extracts a dense numerical feature vector for classical ML models (SVM, Random Forest, etc.)
    Comprises color histograms (HSV), spatial gradients, edge statistics, and downsampled intensity.
    """
    img_rgb = image.convert("RGB").resize((128, 128), RESAMPLE_BILINEAR)
    arr = np.array(img_rgb)
    hsv = rgb_to_hsv_np(arr)
    
    # Hue histogram (16 bins)
    h_hist, _ = np.histogram(hsv[..., 0], bins=16, range=(0, 360), density=True)
    # Saturation histogram (8 bins)
    s_hist, _ = np.histogram(hsv[..., 1], bins=8, range=(0, 1), density=True)
    # Value histogram (8 bins)
    v_hist, _ = np.histogram(hsv[..., 2], bins=8, range=(0, 1), density=True)
    
    # Statistical moments
    mean_rgb = np.mean(arr, axis=(0, 1)) / 255.0
    std_rgb = np.std(arr, axis=(0, 1)) / 255.0
    mean_hsv = np.mean(hsv, axis=(0, 1))
    mean_hsv[0] /= 360.0
    
    # Edge energy
    gray = image.convert("L").resize((128, 128), RESAMPLE_BILINEAR)
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_hist, _ = np.histogram(edges, bins=8, range=(0, 1), density=True)
    
    # Spatial block color averages (4x4 grid = 16 blocks x 3 channels = 48 features)
    blocks = []
    for r in range(4):
        for c in range(4):
            block = arr[r*32:(r+1)*32, c*32:(c+1)*32]
            blocks.extend(np.mean(block, axis=(0, 1)) / 255.0)
            
    # Morphological diagnostic scalars
    morph = extract_interpretable_features(image)
    morph_scalars = np.array([
        morph["aspect_ratio"],
        morph["edge_density"],
        morph["edge_variance"],
        morph["hue_entropy"],
        morph["vibrant_ratio"],
        morph["foliage_ratio"],
        morph["mean_saturation"]
    ], dtype=np.float32)

    vector = np.concatenate([
        h_hist, s_hist, v_hist,
        mean_rgb, std_rgb, mean_hsv,
        edge_hist, blocks,
        morph_scalars
    ])
    return vector.astype(np.float32)

