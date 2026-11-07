"""
Dataset generator and sample synthesizer for Butterfly vs Moth Classification.
Generates:
1. Procedural/realistic Lepidopteran image dataset with distinct biological phenotypes:
   - Butterfly: vibrant iridescent hues, crisp wing borders, slender bodies, clubbed antennae.
   - Moth: cryptic mottled camouflage, bark patterns, stout furry bodies, feathery/broad antennae.
2. Curated showcase samples for the interactive UI test gallery.
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from src.config import DATA_DIR, SAMPLES_DIR, IMG_SIZE, CLASSES

def draw_butterfly_specimen(size=(224, 224), variety="monarch", seed=None) -> Image.Image:
    """
    Renders a realistic biological representation of a butterfly specimen.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    w, h = size
    # Background: natural foliage, leaf or studio neutral
    bg_types = ["foliage", "neutral", "bark"]
    bg_type = random.choice(bg_types)
    
    if bg_type == "foliage":
        base_color = (random.randint(25, 45), random.randint(60, 95), random.randint(30, 50))
    elif bg_type == "neutral":
        base_color = (random.randint(220, 245), random.randint(220, 245), random.randint(215, 240))
    else:
        base_color = (random.randint(40, 60), random.randint(35, 55), random.randint(30, 45))
        
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add subtle background texture / bokeh
    for _ in range(15):
        bx = random.randint(0, w)
        by = random.randint(0, h)
        br = random.randint(15, 50)
        alpha_color = tuple(min(255, c + random.randint(-15, 25)) for c in base_color)
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=alpha_color)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.0)))
    draw = ImageDraw.Draw(img)
    
    cx, cy = w // 2, h // 2
    scale = random.uniform(0.75, 1.05)
    
    # Varieties: monarch, morpho, swallowtail, emerald
    varieties = ["monarch", "morpho", "swallowtail", "emerald"]
    chosen_variety = variety if variety in varieties else random.choice(varieties)
    
    if chosen_variety == "monarch":
        wing_color = (random.randint(225, 255), random.randint(110, 140), random.randint(10, 30)) # Orange
        border_color = (20, 20, 22)
        spot_color = (255, 255, 255)
    elif chosen_variety == "morpho":
        wing_color = (random.randint(10, 40), random.randint(130, 190), random.randint(220, 255)) # Electric Blue
        border_color = (15, 20, 35)
        spot_color = (200, 230, 255)
    elif chosen_variety == "swallowtail":
        wing_color = (random.randint(235, 255), random.randint(210, 240), random.randint(40, 70)) # Yellow
        border_color = (10, 15, 20)
        spot_color = (40, 90, 200)
    else: # emerald
        wing_color = (random.randint(20, 50), random.randint(190, 235), random.randint(120, 160)) # Emerald green
        border_color = (10, 25, 20)
        spot_color = (255, 255, 200)

    # 1. Wings (Forewings and Hindwings with characteristic Lepidopteran curvature)
    # Forewings
    fw_w = int(60 * scale)
    fw_h = int(75 * scale)
    
    # Left forewing
    draw.polygon([
        (cx - 4, cy - 10),
        (cx - int(fw_w * 1.3), cy - int(fw_h * 0.9)),
        (cx - int(fw_w * 1.4), cy - int(fw_h * 0.3)),
        (cx - int(fw_w * 0.7), cy + int(fw_h * 0.2)),
        (cx - 4, cy + 15)
    ], fill=wing_color, outline=border_color)
    
    # Right forewing
    draw.polygon([
        (cx + 4, cy - 10),
        (cx + int(fw_w * 1.3), cy - int(fw_h * 0.9)),
        (cx + int(fw_w * 1.4), cy - int(fw_h * 0.3)),
        (cx + int(fw_w * 0.7), cy + int(fw_h * 0.2)),
        (cx + 4, cy + 15)
    ], fill=wing_color, outline=border_color)

    # Hindwings
    hw_w = int(45 * scale)
    hw_h = int(55 * scale)
    draw.polygon([
        (cx - 3, cy + 10),
        (cx - int(hw_w * 1.2), cy + int(hw_h * 0.5)),
        (cx - int(hw_w * 0.9), cy + int(hw_h * 1.1)),
        (cx - int(hw_w * 0.3), cy + int(hw_h * 1.2)),
        (cx - 2, cy + 30)
    ], fill=wing_color, outline=border_color)
    
    draw.polygon([
        (cx + 3, cy + 10),
        (cx + int(hw_w * 1.2), cy + int(hw_h * 0.5)),
        (cx + int(hw_w * 0.9), cy + int(hw_h * 1.1)),
        (cx + int(hw_w * 0.3), cy + int(hw_h * 1.2)),
        (cx + 2, cy + 30)
    ], fill=wing_color, outline=border_color)

    # Wing veins (Distinct, crisp black venation)
    for sign in [-1, 1]:
        draw.line([(cx + sign*4, cy - 10), (cx + sign*int(fw_w*1.1), cy - int(fw_h*0.7))], fill=border_color, width=2)
        draw.line([(cx + sign*4, cy - 5), (cx + sign*int(fw_w*1.2), cy - int(fw_h*0.2))], fill=border_color, width=2)
        draw.line([(cx + sign*4, cy + 5), (cx + sign*int(fw_w*0.7), cy + int(fw_h*0.1))], fill=border_color, width=1)
        draw.line([(cx + sign*3, cy + 15), (cx + sign*int(hw_w*0.8), cy + int(hw_h*0.9))], fill=border_color, width=2)

    # Crisp margin spots (Classic butterfly border dots)
    for sign in [-1, 1]:
        for i in range(5):
            sx = cx + sign * (int(fw_w * 1.2) - i * 6)
            sy = cy - int(fw_h * 0.8) + i * 14
            draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=spot_color)

    # 2. Body (Slender, streamlined, smooth butterfly thorax/abdomen)
    body_color = (25, 25, 28)
    draw.ellipse([cx - 4, cy - 30, cx + 4, cy - 15], fill=body_color) # Head
    draw.ellipse([cx - 5, cy - 15, cx + 5, cy + 10], fill=body_color)  # Thorax
    draw.ellipse([cx - 4, cy + 8, cx + 4, cy + 42], fill=body_color)   # Slender Abdomen

    # 3. Antennae (DIAGNOSTIC: Long, slender, with distinct CLUBBED tips)
    draw.line([(cx - 2, cy - 28), (cx - 24, cy - 55)], fill=body_color, width=2)
    draw.ellipse([cx - 27, cy - 60, cx - 21, cy - 54], fill=body_color) # Left Club tip
    
    draw.line([(cx + 2, cy - 28), (cx + 24, cy - 55)], fill=body_color, width=2)
    draw.ellipse([cx + 21, cy - 60, cx + 27, cy - 54], fill=body_color) # Right Club tip

    return img


def draw_moth_specimen(size=(224, 224), variety="luna", seed=None) -> Image.Image:
    """
    Renders a realistic biological representation of a moth specimen.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    w, h = size
    # Moths are often on tree bark, lichen, foliage leaves, or wooden/neutral backgrounds
    bg_types = ["foliage", "bark", "lichen", "night", "neutral"]
    bg_type = random.choice(bg_types)
    
    if bg_type == "foliage":
        base_color = (random.randint(25, 45), random.randint(60, 95), random.randint(30, 50))
    elif bg_type == "bark":
        base_color = (random.randint(65, 85), random.randint(50, 68), random.randint(40, 55))
    elif bg_type == "lichen":
        base_color = (random.randint(110, 130), random.randint(120, 140), random.randint(110, 125))
    elif bg_type == "neutral":
        base_color = (random.randint(200, 230), random.randint(200, 230), random.randint(195, 220))
    else:
        base_color = (random.randint(20, 35), random.randint(25, 38), random.randint(30, 45))
        
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add mottled / foliage bokeh texture
    if bg_type == "foliage":
        for _ in range(15):
            bx = random.randint(0, w)
            by = random.randint(0, h)
            br = random.randint(15, 50)
            alpha_color = tuple(min(255, c + random.randint(-15, 25)) for c in base_color)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=alpha_color)
    else:
        for _ in range(40):
            bx = random.randint(0, w)
            by = random.randint(0, h)
            bw = random.randint(4, 25)
            bh = random.randint(2, 6)
            c_shift = random.randint(-25, 25)
            bark_color = tuple(max(0, min(255, c + c_shift)) for c in base_color)
            draw.rectangle([bx, by, bx + bw, by + bh], fill=bark_color)
        
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    scale = random.uniform(0.8, 1.1)

    # Varieties: luna, peppered, hawk, atlas, cecropia
    varieties = ["luna", "peppered", "hawk", "atlas", "cecropia"]
    chosen_variety = variety if variety in varieties else random.choice(varieties)

    if chosen_variety == "luna":
        wing_color = (random.randint(165, 195), random.randint(215, 235), random.randint(165, 195)) # Pale Mint green
        edge_color = (random.randint(120, 150), random.randint(60, 80), random.randint(40, 60)) # Purplish leading edge
        has_tails = True
    elif chosen_variety == "peppered":
        # Mottled grey/black bark camouflage
        g = random.randint(120, 155)
        wing_color = (g, g - random.randint(2, 8), g - random.randint(5, 12))
        edge_color = (40, 40, 45)
        has_tails = False
    elif chosen_variety == "hawk":
        # Streamlined earthy ochre/olive
        wing_color = (random.randint(115, 145), random.randint(95, 125), random.randint(65, 85))
        edge_color = (60, 45, 30)
        has_tails = False
    elif chosen_variety == "cecropia":
        # Cecropia giant silk moth: rich rust/orange-brown with charcoal borders and red-black collar
        wing_color = (random.randint(130, 160), random.randint(60, 85), random.randint(40, 60))
        edge_color = (35, 25, 25)
        has_tails = False
    else: # atlas
        # Giant silk moth rusty mahogany / orange-brown
        wing_color = (random.randint(150, 185), random.randint(75, 105), random.randint(45, 65))
        edge_color = (80, 40, 25)
        has_tails = False

    # 1. Wings (DIAGNOSTIC: Folded flat / triangular / tent-like resting posture)
    mw_w = int(75 * scale)
    mw_h = int(60 * scale)

    # Left moth wing (broad triangular spread)
    draw.polygon([
        (cx - 3, cy - 8),
        (cx - int(mw_w * 1.25), cy - int(mw_h * 0.1)),
        (cx - int(mw_w * 1.05), cy + int(mw_h * 0.65)),
        (cx - int(mw_w * 0.4), cy + int(mw_h * 0.8)),
        (cx - 3, cy + 20)
    ], fill=wing_color, outline=edge_color)

    # Right moth wing
    draw.polygon([
        (cx + 3, cy - 8),
        (cx + int(mw_w * 1.25), cy - int(mw_h * 0.1)),
        (cx + int(mw_w * 1.05), cy + int(mw_h * 0.65)),
        (cx + int(mw_w * 0.4), cy + int(mw_h * 0.8)),
        (cx + 3, cy + 20)
    ], fill=wing_color, outline=edge_color)

    # Luna moth elongated tails if applicable
    if has_tails:
        draw.polygon([(cx - int(mw_w*0.5), cy + int(mw_h*0.75)), (cx - int(mw_w*0.65), cy + int(mw_h*1.45)), (cx - int(mw_w*0.4), cy + int(mw_h*0.85))], fill=wing_color)
        draw.polygon([(cx + int(mw_w*0.5), cy + int(mw_h*0.75)), (cx + int(mw_w*0.65), cy + int(mw_h*1.45)), (cx + int(mw_w*0.4), cy + int(mw_h*0.85))], fill=wing_color)

    # Moth wing mottled texture & cryptic wavy fascia (zigzag lines)
    for sign in [-1, 1]:
        for offset in [-10, 10, 25]:
            x_a = cx + sign * int(mw_w * 0.8) - 15
            x_b = cx + sign * int(mw_w * 0.2) + 15
            y_a = cy + offset - 10
            y_b = cy + offset + 15
            box = [min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)]
            draw.arc(box, 0, 180, fill=edge_color)
            
        # Eyespot or crescent lunule on moth wing
        ex = cx + sign * int(mw_w * 0.55)
        ey = cy + int(mw_h * 0.2)
        if chosen_variety in ["cecropia", "atlas"]:
            # Characteristic white crescent lunule with red/black border
            draw.arc([ex - 10, ey - 5, ex + 10, ey + 7], 0, 180, fill=(245, 245, 240), width=3)
            draw.arc([ex - 12, ey - 7, ex + 12, ey + 9], 0, 180, fill=(180, 40, 30), width=2)
        else:
            draw.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], outline=edge_color)
            draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(230, 210, 180))

    # 2. Body (DIAGNOSTIC: Stout, thick, heavily cloaked in fuzzy setae / woolly fur)
    if chosen_variety == "cecropia":
        body_color = (random.randint(160, 195), random.randint(35, 60), random.randint(30, 50)) # Crimson rust body
    else:
        body_color = (random.randint(60, 85), random.randint(55, 75), random.randint(50, 65))

    # Furry textured abdomen
    draw.ellipse([cx - 8, cy - 20, cx + 8, cy - 10], fill=body_color) # Broad head
    draw.ellipse([cx - 11, cy - 12, cx + 11, cy + 15], fill=body_color) # Stout thorax
    draw.ellipse([cx - 9, cy + 10, cx + 9, cy + 42], fill=body_color)  # Heavy fuzzy abdomen
    
    # Setae / fur fringes on thorax
    for y_fur in range(cy - 10, cy + 18, 4):
        draw.line([(cx - 13, y_fur), (cx - 9, y_fur)], fill=(body_color[0]+25, body_color[1]+20, body_color[2]+15), width=2)
        draw.line([(cx + 9, y_fur), (cx + 13, y_fur)], fill=(body_color[0]+25, body_color[1]+20, body_color[2]+15), width=2)

    # 3. Antennae (DIAGNOSTIC: Broad, FEATHERY / BIPECTINATE combs, NO clubs)
    for sign in [-1, 1]:
        # Central stem
        stem_start = (cx + sign * 5, cy - 18)
        stem_end = (cx + sign * 35, cy - 45)
        draw.line([stem_start, stem_end], fill=body_color, width=2)
        
        # Bipectinate comb side-branches (feathery plumage)
        steps = 10
        for s in range(steps):
            t = s / steps
            px = stem_start[0] + t * (stem_end[0] - stem_start[0])
            py = stem_start[1] + t * (stem_end[1] - stem_start[1])
            feather_len = int(8 * (1 - 0.3 * t))
            draw.line([(px, py), (px + sign * feather_len, py - 4)], fill=body_color, width=1)
            draw.line([(px, py), (px - sign * (feather_len // 2), py + 4)], fill=body_color, width=1)

    return img


def generate_curated_samples():
    """
    Creates a set of distinct, beautiful biological reference samples
    saved to static/samples/ for instant testing in the UI.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    samples_manifest = [
        # Butterflies
        ("monarch_butterfly.jpg", "butterfly", "monarch", 101, "Monarch Butterfly (Danaus plexippus) - Vibrant orange, crisp black veins"),
        ("blue_morpho_butterfly.jpg", "butterfly", "morpho", 202, "Blue Morpho (Morpho peleides) - Iridescent structural blue"),
        ("swallowtail_butterfly.jpg", "butterfly", "swallowtail", 303, "Swallowtail (Papilionidae) - Crisp yellow with dark margins"),
        ("emerald_swallowtail.jpg", "butterfly", "emerald", 404, "Emerald Swallowtail - Brilliant vivid green scales"),
        
        # Moths
        ("luna_moth.jpg", "moth", "luna", 505, "Luna Moth (Actias luna) - Broad pale-mint wings & feathery antennae"),
        ("peppered_moth.jpg", "moth", "peppered", 606, "Peppered Moth (Biston betularia) - Mottled camouflage pattern"),
        ("hawk_moth.jpg", "moth", "hawk", 707, "Hawk Moth (Sphingidae) - Streamlined stout body, earthy tones"),
        ("atlas_moth.jpg", "moth", "atlas", 808, "Atlas Silk Moth (Attacus atlas) - Rich mahogany patterning & heavy fur"),
    ]
    
    created = []
    for filename, category, variety, seed, description in samples_manifest:
        out_path = SAMPLES_DIR / filename
        if category == "butterfly":
            img = draw_butterfly_specimen(variety=variety, seed=seed)
        else:
            img = draw_moth_specimen(variety=variety, seed=seed)
            
        img.save(out_path, quality=95)
        created.append({
            "filename": filename,
            "category": category,
            "variety": variety,
            "description": description,
            "url": f"/static/samples/{filename}"
        })
        
    print(f"Generated {len(created)} curated Lepidoptera showcase samples in {SAMPLES_DIR}")
    return created


def generate_synthetic_dataset(num_train_per_class: int = 100, num_val_per_class: int = 25):
    """
    Creates complete train and validation splits for binary classification.
    """
    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class
    }
    
    for split_name, count in splits.items():
        for category in CLASSES:
            split_dir = DATA_DIR / split_name / category
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                seed = 1000 * (1 if category == "butterfly" else 2) + i
                if category == "butterfly":
                    img = draw_butterfly_specimen(seed=seed)
                else:
                    img = draw_moth_specimen(seed=seed)
                    
                filepath = split_dir / f"{category}_{i:04d}.jpg"
                img.save(filepath, quality=90)
                
    print(f"Dataset generated at {DATA_DIR}: {num_train_per_class*2} train images, {num_val_per_class*2} val images.")

if __name__ == "__main__":
    generate_curated_samples()
    generate_synthetic_dataset(num_train_per_class=60, num_val_per_class=20)
