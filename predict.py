"""
Command-Line Interface for Butterfly vs Moth Classification.
Usage:
    python predict.py --image path/to/specimen.jpg
    python predict.py --sample monarch_butterfly.jpg
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import SAMPLES_DIR
from src.inference import get_classifier

def main():
    parser = argparse.ArgumentParser(description="Classify an image as Butterfly or Moth.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", type=str, help="Path to input image file.")
    group.add_argument("--sample", "-s", type=str, help="Name of sample image in static/samples/")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    args = parser.parse_args()

    if args.sample:
        target_path = SAMPLES_DIR / args.sample
        if not target_path.exists():
            print(f"Error: Sample '{args.sample}' not found in {SAMPLES_DIR}")
            print(f"Available samples: {[f.name for f in SAMPLES_DIR.glob('*.jpg')]}")
            sys.exit(1)
    else:
        target_path = Path(args.image)
        if not target_path.exists():
            print(f"Error: Image path '{target_path}' does not exist.")
            sys.exit(1)

    classifier = get_classifier()
    result = classifier.predict(target_path)

    if args.json:
        # Exclude base64 strings from stdout json for cleanliness
        clean_result = {k: v for k, v in result.items() if k not in ["heatmap_overlay", "thumbnail"]}
        print(json.dumps(clean_result, indent=2))
        return

    # Beautiful console formatting
    pred_class = result["predicted_class"].upper()
    confidence = result["confidence"]
    probs = result["probabilities"]
    metrics = result["morphological_metrics"]
    taxonomy = result["taxonomy_info"]

    banner = "=" * 60
    print(banner)
    print(f"  LEPIDOPTERA CLASSIFIER - INFERENCE RESULT")
    print(banner)
    print(f"  File: {target_path.name}")
    print(f"  Classification : {pred_class} (Confidence: {confidence:.2f}%)")
    print(f"  Class Probs    : Butterfly: {probs['butterfly']:.1f}% | Moth: {probs['moth']:.1f}%")
    print("-" * 60)
    print("  Biological & Morphological Diagnostic Signals:")
    print(f"   * Color Saturation   : {metrics['mean_saturation'] * 100:.1f}% (Vibrant Chromatic Ratio: {metrics['vibrant_ratio'] * 100:.1f}%)")
    print(f"   * Camouflage Density : {metrics['edge_density'] * 100:.1f}%")
    print(f"   * Aspect Ratio       : {metrics['aspect_ratio']:.2f} ({metrics['posture_tendency']})")
    print(f"   * Color Entropy      : {metrics['hue_entropy']:.2f}")
    print("-" * 60)
    print(f"  Taxonomic Reference ({taxonomy['title']}):")
    print(f"   * Antennae         : {taxonomy['antennae']}")
    print(f"   * Wings at Rest    : {taxonomy['wings_at_rest']}")
    print(f"   * Body Morphology  : {taxonomy['body_shape']}")
    print(f"   * Activity Pattern : {taxonomy['activity']}")
    print("-" * 60)
    print("  Key Reasoning:")
    for point in result["diagnostic_points"]:
        print(f"   - {point}")
    print(banner)

if __name__ == "__main__":
    main()
