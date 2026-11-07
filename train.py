"""
CLI training runner for Butterfly vs Moth Classification.
Usage:
    python train.py --generate-data --epochs 12
"""

import argparse
from src.dataset_generator import generate_curated_samples, generate_synthetic_dataset
from src.trainer import run_full_training

def main():
    parser = argparse.ArgumentParser(description="Train Butterfly vs Moth Classification Models.")
    parser.add_argument("--generate-data", action="store_true", help="Synthesize dataset before training.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs for CNN.")
    args = parser.parse_args()

    print("[Step 1] Verifying / Generating Curated Lepidoptera Samples...")
    generate_curated_samples()

    if args.generate_data:
        print("[Step 2] Generating synthetic training and validation datasets...")
        generate_synthetic_dataset(num_train_per_class=80, num_val_per_class=25)

    print("\n[Step 3] Launching Full Training Pipeline...")
    summary = run_full_training(epochs=args.epochs)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
