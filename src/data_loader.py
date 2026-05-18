# Data Loader — NIH ChestX-ray14
# Loads images and labels for CUS evaluation
# Author: bhoomika9859-jpg

import os
import pandas as pd
import numpy as np
import cv2

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "sample", "sample","images")
LABELS_CSV = os.path.join("data", "sample", "sample_labels.csv")

# ── Load Labels ───────────────────────────────────────────────────
def load_labels(csv_path=LABELS_CSV):
    """
    Load image labels from CSV file.
    Returns a dictionary: {image_name: [list of conditions]}
    """
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        image_name = row["Image Index"]
        conditions = row["Finding Labels"].split("|")
        labels[image_name] = conditions
    return labels


# ── Load Images ───────────────────────────────────────────────────
def load_images(data_dir=DATA_DIR, max_images=500):
    """
    Load chest X-ray images from folder.
    Returns list of (image_name, image_array) tuples.
    """
    images = []
    all_files = os.listdir(data_dir)
    png_files = [f for f in all_files if f.endswith(".png")]

    print(f"Found {len(png_files)} images in dataset!")
    print(f"Loading first {max_images} images...")

    for i, filename in enumerate(png_files[:max_images]):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))

    print(f"Successfully loaded {len(images)} images!")
    return images


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test loading labels
    print("Loading labels...")
    labels = load_labels()
    print(f"Total labelled images: {len(labels)}")

    # Show first 3 labels
    print("\nSample labels:")
    for i, (name, conditions) in enumerate(labels.items()):
        if i >= 3:
            break
        print(f"  {name}: {conditions}")

    # Test loading images
    print("\nLoading images...")
    images = load_images(max_images=500)

    # Show first image shape
    if images:
        name, img = images[0]
        print(f"\nFirst image: {name}")
        print(f"Image size: {img.shape}")