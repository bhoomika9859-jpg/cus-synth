# Compare CUS vs FID vs KID
# CUS-Synth: Clinical Utility Score for Synthetic Medical Image Evaluation
# Author: bhoomika9859-jpg
# Version 2.0 — Updated with verified CUS-O and CUS-H split

import numpy as np
import pandas as pd
import json
import os
from src.cus_metric import compute_cus, RELIABLE_CONDITIONS
from src.data_loader import load_labels, load_images
from src.fid_kid_metric import compute_fid, compute_kid, extract_features

print("=" * 60)
print("CUS-Synth: Metric Comparison")
print("CUS vs FID vs KID on NIH Chest X-ray Images")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────
print("\n📂 Loading data...")
labels = load_labels()
images = load_images(max_images=500)

# ── Split into Real and Synthetic ─────────────────────────────────
half = len(images) // 2
real_images = images[:half]
synthetic_images = images[half:]

print(f"\n✅ Real images:      {len(real_images)}")
print(f"✅ Synthetic images: {len(synthetic_images)}")

# ── Compute FID ───────────────────────────────────────────────────
print("\n⚔️  Computing FID Score...")
real_features = extract_features(real_images)
synthetic_features = extract_features(synthetic_images)
fid_score = compute_fid(real_features, synthetic_features)
print(f"   FID Score: {fid_score}")
print(f"   ⚠️  FID gives ONE global score — no per-image signal!")

# ── Compute KID ───────────────────────────────────────────────────
print("\n⚔️  Computing KID Score...")
kid_score = compute_kid(real_features, synthetic_features)
print(f"   KID Score: {kid_score}")
print(f"   ⚠️  KID gives ONE global score — no per-image signal!")

# ── Compute CUS per image ─────────────────────────────────────────
print("\n🧪 Computing CUS-O and CUS-H per image...")
results = []

for image_name, image_array in synthetic_images:
    true_labels = labels.get(image_name, ["No Finding"])
    # Baseline: model predicts No Finding for all synthetic
    predicted_labels = ["No Finding"]

    cus, cus_o, cus_h, missed, hallucinated = compute_cus(
        predicted_labels, true_labels
    )

    results.append({
        "image": image_name,
        "true_labels": "|".join(true_labels),
        "predicted_labels": "|".join(predicted_labels),
        "cus_score": cus,
        "cus_omission": cus_o,
        "cus_hallucination": cus_h,
        "missed_conditions": "|".join(missed) if missed else "none",
        "fid_score": fid_score,
        "kid_score": kid_score,
    })

df = pd.DataFrame(results)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)
print(f"\n  FID Score:              {fid_score}")
print(f"  ⚠️  No per-image signal!")
print(f"\n  KID Score:              {kid_score}")
print(f"  ⚠️  No per-image signal!")
print(f"\n  Avg CUS Score:          {df['cus_score'].mean():.4f}")
print(f"  Avg CUS-Omission:       {df['cus_omission'].mean():.4f}")
print(f"  Avg CUS-Hallucination:  {df['cus_hallucination'].mean():.4f}")

dangerous = df[df['cus_score'] > 0.0]
print(f"\n🔬 KEY FINDING:")
print(f"  FID/KID per-image findings: 0")
print(f"  CUS images with omissions:  {len(dangerous)}")
print(f"  → CUS catches danger FID/KID completely miss!")

# ── Save ──────────────────────────────────────────────────────────
df.to_csv("results/comparison.csv", index=False)
print(f"\n✅ Results saved to results/comparison.csv")
print("=" * 60)