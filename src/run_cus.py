# Run CUS on all 500 chest X-ray images
# Author: bhoomika9859-jpg

import os
import pandas as pd
from src.cus_metric import compute_cus, ACR_WEIGHTS
from src.data_loader import load_labels, load_images

# ── Load Data ─────────────────────────────────────────────────────
print("=" * 50)
print("CUS-Synth: Clinical Utility Score Evaluation")
print("=" * 50)

print("\n📂 Loading labels...")
labels = load_labels()

print("\n🖼️  Loading images...")
images = load_images(max_images=500)

# ── Run CUS ───────────────────────────────────────────────────────
print("\n🧪 Computing CUS scores...")

results = []

for image_name, image_array in images:
    # Get true labels for this image
    true_labels = labels.get(image_name, ["No Finding"])

    # For now we simulate a predicted label
    # (In Week 4 we'll use a real AI model!)
    predicted_labels = ["No Finding"]

    # Compute CUS
    score = compute_cus(predicted_labels, true_labels)

    results.append({
        "image": image_name,
        "true_labels": "|".join(true_labels),
        "predicted_labels": "|".join(predicted_labels),
        "cus_score": score
    })

# ── Save Results ──────────────────────────────────────────────────
print("\n💾 Saving results...")
df = pd.DataFrame(results)
df.to_csv("results/cus_scores.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────
print("\n📊 Summary:")
print(f"  Total images evaluated: {len(df)}")
print(f"  Average CUS score: {df['cus_score'].mean():.4f}")
print(f"  Max CUS score: {df['cus_score'].max():.4f}")
print(f"  Min CUS score: {df['cus_score'].min():.4f}")
print(f"\n✅ Results saved to results/cus_scores.csv")
print("=" * 50)