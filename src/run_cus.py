# Run CUS on chest X-ray images
# CUS-Synth: Clinical Utility Score for Synthetic Medical Image Evaluation
# Author: bhoomika9859-jpg
# Version 2.0 — Updated with CUS-O and CUS-H split

import os
import pandas as pd
from src.cus_metric import compute_cus, RELIABLE_CONDITIONS
from src.data_loader import load_labels, load_images

print("=" * 55)
print("CUS-Synth: Clinical Utility Score Evaluation")
print("=" * 55)

print("\n📂 Loading labels...")
labels = load_labels()

print("\n🖼️  Loading images...")
images = load_images(max_images=500)

print("\n🧪 Computing CUS scores...")
results = []

for image_name, image_array in images:
    true_labels = labels.get(image_name, ["No Finding"])
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
        "hallucinated_conditions": "|".join(hallucinated) if hallucinated else "none",
    })

df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("📊 Summary:")
print("=" * 55)
print(f"  Total images evaluated: {len(df)}")
print(f"  Avg CUS Score:          {df['cus_score'].mean():.4f}")
print(f"  Avg CUS-Omission:       {df['cus_omission'].mean():.4f}")
print(f"  Avg CUS-Hallucination:  {df['cus_hallucination'].mean():.4f}")
print(f"  Max CUS Score:          {df['cus_score'].max():.4f}")
print(f"  Min CUS Score:          {df['cus_score'].min():.4f}")

images_with_omission = df[df['cus_omission'] > 0]
print(f"\n  Images with omissions:  {len(images_with_omission)}")
print(f"\n✅ Results saved to results/cus_scores.csv")
print("=" * 55)

df.to_csv("results/cus_scores.csv", index=False)