# Visualisation — CUS vs FID vs KID
# Author: bhoomika9859-jpg

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load Results ──────────────────────────────────────────────────
df = pd.read_csv("results/comparison.csv")

# Create plots folder
os.makedirs("results/plots", exist_ok=True)

print("🎨 Generating research visualisations...")

# ── Plot 1: CUS Score Distribution ────────────────────────────────
plt.figure(figsize=(10, 5))
colors = ['#E24B4A' if x > 0.2 else '#378ADD' 
          for x in df['cus_score']]
plt.bar(range(len(df)), df['cus_score'], color=colors, width=1.0)
plt.axhline(y=0.2, color='orange', linestyle='--', 
            linewidth=2, label='Danger threshold (0.2)')
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('CUS Score', fontsize=12)
plt.title('CUS Scores Across 250 Chest X-ray Images\n'
          'Red = Clinically Dangerous, Blue = Safe', fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/plots/cus_distribution.png', dpi=150)
plt.close()
print("  ✅ Plot 1: CUS distribution saved!")

# ── Plot 2: Metric Comparison Bar Chart ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# CUS
axes[0].bar(['CUS'], [df['cus_score'].mean()], 
            color='#378ADD', width=0.4)
axes[0].set_title('CUS Score\n(0-1 scale)', fontsize=12)
axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_ylim(0, 1)
axes[0].text(0, df['cus_score'].mean() + 0.02, 
             f"{df['cus_score'].mean():.4f}", 
             ha='center', fontsize=12, fontweight='bold')

# FID
axes[1].bar(['FID'], [df['fid_score'].mean()], 
            color='#E24B4A', width=0.4)
axes[1].set_title('FID Score\n(unbounded scale)', fontsize=12)
axes[1].set_ylabel('Score', fontsize=11)
axes[1].text(0, df['fid_score'].mean() * 1.02,
             f"{df['fid_score'].mean():.1f}",
             ha='center', fontsize=12, fontweight='bold')

# KID
axes[2].bar(['KID'], [df['kid_score'].mean()], 
            color='#EF9F27', width=0.4)
axes[2].set_title('KID Score\n(unbounded scale)', fontsize=12)
axes[2].set_ylabel('Score', fontsize=11)
axes[2].text(0, df['kid_score'].mean() * 1.02,
             f"{df['kid_score'].mean():.2e}",
             ha='center', fontsize=11, fontweight='bold')

plt.suptitle('CUS vs FID vs KID — Why CUS is Better for Medical Images',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/plots/metric_comparison.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 2: Metric comparison saved!")

# ── Plot 3: Dangerous Images Found ────────────────────────────────
dangerous = df[df['cus_score'] > 0.2]
safe = df[df['cus_score'] <= 0.2]

fig, ax = plt.subplots(figsize=(8, 8))
sizes = [len(dangerous), len(safe)]
colors = ['#E24B4A', '#378ADD']
labels = [f'Dangerous\n({len(dangerous)} images)', 
          f'Safe\n({len(safe)} images)']
explode = (0.05, 0)

ax.pie(sizes, explode=explode, labels=labels, 
       colors=colors, autopct='%1.1f%%',
       shadow=False, startangle=90,
       textprops={'fontsize': 13})
ax.set_title('Images FID/KID Miss vs CUS Catches\n'
             'CUS identifies clinically dangerous images!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/dangerous_vs_safe.png', dpi=150)
plt.close()
print("  ✅ Plot 3: Dangerous vs safe pie chart saved!")

# ── Plot 4: CUS by Condition ──────────────────────────────────────
condition_scores = {}
for _, row in df.iterrows():
    conditions = row['true_labels'].split('|')
    for cond in conditions:
        if cond not in condition_scores:
            condition_scores[cond] = []
        condition_scores[cond].append(row['cus_score'])

cond_means = {k: np.mean(v) 
              for k, v in condition_scores.items()}
cond_means = dict(sorted(cond_means.items(), 
                         key=lambda x: x[1], reverse=True))

plt.figure(figsize=(12, 6))
bars = plt.bar(cond_means.keys(), cond_means.values(),
               color=['#E24B4A' if v > 0.1 else '#378ADD' 
                      for v in cond_means.values()])
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Average CUS Score', fontsize=12)
plt.title('Average CUS Score by Medical Condition\n'
          'Higher = More Dangerous to Miss!', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('results/plots/cus_by_condition.png', dpi=150)
plt.close()
print("  ✅ Plot 4: CUS by condition saved!")

print("\n🎉 All plots saved to results/plots/")
print("Open the results/plots/ folder to see your graphs!")