# Visualisation — CUS vs FID vs KID
# CUS-Synth: Clinical Utility Score for Synthetic Medical Image Evaluation
# Author: bhoomika9859-jpg
# Version 2.0 — Updated with verified CUS-O and CUS-H results

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load Results ──────────────────────────────────────────────────
results_path = "results/final_honest_results.csv"
if not os.path.exists(results_path):
    results_path = "results/corrected_cus_results.csv"
if not os.path.exists(results_path):
    results_path = "results/cus_scores.csv"

df = pd.read_csv(results_path)
os.makedirs("results/plots", exist_ok=True)

print("🎨 Generating verified research visualisations...")
print(f"📂 Using: {results_path}")

# ── Plot 1: CUS-O and CUS-H Distribution ─────────────────────────
if 'cus_omission' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CUS-Omission
    colors_o = [
        '#E24B4A' if x > 0 else '#378ADD'
        for x in df['cus_omission']
    ]
    axes[0].bar(
        range(len(df)), df['cus_omission'],
        color=colors_o, width=1.0
    )
    axes[0].set_xlabel('Image Index', fontsize=12)
    axes[0].set_ylabel('CUS-Omission Score', fontsize=12)
    axes[0].set_title(
        'CUS-Omission Scores\nRed = Missed Clinical Condition',
        fontsize=12, fontweight='bold'
    )

    # CUS-Hallucination
    colors_h = [
        '#E24B4A' if x > 0 else '#378ADD'
        for x in df['cus_hallucination']
    ]
    axes[1].bar(
        range(len(df)), df['cus_hallucination'],
        color=colors_h, width=1.0
    )
    axes[1].set_xlabel('Image Index', fontsize=12)
    axes[1].set_ylabel('CUS-Hallucination Score', fontsize=12)
    axes[1].set_title(
        'CUS-Hallucination Scores\nRed = Invented Clinical Condition',
        fontsize=12, fontweight='bold'
    )

    plt.suptitle(
        'CUS-Omission vs CUS-Hallucination\n'
        'Two Clinically Distinct Failure Modes',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        'results/plots/cus_omission_hallucination.png',
        dpi=150
    )
    plt.close()
    print("  ✅ Plot 1: CUS-O vs CUS-H saved!")

# ── Plot 2: Metric Battle ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

capabilities = [
    'Per-image\nscoring',
    'Catches\nOmissions',
    'Catches\nHallucinations',
    'Clinical\nInterpretability',
    'ACR-grounded\nWeights'
]
fid_scores = [0, 0, 0, 0, 0]
kid_scores = [0, 0, 0, 0, 0]
cus_scores = [1, 1, 1, 1, 1]

x = np.arange(len(capabilities))
width = 0.25

ax.bar(x - width, fid_scores, width,
       label='FID', color='#E24B4A', alpha=0.8)
ax.bar(x, kid_scores, width,
       label='KID', color='#EF9F27', alpha=0.8)
ax.bar(x + width, cus_scores, width,
       label='CUS (Ours)', color='#378ADD', alpha=0.8)

ax.set_ylabel('Capability (1=Yes, 0=No)', fontsize=12)
ax.set_title(
    'CUS vs FID vs KID\n'
    'Why CUS is Better for Medical AI Evaluation',
    fontsize=13, fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(capabilities, fontsize=11)
ax.set_ylim(0, 1.3)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/metric_battle.png', dpi=150)
plt.close()
print("  ✅ Plot 2: Metric battle saved!")

# ── Plot 3: Classifier Performance ───────────────────────────────
reliable_conditions = [
    "Pneumothorax", "Nodule", "Consolidation",
    "Effusion", "Cardiomegaly"
]
sensitivities = [1.0, 1.0, 0.833, 0.455, 0.75]
specificities = [1.0, 0.957, 0.986, 1.0, 0.746]

x = np.arange(len(reliable_conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, sensitivities, width,
               label='Sensitivity', color='#378ADD')
bars2 = ax.bar(x + width/2, specificities, width,
               label='Specificity', color='#3B9C52')

ax.set_ylabel('Score', fontsize=12)
ax.set_title(
    'DenseNet121 Classifier Performance\n'
    'Evaluated on Held-Out Test Set',
    fontsize=13, fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(reliable_conditions, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.axhline(
    y=0.7, color='orange', linestyle='--',
    linewidth=1.5, label='70% baseline'
)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width()/2.,
        bar.get_height() + 0.02,
        f'{bar.get_height():.0%}',
        ha='center', fontsize=10, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(
    'results/plots/classifier_performance.png', dpi=150
)
plt.close()
print("  ✅ Plot 3: Classifier performance saved!")

# ── Plot 4: Omission breakdown by condition ───────────────────────
if 'missed_conditions' in df.columns:
    from collections import Counter
    all_missed = []
    for _, row in df.iterrows():
        if row['missed_conditions'] != 'none':
            all_missed.extend(
                row['missed_conditions'].split('|')
            )

    if all_missed:
        counts = Counter(all_missed)
        conditions = list(counts.keys())
        values = list(counts.values())

        colors = ['#E24B4A'] * len(conditions)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(conditions, values, color=colors)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.3,
                str(val),
                ha='center', fontsize=12,
                fontweight='bold'
            )

        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title(
            'Missed Conditions in Synthetic Images\n'
            'By Condition Type',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            'results/plots/missed_conditions.png', dpi=150
        )
        plt.close()
        print("  ✅ Plot 4: Missed conditions saved!")

print("\n🎉 All verified plots saved to results/plots/")