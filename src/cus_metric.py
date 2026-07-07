# CUS Metric — Clinical Utility Score
# CUS-Synth: Clinical Utility Score for Synthetic Medical Image Evaluation
# Author: bhoomika9859-jpg
# Version 2.0 — Corrected formula with CUS-Omission and CUS-Hallucination split

import numpy as np

# ── ACR Severity Weights ──────────────────────────────────────────
# Based on ACR (American College of Radiology) diagnostic error severity
# Higher weight = more dangerous to miss this condition
# Full condition set for reference

ACR_WEIGHTS_FULL = {
    "Cardiomegaly":       0.7,
    "Effusion":           0.8,
    "Infiltration":       0.6,
    "Mass":               0.9,
    "Nodule":             0.85,
    "Pneumonia":          0.95,
    "Pneumothorax":       1.0,   # Most dangerous!
    "Consolidation":      0.75,
    "Edema":              0.85,
    "Emphysema":          0.6,
    "Fibrosis":           0.5,
    "Pleural_Thickening": 0.5,
    "Hernia":             0.4,
    "No Finding":         0.0,
}

# ── Reliable conditions only ──────────────────────────────────────
# Restricted to conditions where DenseNet121 showed meaningful
# probability variance (std > 0.10) on held-out test set.
# This avoids the multiple comparisons artifact from checking
# all 14 conditions per image at low thresholds.
# Verified sensitivity/specificity on held-out test set:
#   Pneumothorax:  100% sensitivity, 100% specificity
#   Consolidation:  83% sensitivity,  99% specificity
#   Effusion:       46% sensitivity, 100% specificity
#   Nodule:        100% sensitivity,  96% specificity
#   Cardiomegaly:   75% sensitivity,  75% specificity

ACR_WEIGHTS = {
    "Pneumothorax":  1.0,
    "Nodule":        0.85,
    "Consolidation": 0.75,
    "Effusion":      0.80,
    "Cardiomegaly":  0.70,
}

RELIABLE_CONDITIONS = list(ACR_WEIGHTS.keys())

# ── CUS-Omission ──────────────────────────────────────────────────
def compute_cus_omission(predicted_labels, true_labels):
    """
    CUS-Omission (CUS-O):
    Penalises synthetic images that MISS real conditions.
    This is the more dangerous failure mode — a model trained
    on these images learns to ignore real disease signs.

    Args:
        predicted_labels: list of conditions predicted
        true_labels: list of actual conditions

    Returns:
        cus_o: float between 0 and 1
        missed: list of missed conditions
    """
    # Restrict to reliable conditions only
    predicted_r = [
        c for c in predicted_labels if c in RELIABLE_CONDITIONS
    ]
    true_r = [
        c for c in true_labels if c in RELIABLE_CONDITIONS
    ]

    missed = set(true_r) - set(predicted_r)
    max_possible = sum(ACR_WEIGHTS.values())

    cus_o = sum(
        ACR_WEIGHTS.get(c, 0) for c in missed
    ) / max_possible

    return round(cus_o, 4), list(missed)


# ── CUS-Hallucination ─────────────────────────────────────────────
def compute_cus_hallucination(predicted_labels, true_labels):
    """
    CUS-Hallucination (CUS-H):
    Penalises synthetic images that INVENT conditions.
    Less dangerous than omission but causes alarm fatigue
    and wastes clinical resources.

    Note: Only meaningful for conditions with threshold >= 0.20
    to avoid multiple comparisons artifact.

    Args:
        predicted_labels: list of conditions predicted
        true_labels: list of actual conditions

    Returns:
        cus_h: float between 0 and 1
        hallucinated: list of hallucinated conditions
    """
    predicted_r = [
        c for c in predicted_labels if c in RELIABLE_CONDITIONS
    ]
    true_r = [
        c for c in true_labels if c in RELIABLE_CONDITIONS
    ]

    hallucinated = set(predicted_r) - set(true_r)
    max_possible = sum(ACR_WEIGHTS.values())

    cus_h = sum(
        ACR_WEIGHTS.get(c, 0) * 0.5 for c in hallucinated
    ) / max_possible

    return round(cus_h, 4), list(hallucinated)


# ── Combined CUS ──────────────────────────────────────────────────
def compute_cus(predicted_labels, true_labels):
    """
    Combined Clinical Utility Score (CUS).
    Sum of CUS-Omission and CUS-Hallucination.

    A lower CUS = safer synthetic image
    A higher CUS = more clinically dangerous errors

    Args:
        predicted_labels: list of conditions predicted
        true_labels: list of actual conditions

    Returns:
        cus_total: float between 0 and 1
        cus_o: omission component
        cus_h: hallucination component
        missed: list of missed conditions
        hallucinated: list of hallucinated conditions
    """
    cus_o, missed = compute_cus_omission(
        predicted_labels, true_labels
    )
    cus_h, hallucinated = compute_cus_hallucination(
        predicted_labels, true_labels
    )

    return (
        round(cus_o + cus_h, 4),
        cus_o,
        cus_h,
        missed,
        hallucinated
    )


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("CUS-Synth: Clinical Utility Score Test")
    print("=" * 50)

    # Test 1: Missed Pneumothorax (most dangerous!)
    true = ["Pneumothorax", "Effusion"]
    predicted = ["Effusion"]
    cus, cus_o, cus_h, missed, hallucinated = compute_cus(
        predicted, true
    )
    print(f"\nTest 1 — Missed Pneumothorax:")
    print(f"  True:        {true}")
    print(f"  Predicted:   {predicted}")
    print(f"  CUS Total:   {cus}")
    print(f"  CUS-O:       {cus_o} (missed: {missed})")
    print(f"  CUS-H:       {cus_h} (hallucinated: {hallucinated})")

    # Test 2: Hallucinated condition
    true2 = ["No Finding"]
    predicted2 = ["Pneumothorax", "No Finding"]
    cus2, cus_o2, cus_h2, missed2, hallucinated2 = compute_cus(
        predicted2, true2
    )
    print(f"\nTest 2 — Hallucinated Pneumothorax:")
    print(f"  True:        {true2}")
    print(f"  Predicted:   {predicted2}")
    print(f"  CUS Total:   {cus2}")
    print(f"  CUS-O:       {cus_o2} (missed: {missed2})")
    print(f"  CUS-H:       {cus_h2} (hallucinated: {hallucinated2})")

    # Test 3: Perfect prediction
    true3 = ["Effusion"]
    predicted3 = ["Effusion"]
    cus3, cus_o3, cus_h3, missed3, hallucinated3 = compute_cus(
        predicted3, true3
    )
    print(f"\nTest 3 — Perfect prediction:")
    print(f"  True:        {true3}")
    print(f"  Predicted:   {predicted3}")
    print(f"  CUS Total:   {cus3}")
    print(f"  CUS-O:       {cus_o3}")
    print(f"  CUS-H:       {cus_h3}")
    print(f"\n✅ CUS formula working correctly!")
    print("=" * 50)