# CUS Metric — Clinical Utility Score
# Based on ACR diagnostic error severity weights
# Author: bhoomika9859-jpg

import numpy as np

# ── ACR Severity Weights ──────────────────────────────────────────
# These weights are based on ACR (American College of Radiology)
# diagnostic error severity levels
# Higher weight = more dangerous to miss this condition

ACR_WEIGHTS = {
    "Cardiomegaly":           0.7,
    "Effusion":               0.8,
    "Infiltration":           0.6,
    "Mass":                   0.9,
    "Nodule":                 0.85,
    "Pneumonia":              0.95,
    "Pneumothorax":           1.0,   # Most dangerous!
    "Consolidation":          0.75,
    "Edema":                  0.85,
    "Emphysema":              0.6,
    "Fibrosis":               0.5,
    "Pleural_Thickening":     0.5,
    "Hernia":                 0.4,
    "No Finding":             0.0,   # Healthy — no risk
}

# ── CUS Formula ───────────────────────────────────────────────────
def compute_cus(predicted_labels, true_labels):
    """
    Compute the Clinical Utility Score (CUS) between
    predicted and true labels.

    A lower CUS = safer synthetic image
    A higher CUS = more clinically dangerous errors

    Args:
        predicted_labels: list of condition names predicted
        true_labels: list of actual condition names

    Returns:
        cus_score: float between 0 and 1
    """

    missed = set(true_labels) - set(predicted_labels)
    false_alarms = set(predicted_labels) - set(true_labels)

    # Penalty for missing real conditions (more dangerous!)
    missed_penalty = sum(ACR_WEIGHTS.get(c, 0.5) for c in missed)

    # Penalty for false alarms (less dangerous but still bad)
    false_alarm_penalty = sum(
        ACR_WEIGHTS.get(c, 0.5) * 0.5 for c in false_alarms
    )

    total_penalty = missed_penalty + false_alarm_penalty

    # Normalise to 0-1 range
    max_possible = sum(ACR_WEIGHTS.values())
    cus_score = total_penalty / max_possible

    return round(cus_score, 4)


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: real image has Pneumothorax and Effusion
    # but model predicted only Effusion — missed Pneumothorax!
    true =      ["Pneumothorax", "Effusion"]
    predicted = ["Effusion"]

    score = compute_cus(predicted, true)
    print(f"CUS Score: {score}")
    print("(0 = perfect, 1 = most dangerous)")