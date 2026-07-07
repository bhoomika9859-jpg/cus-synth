# FID and KID Metrics
# CUS-Synth: Clinical Utility Score for Synthetic Medical Image Evaluation
# Author: bhoomika9859-jpg
# Version 2.0 — Updated with honest framing

import numpy as np
from scipy import linalg

# ── Important Note ────────────────────────────────────────────────
# FID and KID are DISTRIBUTIONAL metrics — they give ONE global
# score for an entire set of images, not per-image scores.
# This is a fundamental limitation for medical image evaluation
# where per-image clinical safety matters.
# CUS addresses this by giving per-image clinical risk scores.

# ── Helper: Calculate Stats ───────────────────────────────────────
def calculate_stats(features):
    """
    Calculate mean and covariance of image features.
    Foundation of both FID and KID.
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ── FID Score ─────────────────────────────────────────────────────
def compute_fid(real_features, synthetic_features):
    """
    Frechet Inception Distance (FID)

    Measures distributional similarity between two sets
    of images. Lower = more similar looking images.

    ⚠️ LIMITATION: Gives ONE global score for entire dataset.
    Cannot identify which individual images are dangerous.
    Not designed for medical image evaluation.
    """
    mu1, sigma1 = calculate_stats(real_features)
    mu2, sigma2 = calculate_stats(synthetic_features)

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )

    return round(float(fid), 4)


# ── KID Score ─────────────────────────────────────────────────────
def compute_kid(real_features, synthetic_features,
                num_subsets=10):
    """
    Kernel Inception Distance (KID)

    More reliable than FID with small datasets.
    Lower = more similar looking images.

    ⚠️ LIMITATION: Also gives ONE global score.
    Cannot identify which individual images are dangerous.
    Not designed for medical image evaluation.
    """
    subset_size = min(
        len(real_features),
        len(synthetic_features),
        100
    )

    kid_scores = []
    for _ in range(num_subsets):
        real_idx = np.random.choice(
            len(real_features), subset_size, replace=False
        )
        syn_idx = np.random.choice(
            len(synthetic_features), subset_size, replace=False
        )

        real_sub = real_features[real_idx]
        syn_sub = synthetic_features[syn_idx]

        def kernel(x, y):
            return (x.dot(y.T) / x.shape[1] + 1) ** 3

        kid = (
            np.mean(kernel(real_sub, real_sub)) +
            np.mean(kernel(syn_sub, syn_sub)) -
            2 * np.mean(kernel(real_sub, syn_sub))
        )
        kid_scores.append(kid)

    return round(float(np.mean(kid_scores)), 4)


# ── Feature Extraction ────────────────────────────────────────────
def extract_features(images, feature_dim=64):
    """
    Extract simple features from images for FID/KID.
    In production this would use InceptionV3 features.
    For this study we use flattened pixel features as proxy.
    """
    features = []
    for _, img in images:
        import cv2
        resized = cv2.resize(img, (8, 8))
        flat = resized.flatten().astype(np.float32)
        if len(flat) < feature_dim:
            flat = np.pad(
                flat, (0, feature_dim - len(flat))
            )
        else:
            flat = flat[:feature_dim]
        features.append(flat)
    return np.array(features)


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("CUS-Synth: FID and KID Test")
    print("=" * 55)

    np.random.seed(42)
    real_feats = np.random.randn(100, 64)
    syn_feats = np.random.randn(100, 64) + 0.5

    fid = compute_fid(real_feats, syn_feats)
    kid = compute_kid(real_feats, syn_feats)

    print(f"\nFID Score: {fid}")
    print(f"KID Score: {kid}")
    print(f"\n⚠️  These are GLOBAL scores — no per-image signal!")
    print(f"⚠️  Cannot identify clinically dangerous images!")
    print(f"✅ CUS solves this with per-image clinical scores!")
    print("=" * 55)