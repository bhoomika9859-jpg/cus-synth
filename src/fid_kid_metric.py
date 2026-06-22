# FID and KID Metrics for comparison with CUS
# Author: bhoomika9859-jpg

import numpy as np
from scipy import linalg

# ── Helper: Calculate Mean and Covariance ─────────────────────────
def calculate_stats(features):
    """
    Calculate mean and covariance of image features.
    This is the foundation of both FID and KID!
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ── FID Score ─────────────────────────────────────────────────────
def compute_fid(real_features, synthetic_features):
    """
    Frechet Inception Distance (FID)
    Measures how similar two sets of images LOOK.
    Lower = more similar looking images.

    BUT — it doesn't care if a dangerous condition
    is missed! That's the problem CUS solves!
    """
    mu1, sigma1 = calculate_stats(real_features)
    mu2, sigma2 = calculate_stats(synthetic_features)

    # Difference between means
    diff = mu1 - mu2

    # Square root of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )

    return round(float(fid), 4)


# ── KID Score ─────────────────────────────────────────────────────
def compute_kid(real_features, synthetic_features, num_subsets=10):
    """
    Kernel Inception Distance (KID)
    Similar to FID but more reliable with small datasets.
    Lower = more similar looking images.

    Still doesn't catch clinical danger though!
    That's why CUS is needed!
    """
    subset_size = min(
        len(real_features),
        len(synthetic_features),
        100
    )

    kid_scores = []

    for _ in range(num_subsets):
        # Random subsets
        real_idx = np.random.choice(
            len(real_features), subset_size, replace=False
        )
        syn_idx = np.random.choice(
            len(synthetic_features), subset_size, replace=False
        )

        real_sub = real_features[real_idx]
        syn_sub = synthetic_features[syn_idx]

        # Polynomial kernel
        def kernel(x, y):
            return (x.dot(y.T) / x.shape[1] + 1) ** 3

        kid = (
            np.mean(kernel(real_sub, real_sub)) +
            np.mean(kernel(syn_sub, syn_sub)) -
            2 * np.mean(kernel(real_sub, syn_sub))
        )
        kid_scores.append(kid)

    return round(float(np.mean(kid_scores)), 4)


# ── Simulate Image Features ───────────────────────────────────────
def extract_features(images, feature_dim=64):
    """
    Simulate feature extraction from images.
    In Week 4 we'll replace this with a real
    AI model (DenseNet121)!
    """
    features = []
    for _, img in images:
        # Resize and flatten image as simple features
        import cv2
        resized = cv2.resize(img, (8, 8))
        flat = resized.flatten().astype(np.float32)
        # Pad or trim to feature_dim
        if len(flat) < feature_dim:
            flat = np.pad(flat, (0, feature_dim - len(flat)))
        else:
            flat = flat[:feature_dim]
        features.append(flat)
    return np.array(features)


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing FID and KID with random features...")
    # Simulate real vs synthetic features
    np.random.seed(42)
    real_feats = np.random.randn(100, 64)
    syn_feats = np.random.randn(100, 64) + 0.5

    fid = compute_fid(real_feats, syn_feats)
    kid = compute_kid(real_feats, syn_feats)

    print(f"FID Score: {fid}")
    print(f"KID Score: {kid}")
    print("(Lower = more similar images)")