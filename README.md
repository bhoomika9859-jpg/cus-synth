# CUS-Synth

## Hypothesis
Current metrics measure realism, not clinical risk.
FID and KID were designed for natural image synthesis
and have never been validated for clinical utility in
diagnostic imaging. We test whether a harm-aware metric
grounded in ACR diagnostic error severity (CUS) better
aligns with downstream diagnostic performance than
distributional similarity metrics.

## Dataset
NIH ChestX-ray14 (Sample) — 5606 chest X-ray images
with 14 disease labels. MIDI-B access pending PhysioNet
credentialing approval for future validation.

## Project Structure
- `data/` — datasets (NIH ChestX-ray14 images)
- `src/` — all source code
- `notebooks/` — experiments and analysis
- `results/` — outputs, plots, scores

## Status
## Status
✅ Week 1 — Project setup complete
🔄 Week 2 — Data pipeline in progress
