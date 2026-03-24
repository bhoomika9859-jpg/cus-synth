# CUS-Synth

## Hypothesis
Current metrics measure realism, not clinical risk.
FID and KID were designed for natural image synthesis
and have never been validated for clinical utility in
diagnostic imaging. We test whether a harm-aware metric
grounded in ACR diagnostic error severity (CUS) better
aligns with downstream diagnostic performance than
distributional similarity metrics.

## Project Structure
- `data/` — datasets (MIDI-B CXR images)
- `src/` — all source code
- `notebooks/` — experiments and analysis
- `results/` — outputs, plots, scores

## Status
🚧 Week 1 — Setup in progress