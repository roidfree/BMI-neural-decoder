>> # 🧠 Real-Time Neural Decoding of Hand Trajectories

## Overview

This project implements a **causal neural decoder** that reconstructs continuous 2D hand trajectories from intracortical spike recordings.

Using spike trains from 98 motor cortical units recorded during center-out reaching movements, the model estimates hand position (X, Y) at millisecond resolution under strict real-time constraints.

The objective is to simulate the core computational problem behind prosthetic control systems: translating raw neural activity into continuous motor output.

---

## Problem Formulation

- **Input:** 98 binary spike trains (1 ms resolution)
- **Output:** Continuous 2D hand position estimates
- **Constraint:** Fully causal (no future information allowed)
- **Evaluation Metric:** Root Mean Squared Error (RMSE, cm)

The decoder is evaluated incrementally in 20 ms intervals to simulate streaming neural data and enforce real-time feasibility.

---

## Dataset

Neural recordings were provided by the laboratory of Prof. Krishna Shenoy (Stanford University) for educational and research purposes.

Dataset characteristics:
- 98 neural units
- 8 reaching directions
- 100 trials per direction (training set)
- 1 ms temporal resolution
- Time-aligned spike trains and hand trajectories

Only planar movement (X, Y) is decoded.

---

## Methodology

### 1. Signal Processing
- Spike binning and firing rate estimation
- Temporal smoothing
- Sliding window feature extraction
- Directional tuning analysis

### 2. Decoder Design
- Causal regression-based architecture
- Lightweight, interpretable modeling
- Runtime-optimized implementation

### 3. Real-Time Constraint
- No "time travel"
- Model evaluated progressively with partial time-series input
- Total execution time < 5 minutes

---

## Performance Metric

\[
RMSE = \sqrt{\frac{1}{N} \sum (x_{true} - x_{pred})^2 + (y_{true} - y_{pred})^2}
\]

Error is averaged across dimensions and trials and reported in centimeters.

---

## Key Insights

- Neural firing demonstrates clear directional tuning.
- Temporal integration significantly improves stability.
- Feature engineering impacts performance more than model complexity.
- Simpler, well-regularized models generalize better under causal constraints.

---

## Repository Structure