# PredictiveEnforcement: Early Anomaly Prediction & Runtime Enforcement

PredictiveEnforcement is a research framework that integrates **early time-series anomaly prediction** with **runtime enforcement** in industrial cyber-physical systems. Built on the APNOMO framework, it forecasts pre-anomaly patterns in multi-sensor telemetry and enables proactive, graduated system corrections—shifting the enforcement paradigm from reactive intervention to anticipatory safeguarding.

The framework is evaluated on the **PHM 2018 Data Challenge** dataset, comprising telemetry from ion mill etching tools used in semiconductor wafer manufacturing at Seagate Technology.

---

## Overview

Traditional runtime enforcement (RE) mechanisms intervene only when a safety violation is imminent or already in progress, forcing abrupt corrective actions that can destabilise downstream control systems and result in in-process wafer losses. This work addresses that limitation by coupling a predictive anomaly model with the enforcement loop:

1. The **APNOMO prediction model** continuously analyses sliding windows of multi-sensor telemetry and forecasts FlowCool system anomalies up to **3 hours** in advance.
2. Upon an early warning, the **runtime enforcer** applies an exponential attenuation schedule to primary sensor signals, producing smooth, graduated corrections before the monitored Signal Temporal Logic (STL) safety property is violated.

The safety property monitored is:

```
G(p_min ≤ FLOWCOOLPRESSURE ≤ p_max)
```

A violation corresponds to one of three FlowCool fault classes: pressure drop below limit, over-pressure, or confirmed helium leak.

---

## Key Results

| Metric | Value |
|---|---|
| Recall | 0.75 |
| Precision | 0.0033 |
| F1 Score | 0.0066 |
| Optimal Decision Threshold (τ*) | 0.00195 |
| Total Alarm Events | 4,170 |
| True-Positive Alarms | 102 (2.4%) |
| TP Windows — Strong Enforcement (fault prevented) | 51 (50%) |
| TP Windows — Weak Enforcement (fault delayed) | 51 (50%) |
| Mean Peak Attenuation | 0.1178 |
| Avg Enforcement Duration (TP) | 9,624.7 s |

- **High recall (0.75)** ensures the majority of upcoming FlowCool anomalies are detected before they manifest as operator-observed faults.
- **50% of true-positive windows** admit strong enforcement sufficient to keep the monitored signal within the specified operating band for the full 3-hour horizon.
- The recall-biased operating point intentionally accepts a high false-positive rate; false alarms trigger only conservative, non-disruptive actions (e.g., end-of-lot inspection scheduling) rather than emergency shutdowns.

---

## Features

- **Early Anomaly Prediction** via the APNOMO framework (multi-objective optimisation of F1-score and earliness).
- **GRU-Based Temporal Feature Extraction** with a two-layer GRU network over sliding windows of 2,700 samples (3-hour look-ahead horizon at 4 s/sample).
- **Class Imbalance Handling** through minority-class oversampling (30% positive-class ratio per training batch).
- **Ensemble Logistic Classifiers** — S = 10 sub-segment classifiers for fine-grained earliness-aware prediction.
- **Exponential Attenuation Enforcement** with a configurable decay constant (λ = 0.001 sample⁻¹) and signal floor (f = 0.05 × |s₀|).
- **Spike Detection** via rolling z-score monitoring (window w = 30, threshold z_θ = 3.0).
- **STL Property Monitoring** — formal specification of safety properties over real-valued signals.
- **Proactive vs. Reactive Comparison** — quantitative case studies demonstrating smoother trajectories and extended operating windows.

---

## Dataset

The framework uses the **PHM 2018 Data Challenge** dataset (Seagate Technology), containing multivariate sensor telemetry from 20 ion mill etching tools sampled at 0.25 Hz (one observation every 4 seconds).

- **Training:** 16 machines
- **Testing:** 4 machines
- **Signals:** 24 parameters including FlowCool pressure & flow rate, ion gauge pressure, beam/suppressor currents, fixture mechanics, and consumable usage counters.
- **Fault Classes:** FlowCool Pressure Dropped Below Limit · FlowCool Pressure Too High · FlowCool Leak

**Primary model input signals:** `FLOWCOOLPRESSURE (S14)`, `FLOWCOOLFLOWRATE (S13)`, `IONGAUGEPRESSURE (S8)`, `ETCHBEAMCURRENT (S10)`, `ETCHSUPPRESSORCURRENT (S12)`, `ETCHSOURCEUSAGE (S21)`, `ACTUALSTEPDURATION (S24)`.

> The PHM 2018 dataset is publicly available via the [PHM Society Data Challenge](https://phmsociety.org/phm_competition/2018-phm-data-challenge/).

---

## Architecture

```
Sensor Streams
      │
      ▼
Sliding Window Segmentation (α = 2700 samples)
      │
      ▼
GRU Feature Extractor (2-layer, H = 40)
      │
      ▼
Ensemble of S=10 Logistic Classifiers (per sub-segment)
      │
      ▼
Threshold Optimisation (τ* via macro-F1 search on validation set)
      │
      ▼
Early Warning Signal
      │
      ▼
Runtime Enforcer ──► Exponential Attenuation Schedule m(t) = max(e^{-λt}, f)
      │
      ▼
Controlled System Output ──► feedback ──► Sensor Streams
```

---

## Project Structure

```
PredictiveEnforcement/
├── data/                      # PHM 2018 dataset (raw and preprocessed)
├── src/
│   ├── predictor.ipynb        # APNOMO-based GRU anomaly prediction model
│   └── enforcer.ipynb         # Runtime enforcement logic and simulations
├── results/                   # Evaluation metrics, logs, and visualisations
├── signal_analysis.ipynb      # Exploratory data analysis for time-series signals
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU inference is supported but slower)
- PyTorch with CUDA support

### 1. Clone the Repository

```bash
git clone https://github.com/ani3h/pred-enforce.git
cd pred-enforce
```

### 2. Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the Dataset

Download the PHM 2018 Data Challenge dataset from the [PHM Society](https://phmsociety.org/phm_competition/2018-phm-data-challenge/) and place the files under `data/`.

---

## Usage

All components are implemented as Jupyter Notebooks for reproducibility and interactive inspection.

### Exploratory Data Analysis

```bash
jupyter notebook signal_analysis.ipynb
```

Explore time-series signal characteristics, class distributions, and preprocessing steps (standardisation, sliding-window segmentation).

### Early Anomaly Prediction

```bash
jupyter notebook src/predictor.ipynb
```

Trains the GRU-based APNOMO model, performs threshold optimisation on the validation set, and evaluates prediction performance (recall, precision, F1, earliness) on the held-out test machines.

### Runtime Enforcement Simulation

```bash
jupyter notebook src/enforcer.ipynb
```

Simulates the full predictive enforcement loop: applies the exponential attenuation schedule on alarmed windows, computes per-signal counterfactual deltas, and produces case-study visualisations (strong vs. weak enforcement regimes).

---

## Methodology

### Anomaly Prediction (APNOMO)

The APNOMO framework addresses three core challenges in industrial anomaly prediction:

- **Long sequences** — handled via fixed-length sliding-window segmentation.
- **Severe class imbalance** — addressed by minority-class oversampling (pre-anomaly windows constitute < 5% of all segments).
- **Accuracy–earliness trade-off** — resolved through multi-objective threshold optimisation jointly maximising F1-score and prediction earliness.

The GRU processes each 2,700-sample segment as a sequence of *d*-dimensional sensor vectors, producing a 40-dimensional embedding. Ten logistic regression classifiers (one per temporal sub-segment) score this embedding; a positive prediction is issued if any sub-segment probability exceeds the calibrated threshold τ*.

### Runtime Enforcement

The enforcer applies an exponential attenuation schedule to the seven primary sensor signals upon receiving an early warning:

```
m(t) = max(e^{-λt}, f)
ŝ(t) = s(t) · m(t)
```

- **λ = 0.001 sample⁻¹** — gradual decay reaching the floor within the 3-hour horizon.
- **f = 0.05 × |s₀|** — sign-preserving floor ensuring signals are never driven to zero.

This produces smooth, graduated corrections rather than abrupt shutdowns, preserving in-process wafer integrity while maximising the probability of keeping the monitored property within its STL-specified operating band.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Sampling rate | 0.25 Hz (4 s/observation) |
| Look-ahead horizon (α) | 3 hours / 2,700 samples |
| Sub-segments (S) | 10 |
| GRU hidden size (H) | 40 |
| Training epochs | 20 |
| Batch size | 32 |
| Learning rate | 1 × 10⁻³ |
| Oversampling positive ratio | 30% |
| Decay constant (λ) | 0.001 sample⁻¹ |
| Signal floor fraction (f) | 0.05 |
| Spike detection window (w) | 30 samples |
| Spike z-score threshold (z_θ) | 3.0 |

> **Note:** Approximately 60% of the available training data was used due to GPU memory constraints. Full-dataset training is part of future work.