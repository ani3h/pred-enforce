# PredictiveEnforcement: Early Anomaly Prediction & Runtime Control

PredictiveEnforcement is an AI-driven system for **early time-series anomaly prediction and runtime enforcement** in industrial environments. It leverages the APNOMO framework to detect pre-anomaly patterns and enables proactive system control through continuous monitoring and corrective actions.

## Features

* Early anomaly prediction using APNOMO framework
* Multi-objective optimization for balancing **F1-score and earliness**
* GRU-based time-series learning with sliding window segmentation
* Runtime enforcement for real-time monitoring and correction
* Proactive intervention via early warning signals

## Project Structure

```id="9r7m2x"
PredictiveEnforcement/
│── data/                 # PHM Dataset
│── src/                  # Core APNOMO implementation (training + prediction)
│── results/              # Logs, metrics, and visualizations
│── README.md
│── requirements.txt
```

## Getting Started

1. **Clone the Repository**

```bash id="3k9x2p"
git clone https://github.com/ani3h/pred-enforce.git
cd PredictiveEnforcement
```

2. **Setup Environment**

```bash id="p8d2ks"
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash id="c6y8d1"
pip install -r requirements.txt
```

4. **Run Early Prediction**

```bash id="h2n7az"
python src/predict.py
```

5. **Run Runtime Enforcement**

```bash id="m5x8kl"
python src/enforcer.py
```

## Results

* High recall (~0.75) enables reliable early anomaly detection
* Optimized threshold balances detection accuracy and earliness
* Supports proactive system control over reactive failure handling

## Overview

This project combines **early anomaly prediction + runtime enforcement**, allowing systems to:

* Detect issues before failure
* Continuously monitor live data streams
* Apply gradual corrective actions for stability
