# ==========================================================
# COMPLETE APNOMO IMPLEMENTATION
# Based on: Early Time Series Anomaly Prediction With
# Multi-Objective Optimization (APNOMO)
# ==========================================================

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", DEVICE)

# ==========================================================
# CONFIG (MATCH PAPER)
# ==========================================================

ALPHA_HOURS = 3
SAMPLING_RATE = 4
ALPHA = (ALPHA_HOURS * 3600) // SAMPLING_RATE

SEGMENTS = 10
NEIGHBOR_N = 2
HIDDEN_SIZE = 40
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
OVERSAMPLE_RATE = 0.3  # r in paper

TRAIN_DATA_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train"
TRAIN_FAULT_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train/train_faults"
TEST_DATA_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/test"

# ==========================================================
# DATA LOADING
# ==========================================================


def load_machine(file_path, fault_path=None, scaler=None, fit_scaler=False):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])
    df = df.ffill().bfill()

    if fit_scaler:
        data = scaler.fit_transform(df.values)
    else:
        data = scaler.transform(df.values)

    labels = np.zeros(len(data))

    if fault_path and os.path.exists(fault_path):
        fault_df = pd.read_csv(fault_path)
        faults = fault_df.iloc[:, 0].values
        for f in faults:
            start = max(0, f - ALPHA)
            labels[start:f] = 1

    return data, labels


def create_segments(data, labels):
    X, y = [], []
    for i in range(0, len(data) - ALPHA, ALPHA):
        seg = data[i:i+ALPHA]
        label = 1 if np.any(labels[i:i+ALPHA]) else 0
        X.append(seg)
        y.append(label)
    return np.array(X), np.array(y)


def load_dataset(data_dir, fault_dir=None, scaler=None, fit_scaler=False):
    files = sorted(glob.glob(os.path.join(data_dir, "*_DC_*.csv")))
    all_X, all_y = [], []

    for file in files:
        base = os.path.basename(file).replace(
            "_DC_train.csv", "").replace("_DC_test.csv", "")
        fault_file = None

        if fault_dir:
            fault_file = os.path.join(
                fault_dir, f"{base}_train_fault_data.csv")

        data, labels = load_machine(file, fault_file, scaler, fit_scaler)
        X, y = create_segments(data, labels)

        all_X.append(X)
        all_y.append(y)

    return np.concatenate(all_X), np.concatenate(all_y)

# ==========================================================
# SLIDING WINDOW SEGMENTATION (CUMULATIVE)
# ==========================================================


def cumulative_segments(X):
    sub_len = ALPHA // SEGMENTS
    segmented = []

    for seg in X:
        machine_segments = []
        for j in range(SEGMENTS):
            machine_segments.append(seg[: (j+1)*sub_len])
        segmented.append(machine_segments)

    return segmented

# ==========================================================
# MODEL
# ==========================================================


class GRUExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, HIDDEN_SIZE,
                          num_layers=2, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

# ==========================================================
# FEATURE EXTRACTOR TRAINING (WITH OVERSAMPLING)
# ==========================================================


def oversample_minority(X, y):

    X_min = X[y == 1]
    y_min = y[y == 1]

    n_samples = int(len(X) * OVERSAMPLE_RATE)

    X_resampled, y_resampled = resample(
        X_min, y_min,
        replace=True,
        n_samples=n_samples,
        random_state=42
    )

    X_balanced = np.concatenate([X, X_resampled])
    y_balanced = np.concatenate([y, y_resampled])

    return X_balanced, y_balanced


def train_feature_model(X, y):

    X_bal, y_bal = oversample_minority(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = GRUExtractor(X.shape[2]).to(DEVICE)
    clf_head = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(clf_head.parameters()), lr=LR
    )

    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            with autocast():
                features = model(xb)
                logits = clf_head(features).squeeze(-1)
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_losses.append(total_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                features = model(xb)
                logits = clf_head(features).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss += loss.item()

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")

    # plot curves
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Training Curve")
    plt.show()

    return model

# ==========================================================
# NEIGHBOR OVER-SAMPLING PREDICTORS (EXACT PAPER)
# ==========================================================


def train_segment_predictors(model, segmented_X, y):

    model.eval()
    predictors = []

    for q in range(SEGMENTS):

        X_seg, y_seg = [], []

        for i in range(len(segmented_X)):

            for j in range(max(0, q-NEIGHBOR_N),
                           min(SEGMENTS, q+NEIGHBOR_N+1)):

                sub_tensor = torch.tensor(
                    segmented_X[i][j],
                    dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = model(sub_tensor).cpu().numpy().flatten()

                X_seg.append(feat)
                y_seg.append(y[i])

        clf = LogisticRegression(max_iter=500)
        clf.fit(np.array(X_seg), np.array(y_seg))
        predictors.append(clf)

        print(f"Segment {q+1} predictor trained.")

    return predictors

# ==========================================================
# MULTI-OBJECTIVE OPTIMIZATION (PARETO + MMD)
# ==========================================================


def compute_earliness_T(pred_segments, true_labels):
    vals = []
    for seg_idx, y in zip(pred_segments, true_labels):
        if y == 1 and seg_idx is not None:
            vals.append(seg_idx / SEGMENTS)
    if len(vals) == 0:
        return 1.0
    return np.mean(vals)


def find_best_threshold(prob_matrix, y_true):

    thresholds = np.linspace(0, 1, 200)
    results = []

    for t in thresholds:

        preds, segs = [], []

        for probs in prob_matrix:

            pred = 0
            seg_choice = None

            for i, p in enumerate(probs):
                if p >= t:
                    pred = 1
                    seg_choice = i
                    break

            preds.append(pred)
            segs.append(seg_choice)

        f1 = f1_score(y_true, preds, zero_division=0)
        eT = compute_earliness_T(segs, y_true)

        results.append((t, f1, eT))

    # Pareto front
    pareto = []
    for r in results:
        dominated = False
        for r2 in results:
            if (r2[1] >= r[1] and r2[2] <= r[2]) and \
               (r2[1] > r[1] or r2[2] < r[2]):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    f1_vals = [p[1] for p in pareto]
    e_vals = [p[2] for p in pareto]

    f1_min, f1_max = min(f1_vals), max(f1_vals)
    e_min, e_max = min(e_vals), max(e_vals)

    best_dist = float("inf")
    best_thresh = 0.5

    for t, f1, e in pareto:
        norm_f1 = (f1 - f1_min) / (f1_max - f1_min + 1e-8)
        norm_e = (e - e_min) / (e_max - e_min + 1e-8)
        dist = abs(1 - norm_f1) + abs(norm_e)

        if dist < best_dist:
            best_dist = dist
            best_thresh = t

    return best_thresh

# ==========================================================
# MAIN
# ==========================================================


def main():

    os.makedirs("checkpoints", exist_ok=True)
    scaler = StandardScaler()

    print("Loading TRAIN...")
    X_train, y_train = load_dataset(
        TRAIN_DATA_DIR, TRAIN_FAULT_DIR, scaler, fit_scaler=True
    )

    print("Loading TEST...")
    X_test, y_test = load_dataset(
        TEST_DATA_DIR, None, scaler, fit_scaler=False
    )

    print("Training Feature Extractor...")
    model = train_feature_model(X_train, y_train)

    segmented_train = cumulative_segments(X_train)
    segmented_test = cumulative_segments(X_test)

    print("Training Segment Predictors...")
    predictors = train_segment_predictors(model, segmented_train, y_train)

    torch.save(model.state_dict(), "checkpoints/feature_model.pt")
    joblib.dump(predictors, "checkpoints/segment_predictors.pkl")

    print("Building probability matrix...")

    prob_matrix = []

    for i in range(len(segmented_test)):
        probs = []

        for q in range(SEGMENTS):

            sub_tensor = torch.tensor(
                segmented_test[i][q],
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = model(sub_tensor).cpu().numpy().flatten()

            prob = predictors[q].predict_proba([feat])[0][1]
            probs.append(prob)

        prob_matrix.append(probs)

    best_threshold = find_best_threshold(prob_matrix, y_test)

    print("Optimal Threshold:", best_threshold)

    with open("checkpoints/best_threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold)}, f)

    print("APNOMO Training Complete.")


if __name__ == "__main__":
    main()
