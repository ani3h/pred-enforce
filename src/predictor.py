import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors

# GPU SETUP
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# CONFIG

# Look-ahead horizon α = 3 h at one sample per 4 s
ALPHA_HOURS = 3
SAMPLING_RATE = 4                                        # seconds per sample
ALPHA = (ALPHA_HOURS * 3600) // SAMPLING_RATE            # samples in 3 h

SEGMENTS = 10    # APNOMO sub-segments per α-window
NEIGHBOR_N = 4   # neighbours for SMOTE-style oversampling

HIDDEN_SIZE = 64  # GRU hidden units
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32

TRAIN_DATA_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train"
TRAIN_FAULT_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train/train_faults"
TRAIN_TTF_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train/train_ttf"
TEST_DATA_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/test"

CHECKPOINT_DIR = "/kaggle/working/apnomo_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Enforcer output directory
# The enforcer module reads alarm_events.pkl from here.
ENFORCER_DIR = "/kaggle/working/enforcer_inputs"
os.makedirs(ENFORCER_DIR, exist_ok=True)

# Primary sensors (direct fault indicators)
PRIMARY_SIGNALS = [
    "FLOWCOOLPRESSURE",       # S14 – primary monitored signal
    "FLOWCOOLFLOWRATE",       # S13 – flow setpoint
    "IONGAUGEPRESSURE",       # S8  – indirect helium-leak indicator
    "ETCHBEAMCURRENT",        # S10 – beam perturbations
    "ETCHSUPPRESSORCURRENT",  # S12
    "ETCHSOURCEUSAGE",        # S21 – cumulative wear
    "ACTUALSTEPDURATION",     # S24 – anomalous durations
]

# Auxiliary setpoint / contextual signals
AUX_SIGNALS = [
    "ETCHBEAMVOLTAGE",          # S9
    "ETCHSUPPRESSORVOLTAGE",    # S11
    "ETCHGASCHANNEL1READBACK",  # S15
    "ETCHPBNGASREADBACK",       # S16
    "FIXTURETILTANGLE",         # S17
    "ROTATIONSPEED",            # S18
    "ACTUALROTATIONANGLE",      # S19
    "FIXTURESHUTTERPOSITION",   # S20
    "ETCHAUXSOURCETIMER",       # S22
    "ETCHAUX2SOURCETIMER",      # S23
]

ALL_FEATURE_COLS = PRIMARY_SIGNALS + AUX_SIGNALS   # 17 features


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for col in ALL_FEATURE_COLS:
        if col in df.columns:
            cols.append(col)
        else:
            df[col] = 0.0
            cols.append(col)
    return df[cols]


# ENGINEERED FEATURES

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "FLOWCOOLPRESSURE" in df.columns and "FLOWCOOLFLOWRATE" in df.columns:
        denom = df["FLOWCOOLFLOWRATE"].replace(0, np.nan)
        df["FC_PRESSURE_FLOW_RATIO"] = (
            df["FLOWCOOLPRESSURE"] / denom).fillna(0)

    if "FLOWCOOLPRESSURE" in df.columns:
        df["FC_PRESSURE_ROLLMEAN"] = (
            df["FLOWCOOLPRESSURE"].rolling(60, min_periods=1).mean()
        )
        df["FC_PRESSURE_ROLLSTD"] = (
            df["FLOWCOOLPRESSURE"].rolling(60, min_periods=1).std().fillna(0)
        )

    return df


ENGINEERED_COLS = [
    "FC_PRESSURE_FLOW_RATIO",
    "FC_PRESSURE_ROLLMEAN",
    "FC_PRESSURE_ROLLSTD",
]

N_FEATURES = len(ALL_FEATURE_COLS) + len(ENGINEERED_COLS)   # 20


# MODEL

class GRUExtractor(nn.Module):
    """Two-layer GRU feature extractor with dropout regularisation."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]   # last-layer hidden state


# DATA LOADING

def load_machine(file_path: str, fault_path: str | None = None):
    df = pd.read_csv(file_path)

    timestamps = df["time"].values if "time" in df.columns else np.arange(
        len(df))

    df = add_engineered_features(df)

    # This is the unscaled, unmodified sensor trace the enforcer will act on.
    raw_primary = {}
    for col in PRIMARY_SIGNALS:
        raw_primary[col] = df[col].values.copy(
        ) if col in df.columns else np.zeros(len(df))

    df = select_features(df)
    df = df.ffill().bfill().fillna(0)

    data = df.values.astype(np.float32)
    labels = np.zeros(len(data), dtype=np.float32)

    if fault_path and os.path.exists(fault_path):
        fault_df = pd.read_csv(fault_path)
        fault_times = fault_df.iloc[:, 0].values

        for ft in fault_times:
            horizon_start = ft - ALPHA * SAMPLING_RATE
            mask = (timestamps >= horizon_start) & (timestamps < ft)
            labels[mask] = 1

    return data, labels, timestamps, raw_primary


# SEGMENTATION

def create_segments(data: np.ndarray, labels: np.ndarray):
    X, y = [], []
    for i in range(0, len(data) - ALPHA, ALPHA):
        seg = data[i: i + ALPHA]
        label = 1 if np.any(labels[i: i + ALPHA]) else 0
        X.append(seg)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# segment raw

def create_raw_segments(raw_primary: dict, n_windows: int) -> list[dict]:
    segs = []
    for w in range(n_windows):
        start = w * ALPHA
        end = start + ALPHA
        window = {}
        for col, arr in raw_primary.items():
            window[col] = arr[start:end] if end <= len(arr) else arr[start:]
        segs.append(window)
    return segs


# OVERSAMPLING

def neighbor_oversample(X: np.ndarray, y: np.ndarray) -> tuple:
    X_min = X[y == 1]
    X_maj = X[y == 0]

    if len(X_min) == 0:
        return X, y

    flat = X_min.reshape(len(X_min), -1)
    k = min(NEIGHBOR_N, len(X_min) - 1)

    n_synthetic = len(X_maj) - len(X_min)
    if n_synthetic <= 0:
        return X, y

    if k < 1:
        idx = np.random.choice(len(X_min), size=n_synthetic, replace=True)
        X_syn = X_min[idx]
    else:
        nbrs = NearestNeighbors(
            n_neighbors=k + 1, algorithm="ball_tree").fit(flat)
        _, indices = nbrs.kneighbors(flat)

        X_syn_list = []
        for _ in range(n_synthetic):
            i = np.random.randint(len(X_min))
            j = indices[i, np.random.randint(1, k + 1)]
            lam = np.random.rand()
            X_syn_list.append(X_min[i] + lam * (X_min[j] - X_min[i]))
        X_syn = np.array(X_syn_list, dtype=np.float32)

    y_syn = np.ones(len(X_syn), dtype=np.float32)
    return (
        np.concatenate([X, X_syn]),
        np.concatenate([y, y_syn]),
    )


# TRAIN FEATURE EXTRACTOR

def train_feature_extractor(files: list, scaler: StandardScaler):
    print("Fitting global scaler on training data …")
    all_data = []
    for file in files:
        base = os.path.basename(file).replace("_DC_train.csv", "")
        fault_file = os.path.join(
            TRAIN_FAULT_DIR, f"{base}_train_fault_data.csv")
        # load_machine now returns 4 values; unpack accordingly
        data, _, _, _ = load_machine(file, fault_file)
        all_data.append(data)

    scaler.fit(np.concatenate(all_data, axis=0))
    print("Scaler fitted.")

    model = GRUExtractor(N_FEATURES).to(DEVICE)
    classifier = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=LR
    )
    pos_weight = torch.tensor([10.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()

        for file in files:
            base = os.path.basename(file).replace("_DC_train.csv", "")
            fault_file = os.path.join(
                TRAIN_FAULT_DIR, f"{base}_train_fault_data.csv")

            data, labels, _, _ = load_machine(file, fault_file)
            data = scaler.transform(data)

            X, y = create_segments(data, labels)
            if len(X) == 0:
                continue

            X, y = neighbor_oversample(X, y)

            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]

            for i in range(0, len(X), BATCH_SIZE):
                bX = torch.tensor(X[i: i + BATCH_SIZE]).to(DEVICE)
                by = torch.tensor(y[i: i + BATCH_SIZE]).to(DEVICE)

                feat = model(bX)
                logits = classifier(feat).squeeze(-1)
                loss = criterion(logits, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{EPOCHS}  loss {epoch_loss:.3f}")

    return model


# FEATURE EXTRACTION

def extract_features(
    model: GRUExtractor,
    files: list,
    scaler: StandardScaler,
    fault_dir: str | None = None,
):
    model.eval()
    feats, all_labels = [], []
    raw_segs_by_machine = []   # NEW

    with torch.no_grad():
        for file in files:
            base = os.path.basename(file).replace(
                "_DC_train.csv", "").replace("_DC_test.csv", "")

            fault_file = None
            if fault_dir:
                candidate = os.path.join(
                    fault_dir, f"{base}_train_fault_data.csv")
                if os.path.exists(candidate):
                    fault_file = candidate

            data, labels, _, raw_primary = load_machine(file, fault_file)
            data = scaler.transform(data)

            X, y = create_segments(data, labels)
            if len(X) == 0:
                raw_segs_by_machine.append([])
                continue

            # Raw segments aligned with X/y windows
            raw_segs = create_raw_segments(raw_primary, n_windows=len(X))
            raw_segs_by_machine.append(raw_segs)

            for seg, label in zip(X, y):
                tensor = torch.tensor(seg).unsqueeze(0).to(DEVICE)
                feat = model(tensor).cpu().numpy().flatten()
                feats.append(feat)
                all_labels.append(label)

    return (
        np.array(feats, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
        raw_segs_by_machine,
    )


# PER-SEGMENT PREDICTORS

def build_segment_features(full_feats: np.ndarray, full_labels: np.ndarray):
    n = len(full_feats)
    seg_size = max(1, n // SEGMENTS)
    seg_X, seg_y = [], []

    for q in range(SEGMENTS):
        start = q * seg_size
        end = (q + 1) * seg_size if q < SEGMENTS - 1 else n
        seg_X.append(full_feats[start:end])
        seg_y.append(full_labels[start:end])

    return seg_X, seg_y


def train_predictors(features: np.ndarray, labels: np.ndarray):
    seg_X, seg_y = build_segment_features(features, labels)
    predictors = []

    for q in range(SEGMENTS):
        X_cum = np.concatenate(seg_X[q:], axis=0)
        y_cum = np.concatenate(seg_y[q:], axis=0)

        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5)
        clf.fit(X_cum, y_cum)
        predictors.append(clf)
        print(f"Segment {q+1} predictor trained")

    return predictors


# THRESHOLD SEARCH

def find_best_threshold(prob_matrix: list, y_true: np.ndarray):
    BETA = 0.5

    all_probs = np.unique(np.array(prob_matrix).flatten())
    if len(all_probs) > 200:
        all_probs = np.percentile(all_probs, np.linspace(0, 100, 200))

    thresholds = [(all_probs[i] + all_probs[i + 1]) / 2
                  for i in range(len(all_probs) - 1)]
    if not thresholds:
        return 0.5

    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        preds, segs = [], []
        for row in prob_matrix:
            pred, seg = 0, None
            for i, p in enumerate(row):
                if p >= t:
                    pred, seg = 1, i
                    break
            preds.append(pred)
            segs.append(seg)

        f1 = f1_score(y_true, preds, zero_division=0)
        earliness = compute_earliness(segs, y_true)
        score = BETA * f1 + (1 - BETA) * (1 - earliness)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t


# EARLINESS

def compute_earliness(segs: list, y: np.ndarray) -> float:
    vals = [s / SEGMENTS for s, label in zip(segs, y)
            if label == 1 and s is not None]
    return float(np.mean(vals)) if vals else 1.0


# BUILD & SAVE ALARM EVENTS

def build_alarm_events(
    prob_matrix: list,
    threshold: float,
    test_labels: np.ndarray,
    raw_segs_by_machine: list,
    test_files: list,
) -> list[dict]:
    """
    For every test window where APNOMO fires an alarm, package:

      machine_id        – basename of the source CSV
      window_idx        – global window index across all test machines
      alarm_segment     – sub-segment index (0..SEGMENTS-1) where threshold crossed
      remaining_horizon – number of α-windows left until end of prediction horizon
                          (= SEGMENTS - alarm_segment - 1)
      remaining_seconds – remaining_horizon expressed in wall-clock seconds
      prob_trace        – full SEGMENTS-length probability vector for this window
      true_label        – 1 if this is a genuine pre-anomaly window, else 0
      raw_signals       – dict {signal_name -> np.array}  unscaled sensor values
                          for THIS α-window, available for the enforcer to act on

    alarm_events with pred == 0 are excluded (no alarm → no enforcement needed).
    """
    alarm_events = []

    # Flatten raw_segs_by_machine back to a per-window list aligned with prob_matrix
    flat_raw = []
    for machine_segs in raw_segs_by_machine:
        flat_raw.extend(machine_segs)

    # Build machine-id lookup (one entry per window, same order as flat_raw)
    machine_ids = []
    for file, machine_segs in zip(test_files, raw_segs_by_machine):
        base = os.path.basename(file).replace("_DC_test.csv", "")
        machine_ids.extend([base] * len(machine_segs))

    for w_idx, (row, true_label) in enumerate(zip(prob_matrix, test_labels)):
        pred, alarm_seg = 0, None
        for seg_idx, p in enumerate(row):
            if p >= threshold:
                pred, alarm_seg = 1, seg_idx
                break

        if pred == 0:
            continue   # no alarm → nothing for the enforcer to do

        remaining_horizon = SEGMENTS - alarm_seg - 1
        remaining_seconds = remaining_horizon * ALPHA * SAMPLING_RATE  # wall-clock

        event = {
            "machine_id":        machine_ids[w_idx] if w_idx < len(machine_ids) else f"machine_{w_idx}",
            "window_idx":        w_idx,
            "alarm_segment":     alarm_seg,
            "remaining_horizon": remaining_horizon,   # in sub-segments
            "remaining_seconds": remaining_seconds,   # in seconds
            "prob_trace":        np.array(row),
            "true_label":        int(true_label),
            # Unscaled primary-signal snapshots for the enforcer to act on.
            # Each value is a 1-D array of ALPHA samples.
            "raw_signals":       flat_raw[w_idx] if w_idx < len(flat_raw) else {},
        }
        alarm_events.append(event)

    return alarm_events


# MAIN

def main():
    train_files = sorted(
        glob.glob(os.path.join(TRAIN_DATA_DIR, "*_DC_train.csv")))
    test_files = sorted(
        glob.glob(os.path.join(TEST_DATA_DIR,  "*_DC_test.csv")))

    if not test_files:
        print("No separate test files found – using last 4 train machines as test set.")
        np.random.seed(42)
        np.random.shuffle(train_files)
        train_files, test_files = train_files[:-4], train_files[-4:]

    print(f"Train machines: {len(train_files)}")
    print(f"Test  machines: {len(test_files)}")

    scaler = StandardScaler()

    # Train
    model = train_feature_extractor(train_files, scaler)

    torch.save(model.state_dict(), os.path.join(
        CHECKPOINT_DIR, "feature_extractor.pt"))
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))

    # Extract features
    train_feats, train_labels, _ = extract_features(
        model, train_files, scaler, fault_dir=TRAIN_FAULT_DIR
    )
    test_feats, test_labels, raw_segs_by_machine = extract_features(
        model, test_files, scaler, fault_dir=TRAIN_FAULT_DIR
    )

    print(f"Train anomalies: {int(train_labels.sum())}")
    print(f"Test  anomalies: {int(test_labels.sum())}")

    # Per-segment predictors
    predictors = train_predictors(train_feats, train_labels)
    joblib.dump(predictors, os.path.join(CHECKPOINT_DIR, "predictors.pkl"))

    # Probability matrix
    prob_matrix = []
    for feat in test_feats:
        probs = [clf.predict_proba(feat.reshape(1, -1))[0][1]
                 for clf in predictors]
        prob_matrix.append(probs)

    # Threshold search
    threshold = find_best_threshold(prob_matrix, test_labels)
    print(f"\nBest threshold: {threshold:.6f}")

    # Final predictions
    preds, segs = [], []
    for row in prob_matrix:
        pred, seg = 0, None
        for i, p in enumerate(row):
            if p >= threshold:
                pred, seg = 1, i
                break
        preds.append(pred)
        segs.append(seg)

    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels,    preds, zero_division=0)
    f1 = f1_score(test_labels,        preds, zero_division=0)
    earliness = compute_earliness(segs, test_labels)

    print("\n========= FINAL METRICS =========")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"Earliness-T : {earliness:.4f}")

    alarm_events = build_alarm_events(
        prob_matrix=prob_matrix,
        threshold=threshold,
        test_labels=test_labels,
        raw_segs_by_machine=raw_segs_by_machine,
        test_files=test_files,
    )

    enforcer_payload = {
        "alarm_events":   alarm_events,
        "threshold":      threshold,
        "alpha":          ALPHA,
        "sampling_rate":  SAMPLING_RATE,
        "segments":       SEGMENTS,
        "primary_signals": PRIMARY_SIGNALS,
    }
    payload_path = os.path.join(ENFORCER_DIR, "alarm_events.pkl")
    joblib.dump(enforcer_payload, payload_path)

    print(f"\nEnforcer payload saved → {payload_path}")
    print(f"  Total alarm events  : {len(alarm_events)}")
    print(
        f"  True-positive alarms: {sum(e['true_label'] for e in alarm_events)}")
    print(
        f"  False-positive alarms: {sum(1 - e['true_label'] for e in alarm_events)}")
    print("  Keys in each event  : machine_id, window_idx, alarm_segment,")
    print("                        remaining_horizon, remaining_seconds,")
    print("                        prob_trace, true_label, raw_signals")


if __name__ == "__main__":
    main()
