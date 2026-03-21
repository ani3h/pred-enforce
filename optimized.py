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
from sklearn.utils import resample

# GPU SETUP
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# CONFIG
ALPHA_HOURS = 3
SAMPLING_RATE = 4
ALPHA = (ALPHA_HOURS * 3600) // SAMPLING_RATE

SEGMENTS = 10
NEIGHBOR_N = 4

HIDDEN_SIZE = 40
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32

TRAIN_RATIO = 0.8
OVERSAMPLE_RATE = 0.3

TRAIN_DATA_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train"
TRAIN_FAULT_DIR = "/kaggle/input/datasets/ani3hhh/phm-data/data/train/train_faults"

CHECKPOINT_DIR = "/kaggle/working/apnomo_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# MODEL


class GRUExtractor(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            HIDDEN_SIZE,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):

        _, h = self.gru(x)

        return h[-1]

# DATA FUNCTIONS


def load_machine(file_path, fault_path=None):

    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])
    df = df.ffill().bfill()

    data = df.values

    labels = np.zeros(len(data))

    if fault_path and os.path.exists(fault_path):

        fault_df = pd.read_csv(fault_path)
        faults = fault_df.iloc[:, 0].values

        for f in faults:

            start = max(0, f - ALPHA)
            labels[start:f] = 1

    return data, labels


def create_segments(data, labels):

    X = []
    y = []

    for i in range(0, len(data) - ALPHA, ALPHA):

        seg = data[i:i+ALPHA]
        label = 1 if np.any(labels[i:i+ALPHA]) else 0

        X.append(seg)
        y.append(label)

    return np.array(X), np.array(y)


def oversample(X, y):

    X_min = X[y == 1]
    y_min = y[y == 1]

    if len(X_min) == 0:
        return X, y

    n = int(len(X) * OVERSAMPLE_RATE)

    X_res, y_res = resample(
        X_min,
        y_min,
        replace=True,
        n_samples=n,
        random_state=42
    )

    return np.concatenate([X, X_res]), np.concatenate([y, y_res])

# TRAIN FEATURE EXTRACTOR (GPU BATCHED)


def train_feature_extractor(files):

    scaler = StandardScaler()

    model = None
    classifier = None
    optimizer = None

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):

        epoch_loss = 0

        for file in files:

            base = os.path.basename(file).replace("_DC_train.csv", "")

            fault_file = os.path.join(
                TRAIN_FAULT_DIR,
                f"{base}_train_fault_data.csv"
            )

            data, labels = load_machine(file, fault_file)

            data = scaler.fit_transform(data)

            X, y = create_segments(data, labels)

            X, y = oversample(X, y)

            if model is None:

                model = GRUExtractor(X.shape[2]).to(DEVICE)

                classifier = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(classifier.parameters()),
                    lr=LR
                )

            for i in range(0, len(X), BATCH_SIZE):

                batch_X = X[i:i+BATCH_SIZE]
                batch_y = y[i:i+BATCH_SIZE]

                X_tensor = torch.tensor(
                    batch_X, dtype=torch.float32).to(DEVICE)
                y_tensor = torch.tensor(
                    batch_y, dtype=torch.float32).to(DEVICE)

                feat = model(X_tensor)

                logits = classifier(feat).squeeze(-1)

                loss = criterion(logits, y_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{EPOCHS} loss {epoch_loss:.3f}")

    return model

# FEATURE EXTRACTION


def extract_features(model, files):

    scaler = StandardScaler()

    feats = []
    labels = []

    with torch.no_grad():

        for file in files:

            base = os.path.basename(file).replace("_DC_train.csv", "")

            fault_file = os.path.join(
                TRAIN_FAULT_DIR,
                f"{base}_train_fault_data.csv"
            )

            data, lab = load_machine(file, fault_file)

            data = scaler.fit_transform(data)

            X, y = create_segments(data, lab)

            for seg, label in zip(X, y):

                tensor = torch.tensor(seg, dtype=torch.float32)\
                    .unsqueeze(0).to(DEVICE)

                feat = model(tensor).cpu().numpy().flatten()

                feats.append(feat)
                labels.append(label)

    return np.array(feats), np.array(labels)

# TRAIN PREDICTORS


def train_predictors(features, labels):

    predictors = []

    for q in range(SEGMENTS):

        clf = LogisticRegression(
            max_iter=300,
            class_weight="balanced"
        )

        clf.fit(features, labels)

        predictors.append(clf)

        print("Segment", q+1, "predictor trained")

    return predictors

# THRESHOLD SEARCH


def find_best_threshold(prob_matrix, y_true):

    probs = np.array(prob_matrix).flatten()
    probs = np.sort(probs)

    thresholds = [(probs[i]+probs[i+1])/2
                  for i in range(len(probs)-1)]

    best_t = 0.5
    best_f1 = 0

    for t in thresholds:

        preds = []

        for probs in prob_matrix:

            pred = 0

            for p in probs:
                if p >= t:
                    pred = 1
                    break

            preds.append(pred)

        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t

# EARLINESS


def compute_earliness(segs, y):

    vals = []

    for s, label in zip(segs, y):

        if label == 1 and s is not None:

            vals.append(s/SEGMENTS)

    if len(vals) == 0:
        return 1

    return np.mean(vals)

# MAIN


def main():

    files = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, "*_DC_*.csv")))

    np.random.seed(42)
    np.random.shuffle(files)

    split = int(len(files) * TRAIN_RATIO)

    train_files = files[:split]
    test_files = files[split:]

    print("Train machines:", len(train_files))
    print("Test machines:", len(test_files))

    model = train_feature_extractor(train_files)

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "feature_extractor.pt")
    )

    train_feats, train_labels = extract_features(model, train_files)
    test_feats, test_labels = extract_features(model, test_files)

    print("Train anomalies:", np.sum(train_labels))
    print("Test anomalies:", np.sum(test_labels))

    predictors = train_predictors(train_feats, train_labels)

    joblib.dump(
        predictors,
        os.path.join(CHECKPOINT_DIR, "predictors.pkl")
    )

    prob_matrix = []

    for feat in test_feats:

        probs = [p.predict_proba([feat])[0][1] for p in predictors]

        prob_matrix.append(probs)

    threshold = find_best_threshold(prob_matrix, test_labels)

    print("Best threshold:", threshold)

    preds = []
    segs = []

    for probs in prob_matrix:

        pred = 0
        seg = None

        for i, p in enumerate(probs):

            if p >= threshold:
                pred = 1
                seg = i
                break

        preds.append(pred)
        segs.append(seg)

    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    earliness = compute_earliness(segs, test_labels)

    print("\n========= FINAL METRICS =========")

    print("Precision :", precision)
    print("Recall    :", recall)
    print("F1-score  :", f1)
    print("Earliness-T :", earliness)


if __name__ == "__main__":
    main()
