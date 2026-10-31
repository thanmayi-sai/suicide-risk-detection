import os, json, random, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA = BASE / "Datasets"
MODEL_DIR = BASE / "models" / "text_model"
REPORT_DIR = BASE / "reports" / "text_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA / "training-set-clean.csv"
TEST_CSV = DATA / "testing-set-clean-leakage-free.csv"

VOCAB_SIZE = 20000
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20
VAL_SIZE = 0.15
PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(text):
    return str(text).strip().split()

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    required = {"model_text", "is_suicide"}
    if not required.issubset(train_df.columns) or not required.issubset(test_df.columns):
        raise ValueError("Required columns model_text and is_suicide are missing.")
    train_df = train_df.dropna(subset=["model_text", "is_suicide"]).copy()
    test_df = test_df.dropna(subset=["model_text", "is_suicide"]).copy()
    train_df["model_text"] = train_df["model_text"].astype(str)
    test_df["model_text"] = test_df["model_text"].astype(str)
    train_df["is_suicide"] = train_df["is_suicide"].astype(int)
    test_df["is_suicide"] = test_df["is_suicide"].astype(int)
    return train_df, test_df

def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>":0, "<OOV>":1}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)]
    ids = ids[:max_len]
    length = max(1, len(ids))
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids, length

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        hcat = torch.cat((h[-2], h[-1]), dim=1)
        z = self.dropout(self.relu(self.fc1(hcat)))
        return self.fc2(z).squeeze(1)

def evaluate(model, loader):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, lb, yb in loader:
            xb, lb = xb.to(DEVICE), lb.to(DEVICE)
            logits = model(xb, lb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(yb.numpy().tolist())
    probs = np.array(probs)
    ys = np.array(ys).astype(int)
    preds = (probs >= 0.5).astype(int)
    return ys, preds, probs

def main():
    train_df, test_df = load_data()

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        train_df["model_text"].values,
        train_df["is_suicide"].values,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_df["is_suicide"].values,
    )

    vocab = build_vocab(X_train_text)
    train_lengths = np.array([len(tokenize(t)) for t in X_train_text])
    max_len = int(np.percentile(train_lengths, 95))
    max_len = max(80, min(max_len, 400))

    train_encoded = [encode(t, vocab, max_len) for t in X_train_text]
    val_encoded = [encode(t, vocab, max_len) for t in X_val_text]
    test_encoded = [encode(t, vocab, max_len) for t in test_df["model_text"].values]

    X_train = np.array([x[0] for x in train_encoded], dtype=np.int64)
    L_train = np.array([x[1] for x in train_encoded], dtype=np.int64)
    X_val = np.array([x[0] for x in val_encoded], dtype=np.int64)
    L_val = np.array([x[1] for x in val_encoded], dtype=np.int64)
    X_test = np.array([x[0] for x in test_encoded], dtype=np.int64)
    L_test = np.array([x[1] for x in test_encoded], dtype=np.int64)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(L_train), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(L_val), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(L_test), torch.tensor(test_df["is_suicide"].values, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TextBiLSTM(len(vocab), EMBED_DIM, LSTM_UNITS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    no_improve = 0
    history = {"train_loss":[], "val_loss":[]}
    epochs_completed = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, lb, yb in val_loader:
                xb, lb, yb = xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
                vlosses.append(criterion(model(xb, lb), yb).item())

        tr_loss = float(np.mean(losses))
        va_loss = float(np.mean(vlosses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        epochs_completed = epoch
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_text_lstm.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    model.load_state_dict(torch.load(MODEL_DIR / "best_text_lstm.pt", map_location=DEVICE))
    torch.save(model.state_dict(), MODEL_DIR / "final_text_lstm.pt")

    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    y_test, preds, probs = evaluate(model, test_loader)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "framework": "PyTorch",
        "device": str(DEVICE),
        "seed": SEED,
        "train_rows_total": int(len(train_df)),
        "train_rows_model_fit": int(len(X_train_text)),
        "validation_rows": int(len(X_val_text)),
        "test_rows": int(len(test_df)),
        "vocab_size": int(len(vocab)),
        "max_sequence_length": int(max_len),
        "embedding_dim": EMBED_DIM,
        "lstm_units": LSTM_UNITS,
        "epochs_requested": EPOCHS,
        "epochs_completed": epochs_completed,
        "batch_size": BATCH_SIZE,
        "threshold": 0.5,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }
    with open(REPORT_DIR / "text_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds, digits=4, zero_division=0))

    pd.DataFrame({
        "record_id": test_df.get("record_id", pd.Series(range(1, len(test_df)+1))),
        "true_label": y_test,
        "predicted_label": preds,
        "suicide_text_probability": probs,
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Text BiLSTM Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["0","1"])
    plt.yticks([0,1], ["0","1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Text BiLSTM Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "training_history.png", dpi=180)
    plt.close()

    config = {
        "input_column": "model_text",
        "target_column": "is_suicide",
        "train_file": str(TRAIN_CSV),
        "test_file": str(TEST_CSV),
        "vocab_fit_scope": "training subset only",
        "validation_split_method": "15% stratified from cleaned training set",
        "test_set_policy": "held out and used only for final evaluation",
        "architecture": "Embedding -> Bidirectional LSTM -> Dense -> Dropout -> Sigmoid",
        "class_balancing": "No SMOTE/oversampling; source classes are already near-balanced",
    }
    with open(MODEL_DIR / "text_model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(metrics, indent=2), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
