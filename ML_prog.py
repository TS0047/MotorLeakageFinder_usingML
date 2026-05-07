import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "leaknet.pth"
FEATURES   = ["RPM", "AF", "EGT", "IP", "EP"]
ZONE_MAP   = {"NONE": 0, "A": 1, "B": 2, "C": 3, "D": 4}
INV_MAP    = {v: k for k, v in ZONE_MAP.items()}
print(f"Using: {DEVICE}")

# ── Model Definition ─────────────────────────────────────────────────────────
class LeakNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 5)
        )
    def forward(self, x): return self.net(x)

# ── Predict Function ──────────────────────────────────────────────────────────
def predict(RPM, AF, EGT, IP, EP):
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE,weights_only=False)
    model = LeakNet().to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    x = np.array([[RPM, AF, EGT, IP, EP]], dtype=np.float32)
    x = (x - ckpt['scaler_mean']) / ckpt['scaler_scale']
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
    zone = INV_MAP[out.argmax(1).item()]
    label = "HEALTHY" if zone == "NONE" else f"LEAK — Zone {zone}"
    print(f"Prediction: {label}")
    return zone

# ── Train & Save ──────────────────────────────────────────────────────────────
def train():
    healthy = pd.read_csv("healthy_train.csv")
    leak    = pd.read_csv("leak_test.csv")
    df      = pd.concat([healthy, leak], ignore_index=True)

    X = df[FEATURES].values
    y = df["zone"].map(ZONE_MAP).values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                 stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)

    def to_tensor(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.long).to(DEVICE)
        )

    train_loader = DataLoader(to_tensor(X_tr, y_tr),  batch_size=64, shuffle=True)
    val_loader   = DataLoader(to_tensor(X_val, y_val), batch_size=64)

    model     = LeakNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(1, 101):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb).argmax(1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            print(f"Epoch {epoch:3d} | Val Acc: {correct/total:.4f}")

    # Final report
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).argmax(1).cpu().numpy()
    print("\n", classification_report(y_val, preds,
          target_names=[INV_MAP[i] for i in range(5)]))

    torch.save({
        'model_state':  model.state_dict(),
        'scaler_mean':  scaler.mean_,
        'scaler_scale': scaler.scale_,
    }, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

# ── Entry Point ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("No saved model found. Training now...")
    train()
else:
    print(f"Loaded existing model from {MODEL_PATH} (delete it to retrain)")

# ── Example Inference ─────────────────────────────────────────────────────────
predict(RPM=1700, AF=7.5, EGT=335, IP=1.24, EP=1.14)   # expected: HEALTHY
predict(RPM=1700, AF=5.5, EGT=335, IP=1.24, EP=1.14)   # expected: Zone A (low AF)
predict(RPM=1700, AF=7.5, EGT=385, IP=1.05, EP=1.14)   # expected: Zone B (low IP + high EGT)