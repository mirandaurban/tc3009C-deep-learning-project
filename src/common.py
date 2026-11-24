"""Funciones compartidas por los notebooks 03 y 04."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = "velocity_kmh"

# Features (ver notebook 02, sección 2)
DEMO = ["age", "bmi", "female"]
CARRERA_PRE = ["distancia_km"]
OTRAS = ["footwear", "injury"]
ENTRENO = ["typical", "max", "sprint", "tempo", "endurancecat", "endurancespeed"]
FEATS_A = DEMO + CARRERA_PRE + OTRAS
FEATS_B = FEATS_A + ENTRENO
EXCLUIDAS = {"forma_fisica", "dificultad", "tiempo_ajustado_s", "id", "distancia"}

def load_data(root):
    df = pd.read_parquet(Path(root) / "data" / "processed" / "df_long.parquet")
    assert df.drop(columns=["distancia"]).isnull().sum().sum() == 0
    v = df["distancia_km"] / (df["tiempo_ajustado_s"] / 3600)
    assert (v - df[TARGET]).abs().max() < 1e-9
    return df


def split_by_runner(d, val_size=0.15, test_size=0.15, seed=SEED):
    """Partición agrupada por corredor. Devuelve (train, val, test)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(gss1.split(d, groups=d["id"]))
    trval = d.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr_idx, val_idx = next(gss2.split(trval, groups=trval["id"]))
    train, val, test = trval.iloc[tr_idx], trval.iloc[val_idx], d.iloc[test_idx]
    ids = [set(x["id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    return train, val, test


def prepare(train, val, feats):
    """Escalado ajustado sólo con train. Devuelve (X_train, X_val) como arrays."""
    sc = StandardScaler().fit(train[feats])
    return sc.transform(train[feats]), sc.transform(val[feats])


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_optimizer(params, name, lr, weight_decay):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizador desconocido: {name}")


def train_mlp(Xtr, ytr, Xva, yva, seed,
              hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
              batch_size=64, optimizer="Adam", epochs=80, early_stopping=None):
    """Entrena una MLP con MSE.

    early_stopping=None: entrena `epochs` épocas y devuelve el modelo final.
    early_stopping={"patience": p, "max_epochs": m}: detiene si la pérdida de
    validación no mejora en p épocas (máximo m) y restaura los pesos de la
    mejor época.

    Devuelve (modelo, historia, mejor_época). La mejor época es la de menor
    pérdida de validación (índice base 0).
    """
    set_seed(seed)
    model = MLP(Xtr.shape[1], hidden, dropout).to(DEVICE)
    opt = make_optimizer(model.parameters(), optimizer, lr, weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=DEVICE)

    max_ep = early_stopping["max_epochs"] if early_stopping else epochs
    patience = early_stopping["patience"] if early_stopping else None
    hist = {"train": [], "val": []}
    best_loss, best_state, bad = np.inf, None, 0

    for ep in range(max_ep):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        hist["train"].append(tot / n)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        hist["val"].append(vl)

        if early_stopping:
            if vl < best_loss:
                best_loss, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

    if early_stopping:
        model.load_state_dict(best_state)
    best_epoch = int(np.argmin(hist["val"]))
    return model, hist, best_epoch


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def metrics(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
    }
