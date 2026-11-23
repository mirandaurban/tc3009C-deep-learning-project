"""Deterministic-seeded MLP fitting with optional validation early stopping."""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fit_mlp(Xtr, ytr, Xva, yva, seed=42, hidden=(64,32), dropout=.2,
            lr=1e-3, weight_decay=1e-4, batch_size=64, optimizer="Adam",
            epochs=80, early_stopping=None):
    if len(Xtr) == 0 or len(Xva) == 0: raise ValueError("Train y validación deben tener datos")
    set_seed(seed)
    model = MLP(Xtr.shape[1], hidden, dropout).to(DEVICE)
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer not in optimizers: raise ValueError(f"Optimizador desconocido: {optimizer}")
    opt = optimizers[optimizer](model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr,dtype=torch.float32), torch.tensor(ytr,dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    xv=torch.tensor(Xva,dtype=torch.float32,device=DEVICE); yv=torch.tensor(yva,dtype=torch.float32,device=DEVICE)
    maximum = early_stopping["max_epochs"] if early_stopping else epochs
    patience = early_stopping["patience"] if early_stopping else None
    history={"train":[],"val":[]}; best=np.inf; best_state=None; bad=0
    for _ in range(maximum):
        model.train(); total=0.; count=0
        for xb,yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad()
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            total += loss.item()*len(xb); count += len(xb)
        history["train"].append(total/count)
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(xv),yv).item()
        history["val"].append(val_loss)
        if early_stopping:
            if val_loss < best:
                best=val_loss; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                bad+=1
                if bad >= patience: break
    if early_stopping: model.load_state_dict(best_state)
    return model, history, int(np.argmin(history["val"]))

def predict(model, X):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(X,dtype=torch.float32,device=DEVICE)).cpu().numpy()
