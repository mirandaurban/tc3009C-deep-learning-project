"""Validation-only random hyperparameter sampling; does not execute automatically."""
import itertools
import numpy as np

SEARCH_SPACE = {"hidden": [(32,), (64,), (128,), (64,32), (128,64), (128,64,32), (256,128)],
 "lr": [1e-4,5e-4,1e-3,3e-3], "dropout": [0.,.1,.2,.3], "weight_decay": [0.,1e-4,1e-3],
 "batch_size": [32,64,128], "optimizer": ["Adam","AdamW"]}

def sample_configs(n=150, seed=42):
    keys=list(SEARCH_SPACE); space=list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    if not 1 <= n <= len(space): raise ValueError("n fuera del espacio de búsqueda")
    indices=np.random.default_rng(seed).choice(len(space),size=n,replace=False)
    return [dict(zip(keys,space[i])) for i in indices]
