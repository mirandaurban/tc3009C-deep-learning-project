"""Safe reusable data split and feature preparation primitives."""
from sklearn.model_selection import GroupShuffleSplit
from .features import GROUP, TARGET, get_features
from .preprocessing import fit_transform_pair

def split_by_runner(frame, val_size=.15, test_size=.15, seed=42):
    outer=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed)
    trainval_idx,test_idx=next(outer.split(frame,groups=frame[GROUP]))
    trainval=frame.iloc[trainval_idx]
    inner=GroupShuffleSplit(n_splits=1,test_size=val_size/(1-test_size),random_state=seed)
    train_idx,val_idx=next(inner.split(trainval,groups=trainval[GROUP]))
    train,val,test=trainval.iloc[train_idx],trainval.iloc[val_idx],frame.iloc[test_idx]
    ids=[set(part[GROUP]) for part in (train,val,test)]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError("Corredores compartidos entre particiones")
    return train,val,test

def prepare_pair(train, other, variant):
    return fit_transform_pair(train,other,get_features(variant))
