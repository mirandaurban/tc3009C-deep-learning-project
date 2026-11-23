"""Train-only feature scaling."""
from sklearn.preprocessing import StandardScaler
from .features import validate_predictors

def fit_transform_pair(train, other, features):
    validate_predictors(features)
    scaler = StandardScaler().fit(train[features])
    return scaler.transform(train[features]), scaler.transform(other[features]), scaler
