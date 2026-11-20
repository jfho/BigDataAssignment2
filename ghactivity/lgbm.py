import lightgbm as lgb
import pandas as pd

from ghactivity.features import training_table, user_features
from ghactivity.task import ACTIVITIES


def fit(events, cutoffs, seed=0, features=None):
    X, y = training_table(events, cutoffs)
    if features is not None:
        X = X[features]
    params = {"objective": "binary", "verbose": -1, "seed": seed}
    return {a: lgb.train(params, lgb.Dataset(X, y[a])) for a in ACTIVITIES}


def predict(models, events, t, users):
    X = user_features(events, t, users)
    return pd.DataFrame({a: m.predict(X[m.feature_name()]) for a, m in models.items()}, index=X.index)
