from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ghactivity.metrics import label_f1
from ghactivity.task import ACTIVITIES, USER_PICK_GAP, VAL_WEEKS, active_users, labels, weekly_cutoffs

THRESHOLDS = np.arange(0.1, 0.65, 0.05).round(2)


def variant(model, **options):
    return SimpleNamespace(fit=partial(model.fit, **options), predict=model.predict)


def validate(model, events, seed=0, cutoffs=None):
    if cutoffs is None:
        cutoffs = weekly_cutoffs(events, VAL_WEEKS[0])
    fitted = model.fit(events, cutoffs, seed)
    prob, truth = {}, {}
    for t in VAL_WEEKS:
        users = active_users(events, t - USER_PICK_GAP)
        prob[t] = model.predict(fitted, events, t, users)
        truth[t] = labels(events, t, users)
    return pd.concat(prob, names=["cutoff"]), pd.concat(truth, names=["cutoff"])


def best_thresholds(prob, truth):
    thresholds = {}
    for a in ACTIVITIES:
        sweep = {float(th): label_f1(prob[a] >= th, truth[a]) for th in THRESHOLDS}
        thresholds[a] = max(sweep, key=sweep.get)
    return thresholds


def to_labels(prob, thresholds):
    return pd.DataFrame({a: (prob[a] >= thresholds[a]).astype(int) for a in ACTIVITIES})


def held_out_labels(prob, truth):
    # each week gets the thresholds picked on the other weeks
    weeks = prob.index.get_level_values("cutoff")
    parts = []
    for t in weeks.unique():
        others = weeks != t
        parts.append(to_labels(prob[~others], best_thresholds(prob[others], truth[others])))
    return pd.concat(parts)


def train_for_week(model, events, t):
    thresholds = best_thresholds(*validate(model, events))
    return model.fit(events, weekly_cutoffs(events, t)), thresholds
