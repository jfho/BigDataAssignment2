import pandas as pd

from ghactivity.task import ACTIVITIES, USER_PICK_GAP, active_users, count_activity, labels, visible

MAX_DAYS = 90
WINDOWS = [1, 7, 28, 90]

FEATURES = [f"{a}_{d}d" for d in WINDOWS for a in ACTIVITIES] + [f"{a}_days_since" for a in ACTIVITIES]
NO_PUSH = [c for c in FEATURES if not c.startswith("push")]


def days_since_last(past, t, users):
    last = past.groupby(["github_user_id", "activity"])["time"].max().unstack()
    last = last.reindex(index=pd.Index(users, name="github_user_id"), columns=ACTIVITIES)
    days = (t - last) / pd.Timedelta(days=1)
    return days.fillna(MAX_DAYS).clip(upper=MAX_DAYS)


def user_features(events, t, users):
    past = visible(events, t)
    tables = [
        count_activity(past, t - pd.Timedelta(days=days), t, users).add_suffix(f"_{days}d")
        for days in WINDOWS
    ]
    tables.append(days_since_last(past, t, users).add_suffix("_days_since"))
    return pd.concat(tables, axis=1)


def training_table(events, cutoffs):
    X_parts, y_parts = [], []
    for t in cutoffs:
        users = active_users(events, t - USER_PICK_GAP)
        X_parts.append(user_features(events, t, users))
        y_parts.append(labels(events, t, users))
    X = pd.concat(X_parts, keys=cutoffs, names=["cutoff"])
    y = pd.concat(y_parts, keys=cutoffs, names=["cutoff"])
    return X, y
