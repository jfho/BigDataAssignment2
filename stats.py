from pathlib import Path

import pandas as pd

from compare import training_windows
from ghactivity import lgbm
from ghactivity.data import load_db
from ghactivity.features import training_table, user_features
from ghactivity.gnn import HISTORY
from ghactivity.metrics import WITHOUT_PUSH, label_f1
from ghactivity.pipeline import best_thresholds, validate
from ghactivity.task import (
    ACTIVITIES, FINAL_WEEK, LABEL_WINDOW, USER_PICK_GAP, VAL_WEEKS, active_users, activity_events, labels, visible,
)

RESULTS = Path(__file__).parent / "results"
EXAMPLE_USER = 104638223706207


def recent(events, t):
    past = visible(events, t)
    return past[past["time"] >= t - HISTORY]


def graph_size(events, t):
    past = recent(events, t)
    edges = past[["github_user_id", "repo_id", "activity"]].drop_duplicates()
    return f"{t.date()}: {len(past):,} rows become {len(edges):,} user-repo edges"


def overlap(events, t):
    users = active_users(events, t - USER_PICK_GAP)
    pairs = recent(events, t)[["github_user_id", "repo_id"]].drop_duplicates()
    users_per_repo = pairs.groupby("repo_id").size()
    shared = pairs[pairs["repo_id"].map(users_per_repo) > 1]["github_user_id"]
    in_graph = pairs["github_user_id"][pairs["github_user_id"].isin(users)].unique()
    alone = 1 - pd.Index(in_graph).isin(shared).mean()

    known = set(zip(pairs["github_user_id"], pairs["repo_id"]))
    future = events[(events["time"] >= t) & (events["time"] < t + LABEL_WINDOW) & events["github_user_id"].isin(users)]
    future = future.assign(known=[pair in known for pair in zip(future["github_user_id"], future["repo_id"])])
    positives = future.groupby(["github_user_id", "activity"])["known"].any()
    return (f"{t.date()}: repos with one cohort user {(users_per_repo == 1).mean():.1%}, "
            f"users sharing no repo {alone:.1%}, positive labels in a known repo {positives.mean():.1%}")


def window_sizes(events):
    lines = []
    for name, cutoffs in training_windows(events).items():
        _, y = training_table(events, cutoffs)
        lines.append(f"{name}: {len(cutoffs)} weeks, {len(y):,} examples, {int(y.to_numpy().sum()):,} positive "
                     f"labels, {int(y[WITHOUT_PUSH].to_numpy().sum()):,} without push")
    return lines


def threshold_table(events):
    prob, truth = validate(lgbm, events)
    chosen = best_thresholds(prob, truth)
    return [f"{a}: threshold {chosen[a]:.2f}, F1 at 0.5 {label_f1(prob[a] >= 0.5, truth[a]):.3f}, "
            f"F1 at threshold {label_f1(prob[a] >= chosen[a], truth[a]):.3f}" for a in ACTIVITIES]


def main():
    events = activity_events(load_db())
    october = VAL_WEEKS[-1]
    hours = events[events["activity"] == "push"].set_index("time").resample("1h").size()["2026-10-26":]
    lines = [
        "Rows and edges in the last 90 days", graph_size(events, october), "",
        "Cohort users per repository, last 90 days", *[overlap(events, t) for t in VAL_WEEKS], "",
        f"User {EXAMPLE_USER} at {october.date()}",
        user_features(events, october, pd.Index([EXAMPLE_USER])).T.round(1).to_string(), "",
        f"Hours without pushes since 2026-10-26: {(hours == 0).sum()} of {len(hours)}", "",
        "Users who push in the next week",
        *[f"{t.date()}: {labels(events, t, active_users(events, t - USER_PICK_GAP))['push'].mean():.1%}"
          for t in [*VAL_WEEKS, FINAL_WEEK]], "",
        "Training weeks", *window_sizes(events), "",
        "LightGBM thresholds on the validation weeks", *threshold_table(events),
    ]
    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stats.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
