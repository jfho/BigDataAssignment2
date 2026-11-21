import sys
from pathlib import Path

import pandas as pd

from ghactivity import gnn, lgbm, rdl
from ghactivity.data import load_db
from ghactivity.features import FEATURES, NO_PUSH
from ghactivity.metrics import WITHOUT_PUSH, score
from ghactivity.pipeline import best_thresholds, held_out_labels, to_labels, validate, variant
from ghactivity.task import (
    ACTIVITIES, FINAL_WEEK, USER_PICK_GAP, VAL_WEEKS, active_users, activity_events, labels,
    recent_activity, weekly_cutoffs,
)

SEEDS = range(5)
RDL_SEEDS = range(3)
ROWS = ACTIVITIES + ["score, all six", "macro", "micro", "score"]
RESULTS = Path(__file__).parent / "results"

# GH Archive has almost no pushes from 2026-10-26 on, 26 days before the test cutoff
OUTAGE = pd.Timedelta(days=26)

# the week of the reference scores in the course README
REFERENCE_WEEK = pd.Timestamp("2026-09-26 16:00")


def scores(pred, truth):
    return {**score(pred, truth, WITHOUT_PUSH), "score, all six": score(pred, truth)["score"]}


def summary(pred, truth):
    weeks = pred.index.get_level_values("cutoff").unique()
    weekly = {str(t.date()): score(pred.xs(t), truth.xs(t), WITHOUT_PUSH)["score"] for t in weeks}
    pooled = scores(pred, truth)
    return pd.Series({**{row: pooled[row] for row in ROWS}, **weekly})


def seeds(name, model, events, cutoffs=None, seed_range=SEEDS):
    runs = {}
    for seed in seed_range:
        prob, truth = validate(model, events, seed, cutoffs)
        runs[seed] = summary(held_out_labels(prob, truth), truth)
        print(f"{name} seed {seed}: score {runs[seed]['score']:.4f}", flush=True)
    return pd.DataFrame(runs)


def mean_sd(runs):
    if runs.shape[1] == 1:
        return runs.iloc[:, 0].map(lambda m: f"{m:.3f}")
    return runs.mean(axis=1).map(lambda m: f"{m:.3f}") + runs.std(axis=1).map(lambda s: f" ± {s:.3f}")


def table(columns, reference):
    weeks = [str(t.date()) for t in VAL_WEEKS]
    ref = columns[reference].mean(axis=1)
    out = pd.DataFrame({name: mean_sd(runs) for name, runs in columns.items()})
    wins = {name: f"{(runs.mean(axis=1)[weeks] > ref[weeks]).sum()} of {len(weeks)}" for name, runs in columns.items()}
    out.loc[f"weeks better than {reference}"] = pd.Series(wins)
    return out


def models(events):
    truth = pd.concat({t: labels(events, t, active_users(events, t - USER_PICK_GAP)) for t in VAL_WEEKS},
                      names=["cutoff"])
    baseline = pd.concat({t: recent_activity(events, t, truth.xs(t).index) for t in VAL_WEEKS}, names=["cutoff"])
    columns = {"Baseline": summary(baseline, truth).to_frame(),
               "LightGBM": seeds("LightGBM", lgbm, events), "GNN": seeds("GNN", gnn, events)}
    return table(columns, "Baseline")


def graph(events):
    edges = {"no edges": "none", "shuffled edges": "shuffled", "real edges": "real", "temporal edges": "temporal"}
    return table({name: seeds(name, variant(gnn, edges=e), events) for name, e in edges.items()}, "no edges")


FEATURE_SETS = {
    "last 7 days": [c for c in FEATURES if c.endswith("_7d")],
    "last 90 days": [c for c in FEATURES if c.endswith("_90d")],
    "1, 7, 28, 90 days": [c for c in FEATURES if not c.endswith("_days_since")],
    "+ days since last": FEATURES,
    "all but push": NO_PUSH,
}


def features(events):
    return table({name: seeds(name, variant(lgbm, features=cols), events) for name, cols in FEATURE_SETS.items()},
                 "last 90 days")


def training_windows(events):
    weeks = weekly_cutoffs(events, VAL_WEEKS[0])
    return {
        "all 30 weeks": weeks,
        "without Jun to Aug": [t for t in weeks if t.month not in (6, 7, 8)],
        "first 13 (Mar to May)": weeks[:13],
        "last 13 (Jul to Sep)": weeks[-13:],
        "last 4": weeks[-4:],
    }


def window(events):
    windows = training_windows(events)
    return table({name: seeds(name, lgbm, events, cutoffs) for name, cutoffs in windows.items()}, "all 30 weeks")


def with_outage(name, model, events):
    normal, blind = {}, {}
    for seed in SEEDS:
        fitted = model.fit(events, weekly_cutoffs(events, VAL_WEEKS[0]), seed)
        prob, prob_blind, truth = {}, {}, {}
        for t in VAL_WEEKS:
            users = active_users(events, t - USER_PICK_GAP)
            missing = (events["activity"] == "push") & (events["time"] >= t - OUTAGE)
            prob[t] = model.predict(fitted, events, t, users)
            prob_blind[t] = model.predict(fitted, events[~missing], t, users)
            truth[t] = labels(events, t, users)
        prob, prob_blind, truth = (pd.concat(d, names=["cutoff"]) for d in (prob, prob_blind, truth))
        normal[seed] = summary(held_out_labels(prob, truth), truth)
        blind[seed] = summary(held_out_labels(prob, truth, prob_blind), truth)
        print(f"{name} seed {seed}: {normal[seed]['score']:.4f} -> {blind[seed]['score']:.4f}", flush=True)
    return pd.DataFrame(normal), pd.DataFrame(blind)


def outage(events):
    columns = {}
    for name, model in [("LightGBM", lgbm), ("LightGBM, no push inputs", variant(lgbm, features=NO_PUSH)),
                        ("GNN", gnn)]:
        columns[name], columns[f"{name}, pushes missing"] = with_outage(name, model, events)
    return table(columns, "LightGBM")


MODELS = {"LightGBM, no push inputs": variant(lgbm, features=NO_PUSH),
          "LightGBM, all 30 (submitted)": lgbm, "GNN": gnn}


def one_week(events, t, val_weeks, models=MODELS, seed_range=SEEDS):
    users = active_users(events, t - USER_PICK_GAP)
    truth = labels(events, t, users)
    columns = {"Baseline": pd.Series(scores(recent_activity(events, t, users), truth)).to_frame()}
    for name, model in models.items():
        runs = {}
        for seed in seed_range:
            thresholds = best_thresholds(*validate(model, events, seed, weeks=val_weeks))
            fitted = model.fit(events, weekly_cutoffs(events, t), seed)
            runs[seed] = pd.Series(scores(to_labels(model.predict(fitted, events, t, users), thresholds), truth))
            print(f"{name} seed {seed}: score {runs[seed]['score']:.4f}", flush=True)
        columns[name] = pd.DataFrame(runs)
    return pd.DataFrame({name: mean_sd(runs.loc[ROWS]) for name, runs in columns.items()})


def final(events):
    return one_week(events, FINAL_WEEK, VAL_WEEKS)


def reference(events, models=MODELS, seed_range=SEEDS):
    week = pd.Timedelta(days=7)
    return one_week(events, REFERENCE_WEEK, [REFERENCE_WEEK - k * week for k in (4, 3, 2, 1)], models, seed_range)


def recency(events):
    names = ["last 90 days", "1, 7, 28, 90 days", "+ days since last"]
    return reference(events, {f"LightGBM, {n}": variant(lgbm, features=FEATURE_SETS[n]) for n in names})


RDL = {"RDL, no time": variant(rdl, time=False), "RDL, time": rdl}


def relational(events):
    columns = {"LightGBM": seeds("LightGBM", lgbm, events),
               **{name: seeds(name, model, events, seed_range=RDL_SEEDS) for name, model in RDL.items()}}
    return table(columns, "LightGBM")


def relational_reference(events):
    return reference(events, RDL, RDL_SEEDS)


def markdown(df):
    lines = ["| | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    lines += ["| " + " | ".join([str(row), *map(str, values)]) + " |" for row, values in df.iterrows()]
    return "\n".join(lines)


EXPERIMENTS = {"models": models, "graph": graph, "features": features, "window": window,
               "outage": outage, "final": final, "reference": reference, "recency": recency,
               "rdl": relational, "rdl_reference": relational_reference}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        sys.exit("usage: python compare.py " + "|".join(EXPERIMENTS))
    name = sys.argv[1]
    text = markdown(EXPERIMENTS[name](activity_events(load_db())))
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
