import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ghactivity import lgbm
from ghactivity.data import load_db
from ghactivity.metrics import label_f1
from ghactivity.pipeline import best_thresholds, validate
from ghactivity.task import (
    ACTIVITIES, FINAL_WEEK, LABEL_WINDOW, TEST_CUTOFF, USER_PICK_GAP, VAL_WEEKS, active_users,
    activity_events, labels, weekly_cutoffs,
)

FIGURES = Path(__file__).parent / "figures"
COLORS = dict(zip(ACTIVITIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def mark_weeks(ax):
    for i, t in enumerate(VAL_WEEKS):
        ax.axvspan(t, t + LABEL_WINDOW, color="#9ec3ef", lw=0, label="validation" if i == 0 else None)
    ax.axvspan(FINAL_WEEK, FINAL_WEEK + LABEL_WINDOW, color="#f6c19b", lw=0, label="final week")
    ax.axvspan(TEST_CUTOFF, TEST_CUTOFF + LABEL_WINDOW, color="#cfcdc6", lw=0, label="test week")


def activity_per_week(events):
    weekly = (events.assign(week=events["time"].dt.to_period("W-FRI").dt.start_time)
              .query("week >= '2025-12-01' and week < '2026-11-07'")
              .groupby(["week", "activity"]).size().unstack(fill_value=0)[ACTIVITIES])
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for ax, a in zip(axes.flat, ACTIVITIES):
        mark_weeks(ax)
        ax.plot(weekly.index, weekly[a], color=COLORS[a], lw=2)
        ax.set_title(a, loc="left", color=INK)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=45)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False)
    fig.suptitle("Cohort activity per week (events by when they happened)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "activity_per_week.png")


def label_rate(events):
    rates = {t: labels(events, t, active_users(events, t - USER_PICK_GAP)).mean()
             for t in weekly_cutoffs(events, FINAL_WEEK + LABEL_WINDOW)}
    rates = pd.DataFrame(rates).T
    fig, ax = plt.subplots(figsize=(9, 4))
    mark_weeks(ax)
    for a in ACTIVITIES:
        ax.plot(rates.index, rates[a], color=COLORS[a], lw=2, label=a)
    ax.set_xlim(right=TEST_CUTOFF + LABEL_WINDOW + pd.Timedelta(days=3))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_ylabel("share of users (label = 1)")
    ax.set_title("How often each label is 1, per week (cutoff on the x-axis)", loc="left", color=INK)
    fig.savefig(FIGURES / "label_rate.png")


def late_arrival(events):
    events = events[events["time"] >= "2026-03-01"]
    late = (events["seen"] - events["time"]) > pd.Timedelta(days=1)
    share = late.groupby(events["activity"]).mean().reindex(ACTIVITIES)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(share.index, share * 100, color=[COLORS[a] for a in share.index], height=0.6)
    for y, v in enumerate(share * 100):
        ax.annotate(f"{v:.1f}%", (v, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xlabel("% of rows visible more than 1 day after they happened (since March)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Late arrival per activity", loc="left", color=INK)
    fig.savefig(FIGURES / "late_arrival.png")


def thresholds(events):
    prob, truth = validate(lgbm, events)
    chosen = best_thresholds(prob, truth)
    grid = np.arange(0.02, 0.81, 0.02)
    fig, ax = plt.subplots(figsize=(8, 4))
    for a in ACTIVITIES:
        f1 = [label_f1(prob[a] >= th, truth[a]) for th in grid]
        ax.plot(grid, f1, color=COLORS[a], lw=2, label=a)
        ax.plot(chosen[a], label_f1(prob[a] >= chosen[a], truth[a]), "o", ms=7, color=COLORS[a], mec="white", mew=1.5)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("decision threshold (predict 1 if probability >= threshold)")
    ax.set_ylabel("F1, four validation weeks")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("F1 against the threshold, submitted LightGBM (dots are the thresholds picked)", loc="left", color=INK)
    fig.savefig(FIGURES / "thresholds.png")


def importance(events):
    models = lgbm.fit(events, weekly_cutoffs(events, VAL_WEEKS[0]))
    windows = {"last day": "_1d", "last 7 days": "_7d", "last 28 days": "_28d", "last 90 days": "_90d",
               "days since last": "_days_since"}
    share = {}
    for a, m in models.items():
        gain = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        share[a] = {w: gain[gain.index.str.endswith(end)].sum() / gain.sum() for w, end in windows.items()}
    share = pd.DataFrame(share).T.loc[ACTIVITIES[::-1]] * 100
    colors = ["#c6dcf5", "#8fbbea", "#4f93de", "#2a6fc0", "#1b4a85"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    left = np.zeros(len(share))
    for w, color in zip(windows, colors):
        ax.barh(share.index, share[w], left=left, color=color, height=0.6, label=w, edgecolor="white", linewidth=2)
        left += share[w]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the trees' total gain")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("Share of LightGBM's split gain per time window", loc="left", color=INK)
    fig.savefig(FIGURES / "importance.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    events = activity_events(load_db())
    activity_per_week(events)
    label_rate(events)
    late_arrival(events)
    thresholds(events)
    importance(events)
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", "-Gdpi=200", FIGURES / "schema.dot", "-o", FIGURES / "schema.png"], check=True)
    else:
        print("Graphviz not installed, schema.png skipped")


if __name__ == "__main__":
    main()
