import pandas as pd

# activity -> (table, user column, when it happened, when it became visible)
SOURCES = {
    "open_pr": ("PullRequests", "author_id", "created_at", "first_seen_at"),
    "open_issue": ("Issues", "author_id", "created_at", "first_seen_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at", "first_seen_at"),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at", "created_at"),
    "push": ("Pushes", "user_id", "created_at", "created_at"),
}
ACTIVITIES = list(SOURCES)

LABEL_WINDOW = pd.Timedelta(days=7)
TEST_CUTOFF = pd.Timestamp("2026-11-21 16:00")  # 2026-11-22 00:00 in Taiwan

# active_users at this time gives exactly the users in test_input.csv
USER_PICK_TIME = pd.Timestamp("2026-11-04 12:00")
USER_PICK_GAP = TEST_CUTOFF - USER_PICK_TIME

# the data starts on 2025-12-01 and features look back 90 days
FIRST_CUTOFF = pd.Timestamp("2026-03-01")

VAL_WEEKS = pd.to_datetime(["2026-10-03 16:00", "2026-10-10 16:00", "2026-10-17 16:00", "2026-10-24 16:00"])
FINAL_WEEK = pd.Timestamp("2026-10-31 16:00")


def activity_events(db):
    users = db["Users"]
    cohort = users[users["is_cohort"]]
    to_github = dict(zip(cohort["user_id"], cohort["github_user_id"]))

    parts = []
    for activity, (table, who, happened, seen) in SOURCES.items():
        df = db[table]
        if table == "Issues":
            df = df[~df["is_pull_request"]]  # PRs are issues too on GitHub
        df = df[df[who].isin(to_github)]
        parts.append(pd.DataFrame({
            "github_user_id": df[who].map(to_github),
            "repo_id": df["repo_id"],
            "activity": activity,
            "time": df[happened],
            "seen": df[seen],
        }))
    events = pd.concat(parts).dropna(subset=["time"])
    return events.sort_values("time", ignore_index=True)


def visible(events, t):
    return events[events["seen"] < t]


def count_activity(events, start, end, ids, by="github_user_id"):
    window = events[(events["time"] >= start) & (events["time"] < end)]
    counts = window.groupby([by, "activity"]).size().unstack(fill_value=0)
    return counts.reindex(index=pd.Index(ids, name=by), columns=ACTIVITIES, fill_value=0)


def did_activity(events, start, end, users):
    return (count_activity(events, start, end, users) > 0).astype(int)


def labels(events, t, users):
    return did_activity(events, t, t + LABEL_WINDOW, users)


def active_users(events, t, min_records=3, days=90):
    history = visible(events, t)
    history = history[history["time"] >= t - pd.Timedelta(days=days)]
    counts = history.groupby("github_user_id").size()
    return counts[counts >= min_records].index


def recent_activity(events, t, users, days=7):
    return did_activity(visible(events, t), t - pd.Timedelta(days=days), t, users)


def weekly_cutoffs(events, end):
    last_label_end = min(end, events["seen"].max())
    weeks = [TEST_CUTOFF - k * LABEL_WINDOW for k in range((TEST_CUTOFF - FIRST_CUTOFF) // LABEL_WINDOW, 0, -1)]
    return [t for t in weeks if t + LABEL_WINDOW <= last_label_end]
