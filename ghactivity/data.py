import os
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
COURSE_DIR = Path(os.environ.get("A2_DIR", REPO_DIR.parent / "assignment2"))
SCRIPTS_DIR = COURSE_DIR / "scripts"
SUBMISSION_DIR = COURSE_DIR / "submission"
DATA_DIR = Path(os.environ.get("A2_DATA", REPO_DIR.parent / "data" / "db"))

TABLES = [
    "Users",
    "Repositories",
    "PullRequests",
    "Issues",
    "IssueComments",
    "PRReviews",
    "PRReviewComments",
    "Pushes",
]


def load_db():
    return {name: pd.read_parquet(DATA_DIR / f"{name}.parquet") for name in TABLES}
