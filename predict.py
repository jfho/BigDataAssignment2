import argparse
import subprocess
import sys

import pandas as pd

from ghactivity.data import SCRIPTS_DIR, SUBMISSION_DIR, load_db
from ghactivity.task import ACTIVITIES, TEST_CUTOFF, activity_events, recent_activity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output_315552801.csv")
    args = parser.parse_args()

    test_input = SUBMISSION_DIR / "test_input.csv"
    users = pd.read_csv(test_input)["UserID"]
    events = activity_events(load_db())

    pred = recent_activity(events, TEST_CUTOFF, users)
    pred[ACTIVITIES].rename_axis("UserID").reset_index().to_csv(args.out, index=False)

    check = [sys.executable, SCRIPTS_DIR / "evaluate.py", "--pred", args.out, "--input", test_input]
    sys.exit(subprocess.run(check).returncode)


if __name__ == "__main__":
    main()
