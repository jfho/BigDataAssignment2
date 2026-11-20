import argparse
import subprocess
import sys

import pandas as pd

from ghactivity import gnn, lgbm
from ghactivity.data import SCRIPTS_DIR, SUBMISSION_DIR, load_db
from ghactivity.pipeline import to_labels, train_for_week
from ghactivity.task import ACTIVITIES, TEST_CUTOFF, activity_events

MODELS = {"lightgbm": lgbm, "gnn": gnn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output_315552801.csv")
    parser.add_argument("--model", choices=MODELS, default="lightgbm")
    args = parser.parse_args()

    test_input = SUBMISSION_DIR / "test_input.csv"
    users = pd.read_csv(test_input)["UserID"]
    model = MODELS[args.model]
    events = activity_events(load_db())
    fitted, thresholds = train_for_week(model, events, TEST_CUTOFF)
    print("thresholds:", thresholds, flush=True)

    pred = to_labels(model.predict(fitted, events, TEST_CUTOFF, users), thresholds)
    pred[ACTIVITIES].rename_axis("UserID").reset_index().to_csv(args.out, index=False)

    check = [sys.executable, SCRIPTS_DIR / "evaluate.py", "--pred", args.out, "--input", test_input]
    sys.exit(subprocess.run(check).returncode)


if __name__ == "__main__":
    main()
