import importlib.util

import pandas as pd
import pytest

from ghactivity.data import SCRIPTS_DIR, SUBMISSION_DIR, load_db
from ghactivity.features import FEATURES, user_features
from ghactivity.metrics import score
from ghactivity.task import (
    USER_PICK_TIME, VAL_WEEKS, active_users, activity_events, labels, recent_activity, visible,
)


@pytest.fixture(scope="module")
def events():
    return activity_events(load_db())


def test_rule_gives_the_test_users(events):
    test_users = pd.read_csv(SUBMISSION_DIR / "test_input.csv")["UserID"]
    assert set(active_users(events, USER_PICK_TIME)) == set(test_users)


def test_nothing_after_t_is_used(events):
    t = VAL_WEEKS[0]
    past = visible(events, t)
    users = active_users(events, t)
    assert active_users(past, t).equals(users)
    assert recent_activity(past, t, users).equals(recent_activity(events, t, users))
    assert user_features(past, t, users).equals(user_features(events, t, users))
    assert list(user_features(events, t, users).columns) == FEATURES


def test_same_score_as_evaluate_py(events):
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS_DIR / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    t = VAL_WEEKS[0]
    users = active_users(events, t)
    pred, truth = recent_activity(events, t, users), labels(events, t, users)
    theirs = official.scores(pred.rename_axis("UserID").reset_index(), truth.rename_axis("UserID").reset_index())
    ours = score(pred, truth)
    assert ours["score"] == pytest.approx(theirs["score"])
    assert ours["macro"] == pytest.approx(theirs["macro_f1"])
    assert ours["micro"] == pytest.approx(theirs["micro_f1"])
