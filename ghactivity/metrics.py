from ghactivity.task import ACTIVITIES


def f1(tp, fp, fn):
    # evaluate.py scores an activity with no true and no predicted 1s as 1.0
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def label_f1(pred, truth):
    tp = ((pred == 1) & (truth == 1)).sum()
    fp = ((pred == 1) & (truth == 0)).sum()
    fn = ((pred == 0) & (truth == 1)).sum()
    return f1(tp, fp, fn)


def score(pred, truth, activities=ACTIVITIES):
    pred = pred.loc[truth.index, ACTIVITIES]
    truth = truth[ACTIVITIES]
    tp = ((pred == 1) & (truth == 1)).sum()
    fp = ((pred == 1) & (truth == 0)).sum()
    fn = ((pred == 0) & (truth == 1)).sum()

    per_activity = {a: float(f1(tp[a], fp[a], fn[a])) for a in ACTIVITIES}
    macro = sum(per_activity[a] for a in activities) / len(activities)
    micro = float(f1(tp[activities].sum(), fp[activities].sum(), fn[activities].sum()))
    return {"score": (macro + micro) / 2, "macro": macro, "micro": micro, **per_activity}
