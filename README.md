# Predicting GitHub activity (Big Data Assignment 2)

For each test user, the code predicts which of six activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they do in the test week, 2026-11-22
00:00 to 11-29 00:00 Taiwan time (2026-11-21 16:00 to 11-28 16:00 UTC). I compare LightGBM on
per-user features against copying last week.

## Data Source

All data comes from [GH Archive](https://www.gharchive.org/), the public archive of GitHub
events, turned into 8 tables by the course's `preprocess.py`. I used every GH Archive hour
file from `2025-12-01-0` to `2026-11-14-3` (UTC), 8,356 files, and none of them was missing.

## Setup

The course repo, the GH Archive files and the built tables sit next to this repo.

```
some_folder/
    assignment2/          course repo (scripts/, submission/)
    gharchive/            GH Archive hour files
    data/db/              the 8 tables built by preprocess.py
    BigDataAssignment2/   this repo
```

```bash
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
mkdir gharchive
cd gharchive
wget https://data.gharchive.org/2025-12-{01..31}-{0..23}.json.gz
wget https://data.gharchive.org/2026-{01..10}-{01..31}-{0..23}.json.gz
wget https://data.gharchive.org/2026-11-{01..13}-{0..23}.json.gz
wget https://data.gharchive.org/2026-11-14-{0..3}.json.gz
cd ..
python assignment2/scripts/preprocess.py --raw gharchive --out data
cd BigDataAssignment2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The second `wget` also asks for dates that do not exist, and those requests fail harmlessly.
The plain hour files are about 225 GB, which did not fit on my disk. So I filtered each hour
file down to the lines `preprocess.py` keeps, with a small script outside this repo. On 4
hours I checked that the filtered and the plain files give the same tables.

The code runs on Python 3.14 with the versions in `requirements.txt`.

## Run

```bash
python predict.py
python compare.py models
python compare.py features
python compare.py window
```

`predict.py` retrains from scratch with a fixed seed and writes the submission
`output_315552801.csv` for the users in `test_input.csv`. It then checks the file with the
course's `evaluate.py`. `compare.py <experiment>` runs one of the experiments below and saves
its table in `results/`. Without an argument it lists them.

## Method

The test users are the cohort users with at least 3 activities in the 90 days before
2026-11-04 12:00 UTC, counting only rows visible at that time. This rule gives exactly the
4,465 users of `test_input.csv`.

For training I build the same task for earlier weeks. A training row is a cutoff t and a
user, with a cutoff every Saturday 16:00 UTC from March 2026 on, the same weekday and hour
as the test cutoff. Users are picked with the same rule 17 days and 4 hours before t, which
is the gap between 2026-11-04 12:00 and the test cutoff. The labels say whether the user did
each activity in the 7 days from t.

Every activity row has two times, when it happened and its row time, which is when it
became visible in the data. Features only use rows whose row time is before t. I never read
the late-known columns such as `merged_at` and `closed_at`.

Each user is described by 30 numbers, the counts of each activity over the last 1, 7, 28
and 90 days and the days since the last time, capped at 90. LightGBM fits one
gradient-boosted tree model per activity on these numbers.

To validate, I train on the weeks before 2026-10-03 and score the four weeks from cutoffs
2026-10-03, 10-10, 10-17 and 10-24. Each activity gets its own decision threshold, picked
from 0.10 to 0.60 in steps of 0.05, and each week is scored with thresholds picked on the
other three. Every model runs with 5 seeds.

GH Archive has recorded almost no push events since 2026-10-26. The hour 2026-10-28 12:00,
for example, contains no PushEvent at all. An activity that GH Archive is missing in the
test week is not scored, and the push labels of the latest weeks are broken anyway. So I
compare designs on the score of the other five activities, the same as `evaluate.py --skip
push`, and report the score over all six as well.

## Results

The scores are for the five activities without push, with the four validation weeks pooled,
as mean ± sd over the seeds (full tables with F1 per activity and each week in `results/`).
"Weeks better" counts the validation weeks in which a variant beats the first row of its
group.

| Experiment | Variant | Score | Weeks better |
|---|---|---|---|
| Models (`models.md`) | copy last week | 0.368 | |
| | LightGBM | 0.428 ± 0.003 | 4 of 4 |
| Feature windows (`features.md`, LightGBM) | last 90 days only | 0.385 ± 0.003 | |
| | last 7 days only | 0.380 ± 0.001 | 1 of 4 |
| | 1, 7, 28 and 90 days | 0.425 ± 0.002 | 4 of 4 |
| | and days since last (all 30) | 0.428 ± 0.003 | 4 of 4 |
| | all 30 except the 6 push features | 0.427 ± 0.002 | 4 of 4 |
| Training weeks (`window.md`, LightGBM) | all 30 weeks from March | 0.428 ± 0.003 | |
| | without June to August | 0.429 ± 0.000 | 2 of 4 |
| | first 13 weeks (March to May) | 0.423 ± 0.000 | 0 of 4 |
| | last 13 weeks (July to September) | 0.397 ± 0.000 | 0 of 4 |
| | last 4 weeks | 0.387 ± 0.000 | 0 of 4 |

Counting each activity over several windows adds about 0.04 over a single window.

The two 13-week windows have the same length, but March to May has 50,400 positive labels
over the five activities and July to September only 18,406, from more rows (81,200 vs
107,695). I think the number of positive examples matters more here than how recent the
weeks are.

## Submitted model

I submit LightGBM retrained on every complete week from March, with thresholds picked on the
four validation weeks.

## Tests

`pytest` checks that the user rule gives exactly the users of `test_input.csv`, that the
baseline and the user features use nothing after the cutoff, and that my score matches
`evaluate.py`.

## Code

| File | What it does |
|---|---|
| `ghactivity/data.py` | where the course files and tables are, loads the 8 tables |
| `ghactivity/task.py` | activity events, test-user rule, labels, weekly cutoffs, baseline |
| `ghactivity/features.py` | the 30 per-user features at a cutoff, and the training table |
| `ghactivity/metrics.py` | F1 and the official score |
| `ghactivity/lgbm.py` | the LightGBM models |
| `ghactivity/pipeline.py` | validation, thresholds and retraining |
| `predict.py` | writes and checks the submission file |
| `compare.py` | the experiments in the results |
