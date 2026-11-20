# Predicting GitHub activity (Big Data Assignment 2)

For each test user, the code predicts which of six activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they do in the test week, 2026-11-22
00:00 to 11-29 00:00 Taiwan time (2026-11-21 16:00 to 11-28 16:00 UTC). I compare LightGBM on
per-user features and two graph neural networks against copying last week, and submit
LightGBM.

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
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
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
```

`predict.py` retrains from scratch with a fixed seed and writes the submission
`output_315552801.csv` for the users in `test_input.csv`. It then checks the file with the
course's `evaluate.py`. With `--model gnn` it uses the GNN on users and repositories instead.
`compare.py <experiment>` runs one of the experiments below and saves its table in
`results/`. Without an argument it lists them.

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

The GNN on users and repositories builds one graph per cutoff from the last 90 days. User
nodes carry the same 30 numbers and repository nodes their activity counts over 7, 28 and
90 days. A user and a repository they worked in are linked by one edge per activity, in both
directions. In each of the two message-passing layers, a node's new vector is a linear map of
its own vector plus the mean of its neighbours' messages. A last layer gives one output per
activity. To test the edges I also run it with no edges and with every edge moved to a
random repository, where each node keeps its number of edges. A last variant has edges that
also carry how often and how recently the user worked there.

Since the course's reference GNN follows the RelBench examples, I also built a GNN directly
on the 8 tables (`rdl.py`) to compare with. Every row visible before t is a node, with activity
rows only from the last 90 days. Each foreign key is an edge in both directions, except the
late-known `merged_by_id`. The node inputs are the few columns that survive the masking.
Its time encoding adds to each row's vector how many days before t the row happened,
written as sine and cosine waves like in RelBench. It has two layers with sum aggregation
and the same training settings as the GNN on users and repositories, without any tuning.

To validate, I train on the weeks before 2026-10-03 and score the four weeks from cutoffs
2026-10-03, 10-10, 10-17 and 10-24. Each activity gets its own decision threshold, picked
from 0.10 to 0.60 in steps of 0.05, and each week is scored with thresholds picked on the
other three. Every model runs with 5 seeds, except the GNN on the 8 tables with 3, because
one fit takes about 25 minutes.

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
| | GNN on users and repositories | 0.422 ± 0.006 | 4 of 4 |
| Edges (`graph.md`) | GNN, no edges | 0.418 ± 0.005 | |
| | random edges | 0.420 ± 0.004 | 2 of 4 |
| | real edges | 0.422 ± 0.006 | 3 of 4 |
| | real edges with counts and recency | 0.422 ± 0.002 | 4 of 4 |
| GNN on the 8 tables (`rdl.md`, 3 seeds) | no time encoding | 0.384 ± 0.002 | |
| | with time encoding | 0.413 ± 0.006 | 4 of 4 |
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

Counting each activity over several windows adds about 0.04 over a single window. The real
edges add about 0.004 over no edges, which is within the seed sd, and random edges score
about the same.

The time encoding helps the GNN on the 8 tables in every validation week, and by 0.029 with
the four weeks pooled. Even with it, that GNN stays below LightGBM in all four weeks and below the GNN on
users and repositories. I think the reason is that LightGBM gets the time information
already counted per user, while the GNN on the 8 tables has to learn the counting from
single rows.

The two 13-week windows have the same length, but March to May has 50,400 positive labels
over the five activities and July to September only 18,406, from more rows (81,200 vs
107,695). I think the number of positive examples matters more here than how recent the
weeks are.

For the missing pushes I predict each validation week twice, once as usual and once with the
pushes of the 26 days before the cutoff removed (`outage.md`). Before the test week GH Archive
still has a few hundred pushes per week, so removing all of them is the harder case.

| Model | Pushes complete | Last 26 days of pushes removed |
|---|---|---|
| LightGBM | 0.428 ± 0.003 | 0.422 ± 0.002 |
| LightGBM without push inputs | 0.427 ± 0.002 | 0.427 ± 0.002 |
| GNN on users and repositories | 0.422 ± 0.006 | 0.414 ± 0.005 |

## Final week

The week from cutoff 2026-10-31 was kept aside until the design was fixed. Each model is
trained on every week before it with 5 seeds, and the inputs already contain the push outage
(`final.md`).

| | Copy last week | LightGBM without push inputs | LightGBM, all 30 (submitted) | GNN on users and repositories |
|---|---|---|---|---|
| Score | 0.345 | 0.410 ± 0.006 | 0.409 ± 0.002 | 0.400 ± 0.025 |

The order is the same as on the validation weeks, with all scores about 0.02 lower. On this
week the GNN's score moves between 0.358 and 0.422 over the 5 seeds. In the first three
seeds, the share of users it predicts as 1 changes a lot (open_pr between 3% and 14%, with 7%
in the labels), so the same thresholds give very different predictions. Removing pushes from
all its data does not change this, so the push outage is not the main cause.

## Reference week

The course README gives reference scores for the week 2026/09/27 to 10/04 Taiwan time, which
starts at the cutoff 2026-09-26 16:00 UTC. To compare, I trained each model on the weeks
before that cutoff, with thresholds picked on the four weeks before it. Then I scored all six
activities like the course does (`reference.md`, `rdl_reference.md`).

| Model | Course | Mine, all six | Mine, without push |
|---|---|---|---|
| Copy last week | 0.496 | 0.499 | 0.382 |
| GNN without time encoding | 0.357 | | |
| GNN with time encoding | 0.548 | | |
| GNN on the 8 tables, no time encoding (3 seeds) | | 0.506 ± 0.001 | 0.454 ± 0.001 |
| GNN on the 8 tables, with time encoding (3 seeds) | | 0.514 ± 0.012 | 0.423 ± 0.010 |
| GNN on users and repositories | | 0.504 ± 0.022 | 0.428 ± 0.009 |
| LightGBM, all 30 (submitted) | | 0.524 ± 0.001 | 0.438 ± 0.001 |
| LightGBM without push inputs | | 0.502 ± 0.001 | 0.450 ± 0.002 |

Copying last week gives 0.499 here and 0.496 in the course README, which suggests my data and
user rule are close to what the course used for that week. The best score is the course's
GNN with time encoding. The course README does not say how that GNN was trained or how its
thresholds were picked. Without time encoding my GNN on the 8 tables scores just above
copying last week, while the course's scores far below it.

Without push, the versions with less time information come out ahead on this week. LightGBM
with only the 90-day counts beats the full 30 features (0.462 vs 0.438, `recency.md`), and the
GNN on the 8 tables scores higher without its time encoding.

I looked into the LightGBM case with a separate script, one seed. The full features still
rank users better, with a higher average precision in every activity. With thresholds picked
on the reference week itself they win again, so I think the loss comes mostly from the
thresholds. The thresholds for this week are picked with a model trained on the weeks up to
2026-08-22, when activity was very low, and for every activity except push they land at
0.10, the lowest value of the grid. The final model is trained up to 2026-09-19 and has seen
the jump in activity in early September. With the old thresholds it predicts open_pr for 24%
of users, while 9% do it. The full features follow the jump more closely than the 90-day
counts (mean predicted open_pr 0.092 vs 0.073), so they lose more.

## Submitted model

I submit LightGBM with all 30 features, retrained on every complete week from March with
thresholds picked on the four validation weeks. Over all six activities it scores best on the
validation weeks, the final week and the reference week. On the five activities without push,
it and LightGBM without push inputs are within 0.012 of each other on every week. If GH Archive
still misses pushes in the test week, push is not scored, and removing the pushes from the
inputs cost LightGBM only 0.006 on the other five activities (`outage.md`). Running
`predict.py` twice gives the same file.

## Tests

`pytest` checks that the user rule gives exactly the users of `test_input.csv`, that the
baseline, the user features and the graphs of both GNNs use nothing after the cutoff, and
that my score matches `evaluate.py`.

## Code

| File | What it does |
|---|---|
| `ghactivity/data.py` | where the course files and tables are, loads the 8 tables |
| `ghactivity/task.py` | activity events, test-user rule, labels, weekly cutoffs, baseline |
| `ghactivity/features.py` | the 30 per-user features at a cutoff, and the training table |
| `ghactivity/metrics.py` | F1 and the official score |
| `ghactivity/lgbm.py` | the LightGBM models |
| `ghactivity/gnn.py` | the GNN on users and repositories |
| `ghactivity/rdl.py` | the GNN on the 8 tables, with every row a node |
| `ghactivity/pipeline.py` | validation, thresholds and retraining, shared by all models |
| `predict.py` | writes and checks the submission file |
| `compare.py` | the experiments in the results |
