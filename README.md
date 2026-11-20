# Predicting GitHub activity (Big Data Assignment 2)

For each test user, the code predicts which of six activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they do in the test week, 2026-11-22
00:00 to 11-29 00:00 Taiwan time (2026-11-21 16:00 to 11-28 16:00 UTC). For now it predicts
that each user repeats what they did in the last 7 days.

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
```

`predict.py` writes the submission `output_315552801.csv` for the users in `test_input.csv`
and checks the file with the course's `evaluate.py`.

## Method

The test users are the cohort users with at least 3 activities in the 90 days before
2026-11-04 12:00 UTC, counting only rows visible at that time. This rule gives exactly the
4,465 users of `test_input.csv`.

Every activity row has two times, when it happened and its row time, which is when it
became visible in the data. Anything known at a cutoff t only uses rows whose row time is
before t. I never read the late-known columns such as `merged_at` and `closed_at`.

## Tests

`pytest` checks that the user rule gives exactly the users of `test_input.csv`, that the
baseline uses nothing after the cutoff, and that my score matches `evaluate.py`.
