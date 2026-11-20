from functools import cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from ghactivity.data import DATA_DIR
from ghactivity.task import ACTIVITIES, USER_PICK_GAP, active_users, labels

HISTORY = pd.Timedelta(days=90)

# node type -> (table, primary key, when visible, when it happened, foreign keys)
# merged_by_id is left out, it is only known after the merge
ROWS = {
    "user": ("Users", "user_id", "first_seen_at", "first_seen_at", {}),
    "repo": ("Repositories", "repo_id", "first_seen_at", "first_seen_at", {}),
    "pr": ("PullRequests", "pr_id", "first_seen_at", "created_at",
           {"repo_id": "repo", "author_id": "user"}),
    "issue": ("Issues", "issue_id", "first_seen_at", "created_at",
              {"repo_id": "repo", "author_id": "user", "assignee_id": "user", "pull_request_id": "pr"}),
    "issue_comment": ("IssueComments", "comment_id", "created_at", "created_at",
                      {"issue_id": "issue", "repo_id": "repo", "author_id": "user"}),
    "review": ("PRReviews", "review_id", "first_seen_at", "submitted_at",
               {"pr_id": "pr", "repo_id": "repo", "reviewer_id": "user"}),
    "review_comment": ("PRReviewComments", "comment_id", "created_at", "created_at",
                       {"pr_id": "pr", "repo_id": "repo", "author_id": "user",
                        "pull_request_review_id": "review", "in_reply_to_id": "review_comment"}),
    "push": ("Pushes", "push_id", "created_at", "created_at", {"user_id": "user", "repo_id": "repo"}),
}

FLAGS = {"user": ["is_bot"], "pr": ["draft", "is_cross_repo"], "issue": ["is_pull_request", "locked"],
         "issue_comment": ["is_pull_request"]}
REVIEW_STATES = ["approved", "commented", "changes_requested"]  # not "dismissed", that happens later

HIDDEN = 64
EPOCHS = 15
LEARNING_RATE = 1e-3


@cache
def tables():
    extra = {"Issues": ["num_labels"], "PRReviews": ["state"], "PRReviewComments": ["subject_type"],
             "Users": ["github_user_id", "is_cohort"]}
    out = {}
    for kind, (table, pk, seen, happened, fks) in ROWS.items():
        columns = {pk, seen, happened, *fks, *FLAGS.get(kind, []), *extra.get(table, [])}
        out[kind] = pd.read_parquet(DATA_DIR / f"{table}.parquet", columns=sorted(columns))
    return out


def row_features(kind, rows):
    x = {"one": np.ones(len(rows))}
    for flag in FLAGS.get(kind, []):
        x[flag] = rows[flag].fillna(False).to_numpy(dtype=float)
    if kind == "issue":
        x["labels"] = np.log1p(rows["num_labels"].fillna(0).to_numpy(dtype=float))
    if kind == "review":
        for state in REVIEW_STATES:
            x[state] = (rows["state"] == state).to_numpy(dtype=float)
    if kind == "review_comment":
        x["line"] = (rows["subject_type"] == "line").to_numpy(dtype=float)
    return torch.tensor(np.stack(list(x.values()), axis=1), dtype=torch.float)


def build_graph(t, users):
    db = tables()
    cohort = db["user"][db["user"]["is_cohort"]]
    predict_ids = pd.Index(cohort["user_id"].to_numpy())[
        pd.Index(cohort["github_user_id"]).get_indexer(users)]

    graph, index, kept = HeteroData(), {}, {}
    for kind, (table, pk, seen, happened, fks) in ROWS.items():
        rows = db[kind]
        when = rows[happened].fillna(rows[seen])
        keep = rows[seen] < t
        if fks:
            keep &= when >= t - HISTORY
        if kind == "user":
            keep |= rows[pk].isin(predict_ids)
        rows = rows[keep]
        kept[kind] = rows
        index[kind] = pd.Index(rows[pk])
        graph[kind].x = row_features(kind, rows)
        graph[kind].days = torch.tensor(((t - when[keep]) / pd.Timedelta(days=1)).to_numpy(), dtype=torch.float)

    for kind, (_, _, _, _, fks) in ROWS.items():
        for column, target in fks.items():
            target_row = index[target].get_indexer(kept[kind][column].to_numpy(dtype=float, na_value=np.nan))
            linked = target_row >= 0
            edges = torch.tensor(np.stack([np.flatnonzero(linked), target_row[linked]]))
            graph[kind, column, target].edge_index = edges
            graph[target, f"{kind}_{column}", kind].edge_index = edges.flip(0)
    graph["user"].predict = torch.tensor(index["user"].get_indexer(predict_ids))
    return graph


class TimeEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", 1 / 10000 ** (torch.arange(0, HIDDEN, 2) / HIDDEN))
        self.lin = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, days):
        angle = days[:, None] * self.freq
        return self.lin(torch.cat([angle.sin(), angle.cos()], dim=1))


class Net(nn.Module):
    def __init__(self, graph, time):
        super().__init__()
        kinds = graph.node_types
        self.time = time
        self.embed = nn.ModuleDict({k: nn.Linear(graph[k].x.shape[1], HIDDEN) for k in kinds})
        self.encode_time = nn.ModuleDict({k: TimeEncoding() for k in kinds})
        self.layers = nn.ModuleList(
            HeteroConv({e: SAGEConv(HIDDEN, HIDDEN, aggr="sum") for e in graph.edge_types}, aggr="sum")
            for _ in range(2)
        )
        self.norms = nn.ModuleList(nn.ModuleDict({k: nn.LayerNorm(HIDDEN) for k in kinds}) for _ in range(2))
        self.out = nn.Linear(HIDDEN, len(ACTIVITIES))

    def forward(self, graph):
        h = {k: self.embed[k](graph[k].x) for k in graph.node_types}
        if self.time:
            h = {k: h[k] + self.encode_time[k](graph[k].days) for k in h}
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, graph.edge_index_dict)
            h = {k: F.relu(norm[k](v)) for k, v in h.items()}
        return self.out(h["user"][graph["user"].predict])


def fit(events, cutoffs, seed=0, time=True):
    torch.manual_seed(seed)
    weeks = []
    for t in cutoffs:
        users = active_users(events, t - USER_PICK_GAP)
        y = torch.tensor(labels(events, t, users).to_numpy(), dtype=torch.float)
        weeks.append((build_graph(t, users), y))

    net = Net(weeks[0][0], time)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(weeks)).tolist():
            graph, y = weeks[i]
            loss = F.binary_cross_entropy_with_logits(net(graph), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net


def predict(net, events, t, users):
    with torch.no_grad():
        prob = torch.sigmoid(net(build_graph(t, users)))
    return pd.DataFrame(prob.numpy(), index=pd.Index(users, name="github_user_id"), columns=ACTIVITIES)
