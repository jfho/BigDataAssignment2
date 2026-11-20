import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing

from ghactivity.features import MAX_DAYS, user_features
from ghactivity.task import ACTIVITIES, USER_PICK_GAP, active_users, count_activity, labels, visible

HISTORY = pd.Timedelta(days=90)
REPO_WINDOWS = [7, 28, 90]
EDGES = [("user", a, "repo") for a in ACTIVITIES] + [("repo", f"{a}_by", "user") for a in ACTIVITIES]

HIDDEN = 64
EPOCHS = 15
LEARNING_RATE = 1e-3


def as_tensor(table):
    return torch.tensor(np.log1p(table.to_numpy(dtype=float)), dtype=torch.float)


def edge_table(past, t, activity):
    rows = past[past["activity"] == activity]
    rows = rows.assign(days_ago=(t - rows["time"]) / pd.Timedelta(days=1))
    for d in REPO_WINDOWS:
        rows[f"{d}d"] = rows["days_ago"] < d
    pairs = rows.groupby(["github_user_id", "repo_id"]).agg(
        **{f"{d}d": (f"{d}d", "sum") for d in REPO_WINDOWS}, days_since=("days_ago", "min"))
    pairs["days_since"] = pairs["days_since"].clip(upper=MAX_DAYS)
    return pairs.reset_index()


def build_graph(events, t, users, edges="real"):
    past = visible(events, t)
    past = past[past["time"] >= t - HISTORY]
    user_ids = pd.Index(past["github_user_id"].unique()).union(users)
    repo_ids = pd.Index(past["repo_id"].unique())

    graph = HeteroData()
    graph["user"].x = as_tensor(user_features(events, t, user_ids))
    graph["user"].predict = torch.tensor(user_ids.get_indexer(users))
    graph["repo"].x = as_tensor(pd.concat(
        [count_activity(past, t - pd.Timedelta(days=d), t, repo_ids, by="repo_id") for d in REPO_WINDOWS],
        axis=1,
    ))
    for a in ACTIVITIES:
        pairs = edge_table(past, t, a)
        index = torch.tensor(np.stack([
            user_ids.get_indexer(pairs["github_user_id"]),
            repo_ids.get_indexer(pairs["repo_id"]),
        ]))
        attr = torch.zeros(len(pairs), 0)
        if edges == "temporal":
            recency = torch.tensor(pairs[["days_since"]].to_numpy() / MAX_DAYS, dtype=torch.float)
            attr = torch.cat([as_tensor(pairs[["7d", "28d", "90d"]]), recency], dim=1)
        graph["user", a, "repo"].edge_index = index
        graph["repo", f"{a}_by", "user"].edge_index = index.flip(0)
        graph["user", a, "repo"].edge_attr = attr
        graph["repo", f"{a}_by", "user"].edge_attr = attr
    return graph


class Layer(MessagePassing):
    # GraphSAGE, with the edge attributes added to each message
    def __init__(self, edge_dim):
        super().__init__(aggr="mean")
        self.message_lin = nn.Linear(HIDDEN + edge_dim, HIDDEN)
        self.self_lin = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.self_lin(x[1])

    def message(self, x_j, edge_attr):
        return self.message_lin(torch.cat([x_j, edge_attr], dim=1))


class Net(nn.Module):
    def __init__(self, user_dim, repo_dim, edge_dim):
        super().__init__()
        self.embed = nn.ModuleDict({"user": nn.Linear(user_dim, HIDDEN), "repo": nn.Linear(repo_dim, HIDDEN)})
        self.layers = nn.ModuleList(
            HeteroConv({edge: Layer(edge_dim) for edge in EDGES}, aggr="sum") for _ in range(2)
        )
        self.out = nn.Linear(HIDDEN, len(ACTIVITIES))

    def forward(self, graph):
        h = {node: F.relu(self.embed[node](graph[node].x)) for node in self.embed}
        for layer in self.layers:
            update = layer(h, graph.edge_index_dict, graph.edge_attr_dict)
            h = {node: h[node] + F.relu(update[node]) for node in h}
        return self.out(h["user"])


def fit(events, cutoffs, seed=0, edges="real"):
    torch.manual_seed(seed)
    weeks = []
    for t in cutoffs:
        users = active_users(events, t - USER_PICK_GAP)
        y = torch.tensor(labels(events, t, users).to_numpy(), dtype=torch.float)
        weeks.append((build_graph(events, t, users, edges), y))

    first = weeks[0][0]
    net = Net(first["user"].x.shape[1], first["repo"].x.shape[1], first[EDGES[0]].edge_attr.shape[1])
    net.edges = edges
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(weeks)).tolist():
            graph, y = weeks[i]
            loss = F.binary_cross_entropy_with_logits(net(graph)[graph["user"].predict], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net


def predict(net, events, t, users):
    graph = build_graph(events, t, users, net.edges)
    with torch.no_grad():
        prob = torch.sigmoid(net(graph)[graph["user"].predict])
    return pd.DataFrame(prob.numpy(), index=pd.Index(users, name="github_user_id"), columns=ACTIVITIES)
