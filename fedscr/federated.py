"""Clients, a small numpy MLP, federated averaging and the per group reliability scores."""
import time
from itertools import combinations

import numpy as np

from . import config as C
from .features import env_cols


def make_clients(tab, seed, conds=None, n_shards=None):
    """Clients are operating condition x Dirichlet label skew shard. Returns the client id of every window (-1: unused)."""
    conds = list(C.CONDS) if conds is None else conds
    n_shards = n_shards or C.N_SHARDS
    rng = np.random.default_rng(seed)
    props = {(c, k): rng.dirichlet([C.DIR_ALPHA] * n_shards) if n_shards > 1 else np.ones(1)
             for c in C.CONDS for k in range(len(C.CLASSES))}
    client = -np.ones(len(tab["cond"]), int)
    for r in np.unique(tab["rec"]):
        m = tab["rec"] == r
        c, y = tab["cond"][m][0], tab["cls"][m][0]
        if c not in conds:
            continue
        p = props[(c, y)]
        for b in ("train", "cal", "test"):
            ids = np.where(m & (tab["block"] == b))[0]
            ids = ids[np.argsort(tab["pos"][ids])]
            for sh, part in enumerate(np.split(ids, np.round(np.cumsum(p)[:-1] * len(ids)).astype(int))):
                client[part] = conds.index(c) * n_shards + sh
    return client


class MLP:
    """Two hidden layer ReLU network (hidden 64, embedding 32) trained with Adam."""

    def __init__(self, n_in, n_out, h=64, emb=32, kind="clf", seed=0):
        r = np.random.default_rng(seed)
        self.kind = kind
        self.P = []
        for a, b in zip([n_in, h, emb], [h, emb, n_out]):
            self.P += [(r.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float32), np.zeros(b, np.float32)]

    def copy(self):
        m = MLP.__new__(MLP)
        m.kind = self.kind
        m.P = [p.copy() for p in self.P]
        return m

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.P
        h1 = np.maximum(X @ W1 + b1, 0)
        h2 = np.maximum(h1 @ W2 + b2, 0)
        return h1, h2, h2 @ W3 + b3

    def __call__(self, X, emb=False):
        _, h2, o = self.forward(X)
        return (o, h2) if emb else o

    def grads(self, X, Y):
        _, _, W2, _, W3, _ = self.P
        h1, h2, o = self.forward(X)
        n = len(X)
        if self.kind == "clf":
            p = np.exp(o - o.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            d3 = p
            d3[np.arange(n), Y] -= 1
            d3 /= n
        else:
            d3 = 2 * (o - Y) / (n * o.shape[1])
        d2 = (d3 @ W3.T) * (h2 > 0)
        d1 = (d2 @ W2.T) * (h1 > 0)
        return [X.T @ d1, d1.sum(0), h1.T @ d2, d2.sum(0), h2.T @ d3, d3.sum(0)]

    def train(self, X, Y, epochs, rng):
        m = [np.zeros_like(p) for p in self.P]
        v = [np.zeros_like(p) for p in self.P]
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for i in range(0, len(X), C.BATCH):
                b = perm[i:i + C.BATCH]
                t += 1
                for j, gr in enumerate(self.grads(X[b], Y[b])):
                    m[j] = 0.9 * m[j] + 0.1 * gr
                    v[j] = 0.999 * v[j] + 0.001 * gr * gr
                    self.P[j] -= C.LR * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)


def fedavg(make, data, rounds, seed=0):
    rng = np.random.default_rng(seed)
    g = make()
    n = np.array([len(x) for x, _ in data], float)
    for _ in range(rounds):
        new = [np.zeros_like(p) for p in g.P]
        for (X, Y), nk in zip(data, n):
            if nk == 0:
                continue
            m = g.copy()
            m.train(X, Y, C.LOCAL_EPOCHS, rng)
            for j, p in enumerate(m.P):
                new[j] += (nk / n.sum()) * p
        g.P = new
    return g


def fed_stats(X_list):
    """Feature mean and standard deviation from client sums."""
    n = sum(len(x) for x in X_list)
    s1 = sum(x.sum(0) for x in X_list)
    s2 = sum((x.astype(np.float64) ** 2).sum(0) for x in X_list)
    mu = s1 / n
    return mu.astype(np.float32), np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-8)).astype(np.float32)


def softmax_log(o):
    o = o - o.max(1, keepdims=True)
    return o - np.log(np.exp(o).sum(1, keepdims=True))


class System:
    """For every evidence group: a FedAvg classifier head, FedAvg predictors of its features from every non empty
    subset of the other groups, and class means and pooled covariance of the head embeddings."""

    def __init__(self, tab, client, train_clients, seed, drop_env=False):
        t0 = time.time()
        tr = tab["block"] == "train"
        self.cols = {g: (np.setdiff1d(np.arange(tab["F"][g].shape[1]), env_cols(tab["F"][g].shape[1] // (C.NB + 5 + C.NE)))
                         if drop_env and C.GTYPE[g] == "dyn" else np.arange(tab["F"][g].shape[1])) for g in C.GROUPS}
        self.norm = {g: fed_stats([tab["F"][g][tr & (client == k)] for k in train_clients]) for g in C.GROUPS}
        Z = {g: self.z(g, tab["F"][g]) for g in C.GROUPS}
        y = tab["cls"]
        self.head, self.pred, self.subsets, self.partners = {}, {}, {}, {}
        for g in C.EVID:
            data = [(Z[g][tr & (client == k)], y[tr & (client == k)]) for k in train_clients]
            self.head[g] = fedavg(lambda g=g: MLP(Z[g].shape[1], len(C.CLASSES), seed=seed), data, C.FL_ROUNDS, seed)
            P = [j for j in C.GROUPS if j != g]
            self.partners[g] = P
            self.subsets[g] = [S for r in range(1, len(P) + 1) for S in combinations(P, r)]
            for S in self.subsets[g]:
                XS = np.concatenate([Z[j] for j in S], 1)
                data = [(XS[tr & (client == k)], Z[g][tr & (client == k)]) for k in train_clients]
                self.pred[(g, S)] = fedavg(lambda d=XS.shape[1], g=g: MLP(d, Z[g].shape[1], kind="reg", seed=seed),
                                           data, C.PRED_ROUNDS, seed)
        self.typ = {}
        for g in C.EVID:
            E = self.head[g](Z[g][tr], emb=True)[1]
            yy, cc = y[tr], client[tr]
            inside = np.isin(cc, train_clients)
            mus = np.stack([E[inside & (yy == c)].mean(0) if np.any(inside & (yy == c)) else np.zeros(E.shape[1])
                            for c in range(len(C.CLASSES))])
            R = (E - mus[yy])[inside]
            S_ = R.T @ R / max(len(R) - len(C.CLASSES), 1)
            S_ += 0.1 * np.trace(S_) / S_.shape[0] * np.eye(S_.shape[0])
            self.typ[g] = (mus, np.linalg.inv(S_))
        self.train_time = time.time() - t0

    def z(self, g, X):
        mu, sd = self.norm[g] if hasattr(self, "norm") else (0, 1)
        return (((X - mu) / sd)[:, self.cols[g]]).astype(np.float32)

    def evaluate(self, feats):
        """Class log probabilities, cross sensor residuals of every subset, typicality scores and embeddings."""
        Z = {g: self.z(g, feats[g]) for g in C.GROUPS}
        logp, R, typ, emb = [], [], [], []
        for g in C.EVID:
            o, E = self.head[g](Z[g], emb=True)
            logp.append(softmax_log(o))
            emb.append(E)
            R.append(np.stack([((self.pred[(g, S)](np.concatenate([Z[j] for j in S], 1)) - Z[g]) ** 2).mean(1)
                               for S in self.subsets[g]], 1))
            mus, Pm = self.typ[g]
            typ.append(np.log(np.stack([np.einsum("nd,de,ne->n", E - m, Pm, E - m) for m in mus], 1).min(1) + 1e-6))
        return dict(logp=np.stack(logp), R=R, typ=np.stack(typ), emb=np.concatenate(emb, 1),
                    zpart=np.concatenate([Z[g] for g in C.PARTNER], 1) if C.PARTNER else None)


def scores_for(sysm, tab, rows, variant=None):
    """Scores of the given windows; variant lists (group, operator, severity) degradations to apply."""
    feats = {g: tab["F"][g][rows] for g in C.GROUPS}
    if variant:
        pos = np.searchsorted(tab["D_rows"], rows)
        assert np.all(tab["D_rows"][pos] == rows)
        for (g, kind, s) in variant:
            feats[g] = tab["D"][(g, kind, s)][pos]
    return sysm.evaluate(feats)


def sc_take(sc, idx):
    return dict(logp=sc["logp"][:, idx], R=[r[idx] for r in sc["R"]], typ=sc["typ"][:, idx], emb=sc["emb"][idx],
                zpart=None if sc["zpart"] is None else sc["zpart"][idx])


def sc_set(sc, i, other):
    sc["logp"][:, i] = other["logp"][:, 0]
    sc["typ"][:, i] = other["typ"][:, 0]
    for g in range(len(sc["R"])):
        sc["R"][g][i] = other["R"][g][0]


_SYSTEMS = {}


def get_system(tab, key, client, train_clients, seed, drop_env=False):
    if key not in _SYSTEMS:
        _SYSTEMS[key] = System(tab, client, train_clients, seed, drop_env)
        print(f"  trained system {key} in {_SYSTEMS[key].train_time:.0f}s")
    return _SYSTEMS[key]


def trained_systems():
    return list(_SYSTEMS.values())
