"""Calibration on quantile sketches: local, pooled, FedSCR, PFWCP and trimmed mean, with the decision rules."""
from itertools import combinations

import numpy as np

from . import config as C
from .federated import MLP, fedavg, scores_for, softmax_log


def make_grid(n_body=91, n_tail=201):
    return np.unique(np.concatenate([np.linspace(0, 0.9, n_body), np.linspace(0.9, 1.0, n_tail)]))


UGRID = make_grid()


def sketch(v, w=None, grid=None):
    """Quantiles of v on a fixed grid; with weights w, a weighted sketch with its effective sample size."""
    grid = UGRID if grid is None else grid
    if w is None:
        return dict(q=np.quantile(v, grid), n=len(v), W=float(len(v)), u=grid)
    o = np.argsort(v)
    v, w = v[o], np.maximum(w[o], 0) + 1e-12
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return dict(q=np.interp(grid, cw, v), n=float(w.sum() ** 2 / (w ** 2).sum()), W=float(w.sum()), u=grid)


def cdf(sk, t):
    return np.interp(t, sk["q"], sk["u"], left=0.0, right=1.0)


def mix_pvalue(sks, w, x):
    """Conformal p value and fusion weight p value (without the +1 term) from a weighted mixture of sketches."""
    w = np.asarray(w, float)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2 / np.array([s["n"] for s in sks]))
    F = sum(wj * cdf(s, x) for wj, s in zip(w, sks) if wj > 1e-9)
    return ((1 - F) * n_eff + 1) / (n_eff + 1), (1 - F) * n_eff / (n_eff + 1)


def w1_std(a, b):
    """Wasserstein distance between two sketches on the levels 0.05 to 0.95, scaled by the IQR of a."""
    m = (a["u"] >= 0.05) & (a["u"] <= 0.95)
    iqr = np.interp(0.75, a["u"], a["q"]) - np.interp(0.25, a["u"], a["q"]) + 1e-12
    return np.mean(np.abs(a["q"][m] - b["q"][m])) / iqr


def w1_tail(a, b, lo=0.95, hi=0.99):
    """Distance on the upper tail levels, scaled by the IQR of a."""
    m = (a["u"] >= lo) & (a["u"] <= hi)
    iqr = np.interp(0.75, a["u"], a["q"]) - np.interp(0.25, a["u"], a["q"]) + 1e-12
    return np.mean(np.abs(a["q"][m] - b["q"][m])) / iqr


def mix_weights(method, k, lst, h_c=None):
    n = np.array([s["n"] for s in lst], float)
    if method == "local":
        w = np.zeros(len(lst))
        w[k] = 1
        return w
    if method == "pooled":
        return n
    if method in ("fedscr", "fedscr_tail"):
        h_c = C.H_C if h_c is None else h_c
        if method == "fedscr":
            d = np.array([w1_std(lst[k], s) for s in lst])
        else:                                   # distance that also compares the upper tail levels
            d = np.array([max(w1_std(lst[k], s), w1_tail(lst[k], s)) for s in lst])
        h = h_c * np.sqrt(1 / n[k] + 1 / n)
        return n * np.exp(-(d / h) ** 2)
    raise ValueError(method)


def stat_values(sc, gi, key, subsets, partners):
    if key == "typ":
        return sc["typ"][gi]
    R = sc["R"][gi]
    if key == "min":
        return R.min(1)
    if key == "max":
        return R.max(1)
    if key[0] == "S":
        return R[:, int(key[1:])]
    q = int(key[1])          # q1, q2: max over excluded partner sets E with |E| = q of the min over the other subsets
    vals = [R[:, [i for i, S in enumerate(subsets) if not set(S) & set(E)]].min(1) for E in combinations(partners, q)]
    return np.max(vals, 0)


class Cal:
    """Sketches of every client for every (group, statistic), and p values for a target client."""

    def __init__(self, sysm, sc_list, grid=None, pf_ratios=None, h_c=None):
        self.sysm, self.grid, self.h_c, self.cache = sysm, grid, h_c, {}
        self.keys = {}
        for gi, g in enumerate(C.EVID):
            P = sysm.partners[g]
            self.keys[gi] = ["typ", "min", "max", "q1"] + (["q2"] if len(P) > 2 else []) + \
                [f"S{i}" for i in range(len(sysm.subsets[g]))]
        self.vals = [{(gi, key): stat_values(sc, gi, key, sysm.subsets[C.EVID[gi]], sysm.partners[C.EVID[gi]])
                      for gi in range(len(C.EVID)) for key in self.keys[gi]} for sc in sc_list]
        self.sk = {kk: [sketch(v[kk], grid=grid) for v in self.vals] for kk in self.vals[0]}
        self.pf = None
        if pf_ratios is not None:           # PFWCP: density ratio weighted sketches for every target client
            self.pf = {kk: {t: [sketch(v[kk], pf_ratios[j][:, t], grid) for j, v in enumerate(self.vals)]
                            for t in range(pf_ratios[0].shape[1])} for kk in self.vals[0]}

    def mixture(self, method, k, kk):
        key = (method, k, kk)
        if key not in self.cache:
            if method == "pfwcp":
                lst = self.pf[kk][k]
                self.cache[key] = (lst, np.array([x["W"] for x in lst]))
            elif method == "trimmed_pooled":
                Q = np.stack([x["q"] for x in self.sk[kk]])
                t = int(0.25 * len(Q))
                agg = dict(q=np.maximum.accumulate(np.sort(Q, 0)[t:len(Q) - t].mean(0)),
                           n=sum(x["n"] for x in self.sk[kk]), W=1.0, u=self.sk[kk][0]["u"])
                self.cache[key] = ([agg], np.ones(1))
            else:
                lst = self.sk[kk]
                self.cache[key] = (lst, mix_weights(method, k, lst, self.h_c))
        return self.cache[key]

    def pval(self, method, k, kk, x):
        sks, w = self.mixture(method, k, kk)
        return mix_pvalue(sks, w, x)


def decide(cal, method, k, sc, variant="adaptive"):
    """Sensor alarms (groups x windows, bool) and fusion weights for target client k."""
    sysm = cal.sysm
    G = len(C.EVID)
    n = sc["typ"].shape[1]
    pv_typ = [cal.pval(method, k, (gi, "typ"), sc["typ"][gi]) for gi in range(G)]

    def merged(pc, pt):
        return np.minimum(1, 2 * np.minimum(pc, pt))

    if variant == "typ_only":
        P = [pv_typ[gi] for gi in range(G)]
    elif variant in ("min", "max", "q1", "q2", "cross_only"):
        P = []
        for gi in range(G):
            key = "q2" if (variant == "cross_only" and "q2" in cal.keys[gi]) else ("q1" if variant == "cross_only" else variant)
            if key not in cal.keys[gi]:
                key = "max"
            pc = cal.pval(method, k, (gi, key), sc["R"][gi].min(1))
            P.append(pc if variant == "cross_only" else (merged(pc[0], pv_typ[gi][0]), merged(pc[1], pv_typ[gi][1])))
    else:                                   # adaptive partner selection
        st1 = []
        for gi in range(G):
            key = "q2" if "q2" in cal.keys[gi] else "q1"
            pc = cal.pval(method, k, (gi, key), sc["R"][gi].min(1))[0]
            st1.append(merged(pc, pv_typ[gi][0]) <= C.ALPHA)
        st1 = np.stack(st1)
        P = []
        for gi, g in enumerate(C.EVID):
            subs, Pg = sysm.subsets[g], sysm.partners[g]
            allowed = np.stack([~st1[C.EVID.index(j)] if j in C.EVID else np.ones(n, bool) for j in Pg], 1)
            pc_v, pc_w = np.ones(n), np.ones(n)
            has = np.zeros(n, bool)
            for pattern in np.unique(allowed, axis=0):
                rows = np.all(allowed == pattern, 1)
                S = tuple(j for j, a in zip(Pg, pattern) if a)
                if not S:
                    continue
                i = subs.index(S)
                has[rows] = True
                pc_v[rows], pc_w[rows] = cal.pval(method, k, (gi, f"S{i}"), sc["R"][gi][rows, i])
            pv = np.where(has, merged(pc_v, pv_typ[gi][0]), pv_typ[gi][0])
            pw = np.where(has, merged(pc_w, pv_typ[gi][1]), pv_typ[gi][1])
            P.append((pv, pw))
    flags = np.stack([p[0] <= C.ALPHA for p in P])
    wts = np.stack([np.minimum(1, p[1] / C.ALPHA) for p in P])
    return flags, wts


def pf_classifier(sysm, tab, client, clients, labels, seed):
    """Federated client classifier on head embeddings and partner features; returns density ratios for PFWCP."""
    data = []
    for k, lab in zip(clients, labels):
        rows = np.where((client == k) & (tab["block"] != "test"))[0]
        sc = scores_for(sysm, tab, rows)
        X = sc["emb"] if sc["zpart"] is None else np.concatenate([sc["emb"], sc["zpart"]], 1)
        data.append((X.astype(np.float32), np.full(len(rows), lab)))
    mu = np.concatenate([d[0] for d in data]).mean(0)
    sd = np.concatenate([d[0] for d in data]).std(0) + 1e-6
    data = [((x - mu) / sd, y) for x, y in data]
    K = max(labels) + 1
    clf = fedavg(lambda: MLP(data[0][0].shape[1], K, seed=seed), data, C.FL_ROUNDS, seed)
    prior = np.bincount(np.concatenate([d[1] for d in data]), minlength=K) / sum(len(d[1]) for d in data)

    def ratios(sc):
        X = sc["emb"] if sc["zpart"] is None else np.concatenate([sc["emb"], sc["zpart"]], 1)
        p = np.exp(softmax_log(clf((X - mu) / sd)))
        return np.clip(p / prior[None, :], 0, 20)
    return ratios


def trim_rule(own, donors, margin=1.5):
    """Keep scores below median + margin x tail shape x IQR, with the tail shape taken from the other clients."""
    k = max((np.interp(0.995, s["u"], s["q"]) - np.interp(0.5, s["u"], s["q"])) /
            (np.interp(0.75, s["u"], s["q"]) - np.interp(0.25, s["u"], s["q"]) + 1e-12) for s in donors)
    med, q1, q3 = np.quantile(own, [0.5, 0.25, 0.75])
    return own <= med + margin * k * (q3 - q1 + 1e-12)


class TrimCal(Cal):
    """Calibration of a new site from its unlabelled windows, each statistic trimmed with the donor tail shapes."""

    def __init__(self, sysm, sc_own, donor_cal=None, margin=1.5):
        super().__init__(sysm, [sc_own])
        self.trimmed = []
        if donor_cal is not None:
            for kk in list(self.sk):
                v = self.vals[0][kk]
                keep = trim_rule(v, donor_cal.sk[kk], margin)
                self.trimmed.append(1 - keep.mean())
                self.sk[kk] = [sketch(v[keep])]


def attack(sk, kind, f):
    """Inflate (f > 1) or deflate (f < 1) a sketch about its median."""
    if kind == "none":
        return sk
    med = np.interp(0.5, sk["u"], sk["q"])
    sc_ = np.interp(0.75, sk["u"], sk["q"]) - np.interp(0.25, sk["u"], sk["q"])
    return dict(sk, q=med + f * (sk["q"] - med) + (f - 1) * (abs(med) + sc_))


def attack_tail(sk, f, u0=0.90):
    """Keep the sketch below level u0 and stretch (f > 1) or shrink (f < 1) the upper tail about Q(u0)."""
    u, q = sk["u"], sk["q"].copy()
    q0 = np.interp(u0, u, q)
    m = u >= u0
    q[m] = q0 + f * (q[m] - q0)
    return dict(sk, q=np.maximum.accumulate(q))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", labels=np.unique(y))


def fused_pred(logp, w):
    w = np.where(w.sum(0, keepdims=True) == 0, 1.0, w)        # all groups distrusted: equal weights
    return np.argmax((w[:, :, None] * logp).sum(0), 1)


def entropy_w(logp):
    return 1 / (-(np.exp(logp) * logp).sum(2) + 1e-3)
