"""Window features, sensor degradation operators and the window table."""
import os
import pickle
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np

from . import config as C
from .data import recordings


@lru_cache(maxsize=None)
def _filters(win, fs):
    freqs = np.fft.rfftfreq(win, 1 / fs)
    band = np.digitize(freqs, np.geomspace(10, fs / 2, C.NB + 1)) - 1
    eband = np.digitize(freqs, np.geomspace(2, min(1000.0, fs / 8), C.NE + 1)) - 1
    h = np.zeros(win)
    h[0] = 1
    h[1:win // 2] = 2
    h[win // 2] = 1                       # analytic signal filter (Hilbert transform)
    return band, eband, h


def windows(x):
    n = 1 + (len(x) - C.WIN) // C.HOP
    return x[np.arange(C.WIN)[None, :] + C.HOP * np.arange(max(n, 0))[:, None]]


def _bandsum(P, idx, nb):
    out = np.zeros((P.shape[0], nb, P.shape[2]), np.float32)
    ok = (idx >= 0) & (idx < nb)
    np.add.at(out, (slice(None), idx[ok]), P[:, ok])
    return np.log(out + 1e-10)


def feat_dyn(w):
    """Per channel: 48 log band powers, 5 statistics and 24 log envelope spectrum band powers."""
    band, eband, h = _filters(C.WIN, C.FS)
    outs = []
    for i in range(0, len(w), 512):
        x = w[i:i + 512].astype(np.float32)
        n, L, _ = x.shape
        mu = x.mean(1, keepdims=True)
        xc = x - mu
        sd = xc.std(1) + 1e-8
        F = np.fft.fft(xc, axis=1)
        P = (np.abs(F[:, :L // 2 + 1]) ** 2) / L
        env = np.abs(np.fft.ifft(F * h[None, :, None], axis=1))
        env -= env.mean(1, keepdims=True)
        Pe = (np.abs(np.fft.rfft(env, axis=1)) ** 2) / L
        st = np.stack([mu[:, 0], np.log(sd), (xc ** 3).mean(1) / sd ** 3, np.log((xc ** 4).mean(1) / sd ** 4 + 1e-6),
                       np.log(np.abs(xc).max(1) / sd + 1e-6)], 1)
        outs.append(np.concatenate([_bandsum(P, band, C.NB), st, _bandsum(Pe, eband, C.NE)], 1).reshape(n, -1))
    return np.concatenate(outs).astype(np.float32)


def feat_slow(w):
    """Mean, log standard deviation and slope of each channel."""
    t = np.arange(w.shape[1]) / w.shape[1]
    slope = ((w - w.mean(1, keepdims=True)) * (t - 0.5)[None, :, None]).mean(1) * 12
    return np.concatenate([w.mean(1), np.log(w.std(1) + 1e-6), slope], 1).astype(np.float32)


def feature(g, w):
    return feat_dyn(w) if C.GTYPE[g] == "dyn" else feat_slow(w)


def env_cols(n_ch):
    """Column indices of the envelope features of a group with n_ch channels."""
    return np.array([j for j in range((C.NB + 5 + C.NE) * n_ch) if (j // n_ch) >= C.NB + 5])


def degrade(w, kind, s, rng):
    """Apply a sensor degradation of severity s to windows w of shape (n, length, channels)."""
    w = w.astype(np.float32).copy()
    n, L, _ = w.shape
    sd = w.std(1, keepdims=True) + 1e-8
    if kind == "noise":
        w += rng.standard_normal(w.shape).astype(np.float32) * (2.0 * s) * sd
    elif kind == "gain":
        w *= (1 + 3.0 * s)
    elif kind == "bias":
        w += 4.0 * s * sd
    elif kind == "stuck":
        st = int((1 - 0.9 * s) * L)
        w[:, st:, :] = w[:, st:st + 1, :]
    elif kind == "clip":
        c = (1 - 0.85 * s) * np.abs(w).max(1, keepdims=True)
        w = np.clip(w, -c, c)
    elif kind == "dropout":
        nb = 8
        bl = L // nb
        k = int(round(0.6 * s * nb))
        drop = np.argsort(rng.random((n, nb)), 1)[:, :k]
        mask = np.ones((n, nb), np.float32)
        np.put_along_axis(mask, drop, 0.0, 1)
        w[:, :nb * bl] *= np.repeat(mask, bl, 1)[:, :, None]
    elif kind == "bandwidth":
        F = np.fft.rfft(w, axis=1)
        F[:, int((1 - 0.95 * s) * F.shape[1]):] = 0
        w = np.fft.irfft(F, n=L, axis=1).astype(np.float32)
    elif kind == "fail":
        w = (1e-3 * sd * rng.standard_normal(w.shape)).astype(np.float32)
    return w


def build_table():
    """Clean features of every window and degraded features of every calibration and test window."""
    rng = np.random.default_rng(1000)
    rows = {k: [] for k in ("cond", "cls", "rec", "block", "pos")}
    F = {g: [] for g in C.GROUPS}
    D = defaultdict(list)
    D_rows = []
    row0 = rid = 0
    t0 = time.time()
    for cond, cls, _, sig, blk in recordings():
        W = {g: windows(sig[g]) for g in C.GROUPS}
        n = min(len(W[g]) for g in C.GROUPS)
        if n < 3:
            continue
        if blk is None and C.SPLIT_MODE == "interleaved":
            pat = np.array(["train", "train", "train", "cal", "test"])
            block = pat[(np.arange(n) // C.CHUNK + rid) % len(pat)]
            keep = (np.arange(n) % C.CHUNK) != 0          # first window of every chunk is dropped
        elif blk is None:
            c1, c2 = int(C.SPLIT[0] * n), int((C.SPLIT[0] + C.SPLIT[1]) * n)
            block = np.array(["train"] * c1 + ["cal"] * (c2 - c1) + ["test"] * (n - c2))
            keep = np.ones(n, bool)
            keep[[c1, c2]] = False if C.HOP < C.WIN else keep[[c1, c2]]   # overlapping windows at block edges
        else:
            block = np.array([blk] * n)
            keep = np.ones(n, bool)
        idx = np.where(keep)[0]
        for g in C.GROUPS:
            F[g].append(feature(g, W[g][idx]))
        rows["cond"] += [cond] * len(idx)
        rows["cls"] += [C.CLASSES.index(cls)] * len(idx)
        rows["rec"] += [rid] * len(idx)
        rows["block"] += list(block[idx])
        rows["pos"] += list(idx / n)
        ct = idx[block[idx] != "train"]
        if len(ct):
            for g in C.EVID:
                for kind in C.KINDS:
                    for s in C.SEVS:
                        D[(g, kind, s)].append(feature(g, degrade(W[g][ct], kind, s, rng)))
        D_rows += list(row0 + np.where(block[idx] != "train")[0])
        row0 += len(idx)
        rid += 1
        if rid % 10 == 0:
            print(f"  {rid} recordings, {row0} windows ({time.time() - t0:.0f}s)")
    tab = {k: np.array(v) for k, v in rows.items()}
    tab["F"] = {g: np.concatenate(F[g]) for g in C.GROUPS}
    tab["D"] = {k: np.concatenate(v) for k, v in D.items()}
    tab["D_rows"] = np.array(D_rows)
    return tab


def load_table():
    path = os.path.join(C.OUT_DIR, f"table_{C.DATASET}_{C.WIN}_{C.SPLIT_MODE}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    tab = build_table()
    pickle.dump(tab, open(path, "wb"), protocol=4)
    return tab
