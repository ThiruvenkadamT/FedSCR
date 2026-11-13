"""File inventory, signal loaders and the MCC5-THU download helper."""
import glob
import json
import os
import re
import shutil
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from . import config as C

MCC5_ID, MCC5_VER = "p92gj2732w", 1


# ---------------------------------------------------------------- MCC5-THU download (Mendeley Data)
def _mcc5_convert_csv(fileobj, name, dst):
    import pandas as pd
    out = os.path.join(dst, os.path.splitext(os.path.basename(name))[0] + ".npy")
    if os.path.exists(out):
        return
    df = pd.read_csv(fileobj)
    np.save(out, df.values.astype(np.float32))
    json.dump(list(map(str, df.columns)), open(out[:-4] + ".cols.json", "w"))


def _mcc5_convert_zip(path, dst):
    with zipfile.ZipFile(path) as z:
        for m in z.namelist():
            low = m.lower()
            if low.endswith(".zip"):
                tmp = os.path.join(dst, "_inner.zip")
                with z.open(m) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, length=16 << 20)
                _mcc5_convert_zip(tmp, dst)
                os.remove(tmp)
            elif low.endswith(".csv"):
                with z.open(m) as src:
                    _mcc5_convert_csv(src, m, dst)


def _download(url, dst):
    import requests
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=16 << 20)


def _mendeley_files():
    import requests
    api = f"https://data.mendeley.com/public-api/datasets/{MCC5_ID}"
    hdr = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    out = []
    folders = requests.get(f"{api}/folders/{MCC5_VER}", headers=hdr, timeout=60)
    ids = ["root"] + ([f["id"] for f in folders.json()] if folders.ok else [])
    for fid in ids:
        r = requests.get(f"{api}/files", params={"folder_id": fid, "version": MCC5_VER}, headers=hdr, timeout=60)
        if r.ok:
            for f in r.json():
                cd = f.get("content_details", {})
                out.append((f.get("filename"), cd.get("size", 0), cd.get("download_url")))
    return out


def download_mcc5(dst):
    """Download MCC5-THU file by file and store every CSV as float32 .npy with its column names."""
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    files = _mendeley_files()
    print(f"{len(files)} files listed, {sum(f[1] or 0 for f in files) / 2 ** 30:.2f} GiB")
    for i, (name, _, url) in enumerate(files):
        if not url or not name:
            continue
        low = name.lower()
        if low.endswith(".csv") and os.path.exists(os.path.join(dst, os.path.splitext(name)[0] + ".npy")):
            continue
        tmp = os.path.join(dst, "_dl_" + os.path.basename(name))
        _download(url, tmp)
        if low.endswith(".csv"):
            _mcc5_convert_csv(tmp, name, dst)
        elif low.endswith(".zip"):
            _mcc5_convert_zip(tmp, dst)
        os.remove(tmp)
        if i % 20 == 0:
            print(f"  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    print(len(glob.glob(os.path.join(dst, "*.npy"))), "recordings converted")


# ---------------------------------------------------------------- inventory
def kaist_parse(path):
    name = os.path.basename(path).lower()
    m = re.search(r"(\d)\s*nm", name) or re.search(r"(\d)\s*nm", path.lower())
    cls = next((c for c in ["normal", "bpfi", "bpfo", "misalign", "unbalance"] if c in name), None)
    if not m or cls is None:
        return None
    sev = re.sub(r"\.(mat|tdms)$", "", name.split(cls, 1)[1]).strip("_- ")
    d = re.sub(r"[^0-9]", "", sev).lstrip("0") or "0"
    return int(m.group(1)), cls, d


def mcc5_parse(path):
    toks = re.sub(r"\.(csv|npy)$", "", os.path.basename(path).lower()).split("_")
    i = next((j for j, t in enumerate(toks) if t in ("speed", "torque")), None)
    if i is None or i == 0:
        return None
    cls_t = toks[:i]
    sev = cls_t[-1] if cls_t[-1] in ("l", "m", "h") else ""
    if sev:
        cls_t = cls_t[:-1]
    return "_".join(toks[i:]), "_".join(cls_t), sev or "0"


def inventory():
    """List the recordings of the configured dataset and set C.CONDS and C.CLASSES."""
    inv = []
    root = C.DATA_ROOT
    if C.DATASET == "kaist":
        for f in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
            fl = f.lower()
            if fl.endswith((".mat", ".tdms")) and "acous" not in fl:
                p = kaist_parse(f)
                if p:
                    inv.append(dict(path=f, cond=p[0], cls=p[1], rec=f"{p[0]}_{p[1]}_{p[2]}",
                                    mod="vibration" if fl.endswith(".mat") else "tdms"))
    elif C.DATASET == "paderborn":
        want = {b: c for c, bs in C.PU_BEARINGS.items() for b in bs}
        for f in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True):
            m = re.search(r"(N\d\d_M\d\d_F\d\d)_(K[AIB]?\d+)_(\d+)\.mat$", os.path.basename(f))
            if m and m.group(1) in C.PU_CONDS and m.group(2) in want:
                inv.append(dict(path=f, cond=m.group(1), cls=want[m.group(2)], rec=f"{m.group(1)}_{m.group(2)}_{m.group(3)}",
                                bearing=m.group(2), idx=int(m.group(3)), mod="pu"))
    elif C.DATASET == "mcc5":
        for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) + \
                glob.glob(os.path.join(root, "**", "*.npy"), recursive=True):
            p = mcc5_parse(f)
            if p:
                inv.append(dict(path=f, cond=p[0], cls=p[1], rec=f"{p[0]}|{p[1]}|{p[2]}", mod="csv"))
    C.INVENTORY = inv
    if C.DATASET == "synthetic":
        C.CONDS, C.CLASSES = [0, 2, 4], ["normal", "bpfi", "bpfo", "misalign", "unbalance"]
        return
    assert inv, f"no files found under {root}"
    print(len(inv), "files; conditions:", dict(Counter(d["cond"] for d in inv)))
    C.CONDS = sorted({d["cond"] for d in inv}, key=str)
    C.CLASSES = sorted({d["cls"] for d in inv})
    if C.DATASET == "kaist":
        C.CLASSES = [c for c in ["normal", "bpfi", "bpfo", "misalign", "unbalance"] if c in C.CLASSES]
    if C.DATASET == "paderborn":
        C.CLASSES = ["healthy", "outer", "inner"]


# ---------------------------------------------------------------- loaders
def to_fs(x, fs):
    if abs(fs - C.FS) / C.FS < 0.01:
        return x
    fr = Fraction(C.FS / fs).limit_denominator(200)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)


def _numeric_arrays(obj, out, name="root"):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiu" and obj.size >= 50000:
            out.append((name, np.asarray(obj, dtype=np.float32)))
        elif obj.dtype.kind == "O":
            for i, el in enumerate(obj.flat):
                _numeric_arrays(el, out, f"{name}[{i}]")
        elif obj.dtype.names:
            for n in obj.dtype.names:
                for el in obj[n].flat:
                    _numeric_arrays(el, out, f"{name}.{n}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not str(k).startswith("__"):
                _numeric_arrays(v, out, str(k))


def mat_arrays(path):
    try:
        out = []
        _numeric_arrays(sio.loadmat(path, squeeze_me=False), out)
    except NotImplementedError:          # MATLAB v7.3 files
        import h5py
        out = []
        with h5py.File(path, "r") as h:
            h.visit(lambda k: out.append((k, np.asarray(h[k], np.float32)))
                    if isinstance(h[k], h5py.Dataset) and h[k].size >= 50000 else None)
    return out


def load_mat(path):
    cols = []
    for _, a in mat_arrays(path):
        a = np.squeeze(a)
        if a.ndim == 1:
            cols.append(a[:, None])
        elif a.ndim == 2:
            cols.append(a if a.shape[0] >= a.shape[1] else a.T)
    if not cols:
        raise ValueError(f"no signal array in {path}")
    n = max(c.shape[0] for c in cols)
    X = np.concatenate([c for c in cols if c.shape[0] == n], 1)
    return X[:, [i for i in range(X.shape[1]) if not np.all(np.diff(X[:2000, i]) > 0)]]   # drop time axes


def load_tdms(path):
    from nptdms import TdmsFile
    chans = {}
    for g in TdmsFile.read(path).groups():
        for c in g.channels():
            x = np.asarray(c[:], np.float32)
            if x.size > 1000 and np.std(x) > 0:
                inc = c.properties.get("wf_increment")
                chans[f"{g.name}/{c.name}"] = (x, 1.0 / inc if inc else C.FS)
    return chans


def tdms_kind(name, x):
    low = name.lower()
    if "temp" in low or "thermo" in low:
        return "temp"
    if "current" in low:
        return "current"
    return "temp" if abs(float(np.mean(x))) > 3 * float(np.std(x)) else "current"


def load_kaist(files):
    out = {}
    for d in files:
        if d["mod"] == "vibration":
            v = load_mat(d["path"])
            if v.shape[1] < 4:
                raise ValueError(f"expected 4 vibration channels in {d['path']}, got {v.shape}")
            out["vibA"], out["vibB"] = v[:, :2], v[:, 2:4]
        else:
            for name, (x, fs) in load_tdms(d["path"]).items():
                out.setdefault(tdms_kind(name, x), []).append(to_fs(x, fs)[:, None])
    for k in ("current", "temp"):
        if k in out:
            n = min(len(x) for x in out[k])
            out[k] = np.concatenate([x[:n] for x in out[k]], 1)
    if "current" in out:
        out["current"] = out["current"][:, :1]
    if "temp" in out:
        out["temp"] = out["temp"][:, :2]
    return out


def load_pu(files):
    m = sio.loadmat(files[0]["path"], squeeze_me=True, struct_as_record=False)
    rec = m[[k for k in m if not k.startswith("__")][0]]
    ch = {str(y.Name): np.asarray(y.Data, np.float32).ravel() for y in np.atleast_1d(rec.Y)}
    n_ref = len(ch["vibration_1"])

    def up(x):        # 4 kHz mechanical channels to the vibration rate by linear interpolation
        return np.interp(np.linspace(0, len(x) - 1, n_ref), np.arange(len(x)), x).astype(np.float32)
    mech = np.stack([up(ch[k]) for k in ("force", "speed", "torque") if k in ch], 1)
    return {"vib": ch["vibration_1"][:, None], "cur1": ch["phase_current_1"][:, None],
            "cur2": ch["phase_current_2"][:, None], "mech": mech}


def load_mcc5(files):
    import pandas as pd
    path = files[0]["path"]
    if path.endswith(".npy"):
        X = np.load(path)
        cj = path[:-4] + ".cols.json"
        names = json.load(open(cj)) if os.path.exists(cj) else [f"c{i}" for i in range(X.shape[1])]
    else:
        df = pd.read_csv(path)
        X, names = df.values.astype(np.float32), list(df.columns)
    cols = [str(c).lower() for c in names]

    def pick(*keys, excl=()):
        return [i for i, c in enumerate(cols) if all(k in c for k in keys) and not any(e in c for e in excl)]
    mot, gear = pick("motor", excl=("speed", "torque")), pick("gear", excl=("speed", "torque"))
    sp, tq = pick("speed"), pick("torque")
    if len(mot) < 3 or len(gear) < 3:          # documented column order: speed, torque, motor x/y/z, gearbox x/y/z
        sp, tq, mot, gear = [0], [1], [2, 3, 4], [5, 6, 7]
    return {"motorVib": X[:, mot], "gearVib": X[:, gear], "mech": X[:, sp + tq]}


def load_synth(cond, cls, sev_idx, rng, seconds=24):
    """Small synthetic machine with the KAIST sensor layout, used for testing the pipeline."""
    T = int(seconds * C.FS)
    t = np.arange(T) / C.FS
    f0 = {0: 29.0, 2: 28.6, 4: 28.2}[cond]
    li = C.CONDS.index(cond)
    src = (1.0 + 0.25 * li) * (np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t))
    ff = {"bpfi": 5.4 * f0, "bpfo": 3.6 * f0, "misalign": 2 * f0, "unbalance": f0}
    if cls in ff:
        k = 0.4 + 0.2 * sev_idx
        src = src + (k * np.sin(2 * np.pi * ff[cls] * t) * (1 + np.sin(2 * np.pi * f0 * t)) if cls in ("bpfi", "bpfo")
                     else k * np.sin(2 * np.pi * ff[cls] * t + 0.5))
    nz = 0.25 * (1 + 0.6 * li)
    vibA = np.stack([src + nz * rng.standard_normal(T), 0.8 * src + nz * rng.standard_normal(T)], 1)
    vibB = np.stack([0.6 * np.roll(src, 7) + nz * rng.standard_normal(T), 0.5 * np.roll(src, 11) + nz * rng.standard_normal(T)], 1)
    cur = ((1 + 0.15 * cond) * np.sin(2 * np.pi * 60 * t) * (1 + 0.05 * src) + 0.05 * rng.standard_normal(T))[:, None]
    temp = (25 + 2 * cond + 0.2 * np.cumsum(rng.standard_normal(T)) / np.sqrt(T) + 0.02 * rng.standard_normal(T))[:, None]
    return {"vibA": vibA, "vibB": vibB, "current": cur, "temp": temp}


LOADERS = {"kaist": load_kaist, "paderborn": load_pu, "mcc5": load_mcc5}


def recording_groups():
    g = defaultdict(list)
    for d in C.INVENTORY:
        g[d["rec"]].append(d)
    return dict(sorted(g.items()))


def pu_block(files):
    """Paderborn: the 20 measurements of each bearing and condition are permuted once and split 14/3/3."""
    d = files[0]
    perm = np.random.default_rng(zlib.crc32((d["cond"] + d["bearing"]).encode())).permutation(20)
    pos = int(np.where(perm == (d["idx"] - 1) % 20)[0][0])
    return "train" if pos < 14 else ("cal" if pos < 17 else "test")


def recordings():
    """Yield (condition, class, key, signals by group, block or None)."""
    if C.DATASET == "synthetic":
        rng = np.random.default_rng(123)
        for c in C.CONDS:
            for cls in C.CLASSES:
                for si in range(1 if cls == "normal" else 2):
                    yield c, cls, f"{c}_{cls}_{si}", load_synth(c, cls, si, rng), None
        return
    loader = LOADERS[C.DATASET]
    for key, files in recording_groups().items():
        try:
            sig = loader(files)
            miss = [g for g in C.GROUPS if g not in sig]
            if miss:
                raise ValueError(f"missing groups {miss}, found {sorted(sig)}")
            n = min(len(sig[g]) for g in C.GROUPS)
            yield files[0]["cond"], files[0]["cls"], key, {g: sig[g][:n] for g in C.GROUPS}, \
                (pu_block(files) if C.SPLIT_MODE == "recording" else None)
        except Exception as e:
            print("skipped", key, ":", e)
