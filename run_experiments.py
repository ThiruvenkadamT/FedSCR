"""Run the FedSCR experiments on one dataset and write the raw results as CSV files.

Example:
    python run_experiments.py --dataset kaist --data /path/to/KAIST --out results/kaist
    python run_experiments.py --dataset synthetic --seeds 0
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from fedscr import config as C
from fedscr import data, experiments
from fedscr.calibration import UGRID, Cal
from fedscr.federated import sc_take, scores_for, trained_systems, get_system, make_clients
from fedscr.features import load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kaist", "paderborn", "mcc5", "synthetic"])
    ap.add_argument("--data", default=".", help="root folder of the dataset files")
    ap.add_argument("--out", default=None, help="output folder (default results/<dataset>)")
    ap.add_argument("--seeds", default=None, help="comma separated seeds (default 0,1,2,3,4)")
    ap.add_argument("--split", default=None, choices=["time", "recording", "interleaved"])
    ap.add_argument("--download-mcc5", action="store_true", help="download MCC5-THU into --data first")
    ap.add_argument("--only", default=None, help="comma separated subset of: calibration,new_site,robustness,tail,scalability,ablations")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    C.configure(a.dataset, a.data, a.out, seeds, a.split)
    if a.download_mcc5:
        data.download_mcc5(a.data)
    data.inventory()
    print(f"dataset={C.DATASET} conditions={C.CONDS} classes={C.CLASSES} seeds={C.SEEDS} -> {C.OUT_DIR}")
    tab = load_table()
    print("windows:", len(tab["cond"]), {b: int((tab["block"] == b).sum()) for b in ("train", "cal", "test")})

    steps = [("calibration", experiments.calibration_and_fusion), ("new_site", experiments.new_site),
             ("robustness", experiments.robustness),
             ("tail", experiments.tail_manipulation), ("scalability", experiments.scalability),
             ("ablations", experiments.ablations)]
    only = set(a.only.split(",")) if a.only else None
    res, timing = defaultdict(list), {}
    for name, fn in steps:
        if only and name not in only:
            continue
        t0 = time.time()
        fn(tab, res)
        timing[name] = time.time() - t0
    for k, v in res.items():
        pd.DataFrame(v).to_csv(os.path.join(C.OUT_DIR, f"{k}.csv"), index=False)

    K = len(C.CONDS) * C.N_SHARDS
    sysm = get_system(tab, ("all", C.SEEDS[0]), make_clients(tab, C.SEEDS[0]), list(range(K)), C.SEEDS[0])
    n_stat = sum(len(v) for v in Cal(sysm, [sc_take(scores_for(sysm, tab, np.arange(20)), np.arange(20))]).keys.values())
    timing["train_s_per_system"] = float(np.mean([s.train_time for s in trained_systems()]))
    meta = dict(dataset=C.DATASET, seeds=C.SEEDS, timing=timing, comm_kib=n_stat * len(UGRID) * 4 / 1024,
                conds=[str(c) for c in C.CONDS], classes=C.CLASSES, n_windows=int(len(tab["cond"])))
    json.dump(meta, open(os.path.join(C.OUT_DIR, "meta.json"), "w"), indent=2)
    print(f"sketch size: {n_stat} statistics x {len(UGRID)} levels x 4 B = {meta['comm_kib']:.1f} KiB per client")
    print("results written to", C.OUT_DIR)


if __name__ == "__main__":
    main()
