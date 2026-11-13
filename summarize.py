"""Summarise the CSV files written by run_experiments.py (mean and standard deviation over seeds).

Example:
    python summarize.py results/kaist
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def ms(s):
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n_cal", type=int, default=100)
    a = ap.parse_args()
    p = a.results
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    if os.path.exists(f"{p}/c1.csv"):
        c1 = pd.read_csv(f"{p}/c1.csv")
        rows = []
        for (var, n, m), x in c1.groupby(["variant", "n_cal", "method"]):
            cf = x.groupby(["seed", "client"]).far.mean()
            rows.append(dict(variant=var, n_cal=n, method=m, far=ms(x.groupby("seed").far.mean()),
                             worst_client=ms(cf.groupby("seed").max()), client_sd=ms(cf.groupby("seed").std()),
                             power_s06=ms(x.groupby("seed")["power_s0.6"].mean()),
                             power_s10=ms(x.groupby("seed")["power_s1.0"].mean())))
        print("\nCalibration across clients\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/fusion.csv"):
        f = pd.read_csv(f"{p}/fusion.csv")
        t = f.groupby(["method", "scenario", "seed"]).f1.mean().groupby(["method", "scenario"]).agg(["mean", "std"])
        print("\nFused macro F1 (mean and sd over seeds)\n", t.unstack("scenario").round(4).to_string())
        qv = "fedscr[q2]" if "fedscr[q2]" in f.method.unique() else "fedscr[q1]"
        rows = []
        for sc in [s for s in f.scenario.unique() if s != "clean"]:
            pv = f[f.scenario == sc].pivot_table(index=["seed", "client", "detail"], columns="method", values="f1")
            for b in ("equal", "entropy", "local", "pooled", "pfwcp", "oracle_drop"):
                d = (pv[qv] - pv[b]).dropna()
                rows.append(dict(scenario=sc, versus=b, diff=round(d.mean(), 4), wins=int((d > 0).sum()),
                                 losses=int((d < 0).sum()), cells=len(d),
                                 p=wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
        print(f"\nWilcoxon signed rank tests, {qv} against each method\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/partner_far.csv"):
        pf = pd.read_csv(f"{p}/partner_far.csv")
        print("\nFalse alarm rate of healthy groups with one degraded partner")
        for m, x in pf.groupby("method"):
            print(f"  {m:24s} {ms(x.groupby('seed').far.mean())}")

    if os.path.exists(f"{p}/c2.csv"):
        c2 = pd.read_csv(f"{p}/c2.csv")
        rows = [dict(method=m, eps=e, far=ms(x.groupby("seed").far.mean()), power=ms(x.groupby("seed").power.mean()))
                for (m, e), x in c2.groupby(["method", "eps"])]
        print("\nNew site calibration\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/c3.csv"):
        c3 = pd.read_csv(f"{p}/c3.csv")
        rows = [dict(attack=at, method=m, far=ms(x.groupby("seed").far.mean()), worst=ms(x.groupby("seed").far.max()),
                     power=ms(x.groupby("seed").power.mean())) for (at, m), x in c3.groupby(["attack", "method"])]
        print("\nManipulated sketches\n", pd.DataFrame(rows).to_string(index=False))
        h = pd.read_csv(f"{p}/c3_honest.csv")
        rows = [dict(method=m, group=g, power=ms(x.groupby("seed").power.mean())) for (m, g), x in h.groupby(["method", "group"])]
        print("\nHonest client with a degraded sensor during calibration\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/tail_attack.csv"):
        t = pd.read_csv(f"{p}/tail_attack.csv")
        rows = [dict(u0=u, attack=at, method=m, far=ms(x.groupby("seed").far.mean()), worst=ms(x.groupby("seed").far.max()),
                     power=ms(x.groupby("seed").power.mean())) for (u, at, m), x in t.groupby(["u0", "attack", "method"])]
        print("\nManipulation of the upper tail only\n", pd.DataFrame(rows).to_string(index=False))
        c = pd.read_csv(f"{p}/tail_cal.csv")
        cf = c.groupby(["n_cal", "method", "seed", "client"]).far.mean()
        rows = [dict(n_cal=n, method=m, far=ms(x.groupby("seed").far.mean()),
                     worst_client=ms(cf.loc[(n, m)].groupby("seed").max()), power=ms(x.groupby("seed").power.mean()))
                for (n, m), x in c.groupby(["n_cal", "method"])]
        print("\nTail aware distance without manipulation\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/scaling.csv"):
        s = pd.read_csv(f"{p}/scaling.csv")
        rows = [dict(n_clients=n, method=m, client_far_sd=ms(x.groupby("seed").far.std()),
                     power=ms(x.groupby("seed").power.mean())) for (n, m), x in s.groupby(["n_clients", "method"])]
        print("\nNumber of clients\n", pd.DataFrame(rows).to_string(index=False))

    if os.path.exists(f"{p}/ablation.csv"):
        ab = pd.read_csv(f"{p}/ablation.csv")
        t = ab.groupby(["features", "setting", "value", "seed"])[["far", "power", "f1_clean", "f1_1g"]].mean() \
              .groupby(["features", "setting", "value"]).mean().round(4)
        print("\nAblations (mean over seeds)\n", t.to_string())

    if os.path.exists(f"{p}/meta.json"):
        meta = json.load(open(f"{p}/meta.json"))
        print("\nwindows:", meta["n_windows"], "| sketch KiB per client:", round(meta["comm_kib"], 1),
              "| timing (s):", {k: round(v) for k, v in meta["timing"].items()})


if __name__ == "__main__":
    main()
