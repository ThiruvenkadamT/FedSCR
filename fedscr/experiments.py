"""Experiments: calibration across clients and fusion, new site calibration, robustness, scalability, ablations."""
import time
from itertools import combinations

import numpy as np

from . import config as C
from .calibration import (Cal, TrimCal, attack, attack_tail, decide, entropy_w, fused_pred, macro_f1, make_grid,
                          pf_classifier)
from .federated import get_system, make_clients, sc_set, sc_take, scores_for


def _split_rows(tab, client, K):
    cal_pool = [np.where((client == k) & (tab["block"] == "cal"))[0] for k in range(K)]
    test_rows = [np.where((client == k) & (tab["block"] == "test"))[0] for k in range(K)]
    live = [k for k in range(K) if len(cal_pool[k]) >= 5 and len(test_rows[k]) >= 5]
    return cal_pool, test_rows, live


def calibration_and_fusion(tab, res):
    """Per client false alarm rate and power for every calibration method, and fused macro F1."""
    K = len(C.CONDS) * C.N_SHARDS
    pairs = list(combinations(range(len(C.EVID)), 2)) if len(C.EVID) > 2 else []
    for seed in C.SEEDS:
        t0 = time.time()
        client = make_clients(tab, seed)
        sysm = get_system(tab, ("all", seed), client, list(range(K)), seed)
        rng = np.random.default_rng(seed)
        cal_pool, test_rows, live = _split_rows(tab, client, K)
        sc_cal_all = [scores_for(sysm, tab, r) if len(r) else None for r in cal_pool]
        sc_test = {k: scores_for(sysm, tab, test_rows[k]) for k in live}
        deg = {k: {(g, kd, sv): scores_for(sysm, tab, test_rows[k], [(g, kd, sv)])
                   for g in C.EVID for kd in C.KINDS for sv in C.SEVS} for k in live}
        ratios = pf_classifier(sysm, tab, client, live, list(range(len(live))), seed)
        for n_cal in C.N_CAL_GRID:
            pick = {k: rng.choice(len(cal_pool[k]), min(n_cal, len(cal_pool[k])), replace=False) for k in live}
            scl = [sc_take(sc_cal_all[k], pick[k]) for k in live]
            cal = Cal(sysm, scl, pf_ratios=[ratios(s) for s in scl])
            for method in ("local", "pooled", "pfwcp", "fedscr"):
                for ki, k in enumerate(live):
                    for variant in ("adaptive", "q2" if len(C.EVID) > 2 else "q1"):
                        fl, _ = decide(cal, method, ki, sc_test[k], variant)
                        for gi, g in enumerate(C.EVID):
                            rec = dict(seed=seed, n_cal=n_cal, method=method, variant=variant, client=k,
                                       cond=str(C.CONDS[k // C.N_SHARDS]), group=g, far=fl[gi].mean())
                            for sv in C.SEVS:
                                rec[f"power_s{sv}"] = float(np.mean(
                                    [decide(cal, method, ki, deg[k][(g, kd, sv)], variant)[0][gi].mean() for kd in C.KINDS]))
                            res["c1"].append(rec)
            print(f"[seed {seed}] calibration n_cal={n_cal} ({time.time() - t0:.0f}s)")
        pick = {k: rng.choice(len(cal_pool[k]), min(C.N_CAL_MAIN, len(cal_pool[k])), replace=False) for k in live}
        scl = [sc_take(sc_cal_all[k], pick[k]) for k in live]
        cal = Cal(sysm, scl, pf_ratios=[ratios(s) for s in scl])
        for ki, k in enumerate(live):
            rows = test_rows[k]
            y = tab["cls"][rows]
            scen = [("clean", None)] + \
                [(f"1g:{C.EVID[g]}:{kd}", [(C.EVID[g], kd, 1.0)]) for g in range(len(C.EVID)) for kd in C.KINDS] + \
                [(f"2g:{C.EVID[a]}+{C.EVID[b]}:{kd}", [(C.EVID[a], kd, 1.0), (C.EVID[b], kd, 1.0)])
                 for a, b in pairs for kd in C.KINDS]
            for name, var in scen:
                sc = deg[k][(var[0][0], var[0][1], 1.0)] if (var and len(var) == 1) else scores_for(sysm, tab, rows, var)
                degr = {v[0] for v in var} if var else set()
                W = {"equal": np.ones((len(C.EVID), len(rows))), "entropy": entropy_w(sc["logp"]),
                     "oracle_drop": np.array([[0.0 if g in degr else 1.0] * len(rows) for g in C.EVID])}
                for method in ("local", "pooled", "pfwcp"):
                    W[method] = decide(cal, method, ki, sc, "adaptive")[1]
                for v in C.VARIANTS_FUSION:
                    fl, W[f"fedscr[{v}]"] = decide(cal, "fedscr", ki, sc, v)
                    if var and len(var) == 1:
                        for gi, g in enumerate(C.EVID):
                            if g not in degr:
                                res["partner_far"].append(dict(seed=seed, client=k, method=f"fedscr[{v}]", degraded=var[0][0],
                                                               kind=var[0][1], group=g, far=fl[gi].mean()))
                if var and len(var) == 1:
                    for method in ("local", "pooled", "pfwcp"):
                        fl = decide(cal, method, ki, sc, "adaptive")[0]
                        for gi, g in enumerate(C.EVID):
                            if g not in degr:
                                res["partner_far"].append(dict(seed=seed, client=k, method=method, degraded=var[0][0],
                                                               kind=var[0][1], group=g, far=fl[gi].mean()))
                for m, w in W.items():
                    res["fusion"].append(dict(seed=seed, client=k, scenario=name.split(":")[0], detail=name, method=m,
                                              f1=macro_f1(y, fused_pred(sc["logp"], w))))
        print(f"[seed {seed}] fusion ({time.time() - t0:.0f}s)")


def new_site(tab, res):
    """Leave one operating condition out; the new site calibrates from unlabelled, partly contaminated windows."""
    folds = C.CONDS[:C.C2_MAX_FOLDS] if len(C.CONDS) > C.C2_MAX_FOLDS else C.CONDS
    for seed in C.SEEDS:
        for new in folds:
            t0 = time.time()
            tr_conds = [c for c in C.CONDS if c != new]
            client = make_clients(tab, seed, conds=tr_conds)
            Ktr = len(tr_conds) * C.N_SHARDS
            sysm = get_system(tab, ("lolo", str(new), seed), client, list(range(Ktr)), seed)
            rng = np.random.default_rng(seed + 7)
            tr_cal = [np.where((client == k) & (tab["block"] == "cal"))[0] for k in range(Ktr)]
            tr_live = [k for k in range(Ktr) if len(tr_cal[k]) >= 5]
            tr_sc = [scores_for(sysm, tab, rng.choice(tr_cal[k], min(C.N_CAL_MAIN, len(tr_cal[k])), replace=False))
                     for k in tr_live]
            donor = Cal(sysm, tr_sc)
            pool = np.where((tab["cond"] == new) & (tab["block"] == "cal"))[0]
            test = np.where((tab["cond"] == new) & (tab["block"] == "test"))[0]
            test = rng.choice(test, min(1500, len(test)), replace=False)
            sc_t = scores_for(sysm, tab, test)
            deg_t = {(g, kd): scores_for(sysm, tab, test, [(g, kd, 1.0)]) for g in C.EVID for kd in C.KINDS}
            client_pf = client.copy()
            client_pf[pool] = Ktr
            ratios = pf_classifier(sysm, tab, client_pf, tr_live + [Ktr], [0] * len(tr_live) + [1], seed)
            pf_cal = Cal(sysm, tr_sc, pf_ratios=[ratios(s)[:, [1]] for s in tr_sc])
            for eps in C.EPS_GRID:
                cal_rows = rng.choice(pool, min(300, len(pool)), replace=False)
                sc_c = scores_for(sysm, tab, cal_rows)
                bad = rng.choice(len(cal_rows), int(round(eps * len(cal_rows))), replace=False)
                for i in bad:
                    sc_set(sc_c, i, scores_for(sysm, tab, cal_rows[i:i + 1],
                                               [(C.EVID[rng.integers(len(C.EVID))], C.KINDS[rng.integers(len(C.KINDS))], 1.0)]))
                clean = np.setdiff1d(np.arange(len(cal_rows)), bad)
                cals = {"pooled_others": (donor, "pooled", 0), "pfwcp_others": (pf_cal, "pfwcp", 0),
                        "naive_local": (TrimCal(sysm, sc_c), "local", 0),
                        "clean_local_oracle": (TrimCal(sysm, sc_take(sc_c, clean)), "local", 0)}
                for mg in ([1.0, 1.5, 2.5, 4.0] if C.RUN_ABLATIONS else [1.5]):
                    cals[f"fedscr_trim[m={mg}]"] = (TrimCal(sysm, sc_c, donor, mg), "local", 0)
                for name, (cl, method, k) in cals.items():
                    far = decide(cl, method, k, sc_t)[0].mean(1)
                    for gi, g in enumerate(C.EVID):
                        pw = np.mean([decide(cl, method, k, deg_t[(g, kd)])[0][gi].mean() for kd in C.KINDS])
                        res["c2"].append(dict(seed=seed, new_cond=str(new), eps=eps, method=name, group=g, far=far[gi],
                                              power=pw, trimmed=float(np.mean(cl.trimmed)) if getattr(cl, "trimmed", None)
                                              else np.nan))
            print(f"[seed {seed}] new site {new} ({time.time() - t0:.0f}s)")


def robustness(tab, res):
    """Manipulated sketches from malicious clients, and an honest client calibrating with a degraded sensor."""
    K = len(C.CONDS) * C.N_SHARDS
    for seed in C.SEEDS:
        t0 = time.time()
        client = make_clients(tab, seed)
        sysm = get_system(tab, ("all", seed), client, list(range(K)), seed)
        rng = np.random.default_rng(seed + 11)
        cal_pool, test_rows, live = _split_rows(tab, client, K)
        picks = {k: rng.choice(cal_pool[k], min(C.N_CAL_MAIN, len(cal_pool[k])), replace=False) for k in live}
        scl = [scores_for(sysm, tab, picks[k]) for k in live]
        tests = {k: (scores_for(sysm, tab, test_rows[k]),
                     {(g, kd): scores_for(sysm, tab, test_rows[k], [(g, kd, 0.6)]) for g in C.EVID for kd in C.KINDS})
                 for k in live}
        base = Cal(sysm, scl)
        byz = list(rng.choice(len(live), min(C.N_BYZ, len(live) - 1), replace=False))

        def run(cal, honest, tag, extra):
            for method in ("pooled", "trimmed_pooled", "fedscr"):
                for ki in honest:
                    sc0, dg = tests[live[ki]]
                    far = decide(cal, method, ki, sc0)[0].mean(1)
                    for gi, g in enumerate(C.EVID):
                        pw = np.mean([decide(cal, method, ki, dg[(g, kd)])[0][gi].mean() for kd in C.KINDS])
                        res[tag].append(dict(seed=seed, method=method, client=live[ki], group=g, far=far[gi], power=pw, **extra))
        honest = [i for i in range(len(live)) if i not in byz]
        for kind, f in C.ATTACKS:
            cal = Cal.__new__(Cal)
            cal.__dict__.update(base.__dict__)
            cal.cache = {}
            cal.sk = {kk: [attack(s, kind, f) if j in byz else s for j, s in enumerate(lst)] for kk, lst in base.sk.items()}
            run(cal, honest, "c3", dict(attack=f"{kind}x{f}"))
        bad = int(rng.integers(len(live)))
        scl2 = list(scl)
        scl2[bad] = scores_for(sysm, tab, picks[live[bad]], [(C.EVID[0], "gain", 1.0)])
        run(Cal(sysm, scl2), [i for i in range(len(live)) if i != bad], "c3_honest", dict(bad_site=live[bad]))
        print(f"[seed {seed}] robustness ({time.time() - t0:.0f}s)")


TAIL_RUNS = {0.90: [("none", 1.0), ("tail", 3.0), ("tail", 10.0), ("tail", 0.5), ("inflate", 10.0), ("deflate", 0.2)],
             0.95: [("none", 1.0), ("tail", 3.0), ("tail", 10.0), ("tail", 30.0), ("tail", 0.5)]}


def tail_manipulation(tab, res):
    """Two malicious clients change only the upper tail of their sketches (above level u0), and the cost of a
    tail aware distance without manipulation."""
    K = len(C.CONDS) * C.N_SHARDS
    for u0, attacks in TAIL_RUNS.items():
        for seed in C.SEEDS:
            t0 = time.time()
            client = make_clients(tab, seed)
            sysm = get_system(tab, ("all", seed), client, list(range(K)), seed)
            rng = np.random.default_rng(seed + 11)        # same draws as the robustness experiment
            cal_pool, test_rows, live = _split_rows(tab, client, K)
            picks = {k: rng.choice(cal_pool[k], min(C.N_CAL_MAIN, len(cal_pool[k])), replace=False) for k in live}
            scl = [scores_for(sysm, tab, picks[k]) for k in live]
            tests = {k: (scores_for(sysm, tab, test_rows[k]),
                         {(g, kd): scores_for(sysm, tab, test_rows[k], [(g, kd, 0.6)]) for g in C.EVID for kd in C.KINDS})
                     for k in live}
            base = Cal(sysm, scl)
            byz = list(rng.choice(len(live), min(C.N_BYZ, len(live) - 1), replace=False))
            honest = [i for i in range(len(live)) if i not in byz]
            for kind, f in attacks:
                cal = Cal.__new__(Cal)
                cal.__dict__.update(base.__dict__)
                cal.cache = {}
                change = (lambda s: attack_tail(s, f, u0)) if kind == "tail" else (lambda s: attack(s, kind, f))
                cal.sk = {kk: [change(s) if j in byz else s for j, s in enumerate(lst)] for kk, lst in base.sk.items()}
                for method in ("pooled", "trimmed_pooled", "fedscr", "fedscr_tail"):
                    for ki in honest:
                        sc0, dg = tests[live[ki]]
                        far = decide(cal, method, ki, sc0)[0].mean(1)
                        for gi, g in enumerate(C.EVID):
                            pw = np.mean([decide(cal, method, ki, dg[(g, kd)])[0][gi].mean() for kd in C.KINDS])
                            res["tail_attack"].append(dict(seed=seed, u0=u0, attack=f"{kind}x{f}", method=method,
                                                           client=live[ki], group=g, far=far[gi], power=pw))
            if u0 == 0.90:
                rng2 = np.random.default_rng(seed + 21)
                for n_cal in (20, 100):
                    pk = {k: rng2.choice(cal_pool[k], min(n_cal, len(cal_pool[k])), replace=False) for k in live}
                    cal = Cal(sysm, [scores_for(sysm, tab, pk[k]) for k in live])
                    for method in ("local", "pooled", "fedscr", "fedscr_tail"):
                        for ki, k in enumerate(live):
                            sc0, dg = tests[k]
                            far = decide(cal, method, ki, sc0)[0].mean(1)
                            for gi, g in enumerate(C.EVID):
                                pw = np.mean([decide(cal, method, ki, dg[(g, kd)])[0][gi].mean() for kd in C.KINDS])
                                res["tail_cal"].append(dict(seed=seed, n_cal=n_cal, method=method, client=k, group=g,
                                                            far=far[gi], power=pw))
            print(f"[seed {seed}] tail manipulation above {u0} ({time.time() - t0:.0f}s)")


def scalability(tab, res):
    if not (C.RUN_SCALING and C.SCAL_SHARDS):
        return
    K = len(C.CONDS) * C.N_SHARDS
    for seed in C.SEEDS[:3]:
        client0 = make_clients(tab, seed)
        sysm = get_system(tab, ("all", seed), client0, list(range(K)), seed)
        for S in C.SCAL_SHARDS:
            client = make_clients(tab, seed + 100, n_shards=S)
            KS = len(C.CONDS) * S
            rng = np.random.default_rng(seed)
            cal_pool, test_rows, live = _split_rows(tab, client, KS)
            scl = [scores_for(sysm, tab, rng.choice(cal_pool[k], min(C.N_CAL_MAIN, len(cal_pool[k])), replace=False))
                   for k in live]
            cal = Cal(sysm, scl)
            for ki, k in enumerate(live):
                sc0 = scores_for(sysm, tab, test_rows[k])
                dg = {(g, kd): scores_for(sysm, tab, test_rows[k], [(g, kd, 0.6)]) for g in C.EVID for kd in C.KINDS}
                for method in ("local", "pooled", "fedscr"):
                    far = decide(cal, method, ki, sc0)[0].mean(1)
                    pw = np.mean([decide(cal, method, ki, dg[(g, kd)])[0][gi].mean()
                                  for gi, g in enumerate(C.EVID) for kd in C.KINDS])
                    res["scaling"].append(dict(seed=seed, n_clients=len(live), method=method, client=k, far=far.mean(), power=pw))
        print(f"[seed {seed}] scalability")


def ablations(tab, res):
    if not C.RUN_ABLATIONS:
        return
    K = len(C.CONDS) * C.N_SHARDS
    for seed in C.SEEDS[:3]:
        client = make_clients(tab, seed)
        rng = np.random.default_rng(seed + 3)
        cal_pool, test_rows, live = _split_rows(tab, client, K)
        picks = {k: rng.choice(cal_pool[k], min(C.N_CAL_MAIN, len(cal_pool[k])), replace=False) for k in live}
        for feat_tag, drop in (("full", False), ("no_envelope", True)):
            sysm = get_system(tab, ("all", seed) if not drop else ("noenv", seed), client, list(range(K)), seed, drop_env=drop)
            scl = [scores_for(sysm, tab, picks[k]) for k in live]
            settings = [("h_c", h, Cal(sysm, scl, h_c=h)) for h in ([0.5, 1.0, 1.5, 3.0, 6.0] if not drop else [C.H_C])]
            if not drop:
                settings += [("grid", gs, Cal(sysm, scl, grid=make_grid(*gs))) for gs in ((10, 11), (19, 21), (46, 101))]
            for name, val, cal in settings:
                for ki, k in enumerate(live):
                    rows = test_rows[k]
                    y = tab["cls"][rows]
                    sc0 = scores_for(sysm, tab, rows)
                    far = decide(cal, "fedscr", ki, sc0)[0].mean(1)
                    dg = {(g, kd): scores_for(sysm, tab, rows, [(g, kd, 0.6)]) for g in C.EVID for kd in C.KINDS}
                    pw = np.mean([decide(cal, "fedscr", ki, dg[(g, kd)])[0][gi].mean()
                                  for gi, g in enumerate(C.EVID) for kd in C.KINDS])
                    f1 = {}
                    for sc_name, var in [("clean", None)] + [("1g", [(g, kd, 1.0)]) for g in C.EVID for kd in C.KINDS[:4]]:
                        sc = scores_for(sysm, tab, rows, var) if var else sc0
                        f1.setdefault(sc_name, []).append(macro_f1(y, fused_pred(sc["logp"], decide(cal, "fedscr", ki, sc)[1])))
                    res["ablation"].append(dict(seed=seed, features=feat_tag, setting=name, value=str(val), client=k,
                                                far=far.mean(), power=pw, f1_clean=np.mean(f1["clean"]),
                                                f1_1g=np.mean(f1["1g"])))
        print(f"[seed {seed}] ablations")
