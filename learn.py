"""Train a model that guides the solver, using synthetic networks only.

For each candidate connection, a gradient boosting classifier predicts whether the
best schedule found keeps it valid. The most confident predictions are fixed before
solving, which leaves the solver fewer decisions. The result is then compared with
solving the full model, under the same time limit, on networks not used in training.

Run:  python learn.py
"""
from contextlib import redirect_stdout
from pathlib import Path
import io
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "data"))

from generate_dummy_data import build as build_network
from preprocess import preprocess
import model

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

C = model.consts
TRAIN_SEEDS, TEST_SEEDS = range(100, 130), range(900, 910)
TRAIN_SPOKES, TEST_SPOKES = (6, 7), 8
TIME_TRAIN, TIME_TEST = 20, 30
P_OFF, P_ON = .05, .95
FEATURES = ["gap", "gap_min", "gap_max", "to_min", "to_max", "trip", "competitors", "init_rank", "beats_now", "beats_best", "demand", "market_size", "flight_load"]


def _solve(D, m, cand, limit):
    with redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        res = model.solve(D, m, cand, time_limit=limit, log_output=False)
        spent = time.perf_counter() - t0
    return pyo.value(m.obj), spent, str(res.solver.termination_condition)


def _instance(seed, spokes):
    with redirect_stdout(io.StringIO()):
        data = build_network(seed, spokes)
        for col in ("col_dur_arr", "col_dur_dep"):
            name = model.data_transfer[col]
            data["data_transfer"][name] = pd.to_datetime(data["data_transfer"][name].astype(str), format="%H:%M:%S")
        D = preprocess(data)
        m, cand, _ = model.build(D)
    return D, m, cand


def features(D, cand):
    """One row per candidate connection, using only what is known before solving."""
    load, size = {}, {}
    for od, day, no_arr, no_dep, *_ in cand:
        for key in ((no_arr, day, "a"), (no_dep, day, "d")): load[key] = load.get(key, 0) + 1
        size[(od, day)] = size.get((od, day), 0) + 1
    rows = []
    for od, day, no_arr, no_dep, d1, d2 in cand:
        mk = D["markets"][(od, day)]
        off = model._off(D, no_arr, no_dep, day)
        arr, dep = D["prog_no_arr"][(no_arr, day)][0], D["prog_no_dep"][(no_dep, day)][0]
        (a0, a1), (d0, d1_) = model._calc_bounds(D, "arr", (no_arr, day)), model._calc_bounds(D, "dep", (no_dep, day))
        gap, gap_min, gap_max = dep + off - arr, d0 + off - a1, d1_ + off - a0
        rows.append(dict(gap=gap, gap_min=gap_min, gap_max=gap_max, to_min=gap - C.DUR_MIN_TRANSF, to_max=C.DUR_MAX_TRANSF - gap, trip=d1 + gap + d2,
                         competitors=mk["n"], init_rank=mk["init"], beats_now=sum(c >= d1 + gap + d2 for c in mk["comp"]), beats_best=sum(c >= d1 + gap_min + d2 for c in mk["comp"]),
                         demand=D["dict_demand"].get(tuple(od.split("-")), 0.), market_size=size[(od, day)], flight_load=load[(no_arr, day, "a")] + load[(no_dep, day, "d")]))
    return pd.DataFrame(rows, columns=FEATURES)


def collect(seeds, spokes_list):
    X, y, group = [], [], []
    for i, seed in enumerate(seeds):
        spokes = spokes_list[i % len(spokes_list)]
        D, m, cand = _instance(seed, spokes)
        try:
            _solve(D, m, cand, TIME_TRAIN)
            labels = [int(pyo.value(m.var_y[c]) > .5) for c in m.C]
        except Exception as e:
            print(f"  seed {seed}: skipped, {type(e).__name__}")
            continue
        X.append(features(D, cand)); y += labels; group += [seed] * len(cand)
        print(f"  seed {seed}: {spokes} spokes per side, {len(cand)} candidates")
    return pd.concat(X, ignore_index=True), np.array(y), np.array(group)


def main():
    print("Solving training networks")
    X, y, _ = collect(TRAIN_SEEDS, TRAIN_SPOKES)
    clf = GradientBoostingClassifier(random_state=0).fit(X, y)

    print("\nTesting on unseen networks")
    rows, X_test, y_test = [], [], []
    for seed in TEST_SEEDS:
        D, full, cand = _instance(seed, TEST_SPOKES)
        obj_full, t_full, status_full = _solve(D, full, cand, TIME_TEST)
        F = features(D, cand)
        prob = clf.predict_proba(F)[:, 1]
        X_test.append(prob); y_test += [int(pyo.value(full.var_y[c]) > .5) for c in full.C]

        D2, guided, cand2 = _instance(seed, TEST_SPOKES)
        off, on = [i for i in guided.C if prob[i] < P_OFF], [i for i in guided.C if prob[i] > P_ON]
        guided.con_ml = pyo.Constraint(off + on, rule=lambda mo, i: mo.var_y[i] == (1 if prob[i] > P_ON else 0))
        t0 = time.perf_counter()
        try:
            obj_ml, t_ml, status_ml = _solve(D2, guided, cand2, TIME_TEST)
        except Exception as e:
            obj_ml, t_ml, status_ml = float("nan"), time.perf_counter() - t0, type(e).__name__
        rows.append(dict(seed=seed, candidates=len(cand), fixed=len(off) + len(on), obj_full=round(obj_full, 2), obj_ml=round(obj_ml, 2), sec_full=round(t_full, 1), sec_ml=round(t_ml, 1), full=status_full, ml=status_ml))
        print(" ", rows[-1])

    res = pd.DataFrame(rows)
    prob = np.concatenate(X_test)
    print(f"\nCandidates that stay valid: {np.mean(y_test):.1%}")
    print(f"ROC AUC {roc_auc_score(y_test, prob):.3f}, average precision {average_precision_score(y_test, prob):.3f}")
    fixed = [(p, t) for p, t in zip(prob, y_test) if p < P_OFF or p > P_ON]
    print(f"Fixed: {len(fixed)} of {len(prob)}, agreeing with the full solve on {np.mean([(p > P_ON) == bool(t) for p, t in fixed]):.1%}")
    ok = res.dropna(subset=["obj_ml"])
    print(f"Median solve time: full {res.sec_full.median():.1f} s, guided {ok.sec_ml.median():.1f} s")
    print(f"Networks where guided beats full: {(ok.obj_ml > ok.obj_full).sum()}, ties {(ok.obj_ml == ok.obj_full).sum()}, of {len(ok)}")
    print(f"Total objective: full {ok.obj_full.sum():.2f}, guided {ok.obj_ml.sum():.2f} ({ok.obj_ml.sum() / ok.obj_full.sum() - 1:+.2%}), failed {len(res) - len(ok)}")
    out = Path(__file__).parent / "result"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "learn_results.csv", index=False)
    return res


if __name__ == "__main__":
    main()
