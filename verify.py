"""Check result/result.xlsx without using the optimization model.

Rebuilds every leg time from the raw data and the adjusted clock times, then tests
each operational rule and recomputes the objective. It also scores the original
schedule, to show what the optimization gained.

Run after run.py:  python verify.py
"""
from pathlib import Path
import json
import sys

import pandas as pd

HERE = Path(__file__).parent
cfg = json.load(open(HERE / "config.json", encoding="utf-8"))
C, T, R, D, F = cfg["consts"], cfg["data_contents"]["data_transfer"], cfg["data_contents"]["data_route"], cfg["data_contents"]["data_demand"], cfg["data_contents"]["data_func"]


def minutes(s):
    t = pd.to_datetime(s.astype(str), format="mixed")
    return t.dt.hour * 60 + t.dt.minute


def load():
    get = lambda name: pd.read_excel(HERE / "data" / cfg["data_files"][name])
    df = get("data_transfer")
    ref = df[T["col_time_arr"]].min().normalize()
    df["arr"], df["dep"] = [(df[T[c]] - ref).dt.total_seconds() / 60 for c in ("col_time_arr", "col_time_dep")]
    df["d1"], df["d2"] = minutes(df[T["col_dur_arr"]]), minutes(df[T["col_dur_dep"]])
    func = {(int(a), int(b), int(c)): w for a, b, c, w in get("data_func")[[F["n_comp"], F["rank_init"], F["rank_ulti"], F["weight"]]].values}
    demand = get("data_demand").set_index([D["col_port_orig"], D["col_port_dest"]])[D["col_demand"]].to_dict()
    return df, get("data_route"), func, demand


def score(df, func, demand, shift_arr, shift_dep):
    """Objective for a schedule given as minute shifts per (flight, day) leg."""
    total, cap = 0., max(k[0] for k in func)
    best_by_market = {}
    for (od, day), g in df.groupby([T["col_od"], T["col_day"]]):
        loc = g[g[T["col_port_transf"]] == T["elem_local"]]
        if loc.empty: continue
        comp = (g.loc[g[T["col_port_transf"]] != T["elem_local"], "d1"] + g["d2"] + g["dep"] - g["arr"]).dropna().tolist()
        n = min(len(comp) + 1, cap)
        best0 = (loc.d1 + loc.d2 + loc.dep - loc.arr).min()
        init = min(max(1, n - sum(c >= best0 for c in comp)), n)
        valid = []
        for _, r in loc.iterrows():
            a = r.arr + shift_arr.get((r[T["col_no_arr"]], day), 0)
            d = r.dep + shift_dep.get((r[T["col_no_dep"]], day), 0)
            if C["DUR_MIN_TRANSF"] <= d - a <= C["DUR_MAX_TRANSF"]: valid.append(r.d1 + (d - a) + r.d2)
        best = min(valid) if valid else None
        beaten = sum(c >= best for c in comp) if valid else 0
        rho = demand.get(tuple(od.split("-")), 0.)
        total += rho * 2 * (1 - .5 ** len(valid)) + rho * func.get((n, init, max(1, n - min(beaten, n - 1))), 0.)
        best_by_market[(od, day)] = (len(valid), best)
    return total, best_by_market


def main():
    df, route, func, demand = load()
    res = pd.read_excel(HERE / "result" / "result.xlsx", sheet_name=None)
    times, reported = res["adjusted_times"], float(res["summary"]["objective"].iloc[0])
    loc = df[df[T["col_port_transf"]] == T["elem_local"]]
    nom = {"arr": loc.groupby([T["col_no_arr"], T["col_day"]])["arr"].min(), "dep": loc.groupby([T["col_no_dep"], T["col_day"]])["dep"].min()}
    shift, abs_new = {"arr": {}, "dep": {}}, {"arr": {}, "dep": {}}
    for _, r in times.iterrows():
        key = (r["no"], r["day"])
        base = nom[r["kind"]][key]
        delta = (r["new"] - base % 1440 + 720) % 1440 - 720
        shift[r["kind"]][key], abs_new[r["kind"]][key] = delta, base + delta

    fails = []
    def check(name, bad, detail=""):
        print(f"[{'ok' if not bad else 'FAIL'}] {name}{' ' + detail if detail else ''}")
        if bad: fails.append(name)

    for kind, cap in (("arr", C["CAP_ARR"]), ("dep", C["CAP_DEP"])):
        slots = pd.Series(abs_new[kind]).floordiv(C["INT_TIM"]).value_counts()
        check(f"Hub capacity {kind} <= {cap} per {C['INT_TIM']} min", (slots > cap).any(), f"(peak {slots.max()})")

    spread = 0.
    for kind in ("arr", "dep"):
        by_no = {}
        for (no, _), v in abs_new[kind].items(): by_no.setdefault(no, []).append(v % 1440)
        for vals in by_no.values():
            vals = [(v - vals[0] + 720) % 1440 - 720 for v in vals]
            spread = max(spread, max(vals) - min(vals))
    check(f"Same flight stays within {C['DEV_TIM_PROG']} min across days", spread > C["DEV_TIM_PROG"] + 1, f"(widest {spread:.0f})")

    total, markets = score(df, func, demand, shift["arr"], shift["dep"])
    baseline, markets0 = score(df, func, demand, {}, {})

    worst = 0.
    for _, g in route.groupby(R["col_pairs"]):
        outs = g[g[R["col_port_orig"]] == T["elem_local"]][R["col_no"]]
        ins = g[g[R["col_port_dest"]] == T["elem_local"]][R["col_no"]]
        for no_dep in outs:
            for (no, day), dep_abs in abs_new["dep"].items():
                if no != no_dep: continue
                r = loc[(loc[T["col_no_dep"]] == no) & (loc[T["col_day"]] == day)].iloc[0]
                after = [(a, n2) for (n2, d2), a in abs_new["arr"].items() if n2 in set(ins)]
                nominal_after = [a for (n2, d2), a in nom["arr"].items() if n2 in set(ins) and a >= r.dep + r.d2 + C["DUR_MIN_PORT"]]
                if not nominal_after: continue
                first = min(nominal_after)
                match = [abs_new["arr"][k] for k, a in nom["arr"].items() if a == first and k[0] in set(ins)]
                back = loc[loc[T["col_no_arr"]].isin(ins)].iloc[0].d1
                worst = min(worst, match[0] - (dep_abs + r.d2 + C["DUR_MIN_PORT"] + back))
    check(f"Aircraft rotation leaves {C['DUR_MIN_PORT']} min on the ground", worst < -1, f"(smallest margin {worst:.0f})")

    bad = dev = 0
    for (od, day), (k, best) in markets.items():
        o, d = od.split("-")
        rev = markets.get((f"{d}-{o}", day))
        if not rev or not k or not rev[0]: continue
        bad += abs(k - rev[0]) > C["RTO_2WAY"] * (k + rev[0]) + 1e-6
        dev = max(dev, abs(best - rev[1]))
    check(f"Both directions balanced within {C['RTO_2WAY']:.0%}", bad > 0)
    check(f"Best trip times of both directions within {C['DEV_GTG_2WAY']} min", dev > C["DEV_GTG_2WAY"] + 1, f"(widest {dev:.0f})")
    check("Recomputed objective matches the solver", abs(total - reported) > .01 * max(1, abs(reported)), f"({total:.2f} vs {reported:.2f})")

    served0 = sum(1 for k, _ in markets0.values() if k)
    served = sum(1 for k, _ in markets.values() if k)
    print(f"\nOriginal schedule: objective {baseline:.2f}, {served0} markets with a valid connection")
    print(f"Optimized schedule: objective {total:.2f}, {served} markets with a valid connection")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
