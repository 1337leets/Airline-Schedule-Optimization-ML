from pathlib import Path
from types import SimpleNamespace
import json

import log

import pandas as pd
import pyomo.environ as pyo


### config #####################################################################################################################################################
data_contents, consts = (cfg := json.load(open(Path(__file__).parent / "config.json", "r", encoding="utf-8"))).get("data_contents", {}), SimpleNamespace(**cfg.get("consts", {}))
data_transfer, data_route = [data_contents[key] for key in ("data_transfer", "data_route")]
################################################################################################################################################################


### helpers ####################################################################################################################################################
################################################################################################################################################################

def _calc_bounds(D, tag, key):
    prog_no, mgn = (D["prog_no_arr"], D["mgn_arr"]) if tag == "arr" else (D["prog_no_dep"], D["mgn_dep"])
    c, w = prog_no[key][0], mgn[key[0]]
    return c - w, c + w    


def _extract_rotations(D):
    arr_local = {(r[data_transfer["col_no_arr"]], r[data_transfer["col_day"]]): r["dur_arr"] for _, r in D["df_transfer_local"].iterrows()}
    dep_local = {(r[data_transfer["col_no_dep"]], r[data_transfer["col_day"]]): r["dur_dep"] for _, r in D["df_transfer_local"].iterrows()}
    by_dep, by_arr = {}, {}
    for (no, day) in D["prog_no_dep"]: by_dep.setdefault(no, set()).add(day)
    for (no, day) in D["prog_no_arr"]: by_arr.setdefault(no, set()).add(day)

    rot = []
    for _, grp in D["df_route"].groupby("Pair"):
        arrs = sorted(set(grp[grp[data_route["col_port_dest"]] == data_transfer["elem_local"]][data_route["col_no"]]))
        deps = sorted(set(grp[grp[data_route["col_port_orig"]] == data_transfer["elem_local"]][data_route["col_no"]]))
        if not deps or not arrs: continue
        for no_dep in deps:
            for day_dep in sorted(by_dep.get(no_dep, ())):
                dur_dep = dep_local.get((no_dep, day_dep))
                if dur_dep is None: continue
                dep_nom = sum(D["prog_no_dep"][(no_dep, day_dep)])
                best = None
                for no_arr in arrs:
                    if no_arr == no_dep and len(grp) > 2: continue
                    for day_arr in sorted(by_arr.get(no_arr, ())):
                        dur_arr = arr_local.get((no_arr, day_arr))
                        if dur_arr is None: continue
                        arr_nom, dur_min = sum(D["prog_no_arr"][(no_arr, day_arr)]), dur_dep + consts.DUR_MIN_PORT + dur_arr
                        if arr_nom >= dep_nom + dur_min and (best is None or arr_nom < best[0]): best = (arr_nom, no_arr, day_arr, dur_min)
                if best is not None: rot.append(((no_dep, day_dep), (best[1], best[2]), best[3]))
    return rot


def _add_hard_cons(m, D):
    m.var_arr = pyo.Var(list(D["prog_no_arr"]), domain=pyo.Integers, bounds=lambda mo, no, day: _calc_bounds(D, "arr", (no, day)))
    m.var_dep = pyo.Var(list(D["prog_no_dep"]), domain=pyo.Integers, bounds=lambda mo, no, day: _calc_bounds(D, "dep", (no, day)))
    m.con_hub, m.bkt = pyo.ConstraintList(), {}
    for var, tag, cap in ((m.var_arr, "arr", consts.CAP_ARR), (m.var_dep, "dep", consts.CAP_DEP)):
        buckets = {}
        for key in var:
            lo, hi = _calc_bounds(D, tag, key)
            bset = range(int(lo // consts.INT_TIM), int(hi // consts.INT_TIM) + 1)
            bv, sv = pyo.Var(bset, within=pyo.Binary), pyo.Var(domain=pyo.Integers, bounds=(0, consts.INT_TIM - 1))
            setattr(m, f"bkt_{tag}_{key[0]}_{key[1]}", bv); setattr(m, f"slk_{tag}_{key[0]}_{key[1]}", sv)
            m.bkt[(tag, key)] = bv
            m.con_hub.add(sum(bv.values()) == 1)
            m.con_hub.add(var[key] == sum(b * consts.INT_TIM * bv[b] for b in bset) + sv)
            for b in bset: buckets.setdefault((key[1], b), []).append(bv[b])
        for vs in buckets.values(): m.con_hub.add(sum(vs) <= cap)
    m.con_rot = pyo.ConstraintList()
    for k_dep, k_arr, need in D["rot"]: m.con_rot.add(m.var_arr[k_arr] + D["prog_no_arr"][k_arr][1] >= m.var_dep[k_dep] + D["prog_no_dep"][k_dep][1] + need)
    m.con_reg, m.reg_ref, ri = pyo.ConstraintList(), {}, 0
    for tag, var, prog_no in (("arr", m.var_arr, D["prog_no_arr"]), ("dep", m.var_dep, D["prog_no_dep"])):
        by_no = {}
        for (no, day) in prog_no: by_no.setdefault(no, []).append(day)
        for no, days in by_no.items():
            if len(days) < 2: continue
            ri += 1; rv = pyo.Var(); setattr(m, f"reg_ref_{ri}", rv)
            m.reg_ref[(tag, no)] = rv
            for day in days: m.con_reg.add(var[(no, day)] - rv <= consts.DEV_TIM_PROG / 2); m.con_reg.add(rv - var[(no, day)] <= consts.DEV_TIM_PROG / 2)


def _to_time(d):
    d = round(d) % 1440
    return f"{d // 60:02d}:{d % 60:02d}"

################################################################################################################################################################
################################################################################################################################################################


### exec #######################################################################################################################################################
################################################################################################################################################################

@log.build
def build(D):
    D["rot"] = _extract_rotations(D)
    m = pyo.ConcreteModel()
    _add_hard_cons(m, D)

    cand = []
    for (od, day), mk in D["markets"].items():
        for r in mk["rows"]:
            k_arr, k_dep = (r._asdict()[data_transfer["col_no_arr"]], day), (r._asdict()[data_transfer["col_no_dep"]], day)
            if k_arr not in D["prog_no_arr"] or k_dep not in D["prog_no_dep"]: continue
            (arr_min, arr_max), (dep_min, dep_max) = _calc_bounds(D, "arr", k_arr), _calc_bounds(D, "dep", k_dep)
            if dep_max - arr_min < consts.DUR_MIN_TRANSF or dep_min - arr_max > consts.DUR_MAX_TRANSF: continue
            cand.append((od, day, k_arr[0], k_dep[0], float(r.dur_arr), float(r.dur_dep)))
    m.C = pyo.Set(initialize=range(len(cand)))
    m.var_y, m.var_below, m.var_above = pyo.Var(m.C, domain=pyo.Binary), pyo.Var(m.C, domain=pyo.Binary), pyo.Var(m.C, domain=pyo.Binary)

    _gap_expr = lambda cand_i: m.var_dep[(cand[cand_i][3], cand[cand_i][1])] - m.var_arr[(cand[cand_i][2], cand[cand_i][1])]
    _gtg_expr = lambda cand_i: cand[cand_i][4] + _gap_expr(cand_i) + cand[cand_i][5]
    def _calc_gap_bounds(cand_i):
        _, day, no_arr, no_dep, dur_arr, dur_dep = cand[cand_i]
        (arr_min, arr_max), (dep_min, dep_max) = _calc_bounds(D, "arr", (no_arr, day)), _calc_bounds(D, "dep", (no_dep, day))
        gap_min, gap_max = dep_min - arr_max, dep_max - arr_min
        return gap_min, gap_max, dur_arr + gap_min + dur_dep, dur_arr + gap_max + dur_dep

    m.con_ind = pyo.ConstraintList()
    for cand_i in m.C:
        gap_min, gap_max, _, _ = _calc_gap_bounds(cand_i)
        m.con_ind.add(m.var_y[cand_i] == 1 - m.var_below[cand_i] - m.var_above[cand_i])
        m.con_ind.add(m.var_below[cand_i] + m.var_above[cand_i] <= 1)
        m.con_ind.add(_gap_expr(cand_i) <= (consts.DUR_MIN_TRANSF - consts.EPS) + (max(0., gap_max - consts.DUR_MIN_TRANSF + consts.EPS) + 1) * (1 - m.var_below[cand_i]))
        m.con_ind.add(_gap_expr(cand_i) >= consts.DUR_MIN_TRANSF - (max(0., consts.DUR_MIN_TRANSF - gap_min) + 1) * m.var_below[cand_i])
        m.con_ind.add(_gap_expr(cand_i) >= (consts.DUR_MAX_TRANSF + consts.EPS) - (max(0., consts.DUR_MAX_TRANSF + consts.EPS - gap_min) + 1) * (1 - m.var_above[cand_i]))
        m.con_ind.add(_gap_expr(cand_i) <= consts.DUR_MAX_TRANSF + (max(0., gap_max - consts.DUR_MAX_TRANSF) + 1) * m.var_above[cand_i])

    mkt_c = {}
    for cand_i, (od, day, *_ ) in enumerate(cand): mkt_c.setdefault((od, day), []).append(cand_i)

    m.var_conn = pyo.Var(list(mkt_c), domain=pyo.NonNegativeReals)
    m.con_cap = pyo.ConstraintList()
    for key, cs in mkt_c.items():
        k_expr = sum(m.var_y[cand_i] for cand_i in cs)
        for t in range(1, len(cs) + 1):
            slope = .5 ** (t - 1)
            m.con_cap.add(m.var_conn[key] <= slope * k_expr + (2 * (1 - .5 ** t) - slope * t))

    m.var_rank = pyo.Var(list(D["markets"]), domain=pyo.Reals)
    m.con_rank, m.z, m.beat, m.rk = pyo.ConstraintList(), {}, {}, {}
    for (od, day), mk in D["markets"].items():
        cs, n = mkt_c.get((od, day), []), mk["n"]
        beat_vars = []
        for ki, tcomp in enumerate(mk["comp"]):
            zs = []
            for cand_i in cs:
                _, _, gap_min_j, gap_max_j = _calc_gap_bounds(cand_i)
                if gap_min_j > tcomp: continue
                zv = pyo.Var(within=pyo.Binary); m.add_component(f"z_{od}_{day}_{ki}_{cand_i}", zv)
                m.z[((od, day, ki), cand_i)] = zv
                m.con_rank.add(zv <= m.var_y[cand_i])
                m.con_rank.add(_gtg_expr(cand_i) <= tcomp + (max(0., gap_max_j - tcomp) + 1) * (1 - zv))
                zs.append(zv)
            bv = pyo.Var(within=pyo.Binary); m.add_component(f"beat_{od}_{day}_{ki}", bv)
            m.beat[(od, day, ki)] = bv
            m.con_rank.add(bv <= (sum(zs) if zs else 0))
            beat_vars.append(bv)
        tmax = min(len(beat_vars), n - 1) if n > 1 else 0
        rk = {t: pyo.Var(within=pyo.Binary) for t in range(tmax + 1)}
        for t, v in rk.items(): m.add_component(f"rk_{od}_{day}_{t}", v)
        m.rk[(od, day)] = rk
        m.con_rank.add(sum(rk.values()) == 1)
        m.con_rank.add(sum(beat_vars) <= sum(t * v for t, v in rk.items()))
        m.con_rank.add(sum(beat_vars) >= sum(t * v for t, v in rk.items()))
        m.con_rank.add(m.var_rank[(od, day)] == sum(rk[t] * D["dict_weight"].get((n, mk["init"], max(1, n - t)), 0.) for t in rk))

    m.con_bal, m.hbo, m.bJbs, seen = pyo.ConstraintList(), {}, {}, set()
    for (od, day) in D["markets"]:
        o, d = od.split("-"); rev = f"{d}-{o}"
        if (rev, day) not in D["markets"] or (od, day) in seen: continue
        seen.add((od, day)); seen.add((rev, day))
        cands_fwd, cands_rev = mkt_c.get((od, day), []), mkt_c.get((rev, day), [])
        if not cands_fwd or not cands_rev: continue
        count_fwd, count_rev = sum(m.var_y[cand_i] for cand_i in cands_fwd), sum(m.var_y[cand_i] for cand_i in cands_rev)
        has_fwd, has_rev, has_both = (pyo.Var(within=pyo.Binary) for _ in range(3))
        m.add_component(f"has_fwd_{od}_{day}", has_fwd); m.add_component(f"has_rev_{od}_{day}", has_rev); m.add_component(f"has_both_{od}_{day}", has_both)
        m.hbo[(od, day)] = (has_fwd, has_rev, has_both)
        m.con_bal.add(count_fwd <= len(cands_fwd) * has_fwd); m.con_bal.add(has_fwd <= count_fwd)
        m.con_bal.add(count_rev <= len(cands_rev) * has_rev); m.con_bal.add(has_rev <= count_rev)
        m.con_bal.add(has_both <= has_fwd); m.con_bal.add(has_both <= has_rev); m.con_bal.add(has_both >= has_fwd + has_rev - 1)
        slack_cnt = max(len(cands_fwd), len(cands_rev)) + 1.
        m.con_bal.add(count_fwd - count_rev <= consts.RTO_2WAY * (count_fwd + count_rev) + slack_cnt * (1 - has_both))
        m.con_bal.add(count_rev - count_fwd <= consts.RTO_2WAY * (count_fwd + count_rev) + slack_cnt * (1 - has_both))
        gtg_lo = min(_calc_gap_bounds(cand_i)[2] for cand_i in cands_fwd + cands_rev)
        gtg_hi = max(_calc_gap_bounds(cand_i)[3] for cand_i in cands_fwd + cands_rev)
        best_gtg, argmin_flags = {}, {}
        for direction, cs, active_flag in (("fwd", cands_fwd, has_fwd), ("rev", cands_rev, has_rev)):
            best_gtg_var = pyo.Var(bounds=(gtg_lo, gtg_hi)); m.add_component(f"best_gtg_{direction}_{od}_{day}", best_gtg_var)
            flag_vars = []
            for cand_i in cs:
                flag_var = pyo.Var(within=pyo.Binary); m.add_component(f"argmin_{direction}_{od}_{day}_{cand_i}", flag_var); flag_vars.append(flag_var)
                argmin_flags[(direction, cand_i)] = flag_var
                _, _, gap_min_c, gap_max_c = _calc_gap_bounds(cand_i)
                m.con_bal.add(flag_var <= m.var_y[cand_i])
                m.con_bal.add(best_gtg_var <= _gtg_expr(cand_i) + (gtg_hi - gap_min_c) * (1 - m.var_y[cand_i]))
                m.con_bal.add(best_gtg_var >= _gtg_expr(cand_i) - (gap_max_c - gtg_lo) * (1 - flag_var))
            m.con_bal.add(sum(flag_vars) == active_flag)
            best_gtg[direction] = best_gtg_var
        m.bJbs[(od, day)] = (best_gtg, argmin_flags)
        slack_gtg = (gtg_hi - gtg_lo) + 1.
        m.con_bal.add(best_gtg["fwd"] - best_gtg["rev"] <= consts.DEV_GTG_2WAY + slack_gtg * (1 - has_both))
        m.con_bal.add(best_gtg["rev"] - best_gtg["fwd"] <= consts.DEV_GTG_2WAY + slack_gtg * (1 - has_both))

    rho = lambda od: D["dict_demand"].get(tuple(od.split("-")), 0.)
    m.obj = pyo.Objective(expr=sum(rho(od) * (m.var_conn[(od, day)] if (od, day) in mkt_c else 0) + rho(od) * m.var_rank[(od, day)]
                                   for (od, day) in D["markets"]), sense=pyo.maximize)
    return m, cand, mkt_c


@log.start
def start(D, time_limit = consts.LIM_TIM_START):
    fm = pyo.ConcreteModel()
    _add_hard_cons(fm, D)
    fm.dev_arr, fm.dev_dep = pyo.Var(list(D["prog_no_arr"]), domain=pyo.NonNegativeReals), pyo.Var(list(D["prog_no_dep"]), domain=pyo.NonNegativeReals)
    fm.con_dev = pyo.ConstraintList()
    for key, (c, _) in D["prog_no_arr"].items(): fm.con_dev.add(fm.var_arr[key] - c <= fm.dev_arr[key]); fm.con_dev.add(c - fm.var_arr[key] <= fm.dev_arr[key])
    for key, (c, _) in D["prog_no_dep"].items(): fm.con_dev.add(fm.var_dep[key] - c <= fm.dev_dep[key]); fm.con_dev.add(c - fm.var_dep[key] <= fm.dev_dep[key])
    fm.obj = pyo.Objective(expr=sum(fm.dev_arr.values()) + sum(fm.dev_dep.values()), sense=pyo.minimize)

    opt = pyo.SolverFactory("appsi_highs")
    opt.config.time_limit, opt.config.mip_gap, opt.config.stream_solver = time_limit, 0.5, True
    opt.solve(fm)
    arr0 = {key: pyo.value(fm.var_arr[key]) for key in D["prog_no_arr"]}
    dep0 = {key: pyo.value(fm.var_dep[key]) for key in D["prog_no_dep"]}
    return arr0, dep0, pyo.value(fm.obj)


@log.solve
def solve(D, m, cand, time_limit=consts.LIM_TIM_SOLVE, mip_gap=.01, warm=True):
    arr0, dep0, _ = start(D)
    for key, v in arr0.items(): m.var_arr[key].value = round(v)
    for key, v in dep0.items(): m.var_dep[key].value = round(v)
    for (tag, key), bv in m.bkt.items():
        chosen = int(round((arr0 if tag == "arr" else dep0)[key]) // consts.INT_TIM)
        for b in bv: bv[b].value = 1 if b == chosen else 0
        getattr(m, f"slk_{tag}_{key[0]}_{key[1]}").value = round((arr0 if tag == "arr" else dep0)[key]) - chosen * consts.INT_TIM

    gtg0 = {}
    for cand_i, (od, day, no_arr, no_dep, dur_arr, dur_dep) in enumerate(cand):
        dur_gap = round(dep0[(no_dep, day)]) - round(arr0[(no_arr, day)])
        below, above = int(dur_gap < consts.DUR_MIN_TRANSF), int(dur_gap > consts.DUR_MAX_TRANSF)
        m.var_below[cand_i].value, m.var_above[cand_i].value = below, above
        m.var_y[cand_i].value = 1 - below - above
        gtg0[cand_i] = dur_arr + dur_gap + dur_dep

    for (tag, no), rv in m.reg_ref.items():
        vals = [(arr0 if tag == "arr" else dep0)[(no, day)] for (n2, day) in (D["prog_no_arr"] if tag == "arr" else D["prog_no_dep"]) if n2 == no]
        rv.value = round(sum(vals) / len(vals))

    mkt_c = {}
    for cand_i, (od, day, *_ ) in enumerate(cand): mkt_c.setdefault((od, day), []).append(cand_i)

    for key, cs in mkt_c.items():
        k0 = int(sum(m.var_y[cand_i].value for cand_i in cs))
        m.var_conn[key].value = 2 * (1 - .5 ** k0) if cs else 0.

    for (od, day), mk in D["markets"].items():
        cs, n = mkt_c.get((od, day), []), mk["n"]
        beat_vars0 = []
        for ki, tcomp in enumerate(mk["comp"]):
            beaten = 0
            for cand_i in cs:
                zv = m.z.get(((od, day, ki), cand_i))
                if zv is None: continue
                zval = 1 if m.var_y[cand_i].value > .5 and gtg0[cand_i] <= tcomp else 0
                zv.value = zval
                beaten = max(beaten, zval)
            m.beat[(od, day, ki)].value = beaten
            beat_vars0.append(beaten)
        tmax = min(len(beat_vars0), n - 1) if n > 1 else 0
        t0 = min(sum(beat_vars0), tmax)
        for t, v in m.rk[(od, day)].items(): v.value = 1 if t == t0 else 0
        m.var_rank[(od, day)].value = D["dict_weight"].get((n, mk["init"], max(1, n - t0)), 0.)

    for (od, day), (has_fwd, has_rev, has_both) in m.hbo.items():
        o, d = od.split("-"); rev = f"{d}-{o}"
        cands_fwd, cands_rev = mkt_c.get((od, day), []), mkt_c.get((rev, day), [])
        active_fwd, active_rev = [cand_i for cand_i in cands_fwd if m.var_y[cand_i].value > .5], [cand_i for cand_i in cands_rev if m.var_y[cand_i].value > .5]
        has_fwd.value, has_rev.value = int(bool(active_fwd)), int(bool(active_rev))
        has_both.value = int(bool(active_fwd) and bool(active_rev))
        best_gtg, argmin_flags = m.bJbs[(od, day)]
        for direction, cs, active in (("fwd", cands_fwd, active_fwd), ("rev", cands_rev, active_rev)):
            for cand_i in cs: argmin_flags[(direction, cand_i)].value = 1 if cand_i in active else 0
            best_gtg[direction].value = min((gtg0[cand_i] for cand_i in active), default=best_gtg[direction].bounds[0])

    solver = consts.NAME_SOLVER
    opt = pyo.SolverFactory(solver)
    if solver == "gurobi":
        opt.options["TimeLimit"], opt.options["MIPGap"] = time_limit, mip_gap
        opt.options["Seed"], opt.options["Threads"] = 0, 8
        opt.options["NoRelHeurTime"], opt.options["MIPFocus"] = 60, 1
        res = opt.solve(m, tee=True, warmstart=warm)
    else:
        opt.config.time_limit, opt.config.mip_gap, opt.config.stream_solver = time_limit, mip_gap, True
        res = opt.solve(m)
    return res


@log.extract
def extract(D, m, cand):
    arr_c, dep_c = {key: pyo.value(m.var_arr[key]) for key in D["prog_no_arr"]}, {key: pyo.value(m.var_dep[key]) for key in D["prog_no_dep"]}

    sel = [dict(od=od, day=day, no_arr=no_arr, no_dep=no_dep,
                t_arr=round(arr_c[(no_arr, day)] % 1440, 1), t_dep=round(dep_c[(no_dep, day)] % 1440, 1),
                t_arr_time=_to_time(arr_c[(no_arr, day)]), t_dep_time=_to_time(dep_c[(no_dep, day)]),
                gtg=round(dur_arr + (dep_c[(no_dep, day)] - arr_c[(no_arr, day)]) + dur_dep, 1))
           for cand_i, (od, day, no_arr, no_dep, dur_arr, dur_dep) in enumerate(cand) if pyo.value(m.var_y[cand_i]) > .5]

    times = [dict(tip="arr", no=no, day=day, nominal=round(D["prog_no_arr"][(no, day)][0] % 1440, 1),
                  nominal_time=_to_time(D["prog_no_arr"][(no, day)][0]), new=round(clock % 1440, 1), new_time=_to_time(clock))
             for (no, day), clock in arr_c.items()]
    times += [dict(tip="dep", no=no, day=day, nominal=round(D["prog_no_dep"][(no, day)][0] % 1440, 1),
                   nominal_time=_to_time(D["prog_no_dep"][(no, day)][0]), new=round(clock % 1440, 1), new_time=_to_time(clock))
              for (no, day), clock in dep_c.items()]

    per_mkt = {}
    for (od, day), mk in D["markets"].items():
        gtgs = []
        for r in mk["rows"]:
            k_arr, k_dep = (r._asdict()[data_transfer["col_no_arr"]], day), (r._asdict()[data_transfer["col_no_dep"]], day)
            if k_arr in arr_c and k_dep in dep_c: gtgs.append(r.dur_arr + (dep_c[k_dep] - arr_c[k_arr]) + r.dur_dep)
        beaten = sum(1 for tc in mk["comp"] if gtgs and min(gtgs) <= tc)
        per_mkt[(od, day)] = dict(k=len(gtgs), beat=beaten, rank=max(1, mk["n"] - beaten))

    return dict(sel=sel, times=times, per_mkt=per_mkt, obj=pyo.value(m.obj))


@log.export
def export(result, out_path=None):
    out_path = out_path or Path(__file__).parent / "result" / "result.xlsx"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_sel = pd.DataFrame(result["sel"])
    df_tim = pd.DataFrame(result["times"])
    df_mkt = pd.DataFrame([dict(od=od, day=day, **v) for (od, day), v in result["per_mkt"].items()])
    df_obj = pd.DataFrame([dict(objective=result["obj"])])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        df_sel.to_excel(xl, sheet_name="secilen_baglantilar", index=False)
        df_tim.to_excel(xl, sheet_name="ayarlanan_saatler", index=False)
        df_mkt.to_excel(xl, sheet_name="od_rakip_sonuclari", index=False)
        df_obj.to_excel(xl, sheet_name="ozet", index=False)
    return out_path

################################################################################################################################################################
################################################################################################################################################################