from types import SimpleNamespace
from pathlib import Path
import json

import log

import pandas as pd


### config #####################################################################################################################################################
data_contents, consts = (cfg := json.load(open(Path(__file__).parent / "config.json", "r", encoding="utf-8"))).get("data_contents", {}), SimpleNamespace(**cfg.get("consts", {}))
data_transfer, data_demand, data_func = [data_contents[key] for key in ("data_transfer", "data_demand", "data_func")]
################################################################################################################################################################


### helpers ####################################################################################################################################################
################################################################################################################################################################

def _add_crit_data(df):
    ref = df[data_transfer["col_time_arr"]].min().normalize()
    df["time_arr"] = (df[data_transfer["col_time_arr"]] - ref).dt.total_seconds() / 60
    df["time_dep"] = (df[data_transfer["col_time_dep"]] - ref).dt.total_seconds() / 60

    for col, out in [(data_transfer["col_dur_arr"], "dur_arr"), (data_transfer["col_dur_dep"], "dur_dep")]:
        d = df[col]
        t = d if pd.api.types.is_datetime64_any_dtype(d) else pd.to_datetime(d, dayfirst=True)
        df[out] = t.dt.hour * 60 + t.dt.minute

    df["dur_gap"] = df["time_dep"] - df["time_arr"]
    df["dur_gtg"] = df["dur_arr"] + df["dur_dep"] + df["dur_gap"]
    return df


def _get_markets(df, cap_comp):
    markets = {}
    for (od, day), space in df.groupby([data_transfer[k] for k in ("col_od", "col_day")]):
        local = space[space[data_transfer["col_port_transf"]] == data_transfer["elem_local"]]
        if local.empty: continue

        dur_gtg_local_min = local["dur_gtg"].min()
        dur_gtg_comp = space.loc[space[data_transfer["col_port_transf"]] != data_transfer["elem_local"], "dur_gtg"].tolist()
        n_tot = min(len(dur_gtg_comp) + 1, cap_comp)

        rank_init = max(1, n_tot - sum(1 for c in dur_gtg_comp if c >= dur_gtg_local_min))
        markets[(od, day)] = dict(n=n_tot, init=min(rank_init, n_tot), comp=dur_gtg_comp, rows=list(local.itertuples()))
    return markets


def _get_prog_no_uniq(df, col_no, col_time, col_day = data_transfer["col_day"]):
    d = df[[col_no, col_day, col_time]].copy()
    d["day"] = (d[col_time] // 1440).astype(int)
    d["clock"] = d[col_time] - d["day"] * 1440
    d = d.sort_values("day").drop_duplicates([col_no, col_day], keep="first")
    idx = {(r[0], r[1]): [r[2], r[3] * 1440] for r in zip(d[col_no], d[col_day], d["clock"], d["day"])}
    
    by_no = {}
    for (no, day), cv in idx.items(): by_no.setdefault(no, []).append(cv)
    for cvs in by_no.values():
        clocks = [cv[0] for cv in cvs]
        if max(clocks) - min(clocks) > 720:
            for cv in cvs:
                if cv[0] > 720: cv[0] -= 1440; cv[1] += 1440

    return {k: tuple(v) for k, v in idx.items()}


def _get_mgn(prog_no, mgn_tim_port, dev_tim_prog):
    mgn = 5
    by_no = {}
    for (no, day), (c, _) in prog_no.items(): by_no.setdefault(no, []).append(c)
    return {no: max(mgn_tim_port, (max(cs) - min(cs) - dev_tim_prog) / 2 + mgn) if len(cs) > 1 else mgn_tim_port
            for no, cs in by_no.items()}

################################################################################################################################################################
################################################################################################################################################################


### exec #######################################################################################################################################################
@log.preprocess
def preprocess(dict_data):
    df_transfer, df_demand, df_func, df_route = (dict_data[k] for k in ("data_transfer", "data_demand", "data_func", "data_route"))

    df_transfer_space = _add_crit_data(df_transfer.copy())
    cap_comp = int(df_func[data_func["n_comp"]].max())
    markets = _get_markets(df_transfer_space, cap_comp)

    df_transfer_local = df_transfer_space[df_transfer_space[data_transfer["col_port_transf"]] == data_transfer["elem_local"]].reset_index(drop=True)
    prog_no_arr = _get_prog_no_uniq(df_transfer_local, data_transfer["col_no_arr"], "time_arr")
    prog_no_dep = _get_prog_no_uniq(df_transfer_local, data_transfer["col_no_dep"], "time_dep")

    mgn_arr = _get_mgn(prog_no_arr, consts.MGN_TIM_PORT, consts.DEV_TIM_PROG)
    mgn_dep = _get_mgn(prog_no_dep, consts.MGN_TIM_PORT, consts.DEV_TIM_PROG)

    dict_demand = df_demand.set_index([data_demand["col_port_orig"], data_demand["col_port_dest"]])[data_demand["col_demand"]].to_dict()
    dict_weight = {(int(r[0]), int(r[1]), int(r[2])): r[3] for r in df_func[[data_func[k] for k in ["n_comp", "rank_init", "rank_ulti", "weight"]]].to_numpy(dtype=float)}

    return {k: locals()[k] for k in ("df_transfer_local", "prog_no_arr", "prog_no_dep", "mgn_arr", "mgn_dep", "dict_demand", "dict_weight", "cap_comp", "markets", "df_route")}
################################################################################################################################################################