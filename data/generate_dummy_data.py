"""Generate the synthetic dataset used by this repository.

Everything here is invented: the network, the airports, the flight numbers, the
times, the demand factors and the scoring table. No value comes from a real
airline or from the competition data. A fixed seed makes the output repeatable.

Run:  python data/generate_dummy_data.py
"""
from datetime import datetime, time, timedelta
from pathlib import Path
import random

import pandas as pd

SEED = 2026
DAYS = 3
BASE = datetime(2030, 1, 7)
HUB = "HUB"
RIVAL_HUBS = ["RH1", "RH2", "RH3"]
MAX_COMPETITORS = 8
OUT = Path(__file__).parent


def _at(day, minutes):
    return BASE + timedelta(days=day, minutes=minutes)


def _dur(minutes):
    return time(minutes // 60, minutes % 60)


def build(seed=SEED, spokes=5):
    rnd = random.Random(seed)
    WEST, EAST = [f"W{i}" for i in range(1, spokes + 1)], [f"E{i}" for i in range(1, spokes + 1)]
    block = {s: rnd.randrange(90, 240, 5) for s in WEST + EAST}
    flights, routes, pair = {}, [], 0
    for s in WEST + EAST:
        pair += 1
        into, outof = 1000 + pair * 2, 1001 + pair * 2
        flights[s] = (into, outof)
        routes += [dict(FlNo=into, Orig=s, Dest=HUB, Pair=pair), dict(FlNo=outof, Orig=HUB, Dest=s, Pair=pair)]

    # Morning bank: west to east. Evening bank: east to west. The last west
    # arrival and the last east departure are late enough to cross midnight.
    arr_w = {s: 420 + 18 * i for i, s in enumerate(WEST)}
    dep_e = {s: 500 + 22 * i for i, s in enumerate(EAST)}
    arr_e = {s: 1040 + 16 * i for i, s in enumerate(EAST)}
    dep_w = {s: 1190 + 20 * i for i, s in enumerate(WEST)}
    dep_w[WEST[-1]] = 1440 + 25
    arr_e[EAST[-1]] = 1395

    rows, jitter = [], {}
    leg = lambda no, day: jitter.setdefault((no, day), rnd.randint(-4, 4))   # one time per flight and day
    for day in range(DAYS):
        for w in WEST:
            for e in EAST:
                for orig, dest, a0, d0 in ((w, e, arr_w[w], dep_e[e]), (e, w, arr_e[e], dep_w[w])):
                    a, d = a0 + leg(flights[orig][0], day), d0 + leg(flights[dest][1], day)
                    rows.append({"O&D": f"{orig}-{dest}", "Day": day + 1, "Arr1": HUB, "FlNo1": flights[orig][0], "FlNo2": flights[dest][1],
                                 "ElapsedTime1": _dur(block[orig]), "ElapsedTime2": _dur(block[dest]), "Arr Time": _at(day, a), "Dep Time": _at(day, d)})
                    nominal = block[orig] + (d - a) + block[dest]
                    for k in range(rnd.randint(0, 3)):
                        ra = a + rnd.randint(-120, 120)
                        total = nominal + rnd.randint(-70, 90)
                        b1, b2 = rnd.randrange(90, 240, 5), rnd.randrange(90, 240, 5)
                        rows.append({"O&D": f"{orig}-{dest}", "Day": day + 1, "Arr1": RIVAL_HUBS[k], "FlNo1": 5000 + len(rows), "FlNo2": 7000 + len(rows),
                                     "ElapsedTime1": _dur(b1), "ElapsedTime2": _dur(b2), "Arr Time": _at(day, ra), "Dep Time": _at(day, ra + max(45, total - b1 - b2))})

    transfer = pd.DataFrame(rows)
    pairs = sorted({tuple(od.split("-")) for od in transfer["O&D"]})
    demand = pd.DataFrame([{"Orig Airport Code": o, "Dest Airport Code": d, "OD importance factor": round(rnd.uniform(.1, 1.), 3)} for o, d in pairs])

    # Invented scoring table: a higher final rank scores more, and so does climbing.
    func = [dict(zip(("Competitors", "Initial Rank", "Final Rank", "Weight"), (n, i, f, 0. if n == 1 else round((n - f + 1) / n + max(0, i - f) / n, 6))))
            for n in range(1, MAX_COMPETITORS + 1) for i in range(1, n + 1) for f in range(1, n + 1)]
    return dict(data_transfer=transfer, data_route=pd.DataFrame(routes), data_demand=demand, data_func=pd.DataFrame(func))


if __name__ == "__main__":
    for name, df in build().items():
        with pd.ExcelWriter(OUT / f"{name}.xlsx", engine="openpyxl") as xl:
            df.to_excel(xl, index=False)
            xl.book.properties.creator, xl.book.properties.lastModifiedBy = "synthetic", "synthetic"
        print(f"{name}: {len(df)} rows")
