# Airline Schedule Optimization

A mixed-integer model that shifts flight times at a hub to create more connections and shorter trips than competing routes. It was built by our team for the [TEKNOFEST 2026 AI-Assisted Airline Optimization Competition](https://www.teknofest.org/tr/yarismalar/yapay-zeka-destekli-havayolu-optimizasyonu-yarismasi/), held with Turkish Technology.

**This repository contains no competition data.** The rules do not allow sharing the provided datasets, so everything in `data/` is invented by `data/generate_dummy_data.py`. The rule values in `config.json` are invented too. The results below describe that synthetic network and say nothing about any real airline schedule.

## The Problem

An airline can move its flights by a limited amount. Each move changes which arriving flights connect to which departing flights at the hub. For every origin and destination pair on every day (a market), the goal is more usable connections and a total trip time that beats more of the competing routes.

## The Model

The model is written in Pyomo. Flight times are integer variables in minutes.

**Objective.** Maximize, over all markets and weighted by each market's importance:

- A connection score, `2(1 - 0.5^k)` for `k` valid connections, so each extra connection is worth half of the one before it. Its concave shape lets it be written as a set of linear cuts.
- A rank score from a lookup table, based on the number of competitors, the initial rank and the final rank.

**Constraints.**

| Rule | Default |
|------|---------|
| A valid connection has a ground time between the two limits | 50 to 280 min |
| Each flight moves only within its time window | ±50 min or more |
| The same flight stays regular across days | 20 min spread |
| Hub capacity per 15-minute slot | 3 arrivals, 2 departures |
| An aircraft has enough time for its round trip plus ground time | 40 min |
| Both directions of a market have a similar number of connections | 25 % |
| Both directions of a market have similar best trip times | 40 min |

Connection validity, the ranking and the direction balance use binary variables with big-M constraints, tightened from the time windows of each connection. Binaries that can never be active are not created, which keeps the model small. A first model that only minimizes the deviation from the original schedule provides the starting point.

## Results on the Synthetic Network

The dataset has 10 spoke airports, 30 adjustable flight legs per direction and 150 markets over 3 days, with up to three competing routes per market.

| | Original schedule | Optimized schedule |
|---|---|---|
| Objective | 108.15 | 135.65 |
| Markets with a valid connection | 112 | 126 |

The solver proves this optimal in a few seconds with HiGHS. The network is much smaller than a real one, so this says little about solve times at full scale.

## Machine Learning Layer

**This layer was added after the competition.** The competition code in this repository uses no machine learning.

`learn.py` generates synthetic networks and solves them. It then trains a gradient boosting classifier to predict whether each candidate connection stays valid, using only what is known before solving, such as ground time limits, trip time and competitors. The most confident predictions are fixed before the solve, so the solver has fewer decisions left.

The test used 10 larger networks that were not used in training, each with 8 spoke airports per side and about 350 candidates. Both versions had a 30 second limit, and neither finished within it.

| | Full model | With the classifier |
|---|---|---|
| Total objective | 2999.52 | 3088.01 |
| Networks won | 3 | 7 |

The classifier fixed about 65 % of the candidates, and 96.6 % of those choices agreed with the full model's solution. Its ROC AUC was 0.918. The gain is 2.95 % in total, and in 3 of the 10 networks the result was worse.

Fixing a wrong choice can remove the best schedule, so this is a way to get a good answer quickly and not a replacement for the full model. Given more time, the full model can overtake it. The time limits also make these numbers depend on the computer. The training labels come from time limited solves and may not be optimal, and one of the 30 training networks gave no solution within its limit and was skipped.

## Running It

Python 3.12 or newer is required.

```
pip install -r requirements.txt
python data/generate_dummy_data.py
python run.py
python verify.py
```

To repeat the machine learning comparison, run `python learn.py`. It can take around 20 minutes because of the solver time limits.

`run.py` writes the new times, the selected connections and the per-market results to `result/result.xlsx`. The generator is optional because its output is committed.

`verify.py` does not use the model. It rebuilds the schedule from the raw data, tests each rule, and recomputes the objective to compare it with the solver.

The solver is set by `NAME_SOLVER` in `config.json`. It uses the free HiGHS solver by default. `gurobi` is also supported if you have a license, but it was not tested with this version. Other solvers or versions can return a different schedule with the same objective.

## Fixes After the Competition

Working through the synthetic data exposed these problems in the competition version of the code.

- **Overnight connections were lost.** Day offsets were computed from a time of day, so they were always zero, and connection gaps ignored them. A connection that crossed midnight looked like a negative gap and was dropped. On the synthetic network this removed 5 of 128 candidate connections and lowered the objective from 135.65 to 132.58.
- **Hub capacity split across midnight.** Flights near midnight could be counted in separate slots. Capacity is now counted in absolute time. This one did not change the result on the synthetic network.
- **Reported ranks counted invalid connections.** The results file now only uses connections inside the ground time limits. The objective was not affected.
- **HiGHS ignored the starting point and could overrun its time limit.** The starting point was never passed to the solver, and on larger networks the Pyomo interface kept running past the limit here. The code now calls HiGHS directly.
- **A column name was hardcoded.** The pair column now comes from `config.json`.

## Limitations

- The rule values in `config.json` are invented and were not taken from the competition.
- The rank table and the demand factors are invented, so the scores only mean something within this dataset.
- A connection counts as local when it goes through the hub, so a competitor at the same hub would not be modeled correctly.
- Each flight number has one time per day, which the generator guarantees and real data should be checked for.

## License

MIT
