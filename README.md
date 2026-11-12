# Workplace Stress Agent-Based Model

An agent-based model built in Python with [Mesa](https://mesa.readthedocs.io) (3.x). It simulates 600 investment-bank employees (5 banks × 120 people) whose stress levels move between **Healthy**, **Stressed** and **Burnout** states. The model measures how workforce wellbeing cascades into **firm** and **macro** volatility.

## The feedback loop

```
market turbulence ─► heavier workload ─► longer hours ─► employee stress
        ▲                                                     │
        │                                  contagion over the office network
        │                                                     ▼
 bank losses feed       noisier bank P&L ◄── lower productivity, more operational
 market volatility ◄──  (firm volatility)     errors, more resignations
                                                              │
                                vacancies push work onto remaining colleagues
```

One model step is one week. A run lasts 260 weeks (5 years).

## Model components

| Piece | What it does |
|---|---|
| `Employee` agent ([agents.py](workplace_stress/agents.py)) | Has a role (Analyst → MD), a desk (M&A, Trading, DCM/ECM, Research, Risk/Ops), a resilience trait and a continuous stress level in [0, 1]. Each week it works some hours, updates its stress, produces output, may make a costly error, and may quit. |
| Stress dynamics | `Δs = 0.10·overwork + contagion·max(0, s̄_neighbors − s) − recovery·s + noise`. Overwork starts at 60 h/week. Recovery depends on resilience and firm wellbeing support, and disappears when someone is buried in work. Contagion only flows from more-stressed colleagues (negative affect spreads more readily). |
| Performance | Follows a Yerkes–Dodson inverted U that peaks at moderate stress (0.3). New hires ramp up over about 6 months. People in burnout are absent 25% of the time. |
| Operational risk | Weekly error probability is `base·(1 + 10·s²)`. Loss size is lognormal and scaled by role exposure and desk risk (Trading is highest). |
| Attrition and hiring | The quit hazard rises with each consecutive week of burnout. A vacancy is refilled after an 8-week hiring lag. Until then, 60% of the missing work is pushed onto the people left on that desk. |
| Office network ([model.py](workplace_stress/model.py)) | Each desk is a Watts–Strogatz small-world graph, with reporting lines to more senior staff and some cross-desk friendships. Built with networkx. |
| `Bank` | Weekly P&L is revenue minus sticky costs minus operational losses. Equity return is `β·market + P&L/capital`. The second term is the **workforce-driven (idiosyncratic) firm volatility**. |
| Market / macro | Volatility behaves like GARCH: it is persistent, mean-reverting, and pushed up by last week's sector losses. The sector index mixes the broad market with bank equities. This index is the **macro volatility** measure. An optional one-time volatility shock can be added. |

All data is recorded with `mesa.DataCollector`. The experiments run through `mesa.batch_run` in parallel.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py                  # one 5-year run with a market shock at week 104
python experiments.py          # 4 scenarios × 20 seeds + contagion sweep (~1 min on 8+ cores)
```

`run.py` accepts `--contagion`, `--wellbeing`, `--hours-cap`, `--shock-week`, `--shock-size`, `--banks`, `--employees` and `--seed`.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `single_run.png` | Stress-state mix, resignations, firm and macro volatility, productivity |
| `single_run_model.csv` / `single_run_agents.csv` | Weekly model-level and per-employee data |
| `scenarios.png`, `scenario_summary.csv` | Scenario comparison (mean ± s.d. over seeds) |
| `contagion_sweep.png`, `contagion_sweep.csv` | Contagion strength (0–0.4) vs. burnout and volatility |
| `stress_vol_link.csv` | Correlation of burnout share with volatility 0–13 weeks later |

## Results (20 seeds, 600 employees, 5 years)

| Scenario | Workforce-driven firm vol | Macro vol | Resignations | Peak burnout |
|---|---|---|---|---|
| Baseline | 1.5% | 18.7% | 660 | 9% |
| Market shock (week 104) | 2.4% | 20.0% | 1,364 | 40% |
| Shock + high stress contagion | 3.4% | 20.6% | 1,821 | 46% |
| Shock + wellbeing program (support + 80 h cap) | 1.4% | 20.4% | 317 | 2% |

What the results show:

- **The shock spreads through the workforce.** A short-lived volatility shock (half-life about 4 weeks) leaves burnout high for about a year, because resignations pile work onto the people who stay.
- **Contagion multiplies the effect on firms.** Raising contagion from 0 to 0.4 increases workforce-driven firm volatility about 2.6× (2.0% → 5.2%) and peak burnout from 39% to 60%.
- **Wellbeing policy absorbs the shock.** With support and an hours cap, firm idiosyncratic volatility stays at baseline levels and resignations fall 77% compared with the shock-only scenario.
- **The macro effect is real but small.** Burnout share correlates with later macro volatility (ρ ≈ 0.6–0.7 at 0–4 week lags in the shock scenarios). Across scenarios, though, macro volatility differs by less than one seed-standard-deviation, because the broad market dominates the sector index. A larger share of financials in the index, or more banks, would make this channel stronger.

## Calibration notes

Parameters are stylised and were chosen to be plausible, not fitted to data. Baseline attrition is about 22% a year and average hours are about 66 a week, both broadly in line with reported IB figures. Market volatility is around 18–20% annualised. The main levers are in the signature of `WorkplaceStressModel.__init__`. Notable ones include `vol_feedback` (how strongly bank losses feed market volatility; at 0.6 the loop ran away to the volatility cap in testing) and `Bank.vacancy_passthrough`.
