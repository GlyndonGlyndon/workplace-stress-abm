"""Scenario experiments and a contagion sweep, run with mesa.batch_run.

    python experiments.py --seeds 20

Outputs (in results/):
  scenario_timeseries.csv, scenario_summary.csv, scenarios.png
  contagion_sweep.csv, contagion_sweep.png
  stress_vol_link.csv   (does workforce stress lead volatility?)
"""

import argparse
import warnings
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from workplace_stress import WorkplaceStressModel
from workplace_stress.plots import plot_scenarios, plot_sweep

warnings.filterwarnings("ignore", category=FutureWarning)

WEEKS = 260
SHOCK = {"shock_week": 104, "shock_size": 0.04}

SCENARIOS = {
    "Baseline": {},
    "Market shock": {**SHOCK},
    "Shock + high contagion": {**SHOCK, "contagion": 0.20},
    "Shock + wellbeing program": {**SHOCK, "wellbeing_support": 0.3, "hours_cap": 80},
}

SUMMARY_COLS = ["FirmVolFull", "FirmIdioVolFull", "MacroVolFull", "TotalExits", "PeakBurnout", "CumOpLosses", "MaxDrawdown"]


def run_batch(params: dict, seeds, every_step: bool) -> pd.DataFrame:
    results = mesa.batch_run(
        WorkplaceStressModel,
        parameters={**params, "collect_agents": False},
        rng=list(seeds),
        max_steps=WEEKS,
        data_collection_period=1 if every_step else -1,
        number_processes=None,
        display_progress=False,
    )
    return pd.DataFrame(results)


def stress_leads_volatility(ts: pd.DataFrame, lags=(0, 4, 8, 13)) -> pd.DataFrame:
    """Correlation of burnout share at week t with firm/macro vol at week t+lag."""
    rows = []
    for (scenario, run), g in ts.groupby(["Scenario", "RunId"]):
        g = g.sort_values("Step").iloc[26:]  # skip the warm-up window
        for lag in lags:
            rows.append({
                "Scenario": scenario,
                "lag_weeks": lag,
                "corr_firm_idio": g["Burnout"].corr(g["FirmIdioVol"].shift(-lag)),
                "corr_macro": g["Burnout"].corr(g["MacroVol"].shift(-lag)),
            })
    return pd.DataFrame(rows).groupby(["Scenario", "lag_weeks"]).mean().round(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    seeds = range(args.seeds)

    # 1. Scenario comparison.
    frames = []
    for name, params in SCENARIOS.items():
        print(f"Running scenario: {name}")
        df = run_batch(params, seeds, every_step=True)
        df["Scenario"] = name
        frames.append(df)
    ts = pd.concat(frames, ignore_index=True)
    ts.to_csv(out / "scenario_timeseries.csv", index=False)

    final = ts[ts.Step == WEEKS]
    summary = final.groupby("Scenario", sort=False)[SUMMARY_COLS].agg(["mean", "std"])
    summary.to_csv(out / "scenario_summary.csv")
    plot_scenarios(ts, summary, out / "scenarios.png", shock_week=SHOCK["shock_week"])

    link = stress_leads_volatility(ts)
    link.to_csv(out / "stress_vol_link.csv")

    # 2. Contagion sweep under a market shock.
    print("Running contagion sweep")
    sweep = run_batch({**SHOCK, "contagion": list(np.round(np.linspace(0, 0.4, 9), 2))},
                      seeds, every_step=False)
    sweep = sweep[["contagion", *SUMMARY_COLS]]
    sweep.to_csv(out / "contagion_sweep.csv", index=False)
    plot_sweep(sweep, out / "contagion_sweep.png")

    # Console report.
    pd.set_option("display.width", 140)
    means = summary.xs("mean", axis=1, level=1).copy()
    for c in ["FirmVolFull", "FirmIdioVolFull", "MacroVolFull", "PeakBurnout", "MaxDrawdown"]:
        means[c] = (means[c] * 100).round(1)
    means["CumOpLosses"] = (means["CumOpLosses"] / 1000).round(1)  # $M
    means["TotalExits"] = means["TotalExits"].round(0)
    print("\nScenario means (vol/burnout/drawdown in %, op losses in $M):")
    print(means.to_string())
    print("\nCorrelation of burnout share with future volatility:")
    print(link.to_string())
    print(f"\nSaved outputs to {out}/")


if __name__ == "__main__":
    main()
