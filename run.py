"""Run one simulation and save its time series and a dashboard figure.

    python run.py --weeks 260 --shock-week 104 --shock-size 0.04 --seed 7
"""

import argparse
import warnings
from pathlib import Path

from workplace_stress import WorkplaceStressModel
from workplace_stress.plots import plot_single_run

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weeks", type=int, default=260)
    p.add_argument("--banks", type=int, default=5)
    p.add_argument("--employees", type=int, default=120, help="employees per bank")
    p.add_argument("--contagion", type=float, default=0.08)
    p.add_argument("--wellbeing", type=float, default=0.0, help="wellbeing support (0 = none)")
    p.add_argument("--hours-cap", type=float, default=None)
    p.add_argument("--shock-week", type=int, default=104)
    p.add_argument("--shock-size", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    model = WorkplaceStressModel(
        n_banks=args.banks,
        employees_per_bank=args.employees,
        contagion=args.contagion,
        wellbeing_support=args.wellbeing,
        hours_cap=args.hours_cap,
        shock_week=args.shock_week,
        shock_size=args.shock_size,
        seed=args.seed,
    )
    for _ in range(args.weeks):
        model.step()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = model.datacollector.get_model_vars_dataframe()
    df.index.name = "Week"
    df.to_csv(out / "single_run_model.csv")
    model.datacollector.get_agent_vars_dataframe().to_csv(out / "single_run_agents.csv")
    plot_single_run(df, out / "single_run.png", shock_week=args.shock_week)

    last = df.iloc[-1]
    print(f"Employees: {args.banks * args.employees} across {args.banks} banks, {args.weeks} weeks")
    print(f"Total resignations:  {model.total_exits}")
    print(f"Peak burnout share:  {model.peak_burnout:.1%}")
    print(f"Firm volatility:     {last.FirmVolFull:.1%} (annualised, whole run)")
    print(f"Macro volatility:    {last.MacroVolFull:.1%}")
    print(f"Max index drawdown:  {last.MaxDrawdown:.1%}")
    print(f"Saved outputs to {out}/")


if __name__ == "__main__":
    main()
