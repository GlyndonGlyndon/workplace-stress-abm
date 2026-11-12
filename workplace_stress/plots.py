"""Matplotlib figures for single runs and scenario experiments."""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"Healthy": "#0ca30c", "Stressed": "#fab219", "Burnout": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def _shade_shock(ax, shock_week):
    if shock_week is not None:
        ax.axvline(shock_week, color=MUTED, lw=1, ls="--")


def plot_single_run(df: pd.DataFrame, path: str, shock_week: int | None = None):
    """Four-panel dashboard of one simulation's time series."""
    df = df.iloc[1:]  # step 0 is the pre-simulation snapshot
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    weeks = df.index
    if shock_week is not None:
        fig.suptitle(f"Dashed line = market shock (week {shock_week})", x=0.01, ha="left",
                     color=INK_2, fontsize=9)

    ax = axes[0, 0]
    cols = ["Healthy", "Stressed", "Burnout"]
    ax.stackplot(weeks, *[df[c] * 100 for c in cols], labels=cols,
                 colors=[STATUS[c] for c in cols], edgecolor=SURFACE, linewidth=0.5, alpha=0.9)
    ax.set(title="Workforce stress states", ylabel="% of employees", xlabel="Week", ylim=(0, 100))
    ax.legend(loc="lower left", ncol=3)
    _shade_shock(ax, shock_week)

    ax = axes[0, 1]
    ax.plot(weeks, df["Exits"].rolling(4, min_periods=1).sum(), color=SERIES[0])
    ax.set(title="Resignations (4-week rolling)", ylabel="Employees", xlabel="Week")
    _shade_shock(ax, shock_week)

    ax = axes[1, 0]
    ax.plot(weeks, df["FirmVol"] * 100, color=SERIES[0], label="Firm (bank equity)")
    ax.plot(weeks, df["MacroVol"] * 100, color=SERIES[1], label="Macro (sector index)")
    ax.set(title="Volatility (26-week, annualised)", ylabel="%", xlabel="Week")
    ax.legend(loc="upper left")
    _shade_shock(ax, shock_week)

    ax = axes[1, 1]
    ax.plot(weeks, df["Productivity"] * 100, color=SERIES[0])
    ax.set(title="Productivity vs. healthy-workforce benchmark", ylabel="%", xlabel="Week")
    _shade_shock(ax, shock_week)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(ts: pd.DataFrame, summary: pd.DataFrame, path: str, shock_week: int | None):
    """Scenario comparison: burnout and macro-vol paths plus summary bars."""
    scenarios = list(summary.index)
    colors = {s: SERIES[i] for i, s in enumerate(scenarios)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

    for ax, col, title, ylabel in [
        (axes[0, 0], "Burnout", "Share of employees in burnout (mean across seeds)", "%"),
        (axes[0, 1], "MacroVol", "Macro volatility, 26-week annualised", "%"),
    ]:
        for s in scenarios:
            g = ts[ts.Scenario == s].groupby("Step")[col].mean() * 100
            ax.plot(g.index, g.values, color=colors[s], label=s)
        ax.set(title=title, ylabel=ylabel, xlabel="Week")
        _shade_shock(ax, shock_week)
    axes[0, 0].legend(loc="upper right")

    for ax, col, title, scale, fmt in [
        (axes[1, 0], "FirmIdioVolFull", "Workforce-driven firm volatility (annualised %)", 100, "{:.1f}"),
        (axes[1, 1], "TotalExits", "Total resignations over 5 years", 1, "{:.0f}"),
    ]:
        vals = summary[(col, "mean")] * scale
        errs = summary[(col, "std")] * scale
        labels = [textwrap.fill(s, 16) for s in scenarios]
        bars = ax.bar(labels, vals, yerr=errs, color=[colors[s] for s in scenarios],
                      width=0.6, edgecolor=SURFACE, linewidth=2,
                      error_kw={"ecolor": MUTED, "lw": 1, "capsize": 3})
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, 0), xytext=(0, 4),
                        textcoords="offset points", ha="center", color="white", fontsize=9,
                        fontweight="bold")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8, colors=INK_2)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, path: str):
    """Contagion strength vs. firm/macro volatility and peak burnout."""
    g = sweep.groupby("contagion").agg(["mean", "std"])
    x = g.index
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("PeakBurnout", "Peak burnout share", "% of employees", STATUS["Burnout"]),
        ("FirmIdioVolFull", "Workforce-driven firm volatility", "Annualised vol, %", SERIES[0]),
        ("MacroVolFull", "Macro (sector index) volatility", "Annualised vol, %", SERIES[1]),
    ]
    for ax, (col, title, ylabel, c) in zip(axes, panels):
        m, s = g[(col, "mean")] * 100, g[(col, "std")] * 100
        ax.plot(x, m, color=c, marker="o", ms=4)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.set(title=title, xlabel="Stress-contagion coefficient", ylabel=ylabel)
    fig.suptitle("Shaded band = ±1 s.d. across seeds; all runs include the week-104 market shock",
                 x=0.01, ha="left", color=INK_2, fontsize=9)

    fig.savefig(path, dpi=150)
    plt.close(fig)
