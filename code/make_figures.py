"""
Publication figures for the revised manuscript.

Reads only from results/ (run random_walk_submission_analysis.py and
extra_analyses.py first) and writes 300-dpi PNGs to figures/.

One visual system across all figures:
    blue dots, white ring   simulation estimate (bars = 95% CI where visible)
    black dashed line       exact finite-n theory
    grey dotted line        large-n asymptotic approximation
    pale blue band          range expected from Monte Carlo noise / spread
No in-figure titles: each panel carries a letter and a short heading, and the
manuscript captions carry the message. Palettes were checked with the
dataviz validate_palette.py script on a white surface:
    categorical #2a78d6,#eb6834,#1baf7a,#eda100   adjacent CVD dE 9.1 (pass);
        green/yellow < 3:1 contrast -> lines are direct-labelled + marker shapes
    ordinal ramp #86b6ef,#3987e5,#1c5cab,#0d366b  monotone, one hue (pass)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

INK = "#0b0b0b"          # theory lines, primary text
INK_2 = "#52514e"        # secondary text, axes, ticks
INK_3 = "#9a9893"        # asymptotic lines, sample paths
GRID = "#e8e7e3"         # hairline grid
SURFACE = "#ffffff"

SIM = "#2a78d6"          # categorical slot 1 - simulation
ACCENT = "#eb6834"       # categorical slot 2 - normal approximation
BAND_1 = "#cde2fb"       # blue 100 - expected-noise band / +-1 SD
BAND_2 = "#e6f0fc"       # lighter tint - +-2 SD
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ORDINAL = {0.50: "#86b6ef", 0.55: "#3987e5", 0.60: "#1c5cab", 0.70: "#0d366b"}
MARKERS = ["o", "s", "^", "D"]

FULL_WIDTH = 6.5   # inches: AJSR text width on Letter with 1-inch margins

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)

THEORY = dict(color=INK, ls=(0, (4, 2.5)), lw=1.3)
ASYMPTOTIC = dict(color=INK_3, ls=(0, (1, 1.6)), lw=1.6)
SIM_POINT = dict(fmt="o", color=SIM, ms=6, mec=SURFACE, mew=1.0, ecolor=SIM, elinewidth=1.0,
                 capsize=0, zorder=4)


def heading(ax, letter: str, text: str) -> None:
    """Panel letter in bold plus a short heading, left-aligned above the axes."""
    ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", fontsize=9.5, color=INK, pad=8)


def log_steps_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 50, 100, 500, 1000, 5000]))
    ax.set_xticklabels(["10", "50", "100", "500", "1,000", "5,000"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(7, 7000)
    ax.set_xlabel("Number of steps $n$")


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", FIGURES / name)


def exact_p0(n: np.ndarray) -> np.ndarray:
    return np.exp([math.lgamma(k + 1) - 2 * math.lgamma(k / 2 + 1) - k * math.log(2) for k in n])


# ---------------------------------------------------------------------------
# Figure 1: sample paths
# ---------------------------------------------------------------------------

def figure1_paths() -> None:
    paths = np.load(RESULTS / "sample_paths.npz")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7), sharey=True)
    for ax, p, letter, text in zip(axes, [0.50, 0.55], "AB",
                                   ["Symmetric walk, $p = 0.50$", "Biased walk, $p = 0.55$"]):
        walks = paths[f"paths_p{p:.2f}"]
        t = np.arange(walks.shape[1])
        mean = t * (2 * p - 1)
        sd = np.sqrt(4 * t * p * (1 - p))
        ax.fill_between(t, mean - 2 * sd, mean + 2 * sd, color=BAND_2, lw=0, zorder=0)
        ax.fill_between(t, mean - sd, mean + sd, color=BAND_1, lw=0, zorder=0)
        for walk in walks[1:]:
            ax.plot(t, walk, color=INK_3, lw=0.45, alpha=0.75, zorder=1)
        ax.plot(t, walks[0], color=SIM, lw=0.9, zorder=2)   # one walk highlighted
        ax.plot(t, mean, **THEORY, zorder=3)
        end = t[-1]
        ax.text(end * 1.01, mean[-1], "mean", va="center", fontsize=8, color=INK)
        ax.text(end * 1.01, mean[-1] + 0.8 * sd[-1], "±1 SD", va="center", fontsize=8, color=INK_2)
        ax.text(end * 1.01, mean[-1] + 1.75 * sd[-1], "±2 SD", va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, end)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels(["0", "250", "500", "750", "1,000"])
        ax.set_xlabel("Step $t$")
        ax.grid(axis="x", visible=False)
        heading(ax, letter, text)
    axes[0].set_ylabel("Position $S_t$ (steps)")
    fig.tight_layout(w_pad=3.0)
    save(fig, "fig1_sample_paths.png")


# ---------------------------------------------------------------------------
# Figure 2: symmetric walks vs exact theory
# ---------------------------------------------------------------------------

def figure2_symmetric() -> None:
    d = pd.read_csv(RESULTS / "symmetric_with_uncertainty.csv")
    trials = int(d["trials"].iloc[0])
    n = d["steps"].to_numpy(dtype=float)
    grid = np.unique(np.round(np.logspace(np.log10(8), np.log10(6000), 300) / 2) * 2)

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def sim(ax, metric):
        ax.errorbar(n, d[metric], yerr=1.96 * d[f"{metric}_se"], **SIM_POINT)

    # A: mean - points against the band expected from noise alone
    band = 1.96 * np.sqrt(grid / trials)
    ax_a.fill_between(grid, -band, band, color=BAND_1, lw=0)
    ax_a.axhline(0, **THEORY)
    sim(ax_a, "mean_position")
    ax_a.text(5600, band[-1] * 0.55, "95% noise\nrange", fontsize=7.5, color=INK_2, ha="right", va="center")
    ax_a.set_ylabel(r"Mean position $\bar S_n$ (steps)")
    heading(ax_a, "A", "Mean final position")

    # B: variance
    ax_b.plot(grid, grid, **THEORY)
    sim(ax_b, "variance")
    ax_b.set_yscale("log")
    ax_b.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax_b.set_yticklabels(["10", "100", "1,000", "10,000"])
    ax_b.yaxis.set_minor_locator(NullLocator())
    ax_b.set_ylabel(r"Variance of $S_n$ (steps$^2$)")
    heading(ax_b, "B", "Variance")

    # C: E|S_n|
    exact_abs = grid * exact_p0(grid)
    ax_c.plot(grid, np.sqrt(2 * grid / np.pi), **ASYMPTOTIC)
    ax_c.plot(grid, exact_abs, **THEORY)
    sim(ax_c, "average_absolute_distance")
    ax_c.set_yscale("log")
    ax_c.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    ax_c.set_yticklabels(["2", "5", "10", "20", "50"])
    ax_c.yaxis.set_minor_locator(NullLocator())
    ax_c.set_ylabel(r"Mean distance $\overline{|S_n|}$ (steps)")
    heading(ax_c, "C", "Mean distance from origin")

    # D: P(S_n = 0)
    ax_d.plot(grid, np.sqrt(2 / (np.pi * grid)), **ASYMPTOTIC)
    ax_d.plot(grid, exact_p0(grid), **THEORY)
    sim(ax_d, "endpoint_origin_probability")
    ax_d.set_yscale("log")
    ax_d.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2]))
    ax_d.set_yticklabels(["0.01", "0.02", "0.05", "0.10", "0.20"])
    ax_d.yaxis.set_minor_locator(NullLocator())
    ax_d.set_ylabel(r"Probability $P(S_n = 0)$")
    heading(ax_d, "D", "Probability of ending at the origin")

    for ax in axes.flat:
        log_steps_axis(ax)

    # One shared key for the whole figure
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=SIM, mec=SURFACE, mew=1.0, ms=6),
        plt.Line2D([], [], **THEORY),
        plt.Line2D([], [], **ASYMPTOTIC),
        plt.Rectangle((0, 0), 1, 1, color=BAND_1, lw=0),
    ]
    labels = ["Simulation ($N$ = 10,000; bars = 95% CI)", "Exact theory",
              "Large-step approximation", "Expected noise range"]
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.5)
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               fontsize=7.8, columnspacing=1.4, handletextpad=0.5)
    save(fig, "fig2_symmetric_vs_theory.png")


# ---------------------------------------------------------------------------
# Figure 3: when the large-step approximation fails a Monte Carlo test
# ---------------------------------------------------------------------------

def figure3_approximation() -> None:
    sim = pd.read_csv(RESULTS / "approximation_rejection_rates.csv")
    crit = pd.read_csv(RESULTS / "approximation_critical_trials.csv")
    rule = pd.read_csv(RESULTS / "approximation_rule_constant.csv").iloc[0]
    limit = float(rule["limit_critical_trials_over_n_squared"])        # 50% rejection probability
    limit_10 = float(rule["limit_10pct_trials_over_n_squared"])        # 10% rejection probability
    steps_order = sorted(sim["steps"].unique())
    ramp = dict(zip(steps_order, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))

    # A: observed rejection rate vs N / n^2 (Wilson 95% CIs), with predicted curves
    ratio_grid = np.logspace(-0.2, 2.7, 200)
    offsets = dict(zip(steps_order, [0.88, 0.96, 1.04, 1.12]))   # small x-offsets so markers don't hide each other
    for n in steps_order:
        row = crit[(crit.metric == "average_absolute_distance") & (crit.steps == n)].iloc[0]
        se_rel = row.cv / np.sqrt(ratio_grid * n**2)
        delta = -row.relative_error / se_rel
        predicted = stats.norm.cdf(-1.96 - delta) + 1 - stats.norm.cdf(1.96 - delta)
        ax_a.plot(ratio_grid, predicted, color=ramp[n], lw=1.0)
        d = sim[sim.steps == n]
        rate = d.rejection_rate_vs_approximation
        wilson = [rate - d.wilson_lower_vs_approximation, d.wilson_upper_vs_approximation - rate]
        ax_a.errorbar(d.trials_over_n_squared * offsets[n], rate,
                      yerr=wilson, fmt="o", color=ramp[n], ms=5.5, mec=SURFACE,
                      mew=0.9, elinewidth=0.8, capsize=0, label=f"$n$ = {n}", zorder=3)
    ax_a.axhline(0.05, color=INK_3, ls=ASYMPTOTIC["ls"], lw=1.2)
    ax_a.text(420, 0.075, "nominal 5%", fontsize=7.5, color=INK_2, ha="right")
    ax_a.axvline(limit, color=INK, ls=THEORY["ls"], lw=1.0)
    ax_a.text(limit / 1.12, 0.72, f"$N = {limit:.0f}\\,n^2$\n(50%)", fontsize=8.5, color=INK, ha="right")
    ax_a.axvline(limit_10, color=INK_2, ls="-.", lw=0.9)
    ax_a.text(limit_10 * 1.1, 0.30, f"$N = {limit_10:.1f}\\,n^2$\n(10%)", fontsize=8.5, color=INK_2, ha="left")
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax_a.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.set_ylim(0, 1.04)
    ax_a.set_xlabel("Trials per squared step count, $N/n^2$")
    ax_a.set_ylabel("Rejection rate")
    ax_a.legend(loc="upper left", fontsize=7.5, handletextpad=0.3)
    heading(ax_a, "A", "Rejections of $\\sqrt{2n/\\pi}$, mean distance")

    # B: critical number of trials vs n for both statistics
    for metric, colour, marker, label in [
        ("average_absolute_distance", SIM, "o", "Mean distance"),
        ("endpoint_origin_probability", ACCENT, "s", r"$P(S_n=0)$"),
    ]:
        d = crit[crit.metric == metric]
        ax_b.plot(d.steps, d.critical_trials, color=colour, marker=marker, ms=6, mec=SURFACE, mew=0.9,
                  ls="none", label=label, zorder=3)
    n_grid = np.logspace(1, np.log10(5000), 50)
    ax_b.plot(n_grid, limit * n_grid**2, color=INK, ls=THEORY["ls"], lw=1.2, label=f"${limit:.0f}\\,n^2$ (50%)", zorder=4)
    ax_b.plot(n_grid, limit_10 * n_grid**2, color=INK_2, ls="-.", lw=0.9,
              label=f"${limit_10:.1f}\\,n^2$ (10%)", zorder=4)
    ax_b.axhline(10_000, color=INK_3, ls=ASYMPTOTIC["ls"], lw=1.2)
    ax_b.text(1100, 13_000, "$N$ = 10,000", fontsize=7.5, color=INK_2)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 5000]))
    ax_b.set_xticklabels(["10", "100", "1,000", "5,000"])
    ax_b.yaxis.set_major_locator(FixedLocator([1e3, 1e5, 1e7, 1e9, 1e11]))
    ax_b.set_yticklabels(["$10^3$", "$10^5$", "$10^7$", "$10^9$", "$10^{11}$"])
    ax_b.xaxis.set_minor_locator(NullLocator()); ax_b.yaxis.set_minor_locator(NullLocator())
    ax_b.set_xlabel("Number of steps $n$")
    ax_b.set_ylabel("Critical number of trials $N^*$")
    ax_b.legend(loc="upper left", fontsize=7.5, handletextpad=0.4)
    heading(ax_b, "B", "Trials for a 50% rejection probability")

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_approximation_failure.png")


# ---------------------------------------------------------------------------
# Figure 5: biased walks
# ---------------------------------------------------------------------------

def figure5_biased() -> None:
    d = pd.read_csv(RESULTS / "biased_with_uncertainty.csv")
    samples = np.load(RESULTS / "endpoint_samples.npz")
    steps = int(d["steps"].iloc[0])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8),
                                     gridspec_kw={"width_ratios": [1, 1.55]})

    p_grid = np.linspace(0.48, 0.72, 100)
    ax_a.plot(p_grid, steps * (2 * p_grid - 1), **THEORY)
    for _, row in d.iterrows():
        p = round(row["p_right"], 2)
        ax_a.errorbar(p, row["mean_position"], yerr=1.96 * row["mean_position_se"],
                      **{**SIM_POINT, "color": ORDINAL[p], "ecolor": ORDINAL[p], "ms": 7})
    ax_a.text(0.66, 175, r"$n(2p-1)$", fontsize=8.5, color=INK, ha="left")
    ax_a.set_xlim(0.48, 0.72)
    ax_a.set_ylim(-40, 450)
    ax_a.set_xticks([0.50, 0.55, 0.60, 0.70])
    ax_a.set_xlabel("Probability of a right step $p$")
    ax_a.set_ylabel(r"Mean position $\bar S_n$ (steps)")
    heading(ax_a, "A", f"Mean final position ($n$ = {steps:,})")

    edges = np.arange(-121, 521, 8)  # bin edges on odd integers: S_n is always even
    x = np.linspace(-130, 520, 700)
    for p, colour in ORDINAL.items():
        positions = samples[f"biased_p{p:.2f}"]
        ax_b.hist(positions, bins=edges, density=True, color=colour, alpha=0.18, lw=0)
        ax_b.hist(positions, bins=edges, density=True, histtype="step", color=colour, lw=0.9)
        mu, sigma = steps * (2 * p - 1), math.sqrt(4 * steps * p * (1 - p))
        ax_b.plot(x, stats.norm.pdf(x, mu, sigma), color=INK, lw=0.7, ls=THEORY["ls"])
        ax_b.text(mu, stats.norm.pdf(mu, mu, sigma) * 1.07, f"$p$ = {p:.2f}", ha="center",
                  fontsize=8, color=INK)
    ax_b.set_xlim(-130, 520)
    ax_b.set_ylim(0, 0.0165)
    ax_b.set_yticks([0, 0.005, 0.010, 0.015])
    ax_b.set_yticklabels(["0", "0.005", "0.010", "0.015"])
    ax_b.set_xlabel("Final position $S_n$ (steps)")
    ax_b.set_ylabel("Density (per step)")
    ax_b.grid(axis="x", visible=False)
    heading(ax_b, "B", "Distribution of final positions")

    handles = [plt.Line2D([], [], marker="s", ls="none", color=c, ms=7) for c in ORDINAL.values()]
    handles += [plt.Line2D([], [], **THEORY)]
    labels = [f"$p$ = {p:.2f}" for p in ORDINAL] + ["Theory"]
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               fontsize=7.8, columnspacing=1.6, handletextpad=0.4)
    save(fig, "fig5_biased_walks.png")


# ---------------------------------------------------------------------------
# Figure 6: finite-sample convergence
# ---------------------------------------------------------------------------

def figure6_convergence() -> None:
    summary = pd.read_csv(RESULTS / "convergence_summary.csv")
    rates = pd.read_csv(RESULTS / "convergence_rates_with_ci.csv").set_index("metric")
    labels = {
        "mean_position": "Mean",
        "variance": "Variance",
        "average_absolute_distance": "Mean distance",
        "endpoint_origin_probability": r"$P(S_n=0)$",
    }

    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    n_grid = np.logspace(2, np.log10(50_000), 50)
    ax.plot(n_grid, (n_grid / 100) ** -0.5, **THEORY, zorder=2)

    handles = [plt.Line2D([], [], **THEORY)]
    labels_out = [r"Theory: slope $-1/2$"]
    for (metric, label), colour, marker in zip(labels.items(), CATEGORICAL, MARKERS):
        sel = summary[summary["metric"] == metric].sort_values("trials")
        reference = float(sel["rmse"].iloc[0])   # normalise so all four share one axis
        r = rates.loc[metric]
        fitted = np.exp(r["intercept"]) * n_grid ** r["slope"] / reference
        ax.plot(n_grid, fitted, color=colour, lw=1.0, alpha=0.9, zorder=3)
        ax.plot(sel["trials"], sel["rmse"] / reference, marker=marker, ls="none", color=colour,
                ms=6, mec=SURFACE, mew=1.0, zorder=4)
        handles.append(plt.Line2D([], [], color=colour, marker=marker, ms=6, mec=SURFACE, mew=1.0, lw=1.0))
        labels_out.append(f"{label}: slope {r['slope']:.3f}".replace("-", "−"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(80, 62_000)
    ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10_000, 50_000]))
    ax.set_xticklabels(["100", "1,000", "10,000", "50,000"])
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0]))
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of Monte Carlo trials $N$")
    ax.set_ylabel("RMSE (relative to $N$ = 100)")
    ax.legend(handles, labels_out, loc="lower left", fontsize=7.8, handletextpad=0.5)
    fig.tight_layout()
    save(fig, "fig6_convergence.png")


# ---------------------------------------------------------------------------
# Figure 4: endpoint distributions vs exact and normal
# ---------------------------------------------------------------------------

def figure4_clt() -> None:
    samples = np.load(RESULTS / "endpoint_samples.npz")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5))
    for ax, n, letter in zip(axes, [10, 100, 1000], "ABC"):
        positions = samples[f"symmetric_n{n}"]
        half_width = int(4 * math.sqrt(n)) // 2 * 2
        support = np.arange(-min(n, half_width), min(n, half_width) + 1, 2)
        observed = np.array([(positions == s).mean() for s in support])
        exact = stats.binom.pmf((support + n) // 2, n, 0.5)
        ax.bar(support, observed, width=1.7, color="#b7d3f6", edgecolor=SURFACE, lw=0.4 if n < 1000 else 0,
               zorder=1)
        x = np.linspace(support.min(), support.max(), 400)
        ax.plot(x, 2 * stats.norm.pdf(x, 0, math.sqrt(n)), color=ACCENT, lw=1.3, zorder=2)
        ax.plot(support, exact, "o", ls="none", color=INK, ms=3.2 if n == 10 else 1.6, zorder=3)
        ax.set_xlim(support.min() - 1.5, support.max() + 1.5)
        ax.set_ylim(0, None)
        ax.set_xlabel("Final position $s$ (steps)")
        ax.grid(axis="x", visible=False)
        heading(ax, letter, f"$n$ = {n:,}")
    axes[0].set_ylabel("Probability $P(S_n = s)$")

    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_1, lw=0),
               plt.Line2D([], [], marker="o", ls="none", color=INK, ms=3.2),
               plt.Line2D([], [], color=ACCENT, lw=1.3)]
    labels = ["Simulation ($N$ = 10,000)", "Exact binomial", "Normal approximation"]
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               fontsize=7.8, columnspacing=1.8)
    save(fig, "fig4_endpoint_distributions.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    figure1_paths()
    figure2_symmetric()
    figure3_approximation()
    figure4_clt()
    figure5_biased()
    figure6_convergence()


if __name__ == "__main__":
    main()
