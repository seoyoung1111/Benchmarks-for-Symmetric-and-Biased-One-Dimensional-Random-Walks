"""
Manuscript tables (pandoc pipe tables) generated from results/*.csv.

Writes manuscript/tables/table{1..4}.md; paper.md includes them, so no
number in a table is ever typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import random_walk_submission_analysis as base

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "manuscript" / "tables"


def pipe_table(header: list[str], rows: list[list[str]], align: str,
               widths: list[int] | None = None) -> str:
    """Pipe table; `widths` sets relative column widths (pandoc reads them from the dashes)."""
    widths = widths or [2] * len(align)
    marks = {"l": lambda w: ":" + "-" * w, "r": lambda w: "-" * w + ":", "c": lambda w: ":" + "-" * w + ":"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(marks[a](w) for a, w in zip(align, widths)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def ci(row, metric: str, digits: int) -> str:
    return f"{row[metric]:,.{digits}f} ({row[metric + '_ci_lower']:,.{digits}f}, {row[metric + '_ci_upper']:,.{digits}f})".replace("-", "−")


def z(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def table1_design() -> str:
    rows = [
        ["Symmetric comparison", ", ".join(f"{n:,}" for n in base.SYMMETRIC_STEP_COUNTS),
         f"{base.SYMMETRIC_TRIALS:,}", "0.50", "1", "1"],
        ["Biased comparison", f"{base.BIASED_STEP_COUNT:,}", f"{base.BIASED_TRIALS:,}",
         ", ".join(f"{p:.2f}" for p in base.BIASED_PROBABILITIES), "1", "2"],
        ["Convergence study", f"{base.CONVERGENCE_STEP_COUNT:,}",
         ", ".join(f"{n:,}" for n in base.CONVERGENCE_TRIAL_COUNTS), f"{base.CONVERGENCE_P_RIGHT:.2f}",
         str(base.CONVERGENCE_REPETITIONS), "3"],
        ["Sample paths", "1,000", "20 paths", "0.50, 0.55", "1", "4"],
        ["Approximation test", "10, 20, 50, 100", "$n^2 \\times$ (1, 3, 10, 30, 100, 300)", "0.50",
         "200", "5"],
    ]
    return pipe_table(["Analysis", "Steps $n$", "Trials $N$", "$p$", "Repetitions", "Seed family"],
                      rows, "llllcc", widths=[19, 21, 22, 12, 15, 11])


def table2_symmetric() -> str:
    """Simplified main-text table: simulated value with exact theory in brackets."""
    d = pd.read_csv(RESULTS / "symmetric_with_uncertainty.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([
            f"{int(r.steps):,}",
            f"{r.mean_position:.3f} (0)".replace("-", "−"),
            f"{r.variance:,.1f} ({r.variance_theory:,.0f})",
            f"{r.average_absolute_distance:.2f} ({r.average_absolute_distance_theory:.2f})",
            f"{r.endpoint_origin_probability:.4f} ({r.endpoint_origin_probability_theory:.4f})",
        ])
    return pipe_table(
        ["Steps $n$", "Mean position", "Variance", "Mean distance from origin",
         "Probability of ending at 0"],
        rows, "rrrrr")


def table3_biased() -> str:
    d = pd.read_csv(RESULTS / "biased_with_uncertainty.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([
            f"{r.p_right:.2f}",
            f"{r.mean_position:.2f} ({r.mean_position_theory:.0f})".replace("-", "−"),
            f"{r.variance:,.1f} ({r.variance_theory:,.0f})",
            f"{r.drift_to_spread_ratio:.1f}",
        ])
    return pipe_table(
        ["Probability of a right step $p$", "Mean position", "Variance", "Drift ÷ spread"],
        rows, "rrrr")


def supplementary_full_table() -> str:
    """Every estimate with its 95% CI and z-score, for the supplementary material."""
    rows = []
    for name in ["symmetric", "biased"]:
        d = pd.read_csv(RESULTS / f"{name}_with_uncertainty.csv")
        for _, r in d.iterrows():
            for metric, label in [("mean_position", "Mean"), ("variance", "Variance"),
                                  ("average_absolute_distance", "Mean distance"),
                                  ("endpoint_origin_probability", "$P(S_n = 0)$")]:
                digits = 4 if metric == "endpoint_origin_probability" else 3
                theory = r[metric + "_theory"]
                if theory >= 10 ** -digits or theory == 0:
                    theory_text = f"{theory:,.{digits}f}"
                else:
                    mantissa, exponent = f"{theory:.1e}".split("e")
                    theory_text = f"{mantissa} × 10^{int(exponent)}^".replace("-", "−")
                rows.append([f"{int(r.steps):,}", f"{r.p_right:.2f}", label,
                             f"{r[metric]:,.{digits}f}".replace("-", "−"), theory_text,
                             ci(r, metric, digits).split(" ", 1)[1], z(r[metric + "_z"])])
    return pipe_table(["$n$", "$p$", "Statistic", "Simulated", "Exact theory", "95% CI", "$z$"],
                      rows, "rrlrrlr", widths=[8, 7, 17, 13, 15, 31, 9])


def supplementary_approximation_table() -> str:
    """Rejection counts of the repeated approximation test (Figure 3A), for the supplementary material."""
    d = pd.read_csv(RESULTS / "approximation_rejection_rates.csv")

    def pct(x: float) -> str:
        if round(x, 9) == 1:
            return "100.0"
        return ">99.9" if x > 0.999 else f"{100 * x:.1f}"

    rows = []
    for _, r in d.iterrows():
        rows.append([
            f"{int(r.steps):,}", f"{int(r.trials):,}", f"{int(r.trials_over_n_squared):,}",
            f"{int(r.rejections_vs_approximation)}",
            f"{pct(r.rejection_rate_vs_approximation)} ({pct(r.wilson_lower_vs_approximation)}, "
            f"{pct(r.wilson_upper_vs_approximation)})",
            f"{100 * r.predicted_rejection_probability:.2f}",
            z(r.observed_minus_predicted_in_se),
            f"{int(r.rejections_vs_exact)}",
        ])
    return pipe_table(
        ["$n$", "$N$", "$N/n^2$", "Rejections of $\\sqrt{2n/\\pi}$ (of 200)",
         "Rejection rate, % (Wilson 95% CI)", "Predicted, % (Eq. 8)", "Difference (SE)",
         "Rejections of exact value (of 200)"],
        rows, "rrrrlrrr", widths=[6, 11, 7, 14, 22, 12, 12, 16])


def table4_convergence() -> str:
    d = pd.read_csv(RESULTS / "convergence_rates_with_ci.csv")
    need = pd.read_csv(RESULTS / "trials_for_1pct_precision.csv").set_index("metric")["trials_needed"]
    names = {
        "mean_position": "Mean position",
        "variance": "Variance",
        "average_absolute_distance": "Mean distance from origin",
        "endpoint_origin_probability": "Probability of ending at 0",
    }
    rows = []
    for _, r in d.iterrows():
        rel = "—" if pd.isna(r.relative_rmse_percent_at_10000) else f"{r.relative_rmse_percent_at_10000:.2f}%"
        n_needed = f"{need[r.metric]:,.0f}" if r.metric in need.index else "—"
        rows.append([
            names[r.metric],
            f"{r.slope:.3f} ({r.slope_ci_lower:.3f}, {r.slope_ci_upper:.3f})".replace("-", "−"),
            f"{r.r_squared:.3f}", rel, n_needed,
        ])
    return pipe_table(
        ["Estimator", "Fitted slope $b$ (95% CI)", "$R^2$", "Relative RMSE at $N$ = 10,000",
         "$N$ for 1% relative SE"],
        rows, "lrrrr", widths=[24, 28, 10, 18, 20])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for i, builder in enumerate([table1_design, table2_symmetric, table3_biased, table4_convergence], 1):
        (TABLES / f"table{i}.md").write_text(builder(), encoding="utf-8")
        print("wrote", TABLES / f"table{i}.md")
    (TABLES / "tableS1.md").write_text(supplementary_full_table(), encoding="utf-8")
    (TABLES / "tableS2.md").write_text(supplementary_approximation_table(), encoding="utf-8")


if __name__ == "__main__":
    main()
