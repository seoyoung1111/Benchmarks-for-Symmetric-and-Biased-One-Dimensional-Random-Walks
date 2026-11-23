"""
When does a large-step approximation fail a Monte Carlo test?

A simulation of N walks estimates a statistic theta with standard error
SE = sigma / sqrt(N), where sigma is the per-walk standard deviation. The
estimate is tested against a benchmark B with a two-sided z-test at the 5% level:

    H0: E(estimate) = B,    z = (estimate - B) / SE,    reject H0 when |z| > 1.96.

If B is the exact value theta, H0 is true and the test rejects in about 5% of
simulations (the nominal rate). If B is an approximation A that differs from
theta, H0 is false, and a rejection correctly detects the approximation's error;
it is not evidence that the simulation is wrong.

Assuming that the estimate is approximately normal, that SE = sigma / sqrt(N),
and that A - theta does not depend on N, z is approximately normal with
standard deviation 1 and mean

    delta = (theta - A) / SE = -(A - theta) * sqrt(N) / sigma,

so the probability of rejecting the approximation is

    P(N) = Phi(-1.96 - delta) + 1 - Phi(1.96 - delta).

P(N) rises smoothly with N. It is 10% when |delta| = 0.652 and 50% when
|delta| = 1.96, that is, when

    N* = [1.96 * sigma / (A - theta)]^2.

These are reference points on a smooth curve, not sharp boundaries. When
theta != 0, N* = (1.96 * CV / r)^2 with CV = sigma / theta and
r = (A - theta) / theta.

For the mean distance from the origin with even n, r ~ 1/(4n) and
CV -> sqrt(pi/2 - 1) = 0.756, so N* ~ 35 n^2 and the 10% point is ~3.9 n^2.
(For odd n the approximation underestimates by about 1/(4n); all step counts
simulated here are even.) For the endpoint origin probability,
CV = sqrt((1 - P) / P) grows like n^(1/4), so N* grows faster, roughly like n^2.5.

This script (analysis_id = 5) checks the predicted rejection probability by
simulation: for several step counts n and trial counts N, it repeats the
simulation 200 times and records how often the simulated mean distance differs
from the approximation sqrt(2n/pi), and from the exact value, by more than 1.96
standard errors. Observed rejection rates are reported with Wilson score 95%
confidence intervals.

Run (after random_walk_submission_analysis.py and extra_analyses.py):
    python code/approximation_study.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

import random_walk_submission_analysis as base
from extra_analyses import absolute_distance_sd_exact, expected_absolute_distance_exact

OUTPUT_DIR = base.OUTPUT_DIR
ANALYSIS_ID_APPROXIMATION = 5
STEP_COUNTS = [10, 20, 50, 100]
TRIALS_PER_N_SQUARED = [1, 3, 10, 30, 100, 300]    # N = ratio * n^2
REPETITIONS = 200
Z_CRITICAL = 1.96


def asymptotic(metric: str, steps: int) -> float:
    if metric == "average_absolute_distance":
        return math.sqrt(2 * steps / math.pi)
    return math.sqrt(2 / (math.pi * steps))


def exact_and_sd(metric: str, steps: int) -> tuple[float, float]:
    if metric == "average_absolute_distance":
        return expected_absolute_distance_exact(steps, 0.5), absolute_distance_sd_exact(steps, 0.5)
    p0 = base.endpoint_origin_probability_exact(steps, 0.5)
    return p0, math.sqrt(p0 * (1 - p0))


def rejection_probability(delta: float) -> float:
    """Probability that |z| > 1.96 when z is normal with mean delta and SD 1 (Equation 8)."""
    return float(stats.norm.cdf(-Z_CRITICAL - delta) + 1 - stats.norm.cdf(Z_CRITICAL - delta))


# |delta| at which the rejection probability is 10%.
DELTA_10PCT = brentq(lambda d: rejection_probability(d) - 0.10, 0.0, Z_CRITICAL)


def wilson_interval(successes: int, trials: int, z: float = Z_CRITICAL) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def critical_trials(metric: str, steps: int) -> dict:
    exact, sd = exact_and_sd(metric, steps)
    approx = asymptotic(metric, steps)
    difference = approx - exact
    n_star = (Z_CRITICAL * sd / difference) ** 2          # 50% rejection probability (Equation 9)
    n_10pct = (DELTA_10PCT * sd / difference) ** 2        # 10% rejection probability
    return {
        "metric": metric,
        "steps": steps,
        "exact": exact,
        "asymptotic": approx,
        "difference": difference,
        "sd_per_walk": sd,
        "relative_error": difference / exact,
        "relative_error_times_4n": difference / exact * 4 * steps,
        "cv": sd / exact,
        "critical_trials": n_star,
        "critical_trials_over_n_squared": n_star / steps**2,
        "trials_for_10pct_rejection": n_10pct,
        "trials_for_10pct_rejection_over_n_squared": n_10pct / steps**2,
        "rejection_probability_at_10000_trials": rejection_probability(-difference / (sd / 100.0)),
    }


def run_approximation_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (summary, raw). `summary` has one row per (n, N) condition; `raw`
    has one row per repetition, with the seed that reproduces it.
    """
    metric = "average_absolute_distance"
    rows = []
    raw_rows = []
    for condition_index, (steps, ratio) in enumerate(
        (s, r) for s in STEP_COUNTS for r in TRIALS_PER_N_SQUARED
    ):
        trials = ratio * steps**2
        exact, sd = exact_and_sd(metric, steps)
        approx = asymptotic(metric, steps)
        se = sd / math.sqrt(trials)
        z_vs_approx, z_vs_exact = [], []
        for repetition in range(REPETITIONS):
            seed = base.condition_seed(ANALYSIS_ID_APPROXIMATION, condition_index, repetition)
            positions = base.simulate_final_positions(steps, trials, 0.5, seed)
            estimate = float(np.mean(np.abs(positions)))
            z_vs_approx.append((estimate - approx) / se)
            z_vs_exact.append((estimate - exact) / se)
            raw_rows.append(
                {
                    "analysis": "approximation_test",
                    "condition_index": condition_index,
                    "repetition_index": repetition,
                    "seed": seed,
                    "steps": steps,
                    "trials": trials,
                    "trials_over_n_squared": ratio,
                    "p_right": 0.5,
                    "average_absolute_distance": estimate,
                    "exact_value": exact,
                    "approximation": approx,
                    "standard_error": se,
                    "z_vs_exact": z_vs_exact[-1],
                    "z_vs_approximation": z_vs_approx[-1],
                    "rejected_vs_exact": int(abs(z_vs_exact[-1]) > Z_CRITICAL),
                    "rejected_vs_approximation": int(abs(z_vs_approx[-1]) > Z_CRITICAL),
                }
            )
        z_vs_approx, z_vs_exact = np.array(z_vs_approx), np.array(z_vs_exact)
        delta = (exact - approx) / se
        predicted = rejection_probability(delta)
        rejections_approx = int(np.sum(np.abs(z_vs_approx) > Z_CRITICAL))
        rejections_exact = int(np.sum(np.abs(z_vs_exact) > Z_CRITICAL))
        rate_approx = rejections_approx / REPETITIONS
        lower_approx, upper_approx = wilson_interval(rejections_approx, REPETITIONS)
        lower_exact, upper_exact = wilson_interval(rejections_exact, REPETITIONS)
        rows.append(
            {
                "steps": steps,
                "trials": trials,
                "trials_over_n_squared": ratio,
                "repetitions": REPETITIONS,
                "condition_index": condition_index,
                "first_seed": base.condition_seed(ANALYSIS_ID_APPROXIMATION, condition_index, 0),
                "last_seed": base.condition_seed(ANALYSIS_ID_APPROXIMATION, condition_index, REPETITIONS - 1),
                # Tests against the approximation sqrt(2n/pi): H0 is false, so
                # rejections detect the approximation's error.
                "rejections_vs_approximation": rejections_approx,
                "rejection_rate_vs_approximation": rate_approx,
                "wilson_lower_vs_approximation": lower_approx,
                "wilson_upper_vs_approximation": upper_approx,
                "predicted_rejection_probability": predicted,
                # Observed rate minus prediction, in binomial standard errors
                # computed from the prediction: sqrt(P (1 - P) / repetitions).
                "observed_minus_predicted_in_se": (rate_approx - predicted)
                / math.sqrt(predicted * (1 - predicted) / REPETITIONS),
                "predicted_within_wilson_interval": bool(lower_approx <= predicted <= upper_approx),
                # Tests against the exact value: H0 is true, so about 5% of
                # repetitions are expected to reject.
                "rejections_vs_exact": rejections_exact,
                "rejection_rate_vs_exact": rejections_exact / REPETITIONS,
                "wilson_lower_vs_exact": lower_exact,
                "wilson_upper_vs_exact": upper_exact,
                "mean_z_vs_approximation": float(np.mean(z_vs_approx)),
                "predicted_mean_z": delta,
            }
        )
        print(f"n={steps:4d} N={trials:>9,d}  rejected {rejections_approx:3d}/{REPETITIONS}"
              f"  predicted {predicted:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


def main() -> None:
    critical = pd.DataFrame(
        [critical_trials(m, s) for m in ["average_absolute_distance", "endpoint_origin_probability"]
         for s in [10, 20, 50, 100, 200, 500, 1000, 5000]]
    )
    critical.to_csv(OUTPUT_DIR / "approximation_critical_trials.csv", index=False)
    simulated, raw = run_approximation_test()
    simulated.to_csv(OUTPUT_DIR / "approximation_rejection_rates.csv", index=False)
    raw.to_csv(OUTPUT_DIR / "approximation_raw_results.csv", index=False)

    se_rate = np.sqrt(simulated.predicted_rejection_probability * (1 - simulated.predicted_rejection_probability)
                      / simulated.repetitions).replace(0, np.nan)
    deviation_in_se = ((simulated.rejection_rate_vs_approximation - simulated.predicted_rejection_probability).abs()
                       / se_rate)
    pd.DataFrame([{
        "conditions": len(simulated),
        "repetitions_per_condition": REPETITIONS,
        "total_rejections_vs_exact": int(simulated.rejections_vs_exact.sum()),
        "total_repetitions": int(simulated.repetitions.sum()),
        "mean_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.mean(),
        "min_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.min(),
        "max_rejection_rate_vs_exact": simulated.rejection_rate_vs_exact.max(),
        "max_abs_observed_minus_predicted": (simulated.rejection_rate_vs_approximation
                                             - simulated.predicted_rejection_probability).abs().max(),
        "max_deviation_in_se": deviation_in_se.max(),
        "conditions_with_prediction_in_wilson_interval": int(simulated.predicted_within_wilson_interval.sum()),
    }]).to_csv(OUTPUT_DIR / "approximation_summary.csv", index=False)

    # Large-n limits for the mean distance, where N* / n^2 -> (1.96 * 4 * CV)^2,
    # and the trial counts giving a 10% rejection probability at each n.
    cv_limit = math.sqrt(math.pi / 2 - 1)
    limit = (Z_CRITICAL * 4 * cv_limit) ** 2
    limit_10 = (DELTA_10PCT * 4 * cv_limit) ** 2
    exact_10 = {}
    for steps in [10, 20, 50, 100]:
        exact, sd = exact_and_sd("average_absolute_distance", steps)
        exact_10[steps] = (DELTA_10PCT * sd / (asymptotic("average_absolute_distance", steps) - exact)) ** 2
    pd.DataFrame([{
        "limit_critical_trials_over_n_squared": limit,
        "cv_limit": cv_limit,
        "delta_for_10pct_rejection": DELTA_10PCT,
        "limit_10pct_trials_over_n_squared": limit_10,
        "steps_needed_for_10pct_at_10000_trials": math.sqrt(10_000 / limit_10),
        **{f"trials_for_10pct_n{k}": v for k, v in exact_10.items()},
        **{f"predicted_rejection_n{k}_N10000": rejection_probability(
            -(asymptotic("average_absolute_distance", k) - exact_and_sd("average_absolute_distance", k)[0])
            / (exact_and_sd("average_absolute_distance", k)[1] / 100.0)) for k in [10, 20, 50, 100]},
    }]).to_csv(OUTPUT_DIR / "approximation_rule_constant.csv", index=False)

    manifest_path = OUTPUT_DIR / "run_manifest.txt"
    if "5. Approximation" not in manifest_path.read_text(encoding="utf-8"):
        with open(manifest_path, "a", encoding="utf-8") as manifest:
            manifest.write(
                "\n5. Approximation test (approximation_study.py)\n"
                f"    Step counts: {STEP_COUNTS}; trials N = ratio * n^2 for ratio in {TRIALS_PER_N_SQUARED}\n"
                f"    Repetitions per condition: {REPETITIONS}; p_right: 0.50; seed family analysis_id = 5.\n"
                "    The seed and outcome of every repetition are stored in approximation_raw_results.csv.\n"
            )
    print(critical.round(4).to_string(index=False))
    print(f"large-n limit of N*/n^2: {limit:.2f}; 10% rejection point: {limit_10:.2f} n^2")


if __name__ == "__main__":
    main()
