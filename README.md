# Supplementary materials

**Article:** When Do Large-Step Approximations Fail a Monte Carlo Test? Exact Benchmarks for Symmetric and Biased One-Dimensional Random Walks

**Author:** Seoyoung Choi

This folder contains what is needed to check the results in the article: the derivation of the exact mean distance, the code, the pinned package versions, and every numerical output, including the random-number seed of every simulated condition and repetition.

## Contents

| Path | What it is |
|:--|:--|
| `Supplementary_Material.docx` | Supplementary Table S1 (every estimate with its exact value, 95% confidence interval, and z-score), Table S2 (rejection counts of the repeated approximation test with Wilson intervals and predicted probabilities), Appendix S1 (proof of Equation 5, the exact mean distance from the origin), and Appendix S2 (parity and the error of the large-step approximation) |
| `code/` | Five Python scripts and `requirements.txt` (pinned package versions) |
| `results/` | All numerical outputs (CSV and NPZ), the run manifest, and a record of the software environment |
| `figures/` | Figures 1–6 of the article, written by `code/make_figures.py` |
| `manuscript/tables/` | Tables 1–4 and Tables S1–S2 in Markdown, written by `code/make_tables.py` |

## Reproducing the results

The analysis was run with Python 3.9.6 and the package versions in `code/requirements.txt` (NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.3, Matplotlib 3.9.4). The versions are pinned because NumPy does not guarantee identical random-number streams across versions.

Run these commands from this folder, in this order (on Windows, replace `.venv/bin/python` with `.venv\Scripts\python`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/random_walk_submission_analysis.py
.venv/bin/python code/extra_analyses.py
.venv/bin/python code/approximation_study.py
.venv/bin/python code/make_figures.py
.venv/bin/python code/make_tables.py
```

| Script | What it does |
|:--|:--|
| `random_walk_submission_analysis.py` | Symmetric, biased, and convergence simulations; exact finite-step theory; seeds and run manifest |
| `extra_analyses.py` | Standard errors, confidence intervals, z-scores, sample paths, drift thresholds, and the trials needed for 1% precision |
| `approximation_study.py` | The repeated approximation test (24 conditions, 200 repetitions each), Wilson confidence intervals for the rejection rates, and the trial counts at which Equation (8) gives a 10% and a 50% rejection probability |
| `make_figures.py` | Figures 1–6 |
| `make_tables.py` | Tables 1–4 and Tables S1–S2 |

The scripts overwrite the files in `results/`, `figures/`, and `manuscript/tables/`. The whole pipeline takes about a minute and a half on a laptop; most of that time is the approximation test.

Two runs from scratch on the computer used for the article produced byte-identical CSV and NPZ files and pixel-identical Figures 1–6. The diagnostic plots in `results/` can differ by a few pixels in cropping between runs. On another operating system or processor, floating-point values may differ in their last digits. `results/environment_info.txt` records the Python build and operating system of the run, so it differs between computers.

## Seeds

Every condition and repetition has its own seed, computed as

```
seed = 20260729 + 10,000,000 × analysis_id + 10,000 × condition_index + repetition_index
```

so a result does not depend on the order in which conditions are run. The generator is NumPy's default, PCG64 (`numpy.random.default_rng(seed)`).

| analysis_id | Analysis | Where each seed is stored |
|:-:|:--|:--|
| 1 | Symmetric walks, n = 10 to 5,000 | `seed` column of `results/symmetric_step_analysis.csv` and `results/symmetric_with_uncertainty.csv` |
| 2 | Biased 1,000-step walks, p = 0.50 to 0.70 | `seed` column of `results/biased_probability_analysis.csv` and `results/biased_with_uncertainty.csv` |
| 3 | Convergence study, 6 trial counts × 30 repetitions | `seed` column of `results/convergence_raw_results.csv` (180 rows) |
| 4 | Sample paths for Figure 1 | listed in `results/run_manifest.txt` |
| 5 | Approximation test, 24 conditions × 200 repetitions | `seed` column of `results/approximation_raw_results.csv` (4,800 rows) |

Any single row can be checked on its own. For example, a row of `approximation_raw_results.csv` with `steps = n`, `trials = N`, and `seed = s` is reproduced by

```python
import numpy as np
k = np.random.default_rng(s).binomial(n=n, p=0.5, size=N)   # number of right steps in each walk
print(np.mean(np.abs(2 * k - n)))                           # equals average_absolute_distance
```

## Which file supports which result

| Article item | Files in `results/` |
|:--|:--|
| Figure 1 | `sample_paths.npz` |
| Table 2, Figure 2, Table S1 (symmetric walks) | `symmetric_with_uncertainty.csv` |
| Figure 3, Table S2, and the approximation results | `approximation_rejection_rates.csv`, `approximation_raw_results.csv`, `approximation_critical_trials.csv`, `approximation_rule_constant.csv`, `approximation_summary.csv` |
| Figure 4 | `endpoint_samples.npz` |
| Table 3, Figure 5, Table S1 (biased walks) | `biased_with_uncertainty.csv`, `endpoint_samples.npz` |
| Table 4, Figure 6 | `convergence_raw_results.csv`, `convergence_summary.csv`, `convergence_rates_with_ci.csv`, `trials_for_1pct_precision.csv` |
| Drift thresholds quoted in the Discussion | `drift_thresholds.csv` |
| Design of every analysis | `run_manifest.txt` |

### The approximation test

Each repetition tests the null hypothesis that the expected simulated mean distance equals a benchmark, rejecting it when |z| > 1.96 (two-sided, 5% level). Against the exact value the null hypothesis is true, so about 5% of repetitions are expected to reject. Against the large-step approximation √(2n/π) the null hypothesis is false, so a rejection detects the approximation's error; it does not indicate an error in the simulation. The rejection probability rises smoothly with N: Equation (8) gives about 10% near N = 3.9n² and 50% near N = 35n² (the critical number of trials N\*) for the even step counts used here.

`approximation_raw_results.csv` has one row per repetition: the seed, the simulated mean distance from the origin, the exact value, the approximation √(2n/π), the standard error, the z-score against each benchmark, and whether each test rejected (`rejected_vs_exact`, `rejected_vs_approximation`; 1 = yes).

`approximation_rejection_rates.csv` has one row per condition (n, N):

| Columns | Meaning |
|:--|:--|
| `rejections_vs_approximation`, `rejection_rate_vs_approximation` | Number and fraction of the 200 repetitions that rejected the approximation |
| `wilson_lower_vs_approximation`, `wilson_upper_vs_approximation` | Wilson score 95% confidence interval for that fraction (shown as error bars in Figure 3A) |
| `predicted_rejection_probability` | Rejection probability given by Equation (8) |
| `observed_minus_predicted_in_se` | Observed fraction minus the predicted probability P, divided by √(P(1 − P)/200) |
| `predicted_within_wilson_interval` | Whether the predicted probability lies inside the Wilson interval (true for all 24 conditions) |
| `rejections_vs_exact`, `rejection_rate_vs_exact`, `wilson_lower_vs_exact`, `wilson_upper_vs_exact` | The same for the tests against the exact value |
| `first_seed`, `last_seed` | Seeds of the first and last repetition of the condition |

`approximation_critical_trials.csv` gives, for both statistics and n = 10 to 5,000, the exact value, the approximation, their difference A − θ, the per-walk standard deviation σ, the trial count N\* = [1.96σ/(A − θ)]² at which the rejection probability is 50%, the trial count at which it is 10%, and the rejection probability at N = 10,000.

The Wilson interval for x rejections in m repetitions, with p̂ = x/m and z = 1.96, is

```
[ p̂ + z²/(2m) ± z·√( p̂(1 − p̂)/m + z²/(4m²) ) ] / (1 + z²/m)
```

Unlike the interval p̂ ± 1.96·√(p̂(1 − p̂)/m), it does not collapse to zero width when all or none of the repetitions reject.

## Notes

- `figures/fig3_approximation_failure.png` is the revised Figure 3: rejection rates with Wilson intervals, and the trial counts for 10% and 50% rejection probabilities marked. All numbers behind it are unchanged from the submitted version.
- `results/figure1_*.png` to `results/figure9_*.png` are diagnostic plots written by the main script. They are not the figures of the article, which are in `figures/`.
- `results/requirements.txt` is written by the main script and lists only the three packages that script imports. Install from `code/requirements.txt`, which also includes SciPy.
