# Topological Recurrence in EEG Dynamics

### A Complementary, Wake-to-REM-Graded Descriptor of Sleep-Stage Organisation

This repository contains the consolidated analysis code for the manuscript:

> **McMahon, E.** — *Topological Recurrence in EEG Dynamics: A Complementary, Wake-to-REM-Graded Descriptor of Sleep-Stage Organisation.*

The pipeline reconstructs single-channel EEG state-space dynamics with delay embedding, quantifies recurrence via persistent homology (the **K0** metric = within-subject z-score of H1 total persistence), and benchmarks K0 against conventional EEG measures across the **Sleep-EDF Expanded** dataset.

The entire analysis runs from a single command, `python run.py`, which
orchestrates four stages: `pipeline.py` (core TDA + baselines + stats),
`collate.py` (→ `results.xlsx`), `figures.py` (all figures), and the six
standalone supplementary-analysis scripts (Stage 4 — see §4c).

```
sleep-edf-topological-recurrence/
├── run.py                 # interactive shell orchestrator — runs all 4 stages
├── scripts/
│   ├── pipeline.py                    # Stage 1 — core analysis (TDA + baselines + stats)
│   ├── collate.py                     # Stage 2 — outputs → single results.xlsx
│   ├── figures.py                     # Stage 3 — all manuscript / supplementary figures
│   ├── demographics_breakdown.py      # Stage 4 — cohort demographics (§4c)
│   ├── stratified_effects.py          # Stage 4 — lifespan/sex/drug strata (§4c)
│   ├── sensitivity_power_analysis.py  # Stage 4 — power / MDE analysis (§4c)
│   ├── supplementary_round3.py        # Stage 4 — diagnostic embedding, LOSO CIs, ICA, PE-order (§4c)
│   ├── supplementary_round4.py        # Stage 4 — regime-specific AUCs, EOG-corrected band power (§4c)
│   └── refit_wake_robustness_stats.py # Stage 4 — wake-robustness inner-LM re-fit (§4c)
├── config.env             # SLEEP_EDF_ROOT (relative path)
├── requirements.txt
├── install_packages.py
├── readme.md
├── license.txt
└── .gitignore
```

Outputs (CSVs, NPZs, figures) land in `outputs/` and are not committed.

---

## 1. Install

Python ≥ 3.10 is required.

```bash
git clone <repository-url> sleep-edf-topological-recurrence
cd sleep-edf-topological-recurrence
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # or:  python install_packages.py
```

The required libraries are NumPy, pandas, SciPy, MNE, ripser, persim,
matplotlib, statsmodels, openpyxl, scikit-learn, joblib (joblib is
optional but strongly recommended; it enables `--n-jobs > 1` for per-night
parallelism). scikit-learn is used by the LOSO classification step.

---

## 2. Get the dataset

Download **Sleep-EDF Expanded (1.0.0)** from PhysioNet:

> https://physionet.org/content/sleep-edfx/1.0.0/

Extract it so the layout looks like:

```
sleep-edf-topological-recurrence/
└── data/
    └── sleep-edfx/
        └── sleep-edf-database-expanded-1.0.0/
            ├── sleep-cassette/
            └── sleep-telemetry/
```

The pipeline auto-detects this location. To use any other path, edit
`config.env`:

```
SLEEP_EDF_ROOT=/absolute/or/relative/path/to/sleep-edf-database-expanded-1.0.0
```

The published analysis uses the 197 paired Sleep-Cassette PSG/hypnogram nights.

---

## 3. Run the analysis

```bash
python run.py
```

`run.py` will:

1. Verify Python version + required packages
2. Locate the dataset and report PSG / hypnogram counts
3. Prompt for the number of parallel workers (`--n-jobs`); pick e.g. 8 on a
   modern laptop. Pass `--n-jobs 1` (or accept the prompt default) to run
   sequentially
4. **Stage 1** — Run `scripts/pipeline.py` end-to-end, narrating each of its 29 steps
5. **Stage 2** — Run `scripts/collate.py` to assemble `outputs/results.xlsx`
6. **Stage 3** — Run `scripts/figures.py` to generate every figure into `outputs/figures/`
7. **Stage 4** — Run the six standalone supplementary-analysis scripts in order
   (`demographics_breakdown.py` → `stratified_effects.py` →
   `sensitivity_power_analysis.py` → `supplementary_round3.py` →
   `supplementary_round4.py` → `refit_wake_robustness_stats.py`; see §4c)
8. Mirror everything to `outputs/logs/run_YYYYMMDD_HHMMSS.log`

A complete run on a typical workstation takes a few hours wall-clock when
parallelised. **Save-as-you-go:** every step writes its output as soon as it
finishes, so re-running `python run.py` resumes from the last completed step.

### Useful flags

```bash
python run.py --n-jobs 8         # skip the interactive prompt
python run.py --force            # re-run every step from scratch
python run.py --only-collate     # rebuild results.xlsx from existing CSVs
python run.py --only-figures     # rebuild figures from existing CSVs
python run.py --only-extras      # rebuild the stage-4 supplementary analyses only
python run.py --skip-figures     # pipeline + collate + extras, no figures
python run.py --skip-extras      # pipeline + collate + figures, no extras
```

### Running a single stage manually

```bash
python scripts/pipeline.py --n-jobs 8 --only main_tda,baselines
python scripts/collate.py
python scripts/figures.py --only heatmap,k0_subject_lines
```

`pipeline.py --help` lists all 29 step names; `figures.py --help` lists all
figure ids.

---

## 4. What `pipeline.py` does

| Step | What it computes | Output(s) |
| --- | --- | --- |
| `main_tda` | Per-epoch H0/H1 persistence on EEG Fpz-Cz, m=10, τ=2, max 30 epochs/stage/night | `tda_epoch_features_all.csv`, `tda_stage_summary_all.csv` |
| `robustness_grid` | TDA across m∈{6,8,10,12} × τ∈{1,2,4} | `tda_robustness_grid_epochs.csv` |
| `baselines` | Welch δ/θ/α/σ/β power, spectral entropy, permutation entropy, LZ76 | `baseline_epoch_features_all.csv` |
| `wake_subclasses` | Track-2 wake labels (recording-percentile + 6-vote rule) → W_quiet / W_active_ocular / W_bad | `wake_epoch_subclasses.csv`, `wake_subclass_summary.csv` |
| `wake_qc` | Track-3 richer wake-QC table (per-subject MAD-based thresholds + 3-vote ocular rule) | `wake_qc_epoch_table.csv` |
| `tda_wake_subclasses` | TDA on REM + W_quiet + W_active_ocular | `tda_epoch_features_wake_subclasses.csv`, `tda_stage_summary_wake_subclasses.csv` |
| `corrected_eeg` | Full-recording linear regression β = cov(eeg,eog)/var(eog), saves NPZ epoch bundles | `outputs/corrected_epochs/*.npz`, `corrected_epochs_manifest.csv` |
| `tda_track3` | TDA on raw / EOG-corrected / EOG-channel for wake subclasses | 6 CSVs (`*_wake_raw`, `*_wake_corrected`, `*_eog`) |
| `wake_subclass_robustness` | Wake-subclass TDA grid m∈{8,10,12} × τ∈{1,2,3} | `wake_subclass_robustness_*.csv` (3 files) |
| `main_robustness_mixedlm` | Mixed-LM (Powell, REML=False) + planned contrasts (REM−W, REM−N3, N1−N3) on K0 | `tda_robustness_mixedlm_omnibus.csv`, `tda_robustness_mixedlm_planned_contrasts.csv` |
| `baseline_mixedlm` | Same on baseline metrics | `baseline_mixedlm_omnibus.csv`, `baseline_mixedlm_planned_contrasts.csv` |
| `wake_subclass_mixedlm` | Mixed-LM on wake-subclass TDA | `tda_wake_subclasses_mixedlm_*.csv` |
| `baseline_wake_mixedlm` | Mixed-LM on baselines restricted to wake subclasses | `baseline_wake_subclasses_mixedlm_*.csv` |
| `track3_mixedlm` | Mixed-LM on raw / corrected / EOG TDA | 6 CSVs (`raw_wake_…`, `corrected_wake_…`, `eog_wake_…`) |
| `incremental_glm` | Binomial GLM with subject FE: does K0_tot add information beyond band power? (Models A/B/C/D + LR tests) | `incremental_k0_vs_bandpower_*.csv` (3 files) |
| `review_*` | Stage means, paired Cohen's d_z, summary tables (TDA + baselines + incremental) | 9 review CSVs |
| `comparison_table` | Headline manuscript table: K0 vs every baseline metric for REM vs wake subclasses | `comparison_table_rem_vs_wake_subclasses{,_long}.csv` |
| `supplementary_table` | Wake-subclass contrasts + incremental K0 | `supplementary_table_wake_subclass_and_incremental_results{,_long}.csv` |

---

## 4b. Robustness, sensitivity, and predictive-validity analyses

Nine additional steps extend the headline analysis with robustness checks,
sensitivity analyses, and out-of-subject predictive performance. They run
after the headline analyses and reuse the same `RNG_SEED=0`, embedding
parameters, mixed-LM specification, and Holm correction.

| Step | What it computes | Output(s) |
| --- | --- | --- |
| `all_pairwise` | All 10 pairwise stage contrasts on K0 (W vs N1, W vs N2, … N3 vs REM) + per-stage descriptives + per-subject Spearman monotonicity test | `stage_descriptives_all.csv`, `stage_all_pairwise_contrasts.csv`, `stage_monotonicity.csv` |
| `cohort_replication` | Refit headline contrasts on Sleep-Cassette (`SC*`) and Sleep-Telemetry (`ST*`) cohorts separately, for K0 and every baseline metric | `cohort_replication_contrasts.csv` |
| `subsampling_stability` | Post-hoc resampling at caps {5, 10, 15, 20, 25, 30} × 10 replicates, refitting the headline mixed-LM each time | `subsampling_stability_contrasts.csv` |
| `bootstrap_contrasts` | 1000 subject-level bootstrap resamples; percentile 95% CIs on REM−W, REM−N3, N1−N3 | `bootstrap_contrasts.csv` |
| `embedding_diagnostics` | Per-recording AMI(τ), τ ∈ 1..20 and FNN(m), m ∈ 1..15 (τ = 2 fixed) on a 30-recording subset; suggested optimal τ and m | `embedding_ami.csv`, `embedding_fnn.csv`, `embedding_diagnostics_summary.csv` |
| `main_tda_pz_oz` | Replicate the headline TDA on the EEG Pz-Oz channel (where present) | `tda_epoch_features_pz_oz.csv`, `tda_pz_oz_mixedlm_omnibus.csv`, `tda_pz_oz_mixedlm_planned_contrasts.csv` |
| `preprocessing_sensitivity` | REM−W contrast across bandpass {0.5–30, 0.5–40, 0.5–45, 1–40} Hz × resample {50, 100, 128} Hz on 30 random nights | `preprocessing_sensitivity_contrasts.csv` |
| `diagnostics` | Mixed-LM assumption checks: Shapiro-Wilk (residuals), Levene's (across stages), residual skew + excess kurtosis | `statistical_diagnostics.csv` |
| `classification` | LOSO logistic regression + random forest, 3 feature sets (K0 alone, band power alone, combined), 3 targets (REM-vs-W, REM-vs-NREM, REM-vs-other); reports AUC, balanced accuracy, F1, sensitivity, specificity per fold | `classification_loso_metrics.csv`, `classification_summary.csv` |

Run any subset directly:

```bash
python scripts/pipeline.py --only classification,bootstrap_contrasts
python scripts/pipeline.py --only all_pairwise,cohort_replication
```

The corresponding figures are emitted by `figures.py` under the ids
`all_pairwise`, `subsampling`, `bootstrap`, `embedding_diag`, `pz_oz`,
`preproc_sensitivity`, `classification`, and `cohort_replication`.

---

## 4c. Stage 4 — standalone supplementary-analysis scripts

Six scripts in `scripts/` cover the demographic, lifespan, pharmacological,
power, and supplementary robustness / sensitivity analyses. They depend on
the main pipeline's output CSVs, so `run.py` runs them **automatically as
Stage 4, after `pipeline.py`**, in the order below.

You only need the commands below if you want to run them standalone (e.g. to
rebuild just these outputs, or after editing one) — or use
`python run.py --only-extras`:

```bash
# 1. Cohort demographics — run first; #2 and #3 depend on its output
python scripts/demographics_breakdown.py

# 2. Lifespan / sex / drug stratified effects
python scripts/stratified_effects.py

# 3. Sensitivity power analysis (MDE / observed d_z)
python scripts/sensitivity_power_analysis.py

# 4. Round-3 supplementary analyses (diagnostic embedding, LOSO CIs, ICA, PE-order)
python scripts/supplementary_round3.py --n-jobs 8

# 5. Round-4 supplementary analyses (regime-specific AUCs, EOG-corrected band power)
python scripts/supplementary_round4.py --n-jobs 8

# 6. Wake-robustness inner-LM re-fit (idempotent; takes ~seconds, not the ~1.5 h TDA)
python scripts/refit_wake_robustness_stats.py
```

`demographics_breakdown.py` must run before `stratified_effects.py` and
`sensitivity_power_analysis.py` (they read `demographics_per_night.csv`).
`supplementary_round3.py` and `supplementary_round4.py` depend only on the
pipeline's output CSVs and accept `--n-jobs`; `supplementary_round3.py`
Section C re-processes raw EDFs for the ICA control.
`refit_wake_robustness_stats.py` is LM-only (no per-night TDA), deterministic,
and idempotent — running it on a successful pipeline rewrites the same two
output CSVs with identical results, and it doubles as a defensive recovery
against the rare statsmodels singular-matrix case seen on some Python /
statsmodels combinations.

| Script | Depends on | What it computes | Output(s) |
| --- | --- | --- | --- |
| `demographics_breakdown.py` | `SC-subjects.xls`, `ST-subjects.xls`, the dataset PSG/hypnogram pairs | Demographic breakdown of the 197-night / 100-subject cohort: age, sex, cohort, and (Telemetry) drug-protocol assignment | `demographics_per_night.csv`, `demographics_per_subject.csv`, `demographics_summary.csv`, `figures/demographics_age_distribution.png` |
| `stratified_effects.py` | `tda_epoch_features_all.csv` (pipeline), `demographics_per_night.csv` | Headline REM−Wake K0 contrast stratified by lifespan × sex, and by the Sleep-Telemetry within-subject Temazepam crossover (placebo vs drug, stage × drug interaction, per-subject paired) | `strat_lifespan_age_sex_remw.csv`, `strat_drug_remw_by_condition.csv`, `strat_drug_stage_x_drug.csv`, `strat_drug_subject_paired_remw.csv`, `figures/strat_forest_age_sex_remw.png`, `figures/strat_drug_remw.png` |
| `sensitivity_power_analysis.py` | `tda_epoch_features_all.csv` (pipeline), `demographics_per_night.csv` | Sensitivity power analysis for the headline contrast across six cohort specifications: observed paired-t d_z, minimum detectable effect at 80% power (α = 0.05), and a power-vs-effect-size grid | `sensitivity_power_analysis.csv`, `sensitivity_power_curves.csv`, `figures/sensitivity_power_curves.png` |
| `supplementary_round3.py` | `tda_epoch_features_all.csv`, `baseline_epoch_features_all.csv`, `tda_epoch_features_wake_subclasses.csv` (pipeline); raw EDFs for the ICA section | Four round-3 supplementary analyses: diagnostic-favoured embedding cell (AMI τ = 11), LOSO classification bootstrap CIs + paired-bootstrap ΔAUC, ICA-based ocular-artefact sensitivity on Pz–Oz, and permutation-entropy order sweep (orders 3–6) | `supp_a_diagnostic_embedding_*.csv`, `supp_b_loso_*.csv`, `supp_c_ica_pz_oz_*.csv`, `supp_d_pe_order_*.csv`, plus ROC / K0-distribution figures |
| `supplementary_round4.py` | `tda_epoch_features_all.csv`, `baseline_epoch_features_all.csv`, `wake_epoch_subclasses.csv`, `corrected_epochs/*.npz` (pipeline) | Two round-4 supplementary analyses: regime-specific LOSO AUCs (REM vs quiet wake, REM vs active-ocular wake) with bootstrap CIs, and EOG-regressed band-power LOSO control | `supp_d_regime_*.csv`, `supp_d_corrected_bandpower_*.csv`, plus regime ROC figures |
| `refit_wake_robustness_stats.py` | `wake_subclass_robustness_grid.csv` (pipeline) | Re-fits the `wake_subclass_robustness` inner mixed-LMs from the existing grid CSV without re-running the ~1.5 h per-night TDA — Powell-first optimiser, deterministic and idempotent. Runs as the final Stage 4 step; also doubles as a defensive recovery against the rare statsmodels singular-matrix case seen on some Python combinations. | `wake_subclass_robustness_mixedlm_omnibus.csv`, `wake_subclass_robustness_planned_contrasts.csv` |

The three analysis scripts each accept `--force` (recompute) and `--no-figure`
(skip the PNG); `stratified_effects.py` additionally accepts `--age-bins` and
`--tel-policy`. None of the four is required to reproduce the headline result —
they generate the demographic, lifespan, pharmacological, and power-analysis
supplementary numbers reported in the manuscript.

---

## 5. What `collate.py` produces

A single Excel workbook, `outputs/results.xlsx`, with **~40 sheets** —
one per logical results section. Sheet `README` is an index that lists
every other sheet, the source CSV, the row count, and a description.

---

## 6. What `figures.py` produces

All figures land in `outputs/figures/`. Available:

| Figure id | File(s) | Notes |
| --- | --- | --- |
| `heatmap` | `robustness_heatmap_K0_tot_REM-W*.png` | **Figure 1** — K0 effect-size heatmap across (m, τ) |
| `fig2_example` | `fig2A_…trajectory.png`, `fig2B_…`, `fig2C_…h1_pd.png`, `fig2D_…`, `fig2_panels.png` | **Figure 2** — embedded trajectories + persistence diagrams (re-processes raw EDFs) |
| `wake_counts` | `wake_subclass_counts.png` | Epoch counts per wake subclass |
| `qc_distributions` | `qc_feature_distributions.png` | EOG / EMG / EEG-EOG distributions |
| `boxplot_raw` / `_corrected` / `_eog` | `*_tda_boxplots_h1tot.png` | H1_totpers boxplots per signal source |
| `contrast_estimates` | `contrast_estimates_h1tot.png` | Planned contrast estimates with 95% CIs |
| `k0_mean_sem` | `fig_k0tot_by_stage_mean_sem.png` | K0_tot bar chart with SEM |
| `k0_subject_lines` | `fig_k0tot_by_stage_subject_lines.png` | Per-subject paired spaghetti |
| `k0_rem_contrasts` | `fig_k0tot_rem_contrasts_subject_lines.png` | Two-panel REM contrasts |
| `all_pairwise` | `all_pairwise_contrasts_heatmap.png` | All 10 pairwise stage contrasts on K0 |
| `subsampling` | `subsampling_stability.png` | REM−W stability across cap sizes |
| `bootstrap` | `bootstrap_contrast_cis.png` | Bootstrap 95% CIs on headline contrasts |
| `embedding_diag` | `embedding_diagnostics.png` | AMI(τ) + FNN(m) curves justifying m=10, τ=2 |
| `pz_oz` | `pz_oz_contrasts.png` | Fpz-Cz vs Pz-Oz multi-channel control |
| `preproc_sensitivity` | `preprocessing_sensitivity.png` | REM−W across bandpass × sfreq grid |
| `classification` | `classification_summary.png` | LOSO AUC for K0 / band power / combined |
| `cohort_replication` | `cohort_replication.png` | Cassette vs Telemetry replication |

---

## 7. Reproducibility

- `RNG_SEED = 0`. Per-night sub-seeds are spawned via
  `numpy.random.SeedSequence(0).spawn(n_pairs)`, so output is bit-for-bit
  identical regardless of `--n-jobs`.
- Filtering: 0.5–40 Hz bandpass on EEG, resample to 50 Hz, 30-second epochs,
  stage assigned at the epoch midpoint.
- Embedding: m=10, τ=2 (main analysis), Ripser up to H1, per-dimension
  z-score with ε = 1e-8, within-epoch downsample by factor 2.
- Mixed-LM: `y ~ C(stage)` with subject random intercepts, `reml=False`,
  Powell optimizer (Track 1 / Track 2) or L-BFGS (Track 3 / robustness),
  `maxiter=2000` where applicable.
- Planned contrasts use Holm correction across the contrast family per metric.

---

## 8. License & citation

MIT. Copyright (c) 2026 mcmahonemmet — see `license.txt`.

If you use this code, please cite the manuscript and the original Sleep-EDF
dataset reference (Kemp et al., *IEEE Trans. Biomed. Eng.*, 2000).

---

## 9. Repository hygiene

`.gitignore` excludes `data/`, `outputs/`, virtualenvs, and Python caches so
the repo stays small and platform-neutral. If you fork this for your own
analysis, keep that pattern: re-running `run.py` regenerates every core CSV,
NPZ, sheet, and figure, and the four standalone scripts in §4c regenerate the
demographic and power-analysis outputs.
