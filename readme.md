# Topological Recurrence in EEG Dynamics
### Distinguishing REM Sleep from Wakefulness via Persistent Homology

This repository contains the analysis code used in the study:

**McMahon, E. – Topological Recurrence in EEG Dynamics: Distinguishing REM Sleep from Wakefulness via Persistent Homology**

The pipeline reconstructs EEG state-space dynamics using delay embedding and quantifies recurrence using persistent homology. The resulting recurrence metric (**K0**) differentiates REM sleep from wakefulness in single-channel EEG.

The code reproduces the full analysis described in the manuscript.

---

# Repository Structure
sleep-edf-topological-recurrence/
│
├ scripts/
│
├ config/
│
├ data/
│
├ outputs/
│
├ README.md
├ requirements.txt
├ install_packages.py
├ run_full_pipeline.py


Outputs are generated automatically during execution and stored in:

```
outputs/
```

---

# Dataset

The analysis uses the **Sleep-EDF Expanded Dataset**.

Download from:

https://physionet.org/content/sleep-edfx/

Dataset reference:
Kemp et al., IEEE Transactions on Biomedical Engineering (2000).

The manuscript analyses **197 full-night recordings** with expert-scored sleep stages.

---

# Downloading the Dataset

1. Create a directory for the dataset:

```
Documents/sleep_edf_expanded_analysis/data/
```

2. Download **Sleep-EDF Expanded**

3. Extract the dataset so the structure looks like:

```
sleep_edf_expanded_analysis/
│
├ data/
│   └ sleep-edf-database-expanded-1.0.0/
│        ├ sleep-cassette/
│        └ sleep-telemetry/
```

4. Create a config file

```
config/config.env
```

Example:

```
SLEEP_EDF_ROOT=../data/sleep-edf-database-expanded-1.0.0
```

This path is used by all scripts.

---

# Installing Required Packages

Python version:

```
Python ≥ 3.10
```

Required libraries:

```
numpy
pandas
scipy
mne
ripser
persim
matplotlib
seaborn
statsmodels
tqdm
python-dotenv
```

Install packages automatically:

```
python install_packages.py
```

or manually:

```
pip install -r requirements.txt
```

---

# Running the Full Pipeline

To reproduce the full analysis described in the manuscript:

```
python scripts/run_full_pipeline.py
```

This sequentially runs:

1. Dataset scanning
2. TDA feature extraction
3. Robustness grid computation
4. Robustness statistical analysis
5. Baseline metric extraction
6. Baseline statistical analysis
7. Figure generation
8. Summary tables

Outputs will be written to:

```
outputs/
```

---

# Running Scripts Individually

Each stage of the analysis can also be run independently.

---

## 1. Scan dataset

```
python scripts/scan_dataset.py
```

Purpose:

Validates PSG–hypnogram file pairing in the Sleep-EDF dataset.

Ensures correct EEG and sleep stage annotation alignment.

---

## 2. Topological feature extraction

```
python scripts/run_tda_all_nights.py
```

Corresponds to:

**Methods — State-space reconstruction and persistent homology**

The script:

• loads EEG  
• filters between **0.5–40 Hz**  
• downsamples to **50 Hz**  
• segments into **30 s epochs**  
• performs **delay embedding (m=10, τ=2)**  
• computes persistent homology (Ripser)

Primary output:

```
outputs/tda_epoch_features_all.csv
```

Contains:

- H1 persistence
- persistence counts
- persistence maxima

These values correspond to **raw persistence values described in Table 1 of the manuscript**.

---

## 3. Robustness grid computation

```
python scripts/compute_tda_robustness_grid.py
```

Corresponds to:

**Results — Robustness across embedding parameters**

The pipeline recomputes recurrence across:

```
m ∈ {6, 8, 10, 12}
τ ∈ {1, 2, 4}
```

Output:

```
outputs/tda_robustness_grid_epochs.csv
```

This table stores epoch-level persistence values for all parameter combinations.

---

## 4. Robustness statistical analysis

```
python scripts/summarise_tda_robustness_grid.py
```

Corresponds to:

**Mixed-effects model analysis of stage contrasts**

The script:

• computes **K0 = z-score(H1 persistence) within subject**  
• aggregates stage means  
• fits mixed-effects models  
• computes planned contrasts

Outputs:

```
outputs/tda_robustness_mixedlm_planned_contrasts.csv
outputs/tda_robustness_mixedlm_omnibus.csv
```

These correspond to the **REM–Wake contrasts shown in Figure 1 of the manuscript.**

---

## 5. Baseline EEG metrics

```
python scripts/compute_baselines_all_nights.py
```

Corresponds to:

**Comparison with Conventional EEG Metrics**

Extracted metrics:

- spectral entropy
- permutation entropy
- Lempel–Ziv complexity
- delta power
- theta power
- alpha power
- beta power

Output:

```
outputs/baseline_epoch_features_all.csv
```

---

## 6. Baseline statistical comparisons

```
python scripts/run_baseline_benchmarks_stats.py
```

Computes identical mixed-effects contrasts for baseline EEG metrics.

Outputs:

```
outputs/baseline_mixedlm_planned_contrasts.csv
outputs/baseline_mixedlm_omnibus.csv
```

These results correspond to **Table 4 in the manuscript**, which compares recurrence to conventional EEG measures.

---

## 7. Robustness heatmap (Figure 1)

```
python scripts/make_planned_contrast_heatmap.py
```

Produces:

```
outputs/K0_robustness_heatmap.png
```

This figure visualises REM–Wake effect sizes across embedding parameters.

---

## 8. Example reconstruction (Figure 2)

```
python scripts/make_figure2_example.py
```

Produces:

- embedded trajectory plots
- persistence diagrams

These correspond to the **illustrative examples in Figure 2**.

---

## 9. Results summary

```
python scripts/review_outputs.py
```

Generates final summary tables for manuscript reporting.

---

# Outputs

All generated data are written to:

```
outputs/
```

Typical outputs include:

```
tda_epoch_features_all.csv
tda_stage_summary_all.csv
tda_robustness_grid_epochs.csv
tda_robustness_mixedlm_planned_contrasts.csv
baseline_epoch_features_all.csv
baseline_mixedlm_planned_contrasts.csv
```

These files contain the statistical results reported in the manuscript.

---

# Reproducibility

All scripts are deterministic.

Random sampling uses a fixed seed:

```
seed = 0
```

The full pipeline reproduces:

- recurrence metric **K0**
- stage contrasts
- robustness heatmaps
- baseline comparisons

---

# Code Availability

The analysis code supporting this study is publicly archived via Zenodo with a DOI-minted release.

The repository provides full reproducibility of all computational analyses described in the manuscript.

---

# License

MIT License
```
IT License

Copyright (c) 2026 mcmahonemmet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---
