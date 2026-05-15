#!/usr/bin/env python3
"""
refit_wake_robustness_stats.py
==============================

Re-fit the wake_subclass_robustness inner mixed-LMs from the existing grid CSV
without re-running the per-night TDA (which takes ~1.5 hours).

Use this if `step_wake_subclass_robustness` produced its grid CSV successfully
but the inner mixed-LM fits failed (e.g. all hit Singular-matrix errors).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from pipeline import (OUT_DIR, _safe_read_csv, _save, _inner_bar,
                       within_subject_z, fit_mixedlm_stage, planned_contrasts,
                       STAGES_WAKE, PLANNED_WAKE, Narrator, LOG_DIR)

n = Narrator(LOG_DIR / "refit_wake_robustness.log")
grid_csv = OUT_DIR / "wake_subclass_robustness_grid.csv"
out_omni = OUT_DIR / "wake_subclass_robustness_mixedlm_omnibus.csv"
out_pc = OUT_DIR / "wake_subclass_robustness_planned_contrasts.csv"

n.banner("Re-fit wake_subclass_robustness inner mixed-LM (powell-first)")
df = _safe_read_csv(grid_csv)
if df.empty:
    raise SystemExit(f"Grid CSV is empty or missing: {grid_csv}")
df = df[df["stage"].isin(STAGES_WAKE)].copy()
n.log(f"  · {len(df):,} grid rows, {df['subject'].nunique()} subjects")

omni_rows, contrast_rows = [], []
grid_groups = list(df.groupby(["channel", "m", "tau"]))
pbar = _inner_bar(len(grid_groups) * 3, "wake-subclass mixed-LM (refit)")
fail = 0
last_err = ""
for (channel, m, tau), d in grid_groups:
    d = d.copy()
    d["K0_tot"] = within_subject_z(d, "H1_totpers")
    d["K0_max"] = within_subject_z(d, "H1_maxpers")
    d["K0_cnt"] = within_subject_z(d, "H1_count")
    for y in ["K0_tot", "K0_max", "K0_cnt"]:
        ss = d.groupby(["subject", "stage"], as_index=False)[y].mean()
        try:
            try:
                res, lr, df_d, p_lr = fit_mixedlm_stage(ss, y, STAGES_WAKE, method="powell")
            except Exception:
                res, lr, df_d, p_lr = fit_mixedlm_stage(ss, y, STAGES_WAKE, method="lbfgs")
            omni_rows.append({"channel": channel, "m": int(m), "tau": int(tau),
                              "metric": y, "LR": lr, "df": df_d, "p": p_lr})
            pc = planned_contrasts(res, y, PLANNED_WAKE, STAGES_WAKE)
            pc["channel"] = channel; pc["m"] = int(m); pc["tau"] = int(tau)
            contrast_rows.append(pc)
        except Exception as ex:
            fail += 1
            last_err = f"{type(ex).__name__}: {ex}"
        if pbar is not None:
            pbar.update(1)
if pbar is not None:
    pbar.close()
if fail:
    n.log(f"  ! {fail} fits still failed (e.g. '{last_err}')")
n.log(f"  ✓ {len(omni_rows)} omnibus rows, "
      f"{sum(len(c) for c in contrast_rows)} contrast rows")
_save(pd.DataFrame(omni_rows), out_omni, n)
_save(pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(),
      out_pc, n)
