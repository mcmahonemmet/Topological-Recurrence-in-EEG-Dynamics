from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

STAGES = ["W", "N1", "N2", "N3", "REM"]
PLANNED = [("REM","W"), ("REM","N3"), ("N1","N3")]

def holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    sorted_adj = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    sorted_adj = np.maximum.accumulate(sorted_adj)
    out = np.empty(m)
    out[order] = sorted_adj
    return out

def within_subject_z(df: pd.DataFrame, col: str) -> pd.Series:
    mu = df.groupby("subject")[col].transform("mean")
    sd = df.groupby("subject")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).fillna(0.0)

def fit_mixedlm_stage(ss: pd.DataFrame, y: str):
    ss = ss.copy()
    ss["stage"] = pd.Categorical(ss["stage"], categories=STAGES, ordered=True)
    model = smf.mixedlm(f"{y} ~ C(stage)", ss, groups=ss["subject"])
    res = model.fit(reml=False, method="powell", maxiter=2000, disp=False)

    null_model = smf.mixedlm(f"{y} ~ 1", ss, groups=ss["subject"])
    null_res = null_model.fit(reml=False, method="powell", maxiter=2000, disp=False)

    lr = 2*(res.llf - null_res.llf)
    df_diff = res.df_modelwc - null_res.df_modelwc
    p_lr = stats.chi2.sf(lr, df_diff)
    return res, float(lr), int(df_diff), float(p_lr)

def planned_contrasts(res, yname: str):
    params = res.params
    cov = res.cov_params()
    idx = params.index.tolist()

    def Lvec(mapping):
        L = np.zeros(len(idx))
        for k, v in mapping.items():
            L[idx.index(k)] = v
        return L

    def contrast(L, name):
        est = float(np.dot(L, params))
        se = float(np.sqrt(np.dot(L, np.dot(cov, L))))
        z = est / se if se > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if se > 0 else np.nan
        lo = est - 1.96 * se
        hi = est + 1.96 * se
        return dict(contrast=name, estimate=est, SE=se, z=z, p=p, CI95_low=lo, CI95_high=hi)

    out = []
    out.append(contrast(Lvec({"C(stage)[T.REM]": 1}), "REM - W"))
    out.append(contrast(Lvec({"C(stage)[T.REM]": 1, "C(stage)[T.N3]": -1}), "REM - N3"))
    out.append(contrast(Lvec({"C(stage)[T.N1]": 1, "C(stage)[T.N3]": -1}), "N1 - N3"))

    dfc = pd.DataFrame(out)
    dfc["metric"] = yname
    dfc["p_holm"] = holm(dfc["p"].values)
    return dfc

def main():
    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "outputs" / "tda_robustness_grid_epochs.csv"
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(in_path)
    df = df[df["stage"].isin(STAGES)].copy()

    # For each (channel, m, tau), compute K0 variants (within-subject z) and run mixed model
    omnibus_rows = []
    contrast_rows = []

    for (channel, m, tau), d in df.groupby(["channel","m","tau"]):
        # define K0 variants
        d = d.copy()
        d["K0_tot"] = within_subject_z(d, "H1_totpers")
        d["K0_max"] = within_subject_z(d, "H1_maxpers")
        d["K0_cnt"] = within_subject_z(d, "H1_count")

        # per-subject stage means
        for y in ["K0_tot", "K0_max", "K0_cnt"]:
            ss = d.groupby(["subject","stage"], as_index=False)[y].mean()

            res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, y)
            omnibus_rows.append({
                "channel": channel, "m": int(m), "tau": int(tau),
                "metric": y, "LR": lr, "df": df_diff, "p": p_lr
            })

            pc = planned_contrasts(res, y).copy()
            pc["channel"] = channel
            pc["m"] = int(m)
            pc["tau"] = int(tau)
            contrast_rows.append(pc)

    omni = pd.DataFrame(omnibus_rows)
    contr = pd.concat(contrast_rows, ignore_index=True)

    omni.to_csv(out_dir / "tda_robustness_mixedlm_omnibus.csv", index=False)
    contr.to_csv(out_dir / "tda_robustness_mixedlm_planned_contrasts.csv", index=False)

    print("Saved:")
    print(" ", out_dir / "tda_robustness_mixedlm_omnibus.csv")
    print(" ", out_dir / "tda_robustness_mixedlm_planned_contrasts.csv")

if __name__ == "__main__":
    main()
