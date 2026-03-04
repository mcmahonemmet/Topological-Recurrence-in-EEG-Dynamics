from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


STAGES = ["W", "N1", "N2", "N3", "REM"]
PLANNED = [("REM", "W"), ("REM", "N3"), ("N1", "N3")]


def holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    sorted_adj = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    sorted_adj = np.maximum.accumulate(sorted_adj)
    out = np.empty(m)
    out[order] = sorted_adj
    return out


def fit_mixedlm(df: pd.DataFrame, y: str) -> tuple[object, float, int, float]:
    # per-subject stage means (avoid pseudoreplication)
    ss = df.groupby(["subject", "stage"], as_index=False)[y].mean()
    ss["stage"] = pd.Categorical(ss["stage"], categories=STAGES, ordered=True)

    model = smf.mixedlm(f"{y} ~ C(stage)", ss, groups=ss["subject"])
    res = model.fit(reml=False, method="powell", maxiter=2000, disp=False)

    null_model = smf.mixedlm(f"{y} ~ 1", ss, groups=ss["subject"])
    null_res = null_model.fit(reml=False, method="powell", maxiter=2000, disp=False)

    lr = 2 * (res.llf - null_res.llf)
    df_diff = res.df_modelwc - null_res.df_modelwc
    p_lr = stats.chi2.sf(lr, df_diff)
    return res, float(lr), int(df_diff), float(p_lr)


def planned_contrasts(res, y: str) -> pd.DataFrame:
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
        return dict(metric=y, contrast=name, estimate=est, SE=se, z=z, p=p, CI95_low=lo, CI95_high=hi)

    # baseline is W in statsmodels coding
    out = []
    out.append(contrast(Lvec({"C(stage)[T.REM]": 1}), "REM - W"))
    out.append(contrast(Lvec({"C(stage)[T.REM]": 1, "C(stage)[T.N3]": -1}), "REM - N3"))
    out.append(contrast(Lvec({"C(stage)[T.N1]": 1, "C(stage)[T.N3]": -1}), "N1 - N3"))

    dfc = pd.DataFrame(out)
    dfc["p_holm"] = holm(dfc["p"].values)
    return dfc


def main():
    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "outputs" / "baseline_epoch_features_all.csv"
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(in_path)
    df = df[df["stage"].isin(STAGES)].copy()

    metrics = [
        "log_delta", "log_theta", "log_alpha", "log_sigma", "log_beta",
        "spec_entropy", "perm_entropy", "lz_complexity",
    ]

    omnibus_rows = []
    contrast_rows = []

    for y in metrics:
        res, lr, df_diff, p_lr = fit_mixedlm(df, y)
        omnibus_rows.append({"metric": y, "LR": lr, "df": df_diff, "p": p_lr})
        contrast_rows.append(planned_contrasts(res, y))

    omni = pd.DataFrame(omnibus_rows)
    contr = pd.concat(contrast_rows, ignore_index=True)

    omni.to_csv(out_dir / "baseline_mixedlm_omnibus.csv", index=False)
    contr.to_csv(out_dir / "baseline_mixedlm_planned_contrasts.csv", index=False)

    print("Saved:")
    print(" ", out_dir / "baseline_mixedlm_omnibus.csv")
    print(" ", out_dir / "baseline_mixedlm_planned_contrasts.csv")


if __name__ == "__main__":
    main()
