#!/usr/bin/env python3
"""
P1_07 — Bootstrap 95% CI for ethnic coefficients in multinomial logit.

Asymptotic Wald CI in multinomial logit can be unreliable for small subgroups
(Batak n≈329, Minang n≈421). 200-replicate cluster bootstrap (resample HHs
with replacement) yields finite-sample valid 95% CI.

Output: outputs/tables/
  - P1_bootstrap_ci.csv     CI percentile + comparison with asymptotic SE
  - fig_P1_07_bootstrap_forest.png   Forest plot
"""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from joblib import Parallel, delayed
from statsmodels.formula.api import mnlogit

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HH14 = PROJECT_ROOT / "data_raw" / "ifls5" / "hh14"
NELM = PROJECT_ROOT / "data_processed" / "ifls5_nelm"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_FIG = PROJECT_ROOT / "outputs" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "figure.dpi": 110,
})

FORMULA = ("kmeans_cluster ~ C(ethnic) + head_age + head_female + "
            "head_edu_lvl + urban + n_members + log_exp_pcm + C(prov_cat)")

N_BOOTSTRAP = 200
N_JOBS = 4


def fit_one(seed_idx, df_orig):
    """Resample HH with replacement and refit logit. Return ethnic coefs."""
    np.random.seed(seed_idx)
    boot_idx = np.random.choice(len(df_orig), size=len(df_orig), replace=True)
    df_boot = df_orig.iloc[boot_idx].copy().reset_index(drop=True)
    try:
        m = mnlogit(FORMULA, data=df_boot)
        r = m.fit(method="newton", disp=False, maxiter=200)
        # Extract ethnic dummies for each cluster outcome
        rows = []
        for outcome in r.params.columns:
            for var in r.params.index:
                if "ethnic" in var:
                    rows.append({
                        "rep": seed_idx,
                        "outcome_cluster": outcome,
                        "variable": var,
                        "coef": r.params.loc[var, outcome],
                    })
        return rows
    except Exception:
        return []


def main():
    print("=" * 78)
    print(f"P1_07 — Bootstrap CI ({N_BOOTSTRAP} replicates, {N_JOBS} parallel jobs)")
    print("=" * 78)

    # ===========================================================
    # Prepare modeling frame
    # ===========================================================
    df = pd.read_csv(NELM / "household_nelm_clustered.csv")
    ar, _ = pyreadstat.read_dta(str(HH14 / "bk_ar1.dta"),
                                  usecols=["hhid14", "ar02b", "ar07", "ar09",
                                            "ar16"])
    heads = ar[ar["ar02b"] == 1].copy().drop_duplicates("hhid14")
    heads["head_female"] = (heads["ar07"] == 3).astype(int)
    heads["head_age"] = pd.to_numeric(heads["ar09"], errors="coerce")
    heads.loc[heads["head_age"] > 110, "head_age"] = np.nan
    heads["head_edu_lvl"] = pd.to_numeric(heads["ar16"], errors="coerce")
    df = df.merge(heads[["hhid14", "head_age", "head_female", "head_edu_lvl"]],
                    on="hhid14", how="left")

    df["ethnic"] = pd.Categorical(df["ethnic_modal"].map(
        {1: "Java", 4: "Batak", 9: "Minang"}),
        categories=["Java", "Batak", "Minang"], ordered=False)
    top_prov = df["prov_bps"].value_counts().head(6).index.tolist()
    df["prov_cat"] = df["prov_bps"].apply(
        lambda v: str(int(v)) if v in top_prov else "Other")
    df["prov_cat"] = pd.Categorical(df["prov_cat"])

    keep = ["kmeans_cluster", "ethnic", "head_age", "head_female",
             "head_edu_lvl", "urban", "n_members", "log_exp_pcm", "prov_cat"]
    df_model = df.dropna(subset=keep).reset_index(drop=True)
    print(f"\nModeling sample: {len(df_model):,}")

    # ===========================================================
    # Original full-sample point estimates
    # ===========================================================
    print("\n[1] Original full-sample fit")
    m_full = mnlogit(FORMULA, data=df_model)
    r_full = m_full.fit(method="newton", disp=False, maxiter=200)
    point_rows = []
    for outcome in r_full.params.columns:
        for var in r_full.params.index:
            if "ethnic" in var:
                point_rows.append({
                    "outcome_cluster": outcome,
                    "variable": var,
                    "point": r_full.params.loc[var, outcome],
                    "asymp_se": r_full.bse.loc[var, outcome],
                    "asymp_p": r_full.pvalues.loc[var, outcome],
                    "asymp_ci_lo": (r_full.params.loc[var, outcome] -
                                      1.96 * r_full.bse.loc[var, outcome]),
                    "asymp_ci_hi": (r_full.params.loc[var, outcome] +
                                      1.96 * r_full.bse.loc[var, outcome]),
                })
    point_df = pd.DataFrame(point_rows)
    print(f"  point estimates extracted for {len(point_df)} ethnic-cluster cells")

    # ===========================================================
    # Run bootstrap in parallel
    # ===========================================================
    print(f"\n[2] Running {N_BOOTSTRAP} bootstrap replicates (this may take 5-15 min)")
    bootstrap_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(fit_one)(seed, df_model) for seed in range(N_BOOTSTRAP))

    # Flatten
    boot_rows = [r for batch in bootstrap_results for r in batch]
    boot_df = pd.DataFrame(boot_rows)
    print(f"\n  Successful bootstrap fits: {boot_df['rep'].nunique()}/{N_BOOTSTRAP}")

    # Compute percentile CI
    print("\n[3] Computing 95% percentile CI")
    ci_rows = []
    for (outcome, var), grp in boot_df.groupby(["outcome_cluster", "variable"]):
        coefs = grp["coef"].values
        if len(coefs) < 50:
            continue
        ci_lo, ci_hi = np.percentile(coefs, [2.5, 97.5])
        ci_rows.append({
            "outcome_cluster": outcome,
            "variable": var,
            "boot_n": len(coefs),
            "boot_mean": float(np.mean(coefs)),
            "boot_se": float(np.std(coefs)),
            "boot_ci_lo": ci_lo,
            "boot_ci_hi": ci_hi,
            "boot_p_two_sided": (
                2 * min(np.mean(coefs > 0), np.mean(coefs < 0))),
        })
    boot_summary = pd.DataFrame(ci_rows)

    # Merge with point estimates
    merged = point_df.merge(boot_summary, on=["outcome_cluster", "variable"],
                                how="left")
    out_path = OUT_TAB / "P1_bootstrap_ci.csv"
    merged.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    print("\n  Asymptotic vs Bootstrap (95% CI):")
    print(merged[["outcome_cluster", "variable", "point",
                     "asymp_ci_lo", "asymp_ci_hi",
                     "boot_ci_lo", "boot_ci_hi"]].round(3).to_string(index=False))

    # ===========================================================
    # Forest plot
    # ===========================================================
    print("\n[4] Forest plot")
    fig, ax = plt.subplots(figsize=(11, 7))
    ypos = np.arange(len(merged))
    # Asymptotic interval
    ax.errorbar(merged["point"], ypos - 0.18,
                  xerr=[merged["point"] - merged["asymp_ci_lo"],
                         merged["asymp_ci_hi"] - merged["point"]],
                  fmt="s", color="#1f77b4", capsize=4,
                  label="Asymptotic 95% CI", markersize=6)
    # Bootstrap interval
    ax.errorbar(merged["point"], ypos + 0.18,
                  xerr=[merged["point"] - merged["boot_ci_lo"],
                         merged["boot_ci_hi"] - merged["point"]],
                  fmt="o", color="#d62728", capsize=4,
                  label="Bootstrap 95% CI", markersize=6)
    ax.axvline(0, color="black", linestyle="--", lw=0.5)
    ax.set_yticks(ypos)
    labels = [f"C{int(o)}: {v.replace('C(ethnic)[T.', '').replace(']', '')}"
                for o, v in zip(merged["outcome_cluster"], merged["variable"])]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Ethnic effects: Asymptotic vs Bootstrap CI ({N_BOOTSTRAP} reps)\n"
                  "Bootstrap (red) provides finite-sample valid inference")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_07_bootstrap_forest.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_07_bootstrap_forest.png")

    print("\n" + "=" * 78)
    print(f"P1_07 done. {boot_df['rep'].nunique()}/{N_BOOTSTRAP} replicates successful.")
    print("=" * 78)


if __name__ == "__main__":
    main()
