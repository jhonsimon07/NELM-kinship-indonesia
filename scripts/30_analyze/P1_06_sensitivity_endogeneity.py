#!/usr/bin/env python3
"""
P1_06 — Sensitivity & endogeneity analyses for ethnic effects on NELM strategies.

ROUND 2 (Targeted): address the most likely reviewer concerns:

  1. SUBSAMPLE SENSITIVITY:
     1a. Urban-only HH
     1b. Rural-only HH
     1c. By expenditure tertile (low/mid/high SES)
     1d. Drop urban-anomaly provinces (DKI Jakarta) for Sumatra-only check

  2. SELECTION-BIAS CHECK:
     2a. Among ethnic-modal HH (current sample) vs mixed-ethnic HH
     2b. Among "homeland" residents (Sumbar for Minang, Sumut for Batak) vs diaspora

  3. ALTERNATIVE OPERATIONALIZATION:
     3a. Multi-ethnic HH excluded (homogeneity ≥0.95)
     3b. Continuous ethnic-share metric instead of categorical

For each subsample, refit the multinomial logit and test ethnic effect via LR.
Findings should be CONSISTENT across subsamples for paper to claim robustness.

Output:
  outputs/tables/P1_sensitivity_results.csv  (subsample × LR p × pseudoR²)
  outputs/figures/fig_P1_06_sensitivity_forest.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2
from statsmodels.formula.api import mnlogit

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

FORMULA_FULL = ("kmeans_cluster ~ C(ethnic) + head_age + head_female + "
                 "head_edu_lvl + urban + n_members + log_exp_pcm + C(prov_cat)")
FORMULA_RED = ("kmeans_cluster ~ head_age + head_female + head_edu_lvl + "
                "urban + n_members + log_exp_pcm + C(prov_cat)")


def fit_lr(df_model, formula_full, formula_red):
    try:
        m_full = mnlogit(formula_full, data=df_model)
        r_full = m_full.fit(method="newton", disp=False, maxiter=200)
        m_red = mnlogit(formula_red, data=df_model)
        r_red = m_red.fit(method="newton", disp=False, maxiter=200)
        lr = 2 * (r_full.llf - r_red.llf)
        df_diff = r_full.df_model - r_red.df_model
        p = 1 - chi2.cdf(lr, df_diff)
        return r_full, lr, df_diff, p
    except Exception:
        return None, None, None, None


def prepare_modeling_frame():
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
             "head_edu_lvl", "urban", "n_members", "log_exp_pcm",
             "prov_cat", "prov_bps", "ethnic_share_modal"]
    return df.dropna(subset=keep).copy()


def main():
    print("=" * 78)
    print("P1_06 — Sensitivity & endogeneity (Round 2)")
    print("=" * 78)

    df_model = prepare_modeling_frame()
    print(f"\nFull modeling sample: {len(df_model):,}")

    results = []

    def run_check(name, df_sub, descr=""):
        if len(df_sub) < 200:
            print(f"  [{name}] N={len(df_sub):,} too small, skip")
            return
        r_full, lr, df_diff, p = fit_lr(df_sub, FORMULA_FULL, FORMULA_RED)
        if r_full is None:
            print(f"  [{name}] convergence error")
            return
        n_minang = (df_sub["ethnic"] == "Minang").sum()
        n_batak = (df_sub["ethnic"] == "Batak").sum()
        print(f"  [{name}]  N={len(df_sub):,}  "
              f"(Minang={n_minang}, Batak={n_batak})  "
              f"LR={lr:.2f} p={p:.5f}  pseudoR²={r_full.prsquared:.3f}")
        results.append({
            "subsample": name,
            "description": descr,
            "n": len(df_sub),
            "n_minang": int(n_minang),
            "n_batak": int(n_batak),
            "lr_ethnic": lr,
            "df": df_diff,
            "p_ethnic": p,
            "pseudoR2": r_full.prsquared,
            "ethnic_significant_p05": p < 0.05,
        })

    # ===========================================================
    # CHECK 0: Baseline (full sample)
    # ===========================================================
    print("\n[0] Baseline (full sample)")
    run_check("baseline_full", df_model, "Full sample")

    # ===========================================================
    # CHECK 1a-1b: Urban / Rural split
    # ===========================================================
    print("\n[1a-1b] Urban / Rural split")
    run_check("urban_only", df_model[df_model["urban"] == 1],
                "HH in urban kabupaten")
    run_check("rural_only", df_model[df_model["urban"] == 0],
                "HH in rural kabupaten")

    # ===========================================================
    # CHECK 1c: By expenditure tertile
    # ===========================================================
    print("\n[1c] By expenditure tertile")
    df_model["exp_tertile"] = pd.qcut(df_model["log_exp_pcm"], q=3,
                                          labels=["low", "mid", "high"])
    for tertile in ["low", "mid", "high"]:
        run_check(f"exp_{tertile}",
                    df_model[df_model["exp_tertile"] == tertile],
                    f"Expenditure tertile = {tertile}")

    # ===========================================================
    # CHECK 1d: Drop DKI Jakarta + Sumatra-only check
    # ===========================================================
    print("\n[1d] Drop DKI Jakarta / Sumatra-only check")
    sumatra_codes = [12, 13, 14, 15, 16, 17, 18, 19, 21]
    run_check("drop_dki", df_model[df_model["prov_bps"] != 31],
                "Drop DKI Jakarta")
    run_check("sumatra_only",
                df_model[df_model["prov_bps"].isin(sumatra_codes)],
                "Sumatra provinces only")

    # ===========================================================
    # CHECK 2a: Ethnic homogeneity
    # ===========================================================
    print("\n[2a] Multi-ethnic homogeneity restriction")
    run_check("homogeneous_ethnic",
                df_model[df_model["ethnic_share_modal"] >= 0.95],
                "HH with ≥95% same-ethnicity members")

    # ===========================================================
    # CHECK 2b: Homeland vs diaspora for Minang/Batak
    # ===========================================================
    print("\n[2b] Homeland vs diaspora (for Sumatran ethnics)")
    df_minang_homeland = df_model[
        (df_model["ethnic"] == "Minang") & (df_model["prov_bps"] == 13)]
    df_batak_homeland = df_model[
        (df_model["ethnic"] == "Batak") & (df_model["prov_bps"] == 12)]
    df_java_baseline = df_model[df_model["ethnic"] == "Java"]
    df_homeland = pd.concat([df_minang_homeland, df_batak_homeland,
                                df_java_baseline.iloc[:1500]])
    run_check("homeland_only", df_homeland,
                "Sumatran ethnics in homeland (Sumbar/Sumut) + Java baseline")

    # ===========================================================
    # CHECK 3: Drop high-leverage observations
    # ===========================================================
    print("\n[3] Drop top/bottom 1% expenditure outliers")
    p1, p99 = df_model["log_exp_pcm"].quantile([0.01, 0.99])
    df_trim = df_model[(df_model["log_exp_pcm"] > p1) &
                          (df_model["log_exp_pcm"] < p99)]
    run_check("trim_1pct_exp", df_trim, "Drop top/bottom 1% expenditure")

    # ===========================================================
    # Save & visualize
    # ===========================================================
    res_df = pd.DataFrame(results)
    out_path = OUT_TAB / "P1_sensitivity_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n\nResults saved → {out_path}")
    print(res_df[["subsample", "n", "lr_ethnic", "p_ethnic",
                     "pseudoR2"]].to_string(index=False))

    # Forest plot of LR p-values
    fig, ax = plt.subplots(figsize=(9, 6))
    res_df["log10p"] = -np.log10(res_df["p_ethnic"].clip(lower=1e-10))
    colors = ["#2ca02c" if p < 0.001 else
                "#1f77b4" if p < 0.01 else
                "#ff7f0e" if p < 0.05 else
                "#d62728" for p in res_df["p_ethnic"]]
    ax.barh(res_df["subsample"], res_df["log10p"], color=colors, height=0.7)
    ax.axvline(-np.log10(0.05), color="black", linestyle="--",
                 label="p=0.05 threshold")
    ax.axvline(-np.log10(0.01), color="gray", linestyle=":",
                 label="p=0.01 threshold")
    ax.set_xlabel("-log10(p-value) for LR test of ethnicity")
    ax.set_title("Sensitivity Forest Plot: Ethnic Effect on NELM Cluster\n"
                  "Across 11 subsamples — Green=p<0.001, Blue=p<0.01, "
                  "Orange=p<0.05, Red=NS")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_06_sensitivity_forest.png",
                  dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_06_sensitivity_forest.png")

    print("\n" + "=" * 78)
    print("P1_06 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
