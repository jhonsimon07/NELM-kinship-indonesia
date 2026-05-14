#!/usr/bin/env python3
"""
P1_03 — Multinomial logistic regression: predict NELM cluster membership
by ethnicity, controlling for HH-head SES and geography.

ROUND 1 REFINEMENT (9 May 2026):
  - Outcome: kmeans_cluster (k=5, primary) AND ward_cluster (robustness)
  - Added log_exp_pcm (per-capita expenditure) as additional SES control
  - Bivariate vs full vs reduced (no ethnic) comparison via LR test

Output: outputs/tables/
  - P1_multinomial_logit_results.csv      coefficients + p-values (k-means)
  - P1_multinomial_logit_ward.csv         same but for Ward outcome
  - P1_marginal_effects.csv                marginal predicted probabilities
  - P1_logit_summary.txt                   full statsmodels summary
  - P1_lr_tests_summary.csv                LR tests across specifications
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2
from statsmodels.formula.api import mnlogit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HH14 = PROJECT_ROOT / "data_raw" / "ifls5" / "hh14"
NELM = PROJECT_ROOT / "data_processed" / "ifls5_nelm"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"


def fit_lr(df_model, formula_full, formula_reduced):
    m_full = mnlogit(formula_full, data=df_model)
    r_full = m_full.fit(method="newton", disp=False, maxiter=200)
    m_red = mnlogit(formula_reduced, data=df_model)
    r_red = m_red.fit(method="newton", disp=False, maxiter=200)
    lr = 2 * (r_full.llf - r_red.llf)
    df_diff = r_full.df_model - r_red.df_model
    p = 1 - chi2.cdf(lr, df_diff)
    return r_full, r_red, lr, df_diff, p


def main():
    print("=" * 78)
    print("P1_03 — Multinomial logit (refined: with expenditure + Ward)")
    print("=" * 78)

    # ===========================================================
    # 1. Load + add HH-head SES
    # ===========================================================
    df = pd.read_csv(NELM / "household_nelm_clustered.csv")
    print(f"\nClustered HH: {len(df):,}")

    print("\n[1] Identifying HH heads")
    ar, _ = pyreadstat.read_dta(str(HH14 / "bk_ar1.dta"),
                                  usecols=["hhid14", "ar02b", "ar07", "ar09",
                                            "ar16", "ar17"])
    heads = ar[ar["ar02b"] == 1].copy().drop_duplicates("hhid14")
    heads["head_female"] = (heads["ar07"] == 3).astype(int)
    heads["head_age"] = pd.to_numeric(heads["ar09"], errors="coerce")
    heads.loc[heads["head_age"] > 110, "head_age"] = np.nan
    heads["head_edu_lvl"] = pd.to_numeric(heads["ar16"], errors="coerce")
    df = df.merge(heads[["hhid14", "head_age", "head_female", "head_edu_lvl"]],
                    on="hhid14", how="left")

    # ===========================================================
    # 2. Build modeling frame
    # ===========================================================
    df["ethnic"] = pd.Categorical(df["ethnic_modal"].map(
        {1: "Java", 4: "Batak", 9: "Minang"}),
        categories=["Java", "Batak", "Minang"], ordered=False)

    top_prov = df["prov_bps"].value_counts().head(6).index.tolist()
    df["prov_cat"] = df["prov_bps"].apply(
        lambda v: str(int(v)) if v in top_prov else "Other")
    df["prov_cat"] = pd.Categorical(df["prov_cat"])

    keep_cols = ["kmeans_cluster", "ward_cluster", "ethnic",
                  "head_age", "head_female", "head_edu_lvl",
                  "urban", "n_members", "log_exp_pcm", "prov_cat"]
    df_model = df.dropna(subset=keep_cols).copy()
    print(f"  modeling sample (complete cases): {len(df_model):,}")

    # ===========================================================
    # 3. PRIMARY: k-means cluster outcome
    # ===========================================================
    print("\n[3] PRIMARY: k-means cluster outcome")
    formula_biv = "kmeans_cluster ~ C(ethnic)"
    formula_full = ("kmeans_cluster ~ C(ethnic) + head_age + head_female + "
                     "head_edu_lvl + urban + n_members + log_exp_pcm + C(prov_cat)")
    formula_red = ("kmeans_cluster ~ head_age + head_female + head_edu_lvl + "
                    "urban + n_members + log_exp_pcm + C(prov_cat)")

    r_full, r_red, lr, df_diff, p_lr = fit_lr(df_model, formula_full, formula_red)
    print(f"  N: {r_full.nobs}")
    print(f"  Pseudo R² (full): {r_full.prsquared:.4f}")
    print(f"  Pseudo R² (no ethnic): {r_red.prsquared:.4f}")
    print(f"  LR test ethnic: stat={lr:.2f} (df={df_diff})  p={p_lr:.6f}")

    # Save full summary
    (OUT_TAB / "P1_logit_summary.txt").write_text(str(r_full.summary()))

    # Coefficient table
    coef_rows = []
    for outcome in r_full.params.columns:
        for var in r_full.params.index:
            coef_rows.append({
                "outcome_cluster": outcome,
                "variable": var,
                "coef": r_full.params.loc[var, outcome],
                "se": r_full.bse.loc[var, outcome],
                "p_value": r_full.pvalues.loc[var, outcome],
                "odds_ratio": np.exp(r_full.params.loc[var, outcome]),
                "sig": "***" if r_full.pvalues.loc[var, outcome] < 0.001
                        else "**" if r_full.pvalues.loc[var, outcome] < 0.01
                        else "*" if r_full.pvalues.loc[var, outcome] < 0.05
                        else "",
            })
    coef_df = pd.DataFrame(coef_rows)
    out_path = OUT_TAB / "P1_multinomial_logit_results.csv"
    coef_df.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    print("\n  ETHNIC EFFECTS (k-means clusters):")
    eth_only = coef_df[coef_df["variable"].str.contains("ethnic")]
    print(eth_only[["outcome_cluster", "variable", "coef", "p_value",
                       "odds_ratio", "sig"]].to_string(index=False))

    # ===========================================================
    # 4. ROBUSTNESS: Ward cluster outcome
    # ===========================================================
    print("\n[4] ROBUSTNESS: Ward cluster outcome")
    formula_full_w = formula_full.replace("kmeans_cluster", "ward_cluster")
    formula_red_w = formula_red.replace("kmeans_cluster", "ward_cluster")
    r_full_w, r_red_w, lr_w, dfd_w, p_w = fit_lr(
        df_model, formula_full_w, formula_red_w)
    print(f"  Pseudo R² (full): {r_full_w.prsquared:.4f}")
    print(f"  LR test ethnic: stat={lr_w:.2f} (df={dfd_w})  p={p_w:.6f}")

    coef_w_rows = []
    for outcome in r_full_w.params.columns:
        for var in r_full_w.params.index:
            coef_w_rows.append({
                "outcome_cluster": outcome,
                "variable": var,
                "coef": r_full_w.params.loc[var, outcome],
                "se": r_full_w.bse.loc[var, outcome],
                "p_value": r_full_w.pvalues.loc[var, outcome],
                "odds_ratio": np.exp(r_full_w.params.loc[var, outcome]),
                "sig": "***" if r_full_w.pvalues.loc[var, outcome] < 0.001
                        else "**" if r_full_w.pvalues.loc[var, outcome] < 0.01
                        else "*" if r_full_w.pvalues.loc[var, outcome] < 0.05
                        else "",
            })
    coef_w_df = pd.DataFrame(coef_w_rows)
    coef_w_df.to_csv(OUT_TAB / "P1_multinomial_logit_ward.csv", index=False)
    print(f"  → P1_multinomial_logit_ward.csv")

    print("\n  ETHNIC EFFECTS (Ward clusters):")
    eth_only_w = coef_w_df[coef_w_df["variable"].str.contains("ethnic")]
    print(eth_only_w[["outcome_cluster", "variable", "coef", "p_value",
                         "sig"]].to_string(index=False))

    # ===========================================================
    # 5. LR test summary across specifications
    # ===========================================================
    print("\n[5] LR test summary across specifications")
    lr_summary = pd.DataFrame([
        {"spec": "kmeans_cluster_full", "lr_ethnic": lr,
          "df": df_diff, "p_value": p_lr,
          "pseudoR2_full": r_full.prsquared, "n": int(r_full.nobs)},
        {"spec": "ward_cluster_full", "lr_ethnic": lr_w,
          "df": dfd_w, "p_value": p_w,
          "pseudoR2_full": r_full_w.prsquared, "n": int(r_full_w.nobs)},
    ])
    print(lr_summary.to_string(index=False))
    lr_summary.to_csv(OUT_TAB / "P1_lr_tests_summary.csv", index=False)
    print(f"  → P1_lr_tests_summary.csv")

    # ===========================================================
    # 6. Marginal predicted probabilities at baseline (k-means)
    # ===========================================================
    print("\n[6] Marginal predicted probabilities by ethnicity (k-means)")
    baseline = df_model[["head_age", "head_female", "head_edu_lvl",
                            "urban", "n_members", "log_exp_pcm"]].mean()
    common_prov = df_model["prov_cat"].mode().iloc[0]

    rows = []
    for eth in ["Java", "Batak", "Minang"]:
        scenario = df_model.iloc[:1].copy()
        scenario["ethnic"] = pd.Categorical([eth],
            categories=["Java", "Batak", "Minang"])
        for col, v in baseline.items():
            scenario[col] = v
        scenario["prov_cat"] = pd.Categorical([common_prov],
            categories=df_model["prov_cat"].cat.categories)
        probs = r_full.predict(scenario).iloc[0]
        for cluster_id, p in probs.items():
            rows.append({"ethnic": eth, "cluster": cluster_id,
                          "predicted_prob": float(p)})

    marg = pd.DataFrame(rows).pivot(index="ethnic", columns="cluster",
                                          values="predicted_prob")
    marg.to_csv(OUT_TAB / "P1_marginal_effects.csv")
    print(f"\n  Predicted P(cluster | ethnic, baseline):")
    print(marg.round(3).to_string())
    print(f"  → P1_marginal_effects.csv")

    print("\n" + "=" * 78)
    print("P1_03 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
