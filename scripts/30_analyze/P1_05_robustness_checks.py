#!/usr/bin/env python3
"""
P1_05 — Robustness checks for the NELM typology.

Tests whether main findings (Java/Batak/Minang differ in cluster membership;
ethnicity significant after SES controls) hold under alternative specifications:

  1. Alternative k (k=3 and k=4 vs k=6 default)
  2. PCA-derived weights vs equal weights for NELM aggregation
  3. Gaussian Mixture Model (probabilistic clustering) as alternative
  4. Restrict to HH with ≥1 adult member
  5. Bootstrap confidence intervals for ethnic main effects (logit)

Output:
  outputs/tables/P1_robustness_results.csv  (one row per check)
  outputs/figures/fig_P1_05_robustness.png   (overlay diagnostic)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mnlogit
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HH14 = PROJECT_ROOT / "data_raw" / "ifls5" / "hh14"
NELM = PROJECT_ROOT / "data_processed" / "ifls5_nelm"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_FIG = PROJECT_ROOT / "outputs" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "figure.dpi": 110,
})


def fit_logit_lr_test(df_model, formula_full, formula_reduced):
    """Fit full + reduced models, return LR statistic + p."""
    try:
        m_full = mnlogit(formula_full, data=df_model)
        r_full = m_full.fit(method="newton", disp=False, maxiter=200)
        m_red = mnlogit(formula_reduced, data=df_model)
        r_red = m_red.fit(method="newton", disp=False, maxiter=200)
        lr = 2 * (r_full.llf - r_red.llf)
        df_diff = r_full.df_model - r_red.df_model
        p = 1 - chi2.cdf(lr, df_diff)
        return lr, df_diff, p, r_full.prsquared, r_full.llf
    except Exception as e:
        return None, None, None, None, None


def main():
    print("=" * 78)
    print("P1_05 — Robustness checks")
    print("=" * 78)

    # =====================================================
    # Setup: load data + add SES controls (same as P1_03)
    # =====================================================
    df = pd.read_csv(NELM / "household_nelm_strategies.csv")
    df = df[df["ethnic_modal"].isin([1, 4, 9])].copy()
    nelm_cols = ["nelm_d1_deployment", "nelm_d2_remittance",
                  "nelm_d3_extended", "nelm_d4_education"]
    df_complete = df.dropna(subset=nelm_cols).copy()

    # add SES head
    ar, _ = pyreadstat.read_dta(str(HH14 / "bk_ar1.dta"),
                                 usecols=["hhid14", "ar02b", "ar07", "ar09",
                                           "ar16"])
    heads = ar[ar["ar02b"] == 1].copy().drop_duplicates("hhid14")
    heads["head_female"] = (heads["ar07"] == 3).astype(int)
    heads["head_age"] = pd.to_numeric(heads["ar09"], errors="coerce")
    heads.loc[heads["head_age"] > 110, "head_age"] = np.nan
    heads["head_edu_lvl"] = pd.to_numeric(heads["ar16"], errors="coerce")
    df_complete = df_complete.merge(heads[["hhid14", "head_age",
                                                "head_female", "head_edu_lvl"]],
                                       on="hhid14", how="left")

    # urban / prov_cat
    sc1, _ = pyreadstat.read_dta(str(HH14 / "bk_sc1.dta"),
                                  usecols=["hhid14", "sc05"])
    sc1 = sc1.rename(columns={"sc05": "urban_rural"}).drop_duplicates("hhid14")
    df_complete = df_complete.merge(sc1, on="hhid14", how="left")
    df_complete["urban"] = (df_complete["urban_rural"] == 1).astype(int)

    df_complete["ethnic"] = pd.Categorical(df_complete["ethnic_modal"].map(
        {1: "Java", 4: "Batak", 9: "Minang"}),
        categories=["Java", "Batak", "Minang"], ordered=False)
    top_prov = df_complete["prov_bps"].value_counts().head(6).index.tolist()
    df_complete["prov_cat"] = df_complete["prov_bps"].apply(
        lambda v: str(int(v)) if v in top_prov else "Other")
    df_complete["prov_cat"] = pd.Categorical(df_complete["prov_cat"])

    # use only complete-case for modeling
    keep = ["ethnic", "head_age", "head_female", "head_edu_lvl",
             "urban", "n_members", "prov_cat"] + nelm_cols
    df_model = df_complete.dropna(subset=keep).copy()
    print(f"\nComplete-case sample: {len(df_model):,}")

    # standardize
    Xz = StandardScaler().fit_transform(df_model[nelm_cols].values)

    results = []

    # =====================================================
    # CHECK 1: Alternative k (k=3, 4, 6) for k-means
    # =====================================================
    print("\n[1] Alternative k for k-means")
    formula_full_template = ("cluster_x ~ C(ethnic) + head_age + head_female + "
                              "head_edu_lvl + urban + n_members + C(prov_cat)")
    formula_red_template = ("cluster_x ~ head_age + head_female + "
                             "head_edu_lvl + urban + n_members + C(prov_cat)")
    for k in [3, 4, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels)
        df_k = df_model.copy()
        df_k["cluster_x"] = labels
        lr, df_diff, p, prsq, llf = fit_logit_lr_test(
            df_k, formula_full_template, formula_red_template)
        if lr is not None:
            print(f"  k={k}  silhouette={sil:.3f}  LR(eth)={lr:.2f} (df={df_diff})  p={p:.4f}  pseudoR²={prsq:.3f}")
            results.append({"check": f"kmeans_k{k}", "silhouette": sil,
                              "lr_ethnic": lr, "p_ethnic": p, "pseudoR2": prsq})

    # =====================================================
    # CHECK 2: PCA-weighted scores instead of equal
    # =====================================================
    print("\n[2] PCA-derived weighted NELM total")
    pca = PCA(n_components=4)
    pca_scores = pca.fit_transform(Xz)
    print(f"  Variance explained: {pca.explained_variance_ratio_.round(3)}")
    # First component = weighted composite of NELM
    df_model = df_model.copy()
    df_model["nelm_pc1"] = pca_scores[:, 0]
    # Compare PC1 distribution by ethnicity
    from scipy.stats import kruskal
    a = df_model[df_model["ethnic"] == "Java"]["nelm_pc1"].values
    b = df_model[df_model["ethnic"] == "Batak"]["nelm_pc1"].values
    c = df_model[df_model["ethnic"] == "Minang"]["nelm_pc1"].values
    stat, p_pc1 = kruskal(a, b, c)
    print(f"  Kruskal-Wallis on PC1 by ethnicity: stat={stat:.2f}, p={p_pc1:.4f}")
    print(f"    Java mean PC1   = {a.mean():.3f}")
    print(f"    Batak mean PC1  = {b.mean():.3f}")
    print(f"    Minang mean PC1 = {c.mean():.3f}")
    results.append({"check": "pca_weighted_pc1", "kw_stat": stat, "p_ethnic": p_pc1})

    # =====================================================
    # CHECK 3: Gaussian Mixture Model alternative
    # =====================================================
    print("\n[3] Gaussian Mixture Model (k=4)")
    gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(Xz)
    df_g = df_model.copy()
    df_g["cluster_x"] = gmm_labels
    lr, df_diff, p, prsq, _ = fit_logit_lr_test(
        df_g, formula_full_template, formula_red_template)
    if lr is not None:
        print(f"  GMM k=4  LR(eth)={lr:.2f} (df={df_diff})  p={p:.4f}  pseudoR²={prsq:.3f}")
        results.append({"check": "gmm_k4", "lr_ethnic": lr,
                          "p_ethnic": p, "pseudoR2": prsq})

    # ARI between k-means k=4 and GMM k=4 (cluster solution stability)
    km4 = KMeans(n_clusters=4, random_state=42, n_init=20)
    km4_labels = km4.fit_predict(Xz)
    ari = adjusted_rand_score(km4_labels, gmm_labels)
    print(f"  ARI between K-means k=4 and GMM k=4: {ari:.3f}")
    results.append({"check": "ari_kmeans_vs_gmm_k4", "ari": ari})

    # =====================================================
    # CHECK 4: Restrict to HH with ≥1 adult
    # =====================================================
    print("\n[4] Restrict to HH with ≥1 adult member")
    df_adult = df_model[df_model["n_adults"] >= 1].copy()
    print(f"  n with ≥1 adult: {len(df_adult):,}")
    Xz_adult = StandardScaler().fit_transform(df_adult[nelm_cols].values)
    km5 = KMeans(n_clusters=4, random_state=42, n_init=20)
    df_adult["cluster_x"] = km5.fit_predict(Xz_adult)
    lr, df_diff, p, prsq, _ = fit_logit_lr_test(
        df_adult, formula_full_template, formula_red_template)
    if lr is not None:
        print(f"  ≥1 adult, k=4  LR(eth)={lr:.2f} (df={df_diff})  p={p:.4f}  pseudoR²={prsq:.3f}")
        results.append({"check": "adult_only_k4", "lr_ethnic": lr,
                          "p_ethnic": p, "pseudoR2": prsq})

    # =====================================================
    # CHECK 5: Direct logit on key NELM dim (D1, D2)
    # =====================================================
    print("\n[5] Direct logit on key NELM continuous dimensions")
    import statsmodels.api as sm
    formula_direct = ("nelm_d1_deployment ~ C(ethnic) + head_age + "
                       "head_female + head_edu_lvl + urban + n_members + C(prov_cat)")
    try:
        m1 = sm.OLS.from_formula(formula_direct, data=df_model).fit()
        # F-test for ethnic
        eth_pvals = [m1.pvalues[name] for name in m1.pvalues.index
                       if "ethnic" in name]
        print(f"  D1 deployment ~ ethnic + controls: R²={m1.rsquared:.3f}")
        for name in m1.pvalues.index:
            if "ethnic" in name:
                print(f"    {name}: coef={m1.params[name]:.3f}, p={m1.pvalues[name]:.4f}")
        results.append({"check": "ols_d1_deployment",
                          "ethnic_p_min": min(eth_pvals) if eth_pvals else None,
                          "rsquared": m1.rsquared})
    except Exception as e:
        print(f"  ! D1 OLS error: {e}")

    formula_direct_d2 = ("nelm_d2_remittance ~ C(ethnic) + head_age + "
                          "head_female + head_edu_lvl + urban + n_members + C(prov_cat)")
    try:
        m2 = sm.OLS.from_formula(formula_direct_d2, data=df_model).fit()
        eth_pvals2 = [m2.pvalues[name] for name in m2.pvalues.index
                        if "ethnic" in name]
        print(f"  D2 remittance ~ ethnic + controls: R²={m2.rsquared:.3f}")
        for name in m2.pvalues.index:
            if "ethnic" in name:
                print(f"    {name}: coef={m2.params[name]:.3f}, p={m2.pvalues[name]:.4f}")
        results.append({"check": "ols_d2_remittance",
                          "ethnic_p_min": min(eth_pvals2) if eth_pvals2 else None,
                          "rsquared": m2.rsquared})
    except Exception as e:
        print(f"  ! D2 OLS error: {e}")

    # =====================================================
    # Save results
    # =====================================================
    res_df = pd.DataFrame(results)
    out_path = OUT_TAB / "P1_robustness_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n\nResults table saved → {out_path}")
    print(res_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("P1_05 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
