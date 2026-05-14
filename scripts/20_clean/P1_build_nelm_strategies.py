#!/usr/bin/env python3
"""
P1 — Build household-level NELM (New Economics of Labor Migration) strategy
indicators from IFLS-5.   [Round 1 refinement, 9 May 2026]

Following Stark & Bloom (1985), Stark (1991), and Massey et al. (1993), NELM
posits that migration is a household-level risk-diversification strategy.

CHANGES from V1:
  - Dropped D4 educational investment (caused 47% sample drop, was NS in OLS)
  - D3 continuous: ratio of non-nuclear members ((n_adults - 2 + n_minors) / n_members)
  - Added expenditure_pcm (per-capita monthly Rp), aggregated from b1_ks1 + b1_ks2 + b1_ks3
  - Ethnic-share metric: % of HH members same ethnicity as modal

THREE NELM DIMENSIONS:
  D1. MIGRATION DEPLOYMENT  — % HH adults out-migrated from kab. of birth
  D2. REMITTANCE DEPENDENCY — log(IDR received by all HH members)
  D3. CO-RESIDENCE EXTENT   — continuous ratio of non-nuclear adult members

PLUS context variable:
  - expenditure_pcm        — per-capita monthly expenditure (wealth proxy)
  - ethnic_share_modal     — homogeneity of HH ethnic composition

Output: data_processed/ifls5_nelm/  (v2 overwrite, v1 archived)
  - household_nelm_strategies.csv      (HH-level, 3 dimensions + ethnicity + expenditure)
  - individual_nelm_components.csv     (individual-level raw inputs)
  - minang_households.csv              (subset)
  - batak_households.csv               (subset)
  - java_households.csv                (subset)
  - nelm_build_log.txt                 (build log)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HH14 = PROJECT_ROOT / "data_raw" / "ifls5" / "hh14"
OUT = PROJECT_ROOT / "data_processed" / "ifls5_nelm"
OUT.mkdir(parents=True, exist_ok=True)

LOG: list[str] = []


def log(m: str):
    print(m)
    LOG.append(m)


def read_dta(name: str, cols: list[str] | None = None) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(str(HH14 / name), usecols=cols)
    log(f"  loaded {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def main():
    log("=" * 78)
    log("P1 NELM strategies builder — Round 1 refinement (3-dim + expenditure)")
    log("=" * 78)

    # =============================================================
    # 1. ROSTER + ETHNICITY (bk_ar1)
    # =============================================================
    log("\n[1/7] Roster bk_ar1.dta")
    ar = read_dta("bk_ar1.dta",
                   cols=["hhid14", "pidlink", "ar02b", "ar07", "ar09",
                          "ar15", "ar15d", "ar16", "ar17"])
    ar = ar.dropna(subset=["hhid14", "pidlink"]).drop_duplicates("pidlink")
    log(f"  unique pidlinks: {len(ar):,}")

    # =============================================================
    # 2. MIGRATION HISTORY (b3a_mg1) — for D1 deployment
    # =============================================================
    log("\n[2/7] Migration history b3a_mg1.dta")
    mg = read_dta("b3a_mg1.dta",
                   cols=["hhid14", "pidlink",
                          "mg01b", "mg01c", "mg01d", "mg01e"])

    sc1 = read_dta("bk_sc1.dta",
                    cols=["hhid14", "sc01_14_14", "sc02_14_14", "sc05"])
    sc1 = sc1.rename(columns={"sc01_14_14": "prov_bps",
                                "sc02_14_14": "kab_bps",
                                "sc05": "urban_rural"})

    mg = mg.merge(sc1, on="hhid14", how="left")
    mg["mg01c"] = pd.to_numeric(mg["mg01c"], errors="coerce")
    mg["out_migrated"] = (mg["mg01c"].notna()
                            & mg["kab_bps"].notna()
                            & (mg["mg01c"] != mg["kab_bps"])).astype(int)
    n_outmig = mg["out_migrated"].sum()
    log(f"  out-migrated individuals: {n_outmig:,} of {len(mg):,}")

    # =============================================================
    # 3. TRANSFERS (b3b_tf) — for D2 remittance
    # =============================================================
    log("\n[3/7] Transfers b3b_tf.dta")
    tf = read_dta("b3b_tf.dta",
                   cols=["hhid14", "pidlink", "tf04a", "tf06a"])
    for c in ["tf04a", "tf06a"]:
        tf[c] = pd.to_numeric(tf[c], errors="coerce")
        tf.loc[tf[c] >= 9_999_998, c] = np.nan
        tf[c] = tf[c].fillna(0).clip(lower=0)
    indiv_remit = tf.groupby("pidlink").agg(
        amount_given=("tf04a", "sum"),
        amount_received=("tf06a", "sum"),
        n_dyads=("tf04a", "size"),
    ).reset_index()
    indiv_remit["log_amount_received"] = np.log1p(indiv_remit["amount_received"])
    log(f"  individuals with transfer data: {len(indiv_remit):,}")

    # =============================================================
    # 4. EXPENDITURE (b1_ks1 + b1_ks2 + b1_ks3) — SES control
    # =============================================================
    log("\n[4/7] Expenditure aggregation (b1_ks1/2/3)")

    # b1_ks1: food items weekly (ks02 purchase + ks03 self-produced)
    ks1, _ = pyreadstat.read_dta(str(HH14 / "b1_ks1.dta"),
                                  usecols=["hhid14", "ks02", "ks03"])
    for c in ["ks02", "ks03"]:
        ks1[c] = pd.to_numeric(ks1[c], errors="coerce")
        ks1.loc[ks1[c] >= 9_999_998, c] = np.nan
        ks1[c] = ks1[c].fillna(0).clip(lower=0)
    ks1_hh = ks1.groupby("hhid14").agg(
        food_weekly=("ks02", "sum"),
        food_self=("ks03", "sum")
    ).reset_index()
    ks1_hh["food_total_monthly"] = (ks1_hh["food_weekly"] +
                                       ks1_hh["food_self"]) * 4.33
    log(f"  ks1: {len(ks1):,} food items → {len(ks1_hh):,} HH")

    # b1_ks2: non-food monthly (ks06)
    ks2, _ = pyreadstat.read_dta(str(HH14 / "b1_ks2.dta"),
                                  usecols=["hhid14", "ks06"])
    ks2["ks06"] = pd.to_numeric(ks2["ks06"], errors="coerce")
    ks2.loc[ks2["ks06"] >= 9_999_998, "ks06"] = np.nan
    ks2["ks06"] = ks2["ks06"].fillna(0).clip(lower=0)
    ks2_hh = ks2.groupby("hhid14").agg(nonfood_monthly=("ks06", "sum")).reset_index()
    log(f"  ks2: {len(ks2):,} non-food items → {len(ks2_hh):,} HH")

    # b1_ks3: non-food annual (ks08), convert to monthly
    ks3, _ = pyreadstat.read_dta(str(HH14 / "b1_ks3.dta"),
                                  usecols=["hhid14", "ks08"])
    ks3["ks08"] = pd.to_numeric(ks3["ks08"], errors="coerce")
    ks3.loc[ks3["ks08"] >= 9_999_998, "ks08"] = np.nan
    ks3["ks08"] = ks3["ks08"].fillna(0).clip(lower=0)
    ks3_hh = ks3.groupby("hhid14").agg(nonfood_annual=("ks08", "sum")).reset_index()
    ks3_hh["nonfood_annual_monthly"] = ks3_hh["nonfood_annual"] / 12
    log(f"  ks3: {len(ks3):,} durables → {len(ks3_hh):,} HH")

    # Combine total monthly expenditure
    exp = ks1_hh.merge(ks2_hh, on="hhid14", how="outer") \
                   .merge(ks3_hh, on="hhid14", how="outer")
    for c in ["food_total_monthly", "nonfood_monthly", "nonfood_annual_monthly"]:
        exp[c] = exp[c].fillna(0)
    exp["expenditure_total_monthly"] = (exp["food_total_monthly"] +
                                              exp["nonfood_monthly"] +
                                              exp["nonfood_annual_monthly"])
    exp["log_exp_monthly"] = np.log1p(exp["expenditure_total_monthly"])
    log(f"  HH with any expenditure data: {len(exp):,}")
    log(f"  Mean monthly HH expenditure: Rp {exp['expenditure_total_monthly'].mean():,.0f}")
    log(f"  Median: Rp {exp['expenditure_total_monthly'].median():,.0f}")

    # =============================================================
    # 5. MERGE INDIVIDUAL TABLE
    # =============================================================
    log("\n[5/7] Merging individual-level table")
    ind = ar[["hhid14", "pidlink", "ar02b", "ar07", "ar09",
                "ar15", "ar15d", "ar16", "ar17"]].copy()
    ind = ind.merge(mg[["pidlink", "out_migrated", "mg01c",
                          "kab_bps", "prov_bps", "urban_rural"]],
                     on="pidlink", how="left")
    ind = ind.merge(indiv_remit[["pidlink", "amount_given",
                                     "amount_received", "log_amount_received"]],
                     on="pidlink", how="left")
    log(f"  individual table: {len(ind):,} rows × {ind.shape[1]} cols")

    out_ind = OUT / "individual_nelm_components.csv"
    ind.to_csv(out_ind, index=False)
    log(f"  saved {out_ind}")

    # =============================================================
    # 6. AGGREGATE TO HOUSEHOLD LEVEL
    # =============================================================
    log("\n[6/7] Aggregating to household level (3-dim NELM + expenditure)")

    def hh_aggregate(g: pd.DataFrame) -> pd.Series:
        adults = g[g["ar09"] >= 15]
        n_members = len(g)
        n_adults = len(adults)
        n_outmig = (adults["out_migrated"] == 1).sum() if n_adults > 0 else 0

        # Ethnicity statistics
        eth = g["ar15d"].dropna()
        ethnic_modal = eth.mode().iloc[0] if not eth.mode().empty else np.nan
        ethnic_share_modal = (eth == ethnic_modal).mean() if len(eth) > 0 else np.nan
        has_minang = (g["ar15d"] == 9).any()
        has_batak = (g["ar15d"] == 4).any()
        has_java = (g["ar15d"] == 1).any()

        # Remittance HH-level
        amount_given = g["amount_given"].fillna(0).sum()
        amount_received = g["amount_received"].fillna(0).sum()

        return pd.Series({
            "n_members": n_members,
            "n_adults": n_adults,
            "n_outmig_adults": n_outmig,
            "amount_given_total": amount_given,
            "amount_received_total": amount_received,
            "log_received_total": np.log1p(amount_received),
            "ethnic_modal": ethnic_modal,
            "ethnic_share_modal": ethnic_share_modal,
            "has_minang": has_minang,
            "has_batak": has_batak,
            "has_java": has_java,
        })

    hh = ind.groupby("hhid14").apply(hh_aggregate, include_groups=False).reset_index()

    # 3-dim NELM scores
    # D1: deployment ratio
    hh["nelm_d1_deployment"] = (hh["n_outmig_adults"] / hh["n_adults"]).where(
        hh["n_adults"] > 0)
    # D2: remittance dependency (log IDR)
    hh["nelm_d2_remittance"] = hh["log_received_total"]
    # D3: CONTINUOUS extended ratio = (members beyond nuclear "2 adults") / total members
    # Nuclear baseline = 2 adults; anything beyond is extended
    hh["non_nuclear_adults"] = (hh["n_adults"] - 2).clip(lower=0)
    hh["nelm_d3_extended_ratio"] = (hh["non_nuclear_adults"] / hh["n_members"]).where(
        hh["n_members"] > 0)

    # Merge expenditure
    hh = hh.merge(exp[["hhid14", "expenditure_total_monthly", "log_exp_monthly"]],
                    on="hhid14", how="left")
    hh["expenditure_pcm"] = (hh["expenditure_total_monthly"] / hh["n_members"]).where(
        hh["n_members"] > 0)
    hh["log_exp_pcm"] = np.log1p(hh["expenditure_pcm"])

    # Province + urban from sc1
    hh = hh.merge(sc1.drop_duplicates("hhid14"), on="hhid14", how="left")
    hh["urban"] = (hh["urban_rural"] == 1).astype(int)

    log(f"  household table: {len(hh):,} rows")
    log(f"  Minang ethnic-modal HH: {(hh['ethnic_modal'] == 9).sum():,}")
    log(f"  Batak  ethnic-modal HH: {(hh['ethnic_modal'] == 4).sum():,}")
    log(f"  Java   ethnic-modal HH: {(hh['ethnic_modal'] == 1).sum():,}")

    # Sample size after dropping missing in 3-dim NELM
    nelm3 = ["nelm_d1_deployment", "nelm_d2_remittance", "nelm_d3_extended_ratio"]
    n_complete = hh.dropna(subset=nelm3).shape[0]
    log(f"  HH with all 3 NELM dim complete: {n_complete:,}  "
        f"(was {hh.dropna(subset=nelm3 + ['expenditure_pcm']).shape[0]:,} with expenditure)")

    out_hh = OUT / "household_nelm_strategies.csv"
    hh.to_csv(out_hh, index=False)
    log(f"  saved {out_hh}")

    # =============================================================
    # 7. ETHNIC SUBSETS + descriptive comparison
    # =============================================================
    log("\n[7/7] Ethnic subsets")
    minang_hh = hh[hh["ethnic_modal"] == 9].copy()
    batak_hh = hh[hh["ethnic_modal"] == 4].copy()
    java_hh = hh[hh["ethnic_modal"] == 1].copy()

    minang_hh.to_csv(OUT / "minang_households.csv", index=False)
    batak_hh.to_csv(OUT / "batak_households.csv", index=False)
    java_hh.to_csv(OUT / "java_households.csv", index=False)
    log(f"  Minang ethnic-modal HH: {len(minang_hh):,}")
    log(f"  Batak  ethnic-modal HH: {len(batak_hh):,}")
    log(f"  Java   ethnic-modal HH: {len(java_hh):,}")

    log("\n=== 3-dim NELM + expenditure: Minang vs Batak (means) ===")
    cols = nelm3 + ["log_exp_pcm", "ethnic_share_modal"]
    cmp = pd.DataFrame({
        "Minang": minang_hh[cols].mean(),
        "Batak":  batak_hh[cols].mean(),
        "Java":   java_hh[cols].mean(),
    })
    log(cmp.round(3).to_string())

    # Mann-Whitney tests
    log("\n=== Mann-Whitney U: Minang vs Batak ===")
    from scipy.stats import mannwhitneyu
    for d in nelm3 + ["log_exp_pcm"]:
        a = minang_hh[d].dropna()
        b = batak_hh[d].dropna()
        if len(a) > 5 and len(b) > 5:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            diff = a.mean() - b.mean()
            sign = "+" if diff > 0 else "-"
            log(f"  {d:30s}: Minang={a.mean():.3f}  Batak={b.mean():.3f}  "
                f"diff={sign}{abs(diff):.3f}  p={p:.4f}")

    # save log
    (OUT / "nelm_build_log.txt").write_text("\n".join(LOG))
    log(f"\nLog saved → {OUT / 'nelm_build_log.txt'}")


if __name__ == "__main__":
    main()
