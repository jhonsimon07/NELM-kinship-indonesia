#!/usr/bin/env python3
"""
P1_04 — Sub-ethnic Bataknese decomposition: refined v2 (Round 2).

CHANGES from V1:
  - Use kab_residence (sc02_14_14) as primary proxy (high coverage)
  - Fallback to kab_birth (mg01c) if available
  - Religion (ar15) as secondary disambiguator
  - Multi-criteria classification with priority logic

CLASSIFICATION RULES (priority order):
  1. If kab_residence ∈ Toba area + Christian → Toba
  2. If kab_residence ∈ Karo area + Christian → Karo
  3. If kab_residence ∈ Mandailing area + Muslim → Mandailing/Angkola
  4. If kab_residence ∈ Simalungun area + Christian → Simalungun
  5. If kab_residence ∈ Pakpak area + Christian → Pakpak
  6. If kab_birth-based fallback (using same logic with mg01c)
  7. Religion-only if no kab info: Muslim → Mandailing/Angkola; Christian → Christian-Batak (combined)
  8. Else → Diaspora-Batak (likely living outside Batak homeland kab)

Output:
  data_processed/ifls5_nelm/batak_subethnic_classified.csv
  outputs/tables/P1_subethnic_batak_nelm_comparison.csv
  outputs/tables/P1_subethnic_kruskal.csv
  outputs/tables/P1_subethnic_pairwise.csv
  outputs/figures/fig_P1_04_subethnic_nelm.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

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

# BPS kab codes per Batak sub-region (2014 codes)
SUB_KAB = {
    "Toba":              [1202, 1204, 1213, 1215, 1271],   # Taput, Tobasa, Humbang, Samosir, Sibolga
    "Karo":              [1209],                              # Karo
    "Mandailing/Angkola": [1203, 1224, 1218, 1219, 1277],   # Tapsel, Madina, Padlawut, Padlaw, P'sidimpuan
    "Simalungun":        [1207, 1273],                        # Simalungun, P'siantar
    "Pakpak":            [1208, 1214],                        # Dairi, Pakpak Bharat
}

RELIGION_CHRISTIAN = (2, 3)   # Catholic, Protestant
RELIGION_MUSLIM = (1,)


def classify_individual(row) -> str:
    """Classify Batak individual using residence-then-birth-then-religion logic."""
    relig = row.get("ar15")
    kab_res = row.get("kab_bps_res")    # current residence
    kab_birth = row.get("mg01c")          # birth (often missing)

    if pd.isna(relig):
        return "Unclassified"
    relig = int(relig)

    # Try residence first (highest coverage)
    for kab_var, source in [(kab_res, "res"), (kab_birth, "birth")]:
        if pd.isna(kab_var):
            continue
        kab = int(kab_var)
        # Toba (Christian)
        if kab in SUB_KAB["Toba"] and relig in RELIGION_CHRISTIAN:
            return "Toba"
        # Karo (mostly Christian, some Muslim)
        if kab in SUB_KAB["Karo"]:
            return "Karo"
        # Mandailing (Muslim)
        if kab in SUB_KAB["Mandailing/Angkola"] and relig in RELIGION_MUSLIM:
            return "Mandailing/Angkola"
        # Simalungun (Christian)
        if kab in SUB_KAB["Simalungun"] and relig in RELIGION_CHRISTIAN:
            return "Simalungun"
        # Pakpak (Christian)
        if kab in SUB_KAB["Pakpak"] and relig in RELIGION_CHRISTIAN:
            return "Pakpak"

    # Religion-only fallback (for diaspora/non-homeland-residing Batak)
    if relig in RELIGION_MUSLIM:
        return "Diaspora-Muslim-Batak"   # likely Mandailing-origin
    elif relig in RELIGION_CHRISTIAN:
        return "Diaspora-Christian-Batak"  # likely Toba/Karo/etc-origin
    else:
        return "Unclassified"


def main():
    print("=" * 78)
    log = []
    print("P1_04 — Sub-ethnic Batak decomposition (Round 2 refined)")
    print("=" * 78)

    # =============================================================
    # 1. Load Batak HH + members
    # =============================================================
    print("\n[1] Loading Batak HH + members")
    hh = pd.read_csv(NELM / "household_nelm_strategies.csv")
    batak_hh = hh[hh["ethnic_modal"] == 4].copy()
    print(f"  Batak ethnic-modal HH: {len(batak_hh):,}")

    ar, _ = pyreadstat.read_dta(str(HH14 / "bk_ar1.dta"),
                                  usecols=["hhid14", "pidlink", "ar15", "ar15d"])
    mg, _ = pyreadstat.read_dta(str(HH14 / "b3a_mg1.dta"),
                                  usecols=["pidlink", "mg01c"])
    sc, _ = pyreadstat.read_dta(str(HH14 / "bk_sc1.dta"),
                                  usecols=["hhid14", "sc02_14_14"])
    sc = sc.rename(columns={"sc02_14_14": "kab_bps_res"}).drop_duplicates("hhid14")

    ar = ar.merge(mg, on="pidlink", how="left")
    ar = ar.merge(sc, on="hhid14", how="left")
    batak_ind = ar[ar["ar15d"] == 4].copy()
    print(f"  Batak individuals: {len(batak_ind):,}")
    print(f"  with mg01c (kab birth): {batak_ind['mg01c'].notna().sum():,}")
    print(f"  with sc02_14_14 (kab res): {batak_ind['kab_bps_res'].notna().sum():,}")

    # =============================================================
    # 2. Classify individuals
    # =============================================================
    print("\n[2] Classifying individuals")
    batak_ind["subethnic"] = batak_ind.apply(classify_individual, axis=1)
    print("  Sub-ethnic distribution (individual level):")
    print(batak_ind["subethnic"].value_counts().to_string())

    # =============================================================
    # 3. Modal sub-ethnic per HH
    # =============================================================
    print("\n[3] Modal sub-ethnic per HH")
    hh_sub = (batak_ind.groupby("hhid14")["subethnic"]
                          .apply(lambda s: s.mode().iloc[0]
                                  if not s.mode().empty else "Unclassified")
                          .reset_index()
                          .rename(columns={"subethnic": "subethnic_modal"}))
    batak_hh = batak_hh.merge(hh_sub, on="hhid14", how="left")
    print("  HH classified by modal sub-ethnic:")
    print(batak_hh["subethnic_modal"].value_counts().to_string())

    out_classified = NELM / "batak_subethnic_classified.csv"
    batak_hh.to_csv(out_classified, index=False)
    print(f"\n  → {out_classified}")

    # =============================================================
    # 4. Compare NELM 3-dim across sub-ethnic groups
    # =============================================================
    print("\n[4] NELM 3-dim by sub-ethnic group")
    nelm_cols = ["nelm_d1_deployment", "nelm_d2_remittance",
                  "nelm_d3_extended_ratio"]
    counts = batak_hh["subethnic_modal"].value_counts()
    use_groups = counts[counts >= 20].index.tolist()
    print(f"  Sub-ethnic groups with n≥20: {use_groups}")

    sub_compare = batak_hh[batak_hh["subethnic_modal"].isin(use_groups)].copy()
    grp_means = sub_compare.groupby("subethnic_modal")[nelm_cols].mean().round(3)
    grp_n = sub_compare.groupby("subethnic_modal").size().rename("n_HH")
    summary = grp_means.copy()
    summary["n_HH"] = grp_n
    print(summary.to_string())
    summary.to_csv(OUT_TAB / "P1_subethnic_batak_nelm_comparison.csv")

    # =============================================================
    # 5. Kruskal-Wallis (overall) + pairwise Mann-Whitney
    # =============================================================
    print("\n[5] Kruskal-Wallis + pairwise tests")
    kw_rows = []
    for d in nelm_cols:
        groups_data = [sub_compare[sub_compare["subethnic_modal"] == g][d].dropna()
                          for g in use_groups]
        valid_groups = [g for g in groups_data if len(g) > 5]
        if len(valid_groups) >= 2:
            stat, p = kruskal(*valid_groups)
            print(f"  {d}: KW stat={stat:.3f}, p={p:.4f}")
            kw_rows.append({"dim": d, "kw_stat": stat, "p_value": p,
                              "n_groups": len(valid_groups)})

    pd.DataFrame(kw_rows).to_csv(OUT_TAB / "P1_subethnic_kruskal.csv", index=False)

    # Pairwise (only main groups)
    print("\n  Pairwise Mann-Whitney comparisons (n≥30):")
    main_groups = [g for g in use_groups if counts[g] >= 30]
    pw_rows = []
    for g1, g2 in combinations(main_groups, 2):
        for d in nelm_cols:
            a = sub_compare[sub_compare["subethnic_modal"] == g1][d].dropna()
            b = sub_compare[sub_compare["subethnic_modal"] == g2][d].dropna()
            if len(a) > 10 and len(b) > 10:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                diff = a.mean() - b.mean()
                pw_rows.append({"group1": g1, "group2": g2, "dim": d,
                                  "mean1": a.mean(), "mean2": b.mean(),
                                  "diff": diff, "p_value": p,
                                  "sig": "*" if p < 0.05 else ""})
                if p < 0.10:
                    print(f"    {g1[:10]:10s} vs {g2[:10]:10s} | {d:25s} "
                          f"diff={diff:+.3f}  p={p:.4f}")

    pd.DataFrame(pw_rows).to_csv(OUT_TAB / "P1_subethnic_pairwise.csv", index=False)

    # =============================================================
    # 6. Figure: NELM 3-dim by sub-ethnic
    # =============================================================
    print("\n[6] Figure")
    if len(use_groups) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for i, d in enumerate(nelm_cols):
            ax = axes[i]
            data_per_group = [sub_compare[sub_compare["subethnic_modal"] == g][d].dropna()
                                  for g in use_groups]
            bp = ax.boxplot(data_per_group, tick_labels=use_groups,
                              showfliers=False, patch_artist=True)
            for patch, c in zip(bp["boxes"], plt.cm.Set2.colors):
                patch.set_facecolor(c)
            ax.set_title(d.replace("nelm_d", "D"))
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("NELM 3 dimensions by Bataknese sub-ethnic (Round 2)",
                       fontsize=12, y=1.02)
        plt.tight_layout()
        out_fig = OUT_FIG / "fig_P1_04_subethnic_nelm.png"
        fig.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_fig}")

    print("\n" + "=" * 78)
    print("P1_04 v2 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
