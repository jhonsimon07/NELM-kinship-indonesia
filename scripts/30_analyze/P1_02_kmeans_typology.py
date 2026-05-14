#!/usr/bin/env python3
"""
P1_02 — Typology of household NELM strategies via k-means + Ward hierarchical.

ROUND 1 REFINEMENT (9 May 2026):
  - Use 3-dim NELM (D4 education dropped)
  - Add Ward hierarchical clustering as 3rd method
  - Compute ARI between k-means and Ward (cluster solution stability)
  - All k=2..6 evaluated

Output: data_processed/ifls5_nelm/
  - household_nelm_clustered.csv      HH + kmeans_cluster + ward_cluster
  - cluster_profiles_kmeans.csv       k-means cluster × NELM mean + ethnic share
  - cluster_profiles_ward.csv         Ward cluster × NELM mean + ethnic share
  - cluster_diagnostics.csv           silhouette + CH per k per method
  - cluster_method_ari.csv            ARI between k-means and Ward

Output: outputs/figures/
  - fig_P1_02_silhouette_k.png        Diagnostic plot for both methods
  - fig_P1_02_cluster_profiles.png    Profile heatmap
  - fig_P1_02_dendrogram.png          Hierarchical dendrogram
  - fig_P1_02_ethnic_composition.png  Ethnic composition per cluster
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                                silhouette_score)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NELM = PROJECT_ROOT / "data_processed" / "ifls5_nelm"
OUT_FIG = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "figure.dpi": 110,
})

ETHNIC_LABELS = {1: "Java", 4: "Batak", 9: "Minang"}
ETHNIC_COLORS = {1: "#aaaaaa", 4: "#d62728", 9: "#1f77b4"}

NELM_COLS = ["nelm_d1_deployment", "nelm_d2_remittance", "nelm_d3_extended_ratio"]


def main():
    print("=" * 78)
    print("P1_02 — Typology of NELM strategies (k-means + Ward, 3-dim)")
    print("=" * 78)

    # =========================================================
    # 1. Load + filter to ethnic-modal HH (Java/Minang/Batak)
    # =========================================================
    df = pd.read_csv(NELM / "household_nelm_strategies.csv")
    print(f"\nFull HH table: {len(df):,}")

    df = df[df["ethnic_modal"].isin([1, 4, 9])].copy()
    print(f"Ethnic-modal Java/Batak/Minang HH: {len(df):,}")

    df_complete = df.dropna(subset=NELM_COLS).copy()
    print(f"HH with all 3 NELM dim complete: {len(df_complete):,}")
    print(f"  Java:  {(df_complete['ethnic_modal'] == 1).sum():,}")
    print(f"  Batak: {(df_complete['ethnic_modal'] == 4).sum():,}")
    print(f"  Minang:{(df_complete['ethnic_modal'] == 9).sum():,}")

    # =========================================================
    # 2. Standardize
    # =========================================================
    print("\n[2] Standardize NELM 3 dim")
    X = df_complete[NELM_COLS].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # =========================================================
    # 3. K-means k=2..6
    # =========================================================
    print("\n[3] K-means k=2..6")
    diag_rows = []
    kmeans_solutions = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels)
        ch = calinski_harabasz_score(Xz, labels)
        diag_rows.append({"method": "kmeans", "k": k,
                            "silhouette": sil, "calinski_harabasz": ch,
                            "inertia": km.inertia_})
        kmeans_solutions[k] = labels
        print(f"  k={k}: silhouette={sil:.4f}  CH={ch:.1f}")

    # =========================================================
    # 4. Ward hierarchical k=2..6
    # =========================================================
    print("\n[4] Ward hierarchical k=2..6")
    ward_solutions = {}
    for k in range(2, 7):
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(Xz)
        sil = silhouette_score(Xz, labels)
        ch = calinski_harabasz_score(Xz, labels)
        diag_rows.append({"method": "ward", "k": k,
                            "silhouette": sil, "calinski_harabasz": ch,
                            "inertia": np.nan})
        ward_solutions[k] = labels
        print(f"  k={k}: silhouette={sil:.4f}  CH={ch:.1f}")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(NELM / "cluster_diagnostics.csv", index=False)

    # Pick optimal k (max silhouette in k-means as primary)
    km_diag = diag_df[diag_df["method"] == "kmeans"]
    best_k = int(km_diag.loc[km_diag["silhouette"].idxmax(), "k"])
    print(f"\n  Optimal k (k-means silhouette): {best_k}")

    # =========================================================
    # 5. ARI: k-means vs Ward at same k
    # =========================================================
    print("\n[5] ARI between k-means and Ward (cluster stability)")
    ari_rows = []
    for k in range(2, 7):
        ari = adjusted_rand_score(kmeans_solutions[k], ward_solutions[k])
        ari_rows.append({"k": k, "ari_kmeans_vs_ward": ari})
        print(f"  k={k}: ARI = {ari:.3f}")
    pd.DataFrame(ari_rows).to_csv(NELM / "cluster_method_ari.csv", index=False)

    # =========================================================
    # 6. Profile clusters at best_k (both methods)
    # =========================================================
    df_complete["kmeans_cluster"] = kmeans_solutions[best_k]
    df_complete["ward_cluster"] = ward_solutions[best_k]

    print(f"\n[6] Profiling at k={best_k}")
    for method in ["kmeans", "ward"]:
        col = f"{method}_cluster"
        prof = df_complete.groupby(col)[NELM_COLS].mean()
        prof["n_HH"] = df_complete.groupby(col).size()
        eth_comp = pd.crosstab(df_complete[col], df_complete["ethnic_modal"],
                                  normalize="index").round(3)
        eth_comp.columns = [ETHNIC_LABELS.get(c, str(c)) for c in eth_comp.columns]
        for c in eth_comp.columns:
            prof[f"share_{c}"] = eth_comp[c]
        prof.to_csv(NELM / f"cluster_profiles_{method}.csv")
        print(f"\n  {method.upper()} profile (k={best_k}):")
        print(prof.round(3).to_string())

    out_clustered = NELM / "household_nelm_clustered.csv"
    df_complete.to_csv(out_clustered, index=False)
    print(f"\n  → {out_clustered}")

    # =========================================================
    # 7. FIGURES
    # =========================================================
    print("\n[7] Figures")

    # 7a: silhouette comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, color, marker in [("kmeans", "#1f77b4", "o"),
                                       ("ward", "#ff7f0e", "s")]:
        sub = diag_df[diag_df["method"] == method]
        axes[0].plot(sub["k"], sub["silhouette"], marker=marker,
                       color=color, label=method)
        axes[1].plot(sub["k"], sub["calinski_harabasz"], marker=marker,
                       color=color, label=method)
    axes[0].axvline(best_k, color="red", linestyle="--", lw=0.5)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title(f"Silhouette (best k-means at k={best_k})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Calinski-Harabasz score")
    axes[1].set_title("CH (higher = better-separated)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_02_silhouette_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_02_silhouette_k.png")

    # 7b: cluster profile heatmap (k-means primary)
    km_prof = pd.read_csv(NELM / "cluster_profiles_kmeans.csv", index_col=0)
    nelm_z = pd.DataFrame(scaler.transform(km_prof[NELM_COLS].values),
                              index=km_prof.index, columns=NELM_COLS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(nelm_z.T, cmap="RdBu_r", aspect="auto",
                     vmin=-2, vmax=2)
    ax.set_xticks(range(len(km_prof.index)))
    ax.set_xticklabels([f"C{i}" for i in km_prof.index])
    ax.set_yticks(range(len(NELM_COLS)))
    ax.set_yticklabels([c.replace("nelm_d", "D") for c in NELM_COLS])
    ax.set_title(f"NELM cluster profiles (k-means k={best_k})\n"
                  "blue = below mean, red = above mean (z-score)")
    # annotate cell values
    for i in range(len(km_prof.index)):
        for j in range(len(NELM_COLS)):
            ax.text(i, j, f"{nelm_z.iloc[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if abs(nelm_z.iloc[i, j]) > 1 else "black",
                     fontsize=9)
    plt.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_02_cluster_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_02_cluster_profiles.png")

    # 7c: ethnic composition stacked bar
    eth_comp = pd.crosstab(df_complete["kmeans_cluster"],
                              df_complete["ethnic_modal"], normalize="index")
    eth_comp.columns = [ETHNIC_LABELS.get(c, str(c)) for c in eth_comp.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    eth_comp.plot(kind="bar", stacked=True, ax=ax,
                    color=[ETHNIC_COLORS[v] for v in [1, 4, 9]
                            if v in df_complete["ethnic_modal"].values],
                    edgecolor="white", lw=0.5)
    ax.set_xlabel("Cluster (k-means)")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Ethnic composition per k-means cluster (k={best_k})")
    ax.set_xticklabels([f"C{i}" for i in eth_comp.index], rotation=0)
    ax.legend(title="Ethnicity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_02_ethnic_composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_02_ethnic_composition.png")

    # 7d: dendrogram (Ward)
    print("\n  Computing dendrogram (subsample 500 HH for visibility)")
    np.random.seed(42)
    if len(Xz) > 500:
        idx = np.random.choice(len(Xz), 500, replace=False)
        Xz_sample = Xz[idx]
    else:
        Xz_sample = Xz
    Z = linkage(Xz_sample, method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=4,
                color_threshold=Z[-(best_k - 1), 2] if best_k > 1 else 0,
                no_labels=True)
    ax.axhline(Z[-(best_k - 1), 2], color="red", linestyle="--", lw=0.7,
                 label=f"cut at k={best_k}")
    ax.set_title(f"Ward hierarchical dendrogram (subsample n=500)")
    ax.set_ylabel("Distance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_P1_02_dendrogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_P1_02_dendrogram.png")

    print("\n" + "=" * 78)
    print(f"P1_02 done. Optimal k={best_k}, sample n={len(df_complete):,}")
    print("=" * 78)


if __name__ == "__main__":
    main()
