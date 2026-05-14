# Data Directory

This directory contains derived datasets generated from the Indonesian Family Life Survey wave 5 (IFLS-5, 2014). Raw IFLS-5 files are NOT included due to RAND's data-use license; see `docs/data_sources.md` for acquisition instructions.

## Folder Structure

```
data/
├── README.md               this file
├── raw/                    NOT INCLUDED (license-restricted, gitignored)
│   └── ifls5/              place IFLS-5 STATA files here after RAND registration
│       └── hh14/           IFLS-5 wave 5 household files (.dta)
└── processed/              derived datasets from cleaning pipeline (7 CSV files)
```

## Processed Datasets

All processed datasets are derived using `scripts/20_clean/P1_build_nelm_strategies.py` from raw IFLS-5 files. The processing pipeline operationalizes three New Economics of Labor Migration (NELM) dimensions and household-level controls per the operationalizations described in Section 3 of the paper.

| File | Rows | Cols | Description |
|---|---|---|---|
| `household_nelm_strategies.csv` | 14,451 | 21 | All IFLS-5 households with NELM 3-dim + expenditure + ethnicity |
| `household_nelm_clustered.csv` | 7,340 | 23 | Ethnic-modal Java/Batak/Minang households with cluster labels (k-means + Ward at k=5) |
| `cluster_profiles_kmeans.csv` | 5 | 9 | k=5 k-means cluster profile (NELM means + ethnic shares) |
| `cluster_profiles_ward.csv` | 5 | 9 | k=5 Ward hierarchical cluster profile |
| `cluster_diagnostics.csv` | 10 | 5 | Silhouette + Calinski-Harabasz for k=2..6 (both methods) |
| `cluster_method_ari.csv` | 5 | 2 | Adjusted Rand Index between k-means and Ward at each k |
| `batak_subethnic_classified.csv` | 647 | 24 | Batak households classified into Christian/Muslim diaspora subgroups |

## Variable Documentation

### Common Variables (across files)
- `hhid14`: IFLS-5 household ID (alphanumeric)
- `pidlink`: IFLS-5 individual link ID
- `ethnic_modal`: modal ethnicity code (1 = Java, 4 = Batak, 9 = Minang)
- `ethnic_share_modal`: proportion of household members sharing modal ethnicity
- `n_members`: total household members
- `n_adults`: members aged 15+

### NELM Dimensions
- `nelm_d1_deployment`: proportion of adults out-migrated (birth-kabupaten differs from 2014 residence)
- `nelm_d2_remittance`: log(1 + total monetary transfers received from external kin in past year, IDR)
- `nelm_d3_extended_ratio`: max(0, n_adults - 2) / n_members (co-residence extent)

### Auxiliary Controls
- `head_age`, `head_female`, `head_edu_lvl`: HH-head characteristics
- `urban`: dummy (1 = urban kabupaten, 0 = rural)
- `log_exp_pcm`: log per-capita monthly household expenditure (IDR)
- `prov_bps`: BPS province code (sc01_14_14)

### Cluster Labels (in `household_nelm_clustered.csv`)
- `kmeans_cluster`: 0..4 (cluster index at k=5, k-means primary)
- `ward_cluster`: 0..4 (cluster index at k=5, Ward robustness)

## Data Provenance

Raw IFLS-5 files: SHA-256 checksums recorded in our project provenance YAML (private; see paper methods section for verification protocol).

Derived data: each CSV traced to exact pipeline script in `scripts/20_clean/`. Re-running the pipeline against the same IFLS-5 input produces byte-identical CSV output (modulo floating-point precision in numpy operations).

## Privacy Notes

Per RAND's IFLS data-use policy, derived datasets in this directory:
- Contain only household-level aggregates and standardized NELM dimensions
- Do NOT permit reconstruction of individual records
- Retain `hhid14` and `pidlink` for replication reference only (these are IFLS-internal codes, not personally identifiable)

Individual-level intermediate file (`individual_nelm_components.csv`) is excluded from this replication package via `.gitignore`. To reproduce it, run `scripts/20_clean/P1_build_nelm_strategies.py` against raw IFLS-5 files locally.
