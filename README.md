# Replication Package — Configuring Translocal Households

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Replication materials for the paper:

> **Simon, J., Ismail, I., Tahir, R., Saragih, S., & Angelia, N. (2026).** Configuring Translocal Households: Kinship Structure and the New Economics of Labor Migration in Bataknese versus Minangkabau Indonesia. *Journal of Ethnic and Migration Studies* (under review).

---

## Overview

This repository contains the analysis pipeline, derived datasets, output tables, and figures supporting the paper. The paper tests whether matrilineal Minangkabau and patrilineal Bataknese households in Indonesia configure their translocal migration economies into systematically different strategy profiles, drawing on the Indonesian Family Life Survey wave 5 (IFLS-5, 2014).

### Key Results
- Sample: 6,936 ethnic-modal households (Java 5,679; Batak 583; Minang 674) drawn from 14,451 households across 13 Indonesian provinces (excluding Aceh).
- Five household-strategy clusters identified via k-means (silhouette 0.628 at k = 5; Ward agreement ARI = 0.68).
- Ethnicity remains a significant predictor of cluster membership after controlling for HH-head age, sex, education, urban residence, household size, log per-capita expenditure, and province (likelihood-ratio chi-squared = 34.15, df = 8, p = 3.8 x 10⁻⁵; pseudo R² = 0.30).
- Findings robust across two clustering methods, three alternative k values, eight subsample sensitivity tests, and 200-replicate household-level bootstrap CIs.
- Sub-ethnic Bataknese decomposition (Christian-Muslim) shows no significant within-Batak differences, supporting the interpretation that kinship structure (not religion) drives the contrast.

---

## Repository Structure

```
replication_package/
├── README.md                                this file
├── LICENSE                                  CC BY 4.0 (data) + MIT (code)
├── CITATION.cff                             machine-readable citation
├── requirements.txt                         Python dependencies
├── .gitignore                               excludes raw IFLS data (license-restricted)
│
├── data/
│   ├── README.md                            data acquisition + provenance
│   └── processed/                           derived datasets (CSV, 7 files)
│       ├── household_nelm_strategies.csv    14,451 HH x 21 cols (NELM dim + ethnicity + SES)
│       ├── household_nelm_clustered.csv     7,340 HH (ethnic-modal Java/Batak/Minang)
│       ├── cluster_profiles_kmeans.csv      5 clusters x NELM means + ethnic share
│       ├── cluster_profiles_ward.csv        Ward clustering robustness
│       ├── cluster_diagnostics.csv          k = 2 to 6 silhouette + Calinski-Harabasz
│       ├── cluster_method_ari.csv           Adjusted Rand Index k-means vs Ward
│       └── batak_subethnic_classified.csv   647 Batak HH (Christian vs Muslim diaspora)
│
├── scripts/                                 analysis pipeline (Python 3.10+)
│   ├── 20_clean/
│   │   └── P1_build_nelm_strategies.py      NELM 3-dim operationalization from IFLS-5
│   ├── 30_analyze/
│   │   ├── P1_02_kmeans_typology.py         K-means + Ward clustering, k = 2 to 6
│   │   ├── P1_03_multinomial_logit.py       Multinomial logit, ethnicity x cluster
│   │   ├── P1_04_subethnic_batak.py         Sub-ethnic Batak classification
│   │   ├── P1_05_robustness_checks.py       Alt k, GMM, ARI, OLS direct evidence
│   │   ├── P1_06_sensitivity_endogeneity.py 8 subsample sensitivity tests
│   │   └── P1_07_bootstrap_ci.py            200-replicate household-level bootstrap
│   ├── 99_tests/
│   │   └── verify_all_provenance.py         SHA-256 integrity check
│   └── lib/
│       └── provenance.py                    provenance utility (write + verify)
│
├── outputs/
│   ├── tables/                              all CSV tables in paper + appendix
│   │   ├── P1_multinomial_logit_results.csv
│   │   ├── P1_multinomial_logit_ward.csv
│   │   ├── P1_lr_tests_summary.csv
│   │   ├── P1_marginal_effects.csv
│   │   ├── P1_logit_summary.txt
│   │   ├── P1_subethnic_batak_nelm_comparison.csv
│   │   ├── P1_subethnic_kruskal.csv
│   │   ├── P1_subethnic_pairwise.csv
│   │   ├── P1_sensitivity_results.csv
│   │   ├── P1_bootstrap_ci.csv
│   │   └── P1_robustness_results.csv
│   └── figures/                             7 PNG figures (300 DPI)
│       ├── fig_P1_02_silhouette_k.png
│       ├── fig_P1_02_cluster_profiles.png
│       ├── fig_P1_02_dendrogram.png
│       ├── fig_P1_02_ethnic_composition.png
│       ├── fig_P1_04_subethnic_nelm.png
│       ├── fig_P1_06_sensitivity_forest.png
│       └── fig_P1_07_bootstrap_forest.png
│
└── docs/
    ├── methodology.md                       NELM operationalization details
    ├── data_sources.md                      how to acquire IFLS-5 from RAND
    └── reproducibility.md                   step-by-step replication guide
```

---

## Quick Start

### 1. Acquire the Source Data

This package does **not** include the raw IFLS-5 microdata, which is licensed by RAND and requires individual user agreement. Follow `docs/data_sources.md` to:

1. Register at RAND IFLS: https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS.html
2. Accept the data-use agreement
3. Download IFLS-5 STATA files (~2 GB)
4. Place files in `data/raw/ifls5/hh14/` (folder you create locally)

The package includes derived datasets (in `data/processed/`) that allow replication of all paper figures and tables without re-running the cleaning pipeline. Re-running the cleaning pipeline requires the raw IFLS-5 files.

### 2. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/jhonsimon07/NELM-kinship-indonesia.git
cd NELM-kinship-indonesia

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Reproduce the Analysis

```bash
# Option A: Re-run from processed data (no raw IFLS-5 needed)
python3 scripts/30_analyze/P1_02_kmeans_typology.py
python3 scripts/30_analyze/P1_03_multinomial_logit.py
python3 scripts/30_analyze/P1_04_subethnic_batak.py
python3 scripts/30_analyze/P1_06_sensitivity_endogeneity.py
python3 scripts/30_analyze/P1_07_bootstrap_ci.py

# Option B: Re-build everything from raw IFLS-5 (requires data/raw/ifls5/)
python3 scripts/20_clean/P1_build_nelm_strategies.py
# ... then re-run Option A scripts
```

Expected runtime: 5-15 minutes on a standard laptop (most time in P1_07 bootstrap with 200 replicates).

### 4. Verify Outputs

Tables generated in `outputs/tables/` should match the paper's Tables 1-5. Figures in `outputs/figures/` correspond to Figures 1-4 in the manuscript.

---

## Data Provenance

IFLS-5 raw files: SHA-256 checksums recorded in our project provenance YAML (not included in this package; see methods statement in the paper for verification protocol).

Derived data: each CSV in `data/processed/` was generated by `scripts/20_clean/P1_build_nelm_strategies.py` from IFLS-5 STATA files using the operationalizations described in Section 3 of the paper.

---

## Software

- Python 3.10+
- pandas, numpy, scipy, scikit-learn, statsmodels, pyreadstat, matplotlib, joblib (versions in `requirements.txt`)

---

## Citation

If you use these materials, please cite:

```bibtex
@article{simon2026configuring,
  author  = {Simon, Jhon and Ismail, Isdawati and Tahir, Rahman and Saragih, Siswati and Angelia, Nina},
  title   = {Configuring Translocal Households: Kinship Structure and the New Economics of Labor Migration in Bataknese versus Minangkabau Indonesia},
  journal = {Journal of Ethnic and Migration Studies},
  year    = {2026},
  note    = {Under review}
}
```

For the replication package itself:

```bibtex
@dataset{simon2026replication,
  author       = {Simon, Jhon and Ismail, Isdawati and Tahir, Rahman and Saragih, Siswati and Angelia, Nina},
  title        = {Replication package for Configuring Translocal Households},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.PLACEHOLDER},
  url          = {https://doi.org/10.5281/zenodo.PLACEHOLDER}
}
```

---

## License

- **Code** (`scripts/`): MIT License (see LICENSE)
- **Derived data + documentation**: CC BY 4.0 (Creative Commons Attribution 4.0)
- **IFLS-5 raw data**: RAND data-use license (separate; not redistributed here)

---

## Contact

**Corresponding author:** Jhon Simon
Faculty of Social and Political Sciences
Universitas Dharmawangsa
Medan, Indonesia
Email: jhon.simon@dharmawangsa.ac.id

---

## Acknowledgements

We thank RAND, the Center for Population and Policy Studies at Universitas Gadjah Mada, and Survey METRE for collecting and disseminating the IFLS-5 data. This research is self-funded by the authors; no external grants were received.
