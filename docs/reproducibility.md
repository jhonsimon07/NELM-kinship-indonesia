# Reproducibility Guide

Step-by-step instructions to reproduce all tables and figures from the paper.

## Environment Setup

### Tested Configuration
- OS: Ubuntu 22.04 / macOS 13+ / Windows 11 (WSL recommended)
- Python: 3.10, 3.11, or 3.12
- Disk: ~5 GB free (4.6 MB for repo + 2 GB if you re-acquire IFLS-5 raw data)
- Memory: 8 GB RAM minimum (16 GB recommended for bootstrap step)

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/[user]/PJ2-paper1-replication.git
cd PJ2-paper1-replication

# Create isolated Python environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install exact dependency versions
pip install -r requirements.txt

# Verify install
python3 -c "import pandas, numpy, scipy, sklearn, statsmodels, pyreadstat, matplotlib, joblib; print('All deps OK')"
```

## Two Replication Paths

### Path A — Replicate from Processed Data (Recommended; ~10 minutes)

This path uses the derived CSV datasets in `data/processed/` and reproduces all tables and figures in the paper. **Raw IFLS-5 files are NOT needed.**

```bash
# 1. Cluster typology (Tables 2, 3 + Figures 1, 2)
python3 scripts/30_analyze/P1_02_kmeans_typology.py

# 2. Multinomial logit (Tables 4a, 4b, 5 + Figure 4)
python3 scripts/30_analyze/P1_03_multinomial_logit.py

# 3. Sub-ethnic Batak (Section 4.5 + supplementary figure)
python3 scripts/30_analyze/P1_04_subethnic_batak.py

# 4. Sensitivity analysis (Figure 3)
python3 scripts/30_analyze/P1_06_sensitivity_endogeneity.py

# 5. Bootstrap CIs (Table 4b values + Figure 4)
python3 scripts/30_analyze/P1_07_bootstrap_ci.py
```

**Expected runtime:** 8-12 minutes total, with P1_07 (bootstrap 200 replicates) taking the most time (5-7 minutes on 4 parallel cores).

### Path B — Full Pipeline from Raw IFLS-5 (~30 minutes)

This path re-creates the derived datasets from raw IFLS-5 files. **Requires raw IFLS-5 data; see `docs/data_sources.md`.**

```bash
# 0. (One-time) Acquire IFLS-5 raw files per docs/data_sources.md
#    Place STATA files in data/raw/ifls5/hh14/

# 1. Build NELM dimensions + auxiliary variables
python3 scripts/20_clean/P1_build_nelm_strategies.py

# Then continue with Path A steps 1-5 above
```

## Outputs

After running the scripts, the following files are generated or updated:

### Tables (in `outputs/tables/`)
- `P1_multinomial_logit_results.csv` — Table 4a coefficients
- `P1_multinomial_logit_ward.csv` — Ward robustness
- `P1_lr_tests_summary.csv` — likelihood-ratio tests
- `P1_marginal_effects.csv` — Table 5 marginal probabilities
- `P1_subethnic_*.csv` — sub-ethnic Batak Section 4.5
- `P1_sensitivity_results.csv` — 8 subsample specifications
- `P1_bootstrap_ci.csv` — Table 4b bootstrap intervals
- `P1_logit_summary.txt` — full statsmodels output

### Figures (in `outputs/figures/`, all 300 DPI PNG)
- `fig_P1_02_silhouette_k.png` — diagnostic (k = 2 to 6, both methods)
- `fig_P1_02_cluster_profiles.png` — Figure 1 in paper
- `fig_P1_02_dendrogram.png` — Ward dendrogram (supplementary)
- `fig_P1_02_ethnic_composition.png` — Figure 2 in paper
- `fig_P1_04_subethnic_nelm.png` — sub-ethnic comparison (supplementary)
- `fig_P1_06_sensitivity_forest.png` — Figure 3 in paper
- `fig_P1_07_bootstrap_forest.png` — Figure 4 in paper

## Verifying Reproducibility

Cross-check generated outputs against the paper's reported numbers:

| Paper Reference | Expected Value (from CSV) |
|---|---|
| Section 4.1 sample N | 6,936 (in `household_nelm_clustered.csv` after listwise) |
| Section 4.1 mean PCM expenditure | Rp 935,312 |
| Table 1 D1 deployment Batak | 0.155 (in `household_nelm_strategies.csv` for `ethnic_modal==4`) |
| Table 1 D2 remittance Minang | 4.843 |
| Table 4a LR test ethnicity | chi-squared = 34.15, p = 3.8e-5 |
| Table 4a pseudo R squared | 0.30 |
| Table 5 Java baseline C4 | 0.262 |
| Section 4.5 sub-ethnic KW | p = 0.55, 0.40, 0.08 for D1, D2, D3 |
| Section 4.5 sensitivity 7 of 8 | p < 0.05 (only homeland-only NS) |

If numbers match within floating-point precision (e.g., +/- 0.001 for percentages), the replication is successful.

## Troubleshooting

**Q: `pyreadstat` fails to install on macOS.**
A: Install via `pip install pyreadstat --no-binary :all:` or use Anaconda's `conda install -c conda-forge pyreadstat`.

**Q: `statsmodels` multinomial logit raises convergence warning.**
A: This is expected for a few subsample specifications in P1_06 (rural-only, low-expenditure tertile, Sumatra-only). These appear in the paper as "convergence error" in Appendix D. They do not affect the main results.

**Q: Bootstrap (P1_07) takes too long.**
A: Default is 200 replicates with 4 parallel jobs. To use fewer cores, edit `N_JOBS` at the top of the script. To reduce replicates (faster but less stable CIs), edit `N_BOOTSTRAP`.

**Q: Output figures look different (color, fonts).**
A: Set `plt.rcParams["font.family"] = "DejaVu Serif"` (default in scripts). Color palettes use matplotlib defaults; replication should match the paper figures within rendering tolerance.

## Reporting Issues

If replication fails or numbers do not match the paper, please open an issue at the GitHub repository with:
1. Your environment (`pip list` output)
2. The script that failed
3. Full error message or numerical discrepancy

## Citation

If your replication or extension produces published work, please cite both:
1. The paper (Simon et al. 2026, *Journal of Ethnic and Migration Studies*)
2. This replication package (Zenodo DOI when assigned)
