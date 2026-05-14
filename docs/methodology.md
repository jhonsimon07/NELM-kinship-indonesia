# Methodology Notes

Supplementary methodological detail for the analysis pipeline. For the full theoretical framework, see Section 2 of the paper.

## NELM 3-Dimension Operationalization

Following Stark and Bloom (1985) and Stark (1991), we operationalize three observable household-level dimensions of the New Economics of Labor Migration framework.

### D1. Migration Deployment

Defined as the proportion of working-age (>= 15 years) household members who are observed as out-migrated:

```
D1_h = n_adults_out_migrated_h / n_adults_h
```

**Operationalization:**
- "Out-migrated" = kabupaten of birth (`mg01c` in `b3a_mg1.dta`) differs from 2014 kabupaten of residence (`sc02_14_14` in `bk_sc1.dta`)
- "Adults" = members aged >= 15 at survey time (`ar09` in `bk_ar1.dta`)
- Households with `n_adults_h == 0` have D1 set to NaN

### D2. Remittance Dependency

Log-transformed total monetary transfers received from external kin:

```
D2_h = log(1 + sum_i in h of IDR_received_i)
```

**Operationalization:**
- Source variable: `tf06a` in `b3b_tf.dta` (amount received)
- Values >= 9,999,998 treated as missing-data codes and set to 0
- Negative values clipped to 0
- Aggregated to household level by summing across all member-dyads
- Log-1-plus transformation accommodates zero values

### D3. Co-Residence Extent

Continuous ratio of non-nuclear adult members to total household members:

```
D3_h = max(0, n_adults_h - 2) / n_members_h
```

**Operationalization:**
- "Non-nuclear" = adults beyond the conjugal pair (2 adults nominal baseline)
- Higher D3 indicates a more extended (multi-generational or multi-conjugal) structure
- Continuous rather than binary, capturing degree of extension
- Households with `n_members_h == 0` have D3 set to NaN

### D4 (Excluded from Main Analysis)

We initially constructed an educational-investment dimension as `n_enrolled_school_age / n_school_age` but excluded it because:
1. Introduces 47% sample loss (households without school-age members are missing)
2. Did not differentiate ethnic groups in pilot analysis (Mann-Whitney U p > 0.27 across all pairs)
3. Conceptually downstream of NELM strategy choice rather than parallel to it

D4 is retained as a control variable via `head_edu_lvl`.

## Auxiliary Controls

| Variable | Source | Coding |
|---|---|---|
| `head_age` | `bk_ar1.dta`, `ar09` for `ar02b == 1` | Years; values > 110 set to NaN |
| `head_female` | `bk_ar1.dta`, `ar07` | 1 if `ar07 == 3` (female), 0 otherwise |
| `head_edu_lvl` | `bk_ar1.dta`, `ar16` | Ordinal 1-15 |
| `urban` | `bk_sc1.dta`, `sc05` | 1 if `sc05 == 1` (urban), 0 otherwise |
| `n_members` | `bk_ar1.dta` aggregation | Count of distinct `pidlink` per `hhid14` |
| `log_exp_pcm` | `b1_ks1` + `b1_ks2` + `b1_ks3` | Monthly per-capita expenditure, log-transformed |
| `prov_bps` | `bk_sc1.dta`, `sc01_14_14` | BPS 2014 province code |
| `ethnic_modal` | `bk_ar1.dta`, `ar15d` | Modal ethnicity code per household |

## Expenditure Aggregation Detail

Per-capita monthly expenditure aggregates three IFLS-5 expenditure modules:

```
expenditure_monthly = (
    food_weekly_sum * 4.33                          # from b1_ks1
    + nonfood_monthly_sum                            # from b1_ks2
    + durables_annual_sum / 12                       # from b1_ks3
)
expenditure_pcm = expenditure_monthly / n_members_h
log_exp_pcm = log(1 + expenditure_pcm)
```

Values >= 9,999,998 (missing-data codes) are set to 0 before aggregation.

## Cluster Typology

### K-Means
- Algorithm: scikit-learn `KMeans` with `n_init=20`, `random_state=42`
- Input: standardized (z-score) 3-dim NELM matrix, n = 7,340 ethnic-modal households
- k tested: 2, 3, 4, 5, 6
- Optimal k by maximum silhouette coefficient (k = 5)

### Ward Hierarchical
- Algorithm: scikit-learn `AgglomerativeClustering(linkage="ward")`
- Same input matrix as k-means
- Cluster solution stability: Adjusted Rand Index (ARI) between methods at same k

## Multinomial Logistic Regression

- Implementation: `statsmodels.formula.mnlogit`
- Outcome: `kmeans_cluster` (5 levels at k = 5)
- Reference cluster: C4 (statsmodels mnlogit default = highest categorical level)
- Predictors: `C(ethnic)` (Java reference), `head_age`, `head_female`, `head_edu_lvl`, `urban`, `n_members`, `log_exp_pcm`, `C(prov_cat)` (top 6 provinces + Other)
- Estimation: Newton-Raphson, max iter = 200
- Inference: asymptotic Wald CIs + 200-replicate household-level bootstrap

### Likelihood-Ratio Test for Ethnicity

```
LR = 2 * (LL_full - LL_reduced)
df = df_model_full - df_model_reduced
p = 1 - chi2.cdf(LR, df)
```

Full model includes `C(ethnic)`; reduced model drops it. Significant LR rejects null that ethnicity adds no information beyond SES + geographic controls.

## Bootstrap CIs

200 replicates with household-level resampling (with replacement). Implementation in `joblib.Parallel` with 4 workers.

```python
for seed in range(200):
    boot_idx = np.random.choice(len(df), size=len(df), replace=True)
    df_boot = df.iloc[boot_idx]
    refit logit
    extract ethnic coefficients
```

Percentile 95% CI: 2.5th and 97.5th percentile of bootstrap distribution per coefficient.

## Sub-Ethnic Batak Classification

Religion + residence-based proxy (after kabupaten-of-birth-based attempt yielded zero classification, see Section 4.5 of paper):

- **Diaspora-Christian-Batak:** modal household religion `ar15` in {2, 3} (Catholic, Protestant)
- **Diaspora-Muslim-Batak:** modal household religion `ar15` == 1 (Islam)

The "diaspora" qualifier reflects that 647 of 647 IFLS-5 Batak ethnic-modal households reside outside their ancestral kabupaten (Tapanuli Utara, Toba, Karo, etc.) because IFLS-5's sample frame does not include those kabupaten as primary sampling units.

## Sensitivity Subsamples

Eight subsamples test ethnic effect robustness:
1. Urban-only (`urban == 1`)
2. Rural-only (`urban == 0`)
3-5. Low / Mid / High log-expenditure tertiles
6. Drop DKI Jakarta (`prov_bps != 31`)
7. Sumatra-only (`prov_bps in [11, 12, ..., 21]`)
8. Homogeneous-ethnic (`ethnic_share_modal >= 0.95`)

Plus diagnostic: Sumatran-homeland-only (Minang in Sumbar, Batak in Sumut, plus 1500 Java).

## Software Versions Used in Paper

```
pandas 2.1.4
numpy 1.26.2
scipy 1.11.4
scikit-learn 1.3.2
statsmodels 0.14.1
pyreadstat 1.2.5
matplotlib 3.8.2
joblib 1.3.2
```

Replication should produce identical results within floating-point precision when using these or compatible versions.

## Random Seed Convention

All scripts set `random_state=42` or `np.random.seed(seed_idx)` for reproducible cluster assignments and bootstrap resampling. Re-runs produce byte-identical CSV outputs given the same input.
