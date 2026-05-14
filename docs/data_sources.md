# Data Sources

This document describes how to acquire the raw data needed for full replication. The repository includes derived datasets (in `data/processed/`) sufficient to replicate Tables 1-5 and Figures 1-4 without re-running the cleaning pipeline. Re-running the cleaning pipeline from scratch requires raw IFLS-5 files.

## IFLS-5 (Required for Re-Cleaning Only)

The Indonesian Family Life Survey wave 5 (IFLS-5) is the primary data source. The paper uses the public-use release maintained by RAND Corporation.

### Step-by-Step Acquisition

1. **Visit the IFLS website:**
   https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS.html

2. **Register for free access:**
   Fill the data-use registration form at:
   https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS/datasets.html

   You will be asked to:
   - Provide name, institutional affiliation, and research purpose
   - Agree to the data-use terms (research-only, no redistribution, no attempt at re-identification)

3. **Download IFLS-5 STATA files (approximately 2 GB):**
   Once registered, download the IFLS-5 public-use dataset, specifically the household-level files (folder typically labeled "hh14" in the archive). The files relevant to this paper are:

   - `bk_ar1.dta` — household roster + ethnicity (ar15d) + religion (ar15)
   - `b3a_mg1.dta` — lifetime migration history
   - `b3b_tf.dta` — dyadic transfers
   - `b1_ks1.dta`, `b1_ks2.dta`, `b1_ks3.dta` — expenditure modules
   - `bk_sc1.dta` — geographic codes (province, kabupaten, urban/rural)
   - `b3a_dl1.dta` — language + self-reported ethnicity (used in sub-ethnic classification)

4. **Place files in:**
   ```
   data/raw/ifls5/hh14/
   ```
   (Create this folder locally. It is gitignored and will not be committed.)

5. **Verify integrity:**
   ```bash
   python3 scripts/99_tests/verify_all_provenance.py
   ```
   The script checks that downloaded files match the SHA-256 checksums recorded in our project provenance (private; if the verification fails, the files may have been re-released; see paper Methods Section for current checksum protocol).

### Citation for IFLS-5

If you use IFLS-5 raw data:

```
Strauss, J., Witoelar, F., & Sikoki, B. (2016). The Fifth Wave of the Indonesia
Family Life Survey: Overview and Field Report. RAND Working Paper WR-1143/1-NIA/NICHD.
Santa Monica: RAND Corporation. https://www.rand.org/pubs/working_papers/WR1143z1.html
```

## Other Data (Already Public)

The following data sources are mentioned in the paper but not used directly in the analytical pipeline; we cite them for context:

- **BNPB Sumatra Floods 2025-2026:** Press releases available at https://bnpb.go.id (Indonesian National Disaster Management Agency)
- **BPS Sensus 2022:** Macro-statistics at https://sensus.bps.go.id (Statistics Indonesia)

No raw files from BNPB or BPS are required to replicate the paper's quantitative analysis. They are referenced in Sections 1.1 and 5.4 for substantive context only.

## Data NOT Included in This Package

Per RAND's data-use license, the following IFLS-5 raw files are NOT included in this repository and must be obtained via the RAND registration process above:

- All `.dta` STATA files
- All raw codebooks and questionnaire files

This package only redistributes:
- Aggregated household-level derived CSVs (in `data/processed/`)
- Analysis scripts (in `scripts/`)
- Output tables and figures (in `outputs/`)
- Documentation (this directory)

All redistributed materials comply with RAND's policy on derived-data sharing for non-commercial research replication.

## Data Citation in Paper

In the paper, IFLS-5 is cited as "Strauss, Witoelar, & Sikoki (2016)" with the RAND Working Paper URL. The derived datasets and code in this replication package should be cited via the Zenodo DOI of this release.
