# Recognition of Eating Episodes via Commercial Smartwatch Sensors Analysis

[![DOI](https://zenodo.org/badge/1212836200.svg)](https://doi.org/10.5281/zenodo.19615761)

Companion repository for the manuscript submitted to *PLOS Digital Health* (PDIG-D-26-00312).

This repository contains the derived datasets, per-subject results, bootstrap confidence intervals, analysis scripts, and supplementary figures needed to reproduce all tables, figures and statistics reported in the paper. The raw 5 Hz sensor streams and video-derived eating labels are publicly available through the [`denotion`](https://github.com/UBESP-DCTV/denotion) R package.

---

## Study overview

Twenty healthy adults (9 women, 11 men, 20--30 years) wore Garmin Fenix 5 smartwatches on both wrists during four standardised meals in a semi-naturalistic university cafeteria. Raw inertial data were recorded at 5 Hz and two independent raters labelled eating movements from video. The analysis classifies 5-second Information Units (IUs) as *eating* or *non-eating* using Leave-One-Subject-Out (LOSO) cross-validation on the 19 right-handed participants (subject 02, left-handed, excluded).

Eleven classifiers were evaluated: Logistic Regression, Decision Tree, Random Forest (default/tuned), XGBoost (default/tuned), LightGBM (default/tuned), Attention pooling + MLP (default/tuned), and a Transformer encoder.

## Key results

| Classifier | Sensitivity | Specificity | Balanced accuracy | AUC |
|---|---|---|---|---|
| XGBoost (tuned) | 0.593 [0.512, 0.677] | 0.692 [0.624, 0.756] | 0.642 [0.612, 0.671] | 0.712 [0.671, 0.749] |
| Transformer (threshold = 0.30) | 0.691 [0.637, 0.735] | 0.504 [0.456, 0.549] | 0.597 [0.570, 0.624] | 0.646 [0.613, 0.679] |
| Logistic Regression | 0.625 [0.547, 0.707] | 0.643 [0.584, 0.705] | 0.634 [0.609, 0.659] | 0.697 [0.665, 0.728] |

95% CIs by subject-level cluster bootstrap (B = 1000, seed = 1812) over 19 LOSO folds. Full results in `results/bootstrap/bootstrap_ci.csv`.

---

## Repository structure

```
recognition_of_eating_episodes/
|
+-- data/                           Data derivation
|   +-- 00_derive_data.R            Derives all datasets from the public denotion package
|   +-- feature_list.csv            75 predictor names fed to ML classifiers (Table 2)
|   |
|   | After running 00_derive_data.R, the following files are generated:
|   +-- dat_all.rds                 Main IU feature matrix (delta_s = 1-5 s, 19 subjects)
|   +-- iu_features_unfiltered.rds  IU features without upstream filter (S7 Table ablation)
|   +-- iu_features_filtered.rds    IU features with Butterworth 0.3 Hz filter (S7 Table)
|
+-- analysis/                       Main analysis pipeline (R scripts, numbered)
|   +-- 00_eda.R                    Exploratory data analysis
|   +-- 01_baseline_models.R        Logistic Regression, Decision Tree
|   +-- 02_random_forest.R          Random Forest with class weights
|   +-- 02b_xgboost_gbm.R          XGBoost and LightGBM with class weights
|   +-- 02c_timeframe_analysis.R    Comparison across delta_s = 1-5 s
|   +-- 03_loso_validation.R        LOSO cross-validation (single model)
|   +-- 03b_loso_all_models.R       LOSO for RF, XGB, LGB (default + tuned)
|   +-- 04_interpretability.R       Feature importance (permutation, SHAP)
|   +-- 05_deep_subject_analysis.R  Per-subject performance breakdown
|   +-- 05b_error_analysis.R        False-negative and distribution-overlap analysis
|   +-- 06_fewshot_calibration.R    Few-shot personalisation experiment
|   +-- 07_outlier_analysis.R       Outlier subject identification
|   +-- 09_hyperparameter_tuning.R  Latin-hypercube tuning with racing
|   +-- 09_hyperparameter_tuning_optimized.R  Optimized tuning pipeline
|   +-- 11_paper_figures.R          Generation of manuscript figures
|   +-- timeseries/                 Deep learning / time-series classifiers (Python)
|   |   +-- minirocket_loso.R       MiniRocket (R, tsfeatures)
|   |   +-- cnn1d_loso.py           1D-CNN (PyTorch)
|   |   +-- contrastive_loso.py     MLP with learned embeddings (PyTorch)
|   |   +-- transformer_loso.py     Transformer encoder (PyTorch)
|   |   +-- cnn_lstm_loso.py        CNN+LSTM Kyritsis-style (PyTorch)
|   +-- optimization/               Threshold / ensemble / tuning (Python)
|       +-- 10a_attention_tuning.py Attention pooling hyperparameter search
|       +-- 10b_transformer_gpu.py  Transformer architecture search
|       +-- 10c_threshold_optimization.py  Decision-threshold grid search
|       +-- 10cd_ensemble_full.py   Weighted ensemble of Attention + Transformer
|
+-- revision/                       Scripts added during peer review
|   +-- bootstrap_ci.R              Subject-level cluster bootstrap (B1)
|   +-- paired_bootstrap_B5.R       Paired between-classifier comparisons (B5)
|   +-- loso_lr_dt_B5.R             LOSO re-evaluation of LR and DT (B5)
|   +-- lgb_tuned_per_subject.R     LightGBM tuned per-subject metrics (B5)
|   +-- loso_timeframe_B4.R         Window-size sensitivity under LOSO (B4)
|   +-- agreement_B6.R              Inter-rater reliability analysis (B6)
|   +-- xgb_tuned_EI_B6.R           XGBoost tuned on Eating Intersect target (B6)
|   +-- fp_pattern_C1_v2.R          False-positive pattern analysis (C1)
|   +-- subgroup_arm_C2.R           Menu A vs Menu B robustness check (C2)
|   +-- mde_C3.R                    Minimum detectable effect analysis (C3)
|   +-- inference_latency_C5.R      Inference latency benchmark (C5)
|   +-- feature_eng_C10.R           Feature engineering for filter ablation (C10)
|   +-- loso_C10.R                  LOSO for filter-vs-no-filter comparison (C10)
|
+-- results/                        All numerical outputs (CSV / RDS)
|   +-- per_subject/                Per-subject LOSO metrics for each classifier
|   |   +-- per_subject_metrics.csv     RF, XGB, LGB (default + tuned)
|   |   +-- per_subject_lr_dt.csv       Logistic Regression, Decision Tree
|   |   +-- per_subject_lgb_tuned.csv   LightGBM tuned
|   |   +-- per_subject_attention_default.csv  Attention pooling (default)
|   |   +-- per_subject_xgb_tuned_EI.csv      XGBoost tuned on EI target
|   |   +-- per_subject_C10.csv         Filter vs no-filter per subject
|   |   +-- deep_per_subject.csv        Attention (tuned) + Transformer
|   +-- bootstrap/                  Bootstrap CI and paired comparisons
|   |   +-- bootstrap_ci.csv           95% CIs for all 11 classifiers (Tables 2-3)
|   |   +-- paired_bootstrap_all_pairs.csv  11x11 pairwise comparisons (S5 Table)
|   |   +-- xgb_tuned_EI_ci.csv        CIs for Eating Intersect target (S6 Table)
|   |   +-- ci_C10.csv                 CIs for filter ablation (S7 Table)
|   |   +-- paired_C10.csv             Paired filter vs no-filter (S7 Table)
|   |   +-- timeframe_ci.csv           CIs across delta_s = 1-5 (S3 Table)
|   |   +-- mde_C3.csv                 Minimum detectable effect at N = 19-200
|   +-- agreement/                  Inter-rater reliability
|   |   +-- per_subject_agreement.csv  Per-subject percent agreement and kappa
|   |   +-- agreement_summary.csv      Overall agreement statistics
|   +-- subgroup_arm_C2.csv         Menu A vs B stratified metrics
|   +-- subgroup_arm_diff_C2.csv    Between-menu bootstrap differences
|   +-- fp_pattern_C1_per_subject.csv  Per-subject false-positive analysis
|   +-- fp_pattern_C1_summary.csv   Overall FP boundary vs isolated
|   +-- inference_latency_C5.csv    XGBoost inference timing benchmark
|   +-- timeframe_per_subject.csv   Per-subject metrics across delta_s
|   +-- timeframe_summary.csv       Aggregated timeframe comparison
|   +-- feature_counts.csv          Feature counts by category
|   +-- fewshot_results.rds         Few-shot personalisation (all subjects x strategies)
|
+-- figures/
    +-- S1_Fig_per_subject_sensitivity.tiff  Forest plot (S1 Fig in the paper)
```

## How to reproduce

### Prerequisites

**R** (>= 4.4) with packages:

```r
install.packages(c("tidymodels", "tidyverse", "ranger", "xgboost", "lightgbm",
                    "yardstick", "finetune", "probably", "vip"))
remotes::install_github("UBESP-DCTV/denotion")
```

**Python** (>= 3.10) with packages (for deep-learning scripts only):

```bash
pip install torch numpy pandas scikit-learn
```

### Reproduction steps

1. **Derive datasets from the denotion package** (required first):
   ```r
   source("data/00_derive_data.R")
   # Produces: data/dat_all.rds, data/iu_features_*.rds, data/feature_list.csv
   # Runtime: ~5 minutes. Requires: denotion, tidyverse, signal packages.
   ```

2. **Main analysis pipeline** (run in order):
   ```r
   # From the repository root:
   source("analysis/00_eda.R")
   source("analysis/03b_loso_all_models.R")
   source("analysis/09_hyperparameter_tuning_optimized.R")
   source("analysis/06_fewshot_calibration.R")
   ```

3. **Deep learning models** (Python, GPU recommended):
   ```bash
   python analysis/timeseries/transformer_loso.py
   python analysis/optimization/10a_attention_tuning.py
   python analysis/optimization/10b_transformer_gpu.py
   ```

4. **Revision analyses** (run after steps 2-3):
   ```r
   source("revision/bootstrap_ci.R")         # Tables 2-3 CIs
   source("revision/paired_bootstrap_B5.R")  # S5 Table
   source("revision/agreement_B6.R")         # S6 Table
   source("revision/feature_eng_C10.R")      # S7 Table feature engineering
   source("revision/loso_C10.R")             # S7 Table LOSO comparison
   ```

### Seeds and reproducibility

| Operation | Seed | Set in |
|---|---|---|
| Cluster bootstrap (CIs, paired tests) | 1812 | `revision/bootstrap_ci.R`, `revision/paired_bootstrap_B5.R` |
| Few-shot personalisation | 42 | `analysis/06_fewshot_calibration.R` |
| XGBoost training | fixed per script | each `analysis/*.R` script header |
| PyTorch models | 42 | each `analysis/timeseries/*.py` |

## Data description

### Source data (not in this repository)

The raw 5 Hz inertial sensor streams and video-derived eating labels are available in the [`denotion`](https://github.com/UBESP-DCTV/denotion) R package maintained by the Unit of Biostatistics, Epidemiology and Public Health (UBEP) of the University of Padova.

```r
remotes::install_github("UBESP-DCTV/denotion")
data("denotion", package = "denotion")       # nested raw time-series per window
data("denotion_beta", package = "denotion")   # slope coefficients per IU
```

The `denotion` dataset contains the raw 5 Hz samples (acc_x/y/z, pitch, roll, power, total_energy) nested by subject, meal, and window size, with frame-level eating_union and eating_intersect labels from two independent video raters.

### Derived data (this repository)

All derived datasets are generated by `data/00_derive_data.R` from the public [`denotion`](https://github.com/UBESP-DCTV/denotion) package. No pre-computed data files need to be downloaded.

**`data/dat_all.rds`** (generated): The main Information Unit (IU) feature matrix used for all machine-learning classifiers (Table 2). Contains ~132,000 IUs across delta_s = 1--5 s (19 subjects, ~26,000 per delta_s) with 7 slope features, 49 right-wrist window statistics (7 statistics x 7 variables), 3 axis cross-correlations, categorical covariates (arm, meal, food, subject), and both EU and EI targets. Most analyses filter to `delta_s == 5`.

**`data/iu_features_unfiltered.rds`** and **`data/iu_features_filtered.rds`** (generated): Feature matrices for the filter-vs-no-filter ablation (S7 Table). The unfiltered version matches the main pipeline; the filtered version applies a 2nd-order zero-phase Butterworth 0.3 Hz low-pass filter upstream of the windowing.

**`data/feature_list.csv`**: The 75 predictor names (after preprocessing) fed to the ML classifiers, grouped by category.

## Mapping to manuscript tables and figures

| Paper element | Source file(s) | Output file(s) |
|---|---|---|
| Table 2 (ML classifiers, LOSO + CIs) | `revision/bootstrap_ci.R` | `results/bootstrap/bootstrap_ci.csv` |
| Table 3 (Deep learning, LOSO + CIs) | `revision/bootstrap_ci.R` | `results/bootstrap/bootstrap_ci.csv` |
| S1 Table (per-subject IU counts) | `revision/bootstrap_ci.R` | `results/per_subject/per_subject_metrics.csv` |
| S2 Table (architectures) | described in text | -- |
| S3 Table (window-size sensitivity) | `revision/loso_timeframe_B4.R` | `results/bootstrap/timeframe_ci.csv` |
| S4 Table (feature list) | preprocessing recipe | `data/feature_list.csv` |
| S5 Table (paired comparisons) | `revision/paired_bootstrap_B5.R` | `results/bootstrap/paired_bootstrap_all_pairs.csv` |
| S6 Table (reliability + EI) | `revision/agreement_B6.R`, `revision/xgb_tuned_EI_B6.R` | `results/agreement/`, `results/bootstrap/xgb_tuned_EI_ci.csv` |
| S7 Table (filter ablation) | `revision/feature_eng_C10.R`, `revision/loso_C10.R` | `results/bootstrap/ci_C10.csv`, `results/bootstrap/paired_C10.csv` |
| S1 Fig (per-subject sensitivity) | `analysis/11_paper_figures.R` | `figures/S1_Fig_per_subject_sensitivity.tiff` |
| Fig 1 (study flowchart) | -- | separate file (author-created) |
| Fig 2 (sensor axes) | -- | separate file (author photograph) |

## License

Code: MIT License. Data and figures: CC-BY 4.0.

## Citation

If you use this repository, please cite:

> Vedovelli L, Bhuyan MJ, Lanera C, Baldi I, Berchialla P, Gregori D, NOTION Working Group. Recognition of Eating Episodes via Commercial Smartwatch Sensors Analysis. *PLOS Digital Health*. 2026. [doi: pending]

## References

- Fuscà E, Bolzon A, Buratin A, et al. Measuring Caloric Intake at the Population Level (NOTION): Protocol for an Experimental Study. *JMIR Res Protoc*. 2019;8(3):e12116. doi:10.2196/12116
- Baldi I, Lanera C, Bhuyan MJ, et al. Classifying Food Items During an Eating Occasion. *Foods*. 2025;14(2):276. doi:10.3390/foods14020276
