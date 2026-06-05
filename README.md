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

The headline performance quoted in the text is the **sensor-only** XGBoost (the 59 motion-derived predictors, with the study-arm, meal and food variables removed, as an autonomous watch would have no menu information at prediction time): balanced accuracy 0.639 [0.607, 0.668], AUC 0.699 [0.653, 0.743], sensitivity 0.594 [0.528, 0.663], specificity 0.683 [0.630, 0.732]. A fully nested subject-level cross-validation confirms the post-selection estimates above (the nested and post-selection balanced accuracies differ by at most 0.005 for the tuned tree models), and the Transformer sensitivity advantage over tuned XGBoost is not statistically significant once the decision threshold is selected by nested cross-validation and the comparisons are adjusted for multiplicity. These analyses live in `results/bootstrap/` (`nested_loso_*`, `deep_nested_*`, `lr_dt_sensor_only_*`, `pairwise_*`) and `results/agreement/kappa_bounds.csv`.

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
|   +-- iu_features_unfiltered.rds  IU features without upstream filter (S6 Table ablation)
|   +-- iu_features_filtered.rds    IU features with Butterworth 0.3 Hz filter (S6 Table)
|
+-- analysis/                       Analysis scripts (R and Python)
|   |  Modelling pipeline (run in numbered order):
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
|   +-- 06_fewshot_calibration.R    Meal-blocked few-shot personalisation (S2 Appendix Table S2.5)
|   +-- 07_outlier_analysis.R       Outlier subject identification
|   +-- 09_hyperparameter_tuning.R  Latin-hypercube tuning with racing
|   +-- 09_hyperparameter_tuning_optimized.R  Optimized tuning pipeline
|   +-- 11_paper_figures.R          Generation of manuscript figures
|   |
|   |  Confidence intervals, comparisons and supplementary analyses:
|   +-- bootstrap_ci.R              Subject-level cluster bootstrap CIs (Tables 2-3, S1 Table)
|   +-- loso_lr_dt.R                LOSO Logistic Regression and Decision Tree (Table 2)
|   +-- lgb_tuned_per_subject.R     LightGBM tuned per-subject metrics (Table 2)
|   +-- loso_timeframe.R            Window-size sensitivity under LOSO (S3 Table)
|   +-- xgb_tuned_EI.R              XGBoost tuned on the Eating Intersect target (S5 Table, Panel B)
|   +-- kappa_bounds.R              Inter-rater agreement: exact percent agreement + kappa interval (S5 Table Panel A; S2 Appendix Table S2.4)
|   +-- feature_eng_filter.R        Feature engineering for the filter ablation (S6 Table)
|   +-- loso_filter.R               LOSO filter-vs-no-filter comparison (S6 Table)
|   +-- fp_pattern.R                False-positive pattern analysis (Discussion)
|   +-- subgroup_arm.R              Menu A vs Menu B robustness check (Section 3.6)
|   +-- mde.R                       Minimum-detectable-effect analysis (Limitations)
|   +-- inference_latency.R         Inference-latency benchmark (Discussion)
|   |
|   |  Subject-independent re-analysis (S2 Appendix):
|   +-- nested_loso.R               Fully nested LOSO, full and sensor-only feature sets (S2 Appendix Tables S2.1, S2.2)
|   +-- lr_dt_sensor_only.R         LR/DT on full and sensor-only feature sets (S2 Appendix Table S2.2)
|   +-- deep_nested_threshold.py    Transformer nested-threshold selection (S2 Appendix Table S2.1)
|   +-- pairwise_multiplicity.R     Multiplicity-adjusted pairwise comparisons (S2 Appendix Table S2.3)
|   |
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
+-- results/                        All numerical outputs (CSV / RDS)
|   +-- per_subject/                Per-subject LOSO metrics for each classifier
|   |   +-- per_subject_metrics.csv     RF, XGB, LGB (default + tuned), LR, DT
|   |   +-- per_subject_lr_dt.csv       Logistic Regression, Decision Tree
|   |   +-- per_subject_lgb_tuned.csv   LightGBM tuned
|   |   +-- per_subject_attention_default.csv  Attention pooling (default)
|   |   +-- per_subject_xgb_tuned_EI.csv      XGBoost tuned on EI target
|   |   +-- per_subject_filter.csv      Filter vs no-filter per subject
|   |   +-- deep_per_subject.csv        Attention (tuned) + Transformer
|   +-- bootstrap/                  Bootstrap CIs and pairwise comparisons
|   |   +-- bootstrap_ci.csv            95% CIs for all 11 classifiers (Tables 2-3)
|   |   +-- xgb_tuned_EI_ci.csv         CIs for the Eating Intersect target (S5 Table)
|   |   +-- timeframe_ci.csv            CIs across delta_s = 1-5 (S3 Table)
|   |   +-- filter_ablation_ci.csv      CIs for the filter ablation (S6 Table)
|   |   +-- filter_ablation_paired.csv  Paired filter vs no-filter (S6 Table)
|   |   +-- mde.csv                     Minimum detectable effect at N = 19-200
|   |   +-- nested_loso_pretty.csv / _ci.csv / _per_subject.csv / .rds  Nested LOSO, tree models (S2 Appendix Tables S2.1, S2.2)
|   |   +-- deep_nested_pretty.csv / _ci.csv / _per_subject.csv  Transformer nested threshold + Attention default (S2 Appendix Table S2.1)
|   |   +-- lr_dt_sensor_only_pretty.csv / _ci.csv / _per_subject.csv  LR/DT full vs sensor-only (S2 Appendix Table S2.2)
|   |   +-- pairwise_nested_adjusted.csv      Full multiplicity-adjusted pairwise matrix (S2 Appendix Table S2.3)
|   |   +-- pairwise_vs_xgb_tuned.csv         Comparisons against tuned XGBoost (S2 Appendix Table S2.3)
|   +-- agreement/                  Inter-rater reliability
|   |   +-- per_subject_agreement.csv  Per-subject percent agreement and kappa
|   |   +-- kappa_bounds.csv           Exact percent agreement + kappa interval, IU and raw-frame levels (S5 Table; S2 Appendix Table S2.4)
|   +-- subgroup_arm.csv            Menu A vs B stratified metrics (Section 3.6)
|   +-- subgroup_arm_diff.csv       Between-menu bootstrap differences (Section 3.6)
|   +-- fp_pattern_per_subject.csv  Per-subject false-positive analysis (Discussion)
|   +-- fp_pattern_summary.csv      Overall FP boundary vs isolated (Discussion)
|   +-- inference_latency.csv       XGBoost inference-timing benchmark (Discussion)
|   +-- timeframe_per_subject.csv   Per-subject metrics across delta_s
|   +-- timeframe_summary.csv       Aggregated timeframe comparison
|   +-- feature_counts.csv          Feature counts by category
|   +-- fewshot_blocked_summary.csv Meal-blocked few-shot, by group and number of shots (S2 Appendix Table S2.5)
|   +-- fewshot_blocked_per_subject.csv  Meal-blocked few-shot, per subject (S2 Appendix Table S2.5)
|   +-- predictions.npz             Deep-model per-fold probabilities (input to deep_nested_threshold.py)
|
+-- figures/
    +-- S1_Fig_per_subject_sensitivity.tiff  Forest plot (S1 Fig in the paper)
```

## How to reproduce

### Prerequisites

**R** (>= 4.4) with packages:

```r
install.packages(c("tidymodels", "tidyverse", "ranger", "xgboost", "lightgbm",
                    "bonsai", "rpart", "yardstick", "finetune", "probably",
                    "vip", "here", "testthat"))
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

2. **Modelling pipeline** (run in order):
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

4. **Confidence intervals and supplementary analyses**:
   ```r
   source("analysis/bootstrap_ci.R")        # Tables 2-3 CIs (results/bootstrap/bootstrap_ci.csv)
   source("analysis/loso_timeframe.R")      # S3 Table
   source("analysis/xgb_tuned_EI.R")        # S5 Table, Panel B
   source("analysis/feature_eng_filter.R")  # S6 Table feature engineering
   source("analysis/loso_filter.R")         # S6 Table LOSO comparison
   ```

5. **Subject-independent re-analysis (S2 Appendix)**:
   ```r
   source("analysis/nested_loso.R")         # Nested LOSO, full and sensor-only (Tables S2.1, S2.2)
   source("analysis/lr_dt_sensor_only.R")   # LR/DT full vs sensor-only (Table S2.2)
   source("analysis/kappa_bounds.R")        # Inter-rater agreement bounds (Table S2.4)
   source("analysis/pairwise_multiplicity.R")  # Multiplicity-adjusted comparisons (Table S2.3)
   ```
   ```bash
   python analysis/deep_nested_threshold.py # Transformer nested threshold (Table S2.1)
   ```

### Seeds and reproducibility

| Operation | Seed | Set in |
|---|---|---|
| Cluster bootstrap (CIs, paired tests) | 1812 | `analysis/bootstrap_ci.R`, `analysis/pairwise_multiplicity.R` |
| Few-shot personalisation | 2026 | `analysis/06_fewshot_calibration.R` |
| Nested LOSO inner selection | 2026 (+ fold index) | `analysis/nested_loso.R` |
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

The `denotion` dataset contains the raw 5 Hz samples (acc_x/y/z, pitch, roll, power, total_energy) nested by subject, meal, and window size, with frame-level eating_union and eating_intersect labels from two independent video raters. The two raters' individual label streams were not retained when these public labels were constructed: only the eating-union (labelled eating by at least one rater) and eating-intersect (labelled eating by both) summaries are available. `analysis/kappa_bounds.R` reads this dataset directly from the package to derive the raw-frame agreement.

### Derived data (this repository)

All derived datasets are generated by `data/00_derive_data.R` from the public [`denotion`](https://github.com/UBESP-DCTV/denotion) package. No pre-computed data files need to be downloaded.

**`data/dat_all.rds`** (generated): The main Information Unit (IU) feature matrix used for all machine-learning classifiers (Table 2). Contains ~132,000 IUs across delta_s = 1--5 s (19 subjects, ~26,000 per delta_s) with 7 slope features, 49 right-wrist window statistics (7 statistics x 7 variables), 3 axis cross-correlations, categorical covariates (arm, meal, food, subject), and both EU and EI targets. Most analyses filter to `delta_s == 5`.

**`data/iu_features_unfiltered.rds`** and **`data/iu_features_filtered.rds`** (generated): Feature matrices for the filter-vs-no-filter ablation (S6 Table). The unfiltered version matches the main pipeline; the filtered version applies a 2nd-order zero-phase Butterworth 0.3 Hz low-pass filter upstream of the windowing.

**`data/feature_list.csv`**: The 75 predictor names (after preprocessing) fed to the ML classifiers, grouped by category.

**`results/predictions.npz`**: Per-fold predicted probabilities of the deep-learning models (produced by the time-series and optimization scripts), used by `analysis/deep_nested_threshold.py` to select the Transformer decision threshold under nested cross-validation.

## Mapping to manuscript tables and figures

| Paper element | Source file(s) | Output file(s) |
|---|---|---|
| Table 2 (ML classifiers, LOSO + CIs) | `analysis/bootstrap_ci.R` | `results/bootstrap/bootstrap_ci.csv` |
| Table 3 (Deep learning, LOSO + CIs) | `analysis/bootstrap_ci.R` | `results/bootstrap/bootstrap_ci.csv` |
| S1 Table (per-subject IU counts) | `analysis/bootstrap_ci.R` | `results/per_subject/per_subject_metrics.csv` |
| S2 Table (architectures) | described in text | -- |
| S3 Table (window-size sensitivity) | `analysis/loso_timeframe.R` | `results/bootstrap/timeframe_ci.csv` |
| S4 Table (feature list) | preprocessing recipe | `data/feature_list.csv` |
| S5 Table (inter-rater reliability + EI) | `analysis/kappa_bounds.R` (Panel A), `analysis/xgb_tuned_EI.R` (Panel B) | `results/agreement/`, `results/bootstrap/xgb_tuned_EI_ci.csv` |
| S6 Table (filter ablation) | `analysis/feature_eng_filter.R`, `analysis/loso_filter.R` | `results/bootstrap/filter_ablation_ci.csv`, `filter_ablation_paired.csv` |
| S2 Appendix Table S2.1 (nested LOSO) | `analysis/nested_loso.R`, `analysis/deep_nested_threshold.py` | `results/bootstrap/nested_loso_*.csv`, `deep_nested_*.csv` |
| S2 Appendix Table S2.2 (sensor-only ablation) | `analysis/nested_loso.R`, `analysis/lr_dt_sensor_only.R` | `results/bootstrap/nested_loso_*.csv`, `lr_dt_sensor_only_*.csv` |
| S2 Appendix Table S2.3 (multiplicity-adjusted comparisons) | `analysis/pairwise_multiplicity.R` | `results/bootstrap/pairwise_nested_adjusted.csv`, `pairwise_vs_xgb_tuned.csv` |
| S2 Appendix Table S2.4 (inter-rater bounds) | `analysis/kappa_bounds.R` | `results/agreement/kappa_bounds.csv` |
| S2 Appendix Table S2.5 (meal-blocked few-shot) | `analysis/06_fewshot_calibration.R` | `results/fewshot_blocked_summary.csv`, `fewshot_blocked_per_subject.csv` |
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
