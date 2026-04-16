# 09_hyperparameter_tuning.R - Tuning RF, XGBoost, LightGBM con LOSO
# Fase 9: Hyperparameter tuning per migliorare baseline

library(tidyverse)
library(tidymodels)
library(xgboost)
library(ranger)
library(bonsai)
library(lightgbm)
library(here)

set.seed(2026)

cat("=== FASE 9: HYPERPARAMETER TUNING (LOSO) ===\n\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating")))

cat("Dataset: ", nrow(dat), "obs,", n_distinct(dat$subject), "soggetti\n")
cat("Eating:", sum(dat$EU == "eating"), "| Non-eating:", sum(dat$EU == "non_eating"), "\n\n")

# Class weight
class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
cat("Scale pos weight:", round(scale_pos_weight, 2), "\n\n")

# =============================================================================
# 2. RECIPE
# =============================================================================

rec <- recipe(EU ~ ., data = dat) |>
  step_rm(subject, ID, EI) |>
  step_mutate(across(c(arm, meal, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# =============================================================================
# 3. LOSO FOLDS
# =============================================================================

loso_folds <- group_vfold_cv(dat, group = subject)
cat("LOSO folds:", nrow(loso_folds), "\n\n")

# =============================================================================
# 4. MODEL SPECIFICATIONS CON TUNING
# =============================================================================

# Random Forest
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) |>
  set_engine("ranger", class.weights = c("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

# XGBoost
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) |>
  set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
  set_mode("classification")

# LightGBM
lgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) |>
  set_engine("lightgbm", class_weight = "balanced", verbose = -1) |>
  set_mode("classification")

# =============================================================================
# 5. TUNING GRIDS
# =============================================================================

rf_grid <- grid_regular(
  mtry(range = c(5, 30)),
  min_n(range = c(5, 20)),
  levels = 4
)

xgb_grid <- grid_regular(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),  # 0.01 to 0.1
  min_n(range = c(5, 15)),
  levels = 3
)

lgb_grid <- grid_regular(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),
  min_n(range = c(5, 15)),
  levels = 3
)

metrics <- metric_set(sens, spec, roc_auc)

# =============================================================================
# 6. TUNING RANDOM FOREST
# =============================================================================

cat("=== TUNING RANDOM FOREST ===\n")

rf_wf <- workflow() |> add_recipe(rec) |> add_model(rf_spec)

rf_tune <- tune_grid(
  rf_wf,
  resamples = loso_folds,
  grid = rf_grid,
  metrics = metrics,
  control = control_grid(verbose = TRUE, save_pred = FALSE)
)

rf_best <- select_best(rf_tune, metric = "sens")
cat("\nBest RF params:\n")
print(rf_best)

rf_best_metrics <- rf_tune |>
  collect_metrics() |>
  filter(mtry == rf_best$mtry, min_n == rf_best$min_n)
cat("\nRF best metrics:\n")
print(rf_best_metrics)

# =============================================================================
# 7. TUNING XGBOOST
# =============================================================================

cat("\n\n=== TUNING XGBOOST ===\n")

xgb_wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)

xgb_tune <- tune_grid(
  xgb_wf,
  resamples = loso_folds,
  grid = xgb_grid,
  metrics = metrics,
  control = control_grid(verbose = TRUE, save_pred = FALSE)
)

xgb_best <- select_best(xgb_tune, metric = "sens")
cat("\nBest XGB params:\n")
print(xgb_best)

xgb_best_metrics <- xgb_tune |>
  collect_metrics() |>
  filter(
    trees == xgb_best$trees,
    tree_depth == xgb_best$tree_depth,
    learn_rate == xgb_best$learn_rate,
    min_n == xgb_best$min_n
  )
cat("\nXGB best metrics:\n")
print(xgb_best_metrics)

# =============================================================================
# 8. TUNING LIGHTGBM
# =============================================================================

cat("\n\n=== TUNING LIGHTGBM ===\n")

lgb_wf <- workflow() |> add_recipe(rec) |> add_model(lgb_spec)

lgb_tune <- tune_grid(
  lgb_wf,
  resamples = loso_folds,
  grid = lgb_grid,
  metrics = metrics,
  control = control_grid(verbose = TRUE, save_pred = FALSE)
)

lgb_best <- select_best(lgb_tune, metric = "sens")
cat("\nBest LGB params:\n")
print(lgb_best)

lgb_best_metrics <- lgb_tune |>
  collect_metrics() |>
  filter(
    trees == lgb_best$trees,
    tree_depth == lgb_best$tree_depth,
    learn_rate == lgb_best$learn_rate,
    min_n == lgb_best$min_n
  )
cat("\nLGB best metrics:\n")
print(lgb_best_metrics)

# =============================================================================
# 9. CONFRONTO FINALE
# =============================================================================

cat("\n\n========================================\n")
cat("=== CONFRONTO TUNED VS DEFAULT ===\n")
cat("========================================\n\n")

# Estrai metriche migliori
get_best_sens <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "sens") |>
    slice_max(mean, n = 1) |>
    pull(mean)
}

get_best_spec <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "spec") |>
    slice_max(mean, n = 1) |>
    pull(mean)
}

get_best_auc <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "roc_auc") |>
    slice_max(mean, n = 1) |>
    pull(mean)
}

comparison <- tibble(
  Model = c("RF default", "RF tuned", "XGB default", "XGB tuned", "LGB default", "LGB tuned", "Attention"),
  Sensitivity = c(
    0.224,  # RF default from LOSO
    get_best_sens(rf_tune),
    0.417,  # XGB default from LOSO
    get_best_sens(xgb_tune),
    0.295,  # LGB default from LOSO
    get_best_sens(lgb_tune),
    0.498   # Attention
  ),
  Specificity = c(
    0.931,
    get_best_spec(rf_tune),
    0.814,
    get_best_spec(xgb_tune),
    0.884,
    get_best_spec(lgb_tune),
    0.709
  ),
  AUC = c(
    0.700,
    get_best_auc(rf_tune),
    0.693,
    get_best_auc(xgb_tune),
    0.691,
    get_best_auc(lgb_tune),
    0.640
  )
)

print(comparison)

cat("\n\nMIGLIORAMENTO TUNING:\n")
cat("RF:  sens", round(0.224, 3), "->", round(get_best_sens(rf_tune), 3), 
    "(", round((get_best_sens(rf_tune) - 0.224) * 100, 1), "%)\n")
cat("XGB: sens", round(0.417, 3), "->", round(get_best_sens(xgb_tune), 3),
    "(", round((get_best_sens(xgb_tune) - 0.417) * 100, 1), "%)\n")
cat("LGB: sens", round(0.295, 3), "->", round(get_best_sens(lgb_tune), 3),
    "(", round((get_best_sens(lgb_tune) - 0.295) * 100, 1), "%)\n")

# =============================================================================
# 10. SAVE
# =============================================================================

saveRDS(list(
  rf_tune = rf_tune,
  rf_best = rf_best,
  xgb_tune = xgb_tune,
  xgb_best = xgb_best,
  lgb_tune = lgb_tune,
  lgb_best = lgb_best,
  comparison = comparison
), here("results", "hyperparameter_tuning.rds"))

cat("\nSaved to results/hyperparameter_tuning.rds\n")
