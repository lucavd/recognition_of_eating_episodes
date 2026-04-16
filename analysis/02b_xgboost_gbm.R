# 02b_xgboost_gbm.R - XGBoost e LightGBM con Class Weights
# Fase 2b: Confronto boosting models

library(tidyverse)
library(tidymodels)
library(xgboost)
library(bonsai)
library(lightgbm)
library(here)

set.seed(2026)

# =============================================================================
# LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI)

cat("Dataset:", nrow(dat), "obs,", ncol(dat), "vars\n")

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
cat("scale_pos_weight:", scale_pos_weight, "\n")

# =============================================================================
# RECIPE
# =============================================================================

rec <- recipe(EU ~ ., data = dat) |>
  step_mutate(across(c(arm, meal, subject, delta_s, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# =============================================================================
# TRAIN/TEST SPLIT + CV FOLDS
# =============================================================================

split <- initial_split(dat, prop = 0.8, strata = EU)
train <- training(split)
test <- testing(split)

folds <- vfold_cv(train, v = 10, strata = EU)

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# =============================================================================
# METRICS
# =============================================================================

metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

# =============================================================================
# XGBOOST CON SCALE_POS_WEIGHT
# =============================================================================

cat("\n=== XGBOOST (scale_pos_weight) ===\n")

xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
  set_mode("classification")

xgb_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(xgb_spec)

cat("Running 10-fold CV...\n")
xgb_cv <- fit_resamples(xgb_wf, folds, metrics = metrics)

xgb_cv_metrics <- collect_metrics(xgb_cv)
cat("\nCV Results:\n")
print(xgb_cv_metrics)

cat("\nFitting final model...\n")
xgb_fit <- last_fit(xgb_wf, split, metrics = metrics)
xgb_test_metrics <- collect_metrics(xgb_fit)
cat("\nTest Results:\n")
print(xgb_test_metrics)

# =============================================================================
# LIGHTGBM CON CLASS_WEIGHT
# =============================================================================

cat("\n=== LIGHTGBM (class_weight) ===\n")

lgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("lightgbm", 
             class_weight = list("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

lgb_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(lgb_spec)

cat("Running 10-fold CV...\n")
lgb_cv <- fit_resamples(lgb_wf, folds, metrics = metrics)

lgb_cv_metrics <- collect_metrics(lgb_cv)
cat("\nCV Results:\n")
print(lgb_cv_metrics)

cat("\nFitting final model...\n")
lgb_fit <- last_fit(lgb_wf, split, metrics = metrics)
lgb_test_metrics <- collect_metrics(lgb_fit)
cat("\nTest Results:\n")
print(lgb_test_metrics)

# =============================================================================
# COMPARISON
# =============================================================================

cat("\n=== COMPARISON ===\n")

rf_results <- readRDS(here("results", "rf_weighted_results.rds"))

xgb_summary <- xgb_test_metrics |>
  mutate(model = "XGBoost (weighted)", set = "Test") |>
  select(model, set, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

lgb_summary <- lgb_test_metrics |>
  mutate(model = "LightGBM (weighted)", set = "Test") |>
  select(model, set, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

comparison <- bind_rows(
  rf_results$comparison,
  xgb_summary,
  lgb_summary
)

print(comparison)

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  xgb_cv = xgb_cv_metrics,
  xgb_test = xgb_test_metrics,
  lgb_cv = lgb_cv_metrics,
  lgb_test = lgb_test_metrics,
  comparison = comparison
), here("results", "boosting_results.rds"))

cat("\nSaved to results/boosting_results.rds\n")
