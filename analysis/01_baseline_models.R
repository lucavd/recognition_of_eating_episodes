# 01_baseline_models.R - Modelli Baseline
# Fase 1: Logistic Regression + Decision Tree con 10-fold CV

library(tidyverse)
library(tidymodels)
library(here)

set.seed(2026)

# =============================================================================
# LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI)  # rimuovi ID e outcome alternativo

cat("Dataset:", nrow(dat), "obs,", ncol(dat), "vars\n")
cat("Outcome:", table(dat$EU), "\n")

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
# METRICS (decisione metodologica documentata in PLAN.md)
# - AUC-ROC: metrica principale, threshold-independent
# - Sensitivity + Specificity: metriche cliniche
# - Balanced Accuracy: summary (media di sens e spec)
# =============================================================================

metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

# =============================================================================
# MODEL 1: LOGISTIC REGRESSION
# =============================================================================

cat("\n=== LOGISTIC REGRESSION ===\n")

lr_spec <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

lr_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(lr_spec)

lr_cv <- fit_resamples(lr_wf, folds, metrics = metrics)

lr_cv_metrics <- collect_metrics(lr_cv)
cat("\nCV Results:\n")
print(lr_cv_metrics)

# Test set
lr_fit <- last_fit(lr_wf, split, metrics = metrics)
lr_test_metrics <- collect_metrics(lr_fit)
cat("\nTest Results:\n")
print(lr_test_metrics)

# =============================================================================
# MODEL 2: DECISION TREE
# =============================================================================

cat("\n=== DECISION TREE ===\n")

dt_spec <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")

dt_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(dt_spec)

dt_cv <- fit_resamples(dt_wf, folds, metrics = metrics)

dt_cv_metrics <- collect_metrics(dt_cv)
cat("\nCV Results:\n")
print(dt_cv_metrics)

# Test set
dt_fit <- last_fit(dt_wf, split, metrics = metrics)
dt_test_metrics <- collect_metrics(dt_fit)
cat("\nTest Results:\n")
print(dt_test_metrics)

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_cv <- bind_rows(
  lr_cv_metrics |> mutate(model = "Logistic Regression", set = "CV"),
  dt_cv_metrics |> mutate(model = "Decision Tree", set = "CV")
)

summary_test <- bind_rows(
  lr_test_metrics |> mutate(model = "Logistic Regression", set = "Test"),
  dt_test_metrics |> mutate(model = "Decision Tree", set = "Test")
)

summary_all <- bind_rows(summary_cv, summary_test) |>
  select(model, set, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

cat("\n=== SUMMARY ===\n")
print(summary_all)

# =============================================================================
# SAVE
# =============================================================================

saveRDS(summary_all, here("results", "baseline_summary.rds"))
cat("\nSaved to results/baseline_summary.rds\n")
