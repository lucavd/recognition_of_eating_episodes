# 02_random_forest.R - Random Forest con Class Weights
# Fase 2: Modello complesso per gestire class imbalance

library(tidyverse)
library(tidymodels)
library(ranger)
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
cat("Outcome:", table(dat$EU), "\n")

# =============================================================================
# CLASS WEIGHTS (inversamente proporzionali alla frequenza)
# =============================================================================

class_freq <- table(dat$EU)
class_weights <- max(class_freq) / class_freq
cat("\nClass weights:\n")
print(class_weights)

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

cat("\nTrain:", nrow(train), "| Test:", nrow(test), "\n")

# =============================================================================
# METRICS
# =============================================================================

metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

# =============================================================================
# RANDOM FOREST CON CLASS WEIGHTS
# =============================================================================

cat("\n=== RANDOM FOREST (class weights) ===\n")

rf_spec <- rand_forest(trees = 500) |>
  set_engine("ranger", 
             class.weights = c("eating" = as.numeric(class_weights["eating"]),
                               "non_eating" = as.numeric(class_weights["non_eating"])),
             importance = "impurity") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(rf_spec)

# CV
cat("\nRunning 10-fold CV...\n")
rf_cv <- fit_resamples(rf_wf, folds, metrics = metrics)

rf_cv_metrics <- collect_metrics(rf_cv)
cat("\nCV Results:\n")
print(rf_cv_metrics)

# Test set
cat("\nFitting final model on test set...\n")
rf_fit <- last_fit(rf_wf, split, metrics = metrics)
rf_test_metrics <- collect_metrics(rf_fit)
cat("\nTest Results:\n")
print(rf_test_metrics)

# =============================================================================
# COMPARISON WITH BASELINE
# =============================================================================

cat("\n=== COMPARISON ===\n")
baseline <- readRDS(here("results", "baseline_summary.rds"))

rf_summary <- rf_test_metrics |>
  mutate(model = "Random Forest (weighted)", set = "Test") |>
  select(model, set, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

comparison <- bind_rows(
  baseline |> filter(set == "Test"),
  rf_summary
)

print(comparison)

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  cv_metrics = rf_cv_metrics,
  test_metrics = rf_test_metrics,
  comparison = comparison
), here("results", "rf_weighted_results.rds"))

cat("\nSaved to results/rf_weighted_results.rds\n")
