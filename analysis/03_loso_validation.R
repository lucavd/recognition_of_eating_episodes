# 03_loso_validation.R - Leave-One-Subject-Out Validation
# Risolve data leakage da finestre sovrapposte
# Usa solo delta_s = 5 sec (miglior performance)

library(tidyverse)
library(tidymodels)
library(xgboost)
library(here)

set.seed(2026)

# =============================================================================
# LOAD DATA - SOLO DELTA_S = 5
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI, -delta_s)

cat("Dataset (delta_s=5):", nrow(dat), "obs\n")
cat("Soggetti:", length(unique(dat$subject)), "\n")

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
cat("scale_pos_weight:", scale_pos_weight, "\n")

# =============================================================================
# LOSO VALIDATION
# =============================================================================

subjects <- unique(dat$subject)
cat("\n=== LOSO VALIDATION ===\n")
cat("N fold:", length(subjects), "\n\n")

metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

results <- list()

for (i in seq_along(subjects)) {
  subj <- subjects[i]
  cat("Fold", i, "/", length(subjects), "- Test subject:", subj, "...")
  
  # Split
  test <- dat |> filter(subject == subj)
  train <- dat |> filter(subject != subj)
  
  # Recipe
  rec <- recipe(EU ~ ., data = train) |>
    step_rm(subject) |>
    step_mutate(across(c(arm, meal, food), as.factor)) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
  
  # XGBoost
  xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
    set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
    set_mode("classification")
  
  xgb_wf <- workflow() |>
    add_recipe(rec) |>
    add_model(xgb_spec)
  
  # Fit
  fit <- xgb_wf |> fit(data = train)
  
  # Predict
  preds <- predict(fit, test, type = "prob") |>
    bind_cols(predict(fit, test)) |>
    bind_cols(test |> select(EU))
  
  # Metrics
  fold_metrics <- preds |>
    metrics(truth = EU, .pred_eating, estimate = .pred_class) |>
    mutate(subject = subj, fold = i)
  
  results[[i]] <- fold_metrics
  
  auc <- fold_metrics |> filter(.metric == "roc_auc") |> pull(.estimate)
  sens <- fold_metrics |> filter(.metric == "sens") |> pull(.estimate)
  cat(" AUC:", round(auc, 3), "Sens:", round(sens, 3), "\n")
}

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

all_results <- bind_rows(results)

summary_stats <- all_results |>
  group_by(.metric) |>
  summarise(
    mean = mean(.estimate),
    sd = sd(.estimate),
    min = min(.estimate),
    max = max(.estimate),
    .groups = "drop"
  )

cat("\n=== LOSO SUMMARY (XGBoost, delta_s=5) ===\n")
print(summary_stats)

# Per-subject results
per_subject <- all_results |>
  select(subject, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

cat("\n=== PER-SUBJECT RESULTS ===\n")
print(per_subject, n = 20)

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  all_results = all_results,
  summary = summary_stats,
  per_subject = per_subject
), here("results", "loso_results.rds"))

cat("\nSaved to results/loso_results.rds\n")
