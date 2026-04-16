# 03b_loso_all_models.R - LOSO Validation per RF, XGBoost, LightGBM
# Usa solo delta_s = 5 sec

library(tidyverse)
library(tidymodels)
library(xgboost)
library(ranger)
library(bonsai)
library(lightgbm)
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
cat("scale_pos_weight:", round(scale_pos_weight, 3), "\n")

# =============================================================================
# MODEL SPECS
# =============================================================================

rf_spec <- rand_forest(trees = 500) |>
  set_engine("ranger", class.weights = c("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
  set_mode("classification")

lgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("lightgbm", class_weight = list("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

models <- list(
  "RF" = rf_spec,
  "XGBoost" = xgb_spec,
  "LightGBM" = lgb_spec
)

# =============================================================================
# LOSO VALIDATION
# =============================================================================

subjects <- unique(dat$subject)
metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

all_results <- list()

for (model_name in names(models)) {
  cat("\n=== ", model_name, " ===\n")
  model_spec <- models[[model_name]]
  
  model_results <- list()
  
  for (i in seq_along(subjects)) {
    subj <- subjects[i]
    cat("Fold", i, "/", length(subjects), "- Subject:", subj, "...")
    
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
    
    # Workflow
    wf <- workflow() |>
      add_recipe(rec) |>
      add_model(model_spec)
    
    # Fit
    fit <- tryCatch(
      wf |> fit(data = train),
      error = function(e) NULL
    )
    
    if (is.null(fit)) {
      cat(" ERROR\n")
      next
    }
    
    # Predict
    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))
    
    # Metrics
    fold_metrics <- preds |>
      metrics(truth = EU, .pred_eating, estimate = .pred_class) |>
      mutate(subject = subj, fold = i, model = model_name)
    
    model_results[[i]] <- fold_metrics
    
    auc <- fold_metrics |> filter(.metric == "roc_auc") |> pull(.estimate)
    sens <- fold_metrics |> filter(.metric == "sens") |> pull(.estimate)
    cat(" AUC:", round(auc, 3), "Sens:", round(sens, 3), "\n")
  }
  
  all_results[[model_name]] <- bind_rows(model_results)
}

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

combined <- bind_rows(all_results)

summary_stats <- combined |>
  group_by(model, .metric) |>
  summarise(
    mean = mean(.estimate),
    sd = sd(.estimate),
    min = min(.estimate),
    max = max(.estimate),
    .groups = "drop"
  )

cat("\n\n=== LOSO SUMMARY (ALL MODELS) ===\n")
summary_wide <- summary_stats |>
  select(model, .metric, mean) |>
  pivot_wider(names_from = .metric, values_from = mean)
print(summary_wide)

cat("\n=== DETAILED STATS ===\n")
print(summary_stats |> arrange(.metric, model))

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  all_results = combined,
  summary = summary_stats,
  summary_wide = summary_wide
), here("results", "loso_all_models.rds"))

cat("\nSaved to results/loso_all_models.rds\n")
