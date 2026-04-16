# 02c_timeframe_analysis.R - Performance per timeframe (delta_s)
# Confronto RF vs XGBoost per ogni finestra temporale

library(tidyverse)
library(tidymodels)
library(ranger)
library(xgboost)
library(here)

set.seed(2026)

dat <- readRDS(here("data", "dat_all.rds"))
dat <- dat |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI)

metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

results <- list()

for (ds in 1:5) {
  cat("\n========== DELTA_S =", ds, "==========\n")
  
  dat_ds <- dat |> filter(delta_s == ds) |> select(-delta_s)
  
  # Class weights
  class_freq <- table(dat_ds$EU)
  class_weights <- max(class_freq) / class_freq
  scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
  
  cat("N obs:", nrow(dat_ds), "\n")
  
  # Split
  split <- initial_split(dat_ds, prop = 0.8, strata = EU)
  train <- training(split)
  test <- testing(split)
  
  # Recipe (senza delta_s)
  rec <- recipe(EU ~ ., data = train) |>
    step_mutate(across(c(arm, meal, subject, food), as.factor)) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
  
  # RF
  cat("RF...")
  rf_spec <- rand_forest(trees = 500) |>
    set_engine("ranger",
               class.weights = c("eating" = as.numeric(class_weights["eating"]),
                                 "non_eating" = as.numeric(class_weights["non_eating"]))) |>
    set_mode("classification")
  
  rf_wf <- workflow() |> add_recipe(rec) |> add_model(rf_spec)
  rf_fit <- last_fit(rf_wf, split, metrics = metrics)
  rf_metrics <- collect_metrics(rf_fit) |> mutate(model = "RF", delta_s = ds)
  
  # XGBoost
  cat("XGB...")
  xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
    set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
    set_mode("classification")
  
  xgb_wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
  xgb_fit <- last_fit(xgb_wf, split, metrics = metrics)
  xgb_metrics <- collect_metrics(xgb_fit) |> mutate(model = "XGBoost", delta_s = ds)
  
  results[[ds]] <- bind_rows(rf_metrics, xgb_metrics)
  cat("done\n")
}

# Combine results
all_results <- bind_rows(results)

# Summary table
summary_table <- all_results |>
  select(model, delta_s, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(model, delta_s)

cat("\n========== SUMMARY ==========\n")
print(summary_table, n = 20)

# Save
saveRDS(list(
  all_results = all_results,
  summary = summary_table
), here("results", "timeframe_results.rds"))

cat("\nSaved to results/timeframe_results.rds\n")
