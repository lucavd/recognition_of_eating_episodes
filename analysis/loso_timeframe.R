# ---------------------------------------------------------------------------
# B4 — Window-size sensitivity analysis (LOSO) + exact feature count.
# Response to Reviewer #1, Comment 4.
#
# (i)  For each δs ∈ {1, 2, 3, 4, 5} we run XGBoost (default) under Leave-One-
#      Subject-Out cross-validation and derive 95% cluster bootstrap CIs
#      (B = 1000, seed = 1812) for sens, spec, bal_acc, AUC.
# (ii) Applying the preprocessing recipe to the δs = 5 s subset we count the
#      exact number of numerical predictors fed to the classifier and save
#      their names for the reproducibility appendix.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(yardstick)
  library(here)
})

set.seed(2026)

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

dat <- readRDS(here("data", "dat_all.rds")) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI)

subjects  <- sort(unique(dat$subject))
n_subj    <- length(subjects)
delta_set <- 1:5

metrics_set <- metric_set(roc_auc, sens, spec, bal_accuracy)

all_results <- list()

for (ds in delta_set) {
  cat("\n===== delta_s =", ds, "=====\n")
  dat_ds <- dat |> filter(delta_s == ds) |> select(-delta_s)

  class_freq <- table(dat_ds$EU)
  spw <- as.numeric(class_freq["non_eating"] / class_freq["eating"])

  xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
    set_engine("xgboost", scale_pos_weight = spw, verbosity = 0) |>
    set_mode("classification")

  fold_res <- vector("list", n_subj)
  for (i in seq_len(n_subj)) {
    subj  <- subjects[i]
    test  <- dat_ds |> filter(subject == subj)
    train <- dat_ds |> filter(subject != subj)

    rec <- recipe(EU ~ ., data = train) |>
      step_rm(subject) |>
      step_mutate(across(c(arm, meal, food), as.factor)) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())

    wf  <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
    fit <- wf |> fit(data = train)

    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))

    fm <- preds |>
      metrics_set(truth = EU, .pred_eating, estimate = .pred_class) |>
      mutate(subject = subj, fold = i, delta_s = ds)

    fold_res[[i]] <- fm
    ba <- fm |> filter(.metric == "bal_accuracy") |> pull(.estimate)
    se <- fm |> filter(.metric == "sens")         |> pull(.estimate)
    cat(sprintf("  fold %2d  subj %s  bal_acc=%.3f  sens=%.3f\n",
                i, subj, ba, se))
  }
  all_results[[paste0("ds", ds)]] <- bind_rows(fold_res)
}

combined <- bind_rows(all_results) |>
  select(delta_s, subject, fold, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy)

summary_by_ds <- combined |>
  pivot_longer(c(sens, spec, bal_acc, auc),
               names_to = "metric", values_to = "value") |>
  group_by(delta_s, metric) |>
  summarise(mean = mean(value), sd = sd(value),
            min = min(value), max = max(value),
            .groups = "drop")

# -- Cluster bootstrap CIs ---------------------------------------------------

set.seed(1812)
B_BOOT <- 1000
metrics_to_boot <- c("sens", "spec", "bal_acc", "auc")

boot_one <- function(vals, B) {
  n <- length(vals)
  reps <- replicate(B, mean(sample(vals, n, replace = TRUE)))
  tibble(mean = mean(vals),
         ci_low  = quantile(reps, 0.025, na.rm = TRUE),
         ci_high = quantile(reps, 0.975, na.rm = TRUE))
}

boot_ci <- combined |>
  pivot_longer(c(sens, spec, bal_acc, auc),
               names_to = "metric", values_to = "value") |>
  group_by(delta_s, metric) |>
  summarise(boot_one(value, B_BOOT), .groups = "drop")

write_csv(combined,  file.path(OUT, "timeframe_per_subject.csv"))
write_csv(summary_by_ds, file.path(OUT, "timeframe_summary.csv"))
write_csv(boot_ci,   file.path(OUT, "timeframe_ci.csv"))

saveRDS(list(per_subject = combined,
             summary     = summary_by_ds,
             ci          = boot_ci),
        file.path(OUT, "loso_timeframe_results.rds"))

cat("\n=== Pretty summary (with 95% CI) ===\n")
pretty <- boot_ci |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  select(delta_s, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell)
print(as.data.frame(pretty), row.names = FALSE)

# ---------------------------------------------------------------------------
# Exact feature count after preprocessing (δs = 5 s recipe)
# ---------------------------------------------------------------------------

cat("\n===== FEATURE COUNT (delta_s = 5) =====\n")
dat5 <- dat |> filter(delta_s == 5) |> select(-delta_s)
train_full <- dat5 |> filter(subject != subjects[1])

rec5 <- recipe(EU ~ ., data = train_full) |>
  step_rm(subject) |>
  step_mutate(across(c(arm, meal, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

prepped <- prep(rec5, training = train_full)
baked   <- bake(prepped, new_data = train_full)

feature_names <- setdiff(names(baked), "EU")
cat("N features after preprocessing:", length(feature_names), "\n")

# Classify features
classify_feature <- function(x) {
  case_when(
    x %in% c("acc_x", "acc_y", "acc_z", "pitch", "roll",
             "power", "total_energy") ~ "slope (window regression)",
    grepl("^mean_",  x) ~ "window statistic: mean",
    grepl("^med_",   x) ~ "window statistic: median",
    grepl("^min_",   x) ~ "window statistic: min",
    grepl("^max_",   x) ~ "window statistic: max",
    grepl("^iqr_",   x) ~ "window statistic: IQR",
    grepl("^sd_",    x) ~ "window statistic: SD",
    grepl("^cv_",    x) ~ "window statistic: CV",
    grepl("^cor_",   x) ~ "axis cross-correlation",
    grepl("^arm_",   x) ~ "categorical (arm) dummy",
    grepl("^meal_",  x) ~ "categorical (meal) dummy",
    grepl("^food_",  x) ~ "categorical (food) dummy",
    TRUE                ~ "other"
  )
}

feat_df <- tibble(feature = feature_names,
                  type    = classify_feature(feature_names))
feat_counts <- feat_df |> count(type, name = "n")
print(feat_counts)
# feature_list.csv is produced canonically by data/00_derive_data.R (S4 Table); only feature_counts.csv is written here
write_csv(feat_counts, file.path(OUT, "feature_counts.csv"))

cat("\nSaved:\n",
    " -", file.path(OUT, "timeframe_per_subject.csv"),     "\n",
    " -", file.path(OUT, "timeframe_summary.csv"),         "\n",
    " -", file.path(OUT, "timeframe_ci.csv"),              "\n",
    " -", file.path(OUT, "loso_timeframe_results.rds"),    "\n",
    " -", file.path(OUT, "feature_counts.csv"),            "\n")
