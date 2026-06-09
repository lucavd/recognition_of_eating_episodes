# =============================================================================
# Round 2 / Reviewer #3 — Comments D1 + D2
# NESTED Leave-One-Subject-Out cross-validation for the tuned tree models,
# on BOTH the full (75-predictor, context-augmented) and the sensor-only
# (59-predictor, motion-only) feature sets.
#
# D1: original tuning selected hyperparameters on the SAME 19 LOSO folds used
#     for reporting (post-selection optimism). Here, for each outer held-out
#     subject, hyperparameters are selected using ONLY the other 18 subjects
#     (inner 5-fold subject-grouped CV), then evaluated once on the held-out
#     subject. This yields an unbiased final-test estimate.
# D2: sensor-only feature set removes the 16 experimental-context dummies
#     (arm/menu, meal, food), keeping the 59 motion-derived predictors.
#
# Selection metric: sensitivity (matches original 09_hyperparameter_tuning).
# Grids / specs: identical to the original Phase-9 hyperparameter-tuning script.
# Inner selection: tune_grid (full 15-pt Latin-hypercube) rather than racing.
# Bootstrap: subject-level cluster bootstrap, B = 1000, seed = 1812 (as Table 2).
#
# Outputs -> results/
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(ranger)
  library(bonsai)
  library(lightgbm)
  library(here)
})

set.seed(2026)
n_cores <- parallel::detectCores()
OUT <- here("results", "bootstrap")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Data (delta_s = 5 s; subject 02 already absent in dat_all.rds)
# ---------------------------------------------------------------------------
dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"),
                     levels = c("eating", "non_eating")))   # "eating" = event

subjects <- sort(unique(dat$subject))
stopifnot(length(subjects) == 19, nrow(dat) == 26304)

# Global class weight (fixed scheme, as in the original tuning script)
cf  <- table(dat$EU)
spw <- as.numeric(cf["non_eating"] / cf["eating"])
cat(sprintf("Data: %d IU, %d subjects, eating prevalence %.1f%%, scale_pos_weight %.3f\n\n",
            nrow(dat), length(subjects), 100 * cf["eating"] / sum(cf), spw))

# ---------------------------------------------------------------------------
# Recipe builder: full (75) vs sensor-only (59)
# ---------------------------------------------------------------------------
make_recipe <- function(tr, feature_set) {
  if (feature_set == "sensor_only") {
    # drop the 4 context categoricals -> only motion-derived numeric predictors remain
    recipe(EU ~ ., data = tr) |>
      step_rm(subject, ID, EI, delta_s, arm, meal, food) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())
  } else {
    recipe(EU ~ ., data = tr) |>
      step_rm(subject, ID, EI, delta_s) |>
      step_mutate(across(c(arm, meal, food), as.factor)) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())
  }
}

# ---------------------------------------------------------------------------
# Model specs + grids (identical ranges to the original Phase-9 script)
# ---------------------------------------------------------------------------
make_spec <- function(model) {
  switch(model,
    rf  = rand_forest(mtry = tune(), min_n = tune(), trees = 500) |>
            set_engine("ranger",
                       class.weights = c("eating" = spw, "non_eating" = 1),
                       num.threads = n_cores) |>
            set_mode("classification"),
    xgb = boost_tree(trees = tune(), tree_depth = tune(),
                     learn_rate = tune(), min_n = tune()) |>
            set_engine("xgboost", scale_pos_weight = spw,
                       verbosity = 0, nthread = n_cores) |>
            set_mode("classification"),
    lgb = boost_tree(trees = tune(), tree_depth = tune(),
                     learn_rate = tune(), min_n = tune()) |>
            set_engine("lightgbm", class_weight = "balanced",
                       verbose = -1, num_threads = n_cores) |>
            set_mode("classification")
  )
}

set.seed(2026)
grids <- list(
  rf  = grid_latin_hypercube(mtry(range = c(5, 30)),
                             min_n(range = c(5, 20)), size = 15),
  xgb = grid_latin_hypercube(trees(range = c(300, 700)),
                             tree_depth(range = c(4, 10)),
                             learn_rate(range = c(-2, -1)),
                             min_n(range = c(5, 15)), size = 15),
  lgb = grid_latin_hypercube(trees(range = c(300, 700)),
                             tree_depth(range = c(4, 10)),
                             learn_rate(range = c(-2, -1)),
                             min_n(range = c(5, 15)), size = 15)
)

inner_metrics <- metric_set(sens, spec, roc_auc)
eval_metrics  <- metric_set(roc_auc, sens, spec, bal_accuracy)

# ---------------------------------------------------------------------------
# Nested LOSO for one (model, feature_set) combination
# ---------------------------------------------------------------------------
nested_loso <- function(model, feature_set) {
  cat(sprintf("\n===== NESTED LOSO | model=%s | features=%s =====\n", model, feature_set))
  spec <- make_spec(model)
  grid <- grids[[model]]
  rows <- vector("list", length(subjects))

  for (i in seq_along(subjects)) {
    s     <- subjects[i]
    train <- dat |> filter(subject != s)   # 18 subjects
    test  <- dat |> filter(subject == s)   # held-out subject

    # inner selection on the 18 training subjects ONLY
    set.seed(2026 + i)
    inner <- group_vfold_cv(train, group = subject, v = 5)
    wf    <- workflow() |>
      add_recipe(make_recipe(train, feature_set)) |>
      add_model(spec)

    tg <- tune_grid(wf, resamples = inner, grid = grid,
                    metrics = inner_metrics,
                    control = control_grid(verbose = FALSE, save_pred = FALSE))
    best <- select_best(tg, metric = "sens")

    # refit best on all 18, predict held-out subject ONCE
    fit <- finalize_workflow(wf, best) |> fit(data = train)
    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))

    fm <- preds |>
      eval_metrics(truth = EU, .pred_eating, estimate = .pred_class) |>
      select(.metric, .estimate) |>
      pivot_wider(names_from = .metric, values_from = .estimate) |>
      rename(auc = roc_auc, bal_acc = bal_accuracy)

    rows[[i]] <- bind_cols(
      tibble(model = model, feature_set = feature_set, subject = s, fold = i),
      fm,
      best |> select(-any_of(".config")) |>
        rename_with(~ paste0("hp_", .x))
    )
    cat(sprintf("  fold %2d subj %s  sens=%.3f spec=%.3f bal_acc=%.3f auc=%.3f\n",
                i, s, fm$sens, fm$spec, fm$bal_acc, fm$auc))
  }
  bind_rows(rows)
}

# ---------------------------------------------------------------------------
# Subject-level cluster bootstrap 95% CI (B = 1000, seed = 1812)
# ---------------------------------------------------------------------------
boot_ci <- function(per_subject) {
  metrics <- c("sens", "spec", "bal_acc", "auc")
  per_subject |>
    group_by(model, feature_set) |>
    group_modify(~{
      df <- .x
      map_dfr(metrics, function(m) {
        vals <- df[[m]]
        set.seed(1812)
        reps <- replicate(1000, {
          idx <- sample(seq_along(vals), length(vals), replace = TRUE)
          mean(vals[idx], na.rm = TRUE)
        })
        tibble(metric = m, mean = mean(vals, na.rm = TRUE),
               ci_low = quantile(reps, 0.025, names = FALSE),
               ci_high = quantile(reps, 0.975, names = FALSE))
      })
    }) |>
    ungroup()
}

# ---------------------------------------------------------------------------
# Run all 6 combinations, write incrementally
# ---------------------------------------------------------------------------
combos <- expand_grid(model = c("xgb", "rf", "lgb"),
                      feature_set = c("full", "sensor_only"))

all_ps <- list()
for (k in seq_len(nrow(combos))) {
  ps <- nested_loso(combos$model[k], combos$feature_set[k])
  all_ps[[k]] <- ps
  # incremental save after each combo
  bind_rows(all_ps) |> write_csv(file.path(OUT, "nested_loso_per_subject.csv"))
  ci_now <- boot_ci(bind_rows(all_ps))
  write_csv(ci_now, file.path(OUT, "nested_loso_ci.csv"))
  saveRDS(list(per_subject = bind_rows(all_ps), ci = ci_now),
          file.path(OUT, "nested_loso.rds"))
  cat(sprintf("  [saved after combo %d/%d]\n", k, nrow(combos)))
}

per_subject <- bind_rows(all_ps)
ci <- boot_ci(per_subject)

pretty <- ci |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  select(model, feature_set, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell)
write_csv(pretty, file.path(OUT, "nested_loso_pretty.csv"))

cat("\n\n================ NESTED-LOSO SUMMARY (point [95% CI]) ================\n")
print(as.data.frame(pretty))
cat("\nSaved to results/: nested_loso_per_subject.csv, ",
    "nested_loso_ci.csv, nested_loso_pretty.csv, nested_loso.rds\n")
cat("\nDONE.\n")
