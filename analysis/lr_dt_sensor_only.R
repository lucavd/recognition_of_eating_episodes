# =============================================================================
# Round 2 / Reviewer #3 follow-up (D2 symmetry)
# Logistic Regression and Decision Tree under LOSO on BOTH the full
# (75-predictor, context-augmented) and the sensor-only (59-predictor,
# motion-only) feature sets.
#
# Rationale: S2 Appendix Table S2.2 re-fits only the three tuned tree models
# sensor-only. LR and DT are naive baselines with NO hyperparameter selection,
# so they need plain LOSO (no nesting) and are valid final-test estimates.
# This script adds their sensor-only rows for symmetry.
#
# Protocol replicates analysis/loso_lr_dt.R (the LR/DT fit used for Table 2):
# class weights via importance_weights (n_neg/n_pos for the
# eating class, 1 otherwise); recipe and bootstrap identical to the tree-model
# sensor-only analysis (analysis/01_nested_loso.R):
#   sensor_only: step_rm(subject, ID, EI, delta_s, arm, meal, food) -> zv -> normalize  (59)
#   full:        step_rm(subject, ID, EI, delta_s) -> factors -> novel -> dummy -> zv -> normalize (75)
# Bootstrap: subject-level cluster bootstrap, B = 1000, seed = 1812 (as Table 2).
#
# Sanity check: the FULL rows must reproduce the Round-1 Table 2 LR/DT numbers
# (LR 0.625/0.643/0.634/0.697; DT 0.572/0.659/0.616/0.649).
#
# Outputs -> results/lr_dt_sensor_only_*.{csv}
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(rpart)
  library(here)
})

set.seed(2026)
OUT <- here("results", "bootstrap")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Data (delta_s = 5 s; subject 02 already absent in dat_all.rds)
# ---------------------------------------------------------------------------
dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"),
                     levels = c("eating", "non_eating")))   # "eating" = event (first level)

subjects <- sort(unique(dat$subject))
stopifnot(length(subjects) == 19, nrow(dat) == 26304)
cat(sprintf("Data: %d IU, %d subjects, eating prevalence %.1f%%\n\n",
            nrow(dat), length(subjects), 100 * mean(dat$EU == "eating")))

# ---------------------------------------------------------------------------
# Recipe builder: full (75) vs sensor-only (59) — identical to 01_nested_loso.R
# ---------------------------------------------------------------------------
make_recipe <- function(tr, feature_set) {
  if (feature_set == "sensor_only") {
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

model_specs <- list(
  LR = logistic_reg() |> set_engine("glm") |> set_mode("classification"),
  DT = decision_tree() |> set_engine("rpart") |> set_mode("classification")
)

eval_metrics <- metric_set(roc_auc, sens, spec, bal_accuracy)

# ---------------------------------------------------------------------------
# Plain LOSO for one (model, feature_set): class weights via importance_weights
# ---------------------------------------------------------------------------
loso_run <- function(mname, feature_set) {
  cat(sprintf("\n===== LOSO | model=%s | features=%s =====\n", mname, feature_set))
  mspec <- model_specs[[mname]]
  rows  <- vector("list", length(subjects))

  for (i in seq_along(subjects)) {
    s     <- subjects[i]
    train <- dat |> filter(subject != s)
    test  <- dat |> filter(subject == s)

    n_pos <- sum(train$EU == "eating")
    n_neg <- sum(train$EU == "non_eating")
    wts   <- ifelse(train$EU == "eating", n_neg / n_pos, 1)
    train$case_wt <- importance_weights(wts)

    wf <- workflow() |>
      add_recipe(make_recipe(train, feature_set)) |>
      add_model(mspec) |>
      add_case_weights(case_wt)

    fit <- tryCatch(wf |> fit(data = train),
                    error = function(e) { cat("  ERR:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit)) next

    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))

    fm <- preds |>
      eval_metrics(truth = EU, .pred_eating, estimate = .pred_class) |>
      select(.metric, .estimate) |>
      pivot_wider(names_from = .metric, values_from = .estimate) |>
      rename(auc = roc_auc, bal_acc = bal_accuracy)

    rows[[i]] <- bind_cols(
      tibble(model = mname, feature_set = feature_set, subject = s, fold = i), fm)
    cat(sprintf("  fold %2d subj %s  sens=%.3f spec=%.3f bal_acc=%.3f auc=%.3f\n",
                i, s, fm$sens, fm$spec, fm$bal_acc, fm$auc))
  }
  bind_rows(rows)
}

# ---------------------------------------------------------------------------
# Subject-level cluster bootstrap 95% CI (B = 1000, seed = 1812) — as 01_nested_loso.R
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
# Run all 4 combinations
# ---------------------------------------------------------------------------
combos <- expand_grid(model = c("LR", "DT"),
                      feature_set = c("full", "sensor_only"))
all_ps <- list()
for (k in seq_len(nrow(combos))) {
  all_ps[[k]] <- loso_run(combos$model[k], combos$feature_set[k])
}
per_subject <- bind_rows(all_ps)
write_csv(per_subject, file.path(OUT, "lr_dt_sensor_only_per_subject.csv"))

ci <- boot_ci(per_subject)
write_csv(ci, file.path(OUT, "lr_dt_sensor_only_ci.csv"))

pretty <- ci |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  select(model, feature_set, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell) |>
  select(model, feature_set, sens, spec, bal_acc, auc)
write_csv(pretty, file.path(OUT, "lr_dt_sensor_only_pretty.csv"))

cat("\n\n================ LR / DT  (point [95% CI]) ================\n")
print(as.data.frame(pretty))

# ---------------------------------------------------------------------------
# Sanity check: FULL must reproduce Round-1 Table 2
# ---------------------------------------------------------------------------
cat("\n=== SANITY CHECK: full-feature means vs Round-1 Table 2 ===\n")
ref <- tribble(
  ~model, ~sens, ~spec, ~bal_acc, ~auc,
  "LR", 0.625, 0.643, 0.634, 0.697,
  "DT", 0.572, 0.659, 0.616, 0.649
)
chk <- ci |>
  filter(feature_set == "full") |>
  select(model, metric, mean) |>
  pivot_wider(names_from = metric, values_from = mean) |>
  select(model, sens, spec, bal_acc, auc)
print(as.data.frame(chk))
cat("\nReference (Round-1 Table 2):\n")
print(as.data.frame(ref))

cat("\nSaved to results/: ",
    "lr_dt_sensor_only_{per_subject,ci,pretty}.csv\n")
cat("\nDONE.\n")
