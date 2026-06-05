# ---------------------------------------------------------------------------
# C10 — LOSO XGBoost (tuned) on the filtered and unfiltered IU datasets
# built by feature_eng_filter.R. Subject 02 is excluded. Cluster-bootstrap
# 95% CIs (B = 1000, seed = 1812) on sens, spec, bal_acc and AUC for both
# pipelines, and a paired cluster-bootstrap comparison of the filtered
# vs unfiltered per-subject metrics.
#
# Hyperparameters: same as the XGB tuned configuration that produced the
# manuscript Table 2 row (paper_figures_data.rds$xgb_params):
#   trees = 599, min_n = 13, tree_depth = 4, learn_rate = 0.01347532
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(here)
})

set.seed(2026)

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

XGB_TUNED <- list(trees = 599, min_n = 13, tree_depth = 4, learn_rate = 0.01347532)

metrics_set <- metric_set(roc_auc, sens, spec, bal_accuracy)

loso_run <- function(iu_df, tag) {
  cat("\n============ LOSO:", tag, "============\n")

  dat <- iu_df |>
    dplyr::filter(subject != "02") |>
    mutate(EU = factor(EU, levels = c("eating", "non_eating")))

  # drop meta cols not needed
  dat$prop_eating <- NULL
  dat$n_samples   <- NULL

  subjects <- sort(unique(dat$subject))
  cat("N subjects:", length(subjects), "| N IU:", nrow(dat), "\n")
  cat("class balance:", sprintf("%.3f eating\n",
                                mean(dat$EU == "eating")))

  cf  <- table(dat$EU)
  spw <- as.numeric(cf["non_eating"] / cf["eating"])

  xgb_spec <- boost_tree(
    trees      = XGB_TUNED$trees,
    min_n      = XGB_TUNED$min_n,
    tree_depth = XGB_TUNED$tree_depth,
    learn_rate = XGB_TUNED$learn_rate
  ) |>
    set_engine("xgboost", scale_pos_weight = spw, verbosity = 0,
               nthread = parallel::detectCores()) |>
    set_mode("classification")

  make_recipe <- function(tr) {
    recipe(EU ~ ., data = tr) |>
      step_rm(subject) |>
      step_mutate(across(c(arm, meal, food), as.factor)) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())
  }

  fold_res <- vector("list", length(subjects))
  for (i in seq_along(subjects)) {
    subj  <- subjects[i]
    train <- dat |> dplyr::filter(subject != subj)
    test  <- dat |> dplyr::filter(subject == subj)

    wf  <- workflow() |> add_recipe(make_recipe(train)) |> add_model(xgb_spec)
    fit <- wf |> fit(data = train)

    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> dplyr::select(EU))

    fm <- preds |>
      metrics_set(truth = EU, .pred_eating, estimate = .pred_class) |>
      mutate(subject = subj, fold = i)
    fold_res[[i]] <- fm

    ba <- fm |> dplyr::filter(.metric == "bal_accuracy") |> pull(.estimate)
    se <- fm |> dplyr::filter(.metric == "sens")         |> pull(.estimate)
    cat(sprintf("  fold %2d subj %s  bal_acc=%.3f  sens=%.3f\n",
                i, subj, ba, se))
  }

  bind_rows(fold_res) |>
    dplyr::select(subject, fold, .metric, .estimate) |>
    pivot_wider(names_from = .metric, values_from = .estimate) |>
    rename(auc = roc_auc, bal_acc = bal_accuracy) |>
    mutate(pipeline = tag)
}

iu_unf <- readRDS(file.path(OUT, "iu_features_unfiltered.rds"))
iu_flt <- readRDS(file.path(OUT, "iu_features_filtered.rds"))

per_subj_unf <- loso_run(iu_unf, "unfiltered")
per_subj_flt <- loso_run(iu_flt, "filtered")

per_subj <- bind_rows(per_subj_unf, per_subj_flt)
write_csv(per_subj, file.path(OUT, "per_subject_filter.csv"))

# --- Cluster bootstrap 95% CI (marginal) -----------------------------------
set.seed(1812)
B_BOOT  <- 1000
metrics <- c("sens", "spec", "bal_acc", "auc")

boot_one <- function(vals, B) {
  n <- length(vals)
  reps <- replicate(B, mean(sample(vals, n, replace = TRUE), na.rm = TRUE))
  tibble(mean = mean(vals, na.rm = TRUE),
         ci_low  = quantile(reps, 0.025, na.rm = TRUE),
         ci_high = quantile(reps, 0.975, na.rm = TRUE))
}

ci_tbl <- map_dfr(unique(per_subj$pipeline), function(p) {
  d <- per_subj |> dplyr::filter(pipeline == p)
  map_dfr(metrics, function(m) {
    boot_one(d[[m]], B_BOOT) |> mutate(pipeline = p, metric = m, .before = 1)
  })
})

ci_pretty <- ci_tbl |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  dplyr::select(pipeline, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell) |>
  dplyr::select(pipeline, sens, spec, bal_acc, auc)

write_csv(ci_pretty, file.path(OUT, "filter_ablation_ci_pretty.csv"))
write_csv(ci_tbl,    file.path(OUT, "filter_ablation_ci.csv"))

cat("\n=== Marginal 95% CI ===\n")
print(as.data.frame(ci_pretty), row.names = FALSE)

# --- Paired cluster bootstrap (filtered - unfiltered) ---------------------
wide <- per_subj |>
  pivot_wider(id_cols = subject,
              names_from = pipeline,
              values_from = all_of(metrics),
              names_sep = "__")

paired <- map_dfr(metrics, function(m) {
  flt <- wide[[paste0(m, "__filtered")]]
  unf <- wide[[paste0(m, "__unfiltered")]]
  diff_vec <- flt - unf
  reps <- replicate(B_BOOT, {
    idx <- sample(seq_along(diff_vec), length(diff_vec), replace = TRUE)
    mean(diff_vec[idx], na.rm = TRUE)
  })
  tibble(metric = m,
         mean_unf = mean(unf, na.rm = TRUE),
         mean_flt = mean(flt, na.rm = TRUE),
         delta = mean(diff_vec, na.rm = TRUE),
         ci_low  = quantile(reps, 0.025, na.rm = TRUE),
         ci_high = quantile(reps, 0.975, na.rm = TRUE),
         p_two_sided = 2 * min(mean(reps > 0, na.rm = TRUE),
                                mean(reps < 0, na.rm = TRUE)))
})

write_csv(paired, file.path(OUT, "filter_ablation_paired.csv"))

cat("\n=== Paired bootstrap filtered − unfiltered ===\n")
print(as.data.frame(paired |>
  mutate(cell = sprintf("%+.3f [%+.3f, %+.3f]  p=%.3f",
                        delta, ci_low, ci_high, p_two_sided)) |>
  dplyr::select(metric, mean_unf, mean_flt, cell)),
  row.names = FALSE)

cat("\nSaved:\n",
    " -", file.path(OUT, "per_subject_filter.csv"), "\n",
    " -", file.path(OUT, "filter_ablation_ci_pretty.csv"),    "\n",
    " -", file.path(OUT, "filter_ablation_ci.csv"),           "\n",
    " -", file.path(OUT, "filter_ablation_paired.csv"),       "\n")
