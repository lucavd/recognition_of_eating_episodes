# ---------------------------------------------------------------------------
# B6 — XGBoost (tuned) LOSO with EI (Eating Intersect) as the target, to
# quantify the robustness of Table 2 main-result to label-boundary noise.
# Parallels the main XGB-tuned run but substitutes EI for EU.
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

dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EI = factor(ifelse(EI == 1, "eating", "non_eating"))) |>
  select(-ID, -EU, -delta_s)

cat("Dataset (EI target):", nrow(dat), "obs\n")
cat("Eating (EI=1):", sum(dat$EI == "eating"),
    "| Non-eating:", sum(dat$EI == "non_eating"), "\n")

subjects <- sort(unique(dat$subject))

# Class weight
cf  <- table(dat$EI)
spw <- as.numeric(cf["non_eating"] / cf["eating"])
cat("scale_pos_weight:", round(spw, 3), "\n\n")

# Latin-hypercube grid (same ranges as Phase 9)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),
  min_n(range = c(5, 15)),
  size = 15
)

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(),
  learn_rate = tune(), min_n = tune()
) |>
  set_engine("xgboost", scale_pos_weight = spw,
             verbosity = 0, nthread = parallel::detectCores()) |>
  set_mode("classification")

make_recipe <- function(tr) {
  recipe(EI ~ ., data = tr) |>
    step_rm(subject) |>
    step_mutate(across(c(arm, meal, food), as.factor)) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}

folds_inner <- group_vfold_cv(dat, group = subject, v = 5)
wf_inner    <- workflow() |>
  add_recipe(make_recipe(dat)) |>
  add_model(xgb_spec)

cat("--- Inner tuning (5-fold subject-group CV, 15 LH points, tune_grid) ---\n")
t0 <- Sys.time()
xgb_tune <- tune_grid(
  wf_inner,
  resamples = folds_inner,
  grid      = xgb_grid,
  metrics   = metric_set(sens, spec, roc_auc),
  control   = control_grid(verbose = FALSE, save_pred = FALSE)
)
t1 <- Sys.time()
cat("Inner tuning time:", round(difftime(t1, t0, units = "mins"), 1), "minutes\n")

xgb_best <- select_best(xgb_tune, metric = "sens")
cat("Best XGB params (on EI, max sens):\n"); print(xgb_best)

xgb_best_spec <- boost_tree(
  trees      = xgb_best$trees,
  tree_depth = xgb_best$tree_depth,
  learn_rate = xgb_best$learn_rate,
  min_n      = xgb_best$min_n
) |>
  set_engine("xgboost", scale_pos_weight = spw,
             verbosity = 0, nthread = parallel::detectCores()) |>
  set_mode("classification")

metrics_set <- metric_set(roc_auc, sens, spec, bal_accuracy)
fold_res <- vector("list", length(subjects))

for (i in seq_along(subjects)) {
  subj  <- subjects[i]
  train <- dat |> filter(subject != subj)
  test  <- dat |> filter(subject == subj)

  wf  <- workflow() |> add_recipe(make_recipe(train)) |> add_model(xgb_best_spec)
  fit <- wf |> fit(data = train)

  preds <- predict(fit, test, type = "prob") |>
    bind_cols(predict(fit, test)) |>
    bind_cols(test |> select(EI))

  fm <- preds |>
    metrics_set(truth = EI, .pred_eating, estimate = .pred_class) |>
    mutate(subject = subj, fold = i)
  fold_res[[i]] <- fm

  ba <- fm |> filter(.metric == "bal_accuracy") |> pull(.estimate)
  se <- fm |> filter(.metric == "sens")         |> pull(.estimate)
  cat(sprintf("  fold %2d subj %s  bal_acc=%.3f  sens=%.3f\n",
              i, subj, ba, se))
}

combined <- bind_rows(fold_res) |>
  select(subject, fold, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy) |>
  mutate(model = "XGB (tuned, EI target)")

write_csv(combined, file.path(OUT, "per_subject_xgb_tuned_EI.csv"))
saveRDS(list(best = xgb_best, per_subject = combined),
        file.path(OUT, "xgb_tuned_EI_loso.rds"))

# Cluster bootstrap 95% CI (same procedure as Table 2) ---------------------
set.seed(1812)
B_BOOT  <- 1000
metrics <- c("sens", "spec", "bal_acc", "auc")
subs    <- combined$subject

boot_means <- map_dfr(metrics, function(m) {
  vals <- combined[[m]]
  reps <- replicate(B_BOOT, {
    idx <- sample(seq_along(subs), length(subs), replace = TRUE)
    mean(vals[idx], na.rm = TRUE)
  })
  tibble(metric = m,
         mean   = mean(vals),
         ci_low  = quantile(reps, 0.025),
         ci_high = quantile(reps, 0.975))
})
write_csv(boot_means, file.path(OUT, "xgb_tuned_EI_ci.csv"))

cat("\n=== XGB tuned, EI target — mean [95% CI] ===\n")
print(as.data.frame(boot_means |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  select(metric, cell)))

cat("\nSaved:\n",
    " -", file.path(OUT, "per_subject_xgb_tuned_EI.csv"), "\n",
    " -", file.path(OUT, "xgb_tuned_EI_ci.csv"),           "\n",
    " -", file.path(OUT, "xgb_tuned_EI_loso.rds"),         "\n")
