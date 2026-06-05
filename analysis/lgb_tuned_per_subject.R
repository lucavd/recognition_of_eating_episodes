# ---------------------------------------------------------------------------
# LGB tuned — LOSO per-subject metrics, to reproduce the 'LightGBM (tuned)'
# row of Table 2 in the submitted manuscript (v1): sens=0.293, spec=0.886,
# bal_acc=0.590, AUC=0.691. We run a compact inner-LOSO Latin-hypercube
# search (15 points) selecting the configuration that maximises sensitivity,
# then refit that configuration on each outer LOSO fold and save per-subject
# predictions for the subject-level cluster bootstrap.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(bonsai)
  library(lightgbm)
  library(here)
})

set.seed(2026)

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI, -delta_s)

subjects <- sort(unique(dat$subject))
class_freq <- table(dat$EU)
spw <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
n_cores <- parallel::detectCores()

# Recipe factory ------------------------------------------------------------
make_recipe <- function(tr) {
  recipe(EU ~ ., data = tr) |>
    step_rm(subject) |>
    step_mutate(across(c(arm, meal, food), as.factor)) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}

# -- Step A: pick best tuned config via 5-fold subject-group CV on the whole
# pool (cheaper than nested LOSO and consistent with how the paper has been
# run in Phase 9) ----------------------------------------------------------

folds_inner <- group_vfold_cv(dat, group = subject, v = 5)

lgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(),
  learn_rate = tune(), min_n = tune()
) |>
  set_engine("lightgbm",
             class_weight = "balanced",
             verbose = -1,
             num_threads = n_cores) |>
  set_mode("classification")

lgb_grid <- grid_latin_hypercube(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),
  min_n(range = c(5, 15)),
  size = 15
)

rec_inner <- make_recipe(dat)
wf_inner  <- workflow() |> add_recipe(rec_inner) |> add_model(lgb_spec)

cat("\n--- Inner tuning (5-fold subject-group CV, 15 LH points, tune_grid) ---\n")
t0 <- Sys.time()
lgb_tune <- tune_grid(
  wf_inner,
  resamples = folds_inner,
  grid = lgb_grid,
  metrics = metric_set(sens, spec, roc_auc),
  control = control_grid(verbose = FALSE, save_pred = FALSE)
)
t1 <- Sys.time()
cat("Inner tuning time:", round(difftime(t1, t0, units = "mins"), 1), "minutes\n")

lgb_best <- select_best(lgb_tune, metric = "sens")
cat("Best LGB params (max sens):\n"); print(lgb_best)

# -- Step B: outer LOSO using the selected config ---------------------------

lgb_best_spec <- boost_tree(
  trees      = lgb_best$trees,
  tree_depth = lgb_best$tree_depth,
  learn_rate = lgb_best$learn_rate,
  min_n      = lgb_best$min_n
) |>
  set_engine("lightgbm",
             class_weight = "balanced",
             verbose = -1,
             num_threads = n_cores) |>
  set_mode("classification")

metrics_set <- metric_set(roc_auc, sens, spec, bal_accuracy)

fold_res <- vector("list", length(subjects))
for (i in seq_along(subjects)) {
  subj  <- subjects[i]
  train <- dat |> filter(subject != subj)
  test  <- dat |> filter(subject == subj)

  rec <- make_recipe(train)
  wf  <- workflow() |> add_recipe(rec) |> add_model(lgb_best_spec)
  fit <- wf |> fit(data = train)

  preds <- predict(fit, test, type = "prob") |>
    bind_cols(predict(fit, test)) |>
    bind_cols(test |> select(EU))

  fm <- preds |>
    metrics_set(truth = EU, .pred_eating, estimate = .pred_class) |>
    mutate(subject = subj, fold = i)
  fold_res[[i]] <- fm
  ba <- fm |> filter(.metric == "bal_accuracy") |> pull(.estimate)
  se <- fm |> filter(.metric == "sens")         |> pull(.estimate)
  cat(sprintf("  fold %2d subj %s  bal_acc=%.3f  sens=%.3f\n", i, subj, ba, se))
}

combined <- bind_rows(fold_res) |>
  select(subject, fold, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy) |>
  mutate(model = "LGB (tuned)")

write_csv(combined, file.path(OUT, "per_subject_lgb_tuned.csv"))
saveRDS(list(best_params = lgb_best, per_subject = combined),
        file.path(OUT, "lgb_tuned_loso.rds"))

cat("\nMeans across subjects:\n")
print(combined |>
  summarise(sens = mean(sens), spec = mean(spec),
            bal_acc = mean(bal_acc), auc = mean(auc)))

cat("\nSaved:", file.path(OUT, "per_subject_lgb_tuned.csv"), "\n")
