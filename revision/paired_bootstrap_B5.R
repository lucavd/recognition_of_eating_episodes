# ---------------------------------------------------------------------------
# B5 — Paired cluster bootstrap test for pairwise comparison of all LOSO
# models. Response to Reviewer #1, Comment 5.
#
# For each pair (A, B) of models we test the null that the mean-across-
# subjects of metric M is equal for A and B. At each of B = 1000 iterations
# (seed = 1812) we sample 19 subjects with replacement; for each sampled
# subject we compute (metric_A - metric_B); we average over the sampled
# subjects; the distribution of these averages is the bootstrap distribution
# of the mean difference Δ(A,B). Reported:
#   - Δ̂ point estimate (mean-across-subjects difference)
#   - 95% CI of Δ (percentile)
#   - P(A > B) empirical probability from the bootstrap distribution
#
# Models (9): LR, DT (Reviewer's baselines) + RF def, RF tuned, XGB def,
# XGB tuned, LGB def + Attention, Transformer.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

set.seed(1812)
B_BOOT <- 1000

PROJ <- here()
RES  <- file.path(PROJ, "results")
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

ORIG_ORDER <- c("01","03","04","05","06","07","08","09","10","11","12","13",
                "14","15","16","17","18","19","20")
to_pos <- function(x) match(sprintf("%02d", as.integer(x)), ORIG_ORDER)

# 1. ML default (RF, XGB, LGB) -----------------------------------------------
ml_def <- readRDS(file.path(RES, "loso_all_models.rds"))$all_results |>
  pivot_wider(id_cols = c(subject, model),
              names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy) |>
  mutate(subject_pos = to_pos(subject),
         model = recode(model,
                        "RF" = "RF (default)",
                        "XGBoost" = "XGB (default)",
                        "LightGBM" = "LGB (default)"))

# 2. Tuned RF and XGB (IDs already 1..19 ordinal) ----------------------------
tuned <- read_csv(file.path(RES, "subject_metrics_tuned.csv"),
                  show_col_types = FALSE) |>
  rename(auc = roc_auc) |>
  mutate(bal_acc = (sens + spec) / 2,
         subject_pos = as.integer(subject),
         model = recode(model,
                        "RF_tuned" = "RF (tuned)",
                        "XGB_tuned" = "XGB (tuned)"))

# 3. Attention (tuned) + Transformer from deep_per_subject.csv ---------------
dl <- read_csv(file.path(OUT, "deep_per_subject.csv"),
               show_col_types = FALSE) |>
  filter(sprintf("%02d", as.integer(subject)) != "02") |>
  mutate(subject_pos = to_pos(subject),
         model = recode(model, "Attention" = "Attention (tuned)")) |>
  select(subject_pos, model, sens, spec, bal_acc, auc)

# 3b. Attention (default) -----------------------------------------------------
att_def <- read_csv(file.path(OUT, "per_subject_attention_default.csv"),
                    show_col_types = FALSE) |>
  filter(sprintf("%02d", as.integer(subject)) != "02") |>
  mutate(subject_pos = to_pos(subject)) |>
  select(subject_pos, model, sens, spec, bal_acc, auc)

# 4. LR + DT (IDs are original) ----------------------------------------------
lrdt <- read_csv(file.path(OUT, "per_subject_lr_dt.csv"),
                 show_col_types = FALSE) |>
  mutate(subject_pos = to_pos(subject)) |>
  select(subject_pos, model, sens, spec, bal_acc, auc)

# 5. LGB tuned ---------------------------------------------------------------
lgbt <- read_csv(file.path(OUT, "per_subject_lgb_tuned.csv"),
                 show_col_types = FALSE) |>
  mutate(subject_pos = to_pos(subject), model = "LGB (tuned)") |>
  select(subject_pos, model, sens, spec, bal_acc, auc)

# Combine ---------------------------------------------------------------------
per_subj <- bind_rows(
  ml_def |> select(subject_pos, model, sens, spec, bal_acc, auc),
  tuned  |> select(subject_pos, model, sens, spec, bal_acc, auc),
  dl,
  att_def,
  lrdt,
  lgbt
) |>
  arrange(model, subject_pos)

stopifnot(all(table(per_subj$model) == 19))
models <- unique(per_subj$model)
stopifnot(length(models) == 11)

cat("Models included:\n"); print(models)

# -- Paired bootstrap --------------------------------------------------------

metrics <- c("sens", "spec", "bal_acc", "auc")
n_subj  <- 19

wide <- per_subj |>
  pivot_wider(id_cols = subject_pos,
              names_from = model,
              values_from = all_of(metrics),
              names_sep = "||")

# wide columns: subject_pos, then for each metric: metric||ModelName
# use long form indexed by (subject_pos, model) and a named array per metric
arr <- list()
for (met in metrics) {
  arr[[met]] <- per_subj |>
    select(subject_pos, model, all_of(met)) |>
    pivot_wider(names_from = model, values_from = all_of(met)) |>
    arrange(subject_pos)
}

paired_boot_one <- function(metric, A, B, n_boot = B_BOOT) {
  df <- arr[[metric]]
  va <- df[[A]]; vb <- df[[B]]
  pairwise_diff <- va - vb
  point_diff <- mean(pairwise_diff, na.rm = TRUE)
  reps <- replicate(n_boot, {
    idx <- sample(seq_len(n_subj), n_subj, replace = TRUE)
    mean(pairwise_diff[idx], na.rm = TRUE)
  })
  tibble(
    metric   = metric,
    model_A  = A, model_B = B,
    mean_A   = mean(va), mean_B = mean(vb),
    delta    = point_diff,
    ci_low   = quantile(reps, 0.025, na.rm = TRUE),
    ci_high  = quantile(reps, 0.975, na.rm = TRUE),
    p_A_gt_B = mean(reps > 0, na.rm = TRUE),
    p_two_sided = 2 * min(mean(reps > 0, na.rm = TRUE),
                          mean(reps < 0, na.rm = TRUE))
  )
}

grid <- expand_grid(A = models, B = models) |> filter(A != B)

results <- map_dfr(metrics, function(met) {
  map_dfr(seq_len(nrow(grid)), function(k) {
    paired_boot_one(met, grid$A[k], grid$B[k])
  })
})

write_csv(results, file.path(OUT, "paired_bootstrap_all_pairs.csv"))

# Matrix view (bal_acc delta, upper triangle) --------------------------------

bal_acc_matrix <- results |>
  filter(metric == "bal_acc") |>
  mutate(cell = sprintf("%+.3f\n[%+.3f,%+.3f]\np=%.3f",
                        delta, ci_low, ci_high, p_two_sided)) |>
  select(model_A, model_B, cell) |>
  pivot_wider(names_from = model_B, values_from = cell)

write_csv(bal_acc_matrix, file.path(OUT, "paired_bootstrap_balacc_matrix.csv"))

# Summary vs XGB tuned (the paper's "best" claim) ----------------------------
vs_xgb <- results |>
  filter(model_B == "XGB (tuned)", model_A != "XGB (tuned)") |>
  arrange(metric, desc(delta))
cat("\n=== Differences (model A − XGB tuned), by metric ===\n")
print(as.data.frame(vs_xgb |>
  mutate(across(c(mean_A, mean_B, delta, ci_low, ci_high), round, 3),
         p_two_sided = round(p_two_sided, 3)) |>
  select(metric, model_A, mean_A, mean_B, delta, ci_low, ci_high, p_two_sided)),
  row.names = FALSE)

cat("\nSaved:\n",
    " -", file.path(OUT, "paired_bootstrap_all_pairs.csv"), "\n",
    " -", file.path(OUT, "paired_bootstrap_balacc_matrix.csv"), "\n")
