# ---------------------------------------------------------------------------
# C1 — False-positive pattern analysis for XGBoost (tuned), v2.
#
# This version replaces the earlier fp_pattern.R (which read
# results/paper_figures_data.rds, producing predictions
# whose aggregate mean specificity was 0.804 — not matching Table 2's 0.692).
#
# v2 regenerates LOSO predictions from scratch using the SAME pipeline as
# loso_filter.R unfiltered (the pipeline whose per-subject specificity mean of
# 0.689 matches Table 2's 0.692 within stochastic tolerance). Specifically:
#   - dataset:   iu_features_unfiltered.rds   (26,058 IU, 19 subjects)
#   - recipe:    step_rm(subject) → step_dummy → step_zv → step_normalize
#   - XGBoost:   trees = 599, min_n = 13, tree_depth = 4, learn_rate = 0.01347532
#                scale_pos_weight = non_eating/eating (per fold)
#   - LOSO loop over 19 subjects (subject 02 already excluded)
#
# Outputs (results/):
#   - fp_pattern_predictions.rds       IU-level predictions (subject, EU, .pred_class)
#   - fp_pattern_per_subject.csv       per-subject FP counts (overwrites v1)
#   - fp_pattern_summary.csv           aggregate summary (overwrites v1)
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

XGB_TUNED <- list(trees = 599, min_n = 13, tree_depth = 4,
                  learn_rate = 0.01347532)

iu <- readRDS(file.path(OUT, "iu_features_unfiltered.rds")) |>
  dplyr::filter(subject != "02") |>
  mutate(EU = factor(EU, levels = c("eating", "non_eating")))
iu$prop_eating <- NULL
iu$n_samples   <- NULL

subjects <- sort(unique(iu$subject))
cat("Running LOSO on", length(subjects), "subjects,", nrow(iu), "IUs\n")

xgb_spec <- boost_tree(
  trees      = XGB_TUNED$trees,
  min_n      = XGB_TUNED$min_n,
  tree_depth = XGB_TUNED$tree_depth,
  learn_rate = XGB_TUNED$learn_rate
) |>
  set_engine("xgboost",
             verbosity = 0,
             nthread   = parallel::detectCores()) |>
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

preds_all <- vector("list", length(subjects))
for (i in seq_along(subjects)) {
  subj  <- subjects[i]
  train <- iu |> dplyr::filter(subject != subj)
  test  <- iu |> dplyr::filter(subject == subj)

  cf  <- table(train$EU)
  spw <- as.numeric(cf["non_eating"] / cf["eating"])

  xgb_spec_fold <- boost_tree(
    trees      = XGB_TUNED$trees,
    min_n      = XGB_TUNED$min_n,
    tree_depth = XGB_TUNED$tree_depth,
    learn_rate = XGB_TUNED$learn_rate
  ) |>
    set_engine("xgboost",
               scale_pos_weight = spw,
               verbosity = 0,
               nthread   = parallel::detectCores()) |>
    set_mode("classification")

  wf  <- workflow() |> add_recipe(make_recipe(train)) |> add_model(xgb_spec_fold)
  fit <- wf |> fit(data = train)

  preds_all[[i]] <- predict(fit, test) |>
    bind_cols(test |> dplyr::select(EU, subject))
  cat(sprintf("  fold %2d subj %s done (n=%d)\n", i, subj, nrow(test)))
}

preds <- bind_rows(preds_all) |>
  mutate(EU = as.character(EU), .pred_class = as.character(.pred_class),
         subject = as.character(subject)) |>
  group_by(subject) |>
  mutate(row_pos = row_number()) |>
  ungroup()

saveRDS(preds, file.path(OUT, "fp_pattern_predictions.rds"))

# --- Type + boundary flags -------------------------------------------------
preds <- preds |>
  group_by(subject) |>
  mutate(truth_prev  = lag(EU,  n = 1),
         truth_next  = lead(EU, n = 1),
         truth_prev2 = lag(EU,  n = 2),
         truth_next2 = lead(EU, n = 2)) |>
  ungroup() |>
  mutate(
    type = case_when(
      .pred_class == "eating"     & EU == "eating"     ~ "TP",
      .pred_class == "eating"     & EU == "non_eating" ~ "FP",
      .pred_class == "non_eating" & EU == "eating"     ~ "FN",
      .pred_class == "non_eating" & EU == "non_eating" ~ "TN"
    ),
    boundary_1 = EU == "non_eating" & (
      replace_na(truth_prev == "eating", FALSE) |
      replace_na(truth_next == "eating", FALSE)),
    boundary_2 = EU == "non_eating" & (
      replace_na(truth_prev  == "eating", FALSE) |
      replace_na(truth_next  == "eating", FALSE) |
      replace_na(truth_prev2 == "eating", FALSE) |
      replace_na(truth_next2 == "eating", FALSE))
  )

fp_df <- preds |> dplyr::filter(type == "FP")
tn_df <- preds |> dplyr::filter(type == "TN")
total_fp <- nrow(fp_df); total_tn <- nrow(tn_df)
total_ne <- total_fp + total_tn

cat("\n=== FP pattern summary (v2, harmonised pipeline) ===\n")
cat("Total non-eating IUs:", total_ne, "\n")
cat(sprintf("  FP:  %d (%.1f%%)\n", total_fp, 100*total_fp/total_ne))
cat(sprintf("  TN:  %d (%.1f%%)\n\n", total_tn, 100*total_tn/total_ne))

fp_b1 <- sum(fp_df$boundary_1, na.rm = TRUE)
fp_b2 <- sum(fp_df$boundary_2, na.rm = TRUE)
tn_b1 <- sum(tn_df$boundary_1, na.rm = TRUE)
tn_b2 <- sum(tn_df$boundary_2, na.rm = TRUE)

enr1 <- (fp_b1/total_fp) / (tn_b1/total_tn)
enr2 <- (fp_b2/total_fp) / (tn_b2/total_tn)

cat(sprintf("FP ±1 IU (±5 s):  %d / %d = %.1f%%  vs TN %.1f%%  (enrichment %.2f-fold)\n",
            fp_b1, total_fp, 100*fp_b1/total_fp, 100*tn_b1/total_tn, enr1))
cat(sprintf("FP ±2 IU (±10 s): %d / %d = %.1f%%  vs TN %.1f%%  (enrichment %.2f-fold)\n",
            fp_b2, total_fp, 100*fp_b2/total_fp, 100*tn_b2/total_tn, enr2))

tab1 <- matrix(c(fp_b1, total_fp - fp_b1,
                 tn_b1, total_tn - tn_b1),
               nrow = 2, byrow = TRUE,
               dimnames = list(c("FP", "TN"),
                               c("boundary", "not_boundary")))
ct1 <- chisq.test(tab1)
cat(sprintf("\nChi-square (FP vs TN, ±1 IU): X^2 = %.1f, df = %d, p = %.2e\n",
            ct1$statistic, ct1$parameter, ct1$p.value))

fp_iso <- fp_df |> mutate(iso = !boundary_2) |>
  summarise(n_iso = sum(iso, na.rm = TRUE),
            pct_iso = 100 * mean(iso, na.rm = TRUE))
cat(sprintf("\nIsolated FPs (no eating within ±10 s): %d / %d = %.1f%%\n",
            fp_iso$n_iso, total_fp, fp_iso$pct_iso))

# --- Per-subject table ----------------------------------------------------
per_subj_fp <- preds |>
  group_by(subject) |>
  summarise(n_iu      = n(),
            n_eating  = sum(EU == "eating"),
            n_non_eat = sum(EU == "non_eating"),
            n_fp      = sum(type == "FP"),
            n_fp_b1   = sum(type == "FP" & boundary_1),
            fp_rate   = n_fp / n_non_eat,
            fp_b1_frac_of_fp = n_fp_b1 / pmax(n_fp, 1),
            .groups   = "drop")
write_csv(per_subj_fp, file.path(OUT, "fp_pattern_per_subject.csv"))

summary_out <- tibble(
  metric = c("Total non-eating IUs",
             "False positives (FP)",
             "FP rate (1 - specificity, IU-pooled)",
             "FP adjacent to eating ±1 IU (%)",
             "TN adjacent to eating ±1 IU (%)",
             "Boundary enrichment (FP vs TN, ±1 IU)",
             "FP adjacent to eating ±2 IU (%)",
             "TN adjacent to eating ±2 IU (%)",
             "Boundary enrichment (FP vs TN, ±2 IU)",
             "Isolated FP (not within ±10 s of any eating) (%)",
             "Chi-square p-value (boundary ±1)"),
  value = c(total_ne, total_fp,
            round(total_fp/total_ne, 3),
            round(100*fp_b1/total_fp, 1),
            round(100*tn_b1/total_tn, 1),
            round(enr1, 2),
            round(100*fp_b2/total_fp, 1),
            round(100*tn_b2/total_tn, 1),
            round(enr2, 2),
            round(as.numeric(fp_iso$pct_iso), 1),
            format.pval(ct1$p.value, digits = 2))
)
write_csv(summary_out, file.path(OUT, "fp_pattern_summary.csv"))

cat("\nSaved:\n",
    " -", file.path(OUT, "fp_pattern_predictions.rds"), "\n",
    " -", file.path(OUT, "fp_pattern_per_subject.csv"), "\n",
    " -", file.path(OUT, "fp_pattern_summary.csv"), "\n")
