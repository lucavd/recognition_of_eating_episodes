# =============================================================================
# Round 2 / Reviewer #3 — Comment D4
# Few-shot personalisation WITHOUT window-overlap leakage.
#
# Reviewer #3: the original few-shot split drew random calibration IUs and
# tested on the remaining same-subject IUs; with 5 s windows at 1 s stride,
# calibration and test windows share up to 4 s of raw signal -> inflated gain.
# (We additionally found the original few-shot recipe left EI in the predictor
#  set -> a second, quasi-circular leak.)
#
# Fix here:
#   * TEMPORAL BLOCKING by meal: calibration shots are drawn ONLY from the
#     target subject's meals {01,02}; evaluation is ONLY on the disjoint meals
#     {03,04}. Different meals are separate recording sessions -> no window
#     overlap between calibration and test.
#   * SENSOR-ONLY features (remove subject, ID, EI, delta_s, arm, meal, food):
#     removes the EI leak and the meal-as-predictor confound; consistent with
#     the D2 sensor-only primary and with deployable (context-free) inference.
#   * Baseline and few-shot are evaluated on the SAME held-out meal block.
#   * eating-only calibration (the best strategy in the original experiment),
#     n in {5,10,20,50,100}, capped at availability.
#
# Output -> results/fewshot_blocked_*.csv
# =============================================================================

suppressPackageStartupMessages({library(tidyverse); library(tidymodels); library(xgboost); library(here)})
set.seed(2026)
OUT <- here("results")

dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"), levels = c("eating", "non_eating")))

CALIB_MEALS <- c("01", "02")
TEST_MEALS  <- c("03", "04")
difficult   <- c("08", "09", "10", "11", "16")
subjects    <- sort(unique(dat$subject))
N_SHOTS     <- c(5, 10, 20, 50, 100)

rec_sensor <- function(tr) {
  recipe(EU ~ ., data = tr) |>
    step_rm(subject, ID, EI, delta_s, arm, meal, food) |>   # sensor-only, no EI leak
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}
xgb_spec <- function(spw) {
  boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
    set_engine("xgboost", scale_pos_weight = spw, verbosity = 0, nthread = parallel::detectCores()) |>
    set_mode("classification")
}
sens_on <- function(fit, test) {
  p <- predict(fit, test) |> bind_cols(test |> select(EU))
  if (sum(p$EU == "eating") == 0) return(NA_real_)
  sensitivity(p, truth = EU, estimate = .pred_class)$.estimate
}

fit_predict <- function(train, test) {
  spw <- as.numeric(table(train$EU)["non_eating"] / table(train$EU)["eating"])
  wf <- workflow() |> add_recipe(rec_sensor(train)) |> add_model(xgb_spec(spw))
  suppressWarnings(sens_on(wf |> fit(data = train), test))
}

rows <- list(); k <- 1
for (s in subjects) {
  others   <- dat |> filter(subject != s)
  test_blk <- dat |> filter(subject == s, meal %in% TEST_MEALS)
  calib_pl <- dat |> filter(subject == s, meal %in% CALIB_MEALS, EU == "eating")
  n_eat_test <- sum(test_blk$EU == "eating")
  if (nrow(test_blk) == 0 || n_eat_test == 0) next

  sens_base <- fit_predict(others, test_blk)            # no personalisation, same test block
  for (n in N_SHOTS) {
    if (nrow(calib_pl) < 1) { sens_fs <- NA_real_; n_used <- 0 } else {
      n_used <- min(n, nrow(calib_pl))
      calib  <- calib_pl |> slice_sample(n = n_used)
      sens_fs <- fit_predict(bind_rows(others, calib), test_blk)
    }
    rows[[k]] <- tibble(subject = s, group = ifelse(s %in% difficult, "difficult", "other"),
                        n_shots = n, n_used = n_used, n_test = nrow(test_blk),
                        n_test_eating = n_eat_test,
                        sens_base = sens_base, sens_fewshot = sens_fs,
                        improvement = sens_fs - sens_base)
    k <- k + 1
  }
  cat(sprintf("subj %s (%s): base=%.3f  fs@100=%.3f  (n_test_eat=%d)\n",
              s, ifelse(s %in% difficult, "diff", "easy"), sens_base,
              rows[[k-1]]$sens_fewshot, n_eat_test))
}

res <- bind_rows(rows)
write_csv(res, file.path(OUT, "fewshot_blocked_per_subject.csv"))

summ <- res |> group_by(group, n_shots) |>
  summarise(mean_base = mean(sens_base, na.rm = TRUE),
            mean_fewshot = mean(sens_fewshot, na.rm = TRUE),
            mean_improvement = mean(improvement, na.rm = TRUE),
            n_subj = n_distinct(subject), .groups = "drop")
write_csv(summ, file.path(OUT, "fewshot_blocked_summary.csv"))

cat("\n=== Few-shot (meal-blocked, sensor-only, eating-only) — mean sensitivity improvement ===\n")
print(as.data.frame(summ |> mutate(across(where(is.numeric), ~round(.x, 3)))))
cat("\nReference (original LEAKY result): difficult subjects +0.176 at 100 eating-only shots.\n")
cat("Saved: fewshot_blocked_per_subject.csv, fewshot_blocked_summary.csv\nDONE.\n")
