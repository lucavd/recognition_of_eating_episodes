# ---------------------------------------------------------------------------
# C5 — Inference-time benchmark for the best-performing classifier
# (XGBoost tuned) in response to Reviewer #2, Comment 5.
#
# The benchmark trains the tuned XGBoost on the full 19-subject pool (to
# obtain an operationally representative model), then times predict_proba
# on a single Information Unit and on a batch of 1000 IUs on the host
# machine. Model size on disk (serialised) is also reported.
#
# Note: the benchmark is run on a commodity CPU (macOS host) and should be
# read as a reference point for the computational envelope of the model,
# not as an estimate of smartwatch on-device latency. The Discussion
# frames the numbers accordingly.
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

cat("=== Loading data ===\n")
dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI, -delta_s)

cf  <- table(dat$EU)
spw <- as.numeric(cf["non_eating"] / cf["eating"])

# Fit XGB tuned config (same as paper_figures_data.rds model; parameters
# from Phase 9 tuned search; trees/depth/learn_rate/min_n representative
# of the selected configuration).
xgb_spec <- boost_tree(
  trees      = 500,
  tree_depth = 10,
  learn_rate = 0.044,
  min_n      = 8
) |>
  set_engine("xgboost", scale_pos_weight = spw, verbosity = 0,
             nthread = 1) |>   # realistic single-thread inference path
  set_mode("classification")

rec <- recipe(EU ~ ., data = dat) |>
  step_rm(subject) |>
  step_mutate(across(c(arm, meal, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

cat("=== Fitting model ===\n")
fit <- workflow() |>
  add_recipe(rec) |>
  add_model(xgb_spec) |>
  fit(data = dat)

# Model binary footprint
mdl <- extract_fit_engine(fit)
tmp <- tempfile(fileext = ".bin")
xgb.save(mdl, tmp)
fsize_kb <- file.info(tmp)$size / 1024
cat("Model size on disk:", round(fsize_kb, 1), "kB\n\n")

# Prepare inference inputs
test_row   <- dat[1, ]
test_batch <- dat[1:1000, ]

# Warm-up (compile path, fill caches)
invisible(predict(fit, test_row, type = "prob"))
invisible(predict(fit, test_batch, type = "prob"))

# Benchmark single-IU latency
n_rep <- 200
times_single_us <- replicate(n_rep, {
  t0 <- Sys.time()
  predict(fit, test_row, type = "prob")
  as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1e6
})

# Benchmark 1000-IU batch
times_batch_ms <- replicate(n_rep, {
  t0 <- Sys.time()
  predict(fit, test_batch, type = "prob")
  as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1e3
})

summary_out <- tibble(
  metric = c("Model file size (kB)",
             "Single-IU latency (median, µs)",
             "Single-IU latency (mean, µs)",
             "Single-IU latency (95th pct, µs)",
             "Batch 1000 IU latency (median, ms)",
             "Per-IU latency in batch (median, µs)",
             "Throughput (IU / second, from batch median)"),
  value = c(round(fsize_kb, 1),
            round(median(times_single_us), 1),
            round(mean(times_single_us), 1),
            round(quantile(times_single_us, 0.95), 1),
            round(median(times_batch_ms), 2),
            round(median(times_batch_ms) * 1000 / 1000, 1),
            round(1000 / (median(times_batch_ms) / 1000), 0))
)

cat("=== Benchmark summary (host CPU, nthread=1) ===\n")
print(as.data.frame(summary_out), row.names = FALSE)

write_csv(summary_out, file.path(OUT, "inference_latency.csv"))
cat("\nSaved:", file.path(OUT, "inference_latency.csv"), "\n")
