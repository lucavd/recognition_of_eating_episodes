# ---------------------------------------------------------------------------
# C3 — Retrospective minimum-detectable-effect (MDE) analysis. Uses the
# between-subject SDs of the per-subject LOSO metrics of the best-performing
# classifier (XGBoost tuned, N = 19) as the best local estimate of the
# between-subject variance that drives uncertainty of the aggregated metrics.
#
# For a paired within-subjects design at alpha = 0.05 and power = 0.80,
# MDE ≈ (t_{alpha/2, N-1} + t_{power, N-1}) * SD / sqrt(N).
#
# Output: manuscript/review/outputs/mde_C3.csv
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

d <- read_csv(file.path(PROJ, "results",
                        "subject_metrics_tuned.csv"),
              show_col_types = FALSE) |>
  filter(model == "XGB_tuned") |>
  mutate(bal_acc = (sens + spec) / 2) |>
  rename(auc = roc_auc)

sd_vec <- sapply(d[, c("sens", "spec", "bal_acc", "auc")], sd)
cat("Observed per-subject SDs (N = ", nrow(d), "):\n", sep = "")
print(round(sd_vec, 3))

mde <- function(n, sd, alpha = 0.05, power = 0.80) {
  df <- n - 1
  (qt(1 - alpha / 2, df) + qt(power, df)) * sd / sqrt(n)
}

Ns  <- c(19, 30, 50, 100, 200)
res <- expand_grid(N = Ns,
                   metric = c("sens", "spec", "bal_acc", "auc")) |>
  mutate(sd  = sd_vec[metric],
         mde = mde(N, sd))

table_wide <- res |>
  select(-sd) |>
  pivot_wider(names_from = metric, values_from = mde) |>
  mutate(across(c(sens, spec, bal_acc, auc), \(x) round(x, 3)))

cat("\n=== Minimum detectable effect (alpha = 0.05, power = 0.80) ===\n")
print(as.data.frame(table_wide), row.names = FALSE)

write_csv(table_wide, file.path(OUT, "mde_C3.csv"))

cat("\nSaved:", file.path(OUT, "mde_C3.csv"), "\n")
