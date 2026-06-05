# ---------------------------------------------------------------------------
# C2 — Sub-group analysis by study arm (Menu A vs Menu B), on the XGBoost
# (tuned) LOSO per-subject metrics. Response to Reviewer #2, Comment 2.
#
# Subjects are randomised to one of two fixed standardised menus (Menu A:
# 10 subjects; Menu B: 9 subjects, after exclusion of subject 02). Each
# menu comprises a different set of food items across four meals. Although
# menu assignment is not a cultural-variability proxy, it provides an
# internal test of whether performance depends on the specific food items
# consumed — a non-trivial check given that the feature set includes food-
# item dummies.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

set.seed(1812)
B_BOOT <- 1000

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

# --- Subject → arm mapping from the NOTION data ----------------------------
subject_arm <- readRDS(here("data", "dat_all.rds")) |>
  distinct(subject, arm) |>
  mutate(subject = sprintf("%02d", as.integer(subject)),
         arm = factor(arm, levels = c("a", "b"),
                      labels = c("Menu A", "Menu B")))

cat("=== Subject → arm mapping ===\n")
print(subject_arm)
cat("\nn per arm:\n"); print(table(subject_arm$arm))

# --- Load per-subject metrics for XGB (tuned) -----------------------------
tuned <- read_csv(file.path(PROJ, "results",
                            "subject_metrics_tuned.csv"),
                  show_col_types = FALSE) |>
  filter(model == "XGB_tuned") |>
  mutate(subject_raw = as.integer(subject))

# subject_raw in tuned CSV: integers 1..19; note that subject 02 is already
# excluded from the full subject list, so 1..19 maps to 01,03,...,20.
ordered_ids <- c("01","03","04","05","06","07","08","09","10","11","12","13",
                 "14","15","16","17","18","19","20")
tuned <- tuned |>
  mutate(subject = ordered_ids[subject_raw],
         bal_acc = (sens + spec) / 2) |>
  rename(auc = roc_auc) |>
  select(subject, sens, spec, bal_acc, auc)

dat <- tuned |> inner_join(subject_arm, by = "subject")

cat("\n=== Joined (XGB tuned per-subject + arm) ===\n")
print(dat)

metrics <- c("sens", "spec", "bal_acc", "auc")

boot_one_group <- function(x, B = B_BOOT) {
  n <- length(x)
  reps <- replicate(B, mean(sample(x, n, replace = TRUE)))
  tibble(mean = mean(x),
         ci_low  = quantile(reps, 0.025, na.rm = TRUE),
         ci_high = quantile(reps, 0.975, na.rm = TRUE))
}

per_arm <- map_dfr(levels(dat$arm), function(a) {
  d_a <- dat |> filter(arm == a)
  map_dfr(metrics, function(m) {
    boot_one_group(d_a[[m]]) |>
      mutate(arm = a, metric = m, n = nrow(d_a), .before = 1)
  })
})

cat("\n=== Mean [95% CI] by arm ===\n")
print(as.data.frame(per_arm |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", mean, ci_low, ci_high)) |>
  select(arm, metric, n, cell)))

# Between-group difference (A − B) -----------------------------------------
boot_diff <- function(grp_a, grp_b, B = B_BOOT) {
  reps <- replicate(B, {
    a_resample <- sample(grp_a, length(grp_a), replace = TRUE)
    b_resample <- sample(grp_b, length(grp_b), replace = TRUE)
    mean(a_resample) - mean(b_resample)
  })
  tibble(diff = mean(grp_a) - mean(grp_b),
         ci_low  = quantile(reps, 0.025, na.rm = TRUE),
         ci_high = quantile(reps, 0.975, na.rm = TRUE),
         p_two_sided = 2 * min(mean(reps > 0, na.rm = TRUE),
                                mean(reps < 0, na.rm = TRUE)))
}

diff_tbl <- map_dfr(metrics, function(m) {
  grp_a <- dat[[m]][dat$arm == "Menu A"]
  grp_b <- dat[[m]][dat$arm == "Menu B"]
  boot_diff(grp_a, grp_b) |> mutate(metric = m, .before = 1)
})

cat("\n=== Difference Menu A − Menu B [95% CI] ===\n")
print(as.data.frame(diff_tbl |>
  mutate(cell = sprintf("%+.3f [%+.3f, %+.3f]  p=%.3f",
                        diff, ci_low, ci_high, p_two_sided)) |>
  select(metric, cell)))

write_csv(per_arm,  file.path(OUT, "subgroup_arm.csv"))
write_csv(diff_tbl, file.path(OUT, "subgroup_arm_diff.csv"))

cat("\nSaved:\n",
    " -", file.path(OUT, "subgroup_arm.csv"),      "\n",
    " -", file.path(OUT, "subgroup_arm_diff.csv"), "\n")
