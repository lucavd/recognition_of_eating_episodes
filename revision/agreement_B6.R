# ---------------------------------------------------------------------------
# B6 — Inter-rater agreement at the IU level (delta_s = 5 s) and overall.
#
# The NOTION labelling protocol produces two independent binary streams
# (one per rater) at the raw-sample level. In the aggregated dataset
# `dat_all.rds` each row is an Information Unit (IU) carrying two summary
# labels:
#   EU = 1 if >50% of the raw samples in the window were labelled 'eating'
#        by AT LEAST ONE rater  (Eating Union);
#   EI = 1 if >50% of the raw samples in the window were labelled 'eating'
#        by BOTH raters           (Eating Intersect).
# For every IU this gives a three-cell classification:
#   EU=1 & EI=1  → both raters agree on 'eating'
#   EU=0 & EI=0  → both raters agree on 'non-eating'
#   EU=1 & EI=0  → raters disagree (at least one rater 'eating', not both)
#   EU=0 & EI=1  → impossible by construction
# From these counts we compute percent agreement and Cohen's kappa at the
# IU level.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

dat <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  select(subject, arm, meal, food, EU, EI)

stopifnot(!any(dat$EU == 0 & dat$EI == 1))   # logical sanity

dat <- dat |>
  mutate(
    class = case_when(
      EU == 1 & EI == 1 ~ "both_eating",
      EU == 1 & EI == 0 ~ "disagreement",
      EU == 0 & EI == 0 ~ "both_non_eating"
    )
  )

# ---- Overall agreement -----------------------------------------------------
overall <- dat |>
  count(class) |>
  mutate(pct = 100 * n / sum(n))

cat("\n=== Overall IU agreement (delta_s = 5 s, 19 subjects, N = ",
    nrow(dat), ") ===\n", sep = "")
print(overall)

# Percent agreement
pa <- (sum(dat$class == "both_eating") +
       sum(dat$class == "both_non_eating")) / nrow(dat)

# Cohen's kappa treating the two raters as equally observable binary labels
# Reconstructed 2x2 table:
#   rater1 \ rater2    eating         non-eating
#   eating             both_eating    disagreement/2
#   non-eating         disagreement/2 both_non_eating
# This assumes disagreements split evenly between the two combinations
# (rater1 eat / rater2 non-eat) and (rater1 non-eat / rater2 eat). Under
# this symmetrical allocation we obtain the 2x2 contingency needed for
# kappa, with the observation that marginal rates are preserved.
n_both_e  <- sum(dat$class == "both_eating")
n_both_ne <- sum(dat$class == "both_non_eating")
n_disagr  <- sum(dat$class == "disagreement")
n_total   <- nrow(dat)
n11 <- n_both_e
n00 <- n_both_ne
n10 <- n_disagr / 2   # rater1 eating, rater2 non-eating
n01 <- n_disagr / 2   # rater1 non-eating, rater2 eating

# Marginals
p_r1_e <- (n11 + n10) / n_total
p_r2_e <- (n11 + n01) / n_total
pe     <- p_r1_e * p_r2_e + (1 - p_r1_e) * (1 - p_r2_e)
kappa  <- (pa - pe) / (1 - pe)

cat("\nPercent agreement (IU level):", sprintf("%.4f", pa), "\n")
cat("Cohen's kappa (IU level, symmetric disagreement assumption):",
    sprintf("%.4f", kappa), "\n")

# ---- Per-subject agreement -------------------------------------------------
per_subj <- dat |>
  group_by(subject) |>
  summarise(
    n_iu       = n(),
    n_both_e   = sum(class == "both_eating"),
    n_both_ne  = sum(class == "both_non_eating"),
    n_disagr   = sum(class == "disagreement"),
    pct_agree  = (n_both_e + n_both_ne) / n_iu,
    .groups    = "drop"
  ) |>
  mutate(
    p_r1_e = (n_both_e + n_disagr/2) / n_iu,
    p_r2_e = (n_both_e + n_disagr/2) / n_iu,
    pe     = p_r1_e * p_r2_e + (1 - p_r1_e) * (1 - p_r2_e),
    kappa  = (pct_agree - pe) / (1 - pe)
  )

cat("\n=== Per-subject agreement ===\n")
print(as.data.frame(per_subj |>
  mutate(across(c(pct_agree, kappa), \(x) round(x, 3))) |>
  select(subject, n_iu, n_both_e, n_both_ne, n_disagr, pct_agree, kappa)))

write_csv(per_subj, file.path(OUT, "per_subject_agreement.csv"))

# ---- Distribution of disagreement per subject ------------------------------
cat("\n\nPct agreement summary across subjects:\n")
cat(sprintf("  min=%.3f  median=%.3f  mean=%.3f  max=%.3f\n",
            min(per_subj$pct_agree), median(per_subj$pct_agree),
            mean(per_subj$pct_agree), max(per_subj$pct_agree)))
cat("Kappa summary across subjects:\n")
cat(sprintf("  min=%.3f  median=%.3f  mean=%.3f  max=%.3f\n",
            min(per_subj$kappa), median(per_subj$kappa),
            mean(per_subj$kappa), max(per_subj$kappa)))

# ---- Save a compact summary for the manuscript ----------------------------
summary_out <- tibble(
  metric = c("N IUs", "Both eating", "Both non-eating", "Disagreement",
             "Percent agreement", "Cohen's kappa (IU level)"),
  value  = c(
    n_total,
    n_both_e,
    n_both_ne,
    n_disagr,
    round(pa, 4),
    round(kappa, 4)
  )
)
write_csv(summary_out, file.path(OUT, "agreement_summary.csv"))
print(summary_out)

cat("\nSaved:\n",
    " -", file.path(OUT, "per_subject_agreement.csv"), "\n",
    " -", file.path(OUT, "agreement_summary.csv"),     "\n")
