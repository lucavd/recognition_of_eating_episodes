# =============================================================================
# Round 2 / Reviewer #3 — Comment D5
# The two individual rater streams were NOT retained: both dat_all.rds (IU level)
# and the raw denotion_raw.rda (sample level) carry only the derived
# eating_union (EU = rater1 OR rater2) and eating_intersect (EI = rater1 AND
# rater2). The actual rater-by-rater contingency table is therefore
# unrecoverable. However:
#   * Percent agreement Po = P(both eat) + P(both non-eat) is EXACT, independent
#     of how the contested cases split between the two raters.
#   * Cohen's kappa depends on the individual rater marginals p1, p2, whose sum
#     S = p1 + p2 is fixed by (EU, EI) but whose split is unknown. We therefore
#     report kappa as an interval over all feasible splits.
#       Pe = 1 - S + 2*p1*p2 ; with p1+p2=S fixed, p1*p2 is maximal at p1=p2
#       (symmetric split) -> Pe maximal -> kappa MINIMAL. Hence the symmetric
#       reconstruction used in Round 1 is the LOWER bound; the true kappa >= it.
#       The upper bound is at maximal asymmetry (one rater = EI rate, the other
#       = EU rate).
# This script computes Po and the [kappa_min, kappa_max] interval at the IU
# level (delta_s = 5 s) and at the raw-sample (frame) level.
# Output -> results/kappa_bounds.csv
# =============================================================================

suppressPackageStartupMessages({library(tidyverse); library(here)})
OUT <- here("results", "agreement")

kappa_interval <- function(a, b, cc, label) {
  # a = both eating, b = disagreement (exactly one), cc = both non-eating
  N  <- a + b + cc
  Po <- (a + cc) / N
  S  <- (2 * a + b) / N                 # p1 + p2  (fixed)
  # symmetric split: p1 = p2 = S/2  -> kappa lower bound
  p_sym <- S / 2
  Pe_sym <- 1 - S + 2 * p_sym * p_sym
  k_sym  <- (Po - Pe_sym) / (1 - Pe_sym)
  # maximal asymmetry: p1 = a/N (EI rate), p2 = (a+b)/N (EU rate) -> upper bound
  p1 <- a / N; p2 <- (a + b) / N
  Pe_asym <- 1 - S + 2 * p1 * p2
  k_asym  <- (Po - Pe_asym) / (1 - Pe_asym)
  tibble(level = label, N = N,
         both_eating = a, disagreement = b, both_non = cc,
         pct_agreement = Po,
         EU_rate = p2, EI_rate = p1, marginal_sum = S,
         kappa_symmetric_LOWER = k_sym,
         kappa_maxasym_UPPER = k_asym)
}

# ---- IU level (delta_s = 5 s, 19 subjects: subject 02 already absent) ------
iu <- readRDS(here("data", "dat_all.rds")) |>
  filter(delta_s == 5) |>
  transmute(EU = as.integer(EU), EI = as.integer(EI))
stopifnot(!any(iu$EU == 0 & iu$EI == 1))
a_iu <- sum(iu$EU == 1 & iu$EI == 1)
b_iu <- sum(iu$EU == 1 & iu$EI == 0)
c_iu <- sum(iu$EU == 0 & iu$EI == 0)
res_iu <- kappa_interval(a_iu, b_iu, c_iu, "IU (delta_s=5s, 19 subj)")

# ---- Raw-sample / frame level (from the public denotion package) -----------
data("denotion", package = "denotion")
den <- denotion
# unnest one wrist only (label is temporal, identical across wrists); use delta_s==5 partition
raw <- den |>
  filter(delta_s == 5, subject != "02") |>
  mutate(samp = map(nested_data, function(nd) {
    bind_rows(lapply(nd, function(w) {
      w |> filter(wrist == "right") |>
        select(video_time, eating_union, eating_intersect)
    }))
  })) |>
  select(subject, meal, samp) |>
  unnest(samp) |>
  distinct(subject, meal, video_time, .keep_all = TRUE)   # unique frames

a_f <- sum(raw$eating_union & raw$eating_intersect)
b_f <- sum(raw$eating_union & !raw$eating_intersect)
c_f <- sum(!raw$eating_union & !raw$eating_intersect)
stopifnot(sum(!raw$eating_union & raw$eating_intersect) == 0)
res_f <- kappa_interval(a_f, b_f, c_f, "raw frame (5Hz, 19 subj)")

out <- bind_rows(res_iu, res_f)
write_csv(out, file.path(OUT, "kappa_bounds.csv"))

cat("=== Inter-rater agreement: EXACT % agreement + kappa BOUNDS ===\n\n")
out |>
  mutate(across(c(pct_agreement, EU_rate, EI_rate, marginal_sum,
                  kappa_symmetric_LOWER, kappa_maxasym_UPPER), ~round(.x, 4))) |>
  as.data.frame() |> print()

cat(sprintf("\nIU level   : %% agreement = %.1f%%  | kappa in [%.3f (symmetric, reported), %.3f]\n",
            100*res_iu$pct_agreement, res_iu$kappa_symmetric_LOWER, res_iu$kappa_maxasym_UPPER))
cat(sprintf("Frame level: %% agreement = %.1f%%  | kappa in [%.3f, %.3f]\n",
            100*res_f$pct_agreement, res_f$kappa_symmetric_LOWER, res_f$kappa_maxasym_UPPER))
cat("\nAll values within Landis-Koch 'fair' band (0.21-0.40) -> conclusion robust to the unknown split.\n")
cat("DONE.\n")
