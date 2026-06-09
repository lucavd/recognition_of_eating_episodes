# =============================================================================
# Inter-rater agreement from the two individual rater streams (exact)
#
# The two video evaluators annotated the eating bites independently. Their two
# per-frame eating indicators (5 Hz) are provided in data/rater_streams.rds,
# reconstructed from the original NOTION labelling pipeline with the project's
# own labelling rule and verified exactly against the public labels: the union
# of the two streams reproduces eating_union with no discrepancy, and the
# Information Unit windowing below reproduces dat_all.rds exactly (26,304 IUs,
# identical eating_union / eating_intersect prevalences).
#
# Columns of data/rater_streams.rds (19 subjects, subject 02 excluded;
# 133,040 frames, i.e. the frames that enter at least one Information Unit):
#   subject, meal, video_time, eat_rater1, eat_rater2, eating_union,
#   eating_intersect
#   - eat_rater1 = the more inclusive evaluator; eat_rater2 = the more
#     conservative evaluator (both logical, per 5 Hz frame).
#
# This script reports the actual rater-by-rater 2x2 contingency table and the
# exact Cohen's kappa at the Information Unit level (delta_s = 5 s) and at the
# raw-frame level, plus the per-subject values.
#
# Outputs -> results/agreement/inter_rater_exact.csv
#            results/agreement/per_subject_agreement.csv
# =============================================================================

suppressPackageStartupMessages({library(tidyverse); library(here)})
OUT <- here("results", "agreement")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

fr <- readRDS(here("data", "rater_streams.rds"))

# ---- exact 2x2 + Cohen's kappa for two logical vectors ---------------------
kappa_exact <- function(a, b, label) {
  both1 <- sum(a & b); r1 <- sum(a & !b); r2 <- sum(!a & b); both0 <- sum(!a & !b)
  N  <- both1 + r1 + r2 + both0
  Po <- (both1 + both0) / N
  p1 <- mean(a); p2 <- mean(b)
  Pe <- p1 * p2 + (1 - p1) * (1 - p2)
  k  <- (Po - Pe) / (1 - Pe)
  band <- cut(k, c(-Inf, .20, .40, .60, .80, Inf),
              c("slight", "fair", "moderate", "substantial", "almost perfect"))
  tibble(level = label, N = N,
         both_eating = both1, rater1_only = r1, rater2_only = r2, both_non = both0,
         pct_agreement = Po, p_rater1 = p1, p_rater2 = p2,
         pe = Pe, kappa = k, landis_koch = as.character(band))
}

# ---- Information Unit level: delta_s = 5 s windowing -----------------------
# 25 frames per window (5 s at 5 Hz), 1 s stride (5 frames), within (subject, meal).
# This reproduces dat_all.rds exactly (26,304 IUs).
DELTA <- 5L; BLOCK <- 5L; W <- DELTA * BLOCK
window_majority <- function(df) {
  df <- arrange(df, video_time); n <- nrow(df)
  if (n < W) return(NULL)
  starts <- seq(1L, n - W + 1L, by = BLOCK)
  maj <- function(x, s) mean(x[s:(s + W - 1L)]) > 0.5
  tibble(
    r1 = vapply(starts, function(s) maj(df$eat_rater1, s), logical(1)),
    r2 = vapply(starts, function(s) maj(df$eat_rater2, s), logical(1)),
    eu = vapply(starts, function(s) maj(df$eating_union, s), logical(1)),
    ei = vapply(starts, function(s) maj(df$eating_intersect, s), logical(1))
  )
}
iu <- fr |> group_by(subject, meal) |> group_modify(~window_majority(.x)) |> ungroup()

# sanity: windowing must reproduce the published Information Unit dataset
stopifnot(nrow(iu) == 26304)
message(sprintf("IU windowing check: N=%d, EU=%.4f (dat_all 0.2692), EI=%.4f (dat_all 0.0808)",
                nrow(iu), mean(iu$eu), mean(iu$ei)))

res_iu <- kappa_exact(iu$r1, iu$r2, "IU (delta_s=5s, 19 subj)")
res_fr <- kappa_exact(fr$eat_rater1, fr$eat_rater2, "raw frame (5Hz, 19 subj)")
out <- bind_rows(res_iu, res_fr)
write_csv(out, file.path(OUT, "inter_rater_exact.csv"))

# ---- per-subject (Information Unit level) ----------------------------------
per_subject <- iu |>
  group_by(subject) |>
  summarise(
    n_iu = n(),
    n_both_e  = sum(r1 & r2),
    n_both_ne = sum(!r1 & !r2),
    n_r1_only = sum(r1 & !r2),
    n_r2_only = sum(!r1 & r2),
    pct_agree = mean(r1 == r2),
    p_r1_e = mean(r1), p_r2_e = mean(r2),
    .groups = "drop"
  ) |>
  mutate(pe = p_r1_e * p_r2_e + (1 - p_r1_e) * (1 - p_r2_e),
         kappa = (pct_agree - pe) / (1 - pe))
write_csv(per_subject, file.path(OUT, "per_subject_agreement.csv"))

# ---- report ----------------------------------------------------------------
cat("=== Inter-rater agreement: actual 2x2 + EXACT Cohen's kappa ===\n\n")
out |> mutate(across(c(pct_agreement, p_rater1, p_rater2, pe, kappa), ~round(.x, 4))) |>
  as.data.frame() |> print()
cat(sprintf("\nIU level   : %% agreement = %.1f%%  | kappa = %.3f (%s)\n",
            100 * res_iu$pct_agreement, res_iu$kappa, res_iu$landis_koch))
cat(sprintf("Frame level: %% agreement = %.1f%%  | kappa = %.3f (%s)\n",
            100 * res_fr$pct_agreement, res_fr$kappa, res_fr$landis_koch))
cat(sprintf("\nDirection: of the %d contested IUs, %d (%.0f%%) were rater-1-only.\n",
            res_iu$rater1_only + res_iu$rater2_only, res_iu$rater1_only,
            100 * res_iu$rater1_only / (res_iu$rater1_only + res_iu$rater2_only)))
cat(sprintf("Per-subject kappa range: %.2f to %.2f; agreement %.0f%% to %.0f%%.\n",
            min(per_subject$kappa), max(per_subject$kappa),
            100 * min(per_subject$pct_agree), 100 * max(per_subject$pct_agree)))
cat("DONE.\n")
