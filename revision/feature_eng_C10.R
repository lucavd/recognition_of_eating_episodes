# ---------------------------------------------------------------------------
# C10 — Feature-engineering pipeline to produce Information Units at
# δs = 5 s with and without an upstream Butterworth 0.3 Hz low-pass filter.
#
# The script starts from the raw 5 Hz sensor stream in signals_bite.rda
# (package `notion`) restricted to the right wrist, processes each
# (subject, meal, food) segment by optionally applying a 2nd-order Butterworth
# zero-phase filter (filtfilt, W = 0.3/2.5 = 0.12), slides a 25-sample window
# with a 5-sample stride (5-second window, 1-second stride, 80% overlap),
# and computes for each window the 7 window statistics (mean, median, min,
# max, IQR, SD, CV) for each of the 7 kinetic channels, the slope of each
# channel vs time, and the three axis cross-correlations. Each window is
# labelled with the >50% majority rule on the raw eating_union labels.
#
# Output: manuscript/review/outputs/iu_features_filtered.rds
#         manuscript/review/outputs/iu_features_unfiltered.rds
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})
# Load signal namespace without attaching (avoids filter/IQR clashes)
stopifnot(requireNamespace("signal", quietly = TRUE))

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# Raw data derived from denotion package (see data/00_derive_data.R)
# If you need signals_bite.rda, reconstruct it from denotion::denotion
stop("This script uses signals_bite.rda from the internal notion package.\nUse data/00_derive_data.R instead, which derives the same features from the public denotion package.")

channels <- c("acc_x", "acc_y", "acc_z",
              "pitch", "roll", "power", "total_energy")

# --- Prepare right-wrist stream --------------------------------------------
raw <- signals_bite |>
  dplyr::filter(wrist == "right") |>
  dplyr::select(arm, subject, meal, food, video_time,
                dplyr::all_of(channels), eating_union) |>
  dplyr::arrange(subject, meal, food, video_time)

cat("Raw right-wrist samples:", nrow(raw), "\n")
cat("Subjects:", paste(sort(unique(raw$subject)), collapse = ","), "\n")

# --- Butterworth (0.3 Hz low-pass, 2nd order, zero-phase via filtfilt) -----
bf <- signal::butter(n = 2, W = 0.3 / 2.5, type = "low")

apply_filter <- function(segment_df, do_filter) {
  if (!do_filter || nrow(segment_df) < 20) return(segment_df)
  # filtfilt requires sufficient length; skip short segments
  for (ch in channels) {
    segment_df[[ch]] <- as.numeric(signal::filtfilt(bf, segment_df[[ch]]))
  }
  segment_df
}

# --- Window statistics -----------------------------------------------------
STATS <- function(x) c(
  mean = mean(x), med = median(x), min = min(x), max = max(x),
  iqr = IQR(x), sd = sd(x),
  cv = if (abs(mean(x)) > 1e-8) sd(x) / abs(mean(x)) else NA_real_
)

window_features <- function(w) {
  # w is a 25-row data frame of raw samples. Returns a flat named vector.
  out <- numeric(0)
  for (ch in channels) {
    v <- w[[ch]]
    stats <- STATS(v)
    names(stats) <- paste0(c("mean_", "med_", "min_", "max_",
                             "iqr_", "sd_", "cv_"), ch, "_right")
    slp <- coef(lm.fit(cbind(1, seq_along(v)), v))[2]
    names(slp) <- ch
    out <- c(out, slp, stats)
  }
  cxy <- suppressWarnings(cor(w$acc_x, w$acc_y))
  cyz <- suppressWarnings(cor(w$acc_y, w$acc_z))
  czx <- suppressWarnings(cor(w$acc_z, w$acc_x))
  out <- c(out,
           cor_xy = cxy, cor_yz = cyz, cor_zx = czx,
           prop_eating = mean(w$eating_union),
           n_samples = as.numeric(nrow(w)))
  out
}

# --- Pipeline for one subject-meal-food segment ----------------------------
make_windows <- function(segment_df, do_filter, window_size = 25, stride = 5) {
  segment_df <- apply_filter(segment_df, do_filter)
  n <- nrow(segment_df)
  if (n < window_size) return(NULL)
  starts <- seq(1, n - window_size + 1, by = stride)
  feats <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    s <- starts[i]
    w <- segment_df[s:(s + window_size - 1), ]
    feats[[i]] <- window_features(w)
  }
  fm <- as.data.frame(do.call(rbind, feats), stringsAsFactors = FALSE)
  fm$arm     <- segment_df$arm[1]
  fm$subject <- segment_df$subject[1]
  fm$meal    <- segment_df$meal[1]
  fm$food    <- segment_df$food[1]
  fm
}

pipeline <- function(do_filter) {
  segments <- raw |>
    group_by(subject, meal, food) |>
    group_split()
  iu_list <- lapply(segments, function(sg) make_windows(sg, do_filter))
  iu <- do.call(rbind, Filter(Negate(is.null), iu_list))
  iu
}

cat("\n--- Building UNFILTERED IUs ---\n")
t0 <- Sys.time()
iu_unf <- pipeline(do_filter = FALSE)
cat("Unfiltered IU:", nrow(iu_unf), " time:",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")

cat("\n--- Building FILTERED IUs (Butterworth 0.3 Hz) ---\n")
t0 <- Sys.time()
iu_flt <- pipeline(do_filter = TRUE)
cat("Filtered IU:", nrow(iu_flt), " time:",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")

# --- Labels: EU = 1 if prop_eating > 0.5 -----------------------------------
finalise <- function(df) {
  df$EU <- factor(ifelse(df$prop_eating > 0.5, "eating", "non_eating"))
  df$prop_eating <- NULL
  df$n_samples   <- NULL
  df
}

iu_unf <- finalise(iu_unf)
iu_flt <- finalise(iu_flt)

saveRDS(iu_unf, file.path(OUT, "iu_features_unfiltered.rds"))
saveRDS(iu_flt, file.path(OUT, "iu_features_filtered.rds"))

cat("\nUnfiltered: ", nrow(iu_unf), "rows,",
    ncol(iu_unf), "cols; EU table:\n")
print(table(iu_unf$EU))
cat("\nFiltered: ", nrow(iu_flt), "rows,",
    ncol(iu_flt), "cols; EU table:\n")
print(table(iu_flt$EU))

# =========================================================================
# VALIDATION AGAINST dat_all.rds (δs = 5 s, no subject 02)
# =========================================================================
cat("\n\n========== VALIDATION SUITE ==========\n")
baseline <- readRDS(here("data", "dat_all.rds")) |>
  dplyr::filter(delta_s == 5, subject != "02")
cat("Baseline dat_all.rds (δs=5, excl. 02): nrow =", nrow(baseline), "\n")

test <- iu_unf |> dplyr::filter(subject != "02")

cat("\n--- Check 1: IU count by (subject, meal, food) ---\n")
mine <- test |>
  dplyr::count(subject, meal, food, name = "n_mine")
base <- baseline |>
  dplyr::count(subject, meal, food, name = "n_baseline")
cmp <- dplyr::full_join(mine, base, by = c("subject", "meal", "food")) |>
  dplyr::mutate(delta = n_mine - n_baseline)
cat("Segments with non-null counts on both sides:",
    sum(!is.na(cmp$n_mine) & !is.na(cmp$n_baseline)), "\n")
cat("Segments only in my pipeline:", sum(is.na(cmp$n_baseline)), "\n")
cat("Segments only in baseline:   ", sum(is.na(cmp$n_mine)), "\n")
cat("Delta n_mine - n_baseline: mean =", round(mean(cmp$delta, na.rm=TRUE),2),
    " max abs =", max(abs(cmp$delta), na.rm=TRUE), "\n")
print(head(cmp |> dplyr::arrange(dplyr::desc(abs(delta))), 6))

cat("\n--- Check 2: class balance (EU) ---\n")
p_mine <- mean(test$EU == "eating")
p_base <- mean(baseline$EU == 1)
cat(sprintf("Mine (unfiltered):    %.4f eating\nBaseline dat_all.rds: %.4f eating\n",
            p_mine, p_base))

cat("\n--- Check 3: per-subject IU count ---\n")
by_subj <- dplyr::full_join(
  test     |> dplyr::count(subject, name = "mine"),
  baseline |> dplyr::count(subject, name = "base"),
  by = "subject"
)
print(as.data.frame(by_subj))
cat("Correlation mine vs base:",
    round(cor(by_subj$mine, by_subj$base, use="pairwise.complete.obs"),4), "\n")

cat("\n--- Check 4: feature means (mean_acc_x_right, etc.) ---\n")
feats_check <- c("mean_acc_x_right", "mean_acc_y_right",
                 "sd_acc_x_right", "max_power_right",
                 "iqr_roll_right", "cor_xy")
for (f in feats_check) {
  if (!(f %in% names(test))) { cat("  MISSING:", f, "\n"); next }
  m_mean <- mean(test[[f]], na.rm = TRUE)
  b_mean <- mean(baseline[[f]], na.rm = TRUE)
  rel    <- (m_mean - b_mean) / (abs(b_mean) + 1e-8)
  cat(sprintf("  %-22s  mine=%.4f  base=%.4f  rel.diff=%+.3f\n",
              f, m_mean, b_mean, rel))
}

cat("\n--- Check 5: filter effect on a single feature (SD should drop) ---\n")
cat(sprintf("mean sd_acc_x_right  UNFILTERED: %.2f\n",
            mean(iu_unf$sd_acc_x_right, na.rm = TRUE)))
cat(sprintf("mean sd_acc_x_right  FILTERED:   %.2f\n",
            mean(iu_flt$sd_acc_x_right, na.rm = TRUE)))

cat("\nSaved to", OUT, "\n")
