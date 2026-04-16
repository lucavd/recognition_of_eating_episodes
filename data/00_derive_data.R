# ---------------------------------------------------------------------------
# Derive all datasets from the public denotion R package.
#
# This script reconstructs the continuous 5 Hz right-wrist sensor stream
# from the denotion package, then computes the Information Unit (IU) feature
# matrix used throughout the paper.
#
# Prerequisites:
#   install.packages(c("tidyverse", "signal"))
#   remotes::install_github("UBESP-DCTV/denotion")
#
# Outputs (saved to data/):
#   dat_all.rds               Main IU feature matrix (delta_s = 1-5 s, 19 subjects)
#   iu_features_unfiltered.rds  Same pipeline, for S7 Table ablation
#   iu_features_filtered.rds    With Butterworth 0.3 Hz, for S7 Table ablation
#   feature_list.csv            75 predictor names
#
# Runtime: ~5 minutes on a modern laptop.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(denotion)
})
stopifnot(requireNamespace("signal", quietly = TRUE))

OUT <- if (file.exists("data")) "data" else "."

# ── 1. Reconstruct continuous right-wrist stream from denotion ──────────────

cat("Loading denotion package data...\n")
data("denotion", package = "denotion")

d1 <- denotion[denotion$delta_s == 1, ]
cat("Reconstructing continuous stream from", nrow(d1), "segments...\n")

channels <- c("acc_x", "acc_y", "acc_z", "pitch", "roll", "power", "total_energy")

raw_list <- vector("list", nrow(d1))
for (i in seq_len(nrow(d1))) {
  nd <- d1$nested_data[[i]]
  right_samples <- do.call(rbind, lapply(nd, function(w) w[w$wrist == "right", ]))
  right_samples <- right_samples[!duplicated(right_samples$video_time), ]
  right_samples <- right_samples[order(right_samples$video_time), ]
  right_samples$arm     <- d1$arm[i]
  right_samples$subject <- d1$subject[i]
  right_samples$meal    <- d1$meal[i]
  raw_list[[i]] <- right_samples
}
raw <- bind_rows(raw_list)
raw <- raw |>
  select(arm, subject, meal, food, video_time,
         all_of(channels), eating_union, eating_intersect) |>
  arrange(subject, meal, food, video_time)

cat("Continuous right-wrist stream:", nrow(raw), "samples,",
    length(unique(raw$subject)), "subjects\n\n")

# ── 2. Window statistics function ───────────────────────────────────────────

STATS <- function(x) c(
  mean = mean(x), med = median(x), min = min(x), max = max(x),
  iqr = IQR(x), sd = sd(x),
  cv = if (abs(mean(x)) > 1e-8) sd(x) / abs(mean(x)) else NA_real_
)

window_features <- function(w) {
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
           prop_eating_union = mean(w$eating_union),
           prop_eating_intersect = mean(w$eating_intersect, na.rm = TRUE),
           n_samples = as.numeric(nrow(w)))
  out
}

# ── 3. Sliding-window pipeline ──────────────────────────────────────────────

bf <- signal::butter(n = 2, W = 0.3 / 2.5, type = "low")

make_iu <- function(segment_df, window_size = 25, stride = 5, do_filter = FALSE) {
  if (do_filter && nrow(segment_df) >= 20) {
    for (ch in channels) {
      segment_df[[ch]] <- as.numeric(signal::filtfilt(bf, segment_df[[ch]]))
    }
  }
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

run_pipeline <- function(data, window_size, stride = 5, do_filter = FALSE) {
  segments <- data |> group_by(subject, meal, food) |> group_split()
  iu_list <- lapply(segments, function(sg) make_iu(sg, window_size, stride, do_filter))
  iu <- bind_rows(Filter(Negate(is.null), iu_list))
  iu$EU <- factor(ifelse(iu$prop_eating_union > 0.5, "eating", "non_eating"))
  iu$EI <- factor(ifelse(iu$prop_eating_intersect > 0.5, "eating", "non_eating"))
  iu$delta_s <- window_size / 5
  iu$prop_eating_union <- NULL
  iu$prop_eating_intersect <- NULL
  iu$n_samples <- NULL
  iu
}

# ── 4. Produce main dataset (all delta_s = 1–5) ────────────────────────────

# Exclude subject 02 (left-handed)
raw19 <- raw |> filter(subject != "02")

dat_list <- vector("list", 5)
for (ds in 1:5) {
  ws <- ds * 5  # window size in samples (5 Hz)
  cat("Building IU features for delta_s =", ds, "s (window =", ws, "samples)...\n")
  t0 <- Sys.time()
  dat_list[[ds]] <- run_pipeline(raw19, window_size = ws, stride = 5, do_filter = FALSE)
  cat("  ", nrow(dat_list[[ds]]), "IUs in",
      round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
}
dat_all <- bind_rows(dat_list)

saveRDS(dat_all, file.path(OUT, "dat_all.rds"))
cat("  Saved:", file.path(OUT, "dat_all.rds"), "(", nrow(dat_all), "total IUs )\n\n")

# ── 5. Filter ablation datasets (S7 Table) ─────────────────────────────────

cat("Building unfiltered IU features (S7 Table)...\n")
t0 <- Sys.time()
iu_unf <- run_pipeline(raw19, window_size = 25, stride = 5, do_filter = FALSE)
cat("  ", nrow(iu_unf), "IUs in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")

cat("Building filtered IU features (Butterworth 0.3 Hz, S7 Table)...\n")
t0 <- Sys.time()
iu_flt <- run_pipeline(raw19, window_size = 25, stride = 5, do_filter = TRUE)
cat("  ", nrow(iu_flt), "IUs in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")

saveRDS(iu_unf, file.path(OUT, "iu_features_unfiltered.rds"))
saveRDS(iu_flt, file.path(OUT, "iu_features_filtered.rds"))
cat("  Saved: iu_features_unfiltered.rds, iu_features_filtered.rds\n\n")

# ── 6. Feature list (S4 Table) ─────────────────────────────────────────────

feature_names <- c(
  channels,
  as.vector(outer(
    c("mean_", "med_", "min_", "max_", "iqr_", "sd_", "cv_"),
    paste0(channels, "_right"),
    paste0
  )),
  "cor_xy", "cor_yz", "cor_zx"
)
feature_df <- tibble(
  category = c(
    rep("slope", 7),
    rep("mean", 7), rep("median", 7), rep("minimum", 7), rep("maximum", 7),
    rep("IQR", 7), rep("SD", 7), rep("CV", 7),
    rep("cross-correlation", 3)
  ),
  feature = feature_names
)
write_csv(feature_df, file.path(OUT, "feature_list.csv"))
cat("Saved: feature_list.csv (", nrow(feature_df), "features)\n\n")

# ── Summary ─────────────────────────────────────────────────────────────────

cat("=== Done ===\n")
dat5 <- dat_all[dat_all$delta_s == 5, ]
cat("Subjects:", paste(sort(unique(dat5$subject)), collapse = ", "), "\n")
cat("Total IUs (all delta_s):", nrow(dat_all), "\n")
cat("IUs (delta_s=5):", nrow(dat5), "\n")
cat("Eating prevalence (EU, delta_s=5):", round(mean(dat5$EU == "eating") * 100, 1), "%\n")
cat("Columns:", ncol(dat_all), "\n")
