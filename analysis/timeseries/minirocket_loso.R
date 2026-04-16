# MiniRocket per Time Series Classification
# Fase 8.1: Estrazione features da time series raw + LOSO

library(tidyverse)
library(tidymodels)
library(here)

set.seed(2026)

cat("=== FASE 8.1: MINIROCKET TIME SERIES ===\n\n")

# =============================================================================
# 1. CARICA E PREPARA DATI RAW
# =============================================================================

cat("1. Caricamento dati raw...\n")
data("denotion", package = "denotion")

# Filtra delta_s = 5 (finestre 5 secondi = 25 campioni a 5Hz)
dat_5s <- denotion |> filter(delta_s == 5)
cat("   Finestre delta_s=5:", nrow(dat_5s), "\n")

# Funzione per estrarre time series da nested data
extract_timeseries <- function(nested_data, subject, arm, meal) {
  if (is.null(nested_data) || length(nested_data) == 0) return(NULL)
  
  map_dfr(seq_along(nested_data), function(i) {
    ts_data <- nested_data[[i]]
    if (is.null(ts_data) || nrow(ts_data) == 0) return(NULL)
    
    # Solo wrist = right (dominante per eating)
    ts_right <- ts_data |> filter(wrist == "right")
    if (nrow(ts_right) == 0) return(NULL)
    
    # Label: maggioranza eating_union nella finestra
    eating_label <- mean(ts_right$eating_union, na.rm = TRUE) > 0.5
    
    tibble(
      subject = subject,
      arm = arm,
      meal = meal,
      window_id = i,
      EU = factor(ifelse(eating_label, "eating", "non_eating")),
      # Time series come liste
      ts_acc_x = list(ts_right$acc_x),
      ts_acc_y = list(ts_right$acc_y),
      ts_acc_z = list(ts_right$acc_z),
      ts_pitch = list(ts_right$pitch),
      ts_roll = list(ts_right$roll),
      ts_power = list(ts_right$power)
    )
  })
}

cat("2. Estrazione time series...\n")
ts_data <- map2_dfr(
  dat_5s$nested_data,
  seq_len(nrow(dat_5s)),
  function(nd, idx) {
    extract_timeseries(
      nd,
      dat_5s$subject[idx],
      dat_5s$arm[idx],
      dat_5s$meal[idx]
    )
  },
  .progress = TRUE
)

cat("   Finestre estratte:", nrow(ts_data), "\n")
cat("   Eating:", sum(ts_data$EU == "eating"), "| Non-eating:", sum(ts_data$EU == "non_eating"), "\n")
cat("   Soggetti:", n_distinct(ts_data$subject), "\n")

# =============================================================================
# 2. ROCKET/MINIROCKET FEATURES (implementazione semplificata)
# =============================================================================

cat("\n3. Estrazione features time series (Rocket-like)...\n")

# Funzione per estrarre features da una singola time series
extract_ts_features <- function(ts) {
  if (is.null(ts) || length(ts) < 3) {
    return(rep(NA, 20))
  }
  ts <- as.numeric(ts)
  n <- length(ts)
  
  # Features statistiche base
  f_mean <- mean(ts, na.rm = TRUE)
  f_sd <- sd(ts, na.rm = TRUE)
  f_min <- min(ts, na.rm = TRUE)
  f_max <- max(ts, na.rm = TRUE)
  f_range <- f_max - f_min
  f_iqr <- IQR(ts, na.rm = TRUE)
  
  # Features temporali
  f_slope <- if (n > 1) coef(lm(ts ~ seq_along(ts)))[2] else 0
  f_diff_mean <- mean(abs(diff(ts)), na.rm = TRUE)
  f_diff_sd <- sd(diff(ts), na.rm = TRUE)
  
  # Zero crossings
  f_zc <- sum(diff(sign(ts - f_mean)) != 0, na.rm = TRUE)
  
  # Autocorrelazione
  f_acf1 <- if (n > 2) acf(ts, lag.max = 1, plot = FALSE)$acf[2] else 0
  f_acf2 <- if (n > 3) acf(ts, lag.max = 2, plot = FALSE)$acf[3] else 0
  
  # Percentili
  f_p10 <- quantile(ts, 0.1, na.rm = TRUE)
  f_p25 <- quantile(ts, 0.25, na.rm = TRUE)
  f_p75 <- quantile(ts, 0.75, na.rm = TRUE)
  f_p90 <- quantile(ts, 0.9, na.rm = TRUE)
  
  # Energy e entropia
  f_energy <- sum(ts^2, na.rm = TRUE) / n
  f_rms <- sqrt(f_energy)
  
  # Peak features
  f_n_peaks <- sum(diff(sign(diff(ts))) == -2, na.rm = TRUE)
  f_peak_mean <- if (f_n_peaks > 0) mean(ts[which(diff(sign(diff(ts))) == -2) + 1]) else f_mean
  
  c(f_mean, f_sd, f_min, f_max, f_range, f_iqr, f_slope, f_diff_mean, f_diff_sd, 
    f_zc, f_acf1, f_acf2, f_p10, f_p25, f_p75, f_p90, f_energy, f_rms, f_n_peaks, f_peak_mean)
}

feature_names <- c("mean", "sd", "min", "max", "range", "iqr", "slope", "diff_mean", "diff_sd",
                   "zc", "acf1", "acf2", "p10", "p25", "p75", "p90", "energy", "rms", "n_peaks", "peak_mean")

# Estrai features per tutti i segnali
signals <- c("ts_acc_x", "ts_acc_y", "ts_acc_z", "ts_pitch", "ts_roll", "ts_power")

cat("   Estraendo features per", length(signals), "segnali...\n")

features_list <- map(signals, function(sig) {
  sig_name <- gsub("ts_", "", sig)
  cat("   -", sig_name, "\n")
  
  feat_matrix <- map(ts_data[[sig]], extract_ts_features)
  feat_df <- do.call(rbind, feat_matrix) |> as.data.frame()
  names(feat_df) <- paste0(sig_name, "_", feature_names)
  feat_df
})

# Combina tutte le features
features_df <- bind_cols(features_list)
cat("   Features totali:", ncol(features_df), "\n")

# Dataset finale
dat_final <- bind_cols(
  ts_data |> select(subject, EU),
  features_df
) |>
  drop_na()

cat("   Righe finali (no NA):", nrow(dat_final), "\n")

# =============================================================================
# 3. LOSO VALIDATION
# =============================================================================

cat("\n4. LOSO Validation con XGBoost...\n")

library(xgboost)

subjects <- unique(dat_final$subject)
cat("   Soggetti:", length(subjects), "\n")

loso_results <- map_dfr(subjects, function(subj) {
  test <- dat_final |> filter(subject == subj)
  train <- dat_final |> filter(subject != subj)
  
  if (nrow(test) == 0 || sum(train$EU == "eating") < 5) {
    return(tibble(subject = subj, sens = NA, spec = NA, auc = NA))
  }
  
  # Class weights
  class_freq <- table(train$EU)
  spw <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
  
  # Recipe
  rec <- recipe(EU ~ ., data = train) |>
    step_rm(subject) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_impute_median(all_numeric_predictors())
  
  xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
    set_engine("xgboost", scale_pos_weight = spw, verbosity = 0) |>
    set_mode("classification")
  
  wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
  
  tryCatch({
    fit <- wf |> fit(data = train)
    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))
    
    tibble(
      subject = subj,
      sens = sensitivity(preds, truth = EU, estimate = .pred_class)$.estimate,
      spec = specificity(preds, truth = EU, estimate = .pred_class)$.estimate,
      auc = tryCatch(roc_auc(preds, truth = EU, .pred_eating)$.estimate, error = function(e) NA)
    )
  }, error = function(e) {
    tibble(subject = subj, sens = NA, spec = NA, auc = NA)
  })
}, .progress = TRUE)

# =============================================================================
# 4. RISULTATI
# =============================================================================

cat("\n\n========================================\n")
cat("=== RISULTATI MINIROCKET + LOSO ===\n")
cat("========================================\n\n")

cat("Per soggetto:\n")
print(loso_results, n = 25)

overall <- loso_results |>
  summarise(
    mean_sens = mean(sens, na.rm = TRUE),
    mean_spec = mean(spec, na.rm = TRUE),
    mean_auc = mean(auc, na.rm = TRUE),
    sd_sens = sd(sens, na.rm = TRUE)
  )

cat("\n\nMETRICHE AGGREGATE:\n")
cat("Sensitivity:", round(overall$mean_sens, 3), "(±", round(overall$sd_sens, 3), ")\n")
cat("Specificity:", round(overall$mean_spec, 3), "\n")
cat("AUC:", round(overall$mean_auc, 3), "\n")

cat("\n\nCONFRONTO CON BASELINE (features aggregate):\n")
cat("Baseline LOSO: sens = 0.417\n")
cat("MiniRocket:    sens =", round(overall$mean_sens, 3), "\n")
cat("Delta:         ", round((overall$mean_sens - 0.417) * 100, 1), "%\n")

# Save
saveRDS(list(
  loso_results = loso_results,
  overall = overall,
  features_df = features_df
), here("analysis", "timeseries", "minirocket_results.rds"))

cat("\nSaved to 08_timeseries/01_minirocket/minirocket_results.rds\n")
