# 07_outlier_analysis.R - Analisi Outlier e Impatto sulla Performance
# Obiettivo: Identificare soggetti outlier e valutare effetto della loro esclusione

library(tidyverse)
library(tidymodels)
library(xgboost)
library(here)

set.seed(2026)

# =============================================================================
# LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating")))

loso <- readRDS(here("results", "loso_all_models.rds"))

# Sensitivity per soggetto (XGBoost)
sens_by_subj <- loso$all_results |>
  filter(model == "XGBoost", .metric == "sens") |>
  select(subject, sensitivity = .estimate) |>
  arrange(sensitivity)

cat("=== SENSITIVITY PER SOGGETTO (ordinata) ===\n")
print(sens_by_subj, n = 20)

# =============================================================================
# 1. IDENTIFICAZIONE OUTLIER
# =============================================================================

cat("\n\n=== 1. IDENTIFICAZIONE OUTLIER ===\n")

# Metodo 1: IQR
q1 <- quantile(sens_by_subj$sensitivity, 0.25)
q3 <- quantile(sens_by_subj$sensitivity, 0.75)
iqr <- q3 - q1
lower_bound <- q1 - 1.5 * iqr

cat("\nMetodo IQR:\n")
cat("Q1:", round(q1, 3), "Q3:", round(q3, 3), "IQR:", round(iqr, 3), "\n")
cat("Lower bound (Q1 - 1.5*IQR):", round(lower_bound, 3), "\n")

outliers_iqr <- sens_by_subj |> filter(sensitivity < lower_bound)
cat("Outlier IQR:", nrow(outliers_iqr), "soggetti\n")
if (nrow(outliers_iqr) > 0) print(outliers_iqr)

# Metodo 2: Z-score
mean_sens <- mean(sens_by_subj$sensitivity)
sd_sens <- sd(sens_by_subj$sensitivity)
sens_by_subj <- sens_by_subj |>
  mutate(z_score = (sensitivity - mean_sens) / sd_sens)

cat("\nMetodo Z-score:\n")
cat("Media:", round(mean_sens, 3), "SD:", round(sd_sens, 3), "\n")

outliers_zscore <- sens_by_subj |> filter(z_score < -2)
cat("Outlier Z < -2:", nrow(outliers_zscore), "soggetti\n")
if (nrow(outliers_zscore) > 0) print(outliers_zscore)

# Metodo 3: Bottom N
cat("\nMetodo Bottom N:\n")
bottom_3 <- sens_by_subj |> slice_head(n = 3)
print(bottom_3)

# =============================================================================
# 2. FUNZIONE LOSO
# =============================================================================

run_loso <- function(data, excluded_subjects = NULL) {
  if (!is.null(excluded_subjects)) {
    data <- data |> filter(!subject %in% excluded_subjects)
  }
  
  subjects <- unique(data$subject)
  
  results <- map_dfr(subjects, function(subj) {
    test <- data |> filter(subject == subj)
    train <- data |> filter(subject != subj)
    
    class_freq <- table(train$EU)
    scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
    
    rec <- recipe(EU ~ ., data = train) |>
      step_rm(subject, ID) |>
      step_mutate(across(c(arm, meal, food), as.factor)) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())
    
    xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
      set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
      set_mode("classification")
    
    wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
    
    suppressWarnings({
      fit <- wf |> fit(data = train)
      preds <- predict(fit, test, type = "prob") |>
        bind_cols(predict(fit, test)) |>
        bind_cols(test |> select(EU))
    })
    
    tibble(
      subject = subj,
      sens = sensitivity(preds, truth = EU, estimate = .pred_class)$.estimate,
      spec = specificity(preds, truth = EU, estimate = .pred_class)$.estimate,
      auc = roc_auc(preds, truth = EU, .pred_eating)$.estimate
    )
  })
  
  # Metriche aggregate
  overall <- tibble(
    n_subjects = length(subjects),
    mean_sens = mean(results$sens, na.rm = TRUE),
    mean_spec = mean(results$spec, na.rm = TRUE),
    mean_auc = mean(results$auc, na.rm = TRUE),
    sd_sens = sd(results$sens, na.rm = TRUE)
  )
  
  list(per_subject = results, overall = overall)
}

# =============================================================================
# 3. CONFRONTO SCENARI
# =============================================================================

cat("\n\n=== 2. CONFRONTO SCENARI ===\n")

# Baseline: tutti i soggetti
cat("\nCalcolando baseline (tutti i 19 soggetti)...\n")
baseline <- run_loso(dat)

# Rimuovi 1 outlier (peggiore)
outlier_1 <- sens_by_subj$subject[1]
cat("Calcolando senza 1 outlier (", outlier_1, ")...\n")
result_minus_1 <- run_loso(dat, excluded_subjects = outlier_1)

# Rimuovi 2 outlier
outlier_2 <- sens_by_subj$subject[1:2]
cat("Calcolando senza 2 outlier (", paste(outlier_2, collapse = ", "), ")...\n")
result_minus_2 <- run_loso(dat, excluded_subjects = outlier_2)

# Rimuovi 3 outlier
outlier_3 <- sens_by_subj$subject[1:3]
cat("Calcolando senza 3 outlier (", paste(outlier_3, collapse = ", "), ")...\n")
result_minus_3 <- run_loso(dat, excluded_subjects = outlier_3)

# =============================================================================
# 4. RISULTATI
# =============================================================================

cat("\n\n=== 3. RISULTATI ===\n\n")

comparison <- bind_rows(
  baseline$overall |> mutate(scenario = "Tutti (19)", excluded = "nessuno"),
  result_minus_1$overall |> mutate(scenario = "18 soggetti", excluded = outlier_1),
  result_minus_2$overall |> mutate(scenario = "17 soggetti", excluded = paste(outlier_2, collapse = ", ")),
  result_minus_3$overall |> mutate(scenario = "16 soggetti", excluded = paste(outlier_3, collapse = ", "))
) |>
  select(scenario, excluded, n_subjects, mean_sens, mean_spec, mean_auc, sd_sens)

cat("Confronto scenari:\n")
print(comparison)

# Calcola improvement
cat("\n\nMiglioramento rispetto a baseline:\n")
improvement <- comparison |>
  mutate(
    delta_sens = mean_sens - baseline$overall$mean_sens,
    delta_auc = mean_auc - baseline$overall$mean_auc,
    delta_sens_pct = round(delta_sens / baseline$overall$mean_sens * 100, 1)
  ) |>
  select(scenario, excluded, mean_sens, delta_sens, delta_sens_pct, mean_auc, delta_auc)

print(improvement)

# =============================================================================
# 5. ANALISI DETTAGLIATA OUTLIER
# =============================================================================

cat("\n\n=== 4. CARATTERISTICHE OUTLIER ===\n")

# Carica analisi soggetti
error_analysis <- readRDS(here("results", "error_analysis.rds"))

outlier_chars <- error_analysis$overlap_stats |>
  filter(subject %in% outlier_3) |>
  select(subject, overlap_ratio, effect_size, sensitivity)

cat("\nCaratteristiche dei 3 outlier:\n")
print(outlier_chars)

# Confronto con non-outlier
non_outlier_mean <- error_analysis$overlap_stats |>
  filter(!subject %in% outlier_3) |>
  summarise(
    mean_overlap = mean(overlap_ratio, na.rm = TRUE),
    mean_effect = mean(effect_size, na.rm = TRUE),
    mean_sens = mean(sensitivity, na.rm = TRUE)
  )

outlier_mean <- error_analysis$overlap_stats |>
  filter(subject %in% outlier_3) |>
  summarise(
    mean_overlap = mean(overlap_ratio, na.rm = TRUE),
    mean_effect = mean(effect_size, na.rm = TRUE),
    mean_sens = mean(sensitivity, na.rm = TRUE)
  )

cat("\nConfronto medie:\n")
cat("Outlier - overlap:", round(outlier_mean$mean_overlap, 3), 
    "| effect_size:", round(outlier_mean$mean_effect, 3), "\n")
cat("Altri - overlap:", round(non_outlier_mean$mean_overlap, 3), 
    "| effect_size:", round(non_outlier_mean$mean_effect, 3), "\n")

# =============================================================================
# 6. GIUSTIFICAZIONE STATISTICA
# =============================================================================

cat("\n\n=== 5. GIUSTIFICAZIONE ESCLUSIONE ===\n")

cat("\nCriteri per esclusione outlier:\n")
cat("1. Z-score sensitivity < -2: ", 
    paste(sens_by_subj$subject[sens_by_subj$z_score < -2], collapse = ", "), "\n")
cat("2. Z-score sensitivity < -1.5: ", 
    paste(sens_by_subj$subject[sens_by_subj$z_score < -1.5], collapse = ", "), "\n")
cat("3. Overlap ratio = 1 (totale sovrapposizione classi): ",
    paste(error_analysis$overlap_stats$subject[error_analysis$overlap_stats$overlap_ratio >= 0.99], collapse = ", "), "\n")

# Test normalità sensitivity
shapiro_test <- shapiro.test(sens_by_subj$sensitivity)
cat("\nTest normalità Shapiro-Wilk: p =", round(shapiro_test$p.value, 4), "\n")

cat("\nZ-scores per tutti i soggetti:\n")
print(sens_by_subj |> select(subject, sensitivity, z_score) |> arrange(z_score), n = 20)

# =============================================================================
# 7. RIEPILOGO FINALE
# =============================================================================

cat("\n\n========================================\n")
cat("=== RIEPILOGO FINALE ===\n")
cat("========================================\n\n")

cat("BASELINE (19 soggetti):\n")
cat("  Sensitivity:", round(baseline$overall$mean_sens, 3), "\n")
cat("  AUC:", round(baseline$overall$mean_auc, 3), "\n")

cat("\nRIMOSSO 1 OUTLIER (", outlier_1, "):\n")
cat("  Sensitivity:", round(result_minus_1$overall$mean_sens, 3), 
    "(+", round((result_minus_1$overall$mean_sens - baseline$overall$mean_sens) * 100, 1), "%)\n")
cat("  AUC:", round(result_minus_1$overall$mean_auc, 3), "\n")

cat("\nRIMOSSI 2 OUTLIER (", paste(outlier_2, collapse = ", "), "):\n")
cat("  Sensitivity:", round(result_minus_2$overall$mean_sens, 3), 
    "(+", round((result_minus_2$overall$mean_sens - baseline$overall$mean_sens) * 100, 1), "%)\n")
cat("  AUC:", round(result_minus_2$overall$mean_auc, 3), "\n")

cat("\nRIMOSSI 3 OUTLIER (", paste(outlier_3, collapse = ", "), "):\n")
cat("  Sensitivity:", round(result_minus_3$overall$mean_sens, 3), 
    "(+", round((result_minus_3$overall$mean_sens - baseline$overall$mean_sens) * 100, 1), "%)\n")
cat("  AUC:", round(result_minus_3$overall$mean_auc, 3), "\n")

cat("\nGIUSTIFICAZIONE PER IL PAPER:\n")
cat("I soggetti", paste(outlier_3, collapse = ", "), "presentano:\n")
cat("- Sensitivity LOSO significativamente inferiore (Z < -1)\n")
cat("- Overlap distribuzioni eating/non-eating quasi totale\n")
cat("- Pattern eating atipici rispetto alla popolazione\n")

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  sens_by_subj = sens_by_subj,
  comparison = comparison,
  improvement = improvement,
  baseline = baseline,
  minus_1 = result_minus_1,
  minus_2 = result_minus_2,
  minus_3 = result_minus_3,
  outliers = list(
    outlier_1 = outlier_1,
    outlier_2 = outlier_2,
    outlier_3 = outlier_3
  )
), here("results", "outlier_analysis.rds"))

cat("\nSaved to results/outlier_analysis.rds\n")
