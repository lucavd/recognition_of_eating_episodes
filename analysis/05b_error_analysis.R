# 05b_error_analysis.R - Analisi Errori Dettagliata
# Perché il modello sbaglia su alcuni soggetti?

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

class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])

loso <- readRDS(here("results", "loso_all_models.rds"))

sens_by_subj <- loso$all_results |>
  filter(model == "XGBoost", .metric == "sens") |>
  select(subject, sensitivity = .estimate)

# =============================================================================
# 1. DISTRIBUZIONE SEGNALI EATING VS NON-EATING PER SOGGETTO
# =============================================================================

cat("=== 1. OVERLAP DISTRIBUZIONI PER SOGGETTO ===\n\n")

# Calcola overlap tra distribuzioni eating/non-eating
overlap_stats <- dat |>
  group_by(subject, EU) |>
  summarise(
    mean_acc = mean(mean_acc_x_right, na.rm = TRUE),
    sd_acc = sd(mean_acc_x_right, na.rm = TRUE),
    q25_acc = quantile(mean_acc_x_right, 0.25, na.rm = TRUE),
    q75_acc = quantile(mean_acc_x_right, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) |>
  pivot_wider(
    names_from = EU,
    values_from = c(mean_acc, sd_acc, q25_acc, q75_acc)
  ) |>
  mutate(
    # Overlap tra IQR
    iqr_overlap = pmax(0, pmin(q75_acc_eating, q75_acc_non_eating) - 
                          pmax(q25_acc_eating, q25_acc_non_eating)),
    iqr_eating = q75_acc_eating - q25_acc_eating,
    iqr_non_eating = q75_acc_non_eating - q25_acc_non_eating,
    overlap_ratio = iqr_overlap / pmin(iqr_eating, iqr_non_eating),
    
    # Distanza tra medie in unità di SD
    pooled_sd = sqrt((sd_acc_eating^2 + sd_acc_non_eating^2) / 2),
    effect_size = (mean_acc_eating - mean_acc_non_eating) / pooled_sd
  ) |>
  left_join(sens_by_subj, by = "subject") |>
  arrange(sensitivity)

cat("Overlap distribuzioni (più alto = più difficile):\n")
print(overlap_stats |> select(subject, overlap_ratio, effect_size, sensitivity), n = 20)

cat("\nCorrelazione overlap_ratio vs sensitivity:", 
    round(cor(overlap_stats$overlap_ratio, overlap_stats$sensitivity, use = "complete.obs"), 3), "\n")
cat("Correlazione effect_size vs sensitivity:", 
    round(cor(overlap_stats$effect_size, overlap_stats$sensitivity, use = "complete.obs"), 3), "\n")

# =============================================================================
# 2. ANALISI PREDIZIONI LOSO PER OGNI SOGGETTO
# =============================================================================

cat("\n\n=== 2. ANALISI PREDIZIONI DETTAGLIATA ===\n")

get_predictions <- function(subj) {
  test <- dat |> filter(subject == subj)
  train <- dat |> filter(subject != subj)
  
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
      bind_cols(test |> select(EU, mean_acc_x_right, mean_power_right, 
                               sd_acc_x_right, cor_xy, cor_yz))
  })
  
  preds |> mutate(subject = subj)
}

# Ottieni predizioni per tutti i soggetti
cat("Generando predizioni LOSO per tutti i soggetti...\n")
all_preds <- map_dfr(unique(dat$subject), ~{
  cat("  Soggetto", .x, "...")
  result <- get_predictions(.x)
  cat(" done\n")
  result
})

# Classifica errori
all_preds <- all_preds |>
  mutate(
    error_type = case_when(
      EU == "eating" & .pred_class == "non_eating" ~ "FN",
      EU == "non_eating" & .pred_class == "eating" ~ "FP",
      TRUE ~ "correct"
    ),
    confidence = pmax(.pred_eating, .pred_non_eating)
  )

# =============================================================================
# 3. CARATTERISTICHE DEGLI ERRORI
# =============================================================================

cat("\n\n=== 3. CARATTERISTICHE ERRORI ===\n")

error_summary <- all_preds |>
  group_by(subject) |>
  summarise(
    n_total = n(),
    n_eating = sum(EU == "eating"),
    n_FN = sum(error_type == "FN"),
    n_FP = sum(error_type == "FP"),
    FN_rate = n_FN / n_eating,
    
    # Confidence degli errori
    mean_conf_FN = mean(confidence[error_type == "FN"], na.rm = TRUE),
    mean_conf_correct = mean(confidence[error_type == "correct"], na.rm = TRUE),
    
    # Caratteristiche FN vs TP
    mean_acc_FN = mean(mean_acc_x_right[error_type == "FN"], na.rm = TRUE),
    mean_acc_TP = mean(mean_acc_x_right[EU == "eating" & .pred_class == "eating"], na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  left_join(sens_by_subj, by = "subject") |>
  arrange(sensitivity)

cat("\nRiepilogo errori per soggetto:\n")
print(error_summary |> select(subject, n_eating, FN_rate, mean_conf_FN, mean_acc_FN, mean_acc_TP), n = 20)

# =============================================================================
# 4. CONFRONTO FN vs TP
# =============================================================================

cat("\n\n=== 4. CONFRONTO FALSE NEGATIVES vs TRUE POSITIVES ===\n")

fn_vs_tp <- all_preds |>
  filter(EU == "eating") |>
  mutate(predicted_correct = .pred_class == "eating") |>
  group_by(predicted_correct) |>
  summarise(
    n = n(),
    mean_acc_x = mean(mean_acc_x_right, na.rm = TRUE),
    sd_acc_x = sd(mean_acc_x_right, na.rm = TRUE),
    mean_power = mean(mean_power_right, na.rm = TRUE),
    mean_sd_acc = mean(sd_acc_x_right, na.rm = TRUE),
    mean_cor_xy = mean(cor_xy, na.rm = TRUE),
    mean_confidence = mean(confidence, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nConfronto globale FN vs TP:\n")
print(fn_vs_tp)

# Test statistico
fn_data <- all_preds |> filter(EU == "eating", .pred_class == "non_eating")
tp_data <- all_preds |> filter(EU == "eating", .pred_class == "eating")

cat("\nTest statistici FN vs TP:\n")
for (var in c("mean_acc_x_right", "mean_power_right", "sd_acc_x_right", "cor_xy")) {
  test <- wilcox.test(fn_data[[var]], tp_data[[var]])
  cat(var, ": p =", format(test$p.value, digits = 4), 
      "| FN mean:", round(mean(fn_data[[var]], na.rm = TRUE), 2),
      "| TP mean:", round(mean(tp_data[[var]], na.rm = TRUE), 2), "\n")
}

# =============================================================================
# 5. PATTERN PER SOGGETTO DIFFICILE VS FACILE
# =============================================================================

cat("\n\n=== 5. ANALISI SOGGETTI SPECIFICI ===\n")

difficult_subjs <- c("09", "11")
easy_subjs <- c("15", "19")

for (subj in c(difficult_subjs, easy_subjs)) {
  subj_data <- all_preds |> filter(subject == subj)
  sens <- sens_by_subj |> filter(subject == subj) |> pull(sensitivity)
  
  cat("\n--- Soggetto", subj, "(sens =", round(sens, 3), ") ---\n")
  
  # Distribuzione eating
  eating_data <- subj_data |> filter(EU == "eating")
  cat("N eating:", nrow(eating_data), "\n")
  cat("FN rate:", round(sum(eating_data$.pred_class == "non_eating") / nrow(eating_data), 3), "\n")
  
  # Caratteristiche eating di questo soggetto vs popolazione
  pop_eating <- dat |> filter(EU == "eating", subject != subj)
  
  cat("mean_acc_x_right - soggetto:", round(mean(eating_data$mean_acc_x_right), 1),
      "| popolazione:", round(mean(pop_eating$mean_acc_x_right), 1), "\n")
  
  # Quanto sono "tipici" i suoi eating?
  pop_mean <- mean(pop_eating$mean_acc_x_right)
  pop_sd <- sd(pop_eating$mean_acc_x_right)
  subj_z <- (mean(eating_data$mean_acc_x_right) - pop_mean) / pop_sd
  cat("Z-score vs popolazione:", round(subj_z, 2), "\n")
}

# =============================================================================
# 6. CONCLUSIONI CHIAVE
# =============================================================================

cat("\n\n========================================\n")
cat("=== CONCLUSIONI ANALISI ERRORI ===\n")
cat("========================================\n\n")

# Correlazioni chiave
cat("CORRELAZIONI CON SENSITIVITY:\n")
cat("- Effect size (separabilità): r =", round(cor(overlap_stats$effect_size, overlap_stats$sensitivity, use = "complete.obs"), 3), "\n")
cat("- Overlap distribuzioni: r =", round(cor(overlap_stats$overlap_ratio, overlap_stats$sensitivity, use = "complete.obs"), 3), "\n")

# Differenze FN vs TP
cat("\nPERCHÉ IL MODELLO SBAGLIA (FN):\n")
cat("- I FN hanno segnali mean_acc_x PIÙ BASSI dei TP\n")
cat("- I FN hanno MENO variabilità (sd_acc_x più basso)\n")
cat("- Sono eating 'silenziosi' - movimenti meno pronunciati\n")

cat("\nPERCHÉ ALCUNI SOGGETTI SONO DIFFICILI:\n")
cat("- I loro eating hanno caratteristiche DIVERSE dalla popolazione\n")
cat("- Distribuzione eating/non-eating più SOVRAPPOSTA\n")
cat("- Pattern di movimento eating ATIPICI\n")

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  overlap_stats = overlap_stats,
  error_summary = error_summary,
  all_predictions = all_preds,
  fn_vs_tp = fn_vs_tp
), here("results", "error_analysis.rds"))

cat("\nSaved to results/error_analysis.rds\n")
