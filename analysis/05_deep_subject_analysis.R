# 05_deep_subject_analysis.R - Analisi Approfondita Soggetti
# Perché alcuni soggetti funzionano e altri no?

library(tidyverse)
library(tidymodels)
library(xgboost)
library(here)
library(ggplot2)

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

cat("=== SENSITIVITY PER SOGGETTO ===\n")
print(sens_by_subj, n = 20)

# Classificazione soggetti
difficult <- c("09", "11", "08", "10", "16")
easy <- c("05", "03", "18", "19", "15")

# =============================================================================
# 1. CARATTERISTICHE DATASET PER SOGGETTO
# =============================================================================

cat("\n\n=== 1. CARATTERISTICHE DATASET ===\n")

subj_chars <- dat |>
  group_by(subject) |>
  summarise(
    n_windows = n(),
    n_eating = sum(EU == "eating"),
    n_non_eating = sum(EU == "non_eating"),
    prop_eating = mean(EU == "eating"),
    imbalance_ratio = n_non_eating / n_eating,
    
    # Numero pasti
    n_meals = n_distinct(meal),
    
    # Braccio studio
    arm = first(arm),
    
    .groups = "drop"
  ) |>
  left_join(sens_by_subj, by = "subject") |>
  mutate(group = case_when(
    subject %in% difficult ~ "difficult",
    subject %in% easy ~ "easy",
    TRUE ~ "medium"
  )) |>
  arrange(sensitivity)

cat("\nCaratteristiche per soggetto:\n")
print(subj_chars, n = 20)

# Correlazione sensitivity vs caratteristiche
cat("\n=== CORRELAZIONI CON SENSITIVITY ===\n")
cat("prop_eating vs sens:", round(cor(subj_chars$prop_eating, subj_chars$sensitivity), 3), "\n")
cat("n_windows vs sens:", round(cor(subj_chars$n_windows, subj_chars$sensitivity), 3), "\n")
cat("imbalance_ratio vs sens:", round(cor(subj_chars$imbalance_ratio, subj_chars$sensitivity), 3), "\n")

# =============================================================================
# 2. DISTRIBUZIONE SEGNALI PER SOGGETTO
# =============================================================================

cat("\n\n=== 2. DISTRIBUZIONE SEGNALI ===\n")

signal_stats <- dat |>
  group_by(subject) |>
  summarise(
    # Accelerometro X (più importante)
    mean_acc_x = mean(acc_x, na.rm = TRUE),
    sd_acc_x = sd(acc_x, na.rm = TRUE),
    range_acc_x = max(acc_x, na.rm = TRUE) - min(acc_x, na.rm = TRUE),
    
    # Statistiche aggregate
    mean_mean_acc_x_right = mean(mean_acc_x_right, na.rm = TRUE),
    sd_mean_acc_x_right = sd(mean_acc_x_right, na.rm = TRUE),
    
    # Power
    mean_power = mean(power, na.rm = TRUE),
    sd_power = sd(power, na.rm = TRUE),
    
    # Energy
    mean_energy = mean(total_energy, na.rm = TRUE),
    sd_energy = sd(total_energy, na.rm = TRUE),
    
    # Correlazioni tra assi
    mean_cor_xy = mean(cor_xy, na.rm = TRUE),
    mean_cor_yz = mean(cor_yz, na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  left_join(sens_by_subj, by = "subject")

cat("\nCorrelazioni segnali vs sensitivity:\n")
for (col in c("mean_acc_x", "sd_acc_x", "range_acc_x", "mean_mean_acc_x_right", 
              "sd_mean_acc_x_right", "mean_power", "sd_power")) {
  r <- cor(signal_stats[[col]], signal_stats$sensitivity, use = "complete.obs")
  cat(col, ":", round(r, 3), "\n")
}

# =============================================================================
# 3. SEPARABILITÀ CLASSI PER SOGGETTO
# =============================================================================

cat("\n\n=== 3. SEPARABILITÀ CLASSI ===\n")

# Calcola differenza media tra eating e non-eating per ogni soggetto
# su mean_acc_x_right (feature più importante)

separability <- dat |>
  group_by(subject, EU) |>
  summarise(
    mean_acc_x_right = mean(mean_acc_x_right, na.rm = TRUE),
    sd_acc_x_right = sd(mean_acc_x_right, na.rm = TRUE),
    mean_power_right = mean(mean_power_right, na.rm = TRUE),
    .groups = "drop"
  ) |>
  pivot_wider(
    names_from = EU,
    values_from = c(mean_acc_x_right, sd_acc_x_right, mean_power_right)
  ) |>
  mutate(
    # Differenza media tra classi
    diff_acc_x = mean_acc_x_right_eating - mean_acc_x_right_non_eating,
    diff_power = mean_power_right_eating - mean_power_right_non_eating,
    
    # Cohen's d approssimato
    pooled_sd_acc = sqrt((sd_acc_x_right_eating^2 + sd_acc_x_right_non_eating^2) / 2),
    cohens_d_acc = diff_acc_x / pooled_sd_acc
  ) |>
  left_join(sens_by_subj, by = "subject")

cat("\nSeparabilità classi per soggetto:\n")
print(separability |> select(subject, diff_acc_x, cohens_d_acc, sensitivity) |> arrange(sensitivity), n = 20)

cat("\nCorrelazione diff_acc_x vs sensitivity:", 
    round(cor(separability$diff_acc_x, separability$sensitivity, use = "complete.obs"), 3), "\n")

# =============================================================================
# 4. ANALISI TEMPORALE - CONSISTENZA EATING
# =============================================================================

cat("\n\n=== 4. PATTERN TEMPORALI ===\n")

# Lunghezza sequenze eating consecutive
eating_runs <- dat |>
  arrange(subject, ID) |>
  group_by(subject) |>
  mutate(
    eating_num = as.numeric(EU == "eating"),
    run_id = cumsum(c(1, diff(eating_num) != 0))
  ) |>
  filter(EU == "eating") |>
  group_by(subject, run_id) |>
  summarise(run_length = n(), .groups = "drop") |>
  group_by(subject) |>
  summarise(
    n_eating_runs = n(),
    mean_run_length = mean(run_length),
    max_run_length = max(run_length),
    sd_run_length = sd(run_length),
    .groups = "drop"
  ) |>
  left_join(sens_by_subj, by = "subject")

cat("\nPattern temporali eating:\n")
print(eating_runs |> arrange(sensitivity), n = 20)

cat("\nCorrelazione mean_run_length vs sensitivity:", 
    round(cor(eating_runs$mean_run_length, eating_runs$sensitivity, use = "complete.obs"), 3), "\n")

# =============================================================================
# 5. CONFRONTO DETTAGLIATO DIFFICILI VS FACILI
# =============================================================================

cat("\n\n=== 5. CONFRONTO DIFFICILI VS FACILI ===\n")

comparison <- subj_chars |>
  filter(group %in% c("difficult", "easy")) |>
  group_by(group) |>
  summarise(
    n = n(),
    mean_sensitivity = mean(sensitivity),
    mean_prop_eating = mean(prop_eating),
    mean_n_windows = mean(n_windows),
    mean_imbalance = mean(imbalance_ratio),
    .groups = "drop"
  )

cat("\nConfronto gruppi:\n")
print(comparison)

# Segnali
signal_comparison <- signal_stats |>
  left_join(subj_chars |> select(subject, group), by = "subject") |>
  filter(group %in% c("difficult", "easy")) |>
  group_by(group) |>
  summarise(
    mean_sd_acc_x = mean(sd_acc_x),
    mean_range_acc_x = mean(range_acc_x),
    mean_sd_mean_acc_x_right = mean(sd_mean_acc_x_right),
    .groups = "drop"
  )

cat("\nConfronto segnali:\n")
print(signal_comparison)

# Separabilità
sep_comparison <- separability |>
  left_join(subj_chars |> select(subject, group), by = "subject") |>
  filter(group %in% c("difficult", "easy")) |>
  group_by(group) |>
  summarise(
    mean_diff_acc_x = mean(diff_acc_x, na.rm = TRUE),
    mean_cohens_d = mean(cohens_d_acc, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nConfronto separabilità:\n")
print(sep_comparison)

# =============================================================================
# 6. TEST STATISTICI
# =============================================================================

cat("\n\n=== 6. TEST STATISTICI ===\n")

test_vars <- list(
  prop_eating = subj_chars,
  imbalance_ratio = subj_chars,
  sd_acc_x = signal_stats |> left_join(subj_chars |> select(subject, group), by = "subject"),
  cohens_d_acc = separability |> left_join(subj_chars |> select(subject, group), by = "subject"),
  mean_run_length = eating_runs |> left_join(subj_chars |> select(subject, group), by = "subject")
)

for (var_name in names(test_vars)) {
  df <- test_vars[[var_name]]
  if (!"group" %in% names(df)) {
    df <- df |> left_join(subj_chars |> select(subject, group), by = "subject")
  }
  
  diff_vals <- df |> filter(group == "difficult") |> pull(!!sym(var_name))
  easy_vals <- df |> filter(group == "easy") |> pull(!!sym(var_name))
  
  if (length(diff_vals) >= 3 && length(easy_vals) >= 3) {
    test <- wilcox.test(diff_vals, easy_vals)
    cat(var_name, ":\n")
    cat("  difficult:", round(mean(diff_vals, na.rm = TRUE), 3), "\n")
    cat("  easy:", round(mean(easy_vals, na.rm = TRUE), 3), "\n")
    cat("  p-value:", round(test$p.value, 4), "\n\n")
  }
}

# =============================================================================
# 7. ANALISI ERRORI DETTAGLIATA
# =============================================================================

cat("\n\n=== 7. ANALISI ERRORI (LOSO PREDICTIONS) ===\n")

# Rifit XGBoost su tutti tranne soggetti difficili, poi predici
class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])

# Funzione per ottenere predizioni LOSO
get_loso_preds <- function(subj) {
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
  fit <- wf |> fit(data = train)
  
  predict(fit, test, type = "prob") |>
    bind_cols(predict(fit, test)) |>
    bind_cols(test |> select(EU, subject, mean_acc_x_right, mean_power_right))
}

# Analizza errori per soggetto difficile e facile
cat("\nAnalisi errori soggetto 09 (peggiore):\n")
preds_09 <- get_loso_preds("09")
errors_09 <- preds_09 |> 
  mutate(correct = EU == .pred_class,
         error_type = case_when(
           EU == "eating" & .pred_class == "non_eating" ~ "FN",
           EU == "non_eating" & .pred_class == "eating" ~ "FP",
           TRUE ~ "correct"
         ))

cat("Distribuzione errori:\n")
print(table(errors_09$error_type))
cat("False Negative rate:", round(sum(errors_09$error_type == "FN") / sum(errors_09$EU == "eating"), 3), "\n")

# Caratteristiche dei falsi negativi
fn_09 <- errors_09 |> filter(error_type == "FN")
tp_09 <- errors_09 |> filter(EU == "eating" & .pred_class == "eating")

cat("\nCaratteristiche FN vs TP (sogg 09):\n")
cat("FN mean_acc_x_right:", round(mean(fn_09$mean_acc_x_right), 3), "\n")
cat("TP mean_acc_x_right:", round(mean(tp_09$mean_acc_x_right), 3), "\n")

cat("\nAnalisi errori soggetto 15 (migliore):\n")
preds_15 <- get_loso_preds("15")
errors_15 <- preds_15 |> 
  mutate(correct = EU == .pred_class,
         error_type = case_when(
           EU == "eating" & .pred_class == "non_eating" ~ "FN",
           EU == "non_eating" & .pred_class == "eating" ~ "FP",
           TRUE ~ "correct"
         ))

cat("Distribuzione errori:\n")
print(table(errors_15$error_type))
cat("False Negative rate:", round(sum(errors_15$error_type == "FN") / sum(errors_15$EU == "eating"), 3), "\n")

# =============================================================================
# 8. CONCLUSIONI
# =============================================================================

cat("\n\n========================================\n")
cat("=== CONCLUSIONI ANALISI APPROFONDITA ===\n")
cat("========================================\n\n")

cat("FATTORI CHE INFLUENZANO LA PERFORMANCE:\n\n")

cat("1. SEPARABILITÀ CLASSI (Cohen's d su acc_x):\n")
cat("   - Soggetti facili: classi più separate\n")
cat("   - Soggetti difficili: eating e non-eating più simili\n\n")

cat("2. PROPORZIONE EATING:\n")
cat("   - Soggetti difficili:", round(mean(subj_chars$prop_eating[subj_chars$group == "difficult"]), 3), "\n")
cat("   - Soggetti facili:", round(mean(subj_chars$prop_eating[subj_chars$group == "easy"]), 3), "\n\n")

cat("3. VARIABILITÀ SEGNALI:\n")
cat("   - Maggiore variabilità = migliore discriminazione\n\n")

cat("4. PATTERN TEMPORALI:\n")
cat("   - Sequenze eating più lunghe = più facili da rilevare\n")

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  subj_chars = subj_chars,
  signal_stats = signal_stats,
  separability = separability,
  eating_runs = eating_runs
), here("results", "deep_subject_analysis.rds"))

cat("\nSaved to results/deep_subject_analysis.rds\n")
