# 06_fewshot_calibration.R - Few-shot Learning per Calibrazione Per-Soggetto
# Obiettivo: Migliorare sensitivity su soggetti difficili con pochi esempi

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

subjects <- unique(dat$subject)

# Soggetti difficili identificati
difficult_subjects <- c("09", "11", "08", "10", "16")

cat("=== FEW-SHOT CALIBRATION ===\n")
cat("Soggetti difficili:", paste(difficult_subjects, collapse = ", "), "\n\n")

# =============================================================================
# FUNZIONE: LOSO STANDARD (baseline)
# =============================================================================

loso_standard <- function(target_subj, data) {
  train <- data |> filter(subject != target_subj)
  test <- data |> filter(subject == target_subj)
  
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
    preds <- predict(fit, test) |> bind_cols(test |> select(EU))
  })
  
  sens <- sensitivity(preds, truth = EU, estimate = .pred_class)$.estimate
  spec <- specificity(preds, truth = EU, estimate = .pred_class)$.estimate
  
  list(sensitivity = sens, specificity = spec)
}

# =============================================================================
# FUNZIONE: FEW-SHOT CALIBRATION
# =============================================================================

fewshot_calibration <- function(target_subj, data, n_shots = 20, strategy = "balanced") {
  # Split dati target
  target_data <- data |> filter(subject == target_subj)
  other_data <- data |> filter(subject != target_subj)
  
  # Seleziona few-shot samples dal target
  if (strategy == "balanced") {
    # Ugual numero eating e non-eating
    n_each <- floor(n_shots / 2)
    shots_eating <- target_data |> 
      filter(EU == "eating") |> 
      slice_sample(n = min(n_each, sum(target_data$EU == "eating")))
    shots_non <- target_data |> 
      filter(EU == "non_eating") |> 
      slice_sample(n = min(n_each, sum(target_data$EU == "non_eating")))
    calibration_data <- bind_rows(shots_eating, shots_non)
  } else if (strategy == "random") {
    calibration_data <- target_data |> slice_sample(n = min(n_shots, nrow(target_data)))
  } else if (strategy == "eating_only") {
    # Solo esempi eating (per migliorare sensitivity)
    calibration_data <- target_data |> 
      filter(EU == "eating") |> 
      slice_sample(n = min(n_shots, sum(target_data$EU == "eating")))
  }
  
  # Test set = resto del target
  test_ids <- setdiff(target_data$ID, calibration_data$ID)
  test_data <- target_data |> filter(ID %in% test_ids)
  
  # Training = altri soggetti + calibration data
  train_data <- bind_rows(other_data, calibration_data)
  
  class_freq <- table(train_data$EU)
  scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
  
  rec <- recipe(EU ~ ., data = train_data) |>
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
    fit <- wf |> fit(data = train_data)
    preds <- predict(fit, test_data) |> bind_cols(test_data |> select(EU))
  })
  
  # Calcola metriche solo se ci sono abbastanza dati
  if (sum(preds$EU == "eating") > 0) {
    sens <- sensitivity(preds, truth = EU, estimate = .pred_class)$.estimate
  } else {
    sens <- NA
  }
  if (sum(preds$EU == "non_eating") > 0) {
    spec <- specificity(preds, truth = EU, estimate = .pred_class)$.estimate
  } else {
    spec <- NA
  }
  
  list(
    sensitivity = sens, 
    specificity = spec,
    n_calibration = nrow(calibration_data),
    n_test = nrow(test_data),
    n_test_eating = sum(test_data$EU == "eating")
  )
}

# =============================================================================
# ESPERIMENTO 1: Confronto LOSO vs Few-shot su soggetti difficili
# =============================================================================

cat("=== ESPERIMENTO 1: LOSO vs Few-shot (20 shots) ===\n\n")

results_exp1 <- map_dfr(difficult_subjects, function(subj) {
  cat("Soggetto", subj, "...")
  
  # LOSO standard
  loso_res <- loso_standard(subj, dat)
  
  # Few-shot con 20 esempi bilanciati
  fs_res <- fewshot_calibration(subj, dat, n_shots = 20, strategy = "balanced")
  
  cat(" LOSO sens:", round(loso_res$sensitivity, 3), 
      "| Few-shot sens:", round(fs_res$sensitivity, 3), "\n")
  
  tibble(
    subject = subj,
    loso_sens = loso_res$sensitivity,
    loso_spec = loso_res$specificity,
    fewshot_sens = fs_res$sensitivity,
    fewshot_spec = fs_res$specificity,
    improvement = fs_res$sensitivity - loso_res$sensitivity
  )
})

cat("\nRisultati Esperimento 1:\n")
print(results_exp1)
cat("\nMiglioramento medio sensitivity:", round(mean(results_exp1$improvement, na.rm = TRUE), 3), "\n")

# =============================================================================
# ESPERIMENTO 2: Effetto del numero di shots
# =============================================================================

cat("\n\n=== ESPERIMENTO 2: Effetto numero di shots ===\n\n")

n_shots_values <- c(5, 10, 20, 50, 100)

results_exp2 <- map_dfr(difficult_subjects, function(subj) {
  cat("Soggetto", subj, ":\n")
  
  # Baseline LOSO
  loso_res <- loso_standard(subj, dat)
  
  map_dfr(n_shots_values, function(n) {
    fs_res <- fewshot_calibration(subj, dat, n_shots = n, strategy = "balanced")
    cat("  n=", n, " sens:", round(fs_res$sensitivity, 3), "\n")
    
    tibble(
      subject = subj,
      n_shots = n,
      loso_sens = loso_res$sensitivity,
      fewshot_sens = fs_res$sensitivity,
      improvement = fs_res$sensitivity - loso_res$sensitivity
    )
  })
})

# Riepilogo per n_shots
summary_by_shots <- results_exp2 |>
  group_by(n_shots) |>
  summarise(
    mean_loso_sens = mean(loso_sens, na.rm = TRUE),
    mean_fewshot_sens = mean(fewshot_sens, na.rm = TRUE),
    mean_improvement = mean(improvement, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nRiepilogo per numero di shots:\n")
print(summary_by_shots)

# =============================================================================
# ESPERIMENTO 3: Strategie di selezione shots
# =============================================================================

cat("\n\n=== ESPERIMENTO 3: Strategie di selezione ===\n\n")

strategies <- c("balanced", "random", "eating_only")

results_exp3 <- map_dfr(difficult_subjects, function(subj) {
  cat("Soggetto", subj, ":\n")
  
  loso_res <- loso_standard(subj, dat)
  
  map_dfr(strategies, function(strat) {
    fs_res <- fewshot_calibration(subj, dat, n_shots = 20, strategy = strat)
    cat("  ", strat, ": sens =", round(fs_res$sensitivity, 3), "\n")
    
    tibble(
      subject = subj,
      strategy = strat,
      loso_sens = loso_res$sensitivity,
      fewshot_sens = fs_res$sensitivity,
      improvement = fs_res$sensitivity - loso_res$sensitivity
    )
  })
})

summary_by_strategy <- results_exp3 |>
  group_by(strategy) |>
  summarise(
    mean_improvement = mean(improvement, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nRiepilogo per strategia:\n")
print(summary_by_strategy)

# =============================================================================
# ESPERIMENTO 4: Few-shot su TUTTI i soggetti
# =============================================================================

cat("\n\n=== ESPERIMENTO 4: Few-shot su tutti i soggetti ===\n\n")

results_all <- map_dfr(subjects, function(subj) {
  cat("Soggetto", subj, "...")
  
  loso_res <- loso_standard(subj, dat)
  fs_res <- fewshot_calibration(subj, dat, n_shots = 20, strategy = "balanced")
  
  cat(" improvement:", round(fs_res$sensitivity - loso_res$sensitivity, 3), "\n")
  
  tibble(
    subject = subj,
    loso_sens = loso_res$sensitivity,
    fewshot_sens = fs_res$sensitivity,
    improvement = fs_res$sensitivity - loso_res$sensitivity
  )
})

cat("\nRisultati completi:\n")
print(results_all |> arrange(loso_sens), n = 20)

cat("\n=== RIEPILOGO FINALE ===\n")
cat("Media sensitivity LOSO:", round(mean(results_all$loso_sens, na.rm = TRUE), 3), "\n")
cat("Media sensitivity Few-shot:", round(mean(results_all$fewshot_sens, na.rm = TRUE), 3), "\n")
cat("Miglioramento medio:", round(mean(results_all$improvement, na.rm = TRUE), 3), "\n")

# Confronto difficult vs easy
results_all <- results_all |>
  mutate(group = ifelse(subject %in% difficult_subjects, "difficult", "other"))

summary_by_group <- results_all |>
  group_by(group) |>
  summarise(
    n = n(),
    mean_loso = mean(loso_sens, na.rm = TRUE),
    mean_fewshot = mean(fewshot_sens, na.rm = TRUE),
    mean_improvement = mean(improvement, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nConfronto per gruppo:\n")
print(summary_by_group)

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  exp1_difficult = results_exp1,
  exp2_nshots = results_exp2,
  exp3_strategies = results_exp3,
  exp4_all = results_all,
  summary_by_shots = summary_by_shots,
  summary_by_strategy = summary_by_strategy,
  summary_by_group = summary_by_group
), here("results", "fewshot_results.rds"))

cat("\nSaved to results/fewshot_results.rds\n")
