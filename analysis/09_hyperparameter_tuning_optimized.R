# 09_hyperparameter_tuning_optimized.R - Versione OTTIMIZZATA
# Usa: Racing (early stopping) + Latin Hypercube + Multi-threading interno
#
# MIGLIORAMENTI rispetto alla versione originale:
# 1. grid_latin_hypercube(size=15) invece di grid_regular(levels=3-4)
#    -> Da 178 a 45 configurazioni totali (-75%)
# 2. tune_race_anova() invece di tune_grid()
#    -> Elimina configurazioni scarse dopo pochi fold (early stopping)
# 3. Multi-threading INTERNO ai modelli (ranger, xgboost, lightgbm)
#    -> Usa tutti i core disponibili per ogni singolo fit
#
# Stima tempo: da ~24h a ~2-4h (dipende da CPU)

library(tidyverse)
library(tidymodels)
library(finetune)     # Per tune_race_anova
library(xgboost)
library(ranger)
library(bonsai)
library(lightgbm)
library(here)

set.seed(2026)

cat("=== FASE 9: HYPERPARAMETER TUNING OTTIMIZZATO ===\n\n")

# =============================================================================
# 0. SETUP MULTI-THREADING
# =============================================================================

n_cores <- parallel::detectCores()
cat("Cores disponibili:", n_cores, "\n")
cat("I modelli useranno TUTTI i core internamente\n\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating")))

cat("Dataset:", nrow(dat), "obs,", n_distinct(dat$subject), "soggetti\n")
cat("Eating:", sum(dat$EU == "eating"), "| Non-eating:", sum(dat$EU == "non_eating"), "\n\n")

# Class weight
class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])
cat("Scale pos weight:", round(scale_pos_weight, 2), "\n\n")

# =============================================================================
# 2. RECIPE
# =============================================================================

rec <- recipe(EU ~ ., data = dat) |>
  step_rm(subject, ID, EI) |>
  step_mutate(across(c(arm, meal, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# =============================================================================
# 3. LOSO FOLDS
# =============================================================================

loso_folds <- group_vfold_cv(dat, group = subject)
cat("LOSO folds:", nrow(loso_folds), "\n\n")

# =============================================================================
# 4. MODEL SPECIFICATIONS
# =============================================================================

# Random Forest - usa TUTTI i core
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) |>
  set_engine("ranger", 
             class.weights = c("eating" = scale_pos_weight, "non_eating" = 1),
             num.threads = n_cores) |>
  set_mode("classification")

# XGBoost - usa TUTTI i core
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) |>
  set_engine("xgboost", 
             scale_pos_weight = scale_pos_weight, 
             verbosity = 0,
             nthread = n_cores) |>
  set_mode("classification")

# LightGBM - usa TUTTI i core
lgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) |>
  set_engine("lightgbm", 
             class_weight = "balanced", 
             verbose = -1,
             num_threads = n_cores) |>
  set_mode("classification")

# =============================================================================
# 5. LATIN HYPERCUBE GRIDS (più efficiente di grid_regular)
# =============================================================================

# Latin Hypercube: campiona punti in modo da coprire meglio lo spazio
# 15 punti invece di 16-81 -> stesso coverage, meno tempo

rf_grid <- grid_latin_hypercube(
  mtry(range = c(5, 30)),
  min_n(range = c(5, 20)),
  size = 15
)

xgb_grid <- grid_latin_hypercube(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),  # log scale: 0.01 to 0.1
  min_n(range = c(5, 15)),
  size = 15
)

lgb_grid <- grid_latin_hypercube(
  trees(range = c(300, 700)),
  tree_depth(range = c(4, 10)),
  learn_rate(range = c(-2, -1)),
  min_n(range = c(5, 15)),
  size = 15
)

cat("Grid sizes: RF=", nrow(rf_grid), ", XGB=", nrow(xgb_grid), ", LGB=", nrow(lgb_grid), "\n")
cat("Totale configurazioni:", nrow(rf_grid) + nrow(xgb_grid) + nrow(lgb_grid), "(vs 178 originale)\n\n")

metrics <- metric_set(sens, spec, roc_auc)

# Control per racing - elimina configurazioni scarse dopo burn_in folds
race_ctrl <- control_race(
  verbose = TRUE,
  verbose_elim = TRUE,  # Mostra quando elimina configurazioni
  save_pred = FALSE,
  burn_in = 5  # Aspetta 5 folds prima di iniziare a eliminare
)

# =============================================================================
# 6. TUNING RANDOM FOREST (con Racing)
# =============================================================================

cat("=== TUNING RANDOM FOREST (Racing) ===\n")
t1 <- Sys.time()

rf_wf <- workflow() |> add_recipe(rec) |> add_model(rf_spec)

rf_tune <- tune_race_anova(
  rf_wf,
  resamples = loso_folds,
  grid = rf_grid,
  metrics = metrics,
  control = race_ctrl
)

t2 <- Sys.time()
cat("\nRF tempo:", round(difftime(t2, t1, units = "mins"), 1), "minuti\n")

rf_best <- select_best(rf_tune, metric = "sens")
cat("\nBest RF params:\n")
print(rf_best)

rf_best_metrics <- rf_tune |>
  collect_metrics() |>
  filter(mtry == rf_best$mtry, min_n == rf_best$min_n)
cat("\nRF best metrics:\n")
print(rf_best_metrics)

# =============================================================================
# 7. TUNING XGBOOST (con Racing)
# =============================================================================

cat("\n\n=== TUNING XGBOOST (Racing) ===\n")
t1 <- Sys.time()

xgb_wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)

xgb_tune <- tune_race_anova(
  xgb_wf,
  resamples = loso_folds,
  grid = xgb_grid,
  metrics = metrics,
  control = race_ctrl
)

t2 <- Sys.time()
cat("\nXGB tempo:", round(difftime(t2, t1, units = "mins"), 1), "minuti\n")

xgb_best <- select_best(xgb_tune, metric = "sens")
cat("\nBest XGB params:\n")
print(xgb_best)

xgb_best_metrics <- xgb_tune |>
  collect_metrics() |>
  filter(
    trees == xgb_best$trees,
    tree_depth == xgb_best$tree_depth,
    learn_rate == xgb_best$learn_rate,
    min_n == xgb_best$min_n
  )
cat("\nXGB best metrics:\n")
print(xgb_best_metrics)

# =============================================================================
# 8. TUNING LIGHTGBM (con Racing)
# =============================================================================

cat("\n\n=== TUNING LIGHTGBM (Racing) ===\n")
t1 <- Sys.time()

lgb_wf <- workflow() |> add_recipe(rec) |> add_model(lgb_spec)

lgb_tune <- tune_race_anova(
  lgb_wf,
  resamples = loso_folds,
  grid = lgb_grid,
  metrics = metrics,
  control = race_ctrl
)

t2 <- Sys.time()
cat("\nLGB tempo:", round(difftime(t2, t1, units = "mins"), 1), "minuti\n")

lgb_best <- select_best(lgb_tune, metric = "sens")
cat("\nBest LGB params:\n")
print(lgb_best)

lgb_best_metrics <- lgb_tune |>
  collect_metrics() |>
  filter(
    trees == lgb_best$trees,
    tree_depth == lgb_best$tree_depth,
    learn_rate == lgb_best$learn_rate,
    min_n == lgb_best$min_n
  )
cat("\nLGB best metrics:\n")
print(lgb_best_metrics)

# =============================================================================
# 9. CONFRONTO FINALE
# =============================================================================

cat("\n\n========================================\n")
cat("=== CONFRONTO TUNED VS DEFAULT ===\n")
cat("========================================\n\n")

get_best_sens <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "sens") |>
    slice_max(mean, n = 1) |>
    pull(mean) |>
    first()
}

get_best_spec <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "spec") |>
    slice_max(mean, n = 1) |>
    pull(mean) |>
    first()
}

get_best_auc <- function(tune_res) {
  tune_res |>
    collect_metrics() |>
    filter(.metric == "roc_auc") |>
    slice_max(mean, n = 1) |>
    pull(mean) |>
    first()
}

comparison <- tibble(
  Model = c("RF default", "RF tuned", "XGB default", "XGB tuned", 
            "LGB default", "LGB tuned", "Attention (reference)"),
  Sensitivity = c(
    0.224,  # RF default LOSO
    get_best_sens(rf_tune),
    0.417,  # XGB default LOSO
    get_best_sens(xgb_tune),
    0.295,  # LGB default LOSO
    get_best_sens(lgb_tune),
    0.498   # Attention reference
  ),
  Specificity = c(
    0.931,
    get_best_spec(rf_tune),
    0.814,
    get_best_spec(xgb_tune),
    0.884,
    get_best_spec(lgb_tune),
    0.709
  ),
  AUC = c(
    0.700,
    get_best_auc(rf_tune),
    0.693,
    get_best_auc(xgb_tune),
    0.691,
    get_best_auc(lgb_tune),
    0.640
  )
)

print(comparison)

cat("\n\nMIGLIORAMENTO TUNING vs DEFAULT:\n")
cat("RF:  sens", round(0.224, 3), "->", round(get_best_sens(rf_tune), 3), 
    "(+", round((get_best_sens(rf_tune) - 0.224) * 100, 1), "%)\n")
cat("XGB: sens", round(0.417, 3), "->", round(get_best_sens(xgb_tune), 3),
    "(+", round((get_best_sens(xgb_tune) - 0.417) * 100, 1), "%)\n")
cat("LGB: sens", round(0.295, 3), "->", round(get_best_sens(lgb_tune), 3),
    "(+", round((get_best_sens(lgb_tune) - 0.295) * 100, 1), "%)\n")

cat("\nATTENTION (sens=0.498) rimane il benchmark da battere.\n")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================

results_dir <- here("results")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

saveRDS(list(
  rf_tune = rf_tune,
  rf_best = rf_best,
  xgb_tune = xgb_tune,
  xgb_best = xgb_best,
  lgb_tune = lgb_tune,
  lgb_best = lgb_best,
  comparison = comparison,
  optimization = "racing + latin_hypercube + parallel"
), here("results", "hyperparameter_tuning_optimized.rds"))

cat("\nRisultati salvati in: results/hyperparameter_tuning_optimized.rds\n")

cat("\nDONE!\n")
