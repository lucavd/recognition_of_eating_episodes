# 04_interpretability.R - Feature Importance + Analisi Soggetti Difficili
# Fase 4: Interpretabilità

library(tidyverse)
library(tidymodels)
library(xgboost)
library(ranger)
library(bonsai)
library(lightgbm)
library(vip)
library(here)

set.seed(2026)

# =============================================================================
# LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI, -delta_s)

class_freq <- table(dat$EU)
scale_pos_weight <- as.numeric(class_freq["non_eating"] / class_freq["eating"])

cat("Dataset:", nrow(dat), "obs,", length(unique(dat$subject)), "soggetti\n")

# =============================================================================
# PARTE A: FEATURE IMPORTANCE
# =============================================================================

cat("\n=== PARTE A: FEATURE IMPORTANCE ===\n")

# Recipe per training globale
rec <- recipe(EU ~ ., data = dat) |>
  step_rm(subject) |>
  step_mutate(across(c(arm, meal, food), as.factor)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# --- Random Forest ---
cat("\nFitting RF for feature importance...\n")
rf_spec <- rand_forest(trees = 500) |>
  set_engine("ranger", importance = "impurity",
             class.weights = c("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

rf_wf <- workflow() |> add_recipe(rec) |> add_model(rf_spec)
rf_fit <- rf_wf |> fit(data = dat)

rf_imp <- rf_fit |>
  extract_fit_parsnip() |>
  vip::vi() |>
  mutate(model = "RF")

cat("Top 10 RF features:\n")
print(head(rf_imp, 10))

# --- XGBoost ---
cat("\nFitting XGBoost for feature importance...\n")
xgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
  set_mode("classification")

xgb_wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
xgb_fit <- xgb_wf |> fit(data = dat)

xgb_imp <- xgb_fit |>
  extract_fit_parsnip() |>
  vip::vi() |>
  mutate(model = "XGBoost")

cat("Top 10 XGBoost features:\n")
print(head(xgb_imp, 10))

# --- LightGBM ---
cat("\nFitting LightGBM for feature importance...\n")
lgb_spec <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1) |>
  set_engine("lightgbm", class_weight = list("eating" = scale_pos_weight, "non_eating" = 1)) |>
  set_mode("classification")

lgb_wf <- workflow() |> add_recipe(rec) |> add_model(lgb_spec)
lgb_fit <- lgb_wf |> fit(data = dat)

lgb_imp <- lgb_fit |>
  extract_fit_parsnip() |>
  vip::vi() |>
  mutate(model = "LightGBM")

cat("Top 10 LightGBM features:\n")
print(head(lgb_imp, 10))

# Combine
all_imp <- bind_rows(rf_imp, xgb_imp, lgb_imp)

# Summary: features in top 10 for all models
top10_rf <- head(rf_imp$Variable, 10)
top10_xgb <- head(xgb_imp$Variable, 10)
top10_lgb <- head(lgb_imp$Variable, 10)

common_top10 <- intersect(intersect(top10_rf, top10_xgb), top10_lgb)
cat("\n=== FEATURES IN TOP 10 FOR ALL MODELS ===\n")
print(common_top10)

# =============================================================================
# PARTE B: ANALISI SOGGETTI DIFFICILI
# =============================================================================

cat("\n\n=== PARTE B: ANALISI SOGGETTI DIFFICILI ===\n")

# Load LOSO results
loso <- readRDS(here("results", "loso_all_models.rds"))

# Per-subject sensitivity (XGBoost - best model)
xgb_by_subj <- loso$all_results |>
  filter(model == "XGBoost", .metric == "sens") |>
  arrange(.estimate)

cat("\nSensitivity per soggetto (XGBoost):\n")
print(xgb_by_subj |> select(subject, .estimate))

# Soggetti difficili (bottom 5)
difficult <- head(xgb_by_subj$subject, 5)
cat("\nSoggetti difficili (sens più bassa):", paste(difficult, collapse = ", "), "\n")

# Soggetti facili (top 5)
easy <- tail(xgb_by_subj$subject, 5)
cat("Soggetti facili (sens più alta):", paste(easy, collapse = ", "), "\n")

# Analisi caratteristiche soggetti
cat("\n=== CONFRONTO CARATTERISTICHE ===\n")

subj_stats <- dat |>
  group_by(subject) |>
  summarise(
    n_obs = n(),
    prop_eating = mean(EU == "eating"),
    mean_acc_x = mean(acc_x, na.rm = TRUE),
    mean_acc_y = mean(acc_y, na.rm = TRUE),
    mean_acc_z = mean(acc_z, na.rm = TRUE),
    sd_acc_x = sd(acc_x, na.rm = TRUE),
    mean_power = mean(power, na.rm = TRUE),
    mean_energy = mean(total_energy, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    group = case_when(
      subject %in% difficult ~ "difficult",
      subject %in% easy ~ "easy",
      TRUE ~ "medium"
    )
  )

cat("\nStatistiche soggetti difficili vs facili:\n")
subj_stats |>
  filter(group %in% c("difficult", "easy")) |>
  group_by(group) |>
  summarise(
    n_subj = n(),
    mean_n_obs = mean(n_obs),
    mean_prop_eating = mean(prop_eating),
    mean_acc_x = mean(mean_acc_x),
    mean_sd_acc_x = mean(sd_acc_x),
    mean_power = mean(mean_power),
    .groups = "drop"
  ) |>
  print()

# Test statistico
cat("\n=== TEST STATISTICI ===\n")

for (var in c("prop_eating", "mean_acc_x", "sd_acc_x", "mean_power")) {
  diff_vals <- subj_stats |> filter(group == "difficult") |> pull(!!sym(var))
  easy_vals <- subj_stats |> filter(group == "easy") |> pull(!!sym(var))
  
  if (length(diff_vals) >= 2 && length(easy_vals) >= 2) {
    test <- wilcox.test(diff_vals, easy_vals)
    cat(var, ": p-value =", round(test$p.value, 4), "\n")
  }
}

# =============================================================================
# SAVE
# =============================================================================

saveRDS(list(
  feature_importance = all_imp,
  common_top10 = common_top10,
  subject_stats = subj_stats,
  difficult_subjects = difficult,
  easy_subjects = easy
), here("results", "interpretability_results.rds"))

cat("\nSaved to results/interpretability_results.rds\n")
