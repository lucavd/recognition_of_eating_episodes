# 11_paper_figures.R - Figure per il paper
# 1. Boxplot variabilità inter-soggetto (sensitivity per subject)
# 2. Confusion matrix aggregata XGBoost tuned

library(tidyverse)
library(tidymodels)
library(xgboost)
library(here)
library(ggplot2)
library(patchwork)

set.seed(2026)

# =============================================================================
# LOAD DATA
# =============================================================================

dat <- readRDS(here("data", "dat_all.rds"))

dat <- dat |>
  filter(delta_s == 5) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating")))

# Subject metrics tuned
subject_metrics <- read_csv(here("results", "subject_metrics_tuned.csv"))

# Hyperparameter tuning results
tuning_results <- readRDS(here("results", "hyperparameter_tuning_optimized.rds"))

# =============================================================================
# FIGURA 1: BOXPLOT VARIABILITÀ INTER-SOGGETTO
# =============================================================================

cat("\n=== FIGURA 1: BOXPLOT INTER-SUBJECT VARIABILITY ===\n")

# Prepara dati per boxplot - sensitivity per soggetto e modello
sens_data <- subject_metrics |>
  filter(model %in% c("RF_tuned", "XGB_tuned")) |>
  mutate(
    model = factor(model, 
                   levels = c("RF_tuned", "XGB_tuned"),
                   labels = c("Random Forest\n(Tuned)", "XGBoost\n(Tuned)"))
  )

# Statistiche descrittive
cat("\nStatistiche Sensitivity per modello:\n")
sens_data |>
  group_by(model) |>
  summarise(
    mean = mean(sens),
    sd = sd(sens),
    min = min(sens),
    max = max(sens),
    IQR = IQR(sens),
    .groups = "drop"
  ) |>
  print()

# Boxplot
p1 <- ggplot(sens_data, aes(x = model, y = sens, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 2.5, alpha = 0.8, color = "black") +
  scale_fill_manual(values = c("Random Forest\n(Tuned)" = "#4DAF4A", 
                               "XGBoost\n(Tuned)" = "#E41A1C")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Inter-Subject Variability in Sensitivity",
    subtitle = "Leave-One-Subject-Out Cross-Validation (n = 19 subjects)",
    x = "Model",
    y = "Sensitivity",
    caption = "Each point represents one subject's sensitivity score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 13)
  )

ggsave(here("results", "fig_intersubject_variability.png"),
       p1, width = 8, height = 6, dpi = 300)

ggsave(here("results", "fig_intersubject_variability.pdf"),
       p1, width = 8, height = 6)

cat("\nSalvata: fig_intersubject_variability.png/pdf\n")

# =============================================================================
# FIGURA 2: CONFUSION MATRIX AGGREGATA XGBOOST TUNED
# =============================================================================

cat("\n=== FIGURA 2: CONFUSION MATRIX AGGREGATA XGBOOST TUNED ===\n")

# Ottieni i best parameters da tuning
xgb_best <- tuning_results$xgb_best
cat("\nXGBoost best params:\n")
print(xgb_best)

# Funzione per fare LOSO con parametri tuned
run_loso_tuned <- function(data, xgb_params) {
  subjects <- unique(data$subject)
  all_preds <- list()
  
  for (subj in subjects) {
    cat("Processing subject", subj, "...\n")
    
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
    
    xgb_spec <- boost_tree(
      trees = xgb_params$trees,
      tree_depth = xgb_params$tree_depth,
      learn_rate = xgb_params$learn_rate,
      min_n = xgb_params$min_n
    ) |>
      set_engine("xgboost", scale_pos_weight = scale_pos_weight, verbosity = 0) |>
      set_mode("classification")
    
    wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
    
    suppressWarnings({
      fit <- wf |> fit(data = train)
      preds <- predict(fit, test) |>
        bind_cols(test |> select(EU, subject))
    })
    
    all_preds[[subj]] <- preds
  }
  
  bind_rows(all_preds)
}

# Esegui LOSO
cat("\nEseguendo LOSO con XGBoost tuned...\n")
all_predictions <- run_loso_tuned(dat, xgb_best)

# Calcola confusion matrix aggregata
cm <- conf_mat(all_predictions, truth = EU, estimate = .pred_class)
cat("\nConfusion Matrix Aggregata:\n")
print(cm)

# Estrai valori
cm_table <- cm$table
TP <- cm_table["eating", "eating"]
FN <- cm_table["eating", "non_eating"]
FP <- cm_table["non_eating", "eating"]
TN <- cm_table["non_eating", "non_eating"]

total <- TP + FN + FP + TN
cat("\nMetriche aggregate:\n")
cat("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN, "\n")
cat("Total observations:", total, "\n")
cat("Sensitivity (TPR):", round(TP / (TP + FN), 3), "\n")
cat("Specificity (TNR):", round(TN / (TN + FP), 3), "\n")
cat("Precision (PPV):", round(TP / (TP + FP), 3), "\n")
cat("Accuracy:", round((TP + TN) / total, 3), "\n")

# Crea dataframe per plot
cm_df <- as.data.frame(cm_table) |>
  rename(Truth = Truth, Prediction = Prediction, Count = Freq) |>
  mutate(
    Percentage = Count / total * 100,
    Truth = factor(Truth, levels = c("eating", "non_eating"), 
                   labels = c("Eating", "Non-Eating")),
    Prediction = factor(Prediction, levels = c("eating", "non_eating"), 
                        labels = c("Eating", "Non-Eating")),
    Label = paste0(Count, "\n(", round(Percentage, 1), "%)")
  )

# Plot confusion matrix
p2 <- ggplot(cm_df, aes(x = Prediction, y = Truth)) +
  geom_tile(aes(fill = Count), color = "white", linewidth = 1) +
  geom_text(aes(label = Label), size = 5, fontface = "bold") +
  scale_fill_gradient2(
    low = "white", 
    mid = "#FDB863", 
    high = "#E66101",
    midpoint = max(cm_df$Count) / 2,
    name = "Count"
  ) +
  scale_x_discrete(position = "top") +
  scale_y_discrete(limits = rev) +
  labs(
    title = "Aggregated Confusion Matrix - XGBoost (Tuned)",
    subtitle = sprintf("LOSO-CV | Sensitivity: %.1f%% | Specificity: %.1f%%", 
                       TP / (TP + FN) * 100, TN / (TN + FP) * 100),
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 13),
    panel.grid = element_blank()
  ) +
  coord_fixed()

ggsave(here("results", "fig_confusion_matrix_xgb_tuned.png"),
       p2, width = 8, height = 7, dpi = 300)

ggsave(here("results", "fig_confusion_matrix_xgb_tuned.pdf"),
       p2, width = 8, height = 7)

cat("\nSalvata: fig_confusion_matrix_xgb_tuned.png/pdf\n")

# =============================================================================
# SALVA DATI FIGURE
# =============================================================================

saveRDS(list(
  subject_sensitivity = sens_data,
  confusion_matrix = cm,
  predictions = all_predictions,
  xgb_params = xgb_best
), here("results", "paper_figures_data.rds"))

cat("\n\n========================================\n")
cat("=== FIGURE COMPLETATE ===\n")
cat("========================================\n")
cat("\nFile salvati in results/:\n")
cat("- fig_intersubject_variability.png/pdf\n")
cat("- fig_confusion_matrix_xgb_tuned.png/pdf\n")
cat("- paper_figures_data.rds\n")
