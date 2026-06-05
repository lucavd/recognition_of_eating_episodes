# ---------------------------------------------------------------------------
# Cluster bootstrap 95% CI for LOSO metrics
# Addresses Reviewer #1 Comment B1 (PDIG-D-26-00312)
#
# Rationale: each of the 19 LOSO folds yields one per-subject value for
# sens / spec / bal_acc / AUC. Because subjects are the resampling unit
# (the only source of independent replication), we derive confidence
# intervals for the mean-across-subjects via a cluster (subject-level)
# non-parametric bootstrap. B = 1000 resamples, seed = 1812.
#
# Models included (per-subject metrics already available on disk):
#   - Random Forest (default)
#   - Random Forest (tuned)
#   - XGBoost (default)
#   - XGBoost (tuned)
#   - LightGBM (default)
#   - Attention pooling (default)
#   - Transformer (default)
# Not included (no per-subject data saved; would require re-running):
#   LGB tuned, Logistic Regression, Decision Tree, Attention tuned, Ensemble.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(yardstick)
})

set.seed(1812)
B_BOOT <- 1000

PROJ <- here::here()
RES  <- file.path(PROJ, "results")
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")
DEEP_CSV <- file.path(OUT, "deep_per_subject.csv")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(DEEP_CSV))
  stop("Missing ", DEEP_CSV, ". Run prep_deep_metrics.py first.")

# --- 1. Traditional ML (RF, XGB, LGB) default -------------------------------

loso_all <- readRDS(file.path(RES, "loso_all_models.rds"))$all_results

ml_def <- loso_all |>
  select(subject, model, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy) |>
  mutate(model = recode(model,
                        "RF"       = "RF (default)",
                        "XGBoost"  = "XGB (default)",
                        "LightGBM" = "LGB (default)"),
         subject = sprintf("%02d", as.integer(subject)))

# --- 2. Tuned RF and XGB ----------------------------------------------------

tuned <- read_csv(file.path(RES, "subject_metrics_tuned.csv"),
                  show_col_types = FALSE) |>
  rename(auc = roc_auc) |>
  mutate(bal_acc = (sens + spec) / 2,
         model = recode(model,
                        "RF_tuned"  = "RF (tuned)",
                        "XGB_tuned" = "XGB (tuned)"),
         subject = sprintf("%02d", as.integer(subject)))

# --- 3. Attention + Transformer (per-subject from prep_deep_metrics.py) -----

dl <- read_csv(DEEP_CSV, show_col_types = FALSE) |>
  mutate(subject = sprintf("%02d", as.integer(subject))) |>
  filter(subject != "02") |>   # 02 left-handed, excluded elsewhere in the paper
  select(subject, model, sens, spec, bal_acc, auc)

# --- 3b. LR + DT (baseline) from loso_lr_dt.R -----------------------------

LRDT_CSV <- file.path(OUT, "per_subject_lr_dt.csv")
if (file.exists(LRDT_CSV)) {
  lrdt <- read_csv(LRDT_CSV, show_col_types = FALSE) |>
    mutate(subject = sprintf("%02d", as.integer(subject))) |>
    select(subject, model, sens, spec, bal_acc, auc)
} else {
  lrdt <- tibble(subject = character(), model = character(),
                 sens = numeric(), spec = numeric(),
                 bal_acc = numeric(), auc = numeric())
}

# --- 3c. LGB tuned -----------------------------------------------------------

LGBT_CSV <- file.path(OUT, "per_subject_lgb_tuned.csv")
if (file.exists(LGBT_CSV)) {
  lgbt <- read_csv(LGBT_CSV, show_col_types = FALSE) |>
    mutate(subject = sprintf("%02d", as.integer(subject)),
           model = "LGB (tuned)") |>
    select(subject, model, sens, spec, bal_acc, auc)
} else {
  lgbt <- tibble(subject = character(), model = character(),
                 sens = numeric(), spec = numeric(),
                 bal_acc = numeric(), auc = numeric())
}

# --- 3d. Attention (default) from attention_default_per_subject.py -----------

ATT_DEF_CSV <- file.path(OUT, "per_subject_attention_default.csv")
if (file.exists(ATT_DEF_CSV)) {
  att_def <- read_csv(ATT_DEF_CSV, show_col_types = FALSE) |>
    mutate(subject = sprintf("%02d", as.integer(subject))) |>
    filter(subject != "02") |>
    select(subject, model, sens, spec, bal_acc, auc)
} else {
  att_def <- tibble(subject = character(), model = character(),
                    sens = numeric(), spec = numeric(),
                    bal_acc = numeric(), auc = numeric())
}

# --- 4. Combine -------------------------------------------------------------

dl_renamed <- dl |>
  mutate(model = case_when(
    model == "Attention"   ~ "Attention (tuned)",
    TRUE ~ model
  ))

per_subj <- bind_rows(
  ml_def   |> select(subject, model, sens, spec, bal_acc, auc),
  tuned    |> select(subject, model, sens, spec, bal_acc, auc),
  dl_renamed |> select(subject, model, sens, spec, bal_acc, auc),
  lrdt     |> select(subject, model, sens, spec, bal_acc, auc),
  lgbt     |> select(subject, model, sens, spec, bal_acc, auc),
  att_def  |> select(subject, model, sens, spec, bal_acc, auc)
) |>
  arrange(model, subject)

stopifnot(all(table(per_subj$model) == 19))

write_csv(per_subj, file.path(OUT, "per_subject_metrics.csv"))

# --- 5. Cluster bootstrap ---------------------------------------------------

# Each model has its own 19-subject set (ID schemes differ across sources).
# Cluster bootstrap is performed within each model's own subject pool.
models  <- unique(per_subj$model)
metrics <- c("sens", "spec", "bal_acc", "auc")

boot_one_model <- function(dfm, B) {
  subs   <- sort(unique(dfm$subject))
  n_subj <- length(subs)
  means_mat <- replicate(B, {
    smp  <- sample(subs, n_subj, replace = TRUE)
    rows <- dfm[match(smp, dfm$subject), metrics, drop = FALSE]
    colMeans(rows, na.rm = TRUE)
  })
  tibble(
    metric        = metrics,
    point_mean    = colMeans(dfm[, metrics], na.rm = TRUE),
    boot_mean     = rowMeans(means_mat),
    ci_low        = apply(means_mat, 1, quantile, probs = 0.025, na.rm = TRUE),
    ci_high       = apply(means_mat, 1, quantile, probs = 0.975, na.rm = TRUE),
    boot_se       = apply(means_mat, 1, sd)
  )
}

ci_table <- map_dfr(models, function(m) {
  dfm <- per_subj |> filter(model == m)
  boot_one_model(dfm, B_BOOT) |> mutate(model = m, .before = 1)
})

write_csv(ci_table, file.path(OUT, "bootstrap_ci.csv"))

# --- 6. Pretty printed summary (for quick human inspection) ----------------

pretty <- ci_table |>
  mutate(cell = sprintf("%.3f [%.3f, %.3f]", point_mean, ci_low, ci_high)) |>
  select(model, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell) |>
  select(model, sens, spec, bal_acc, auc)

write_csv(pretty, file.path(OUT, "bootstrap_ci_pretty.csv"))

cat("\n--- Bootstrap 95% CI (B =", B_BOOT, ", seed = 1812) ---\n")
print(as.data.frame(pretty), row.names = FALSE)

# --- 7. Forest plot of per-subject sensitivity -----------------------------

model_order <- c("LR", "DT",
                 "RF (default)", "RF (tuned)",
                 "XGB (default)", "XGB (tuned)",
                 "LGB (default)", "LGB (tuned)",
                 "Attention (default)", "Attention (tuned)",
                 "Transformer")

fp_df <- per_subj |>
  filter(model %in% model_order) |>
  mutate(model = factor(model, levels = model_order)) |>
  group_by(model) |>
  mutate(rank = rank(sens, ties.method = "first")) |>
  ungroup()

pop_mean <- fp_df |>
  group_by(model) |>
  summarise(mean_sens = mean(sens), .groups = "drop")

p_forest <- ggplot(fp_df, aes(x = sens, y = rank)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_vline(data = pop_mean, aes(xintercept = mean_sens),
             colour = "firebrick", linetype = "dotted") +
  geom_point(size = 1.8) +
  geom_text(aes(label = subject), size = 2.2, hjust = -0.4) +
  facet_wrap(~ model, ncol = 4) +
  scale_y_continuous(breaks = seq(1, 19, by = 2), name = "Subject (rank by sensitivity)") +
  scale_x_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
  labs(x = "Per-subject sensitivity (LOSO)") +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95", colour = NA),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

ggsave(file.path(OUT, "forest_per_subject_sens.tiff"),
       p_forest, width = 12, height = 6, dpi = 300,
       device = "tiff", compression = "lzw")
ggsave(file.path(OUT, "forest_per_subject_sens.pdf"),
       p_forest, width = 12, height = 6)
ggsave(file.path(OUT, "forest_per_subject_sens.png"),
       p_forest, width = 12, height = 6, dpi = 300)

cat("\nSaved:\n",
    " -", file.path(OUT, "per_subject_metrics.csv"), "\n",
    " -", file.path(OUT, "bootstrap_ci.csv"), "\n",
    " -", file.path(OUT, "bootstrap_ci_pretty.csv"), "\n",
    " -", file.path(OUT, "forest_per_subject_sens.pdf"), "\n")
