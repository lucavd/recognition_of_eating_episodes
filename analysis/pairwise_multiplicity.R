# =============================================================================
# Round 2 / Reviewer #3 — Comment D6
# Recompute the paired (subject-level) cluster-bootstrap pairwise model
# comparisons on the FINAL per-subject metrics (nested tree models, nested-theta
# Transformer, Attention default), then apply multiplicity correction
# (Benjamini-Hochberg FDR and Holm). Report effect sizes, 95% CIs and adjusted
# p-values, replacing the unadjusted "clearly significant" framing.
#
# Per-subject sources (full feature basis, 19 subjects, subject 02 excluded):
#   - nested tuned tree models  : results/nested_loso_per_subject.csv (feature_set == "full")
#   - defaults + LR + DT        : results/per_subject/per_subject_metrics.csv
#   - Transformer (nested theta), Attention default : results/deep_nested_per_subject.csv
#
# Bootstrap: B = 1000, seed = 1812 (as Tables 2/3 / S5).
# Outputs -> results/
# =============================================================================

suppressPackageStartupMessages({library(tidyverse); library(here)})
OUT <- here("results", "bootstrap")
R1  <- here("results", "per_subject")

norm_subj <- function(x) sprintf("%02d", as.integer(as.character(x)))

# --- nested tuned tree models (full feature) -------------------------------
nested <- read_csv(file.path(OUT, "nested_loso_per_subject.csv"), show_col_types = FALSE) |>
  filter(feature_set == "full") |>
  transmute(subject = norm_subj(subject),
            model = recode(model, xgb = "XGB (tuned)", rf = "RF (tuned)", lgb = "LGB (tuned)"),
            sens, spec, bal_acc, auc)

# --- defaults + LR + DT from Round-1 per-subject ---------------------------
r1 <- read_csv(file.path(R1, "per_subject_metrics.csv"), show_col_types = FALSE) |>
  mutate(subject = norm_subj(subject)) |>
  filter(model %in% c("XGB (default)", "RF (default)", "LGB (default)", "LR", "DT")) |>
  select(subject, model, sens, spec, bal_acc, auc)

# --- deep: Transformer (nested theta) + Attention default ------------------
deep <- read_csv(file.path(OUT, "deep_nested_per_subject.csv"), show_col_types = FALSE) |>
  filter(model %in% c("Transformer (nested theta)", "Attention default (theta=0.50, no selection)")) |>
  transmute(subject = norm_subj(subject),
            model = recode(model,
                           `Transformer (nested theta)` = "Transformer",
                           `Attention default (theta=0.50, no selection)` = "Attention (default)"),
            sens, spec, bal_acc, auc)

ps <- bind_rows(nested, r1, deep) |> filter(subject != "02")

models <- sort(unique(ps$model))
subs   <- sort(unique(ps$subject))
cat("Models:", length(models), "| Subjects:", length(subs), "\n")
cat(paste(" -", models, collapse = "\n"), "\n\n")

# sanity: every model must have all 19 subjects
chk <- ps |> count(model)
stopifnot(all(chk$n == length(subs)))

metrics <- c("sens", "spec", "bal_acc", "auc")
B <- 1000

# wide per metric: subject x model
wide <- map(metrics, function(m) {
  ps |> select(subject, model, !!sym(m)) |>
    pivot_wider(names_from = model, values_from = !!sym(m)) |>
    arrange(subject)
}) |> set_names(metrics)

# bootstrap index matrix shared across all pairs/metrics (paired design)
set.seed(1812)
n <- length(subs)
boot_idx <- replicate(B, sample(seq_len(n), n, replace = TRUE))

pairs <- combn(models, 2, simplify = FALSE)

res <- map_dfr(metrics, function(m) {
  W <- wide[[m]]
  map_dfr(pairs, function(pr) {
    a <- W[[pr[1]]]; b <- W[[pr[2]]]
    d <- a - b
    delta <- mean(d, na.rm = TRUE)
    reps <- apply(boot_idx, 2, function(ix) mean(d[ix], na.rm = TRUE))
    p_two <- 2 * min(mean(reps <= 0), mean(reps >= 0))
    p_two <- min(p_two, 1)
    tibble(metric = m, model_A = pr[1], model_B = pr[2],
           mean_A = mean(a, na.rm = TRUE), mean_B = mean(b, na.rm = TRUE),
           delta = delta,
           ci_low = quantile(reps, 0.025, names = FALSE),
           ci_high = quantile(reps, 0.975, names = FALSE),
           p_raw = p_two)
  })
})

# multiplicity adjustment across the WHOLE family of comparisons
res <- res |>
  mutate(p_BH = p.adjust(p_raw, method = "BH"),
         p_holm = p.adjust(p_raw, method = "holm"),
         sig_raw  = p_raw  < 0.05,
         sig_BH   = p_BH   < 0.05,
         sig_holm = p_holm < 0.05)

write_csv(res, file.path(OUT, "pairwise_nested_adjusted.csv"))

cat("=== Comparisons significant at raw p<0.05 but NOT after BH-FDR ===\n")
res |> filter(sig_raw & !sig_BH) |>
  mutate(cell = sprintf("%-9s %-14s vs %-14s d=%+.3f raw=%.3f BH=%.3f",
                        metric, model_A, model_B, delta, p_raw, p_BH)) |>
  pull(cell) |> walk(~cat(" ", .x, "\n"))

cat("\n=== Survive BH-FDR (p_BH < 0.05) ===\n")
res |> filter(sig_BH) |>
  mutate(cell = sprintf("%-9s %-14s vs %-14s d=%+.3f [%.3f,%.3f] raw=%.3f BH=%.3f Holm=%.3f",
                        metric, model_A, model_B, delta, ci_low, ci_high, p_raw, p_BH, p_holm)) |>
  pull(cell) |> walk(~cat(" ", .x, "\n"))

# Focused: everything vs XGB (tuned) — the headline model
cat("\n=== Focused: vs XGB (tuned), bal_acc ===\n")
vs_xgb <- res |>
  filter((model_A == "XGB (tuned)" | model_B == "XGB (tuned)")) |>
  mutate(other = ifelse(model_A == "XGB (tuned)", model_B, model_A),
         d_xgb_minus_other = ifelse(model_A == "XGB (tuned)", delta, -delta))
vs_xgb |> filter(metric == "bal_acc") |>
  mutate(cell = sprintf("XGB(tuned) - %-15s  d=%+.3f  raw=%.3f  BH=%.3f  Holm=%.3f",
                        other, d_xgb_minus_other, p_raw, p_BH, p_holm)) |>
  pull(cell) |> walk(~cat(" ", .x, "\n"))
write_csv(vs_xgb, file.path(OUT, "pairwise_vs_xgb_tuned.csv"))

cat("\nN comparisons:", nrow(res),
    "| sig raw:", sum(res$sig_raw),
    "| sig BH:", sum(res$sig_BH),
    "| sig Holm:", sum(res$sig_holm), "\n")
cat("Saved: pairwise_nested_adjusted.csv, pairwise_vs_xgb_tuned.csv\nDONE.\n")
