# 00_eda.R - Analisi Esplorativa Dataset
# Fase 0 del piano analisi paper 2026

library(tidyverse)
library(here)

# Carica dataset principale
dat <- readRDS(here("data", "dat_all.rds"))

# =============================================================================
# 0.1 STRUTTURA DATASET
# =============================================================================

cat("=== DIMENSIONI ===\n")
cat("Righe:", nrow(dat), "\n")
cat("Colonne:", ncol(dat), "\n")

cat("\n=== VARIABILI ===\n")
print(names(dat))

cat("\n=== STRUTTURA ===\n")
str(dat, list.len = 20)

cat("\n=== SOGGETTI ===\n")
if ("subject" %in% names(dat)) {
  cat("N soggetti:", n_distinct(dat$subject), "\n")
  print(table(dat$subject))
}

# =============================================================================
# 0.2 OUTCOME
# =============================================================================

cat("\n=== DISTRIBUZIONE OUTCOME (EU) ===\n")
if ("EU" %in% names(dat)) {
  print(table(dat$EU))
  cat("Proporzione eating:", mean(dat$EU == "eating" | dat$EU == 1, na.rm = TRUE), "\n")
}

if ("eating_union" %in% names(dat)) {
  cat("\n=== eating_union ===\n")
  print(table(dat$eating_union))
}

# =============================================================================
# 0.3 VARIABILI CATEGORICHE
# =============================================================================

cat("\n=== VARIABILI CATEGORICHE ===\n")

categorical_vars <- c("subject", "arm", "meal", "food", "delta_s", "wrist", "dominant")
for (v in categorical_vars) {
  if (v %in% names(dat)) {
    cat("\n", v, ":\n")
    print(table(dat[[v]], useNA = "ifany"))
  }
}

# =============================================================================
# 0.4 VARIABILI CONTINUE - SOMMARIO
# =============================================================================

cat("\n=== VARIABILI CONTINUE ===\n")
numeric_vars <- dat %>% select(where(is.numeric)) %>% names()
cat("N variabili numeriche:", length(numeric_vars), "\n")
print(numeric_vars)

cat("\n=== SOMMARIO STATISTICO ===\n")
summary_stats <- dat %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(
    n = sum(!is.na(value)),
    missing = sum(is.na(value)),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    q25 = quantile(value, 0.25, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    q75 = quantile(value, 0.75, na.rm = TRUE),
    max = max(value, na.rm = TRUE)
  ) %>%
  arrange(variable)

print(summary_stats, n = 100)

# =============================================================================
# 0.5 BILANCIAMENTO PER SOGGETTO
# =============================================================================

cat("\n=== OUTCOME PER SOGGETTO ===\n")
if (all(c("subject", "EU") %in% names(dat))) {
  by_subject <- dat %>%
    group_by(subject) %>%
    summarise(
      n_obs = n(),
      n_eating = sum(EU == "eating" | EU == 1, na.rm = TRUE),
      prop_eating = n_eating / n_obs
    ) %>%
    arrange(subject)
  
  print(by_subject, n = 30)
  
  cat("\nMedia proporzione eating per soggetto:", mean(by_subject$prop_eating), "\n")
  cat("Range:", min(by_subject$prop_eating), "-", max(by_subject$prop_eating), "\n")
}

# =============================================================================
# 0.6 SALVA RISULTATI
# =============================================================================

saveRDS(summary_stats, here("results", "eda_summary_stats.rds"))
cat("\n=== Risultati salvati in results/ ===\n")
