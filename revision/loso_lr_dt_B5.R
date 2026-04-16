# ---------------------------------------------------------------------------
# B5 — Baseline Logistic Regression and Decision Tree under LOSO, with class
# weights, to extend Table 2 with comparable per-subject metrics. Required to
# address Reviewer #1, Comment 5 ("LR has higher bal_acc than XGBoost").
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(here)
})

set.seed(2026)

PROJ <- here()
OUT  <- file.path(PROJ, "manuscript", "review", "outputs")

dat <- readRDS(here("data", "dat_all.rds")) |>
  mutate(EU = factor(ifelse(EU == 1, "eating", "non_eating"))) |>
  select(-ID, -EI) |>
  filter(delta_s == 5) |>
  select(-delta_s)

subjects <- sort(unique(dat$subject))

metrics_set <- metric_set(roc_auc, sens, spec, bal_accuracy)

model_specs <- list(
  LR = logistic_reg() |> set_engine("glm") |> set_mode("classification"),
  DT = decision_tree() |> set_engine("rpart") |> set_mode("classification")
)

all_results <- list()

for (mname in names(model_specs)) {
  cat("\n===== ", mname, "=====\n")
  mspec <- model_specs[[mname]]
  fold_res <- vector("list", length(subjects))

  for (i in seq_along(subjects)) {
    subj  <- subjects[i]
    train <- dat |> filter(subject != subj)
    test  <- dat |> filter(subject == subj)

    n_pos <- sum(train$EU == "eating")
    n_neg <- sum(train$EU == "non_eating")
    wts   <- ifelse(train$EU == "eating", n_neg / n_pos, 1)
    train$case_wt <- importance_weights(wts)

    rec <- recipe(EU ~ ., data = train) |>
      step_rm(subject) |>
      step_mutate(across(c(arm, meal, food), as.factor)) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_zv(all_predictors()) |>
      step_normalize(all_numeric_predictors())

    wf <- workflow() |>
      add_recipe(rec) |>
      add_model(mspec) |>
      add_case_weights(case_wt)

    fit <- tryCatch(wf |> fit(data = train),
                    error = function(e) { cat("  ERR:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit)) next

    preds <- predict(fit, test, type = "prob") |>
      bind_cols(predict(fit, test)) |>
      bind_cols(test |> select(EU))

    fm <- preds |>
      metrics_set(truth = EU, .pred_eating, estimate = .pred_class) |>
      mutate(subject = subj, fold = i, model = mname)
    fold_res[[i]] <- fm

    ba <- fm |> filter(.metric == "bal_accuracy") |> pull(.estimate)
    se <- fm |> filter(.metric == "sens")         |> pull(.estimate)
    cat(sprintf("  fold %2d  subj %s  bal_acc=%.3f  sens=%.3f\n", i, subj, ba, se))
  }
  all_results[[mname]] <- bind_rows(fold_res)
}

combined <- bind_rows(all_results) |>
  select(model, subject, fold, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  rename(auc = roc_auc, bal_acc = bal_accuracy)

write_csv(combined, file.path(OUT, "per_subject_lr_dt.csv"))

cat("\n=== Mean-across-subjects ===\n")
summary <- combined |>
  group_by(model) |>
  summarise(sens = mean(sens), spec = mean(spec),
            bal_acc = mean(bal_acc), auc = mean(auc))
print(as.data.frame(summary))

cat("\nSaved:", file.path(OUT, "per_subject_lr_dt.csv"), "\n")
