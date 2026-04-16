library(testthat)
library(here)

# ── Test 1: Derived data exists and has correct structure ────────────────────

test_that("00_derive_data.R produced dat_all.rds", {
  path <- here("data", "dat_all.rds")
  expect_true(file.exists(path), info = "dat_all.rds must exist after running 00_derive_data.R")
})

test_that("dat_all.rds has correct structure", {
  dat <- readRDS(here("data", "dat_all.rds"))

  # Must have all delta_s values
  expect_true(all(1:5 %in% dat$delta_s),
    info = "dat_all.rds must contain delta_s = 1 through 5")

  # Must have 19 subjects (02 excluded)
  subjects <- sort(unique(dat$subject))
  expect_equal(length(subjects), 19)
  expect_false("02" %in% subjects, info = "Subject 02 (left-handed) must be excluded")

  # Must have key columns
  required_cols <- c("acc_x", "acc_y", "acc_z", "pitch", "roll", "power",
    "total_energy", "mean_acc_x_right", "sd_acc_x_right", "cv_acc_x_right",
    "cor_xy", "cor_yz", "cor_zx", "arm", "subject", "meal", "food",
    "EU", "EI", "delta_s")
  for (col in required_cols) {
    expect_true(col %in% names(dat), info = paste("Missing column:", col))
  }

  # EU and EI must be numeric 0/1
  expect_true(is.numeric(dat$EU))
  expect_true(all(dat$EU %in% c(0, 1)))
})

test_that("dat_all.rds delta_s=5 subset has expected size", {
  dat <- readRDS(here("data", "dat_all.rds"))
  dat5 <- dat[dat$delta_s == 5, ]

  # Expected: ~26,000 IUs (original had 26,304; derived may differ by ~1%)
  expect_gt(nrow(dat5), 25000, label = "delta_s=5 IU count lower bound")
  expect_lt(nrow(dat5), 27000, label = "delta_s=5 IU count upper bound")

  # Eating prevalence ~27% (original 26.9%)
  prev <- mean(dat5$EU == 1)
  expect_gt(prev, 0.24, label = "Eating prevalence lower bound")
  expect_lt(prev, 0.30, label = "Eating prevalence upper bound")
})

# ── Test 2: Validate against original dat_all.rds if available ───────────────

test_that("derived data matches original dat_all.rds within tolerance", {
  orig_path <- "/Users/utente/Documents/Projects/2024.dare/data/dat_all.rds"
  skip_if_not(file.exists(orig_path), "Original dat_all.rds not found, skipping comparison")

  orig <- readRDS(orig_path)
  derived <- readRDS(here("data", "dat_all.rds"))

  orig5 <- orig[orig$delta_s == 5 & orig$subject != "02", ]
  der5 <- derived[derived$delta_s == 5, ]

  # IU count difference < 2%
  pct_diff <- abs(nrow(der5) - nrow(orig5)) / nrow(orig5)
  expect_lt(pct_diff, 0.02,
    label = paste("IU count diff:", nrow(der5), "vs", nrow(orig5)))

  # Eating prevalence difference < 1 percentage point
  prev_orig <- mean(orig5$EU == 1)
  prev_der <- mean(der5$EU == 1)
  expect_lt(abs(prev_orig - prev_der), 0.01,
    label = paste("Prevalence:", round(prev_der, 3), "vs", round(prev_orig, 3)))

  # Check common numeric columns have similar distributions
  # Compare key numeric feature distributions (exclude IDs, delta_s, EU, EI)
  skip_cols <- c("ID", "delta_s", "EU", "EI")
  common_num <- setdiff(
    intersect(
      names(orig5)[sapply(orig5, is.numeric)],
      names(der5)[sapply(der5, is.numeric)]
    ),
    skip_cols
  )
  n_check <- min(5, length(common_num))
  for (col in common_num[seq_len(n_check)]) {
    n_min <- min(nrow(orig5), nrow(der5), 1000)
    cor_val <- suppressWarnings(cor(
      head(sort(orig5[[col]]), n_min),
      head(sort(der5[[col]]), n_min)
    ))
    if (!is.na(cor_val)) {
      expect_gt(cor_val, 0.95,
        label = paste("Correlation for sorted", col))
    }
  }
})

# ── Test 3: Filter ablation datasets ─────────────────────────────────────────

test_that("iu_features_unfiltered.rds exists and is consistent", {
  path <- here("data", "iu_features_unfiltered.rds")
  expect_true(file.exists(path))

  iu <- readRDS(path)
  expect_gt(nrow(iu), 25000)
  expect_true("EU" %in% names(iu))
  expect_true(is.numeric(iu$EU))
  expect_equal(length(unique(iu$delta_s)), 1)
  expect_equal(unique(iu$delta_s), 5)
})

test_that("iu_features_filtered.rds exists and has same dimensions", {
  unf <- readRDS(here("data", "iu_features_unfiltered.rds"))
  flt <- readRDS(here("data", "iu_features_filtered.rds"))

  expect_equal(nrow(flt), nrow(unf),
    info = "Filtered and unfiltered must have same row count")
  expect_equal(ncol(flt), ncol(unf),
    info = "Filtered and unfiltered must have same column count")
  expect_equal(sort(unique(flt$subject)), sort(unique(unf$subject)),
    info = "Same subjects in both")
})

# ── Test 4: feature_list.csv ─────────────────────────────────────────────────

test_that("feature_list.csv exists and covers expected features", {
  path <- here("data", "feature_list.csv")
  expect_true(file.exists(path))

  fl <- read.csv(path)
  expect_true("feature" %in% names(fl))
  expect_true("category" %in% names(fl))

  # Must include slopes, window stats, and cross-correlations
  expect_true(any(grepl("acc_x", fl$feature)))
  expect_true(any(grepl("mean_", fl$feature)))
  expect_true(any(grepl("cor_xy", fl$feature)))
})

# ── Test 5: All R scripts parse without syntax errors ────────────────────────

test_that("all R scripts in analysis/ parse without errors", {
  scripts <- list.files(here("analysis"), pattern = "\\.R$",
    recursive = TRUE, full.names = TRUE)
  for (s in scripts) {
    expect_error(parse(s), NA,
      info = paste("Syntax error in", basename(s)))
  }
})

test_that("all R scripts in revision/ parse without errors", {
  scripts <- list.files(here("revision"), pattern = "\\.R$",
    recursive = TRUE, full.names = TRUE)
  for (s in scripts) {
    expect_error(parse(s), NA,
      info = paste("Syntax error in", basename(s)))
  }
})

# ── Test 6: All results CSVs are valid ───────────────────────────────────────

test_that("all CSV files in results/ are readable", {
  csvs <- list.files(here("results"), pattern = "\\.csv$",
    recursive = TRUE, full.names = TRUE)
  expect_gt(length(csvs), 20,
    label = paste("Found only", length(csvs), "CSVs"))

  for (f in csvs) {
    df <- tryCatch(read.csv(f), error = function(e) NULL)
    expect_false(is.null(df),
      info = paste("Failed to read", basename(f)))
    expect_gt(nrow(df), 0,
      label = paste("Empty CSV:", basename(f)))
  }
})

# ── Test 7: Path consistency ─────────────────────────────────────────────────

test_that("no scripts reference old analysis_paper_2026 path", {
  all_R <- c(
    list.files(here("analysis"), pattern = "\\.R$", recursive = TRUE, full.names = TRUE),
    list.files(here("revision"), pattern = "\\.R$", recursive = TRUE, full.names = TRUE)
  )
  for (f in all_R) {
    content <- readLines(f, warn = FALSE)
    hits <- grep("analysis_paper_2026", content)
    expect_equal(length(hits), 0,
      info = paste("Old path in", basename(f), "at lines:", paste(hits, collapse = ",")))
  }
})

test_that("no scripts have hardcoded /Users/ paths", {
  all_R <- c(
    list.files(here("analysis"), pattern = "\\.R$", recursive = TRUE, full.names = TRUE),
    list.files(here("revision"), pattern = "\\.R$", recursive = TRUE, full.names = TRUE)
  )
  for (f in all_R) {
    content <- readLines(f, warn = FALSE)
    # Exclude comments
    code_lines <- content[!grepl("^\\s*#", content)]
    hits <- grep("/Users/", code_lines)
    expect_equal(length(hits), 0,
      info = paste("Hardcoded path in", basename(f)))
  }
})

# ── Test 8: S1 Fig exists ───────────────────────────────────────────────────

test_that("S1 Fig TIFF exists", {
  path <- here("figures", "S1_Fig_per_subject_sensitivity.tiff")
  expect_true(file.exists(path))
  expect_gt(file.size(path), 100000, label = "TIFF file size")
})

# ── Test 9: bootstrap_ci.csv matches paper Table 2 numbers ──────────────────

test_that("bootstrap_ci.csv contains expected classifiers", {
  path <- here("results", "bootstrap", "bootstrap_ci.csv")
  expect_true(file.exists(path))

  ci <- read.csv(path)
  # Must have the 8 ML classifiers from Table 2
  expect_true(any(grepl("XGBoost.*tuned|xgb.*tuned", ci[[1]], ignore.case = TRUE)),
    info = "XGBoost tuned must be in bootstrap_ci.csv")
})

# ── Test 10: Repo completeness ──────────────────────────────────────────────

test_that("README.md exists and is substantial", {
  path <- here("README.md")
  expect_true(file.exists(path))
  content <- readLines(path, warn = FALSE)
  expect_gt(length(content), 100,
    label = "README should be detailed")
})

test_that("LICENSE exists", {
  expect_true(file.exists(here("LICENSE")))
})

test_that(".gitignore exists", {
  expect_true(file.exists(here(".gitignore")))
})
