"""
Round 2 / Reviewer #3 — Comment D1, Step 2 (deep models).

Transformer: NESTED decision-threshold selection.
  The original Table-3 Transformer used theta = 0.30, selected post hoc by
  maximising POOLED balanced accuracy over ALL 19 LOSO subjects' held-out
  predictions (Phase 10c). That is the post-selection knob Reviewer #3 names.
  Here we nest it: for each held-out subject s, theta* is chosen by the SAME
  rule (pooled balanced accuracy, grid 0.10-0.90 step 0.01) on the OTHER 18
  subjects only, then applied once to subject s. No retraining: we reprocess
  the saved held-out probabilities (predictions.npz). Architecture is fixed a
  priori (best_transformer_config.json; +-0.02 insensitivity across variants).

Attention: the "default" row (theta = 0.50, hand-crafted pooling, no tuning,
  no threshold selection) is already a valid final-test estimate -> carried
  forward from Round 1 (per_subject_attention_default.csv), no nesting needed.

Aggregation: mean across the 19 subjects + subject-level cluster bootstrap
  (B = 1000, seed = 1812), identical to Tables 2/3.

Subject 02 (left-handed) is excluded to match the manuscript LOSO design.
Outputs -> results/
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

PROJ = Path(__file__).resolve().parents[1]
NPZ  = PROJ / "results" / "predictions.npz"
ATT  = PROJ / "results" / "per_subject" / "per_subject_attention_default.csv"
OUT  = PROJ / "results" / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

GRID = np.arange(0.10, 0.90 + 1e-9, 0.01)   # same grid as Phase 10c
EXCLUDE = 2                                  # subject 02 (left-handed)

# ---------------------------------------------------------------------------
def pooled_bal_acc_threshold(y_true, y_prob, grid=GRID):
    """theta maximising POOLED balanced accuracy (Phase 10c rule)."""
    best_t, best_ba = 0.5, -1.0
    for t in grid:
        yp = (y_prob >= t).astype(int)
        sens = recall_score(y_true, yp, pos_label=1, zero_division=0)
        spec = recall_score(y_true, yp, pos_label=0, zero_division=0)
        ba = (sens + spec) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t

def per_subject_metrics(y_true_s, y_prob_s, theta):
    yp = (y_prob_s >= theta).astype(int)
    sens = recall_score(y_true_s, yp, pos_label=1, zero_division=0)
    spec = recall_score(y_true_s, yp, pos_label=0, zero_division=0)
    auc = roc_auc_score(y_true_s, y_prob_s) if len(np.unique(y_true_s)) == 2 else np.nan
    return sens, spec, (sens + spec) / 2, auc

def boot_ci(per_subject_df, metrics=("sens", "spec", "bal_acc", "auc"), B=1000, seed=1812):
    rng = np.random.default_rng(seed)
    n = len(per_subject_df)
    rows = []
    for m in metrics:
        vals = per_subject_df[m].to_numpy(dtype=float)
        reps = np.array([np.nanmean(vals[rng.integers(0, n, n)]) for _ in range(B)])
        rows.append(dict(metric=m, mean=float(np.nanmean(vals)),
                         ci_low=float(np.nanpercentile(reps, 2.5)),
                         ci_high=float(np.nanpercentile(reps, 97.5))))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
d = np.load(NPZ, allow_pickle=True)
y_true = d["y_true"].astype(int)
y_prob_tr = d["y_prob_transformer"].astype(float)
subj = d["subjects"].astype(int)

keep = subj != EXCLUDE
y_true, y_prob_tr, subj = y_true[keep], y_prob_tr[keep], subj[keep]
subjects = sorted(np.unique(subj))
print(f"Transformer: {len(subjects)} subjects, {len(y_true)} IU (subject {EXCLUDE} excluded)")

# Sanity check: pooled theta on ALL 19 should reproduce the original ~0.30
theta_all = pooled_bal_acc_threshold(y_true, y_prob_tr)
print(f"Sanity: pooled theta on all 19 = {theta_all:.2f} (original reported 0.30)")

# Nested threshold per outer subject
rows, thetas = [], []
for s in subjects:
    te = subj == s
    inner = ~te
    theta_s = pooled_bal_acc_threshold(y_true[inner], y_prob_tr[inner])  # 18 inner only
    sens, spec, ba, auc = per_subject_metrics(y_true[te], y_prob_tr[te], theta_s)
    rows.append(dict(model="Transformer (nested theta)", subject=int(s),
                     theta=float(theta_s), sens=sens, spec=spec, bal_acc=ba, auc=auc))
    thetas.append(theta_s)
    print(f"  subj {s:2d}  theta*={theta_s:.2f}  sens={sens:.3f} spec={spec:.3f} bal_acc={ba:.3f} auc={auc:.3f}")

tr_ps = pd.DataFrame(rows)
print(f"\nNested theta: mean={np.mean(thetas):.3f}, range=[{min(thetas):.2f}, {max(thetas):.2f}], "
      f"sd={np.std(thetas):.3f}")

# Reference: original post-selection theta = 0.30 applied to everyone
rows_ref = []
for s in subjects:
    te = subj == s
    sens, spec, ba, auc = per_subject_metrics(y_true[te], y_prob_tr[te], 0.30)
    rows_ref.append(dict(model="Transformer (orig theta=0.30)", subject=int(s),
                         theta=0.30, sens=sens, spec=spec, bal_acc=ba, auc=auc))
tr_ref = pd.DataFrame(rows_ref)

# Attention default (no selection -> valid final-test) carried forward
att_ps = None
if ATT.exists():
    a = pd.read_csv(ATT)
    a.columns = [c.lower() for c in a.columns]
    ren = {}
    for c in a.columns:
        if c in ("bal_accuracy", "balanced_accuracy"): ren[c] = "bal_acc"
        if c in ("roc_auc",): ren[c] = "auc"
    a = a.rename(columns=ren)
    a["subject"] = pd.to_numeric(a["subject"], errors="coerce")
    a = a[a["subject"] != EXCLUDE]            # exclude subject 02 -> 19 subjects
    if "subject" in a.columns and {"sens", "spec", "bal_acc", "auc"}.issubset(a.columns):
        att_ps = a[["subject", "sens", "spec", "bal_acc", "auc"]].copy()
        att_ps.insert(0, "model", "Attention default (theta=0.50, no selection)")
        print(f"\nAttention default carried forward: {len(att_ps)} subjects")

# ---------------------------------------------------------------------------
# Assemble per-subject + bootstrap CI
all_ps = pd.concat([tr_ps, tr_ref] + ([att_ps] if att_ps is not None else []),
                   ignore_index=True)
all_ps.to_csv(OUT / "deep_nested_per_subject.csv", index=False)

ci_list = []
for m, g in all_ps.groupby("model"):
    ci = boot_ci(g)
    ci.insert(0, "model", m)
    ci_list.append(ci)
ci = pd.concat(ci_list, ignore_index=True)
ci.to_csv(OUT / "deep_nested_ci.csv", index=False)

pretty = (ci.assign(cell=lambda x: x.apply(
            lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1))
          .pivot(index="model", columns="metric", values="cell")
          .reindex(columns=["sens", "spec", "bal_acc", "auc"]))
pretty.to_csv(OUT / "deep_nested_pretty.csv")

print("\n================ DEEP MODELS — point [95% CI] ================")
print(pretty.to_string())
print("\nSaved: deep_nested_per_subject.csv, deep_nested_ci.csv, deep_nested_pretty.csv")
print("DONE.")
