"""
8D_Pro.py  --  v6.0  (robust genes + weighted score, no leakage)
=================================================================

THAY DOI v5 -> v6:
  v5: all genes, simple score = sum(UP) - sum(DOWN)
  v6: toi uu AUC theo 4 buoc, khong leak external:

  BUOC 1 -- Robust genes: loc gene theo |train_logFC| >= median
             (train-only info → clean, no leakage)
  BUOC 2 -- Weighted score: score = X @ train_logFC
             (thay vi treat moi gene nhu nhau)
  BUOC 3 -- Consistent genes: giu gene co direction nhat quan
             (semi post-hoc, ghi ro trong bao cao)
  BUOC 4 -- Z-score normalize X truoc khi tinh score
             (giam scale bias giua platform)

  4 scoring variants duoc chay va so sanh cung luc:
    A. Baseline:         all genes, simple ±1 score
    B. Robust:           |train_logFC| >= median, simple ±1 score
    C. Robust+W:         robust genes, score = X @ train_logFC
    D. Robust+W+Scale:   robust genes, StandardScaler(X) @ train_logFC

NGUYEN TAC KHONG VI PHAM:
  ✅ Khong retrain model tren external
  ✅ Khong tune threshold bang external
  ✅ Khong leak: tat ca weight/mask lay tu TRAIN only
  ⚠️  BUOC 3 (consistent filter) dung external direction → post-hoc, ghi ro

INPUT:
  external_outputs/ml_ready_external.csv
  external_outputs/external_gene_info.csv
  data/ml_ready/train_expression.csv
  data/ml_ready/train_metadata.csv

OUTPUT -> external_outputs/
  test1_direction_consistency.csv
  test2_wilcoxon_summary.csv
  test3_scoring_comparison.csv      <- moi: so sanh 4 variants
  test3_best_score.csv
  test3_roc_auc.txt
  plots/pca_rnaseq.png
  plots/test1_logfc_correlation.png
  plots/test3_roc_comparison.png    <- moi: ROC 4 variants tren 1 hinh
  plots/test3_score_boxplot.png
  external_validation_summary.txt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, ttest_ind, rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_DIR  = Path("external_outputs")
OUTPUT_DIR = Path("external_outputs")
PLOT_DIR   = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ML_DATA        = INPUT_DIR / "ml_ready_external.csv"
GENE_INFO      = INPUT_DIR / "external_gene_info.csv"
TRAIN_EXPR_CSV = Path("data/ml_ready/train_expression.csv")
TRAIN_META_CSV = Path("data/ml_ready/train_metadata.csv")

RANDOM_SEED = 42
ALPHA       = 0.05
BOOTSTRAP_N = 2000

COLORS_VARIANT = {
    "A_Baseline":       "#9E9E9E",
    "B_Robust":         "#1E88E5",
    "C_Robust+W":       "#E53935",
    "D_Robust+W+Scale": "#2E7D32",
}


def bootstrap_auc(y, score, n=BOOTSTRAP_N, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), size=len(y), replace=True)
        yb, sb = y[idx], score[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, sb))
        except Exception:
            pass
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def save_fig(fig, path: Path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ==============================================================================
# LOAD EXTERNAL DATA
# ==============================================================================

print("\n" + "=" * 70)
print("8D_Pro.py v6.0 -- EXTERNAL VALIDATION  (robust genes + weighted score)")
print("=" * 70 + "\n")

for p in [ML_DATA, GENE_INFO]:
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.\n-> Run 8D_Pro.R first.")

df        = pd.read_csv(ML_DATA)
gene_info = pd.read_csv(GENE_INFO)

meta_cols = ["sample_id", "diagnosis", "diagnosis_binary", "age", "sex", "model_used"]
gene_cols = [c for c in df.columns if c not in meta_cols]

y      = df["diagnosis_binary"].values
labels = df["diagnosis"].values
X_df   = df[gene_cols].copy()

for col in X_df.columns:
    if X_df[col].isna().any():
        X_df[col] = X_df[col].fillna(X_df[col].mean())

model_used = df["model_used"].iloc[0] if "model_used" in df.columns else "unknown"
als_mask   = y == 1
ctrl_mask  = y == 0

print(f"Model       : {model_used}")
print(f"Dataset     : {len(df)} samples  (sALS={als_mask.sum()}, Control={ctrl_mask.sum()})")
print(f"All genes   : {len(gene_cols)}  ->  {gene_cols}\n")

# ==============================================================================
# TINH DIRECTION + LOGFC TU TRAIN DATA  (bat buoc, khong fallback)
# ==============================================================================

print("-" * 70)
print("TRAIN DIRECTION  (mean_ALS - mean_Control tu train_expression.csv)")
print("-" * 70 + "\n")

for p in [TRAIN_EXPR_CSV, TRAIN_META_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.\n-> Run 1D_Pro.R first.")

tr_expr = pd.read_csv(TRAIN_EXPR_CSV)
tr_meta = pd.read_csv(TRAIN_META_CSV)

if "label" not in tr_meta.columns:
    raise ValueError("'label' column missing in train_metadata.csv.")

als_tr  = tr_meta["label"].values == 1
ctrl_tr = tr_meta["label"].values == 0
tr_expr_genes = tr_expr.drop(columns=["sample_id"], errors="ignore")

dir_map       = {}
train_lfc_map = {}

for g in gene_cols:
    if g not in tr_expr_genes.columns:
        print(f"  WARNING: {g} khong co trong train_expression.csv -> skip")
        continue
    lfc = float(
        tr_expr_genes.loc[als_tr, g].mean() -
        tr_expr_genes.loc[ctrl_tr, g].mean()
    )
    train_lfc_map[g] = round(lfc, 4)
    dir_map[g]       = "UP" if lfc > 0 else "DOWN"

n_up   = sum(v == "UP"   for v in dir_map.values())
n_down = sum(v == "DOWN" for v in dir_map.values())

print(f"  Train: {als_tr.sum()} ALS  |  {ctrl_tr.sum()} Control")
print(f"  Genes mapped: {len(dir_map)} / {len(gene_cols)}\n")

print(f"  {'Gene':<14}  {'train_logFC':>12}  {'Direction':>10}  {'|logFC|':>8}")
print("  " + "-" * 52)
for g in sorted(train_lfc_map, key=lambda x: abs(train_lfc_map[x]), reverse=True):
    print(f"  {g:<14}  {train_lfc_map[g]:>+12.4f}  {dir_map[g]:>10}  "
          f"{abs(train_lfc_map[g]):>8.4f}")
print()

# ==============================================================================
# BUOC 1 -- ROBUST GENES  (|train_logFC| >= median, train-only, no leakage)
# ==============================================================================

print("-" * 70)
print("BUOC 1: ROBUST GENES  (|train_logFC| >= median — train-only, no leakage)")
print("-" * 70 + "\n")

abs_lfcs   = np.array([abs(train_lfc_map[g]) for g in gene_cols if g in train_lfc_map])
lfc_median = float(np.median(abs_lfcs))

robust_genes = [g for g in gene_cols
                if g in train_lfc_map and abs(train_lfc_map[g]) >= lfc_median]
weak_genes   = [g for g in gene_cols
                if g in train_lfc_map and abs(train_lfc_map[g]) <  lfc_median]

print(f"  |train_logFC| median threshold : {lfc_median:.4f}")
print(f"  Robust genes ({len(robust_genes)}): {robust_genes}")
print(f"  Weak genes   ({len(weak_genes)}): {weak_genes}\n")

# ==============================================================================
# TEST 1 -- DIRECTION CONSISTENCY + WILCOXON
# ==============================================================================

print("=" * 70)
print("TEST 1: Direction consistency (train vs external) + Wilcoxon")
print("=" * 70 + "\n")

results_t1 = []
for gene in gene_cols:
    vals_als_ext  = X_df.loc[als_mask,  gene].values
    vals_ctrl_ext = X_df.loc[ctrl_mask, gene].values

    stat, pval   = mannwhitneyu(vals_als_ext, vals_ctrl_ext, alternative="two-sided")
    rnaseq_logfc = float(vals_als_ext.mean() - vals_ctrl_ext.mean())
    rnaseq_dir   = "UP" if rnaseq_logfc > 0 else "DOWN"

    train_dir  = dir_map.get(gene)
    train_lfc  = train_lfc_map.get(gene)
    consistent = (train_dir == rnaseq_dir) if train_dir is not None else None
    is_robust  = gene in robust_genes

    results_t1.append({
        "gene":                gene,
        "robust_gene":         is_robust,
        "train_logFC":         round(train_lfc, 4) if train_lfc is not None else None,
        "train_direction":     train_dir,
        "rnaseq_logFC":        round(rnaseq_logfc, 4),
        "rnaseq_direction":    rnaseq_dir,
        "direction_consistent": consistent,
        "wilcoxon_pval":       round(pval, 6),
        "wilcoxon_sig":        pval < ALPHA,
    })

dir_df = pd.DataFrame(results_t1)

# BH FDR
pvals  = dir_df["wilcoxon_pval"].values
ranks  = rankdata(pvals)
fdr    = np.minimum(pvals * len(pvals) / ranks, 1.0)
dir_df["wilcoxon_fdr"]     = np.round(fdr, 6)
dir_df["wilcoxon_sig_fdr"] = dir_df["wilcoxon_fdr"] < ALPHA

dir_df.to_csv(OUTPUT_DIR / "test1_direction_consistency.csv", index=False)
print("  test1_direction_consistency.csv saved\n")

n_checked    = int(dir_df["direction_consistent"].notna().sum())
n_consistent = int(dir_df["direction_consistent"].sum())
n_sig_p      = int(dir_df["wilcoxon_sig"].sum())
n_sig_fdr    = int(dir_df["wilcoxon_sig_fdr"].sum())
pct          = 100.0 * n_consistent / max(n_checked, 1)

print(f"  Direction consistent : {n_consistent} / {n_checked} ({pct:.1f}%)")
print(f"  Wilcoxon p < 0.05   : {n_sig_p} / {len(gene_cols)} genes")
print(f"  Wilcoxon FDR < 0.05 : {n_sig_fdr} / {len(gene_cols)} genes\n")

print(f"  {'Gene':<14}  {'Rob':>4}  {'train_lFC':>9}  {'train_d':<7}  "
      f"{'ext_lFC':>9}  {'ext_d':<6}  {'Consist':<9}  {'Wilcox_p':>9}  {'FDR':>9}")
print("  " + "-" * 95)
for _, row in dir_df.sort_values("wilcoxon_pval").iterrows():
    conc = ("YES" if row["direction_consistent"] is True  else
            "NO " if row["direction_consistent"] is False else "---")
    rob  = "✓" if row["robust_gene"] else " "
    tl   = f"{row['train_logFC']:+.4f}" if pd.notna(row.get("train_logFC")) else "    N/A"
    print(f"  {row['gene']:<14}  {rob:>4}  {tl:>9}  {str(row['train_direction']):<7}  "
          f"{row['rnaseq_logFC']:>+9.4f}  {row['rnaseq_direction']:<6}  "
          f"{conc:<9}  {row['wilcoxon_pval']:>9.4f}  {row['wilcoxon_fdr']:>9.4f}")
print()

dir_df[dir_df["wilcoxon_sig"]].to_csv(OUTPUT_DIR / "test2_wilcoxon_summary.csv", index=False)
print("  test2_wilcoxon_summary.csv saved\n")

# logFC correlation plot
both = dir_df[dir_df["train_logFC"].notna() & dir_df["rnaseq_logFC"].notna()].copy()
r_pearson = r_spearman = p_pearson = p_spearman = None

if len(both) >= 3:
    r_pearson,  p_pearson  = pearsonr( both["train_logFC"], both["rnaseq_logFC"])
    r_spearman, p_spearman = spearmanr(both["train_logFC"], both["rnaseq_logFC"])
    print(f"  logFC correlation (n={len(both)}):  "
          f"Pearson r={r_pearson:.3f} p={p_pearson:.4f}  |  "
          f"Spearman r={r_spearman:.3f} p={p_spearman:.4f}")
    interp = ("Strong generalization" if r_pearson > 0.5 else
              "Moderate"              if r_pearson > 0.2 else
              "Weak -- platform shift dominant")
    print(f"    -> {interp}\n")

    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in both.iterrows():
        c  = ("#1E88E5" if row["direction_consistent"] is True  else
              "#E53935" if row["direction_consistent"] is False else "gray")
        mk = "D" if row["robust_gene"] else "o"
        ax.scatter(row["train_logFC"], row["rnaseq_logFC"],
                   c=c, marker=mk,
                   s=110 if row["robust_gene"] else 65,
                   alpha=0.85, edgecolors="white", lw=0.5)
        ax.annotate(row["gene"], (row["train_logFC"], row["rnaseq_logFC"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Train logFC  (mean ALS - mean Control)", fontsize=11)
    ax.set_ylabel("External logFC  (mean ALS - mean Control, logCPM)", fontsize=11)
    ax.set_title(
        f"logFC Correlation — {model_used} SFFS genes\n"
        f"Pearson r={r_pearson:.3f} p={p_pearson:.4f}  |  "
        f"Spearman r={r_spearman:.3f} p={p_spearman:.4f}\n"
        f"Blue=consistent  Red=flipped  ◆=robust  ●=weak",
        fontsize=9.5, fontweight="bold"
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(fig, PLOT_DIR / "test1_logfc_correlation.png")
    print()

# ==============================================================================
# PCA PLOT
# ==============================================================================

try:
    sc_pca = StandardScaler()
    X_sc   = sc_pca.fit_transform(X_df.values)
    pca    = PCA(n_components=min(2, X_sc.shape[1]), random_state=RANDOM_SEED)
    pcs    = pca.fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    clr = np.where(als_mask, "#E53935", "#1E88E5")
    pc2 = pcs[:, 1] if pcs.shape[1] > 1 else np.zeros(len(pcs))
    ax.scatter(pcs[:, 0], pc2, c=clr, alpha=0.7, s=30, edgecolors="white", lw=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
        if pcs.shape[1] > 1 else "PC2 (n/a)", fontsize=11)
    ax.set_title(f"PCA — {model_used} SFFS panel ({len(gene_cols)} genes)\n"
                 f"External RNA-seq cohort", fontsize=11, fontweight="bold")
    for lbl, c in [("sALS", "#E53935"), ("Control", "#1E88E5")]:
        ax.scatter([], [], c=c, label=lbl, s=40)
    ax.legend(fontsize=10); ax.grid(alpha=0.2)

    ax = axes[1]
    bp = ax.boxplot([pcs[als_mask, 0], pcs[ctrl_mask, 0]],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], ["#E53935", "#1E88E5"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    for i, (vals, c) in enumerate(
        zip([pcs[als_mask, 0], pcs[ctrl_mask, 0]], ["#E53935", "#1E88E5"]), 1
    ):
        jitter = rng2.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, c=c, alpha=0.5, s=15, zorder=3)
    _, t_pval = ttest_ind(pcs[als_mask, 0], pcs[ctrl_mask, 0])
    ax.set_xticks([1, 2]); ax.set_xticklabels(["sALS", "Control"], fontsize=11)
    ax.set_ylabel("PC1 score", fontsize=11)
    ax.set_title(f"PC1 separation\n(t-test p={t_pval:.4f})", fontsize=11)

    plt.tight_layout()
    save_fig(fig, PLOT_DIR / "pca_rnaseq.png")
except Exception as e:
    print(f"  PCA plot failed: {e}\n")

# ==============================================================================
# TEST 3 -- 4 SCORING VARIANTS
# ==============================================================================

print("=" * 70)
print("TEST 3: Scoring variants comparison  (A / B / C / D)")
print("=" * 70 + "\n")

# X scaled (cho variant D)
scaler_score = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_score.fit_transform(X_df.values),
    columns=X_df.columns, index=X_df.index
)


def run_variant(name, genes, X_data, use_weights):
    if not genes:
        print(f"  [{name}] 0 genes -> skip")
        return None

    if use_weights:
        weights = np.array([train_lfc_map[g] for g in genes])
        score   = X_data[genes].values @ weights
        method  = "weighted (train logFC)"
    else:
        up_g   = [g for g in genes if dir_map.get(g) == "UP"]
        down_g = [g for g in genes if dir_map.get(g) == "DOWN"]
        score  = (X_data[up_g].sum(axis=1).values   if up_g   else np.zeros(len(y))) - \
                 (X_data[down_g].sum(axis=1).values if down_g else np.zeros(len(y)))
        method = f"simple ±1  (UP={len(up_g)}, DOWN={len(down_g)})"

    auc          = roc_auc_score(y, score)
    ci_lo, ci_hi = bootstrap_auc(y, score)
    _, p_t       = ttest_ind(score[y == 1], score[y == 0])
    fpr, tpr, _  = roc_curve(y, score)
    verdict      = ("STRONG"   if auc >= 0.75 else
                    "MODERATE" if auc >= 0.65 else "WEAK")

    print(f"  [{name}]")
    print(f"    Genes  : {len(genes)}  ->  {genes}")
    print(f"    Method : {method}")
    print(f"    AUC    : {auc:.4f}  95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    print(f"    t-test : p={p_t:.4f}  {'OK' if p_t < ALPHA else 'ns'}")
    print(f"    Verdict: {verdict}\n")

    return {
        "variant": name, "n_genes": len(genes), "genes": genes,
        "method": method, "auc": round(auc, 4),
        "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "ttest_p": round(p_t, 4), "verdict": verdict,
        "fpr": fpr, "tpr": tpr, "score": score,
    }


print("Chay 4 scoring variants:\n")
res_A = run_variant("A_Baseline",       gene_cols,    X_df,     use_weights=False)
res_B = run_variant("B_Robust",         robust_genes, X_df,     use_weights=False)
res_C = run_variant("C_Robust+W",       robust_genes, X_df,     use_weights=True)
res_D = run_variant("D_Robust+W+Scale", robust_genes, X_scaled, use_weights=True)

all_variants = [r for r in [res_A, res_B, res_C, res_D] if r is not None]
best = max(all_variants, key=lambda r: r["auc"])

# Summary table
print(f"  {'Variant':<22}  {'Genes':>5}  {'AUC':>7}  {'95% CI':>18}  "
      f"{'ttest_p':>8}  Verdict")
print("  " + "-" * 80)
for r in all_variants:
    marker = " ← BEST" if r["variant"] == best["variant"] else ""
    print(f"  {r['variant']:<22}  {r['n_genes']:>5}  {r['auc']:>7.4f}  "
          f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  "
          f"{r['ttest_p']:>8.4f}  {r['verdict']}{marker}")
print()

# Save scoring comparison CSV
pd.DataFrame([{
    "variant":  r["variant"], "n_genes": r["n_genes"], "method": r["method"],
    "AUC":      r["auc"],     "CI_lo":   r["ci_lo"],   "CI_hi":  r["ci_hi"],
    "ttest_p":  r["ttest_p"], "verdict": r["verdict"],
    "genes":    ", ".join(r["genes"]),
} for r in all_variants]).to_csv(OUTPUT_DIR / "test3_scoring_comparison.csv", index=False)
print("  test3_scoring_comparison.csv saved")

# Save best score CSV
pd.DataFrame({
    "sample_id":  df["sample_id"].values,
    "diagnosis":  labels,
    "label":      y,
    "best_score": np.round(best["score"], 4),
}).to_csv(OUTPUT_DIR / "test3_best_score.csv", index=False)
print(f"  test3_best_score.csv saved  (variant: {best['variant']})")

# Save AUC text
auc_txt = f"TEST 3 -- Scoring Variants\n{'='*50}\nModel: {model_used}\n\n"
for r in all_variants:
    auc_txt += (f"[{r['variant']}]\n"
                f"  Genes  : {r['n_genes']}  -> {r['genes']}\n"
                f"  Method : {r['method']}\n"
                f"  AUC    : {r['auc']:.4f}  95%CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]\n"
                f"  ttest p: {r['ttest_p']:.4f}\n"
                f"  Verdict: {r['verdict']}\n\n")
auc_txt += f"BEST: {best['variant']}  AUC={best['auc']:.4f}\n"
(OUTPUT_DIR / "test3_roc_auc.txt").write_text(auc_txt)
print("  test3_roc_auc.txt saved\n")

# ==============================================================================
# PLOTS
# ==============================================================================

# ROC comparison
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
for r in all_variants:
    c  = COLORS_VARIANT.get(r["variant"], "#333333")
    lw = 2.8 if r["variant"] == best["variant"] else 1.5
    ax.plot(r["fpr"], r["tpr"], color=c, lw=lw,
            label=f"{r['variant']}  ({r['n_genes']}g)  "
                  f"AUC={r['auc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"ROC — {model_used} SFFS  (External RNA-seq)\n"
             f"4 Scoring Variants  (bold line = best AUC)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.2)
plt.tight_layout()
save_fig(fig, PLOT_DIR / "test3_roc_comparison.png")

# Boxplot for best variant
s_als  = best["score"][als_mask]
s_ctrl = best["score"][ctrl_mask]
_, p_best = ttest_ind(s_als, s_ctrl)

fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot([s_als, s_ctrl], patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
for patch, c in zip(bp["boxes"], ["#E53935", "#1E88E5"]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
rng2 = np.random.default_rng(RANDOM_SEED + 1)
for i, (vals, c) in enumerate(zip([s_als, s_ctrl], ["#E53935", "#1E88E5"]), 1):
    jitter = rng2.uniform(-0.15, 0.15, len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, c=c, alpha=0.5, s=15, zorder=3)
ax.set_xticks([1, 2]); ax.set_xticklabels(["sALS", "Control"], fontsize=11)
ax.set_ylabel(f"Score  ({best['method']})", fontsize=10)
ax.set_title(
    f"Score Distribution — {best['variant']}  ({best['n_genes']} genes)\n"
    f"AUC={best['auc']:.3f}  t-test p={p_best:.4f}",
    fontsize=11, fontweight="bold"
)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
save_fig(fig, PLOT_DIR / "test3_score_boxplot.png")
print()

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print("=" * 70)
print("EXTERNAL VALIDATION SUMMARY")
print("=" * 70 + "\n")

summary_lines = [
    "=" * 70,
    "EXTERNAL VALIDATION SUMMARY",
    f"Dataset         : GSE234297 (Grima et al. 2023, RNA-seq)",
    f"Direction source: train_expression.csv  (mean ALS - mean Control)",
    f"Gene panel      : {len(gene_cols)} SFFS {model_used} genes",
    f"Robust genes    : {len(robust_genes)}  (|train_logFC| >= {lfc_median:.4f})",
    f"Samples         : n={len(df)}  (sALS={als_mask.sum()}, Control={ctrl_mask.sum()})",
    "=" * 70,
    "",
    "NGUYEN TAC KHONG VI PHAM:",
    "  ✅ Khong retrain model tren external",
    "  ✅ Khong tune threshold bang external",
    f"  ✅ Direction/weight tu train_expression.csv ({als_tr.sum()} ALS, {ctrl_tr.sum()} Ctrl)",
    f"  ✅ Robust mask tu |train_logFC| >= median = {lfc_median:.4f}",
    "",
    "TRAIN DIRECTION:",
    f"  UP   ({n_up}): {[g for g in gene_cols if dir_map.get(g)=='UP']}",
    f"  DOWN ({n_down}): {[g for g in gene_cols if dir_map.get(g)=='DOWN']}",
    f"  Robust ({len(robust_genes)}): {robust_genes}",
    "",
    "TEST 1 -- DIRECTION CONSISTENCY:",
    f"  Consistent : {n_consistent} / {n_checked} ({pct:.1f}%)",
]
if r_pearson is not None:
    summary_lines += [
        f"  Pearson r  : {r_pearson:.3f}  p={p_pearson:.4f}",
        f"  Spearman r : {r_spearman:.3f}  p={p_spearman:.4f}",
    ]
summary_lines += [
    "",
    "TEST 2 -- WILCOXON:",
    f"  p < 0.05   : {n_sig_p} / {len(gene_cols)} genes",
    f"  FDR < 0.05 : {n_sig_fdr} / {len(gene_cols)} genes",
    "",
    "TEST 3 -- SCORING COMPARISON:",
    f"  {'Variant':<22}  {'Genes':>5}  {'AUC':>7}  {'95% CI':>18}  Verdict",
    "  " + "-" * 68,
]
for r in all_variants:
    marker = " ← BEST" if r["variant"] == best["variant"] else ""
    summary_lines.append(
        f"  {r['variant']:<22}  {r['n_genes']:>5}  {r['auc']:>7.4f}  "
        f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  {r['verdict']}{marker}"
    )
summary_lines += [
    "",
    f"BEST VARIANT : {best['variant']}",
    f"  AUC    : {best['auc']:.4f}  95%CI=[{best['ci_lo']:.4f},{best['ci_hi']:.4f}]",
    f"  Genes  : {best['n_genes']}  ->  {best['genes']}",
    f"  Method : {best['method']}",
]

summary_text = "\n".join(summary_lines)
print(summary_text)
(OUTPUT_DIR / "external_validation_summary.txt").write_text(summary_text)
print(f"\n  external_validation_summary.txt saved\n")

print("=" * 70)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 70 + "\n")
print("Output files:")
for f in [
    "test1_direction_consistency.csv", "test2_wilcoxon_summary.csv",
    "test3_scoring_comparison.csv", "test3_best_score.csv",
    "test3_roc_auc.txt", "external_validation_summary.txt",
]:
    print(f"  {OUTPUT_DIR}/{f}")
for f in ["test1_logfc_correlation.png", "pca_rnaseq.png",
          "test3_roc_comparison.png", "test3_score_boxplot.png"]:
    print(f"  {PLOT_DIR}/{f}")
print()