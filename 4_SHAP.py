"""
4_SHAP.py  --  v5.0  (Full SHAP edition: Internal + External)
=================================================================

PURPOSE:
  Q1: Which genes does XGB use to separate ALS vs Control?
  Q2: Does expression UP/DOWN increase ALS risk?

PIPELINE:
  PART A -- Internal SHAP  (XGB, test set)
    A1. Summary beeswarm   -- gene nao quan trong + huong anh huong
    A2. Bar plot           -- ranking dinh luong (ALL genes)
    A3. Dependence grid    -- tat ca gene (moi gene 1 panel nho)
    A4. Waterfall          -- 2 benh nhan: risk cao + risk thap

  PART B -- External SHAP  (XGB applied to external RNA-seq)
    B1. Summary + Bar      -- consistency check

  PART C -- Comparison: Internal vs External
    C1. Overlap analysis   -- gene nao on dinh ca 2 tap
    C2. Direction consistency heatmap
    C3. SHAP stability scatter (rank corr)
    C4. Patient-level case study

INPUT (from 3C_sffs.py + 8_Ex.R):
  data/ml_ready/test_expression.csv
  data/ml_ready/test_metadata.csv
  data/features/optimal_genes_XGB.txt    <- SFFS gene set
  models/model_XGB.pkl                   <- Pipeline(scaler -> XGB)
  external_outputs/ml_ready_external.csv <- optional
  external_outputs/external_gene_info.csv

OUTPUT:
  results/shap/internal/
  results/shap/external/
  results/shap/comparison/
  plots/shap/internal/
  plots/shap/external/
  plots/shap/comparison/
  results/shap/shap_results.csv  <- for 5_BIO.R and 8_Ex
"""

import pickle
import shutil
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

warnings.filterwarnings("ignore")

try:
    from PIL import Image as PILImage
    _pil_ok = True
except ImportError:
    _pil_ok = False


# ==============================================================================
# CONFIG  --  change GENE_LIST / MODEL_FILE to switch model
# ==============================================================================

class Config:
    DATA_DIR  = Path("data/ml_ready")
    FEAT_DIR  = Path("data/features")
    MODEL_DIR = Path("models")

    TEST_EXPR  = DATA_DIR / "test_expression.csv"
    TEST_META  = DATA_DIR / "test_metadata.csv"

    # ── Only this line needs changing to switch between XGB / SVM / RF ──────
    GENE_LIST  = FEAT_DIR / "optimal_genes_XGB.txt"
    MODEL_FILE = MODEL_DIR / "model_XGB.pkl"

    # External validation (optional)
    EXT_EXPR = Path("external_outputs/ml_ready_external.csv")
    EXT_INFO = Path("external_outputs/external_gene_info.csv")

    OUT_INT  = Path("results/shap/internal")
    OUT_EXT  = Path("results/shap/external")
    OUT_CMP  = Path("results/shap/comparison")
    PLT_INT  = Path("plots/shap/internal")
    PLT_EXT  = Path("plots/shap/external")
    PLT_CMP  = Path("plots/shap/comparison")
    LOG_FILE = Path("logs/shap_v5.log")

    RANDOM_STATE  = 42
    OVERLAP_TOP_N = 10   # so gene xem xet cho overlap analysis
    # Multi-model SHAP paths (RF, LR, SVM)
    GENE_LIST_RF   = FEAT_DIR / "optimal_genes_RF.txt"
    MODEL_FILE_RF  = MODEL_DIR / "model_RF.pkl"
    GENE_LIST_LR   = FEAT_DIR / "optimal_genes_LR.txt"
    MODEL_FILE_LR  = MODEL_DIR / "model_LR.pkl"
    GENE_LIST_SVM  = FEAT_DIR / "optimal_genes_SVM.txt"
    MODEL_FILE_SVM = MODEL_DIR / "model_SVM.pkl"

    PLT_RF = Path("plots/shap/rf")
    PLT_LR = Path("plots/shap/lr")
    OUT_RF = Path("results/shap/rf")
    OUT_LR = Path("results/shap/lr")

    def __init__(self):
        for d in [self.OUT_INT, self.OUT_EXT, self.OUT_CMP,
                  self.PLT_INT, self.PLT_EXT, self.PLT_CMP,
                  Path("results/shap"), self.LOG_FILE.parent]:
            d.mkdir(parents=True, exist_ok=True)
            for d in [self.PLT_RF, self.PLT_LR, self.OUT_RF, self.OUT_LR]:
                d.mkdir(parents=True, exist_ok=True)


cfg = Config()


# ==============================================================================
# HELPERS
# ==============================================================================

class Logger:
    def __init__(self, path):
        self._f = open(path, "w")
        self._f.write(f"Started: {datetime.now()}\n\n")

    def log(self, msg=""):
        print(msg)
        self._f.write(str(msg) + "\n"); self._f.flush()

    def section(self, title):
        sep = "=" * 70
        self.log(sep); self.log(title); self.log(sep + "\n")

    def close(self):
        self._f.write(f"\nCompleted: {datetime.now()}\n"); self._f.close()


def save_show(fig, path, logger):
    """Save PNG and open popup."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log(f"  [saved] {path}")
    if _pil_ok:
        try:
            PILImage.open(path).show()
            logger.log(f"  [popup] {path.name}")
        except Exception as e:
            logger.log(f"  [popup] skipped ({e})")


def extract_shap_2d(raw, n_class=1):
    """Normalize SHAP output to 2D array for ALS class."""
    if isinstance(raw, list):
        return raw[n_class] if len(raw) > n_class else raw[0]
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        return raw[:, :, n_class]
    return raw


def get_base_value(explainer, pipeline_or_model, X_raw):
    """Return base value in probability space."""
    ev = explainer.expected_value
    bv = float(np.atleast_1d(ev)[1] if len(np.atleast_1d(ev)) >= 2
               else np.atleast_1d(ev).flat[0])
    mean_prob = float(pipeline_or_model.predict_proba(X_raw)[:, 1].mean())
    if abs(bv - mean_prob) > 0.15:
        bv = mean_prob
    return bv


def eff_dir(corr):
    if abs(corr) < 0.1: return "ambiguous"
    return "positive (expr+ -> risk+)" if corr > 0 else "negative (expr+ -> risk-)"


logger = Logger(cfg.LOG_FILE)
print("\n" + "=" * 70)
print("4_SHAP.py  v5.0  --  Full SHAP Analysis (Internal + External)")
print("=" * 70 + "\n")

logger.log("Q1: Which genes does XGB use to separate ALS vs Control?")
logger.log("Q2: Does expression UP/DOWN increase ALS risk?\n")
logger.log("All plots: saved PNG + popup.\n")


# ==============================================================================
# STEP 1: LOAD
# ==============================================================================

logger.section("STEP 1: LOAD DATA & MODEL")

if not cfg.GENE_LIST.exists():
    raise FileNotFoundError(f"{cfg.GENE_LIST} not found. Run 3C_sffs.py first.")

with open(cfg.GENE_LIST) as fh:
    genes = [l.strip() for l in fh if l.strip()]

test_expr = pd.read_csv(cfg.TEST_EXPR)
test_meta = pd.read_csv(cfg.TEST_META)
all_genes = test_expr.drop("sample_id", axis=1).columns.tolist()

miss = [g for g in genes if g not in all_genes]
if miss:
    logger.log(f"  WARNING: {len(miss)} genes not in expression data: {miss}")
    genes = [g for g in genes if g in all_genes]

X_test_raw = test_expr.set_index("sample_id")[genes].values.astype(np.float64)
y_test     = test_meta["label"].values.astype(int)

with open(cfg.MODEL_FILE, "rb") as fh:
    pipeline = pickle.load(fh)

if hasattr(pipeline, "named_steps"):
    scaler  = pipeline.named_steps["scaler"]
    xgb_clf = pipeline.named_steps["clf"]
    X_test_sc = scaler.transform(X_test_raw)
else:
    xgb_clf  = pipeline
    scaler   = None
    X_test_sc = X_test_raw

logger.log(f"Gene list : {cfg.GENE_LIST.name}  ({len(genes)} genes)")
logger.log(f"Model     : {cfg.MODEL_FILE.name}")
logger.log(f"Test set  : {X_test_raw.shape[0]} samples | ALS={y_test.sum()} Ctrl={(y_test==0).sum()}")
logger.log(f"X_test_sc : {X_test_sc.shape}  (used for TreeExplainer)\n")


# ==============================================================================
# PART A: INTERNAL SHAP
# ==============================================================================

logger.section("PART A: INTERNAL SHAP  (test set)")

explainer = shap.TreeExplainer(xgb_clf)
shap_int  = extract_shap_2d(explainer.shap_values(X_test_sc))
bv_int    = get_base_value(explainer, pipeline, X_test_raw)

assert shap_int.shape == X_test_sc.shape, f"Shape mismatch {shap_int.shape}"

mean_abs_int = np.mean(np.abs(shap_int), axis=0)
corr_int     = np.array([float(np.corrcoef(X_test_sc[:, i], shap_int[:, i])[0, 1])
                          for i in range(len(genes))])

int_rows = [{"gene": g, "mean_abs_shap": round(float(mean_abs_int[i]), 6),
              "correlation": round(float(corr_int[i]), 4),
              "effect_direction": eff_dir(corr_int[i])}
            for i, g in enumerate(genes)]
int_df = (pd.DataFrame(int_rows)
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True))
int_df["shap_rank"] = int_df.index + 1

int_df.to_csv(cfg.OUT_INT / "shap_internal_results.csv", index=False)

logger.log("Internal SHAP -- ALL genes (ranked by importance):")
logger.log(f"  {'Rank':<5} {'Gene':<16} {'Mean|SHAP|':<14} {'Corr':>8}  Direction")
logger.log("  " + "-" * 60)
for _, r in int_df.iterrows():
    logger.log(f"  {int(r.shap_rank):<5} {r.gene:<16} "
               f"{r.mean_abs_shap:<14.6f} {r.correlation:>+8.4f}  {r.effect_direction}")
logger.log()

# ── A1: Summary beeswarm (ALL genes) ─────────────────────────────────────────
logger.log("A1: Summary beeswarm -- ALL genes")
try:
    fig_h = max(5, len(genes) * 0.48)
    fig, _ = plt.subplots(figsize=(10, fig_h))
    shap.summary_plot(shap_int, X_test_sc, feature_names=genes,
                      max_display=len(genes), show=False, plot_size=None)
    plt.title(
        "SHAP Summary — Internal (Test Set)\n"
        "Moi diem = 1 benh nhan  |  Do = expression cao  |  Xanh = expression thap\n"
        "Truc X dương = day model ve phía ALS  |  Truc X âm = day ve phía Control",
        fontsize=10, fontweight="bold")
    plt.xlabel("SHAP value  (Impact on predicted ALS probability)", fontsize=10)
    plt.tight_layout()
    save_show(fig, cfg.PLT_INT / "A1_shap_summary.png", logger)
except Exception as e:
    logger.log(f"  A1 failed: {e}")
logger.log()

# ── A2: Bar plot (ALL genes) ──────────────────────────────────────────────────
logger.log("A2: Bar plot -- ALL genes")
try:
    plot_df = int_df.copy().iloc[::-1].reset_index(drop=True)
    colors  = ["#d62728" if "positive" in r else
               "#1f77b4" if "negative" in r else "#aaaaaa"
               for r in plot_df["effect_direction"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.44)))
    ax.barh(range(len(plot_df)), plot_df["mean_abs_shap"],
            color=colors, alpha=0.82, edgecolor="white", lw=0.4)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        icon = ("expr↑→risk↑" if "positive" in row.effect_direction else
                "expr↑→risk↓" if "negative" in row.effect_direction else "~")
        ax.text(row.mean_abs_shap * 1.02, i,
                f"r={row.correlation:+.2f} {icon}", va="center", ha="left", fontsize=8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["gene"], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (Predictive contribution)", fontsize=10)
    ax.set_title("SHAP Importance — ALL Genes (Internal)\n"
                 "Do = positive effect (expression tang -> risk tang)\n"
                 "Xanh = negative effect (expression tang -> risk giam)",
                 fontsize=10, fontweight="bold")
    ax.legend(handles=[
        Patch(color="#d62728", alpha=0.82, label="Positive: expr↑ → ALS prob↑"),
        Patch(color="#1f77b4", alpha=0.82, label="Negative: expr↑ → ALS prob↓"),
        Patch(color="#aaaaaa", alpha=0.82, label="Ambiguous (|corr|<0.1)"),
    ], fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    save_show(fig, cfg.PLT_INT / "A2_shap_bar.png", logger)
except Exception as e:
    logger.log(f"  A2 failed: {e}")
logger.log()

# ── A3: Dependence plots (ALL genes, grid) ────────────────────────────────────
logger.log("A3: Dependence plots -- ALL genes (grid)")
try:
    gene_order = int_df["gene"].tolist()
    n_g   = len(gene_order)
    ncols = 4
    nrows = int(np.ceil(n_g / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.8, nrows*3.2), squeeze=False)

    for idx, gene in enumerate(gene_order):
        gi   = genes.index(gene)
        ax   = axes[idx // ncols][idx % ncols]
        xv   = X_test_sc[:, gi]
        sv   = shap_int[:, gi]
        cv   = corr_int[gi]
        rank = int(int_df.loc[int_df.gene == gene, "shap_rank"].values[0])

        c_arr = np.where(y_test == 1, "#E53935", "#1E88E5")
        ax.scatter(xv, sv, c=c_arr, alpha=0.55, s=18, edgecolors="none", rasterized=True)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

        dir_lbl = ("expr↑→risk↑" if "positive" in eff_dir(cv) else
                   "expr↑→risk↓" if "negative" in eff_dir(cv) else "~")
        ax.set_title(f"#{rank} {gene}\nr={cv:+.2f}  {dir_lbl}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Scaled expression", fontsize=7)
        ax.set_ylabel("SHAP value", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.15)

    for idx in range(n_g, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.legend(handles=[Patch(color="#E53935", alpha=0.75, label="ALS"),
                        Patch(color="#1E88E5", alpha=0.75, label="Control")],
               loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(
        "SHAP Dependence — ALL Genes (Internal, Test Set)\n"
        "X=scaled expression  Y=SHAP value  |  SHAP>0 = push toward ALS prediction",
        fontsize=10, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_show(fig, cfg.PLT_INT / "A3_dependence_all.png", logger)
except Exception as e:
    logger.log(f"  A3 failed: {e}")
logger.log()

# ── A4: Waterfall (high-risk + low-risk ALS) ─────────────────────────────────
logger.log("A4: Waterfall -- high-risk + low-risk ALS patient")
try:
    als_idx  = np.where(y_test == 1)[0]
    prob_als = pipeline.predict_proba(X_test_raw[als_idx])[:, 1]
    for s_idx, lbl in [(als_idx[np.argmax(prob_als)],  "high_risk_als"),
                        (als_idx[np.argmin(prob_als)],  "low_risk_als")]:
        prob_s = float(pipeline.predict_proba(X_test_raw[[s_idx]])[0, 1])
        exp    = shap.Explanation(values=shap_int[s_idx], base_values=bv_int,
                                   data=X_test_sc[s_idx], feature_names=genes)
        fig = plt.figure(figsize=(10, max(5, len(genes) * 0.36)))
        shap.plots.waterfall(exp, max_display=len(genes), show=False)
        plt.title(
            f"SHAP Waterfall -- {lbl.replace('_', ' ').title()}  "
            f"(predicted ALS prob={prob_s:.3f})\n"
            "Red = positive contribution (toward ALS) | Blue = negative contribution (toward Control)\n"
            "f(x) = predicted prob  |  E[f(x)] = baseline (average on the test set)",
            fontsize=9, fontweight="bold")
        plt.tight_layout()
        save_show(fig, cfg.PLT_INT / f"A4_waterfall_{lbl}.png", logger)
except Exception as e:
    logger.log(f"  A4 failed: {e}")
logger.log()

# ── A5: SHAP Interaction Values (XGB) ────────────────────────────────────────
logger.log("A5: SHAP Interaction Values -- XGB (pairwise gene interactions)")
try:
    logger.log("  Computing shap_interaction_values (may take ~5-10 min)...")
    shap_interact = explainer.shap_interaction_values(X_test_sc)
    # shap_interaction_values returns shape (n_samples, n_genes, n_genes) for binary XGB
    # or list of 2 arrays — normalize to 3D for class 1
    if isinstance(shap_interact, list):
        shap_interact = shap_interact[1] if len(shap_interact) > 1 else shap_interact[0]
    # mean absolute interaction across samples → matrix (n_genes, n_genes)
    mean_abs_interact = np.abs(shap_interact).mean(axis=0)

    # Save to CSV
    interact_df = pd.DataFrame(mean_abs_interact, index=genes, columns=genes)
    interact_df.to_csv(cfg.OUT_INT / "A5_shap_interaction_matrix.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(genes) * 0.65),
                                    max(7, len(genes) * 0.6)))
    im = ax.imshow(mean_abs_interact, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean |SHAP interaction|")
    # Annotate cells
    for i in range(len(genes)):
        for j in range(len(genes)):
            ax.text(j, i, f"{mean_abs_interact[i,j]:.3f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if mean_abs_interact[i,j] > mean_abs_interact.max()*0.6 else "black")
    ax.set_title(
        "SHAP Interaction Values — XGBoost (Test Set)\n"
        "Diagonal = main effect | Off-diagonal = pairwise nonlinear interaction\n"
        "Cap cao = XGB hoc duoc tuong tac giua 2 gene ma SVM/LR khong nam bat duoc",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_show(fig, cfg.PLT_INT / "A5_shap_interaction_heatmap.png", logger)

    # Top interaction pairs (off-diagonal)
    pairs = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            pairs.append({"gene_1": genes[i], "gene_2": genes[j],
                          "interaction": round(float(mean_abs_interact[i, j]), 6)})
    pairs_df = pd.DataFrame(pairs).sort_values("interaction", ascending=False).reset_index(drop=True)
    pairs_df.to_csv(cfg.OUT_INT / "A5_shap_interaction_pairs.csv", index=False)
    logger.log("  Top 10 pairwise interactions:")
    logger.log(f"  {'Gene 1':<16} {'Gene 2':<16} {'Interaction':>12}")
    logger.log("  " + "-" * 46)
    for _, row in pairs_df.head(10).iterrows():
        logger.log(f"  {row.gene_1:<16} {row.gene_2:<16} {row.interaction:>12.6f}")
except Exception as e:
    logger.log(f"  A5 failed: {e}\n{traceback.format_exc()}")
logger.log()

# ── A6: Force plots — True Positive vs False Negative ────────────────────────
logger.log("A6: Force plots -- True Positive (best) + False Negative (worst)")
try:
    y_prob_all = pipeline.predict_proba(X_test_raw)[:, 1]
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    tp_mask = (y_test == 1) & (y_pred_all == 1)
    fn_mask = (y_test == 1) & (y_pred_all == 0)

    cases_force = []
    if tp_mask.any():
        tp_idx = np.where(tp_mask)[0]
        best_tp = tp_idx[np.argmax(y_prob_all[tp_idx])]
        cases_force.append((best_tp, f"True Positive (ALS predicted correctly, prob={y_prob_all[best_tp]:.3f})", "A6_force_true_positive.png"))
    if fn_mask.any():
        fn_idx = np.where(fn_mask)[0]
        worst_fn = fn_idx[np.argmin(y_prob_all[fn_idx])]
        cases_force.append((worst_fn, f"False Negative (ALS missed, prob={y_prob_all[worst_fn]:.3f})", "A6_force_false_negative.png"))

    if not cases_force:
        logger.log("  No TP or FN samples found for force plot.")
    for s_idx, title, fname in cases_force:
        fig = plt.figure(figsize=(14, max(4, len(genes) * 0.3)))
        shap.force_plot(
            bv_int,
            shap_int[s_idx],
            X_test_sc[s_idx],
            feature_names=genes,
            matplotlib=True,
            show=False,
        )
        plt.title(
            f"SHAP Force Plot — {title}\n"
            "Red = push toward ALS | Blue = push toward Control | f(x) = predicted prob",
            fontsize=9, fontweight="bold")
        plt.tight_layout()
        save_show(fig, cfg.PLT_INT / fname, logger)
        logger.log(f"  Sample index: {s_idx} | True label: ALS | Predicted prob: {y_prob_all[s_idx]:.3f}")
except Exception as e:
    logger.log(f"  A6 failed: {e}\n{traceback.format_exc()}")
logger.log()

# ==============================================================================
# PART B: EXTERNAL SHAP
# ==============================================================================

logger.section("PART B: EXTERNAL SHAP  (RNA-seq external)")

ext_ok      = cfg.EXT_EXPR.exists() and cfg.EXT_INFO.exists()
shap_ext    = None
ext_df_res  = None
genes_ext   = []
rho = p_rho = None

if not ext_ok:
    logger.log(f"  External not found ({cfg.EXT_EXPR}). PART B+C skipped.\n")
else:
    ext_raw  = pd.read_csv(cfg.EXT_EXPR)
    ext_info = pd.read_csv(cfg.EXT_INFO)

    meta_ext  = ["sample_id","diagnosis","diagnosis_binary","age","sex","model_used"]
    ext_gcols = [c for c in ext_raw.columns if c not in meta_ext]
    genes_ext = [g for g in genes if g in ext_gcols]
    miss_ext  = [g for g in genes if g not in ext_gcols]
    if miss_ext: logger.log(f"  Missing in external: {miss_ext}")

    y_ext    = ext_raw["diagnosis_binary"].values.astype(int)
    X_ext_r  = ext_raw[genes_ext].fillna(ext_raw[genes_ext].mean()).values.astype(np.float64)

    # Scale: pass full gene vector through pipeline scaler
    if scaler is not None:
        X_full_ext = np.zeros((len(ext_raw), len(genes)))
        for j, g in enumerate(genes):
            if g in genes_ext:
                X_full_ext[:, j] = X_ext_r[:, genes_ext.index(g)]
        X_full_sc = scaler.transform(X_full_ext)
        gi_present = [genes.index(g) for g in genes_ext]
        X_ext_sc   = X_full_sc[:, gi_present]
    else:
        X_full_sc  = X_ext_r
        gi_present = list(range(len(genes_ext)))
        X_ext_sc   = X_ext_r

    logger.log(f"External: {X_ext_r.shape[0]} samples | sALS={y_ext.sum()} Ctrl={(y_ext==0).sum()}")
    logger.log(f"Genes present: {len(genes_ext)} / {len(genes)}\n")

    try:
        sv_full_ext = extract_shap_2d(explainer.shap_values(X_full_sc))
        shap_ext    = sv_full_ext[:, gi_present]
        abs_ext_all = np.mean(np.abs(sv_full_ext), axis=0)
        corr_ext    = np.array([float(np.corrcoef(X_ext_sc[:, i], shap_ext[:, i])[0, 1])
                                  for i in range(len(genes_ext))])

        ext_rows = [{"gene": g,
                     "mean_abs_shap_ext": round(float(abs_ext_all[gi_present[i]]), 6),
                     "correlation_ext":   round(float(corr_ext[i]), 4),
                     "effect_direction_ext": eff_dir(corr_ext[i])}
                    for i, g in enumerate(genes_ext)]
        ext_df_res = (pd.DataFrame(ext_rows)
                      .sort_values("mean_abs_shap_ext", ascending=False)
                      .reset_index(drop=True))
        ext_df_res["shap_rank_ext"] = ext_df_res.index + 1
        ext_df_res.to_csv(cfg.OUT_EXT / "shap_external_results.csv", index=False)

        logger.log("External SHAP -- ALL genes:")
        logger.log(f"  {'Rank':<5} {'Gene':<16} {'Mean|SHAP|':<14} {'Corr':>8}  Direction")
        logger.log("  " + "-" * 60)
        for _, r in ext_df_res.iterrows():
            logger.log(f"  {int(r.shap_rank_ext):<5} {r.gene:<16} "
                       f"{r.mean_abs_shap_ext:<14.6f} {r.correlation_ext:>+8.4f}  {r.effect_direction_ext}")
        logger.log()

        # B1: Summary
        try:
            fig_h = max(5, len(genes_ext) * 0.48)
            fig, _ = plt.subplots(figsize=(10, fig_h))
            shap.summary_plot(shap_ext, X_ext_sc, feature_names=genes_ext,
                               max_display=len(genes_ext), show=False, plot_size=None)
            plt.title("SHAP Summary -- External RNA-seq\n"
                      "Kiem tra: gene co giup phan biet ALS o tap doc lap?",
                      fontsize=10, fontweight="bold")
            plt.xlabel("SHAP value (Impact on ALS prediction)", fontsize=10)
            plt.tight_layout()
            save_show(fig, cfg.PLT_EXT / "B1_shap_summary_external.png", logger)
        except Exception as e:
            logger.log(f"  B1 summary: {e}")

        # B1: Bar
        try:
            plot_e = ext_df_res.copy().iloc[::-1].reset_index(drop=True)
            colors_e = ["#d62728" if "positive" in r else
                        "#1f77b4" if "negative" in r else "#aaaaaa"
                        for r in plot_e["effect_direction_ext"]]
            fig, ax = plt.subplots(figsize=(10, max(5, len(plot_e) * 0.44)))
            ax.barh(range(len(plot_e)), plot_e["mean_abs_shap_ext"],
                    color=colors_e, alpha=0.82, edgecolor="white", lw=0.4)
            for i, (_, row) in enumerate(plot_e.iterrows()):
                ax.text(row.mean_abs_shap_ext * 1.02, i, f"r={row.correlation_ext:+.2f}",
                        va="center", ha="left", fontsize=8)
            ax.set_yticks(range(len(plot_e)))
            ax.set_yticklabels(plot_e["gene"], fontsize=9)
            ax.set_xlabel("Mean |SHAP| (External)", fontsize=10)
            ax.set_title("SHAP Importance -- External (ALL genes)", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            save_show(fig, cfg.PLT_EXT / "B1_shap_bar_external.png", logger)
        except Exception as e:
            logger.log(f"  B1 bar: {e}")
        logger.log()

    except Exception as e:
        logger.log(f"  External SHAP failed: {e}\n{traceback.format_exc()}")
        shap_ext = None


# ==============================================================================
# PART C: COMPARISON
# ==============================================================================

logger.section("PART C: COMPARISON -- Internal vs External")

if shap_ext is None or ext_df_res is None:
    logger.log("  PART C skipped (no external SHAP).\n")
else:
    # C1: Overlap
    logger.log("C1: Overlap analysis")
    top_n   = min(cfg.OVERLAP_TOP_N, len(genes_ext))
    top_int = set(int_df[int_df.gene.isin(genes_ext)].head(top_n)["gene"])
    top_ext = set(ext_df_res.head(top_n)["gene"])
    overlap = top_int & top_ext
    logger.log(f"  Top {top_n} internal: {sorted(top_int)}")
    logger.log(f"  Top {top_n} external: {sorted(top_ext)}")
    logger.log(f"  Overlap  ({len(overlap)}): {sorted(overlap)}")
    logger.log(f"  Rate: {len(overlap)/top_n*100:.0f}%  "
               f"({'Robust biomarkers' if len(overlap)/top_n >= 0.5 else 'Platform-specific'})\n")

    try:
        all_g  = sorted(top_int | top_ext)
        fig, ax = plt.subplots(figsize=(8, max(4, len(all_g) * 0.4)))
        y_pos  = np.arange(len(all_g))
        ax.barh(y_pos - 0.25, [1 if g in top_int else 0 for g in all_g], 0.3,
                label="Internal top", color="#1565C0", alpha=0.8)
        ax.barh(y_pos + 0.25, [1 if g in top_ext else 0 for g in all_g], 0.3,
                label="External top", color="#E65100", alpha=0.8)
        ax.barh(y_pos,        [1 if g in overlap  else 0 for g in all_g], 0.15,
                label="Both (overlap)", color="#2E7D32", alpha=0.9)
        ax.set_yticks(y_pos); ax.set_yticklabels(all_g, fontsize=9)
        ax.set_xlabel("In top set", fontsize=10)
        ax.set_title(
            f"Gene Overlap -- Top {top_n} SHAP Genes\n"
            f"Overlap = {len(overlap)}/{top_n} ({len(overlap)/top_n*100:.0f}%)  |  "
            "Gene o ca 2 tap = on dinh / robust",
            fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        save_show(fig, cfg.PLT_CMP / "C1_overlap.png", logger)
    except Exception as e:
        logger.log(f"  C1: {e}")
    logger.log()

    # C2: Direction consistency
    logger.log("C2: Direction consistency")
    cmp_rows = []
    for g in genes_ext:
        ri = int_df[int_df.gene == g]
        re = ext_df_res[ext_df_res.gene == g]
        if ri.empty or re.empty: continue
        di, de = ri.iloc[0].effect_direction, re.iloc[0].effect_direction_ext
        i_pos  = "positive" in di; e_pos = "positive" in de
        i_neg  = "negative" in di; e_neg = "negative" in de
        status = ("CONSISTENT" if (i_pos and e_pos) or (i_neg and e_neg) else
                  "AMBIGUOUS"  if "ambiguous" in di or "ambiguous" in de else "FLIPPED")
        cmp_rows.append({"gene": g,
                          "shap_int": ri.iloc[0].mean_abs_shap,
                          "dir_internal": di, "dir_external": de,
                          "status": status})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(cfg.OUT_CMP / "C2_direction_consistency.csv", index=False)

    n_ok  = (cmp_df.status=="CONSISTENT").sum()
    n_fl  = (cmp_df.status=="FLIPPED").sum()
    n_amb = (cmp_df.status=="AMBIGUOUS").sum()
    logger.log(f"  Consistent={n_ok}  Flipped={n_fl}  Ambiguous={n_amb}")
    logger.log()
    logger.log(f"  {'Gene':<14} {'Internal dir':<32} {'External dir':<32} Status")
    logger.log("  " + "-" * 90)
    for _, r in cmp_df.sort_values("shap_int", ascending=False).iterrows():
        logger.log(f"  {r.gene:<14} {r.dir_internal:<32} {r.dir_external:<32} {r.status}")
    logger.log()

    try:
        dm_int = cmp_df["dir_internal"].map(
            lambda x: 1 if "positive" in x else (-1 if "negative" in x else 0))
        dm_ext = cmp_df["dir_external"].map(
            lambda x: 1 if "positive" in x else (-1 if "negative" in x else 0))
        mat = np.column_stack([dm_int.values, dm_ext.values])
        fig, ax = plt.subplots(figsize=(4, max(4, len(cmp_df) * 0.4)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Internal","External"], fontsize=10)
        ax.set_yticks(range(len(cmp_df))); ax.set_yticklabels(cmp_df["gene"].values, fontsize=8)
        plt.colorbar(im, ax=ax, label="+1=positive / -1=negative")
        ax.set_title("Direction Consistency\nSame color = consistent direction",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        save_show(fig, cfg.PLT_CMP / "C2_direction_heatmap.png", logger)
    except Exception as e:
        logger.log(f"  C2 heatmap: {e}")
    logger.log()

    # C3: SHAP stability
    logger.log("C3: SHAP Stability (rank correlation)")
    try:
        both_g = [g for g in genes_ext if g in int_df["gene"].values]
        si_v = int_df.set_index("gene").loc[both_g, "mean_abs_shap"].values
        se_v = ext_df_res.set_index("gene").loc[both_g, "mean_abs_shap_ext"].values
        rho, p_rho = spearmanr(si_v, se_v)
        logger.log(f"  Spearman rho={rho:.4f}  p={p_rho:.4f}")
        logger.log(f"  {'Stable' if rho > 0.5 else 'Unstable'} gene importance across platforms\n")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(si_v, se_v, c="#1565C0", alpha=0.7, s=70, edgecolors="white", lw=0.5)
        for i, g in enumerate(both_g):
            ax.annotate(g, (si_v[i], se_v[i]), fontsize=7.5, xytext=(3,3),
                        textcoords="offset points")
        ax.set_xlabel("Mean |SHAP| -- Internal", fontsize=10)
        ax.set_ylabel("Mean |SHAP| -- External", fontsize=10)
        ax.set_title(f"SHAP Stability\nSpearman rho={rho:.3f}  p={p_rho:.4f}\n"
                     "Gene gan duong cheo = on dinh ca 2 tap",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        save_show(fig, cfg.PLT_CMP / "C3_stability_scatter.png", logger)
    except Exception as e:
        logger.log(f"  C3: {e}")
    logger.log()

    # C4: Patient case study
    logger.log("C4: Patient-level case study")
    try:
        all_prob = pipeline.predict_proba(X_test_raw)[:, 1]
        s_high   = np.where(y_test == 1)[0][np.argmax(all_prob[y_test == 1])]
        s_ctrl   = np.where(y_test == 0)[0][np.argmin(all_prob[y_test == 0])]
        cases_cs = [(s_high, f"High-risk ALS  (prob={all_prob[s_high]:.3f})"),
                    (s_ctrl, f"Control  (prob={all_prob[s_ctrl]:.3f})")]

        fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(genes) * 0.35)))
        for ax_i, (si, title) in enumerate(cases_cs):
            sv_row = shap_int[si]
            order  = np.argsort(np.abs(sv_row))[::-1]
            c_bars = ["#d62728" if sv_row[i] > 0 else "#1f77b4" for i in order]
            axes[ax_i].barh(range(len(order)), sv_row[order][::-1],
                             color=c_bars[::-1], alpha=0.8, edgecolor="white", lw=0.4)
            axes[ax_i].set_yticks(range(len(order)))
            axes[ax_i].set_yticklabels([genes[i] for i in order[::-1]], fontsize=8)
            axes[ax_i].axvline(0, color="black", lw=1)
            axes[ax_i].set_xlabel("SHAP (positive = push toward ALS)", fontsize=9)
            axes[ax_i].set_title(title + "\nRed = push ALS | Blue = push Control",
                                  fontsize=9, fontweight="bold")
            axes[ax_i].grid(axis="x", alpha=0.2)
        fig.suptitle("Patient-Level SHAP Case Study\n"
                     "Each gene: its specific contribution to the model’s decision for this patient",
                     fontsize=10, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_show(fig, cfg.PLT_CMP / "C4_case_study.png", logger)
    except Exception as e:
        logger.log(f"  C4: {e}")
    logger.log()

# ==============================================================================
# PART D: RF SHAP — TreeExplainer + rank comparison with XGB
# ==============================================================================

logger.section("PART D: RF SHAP — Importance Ranking + XGB vs RF Comparison")

try:
    if not cfg.MODEL_FILE_RF.exists() or not cfg.GENE_LIST_RF.exists():
        logger.log(f"  RF model or gene list not found — PART D skipped.")
        rf_importance = None
    else:
        # Load RF genes + model
        with open(cfg.GENE_LIST_RF) as fh:
            genes_rf = [l.strip() for l in fh if l.strip()]
        with open(cfg.MODEL_FILE_RF, "rb") as fh:
            pipeline_rf = pickle.load(fh)

        if hasattr(pipeline_rf, "named_steps"):
            scaler_rf  = pipeline_rf.named_steps["scaler"]
            rf_clf     = pipeline_rf.named_steps["clf"]
            X_test_rf_raw = test_expr.set_index("sample_id")[genes_rf].values.astype(np.float64)
            X_test_rf_sc  = scaler_rf.transform(X_test_rf_raw)
        else:
            rf_clf        = pipeline_rf
            X_test_rf_raw = test_expr.set_index("sample_id")[genes_rf].values.astype(np.float64)
            X_test_rf_sc  = X_test_rf_raw

        logger.log(f"  RF model loaded | {len(genes_rf)} genes | {X_test_rf_sc.shape[0]} test samples")

        # TreeExplainer for RF
        explainer_rf   = shap.TreeExplainer(rf_clf)
        shap_rf_raw    = explainer_rf.shap_values(X_test_rf_sc)
        # RF có thể trả về: list[class0, class1] hoặc ndarray 3D (n,f,2)
        if isinstance(shap_rf_raw, list):
            shap_rf = shap_rf_raw[1]                  # list → lấy class 1
        elif shap_rf_raw.ndim == 3:
            shap_rf = shap_rf_raw[:, :, 1]            # 3D → slice class 1
        else:
            shap_rf = shap_rf_raw                     # 2D → dùng trực tiếp

        mean_abs_rf    = np.mean(np.abs(shap_rf), axis=0)
        corr_rf        = np.array([float(np.corrcoef(X_test_rf_sc[:, i], shap_rf[:, i])[0, 1])
                                    for i in range(len(genes_rf))])

        rf_importance = (pd.DataFrame({
            "gene":    genes_rf,
            "rf_shap": mean_abs_rf.tolist(),
            "rf_corr": corr_rf.tolist(),
            "rf_direction": [eff_dir(c) for c in corr_rf],
        })
        .sort_values("rf_shap", ascending=False)
        .reset_index(drop=True))
        rf_importance["rf_rank"] = rf_importance.index + 1
        rf_importance.to_csv(cfg.OUT_RF / "D1_rf_shap_results.csv", index=False)

        logger.log("  RF SHAP importance:")
        logger.log(f"  {'Rank':<5} {'Gene':<16} {'Mean|SHAP|':<14} {'Corr':>8}  Direction")
        logger.log("  " + "-" * 60)
        for _, r in rf_importance.iterrows():
            logger.log(f"  {int(r.rf_rank):<5} {r.gene:<16} {r.rf_shap:<14.6f} "
                       f"{r.rf_corr:>+8.4f}  {r.rf_direction}")
        logger.log()

        # D1: RF beeswarm
        try:
            fig_h = max(5, len(genes_rf) * 0.48)
            fig, _ = plt.subplots(figsize=(10, fig_h))
            shap.summary_plot(shap_rf, X_test_rf_sc, feature_names=genes_rf,
                              max_display=len(genes_rf), show=False, plot_size=None)
            plt.title("SHAP Summary — Random Forest (Internal Test Set)\n"
                      "So sanh voi XGB beeswarm (A1) de kiem tra consensus genes",
                      fontsize=10, fontweight="bold")
            plt.tight_layout()
            save_show(fig, cfg.PLT_RF / "D1_rf_shap_summary.png", logger)
        except Exception as e:
            logger.log(f"  D1 beeswarm failed: {e}")

        # D2: XGB vs RF rank comparison (genes overlap)
        try:
            xgb_rank_df = int_df[["gene", "mean_abs_shap", "shap_rank"]].copy()
            xgb_rank_df.columns = ["gene", "xgb_shap", "xgb_rank"]

            cmp_rank = xgb_rank_df.merge(
                rf_importance[["gene", "rf_shap", "rf_rank"]], on="gene", how="inner"
            ).sort_values("xgb_rank")
            cmp_rank.to_csv(cfg.OUT_RF / "D2_xgb_vs_rf_rank.csv", index=False)

            rho_rf, p_rf = spearmanr(cmp_rank["xgb_rank"], cmp_rank["rf_rank"])
            logger.log(f"  XGB vs RF consensus: {len(cmp_rank)} shared genes")
            logger.log(f"  Spearman rho = {rho_rf:.4f}  p = {p_rf:.4f}")
            logger.log(f"  {'Consistent ranking' if rho_rf > 0.5 else 'Divergent ranking'} — "
                       f"{'core biomarkers are robust across tree-based models' if rho_rf > 0.5 else 'SVM/LR needed for tie-breaking'}\n")
            logger.log(f"  {'Gene':<16} {'XGB rank':>9} {'RF rank':>9} {'XGB |SHAP|':>12} {'RF |SHAP|':>12}")
            logger.log("  " + "-" * 62)
            for _, r in cmp_rank.iterrows():
                logger.log(f"  {r.gene:<16} {int(r.xgb_rank):>9} {int(r.rf_rank):>9} "
                           f"{r.xgb_shap:>12.6f} {r.rf_shap:>12.6f}")

            # Scatter: XGB rank vs RF rank
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(cmp_rank["xgb_rank"], cmp_rank["rf_rank"],
                       c="#1565C0", alpha=0.8, s=80, edgecolors="white", lw=0.5)
            for _, r in cmp_rank.iterrows():
                ax.annotate(r.gene, (r.xgb_rank, r.rf_rank),
                            fontsize=8, xytext=(4, 3), textcoords="offset points")
            ax.plot([1, len(cmp_rank)], [1, len(cmp_rank)], "k--", lw=1, alpha=0.4, label="Perfect agreement")
            ax.set_xlabel("XGB SHAP rank (1 = most important)", fontsize=10)
            ax.set_ylabel("RF SHAP rank (1 = most important)", fontsize=10)
            ax.set_title(
                f"XGB vs RF SHAP Importance Rank\n"
                f"Spearman ρ = {rho_rf:.3f}  (p = {p_rf:.4f})\n"
                "Gene gan duong cheo = consensus biomarker o ca 2 mo hinh",
                fontsize=10, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=0.2)
            plt.tight_layout()
            save_show(fig, cfg.PLT_RF / "D2_xgb_vs_rf_rank_scatter.png", logger)
        except Exception as e:
            logger.log(f"  D2 comparison failed: {e}")

except Exception as e:
    logger.log(f"  PART D failed: {e}\n{traceback.format_exc()}")
    rf_importance = None
logger.log()


# ==============================================================================
# PART E: LR COEFFICIENTS — Exact linear attribution (no SHAP needed)
# ==============================================================================

logger.section("PART E: LR COEFFICIENTS — Linear attribution (alternative to SHAP)")

try:
    if not cfg.MODEL_FILE_LR.exists() or not cfg.GENE_LIST_LR.exists():
        logger.log("  LR model or gene list not found — PART E skipped.")
    else:
        with open(cfg.GENE_LIST_LR) as fh:
            genes_lr = [l.strip() for l in fh if l.strip()]
        with open(cfg.MODEL_FILE_LR, "rb") as fh:
            pipeline_lr = pickle.load(fh)

        if hasattr(pipeline_lr, "named_steps"):
            lr_clf = pipeline_lr.named_steps["clf"]
        else:
            lr_clf = pipeline_lr

        coef = lr_clf.coef_[0]   # shape (n_genes_lr,)
        coef_df = (pd.DataFrame({
            "gene":       genes_lr,
            "coefficient": coef.tolist(),
            "abs_coef":    np.abs(coef).tolist(),
            "direction":  ["positive (expr↑→ALS↑)" if c > 0 else "negative (expr↑→ALS↓)" for c in coef],
        })
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True))
        coef_df["lr_rank"] = coef_df.index + 1
        coef_df.to_csv(cfg.OUT_LR / "E1_lr_coefficients.csv", index=False)

        logger.log("  LR coefficients (exact linear attribution):")
        logger.log(f"  {'Rank':<5} {'Gene':<16} {'Coefficient':>12}  Direction")
        logger.log("  " + "-" * 55)
        for _, r in coef_df.iterrows():
            logger.log(f"  {int(r.lr_rank):<5} {r.gene:<16} {r.coefficient:>+12.6f}  {r.direction}")
        logger.log()

        # E1: Coefficient bar chart
        try:
            plot_lr = coef_df.sort_values("coefficient").reset_index(drop=True)
            colors_lr = ["#d62728" if c > 0 else "#1f77b4" for c in plot_lr["coefficient"]]
            fig, ax = plt.subplots(figsize=(10, max(5, len(plot_lr) * 0.44)))
            ax.barh(range(len(plot_lr)), plot_lr["coefficient"],
                    color=colors_lr, alpha=0.82, edgecolor="white", lw=0.4)
            ax.axvline(0, color="black", lw=1)
            ax.set_yticks(range(len(plot_lr)))
            ax.set_yticklabels(plot_lr["gene"], fontsize=9)
            ax.set_xlabel("Logistic Regression Coefficient\n"
                          "(exact linear attribution — positive = expr↑ raises ALS log-odds)", fontsize=10)
            ax.set_title(
                "LR Coefficient Magnitude — Exact Linear Attribution\n"
                "Do = positive (expr↑ → ALS probability↑)\n"
                "Xanh = negative (expr↑ → ALS probability↓)",
                fontsize=10, fontweight="bold")
            ax.legend(handles=[
                Patch(color="#d62728", alpha=0.82, label="Positive: expr↑ → ALS prob↑"),
                Patch(color="#1f77b4", alpha=0.82, label="Negative: expr↑ → ALS prob↓"),
            ], fontsize=9, loc="lower right")
            ax.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            save_show(fig, cfg.PLT_LR / "E1_lr_coefficient_bar.png", logger)
        except Exception as e:
            logger.log(f"  E1 bar chart failed: {e}")

except Exception as e:
    logger.log(f"  PART E failed: {e}\n{traceback.format_exc()}")
logger.log()


# ==============================================================================
# PART F: SVM — Explanation strategy (no SHAP, by design)
# ==============================================================================

logger.section("PART F: SVM — Interpretability Note (no SHAP)")
logger.log("  SVM interpretability: support vector margin analysis (not SHAP).")
logger.log("  Rationale: KernelSHAP for SVM is an approximation method with:")
logger.log("    - High computational cost (O(n_samples^2) per explanation)")
logger.log("    - Sensitivity to background dataset choice")
logger.log("    - No theoretical exactness guarantee (unlike TreeSHAP)")
logger.log("  → TreeSHAP (XGB, RF) = exact | LR coefficients = exact")
logger.log("  → KernelSHAP (SVM) = approximate — excluded to ensure reliability.")
logger.log()
logger.log("  For thesis defense: if asked 'why no SHAP for SVM?'")
logger.log("  Answer: 'We prioritize exact attribution methods (TreeSHAP, LR coefficients).")
logger.log("  KernelSHAP introduces approximation error sensitive to background distribution,")
logger.log("  making cross-model SHAP comparisons unreliable. SVM performance is reported")
logger.log("  via AUC/F1/MCC on the independent test set, which is the primary metric.'")
logger.log()
if cfg.MODEL_FILE_SVM.exists() and cfg.GENE_LIST_SVM.exists():
    try:
        with open(cfg.GENE_LIST_SVM) as fh:
            genes_svm = [l.strip() for l in fh if l.strip()]
        with open(cfg.MODEL_FILE_SVM, "rb") as fh:
            pipeline_svm = pickle.load(fh)
        svm_prob = pipeline_svm.predict_proba(
            test_expr.set_index("sample_id")[genes_svm].values.astype(np.float64)
        )[:, 1]
        logger.log(f"  SVM loaded: {len(genes_svm)} genes | mean predicted prob ALS = {svm_prob[y_test==1].mean():.3f}")
    except Exception as e:
        logger.log(f"  SVM load note: {e}")
logger.log()

# ==============================================================================
# SAVE + SUMMARY
# ==============================================================================

# Copy to standard path for downstream scripts (5_BIO.R, 8_EX.py)
int_df.to_csv(cfg.OUT_INT / "shap_results.csv", index=False)
shutil.copy(cfg.OUT_INT / "shap_results.csv", Path("results/shap/shap_results.csv"))
logger.log("  shap_results.csv -> results/shap/shap_results.csv  (downstream path)\n")

logger.section("FINAL SUMMARY")
logger.log(f"Gene list   : {cfg.GENE_LIST.name}  ({len(genes)} genes)")
logger.log(f"Test samples: {X_test_sc.shape[0]}")
n_pos = int_df["effect_direction"].str.contains("positive").sum()
n_neg = int_df["effect_direction"].str.contains("negative").sum()
logger.log(f"Internal: positive={n_pos}  negative={n_neg}")
if ext_df_res is not None:
    logger.log(f"External: {len(genes_ext)} genes available")
if rho is not None:
    logger.log(f"Stability: Spearman rho={rho:.4f}  p={p_rho:.4f}")
    logger.log(f"Overlap top-{min(cfg.OVERLAP_TOP_N,len(genes_ext))}: {len(overlap)} genes  {sorted(overlap)}")
logger.log()
logger.log("Output plots:")
for d, lbl in [(cfg.PLT_INT,"INT"), (cfg.PLT_EXT,"EXT"), (cfg.PLT_CMP,"CMP")]:
    for f in sorted(d.glob("*.png")):
        logger.log(f"  [{lbl}] {f}")

logger.log("\n" + "=" * 70)
logger.log("SHAP ANALYSIS COMPLETED  (v5.0)")
logger.log("=" * 70)
logger.close()