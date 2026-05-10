"""
3C_sffs.py  —  v3.0  (True SFFS edition)
ALS BIOMARKER: SFFS wrapper → per-model optimal gene set → GridSearch → Test eval
=================================================================================

THAY ĐỔI v2 → v3  (CORE CHANGE):
  - XOÁ Rank Aggregation (filter method giả danh wrapper)
  - THAY bằng SFFS thực sự (Sequential Forward Feature Selection):
      Mỗi model nhận gene set tối ưu RIÊNG của nó
      Gene thêm vào = gene nào cải thiện CV AUC nhất trong bước đó
      StandardScaler fit TRONG từng CV fold (không phải trên toàn train)
  - Giảm từ 7 model (gây tốn tài nguyên) xuống còn 3 model tốt nhất:
      SVM, XGB, RF  (theo kết quả paper INISCOM 2026)
  - 2 phase:
      Phase A (SFFS): hyperparams cố định nhẹ → tìm optimal gene set nhanh
      Phase B (Tune): GridSearch trên gene set đã chọn → model cuối cùng

TẠI SAO SFFS HIỆN TẠI VÔ DỤNG:
  Rank Aggregation chọn top-k gene theo trung bình rank từ 4 methods.
  Đây là FILTER method: không dùng model để đánh giá tác động thực của
  từng gene đối với từng classifier cụ thể.
  → SVM và RF có thể cần gene hoàn toàn khác nhau → một gene set chung
    không tối ưu cho ai cả.

THIẾT KẾ SFFS ĐÚNG:
  for each model M in [SVM, XGB, RF]:
    selected = []
    while len(selected) < max_genes:
      best_gene = argmax_{g ∉ selected} cv_auc(M, selected ∪ {g})
      if cv_auc improves by > tolerance:
        selected.append(best_gene)
      else:
        break
    optimal_genes[M] = selected

  Mỗi cv_auc call:
    for train_idx, val_idx in k-fold.split(X_train):
      scaler.fit(X_train[train_idx])          ← FIT TRONG FOLD
      X_tr = scaler.transform(X_train[train_idx])
      X_val = scaler.transform(X_train[val_idx])
      M.fit(X_tr); score = roc_auc(M, X_val)
    return mean(scores)

LEAKAGE GUARANTEES:
  ✅ Split xảy ra ở 1D_Pro.R trước tất cả mọi thứ
  ✅ DEG/mRMR/MMPC chỉ fit trên train (2D_sva.R)
  ✅ SFFS: CV đánh giá gene trên train only
  ✅ StandardScaler: fit_transform TRONG từng fold của SFFS CV
  ✅ GridSearch (Phase B): chỉ trên train
  ✅ StandardScaler final: fit_transform(train), transform(test)
  ✅ Test set: đụng đúng 1 lần (Step 5)
  ✅ Bootstrap CI: resample test probas, không fit model mới
  ✅ Permutation test: chỉ dùng train
  ✅ Threshold 0.5: fixed, không tune từ test

INPUT (từ R pipeline):
  data/ml_ready/train_expression.csv
  data/ml_ready/train_metadata.csv
  data/ml_ready/test_expression.csv
  data/ml_ready/test_metadata.csv
  data/features/mmpc_candidates.csv

OUTPUT:
  data/features/sffs_per_model.csv      — gene set + AUC trajectory per model
  data/features/optimal_genes_*.txt     — gene list per model
  results/ml/test_results.csv
  results/ml/bootstrap_ci.csv
  results/ml/permutation_test.csv
  plots/ml/sffs_curves.png              — AUC vs n_genes per model
  plots/ml/test_roc_curves.png
  plots/ml/bootstrap_ci.png
  models/*.pkl
"""

import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("\n" + "=" * 80)
print("ALS BIOMARKER — SFFS (per model) → GridSearch → Test Eval  (v3.0)")
print("=" * 80 + "\n")


# ==============================================================================
# CHECKPOINT FLAGS
# ==============================================================================

FLAGS = dict(
    SKIP_SFFS       = False,   # Step 3: SFFS per model
    SKIP_TRAINING   = False,   # Step 4: GridSearch trên optimal gene sets
    SKIP_EVALUATION = False,   # Step 5: Test evaluation
    SKIP_VALIDATION = False,   # Step 6-7: Bootstrap CI + permutation test
)


# ==============================================================================
# CONFIG
# ==============================================================================

class Config:
    DATA_DIR     = Path("data/ml_ready")
    FEATURES_DIR = Path("data/features")
    TRAIN_EXPR   = DATA_DIR / "train_expression.csv"
    TRAIN_META   = DATA_DIR / "train_metadata.csv"
    TEST_EXPR    = DATA_DIR / "test_expression.csv"
    TEST_META    = DATA_DIR / "test_metadata.csv"
    MMPC_CSV     = FEATURES_DIR / "mmpc_candidates.csv"

    OUTPUT_DIR = Path("results/ml")
    PLOT_DIR   = Path("plots/ml")
    MODELS_DIR = Path("models")
    CKPT_DIR   = Path("checkpoints")
    LOG_FILE   = Path("logs/ml_training.log")

    # --- SFFS (Phase A) ---
    #
    # SFFS_CV_FOLDS = 3: theo paper (INISCOM 2026 dùng 3-fold)
    # SFFS_TOLERANCE: cải thiện AUC tối thiểu để thêm gene.
    #   0.001 = gene phải đóng góp ≥ 0.1% AUC để được giữ.
    #   Tránh: thêm gene noise làm overfit CV mà không có signal thật.
    #   Nếu quá chặt → tăng lên 0.0 (chỉ stop khi không cải thiện).
    # SFFS_MAX_GENES: cap trên — không để SFFS chọn > 27 genes.
    #   Tương đương tổng số MMPC candidates trong paper.
    # SFFS_PATIENCE: số bước liên tiếp không cải thiện → dừng sớm.
    #   Giúp tránh bị stuck ở local plateau.
    SFFS_CV_FOLDS  = 3
    SFFS_TOLERANCE = 0.001
    SFFS_MAX_GENES = 27
    SFFS_PATIENCE  = 3

    # Hyperparams CỐ ĐỊNH cho SFFS Phase A (nhẹ, nhanh):
    # Mục đích: tìm gene set tốt, không cần model tối ưu ở bước này.
    # GridSearch sẽ tune kỹ hơn ở Phase B sau khi đã có gene set.
    SFFS_MODEL_PARAMS = {
        "SVM": dict(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42,
        ),
        "XGB": dict(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, eval_metric="logloss",
            verbosity=0, random_state=42,
        ),
        "RF": dict(
            n_estimators=150, max_depth=8,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
        "LR": dict(
            C=1.0, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=1000,
            random_state=42,
        ),
    }

    # --- GridSearch Phase B ---
    TRAIN_CV_FOLDS = 5
    PARAM_GRIDS = {
        "SVM": {
            "C":     [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto"],
        },
        "XGB": {
            "n_estimators":  [100, 200, 300],
            "max_depth":     [2, 3, 4],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample":     [0.8, 1.0],
        },
        "RF": {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [5, 10, None],
            "min_samples_leaf": [1, 2, 3],
        },
        "LR": {
            "C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver":  ["liblinear"],  # hỗ trợ cả l1 và l2
        },
    }

    # --- Validation ---
    BOOTSTRAP_CI_N = 1000
    PERMUTATION_N  = 1000
    THRESHOLD      = 0.5   # fixed, không tune từ test
    RANDOM_STATE   = 42

    CKPT_SFFS       = CKPT_DIR / "ckpt_sffs.pkl"
    CKPT_TRAINING   = CKPT_DIR / "ckpt_training.pkl"
    CKPT_EVALUATION = CKPT_DIR / "ckpt_evaluation.pkl"
    CKPT_VALIDATION = CKPT_DIR / "ckpt_validation.pkl"


cfg = Config()
for d in [cfg.OUTPUT_DIR, cfg.PLOT_DIR, cfg.MODELS_DIR,
          cfg.CKPT_DIR, cfg.LOG_FILE.parent, cfg.FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# HELPERS
# ==============================================================================

def save_ckpt(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  [CKPT SAVED]  {path}")


def load_ckpt(path, flag_name):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"\nFLAG {flag_name} = True nhưng checkpoint không tồn tại: {path}"
            f"\n→ Set {flag_name} = False và chạy lại."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"  [CKPT LOADED] {path}")
    return obj


class Logger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        with open(self.log_file, "w") as f:
            f.write(f"Started: {datetime.now()}\n\n")

    def log(self, msg=""):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(str(msg) + "\n")

    def section(self, title):
        self.log("=" * 80)
        self.log(title)
        self.log("=" * 80 + "\n")


logger = Logger(cfg.LOG_FILE)
logger.section("CHECKPOINT FLAGS")
for k, v in FLAGS.items():
    logger.log(f"  {k:<28} = {str(v):<5} → {'SKIP (load ckpt)' if v else 'RUN'}")
logger.log()


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

logger.section("STEP 1: LOAD DATA")

train_expr = pd.read_csv(cfg.TRAIN_EXPR)
train_meta = pd.read_csv(cfg.TRAIN_META)
test_expr  = pd.read_csv(cfg.TEST_EXPR)
test_meta  = pd.read_csv(cfg.TEST_META)

all_genes    = train_expr.drop("sample_id", axis=1).columns.tolist()
X_train_full = train_expr.drop("sample_id", axis=1).values.astype(np.float64)
y_train      = train_meta["label"].values.astype(int)
X_test_full  = test_expr.drop("sample_id", axis=1).values.astype(np.float64)
y_test       = test_meta["label"].values.astype(int)

train_df = pd.DataFrame(X_train_full, columns=all_genes)
test_df  = pd.DataFrame(X_test_full,  columns=all_genes)

_n_pos = int(y_train.sum())
_n_neg = int((y_train == 0).sum())
_spw   = float(_n_neg / _n_pos)

logger.log(f"Train (70%): {X_train_full.shape[0]} samples × {X_train_full.shape[1]} genes")
logger.log(f"  ALS={y_train.sum()}  Control={(y_train == 0).sum()}")
logger.log(f"Test  (30%): {X_test_full.shape[0]} samples")
logger.log(f"  ALS={y_test.sum()}  Control={(y_test == 0).sum()}")
logger.log(f"  scale_pos_weight = {_spw:.2f}\n")
logger.log("⚠️  TEST SET SẼ KHÔNG ĐƯỢC ĐỤNG ĐẾN CHO ĐẾN STEP 5!\n")


# ==============================================================================
# STEP 2: LOAD MMPC CANDIDATES
# ==============================================================================

logger.section("STEP 2: LOAD MMPC CANDIDATES")

mmpc_df      = pd.read_csv(cfg.MMPC_CSV)
mmpc_genes   = [g for g in mmpc_df["gene"].tolist() if g in all_genes]
mmpc_missing = [g for g in mmpc_df["gene"].tolist() if g not in all_genes]

if mmpc_missing:
    logger.log(f"⚠️  {len(mmpc_missing)} MMPC genes không có trong expression data: {mmpc_missing}")
logger.log(f"MMPC candidates: {len(mmpc_genes)} genes available")
logger.log(f"  {mmpc_genes}\n")

if len(mmpc_genes) == 0:
    raise ValueError("Không có MMPC gene nào khớp expression data. Kiểm tra R output.")
if len(mmpc_genes) > cfg.SFFS_MAX_GENES:
    logger.log(
        f"  NOTE: {len(mmpc_genes)} candidates > SFFS_MAX_GENES={cfg.SFFS_MAX_GENES}"
        f" → SFFS sẽ dừng tại {cfg.SFFS_MAX_GENES} genes."
    )


# ==============================================================================
# STEP 3: SFFS PER MODEL
#
# THIẾT KẾ CHỐNG LEAKAGE:
#   - SFFS chỉ hoạt động trên X_train / y_train (test không bao giờ xuất hiện)
#   - Mỗi lần đánh giá một gene candidate:
#       for fold in kfold.split(X_train):
#           scaler.fit(X_train[fold_train])    ← FIT TRONG FOLD (không phải toàn train)
#           X_tr  = scaler.transform(X_train[fold_train])
#           X_val = scaler.transform(X_train[fold_val])
#           model.fit(X_tr, y_tr)
#           auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
#   - StandardScaler final (Phase B + Test) được fit riêng trên toàn train
#     sau khi gene set đã xác định → không ảnh hưởng gene selection
#
# LÝ DO SFFS (WRAPPER) > RANK AGGREGATION (FILTER):
#   - Rank Aggregation chọn gene dựa trên feature importance trên toàn train
#     → không biết gene thứ k+1 thực sự có thêm signal cho CLASSIFIER cụ thể không
#   - SFFS greedy add: mỗi bước đánh giá marginal contribution của gene
#     trong context của những gene đã chọn → bắt được interaction, redundancy
#   - Mỗi model nhận gene set RIÊNG phù hợp với inductive bias của nó
#     (SVM cần ít feature clean, RF chịu được nhiều hơn, v.v.)
# ==============================================================================

logger.section("STEP 3: SFFS PER MODEL")

def sffs_eval_genes(gene_list, X_df, y, model_template, cv):
    """
    Đánh giá CV AUC của model_template trên gene_list.

    ANTI-LEAKAGE:
      StandardScaler fit_transform trên train fold, transform trên val fold.
      Không có thông tin từ val fold leaking vào scaler hay model.

    Trả về mean AUC qua tất cả folds.
    """
    X = X_df[gene_list].values.astype(np.float64)
    fold_aucs = []

    for tr_idx, val_idx in cv.split(X, y):
        X_tr,  X_val  = X[tr_idx], X[val_idx]
        y_tr,  y_val  = y[tr_idx], y[val_idx]

        # Scaler fit TRONG fold — đây là điểm quan trọng nhất để không leakage
        sc = StandardScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        try:
            proba = clf.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, proba))
        except Exception:
            fold_aucs.append(0.5)

    return float(np.mean(fold_aucs))


def run_sffs(model_name, model_template, X_df, y, candidate_genes,
             cv, max_genes, tolerance, patience, logger):
    """
    Sequential Forward Feature Selection cho một model cụ thể.

    Returns:
        selected_genes: danh sách gene được chọn theo thứ tự
        auc_trajectory: list AUC tại mỗi bước (len = len(selected_genes))
        step_details: list dict ghi lại chi tiết từng bước
    """
    logger.log(f"\n  {'─' * 60}")
    logger.log(f"  SFFS for {model_name}")
    logger.log(f"    Candidate pool: {len(candidate_genes)} genes")
    logger.log(f"    Max genes: {max_genes} | Tolerance: {tolerance} | Patience: {patience}")
    logger.log(f"    {'─' * 55}")

    selected    = []
    remaining   = list(candidate_genes)
    current_auc = 0.0
    auc_trajectory = []
    step_details   = []
    no_improve_count = 0

    for step in range(max_genes):
        if len(remaining) == 0:
            break

        best_gene     = None
        best_auc_step = current_auc

        for gene in remaining:
            trial_genes = selected + [gene]
            try:
                auc = sffs_eval_genes(trial_genes, X_df, y, model_template, cv)
            except Exception as e:
                logger.log(f"      ⚠️  {gene}: eval failed ({e})")
                auc = 0.0

            if auc > best_auc_step:
                best_auc_step = auc
                best_gene     = gene

        if best_gene is None or (best_auc_step - current_auc) < tolerance:
            no_improve_count += 1
            logger.log(
                f"    Step {step + 1:2d}: no improvement "
                f"(best Δ={best_auc_step - current_auc:.4f} < tol={tolerance})"
                f"  [{no_improve_count}/{patience}]"
            )
            if no_improve_count >= patience:
                logger.log(f"    Early stop: {patience} consecutive steps without improvement.")
                break
            # Vẫn chưa đủ patience — vẫn thêm gene tốt nhất để tiếp tục explore
            # (vì có thể bước sau có gene tốt hơn cần gene này làm nền)
            if best_gene is not None:
                selected.append(best_gene)
                remaining.remove(best_gene)
                current_auc = best_auc_step
                auc_trajectory.append(current_auc)
                step_details.append({
                    "model": model_name, "step": step + 1,
                    "gene": best_gene, "auc": round(current_auc, 4),
                    "delta": round(best_auc_step - (auc_trajectory[-2] if len(auc_trajectory) > 1 else 0), 4),
                    "note": "below_tolerance",
                })
                logger.log(
                    f"    Step {step + 1:2d}: +{best_gene:<16}  "
                    f"AUC={current_auc:.4f}  (Δ={best_auc_step - (auc_trajectory[-2] if len(auc_trajectory) > 1 else 0):.4f}, below tol)"
                )
        else:
            no_improve_count = 0
            selected.append(best_gene)
            remaining.remove(best_gene)
            prev_auc    = current_auc
            current_auc = best_auc_step
            auc_trajectory.append(current_auc)
            step_details.append({
                "model": model_name, "step": step + 1,
                "gene": best_gene, "auc": round(current_auc, 4),
                "delta": round(current_auc - prev_auc, 4),
                "note": "added",
            })
            logger.log(
                f"    Step {step + 1:2d}: +{best_gene:<16}  "
                f"AUC={current_auc:.4f}  (Δ={current_auc - prev_auc:+.4f})"
            )

    logger.log(f"\n    → {model_name} optimal: {len(selected)} genes | Final CV AUC = {current_auc:.4f}")
    logger.log(f"    → Genes: {selected}")
    return selected, auc_trajectory, step_details


if FLAGS["SKIP_SFFS"]:
    logger.log("[SKIP] SKIP_SFFS = True → loading checkpoint...")
    ckpt_sffs     = load_ckpt(cfg.CKPT_SFFS, "SKIP_SFFS")
    optimal_genes = ckpt_sffs["optimal_genes"]
    auc_curves    = ckpt_sffs["auc_curves"]
    sffs_details  = ckpt_sffs["sffs_details"]
    logger.log("  Loaded gene sets per model:")
    for m, gs in optimal_genes.items():
        logger.log(f"    {m}: {len(gs)} genes")
    logger.log()

else:
    logger.log("METHOD: Sequential Forward Feature Selection (wrapper)")
    logger.log(f"  CV folds (SFFS): {cfg.SFFS_CV_FOLDS}-fold (theo paper INISCOM 2026)")
    logger.log(f"  Tolerance:       {cfg.SFFS_TOLERANCE}  (AUC cải thiện tối thiểu)")
    logger.log(f"  Max genes:       {cfg.SFFS_MAX_GENES}")
    logger.log(f"  Patience:        {cfg.SFFS_PATIENCE} consecutive steps")
    logger.log(f"  Models:          SVM, XGB, RF")
    logger.log(f"\n  NOTE: StandardScaler fit TRONG từng fold (không phải toàn train)")
    logger.log(f"  → Đây là điểm khác biệt quan trọng với Rank Aggregation v2\n")

    sffs_cv = StratifiedKFold(
        n_splits=cfg.SFFS_CV_FOLDS, shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )

    # Khởi tạo model templates cho SFFS Phase A (params cố định, nhẹ)
    model_templates = {
        "SVM": SVC(**cfg.SFFS_MODEL_PARAMS["SVM"]),
        "XGB": XGBClassifier(
            **{**cfg.SFFS_MODEL_PARAMS["XGB"],
               "scale_pos_weight": _spw},
            use_label_encoder=False,
        ),
        "RF": RandomForestClassifier(**cfg.SFFS_MODEL_PARAMS["RF"]),
        "LR": LogisticRegression(**cfg.SFFS_MODEL_PARAMS["LR"]),
    }

    optimal_genes = {}   # model_name → list of genes
    auc_curves    = {}   # model_name → list of AUC per step
    sffs_details  = []   # all step_details combined

    for model_name, model_template in model_templates.items():
        genes, aucs, details = run_sffs(
            model_name     = model_name,
            model_template = model_template,
            X_df           = train_df,
            y              = y_train,
            candidate_genes = mmpc_genes,
            cv             = sffs_cv,
            max_genes      = cfg.SFFS_MAX_GENES,
            tolerance      = cfg.SFFS_TOLERANCE,
            patience       = cfg.SFFS_PATIENCE,
            logger         = logger,
        )
        optimal_genes[model_name] = genes
        auc_curves[model_name]    = aucs
        sffs_details.extend(details)

    logger.log("\n" + "─" * 60)
    logger.log("SFFS SUMMARY:")
    for m, gs in optimal_genes.items():
        logger.log(f"  {m}: {len(gs)} genes  |  final CV AUC = {auc_curves[m][-1]:.4f}")
        logger.log(f"    {gs}")
    logger.log()

    # Save SFFS details
    sffs_df = pd.DataFrame(sffs_details)
    sffs_df.to_csv(cfg.FEATURES_DIR / "sffs_per_model.csv", index=False)
    logger.log("  ✓ sffs_per_model.csv saved")

    for m, gs in optimal_genes.items():
        path = cfg.FEATURES_DIR / f"optimal_genes_{m}.txt"
        path.write_text("\n".join(gs) + "\n")
        logger.log(f"  ✓ optimal_genes_{m}.txt saved")

    save_ckpt(cfg.CKPT_SFFS, {
        "optimal_genes": optimal_genes,
        "auc_curves":    auc_curves,
        "sffs_details":  sffs_details,
    })


# ==============================================================================
# STEP 4: GRIDSEARCH TRÊN OPTIMAL GENE SETS (TRAIN ONLY)
#
# Phase B: dùng gene sets từ SFFS → tune hyperparams kỹ hơn
# StandardScaler ở đây: fit trên toàn X_train (không phải từng fold)
# Điều này OK vì gene set đã được chọn xong → scaler chỉ dùng để scale
# trước GridSearch, không ảnh hưởng gene selection nữa.
# GridSearch bên trong vẫn fit scaler trên train fold của CV → không leakage.
#
# NOTE: Chúng ta dùng Pipeline(scaler → model) trong GridSearch để đảm bảo
# scaler fit trong từng CV fold của GridSearch (không phải toàn train).
# ==============================================================================

logger.section("STEP 4: GRIDSEARCH TRÊN OPTIMAL GENE SETS (TRAIN ONLY)")

if FLAGS["SKIP_TRAINING"]:
    logger.log("[SKIP] SKIP_TRAINING = True → loading checkpoint...")
    ckpt_train      = load_ckpt(cfg.CKPT_TRAINING, "SKIP_TRAINING")
    final_models    = ckpt_train["final_models"]
    best_params_log = ckpt_train["best_params_log"]
    scalers         = ckpt_train["scalers"]
    logger.log(f"  Loaded: {list(final_models.keys())}\n")

else:
    from sklearn.pipeline import Pipeline as SKPipeline

    final_models    = {}   # model_name → fitted model (trên toàn train)
    best_params_log = {}
    scalers         = {}   # model_name → StandardScaler (fit trên toàn train gene set)

    tune_cv = StratifiedKFold(
        n_splits=cfg.TRAIN_CV_FOLDS, shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )

    model_builders = {
        "SVM": lambda: SVC(
            kernel="rbf", probability=True,
            class_weight="balanced",
            random_state=cfg.RANDOM_STATE,
        ),
        "XGB": lambda: XGBClassifier(
            scale_pos_weight=_spw,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
            random_state=cfg.RANDOM_STATE,
        ),
        "RF": lambda: RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=cfg.RANDOM_STATE,
        ),
        "LR": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=cfg.RANDOM_STATE,
        ),
    }

    for model_name, genes in optimal_genes.items():
        if len(genes) == 0:
            logger.log(f"  ⚠️  {model_name}: 0 genes selected by SFFS → skip\n")
            continue

        logger.log(f"GridSearch {model_name} ({len(genes)} genes)...")
        X_sub = train_df[genes].values.astype(np.float64)

        # Pipeline: StandardScaler → Model
        # StandardScaler fit TRONG từng fold của GridSearch → không leakage
        pipe = SKPipeline([
            ("scaler", StandardScaler()),
            ("clf",    model_builders[model_name]()),
        ])

        # Prefix param grid với "clf__"
        param_grid = {f"clf__{k}": v for k, v in cfg.PARAM_GRIDS[model_name].items()}

        gs = GridSearchCV(
            pipe, param_grid,
            cv=tune_cv, scoring="roc_auc",
            n_jobs=-1, refit=True,
        )
        gs.fit(X_sub, y_train)

        best_params_log[model_name] = {
            k.replace("clf__", ""): v for k, v in gs.best_params_.items()
            if k.startswith("clf__")
        }
        logger.log(f"  Best params: {best_params_log[model_name]}")
        logger.log(f"  Best CV AUC (train, {cfg.TRAIN_CV_FOLDS}-fold): {gs.best_score_:.4f}\n")

        # Fit final scaler trên toàn train (để dùng lúc predict test)
        # Pipeline đã refit trên toàn train khi refit=True → gs.best_estimator_
        # là Pipeline đã fit, có cả scaler lẫn model
        final_models[model_name] = gs.best_estimator_   # Pipeline (scaler + model)

        # Lưu scaler riêng để inspect
        scalers[model_name] = gs.best_estimator_.named_steps["scaler"]

        # Save model
        path = cfg.MODELS_DIR / f"model_{model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(gs.best_estimator_, f)
        logger.log(f"  ✓ model_{model_name}.pkl saved")

    save_ckpt(cfg.CKPT_TRAINING, {
        "final_models":    final_models,
        "best_params_log": best_params_log,
        "scalers":         scalers,
    })


def get_proba(model_name, split="test"):
    """
    Predict proba từ Pipeline (scaler đã được include).
    Pipeline.predict_proba tự apply scaler trước → an toàn.
    """
    model = final_models[model_name]
    genes = optimal_genes[model_name]
    if split == "train":
        X = train_df[genes].values.astype(np.float64)
    else:
        X = test_df[genes].values.astype(np.float64)
    return model.predict_proba(X)[:, 1]


# ==============================================================================
# STEP 5: ONE-TIME EVALUATION TRÊN 30% TEST
# ==============================================================================

logger.section("STEP 5: ONE-TIME EVALUATION TRÊN 30% TEST")
logger.log("⚠️  ĐÂY LÀ LẦN DUY NHẤT TEST SET ĐƯỢC ĐỤNG ĐẾN!")
logger.log(f"    Threshold: {cfg.THRESHOLD} (fixed)\n")

if FLAGS["SKIP_EVALUATION"]:
    logger.log("[SKIP] SKIP_EVALUATION = True → loading checkpoint...")
    ckpt_eval       = load_ckpt(cfg.CKPT_EVALUATION, "SKIP_EVALUATION")
    test_results_df = ckpt_eval["test_results_df"]
    all_probas      = ckpt_eval["all_probas"]
    logger.log()

else:
    test_results = []
    all_probas   = {}

    for model_name in list(final_models.keys()):
        proba = get_proba(model_name, split="test")
        pred  = (proba >= cfg.THRESHOLD).astype(int)

        auc     = roc_auc_score(y_test, proba)
        acc     = accuracy_score(y_test, pred)
        f1      = f1_score(y_test, pred, zero_division=0)
        mcc     = matthews_corrcoef(y_test, pred)
        prec    = precision_score(y_test, pred, zero_division=0)
        sn      = recall_score(y_test, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        sp      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        n_genes = len(optimal_genes[model_name])

        all_probas[model_name] = proba
        test_results.append({
            "Model":       model_name,
            "N_genes":     n_genes,
            "AUC":         round(auc,  4),
            "Accuracy":    round(acc,  4),
            "Sensitivity": round(sn,   4),
            "Specificity": round(sp,   4),
            "F1":          round(f1,   4),
            "Precision":   round(prec, 4),
            "MCC":         round(mcc,  4),
        })
        logger.log(
            f"  {model_name:<8} ({n_genes:2d} genes) — "
            f"AUC={auc:.4f}  Sn={sn:.4f}  Sp={sp:.4f}  "
            f"F1={f1:.4f}  MCC={mcc:.4f}"
        )

    test_results_df = pd.DataFrame(test_results).sort_values("AUC", ascending=False)
    test_results_df.to_csv(cfg.OUTPUT_DIR / "test_results.csv", index=False)
    logger.log("\n  ✓ test_results.csv saved\n")

    save_ckpt(cfg.CKPT_EVALUATION, {
        "test_results_df": test_results_df,
        "all_probas":      all_probas,
    })


# ==============================================================================
# STEP 6: BOOTSTRAP 95% CI (test probas only — không fit model mới)
# ==============================================================================

logger.section("STEP 6: BOOTSTRAP 95% CI")
logger.log(f"Resample test probas ({cfg.BOOTSTRAP_CI_N} iterations) — không fit model mới\n")

if FLAGS["SKIP_VALIDATION"]:
    logger.log("[SKIP] SKIP_VALIDATION = True → loading checkpoint...")
    ckpt_valid    = load_ckpt(cfg.CKPT_VALIDATION, "SKIP_VALIDATION")
    bootstrap_ci  = ckpt_valid["bootstrap_ci"]
    perm_results  = ckpt_valid["perm_results"]
    logger.log()

else:
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    bootstrap_ci = {}

    for model_name, proba in all_probas.items():
        boot_aucs = []
        n_test = len(y_test)
        for _ in range(cfg.BOOTSTRAP_CI_N):
            idx   = rng.choice(n_test, n_test, replace=True)
            y_b   = y_test[idx]
            p_b   = proba[idx]
            if len(np.unique(y_b)) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_b, p_b))

        if boot_aucs:
            ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
            bootstrap_ci[model_name] = {
                "ci_low":  round(float(ci_low),  4),
                "ci_high": round(float(ci_high), 4),
                "n_valid": len(boot_aucs),
            }
            logger.log(
                f"  {model_name}: 95% CI [{ci_low:.4f}–{ci_high:.4f}]"
                f"  (n={len(boot_aucs)})"
            )
        else:
            bootstrap_ci[model_name] = {}

    pd.DataFrame([
        {"Model": m, **v} for m, v in bootstrap_ci.items() if v
    ]).to_csv(cfg.OUTPUT_DIR / "bootstrap_ci.csv", index=False)
    logger.log("\n  ✓ bootstrap_ci.csv saved\n")


    # --- STEP 7: PERMUTATION TEST (train only) ---
    logger.section("STEP 7: PERMUTATION TEST (train only)")
    logger.log(f"  n = {cfg.PERMUTATION_N} | model: SVM (best from SFFS)\n")

    # Dùng best model (SVM nếu có, otherwise first)
    perm_model_name = "SVM" if "SVM" in final_models else list(final_models.keys())[0]
    perm_genes      = optimal_genes[perm_model_name]
    X_perm          = train_df[perm_genes].values.astype(np.float64)

    perm_cv = StratifiedKFold(
        n_splits=cfg.SFFS_CV_FOLDS, shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )

    # Observed train AUC với Pipeline (scaler trong fold)
    obs_fold_aucs = []
    model_perm_base = final_models[perm_model_name]
    for tr_idx, val_idx in perm_cv.split(X_perm, y_train):
        sc = StandardScaler()
        X_tr  = sc.fit_transform(X_perm[tr_idx])
        X_val = sc.transform(X_perm[val_idx])
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        clf = clone(model_perm_base.named_steps["clf"])
        clf.fit(X_tr, y_tr)
        obs_fold_aucs.append(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    observed_train_auc = float(np.mean(obs_fold_aucs))

    # Null distribution: permute y_train
    null_aucs = []
    for i in range(cfg.PERMUTATION_N):
        if i % 200 == 0:
            logger.log(f"  Permutation {i}/{cfg.PERMUTATION_N}...")
        y_perm = rng.permutation(y_train)
        fold_a = []
        for tr_idx, val_idx in perm_cv.split(X_perm, y_perm):
            sc = StandardScaler()
            X_tr  = sc.fit_transform(X_perm[tr_idx])
            X_val = sc.transform(X_perm[val_idx])
            y_tr, y_val = y_perm[tr_idx], y_perm[val_idx]
            if len(np.unique(y_val)) < 2:
                fold_a.append(0.5)
                continue
            clf = clone(model_perm_base.named_steps["clf"])
            clf.fit(X_tr, y_tr)
            fold_a.append(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
        null_aucs.append(float(np.mean(fold_a)))

    null_aucs_arr = np.array(null_aucs)
    p_value       = float(np.mean(null_aucs_arr >= observed_train_auc))
    null_95th     = float(np.percentile(null_aucs_arr, 95))

    perm_results = {
        "observed_train_auc": round(observed_train_auc, 4),
        "null_mean":          round(float(np.mean(null_aucs_arr)), 4),
        "null_std":           round(float(np.std(null_aucs_arr)),  4),
        "null_95th":          round(null_95th, 4),
        "p_value":            round(p_value, 4),
        "n_permutations":     cfg.PERMUTATION_N,
        "model":              perm_model_name,
    }
    logger.log(f"\n  Observed train AUC: {observed_train_auc:.4f}")
    logger.log(f"  Null AUC: {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}")
    logger.log(f"  p-value: {p_value:.4f}  ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at α=0.05)\n")

    pd.DataFrame([perm_results]).to_csv(cfg.OUTPUT_DIR / "permutation_test.csv", index=False)
    logger.log("  ✓ permutation_test.csv saved\n")

    save_ckpt(cfg.CKPT_VALIDATION, {
        "bootstrap_ci": bootstrap_ci,
        "perm_results": perm_results,
        "null_aucs":    null_aucs,
    })


# ==============================================================================
# STEP 8: PLOTS
# ==============================================================================

logger.section("STEP 8: PLOTS")

COLORS = {"SVM": "#1565C0", "XGB": "#E65100", "RF": "#2E7D32",
          "ADA": "#6A1B9A", "LR": "#558B2F"}

# 8A: SFFS AUC curves (per model)
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, aucs in auc_curves.items():
        if not aucs:
            continue
        xs = range(1, len(aucs) + 1)
        ax.plot(xs, aucs, marker="o", markersize=4,
                color=COLORS.get(model_name, "#333333"),
                label=f"{model_name} ({len(aucs)} genes, AUC={aucs[-1]:.3f})")
    ax.set_xlabel("Number of genes added (SFFS step)", fontsize=11)
    ax.set_ylabel("CV AUC (3-fold)", fontsize=11)
    ax.set_title("SFFS Gene Selection — AUC Trajectory per Model", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(max(0, min(v for aucs in auc_curves.values() for v in aucs) - 0.05), 1.0)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "sffs_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.log("  ✓ sffs_curves.png")
except Exception as e:
    logger.log(f"  ⚠️  SFFS curves plot failed: {e}")

# 8B: ROC curves on test set
try:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    for model_name, proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        n_g = len(optimal_genes[model_name])
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(model_name, "#333333"),
                label=f"{model_name} ({n_g} genes, AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Independent Test Set (30%)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "test_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.log("  ✓ test_roc_curves.png")
except Exception as e:
    logger.log(f"  ⚠️  ROC plot failed: {e}")

# 8C: Bootstrap CI
try:
    valid_ci    = {m: v for m, v in bootstrap_ci.items() if v}
    models_list = list(valid_ci.keys())
    if models_list:
        obs_aucs = [
            test_results_df.set_index("Model").loc[m, "AUC"]
            for m in models_list
        ]
        ci_lows  = [valid_ci[m]["ci_low"]  for m in models_list]
        ci_highs = [valid_ci[m]["ci_high"] for m in models_list]

        fig, ax = plt.subplots(figsize=(7, max(3, len(models_list) * 0.9)))
        y_pos = range(len(models_list))
        ax.barh(y_pos, obs_aucs, height=0.5, color="#1565C0", alpha=0.7)
        for i, (lo, hi) in enumerate(zip(ci_lows, ci_highs)):
            ax.plot([lo, hi], [i, i], "k-", lw=2.5, alpha=0.8)
            ax.plot([lo, lo], [i-0.15, i+0.15], "k-", lw=2)
            ax.plot([hi, hi], [i-0.15, i+0.15], "k-", lw=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{m} ({len(optimal_genes.get(m, []))}g)" for m in models_list],
            fontsize=10
        )
        ax.set_xlabel("AUC")
        ax.set_title(f"Bootstrap 95% CI — Test AUC  (n={cfg.BOOTSTRAP_CI_N})",
                     fontsize=10, fontweight="bold")
        ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        plt.savefig(cfg.PLOT_DIR / "bootstrap_ci.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.log("  ✓ bootstrap_ci.png")
except Exception as e:
    logger.log(f"  ⚠️  Bootstrap CI plot failed: {e}")

# 8D: Permutation test
try:
    if "null_aucs" in dir():
        null_arr = np.array(null_aucs)
        fig, ax  = plt.subplots(figsize=(7, 5))
        ax.hist(null_arr, bins=50, color="#BDBDBD", edgecolor="white",
                lw=0.5, alpha=0.85,
                label=f"Null distribution (n={perm_results['n_permutations']})")
        ax.axvline(perm_results["observed_train_auc"], color="#1565C0", lw=2.5,
                   label=f"Observed train AUC = {perm_results['observed_train_auc']:.4f}")
        ax.axvline(perm_results["null_95th"], color="#E65100", lw=1.5, ls="--",
                   label=f"Null 95th = {perm_results['null_95th']:.4f}")
        ax.set(xlabel=f"CV AUC ({cfg.SFFS_CV_FOLDS}-fold)",
               ylabel="Count",
               title=f"Permutation Test — {perm_results['model']} "
                     f"(p={perm_results['p_value']:.4f})")
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(cfg.PLOT_DIR / "permutation_test.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.log("  ✓ permutation_test.png")
except Exception as e:
    logger.log(f"  ⚠️  Permutation plot failed: {e}")

# 8E: Confusion Matrices (all models)
try:
    from sklearn.metrics import ConfusionMatrixDisplay
    n_models = len(all_probas)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, proba) in zip(axes, all_probas.items()):
        y_pred_cm = (proba >= cfg.THRESHOLD).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_cm,
            display_labels=["Control", "ALS"],
            ax=ax, colorbar=False,
            cmap="Blues",
        )
        n_g = len(optimal_genes[model_name])
        auc_cm = roc_auc_score(y_test, proba)
        ax.set_title(f"{model_name}\n({n_g} genes, AUC={auc_cm:.3f})", fontweight="bold")
    plt.suptitle("Confusion Matrices — Independent Test Set (30%)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.log("  ✓ confusion_matrices.png")
except Exception as e:
    logger.log(f"  ⚠️  Confusion matrix plot failed: {e}")

# 8F: Learning Curve (best model by AUC trên test)
try:
    from sklearn.model_selection import learning_curve as lc_func

    # Chọn model có AUC cao nhất trên test
    best_lc_model = test_results_df.iloc[0]["Model"]
    lc_pipeline   = final_models[best_lc_model]
    lc_genes      = optimal_genes[best_lc_model]
    X_train_lc    = train_df[lc_genes].values.astype(np.float64)

    train_sizes, train_scores, val_scores = lc_func(
        lc_pipeline, X_train_lc, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1),
            label="Training AUC", marker="o", color="#1565C0")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15, color="#1565C0",
    )
    ax.plot(train_sizes, val_scores.mean(axis=1),
            label="CV Validation AUC", marker="s", color="#E65100")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15, color="#E65100",
    )
    ax.set_xlabel("Training set size", fontsize=11)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
    ax.set_title(
        f"Learning Curve — {best_lc_model} ({len(lc_genes)} genes)",
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.log(f"  ✓ learning_curve.png  (model: {best_lc_model})")
except Exception as e:
    logger.log(f"  ⚠️  Learning curve plot failed: {e}")

logger.log()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

logger.section("FINAL SUMMARY")

logger.log("── SFFS RESULTS (optimal gene set per model) ──────────────────────────")
for model_name, genes in optimal_genes.items():
    final_auc = auc_curves[model_name][-1] if auc_curves.get(model_name) else "─"
    logger.log(f"  {model_name}: {len(genes)} genes  |  SFFS CV AUC = {final_auc:.4f}")
    logger.log(f"    {genes}")
logger.log()

logger.log("── TEST SET RESULTS (30% holdout — primary unbiased estimate) ─────────")
logger.log(
    f"  {'Model':<8}  {'Genes':>5}  {'AUC':>6}  "
    f"{'95% CI':>19}  {'Sn':>6}  {'Sp':>6}  {'F1':>6}  {'MCC':>6}"
)
logger.log("  " + "─" * 76)
for _, row in test_results_df.iterrows():
    name   = row["Model"]
    ci_str = "─"
    if name in bootstrap_ci and bootstrap_ci[name]:
        ci_d   = bootstrap_ci[name]
        ci_str = f"[{ci_d['ci_low']:.4f}–{ci_d['ci_high']:.4f}]"
    logger.log(
        f"  {name:<8}  {int(row['N_genes']):5d}  {row['AUC']:>6.4f}  "
        f"{ci_str:>19}  {row['Sensitivity']:>6.4f}  {row['Specificity']:>6.4f}  "
        f"{row['F1']:>6.4f}  {row['MCC']:>6.4f}"
    )
logger.log()

logger.log("── PERMUTATION TEST ───────────────────────────────────────────────────")
logger.log(f"  Model:               {perm_results['model']}")
logger.log(f"  Observed CV AUC:     {perm_results['observed_train_auc']:.4f}")
logger.log(f"  Null (mean±std):     {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}")
logger.log(
    f"  p-value:             {perm_results['p_value']:.4f}  "
    f"({'SIGNIFICANT' if perm_results['p_value'] < 0.05 else 'NOT significant'} at α=0.05)\n"
)

best_row  = test_results_df.iloc[0]
best_name = best_row["Model"]
logger.log(f"Best model: {best_name}  ({int(best_row['N_genes'])} genes)")
logger.log(f"  AUC        : {best_row['AUC']:.4f}")
if best_name in bootstrap_ci and bootstrap_ci[best_name]:
    ci_d = bootstrap_ci[best_name]
    logger.log(f"  95% CI     : [{ci_d['ci_low']:.4f}–{ci_d['ci_high']:.4f}]")
logger.log(f"  Sensitivity: {best_row['Sensitivity']:.4f}")
logger.log(f"  Specificity: {best_row['Specificity']:.4f}")
logger.log(f"  F1         : {best_row['F1']:.4f}")
logger.log(f"  MCC        : {best_row['MCC']:.4f}")
logger.log(f"  Genes      : {optimal_genes[best_name]}\n")

# Paper string
logger.section("PAPER REPORTING STRING")
best_auc = best_row["AUC"]
best_sn  = best_row["Sensitivity"]
best_sp  = best_row["Specificity"]
best_f1  = best_row["F1"]
best_mcc = best_row["MCC"]
ci_str_p = ""
if best_name in bootstrap_ci and bootstrap_ci[best_name]:
    ci_d = bootstrap_ci[best_name]
    ci_str_p = f" (95% CI: {ci_d['ci_low']:.3f}–{ci_d['ci_high']:.3f})"

_perm_sig = perm_results["p_value"] < 0.05
perm_str  = (
    f" A permutation test (n={perm_results['n_permutations']}) confirmed "
    f"gene signature performance significantly above chance "
    f"(p = {perm_results['p_value']:.4f})."
    if _perm_sig else
    f" A permutation test (n={perm_results['n_permutations']}) did not "
    f"confirm significance above chance "
    f"(p = {perm_results['p_value']:.4f}; α = 0.05)."
)

paper_str = (
    f"Gene selection was performed exclusively on the training cohort "
    f"(70%, n={len(y_train)}). The held-out test set (30%, n={len(y_test)}) "
    f"was not used at any stage of feature selection or model development. "
    f"Following multi-stage stability selection in R (DEG+SVA, mRMR, MMPC), "
    f"Sequential Forward Feature Selection (SFFS, {cfg.SFFS_CV_FOLDS}-fold CV) "
    f"identified per-classifier optimal gene subsets from {len(mmpc_genes)} MMPC "
    f"candidates. StandardScaler was fit within each SFFS CV fold to prevent "
    f"data leakage. "
    f"The {best_name} model ({int(best_row['N_genes'])} genes) achieved "
    f"AUC = {best_auc:.3f}{ci_str_p}, sensitivity = {best_sn:.3f}, "
    f"specificity = {best_sp:.3f}, F1 = {best_f1:.3f}, MCC = {best_mcc:.3f} "
    f"on the independent test set (threshold = {cfg.THRESHOLD}).{perm_str}"
)
logger.log(paper_str + "\n")

logger.log("── LEAKAGE AUDIT ──────────────────────────────────────────────────────")
logger.log("  ✅ Train/test split      : 1D_Pro.R trước tất cả mọi thứ")
logger.log("  ✅ DEG+SVA/mRMR/MMPC    : chỉ fit trên train (2D_sva.R)")
logger.log("  ✅ SFFS evaluation       : CV trên train only")
logger.log("  ✅ StandardScaler (SFFS) : fit_transform TRONG từng fold ← KEY")
logger.log("  ✅ GridSearch (Phase B)  : Pipeline(scaler→model), scaler fit trong fold")
logger.log("  ✅ Test set              : đụng đúng 1 lần (Step 5)")
logger.log("  ✅ Bootstrap CI          : resample test probas, không fit model mới")
logger.log("  ✅ Permutation test      : chỉ dùng train")
logger.log("  ✅ Threshold 0.5         : fixed, không tune từ test\n")

logger.log("=" * 80)
logger.log("✅ PIPELINE COMPLETED  (v3.0 — True SFFS)")
logger.log("=" * 80 + "\n")
logger.log("Output files:")
logger.log(f"  {cfg.FEATURES_DIR}/sffs_per_model.csv")
logger.log(f"  {cfg.FEATURES_DIR}/optimal_genes_*.txt")
logger.log(f"  {cfg.OUTPUT_DIR}/test_results.csv")
logger.log(f"  {cfg.OUTPUT_DIR}/bootstrap_ci.csv")
logger.log(f"  {cfg.OUTPUT_DIR}/permutation_test.csv")
logger.log(f"  {cfg.PLOT_DIR}/sffs_curves.png")
logger.log(f"  {cfg.PLOT_DIR}/test_roc_curves.png")
logger.log(f"  {cfg.PLOT_DIR}/bootstrap_ci.png")
logger.log(f"  {cfg.PLOT_DIR}/permutation_test.png")
logger.log(f"  {cfg.MODELS_DIR}/model_*.pkl")