################################################################################
# 2_FeatureSelection.R  —  v1.0  (Adaptive thresholds edition)
#
#   version hien tai: ADAPTIVE THRESHOLDS — moi tang tu dong chon nguong:
#       SVA    : auto (permutation test), tuy chon cap
#       DEG    : binary search (FDR x logFC grid) → target [400, 600] genes
#       mRMR   : scan stability threshold           → target [60, 100] genes
#       MMPC   : alpha/max_k tu n_input             → target [20, 50] genes
#                stability threshold scan           → target [20, 50] genes
#
# TAI SAO ADAPTIVE:
#   Run 41SV: DEG=837 ok, MMPC alpha=0.30 max_k=1 → 73/75 pass (qua long)
#   Run 15SV: DEG=1488 nhieu, MMPC alpha=0.05 max_k=3 → 11 (qua chat)
#   → Can chon params phu hop voi so gene dau vao, khong hardcode
################################################################################

cat("\n")
cat(strrep("=", 80), "\n")
cat("ALS BIOMARKER — SVA + DEG(adaptive) → mRMR → MMPC  (v7.0)\n")
cat(strrep("=", 80), "\n\n")

# ==============================================================================
# 0. SETUP
# ==============================================================================

required_packages <- c("data.table", "dplyr", "limma", "MXM", "mRMRe")
bioc_packages     <- c("sva")

for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        install.packages(pkg, repos = "http://cran.rstudio.com/")
        library(pkg, character.only = TRUE)
    }
}
for (pkg in bioc_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager", repos = "http://cran.rstudio.com/")
        BiocManager::install(pkg, update = FALSE, ask = FALSE)
        library(pkg, character.only = TRUE)
    }
}

set.seed(42)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG <- list(
    train_expr = "data/ml_ready/train_expression.csv",
    train_meta = "data/ml_ready/train_metadata.csv",
    output_dir = "data/features",
    log_file   = "logs/feature_selection_sva.log",

    # ── SVA ──────────────────────────────────────────────────────────────────
    sva_method = "irw",
    # NULL = auto (khuyen dung). Override khi can: dat so nguyen (vi du 15).
    sva_n_sv   = NULL,

    # ── DEG TARGET RANGE ─────────────────────────────────────────────────────
    # Grid search tim (FDR, logFC) cho output ∈ [target_min, target_max]
    # Combinations sap xep tu chat nhat → chon combo dau tien dat range
    deg_target_min  = 400L,
    deg_target_max  = 600L,
    deg_fdr_grid    = c(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    deg_logfc_grid  = c(0.50, 0.40, 0.30, 0.25, 0.20),

    # Sub-filters (co dinh, ap sau khi chon nguong limma)
    deg_presence_threshold  = 0.50,
    deg_variance_percentile = 0.20,
    deg_bootstrap_iter      = 30,
    deg_stability_threshold = 0.50,

    # ── mRMR TARGET RANGE ────────────────────────────────────────────────────
    # Scan stability threshold de output ∈ [target_min, target_max]
    mrmr_bootstrap_iter = 30, 
    mrmr_k_per_boot     = 80, # 80
    mrmr_target_min     = 60L,
    mrmr_target_max     = 100L,  #100
    mrmr_stab_grid      = seq(0.80, 0.20, by = -0.05), # mrmr_stab_grid      = seq(0.80, 0.20, by = -0.05),
    mrmr_method         = "classic",

    # ── MMPC TARGET RANGE ────────────────────────────────────────────────────
    # alpha/max_k tinh tu n_input (khong fixed)
    # Stability threshold scan de output ∈ [target_min, target_max]
    mmpc_bootstrap_iter = 50, # 30
    mmpc_target_min     = 20L,
    mmpc_target_max     = 80L,
    mmpc_stab_grid      = seq(0.80, 0.10, by = -0.05),

    seed = 42
)

# Ham tinh (alpha, max_k) cua MMPC tu n_input
# Logic: n lon → can alpha lon hon de MMPC co the filter (neu alpha nho qua,
# co qua nhieu gene "significantly dependent" → MMPC giu het)
mmpc_params_from_n <- function(n) {
    if      (n <= 25)  list(alpha = 0.05, max_k = 3)
    else if (n <= 40)  list(alpha = 0.08, max_k = 3)
    else if (n <= 60)  list(alpha = 0.10, max_k = 2)
    else if (n <= 80)  list(alpha = 0.12, max_k = 2)
    else if (n <= 100) list(alpha = 0.15, max_k = 2)
    else               list(alpha = 0.20, max_k = 1)
}

# ==============================================================================
# CHECKPOINT FLAGS
# ==============================================================================

FLAGS <- list(
    SKIP_1B     = TRUE,
    SKIP_LAYER1 = TRUE,
    SKIP_LAYER3 = TRUE,
    SKIP_LAYER4 = FALSE
)

CKPT <- list(
    dir    = "checkpoints",
    step1b = "checkpoints/ckpt_sva_1b_sva.rds",
    layer1 = "checkpoints/ckpt_sva_layer1_deg.rds",
    layer3 = "checkpoints/ckpt_sva_layer3_mrmr.rds",
    layer4 = "checkpoints/ckpt_sva_layer4_mmpc.rds"
)

save_ckpt <- function(path, obj) {
    saveRDS(obj, path)
    cat(sprintf("  [CKPT SAVED] %s\n\n", path))
}
load_ckpt <- function(path, flag_name) {
    if (!file.exists(path))
        stop(sprintf("FLAG %s = TRUE nhung ckpt khong ton tai: %s", flag_name, path))
    obj <- readRDS(path)
    cat(sprintf("  [CKPT LOADED] %s\n\n", path))
    obj
}

dir.create(CKPT$dir, showWarnings = FALSE, recursive = TRUE)
dir.create(CONFIG$output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(CONFIG$log_file), showWarnings = FALSE, recursive = TRUE)

sink(CONFIG$log_file, split = TRUE)
cat("Started:", format(Sys.time()), "\n\n")

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================

cat(strrep("=", 80), "\n"); cat("LOAD DATA\n"); cat(strrep("=", 80), "\n\n")

train_expr <- fread(CONFIG$train_expr, data.table = FALSE)
train_meta <- fread(CONFIG$train_meta, data.table = FALSE)

sample_ids <- train_expr$sample_id
train_mat  <- as.matrix(train_expr[, -1])
rownames(train_mat) <- sample_ids
y           <- train_meta$label
cohort_bin  <- as.integer(train_meta$cohort == "V4")

cat("Train:", nrow(train_mat), "samples x", ncol(train_mat), "genes\n")
cat("  ALS:", sum(y==1), "| Control:", sum(y==0),
    "| V3:", sum(cohort_bin==0), "| V4:", sum(cohort_bin==1), "\n\n")

# ==============================================================================
# STEP 1B: SVA
# ==============================================================================

cat(strrep("=", 80), "\n")
cat("STEP 1B: SVA\n")
cat(strrep("=", 80), "\n\n")

if (FLAGS$SKIP_1B) {
    ckpt_1b    <- load_ckpt(CKPT$step1b, "SKIP_1B")
    sva_design <- ckpt_1b$sva_design
    n_sv       <- ckpt_1b$n_sv
    sv_info_df <- ckpt_1b$sv_info_df
    cat("  n_sv:", n_sv, "\n\n")
} else {
    expr_t   <- t(train_mat)
    mod_full <- model.matrix(~ y + cohort_bin)
    mod_null <- model.matrix(~ cohort_bin)

    cat("Running SVA...\n")
    cat("  Full: ~ ALS + cohort  |  Null: ~ cohort\n\n")

    sva_obj <- tryCatch({
        if (!is.null(CONFIG$sva_n_sv)) {
            cat(sprintf("  Override: n_sv = %d\n", CONFIG$sva_n_sv))
            sva(expr_t, mod=mod_full, mod0=mod_null,
                method=CONFIG$sva_method, n.sv=CONFIG$sva_n_sv)
        } else {
            sva(expr_t, mod=mod_full, mod0=mod_null, method=CONFIG$sva_method)
        }
    }, error = function(e) {
        cat("  WARNING: SVA failed:", conditionMessage(e), "\n")
        cat("  -> Fallback: design ~ ALS + cohort\n\n")
        NULL
    })

    if (!is.null(sva_obj) && sva_obj$n.sv > 0) {
        n_sv <- sva_obj$n.sv
        SVs  <- sva_obj$sv

        if (n_sv > 30)
            cat(sprintf("  NOTE: %d SVs found — may over-control expression.\n",  n_sv),
                "  If DEG target not reachable, try sva_n_sv = 15-20 in CONFIG.\n\n")

        cat(sprintf("  SVA found %d SV(s)\n\n", n_sv))

        sv_info_df <- data.frame(
            sv         = paste0("SV", seq_len(n_sv)),
            cor_cohort = apply(SVs, 2, function(v) cor(v, cohort_bin, method="spearman")),
            cor_label  = apply(SVs, 2, function(v) cor(v, y,          method="spearman")),
            stringsAsFactors = FALSE
        )
        for (i in seq_len(nrow(sv_info_df)))
            cat(sprintf("  %-6s  cor_cohort=%+.3f  cor_label=%+.3f\n",
                        sv_info_df$sv[i], sv_info_df$cor_cohort[i], sv_info_df$cor_label[i]))
        cat("\n")

        sv_df <- as.data.frame(SVs)
        colnames(sv_df) <- paste0("SV", seq_len(n_sv))
        sva_design <- cbind(model.matrix(~ y + cohort_bin), as.matrix(sv_df))
        colnames(sva_design)[1:3] <- c("Intercept", "ALS_vs_Control", "Cohort_V4")
    } else {
        n_sv <- 0
        cat("  n_sv = 0 → design ~ ALS + cohort\n\n")
        sva_design <- model.matrix(~ y + cohort_bin)
        colnames(sva_design)[1:3] <- c("Intercept", "ALS_vs_Control", "Cohort_V4")
        sv_info_df <- data.frame(sv=character(0), cor_cohort=numeric(0), cor_label=numeric(0))
    }

    cat(sprintf("  Final design: %d samples x %d columns\n\n", nrow(sva_design), ncol(sva_design)))

    write.csv(sv_info_df, file.path(CONFIG$output_dir, "step1b_sva_info.csv"), row.names=FALSE)
    cat("  step1b_sva_info.csv saved\n\n")

    save_ckpt(CKPT$step1b, list(sva_design=sva_design, n_sv=n_sv, sv_info_df=sv_info_df))
}

# ==============================================================================
# LAYER 1: DEG — ADAPTIVE THRESHOLD SEARCH
#
# Thay vi fix FDR=0.20:
#   1. Chay limma 1 lan, luu full topTable
#   2. Build grid (FDR x logFC), sap xep tu chat → long
#   3. Voi moi combo: ap sub-filters 1.2-1.4, dem gene
#   4. Chon combo dau tien dat [target_min, target_max]
#   5. Chay bootstrap stability voi combo da chon
# ==============================================================================

cat(strrep("=", 80), "\n")
cat("LAYER 1: DEG (ADAPTIVE THRESHOLD SEARCH)\n")
cat(strrep("=", 80), "\n\n")
cat(sprintf("TARGET: [%d, %d] genes  after sub-filters, before stability\n",
            CONFIG$deg_target_min, CONFIG$deg_target_max))
cat(sprintf("DESIGN: %s\n\n", paste(colnames(sva_design), collapse=" + ")))

if (FLAGS$SKIP_LAYER1) {
    ckpt_l1      <- load_ckpt(CKPT$layer1, "SKIP_LAYER1")
    deg_genes    <- ckpt_l1$deg_genes
    deg_results  <- ckpt_l1$deg_results
    deg_freq_df  <- ckpt_l1$deg_freq_df
    chosen_fdr   <- ckpt_l1$chosen_fdr
    chosen_logfc <- ckpt_l1$chosen_logfc
    cat("  deg_genes:", length(deg_genes), "\n\n")
} else {

    # ── 1.0: limma 1 lan ──────────────────────────────────────────────────
    cat("Running limma (1 run, reused for grid search)...\n")
    fit <- lmFit(t(train_mat), sva_design)
    fit <- eBayes(fit)
    deg_results <- topTable(fit, coef="ALS_vs_Control", number=Inf, sort.by="P")
    deg_results$gene <- rownames(deg_results)
    cat(sprintf("  %d genes tested\n\n", nrow(deg_results)))

    # ── Pre-compute helper stats cho grid search ───────────────────────────
    compute_cohens_d <- function(x, group) {
        g1 <- x[group==1]; g2 <- x[group==0]
        sd_p <- sqrt(((length(g1)-1)*var(g1)+(length(g2)-1)*var(g2)) /
                     (length(g1)+length(g2)-2))
        if (sd_p==0) 0 else abs(mean(g1)-mean(g2))/sd_p
    }

    max_fdr_try   <- max(CONFIG$deg_fdr_grid)
    min_logfc_try <- min(CONFIG$deg_logfc_grid)

    # Pre-filter: genes eligible for any grid combo
    pre_mask  <- deg_results$adj.P.Val < max_fdr_try &
                 abs(deg_results$logFC) > min_logfc_try
    pre_genes <- deg_results$gene[pre_mask]
    cat(sprintf("Pre-filtered candidate pool: %d genes (FDR<%.2f, |logFC|>%.2f)\n",
                length(pre_genes), max_fdr_try, min_logfc_try))

    cat("Computing Cohen's d for candidate pool...\n")
    eff_all <- setNames(
        apply(train_mat[, pre_genes, drop=FALSE], 2,
              function(g) compute_cohens_d(g, y)),
        pre_genes
    )
    var_all <- apply(train_mat[, pre_genes, drop=FALSE], 2, var)
    med_var <- median(apply(train_mat, 2, var))
    cat("  Done.\n\n")

    # ── Grid search ───────────────────────────────────────────────────────
    grid <- expand.grid(fdr=CONFIG$deg_fdr_grid, logfc=CONFIG$deg_logfc_grid,
                        stringsAsFactors=FALSE)
    # Sort: strict first (low fdr, high logfc)
    grid$score <- (1 - grid$fdr/max(CONFIG$deg_fdr_grid)) +
                  (grid$logfc/max(CONFIG$deg_logfc_grid))
    grid <- grid[order(-grid$score), ]

    t_min <- CONFIG$deg_target_min
    t_max <- CONFIG$deg_target_max
    t_mid <- as.integer((t_min+t_max)/2)

    cat(sprintf("Grid search (%d combos): FDR in {%s}  x  |logFC| in {%s}\n",
                nrow(grid),
                paste(CONFIG$deg_fdr_grid, collapse=","),
                paste(CONFIG$deg_logfc_grid, collapse=",")))
    cat(sprintf("  %-8s %-8s %-8s\n", "FDR", "|logFC|", "N_genes"))
    cat("  ", strrep("-", 30), "\n")

    chosen_fdr   <- NULL
    chosen_logfc <- NULL
    fallback_best_dist <- Inf
    fallback_fdr   <- tail(CONFIG$deg_fdr_grid, 1)
    fallback_logfc <- head(CONFIG$deg_logfc_grid, 1)

    for (i in seq_len(nrow(grid))) {
        f  <- grid$fdr[i]
        lf <- grid$logfc[i]

        g <- pre_genes[
            deg_results$adj.P.Val[match(pre_genes, deg_results$gene)] < f &
            abs(deg_results$logFC[match(pre_genes, deg_results$gene)]) > lf
        ]
        if (length(g) == 0) {
            cat(sprintf("  %-8.3f %-8.2f %-8d\n", f, lf, 0)); next
        }

        # Sub-filter 1.2: Cohen's d >= Q25
        eff_g <- eff_all[g[g %in% names(eff_all)]]
        g     <- names(eff_g)[eff_g >= quantile(eff_g, 0.25)]

        # Sub-filter 1.3: presence
        var_g <- var_all[g[g %in% names(var_all)]]
        g     <- names(var_g)[!is.na(var_g) & var_g > med_var * CONFIG$deg_presence_threshold]

        # Sub-filter 1.4: variance percentile
        if (length(g) > 0) {
            var_g2 <- var_all[g[g %in% names(var_all)]]
            thr_v  <- quantile(var_g2, CONFIG$deg_variance_percentile, na.rm=TRUE)
            g      <- names(var_g2)[!is.na(var_g2) & var_g2 > thr_v]
        }

        n_g    <- length(g)
        in_rng <- n_g >= t_min && n_g <= t_max
        dist   <- abs(n_g - t_mid)

        cat(sprintf("  %-8.3f %-8.2f %-8d%s\n", f, lf, n_g,
                    if (in_rng) " <- IN RANGE" else ""))

        if (in_rng && is.null(chosen_fdr)) {
            chosen_fdr   <- f
            chosen_logfc <- lf
        }

        if (is.null(chosen_fdr) && dist < fallback_best_dist) {
            fallback_best_dist <- dist
            fallback_fdr       <- f
            fallback_logfc     <- lf
        }
    }

    if (is.null(chosen_fdr)) {
        cat(sprintf("\n  WARNING: No combo in [%d, %d]. Using closest to %d.\n",
                    t_min, t_max, t_mid))
        chosen_fdr   <- fallback_fdr
        chosen_logfc <- fallback_logfc
    }

    cat(sprintf("\n  CHOSEN: FDR=%.3f  |logFC|=%.2f\n\n", chosen_fdr, chosen_logfc))

    # ── Apply chosen thresholds ────────────────────────────────────────────
    cat(sprintf("Applying: FDR=%.3f, |logFC|=%.2f\n", chosen_fdr, chosen_logfc))

    deg_mask_basic <- deg_results$adj.P.Val < chosen_fdr &
                      abs(deg_results$logFC) > chosen_logfc
    genes_11 <- deg_results$gene[deg_mask_basic]
    cat(sprintf("  After 1.1: %d genes (UP=%d DOWN=%d)\n",
                length(genes_11),
                sum(deg_results$logFC[deg_mask_basic]>0),
                sum(deg_results$logFC[deg_mask_basic]<0)))

    eff_11 <- apply(train_mat[, genes_11, drop=FALSE], 2, function(g) compute_cohens_d(g,y))
    genes_12 <- genes_11[eff_11 >= quantile(eff_11, 0.25)]
    cat(sprintf("  After 1.2 (effect Q25): %d genes\n", length(genes_12)))

    var_12 <- apply(train_mat[, genes_12, drop=FALSE], 2, var)
    genes_13 <- genes_12[var_12 > med_var*CONFIG$deg_presence_threshold]
    cat(sprintf("  After 1.3 (presence): %d genes\n", length(genes_13)))

    var_13 <- apply(train_mat[, genes_13, drop=FALSE], 2, var)
    genes_14 <- genes_13[var_13 > quantile(var_13, CONFIG$deg_variance_percentile)]
    cat(sprintf("  After 1.4 (variance): %d genes\n\n", length(genes_14)))

    # ── 1.5: Bootstrap stability ───────────────────────────────────────────
    cat(sprintf("Bootstrap stability (%d iter, threshold %.0f%%)...\n",
                CONFIG$deg_bootstrap_iter, CONFIG$deg_stability_threshold*100))

    deg_frequency <- list()
    for (boot_iter in seq_len(CONFIG$deg_bootstrap_iter)) {
        if (boot_iter %% 5 == 0)
            cat("  Bootstrap", boot_iter, "/", CONFIG$deg_bootstrap_iter, "\n")
        set.seed(CONFIG$seed + boot_iter)

        als_i <- which(y==1); ctl_i <- which(y==0)
        b_idx <- c(sample(als_i, round(length(als_i)*0.8), replace=TRUE),
                   sample(ctl_i, round(length(ctl_i)*0.8), replace=TRUE))
        b_mat <- train_mat[b_idx, genes_14, drop=FALSE]
        b_des <- sva_design[b_idx, , drop=FALSE]

        rank_ok <- tryCatch(qr(b_des)$rank==ncol(b_des), error=function(e) FALSE)
        if (!rank_ok) {
            b_des <- model.matrix(~b_des[,"ALS_vs_Control"]+b_des[,"Cohort_V4"])
            colnames(b_des) <- c("Intercept","ALS_vs_Control","Cohort_V4")
        }

        b_fit  <- eBayes(lmFit(t(b_mat), b_des))
        b_top  <- topTable(b_fit, coef="ALS_vs_Control", number=Inf, sort.by="P")
        b_mask <- b_top$adj.P.Val < chosen_fdr & abs(b_top$logFC) > chosen_logfc
        for (gene in rownames(b_top)[b_mask]) {
            if (is.null(deg_frequency[[gene]])) deg_frequency[[gene]] <- 0
            deg_frequency[[gene]] <- deg_frequency[[gene]] + 1
        }
    }

    deg_freq_df <- data.frame(gene=names(deg_frequency),
                               frequency=unlist(deg_frequency),
                               stringsAsFactors=FALSE)
    deg_freq_df$frequency_pct <- (deg_freq_df$frequency/CONFIG$deg_bootstrap_iter)*100
    deg_freq_df <- deg_freq_df[order(-deg_freq_df$frequency), ]

    stab_pct <- CONFIG$deg_stability_threshold*100
    deg_genes <- intersect(
        deg_freq_df$gene[deg_freq_df$frequency_pct >= stab_pct],
        genes_14
    )
    cat(sprintf("\n  After 1.5 (stability >= %.0f%%): %d genes\n\n", stab_pct, length(deg_genes)))

    cat(strrep("-", 60), "\n")
    cat("LAYER 1 SUMMARY:\n")
    cat(sprintf("  %d → %d → %d → %d → %d → %d genes\n",
                ncol(train_mat), length(genes_11), length(genes_12),
                length(genes_13), length(genes_14), length(deg_genes)))
    cat(sprintf("  Chosen: FDR=%.3f | logFC=%.2f | stability=%.0f%%\n\n",
                chosen_fdr, chosen_logfc, stab_pct))

    if (length(deg_genes) < 100)
        warning("< 100 DEGs — consider relaxing deg_target or check n_sv")
    else if (length(deg_genes) > 1000)
        warning("> 1000 DEGs — target not reachable, adjust CONFIG")
    else
        cat("  DEG count: OK\n\n")

    refined <- deg_results[deg_results$gene %in% deg_genes, ]
    refined$stability_frequency <- deg_freq_df$frequency_pct[match(refined$gene, deg_freq_df$gene)]
    write.csv(refined, file.path(CONFIG$output_dir, "layer1_refined_DEGs.csv"), row.names=FALSE)
    cat("  layer1_refined_DEGs.csv saved\n\n")

    save_ckpt(CKPT$layer1, list(deg_genes=deg_genes, deg_results=deg_results,
                                 deg_freq_df=deg_freq_df, chosen_fdr=chosen_fdr,
                                 chosen_logfc=chosen_logfc))
}

deg_genes_post2 <- deg_genes
train_mat_gsva  <- train_mat[, deg_genes_post2, drop=FALSE]
cat(sprintf("Post-DEG → mRMR input: %d genes\n\n", length(deg_genes_post2)))

# ==============================================================================
# LAYER 3: mRMR — ADAPTIVE STABILITY THRESHOLD
# ==============================================================================

cat(strrep("=", 80), "\n")
cat("LAYER 3: mRMR STABILITY (ADAPTIVE THRESHOLD)\n")
cat(strrep("=", 80), "\n\n")
cat(sprintf("TARGET: [%d, %d] genes\n", CONFIG$mrmr_target_min, CONFIG$mrmr_target_max))
cat(sprintf("INPUT:  %d genes | k=%d | %d bootstraps\n\n",
            length(deg_genes_post2), CONFIG$mrmr_k_per_boot, CONFIG$mrmr_bootstrap_iter))

if (FLAGS$SKIP_LAYER3) {
    ckpt_l3          <- load_ckpt(CKPT$layer3, "SKIP_LAYER3")
    mrmr_genes       <- ckpt_l3$mrmr_genes
    mrmr_freq_df     <- ckpt_l3$mrmr_freq_df
    chosen_mrmr_stab <- ckpt_l3$chosen_mrmr_stab
    cat("  mrmr_genes:", length(mrmr_genes), "\n\n")
} else {
    mrmr_frequency        <- list()
    mrmr_score_accumulator <- list()

    cat("Running bootstrap mRMR...\n\n")
    for (boot_iter in seq_len(CONFIG$mrmr_bootstrap_iter)) {
        cat(sprintf("  mRMR %d/%d... ", boot_iter, CONFIG$mrmr_bootstrap_iter))
        set.seed(CONFIG$seed + boot_iter + 3000)

        als_i <- which(y==1); ctl_i <- which(y==0)
        b_idx <- c(sample(als_i, round(length(als_i)*0.8), replace=TRUE),
                   sample(ctl_i, round(length(ctl_i)*0.8), replace=TRUE))
        b_mat <- train_mat_gsva[b_idx, , drop=FALSE]
        b_y   <- y[b_idx]

        tryCatch({
            b_df     <- data.frame(target=as.numeric(b_y), b_mat)
            b_dd     <- mRMR.data(data=b_df)
            k_use    <- min(CONFIG$mrmr_k_per_boot, ncol(b_mat))
            b_res    <- mRMR.classic(data=b_dd, target_indices=1L, feature_count=k_use)
            b_genes  <- colnames(b_mat)[solutions(b_res)[[1]]-1L]
            b_sc     <- as.numeric(scores(b_res)[[1]]); names(b_sc) <- b_genes
            for (g in b_genes) {
                if (is.null(mrmr_frequency[[g]])) mrmr_frequency[[g]] <- 0
                mrmr_frequency[[g]] <- mrmr_frequency[[g]] + 1
                mrmr_score_accumulator[[g]] <- c(mrmr_score_accumulator[[g]], b_sc[g])
            }
            cat(sprintf("%d genes\n", length(b_genes)))
        }, error = function(e) cat(sprintf("FAILED (%s)\n", e$message)))
    }
    cat("\n  Bootstrap mRMR done.\n\n")

    if (length(mrmr_frequency) == 0) {
        mrmr_genes <- head(deg_genes_post2, CONFIG$mrmr_target_min)
        mrmr_freq_df <- data.frame(gene=mrmr_genes, frequency=CONFIG$mrmr_bootstrap_iter,
                                    frequency_pct=100, mean_score=NA_real_, stringsAsFactors=FALSE)
        chosen_mrmr_stab <- 1.0
    } else {
        mrmr_freq_df <- data.frame(gene=names(mrmr_frequency),
                                    frequency=unlist(mrmr_frequency), stringsAsFactors=FALSE)
        mrmr_freq_df$frequency_pct <- (mrmr_freq_df$frequency/CONFIG$mrmr_bootstrap_iter)*100
        mrmr_freq_df$mean_score <- sapply(mrmr_freq_df$gene, function(g) {
            sc <- mrmr_score_accumulator[[g]]
            if (!is.null(sc)) mean(sc, na.rm=TRUE) else NA_real_
        })
        mrmr_freq_df <- mrmr_freq_df[order(-mrmr_freq_df$frequency), ]

        cat("mRMR distribution:\n")
        for (t in c(80,60,50,40,30,20))
            cat(sprintf("  Freq >= %2d%%: %d genes\n", t, sum(mrmr_freq_df$frequency_pct >= t)))

        # Adaptive scan
        cat("\nScanning stability for mRMR target...\n")
        chosen_mrmr_stab <- NA
        for (stab in CONFIG$mrmr_stab_grid) {
            n_g  <- sum(mrmr_freq_df$frequency_pct >= stab*100)
            in_r <- n_g >= CONFIG$mrmr_target_min && n_g <= CONFIG$mrmr_target_max
            cat(sprintf("  >= %.0f%%: %d%s\n", stab*100, n_g, if (in_r) " <- CHOSEN" else ""))
            if (in_r && is.na(chosen_mrmr_stab)) {
                chosen_mrmr_stab <- stab
                mrmr_genes <- mrmr_freq_df$gene[mrmr_freq_df$frequency_pct >= stab*100]
            }
        }
        if (is.na(chosen_mrmr_stab)) {
            # Fallback: closest to target_mid
            t_mid_m <- as.integer((CONFIG$mrmr_target_min+CONFIG$mrmr_target_max)/2)
            best_d  <- Inf; chosen_mrmr_stab <- 0.20
            for (stab in CONFIG$mrmr_stab_grid) {
                n_g <- sum(mrmr_freq_df$frequency_pct >= stab*100)
                if (abs(n_g-t_mid_m) < best_d) { best_d <- abs(n_g-t_mid_m); chosen_mrmr_stab <- stab }
            }
            mrmr_genes <- mrmr_freq_df$gene[mrmr_freq_df$frequency_pct >= chosen_mrmr_stab*100]
            cat(sprintf("  WARNING: fallback stab=%.0f%% -> %d genes\n", chosen_mrmr_stab*100, length(mrmr_genes)))
        }
        cat(sprintf("\n  CHOSEN: %.0f%% -> %d genes\n\n", chosen_mrmr_stab*100, length(mrmr_genes)))
    }

    mrmr_df <- data.frame(gene=mrmr_freq_df$gene, frequency=mrmr_freq_df$frequency,
                           frequency_pct=mrmr_freq_df$frequency_pct,
                           mean_mrmr_score=mrmr_freq_df$mean_score,
                           selected=mrmr_freq_df$gene %in% mrmr_genes,
                           layer1_logFC=deg_results$logFC[match(mrmr_freq_df$gene, deg_results$gene)],
                           layer1_adj_pval=deg_results$adj.P.Val[match(mrmr_freq_df$gene, deg_results$gene)],
                           stringsAsFactors=FALSE)
    mrmr_df$direction <- ifelse(mrmr_df$layer1_logFC > 0, "UP", "DOWN")
    write.csv(mrmr_df, file.path(CONFIG$output_dir,"layer3_mRMR_selected.csv"), row.names=FALSE)
    cat("  layer3_mRMR_selected.csv saved\n\n")

    cat("LAYER 3 SUMMARY:\n")
    cat(sprintf("  Input: %d | Stability: %.0f%% | Output: %d\n\n",
                length(deg_genes_post2), chosen_mrmr_stab*100, length(mrmr_genes)))

    save_ckpt(CKPT$layer3, list(mrmr_genes=mrmr_genes, mrmr_freq_df=mrmr_freq_df,
                                 chosen_mrmr_stab=chosen_mrmr_stab))
}

train_mat_mrmr <- train_mat[, mrmr_genes, drop=FALSE]

# ==============================================================================
# LAYER 4: MMPC — ADAPTIVE alpha/max_k + ADAPTIVE STABILITY
# ==============================================================================

cat(strrep("=", 80), "\n")
cat("LAYER 4: MMPC STABILITY (ADAPTIVE params + ADAPTIVE threshold)\n")
cat(strrep("=", 80), "\n\n")
cat(sprintf("TARGET: [%d, %d] genes\n", CONFIG$mmpc_target_min, CONFIG$mmpc_target_max))

mmpc_p     <- mmpc_params_from_n(length(mrmr_genes))
mmpc_alpha <- mmpc_p$alpha
mmpc_max_k <- mmpc_p$max_k

cat(sprintf("INPUT:  %d genes\n", length(mrmr_genes)))
cat(sprintf("ADAPTIVE params: alpha=%.2f  max_k=%d  (derived from n=%d)\n\n",
            mmpc_alpha, mmpc_max_k, length(mrmr_genes)))

if (FLAGS$SKIP_LAYER4) {
    ckpt_l4          <- load_ckpt(CKPT$layer4, "SKIP_LAYER4")
    mmpc_genes       <- ckpt_l4$mmpc_genes
    mmpc_freq_df     <- ckpt_l4$mmpc_freq_df
    chosen_mmpc_stab <- ckpt_l4$chosen_mmpc_stab
    cat("  mmpc_genes:", length(mmpc_genes), "\n\n")
} else {
    mmpc_frequency <- list()
    cat("Running bootstrap MMPC...\n\n")

    for (mmpc_iter in seq_len(CONFIG$mmpc_bootstrap_iter)) {
        cat(sprintf("  MMPC %d/%d... ", mmpc_iter, CONFIG$mmpc_bootstrap_iter))
        set.seed(CONFIG$seed + mmpc_iter + 2000)

        als_i <- which(y==1); ctl_i <- which(y==0)
        b_idx <- c(sample(als_i, round(length(als_i)*0.8), replace=TRUE),
                   sample(ctl_i, round(length(ctl_i)*0.8), replace=TRUE))
        b_mat <- train_mat_mrmr[b_idx, , drop=FALSE]
        b_y   <- y[b_idx]

        tryCatch({
            res <- MMPC(target=b_y, dataset=b_mat,
                        max_k=mmpc_max_k, threshold=mmpc_alpha,
                        test="testIndLogistic")
            if (length(res@selectedVars) > 0) {
                sel <- mrmr_genes[res@selectedVars]
                for (g in sel) {
                    if (is.null(mmpc_frequency[[g]])) mmpc_frequency[[g]] <- 0
                    mmpc_frequency[[g]] <- mmpc_frequency[[g]] + 1
                }
                cat(length(sel), "genes\n")
            } else cat("0 genes\n")
        }, error = function(e) cat("FAILED (", e$message, ")\n"))
    }

    cat("\n  Bootstrap MMPC done.\n\n")

    if (length(mmpc_frequency) == 0) {
        warning("No MMPC output — fallback to all mRMR genes")
        mmpc_genes   <- mrmr_genes
        mmpc_freq_df <- data.frame(gene=mrmr_genes, frequency=CONFIG$mmpc_bootstrap_iter,
                                    frequency_pct=100, stringsAsFactors=FALSE)
        chosen_mmpc_stab <- 0.0
    } else {
        mmpc_freq_df <- data.frame(gene=names(mmpc_frequency),
                                    frequency=unlist(mmpc_frequency), stringsAsFactors=FALSE)
        mmpc_freq_df$frequency_pct <- (mmpc_freq_df$frequency/CONFIG$mmpc_bootstrap_iter)*100
        mmpc_freq_df <- mmpc_freq_df[order(-mmpc_freq_df$frequency), ]

        cat("MMPC distribution:\n")
        for (t in c(80,60,50,40,30,20))
            cat(sprintf("  Freq >= %2d%%: %d genes\n", t, sum(mmpc_freq_df$frequency_pct >= t)))

        # Adaptive scan
        cat("\nScanning stability for MMPC target...\n")
        chosen_mmpc_stab <- NA
        for (stab in CONFIG$mmpc_stab_grid) {
            n_g  <- sum(mmpc_freq_df$frequency_pct >= stab*100)
            in_r <- n_g >= CONFIG$mmpc_target_min && n_g <= CONFIG$mmpc_target_max
            cat(sprintf("  >= %.0f%%: %d%s\n", stab*100, n_g, if (in_r) " <- CHOSEN" else ""))
            if (in_r && is.na(chosen_mmpc_stab)) {
                chosen_mmpc_stab <- stab
                mmpc_genes <- mmpc_freq_df$gene[mmpc_freq_df$frequency_pct >= stab*100]
            }
        }
        if (is.na(chosen_mmpc_stab)) {
            t_mid_c <- as.integer((CONFIG$mmpc_target_min+CONFIG$mmpc_target_max)/2)
            best_d  <- Inf; chosen_mmpc_stab <- 0.20
            for (stab in CONFIG$mmpc_stab_grid) {
                n_g <- sum(mmpc_freq_df$frequency_pct >= stab*100)
                if (abs(n_g-t_mid_c) < best_d) { best_d <- abs(n_g-t_mid_c); chosen_mmpc_stab <- stab }
            }
            mmpc_genes <- mmpc_freq_df$gene[mmpc_freq_df$frequency_pct >= chosen_mmpc_stab*100]
            cat(sprintf("  WARNING: fallback stab=%.0f%% -> %d genes\n", chosen_mmpc_stab*100, length(mmpc_genes)))
        }
        cat(sprintf("\n  CHOSEN: %.0f%% -> %d MMPC candidates\n\n", chosen_mmpc_stab*100, length(mmpc_genes)))
    }

    write.csv(mmpc_freq_df, file.path(CONFIG$output_dir,"layer2_MMPC_stability.csv"), row.names=FALSE)

    mmpc_df <- data.frame(
        gene=mmpc_genes,
        layer1_logFC=deg_results$logFC[match(mmpc_genes, deg_results$gene)],
        layer1_adj_pval=deg_results$adj.P.Val[match(mmpc_genes, deg_results$gene)],
        layer1_stability=deg_freq_df$frequency_pct[match(mmpc_genes, deg_freq_df$gene)],
        layer4_stability=mmpc_freq_df$frequency_pct[match(mmpc_genes, mmpc_freq_df$gene)],
        mrmr_rank=match(mmpc_genes, mrmr_genes),
        mrmr_score=mrmr_freq_df$mean_score[match(mmpc_genes, mrmr_freq_df$gene)]
    )
    write.csv(mmpc_df, file.path(CONFIG$output_dir,"layer2_MMPC.csv"), row.names=FALSE)
    cat("  layer2_MMPC.csv saved\n\n")

    save_ckpt(CKPT$layer4, list(mmpc_genes=mmpc_genes, mmpc_freq_df=mmpc_freq_df,
                                 chosen_mmpc_stab=chosen_mmpc_stab,
                                 mmpc_alpha=mmpc_alpha, mmpc_max_k=mmpc_max_k))
}

# ==============================================================================
# SAVE + SUMMARY
# ==============================================================================

cat(strrep("=", 80), "\n"); cat("SAVING RESULTS\n"); cat(strrep("=", 80), "\n\n")

write.csv(mmpc_freq_df, file.path(CONFIG$output_dir,"layer3_gene_frequency.csv"), row.names=FALSE)
cat("  layer3_gene_frequency.csv\n")

candidates_df <- data.frame(
    gene=mmpc_genes,
    mmpc_frequency_pct=mmpc_freq_df$frequency_pct[match(mmpc_genes, mmpc_freq_df$gene)],
    layer1_logFC=deg_results$logFC[match(mmpc_genes, deg_results$gene)],
    layer1_adj_pval=deg_results$adj.P.Val[match(mmpc_genes, deg_results$gene)],
    layer1_stability=deg_freq_df$frequency_pct[match(mmpc_genes, deg_freq_df$gene)],
    mrmr_frequency_pct=mrmr_freq_df$frequency_pct[match(mmpc_genes, mrmr_freq_df$gene)],
    mrmr_mean_score=mrmr_freq_df$mean_score[match(mmpc_genes, mrmr_freq_df$gene)],
    stringsAsFactors=FALSE
)
candidates_df$direction <- ifelse(candidates_df$layer1_logFC > 0, "UP", "DOWN")
candidates_df <- candidates_df[order(-candidates_df$mmpc_frequency_pct), ]
write.csv(candidates_df, file.path(CONFIG$output_dir,"mmpc_candidates.csv"), row.names=FALSE)
cat(sprintf("  mmpc_candidates.csv  (%d genes)\n", nrow(candidates_df)))
writeLines(candidates_df$gene, file.path(CONFIG$output_dir,"mmpc_candidates.txt"))
cat("  mmpc_candidates.txt\n\n")

cat(strrep("=", 80), "\n")
cat("PIPELINE SUMMARY  (v7.0 — Adaptive)\n")
cat(strrep("=", 80), "\n\n")

cat("ADAPTIVE DECISIONS:\n")
cat(sprintf("  SVA   n_sv              = %d  (auto)\n", n_sv))
cat(sprintf("  DEG   FDR               = %.3f (searched from grid)\n", chosen_fdr))
cat(sprintf("  DEG   logFC             = %.2f  (searched from grid)\n", chosen_logfc))
cat(sprintf("  mRMR  stability         = %.0f%% (scanned to hit [%d,%d])\n",
            chosen_mrmr_stab*100, CONFIG$mrmr_target_min, CONFIG$mrmr_target_max))
cat(sprintf("  MMPC  alpha             = %.2f  (from n_input=%d)\n", mmpc_alpha, length(mrmr_genes)))
cat(sprintf("  MMPC  max_k             = %d    (from n_input=%d)\n", mmpc_max_k, length(mrmr_genes)))
cat(sprintf("  MMPC  stability         = %.0f%% (scanned to hit [%d,%d])\n\n",
            chosen_mmpc_stab*100, CONFIG$mmpc_target_min, CONFIG$mmpc_target_max))

cat("GENE COUNTS:\n")
cat(sprintf("  Input:       %6d\n", ncol(train_mat)))
cat(sprintf("  After DEG:   %6d  (target %d-%d)\n",
            length(deg_genes), CONFIG$deg_target_min, CONFIG$deg_target_max))
cat(sprintf("  After mRMR:  %6d  (target %d-%d)\n",
            length(mrmr_genes), CONFIG$mrmr_target_min, CONFIG$mrmr_target_max))
cat(sprintf("  After MMPC:  %6d  (target %d-%d) -> Python\n\n",
            length(mmpc_genes), CONFIG$mmpc_target_min, CONFIG$mmpc_target_max))

cat(sprintf("REDUCTION: %d -> %d (%.1f%%)\n\n",
            ncol(train_mat), length(mmpc_genes),
            (1-length(mmpc_genes)/ncol(train_mat))*100))

cat(strrep("=", 80), "\n")
cat("FEATURE SELECTION COMPLETED  (v7.0 — Adaptive)\n")
cat(strrep("=", 80), "\n\n")

sink()