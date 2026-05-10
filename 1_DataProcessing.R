################################################################################
# 1_DataProcessing.R  —  v1.0
# PURPOSE: DATA CLEANING & PREPROCESSING  (đơn giản, đúng, không data leakage)
#
# PIPELINE:
#   PHASE 1 — Preprocessing kỹ thuật  (toàn bộ data — không leak label/split)
#     1.  Parse raw GEO series matrix files
#     2.  Probe → gene mapping  (mean expression across all probes per gene)
#     3.  Log2 normalization
#     4.  Metadata cleaning  (giữ ALS + Control, bỏ MIM)
#     5.  Merge cohorts  (intersect genes present in BOTH datasets, sau probe→gene)
#
#   PHASE 2 — SPLIT
#     6.  Stratified split 70/30  (Diagnosis × Cohort)
#
#   PHASE 3 — Train-fitted transforms  (fit trên TRAIN → apply sang TEST)
#     7.  Variance filter  (bỏ bottom 10% variance, threshold từ train)
#     8.  Z-score per cohort  (mean + sd fit trên train per cohort)
#
#   PHASE 4 — Save
#     9.  Save outputs
#
# TẠI SAO PIPELINE NÀY ĐÚNG VỀ MẶT ML:
#   - Phase 1 là kỹ thuật thuần túy (không dùng label, không dùng split info)
#     → không data leakage dù làm trên toàn bộ data
#   - Z-score được fit CHỈ trên train → test không "biết" mean/SD của toàn bộ data
#   - Variance threshold được tính CHỈ từ train → test giữ nguyên gene mask của train
#   - Không có ComBat / sex correction / lmFit → đơn giản hơn, ít tham số hơn,
#     ML model sẽ handle phần còn lại
#
# SO SÁNH VỚI PAPER (ICCE 2024):
#   Paper dùng StandardScaler trên toàn bộ 1042 samples → có thể có leakage nhỏ
#   Pipeline này làm đúng hơn: Z-score fit trên 70% train rồi apply sang 30% test
#
# OUTPUTS → data/ml_ready/:
#   train_expression.csv   (samples × genes, Z-scored)
#   test_expression.csv    (samples × genes, Z-scored, dùng train params)
#   train_metadata.csv
#   test_metadata.csv
#   zscore_params.csv      (mean + sd per cohort per gene — để reproduce)
################################################################################

cat("\n")
cat(strrep("=", 80), "\n")
cat("1C: DATA CLEANING & PREPROCESSING  (v1.0 — SIMPLE PIPELINE)\n")
cat(strrep("=", 80), "\n\n")

# ==============================================================================
# 0. SETUP
# ==============================================================================

required_packages <- c("data.table", "matrixStats")

for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        install.packages(pkg, repos = "http://cran.rstudio.com/")
        library(pkg, character.only = TRUE)
    }
}

set.seed(42)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG <- list(
    gse1_matrix = "data/raw/GSE112676_series_matrix.txt.gz",
    gse2_matrix = "data/raw/GSE112680_series_matrix.txt.gz",
    annot_v3    = "data/raw/GPL6947.annot.gz",
    annot_v4    = "data/raw/GPL10558.annot.gz",

    # Filter: bỏ bottom 10% variance (threshold từ train)
    min_variance_percentile = 0.10,

    # Z-score: luôn bật — fit per cohort trên train
    apply_zscore = TRUE,

    train_ratio  = 0.70,
    seed         = 42,

    output_dir = "data/ml_ready",
    log_file   = "logs/1C_preprocessing.log"
)

dir.create(CONFIG$output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(CONFIG$log_file), showWarnings = FALSE, recursive = TRUE)
sink(CONFIG$log_file, split = TRUE)
cat("Started:", format(Sys.time()), "\n\n")

# ==============================================================================
# FUNCTIONS — PHASE 1: PARSE & PREPROCESS
# ==============================================================================

parse_series_matrix <- function(file_path) {
    cat("Parsing:", basename(file_path), "\n")
    con   <- if (grepl("\\.gz$", file_path)) gzfile(file_path, "rt") else file(file_path, "rt")
    lines <- readLines(con); close(con)

    table_start <- grep("^!series_matrix_table_begin", lines)
    table_end   <- grep("^!series_matrix_table_end",   lines)
    expr_lines  <- lines[(table_start + 1):(table_end - 1)]
    expr_list   <- strsplit(expr_lines, "\t")

    header     <- gsub('"', "", expr_list[[1]])
    sample_ids <- header[-1]

    expr_matrix <- do.call(rbind, lapply(expr_list[-1], function(row) {
        row <- gsub('"', "", row); as.numeric(row[-1])
    }))
    probe_ids <- sapply(expr_list[-1], function(row) gsub('"', "", row[1]))
    rownames(expr_matrix) <- probe_ids
    colnames(expr_matrix) <- sample_ids

    sample_lines <- grep("^!Sample_", lines, value = TRUE)
    metadata     <- list()
    for (line in sample_lines) {
        parts  <- strsplit(line, "\t")[[1]]
        field  <- gsub("_ch1$", "", sub("^!Sample_", "", parts[1]))
        values <- gsub('"', "", parts[-1])
        if (length(values) == ncol(expr_matrix)) {
            if (field == "characteristics") {
                parsed      <- strsplit(values, ":", fixed = TRUE)
                char_field  <- trimws(parsed[[1]][1])
                char_values <- sapply(parsed, function(x)
                    if (length(x) >= 2) trimws(paste(x[-1], collapse = ":")) else NA)
                metadata[[char_field]] <- char_values
            } else {
                metadata[[field]] <- values
            }
        }
    }
    metadata_df <- as.data.frame(metadata, stringsAsFactors = FALSE)
    cat("  Probes:", nrow(expr_matrix), "| Samples:", ncol(expr_matrix), "\n\n")
    list(expression = expr_matrix, metadata = metadata_df)
}

parse_annotation <- function(file_path) {
    cat("Parsing annotation:", basename(file_path), "\n")
    con   <- if (grepl("\\.gz$", file_path)) gzfile(file_path, "rt") else file(file_path, "rt")
    lines <- readLines(con); close(con)

    table_start <- grep("^!platform_table_begin", lines)
    if (length(table_start) == 0) table_start <- grep("^ID\t", lines)[1] - 1

    annot <- fread(file_path, skip = table_start, sep = "\t", header = TRUE,
                   fill = TRUE, quote = "", na.strings = c("", "NA", "---"))

    id_col     <- grep("^ID$",               colnames(annot), ignore.case = TRUE)[1]
    symbol_col <- grep("Gene.symbol|Symbol", colnames(annot), ignore.case = TRUE)[1]

    annot_clean <- data.frame(
        ID          = annot[[id_col]],
        Gene_Symbol = trimws(annot[[symbol_col]]),
        stringsAsFactors = FALSE
    )
    annot_clean <- annot_clean[!is.na(annot_clean$Gene_Symbol) &
                               annot_clean$Gene_Symbol != "", ]
    cat("  Valid probes:", nrow(annot_clean), "\n\n")
    annot_clean
}

aggregate_to_genes <- function(expr_matrix, annotation) {
    # Theo paper (ICCE 2024):
    #   "the mean expression value of the gene symbol assigned to multiple probes
    #    will be considered for further analysis"
    # => Voi moi gene co nhieu probe: tinh MEAN expression qua tat ca probe (per sample)
    # => KHONG chon 1 probe dai dien (khong dung max/best probe)
    cat("  Probe -> gene (mean aggregation: trung binh tat ca probe cung gene)\n")

    probe_order <- match(rownames(expr_matrix), annotation$ID)
    genes       <- annotation$Gene_Symbol[probe_order]
    valid       <- !is.na(genes)
    expr_matrix <- expr_matrix[valid, , drop = FALSE]
    genes       <- genes[valid]

    # data.table group-by: tinh mean per sample qua tat ca probe cua cung 1 gene
    dt        <- data.table(gene = genes, as.data.frame(expr_matrix))
    gene_expr <- dt[, lapply(.SD, mean, na.rm = TRUE), by = gene]

    mat <- as.matrix(gene_expr[, -1, with = FALSE])
    rownames(mat) <- gene_expr$gene

    n_multi <- sum(table(genes) > 1)
    cat("  Probes vao:", length(genes),
        "| Gene unique:", nrow(mat),
        "| Gene co >= 2 probes (da mean):", n_multi, "\n\n")
    mat
}

log2_normalize <- function(expr_matrix, label = "") {
    cat("  Log2 normalization:", label, "\n")
    max_val <- max(expr_matrix, na.rm = TRUE)
    if (max_val > 50) {
        cat("    Raw data → log2(x + 1)\n")
        expr_matrix <- log2(expr_matrix + 1)
    } else {
        cat("    Already log-scaled → no transformation\n")
    }
    cat("    Mean:", sprintf("%.3f", mean(expr_matrix)),
        "| SD:", sprintf("%.3f", sd(expr_matrix)), "\n\n")
    expr_matrix
}

clean_metadata <- function(metadata, platform) {
    df <- metadata
    if ("diagnosis" %in% colnames(df)) {
        df$diagnosis <- toupper(trimws(df$diagnosis))
        df$diagnosis[df$diagnosis %in% c("CONTROL","CON","CTRL")] <- "Control"
        df$diagnosis[df$diagnosis %in% c("ALS","PATIENT")]        <- "ALS"
        df$diagnosis[df$diagnosis %in% c("MIMIC","MIM")]          <- "MIM"
    }
    df$sample_id    <- if ("geo_accession" %in% colnames(df)) df$geo_accession else
                           paste0(platform, "_", seq_len(nrow(df)))
    df$cohort       <- platform
    df$sample_index <- seq_len(nrow(df))
    df[, c("sample_id","cohort","sample_index","diagnosis")]
}

# ==============================================================================
# FUNCTIONS — PHASE 2: SPLIT
# Stratified by Diagnosis × Cohort — đủ cho dataset này
# ==============================================================================

stratified_split <- function(metadata, train_ratio = 0.70, seed = 42) {
    cat("Stratified split: Diagnosis × Cohort\n")
    set.seed(seed)

    # Strata: diagnosis × cohort (không cần batch — đã bỏ batch detection)
    metadata$strata <- paste(metadata$diagnosis, metadata$cohort, sep = "_")
    train_idx <- c()
    for (stratum in unique(metadata$strata)) {
        s_idx   <- which(metadata$strata == stratum)
        n_train <- max(1L, round(length(s_idx) * train_ratio))
        train_idx <- c(train_idx, sample(s_idx, n_train))
        cat("  Stratum:", stratum,
            "| Total:", length(s_idx),
            "| Train:", n_train,
            "| Test:",  length(s_idx) - n_train, "\n")
    }
    test_idx <- setdiff(seq_len(nrow(metadata)), train_idx)
    cat("  Total → Train:", length(train_idx), "| Test:", length(test_idx), "\n\n")
    list(train = train_idx, test = test_idx)
}

check_split_distribution <- function(train_meta, test_meta) {
    cat(strrep("-", 60), "\n")
    cat("SPLIT DISTRIBUTION (Diagnosis × Cohort)\n")
    cat(strrep("-", 60), "\n")
    for (cohort in unique(c(train_meta$cohort, test_meta$cohort))) {
        tr <- train_meta[train_meta$cohort == cohort, ]
        te <- test_meta[ test_meta$cohort  == cohort, ]
        cat(sprintf("  %s — Train: ALS=%d Control=%d (n=%d)  |  Test: ALS=%d Control=%d (n=%d)\n",
                    cohort,
                    sum(tr$diagnosis=="ALS"), sum(tr$diagnosis=="Control"), nrow(tr),
                    sum(te$diagnosis=="ALS"), sum(te$diagnosis=="Control"), nrow(te)))
    }
    cat("\n")
}

# ==============================================================================
# FUNCTIONS — PHASE 3: TRAIN-FITTED TRANSFORMS
# ==============================================================================

# --- Variance filter ---
# Fit threshold dari train; apply mask yang sama ke test.
# Tidak ada detection rate filter — cukup variance saja.
filter_by_variance <- function(train_mat, test_mat, min_percentile = 0.10) {
    # train_mat, test_mat: genes × samples
    cat("Variance filter (bottom", min_percentile * 100, "% removed, threshold from TRAIN)\n")
    gene_vars     <- rowVars(train_mat, na.rm = TRUE)
    var_threshold <- quantile(gene_vars, min_percentile)
    keep          <- gene_vars > var_threshold

    cat("  Variance threshold:", sprintf("%.4f", var_threshold), "\n")
    cat("  Before:", nrow(train_mat), "→ After:", sum(keep), "genes\n\n")

    list(train = train_mat[keep, , drop = FALSE],
         test  = test_mat[ keep, , drop = FALSE],
         kept_genes = rownames(train_mat)[keep])
}

# --- Z-score per cohort ---
# Ý nghĩa: loại batch effect đơn giản bằng cách chuẩn hóa từng cohort riêng.
# Fit mean + sd trên TRAIN của mỗi cohort → apply sang TEST cùng cohort.
# Không có leakage vì test không ảnh hưởng đến tham số.
zscore_per_cohort <- function(train_mat, test_mat, train_meta, test_meta) {
    # train_mat, test_mat: samples × genes
    cat("Z-score per cohort (fit on TRAIN, apply to TEST)\n")
    cat("  Note: mean=0, SD=1 per gene per cohort (trong train)\n\n")

    zscore_params <- list()   # lưu để reproduce

    for (co in unique(train_meta$cohort)) {
        tr_idx <- which(train_meta$cohort == co)
        te_idx <- which(test_meta$cohort  == co)

        if (length(tr_idx) == 0) {
            cat("  Cohort", co, ": no train samples — skip\n")
            next
        }

        # Fit trên TRAIN
        g_means <- colMeans(train_mat[tr_idx, , drop = FALSE], na.rm = TRUE)
        g_sds   <- apply(train_mat[tr_idx, , drop = FALSE], 2, sd, na.rm = TRUE)
        g_sds[g_sds == 0] <- 1e-6   # tránh chia 0

        # Apply sang TRAIN
        train_mat[tr_idx, ] <- sweep(
            sweep(train_mat[tr_idx, , drop = FALSE], 2, g_means, "-"),
            2, g_sds, "/")

        # Apply sang TEST dùng CÙNG params từ train
        if (length(te_idx) > 0) {
            test_mat[te_idx, ] <- sweep(
                sweep(test_mat[te_idx, , drop = FALSE], 2, g_means, "-"),
                2, g_sds, "/")
        }

        zscore_params[[co]] <- list(mean = g_means, sd = g_sds)
        cat("  Cohort", co,
            "— Train n =", length(tr_idx),
            "| Test n =", length(te_idx), "✓\n")
    }

    cat("\n")
    list(train = train_mat, test = test_mat, params = zscore_params)
}

# ==============================================================================
# MAIN
# ==============================================================================

main <- function() {

    # --------------------------------------------------------------------------
    cat(strrep("=", 80), "\n")
    cat("PHASE 1: PREPROCESSING KỸ THUẬT\n")
    cat("(Toàn bộ data — không leak vì không dùng label / split info)\n")
    cat(strrep("=", 80), "\n\n")

    # STEP 1 — Load
    cat("STEP 1: LOAD RAW DATA\n")
    cat(strrep("-", 40), "\n")
    v3_raw   <- parse_series_matrix(CONFIG$gse1_matrix)
    v3_annot <- parse_annotation(CONFIG$annot_v3)
    v4_raw   <- parse_series_matrix(CONFIG$gse2_matrix)
    v4_annot <- parse_annotation(CONFIG$annot_v4)

    # STEP 2 — Probe → Gene (mean aggregation per gene)
    cat("STEP 2: PROBE -> GENE MAPPING  (mean aggregation, theo paper ICCE 2024)\n")
    cat(strrep("-", 40), "\n")
    cat("V3:\n"); v3_genes <- aggregate_to_genes(v3_raw$expression, v3_annot)
    cat("V4:\n"); v4_genes <- aggregate_to_genes(v4_raw$expression, v4_annot)

    # STEP 3 — Log2
    cat("STEP 3: LOG2 NORMALIZATION\n")
    cat(strrep("-", 40), "\n")
    v3_norm <- log2_normalize(v3_genes, "V3")
    v4_norm <- log2_normalize(v4_genes, "V4")

    # STEP 4 — Metadata: giữ ALS + Control, bỏ MIM
    cat("STEP 4: METADATA CLEANING  (giữ ALS + Control)\n")
    cat(strrep("-", 40), "\n")
    meta_v3 <- clean_metadata(v3_raw$metadata, "V3")
    keep_v3 <- meta_v3$diagnosis %in% c("ALS", "Control")
    v3_norm <- v3_norm[, keep_v3, drop = FALSE]
    meta_v3 <- meta_v3[keep_v3, ]
    cat("V3 kept:", ncol(v3_norm),
        "(ALS =", sum(meta_v3$diagnosis=="ALS"),
        "| Control =", sum(meta_v3$diagnosis=="Control"), ")\n\n")

    meta_v4 <- clean_metadata(v4_raw$metadata, "V4")
    keep_v4 <- meta_v4$diagnosis %in% c("ALS", "Control")
    v4_norm <- v4_norm[, keep_v4, drop = FALSE]
    meta_v4 <- meta_v4[keep_v4, ]
    cat("V4 kept:", ncol(v4_norm),
        "(ALS =", sum(meta_v4$diagnosis=="ALS"),
        "| Control =", sum(meta_v4$diagnosis=="Control"), ")\n\n")

    # STEP 5 — Merge: intersect genes CÓ MẶT TRONG CẢ 2 DATASET
    # (thực hiện SAU probe->gene mapping, đúng theo paper ICCE 2024)
    cat("STEP 5: MERGE COHORTS  (intersect genes co mat trong ca 2 dataset)\n")
    cat(strrep("-", 40), "\n")
    common_genes <- intersect(rownames(v3_norm), rownames(v4_norm))
    merged_gxs   <- cbind(v3_norm[common_genes, ], v4_norm[common_genes, ])
    merged_meta  <- rbind(meta_v3, meta_v4)
    merged_meta$label <- ifelse(merged_meta$diagnosis == "ALS", 1L, 0L)

    cat("Common genes:", length(common_genes), "\n")
    cat("Merged: Samples =", ncol(merged_gxs),
        "| ALS =", sum(merged_meta$label == 1),
        "| Control =", sum(merged_meta$label == 0), "\n")
    cat("Cohorts: V3 =", sum(merged_meta$cohort == "V3"),
        "| V4 =", sum(merged_meta$cohort == "V4"), "\n\n")

    # --------------------------------------------------------------------------
    cat(strrep("=", 80), "\n")
    cat("PHASE 2: SPLIT 70/30\n")
    cat(strrep("=", 80), "\n\n")

    # STEP 6 — Stratified split
    cat("STEP 6: STRATIFIED SPLIT  (Diagnosis × Cohort)\n")
    cat(strrep("-", 40), "\n")
    split     <- stratified_split(merged_meta, CONFIG$train_ratio, CONFIG$seed)
    train_idx <- split$train
    test_idx  <- split$test

    # genes × samples
    train_gxs  <- merged_gxs[, train_idx, drop = FALSE]
    test_gxs   <- merged_gxs[, test_idx,  drop = FALSE]
    train_meta <- merged_meta[train_idx, ]
    test_meta  <- merged_meta[test_idx,  ]

    check_split_distribution(train_meta, test_meta)

    # --------------------------------------------------------------------------
    cat(strrep("=", 80), "\n")
    cat("PHASE 3: TRAIN-FITTED TRANSFORMS\n")
    cat("(Tất cả params đều fit từ TRAIN → apply sang TEST — không có data leakage)\n")
    cat(strrep("=", 80), "\n\n")

    # STEP 7 — Variance filter (threshold từ train)
    cat("STEP 7: VARIANCE FILTER\n")
    cat(strrep("-", 40), "\n")
    vf        <- filter_by_variance(train_gxs, test_gxs, CONFIG$min_variance_percentile)
    train_gxs <- vf$train
    test_gxs  <- vf$test
    stopifnot(identical(rownames(train_gxs), rownames(test_gxs)))
    cat("  Gene alignment: ✓\n\n")

    # STEP 8 — Z-score per cohort (fit trên train)
    cat("STEP 8: Z-SCORE PER COHORT\n")
    cat(strrep("-", 40), "\n")
    # Chuyển sang samples × genes cho z-score function
    train_sxg <- t(train_gxs)
    test_sxg  <- t(test_gxs)

    zr        <- zscore_per_cohort(train_sxg, test_sxg, train_meta, test_meta)
    train_expr <- zr$train   # samples × genes
    test_expr  <- zr$test

    # Verify: mean ~ 0, sd ~ 1 per cohort trên train
    cat("  Verification (train):\n")
    for (co in unique(train_meta$cohort)) {
        tr_idx <- which(train_meta$cohort == co)
        tr_sub <- train_expr[tr_idx, ]
        cat(sprintf("    %s: mean=%.4f  sd=%.4f  (kỳ vọng: ~0, ~1)\n",
                    co, mean(tr_sub), sd(tr_sub)))
    }
    cat("\n")

    # --------------------------------------------------------------------------
    cat(strrep("=", 80), "\n")
    cat("PHASE 4: SAVE OUTPUTS\n")
    cat(strrep("=", 80), "\n\n")

    # Thêm binary batch dummy vào metadata (để ML dùng nếu cần)
    train_meta$batch_V3 <- as.integer(train_meta$cohort == "V3")
    train_meta$batch_V4 <- as.integer(train_meta$cohort == "V4")
    test_meta$batch_V3  <- as.integer(test_meta$cohort  == "V3")
    test_meta$batch_V4  <- as.integer(test_meta$cohort  == "V4")

    save_expr <- function(mat, path) {
        df <- cbind(sample_id = rownames(mat), as.data.frame(mat))
        write.csv(df, path, row.names = FALSE, quote = FALSE)
    }

    save_expr(train_expr, file.path(CONFIG$output_dir, "train_expression.csv"))
    save_expr(test_expr,  file.path(CONFIG$output_dir, "test_expression.csv"))
    write.csv(train_meta, file.path(CONFIG$output_dir, "train_metadata.csv"),
              row.names = FALSE, quote = FALSE)
    write.csv(test_meta,  file.path(CONFIG$output_dir, "test_metadata.csv"),
              row.names = FALSE, quote = FALSE)

    # Lưu zscore params để reproduce (apply lên external data nếu cần)
    zscore_df_list <- lapply(names(zr$params), function(co) {
        p <- zr$params[[co]]
        data.frame(cohort = co,
                   gene   = names(p$mean),
                   mean   = p$mean,
                   sd     = p$sd,
                   row.names = NULL)
    })
    zscore_df <- do.call(rbind, zscore_df_list)
    write.csv(zscore_df, file.path(CONFIG$output_dir, "zscore_params.csv"),
              row.names = FALSE, quote = FALSE)

    cat("Files saved to:", CONFIG$output_dir, "\n")
    cat("  train_expression.csv  — Samples:", nrow(train_expr),
        "| Genes:", ncol(train_expr), "\n")
    cat("  test_expression.csv   — Samples:", nrow(test_expr),
        "| Genes:", ncol(test_expr), "\n")
    cat("  train_metadata.csv\n")
    cat("  test_metadata.csv\n")
    cat("  zscore_params.csv     — Z-score mean/sd per cohort per gene\n\n")

    # --------------------------------------------------------------------------
    cat(strrep("=", 80), "\n")
    cat("SUMMARY\n")
    cat(strrep("=", 80), "\n\n")
    cat("Input genes  : V3 =", nrow(v3_genes), "| V4 =", nrow(v4_genes), "\n")
    cat("Common genes : ", length(common_genes), "\n")
    cat("After QC     : ", ncol(train_expr), "genes\n")
    cat("Total samples: ", nrow(merged_meta),
        "(ALS =", sum(merged_meta$label==1),
        "| Control =", sum(merged_meta$label==0), ")\n")
    cat("Train / Test : ", nrow(train_expr), "/", nrow(test_expr), "\n")
    cat("Split strata : Diagnosis × Cohort\n")
    cat("Variance filt: bottom", CONFIG$min_variance_percentile*100, "% removed\n")
    cat("Z-score      : per cohort, fit on train ✓\n")
    cat("Leakage check: ✓ (tất cả transforms fit trên train only)\n\n")

    cat(strrep("=", 80), "\n")
    cat("✅ 1C PREPROCESSING COMPLETE  (v1.0 — SIMPLE PIPELINE)\n")
    cat(strrep("=", 80), "\n\n")
    cat("Completed:", format(Sys.time()), "\n")
}

# ==============================================================================
# RUN
# ==============================================================================

tryCatch({
    main()
}, error = function(e) {
    cat("\nERROR:\n", conditionMessage(e), "\n")
    traceback()
}, finally = {
    sink()
})