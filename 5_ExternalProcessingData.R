################################################################################
# 8D_Pro.R  (v5.0 -- external validation prep, no filtering)
#
# THAY DOI v4 -> v5:
#   v4: direction consistency check + logFC/Wilcoxon filter -> SAI logic
#       (mix train-space direction voi external-space, over-process external)
#   v5: load -> normalize -> extract SFFS genes -> export THANG
#       External validation = test generalization cua model, KHONG filter gene
#
# INPUT:
#   data/features/optimal_genes_XGB.txt      <- SFFS output (3C_sffs.py)
#   data/features/optimal_genes_SVM.txt      <- fallback
#   data/features/optimal_genes_RF.txt       <- fallback
#   data/raw/gene_raw_counts.txt             <- Salmon counts (GSE234297)
#   data/raw/gene_offset_matrix.txt          <- Salmon offset (GSE234297)
#   data/raw/blood_RNAseq_supplementary_tables.xlsx  <- metadata S1
#   data/raw/gene_annotation.txt             <- EntrezID -> Symbol
#
# OUTPUT -> external_outputs/
#   ml_ready_external.csv     <- input cho 8_EX.py (ALL SFFS genes present)
#   external_gene_info.csv    <- per-gene info (reference only, no filter)
################################################################################

cat("\n", strrep("=", 70), "\n")
cat("8D_Pro.R -- PREPARE EXTERNAL VALIDATION DATA  (v5.0)\n")
cat(strrep("=", 70), "\n\n")

# ==============================================================================
# 0. PACKAGES
# ==============================================================================

bioc_pkgs <- c("edgeR", "limma")
cran_pkgs <- c("data.table", "readxl", "dplyr")

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "http://cran.rstudio.com/")

for (pkg in bioc_pkgs) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        BiocManager::install(pkg, update = FALSE, ask = FALSE)
        library(pkg, character.only = TRUE)
    }
}
for (pkg in cran_pkgs) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        install.packages(pkg, repos = "http://cran.rstudio.com/")
        library(pkg, character.only = TRUE)
    }
}

set.seed(42)

# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG <- list(
    data_dir           = "data/raw/",
    raw_counts_file    = "gene_raw_counts.txt",
    offset_matrix_file = "gene_offset_matrix.txt",
    metadata_file      = "blood_RNAseq_supplementary_tables.xlsx",
    annotation_file    = "gene_annotation.txt",

    # SFFS gene sets (output cua 3C_sffs.py) -- uu tien XGB -> SVM -> RF
    gene_list_xgb = "data/features/optimal_genes_XGB.txt",
    gene_list_svm = "data/features/optimal_genes_SVM.txt",
    gene_list_rf  = "data/features/optimal_genes_RF.txt",

    output_dir = "external_outputs"
)

dir.create(CONFIG$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# STEP 1: LOAD SFFS GENE SET
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 1: Load SFFS gene set\n")
cat(strrep("-", 70), "\n\n")

if (file.exists(CONFIG$gene_list_xgb)) {
    gene_file  <- CONFIG$gene_list_xgb
    model_name <- "XGB"
} else if (file.exists(CONFIG$gene_list_svm)) {
    gene_file  <- CONFIG$gene_list_svm
    model_name <- "SVM"
} else if (file.exists(CONFIG$gene_list_rf)) {
    gene_file  <- CONFIG$gene_list_rf
    model_name <- "RF"
} else {
    stop(paste0(
        "Khong tim thay SFFS gene list.\n",
        "Kiem tra: ", CONFIG$gene_list_xgb, "\n",
        "Hay chay 3C_sffs.py truoc."
    ))
}

sffs_genes <- readLines(gene_file)
sffs_genes <- sffs_genes[nchar(trimws(sffs_genes)) > 0]
cat(sprintf("  Model     : %s\n", model_name))
cat(sprintf("  Gene file : %s\n", gene_file))
cat(sprintf("  SFFS genes: %d\n", length(sffs_genes)))
cat(sprintf("  %s\n\n", paste(sffs_genes, collapse = ", ")))

# ==============================================================================
# STEP 2: LOAD RAW COUNTS
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 2: Load raw counts\n")
cat(strrep("-", 70), "\n\n")

raw_path <- file.path(CONFIG$data_dir, CONFIG$raw_counts_file)
if (!file.exists(raw_path)) stop("Khong tim thay: ", raw_path)

counts <- fread(raw_path, data.table = FALSE)
rownames(counts) <- as.character(counts$EntrezGeneID)
counts <- counts[, -1, drop = FALSE]
cat(sprintf("  counts: %d genes x %d samples\n\n", nrow(counts), ncol(counts)))

# ==============================================================================
# STEP 3: LOAD OFFSET MATRIX
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 3: Load Salmon offset matrix\n")
cat(strrep("-", 70), "\n\n")

off_path <- file.path(CONFIG$data_dir, CONFIG$offset_matrix_file)
if (!file.exists(off_path)) stop("Khong tim thay: ", off_path)

salmon_offset <- fread(off_path, data.table = FALSE)
rownames(salmon_offset) <- as.character(salmon_offset$EntrezGeneID)
salmon_offset <- salmon_offset[, -1, drop = FALSE]

stopifnot(
    all(rownames(salmon_offset) == rownames(counts)),
    all(colnames(salmon_offset) == colnames(counts))
)
cat(sprintf("  offset: %d genes x %d samples (alignment OK)\n\n",
    nrow(salmon_offset), ncol(salmon_offset)))

# ==============================================================================
# STEP 4: LOAD METADATA (Sheet S1)
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 4: Load metadata (Sheet S1)\n")
cat(strrep("-", 70), "\n\n")

meta_path <- file.path(CONFIG$data_dir, CONFIG$metadata_file)
if (!file.exists(meta_path)) stop("Khong tim thay: ", meta_path)

# Excel structure: row 1-4 = text, row 5 = empty, row 6 = header, row 7+ = data
# read_excel uses row 1 as colnames -> R rows 1-4 = Excel rows 2-5,
# R row 5 = Excel row 6 (actual header), R rows 6+ = data
samples <- as.data.frame(read_excel(meta_path, sheet = "S1"))
names(samples) <- samples[5, ]
samples <- samples[-c(1:5), , drop = FALSE]
rownames(samples) <- samples$Collection_ID

samples$Disease_status <- factor(samples$Disease_status, levels = c("Control", "sALS"))
samples$y_binary       <- ifelse(samples$Disease_status == "sALS", 1L, 0L)

numeric_cols <- c("Age_at_collection", "Age_of_onset", "Disease_duration_months",
                  "Collection_point", "RIN", "RNA_concentration",
                  "Q20", "Q30", "GC_content", "Total_reads", "Mapped_reads_Salmon")
factor_cols  <- c("Sex", "Flowcell_ID")
for (col in intersect(numeric_cols, names(samples))) samples[[col]] <- as.numeric(samples[[col]])
for (col in intersect(factor_cols,  names(samples))) samples[[col]] <- as.factor(samples[[col]])
samples$GC_content <- scale(samples$GC_content, center = TRUE, scale = TRUE)[, 1]

stopifnot(all(rownames(samples) == colnames(counts)))
cat(sprintf("  %d samples  (sALS=%d, Control=%d)\n\n",
    nrow(samples),
    sum(samples$Disease_status == "sALS"),
    sum(samples$Disease_status == "Control")))

# ==============================================================================
# STEP 5: GENE ANNOTATION (gene_annotation.txt)
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 5: Gene annotation (gene_annotation.txt)\n")
cat(strrep("-", 70), "\n\n")

anno_path <- file.path(CONFIG$data_dir, CONFIG$annotation_file)
if (!file.exists(anno_path)) stop("Khong tim thay: ", anno_path)

genes_anno <- fread(anno_path, data.table = FALSE)

# Detect column names flexibly
id_col  <- intersect(c("Gene_ID", "EntrezGeneID", "gene_id"), names(genes_anno))[1]
sym_col <- intersect(c("Gene_Symbol", "Symbol", "gene_symbol"), names(genes_anno))[1]
if (is.na(id_col))  stop("gene_annotation.txt: khong tim thay cot Gene_ID / EntrezGeneID")
if (is.na(sym_col)) stop("gene_annotation.txt: khong tim thay cot Gene_Symbol / Symbol")

genes_anno[[id_col]] <- as.character(genes_anno[[id_col]])
anno_map <- setNames(genes_anno[[sym_col]], genes_anno[[id_col]])

entrez_ids   <- rownames(counts)
gene_symbols <- anno_map[entrez_ids]
genes <- data.frame(
    Gene_ID     = entrez_ids,
    Gene_Symbol = as.character(gene_symbols),
    row.names   = NULL, stringsAsFactors = FALSE
)
na_sym <- is.na(genes$Gene_Symbol)
genes$Gene_Symbol[na_sym] <- genes$Gene_ID[na_sym]  # fallback: keep EntrezID

stopifnot(all(rownames(counts) == genes$Gene_ID))
cat(sprintf("  ID col: '%s'  |  Symbol col: '%s'\n", id_col, sym_col))
cat(sprintf("  %d genes annotated (mapped: %d, fallback to EntrezID: %d)\n\n",
    nrow(genes), sum(!na_sym), sum(na_sym)))

# ==============================================================================
# STEP 6: DGEList + scaleOffset + filterByExpr
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 6: DGEList + scaleOffset + filterByExpr\n")
cat(strrep("-", 70), "\n\n")

dge  <- DGEList(counts  = counts,
                samples = samples,
                genes   = genes,
                group   = samples$Disease_status)
dge  <- scaleOffset(dge, offset = as.matrix(salmon_offset))
keep <- filterByExpr(dge, group = dge$samples$group)
cat(sprintf("  Before filter: %d genes\n", nrow(dge)))
cat(sprintf("  After  filter: %d genes\n\n", sum(keep)))
dge  <- dge[keep, , keep.lib.sizes = FALSE]

# ==============================================================================
# STEP 7: logCPM (with Salmon offset)
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 7: logCPM\n")
cat(strrep("-", 70), "\n\n")

logCPM_raw <- cpm(dge, offset = dge$offset, log = TRUE)
cat(sprintf("  logCPM range: [%.3f, %.3f]\n\n", min(logCPM_raw), max(logCPM_raw)))

# ==============================================================================
# STEP 8: GC content correction
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 8: GC content correction\n")
cat(strrep("-", 70), "\n\n")

if ("GC_content" %in% names(dge$samples) && !all(is.na(dge$samples$GC_content))) {
    logCPM_gc <- removeBatchEffect(
        logCPM_raw,
        covariates = dge$samples$GC_content,
        design     = model.matrix(~ Disease_status, data = dge$samples)
    )
    cat(sprintf("  GC corrected. Range: [%.3f, %.3f]\n\n",
        min(logCPM_gc), max(logCPM_gc)))
} else {
    logCPM_gc <- logCPM_raw
    cat("  GC_content khong co -- skip\n\n")
}

# Lookup map: Gene_Symbol -> EntrezID (rowname trong logCPM_gc)
symbol_to_entrez <- setNames(rownames(dge$genes), dge$genes$Gene_Symbol)

# ==============================================================================
# STEP 9: EXTRACT SFFS GENE PANEL
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 9: Extract SFFS gene panel from RNA-seq\n")
cat(strrep("-", 70), "\n\n")

genes_present <- sffs_genes[sffs_genes %in% names(symbol_to_entrez)]
genes_missing <- sffs_genes[!sffs_genes %in% names(symbol_to_entrez)]

cat(sprintf("  SFFS genes          : %d\n", length(sffs_genes)))
cat(sprintf("  Present in RNA-seq  : %d\n", length(genes_present)))

if (length(genes_missing) > 0) {
    cat(sprintf("  Missing (filtered by filterByExpr -- low count): %d\n", length(genes_missing)))
    cat("  Raw count info (for reference):\n")
    for (g in genes_missing) {
        eid <- genes$Gene_ID[genes$Gene_Symbol == g]
        if (length(eid) > 0 && eid %in% rownames(counts)) {
            rv <- as.numeric(counts[eid, ])
            cat(sprintf("    %-12s  EntrezID=%-8s  median=%.1f  max=%.1f  n_nonzero=%d/%d\n",
                g, eid, median(rv), max(rv), sum(rv > 0), length(rv)))
        } else {
            cat(sprintf("    %-12s  not found in counts\n", g))
        }
    }
}
cat("\n")

# Extract expression matrix: samples x genes
entrez_present <- symbol_to_entrez[genes_present]
expr_panel     <- t(logCPM_gc[entrez_present, , drop = FALSE])
colnames(expr_panel) <- genes_present

cat(sprintf("  expr_panel: %d samples x %d genes\n\n",
    nrow(expr_panel), ncol(expr_panel)))

# ==============================================================================
# STEP 10: EXPORT
# ==============================================================================

cat(strrep("-", 70), "\n")
cat("STEP 10: Export\n")
cat(strrep("-", 70), "\n\n")

# --- ml_ready_external.csv ---
ml_ext                  <- as.data.frame(expr_panel)
ml_ext$sample_id        <- rownames(expr_panel)
ml_ext$diagnosis        <- as.character(dge$samples[rownames(expr_panel), "Disease_status"])
ml_ext$diagnosis_binary <- ifelse(ml_ext$diagnosis == "sALS", 1L, 0L)
ml_ext$age              <- dge$samples[rownames(expr_panel), "Age_at_collection"]
ml_ext$sex              <- as.character(dge$samples[rownames(expr_panel), "Sex"])
ml_ext$model_used       <- model_name

meta_cols <- c("sample_id", "diagnosis", "diagnosis_binary", "age", "sex", "model_used")
ml_ext    <- ml_ext[, c(meta_cols, genes_present)]

out_ml <- file.path(CONFIG$output_dir, "ml_ready_external.csv")
write.csv(ml_ext, out_ml, row.names = FALSE)
cat(sprintf("  ml_ready_external.csv  -> %d samples x %d genes\n",
    nrow(ml_ext), length(genes_present)))

# --- external_gene_info.csv (reference only, KHONG dung de filter) ---
als_idx  <- dge$samples$Disease_status == "sALS"
ctrl_idx <- dge$samples$Disease_status == "Control"

gene_info <- data.frame(
    gene              = sffs_genes,
    model_name        = model_name,
    present_in_rnaseq = sffs_genes %in% genes_present,
    reason_if_missing = ifelse(
        sffs_genes %in% genes_present, "",
        ifelse(
            sffs_genes %in% genes$Gene_Symbol,
            "filtered_by_filterByExpr (low count in GSE234297)",
            "not_found_in_annotation"
        )
    ),
    mean_logCPM_ALS     = NA_real_,
    mean_logCPM_Control = NA_real_,
    logFC_external      = NA_real_,
    stringsAsFactors = FALSE
)

for (g in genes_present) {
    gene_info[gene_info$gene == g, "mean_logCPM_ALS"]     <- round(mean(expr_panel[als_idx,  g], na.rm=TRUE), 4)
    gene_info[gene_info$gene == g, "mean_logCPM_Control"] <- round(mean(expr_panel[ctrl_idx, g], na.rm=TRUE), 4)
    gene_info[gene_info$gene == g, "logFC_external"]      <- round(
        mean(expr_panel[als_idx,  g], na.rm=TRUE) -
        mean(expr_panel[ctrl_idx, g], na.rm=TRUE), 4)
}

out_gene <- file.path(CONFIG$output_dir, "external_gene_info.csv")
write.csv(gene_info, out_gene, row.names = FALSE)
cat(sprintf("  external_gene_info.csv -> %d genes (reference only)\n\n", nrow(gene_info)))

# ==============================================================================
# SUMMARY
# ==============================================================================

cat(strrep("=", 70), "\n")
cat("DONE -- Ready for 8_EX.py\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("  Dataset           : GSE234297 (Grima 2023, ALS blood RNA-seq)\n"))
cat(sprintf("  Samples           : %d  (sALS=%d, Control=%d)\n",
    nrow(ml_ext),
    sum(ml_ext$diagnosis == "sALS"),
    sum(ml_ext$diagnosis == "Control")))
cat(sprintf("  SFFS model        : %s\n", model_name))
cat(sprintf("  SFFS genes        : %d\n", length(sffs_genes)))
cat(sprintf("  Exported to CSV   : %d genes\n", length(genes_present)))
if (length(genes_missing) > 0)
    cat(sprintf("  Not in RNA-seq    : %d  -> %s\n  (low count, filtered by filterByExpr -- not a code error)\n",
        length(genes_missing), paste(genes_missing, collapse = ", ")))
cat(sprintf("\n  Output  : %s/ml_ready_external.csv\n", CONFIG$output_dir))
cat(sprintf("  Gene log: %s/external_gene_info.csv\n\n", CONFIG$output_dir))
cat("Run next:\n")
cat("  python 8_EX.py\n\n")