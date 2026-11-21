# =====================================================================
#Nephron Segment Functional Scoring 
# =====================================================================
# 0. LIBRARIES & CONFIGURATION
# =====================================================================
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratExtend)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(msigdbr)
  library(edgeR)
  library(IHW)
})

# --- Portable Input / Output Paths & Segment Definitions ---
output_dir              <- "results"
input_seurat_path       <- file.path(output_dir, "kidney_seurat_annotated.rds")
target_nephron_segments <- c("S1", "S2", "S3", "DCT", "CNT", "TAL", "PC")


# =====================================================================
# LOAD ANNOTATED SEURAT OBJECT
# =====================================================================
if (!file.exists(input_seurat_path)) {
  stop("Annotated Seurat object not found at '", input_seurat_path)
}
kidney_seurat <- readRDS(input_seurat_path)

# =====================================================================
 # NEPHRON SEGMENT FUNCTIONAL SCORING (KEGG PATHWAYS & TFs)
# =====================================================================
vehicle_nephron_seurat <- subset(
  kidney_seurat,
  subset = Treatment %in% c("FS_Vehicle", "WT_Vehicle") &
           Cell_Type %in% target_nephron_segments
)
vehicle_nephron_seurat$Treatment <- factor(vehicle_nephron_seurat$Treatment, levels = c("WT_Vehicle", "FS_Vehicle"))
vehicle_nephron_seurat$Cell_Type <- factor(vehicle_nephron_seurat$Cell_Type, levels = target_nephron_segments)
DefaultAssay(vehicle_nephron_seurat) <- "RNA"

# --- 3A. Mouse KEGG Medicus Pathway Delta Z-Scores ---
kegg_msigdb_df <- msigdbr(species = "Mus musculus", category = "C2") %>%
  filter(gs_subcat == "CP:KEGG_MEDICUS")

kegg_genesets <- split(
  x = str_to_title(kegg_msigdb_df$gene_symbol),
  f = kegg_msigdb_df$gs_name
)

message("Running AUCell Gene Set Analysis for KEGG pathways...")
vehicle_nephron_seurat <- GeneSetAnalysis(vehicle_nephron_seurat, genesets = kegg_genesets)
kegg_aucell_mat        <- vehicle_nephron_seurat@misc$AUCell$genesets

vehicle_nephron_seurat$Segment_Treatment <- paste(
  vehicle_nephron_seurat$Cell_Type,
  vehicle_nephron_seurat$Treatment,
  sep = "__"
)

named_tf_df <- msigdbr(species = "Mus musculus", category = "C3", subcategory = "TFT:TFT_LEGACY") %>%
  filter(!grepl("UNKNOWN", gs_name)) %>%
  filter(!grepl("^[A-Z]{5,}_", gs_name))

named_tf_genesets <- split(
  x = str_to_title(named_tf_df$gene_symbol),
  f = named_tf_df$gs_name
)

message("Running AUCell Gene Set Analysis for TFs...")
vehicle_nephron_seurat <- GeneSetAnalysis(vehicle_nephron_seurat, genesets = named_tf_genesets)
tf_aucell_mat          <- vehicle_nephron_seurat@misc$AUCell$genesets


# =====================================================================
# PSEUDOBULK DIFFERENTIAL EXPRESSION PER SEGMENT
# =====================================================================


#' Run pairwise pseudobulk edgeR LRT  for a target cell type
run_pseudobulk_de <- function(seurat_obj,
                              target_segment,
                              control_group,
                              test_group) {
  sub_seurat <- subset(
    seurat_obj,
    subset = Cell_Type == target_segment &
             Treatment %in% c(control_group, test_group)
  )

  sub_seurat$Treatment    <- factor(as.character(sub_seurat$Treatment), levels = c(control_group, test_group))
  sub_seurat$Lane         <- factor(as.character(sub_seurat$Lane))
  sub_seurat$pseudo_group <- paste(sub_seurat$Treatment, sub_seurat$Lane, sep = "_")

  # Aggregate raw counts per Treatment x Lane pseudobulk replicate
  pb_counts <- AggregateExpression(
    sub_seurat,
    assays        = "RNA",
    return.seurat = FALSE,
    group.by      = "pseudo_group",
    slot          = "counts"
  )$RNA

  pb_counts           <- round(pb_counts)
  colnames(pb_counts) <- gsub("-", "_", colnames(pb_counts))

  # Build replicate-level metadata aligned to pseudobulk count columns
  pb_meta <- sub_seurat@meta.data %>%
    group_by(pseudo_group, Treatment, Lane) %>%
    summarise(mean_nFeature = mean(nFeature_RNA), .groups = "drop") %>%
    as.data.frame()

  rownames(pb_meta) <- pb_meta$pseudo_group
  pb_meta           <- pb_meta[colnames(pb_counts), ]

  # Fit edgeR LRT model controlling for batch and number of detected genes
  design_mat     <- model.matrix(~ Lane + mean_nFeature + Treatment, data = pb_meta)
  treatment_coef <- ncol(design_mat)

  dge <- DGEList(counts = pb_counts, samples = pb_meta)
  dge <- dge[filterByExpr(dge, group = dge$samples$Treatment), , keep.lib.sizes = FALSE]
  dge <- calcNormFactors(dge)
  dge <- estimateDisp(dge, design_mat, robust = TRUE)

  fit        <- glmFit(dge, design_mat, robust = TRUE)
  lrt        <- glmLRT(fit, coef = treatment_coef)
  de_results <- topTags(lrt, n = Inf)$table

  # Apply Independent Hypothesis Weighting (IHW) using logCPM as covariate
  ihw_fit <- ihw(
    pvalues    = de_results$PValue,
    covariates = de_results$logCPM
  )
  de_results$IHW_FDR <- adj_pvalues(ihw_fit)

  return(de_results)
}

target_nephron_segments <- c("S1", "S2", "S3", "DCT", "CNT", "TAL", "PC")
target_segment <- "S3"

s3_de_fs_vehicle <- run_pseudobulk_de(
  seurat_obj     = kidney_seurat,
  target_segment = target_segment,
  control_group  = "WT_Vehicle",
  test_group     = "FS_Vehicle"
)
