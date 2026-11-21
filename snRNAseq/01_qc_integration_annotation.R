# =====================================================================
#  Data Loading, Quality Control, CCA Integration & Annotation
# =====================================================================
# 0. LIBRARIES & CONFIGURATION
# =====================================================================
suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(Matrix)
})


data_root_dir      <- Sys.getenv("DATA_DIR", unset = file.path("data", "count"))
output_dir         <- "results"
output_seurat_path <- file.path(output_dir, "kidney_seurat_annotated.rds")

# --- Sample Experimental Design ---
sample_metadata <- tribble(
  ~batch_id, ~sample_id, ~Treatment,   ~Lane,
  "0",       "1301_1",   "WT_Vehicle", "5",
  "1",       "1303_2",   "WT_Vehicle", "6",
  "2",       "1304_1",   "WT_BRD",     "5",
  "3",       "1304_2",   "WT_BRD",     "6",
  "4",       "1307_1",   "FS_Vehicle", "5",
  "5",       "1307_2",   "FS_Vehicle", "6",
  "6",       "1309_2",   "FS_BRD",     "5",
  "7",       "1309_4",   "FS_BRD",     "6",
  "8",       "1312_1",   "FS_Vehicle", "7",
  "9",       "1312_2",   "FS_Vehicle", "8",
  "10",      "1312_3",   "FS_BRD",     "7",
  "11",      "1312_4",   "FS_BRD",     "8",
  "12",      "1327_1",   "WT_BRD",     "7",
  "13",      "1327_2",   "WT_BRD",     "8",
  "14",      "1328_1",   "WT_Vehicle", "7",
  "15",      "1328_2",   "WT_Vehicle", "8"
) %>%
  mutate(
    matrix_dir = file.path(data_root_dir, sample_id, "outs", "filtered_feature_bc_matrix")
  )

missing_dirs <- sample_metadata$matrix_dir[!dir.exists(sample_metadata$matrix_dir)]
if (length(missing_dirs) > 0) {
  stop(
    "Missing 10X matrix directories under '", data_root_dir, "'. ",
    "Place CellRanger outputs in 'data/count/' or set DATA_DIR.\nFirst missing path: ",
    missing_dirs[1]
  )
}

treatment_levels <- c("WT_Vehicle", "WT_BRD", "FS_Vehicle", "FS_BRD")
lane_levels      <- c("5", "6", "7", "8")

# --- QC & Integration Parameters ---
qc_params <- list(
  min_genes_per_cell = 200,
  min_cells_per_gene = 5,
  max_genes_per_cell = 4000,
  max_mt_percent     = 25
)

integration_params <- list(
  n_variable_features = 2000,
  n_pcs               = 30,
  cluster_resolution  = 0.5,
  random_seed         = 42
)

# =====================================================================
# 1. HELPER FUNCTIONS
# =====================================================================


load_10x_batch <- function(matrix_dir, batch_id) {
  counts <- Read10X(data.dir = matrix_dir, gene.column = 2, unique.features = FALSE)
  if (is.list(counts)) {
    counts <- counts[["Gene Expression"]]
  }


  feat_candidates <- file.path(
    matrix_dir,
    c("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
  )
  features_file <- feat_candidates[file.exists(feat_candidates)][1]
  features <- read.delim(features_file, header = FALSE, stringsAsFactors = FALSE)
  if (ncol(features) >= 3) {
    features <- features[features[[3]] == "Gene Expression", ]
  }

  rownames(counts) <- make.unique(features[[2]], sep = "-")
  colnames(counts) <- paste0(colnames(counts), "-", batch_id)

  obj <- CreateSeuratObject(
    counts       = counts,
    project      = "SeuratProject",
    min.cells    = 0,
    min.features = 0
  )
  obj$batch <- batch_id
  return(obj)
}

# =====================================================================
# 2. DATA LOADING & SAMPLE METADATA ASSIGNMENT
# =====================================================================
sample_seurat_list <- map2(
  sample_metadata$matrix_dir,
  sample_metadata$batch_id,
  load_10x_batch
)

kidney_seurat <- merge(x = sample_seurat_list[[1]], y = sample_seurat_list[-1])
kidney_seurat[["RNA"]] <- JoinLayers(kidney_seurat[["RNA"]])
rm(sample_seurat_list)

batch_idx <- match(kidney_seurat$batch, sample_metadata$batch_id)
kidney_seurat$Treatment <- factor(sample_metadata$Treatment[batch_idx], levels = treatment_levels)
kidney_seurat$Lane      <- factor(sample_metadata$Lane[batch_idx],      levels = lane_levels)

# =====================================================================
# 3. QUALITY CONTROL FILTERING
# =====================================================================
kidney_seurat <- subset(kidney_seurat, subset = nFeature_RNA >= qc_params$min_genes_per_cell)

raw_counts      <- LayerData(kidney_seurat, assay = "RNA", layer = "counts")
expressed_genes <- Matrix::rowSums(raw_counts > 0) >= qc_params$min_cells_per_gene
kidney_seurat   <- kidney_seurat[expressed_genes, ]

filtered_counts <- LayerData(kidney_seurat, assay = "RNA", layer = "counts")
is_mt_gene      <- grepl("^mt-", rownames(filtered_counts))

kidney_seurat$n_genes_by_counts <- Matrix::colSums(filtered_counts > 0)
kidney_seurat$total_counts      <- Matrix::colSums(filtered_counts)
kidney_seurat$pct_counts_mt     <- (Matrix::colSums(filtered_counts[is_mt_gene, ]) / kidney_seurat$total_counts) * 100

kidney_seurat <- subset(
  kidney_seurat,
  subset = n_genes_by_counts < qc_params$max_genes_per_cell &
           pct_counts_mt     < qc_params$max_mt_percent
)

# =====================================================================
# 4. NORMALIZATION, PCA & CCA BATCH INTEGRATION
# =====================================================================

kidney_seurat[["RNA"]] <- split(kidney_seurat[["RNA"]], f = kidney_seurat$Lane)

kidney_seurat <- NormalizeData(kidney_seurat)
kidney_seurat <- FindVariableFeatures(
  kidney_seurat,
  selection.method = "vst",
  nfeatures        = integration_params$n_variable_features
)
kidney_seurat <- ScaleData(kidney_seurat)
kidney_seurat <- RunPCA(kidney_seurat, npcs = integration_params$n_pcs, verbose = FALSE)

set.seed(integration_params$random_seed)
kidney_seurat <- IntegrateLayers(
  object         = kidney_seurat,
  method         = CCAIntegration,
  orig.reduction = "pca",
  new.reduction  = "integrated.cca",
  verbose        = FALSE
)

kidney_seurat[["RNA"]] <- JoinLayers(kidney_seurat[["RNA"]])

# =====================================================================
# 5. UMAP EMBEDDING, CLUSTERING & CELL-TYPE ANNOTATION
# =====================================================================
kidney_seurat <- RunUMAP(kidney_seurat, reduction = "integrated.cca", dims = 1:integration_params$n_pcs)
kidney_seurat <- FindNeighbors(kidney_seurat, reduction = "integrated.cca", dims = 1:integration_params$n_pcs)
kidney_seurat <- FindClusters(kidney_seurat, resolution = integration_params$cluster_resolution)

DimPlot(kidney_seurat, reduction = "umap", group.by = "orig.ident")


# Save annotated Seurat object for downstream analysis
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(kidney_seurat, file = output_seurat_path)
