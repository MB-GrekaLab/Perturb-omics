# =====================================================================
# S3 Subclustering, BRD Query Label Transfer & Trajectory Analysis
# =====================================================================
# LIBRARIES & CONFIGURATION
# =====================================================================
suppressPackageStartupMessages({
  library(Seurat)
  library(reticulate)
  library(monocle3)
  library(SingleCellExperiment)
  library(destiny)
  library(PseudotimeDE)
  library(edgeR)
  library(IHW)
  library(dplyr)
  library(tibble)
  library(Matrix)
  library(FNN)
  library(igraph)
})

# --- Portable Input / Output Paths ---
output_dir         <- "results"
input_seurat_path  <- file.path(output_dir, "kidney_seurat_annotated.rds")
subcluster_out_dir <- file.path(output_dir, "BRD_query_results")

# =====================================================================
# LOAD ANNOTATED SEURAT OBJECT & SUBSET 
# =====================================================================
if (!file.exists(input_seurat_path)) {
  stop("Annotated Seurat object not found at '", input_seurat_path)
}
kidney_seurat <- readRDS(input_seurat_path)

# Reference: FS_Vehicle, cluster S3
FS_subset <- subset(
  x      = kidney_seurat,
  subset = Cell_Type %in% c("S3") & Treatment %in% c("FS_Vehicle")
)
DefaultAssay(FS_subset) <- "RNA"
S3_ref_base <- NormalizeData(FS_subset, verbose = FALSE)

# Query: FS_BRD, cluster S3 (normalized once, independent of clustering params)
BRD_query <- subset(
  x      = kidney_seurat,
  subset = Cell_Type %in% c("S3") & Treatment %in% c("FS_BRD")
)
DefaultAssay(BRD_query) <- "RNA"
BRD_query <- NormalizeData(BRD_query, verbose = FALSE)

message("Reference (FS_Vehicle S3) cells: ", ncol(S3_ref_base))
message("Query     (FS_BRD S3) cells:     ", ncol(BRD_query))

# =====================================================================
# SUBCLUSTER
# =====================================================================
set.seed(1234)

S3_subset <- S3_ref_base %>%
  FindVariableFeatures(
    selection.method = "vst",
    nfeatures        = 3000,
    verbose          = FALSE
  ) %>%
  ScaleData(
    vars.to.regress = c("nCount_RNA", "nFeature_RNA"),
    scale.max       = 10,
    verbose         = FALSE
  ) %>%
  RunPCA(verbose = FALSE, weight.by.var = TRUE) %>%
  FindNeighbors(
    dims    = 1:10,
    k.param = 25,
    verbose = FALSE
  ) %>%
  FindClusters(
    resolution = 0.25,
    verbose    = FALSE
  ) %>%
  RunUMAP(
    dims         = 1:10,
    return.model = TRUE,
    min.dist     = 0.001,
    umap.method  = "umap-learn",
    verbose      = FALSE
  )

message("Reference S3 subcluster sizes:")
print(table(S3_subset$seurat_clusters))

anchors <- FindTransferAnchors(
  reference           = S3_subset,
  query               = BRD_query,
  dims                = 1:10,
  k.filter            = 200,
  reference.reduction = "pca"
)

BRD_query <- MapQuery(
  anchorset           = anchors,
  reference           = S3_subset,
  query               = BRD_query,
  refdata             = list(seurat_clusters = "seurat_clusters"),
  reference.reduction = "pca",
  reduction.model     = "umap"
)

BRD_query$confident_mapping <- BRD_query$predicted.seurat_clusters.score >= 0.5
message("BRD query mapped cluster counts (confident >= 0.5):")
print(table(
  Predicted_Cluster = BRD_query$predicted.seurat_clusters,
  Confident         = BRD_query$confident_mapping
))

# =====================================================================
# MONOCLE 3 TRAJECTORY INFERENCE 
# =====================================================================


DefaultAssay(S3_subset) <- "RNA"
counts_matrix   <- LayerData(S3_subset, assay = "RNA", layer = "counts")
cell_metadata   <- S3_subset@meta.data
gene_annotation <- data.frame(
  gene_short_name = rownames(counts_matrix),
  row.names       = rownames(counts_matrix)
)

cds3 <- new_cell_data_set(
  expression_data = counts_matrix,
  cell_metadata   = cell_metadata,
  gene_metadata   = gene_annotation
)


cds3 <- preprocess_cds(cds3)
reducedDims(cds3)[["UMAP"]] <- Embeddings(S3_subset, reduction = "umap")

# Cluster and learn principal graph on Seurat UMAP coordinates
cds3 <- cluster_cells(cds3, k = 10)
cds3 <- learn_graph(cds3)


get_earliest_principal_node <- function(cds, starting_cluster) {
  cell_ids <- which(colData(cds)[, "seurat_clusters"] == starting_cluster)
  if (length(cell_ids) == 0) {
    stop("Starting cluster '", starting_cluster, "' not found in colData(cds)$seurat_clusters.")
  }

  closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  closest_vertex <- as.matrix(closest_vertex[colnames(cds), ])
  root_idx       <- as.numeric(names(which.max(table(closest_vertex[cell_ids, ]))))
  root_pr_nodes  <- igraph::V(principal_graph(cds)[["UMAP"]])$name[root_idx]

  return(root_pr_nodes)
}

root_node <- get_earliest_principal_node(
  cds              = cds3,
  starting_cluster = "2"
)
cds3 <- order_cells(cds3, root_pr_nodes = root_node)

# Store pseudotime back into the Seurat reference object
S3_subset$pseudotime <- pseudotime(cds3)[colnames(S3_subset)]

# =====================================================================
# SPATIAL AUTOCORRELATION 
# =====================================================================

calculate_morans_i <- function(cds,
                               features,
                               k        = 30,
                               expr_mat = NULL,
                               sf       = NULL) {


  umap_coords <- SingleCellExperiment::reducedDim(cds, "UMAP")
  n_cells     <- nrow(umap_coords)
  knn_res     <- FNN::get.knn(umap_coords, k = k)

  W <- Matrix::sparseMatrix(
    i    = rep(seq_len(n_cells), each = k),
    j    = as.vector(t(knn_res$nn.index)),
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  W <- ((W + Matrix::t(W)) > 0) * 1.0
  W <- Matrix::Diagonal(x = 1 / Matrix::rowSums(W)) %*% W

  # --- 2. Graph constants (Cliff & Ord 1981) ---
  S0 <- sum(W)
  S1 <- 0.5 * sum((W + Matrix::t(W))^2)
  S2 <- sum((Matrix::rowSums(W) + Matrix::colSums(W))^2)

  E_I   <- -1 / (n_cells - 1)
  var_I <- (n_cells^2 * S1 - n_cells * S2 + 3 * S0^2) /
           (S0^2 * (n_cells^2 - 1)) - E_I^2
  stopifnot(var_I > 0)
  sd_I <- sqrt(var_I)

  # --- 3. Align requested features with CDS ---
  valid_features <- intersect(features, rownames(cds))
  missing        <- setdiff(features, valid_features)
  if (length(valid_features) == 0) {
    stop("No requested features found in cds.")
  }
  if (length(missing) > 0) {
    warning(
      length(missing), " feature(s) absent from cds, e.g.: ",
      paste(utils::head(missing, 5), collapse = ", ")
    )
  }

  # --- 4. Expression matrix ---
  if (!is.null(expr_mat)) {
    stopifnot(identical(colnames(expr_mat), colnames(cds)))
    expr <- expr_mat[valid_features, , drop = FALSE]
  } else {
    if (is.null(sf)) {
      sf <- Matrix::colSums(SingleCellExperiment::counts(cds)) / 1e4
    }
    stopifnot(length(sf) == ncol(cds))
    expr <- log1p(
      SingleCellExperiment::counts(cds)[valid_features, , drop = FALSE] %*%
        Matrix::Diagonal(x = 1 / sf)
    )
  }
  stopifnot(identical(rownames(expr), valid_features))

  # --- 5. Per-gene Moran's I ---
  out <- data.frame(
    Gene     = valid_features,
    Morans_I = NA_real_,
    Z_score  = NA_real_,
    p_value  = NA_real_,
    stringsAsFactors = FALSE
  )

  for (m in seq_along(valid_features)) {
    x     <- as.numeric(expr[m, ])
    z     <- x - mean(x)
    denom <- sum(z^2)
    if (denom > 0) {
      obs_I           <- (n_cells / S0) * (as.numeric(z %*% (W %*% z)) / denom)
      out$Morans_I[m] <- obs_I
      out$Z_score[m]  <- (obs_I - E_I) / sd_I
      out$p_value[m]  <- pnorm(out$Z_score[m], lower.tail = FALSE)
    }
  }

  out$FDR <- p.adjust(out$p_value, method = "fdr")
  out[order(out$p_value), ]
}

DefaultAssay(S3_subset) <- "RNA"
norm_mat <- LayerData(S3_subset, assay = "RNA", layer = "data")[, colnames(cds3), drop = FALSE]
stopifnot(identical(colnames(norm_mat), colnames(cds3)))

sf_seurat  <- Matrix::colSums(LayerData(S3_subset, assay = "RNA", layer = "counts")[, colnames(cds3)]) / 1e4
sf_monocle <- monocle3::size_factors(cds3)
message(sprintf("cor(Seurat sf, monocle sf) = %.4f", cor(sf_seurat, sf_monocle)))


subset_sparse_counts <- LayerData(S3_subset, assay = "RNA", layer = "counts")
min_cells_expr       <- ceiling(0.03 * ncol(S3_subset))
keep_genes           <- rownames(subset_sparse_counts)[Matrix::rowSums(subset_sparse_counts > 0) >= min_cells_expr]

S3_subset_filt <- subset(S3_subset, features = keep_genes)
S3_subset_filt <- FindVariableFeatures(
  S3_subset_filt,
  selection.method = "vst",
  nfeatures        =5000,
  verbose          = FALSE
)
moran_features <- VariableFeatures(S3_subset_filt)

moran_custom <- calculate_morans_i(
  cds      = cds3,
  features = moran_features,
  k        = 30,
  expr_mat = norm_mat
)

# =====================================================================
# DIFFUSION PSEUDOTIMEDE
# =====================================================================

pca_embeddings <- Embeddings(S3_subset, reduction = "pca")[, 1:10, drop = FALSE]
metrics        <- c("rankcor")

dm_list <- lapply(metrics, function(dist_metric) {
  DiffusionMap(pca_embeddings, distance = dist_metric)
})
names(dm_list) <- metrics

dpt_rankcor <- DPT(dm_list$rankcor)

# --- 6B. Project BRD Query onto Reference Diffusion Map ---
query_pca <- Embeddings(BRD_query, reduction = "ref.pca")
ref_pca   <- Embeddings(S3_subset, reduction = "pca")
n_dims    <- min(ncol(ref_pca), ncol(query_pca))

query_pca           <- query_pca[, seq_len(n_dims), drop = FALSE]
colnames(query_pca) <- colnames(ref_pca)[seq_len(n_dims)]

projected_dm <- dm_predict(dm_list$rankcor, query_pca)

S3_subset$pseudotime_rankcor <- dpt_rankcor$DPT2

valid_cells      <- !is.na(S3_subset$pseudotime_rankcor)
valid_cell_names <- colnames(S3_subset)[valid_cells]
pseudotime_rankcor <- S3_subset$pseudotime_rankcor[valid_cells]

message("Keeping ", length(valid_cell_names), " cells for PseudotimeDE.")

ori_tbl <- tibble(
  cell       = valid_cell_names,
  pseudotime = pseudotime_rankcor
)


pca_full <- Embeddings(S3_subset, reduction = "pca")[valid_cell_names, 1:10, drop = FALSE]

message("Generating empirical pseudotime uncertainty using destiny...")
set.seed(1234)

sub_tbl_list <- lapply(seq_len(200), function(i) {
  sub_cells <- sample(valid_cell_names, size = floor(0.8 * length(valid_cell_names)))
  pca_sub   <- pca_full[sub_cells, , drop = FALSE]

  suppressWarnings({
    dm_sub  <- DiffusionMap(pca_sub, distance = "rankcor")
    dpt_sub <- DPT(dm_sub)
    pt_sub  <- dpt_sub$DPT2
  })

  tibble(
    cell       = sub_cells,
    pseudotime = pt_sub
  )
})

ori_tbl_scaled <- ori_tbl %>%
  filter(!is.na(pseudotime)) %>%
  mutate(
    pseudotime = (pseudotime - min(pseudotime)) / (max(pseudotime) - min(pseudotime)),
    pseudotime = pmin(pmax(pseudotime, 0), 1)
  )

sub_tbl_list_scaled <- lapply(sub_tbl_list, function(tbl) {
  tbl_scaled <- tbl %>%
    filter(!is.na(pseudotime)) %>%
    mutate(
      pseudotime = (pseudotime - min(pseudotime)) / (max(pseudotime) - min(pseudotime)),
      pseudotime = pmin(pmax(pseudotime, 0), 1)
    )

  matched_ori <- ori_tbl_scaled$pseudotime[match(tbl_scaled$cell, ori_tbl_scaled$cell)]
  pt_cor      <- cor(tbl_scaled$pseudotime, matched_ori, use = "complete.obs", method = "spearman")

  if (!is.na(pt_cor) && pt_cor < 0) {
    tbl_scaled$pseudotime <- 1 - tbl_scaled$pseudotime
  }

  return(tbl_scaled)
})


DefaultAssay(S3_subset) <- "RNA"
subset_sparse_counts <- LayerData(S3_subset, assay = "RNA", layer = "counts")[, valid_cell_names, drop = FALSE]

keep_genes <- rownames(subset_sparse_counts)[Matrix::rowSums(subset_sparse_counts > 0) >= ceiling(0.03 * ncol(subset_sparse_counts))]

S3_pde_filt <- subset(S3_subset[, valid_cell_names], features = keep_genes)
S3_pde_filt <- FindVariableFeatures(
  S3_pde_filt,
  selection.method = "vst",
  nfeatures        = 5000,
  verbose          = FALSE
)
pde_test_genes <- VariableFeatures(S3_pde_filt)

dense_counts <- as.matrix(subset_sparse_counts[pde_test_genes, valid_cell_names, drop = FALSE])
sce_counts   <- SingleCellExperiment(assays = list(counts = dense_counts))

lib_sizes <- S3_subset$nCount_RNA[valid_cell_names]
sizeFactors(sce_counts) <- lib_sizes / mean(lib_sizes)

message("Running PseudotimeDE across ", length(pde_test_genes), " genes...")
pseudotime_de_res <- runPseudotimeDE(
  gene.vec = pde_test_genes,
  ori.tbl  = ori_tbl_scaled,
  sub.tbl  = sub_tbl_list_scaled,
  mat      = sce_counts,
  model    = "nb",
  usebam   = FALSE,
  mc.cores = 1
)

# =====================================================================
# PSEUDOBULK DIFFERENTIAL EXPRESSION 
# =====================================================================
subcluster_de_params <- list(
  target_clusters = c("0", "1")
)

s3_de_seurat <- subset(
  x      = S3_subset,
  subset = seurat_clusters %in% subcluster_de_params$target_clusters
)
DefaultAssay(s3_de_seurat) <- "RNA"

s3_de_seurat$seurat_clusters <- factor(
  as.character(s3_de_seurat$seurat_clusters),
  levels = subcluster_de_params$target_clusters
)
s3_de_seurat$Lane         <- droplevels(factor(as.character(s3_de_seurat$Lane)))
s3_de_seurat$pseudo_group <- paste0("Cluster", s3_de_seurat$seurat_clusters, "_", s3_de_seurat$Lane)

# Aggregate raw counts per Subcluster x Lane pseudobulk replicate
pb_counts <- AggregateExpression(
  s3_de_seurat,
  assays        = "RNA",
  return.seurat = FALSE,
  group.by      = "pseudo_group",
  slot          = "counts"
)$RNA

pb_counts           <- round(pb_counts)
colnames(pb_counts) <- gsub("-", "_", colnames(pb_counts))

# Build replicate-level metadata aligned to pseudobulk count columns
meta <- s3_de_seurat@meta.data %>%
  group_by(pseudo_group, seurat_clusters, Lane) %>%
  summarise(n_cells = n(), .groups = "drop") %>%
  as.data.frame()

rownames(meta)       <- meta$pseudo_group
meta                 <- meta[colnames(pb_counts), ]
meta$Lane            <- droplevels(factor(meta$Lane))
meta$seurat_clusters <- droplevels(factor(meta$seurat_clusters, levels = subcluster_de_params$target_clusters))

design       <- model.matrix(~ Lane + seurat_clusters, data = meta)
cluster_coef <- ncol(design)

dge <- DGEList(counts = pb_counts, samples = meta)
dge <- dge[filterByExpr(dge, group = dge$samples$seurat_clusters), , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)
dge <- estimateDisp(dge, design, robust = TRUE)

fit                    <- glmFit(dge, design, robust = TRUE)
lrt                    <- glmLRT(fit, coef = cluster_coef)
s3_cluster_de_res      <- topTags(lrt, n = Inf)$table
s3_cluster_de_res$Gene <- rownames(s3_cluster_de_res)

ihw_fit <- ihw(
  pvalues    = s3_cluster_de_res$PValue,
  covariates = s3_cluster_de_res$logCPM
)
s3_cluster_de_res$IHW_FDR <- adj_pvalues(ihw_fit)

