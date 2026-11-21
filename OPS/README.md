# OPS: optical pooled screening scripts

Scripts for the image-to-profile workflow.

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `align.py` | raw SBS and phenotype images | aligned images, DAPI QC stack, QC report |
| 2 | `feature_extraction.py` | aligned images | masks, per-object features |
| 3 | `assign_gRNAs.py` | SBS and phenotype feature tables, gRNA training cells, barcode whitelist | annotated phenotype tables (gRNA, gene) |
| 4 | `normalize_features.py` | annotated phenotype tables | per-cell z-scored Parquet |
| 5 | `aggregate_gene_level.py` | per-cell Parquet | gene-level CSV |
| 6 | `bootstrap_fdr.py` | per-cell Parquet | gene-by-feature p-values and FDR |
