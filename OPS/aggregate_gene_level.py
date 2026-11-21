#!/usr/bin/env python
"""Gene-level profiles from a per-cell table of z-scored features: the median of each feature over the cells of every gRNA, then the median over the gRNAs of every gene
Cells without an assignment are dropped. Features are processed in column batches to bound memory.
Works on the per-cell Parquet written by normalize_features.py (gene column Input, gRNA column Matched_gRNA).
Output: CSV with Gene, Cell_Number (cells of the gene) and one column per feature.
usage: python aggregate_gene_level.py --input cells.parquet --output gene_level.csv [--gene-col Input] [--guide-col Matched_gRNA] [--exclude Unassigned ...] [--key-cols Well Field ...]"""
import argparse, gc, time
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
ap.add_argument('--gene-col', default='Input'); ap.add_argument('--guide-col', default='Matched_gRNA')
ap.add_argument('--exclude', nargs='*', default=['Unassigned'], help='gene values to drop')
ap.add_argument('--key-cols', nargs='*', default=['Plate', 'Plate_ID', 'Well', 'Site', 'Field', 'label_id', 'Cytoplasm_UUID', 'WholeCell_UUID', 'Nuclei_UUID', 'Matched_gRNA', 'gRNA_13nt', 'Input', 'Gene'], help='non-feature columns')
ap.add_argument('--batch', type=int, default=60, help='features per batch')
a = ap.parse_args()

names = pq.ParquetFile(a.input).schema_arrow.names
feats = [c for c in names if c not in set(a.key_cols) | {a.gene_col, a.guide_col}]
ids = pq.read_table(a.input, columns=[a.gene_col, a.guide_col]).to_pandas()
keep = (ids[a.gene_col].notna() & ~ids[a.gene_col].astype(str).isin(a.exclude) & ids[a.guide_col].notna()).values
ids = ids[keep].reset_index(drop=True); ids[a.gene_col] = ids[a.gene_col].astype(str)
gi = pd.factorize(ids[a.gene_col] + '|' + ids[a.guide_col].astype(str))[0]
print(len(feats), 'features;', keep.sum(), 'of', len(keep), 'cells kept;', gi.max() + 1, 'gRNAs;', ids[a.gene_col].nunique(), 'genes', flush=True)
guide_gene = ids.assign(g=gi).drop_duplicates('g').sort_values('g')[a.gene_col].values; counts = ids.groupby(a.gene_col).size()
parts = []; t = time.time()
for i in range(0, len(feats), a.batch):
    cols = feats[i:i + a.batch]; X = pq.read_table(a.input, columns=cols).to_pandas()[keep].reset_index(drop=True)
    gm = X.groupby(gi).median(); gm[a.gene_col] = guide_gene[gm.index.values]          # median per gRNA
    parts.append(gm.groupby(a.gene_col)[cols].median()); del X, gm; gc.collect()          # median per gene
    print(i + len(cols), 'features done', round(time.time() - t), 's', flush=True)
G = pd.concat(parts, axis=1)[feats]; G.insert(0, 'Cell_Number', counts.reindex(G.index).values)
G = G.reset_index().rename(columns={a.gene_col: 'Gene'}).sort_values('Gene'); G.to_csv(a.output, index=False); print('saved', a.output, G.shape)
