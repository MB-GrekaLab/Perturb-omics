#!/usr/bin/env python
"""Gene-level bootstrap p-values and Benjamini-Hochberg FDR per feature from a per-cell table of z-scored features (vectorised version of the OPS / Brieflow bootstrap).
  guide = one gRNA; guide statistic = median of the feature over its cells; keep guides with >= min_cells cells and genes with >= min_guides such guides; gene statistic = median of its guide medians
  null for a guide with n cells = median of n cells resampled with replacement from ONE randomly chosen non-targeting guide, repeated `reps` times; one null per distinct n (shared by all guides of that size)
  gene null = median over the gene's guides of one independent draw from each guide's null; p = 2 * max(min(P(null > obs), P(null < obs)), 1/reps); BH across genes, per feature
Input: per-cell Parquet (normalize_features.py output) with a gene column and a gRNA column; non-targeting cells have 'NEG_CONTROL' in the gene name.
usage: python bootstrap_fdr.py prep    --input cells.parquet --out-dir fdr
       python bootstrap_fdr.py run     --out-dir fdr [--chunk N] [--chunk-size 10]    # chunk = $SGE_TASK_ID - 1 by default (one array task per chunk of features); env FDR_REPS (default 200000), FDR_SEED, FDR_FEATURES (comma list, for tests)
       python bootstrap_fdr.py collect --out-dir fdr [--fdr-threshold 0.1]            # merge chunks -> gene_feature_pvals_long.tsv.gz and gene_by_feature_{gene_median,pval,fdr}.tsv.gz"""
import argparse, os, glob, json, time, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

CTRL = 'NEG_CONTROL'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('mode', choices=['prep', 'run', 'collect']); ap.add_argument('--input'); ap.add_argument('--out-dir', required=True)
ap.add_argument('--gene-col', default='Input'); ap.add_argument('--guide-col', default='Matched_gRNA')
ap.add_argument('--key-cols', nargs='*', default=['Plate', 'Plate_ID', 'Well', 'Site', 'Field', 'label_id', 'Cytoplasm_UUID', 'WholeCell_UUID', 'Nuclei_UUID', 'Matched_gRNA', 'gRNA_13nt', 'Input', 'Gene'])
ap.add_argument('--min-cells', type=int, default=10); ap.add_argument('--min-guides', type=int, default=3)
ap.add_argument('--chunk', type=int); ap.add_argument('--chunk-size', type=int, default=10); ap.add_argument('--fdr-threshold', type=float, default=0.10)
A = ap.parse_args(); FD = A.out_dir; PREP = os.path.join(FD, 'prep'); os.makedirs(PREP, exist_ok=True)


def prep():
    import pyarrow.parquet as pq
    names = pq.ParquetFile(A.input).schema_arrow.names; feats = [c for c in names if c not in set(A.key_cols) | {A.gene_col, A.guide_col}]
    sc = pd.read_parquet(A.input, columns=[A.guide_col, A.gene_col] + feats)
    sc = sc[sc[A.gene_col].notna() & (sc[A.gene_col].astype(str) != 'Unassigned') & sc[A.guide_col].notna()]
    is_ctrl = sc[A.gene_col].astype(str).str.contains(CTRL, na=False).values; sc['gene'] = np.where(is_ctrl, CTRL, sc[A.gene_col].astype(str))
    g = sc.groupby(A.guide_col); guides = g[feats].median(); guides.insert(0, 'count', g.size()); guides.insert(0, 'gene', g.gene.first()); guides.reset_index().rename(columns={A.guide_col: 'guide'}).to_parquet(f'{PREP}/guides.parquet', index=False)
    nt = sc[is_ctrl].sort_values(A.guide_col); X = nt[feats].values.astype(np.float32); _, start, size = np.unique(nt[A.guide_col].values, return_index=True, return_counts=True)
    np.save(f'{PREP}/ctrl_X.npy', X); np.savez(f'{PREP}/ctrl_groups.npz', start=start, size=size); open(f'{PREP}/features.txt', 'w').write('\n'.join(feats))
    print(f'prep: {len(feats)} features, {len(guides)} guides ({int((guides.gene == CTRL).sum())} non-targeting), {len(X)} non-targeting cells in {len(start)} guides')


def null_for_n(n, S, X, start, size, rng, batch=500):
    """(S, f) medians of n cells resampled (with replacement) from one random non-targeting guide per simulation; vectorised"""
    out = np.empty((S, X.shape[1]), np.float32)
    for s0 in range(0, S, batch):
        b = min(batch, S - s0); g = rng.integers(0, len(start), b); idx = start[g][:, None] + (rng.random((b, n)) * size[g][:, None]).astype(np.int64)
        out[s0:s0 + b] = np.median(X[idx], axis=1)
    return out


def gene_pvals(obs, nulls, rng, reps):
    """obs: (f,) observed gene median; nulls: list over the gene's guides of (S, f) arrays -> (f,) p-values; independent index per guide"""
    G = len(nulls); S = nulls[0].shape[0]; draws = np.stack([nulls[k][rng.integers(0, S, reps)] for k in range(G)])      # (G, reps, f), independent draws per guide
    gn = np.median(draws, axis=0); gt = (gn > obs).mean(0); lt = (gn < obs).mean(0)
    return 2 * np.maximum(np.minimum(gt, lt), 1.0 / reps)


def run(chunk, size_):
    reps = int(os.environ.get('FDR_REPS', 200000)); feats = open(f'{PREP}/features.txt').read().split('\n'); sel = os.environ.get('FDR_FEATURES')
    cols = sel.split(',') if sel else feats[chunk * size_:(chunk + 1) * size_]
    if not cols: print('empty chunk'); return
    fi = [feats.index(c) for c in cols]; G = pd.read_parquet(f'{PREP}/guides.parquet'); X = np.load(f'{PREP}/ctrl_X.npy', mmap_mode='r')[:, fi]; X = np.ascontiguousarray(X); st = np.load(f'{PREP}/ctrl_groups.npz'); start, size = st['start'], st['size']
    tg = G[(G.gene != CTRL) & (G['count'] >= A.min_cells)]; ng = tg.groupby('gene').size(); tg = tg[tg.gene.isin(ng[ng >= A.min_guides].index)]; ns = np.sort(tg['count'].unique())
    seed = int(os.environ.get('FDR_SEED', 0)); rng = np.random.default_rng([seed, chunk]); t0 = time.time(); print(f'chunk {chunk}: {len(cols)} features, {len(tg)} guides in {tg.gene.nunique()} genes, {len(ns)} distinct cell counts, {reps} reps', flush=True)
    nulls = {int(n): null_for_n(int(n), reps, X, start, size, rng) for n in ns}; t1 = time.time(); print(f'  guide nulls {t1 - t0:.0f}s', flush=True)
    rows = []; med = {g: tg[tg.gene == g] for g in tg.gene.unique()}
    for gene, d in med.items():
        obs = np.median(d[cols].values, axis=0); p = gene_pvals(obs, [nulls[int(n)] for n in d['count']], rng, reps); rows.append(pd.DataFrame(dict(gene=gene, n_guides=len(d), feature=cols, gene_median=obs, pval=p)))
    out = pd.concat(rows); out['fdr'] = out.groupby('feature').pval.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    out.to_csv(f'{FD}/chunk_{chunk:04d}.tsv.gz', sep='\t', index=False); json.dump(dict(chunk=chunk, features=len(cols), reps=reps, guides=len(tg), genes=tg.gene.nunique(), distinct_n=len(ns), sec_nulls=round(t1 - t0), sec_genes=round(time.time() - t1), sec_total=round(time.time() - t0)), open(f'{FD}/chunk_{chunk:04d}.json', 'w'))
    print(f'  genes {time.time() - t1:.0f}s, total {time.time() - t0:.0f}s', flush=True)


def collect():
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(f'{FD}/chunk_*.tsv.gz'))]); d.sort_values('fdr').to_csv(f'{FD}/gene_feature_pvals_long.tsv.gz', sep='\t', index=False)
    for k in ['gene_median', 'pval', 'fdr']: d.pivot(index='gene', columns='feature', values=k).to_csv(f'{FD}/gene_by_feature_{k}.tsv.gz', sep='\t')
    print(len(d), 'gene-feature tests;', d.gene.nunique(), 'genes;', d.feature.nunique(), 'features; FDR<%g:' % A.fdr_threshold, int((d.fdr < A.fdr_threshold).sum()), 'tests in', d[d.fdr < A.fdr_threshold].gene.nunique(), 'genes')


if A.mode == 'prep': prep()
elif A.mode == 'run': run(A.chunk if A.chunk is not None else int(os.environ['SGE_TASK_ID']) - 1, A.chunk_size)
else: collect()
