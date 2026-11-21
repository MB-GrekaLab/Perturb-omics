"""Per-cell feature normalisation.
Loads the annotated phenotype tables (<Row>_Cols<a>-<b>_Pheno_<compartment>_Annotated.csv from assign_gRNAs.py), performs cell/image QC,
regresses the background-cycle intensities out of every other intensity feature (ordinary least squares, per plate) and robust z-scores
every intensity feature per plate ((x - median) / (1.4826 x MAD)); the background columns keep their raw values.
Writes one per-cell table (Parquet): Well, Field, <compartment>_UUID, Matched_gRNA, Input, Plate_ID + features, ready for aggregate_gene_level.py.
usage: python normalize_features.py [Cytoplasm|WholeCell|Nuclei]      (default: normalization.compartment of config.yaml)"""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ops_config as oc

# ==============================================================================
# 1. SETUP
# ==============================================================================
cfg = oc.load(); N = cfg['normalization']; QC = N['qc']; BG = N['background']
oc.require(cfg, 'root_dir', 'imaging.bit_depth', 'normalization.background.cycle', 'normalization.background.channels', 'normalization.background.statistics')
ROOT_DIR = cfg['root_dir']
COMPARTMENT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else N['compartment']      # Cytoplasm, WholeCell or Nuclei
if not COMPARTMENT:
    sys.exit('Set normalization.compartment in the config or pass Cytoplasm, WholeCell or Nuclei as the first argument.')
UUID_COL = f'{COMPARTMENT}_UUID'
OUTPUT_FILE = oc.fmt(cfg, N['output'], compartment=COMPARTMENT)
MIN_DAPI_MAX = QC['min_dapi_max_fraction'] * 2 ** cfg['imaging']['bit_depth']              # fraction of the imager's maximum intensity

# The background (secondary-only) cycle gives the regression confounders: raw column -> Background_<wavelength>_<statistic>
STAT_NAME = {'mean': 'Intensity_Mean', 'max': 'Intensity_Max', 'median': 'Intensity_Median', 'integrated': 'Intensity_Integrated'}
BG_MAP = {f"{STAT_NAME[s]}_Pheno_c{BG['cycle']}_ch{ch}": f"Background_{nm}_{s}" for ch, nm in BG['channels'].items() for s in BG['statistics']}
CONFOUNDERS = list(BG_MAP.values())

files = glob.glob(os.path.join(ROOT_DIR, 'Analysis_Groups', f"**/*_Pheno_{COMPARTMENT}_Annotated.csv"), recursive=True)
if not files:
    sys.exit(f"No {COMPARTMENT} files found.")
header = pd.read_csv(files[0], nrows=0).columns
raw_intensity_cols = [c for c in header if 'SpIDA_' in c or 'Intensity_' in c]
intensity_cols = [BG_MAP.get(c, c) for c in raw_intensity_cols]
FEATURES = [c for c in intensity_cols if c not in CONFOUNDERS]                              # corrected and z-scored
cols_to_load = list({'Input', 'Matched_gRNA', 'Well', 'Field', UUID_COL, *raw_intensity_cols, *BG_MAP})

# ==============================================================================
# 2. SITES WITH A CONCERNING CYCLE (alignment QC reports)
# ==============================================================================
bad_sites = set()
for well in oc.wells(cfg):
    for report in glob.glob(os.path.join(ROOT_DIR, 'Aligned', 'Plate*', well, f"{well}_concerning_cycles.csv")):
        try:
            bad = pd.read_csv(report)
            bad = bad[bad['Cycle'].str.startswith('Pheno_Cy', na=False)]
            bad_sites |= set(bad['Plate'].astype(str) + '_' + bad['Well'].astype(str) + '_' + bad['Field'].astype(float).astype(int).astype(str))
        except Exception:
            continue
print(f"{len(bad_sites)} concerning plate / well / field combinations to exclude.")

# ==============================================================================
# 3. LOAD THE TABLES WITH CELL QC
# ==============================================================================
def plate_of(path):
    """Plate name from the folder name, e.g. .../Plate3_RowB_Cols2-23/... -> Plate3."""
    folders = [p for p in path.split(os.sep) if 'Plate' in p]
    return folders[0].split('_')[0] if folders else 'Unknown'


def dapi_qc(wc):
    """Cells that pass the DAPI (channel 1) checks of the whole-cell table: bright enough, no big drop in integrated signal, in focus."""
    ch1 = lambda stat: [c for c in wc.columns if f'Intensity_{stat}_Pheno_c' in c and '_ch1' in c]
    mx, integrated, std = ch1('Max'), ch1('Integrated'), ch1('Std')
    ok = pd.Series(True, index=wc.index)
    if mx:
        ok &= wc[mx].notna().all(axis=1) & (wc[mx] > MIN_DAPI_MAX).all(axis=1)
    if integrated:
        ok &= wc[integrated].ge(wc[integrated].max(axis=1) * QC['min_integrated_fraction'], axis=0).all(axis=1)
    if std:                                                          # focus: mean DAPI Intensity_Std across cycles, MAD cut-off below the median
        focus = wc[std].mean(axis=1)
        mad = (focus - focus.median()).abs().median()
        mad = mad if mad > 0 else 1.0
        ok &= focus >= focus.median() - QC['focus_mad_cutoff'] * mad * 1.4826
    return ok


def load(path):
    """One table without poor cells, unassigned cells and concerning sites; the QC comes from the matching whole-cell table."""
    try:
        wc_path = path if COMPARTMENT == 'WholeCell' else path.replace(f'_{COMPARTMENT}_', '_WholeCell_')
        if not os.path.exists(wc_path):
            print(f"Warning: no whole-cell table for {path}; skipped.")
            return None
        df = pd.read_csv(path, usecols=cols_to_load).rename(columns=BG_MAP)
        wc_cols = [c for c in pd.read_csv(wc_path, nrows=0).columns if '_ch1' in c and any(f'Intensity_{s}_Pheno_c' in c for s in ('Max', 'Integrated', 'Std'))]
        if wc_cols:
            df = df[dapi_qc(pd.read_csv(wc_path, usecols=wc_cols))]
        df = df[df['Input'] != 'Unassigned'].copy()
        df['Plate_ID'] = plate_of(path)
        if bad_sites and not df.empty:
            df = df[~(df['Plate_ID'] + '_' + df['Well'].astype(str) + '_' + df['Field'].astype(float).astype(int).astype(str)).isin(bad_sites)].copy()
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


print(f"Loading {len(files)} files...")
with ThreadPoolExecutor(max_workers=20) as pool:
    tables = [df for df in pool.map(load, files) if df is not None and not df.empty]
cells = pd.concat(tables, ignore_index=True)
del tables
cells[intensity_cols] = cells[intensity_cols].fillna(0)
for col in ('Input', 'Matched_gRNA', 'Plate_ID'):
    cells[col] = cells[col].astype('category')
print(f"{len(cells)} cells after QC.")

# ==============================================================================
# 4. REGRESS OUT THE BACKGROUND AND Z-SCORE (per plate)
# ==============================================================================
for plate in cells['Plate_ID'].unique():
    rows = cells['Plate_ID'] == plate
    X, Y = cells.loc[rows, CONFOUNDERS].values, cells.loc[rows, FEATURES].values
    residuals = pd.DataFrame(Y - LinearRegression().fit(X, Y).predict(X), columns=FEATURES, index=cells.index[rows])
    median = residuals.median()
    mad = (residuals - median).abs().median().replace(0, 1.0)
    cells.loc[rows, FEATURES] = (residuals - median) / (mad * 1.4826)

# ==============================================================================
# 5. SAVE THE PER-CELL TABLE (aggregate_gene_level.py takes the median per gRNA, then per gene)
# ==============================================================================
out_cols = ['Well', 'Field', UUID_COL, 'Matched_gRNA', 'Input', 'Plate_ID'] + intensity_cols
cells[out_cols].astype({c: 'float32' for c in intensity_cols}).to_parquet(OUTPUT_FILE, index=False)
print(f"Per-cell normalised table: {OUTPUT_FILE} ({len(cells)} cells)")
