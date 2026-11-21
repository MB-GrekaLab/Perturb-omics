#!/usr/bin/env python
# coding: utf-8
"""Assign a gRNA / gene to every cell."""
import itertools
import os
import re
import sys
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ops_config as oc

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
cfg = oc.load(); BC = cfg['barcode_calling']
oc.require(cfg, 'root_dir', 'sbs.n_cycles', 'plate_format', 'col_ranges', 'barcode_calling.whitelist', 'barcode_calling.barcode_column',
           'barcode_calling.training_cells', 'barcode_calling.training_spots')
ROOT_DIR = cfg['root_dir']
BARCODE_COL = BC['barcode_column']
GENE_COLS = [BARCODE_COL] + list(BC['gene_columns'])
N_CYCLES = cfg['sbs']['n_cycles']
CYCLE_COLS = [f'BC_Cycle{c:02d}' for c in range(N_CYCLES)]            # one called base per sequencing cycle

# One array task = one plate row x one column block
ROWS = oc.plate_rows(cfg)
COL_RANGES = [tuple(r) for r in cfg['col_ranges']]
task = oc.task_id()
row_idx, range_idx = (task - 1) // len(COL_RANGES), (task - 1) % len(COL_RANGES)
if row_idx >= len(ROWS):
    sys.exit(f"Task {task} out of range.")
TARGET_ROW, (START_COL, END_COL) = ROWS[row_idx], COL_RANGES[range_idx]
print(f"Task {task}: row {TARGET_ROW}, columns {START_COL}-{END_COL}")

PLATE = oc.plates(cfg)[int(os.environ.get('OPS_PLATE_INDEX') or BC.get('plate_index') or 0)]
ALIGNED_DIR = os.path.join(ROOT_DIR, 'Aligned', PLATE)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'Analysis_Groups', f"{PLATE}_Row{TARGET_ROW}_Cols{START_COL}-{END_COL}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. HELPERS
# ==============================================================================
CYCLE_PATTERN = re.compile(r"_c(\d+)_")


def mismatch_neighbours(code, n):
    """All strings that differ from `code` at exactly n positions (n = 1 or 2)."""
    out = []
    for pos in itertools.combinations(range(len(code)), n):
        for subs in itertools.product('ACGT', repeat=n):
            if all(s != code[p] for p, s in zip(pos, subs)):
                new = list(code)
                for p, s in zip(pos, subs):
                    new[p] = s
                out.append(''.join(new))
    return out


def unique_mismatch_match(codes, valid, n):
    """{called code: whitelist barcode} for the codes that have exactly one whitelist barcode at n mismatches."""
    found = {}
    for code in set(codes):
        hits = [v for v in mismatch_neighbours(code, n) if v in valid]
        if len(hits) == 1:
            found[code] = hits[0]
    return found


def read_codes(df):
    """The called barcode of every row (bases of all cycles joined)."""
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)
    return df[CYCLE_COLS].astype(str).agg(''.join, axis=1)


def without_dominant_base(df):
    """Drop rows in which one base makes up half or more of the cycles (no real barcode looks like that)."""
    if df.empty:
        return df
    return df[df[CYCLE_COLS].apply(lambda r: r.value_counts().iloc[0] < N_CYCLES * 0.5, axis=1)]


def spot_consensus(spots):
    """One barcode per nucleus, and only if all its spots agree."""
    if spots.empty or 'Parent_Nucleus_UUID' not in spots.columns:
        return pd.DataFrame(columns=['Parent_Nucleus_UUID', 'Matched_gRNA'])
    n_codes = spots.groupby('Parent_Nucleus_UUID')['Matched_gRNA'].nunique()
    agreeing = spots[spots['Parent_Nucleus_UUID'].isin(n_codes[n_codes == 1].index)]
    return agreeing.drop_duplicates(subset=['Parent_Nucleus_UUID'])[['Parent_Nucleus_UUID', 'Matched_gRNA']]


def bad_fields_of(well_dir, well):
    """Fields flagged by the alignment QC report of a well."""
    path = os.path.join(well_dir, f"{well}_concerning_cycles.csv")
    if os.path.exists(path):
        try:
            fields = pd.read_csv(path)['Field'].unique().tolist()
            print(f"  QC: excluding fields {fields} in {well}")
            return fields
        except Exception:
            pass
    return []


def to_zero_based(df):
    """Intensity_Max_SBS_c1_ch1 -> Intensity_Max_c0_ch1 (cycles counted from 0, 'SBS' dropped)."""
    rename = {}
    for col in df.columns:
        if 'SBS' in col and '_c' in col:
            parts = []
            for p in col.split('_'):
                if p.startswith('c') and p[1:].isdigit():
                    parts.append(f"c{int(p[1:]) - 1}")
                elif p != 'SBS':
                    parts.append(p)
            rename[col] = '_'.join(parts)
    return df.rename(columns=rename)


def read_table(path, bad_fields, zero_based=False):
    """A feature table without the bad fields; None if it is missing or empty."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if bad_fields and 'Field' in df.columns:
        df = df[~df['Field'].isin(bad_fields)]
    if df.empty:
        return None
    return to_zero_based(df) if zero_based else df


def row_zscores(df):
    """z-score of every row across the columns (a feature across the sequencing cycles)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0).fillna(0)


# ==============================================================================
# 3. BASE CALLING (the same for nuclei and for spots)
# ==============================================================================
def call_barcodes(features, training, valid, kind):
    """Per-cycle base calling with L1 logistic regression trained on the hand-labelled objects and extended with its own
    exact whitelist hits (8 rounds). Returns the objects matched exactly ('perfect'), with one unique mismatch ('off1'),
    with two ('off2'), and the exact matches among objects with a focus failure ('ff'); each with a Matched_gRNA column."""
    n = N_CYCLES
    training = training.copy(); features = features.copy()
    training.index = [uuid.uuid4().hex[:16] for _ in range(len(training))]
    features.fillna(0, inplace=True); training.fillna(0, inplace=True)

    # Objects with an empty channel in some cycle are focus failures and are called separately
    failed = features[[c for c in features.columns if c.startswith('Intensity_Max_')]].eq(0).any(axis=1).values
    focus_failed, ok = features[failed].copy(), features[~failed].copy()

    # Scale the features of every cycle
    for c in range(n):
        cols = [x for x in features.columns if f'c{c}_' in str(x) and x in training.columns]
        if cols:
            ok[cols] = StandardScaler().fit_transform(ok[cols])
            training[cols] = StandardScaler().fit_transform(training[cols])

    # z-score each feature across the cycles
    feature_cols = training.columns[training.columns.str.contains('Intensity|Corr|Texture|Granularity')]
    bases = sorted({CYCLE_PATTERN.sub('_c_', c) for c in feature_cols})

    def add_zscores(df, reference_columns):
        for base in bases:
            cols = [x for x in (base.replace('_c_', f'_c{c}_') for c in range(n)) if x in reference_columns]
            if cols:
                df[[f'{x}_ZScore' for x in cols]] = row_zscores(df[cols])

    add_zscores(training, training.columns); add_zscores(ok, training.columns)
    per_cycle = {c: [x for x in training.columns if f'_c{c}_' in x and 'ch0' not in x and 'BC' not in x and 'Confidence' not in x] for c in range(n)}
    model = LogisticRegression(C=10, class_weight='balanced', random_state=42, penalty='l1', solver='liblinear', max_iter=200)
    original = training.copy(); calls = [''] * len(ok)

    print(f"Iterative {kind} base calling ({len(ok)} objects):")
    for it in range(8):
        calls = [''] * len(ok)
        for c in range(n):
            model.fit(training[per_cycle[c]], training[CYCLE_COLS[c]])
            calls = [a + str(p) for a, p in zip(calls, model.predict(ok[per_cycle[c]]))]
        hits = [i for i, code in enumerate(calls) if code in valid]
        print(f"  > Iteration {it + 1}: {len(hits)} exact whitelist barcodes ({100 * len(hits) / max(len(ok), 1):.2f}%)")
        if not hits:
            print("    Stopping early (no exact matches).")
            break
        confirmed = ok.iloc[hits].copy()
        for c in range(n):
            confirmed[CYCLE_COLS[c]] = [calls[i][c] for i in hits]
        extra = confirmed.sample(frac=0.333)
        extra = extra[extra[CYCLE_COLS].apply(lambda r: r.value_counts().iloc[0] / n <= 0.49, axis=1)]    # no dominant base
        if not extra.empty:                                                                                 # drop barcodes that are already over-represented
            extra = extra[extra.groupby(CYCLE_COLS)[CYCLE_COLS[0]].transform('size') < 4]
        training = pd.concat([original, extra[[c for c in original.columns if c in extra.columns]]], ignore_index=True).fillna(0)

    final = ok.copy()
    for c in range(n):
        final[CYCLE_COLS[c]] = [code[c] for code in calls]
    final['Matched_gRNA'] = read_codes(final)

    perfect = final[final['Matched_gRNA'].isin(valid)]
    rest = without_dominant_base(final[~final['Matched_gRNA'].isin(valid)])

    codes = read_codes(rest); one = unique_mismatch_match(codes, valid, 1)
    off1 = rest[codes.isin(one)].copy(); off1['Matched_gRNA'] = codes[off1.index].map(one)

    rest = rest.drop(off1.index); codes = read_codes(rest); two = unique_mismatch_match(codes, valid, 2)
    off2 = rest[codes.isin(two)].copy(); off2['Matched_gRNA'] = codes[off2.index].map(two)

    # Focus failures: call the bases with the trained model and keep exact whitelist matches
    add_zscores(focus_failed, original.columns)
    ff_calls = [''] * len(focus_failed)
    if not focus_failed.empty:
        for c in range(n):
            model.fit(training[per_cycle[c]], training[CYCLE_COLS[c]])
            ff_calls = [a + ('N' if p in (0, None) else str(p)) for a, p in zip(ff_calls, model.predict(focus_failed[per_cycle[c]]))]
    matched = []
    for code in ff_calls:
        if code.count('N') > 3:
            matched.append(0)
        elif 'N' not in code:
            matched.append(code if code in valid else 0)
        else:
            hits = [v for v in valid if all(b == 'N' or b == v[i] for i, b in enumerate(code))]
            matched.append(hits[0] if len(hits) == 1 else 0)
    focus_failed['Matched_gRNA'] = matched
    ff = focus_failed[focus_failed['Matched_gRNA'].ne(0)]
    return dict(perfect=perfect, off1=off1, off2=off2, ff=ff)


# ==============================================================================
# 4. LOAD THE WELLS OF THIS TASK
# ==============================================================================
print("\n--- LOADING DATA ---")
wells = []
for name in oc.wells(cfg):
    m = re.match(r"([A-Za-z]+)(\d+)", name.split('_')[-1])
    if m and m.group(1).upper() == TARGET_ROW and START_COL <= int(m.group(2)) <= END_COL:
        wells.append(name)
print(f"Found {len(wells)} matching wells.")

tables = {k: [] for k in ('nuclei', 'spots', 'pheno_nuclei', 'pheno_cell', 'pheno_cyto')}
for well in wells:
    well_path = os.path.join(ALIGNED_DIR, well)
    if not os.path.exists(well_path):
        continue
    bad = bad_fields_of(well_path, well)
    for key, name, zero_based in [('nuclei', 'SBS_Nuclei_Features', True), ('spots', 'SBS_Spots_Features', True),
                                  ('pheno_nuclei', 'Pheno_Nuclei_Features', False), ('pheno_cell', 'Pheno_WholeCell_Features', False),
                                  ('pheno_cyto', 'Pheno_Cytoplasm_Features', False)]:
        df = read_table(os.path.join(well_path, f"{well}_{name}_Combined.csv"), bad, zero_based)
        if df is not None:
            tables[key].append(df)

tables = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in tables.items()}
nuclei, spots = tables['nuclei'], tables['spots']
print(f"Loaded data from {len(wells)} wells.")
if nuclei.empty:
    sys.exit("No SBS nuclei found.")

# ==============================================================================
# 5. CALL THE BARCODES OF NUCLEI AND SPOTS
# ==============================================================================
valid_df = pd.read_csv(BC['whitelist'])
valid = set(valid_df[BARCODE_COL])


def read_training(key):
    path = oc.fmt(cfg, BC[key])
    df = pd.read_csv(path)
    if df.empty or df[[c for c in df.columns if c.startswith('BC_Cycle')]].replace('', np.nan).isna().all().any():
        sys.exit(f'Training file {path} is empty or has unlabelled BC_Cycle columns: make the training files for this dataset first.')
    return to_zero_based(df)                                           # training data are named c1..c<n>; the code counts cycles from 0


print("\n--- NUCLEI ---")
cell_buckets = call_barcodes(nuclei, read_training('training_cells'), valid, 'nucleus')

print("\n--- SPOTS ---")
if spots.empty:
    spot_buckets = {k: pd.DataFrame() for k in cell_buckets}
else:
    spot_buckets = {k: spot_consensus(v) for k, v in call_barcodes(spots, read_training('training_spots'), valid, 'spot').items()}

# ==============================================================================
# 6. MERGE IN FIXED ORDER AND ANNOTATE
# ==============================================================================
print("\n--- MERGING & MAPPING GENES ---")
known_nuclei = set(nuclei['Nuclei_UUID'])
master = pd.DataFrame(columns=['Nuclei_UUID', 'Matched_gRNA', 'Method'])
for bucket, label in [('perfect', 'Perfect'), ('off1', 'Off1'), ('off2', 'Off2'), ('ff', 'FF')]:
    for source, buckets in (('Cell', cell_buckets), ('Spot', spot_buckets)):
        df = buckets[bucket]
        if df.empty:
            continue
        if source == 'Spot':
            df = df.rename(columns={'Parent_Nucleus_UUID': 'Nuclei_UUID'})[['Nuclei_UUID', 'Matched_gRNA']]
            df = df[df['Nuclei_UUID'].isin(known_nuclei)]
        df = df[~df['Nuclei_UUID'].isin(set(master['Nuclei_UUID']))][['Nuclei_UUID', 'Matched_gRNA']].assign(Method=f"{source}_{label}")
        master = pd.concat([master, df], ignore_index=True)

print("Mapping barcodes to genes...")
master = master.merge(valid_df[[c for c in GENE_COLS if c in valid_df.columns]], left_on='Matched_gRNA', right_on=BARCODE_COL, how='left')
if BARCODE_COL in master.columns:
    del master[BARCODE_COL]
print(f"Total identified nuclei: {len(master)}")

# ==============================================================================
# 7. SAVE THE ANNOTATED PHENOTYPE TABLES
# ==============================================================================
nuclei_map = nuclei[['Well', 'Field', 'label_id', 'Nuclei_UUID']]
for key, name in [('pheno_nuclei', 'Pheno_Nuclei'), ('pheno_cell', 'Pheno_WholeCell'), ('pheno_cyto', 'Pheno_Cytoplasm')]:
    pheno = tables[key]
    if pheno.empty:
        continue
    if 'Nuclei_UUID' not in pheno.columns:                              # whole-cell and cytoplasm objects are linked through their nucleus
        pheno = pheno.merge(nuclei_map, on=['Well', 'Field', 'label_id'], how='left')
    annotated = pheno.merge(master, on='Nuclei_UUID', how='left')
    for col, empty in (('Matched_gRNA', 'Unassigned'), ('Method', 'Unmatched'), ('Input', 'Unassigned')):
        annotated[col] = annotated[col].fillna(empty)
    out = os.path.join(OUTPUT_DIR, f"Row{TARGET_ROW}_Cols{START_COL}-{END_COL}_{name}_Annotated.csv")
    annotated.to_csv(out, index=False)
    print(f"Saved: {out} (rows: {len(annotated)})")

print("\nAnalysis group complete.")
