#!/usr/bin/env python
# coding: utf-8
"""Segment nuclei (StarDist), whole cells (watershed on the cell-outline channel) and sequencing spots (StarDist on the SBS max projection) of one well and
measure per-object intensity, texture, granularity and channel-pair features in every SBS and phenotype cycle. Writes masks (<Well>_s<site>_<Nuclei|WholeCell|Cytoplasm|Spots>.tif)
and <Well>_<SBS|Pheno>_<Object>_Features_Combined.csv next to the aligned images. One array task = one well: qsub -t 1-<n_wells> ... python feature_extraction.py
Cycle counts, channels, segmentation images and thresholds come from config.yaml (see ops_config.py)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ops_config as oc
import uuid
import math
import numpy as np
import pandas as pd
import glob
import re
from itertools import combinations
from tqdm import tqdm
from scipy.stats import median_abs_deviation, skew, kurtosis
from scipy.ndimage import binary_fill_holes
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops, shannon_entropy
from skimage.morphology import opening, disk
from skimage.filters import gaussian, threshold_multiotsu
from skimage.util import view_as_blocks
from skimage.segmentation import watershed, expand_labels, clear_border

# ==============================================================================
#   CONFIGURATION
# ==============================================================================

cfg = oc.load(); SEG = cfg['segmentation']; WS = SEG['watershed']
oc.require(cfg, 'root_dir', 'experiment', 'sbs.n_cycles', 'phenotype.n_cycles', 'sbs.reference_cycle', 'imaging.fields_per_well', 'imaging.channels_per_cycle', 'segmentation.stardist_model_dir', 'segmentation.nuclei.cycle', 'segmentation.nuclei.channel', 'segmentation.cytoplasm.cycle', 'segmentation.cytoplasm.channel')

# --- PATHS ---
ROOT_DIR = cfg['root_dir']
MODEL_DIR = SEG['stardist_model_dir']
MODEL_NAME = SEG['stardist_model_name']

# --- EXPERIMENT STRUCTURE ---
SBS_CYCLES = cfg['sbs']['n_cycles']
PHENO_CYCLES = cfg['phenotype']['n_cycles']
N_FIELDS = cfg['imaging']['fields_per_well']

# --- SEGMENTATION SETTINGS ---
# 1. Nuclei image (DAPI)
SEG_NUC_MODE, SEG_NUC_CYC, SEG_NUC_CH = SEG['nuclei']['mode'], SEG['nuclei']['cycle'], SEG['nuclei']['channel']

# 2. Cell-outline image (cytoplasmic stain)
SEG_CYTO_MODE, SEG_CYTO_CYC, SEG_CYTO_CH = SEG['cytoplasm']['mode'], SEG['cytoplasm']['cycle'], SEG['cytoplasm']['channel']

# 3. Spots: max projection of these SBS cycles
SPOT_CYCLES = SEG.get('spot_cycles') or list(range(1, SBS_CYCLES + 1))

# 4. QC Thresholds
MIN_DAPI_INTENSITY = SEG['min_dapi_intensity']

# --- STARDIST SETTINGS ---
NUC_PROB_THRESH = SEG['nuc_prob_thresh']
NUC_NMS_THRESH = SEG['nuc_nms_thresh']

# Spot Settings
SPOT_PROB_THRESH = SEG['spot_prob_thresh']
SPOT_NMS_THRESH = SEG['spot_nms_thresh']

# --- WATERSHED SETTINGS ---
CP_ADAPTIVE_WINDOW = WS['adaptive_window']
CP_SMOOTHING_SIGMA = WS['smoothing_sigma']
CP_CORRECTION_FACTOR = WS['correction_factor']
CP_THRESH_BOUNDS = tuple(WS['thresh_bounds'])
CP_MIDDLE_TO_FOREGROUND = WS['middle_to_foreground']
CP_FILL_HOLES = WS['fill_holes']
WATERSHED_COMPACTNESS = WS['compactness']
MAX_EXPANSION_PX = WS['max_expansion_px']

# ==============================================================================
#   SGE SETUP
# ==============================================================================
SGE_TASK_ID = oc.task_id()
list_index = SGE_TASK_ID - 1
PLATES = oc.plates(cfg)
WELL = oc.wells(cfg)[list_index]

# --- PARSE METADATA FROM WELL NAME ---
well_clean = WELL.split('_')[-1] 
match = re.match(r"([A-Za-z]+)(\d+)", well_clean)
if match:
    META_ROW = match.group(1)       # "A"
    META_COL = int(match.group(2))  # 1
else:
    META_ROW = "Unknown"
    META_COL = 0

print(f"Metadata Parsed: Row={META_ROW}, Col={META_COL}")

# ==============================================================================
#   HELPER FUNCTIONS
# ==============================================================================

def _ensure_2d_single_plane(img):
    if img.ndim == 3:
        if img.shape[-1] in (3, 4): img = img[..., 0]
        else: img = img[0]
    return img

def _normalize01(img):
    m, M = float(np.min(img)), float(np.max(img))
    if M <= m: return np.zeros_like(img, dtype=np.float32)
    return ((img - m) / (M - m + 1e-12)).astype(np.float32)

def load_aligned_image(base_dir, well, field, mode, cycle, channel):
    fname = f"{well}_s{field}_{mode}_Cy{cycle}_w{channel}_aligned.tiff"
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path): return None, path
    try:
        img = imread(path)
        img = _ensure_2d_single_plane(img)
        return img, path 
    except: return None, path

def _blockwise_multiotsu_mask(img01, win=100, correction_factor=1.0, 
                              bounds=(0.0, 1.0), middle_to_foreground=True):
    h, w = img01.shape
    ph = (math.ceil(h/win)*win) - h
    pw = (math.ceil(w/win)*win) - w
    img_pad = np.pad(img01, ((0, ph), (0, pw)), mode='edge')
    blocks = view_as_blocks(img_pad, block_shape=(win, win))
    mask_pad = np.zeros_like(img_pad, dtype=bool)
    nbins = 256
    lo, hi = bounds
    try: g_t = threshold_multiotsu(img_pad, classes=3, nbins=nbins)
    except: g_t = [np.mean(img_pad), np.mean(img_pad)]

    for bi in range(blocks.shape[0]):
        for bj in range(blocks.shape[1]):
            tile = blocks[bi, bj]
            flat = (np.ptp(tile) < 1e-6)
            fg_tile = None
            if not flat:
                try:
                    t = threshold_multiotsu(tile, classes=3, nbins=nbins)
                    t1 = np.clip(t[0] * correction_factor, lo, hi)
                    t2 = np.clip(t[1] * correction_factor, lo, hi)
                    fg_tile = tile >= (t1 if middle_to_foreground else t2)
                except: pass
            if fg_tile is None:
                gt1 = np.clip(g_t[0] * correction_factor, lo, hi)
                gt2 = np.clip(g_t[1] * correction_factor, lo, hi)
                fg_tile = tile >= (gt1 if middle_to_foreground else gt2)
            mask_pad[bi*win:(bi+1)*win, bj*win:(bj+1)*win] = fg_tile
    return mask_pad[:h, :w]

# --- SEGMENTATION FUNCTIONS ---
def segment_whole_cell(nuclei_labels, img_cyto):
    print("    [Seg] Whole Cell Watershed...")
    I = _normalize01(img_cyto)
    I = np.log1p(I) / np.log(2.0)
    I_s = gaussian(I, sigma=CP_SMOOTHING_SIGMA)
    mask = _blockwise_multiotsu_mask(I_s, win=CP_ADAPTIVE_WINDOW, correction_factor=CP_CORRECTION_FACTOR,
                                     bounds=CP_THRESH_BOUNDS, middle_to_foreground=CP_MIDDLE_TO_FOREGROUND)
    if CP_FILL_HOLES: mask = binary_fill_holes(mask)
    mask = np.logical_or(mask, nuclei_labels > 0)
    growth_ceiling = expand_labels(nuclei_labels, distance=MAX_EXPANSION_PX) > 0
    mask = np.logical_and(mask, growth_ceiling)
    return watershed(-I_s, nuclei_labels, mask=mask, compactness=WATERSHED_COMPACTNESS)

def segment_spots(base_dir, well, field, nuclei_labels, model):
    print(f"    [Seg] Spot Segmentation (SBS max projection of cycles {SPOT_CYCLES[0]}-{SPOT_CYCLES[-1]})...")
    max_proj = None
    for c in SPOT_CYCLES:
        for ch in oc.channels(cfg, 'SBS', c):
            img, _ = load_aligned_image(base_dir, well, field, 'SBS', c, ch)
            if img is not None:
                max_proj = img if max_proj is None else np.maximum(max_proj, img)
    
    if max_proj is None: return None, None
    
    norm_img = normalize(max_proj)
    labels, _ = model.predict_instances(norm_img, nms_thresh=SPOT_NMS_THRESH, prob_thresh=SPOT_PROB_THRESH)
    
    relations = None
    if labels.max() > 0:
        nuclei_expanded = expand_labels(nuclei_labels, distance=5)
        mask_pairs = (labels > 0) & (nuclei_expanded > 0)
        pairs = np.stack([labels[mask_pairs], nuclei_expanded[mask_pairs]], axis=1)
        if len(pairs) > 0:
            df_pairs = pd.DataFrame(pairs, columns=["spot", "nuc"])
            relations = df_pairs.groupby("spot")["nuc"].agg(lambda x: x.mode()[0]).reset_index()
            relations.rename(columns={"spot": "spot_label", "nuc": "parent_nucleus_label"}, inplace=True)
            
    return labels, relations

# --- FEATURE MEASUREMENTS ---
def measure_intensity_advanced(regionprops_list, is_pheno=False):
    results = []
    for rp in regionprops_list:
        pixels = rp.intensity_image[rp.image]
        if len(pixels) == 0: continue
        y_c, x_c = rp.local_centroid
        y_w, x_w = rp.weighted_local_centroid if rp.weighted_local_centroid is not None else (y_c, x_c)
        mass_disp = np.sqrt((y_c - y_w)**2 + (x_c - x_w)**2)
        
        row = {
            'label_id': rp.label,
            'Intensity_Integrated': np.sum(pixels),
            'Intensity_Mean': np.mean(pixels),
            'Intensity_Median': np.median(pixels),
            'Intensity_Max': np.max(pixels),
            'Intensity_Min': np.min(pixels),
            'Intensity_Std': np.std(pixels),
            'Intensity_MAD': median_abs_deviation(pixels, scale=1),
            'Intensity_MassDisplacement': mass_disp
        }
        
        # Calculate skewness and kurtosis specifically for Pheno
        if is_pheno:
            if len(pixels) > 2 and np.std(pixels) > 0:
                row['Intensity_Skewness'] = float(skew(pixels))
                row['Intensity_Kurtosis'] = float(kurtosis(pixels))
            else:
                row['Intensity_Skewness'] = 0.0
                row['Intensity_Kurtosis'] = 0.0
                
        results.append(row)
    return pd.DataFrame(results)

def measure_texture_fast(regionprops_list):
    results = []
    for rp in regionprops_list:
        pixels = rp.intensity_image[rp.image]
        if len(pixels) == 0: continue
        results.append({
            'label_id': rp.label,
            'Texture_Entropy': shannon_entropy(pixels),
            'Texture_StdIntensity': np.std(pixels)
        })
    return pd.DataFrame(results)

def measure_granularity_fast(regionprops_list):
    results = []
    radii = [1, 3, 5]
    for rp in regionprops_list:
        intensity_crop = rp.intensity_image
        mask = rp.image
        base_sum = np.sum(intensity_crop[mask])
        row = {'label_id': rp.label}
        prev_sum = base_sum
        for r in radii:
            opened = opening(intensity_crop, disk(r))
            curr_sum = np.sum(opened[mask])
            row[f'Granularity_{r}'] = prev_sum - curr_sum
            prev_sum = curr_sum
        results.append(row)
    return pd.DataFrame(results)

def measure_pairwise_features(regionprops_list, image_B, include_extras=True):
    results = []
    for rp in regionprops_list:
        mask = rp.image
        crop_A = rp.intensity_image
        crop_B = image_B[rp.slice]
        
        flat_A = crop_A[mask]
        flat_B = crop_B[mask]
        
        label_id = rp.label
        row = {'label_id': label_id}
        
        # 1. Base Correlations
        if len(flat_A) < 2 or np.std(flat_A)==0 or np.std(flat_B)==0:
            row['Corr_Pearson'] = 0.0
            row['Corr_Overlap'] = 0.0
        else:
            row['Corr_Pearson'] = np.corrcoef(flat_A, flat_B)[0, 1]
            denom = (np.linalg.norm(flat_A) * np.linalg.norm(flat_B))
            row['Corr_Overlap'] = np.dot(flat_A, flat_B) / denom if denom > 0 else 0.0
            
        # 2. Extras: Max Ratio & Cross-Channel Mass Displacement
        if include_extras:
            if len(flat_A) > 0 and len(flat_B) > 0:
                # Max Intensity Ratio (A / B)
                max_A = np.max(flat_A)
                max_B = np.max(flat_B)
                row['Ratio_MaxInt'] = float(max_A) / (float(max_B) + 1e-12)
                
                # Cross-Channel Mass Displacement
                sum_A = np.sum(flat_A)
                sum_B = np.sum(flat_B)
                if sum_A > 0 and sum_B > 0:
                    y_idx, x_idx = np.nonzero(mask)
                    w_A = crop_A[y_idx, x_idx]
                    w_B = crop_B[y_idx, x_idx]
                    
                    # Compute weighted local centroids manually for speed
                    cy_A = np.average(y_idx, weights=w_A)
                    cx_A = np.average(x_idx, weights=w_A)
                    cy_B = np.average(y_idx, weights=w_B)
                    cx_B = np.average(x_idx, weights=w_B)
                    
                    # Distance between centroids
                    disp = np.sqrt((cy_A - cy_B)**2 + (cx_A - cx_B)**2)
                    row['CrossMassDisplacement'] = disp
                else:
                    row['CrossMassDisplacement'] = 0.0
            else:
                row['Ratio_MaxInt'] = 0.0
                row['CrossMassDisplacement'] = 0.0

        results.append(row)
    return pd.DataFrame(results)

# --- MASTER EXTRACTION FUNCTION (With Metadata) ---

def extract_set(labels, obj_name, output_dir, well, field, image_defs, set_name, parent_info=None):
    """
    Extracts Features. Adds Row, Column, Field metadata to every row.
    Calculates intra-cycle correlations (plus mass displacements/ratios) for both SBS and Pheno sets.
    """
    print(f"    [Feat] {set_name}: {obj_name} ({len(image_defs)} images)...")
    
    # 1. Base Morphology
    rps_base = regionprops(labels)
    base_data = []
    for rp in rps_base:
        y, x = rp.centroid
        base_data.append({
            'label_id': rp.label, 
            'Area': rp.area, 
            'Radius': np.sqrt(rp.area / np.pi),
            'Centroid_X': x, 
            'Centroid_Y': y,
            'Eccentricity': rp.eccentricity
        })
    main_df = pd.DataFrame(base_data)
    if main_df.empty: return

    dfs_to_merge = [main_df]

    # Organize definitions by cycle
    cycles_map = {}
    for d in image_defs:
        cycles_map.setdefault(d['cyc'], []).append(d)
    is_pheno_set = (set_name == 'Pheno')

    def tagged(df, prefix):                                          # name every feature column <feature>_<prefix>
        return df.rename(columns=lambda col: col if col == 'label_id' else f"{col}_{prefix}")

    # 2. Iterate Cycles
    sorted_cycles = sorted(cycles_map.keys())
    with tqdm(total=len(image_defs), unit="img", leave=False) as pbar:
        for c in sorted_cycles:
            loaded_imgs = {} # ch -> img
            cycle_defs = cycles_map[c]
            mode = cycle_defs[0]['mode'] 

            for d in cycle_defs:
                ch = d['ch']
                img, _ = load_aligned_image(output_dir, well, field, mode, c, ch)
                if img is not None:
                    loaded_imgs[ch] = img
            
            if not loaded_imgs: 
                pbar.update(len(cycle_defs))
                continue

            # Single-channel features
            for ch, img in loaded_imgs.items():
                rps = regionprops(labels, intensity_image=img)
                for df in (measure_intensity_advanced(rps, is_pheno=is_pheno_set), measure_texture_fast(rps), measure_granularity_fast(rps)):
                    if not df.empty: dfs_to_merge.append(tagged(df, f"{mode}_c{c}_ch{ch}"))
                pbar.update(1)

            # Correlations between the channels of one cycle (extras only for the phenotype set)
            for chA, chB in combinations(sorted(loaded_imgs), 2):
                df_corr = measure_pairwise_features(regionprops(labels, intensity_image=loaded_imgs[chA]), loaded_imgs[chB], include_extras=is_pheno_set)
                if not df_corr.empty: dfs_to_merge.append(tagged(df_corr, f"{mode}_c{c}_ch{chA}_v_ch{chB}"))


    # 3. Merge
    final_df = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        final_df = pd.merge(final_df, df, on='label_id', how='left')
    
    # 4. Metadata Injection
    uuids = [str(uuid.uuid4()) for _ in range(len(final_df))]
    final_df.insert(0, f'{obj_name}_UUID', uuids)
    
    # Add Location Metadata
    final_df.insert(0, 'Field', field)
    final_df.insert(0, 'Column', META_COL)
    final_df.insert(0, 'Row', META_ROW)
    final_df.insert(0, 'Well', well)
    final_df.insert(0, 'Plate', PLATE)
    
    if parent_info is not None:
        final_df = pd.merge(final_df, parent_info, left_on='label_id', right_on='spot_label', how='left')
        if 'spot_label' in final_df.columns: final_df.drop(columns=['spot_label'], inplace=True)
        
    out_name = f"{well}_s{field}_{set_name}_{obj_name}_Features.csv"
    final_df.to_csv(os.path.join(output_dir, out_name), index=False)


# ==============================================================================
#   MAIN EXECUTION
# ==============================================================================

try:
    model = StarDist2D(None, name=MODEL_NAME, basedir=MODEL_DIR)
    print(f"Loaded StarDist: {MODEL_NAME}")
except:
    model = None

# Define Image Sets
sbs_images = [{'mode': 'SBS', 'cyc': c, 'ch': ch} for c in range(1, SBS_CYCLES+1) for ch in oc.channels(cfg, 'SBS', c)]
pheno_images = [{'mode': 'Pheno', 'cyc': c, 'ch': ch} for c in range(1, PHENO_CYCLES+1) for ch in oc.channels(cfg, 'Pheno', c)]


for PLATE_INDEX in range(len(PLATES)):
    PLATE = PLATES[PLATE_INDEX]

    print(f"\nPLATE: {PLATE} | Well: {WELL} (Task {SGE_TASK_ID})")
    aligned_dir = os.path.join(ROOT_DIR, "Aligned", PLATE, WELL)
    
    if not os.path.exists(aligned_dir): continue

    for field in range(1, N_FIELDS + 1):
        print(f"\n--- Field s{field} ---")
        
        # --- 1. QC CHECK ---
        img_nuc, _ = load_aligned_image(aligned_dir, WELL, field, SEG_NUC_MODE, SEG_NUC_CYC, SEG_NUC_CH)
        if img_nuc is None:
            print("  Skipping: DAPI missing.")
            continue
            
        # Check Intensity
        norm_for_check = _normalize01(img_nuc)
        if np.max(norm_for_check) < MIN_DAPI_INTENSITY:
            print(f"  SKIPPING FIELD: Max DAPI {np.max(norm_for_check):.4f} < {MIN_DAPI_INTENSITY}")
            continue
            
        # --- 2. SEGMENTATION ---
        
        # A. Nuclei
        print("  [Seg] Nuclei...")
        norm_img = normalize(img_nuc)
        nuclei_labels, _ = model.predict_instances(norm_img, nms_thresh=NUC_NMS_THRESH, prob_thresh=NUC_PROB_THRESH)
        
        # [Exclude Border Nuclei]
        nuclei_labels = clear_border(nuclei_labels)
        print(f"  Nuclei Count (Border Cleared): {nuclei_labels.max()}")
        
        # B. Spots (SBS max projection)
        spot_labels, spot_relations = segment_spots(aligned_dir, WELL, field, nuclei_labels, model)
        
        # C. Whole cell / cytoplasm
        whole_cell_labels = None
        cytoplasm_labels = None
        img_cyto, _ = load_aligned_image(aligned_dir, WELL, field, SEG_CYTO_MODE, SEG_CYTO_CYC, SEG_CYTO_CH)
        
        if img_cyto is not None and nuclei_labels.max() > 0:
            whole_cell_labels = segment_whole_cell(nuclei_labels, img_cyto)
            cytoplasm_labels = whole_cell_labels.copy()
            cytoplasm_labels[nuclei_labels > 0] = 0
            
            # Save Masks
            imwrite(os.path.join(aligned_dir, f"{WELL}_s{field}_Nuclei.tif"), nuclei_labels.astype(np.uint16))
            if whole_cell_labels is not None:
                imwrite(os.path.join(aligned_dir, f"{WELL}_s{field}_WholeCell.tif"), whole_cell_labels.astype(np.uint16))
            if cytoplasm_labels is not None:
                imwrite(os.path.join(aligned_dir, f"{WELL}_s{field}_Cytoplasm.tif"), cytoplasm_labels.astype(np.uint16))
            if spot_labels is not None:
                imwrite(os.path.join(aligned_dir, f"{WELL}_s{field}_Spots.tif"), spot_labels.astype(np.uint16))

        # --- 3. FEATURE EXTRACTION ---
        
        # Output 1: SBS -> Nuclei + Spots
        if nuclei_labels.max() > 0:
            extract_set(nuclei_labels, "Nuclei", aligned_dir, WELL, field, sbs_images, "SBS")
        if spot_labels is not None and spot_labels.max() > 0:
            extract_set(spot_labels, "Spots", aligned_dir, WELL, field, sbs_images, "SBS", parent_info=spot_relations)
            
        # Output 2: Pheno -> Nuclei + WholeCell + Cytoplasm
        if nuclei_labels.max() > 0:
            extract_set(nuclei_labels, "Nuclei", aligned_dir, WELL, field, pheno_images, "Pheno")
        if whole_cell_labels is not None:
            extract_set(whole_cell_labels, "WholeCell", aligned_dir, WELL, field, pheno_images, "Pheno")
        if cytoplasm_labels is not None:
            extract_set(cytoplasm_labels, "Cytoplasm", aligned_dir, WELL, field, pheno_images, "Pheno")

    # --- 4. SIMPLE WELL AGGREGATION (Concatenate fields 1-4) ---
    patterns = [
        "SBS_Nuclei_Features", "SBS_Spots_Features",
        "Pheno_Nuclei_Features", "Pheno_WholeCell_Features", "Pheno_Cytoplasm_Features"
    ]
    
    for pat in patterns:
        file_list = glob.glob(os.path.join(aligned_dir, f"{WELL}_s*_{pat}.csv"))
        if file_list:
            dfs = []
            for fpath in file_list:
                try:
                    dfs.append(pd.read_csv(fpath))
                except: pass
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                final_out = os.path.join(aligned_dir, f"{WELL}_{pat}_Combined.csv")
                combined_df.to_csv(final_out, index=False)

print("\nProcessing Complete.")