#!/usr/bin/env python
# coding: utf-8
"""Align every SBS and phenotype cycle of one well to the DAPI of a reference cycle (palom: coarse affine + block-wise affine), per plate, and write
<Well>_s<site>_<SBS|Pheno>_Cy<k>_w<ch>_aligned.tiff, a DAPI QC stack and <Well>_concerning_cycles.csv (cycles whose DAPI correlation with the reference is low).
One array task = one well (all plates, all fields): qsub -t 1-<n_wells> ... python align.py        (task number: $SGE_TASK_ID or first argument)
All experiment-specific values (cycle counts, folder names, channels, thresholds) are read from config.yaml (see ops_config.py)."""
import glob
import os
import shutil
import sys

import dask.array as da
import numpy as np
import palom
import pandas as pd
from skimage import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ops_config as oc

cfg = oc.load()
oc.require(cfg, 'root_dir', 'experiment', 'sbs.n_cycles', 'phenotype.n_cycles', 'sbs.reference_cycle', 'imaging.fields_per_well', 'imaging.channels_per_cycle', 'imaging.pixel_size_um')
ROOT_DIR = cfg['root_dir']
SBS_CYCLES, PHENO_CYCLES = cfg['sbs']['n_cycles'], cfg['phenotype']['n_cycles']
REF_CYCLE = cfg['sbs']['reference_cycle']
CORRELATION_THRESHOLD = cfg['alignment']['correlation_threshold']
FIELDS = range(1, cfg['imaging']['fields_per_well'] + 1)
PIXEL_SIZE_UM = cfg['imaging']['pixel_size_um']

task = oc.task_id()
PLATES = oc.plates(cfg)
WELL = oc.wells(cfg)[task - 1]                                         # one task = one well, all plates and fields
print(f"Task {task} | well {WELL}")


def first_match(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def cycle_folder(base, kind, cycle, plate):
    return first_match(os.path.join(base, cfg[kind]['folder_pattern'].format(experiment=cfg['experiment'], plate=plate, cycle=cycle)))


def read_image(path):
    return palom.reader.OmePyramidReader(path).read_level_channels(0, 0)


def fit_affine(ref_full, ref_thumb, mov_full, mov_thumb):
    """palom coarse + block-wise alignment; if it fails, retry with the bright tail of the moving image clipped. Returns the aligner or None."""
    dtype = mov_thumb.dtype
    for q in (None, 99, 95):
        if q is None:
            thumb, full = mov_thumb, mov_full
        else:
            limit = np.percentile(mov_thumb, q)
            thumb, full = np.clip(mov_thumb, 0, limit).astype(dtype), da.clip(mov_full, 0, limit).astype(dtype)
        try:
            aligner = palom.align.Aligner(ref_img=ref_full, moving_img=full, ref_thumbnail=ref_thumb, moving_thumbnail=thumb,
                                          ref_thumbnail_down_factor=1.0, moving_thumbnail_down_factor=1.0)
            aligner.coarse_register_affine(n_keypoints=5000)
            aligner.compute_shifts()
            if not np.any(np.isnan(aligner.affine_matrix)):
                print(f"    Aligned at Q={q}")
                return aligner
        except Exception:
            pass
    return None


def process_plate(plate, index):
    print(f"\n{'=' * 60}\nPLATE {index + 1}/{len(PLATES)}: {plate} | well {WELL}\n{'=' * 60}")
    bases = {'sbs': oc.fmt(cfg, cfg['sbs']['input_dir'], plate=plate), 'phenotype': oc.fmt(cfg, cfg['phenotype']['input_dir'], plate=plate)}
    output = os.path.join(ROOT_DIR, 'Aligned', plate, WELL)
    os.makedirs(output, exist_ok=True)

    ref_folder = cycle_folder(bases['sbs'], 'sbs', REF_CYCLE, plate)          # DAPI of this cycle is the reference of every alignment
    if ref_folder is None:
        print(f"CRITICAL: no folder for the reference SBS cycle of plate {plate}. Skipping plate.")
        return

    dapi = {f: [None] * (SBS_CYCLES + PHENO_CYCLES) for f in FIELDS}          # DAPI of every cycle (SBS first), for the QC stack
    shape = {f: None for f in FIELDS}
    concerning = []

    def process_cycle(cycle, mode, field, folder, layer):
        channels = oc.channels(cfg, mode, cycle)
        ref_path = first_match(os.path.join(ref_folder, f"{WELL}_s{field}_w1_z0_*.tiff"))
        mov_path = first_match(os.path.join(folder, f"{WELL}_s{field}_w1_z0_*.tiff"))
        if ref_path is None or mov_path is None:
            print(f"    Skipping: images missing for {mode} Cy{cycle} field {field}")
            return

        ref_full = read_image(ref_path)
        if shape[field] is None:
            shape[field] = ref_full.shape

        if mode == 'SBS' and cycle == REF_CYCLE:                              # the reference cycle is copied, not aligned
            print("    Reference cycle (copying files)...")
            dapi[field][layer] = ref_full
            for ch in channels:
                src = first_match(os.path.join(ref_folder, f"{WELL}_s{field}_w{ch}_z0_*.tiff"))
                dst = os.path.join(output, f"{WELL}_s{field}_SBS_Cy{REF_CYCLE}_w{ch}_aligned.tiff")
                if src and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            return

        mov_full = read_image(mov_path)
        aligner = fit_affine(ref_full, ref_full.compute(), mov_full, mov_full.compute())
        if aligner is None:
            print(f"    FAILED: {mode} Cy{cycle} field {field} failed alignment.")
            return

        for ch in channels:
            path = first_match(os.path.join(folder, f"{WELL}_s{field}_w{ch}_z0_*.tiff"))
            if path is None:
                continue
            warped = palom.align.block_affine_transformed_moving_img(ref_img=ref_full, moving_img=read_image(path), mxs=aligner.block_affine_matrices_da, is_mask=False)
            if ch == 1:
                dapi[field][layer] = warped
            io.imsave(os.path.join(output, f"{WELL}_s{field}_{mode}_Cy{cycle}_w{ch}_aligned.tiff"), warped, check_contrast=False)

    for mode, kind, n_cycles, first_layer in (('SBS', 'sbs', SBS_CYCLES, 0), ('Pheno', 'phenotype', PHENO_CYCLES, SBS_CYCLES)):
        for cycle in range(1, n_cycles + 1):
            print(f"\n--- {mode} cycle {cycle} ---")
            folder = cycle_folder(bases[kind], kind, cycle, plate)
            if folder is None:
                continue
            for field in FIELDS:
                print(f"  Field {field}...")
                process_cycle(cycle, mode, field, folder, first_layer + cycle - 1)

    # QC: DAPI of every cycle against the reference (every 10th pixel), and the QC stack
    print("\n--- QC stacks & correlation ---")
    for field in FIELDS:
        print(f"  -- field s{field} --")
        layers, target_shape = dapi[field], shape[field]
        if target_shape is None:
            continue
        if layers[REF_CYCLE - 1] is None:
            print("    Reference DAPI missing; cannot calculate correlations.")
            continue
        ref_flat = layers[REF_CYCLE - 1][::10, ::10].compute().flatten()

        stack = []
        for i, layer in enumerate(layers):
            if layer is None:
                stack.append(da.zeros(target_shape, dtype=np.uint16))
                continue
            stack.append(layer)
            if i == REF_CYCLE - 1:
                continue
            try:
                corr = np.corrcoef(ref_flat, layer[::10, ::10].compute().flatten())[0, 1]
                name = f"SBS_Cy{i + 1}" if i < SBS_CYCLES else f"Pheno_Cy{i - SBS_CYCLES + 1}"
                print(f"    {name}: {corr:.3f}")
                if corr < CORRELATION_THRESHOLD:
                    concerning.append({'Plate': plate, 'Well': WELL, 'Field': field, 'Cycle': name, 'Correlation': corr})
            except Exception:
                pass

        palom.pyramid.write_pyramid(mosaics=[da.stack(stack, axis=0)], output_path=os.path.join(output, f"{WELL}_s{field}_QC_Combined_Stack.ome.tif"),
                                    pixel_size=PIXEL_SIZE_UM, downscale_factor=2, compression="lzw", save_RAM=True, verbose=False)
        print("    Saved QC stack")

    if concerning:
        pd.DataFrame(concerning).to_csv(os.path.join(output, f"{WELL}_concerning_cycles.csv"), index=False)
        print(f"    WARNING: {len(concerning)} cycles with a correlation below {CORRELATION_THRESHOLD}.")


for index, plate in enumerate(PLATES):
    process_plate(plate, index)
print("\nAll plates complete.")
