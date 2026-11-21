import argparse
import os
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import spatiomic as so
from tqdm import tqdm

# ---------------------------------------------------------
# 0. Paths & Sample Definitions 
# ---------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--input-dir", type=Path, default=Path("."))
parser.add_argument("--output-dir", type=Path, default=None)
args, _ = parser.parse_known_args()

input_dir = args.input_dir.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_dir
output_dir.mkdir(parents=True, exist_ok=True)

def get_sample_images(prefix: str) -> list[str]:
    def sort_key(p: Path):
        suffix = p.stem.rsplit("_c", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, p.stem)

    return [p.stem for p in sorted(input_dir.glob(f"{prefix}_c*.tif"), key=sort_key)]


wt_1_images = get_sample_images("WT_1")
wt_2_images = get_sample_images("WT_2")
wt_3_images = get_sample_images("WT_3")

fs_1_images = get_sample_images("FS_1")
fs_2_images = get_sample_images("FS_2")
fs_3_images = get_sample_images("FS_3")

wt_images = wt_1_images + wt_2_images + wt_3_images
fs_images = fs_1_images + fs_2_images + fs_3_images
images = wt_images + fs_images

# ---------------------------------------------------------
# 1. Load Channels & Build Subsample
# ---------------------------------------------------------
channels_data = [
    str(ch).strip()
    for ch in pd.read_csv(input_dir / "channel_names.csv", header=0).T.to_numpy()[0]
]
channels = [ch for ch in channels_data if ch != "MUC1-fs"]


subsamples = []
first_shape = None

for image in tqdm(images, desc="Subsampling images"):
    data = so.data.read().read_tiff(
        str(input_dir / f"{image}.tif"),
        input_dimension_order="CYX",
    )

    assert data.shape[-1] == len(channels_data), (
        f"{image}: expected {len(channels_data)} channels, got shape {data.shape}"
    )

    data = so.data.subset(
        data,
        channel_names_data=channels_data,
        channel_names_subset=channels,
    )

    if first_shape is None:
        first_shape = data.shape

    subsample = so.data.subsample().fit_transform(
        data,
        method="count",
        count=40000,
        seed=0,
    )
    subsamples.append(subsample)

subsample = np.concatenate(subsamples, axis=0)
del subsamples
print(f"Subsampled {len(images)} images | First shape: {first_shape} | Subsample: {subsample.shape}")

# ---------------------------------------------------------
# 2. Clip & Normalize
# ---------------------------------------------------------
clipper = so.process.clip(
    method="percentile",
    percentile_min=50.0,  
    percentile_max=99.7,  
    use_gpu=True,
)

normalizer = so.process.normalize(
    min_value=0.0,
    max_value=1.0,
    use_gpu=True,
)

subsample_processed = clipper.fit_transform(subsample)
subsample_processed = normalizer.fit_transform(subsample_processed)
del subsample

for image_name in tqdm(images, desc="Normalizing images"):
    image = so.data.read().read_tiff(
        str(input_dir / f"{image_name}.tif"),
        input_dimension_order="CYX",
    )
    image = so.data.subset(
        image,
        channel_names_data=channels_data,
        channel_names_subset=channels,
    )
    image = clipper.transform(image)
    image = normalizer.transform(image)
    np.save(str(output_dir / f"{image_name}.npy"), image)

# ---------------------------------------------------------
# 3. Train SOM, Leiden Cluster & Label Pixels
# ---------------------------------------------------------
img_data_som = so.dimension.som(
    node_count=(500, 500),
    dimension_count=subsample_processed.shape[-1],
    distance_metric="euclidean",
    use_gpu=True,
    seed=0,
)

img_data_som.fit(
    subsample_processed,
    iteration_count=50,
)
img_data_som.save(str(output_dir / "example_map.p"))

knn_graph_som = so.neighbor.knn_graph().create(
    img_data_som,
    neighbor_count=12,
    use_gpu=True,
)

communities, modularity = so.cluster.leiden().predict(
    knn_graph_som,
    resolution=2.5,
    iteration_count=100,
    seed=42,
    use_gpu=True,
)

with open(output_dir / "modularity.txt", "w") as f:
    f.write(str(modularity))

for image_name in tqdm(images, desc="Labeling images"):
    image = np.load(str(output_dir / f"{image_name}.npy"))
    img_data_som.label(
        image,
        communities,
        save_path=str(output_dir / f"{image_name}-labelled.npy"),
    )



