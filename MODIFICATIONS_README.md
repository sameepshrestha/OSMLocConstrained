# OSMLoc Repository Modifications

This document outlines the modifications made to the original OSMLoc repository, including which files were modified to support our custom datasets and what new scripts were added to facilitate data preprocessing, training, and evaluation.

## 1. Added Files (Not In Original Repo)

These files were newly introduced to process custom datasets, parse ROS1 bags, visualize outputs, and calculate specific evaluation metrics:

### Dataset Extraction & Preprocessing
* `ros1bag_to_dataset.py`: Extracts data straight from ROS1 bag files (pure Python parsing without ROS setup). Handles images, camera intrinsics, GPS, and IMU data, rectifying images and formatting them into the `OSMLoc` dataset structure (e.g., `dump.json`).
* `rectify_dataset_perspective.py`: Applies a learned perspective model to address cropping and incorrect horizons in images collected with uncalibrated IMU data.
* `fix_cameras.py` & `fix_dump_dims.py`: Utilities to patch camera intrinstics and image dimensions in the extracted dataset.
* `inspect_bag.py`: Utility script to inspect the topics and contents of ROS1 bag files.

### Evaluation & Metrics Analysis
* `maploc/evaluation/evaluate_error_levels.py` / `evaluate_csv_levels.py`: Custom evaluation script that calculates location recall percentages (at 1m, 2m, 3m, 5m, 10m) and angular recall percentages (at 1°, 5°, 20°). Used for analyzing inference pipeline results.
* `maploc/evaluate_results.py` & `maploc/evaluate_results_new.py`: Helper evaluation scripts designed to crunch localization models' output, analyze various error thresholds, and save analytical statistics.
* `maploc/evaluation/evaluate_ranges.py`: Evaluates performance specifically within different bounding ranges.
* `maploc/semantic_similarity.py` & `maploc/get_mia_bev.py`: Utilities to extract BEV (Bird's Eye View) maps and evaluate semantic consistency.

### Visualizations
* `evaluate_gps_folium.py` & `maploc/evaluate_gps_folium.py`: Scripts leveraging Folium to dump predicted vs. Ground Truth GPS coordinates as HTML maps (for qualitative assessment).
* `maploc/data_plot.py`: Plots and visualizations for dataset exploration.

## 2. Modified Original Files

These tracked files originating from the official OSMLoc repository were modified to support our environment, masks, and training pipeline:

### Data & Configuration changes
* `maploc/conf/data/mapillary_mgl.yaml` & `maploc/conf/osmloc_base.yaml`: Updated data and base configurations for the OSMLoc pipelines.
* `maploc/data/dataset.py`: Logic modified to accommodate semantic mask loading alongside conventional image data.
* `maploc/data/mapillary/dataset.py`: Tailored configurations for loading mapillary data alongside local test data.

### Core Model Tweaks
* `maploc/models/voting.py`:
  * Added an $\epsilon$ (`+ .000001`) in `depth_loss` to prevent hard division-by-zero crashes.
  * Patched slice casting in `conv2d_fft_batchwise` (e.g., using `tuple()`) to work flawlessly with modern PyTorch.
* `maploc/models/osmloc.py` & `maploc/models/mia_osmloc.py`: Updated logic interfacing with the semantic changes and base model structures.
* `maploc/models/depth_anything/dpt.py`: Adjusted generic model imports/calls slightly.

### Evaluation & Inference Alterations
* `maploc/evaluation/run.py`: General changes allowing smooth evaluation passes over custom datasets without erroring out on expected variables. 
* `maploc/evaluation/mapillary.py`: Minor path patching for predictions context.
* `maploc/evaluation/viz.py` & `maploc/utils/viz_localization.py`: Patched the `copy_image` method to gracefully handle Matplotlib updates. In older versions `.pop()` threw KeyErrors if properties weren't strictly present (e.g. `children`, `size`, `tightbbox`). Now appropriately uses `.pop(k, None)`.

### Training Pipeline
* `maploc/train.py`: 
  * Implemented PyTorch 2.6+ fixes resolving the `Weights only load failed` exception associated with OmegaConf/Hydra loading mechanisms.
  * Added initialization for `CometLogger` allowing comprehensive ML run tracking directly to a custom Workspace (`sameep54` / `mia-osmloc`), complete with timestamp-based experiment naming.

## Note on Test Scripts
Various `test_*.py`, `debug_*.py`, and `cleanup_*.py` files that were generated during rapid iterative development and testing have been removed to keep the repository clean.
