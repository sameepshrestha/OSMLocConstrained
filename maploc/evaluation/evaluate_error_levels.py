import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Imports assuming running as python -m maploc.evaluation.evaluate_error_levels
from maploc.data.mapillary.dataset import MapillaryDataModule
from maploc.module import GenericModule 
from maploc.data.torch import collate, unbatch_to_device
from maploc.utils.geo import Projection, BoundaryBox
from maploc.evaluation.viz import plot_example_single # Added Viz import


def angle_error(t1, t2):
    """Compute angular error between two angles in degrees."""
    diff = (t1 - t2) % 360
    return min(diff, 360 - diff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=Path, default=Path("/home/sameep/phd_research/osmloc/OSMLocConstrained/datasets"))
    parser.add_argument("--base_output_dir", type=Path, default=None,
                        help="Output directory. Defaults to experiments/viz_{dataset}_levels")
    parser.add_argument("--dataset", type=str, default="gmu_robot",
                        help="Dataset/scene folder name inside data_dir")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Skip first N images")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images after start_idx")
    parser.add_argument("--viz", action="store_true", help="Save visualizations")
    args = parser.parse_args()

    # Auto-derive output dir from dataset name if not provided
    if args.base_output_dir is None:
        args.base_output_dir = Path(f"experiments/viz_{args.dataset}_levels")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Config setup
    cfg = OmegaConf.create({
        "data": {
            "name": "mapillary",
            "scenes": [args.dataset],
            "split": None, # Load all
            "loading": {
                "train": {"batch_size": 1, "num_workers": 0},
                "val": {"batch_size": 1, "num_workers": 0},
                "test": {"batch_size": 1, "num_workers": 0}
            },
            "num_classes": {
                "areas": 7,
                "ways": 10,
                "nodes": 33
            },
            "pixel_per_meter": 2,
            "crop_size_meters": 64,  # Large enough to cover predictions
            "max_init_error": 10,   # Initial placeholder
            "add_map_mask": False,
            # Images are already 512xH from extraction; don't force-square them.
            # fn=None + int would squish to 512x512; pad_to_square adds black rows.
            # Instead: let dataset.py pad to nearest multiple-of-32 (the default).
            "resize_image": None,
            "pad_to_square": False,
            "pad_to_multiple": 32,
            "rectify_pitch": True, # Rectify using roll/pitch from dump.json
            "augmentation": {
                "rot90": False,
                "flip": False,
                "image": {"apply": True}
            },
            "return_gps": True,
            "data_dir": str(args.data_dir)
        }
    })

    # Initialize Projection (Use Lat/Lon from rostodataset.py)
    lat0, lon0 = 38.831337, -77.308682
    proj = Projection(lat0, lon0)

    # Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = GenericModule.load_from_checkpoint(args.checkpoint, strict=False, cfg=cfg, find_best=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    # Load Dataset
    print("Setting up dataset...")
    dm = MapillaryDataModule(cfg.data)
    dm.prepare_data()
    dm.setup('val')
    dataset = dm.dataset('val')
    print(f"Total dataset size: {len(dataset)}")

    # Indices to process
    start_idx = args.start_idx
    if len(dataset) <= start_idx:
        print(f"Dataset too small ({len(dataset)}) to skip {start_idx} images.")
        return

    end_idx = len(dataset)
    if args.limit:
        end_idx = min(start_idx + args.limit, len(dataset))
    
    indices = range(start_idx, end_idx)
    print(f"Processing {len(indices)} images from index {start_idx} to {end_idx}")

    # Output Setup
    # Ensure directories exist
    error_levels = [10, 20, 30]
    output_dirs = {}
    for level in error_levels:
        lvl_dir = args.base_output_dir / f"error_{level}m"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[level] = lvl_dir
    
    csv_paths = {level: output_dirs[level] / "predictions.csv" for level in error_levels}
    writers = {}
    csv_files = {}

    csv_header = [
        "name", "method", 
        "prior_lat", "prior_lon", 
        "gt_lat", "gt_lon", 
        "pred_lat", "pred_lon",
        "pred_x", "pred_y", "pred_yaw",
        "gt_x", "gt_y", "gt_yaw",
        "error_level_m"
    ]

    # Open CSVs
    for level, path in csv_paths.items():
        mode = 'a' if path.exists() else 'w'
        f = open(path, mode, newline='')
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(csv_header)
        writers[level] = writer
        csv_files[level] = f

    # Loop
    try:
        for i in tqdm(indices):
            if i <= 1377:
                continue
            sample_name = dataset.names[i][-1] # Get name reliably
            
            for level in error_levels:
                # 1. Dynamically set Error Level
                dataset.cfg.max_init_error = level
                
                # 2. Get Sample
                sample = dataset[i]
                
                # 3. Collate & Transfer
                batch = collate([sample])
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                
                # 4. Inference
                with torch.no_grad():
                    pred = model(batch)
                
                # 5. Extract Results (Metric)
                uv_pred = pred['uv_max'][0] # Pixels
                yaw_pred = pred['yaw_max'][0].item() # Degrees
                
                uv_gt = batch['uv'][0] # Pixels (GT)
                yaw_gt = batch['roll_pitch_yaw'][0][-1].item() # Degrees
                # yaw_gt is loaded from dataset as the GT heading relative to map or whatever
                # batch['roll_pitch_yaw'] is usually [roll, pitch, yaw]
                
                canvas = batch['canvas'][0] # List of objects, take first
                
                # Pixels -> XY (Metric)
                pred_xy = canvas.to_xy(uv_pred.double()).cpu().numpy()
                gt_xy = canvas.to_xy(uv_gt.double()).cpu().numpy()
                
                # Errors for Viz & CSV
                err_x = pred_xy[0] - gt_xy[0]
                err_y = pred_xy[1] - gt_xy[1]
                xy_err_m = np.sqrt(err_x**2 + err_y**2)
                yaw_err_deg = angle_error(yaw_pred, yaw_gt)

                # Prior
                if 'uv_gps' in batch and batch['uv_gps'] is not None:
                     uv_prior = batch['uv_gps'][0]
                     prior_xy = canvas.to_xy(uv_prior.double()).cpu().numpy()
                else: 
                     prior_xy = np.array([0.0, 0.0])

                # 6. Unproject to Lat/Lon
                global_gt_metric = dataset.data["t_c2w"][i][:2].numpy() # Global GT
                shift = global_gt_metric - gt_xy
                
                global_pred_metric = pred_xy + shift
                global_prior_metric = prior_xy + shift
                
                # Now Unproject Global Metric -> Lat/Lon
                pred_latlon = proj.unproject(global_pred_metric)
                gt_latlon = proj.unproject(global_gt_metric)
                prior_latlon = proj.unproject(global_prior_metric)
                
                # Write to CSV
                row = [
                    sample_name, "single_frame",
                    prior_latlon[0], prior_latlon[1],
                    gt_latlon[0], gt_latlon[1],
                    pred_latlon[0], pred_latlon[1],
                    pred_xy[0], pred_xy[1], yaw_pred,
                    gt_xy[0], gt_xy[1], yaw_gt,
                    level
                ]
                writers[level].writerow(row)

                # 7. Visualize (if requested)
                # Reconstruct 'results' dict expected by plot_example_single
                # It expects simple scalars usually
                results = {
                    "xy_max_error": xy_err_m,
                    "yaw_max_error": yaw_err_deg,
                    "xy_gps_error": np.linalg.norm(prior_xy - gt_xy) if 'uv_gps' in batch else 0.0 # Just for display
                }
                
                # Batch needs pass unbatched (dict of tensors, usually squeeze 0) to plot_example_single?
                # Actually plot_example_single takes 'pred' and 'data' (batch). 
                # BUT 'pred' produced by model is batched tensors [B, ...].
                # 'batch' is also batched [B, ...].
                # plot_example_single usually takes ONE sample (idx).
                # Wait, run.py calls:
                # plot_example_single(i, model, unbatch_to_device(pred), unbatch_to_device(batch['cuda']??), results, out_dir=...)
                # Need to unbatch.
                
                pred_unbatched = unbatch_to_device(pred)
                batch_unbatched = unbatch_to_device(batch)
                
                # Need to run visualization on CPU usually
                # pred_unbatched elements are usually tensors on GPU unless moved.
                # plot_example_single handles .cpu().numpy() internally mostly.
                # But unbatch_to_device returns list of dicts if batch size > 1.
                # We have batch_size=1, so unbatch returns [sample_dict].
                # We need sample_dict.
                
                if args.viz:
                    try:
                        # Need to pass ONE dict, not list
                        output_dir_level = output_dirs[level]
                        # Fix: Inject dummy depth if missing (Loca model doesn't predict it, but viz expects it)
                        if "depth" not in pred_unbatched:
                             # Access image from data to get shape
                             h, w = batch_unbatched["image"].shape[-2:]
                             pred_unbatched["depth"] = torch.zeros((1, h, w), device=pred_unbatched["uv_max"].device)

                        plot_example_single(
                            i,
                            model, # Pass model wrapper? Or just for config access
                            pred_unbatched, # Take first/only element
                            batch_unbatched,
                            results,
                            out_dir=output_dir_level,
                            show_gps=True
                        )
                    except Exception as e:
                        print(f"Viz Error {sample_name}: {e}")
                        import traceback
                        traceback.print_exc()

                # Cleanup
                del batch, pred, sample
                torch.cuda.empty_cache()
            
            # Flush periodically
            for f_handle in csv_files.values():
                f_handle.flush()
    
    finally:
        for f_handle in csv_files.values():
            f_handle.close()

    print("Evaluation complete. Results saved with visualizations.")


if __name__ == "__main__":
    main()
