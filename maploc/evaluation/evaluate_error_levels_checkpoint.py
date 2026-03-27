import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Imports assuming running as python -m maploc.evaluation.evaluate_error_levels_checkpoint
from maploc.data.mapillary.dataset import MapillaryDataModule
from maploc.module import GenericModule
from maploc.data.torch import collate, unbatch_to_device
from maploc.utils.geo import Projection
from maploc.evaluation.viz import plot_example_single


def angle_error(t1, t2):
    """Compute angular error between two angles in degrees."""
    diff = (t1 - t2) % 360
    return min(diff, 360 - diff)


def build_eval_cfg(args):
    return OmegaConf.create(
        {
            "data": {
                "name": "mapillary",
                "scenes": [args.dataset],
                "split": None,
                "loading": {
                    "train": {"batch_size": 1, "num_workers": 0},
                    "val": {"batch_size": 1, "num_workers": args.num_workers},
                    "test": {"batch_size": 1, "num_workers": args.num_workers},
                },
                "max_init_error": args.error_levels[0],
                "add_map_mask": args.add_map_mask,
                "resize_image": args.resize_image,
                "pad_to_square": args.pad_to_square,
                "pad_to_multiple": args.pad_to_multiple,
                "rectify_pitch": args.rectify_pitch,
                "augmentation": {
                    "rot90": False,
                    "flip": False,
                    "image": {"apply": False},
                },
                "return_gps": True,
                "data_dir": str(args.data_dir),
            }
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/sameep/phd_research/osmloc/OSMLocConstrained/datasets"),
    )
    parser.add_argument(
        "--base_output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to experiments/viz_{dataset}_levels_checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gmu_robot",
        help="Dataset/scene folder name inside data_dir",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--start_idx", type=int, default=0, help="Skip first N images")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images after start_idx")
    parser.add_argument("--error-levels", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resize-image", type=int, default=None)
    parser.add_argument("--pad-to-multiple", type=int, default=32)
    parser.add_argument(
        "--projection-origin",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        default=(38.831337, -77.308682),
    )
    parser.add_argument("--add-map-mask", action="store_true")
    parser.add_argument("--pad-to-square", action="store_true")
    parser.add_argument("--rectify-pitch", action="store_true", default=True)
    parser.add_argument("--no-rectify-pitch", dest="rectify_pitch", action="store_false")
    parser.add_argument("--viz", action="store_true", help="Save visualizations")
    args = parser.parse_args()

    if args.base_output_dir is None:
        args.base_output_dir = Path(f"experiments/viz_{args.dataset}_levels_checkpoint")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    cfg_override = build_eval_cfg(args)
    lat0, lon0 = args.projection_origin
    proj = Projection(lat0, lon0)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = GenericModule.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        cfg=cfg_override,
        find_best=False,
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    cfg = model.cfg
    print(f"Loaded model: {cfg.model.name}")

    print("Setting up dataset...")
    dm = MapillaryDataModule(cfg.data)
    dm.prepare_data()
    dm.setup(args.split)
    dataset = dm.dataset(args.split)
    print(f"Total dataset size: {len(dataset)}")

    start_idx = args.start_idx
    if len(dataset) <= start_idx:
        print(f"Dataset too small ({len(dataset)}) to skip {start_idx} images.")
        return

    end_idx = len(dataset)
    if args.limit:
        end_idx = min(start_idx + args.limit, len(dataset))

    indices = range(start_idx, end_idx)
    print(f"Processing {len(indices)} images from index {start_idx} to {end_idx}")

    output_dirs = {}
    for level in args.error_levels:
        lvl_dir = args.base_output_dir / f"error_{level}m"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[level] = lvl_dir

    csv_paths = {level: output_dirs[level] / "predictions.csv" for level in args.error_levels}
    writers = {}
    csv_files = {}

    csv_header = [
        "name", "method",
        "prior_lat", "prior_lon",
        "gt_lat", "gt_lon",
        "pred_lat", "pred_lon",
        "pred_x", "pred_y", "pred_yaw",
        "gt_x", "gt_y", "gt_yaw",
        "error_level_m",
    ]

    for level, path in csv_paths.items():
        mode = "a" if path.exists() else "w"
        f_handle = open(path, mode, newline="")
        writer = csv.writer(f_handle)
        if mode == "w":
            writer.writerow(csv_header)
        writers[level] = writer
        csv_files[level] = f_handle

    try:
        for i in tqdm(indices):
            sample_name = dataset.names[i][-1]

            for level in args.error_levels:
                dataset.cfg.max_init_error = level
                sample = dataset[i]
                batch = collate([sample])
                batch = model.transfer_batch_to_device(batch, model.device, 0)

                with torch.no_grad():
                    pred = model(batch)

                uv_pred = pred["uv_max"][0]
                yaw_pred = pred["yaw_max"][0].item()
                uv_gt = batch["uv"][0]
                yaw_gt = batch["roll_pitch_yaw"][0][-1].item()
                canvas = batch["canvas"][0]

                pred_xy = canvas.to_xy(uv_pred.double()).cpu().numpy()
                gt_xy = canvas.to_xy(uv_gt.double()).cpu().numpy()

                err_x = pred_xy[0] - gt_xy[0]
                err_y = pred_xy[1] - gt_xy[1]
                xy_err_m = np.sqrt(err_x**2 + err_y**2)
                yaw_err_deg = angle_error(yaw_pred, yaw_gt)

                if "uv_gps" in batch and batch["uv_gps"] is not None:
                    uv_prior = batch["uv_gps"][0]
                    prior_xy = canvas.to_xy(uv_prior.double()).cpu().numpy()
                else:
                    prior_xy = np.array([0.0, 0.0])

                global_gt_metric = dataset.data["t_c2w"][i][:2].numpy()
                shift = global_gt_metric - gt_xy
                global_pred_metric = pred_xy + shift
                global_prior_metric = prior_xy + shift

                pred_latlon = proj.unproject(global_pred_metric)
                gt_latlon = proj.unproject(global_gt_metric)
                prior_latlon = proj.unproject(global_prior_metric)

                writers[level].writerow(
                    [
                        sample_name,
                        "single_frame",
                        prior_latlon[0],
                        prior_latlon[1],
                        gt_latlon[0],
                        gt_latlon[1],
                        pred_latlon[0],
                        pred_latlon[1],
                        pred_xy[0],
                        pred_xy[1],
                        yaw_pred,
                        gt_xy[0],
                        gt_xy[1],
                        yaw_gt,
                        level,
                    ]
                )

                if args.viz:
                    results = {
                        "xy_max_error": xy_err_m,
                        "yaw_max_error": yaw_err_deg,
                        "xy_gps_error": np.linalg.norm(prior_xy - gt_xy) if "uv_gps" in batch else 0.0,
                    }
                    pred_unbatched = unbatch_to_device(pred)
                    batch_unbatched = unbatch_to_device(batch)
                    try:
                        if "depth" not in pred_unbatched:
                            h, w = batch_unbatched["image"].shape[-2:]
                            pred_unbatched["depth"] = torch.zeros(
                                (1, h, w), device=pred_unbatched["uv_max"].device
                            )

                        plot_example_single(
                            i,
                            model,
                            pred_unbatched,
                            batch_unbatched,
                            results,
                            out_dir=output_dirs[level],
                            show_gps=True,
                        )
                    except Exception as exc:
                        print(f"Viz Error {sample_name}: {exc}")
                        import traceback
                        traceback.print_exc()

                del batch, pred, sample
                torch.cuda.empty_cache()

            for f_handle in csv_files.values():
                f_handle.flush()
    finally:
        for f_handle in csv_files.values():
            f_handle.close()

    print("Checkpoint evaluation complete. Results saved with visualizations.")


if __name__ == "__main__":
    main()
