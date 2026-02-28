
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import shutil

# Add MapItAnywhereLocalization to sys.path to allow imports
MIA_ROOT = Path("/home/sameep/phd_research/osmloc/MapItAnywhereLocalization")
if str(MIA_ROOT) not in sys.path:
    sys.path.append(str(MIA_ROOT))

from mapper.PerspectiveFields.perspective2d.perspectivefields import PerspectiveFields

def rectify_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    print(dataset_dir)
    dump_json = dataset_dir / "dump.json"
    dump_json_backup = dataset_dir / "dump.json.bak"
    image_dir = dataset_dir / "images"

    if not dump_json.exists():
        print(f"Error: {dump_json} not found.")
        return

    # Backup
    if not dump_json_backup.exists():
        shutil.copy(dump_json, dump_json_backup)
        print(f"Backed up dump.json to {dump_json_backup}")

    with open(dump_json, 'r') as f:
        data = json.load(f)

    # Initialize Model
    try:
        # Use GSV pretrained model
        pf_model = PerspectiveFields("PersNet_Paramnet-GSV-centered") 
        pf_model.eval()
        if torch.cuda.is_available():
            pf_model.cuda()
    except Exception as e:
        print(f"Failed to load PerspectiveFields: {e}")
        return

    print("Rectifying images using PerspectiveFields...")

    # Iterate scenes/sequences
    for scene_id, scene_data in data.items():
        views = scene_data.get("views", {})
        
        for view_id, view in tqdm(views.items(), desc=f"Scene {scene_id}"):
            img_path = image_dir / f"{view_id}.jpg"
            if not img_path.exists():
                 continue

            # Read Image (BGR)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Inference to get Pitch/Roll (Takes BGR)
            with torch.no_grad():
                pred = pf_model.inference(img)
            
            # Extract Roll/Pitch (Degrees)
            if 'pred_roll' in pred and 'pred_pitch' in pred:
                 # Ensure we get scalar
                 roll = pred['pred_roll'].item() if isinstance(pred['pred_roll'], torch.Tensor) else pred['pred_roll']
                 pitch = pred['pred_pitch'].item() if isinstance(pred['pred_pitch'], torch.Tensor) else pred['pred_pitch']
            else:
                 print(f"Warning: Unexpected keys in PF output: {pred.keys()}")
                 continue

            # Update Dump
            # view["roll_pitch_yaw"] is [roll, pitch, yaw]
            # Keep original Yaw
            original_rpy = view.get("roll_pitch_yaw", [0, 0, 0])
            original_yaw = original_rpy[2]
            
            view["roll_pitch_yaw"] = [roll, pitch, original_yaw]

    # Save
    with open(dump_json, 'w') as f:
        json.dump(data, f) # No indent to keep file size small? or indent for readable? 
                           # Original usually compact. Rostodataset used default (compact).

    print(f"Done! Updated dump.json with rectified Pitch/Roll.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?",
                        default="/home/sameep/phd_research/osmloc/OSMLocConstrained/datasets/cashor_robot",
                        help="Path to dataset directory containing dump.json and images/")
    args = parser.parse_args()
    rectify_dataset(args.dataset_dir)
