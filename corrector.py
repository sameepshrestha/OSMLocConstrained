import json
import numpy as np
from pathlib import Path
from maploc.utils.geo import Projection

# --- CONFIGURATION ---
# Use the same origin as your projection logic
LAT0, LON0 = 38.831337, -77.308682
proj = Projection(LAT0, LON0)

DUMP_PATH = Path("datasets/gmu_robot/dump.json")

# All frames in the range [Key_A : Key_B] will be set to exactly Key_A's values.
MANUAL_ANCHORS = {
    2450:    [38.828204, -77.306849, 90.0],
    3800:    [38.828165, -77.307045, 180.0],
    4500:    [38.828182, -77.307469, 0.0],
    5300:    [38.828251, -77.307759, 140.0]
}

def repair_path_stepwise():
    with open(DUMP_PATH, "r") as f:
        data = json.load(f)
    
    views = data["0"]["views"]
    all_frames = sorted([int(k) for k in views.keys()])
    anchor_frames = sorted(MANUAL_ANCHORS.keys())

    # We iterate through the segments defined by your anchors
    for i in range(len(anchor_frames)):
        start_f = anchor_frames[i]
        
        # Determine where this "block" ends
        if i < len(anchor_frames) - 1:
            end_f = anchor_frames[i+1]
        else:
            end_f = all_frames[-1] + 1 # Go till the very end of the bag
        
        # Values to apply to this entire block
        fixed_val = MANUAL_ANCHORS[start_f]
        lat, lon, yaw = fixed_val[0], fixed_val[1], fixed_val[2]
        
        # Pre-calculate the metric XY once for this block
        xy_metric = proj.project([lat, lon])
        
        # Apply these values to every frame in this range
        segment_frames = [f for f in all_frames if start_f <= f < end_f]
        
        for f_id in segment_frames:
            v = views[str(f_id)]
            v["latlong"] = [lat, lon]
            v["gps_position"] = [lat, lon, 0.0]
            v["t_c2w"] = [xy_metric[0], xy_metric[1], 0.0]  # Required for MapLoc to work
            v["roll_pitch_yaw"] = [0.0, 0.0, float(yaw)]

    # Save the repaired dump.json
    with open(DUMP_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Block-Repaired {all_frames[-1]} frames.")
    print(f"Frames 2450-3799 are now exactly at {MANUAL_ANCHORS[2450][:2]}")
    print(f"Frames 3800-4499 are now exactly at {MANUAL_ANCHORS[3800][:2]}")

if __name__ == "__main__":
    repair_path_stepwise()