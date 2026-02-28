
import json
from pathlib import Path

# Paths
dump_path = Path("/home/sameep/phd_research/osmloc/OSMLocConstrained/datasets/gmu_robot/dump.json")

def fix_dump():
    print(f"Loading {dump_path}...")
    with open(dump_path, 'r') as f:
        data = json.load(f)

    # 1. Update Camera Models
    cameras = data["0"]["cameras"]
    for cam_id, cam_data in cameras.items():
        print(f"Updating Camera {cam_id}...")
        print(f"  Old Width: {cam_data['width']}, Height: {cam_data['height']}")
        # Fix: Using full resolution 640x480
        cam_data["width"] = 640.0
        cam_data["height"] = 480.0
        # Check params: [fx, fy, cx, cy]
        # Current: [340, 341, 320, 240]
        # If cx=320, cy=240, and width=640, height=480, then center is correct.
        params = cam_data["params"]
        print(f"  Params: {params}")
        # Ensure params are floats
        cam_data["params"] = [float(x) for x in params]
        print(f"  New Width: {cam_data['width']}, Height: {cam_data['height']}")
    
    # Save back
    new_path = dump_path # Overwrite? Yes.
    # Backup first
    backup_path = dump_path.with_suffix(".json.bak_dims")
    if not backup_path.exists():
        print(f"Backing up to {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=4) # Backup valid json
    
    print(f"Saving fixed dump to {new_path}...")
    with open(new_path, 'w') as f:
        json.dump(data, f) # Minified save to save space/time? Or indented? Original was minified-ish?

    print("Done. Image dimensions updated to 640x480 to match actual files and intrinsics.")

if __name__ == "__main__":
    fix_dump()
