import json
from pathlib import Path

def fix_dump_json(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"Skipping {p}, does not exist")
        return
        
    print(f"Fixing {p} ...")
    with open(p, "r") as f:
        data = json.load(f)
        
    for scene_id, scene_data in data.items():
        if "cameras" in scene_data and "robot_camera" in scene_data["cameras"]:
            cam = scene_data["cameras"]["robot_camera"]
            w, h = cam["width"], cam["height"]
            
            # Use same fallback logic as fixed ros1bag_to_dataset.py
            TARGET_SIZE = 512
            f_est = TARGET_SIZE * 0.85
            cx = TARGET_SIZE / 2.0
            cy = h / 2.0  # height is scaled already, so cy is half of actual scaled height
            
            old_params = cam["params"]
            new_params = [f_est, f_est, cx, cy]
            cam["params"] = new_params
            print(f"  Old params: {old_params}")
            print(f"  New params: {new_params}")
            
    with open(p, "w") as f:
        json.dump(data, f)
    print(f"Successfully updated {p}\n")

fix_dump_json("datasets/gmu_robot/dump.json")
fix_dump_json("datasets/cashor_robot/dump.json")
