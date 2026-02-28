import pandas as pd
import json
from pathlib import Path

# --- CONFIGURE ---
dataset_root = Path("/home/sameep/phd_research/osmloc/OSMLocConstrained/datasets/MGL")
cities = [
    "amsterdam", "avignon", "berlin", "helsinki", "lemans",
    "milan", "montrouge", "nantes", "paris",
    "sanfrancisco_hayes", "sanfrancisco_soma", "toulouse", "vilnius"
]

for city in cities:
    city_dir = dataset_root / city
    if not city_dir.exists():
        print(f"Skipping {city} - folder not found")
        continue
    
    dump_json_fp = city_dir / "dump.json"
    if not dump_json_fp.exists():
        print(f"Skipping {city} - dump.json not found")
        continue
    
    output_parquet = city_dir / "image_metadata_filtered_processed.parquet"
    if output_parquet.exists():
        print(f"{city}: Parquet already exists - skipping (delete to regenerate)")
        continue
    
    print(f"Processing {city} from dump.json...")
    
    with open(dump_json_fp) as f:
        data = json.load(f)  # dict of outer panorama/sequence keys
    
    rows = []
    for outer_key, outer_dict in data.items():
        views = outer_dict.get("views", {})
        for view_key, view_dict in views.items():
            if "latlong" not in view_dict or len(view_dict["latlong"]) < 2:
                continue
            
            lat, lon = view_dict["latlong"]  # lat first in your data
            
            # Use yaw from roll_pitch_yaw as heading (most accurate for front view direction)
            rpy = view_dict.get("roll_pitch_yaw", [0, 0, 0])
            heading = float(rpy[2])
            
            # Fallback to compass_angle if yaw is 0/missing
            if heading == 0:
                heading = float(view_dict.get("compass_angle", 0.0))
            
            rows.append({
                "id": str(view_key),  # includes "_front" — matches your image filenames exactly
                "computed_geometry.lat": float(lat),
                "computed_geometry.long": float(lon),
                "computed_compass_angle": heading,
            })
    
    if not rows:
        print(f"{city}: No valid views found")
        continue
    
    df = pd.DataFrame(rows)
    print(f"{city}: Created DataFrame with {len(df)} front-view samples")
    
    # Optional: quick test subset
    # df = df.sample(frac=0.05, random_state=42)  # 5%
    
    df.to_parquet(output_parquet, index=False)
    print(f"{city}: Saved parquet to {output_parquet}\n")
