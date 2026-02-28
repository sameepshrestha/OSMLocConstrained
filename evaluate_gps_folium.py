import pandas as pd
import folium
from folium import PolyLine, CircleMarker, FeatureGroup, LayerControl
from pathlib import Path

def plot_filtered_gps(csv_path, output_html, sample_rate=10, error_threshold=50.0):
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Assuming evaluate_results math approach
    err_x = df['pred_x'] - df['gt_x']
    err_y = df['pred_y'] - df['gt_y']
    df['error_m'] = (err_x**2 + err_y**2) ** 0.5
    
    # 1. Remove significant outliers 
    # (e.g. predictions that jump way beyond the prior bounding box map size)
    # User requested: "donot want to plot the outliers taht move way far and away from gt"
    df_filtered = df[df['error_m'] <= error_threshold]
    outliers_dropped = len(df) - len(df_filtered)
    print(f"Dropped {outliers_dropped} prediction outliers (> {error_threshold}m error)")

    # 2. Downsample
    # "sample like 2 from every 10hz during that plot" -> 2/10 = 1/5 of the data
    df_sampled = df_filtered.iloc[::5].copy()
    print(f"Sampling 1 in 5 frames. Total points to plot: {len(df_sampled)}")

    center = [df_sampled["gt_lat"].mean(), df_sampled["gt_lon"].mean()]
    m = folium.Map(location=center, zoom_start=20, max_zoom=22, tiles=None)
    
    tile_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    folium.TileLayer(tile_url, attr='Esri World Imagery', name="Satellite", max_zoom=22, max_native_zoom=19).add_to(m)
    
    gt_group = FeatureGroup(name="True GT Path", show=True).add_to(m)
    pred_group = FeatureGroup(name="Predicted Points", show=True).add_to(m)

    # Plot GT Path safely
    PolyLine(
        locations=list(zip(df_sampled["gt_lat"], df_sampled["gt_lon"])),
        color="green",
        weight=4,
        opacity=0.8,
        tooltip="Ground Truth Path"
    ).add_to(gt_group)

    # Plot valid predictions as markers with lines to GT
    for _, row in df_sampled.iterrows():
        pred_pos = (row["pred_lat"], row["pred_lon"])
        gt_pos = (row["gt_lat"], row["gt_lon"])
        
        PolyLine([gt_pos, pred_pos], color="white", weight=1, opacity=0.4).add_to(pred_group)
        
        CircleMarker(
            location=pred_pos,
            radius=3,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=1.0,
            tooltip=f"Error: {row['error_m']:.2f}m"
        ).add_to(pred_group)

    folium.LayerControl().add_to(m)
    
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved interactive map to {out_path}")

if __name__ == "__main__":
    import sys
    csv_in = sys.argv[1]
    html_out = sys.argv[2]
    plot_filtered_gps(csv_in, html_out)
