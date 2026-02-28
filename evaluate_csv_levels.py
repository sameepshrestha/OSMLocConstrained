import pandas as pd
import numpy as np
import sys
from pathlib import Path

def angle_error(t1, t2):
    diff = (t1 - t2) % 360
    # pandas vectorised version:
    out = np.minimum(diff, 360 - diff)
    return out

def evaluate(csv_path):
    print(f"\nEvaluating results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    n = len(df)
    if n == 0:
        print("Empty CSV")
        return

    # Compute errors manually from the predictions
    # The output CSV didn't store error_m_max directly, it stored coords.
    err_x = df['pred_x'] - df['gt_x']
    err_y = df['pred_y'] - df['gt_y']
    df['error_m_max'] = np.sqrt(err_x**2 + err_y**2)
    df['rot_err_max'] = angle_error(df['pred_yaw'], df['gt_yaw'])
    
    metrics = {}
    metrics['count'] = n
    
    # Location Recall
    metrics['recall_loc_1m'] = (df['error_m_max'] <= 1.0).mean() * 100
    metrics['recall_loc_2m'] = (df['error_m_max'] <= 2.0).mean() * 100
    metrics['recall_loc_3m'] = (df['error_m_max'] <= 3.0).mean() * 100
    metrics['recall_loc_5m'] = (df['error_m_max'] <= 5.0).mean() * 100
    metrics['recall_loc_10m'] = (df['error_m_max'] <= 10.0).mean() * 100
    
    # Orientation Recall
    metrics['recall_ang_1deg'] = (df['rot_err_max'] <= 1.0).mean() * 100
    metrics['recall_ang_5deg'] = (df['rot_err_max'] <= 5.0).mean() * 100
    metrics['recall_ang_10deg'] = (df['rot_err_max'] <= 10.0).mean() * 100
    metrics['recall_ang_20deg'] = (df['rot_err_max'] <= 20.0).mean() * 100
    
    # Statistics
    metrics['median_loc_error'] = df['error_m_max'].median()
    metrics['mean_loc_error'] = df['error_m_max'].mean()
    metrics['median_ang_error'] = df['rot_err_max'].median()
    metrics['mean_ang_error'] = df['rot_err_max'].mean()

    # Print Report
    print(f"Total Samples: {n}")
    print("Location Recall:")
    print(f"  @ 1m : {metrics['recall_loc_1m']:.2f}%")
    print(f"  @ 2m : {metrics['recall_loc_2m']:.2f}%")
    print(f"  @ 3m : {metrics['recall_loc_3m']:.2f}%")
    print(f"  @ 5m : {metrics['recall_loc_5m']:.2f}%")
    print(f"  @ 10m: {metrics['recall_loc_10m']:.2f}%")
    
    print("Angular Recall:")
    print(f"  @ 1° : {metrics['recall_ang_1deg']:.2f}%")
    print(f"  @ 5° : {metrics['recall_ang_5deg']:.2f}%")
    print(f"  @ 10°: {metrics['recall_ang_10deg']:.2f}%")
    print(f"  @ 20°: {metrics['recall_ang_20deg']:.2f}%")
    
    print("Error Statistics:")
    print(f"  Median Loc Error: {metrics['median_loc_error']:.2f} m")
    print(f"  Mean Loc Error  : {metrics['mean_loc_error']:.2f} m")
    print(f"  Median Ang Error: {metrics['median_ang_error']:.2f}°")
    print(f"  Mean Ang Error  : {metrics['mean_ang_error']:.2f}°")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cashor_robot")
    args = parser.parse_args()
    
    base_dir = Path(f"experiments/viz_{args.dataset}_levels")
    for level in ["10m", "20m", "30m"]:
        csv_path = base_dir / f"error_{level}" / "predictions.csv"
        evaluate(csv_path)
