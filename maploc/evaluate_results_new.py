import pandas as pd
import numpy as np
import sys
from pathlib import Path

def evaluate(csv_path):
    print(f"\n--- Evaluating results from {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    n = len(df)
    if n == 0:
        print("Empty CSV")
        return

    # Calculate error manually from true metric coordinates produced by evaluate_error_levels.py
    err_x = df['pred_x'] - df['gt_x']
    err_y = df['pred_y'] - df['gt_y']
    df['error_m'] = np.sqrt(err_x**2 + err_y**2)
    
    yaw_diff = (df['pred_yaw'] - df['gt_yaw']) % 360
    df['rot_err'] = np.minimum(yaw_diff, 360 - yaw_diff)

    metrics = {}
    metrics['count'] = n
    
    # Location Recall 
    metrics['recall_loc_1m'] = (df['error_m'] <= 1.0).mean() * 100
    metrics['recall_loc_2m'] = (df['error_m'] <= 2.0).mean() * 100
    metrics['recall_loc_3m'] = (df['error_m'] <= 3.0).mean() * 100
    metrics['recall_loc_5m'] = (df['error_m'] <= 5.0).mean() * 100
    metrics['recall_loc_10m'] = (df['error_m'] <= 10.0).mean() * 100
    
    # Orientation Recall 
    metrics['recall_ang_1deg'] = (df['rot_err'] <= 1.0).mean() * 100
    metrics['recall_ang_5deg'] = (df['rot_err'] <= 5.0).mean() * 100
    metrics['recall_ang_10deg'] = (df['rot_err'] <= 10.0).mean() * 100
    metrics['recall_ang_20deg'] = (df['rot_err'] <= 20.0).mean() * 100
    
    # Joint Recall 
    metrics['recall_3m_10deg'] = ((df['error_m'] <= 3.0) & (df['rot_err'] <= 10.0)).mean() * 100
    metrics['recall_5m_10deg'] = ((df['error_m'] <= 5.0) & (df['rot_err'] <= 10.0)).mean() * 100
    metrics['recall_10m_20deg'] = ((df['error_m'] <= 10.0) & (df['rot_err'] <= 20.0)).mean() * 100
    
    # Statistics
    metrics['median_loc_error'] = df['error_m'].median()
    metrics['mean_loc_error'] = df['error_m'].mean()
    metrics['median_ang_error'] = df['rot_err'].median()
    metrics['mean_ang_error'] = df['rot_err'].mean()

    # Print Report
    print(f"Total Samples Processed: {n}")
    print("\nLocation Recall:")
    print(f"  @ 1m : {metrics['recall_loc_1m']:.2f}%")
    print(f"  @ 2m : {metrics['recall_loc_2m']:.2f}%")
    print(f"  @ 3m : {metrics['recall_loc_3m']:.2f}%")
    print(f"  @ 5m : {metrics['recall_loc_5m']:.2f}%")
    print(f"  @ 10m: {metrics['recall_loc_10m']:.2f}%")
    
    print("\nAngular Recall:")
    print(f"  @ 1° : {metrics['recall_ang_1deg']:.2f}%")
    print(f"  @ 5° : {metrics['recall_ang_5deg']:.2f}%")
    print(f"  @ 10°: {metrics['recall_ang_10deg']:.2f}%")
    print(f"  @ 20°: {metrics['recall_ang_20deg']:.2f}%")
    
    print("\nJoint Recall:")
    print(f"  (3m, 10°): {metrics['recall_3m_10deg']:.2f}%")
    print(f"  (5m, 10°): {metrics['recall_5m_10deg']:.2f}%")
    print(f"  (10m, 20°): {metrics['recall_10m_20deg']:.2f}%")
    
    print("\nError Statistics:")
    print(f"  Median Loc Error: {metrics['median_loc_error']:.2f} m")
    print(f"  Mean Loc Error  : {metrics['mean_loc_error']:.2f} m")
    print(f"  Median Ang Error: {metrics['median_ang_error']:.2f}°")
    print(f"  Mean Ang Error  : {metrics['mean_ang_error']:.2f}°")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        print("Please provide a csv path")
