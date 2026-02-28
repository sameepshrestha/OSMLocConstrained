import pandas as pd
import numpy as np
import sys

def evaluate(csv_path):
    print(f"Evaluating results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # Check for required columns
    required_cols = ['error_m_exp', 'rot_err_exp'] # Using expected value prediction
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column {col} missing from CSV.")
            return

    # Metrics
    metrics = {}
    
    # Total samples
    n = len(df)
    metrics['count'] = n
    
    # Location Recall (Max)
    metrics['recall_loc_1m'] = (df['error_m_max'] <= 1.0).mean() * 100
    metrics['recall_loc_3m'] = (df['error_m_max'] <= 3.0).mean() * 100
    metrics['recall_loc_5m'] = (df['error_m_max'] <= 5.0).mean() * 100
    metrics['recall_loc_10m'] = (df['error_m_max'] <= 10.0).mean() * 100
    metrics['recall_loc_20m'] = (df['error_m_max'] <= 20.0).mean() * 100
    
    # Orientation Recall (Max)
    metrics['recall_ang_5deg'] = (df['rot_err_max'] <= 5.0).mean() * 100
    metrics['recall_ang_10deg'] = (df['rot_err_max'] <= 10.0).mean() * 100
    metrics['recall_ang_15deg'] = (df['rot_err_max'] <= 15.0).mean() * 100
    metrics['recall_ang_20deg'] = (df['rot_err_max'] <= 20.0).mean() * 100
    metrics['recall_ang_30deg'] = (df['rot_err_max'] <= 30.0).mean() * 100
    
    # Joint Recall (Loc <= X AND Ang <= Y) (Max)
    metrics['recall_3m_10deg'] = ((df['error_m_max'] <= 3.0) & (df['rot_err_max'] <= 10.0)).mean() * 100
    metrics['recall_5m_10deg'] = ((df['error_m_max'] <= 5.0) & (df['rot_err_max'] <= 10.0)).mean() * 100
    metrics['recall_10m_20deg'] = ((df['error_m_max'] <= 10.0) & (df['rot_err_max'] <= 20.0)).mean() * 100
    
    # Statistics
    metrics['median_loc_error'] = df['error_m_max'].median()
    metrics['mean_loc_error'] = df['error_m_max'].mean()
    metrics['median_ang_error'] = df['rot_err_max'].median()
    metrics['mean_ang_error'] = df['rot_err_max'].mean()

    # Comparison (Expectation)
    metrics['recall_loc_5m_exp'] = (df['error_m_exp'] <= 5.0).mean() * 100
    metrics['median_loc_error_exp'] = df['error_m_exp'].median()

    # Print Report
    print("\n--- Evaluation Report (Using MAX Prediction) ---")
    print(f"Total Samples: {n}")
    print("\nLocation Recall (Max):")
    print(f"  @ 1m : {metrics['recall_loc_1m']:.2f}%")
    print(f"  @ 3m : {metrics['recall_loc_3m']:.2f}%")
    print(f"  @ 5m : {metrics['recall_loc_5m']:.2f}%")
    print(f"  @ 10m: {metrics['recall_loc_10m']:.2f}%")
    print(f"  @ 20m: {metrics['recall_loc_20m']:.2f}%")
    
    print("\nAngular Recall (Max):")
    print(f"  @ 5° : {metrics['recall_ang_5deg']:.2f}%")
    print(f"  @ 10°: {metrics['recall_ang_10deg']:.2f}%")
    print(f"  @ 15°: {metrics['recall_ang_15deg']:.2f}%")
    print(f"  @ 20°: {metrics['recall_ang_20deg']:.2f}%")
    print(f"  @ 30°: {metrics['recall_ang_30deg']:.2f}%")
    
    print("\nJoint Recall (Max):")
    print(f"  (3m, 10°): {metrics['recall_3m_10deg']:.2f}%")
    print(f"  (5m, 10°): {metrics['recall_5m_10deg']:.2f}%")
    print(f"  (10m, 20°): {metrics['recall_10m_20deg']:.2f}%")
    
    print("\nError Statistics (Max):")
    print(f"  Median Loc Error: {metrics['median_loc_error']:.2f} m")
    print(f"  Mean Loc Error  : {metrics['mean_loc_error']:.2f} m")
    print(f"  Median Ang Error: {metrics['median_ang_error']:.2f}°")
    print(f"  Mean Ang Error  : {metrics['mean_ang_error']:.2f}°")
    
    print("\nComparison (Expectation):")
    print(f"  Recall @ 5m (Exp): {metrics['recall_loc_5m_exp']:.2f}%")
    print(f"  Median Loc Error (Exp): {metrics['median_loc_error_exp']:.2f} m")

if __name__ == "__main__":
    csv_path = "results.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    evaluate(csv_path)
