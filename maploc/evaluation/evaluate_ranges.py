
import cv2
import os
import glob
from pathlib import Path
import re

def create_video_from_images(image_folder, output_video_path, fps=5):
    # Get all prediction images
    images = glob.glob(os.path.join(image_folder, "gmu_robot_*_pred.png"))
    
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Extract indices to sort correctly
    def extract_index(filename):
        match = re.search(r'gmu_robot_(\d+)_pred.png', filename)
        return int(match.group(1)) if match else -1

    images.sort(key=extract_index)
    
    # Filter to start from 1000 if needed (though sorting handles order)
    # The user mentioned "make a video from images 1000", imply starting index is 1000
    # We will include all found images sorted.

    if not images:
        print("No valid images found.")
        return

    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Failed to read {images[0]}")
        return
        
    height, width, layers = frame.shape
    size = (width, height)
    
    # Initialize VideoWriter
    # Use mp4v codec for widely compatible mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, size)
    
    print(f"Creating video {output_video_path} with {len(images)} frames...")
    
    for filename in images:
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read {filename}")
            
    out.release()
    print(f"Video saved to {output_video_path}")

def main():
    base_dir = Path("/home/sameep/phd_research/osmloc/OSMLocConstrained/experiments/viz_gmu_robot_levels_2")
    
    # Find all error directories
    error_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("error_")]
    
    if not error_dirs:
        print(f"No error directories found in {base_dir}")
        return

    for error_dir in error_dirs:
        print(f"Processing directory: {error_dir.name}")
        output_video = base_dir / f"{error_dir.name}.mp4"
        create_video_from_images(error_dir, output_video, fps=5)

if __name__ == "__main__":
    main()
