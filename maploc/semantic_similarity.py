import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import cv2
import sys
import torch
from omegaconf import OmegaConf

# Add MapItAnywhereLocalization to sys.path to allow imports
MIA_ROOT = Path("/home/sameep/phd_research/osmloc/MapItAnywhereLocalization")
if str(MIA_ROOT) not in sys.path:
    sys.path.append(str(MIA_ROOT))

# Patch matplotlib style for nuscenes compatibility
try:
    import matplotlib.pyplot as plt
    if 'seaborn-whitegrid' not in plt.style.available:
        if 'seaborn-v0_8-whitegrid' in plt.style.available:
             _orig_use = plt.style.use
             def _safe_use(style):
                 if style == 'seaborn-whitegrid':
                     try:
                         _orig_use('seaborn-v0_8-whitegrid')
                     except OSError:
                         pass 
                 else:
                     _orig_use(style)
             plt.style.use = _safe_use
except ImportError:
    pass

# MIA Imports
try:
    from mapper.utils.exif import EXIF
    from mapper.utils.wrappers import Camera
    from mapper.data.image import rectify_image, resize_image
    from mapper.utils.viz_2d import one_hot_argmax_to_rgb
    from mapper.module import GenericModule
    from perspective2d import PerspectiveFields
except ImportError as e:
    print(f"Error importing MIA modules: {e}")
    print("Ensure you are running in the correct environment and MIA_ROOT is correct.")

# Local Imports
try:
    from models.voting import (
        conv2d_fft_batchwise, 
        TemplateSampler, 
        expectation_xyr, 
        topk_xyr, 
        log_softmax_spatial, 
        argmax_xyr
    )
    from models.utils import make_grid
except ImportError:
    # Fallback if running from a different root
    try:
        from maploc.models.voting import (
            conv2d_fft_batchwise, 
            TemplateSampler, 
            expectation_xyr, 
            topk_xyr, 
            log_softmax_spatial, 
            argmax_xyr
        )
        from maploc.models.utils import make_grid
    except ImportError:
        print("Could not import models.voting or models.utils. Check your python path.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_pipeline(image, roll_pitch, camera):
    # Image expected to be numpy array (H, W, 3)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).to(device)
    camera = camera.to(device)

    image, valid = rectify_image(image, camera.float(), -roll_pitch[0], -roll_pitch[1])

    image, _, camera, valid = resize_image(
        image=image,
        size=512,
        camera=camera,
        fn=max,
        valid=valid
    )

    camera = torch.stack([camera])

    return {
        "image": image.unsqueeze(0).to(device),
        "valid": valid.unsqueeze(0).to(device),
        "camera": camera.float().to(device),
    }

def exhaustive_voting_robust(query_mask, target_map, valid_mask=None, num_rotations=128, device='cpu', grid_xz_bev=None, ppm=2.0):
    """
    Perform exhaustive voting search with validity masking and normalization, matching OSMLoc logic.
    Uses TemplateSampler for rotation if grid_xz_bev is provided.
    
    Args:
        query_mask: (1, C, H, W) - The MIA prediction.
        target_map: (1, C, H_map, W_map) - The Map Crop.
        valid_mask: (1, 1, H, W) - Validity mask for query.
        num_rotations: Number of angles to search.
        grid_xz_bev: Optional grid for TemplateSampler.
        ppm: Pixels per meter.
    """
    import torchvision.transforms.functional as TF
    
    # 1. Mask the Query
    if valid_mask is not None:
        # Resize valid_mask to query size if needed
        if valid_mask.shape[-2:] != query_mask.shape[-2:]:
             valid_mask = torch.nn.functional.interpolate(valid_mask.float(), size=query_mask.shape[-2:], mode='nearest')
        
        valid_mask_bin = valid_mask > 0.5
        query_mask = query_mask.clone()
        query_mask.masked_fill_(~valid_mask_bin, 0.0)
    else:
        valid_mask_bin = torch.ones_like(query_mask[:, :1]) > 0
        valid_mask = valid_mask_bin.float()

    # 2. Generate Rotated Templates
    if grid_xz_bev is not None:
        # Use TemplateSampler
        # Initialize sampler (optimize=False to get full 360 coverage if needed, or True for optimized)
        # TemplateSampler optimization splits 360 into 4 quadrants. 
        # For simplicity/correctness check, let's trust it.
        # But we need to ensure it matches our simple FFT search expectation (stack of N rotations).
        sampler = TemplateSampler(grid_xz_bev, ppm, num_rotations, optimize=True)
        sampler.to(device)
        
        # query_mask is (1, C, H, W). Sampler expects (B, C, H, W).
        query_mask = query_mask.to(device)
        query_kernel = sampler(query_mask) # (1, N, C, H, W)
        
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
            valid_kernel = sampler(valid_mask) # (1, N, 1, H, W)
            # Threshold back to binary
            valid_kernel = valid_kernel > (1 - 1e-4) # As in osmloc.py
    else:
        # Fallback to TF.rotate
        angles = torch.linspace(0, 360, num_rotations + 1)[:-1]
        query_stack = []
        valid_stack = []
        
        for angle in angles:
            q_rot = TF.rotate(query_mask, angle.item(), interpolation=TF.InterpolationMode.BILINEAR)
            query_stack.append(q_rot)
            
            if valid_mask is not None:
                v_rot = TF.rotate(valid_mask_bin.float(), angle.item(), interpolation=TF.InterpolationMode.NEAREST)
                valid_stack.append(v_rot)
                
        query_kernel = torch.stack(query_stack, dim=1) # (1, N, C, H, W)
        valid_kernel = torch.stack(valid_stack, dim=1) if valid_mask is not None else None

    # 3. FFT Correlation
    target_map = target_map.to(device)
    scores = conv2d_fft_batchwise(target_map.float(), query_kernel.float()) # (1, N, H_map, W_map)
    
    # 4. Normalize by Number of Valid Pixels
    if valid_kernel is not None:
        num_valid = valid_kernel.float().sum(dim=(-3, -2, -1)) # (1, N)
        num_valid = num_valid.clamp(min=1.0)
        scores = scores / num_valid.unsqueeze(-1).unsqueeze(-1)
        
    scores = scores.permute(0, 2, 3, 1) # (1, H, W, N)
    return scores

def plot_search_results(
    image_rgb,
    mask_corp, 
    scores, 
    gt_offset_px, 
    pred_offset_px_exp, 
    perturbation_m, 
    gt_angle_deg,
    pred_angle_deg_exp,
    scale_factor, 
    res_m_px, 
    save_path,
    valid_mask=None,
    mia_bev=None,
    pred_offset_px_max=None,
    pred_angle_deg_max=None
):
    """
    Visualize the Search Space (Map), Probability Distribution, and Poses.
    """
    # 1. Prepare Map Visualization
    if mask_corp.ndim == 3:
        if mask_corp.shape[0] == 9: # C, H, W
            mask_viz = np.argmax(mask_corp, axis=0)
        elif mask_corp.shape[2] == 9: # H, W, C
            mask_viz = np.argmax(mask_corp, axis=2)
        else:
            mask_viz = mask_corp.mean(axis=0) # Fallback
    else:
        mask_viz = mask_corp

    h, w = mask_viz.shape[:2]
    center = (w // 2, h // 2)
    
    # GT Pose (in Crop Frame)
    # The crop is centered at (GT + perturbation).
    # So GT is at (Center - perturbation).
    # gt_offset_px should be the perturbation amount (dx, dy).
    gt_px = (center[0] - gt_offset_px[0], center[1] - gt_offset_px[1])
    
    # Pred Pose Exp (in Crop Frame)
    # pred_offset_px_exp is the model prediction (dx_pred, dy_pred).
    # The model predicts the offset FROM center TO GT.
    # So Model says: GT is at (Center - dx_pred).
    # Wait, let's check what the model predicts.
    # The model predicts the shift needed to align Query to Target.
    # If Target is Map Crop and Query is Image.
    # We found a match at (u, v) in the correlation map.
    # If match is at center, it means Query matches Target center.
    # The Target (Map) is centered at (GT + Perturb).
    # If Query matches Center, then Query is at (GT + Perturb).
    # But Query (Image) is at GT (by definition, image captures GT location).
    # So if Query matches Target Center, then Image Location == Map Center Location.
    # GT == GT + Perturb. This implies Perturb = 0.
    # If Perturb is NOT 0, say Perturb = (10, 0).
    # Map Center is at (10, 0) relative to GT.
    # Image is at (0, 0).
    # Image should match at (-10, 0) relative to Map Center.
    # So dx_pred should be -10.
    # So Predicted Location relative to Map Center is (dx_pred, dy_pred).
    # So Pred Pixel = Center + (dx_pred, dy_pred).
    # Wait, if dx_pred is -10, then Center + (-10) = Left of center. Correct.
    
    # So Pred Pixel logic:
    pred_px_exp = (center[0] + pred_offset_px_exp[0], center[1] + pred_offset_px_exp[1])
    
    # Pred Pose Max (in Crop Frame)
    if pred_offset_px_max is not None:
        pred_px_max = (center[0] + pred_offset_px_max[0], center[1] + pred_offset_px_max[1])
    else:
        pred_px_max = None
    
    # 2. Prepare Score Heatmap & Angular Dist
    b, h_fft, w_fft, n_rot = scores.shape
    scores_spatial_max = scores.max(dim=3).values.squeeze().cpu().numpy()
    
    # Angular distribution at MAX location (if available) or Exp location
    if pred_offset_px_max is not None:
        pred_fft_x = int(pred_offset_px_max[0] * scale_factor + w_fft // 2)
        pred_fft_y = int(pred_offset_px_max[1] * scale_factor + h_fft // 2)
    else:
        pred_fft_x = int(pred_offset_px_exp[0] * scale_factor + w_fft // 2)
        pred_fft_y = int(pred_offset_px_exp[1] * scale_factor + h_fft // 2)
    
    pred_fft_x = max(0, min(w_fft-1, pred_fft_x))
    pred_fft_y = max(0, min(h_fft-1, pred_fft_y))
    
    angular_scores = scores[0, pred_fft_y, pred_fft_x, :].cpu().numpy()
    angles = np.linspace(0, 360, n_rot, endpoint=False)
    
    heatmap = cv2.resize(scores_spatial_max, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Handle -inf for normalization
    # Replace -inf with min finite value
    heatmap_finite = heatmap[np.isfinite(heatmap)]
    if len(heatmap_finite) > 0:
        min_val = heatmap_finite.min()
        max_val = heatmap_finite.max()
        heatmap[~np.isfinite(heatmap)] = min_val
    else:
        min_val = 0
        max_val = 1
        heatmap[:] = 0
        
    # Normalize heatmap for display (0-1)
    if max_val > min_val:
        heatmap_norm = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap_norm = np.zeros_like(heatmap)
        
    # Resize valid_mask to match heatmap if provided
    valid_mask_viz = None
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.cpu().numpy()
        valid_mask_viz = cv2.resize(valid_mask.astype(float), (w, h), interpolation=cv2.INTER_NEAREST)

    
    # 3. Plot Layout (2 Rows, 3 Cols)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    
    ax_img = fig.add_subplot(gs[0, 0])
    ax_mia = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[0, 2])
    
    ax_hm = fig.add_subplot(gs[1, 0])
    ax_ang = fig.add_subplot(gs[1, 1])
    
    err_m = perturbation_m
    if isinstance(err_m, tuple): err_m = np.sqrt(err_m[0]**2 + err_m[1]**2)
    elif isinstance(err_m, (list, np.ndarray)): err_m = np.sqrt(err_m[0]**2 + err_m[1]**2)
    
    # Image
    ax_img.imshow(image_rgb)
    ax_img.set_title("Input RGB Image")
    ax_img.axis('off')
    
    # MIA Prediction
    if mia_bev is not None:
        # mia_bev is (C, H, W). Visualize max channel or specific channel
        if mia_bev.ndim == 3:
            mia_viz = np.argmax(mia_bev, axis=0) # (H, W) or (H_fft, W_fft)
        else:
            mia_viz = mia_bev
        ax_mia.imshow(mia_viz, cmap='tab10', interpolation='nearest') # 'tab10' for categorical
        ax_mia.set_title("MIA Prediction (BEV)")
        ax_mia.axis('off')

    
    # Map (Search Space)
    ax_map.imshow(mask_viz, cmap='gray')
    ax_map.scatter([gt_px[0]], [gt_px[1]], c='lime', s=100, label=f'GT ({-gt_angle_deg:.1f}°)', marker='o', edgecolors='black')
    
    # Plot Exp Prediction (Orange Dot)
    ax_map.scatter([pred_px_exp[0]], [pred_px_exp[1]], c='orange', s=80, label=f'Exp ({pred_angle_deg_exp:.1f}°)', marker='o')
    
    # Plot Max Prediction (Red X) - Highlight
    if pred_px_max is not None:
        ax_map.scatter([pred_px_max[0]], [pred_px_max[1]], c='red', s=150, label=f'MAX ({pred_angle_deg_max:.1f}°)', marker='X', linewidths=3, edgecolors='white')
        # Error Line for MAX
        ax_map.plot([gt_px[0], pred_px_max[0]], [gt_px[1], pred_px_max[1]], c='red', linestyle='--', linewidth=2, label='Err Max')
    else:
        # Fallback line for Exp
        ax_map.plot([gt_px[0], pred_px_exp[0]], [gt_px[1], pred_px_exp[1]], c='orange', linestyle='--', linewidth=1, label='Err Exp')

    ax_map.legend(loc='upper right')
    ax_map.set_title(f"Search Space (Perturb: {err_m:.1f}m)")
    ax_map.set_xlim(0, w)
    ax_map.set_ylim(h, 0)
    
    # Heatmap
    im = ax_hm.imshow(heatmap_norm, cmap='jet')
    ax_hm.scatter([gt_px[0]], [gt_px[1]], c='lime', s=80, marker='o', edgecolors='black')
    if pred_px_max is not None:
        ax_hm.scatter([pred_px_max[0]], [pred_px_max[1]], c='red', marker='X', s=150, label='MAX')
    ax_hm.scatter([pred_px_exp[0]], [pred_px_exp[1]], c='orange', marker='o', s=80, label='Exp')
    
    ax_hm.legend()
    ax_hm.set_title("Search Scores (Heatmap)")
    
    if valid_mask_viz is not None:
        # Overlay validity mask (white for valid, black for invalid)
        # We can use alpha to show it
        # valid areas = 1, invalid = 0
        # let's shadow invalid areas
        invalid_mask = 1.0 - valid_mask_viz
        # Create a red overlay for invalid regions
        overlay = np.zeros((h, w, 4))
        overlay[..., 0] = 1.0 # Red
        overlay[..., 3] = invalid_mask * 0.5 # Alpha
        ax_hm.imshow(overlay, origin='upper')

    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    
    # Angular
    ax_ang.plot(angles, angular_scores, 'b-', linewidth=2)
    if pred_angle_deg_max is not None:
        ax_ang.axvline(x=pred_angle_deg_max % 360, color='r', linestyle='-', linewidth=2, label='Pred Max')
    ax_ang.axvline(x=pred_angle_deg_exp % 360, color='orange', linestyle='--', label='Pred Exp')
    ax_ang.axvline(x=gt_angle_deg % 360, color='g', linestyle='--', label='GT Angle')
    ax_ang.set_title("Angular Score Distribution")
    ax_ang.set_xlabel("Rotation (deg)")
    ax_ang.set_ylabel("Correlation Score")
    ax_ang.set_xlim(0, 360)
    ax_ang.grid(True)
    ax_ang.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

class SemanticMaskReader:
    def __init__(self, dataset_root, mia_config_path=None, mia_checkpoint_path=None):
        self.dataset_root = Path(dataset_root)
        self.masks_dir = self.dataset_root / "semantic_masks"
        self.images_dir = self.dataset_root / "images"
        self.metadata_path = self.dataset_root / "image_metadata_filtered_processed.parquet"
        self.dump_json_path = self.dataset_root / "dump.json"
        
        print(f"Loading metadata from {self.metadata_path}...")
        self.metadata = pd.read_parquet(self.metadata_path)
        self.metadata_index = self.metadata.set_index('id')
        
        print(f"Loading camera parameters from {self.dump_json_path}...")
        import json
        with open(self.dump_json_path, 'r') as f:
            self.dump_data = json.load(f)
            
        self.cameras = {}
        self.views = {}
        
        for chunk_id, chunk_data in self.dump_data.items():
            if "views" in chunk_data:
                for view_id, view_data in chunk_data["views"].items():
                    self.views[view_id] = view_data
            if "cameras" in chunk_data:
                for cam_id, cam_data in chunk_data["cameras"].items():
                    self.cameras[cam_id] = cam_data

        self.MASK_SIZE_M = 128.0
        self.MASK_SIZE_PX = 224
        self.RESOLUTION_M_PX = 0.5

        self.mia_model = None
        
        if mia_config_path and mia_checkpoint_path:
            self.init_mia(mia_config_path, mia_checkpoint_path)

    def init_mia(self, config_path, checkpoint_path):
        print("Initializing MIA model...")
        cfg = OmegaConf.load(config_path)
        self.mia_model = GenericModule(cfg)
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
             state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
             state_dict = torch.load(checkpoint_path, map_location=device)

        self.mia_model.load_state_dict(state_dict["state_dict"], strict=False)
        self.mia_model.to(device)
        self.mia_model.eval()
        print("MIA model initialized.")

    def get_crop(self, image_id, crop_size_m=64.0, error_max_m=30.0):
        if error_max_m >= crop_size_m / 2:
            print(f"Warning: error_max_m ({error_max_m}) is >= crop_size_m / 2 ({crop_size_m / 2}). GT might be nearly out of crop.")

        # 1. Load Data
        mask_path = self.masks_dir / f"{image_id}.npz"
        try:
             mask_data = np.load(mask_path)
             if 'arr_0' in mask_data:
                 mask = mask_data['arr_0']
             else:
                 mask = mask_data['arr_0']
        except Exception as e:
            raise RuntimeError(f"Failed to load mask for {image_id}: {e}")

        # Load RGB Image
        image_filename = f"{image_id}_undistorted.jpg"
        image_path = self.images_dir / image_filename
        if not image_path.exists():
             image_path = self.images_dir / f"{image_id}.jpg"
        
        if not image_path.exists():
             raise FileNotFoundError(f"Image not found at {image_path}")
             
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Perturbation
        # We shift the crop center by (dx, dy).
        # This means the Map Crop is centered at (GT + Perturbation).
        # Relative to the Crop Center, the GT is at (-dx, -dy).
        if hasattr(self, 'perturbation_px') and self.perturbation_px is not None:
             dx_px, dy_px = self.perturbation_px
        else:
             if error_max_m > 0:
                 dx = np.random.uniform(-error_max_m, error_max_m)
                 dy = np.random.uniform(-error_max_m, error_max_m)
                 dx_px = int(dx / self.RESOLUTION_M_PX)
                 dy_px = int(dy / self.RESOLUTION_M_PX)
             else:
                 dx_px, dy_px = 0, 0
        
        # 3. Crop
        center_x, center_y = 112, 112 
        new_center_x = center_x + dx_px
        new_center_y = center_y + dy_px
        
        crop_size_px = int(crop_size_m / self.RESOLUTION_M_PX)
        half_crop = crop_size_px // 2
        
        pad_size = crop_size_px
        mask_padded = np.pad(mask, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
        
        x1_pad = new_center_x + pad_size - half_crop
        x2_pad = new_center_x + pad_size + half_crop
        y1_pad = new_center_y + pad_size - half_crop
        y2_pad = new_center_y + pad_size + half_crop
        
        mask_crop = mask_padded[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # 4. MIA Inference
        bev_pred = None
        valid_bev = None
        
        if self.mia_model:
            with torch.no_grad():
                if image_id not in self.views:
                     print(f"Warning: Image ID {image_id} not found in dump.json views.")
                
                view_data = self.views.get(image_id)
                if view_data is None:
                     raise RuntimeError(f"View data for {image_id} not found in dump.json")
                
                cam_id = view_data['camera_id']
                cam_data = self.cameras.get(cam_id)
                if cam_data is None:
                     raise RuntimeError(f"Camera data for {cam_id} not found")
                
                rpy = view_data['roll_pitch_yaw']
                roll_pitch = (rpy[0], rpy[1])
                
                params = cam_data['params']
                width = cam_data['width']
                height = cam_data['height']
                
                if cam_data['model'] == 'PINHOLE':
                    f = params[0]
                    cx = params[2]
                    cy = params[3]
                    camera = Camera.from_dict({
                        "model": "SIMPLE_PINHOLE",
                        "width": width,
                        "height": height,
                        "params": [f, cx, cy]
                    })
                else:
                     camera = Camera.from_dict(cam_data)

                data = preprocess_pipeline(image_rgb, list(roll_pitch), camera)
                res = self.mia_model(data)
                
                prediction = res['output']
                bev_pred = prediction.cpu()
                if 'valid_bev' in res:
                    valid_bev = res['valid_bev'].cpu()

        return {
            "image": image_rgb,
            "mask_crop": mask_crop,
            "perturbation_px": (dx_px, dy_px),
            "perturbation_m": (dx_px * self.RESOLUTION_M_PX, dy_px * self.RESOLUTION_M_PX),
            "image_id": image_id,
            "bev_pred": bev_pred,
            "valid_bev": valid_bev,
            "yaw": rpy[2] # Ground Truth Heading (degrees)
        }

if __name__ == "__main__":
    # Load Configuration
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found. Using defaults/hardcoded paths.")
        # Fallback or Exit? Let's keep hardcoded as fallback or just fail. 
        # For robustness in this context, let's assume it exists as I just created it.
    
    conf = OmegaConf.load(config_path)
    
    dataset_root = conf.paths.dataset_root
    mia_config = conf.paths.mia_config
    mia_checkpoint = conf.paths.mia_checkpoint
    
    viz_dir = Path(conf.paths.viz_dir)
    viz_dir.mkdir(exist_ok=True, parents=True)

    try:
        print("Starting Large Scale Analysis with Semantic Priors...")
        reader = SemanticMaskReader(dataset_root, mia_config, mia_checkpoint)
        
        # Params from Config
        num_images = 500
        results_list = []
        json_results = []
        
        FFT_SIZE = conf.experiment.fft_size
        ppm_val = 1.0 / conf.experiment.resolution_m_px
        CROP_HEIGHT_M = conf.experiment.crop_height_m
        SEMANTIC_PRIOR = conf.experiment.semantic_prior
        
        # Grid for TemplateSampler (covers 70x70m area centered at 0)
        # We assume TemplateSampler handles the rotation of the cropped query internally if we pass the right grid?
        # No, we decided to perform TF rotation on the query directly and pass None as grid to 'exhaustive_voting_robust'.
        # So we don't strictly need grid_bev here unless we revert to TemplateSampler's internal rotation.
        # Keeping it consistent with previous successful run.
        
        import torchvision.transforms.functional as TF
        
        # Get list of valid IDs
        valid_ids = reader.metadata['id'].tolist()
        target_ids = valid_ids[:num_images] # Limit
        
        print(f"Targeting {len(target_ids)} images.")
        
        for idx, sample_id in enumerate(target_ids):
            print(f"Processing {idx+1}/{len(target_ids)}: {sample_id}")
            
            try:
                # 1. Base Prediction (No error)
                base_result = reader.get_crop(sample_id, crop_size_m=80.0, error_max_m=0.0)
                bev_pred_original = base_result['bev_pred']
                valid_bev_original = base_result.get('valid_bev')
                
                if bev_pred_original is None:
                    print(f"  Skipping {sample_id}: MIA Inference failed.")
                    continue

                # 2. CROP TO FOREGROUND 
                crop_h_px = int(CROP_HEIGHT_M / reader.RESOLUTION_M_PX) 
                _, _, h_pred, w_pred = bev_pred_original.shape
                
                if h_pred > crop_h_px:
                    query_base = bev_pred_original[..., -crop_h_px:, :] 
                    if valid_bev_original is not None:
                        if valid_bev_original.ndim == 5: valid_bev_original = valid_bev_original.squeeze(2)
                        elif valid_bev_original.ndim == 3: valid_bev_original = valid_bev_original.unsqueeze(1)
                        valid_base = valid_bev_original[..., -crop_h_px:, :]
                    else:
                        valid_base = None
                else:
                    query_base = bev_pred_original
                    valid_base = valid_bev_original
                
                query_base = torch.softmax(query_base.float(), dim=1)
                
                # 3. Perturbed Search (Single Large Perturbation)
                error_m = conf.experiment.perturbation_m
                result = reader.get_crop(sample_id, crop_size_m=80.0, error_max_m=error_m)
                mask_crop = result['mask_crop']
                
                # Prepare Target
                if mask_crop.ndim == 3 and mask_crop.shape[2] == 9:
                     mask_fft = torch.from_numpy(mask_crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
                     # We select a subset for correlation (optional)
                     mask_fft_reduced = mask_fft[:, [0, 1, 2, 4, 6, 7], :, :] 
                     
                     # Extract Semantic Mask for Prior
                     # Channels (from MIA dataset.md): 
                     # 0: Road, 1: Crossing, 2: Pedestrian (Sidewalk), 3: Park, 
                     # 4: Building, 5: Water, 6: Terrain, 7: Parking, 8: Train
                     # User Request: "Use sidewalk roads park everything except buildings"
                     # Logic: Valid = NOT Building
                     
                     building_mask = mask_fft[:, 4:5, :, :] > 0.5
                     road_mask_unrotated = ~building_mask
                else:
                     mask_perm = torch.from_numpy(mask_crop).permute(2,0,1).unsqueeze(0).float().to(device)
                     # Fallback logic
                     if mask_perm.shape[1] >= 9:
                          mask_fft_reduced = mask_perm[:, [0, 1, 2, 4, 6, 7], :, :]
                          building_mask = mask_perm[:, 4:5, :, :] > 0.5
                          road_mask_unrotated = ~building_mask
                     else:
                          mask_fft_reduced = mask_perm
                          road_mask_unrotated = (mask_perm[:, 0:1, :, :]) > 0 # Default to just road if channels unknown
                 
                
                # Interpolate target to FFT_SIZE
                target = torch.nn.functional.interpolate(mask_fft_reduced, size=(FFT_SIZE, FFT_SIZE), mode='nearest')
                
                # Prepare Query (MIA Prediction)
                # Select same channels as target: [0, 1, 2, 4, 6, 7]
                # query_base is (1, C, H, W).
                if query_base.shape[1] == 9:
                    query_reduced = query_base[:, [0, 1, 2, 4, 6, 7], :, :]
                else:
                    query_reduced = query_base # Assumption
                
                # Interpolate Query to FFT_SIZE
                query_interpolated = torch.nn.functional.interpolate(query_reduced, size=(FFT_SIZE, FFT_SIZE), mode='bilinear', align_corners=False)
                
                if valid_base is not None:
                     # Ensure 4D for 2D interpolation
                     while valid_base.ndim < 4:
                         valid_base = valid_base.unsqueeze(0)
                     valid_interpolated = torch.nn.functional.interpolate(valid_base.float(), size=(FFT_SIZE, FFT_SIZE), mode='nearest')
                else:
                     valid_interpolated = None

                # No Rotation Injection for Map (Target)
                # The map is North-aligned. The User requires using the Image (Query) to localize.
                # We search for the rotation of the Query relative to the North-aligned Map.
                # So 'target' (Map) stays fixed. 'query' (Image) is rotated by the search algorithm.
                
                rot_deg = 0.0
                
                # Device
                query_final = query_interpolated.to(device)
                target = target.to(device)
                if valid_interpolated is not None: 
                    valid_final = valid_interpolated.to(device)
                else:
                    valid_final = None 
                
                # road_mask_unrotated is the binary map mask. 
                # Since we don't rotate the map target, we use unrotated for the prior.
                road_mask_final = road_mask_unrotated.to(device)

                
                # Search
                scores = exhaustive_voting_robust(
                    query_final, target, valid_mask=valid_final, 
                    num_rotations=conf.experiment.num_rotations, device=device, 
                    grid_xz_bev=None, ppm=ppm_val 
                ) # (1, H, W, N)
                
                # --- SEMANTIC PRIOR APPLICATION ---
                road_mask_bc = None
                if SEMANTIC_PRIOR:
                    # Resize road mask to match scores (e.g. 141x141 if FFT_SIZE=140)
                    out_h, out_w = scores.shape[1], scores.shape[2]
                    
                    road_mask = torch.nn.functional.interpolate(
                        road_mask_final.float(), 
                        size=(out_h, out_w), 
                        mode='nearest'
                    ) > 0.5
                    
                    # scores is (1, H, W, N).
                    # road_mask is (1, 1, H, W). Needs to be broadcast to (1, H, W, N).
                    # Permute mask to (1, H, W, 1)
                    road_mask_bc = road_mask.permute(0, 2, 3, 1)
                    
                    # Apply Mask: Set scores at invalid locations to -inf
                    if road_mask_bc.sum() > 0:
                         scores = scores.masked_fill(~road_mask_bc, -float('inf'))
                # ----------------------------------
                
                # 4. Probabilistic Estimation
                log_probs = log_softmax_spatial(scores) 
                probs = log_probs.exp()
                
                xyr_exp, _ = expectation_xyr(probs)
                xyr_max = argmax_xyr(scores)
                
                # Extract Predictions
                dx_px_exp = xyr_exp[0, 0].item() - (scores.shape[2] // 2)
                dy_px_exp = xyr_exp[0, 1].item() - (scores.shape[1] // 2)
                angle_exp = xyr_exp[0, 2].item()
                
                dx_px_max = xyr_max[0, 0].item() - (scores.shape[2] // 2)
                dy_px_max = xyr_max[0, 1].item() - (scores.shape[1] // 2)
                angle_max = xyr_max[0, 2].item()
                
                # Ground Truth
                # If map is North-Aligned, and MIA predicts Ego-BEV, then the rotation is the heading.
                # However, if MIA predicts North-aligned BEV (which it might if using compass?), then rotation is 0.
                # Let's assume standard task: we need to find heading.
                # So gt_angle = yaw.
                gt_angle = base_result['yaw']
                injected_m = result['perturbation_m']
                injected_px = result['perturbation_px']
                
                scale_factor = FFT_SIZE / mask_crop.shape[0]
                eff_ppm = ppm_val * scale_factor
                
                # Expectation Error
                err_x_exp = (dx_px_exp / eff_ppm) + injected_m[0]
                err_y_exp = (dy_px_exp / eff_ppm) + injected_m[1]
                total_err_exp = np.sqrt(err_x_exp**2 + err_y_exp**2)
                rot_err_exp = min(abs(angle_exp - gt_angle), 360 - abs(angle_exp - gt_angle))
                
                # Max Error
                err_x_max = (dx_px_max / eff_ppm) + injected_m[0]
                err_y_max = (dy_px_max / eff_ppm) + injected_m[1]
                total_err_max = np.sqrt(err_x_max**2 + err_y_max**2)
                rot_err_max = min(abs(angle_max - gt_angle), 360 - abs(angle_max - gt_angle))
                
                print(f"  Img {sample_id}: ExpErr {total_err_exp:.2f}m, MaxErr {total_err_max:.2f}m")
                
                # Save Result
                res_dict = {
                    "id": sample_id,
                    "error_m_exp": total_err_exp,
                    "error_m_max": total_err_max,
                    "rot_err_exp": rot_err_exp,
                    "rot_err_max": rot_err_max,
                    "gt_angle": gt_angle,
                    "pred_angle_exp": angle_exp,
                    "pred_angle_max": angle_max,
                    "injected_m_x": injected_m[0],
                    "injected_m_y": injected_m[1],
                    "injected_m_z": 0.0,
                    # Detailed State (Relative to GT center which is 0,0)
                    "gt_x_m": 0.0, # Center of crop is GT location (offset by injected_m)
                    "gt_y_m": 0.0, 
                    # Prediction in Crop Frame (Offset from center)
                    "pred_x_m_exp": dx_px_exp / eff_ppm,
                    "pred_y_m_exp": dy_px_exp / eff_ppm,
                    "pred_x_m_max": dx_px_max / eff_ppm,
                    "pred_y_m_max": dy_px_max / eff_ppm,
                    # Errors (Difference between pred and GT)
                    # GT is at (0,0) in the UNPERTURBED frame.
                    # But due to perturbation, GT is at (-injected_m) relative to crop center.
                    # err_x_exp includes injected_m, so it is the error relative to true location.
                    "error_x_m_exp": err_x_exp,
                    "error_y_m_exp": err_y_exp,
                    "error_x_m_max": err_x_max,
                    "error_y_m_max": err_y_max,
                }
                results_list.append(res_dict)
                json_results.append(res_dict)
                
                # Visualization (ALL IMAGES)
                viz_name = viz_dir / f"viz_{sample_id}.png"
                try:
                    # Plotting
                    plot_search_results(
                        image_rgb=result['image'], 
                        mask_corp=target.squeeze(0).cpu().numpy(),
                        scores=scores,
                        gt_offset_px=result['perturbation_px'], 
                        pred_offset_px_exp=(dx_px_exp/scale_factor, dy_px_exp/scale_factor),
                        perturbation_m=result['perturbation_m'],
                        gt_angle_deg=gt_angle,
                        pred_angle_deg_exp=angle_exp,
                        scale_factor=scale_factor,
                        res_m_px=reader.RESOLUTION_M_PX,
                        save_path=str(viz_name), 
                        valid_mask=road_mask_final.squeeze().cpu().numpy() if road_mask_final is not None else None,
                        mia_bev=query_final.squeeze(0).cpu().numpy(),
                        pred_offset_px_max=(dx_px_max/scale_factor, dy_px_max/scale_factor),
                        pred_angle_deg_max=angle_max
                    )
                except Exception as e:
                      print(f"  Viz failed: {e}")
                          
            except Exception as e:
                print(f"  Error processing {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Summary
        df = pd.DataFrame(results_list)
        print("\nLarge Scale Analysis Summary (With Semantics):")
        print(df.describe())
        
        df.to_csv(conf.paths.results_csv, index=False)
        print(f"{conf.paths.results_csv} saved.")
        
        import json
        with open(conf.paths.results_json, "w") as f:
            json.dump(json_results, f, indent=4)
        print(f"{conf.paths.results_json} saved.")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

