import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra import initialize, compose,initialize_config_dir
from .base import BaseModel
from .map_encoder import MapEncoder
# Since you symlinked mapper to the root, use top-level import
from mapper.module import GenericModule 
from .voting import (
    argmax_xyr, conv2d_fft_batchwise, expectation_xyr,
    log_softmax_spatial, mask_yaw_prior, TemplateSampler, topk_xyr
)
from .bev_projection import make_grid
import os
import yaml
import numpy as np 
class osmloc(BaseModel):
    default_conf = {
        # Your MIA keys
        "mia_config_path": "???",
        "mia_checkpoint": "???",
        
        # Add these baseline keys so the Hydra merge doesn't crash
        "map_encoder": "???",
        "pixel_per_meter": 2, 
        "num_rotations": 256,
        "apply_map_prior": True,
        "latent_dim": 128,      # Matches baseline YAML
        "matching_dim": 8,      # Matches baseline YAML
        "z_max": 32,            # Matches baseline YAML
        "x_max": 32,            # Matches baseline YAML
        "num_scale_bins": 33,   # Matches baseline YAML
        
        # These are in the YAML but ignored by your new model
        "image_encoder": None, 
        "bev_net": None,
    }
    
    # ... rest of your code ...


    def _init(self, conf):
        ckpt_path = conf.mia_checkpoint
        
        resolved_cfg_path = "/home/sameep/phd_research/osmloc/MapItAnywhereLocalization/full_resolved_pretrain_cfg.yaml"
        
        with open(resolved_cfg_path, "r") as f:
            full_cfg_dict = yaml.safe_load(f)
        
        mia_cfg = OmegaConf.create(full_cfg_dict)
        
        self.image_encoder = GenericModule(mia_cfg)
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.image_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        
        self.image_encoder.eval()
        self.image_encoder.freeze()# Lightning's built-in
    # Or manual:
    # for p in self.image_encoder.parameters():
    #     p.requires_grad = False
        # 2. OSM Side (Trainable librarian)
        self.map_encoder = MapEncoder(conf.map_encoder)

        # 3. Semantic Bridge (The 1x1 Embedding Translator)
        # We translate the 6 MIA probability channels into the high-dim Map features (e.g. 128)
        self.feature_adapter = nn.Conv2d(6,48, kernel_size=1)

        # 4. Rotational Matcher Setup
        ppm = conf.pixel_per_meter
        size = 100 # MIA output BEV is 100x100
        
        # This grid defines the "geometry" of the robot's observation.
        # Robot is at bottom-center: orig_x = -width/2, orig_y = 0.
        grid_xz = make_grid(
            size/ppm, size/ppm, 
            step_x=1/ppm, step_y=1/ppm, 
            orig_x=-size/(2*ppm), orig_y=0, 
            y_up=True
        )
        self.template_sampler = TemplateSampler(grid_xz, ppm, conf.num_rotations)

    def _forward(self, data):
        pred = {}
        
        # --- Map Reference Path ---
        # Output dim is usually [B, 128, 256, 256]
        pred_map, embedding_map = self.map_encoder(data)
        pred["map"] = pred_map
        f_map = pred_map["map_features"][0]

        # --- MIA Observation Path ---
        with torch.no_grad():
            res_mia = self.image_encoder(data)
            # Probability map: [B, 6, 100, 100]
            f_bev_raw = res_mia['output']      
            # Observation mask: [B, 1, 100, 100]
            valid_bev = res_mia['valid_bev']    

        print("\n" + "="*40)
        print(f"DEBUG: Shape Analysis")
        print(f"1. f_bev_raw (MIA Output): {f_bev_raw.shape}")
        print(f"2. valid_bev (Data Mask):  {valid_bev.shape}")
        if f_bev_raw.shape[-1] != valid_bev.shape[-1]:
            print(f"!!! MISMATCH DETECTED: {f_bev_raw.shape[-1]} vs {valid_bev.shape[-1]} !!!")
        print("="*40 + "\n")
        if valid_bev.shape[-1] == 101:
            valid_bev = valid_bev[..., :-1] 

        print("\n" + "="*40)
        print(f"DEBUG: Shape Analysis")
        print(f"1. f_bev_raw (MIA Output): {f_bev_raw.shape}")
        print(f"2. valid_bev (Data Mask):  {valid_bev.shape}")
        if f_bev_raw.shape[-1] != valid_bev.shape[-1]:
            print(f"!!! MISMATCH DETECTED: {f_bev_raw.shape[-1]} vs {valid_bev.shape[-1]} !!!")
        # Translate semantic classes into map-features (e.g. 6 -> 128)
        f_bev = self.feature_adapter(f_bev_raw)

        # Matching: Slide rotated image-templates over the global map
        # Output: scores [B, 256, 128, 128] (matches height/width of f_map)
        scores, f_bev_template, valid_template = self.exhaustive_voting(f_bev, f_map, valid_bev)
        
        # Prepare for probability extraction
        scores = scores.moveaxis(1, -1)  # [B, H, W, N]
        
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
            
        log_probs = log_softmax_spatial(scores)
        
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "features_bev": f_bev, 
            "features_map": f_map,
            "features_map_sem": embedding_map,
            # Ensure mask is 2D for loss/metrics
            "valid_bev": valid_bev.squeeze(1) if valid_bev.dim()==4 else valid_bev,
            "valid_template": valid_template,
        }

    def _forward_wrapper(self, data):
        """Routing for evaluation benchmark scripts."""
        res = self._forward(data)
        return res["log_probs"], res["uvr_max"]

    def forward(self, data):
        if "wrapper" in data:
            return self._forward_wrapper(data)
        return self._forward(data)

    def exhaustive_voting(self, f_bev, f_map, valid_bev):
        # Mask out what the camera can't see before matching
        if valid_bev.dim() == 3: valid_bev = valid_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.bool(), 0.0)
        
        templates = self.template_sampler(f_bev)
        
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode="replicate",
            )
        
        # Calculate how much valid area exists per rotation
        valid_templates = self.template_sampler(valid_bev.float()) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores, templates, valid_templates

    def loss(self, pred, data):
        from .voting import nll_loss_xyr, res_l1loss_xyr
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        
        # Localization (Geometry) loss
        nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        
        # Alignment (Semantic) loss - aligns Adapter + Map features to be similar
        # We match MIA-translated-features to OSM-embedded-features
        res = 10.0 * res_l1loss_xyr(
            pred["features_bev"], 
            pred["features_map_sem"], 
            xy_gt, yaw_gt, pred["valid_bev"]
        )
        return {"total": nll + res, "nll": nll, "loss_res": res}

    def metrics(self):
        from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_1m": Location2DRecall(1.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_1°": AngleRecall(1.0, "yaw_max"),
        }