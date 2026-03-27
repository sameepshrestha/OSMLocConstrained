"""Smoke test for generative_osmloc model — verifies shapes and forward pass."""
import sys
sys.path.insert(0, "/home/sameep/phd_research/osmloc/OSMLocConstrained")

import torch
from omegaconf import OmegaConf
from maploc.models.generative_osmloc import generative_osmloc
from maploc.utils.wrappers import Camera

# Minimal config matching osmloc defaults
cfg = OmegaConf.create({
    "image_encoder": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "use_bn": False,
        "use_clstoken": False,
        "localhub": True,
    },
    "map_encoder": {
        "embedding_dim": 128,
        "output_dim": 8,
        "num_classes": {"areas": 7, "ways": 10, "nodes": 24},
        "backbone": "simple",
        "unary_prior": True,
    },
    "latent_dim": 256,
    "matching_dim": 8,
    "sem_channels": 9,
    "z_max": 32,
    "x_max": 32,
    "pixel_per_meter": 2,
    "num_rotations": 256,
    "add_temperature": False,
    "normalize_features": False,
    "padding_matching": "replicate",
    "apply_map_prior": True,
    "do_label_smoothing": False,
    "sigma_xy": 2,
    "sigma_r": 2,
    "num_decoder_blocks": 4,
    "lambda_sem_bev": 10.0,
    "lambda_sem_map": 10.0,
    "lambda_align": 10.0,
    "lambda_kl": 0.01,
    "depth_parameterization": "scale",
    "norm_depth_scores": False,
    "normalize_scores_by_dim": False,
    "normalize_scores_by_num_valid": True,
    "prior_renorm": True,
    "retrieval_dim": None,
})

print("Initializing model...")
model = generative_osmloc(cfg)
model.eval()

device = "cpu"  # use CPU for smoke test
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")
print(f"Frozen:       {total_params - trainable_params:,}")


# Create dummy data
B = 2
H_img, W_img = 224, 224

data = {
    "image": torch.randn(B, 3, H_img, W_img, device=device),
    "map": torch.randint(0, 5, (B, 3, 256, 256), device=device),
    "valid": torch.ones(B, H_img, W_img, dtype=torch.bool, device=device),
    "semantic_mask": torch.rand(B, 9, 256, 256, device=device),
    "uv": torch.tensor([[128.0, 128.0]] * B, device=device),
    "roll_pitch_yaw": torch.tensor([[0.0, 0.0, 45.0]] * B, device=device),
}


print("\nRunning forward pass...")
with torch.no_grad():
    pred = model(data)

print("\n=== Output Shapes ===")
for k, v in pred.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:30s}: {str(list(v.shape)):20s} dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k:30s}: <dict>")
    elif v is None:
        print(f"  {k:30s}: None (inference mode)")

# Test loss computation (switch to training mode)
print("\n=== Testing Loss ===")
model.train()
pred_train = model(data)
loss = model.loss(pred_train, data)
print("Losses:")
for k, v in loss.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:20s}: {v.mean().item():.4f}")

print("\n✅ Smoke test passed!")
