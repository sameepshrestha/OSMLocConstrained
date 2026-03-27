"""
Generative OSMLoc — Conditional VAE for BEV Localization.

Replaces the geometric depth-based BEV projection (polar → cartesian) with a
conditional VAE that generates BEV semantic masks from a single image, then
matches them against the OSM map for localization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torchvision.models.resnet import Bottleneck

from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection
from .voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    crop_map,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    res_l1loss_xyr,
    topk_xyr,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .depth_anything.dpt import DepthAnything
from .utils import make_grid, rotmat2d


class PriorNetwork(nn.Module):

    def __init__(self, img_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(img_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * latent_dim),
        )

    def forward(self, f_img):
        h = self.net(f_img)
        mu, log_sigma = h.chunk(2, dim=-1)
        return mu, log_sigma


class PosteriorEncoder(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 2 * latent_dim),
        )

    def forward(self, x):
        """x: (B, 9, H, W) → mu (B, D), log_sigma (B, D)."""
        h = self.encoder(x)
        mu, log_sigma = h.chunk(2, dim=-1)
        return mu, log_sigma


class ConditionalDecoder(nn.Module):

    def __init__(self, latent_dim, img_dim, matching_dim, sem_channels,
                 bev_h, bev_w, sem_align_dim, num_blocks=4):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Project latent z to a small spatial map
        self.z_init_h = bev_h // 8                 # 8
        self.z_init_w = bev_w // 8                 # ~16
        self.z_proj = nn.Linear(latent_dim, 128 * self.z_init_h * self.z_init_w)

        # Fuse projected z with image features
        self.fuse = nn.Sequential(
            nn.Conv2d(128 + img_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Bottleneck blocks (same structure as BEVNet)
        blocks = []
        for i in range(num_blocks):
            blocks.append(Bottleneck(128, 128 // Bottleneck.expansion))
        self.blocks = nn.Sequential(*blocks)

        # Matching head — taps from the last block output
        self.matching_head = nn.Conv2d(128, matching_dim, 1)

        # Semantic head — predicts 9 MIA classes
        self.semantic_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, sem_channels, 1),
        )

        # Sem classifier for res_l1loss (BEV features → map embedding dim)
        # This matches the shape of MapEncoder embeddings for the L_align loss
        self.sem_align = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, sem_align_dim, 1),
        )

    def forward(self, z, f_img):
        B = z.shape[0]

        # 1. Project z to spatial
        z_spatial = self.z_proj(z).view(B, 128, self.z_init_h, self.z_init_w)
        z_spatial = F.interpolate(z_spatial, (self.bev_h, self.bev_w),
                                  mode="bilinear", align_corners=False)

        # 2. Resize image features to BEV size and concatenate
        f_img_resized = F.interpolate(f_img, (self.bev_h, self.bev_w),
                                      mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([z_spatial, f_img_resized], dim=1))

        # 3. Bottleneck blocks
        features = self.blocks(fused)  # (B, 128, bev_h, bev_w)

        # 4. Heads
        f_bev = self.matching_head(features)          # (B, matching_dim, H, W)
        sem_bev = self.semantic_head(features)         # (B, 9, H, W)
        f_bev_sem = self.sem_align(features)           # (B, 384, H, W)

        return f_bev, sem_bev, f_bev_sem, features


class MapSemanticHead(nn.Module):

    def __init__(self, embedding_dim, sem_channels=9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, sem_channels, 1),
        )

    def forward(self, embeddings):
        return self.head(embeddings)


def reparameterize(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu_q, log_sigma_q, mu_p, log_sigma_p):
    var_q = torch.exp(log_sigma_q)
    var_p = torch.exp(log_sigma_p)
    kl = 0.5 * (
        log_sigma_p - log_sigma_q
        + var_q / var_p
        + (mu_q - mu_p) ** 2 / var_p
        - 1
    )
    return kl.sum(dim=-1)  # (B,)


class generative_osmloc(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "latent_dim": 256,       # VAE latent dimension
        "matching_dim": 8,       # output channels for f_bev / f_map matching
        "sem_channels": 9,       # MIA semantic classes
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 2,
        "sigma_r": 2,
        "num_decoder_blocks": 4,
        # Loss weights
        "lambda_sem_bev": 10.0,
        "lambda_sem_map": 10.0,
        "lambda_align": 10.0,
        "lambda_kl": 0.01,
        # Compatibility (kept for config loading)
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        self.image_encoder = DepthAnything(conf.image_encoder)
        for p in self.image_encoder.pretrained.parameters():
            p.requires_grad = False
        for p in self.image_encoder.depth_head.parameters():
            p.requires_grad = False

        self.map_encoder = MapEncoder(conf.map_encoder)

        ppm = conf.pixel_per_meter
        delta = 1.0 / ppm
        grid_xz = make_grid(
            conf.x_max * 2 + delta, conf.z_max,
            step_y=delta, step_x=delta,
            orig_y=delta, orig_x=-conf.x_max,
            y_up=True,
        )
        self.register_buffer("grid_xz", grid_xz, persistent=False)
        bev_h, bev_w = grid_xz.shape[:2]  # typically (64, 129)

        self.template_sampler = TemplateSampler(
            grid_xz, ppm, conf.num_rotations
        )

        img_dim = 128  # DINOv2 FPN output channels
        self.prior_net = PriorNetwork(img_dim, conf.latent_dim)
        self.posterior_net = PosteriorEncoder(conf.sem_channels, conf.latent_dim)
        self.decoder = ConditionalDecoder(
            latent_dim=conf.latent_dim,
            img_dim=img_dim,
            matching_dim=conf.matching_dim,
            sem_channels=conf.sem_channels,
            bev_h=bev_h,
            bev_w=bev_w,
            sem_align_dim=conf.map_encoder.embedding_dim * 3,
            num_blocks=conf.num_decoder_blocks,
        )

        map_emb_dim = conf.map_encoder.embedding_dim * 3
        self.map_sem_head = MapSemanticHead(map_emb_dim, conf.sem_channels)

        if conf.add_temperature:
            temperature = nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)


    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores, templates, valid_templates

    def _crop_semantic_mask_ego(self, semantic_mask, xy, yaw, bev_h, bev_w):
        """Crop a north-aligned semantic canvas into the ego-centric BEV frame."""
        bev_size = xy.new_tensor([bev_w, bev_h])
        map_size = semantic_mask.new_tensor(semantic_mask.shape[-2:][::-1])
        grid_uv = make_grid(
            bev_size[0],
            bev_size[1],
            step_x=1,
            step_y=1,
            orig_x=-(bev_size[0] // 2),
            orig_y=-bev_size[1],
            y_up=False,
            device=semantic_mask.device,
        )
        rotmats = rotmat2d(yaw / 180 * torch.pi)
        grid_uv_rot = torch.einsum("...nij, ...hwj -> ...nhwi", rotmats, grid_uv)
        grid_uv_map = grid_uv_rot + xy.view(-1, 1, 1, 2)
        grid_uv_map_norm = grid_uv_map / (map_size.view(1, -1) - 1)
        grid_uv_map_norm = grid_uv_map_norm * 2 - 1

        sem_crop = F.grid_sample(
            semantic_mask,
            grid_uv_map_norm,
            align_corners=True,
            mode="nearest",
        )
        sem_valid = torch.all(
            (grid_uv_map_norm >= -1) & (grid_uv_map_norm <= 1),
            dim=-1,
            keepdim=False,
        ).unsqueeze(1)
        sem_valid = sem_valid.expand(-1, semantic_mask.shape[1], -1, -1)
        sem_crop = sem_crop * sem_valid.to(sem_crop.dtype)
        return sem_crop, sem_valid

    def _forward(self, data):
        pred = {}

        pred_map, embedding_map = self.map_encoder(data)
        pred["map"] = pred_map
        f_map = pred_map["map_features"][0]       # (B, matching_dim, 256, 256)

        sem_map = self.map_sem_head(embedding_map)  # (B, 9, 256, 256)

        output = self.image_encoder(data["image"])
        f_image = output["features"]               # (B, 128, H/2, W/2)

        mu_prior, log_sigma_prior = self.prior_net(f_image)

        sem_gt_ego = sem_gt_ego_valid = None
        if "semantic_mask" in data:
            sem_gt_ego, sem_gt_ego_valid = self._crop_semantic_mask_ego(
                data["semantic_mask"],
                data["uv"],
                data["roll_pitch_yaw"][..., -1],
                self.decoder.bev_h,
                self.decoder.bev_w,
            )

        # Posterior: encode GT-centered BEV crop of true MIA mask (training only)
        # We use the same ego-centric crop as the semantic loss and alignment loss:
        # translate to the GT position and rotate by the gravity-corrected camera yaw.
        if self.training and sem_gt_ego is not None:
            mu_post, log_sigma_post = self.posterior_net(sem_gt_ego)
            z = reparameterize(mu_post, log_sigma_post)
        else:
            mu_post = log_sigma_post = None
            z = mu_prior  # deterministic at inference

        # Decode: z + image features → BEV
        f_bev, sem_bev, f_bev_sem, features_raw = self.decoder(z, f_image)
        # f_bev: (B, matching_dim, 64, 129)
        # sem_bev: (B, 9, 64, 129)
        # f_bev_sem: (B, 384, 64, 129) for alignment loss
        # features_raw: (B, 128, 64, 129) raw intermediate

        # Valid mask: the generative decoder produces the entire BEV,
        # so all pixels are valid (no camera frustum clipping).
        valid_bev = torch.ones(
            f_bev.shape[0], f_bev.shape[2], f_bev.shape[3],
            dtype=torch.bool, device=f_bev.device,
        )

        scores, templates, valid_templates = self.exhaustive_voting(
            f_bev, f_map, valid_bev
        )
        scores = scores.moveaxis(1, -1)  # B, H, W, N

        # Apply unary prior from map
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
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            # VAE outputs
            "mu_prior": mu_prior,
            "log_sigma_prior": log_sigma_prior,
            "mu_post": mu_post,
            "log_sigma_post": log_sigma_post,
            # BEV outputs
            "sem_bev": sem_bev,
            "sem_map": sem_map,
            "sem_bev_target": sem_gt_ego,
            "sem_bev_valid": sem_gt_ego_valid,
            "features_bev": f_bev,
            "features_map": f_map,
            "features_bev_sem": f_bev_sem,
            "features_map_sem": embedding_map,
            "features_image": f_image,
            "valid_bev": valid_bev,
            "valid_template": valid_templates,
        }

    def forward(self, data):
        return self._forward(data)

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]

        #  Localization NLL
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"], xy_gt, yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)

        loss_dict = {"nll": nll}
        total = nll

        #  BEV Semantic Loss (generated BEV vs MIA mask, cropped to BEV size)
        if pred.get("sem_bev_target") is not None:
            sem_gt = data["semantic_mask"]  # (B, 9, 256, 256) canvas-aligned
            sem_bev = pred["sem_bev"]       # (B, 9, 64, 129)
            sem_gt_crop = pred["sem_bev_target"]
            sem_valid = pred["sem_bev_valid"].to(sem_bev.dtype)

            sem_bev_bce = F.binary_cross_entropy_with_logits(
                sem_bev, sem_gt_crop, reduction="none"
            )
            denom = sem_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            loss_sem_bev = (sem_bev_bce * sem_valid).sum(dim=(1, 2, 3)) / denom
            loss_dict["loss_sem_bev"] = loss_sem_bev
            total = total + self.conf.lambda_sem_bev * loss_sem_bev

            #  Map Semantic Loss
            sem_map = pred["sem_map"]  # (B, 9, 256, 256)
            # sem_gt already at canvas size (256, 256)
            loss_sem_map = F.binary_cross_entropy_with_logits(
                sem_map, sem_gt, reduction="none"
            ).mean(dim=(1, 2, 3))  # (B,)
            loss_dict["loss_sem_map"] = loss_sem_map
            total = total + self.conf.lambda_sem_map * loss_sem_map

        res = self.conf.lambda_align * res_l1loss_xyr(
            pred["features_bev_sem"], pred["features_map_sem"],
            xy_gt, yaw_gt, pred["valid_bev"],
        )
        loss_dict["loss_align"] = res
        total = total + res

        # KL Divergence
        if pred["mu_post"] is not None:
            loss_kl = kl_divergence(
                pred["mu_post"], pred["log_sigma_post"],
                pred["mu_prior"], pred["log_sigma_prior"],
            )
            loss_dict["loss_kl"] = loss_kl
            total = total + self.conf.lambda_kl * loss_kl

        loss_dict["total"] = total

        if self.training and self.conf.add_temperature:
            loss_dict["temperature"] = self.temperature.expand(len(nll))

        return loss_dict

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_1m": Location2DRecall(1.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_3m": Location2DRecall(3.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_1°": AngleRecall(1.0, "yaw_max"),
            "yaw_recall_3°": AngleRecall(3.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }
