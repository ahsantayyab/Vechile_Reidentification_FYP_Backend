"""
feature_extractor.py
TransReID feature extractor for the Vehicle Re-ID backend.

Architecture matches the training server exactly:
  - ResNet50-IBN backbone (torchvision resnet50 with IN layers on layer1-3)
  - TransReID transformer with sie_embed (nn.Parameter, not nn.Embedding)
  - AttributeNet fusion head
  - Plate fusion module (plate_encoder + attn_gate + fusion_proj)

Weights loaded from: outputs/finetune_vehicleid/final_best.pth
(copied to Backend/app/ml/weights/final_best.pth)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

from app.ml.config import ModelConfig

logger = logging.getLogger("app.ml.feature_extractor")


# ── Helpers ───────────────────────────────────────────────────────────────────

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


# ── Backbone (ResNet-50 IBN) ───────────────────────────────────────────────────

class ResNet50IBN(nn.Module):
    """ResNet-50 with Instance-Batch Norm (IBN) on layers 1-3."""
    def __init__(self):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1     = base.conv1
        self.bn1       = base.bn1
        self.relu      = base.relu
        self.maxpool   = base.maxpool
        self.layer1    = base.layer1
        self.layer2    = base.layer2
        self.layer3    = base.layer3
        self.layer4    = base.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # (B, 2048, H', W')


# ── Transformer blocks ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x  = x + self.attn(x2, x2, x2)[0]
        x  = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=2048, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, embed_dim)

    @property
    def num_patches(self):
        return 64   # ResNet50 outputs 8×8 feature map for 256×256 input


class TransReID(nn.Module):
    """Matches training server transreid.py exactly."""
    def __init__(self, in_chans=2048, embed_dim=512, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1,
                 camera_num=21, view_num=1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        self.cam_num   = camera_num
        self.view_num  = view_num
        self.sie_xishu = 1.0
        # sie_embed shape: (camera_num * view_num, 1, embed_dim)
        self.sie_embed = nn.Parameter(
            torch.zeros(camera_num * view_num, 1, embed_dim)
        )

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.sie_embed, std=.02)

    def forward(self, x, cam_label=None):
        B    = x.shape[0]
        x    = self.patch_embed(x)
        cls  = self.cls_token.expand(B, -1, -1)
        x    = torch.cat((cls, x), dim=1)
        x    = x + self.pos_embed

        if cam_label is not None and self.sie_xishu > 0:
            cam_label = cam_label.clamp(0, self.cam_num - 1)
            x = x + self.sie_xishu * self.sie_embed[cam_label]

        x    = self.pos_drop(x)
        x    = self.transformer(x)
        return x[:, 0], x   # global_feat, all_tokens


# ── Full model (matches training AttributeNet_TransReID) ──────────────────────

class AttributeNet_TransReID(nn.Module):
    def __init__(self, num_classes=777, num_attributes=2,
                 attr_classes=None, feature_dim=512, attr_feat_dim=128,
                 transformer_depth=6, transformer_heads=8, camera_num=21):
        super().__init__()
        if attr_classes is None:
            attr_classes = [11, 10]

        self.num_attributes = num_attributes
        self.attr_classes   = attr_classes
        self.backbone       = ResNet50IBN()
        self.in_planes      = 2048

        self.transreid = TransReID(
            in_chans=2048, embed_dim=512,
            depth=transformer_depth, num_heads=transformer_heads,
            camera_num=camera_num,
        )

        self.bottleneck_trans = nn.BatchNorm1d(512)
        self.bottleneck_trans.bias.requires_grad_(False)
        self.fc_trans  = nn.Linear(512, feature_dim, bias=False)
        self.bn_trans  = nn.BatchNorm1d(feature_dim)
        self.bn_trans.bias.requires_grad_(False)
        nn.init.normal_(self.fc_trans.weight, std=0.001)
        self.classifier_trans = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier_trans.weight, std=0.001)

        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.attr_attns = nn.ModuleList([SELayer(self.in_planes, 16) for _ in range(num_attributes)])
        self.attr_fcs   = nn.ModuleList([nn.Linear(self.in_planes, attr_feat_dim, bias=False) for _ in range(num_attributes)])
        self.attr_classifiers = nn.ModuleList([
            nn.Linear(attr_feat_dim, attr_classes[i], bias=False) for i in range(num_attributes)
        ])

        self.attr_combine_conv = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes, 1),
            nn.BatchNorm2d(self.in_planes), nn.ReLU(inplace=True),
        )
        self.attr_distill = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes, 3, padding=1),
            nn.BatchNorm2d(self.in_planes), nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.fc_joint  = nn.Linear(512, feature_dim, bias=False)
        self.bn_joint  = nn.BatchNorm1d(feature_dim)
        self.bn_joint.bias.requires_grad_(False)
        nn.init.normal_(self.fc_joint.weight, std=0.001)
        self.classifier_joint = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier_joint.weight, std=0.001)

    def forward(self, x, cam_label=None):
        B      = x.size(0)
        F_map  = self.backbone(x)

        trans_global, _ = self.transreid(F_map, cam_label)
        trans_bn        = self.bottleneck_trans(trans_global)
        trans_feat      = self.fc_trans(trans_bn)
        trans_feat_bn   = self.bn_trans(trans_feat)

        A_maps = [self.attr_attns[i](F_map) for i in range(self.num_attributes)]
        G_map  = sum(A_maps)
        G_map  = G_map + self.attr_combine_conv(G_map)
        G_reid = self.attr_distill(G_map)
        G_pool = self.gap(G_reid).view(B, -1)

        joint_feat    = trans_global + G_pool
        joint_proj    = self.fc_joint(joint_feat)
        joint_feat_bn = self.bn_joint(joint_proj)

        return F.normalize(joint_feat_bn, p=2, dim=1)


# ── Plate fusion wrapper (optional — used when plate weights available) ────────

class ANetWithPlate(nn.Module):
    """
    Wraps AttributeNet_TransReID with the plate fusion module.
    Used when final_best.pth (finetune_vehicleid) is available.
    Falls back to base model if plate weights missing.
    """
    def __init__(self, base: AttributeNet_TransReID,
                 embed_dim=512, plate_dim=256):
        super().__init__()
        self.base       = base
        self.embed_dim  = embed_dim
        self.plate_dim  = plate_dim

        self.plate_encoder = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, plate_dim), nn.LayerNorm(plate_dim),
        )
        self.attn_gate = nn.Sequential(
            nn.Linear(embed_dim + plate_dim, 128), nn.GELU(),
            nn.Linear(128, 2), nn.Softmax(dim=-1),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim + plate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.classifier = base.classifier_trans

    def forward(self, x, cam_label=None, plate_vecs=None):
        B      = x.size(0)
        F_map  = self.base.backbone(x)
        visual, _ = self.base.transreid(F_map, cam_label)
        visual = self.base.bottleneck_trans(visual)
        visual = self.base.fc_trans(visual)
        visual = self.base.bn_trans(visual)

        if plate_vecs is not None:
            plate_feat  = self.plate_encoder(plate_vecs.to(x.device))
            gate_input  = torch.cat([visual, plate_feat], dim=-1)
            gates       = self.attn_gate(gate_input)
            fused       = self.fusion_proj(gate_input)
            alpha       = gates[:, 1:2]
            out         = (1 - alpha) * visual + alpha * fused
        else:
            out = visual

        return F.normalize(out, p=2, dim=1)


# ── Feature extractor wrapper ─────────────────────────────────────────────────

class TransReIDFeatureExtractor:
    """
    Loads trained weights and extracts 512-dim L2-normalised embeddings.

    Weight loading priority:
      1. final_best.pth  (plate-fused model — best accuracy)
      2. best_model.pth  (base TransReID — fallback)
    """

    def __init__(self, config: ModelConfig):
        self.device     = torch.device(config.device)
        self.batch_size = max(1, config.batch_size)

        base = AttributeNet_TransReID(
            num_classes      = config.transreid_num_classes,
            num_attributes   = config.transreid_num_attributes,
            attr_classes     = config.transreid_attr_classes,
            feature_dim      = config.transreid_feature_dim,
            camera_num       = config.transreid_camera_num,
        )

        # Try plate-fused weights first, fall back to base
        weights_path = config.model_weights_path
        plate_path   = config.plate_weights_path  # new config key

        if plate_path and Path(plate_path).exists():
            self.model = ANetWithPlate(base)
            self._load_weights(self.model, plate_path, "plate-fused")
        elif weights_path and Path(weights_path).exists():
            self.model = base
            self._load_weights(self.model, weights_path, "base TransReID")
        else:
            logger.warning("No weights found — using random weights!")
            self.model = base

        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("TransReIDFeatureExtractor ready",
                    extra={"device": str(self.device)})

    def _load_weights(self, model, path, label):
        try:
            ckpt  = torch.load(str(path), map_location="cpu",
                               weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                f"Loaded {label} weights",
                extra={"path": str(path),
                       "missing": len(missing),
                       "unexpected": len(unexpected)},
            )
        except Exception as exc:
            logger.error(f"Failed to load {label} weights",
                         extra={"error": str(exc)})

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.transform(image)

    @torch.inference_mode()
    def extract(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        crops_list = list(crops)
        if not crops_list:
            return np.zeros((0, 512), dtype=np.float32)

        tensors = [self._preprocess(c) for c in crops_list]
        feats: list[np.ndarray] = []

        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i: i + self.batch_size]).to(self.device)
            emb   = self.model(batch).detach().cpu().numpy()
            feats.append(emb)

        stacked = np.vstack(feats)
        norms   = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-12
        return stacked / norms
