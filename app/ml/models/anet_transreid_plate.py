"""
anet_transreid_plate.py
Wraps your existing AttributeNet_TransReID with a plate fusion module.

Architecture:
  - backbone (ResNet50-IBN) + transreid transformer  ← your existing model
  - plate_encoder: hash(text) → 512 → 256-d vector   ← new
  - attn_gate: learns visual vs plate weighting        ← new
  - fusion_proj: (512 + 256) → 512                    ← new
  - classifier_trans stays intact for ID loss          ← unchanged
"""

import torch
import torch.nn as nn
from pathlib import Path


class ANetTransReIDWithPlate(nn.Module):
    """
    Wraps AttributeNet_TransReID and adds a plate fusion branch.
    The existing model is loaded as-is; only the new plate modules are added.
    """

    def __init__(
        self,
        base_model,               # loaded AttributeNet_TransReID instance
        plate_weights: str = 'weights/plate_detector_best.pt',
        embed_dim:     int = 512,
        plate_dim:     int = 256,
        device:        str = 'cuda',
    ):
        super().__init__()

        self.base    = base_model   # full existing model
        self.embed_dim = embed_dim
        self.plate_dim = plate_dim
        self.device    = device

        # ── plate text encoder (hash dim 512 → plate_dim 256) ─────────────────
        self.plate_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, plate_dim),
            nn.LayerNorm(plate_dim),
        )

        # ── attention gate: scalar weights for visual vs plate ─────────────────
        self.attn_gate = nn.Sequential(
            nn.Linear(embed_dim + plate_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        # ── fusion projection back to embed_dim ───────────────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim + plate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── re-use existing classifier head ───────────────────────────────────
        # points to base_model.classifier_trans — no copy, same weights
        self.classifier = base_model.classifier_trans

        # ── plate detector (frozen inference module) ──────────────────────────
        from plate_detection.plate_detector import PlateDetector
        self.plate_detector = PlateDetector(plate_weights, device=device, conf=0.15)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, cam_label=None, plate_texts=None):
        """
        Args:
            x           : (B, C, H, W) image batch
            cam_label   : (B,) camera IDs  — passed to base model
            plate_texts : list[str] length B, empty string = no plate

        Returns:
            fused_feat  : (B, embed_dim)  use for triplet + classifier
            visual_feat : (B, embed_dim)  use for auxiliary triplet loss
            logits      : (B, num_classes) from classifier_trans
        """
        # ── visual branch via existing model ──────────────────────────────────
        # We call the base model's internal pipeline up to the feature vector.
        # AttributeNet_TransReID.forward returns (trans_logits, attr_logits_list)
        # We need the intermediate feature — so we replicate the relevant path.

        B = x.shape[0]

        # ResNet backbone → (B, 2048, H', W')
        resnet_feat = self.base.backbone(x)                   # (B, 2048, H', W')

        # TransReID transformer → cls token → (B, 512)
        visual_feat = self.base.transreid(resnet_feat, cam_label=cam_label)
        # transreid forward returns the cls token (B, embed_dim)

        # bottleneck
        visual_feat = self.base.bottleneck_trans(visual_feat)  # (B, 512)

        # fc projection
        visual_feat = self.base.fc_trans(visual_feat)          # (B, feature_dim=512)
        visual_feat = self.base.bn_trans(visual_feat)          # (B, 512)

        # ── plate branch ──────────────────────────────────────────────────────
        if plate_texts is not None and any(t for t in plate_texts):
            plate_vecs = self._encode_texts(plate_texts, x.device)  # (B, 512)
            plate_feat = self.plate_encoder(plate_vecs)              # (B, 256)

            # attention gate
            gate_input = torch.cat([visual_feat, plate_feat], dim=-1)  # (B, 768)
            gates      = self.attn_gate(gate_input)                    # (B, 2)

            # fusion: project concat back to 512
            fused_feat = self.fusion_proj(gate_input)                  # (B, 512)

            # soft gate: interpolate between visual-only and fused
            alpha      = gates[:, 1:2]                                 # plate weight
            fused_feat = (1 - alpha) * visual_feat + alpha * fused_feat

        else:
            fused_feat = visual_feat

        # ── classifier ────────────────────────────────────────────────────────
        logits = self.classifier(fused_feat)                   # (B, num_classes)

        return fused_feat, visual_feat, logits

    def _encode_texts(self, plate_texts: list, device) -> torch.Tensor:
        """Convert list of plate strings → (B, 512) hash embedding tensor."""
        vecs = [self.plate_detector.get_plate_embedding(t, dim=512)
                for t in plate_texts]
        return torch.stack(vecs).to(device)

    def run_plate_detection(self, images_np: list) -> list:
        """
        Run detector + OCR on raw numpy images.
        Returns list of plate text strings.
        """
        texts = []
        for img in images_np:
            _, _, text = self.plate_detector.detect(img)
            texts.append(text)
        return texts


# ── convenience loader ────────────────────────────────────────────────────────

def build_model_with_plate(
    transreid_checkpoint: str,
    plate_weights:        str  = 'weights/plate_detector_best.pt',
    embed_dim:            int  = 512,
    num_classes:          int  = 777,
    device:               str  = 'cuda',
):
    """
    Load best_model.pth and wrap with plate fusion.

    Args:
        transreid_checkpoint : path to weights/best_model.pth
        plate_weights        : path to weights/plate_detector_best.pt
        embed_dim            : must match config.TRANSFORMER_EMBED_DIM (512)
        num_classes          : config.NUM_CLASSES (777)
        device               : 'cuda' or 'cpu'

    Returns:
        ANetTransReIDWithPlate ready for fine-tuning
    """
    from src.config import Config
    from models.anet_transreid import AttributeNet_TransReID

    cfg = Config()

    # ── build base model with exact same args as train.py ────────────────────
    base = AttributeNet_TransReID(
        num_classes      = cfg.NUM_CLASSES,
        num_attributes   = cfg.NUM_ATTRIBUTES,
        attr_classes     = cfg.ATTRIBUTE_CLASSES,
        feature_dim      = cfg.FEATURE_DIM,
        attr_feat_dim    = cfg.ATTR_FEATURE_DIM,
        transformer_depth = cfg.TRANSFORMER_DEPTH,
        transformer_heads = cfg.TRANSFORMER_HEADS,
    )

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(transreid_checkpoint, map_location='cpu')

    # handle DataParallel wrapper (keys start with 'module.')
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}

    missing, unexpected = base.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {transreid_checkpoint}")
    if missing:
        print(f"  Missing keys  : {len(missing)}  "
              f"(first 3: {missing[:3]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # ── wrap with plate fusion ────────────────────────────────────────────────
    model = ANetTransReIDWithPlate(
        base_model    = base,
        plate_weights = plate_weights,
        embed_dim     = embed_dim,
        device        = device,
    ).to(device)

    base_params  = sum(p.numel() for p in base.parameters())
    plate_params = (sum(p.numel() for p in model.plate_encoder.parameters()) +
                    sum(p.numel() for p in model.attn_gate.parameters()) +
                    sum(p.numel() for p in model.fusion_proj.parameters()))

    print(f"Plate fusion module attached.")
    print(f"  Base model params  : {base_params:,}")
    print(f"  New plate params   : {plate_params:,}")
    print(f"  Total              : {base_params + plate_params:,}")

    return model
