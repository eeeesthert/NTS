from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from .losses import (
    fusion_smoothness_loss,
    grid_angle_loss,
    grid_edge_length_loss,
    nipple_heatmap_alignment_loss,
    overlap_l1_warp_loss,
    seam_cost_loss,
    seam_overlap_boundary_loss,
)
from .models.fusion import SoftSeamFusionUNet
from .models.warp import WarpStage


class TwoStageStitcher(torch.nn.Module):
    def __init__(
        self,
        encoder_pretrain_source: str = "imagenet",
        encoder_ckpt: str | None = None,
        encoder_radimagenet_url: str | None = None,
        encoder_strict_load: bool = False,
    ):
        super().__init__()
        self.warp_net = WarpStage(
            encoder_pretrain_source=encoder_pretrain_source,
            encoder_ckpt=encoder_ckpt,
            encoder_radimagenet_url=encoder_radimagenet_url,
            encoder_strict_load=encoder_strict_load,
        )
        self.fusion_net = SoftSeamFusionUNet()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_x: torch.Tensor | None = None,
        right_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        warp_out = self.warp_net(left, right, left_x=left_x, right_x=right_x)
        fus_out = self.fusion_net(warp_out["left_warp"], warp_out["right_warp"])
        return {**warp_out, **fus_out}


@dataclass
class LossWeights:
    # Recommended defaults when overlap is large.
    warp_l1: float = 1.0
    grid_edge: float = 4.0
    grid_angle: float = 2.0
    warp_nipple: float = 0.5
    seam_boundary: float = 1.0
    seam_cost: float = 2.0
    fusion_smooth: float = 0.2
    fusion_nipple: float = 0.5


def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    left_x: torch.Tensor,
    right_x: torch.Tensor,
    w: LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    if w is None:
        w = LossWeights()
    # Warp-stage losses (4)
    l_warp_l1 = overlap_l1_warp_loss(outputs["left_warp"], outputs["right_warp"], outputs["overlap"])
    l_edge = grid_edge_length_loss(outputs["control_disp"])
    l_angle = grid_angle_loss(outputs["control_disp"])
    l_nipple_warp = nipple_heatmap_alignment_loss(outputs["left_warp"], outputs["right_warp"], left_x, right_x)

    # Fusion-stage losses (4)
    l_boundary = seam_overlap_boundary_loss(outputs["seam_soft"], outputs["overlap"])
    l_seam_cost = seam_cost_loss(outputs["left_warp"], outputs["right_warp"], outputs["seam_soft"])
    l_smooth = fusion_smoothness_loss(outputs["stitched"])
    l_nipple_fus = nipple_heatmap_alignment_loss(outputs["stitched"], outputs["right_warp"], left_x, right_x)

    total = (
        w.warp_l1 * l_warp_l1
        + w.grid_edge * l_edge
        + w.grid_angle * l_angle
        + w.warp_nipple * l_nipple_warp
        + w.seam_boundary * l_boundary
        + w.seam_cost * l_seam_cost
        + w.fusion_smooth * l_smooth
        + w.fusion_nipple * l_nipple_fus
    )
    return {
        "total": total,
        "warp_l1": l_warp_l1,
        "grid_edge": l_edge,
        "grid_angle": l_angle,
        "warp_nipple": l_nipple_warp,
        "seam_boundary": l_boundary,
        "seam_cost": l_seam_cost,
        "fusion_smooth": l_smooth,
        "fusion_nipple": l_nipple_fus,
    }


def save_stage_results(outputs: dict[str, torch.Tensor], out_dir: str | Path, prefix: str) -> None:
    save_stage_results_with_crop(outputs, out_dir, prefix, auto_crop=True)


def _bbox_from_valid(valid: torch.Tensor, min_size: int = 8) -> tuple[int, int, int, int]:
    ys, xs = torch.where(valid[0, 0] > 0)
    h, w = valid.shape[-2:]
    if ys.numel() == 0:
        return 0, h, 0, w
    y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
    x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
    if (y2 - y1) < min_size:
        yc = (y1 + y2) // 2
        y1 = max(0, yc - min_size // 2)
        y2 = min(h, y1 + min_size)
    if (x2 - x1) < min_size:
        xc = (x1 + x2) // 2
        x1 = max(0, xc - min_size // 2)
        x2 = min(w, x1 + min_size)
    return y1, y2, x1, x2


def save_stage_results_with_crop(
    outputs: dict[str, torch.Tensor],
    out_dir: str | Path,
    prefix: str,
    auto_crop: bool = True,
) -> None:
    out_dir = Path(out_dir)
    warp_dir = out_dir / "warp"
    fusion_dir = out_dir / "fusion"
    warp_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    left = outputs["left_warp"].detach().cpu()
    right = outputs["right_warp"].detach().cpu()
    stitched = outputs["stitched"].detach().cpu()
    mask_right = outputs["mask_right"].detach().cpu()
    seam_soft = outputs["seam_soft"].detach().cpu()

    if auto_crop:
        valid_left = (left.sum(1, keepdim=True) > 0).float()
        valid_right = (right.sum(1, keepdim=True) > 0).float()
        valid_union = torch.clamp(valid_left + valid_right, 0, 1)
        y1, y2, x1, x2 = _bbox_from_valid(valid_union)
        left = left[:, :, y1:y2, x1:x2]
        right = right[:, :, y1:y2, x1:x2]
        stitched = stitched[:, :, y1:y2, x1:x2]
        mask_right = mask_right[:, :, y1:y2, x1:x2]
        seam_soft = seam_soft[:, :, y1:y2, x1:x2]

    mask_left = 1.0 - mask_right
    bin_left = (mask_left > 0.5).float()
    bin_right = (mask_right > 0.5).float()

    save_image(left, warp_dir / f"{prefix}_left.png")
    save_image(right, warp_dir / f"{prefix}_right.png")
    save_image(stitched, fusion_dir / f"{prefix}_stitched.png")
    save_image(seam_soft, fusion_dir / f"{prefix}_seam_soft.png")
    save_image(mask_left, fusion_dir / f"{prefix}_mask_left_soft.png")
    save_image(mask_right, fusion_dir / f"{prefix}_mask_right_soft.png")
    save_image(bin_left, fusion_dir / f"{prefix}_mask_left_bin.png")
    save_image(bin_right, fusion_dir / f"{prefix}_mask_right_bin.png")