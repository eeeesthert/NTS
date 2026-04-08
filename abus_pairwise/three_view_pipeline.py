from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import (
    fusion_smoothness_loss,
    grid_angle_loss,
    grid_edge_length_loss,
    nipple_heatmap_alignment_loss,
    overlap_l1_warp_loss,
    seam_cost_loss,
    seam_overlap_boundary_loss,
)
from .pipeline import TwoStageStitcher


def _pad_to_width(x: torch.Tensor, width: int) -> torch.Tensor:
    if x.shape[-1] == width:
        return x
    pad_left = (width - x.shape[-1]) // 2
    pad_right = width - x.shape[-1] - pad_left
    return F.pad(x, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)


@dataclass
class ThreeViewLossWeights:
    warp_l1: float = 1.0
    grid_edge: float = 4.0
    grid_angle: float = 2.0
    warp_nipple: float = 0.5
    seam_boundary: float = 1.0
    seam_cost: float = 2.0
    fusion_smooth: float = 0.2
    fusion_nipple: float = 0.5
    fixed_consistency: float = 0.5
    cross13_warp: float = 0.3
    cross13_nipple: float = 0.2


class ThreeViewFixedCenterStitcher(torch.nn.Module):
    """
    input2 as fixed view:
      - warp input1 -> input2 coordinates
      - warp input3 -> input2 coordinates
      - fuse (input1_warp, input2, input3_warp)
    """

    def __init__(self, pair_model: TwoStageStitcher):
        super().__init__()
        self.pair_model = pair_model

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        input3: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b12 = self.pair_model.warp_net(input1, input2, left_x=x1, right_x=x2)
        b32 = self.pair_model.warp_net(input3, input2, left_x=x3, right_x=x2)

        target_w = max(b12["left_warp"].shape[-1], b32["left_warp"].shape[-1])
        i1 = _pad_to_width(b12["left_warp"], target_w)
        i2a = _pad_to_width(b12["right_warp"], target_w)
        i3 = _pad_to_width(b32["left_warp"], target_w)
        i2b = _pad_to_width(b32["right_warp"], target_w)

        fixed = 0.5 * (i2a + i2b)
        fus12 = self.pair_model.fusion_net(i1, fixed)
        fus32 = self.pair_model.fusion_net(i3, fixed)

        w1 = fus12["mask_left"]
        w3 = fus32["mask_left"]
        w2 = 0.5 * (fus12["mask_right"] + fus32["mask_right"])
        wn = (w1 + w2 + w3).clamp_min(1e-6)
        w1, w2, w3 = w1 / wn, w2 / wn, w3 / wn
        stitched = w1 * i1 + w2 * fixed + w3 * i3
        overlap13 = ((i1.sum(1, keepdim=True) > 0) & (i3.sum(1, keepdim=True) > 0)).float()

        return {
            "input1_warp": i1,
            "input2_fixed_12": i2a,
            "input2_fixed_32": i2b,
            "input2_fixed": fixed,
            "input3_warp": i3,
            "weights_1": w1,
            "weights_2": w2,
            "weights_3": w3,
            "seam_12": fus12["seam_soft"],
            "seam_32": fus32["seam_soft"],
            "overlap_12": b12["overlap"],
            "overlap_32": b32["overlap"],
            "overlap_13": overlap13,
            "control_disp_12": b12["control_disp"],
            "control_disp_32": b32["control_disp"],
            "stitched": stitched,
        }


def compute_three_view_total_loss(
    outputs: dict[str, torch.Tensor],
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    w: ThreeViewLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    if w is None:
        w = ThreeViewLossWeights()

    l_warp = 0.5 * (
        overlap_l1_warp_loss(outputs["input1_warp"], outputs["input2_fixed"], outputs["overlap_12"])
        + overlap_l1_warp_loss(outputs["input3_warp"], outputs["input2_fixed"], outputs["overlap_32"])
    )
    l_edge = 0.5 * (grid_edge_length_loss(outputs["control_disp_12"]) + grid_edge_length_loss(outputs["control_disp_32"]))
    l_angle = 0.5 * (grid_angle_loss(outputs["control_disp_12"]) + grid_angle_loss(outputs["control_disp_32"]))
    l_nipple_warp = 0.5 * (
        nipple_heatmap_alignment_loss(outputs["input1_warp"], outputs["input2_fixed"], x1, x2)
        + nipple_heatmap_alignment_loss(outputs["input3_warp"], outputs["input2_fixed"], x3, x2)
    )

    l_boundary = 0.5 * (
        seam_overlap_boundary_loss(outputs["seam_12"], outputs["overlap_12"])
        + seam_overlap_boundary_loss(outputs["seam_32"], outputs["overlap_32"])
    )
    l_seam_cost = 0.5 * (
        seam_cost_loss(outputs["input1_warp"], outputs["input2_fixed"], outputs["seam_12"])
        + seam_cost_loss(outputs["input3_warp"], outputs["input2_fixed"], outputs["seam_32"])
    )
    l_smooth = fusion_smoothness_loss(outputs["stitched"])
    l_nipple_fus = nipple_heatmap_alignment_loss(outputs["stitched"], outputs["input2_fixed"], x2, x2)
    l_fixed_consistency = F.l1_loss(outputs["input2_fixed_12"], outputs["input2_fixed_32"])
    l_cross13_warp = overlap_l1_warp_loss(outputs["input1_warp"], outputs["input3_warp"], outputs["overlap_13"])
    l_cross13_nipple = nipple_heatmap_alignment_loss(outputs["input1_warp"], outputs["input3_warp"], x1, x3)

    total = (
        w.warp_l1 * l_warp
        + w.grid_edge * l_edge
        + w.grid_angle * l_angle
        + w.warp_nipple * l_nipple_warp
        + w.seam_boundary * l_boundary
        + w.seam_cost * l_seam_cost
        + w.fusion_smooth * l_smooth
        + w.fusion_nipple * l_nipple_fus
        + w.fixed_consistency * l_fixed_consistency
        + w.cross13_warp * l_cross13_warp
        + w.cross13_nipple * l_cross13_nipple
    )
    return {
        "total": total,
        "warp_l1": l_warp,
        "grid_edge": l_edge,
        "grid_angle": l_angle,
        "warp_nipple": l_nipple_warp,
        "seam_boundary": l_boundary,
        "seam_cost": l_seam_cost,
        "fusion_smooth": l_smooth,
        "fusion_nipple": l_nipple_fus,
        "fixed_consistency": l_fixed_consistency,
        "cross13_warp": l_cross13_warp,
        "cross13_nipple": l_cross13_nipple,
    }
