from __future__ import annotations

import torch
import torch.nn.functional as F


def nipple_x_heatmap(x_pos: torch.Tensor, width: int, height: int, sigma: float = 10.0) -> torch.Tensor:
    b = x_pos.shape[0]
    xs = torch.arange(width, device=x_pos.device).float().view(1, 1, width)
    line = torch.exp(-0.5 * ((xs - x_pos.view(b, 1, 1)) / sigma) ** 2)
    return line.repeat(1, height, 1).unsqueeze(1)


# -----------------------------
# Warp-stage losses
# -----------------------------
def overlap_l1_warp_loss(left_warp: torch.Tensor, right_warp: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    l1 = (left_warp - right_warp).abs() * overlap
    return l1.sum() / overlap.sum().clamp_min(1.0)


def grid_edge_length_loss(control_disp: torch.Tensor) -> torch.Tensor:
    """Control-grid edge length regularization (avoid over-stretching/compression)."""
    b, _, gh, gw = control_disp.shape
    device, dtype = control_disp.device, control_disp.dtype
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, gh, device=device, dtype=dtype),
        torch.linspace(-1, 1, gw, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=0).unsqueeze(0)
    deformed = base + control_disp

    dx = deformed[:, :, :, 1:] - deformed[:, :, :, :-1]
    dy = deformed[:, :, 1:, :] - deformed[:, :, :-1, :]

    dx_len = torch.sqrt((dx**2).sum(1) + 1e-6)
    dy_len = torch.sqrt((dy**2).sum(1) + 1e-6)
    target_dx = 2.0 / (gw - 1)
    target_dy = 2.0 / (gh - 1)
    return (dx_len - target_dx).abs().mean() + (dy_len - target_dy).abs().mean()


def grid_angle_loss(control_disp: torch.Tensor) -> torch.Tensor:
    """Encourage near-right-angle local grid (horizontal vs vertical edge orthogonality)."""
    b, _, gh, gw = control_disp.shape
    device, dtype = control_disp.device, control_disp.dtype
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, gh, device=device, dtype=dtype),
        torch.linspace(-1, 1, gw, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=0).unsqueeze(0)
    p = base + control_disp

    px = p[:, :, :-1, 1:] - p[:, :, :-1, :-1]
    py = p[:, :, 1:, :-1] - p[:, :, :-1, :-1]

    dot = (px * py).sum(1)
    nx = torch.sqrt((px**2).sum(1) + 1e-6)
    ny = torch.sqrt((py**2).sum(1) + 1e-6)
    cos = dot / (nx * ny + 1e-6)
    return cos.abs().mean()


def nipple_heatmap_alignment_loss(
    left_warp: torch.Tensor,
    right_warp: torch.Tensor,
    left_x: torch.Tensor,
    right_x: torch.Tensor,
) -> torch.Tensor:
    """Nipple-related supervision via x-only heatmap-weighted local mismatch."""
    _, _, h, w = left_warp.shape
    hm = torch.maximum(
        nipple_x_heatmap(left_x, width=w, height=h),
        nipple_x_heatmap(right_x, width=w, height=h),
    )
    diff = (left_warp - right_warp).abs().mean(1, keepdim=True)
    return (diff * hm).mean()


# -----------------------------
# Fusion-stage losses
# -----------------------------
def seam_overlap_boundary_loss(seam_soft: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Restrict seam inside overlap: penalize seam activation outside overlap."""
    outside = seam_soft * (1.0 - overlap)
    return outside.mean()


def seam_cost_loss(left_warp: torch.Tensor, right_warp: torch.Tensor, seam_soft: torch.Tensor) -> torch.Tensor:
    """
    Push seam toward low-difference regions:
    cost map = squared difference, weighted by seam gradients.
    """
    cost = (left_warp - right_warp).pow(2).mean(1, keepdim=True)
    gx = (seam_soft[:, :, :, 1:] - seam_soft[:, :, :, :-1]).abs()
    gy = (seam_soft[:, :, 1:, :] - seam_soft[:, :, :-1, :]).abs()
    cx = cost[:, :, :, 1:]
    cy = cost[:, :, 1:, :]
    return (gx * cx).mean() + (gy * cy).mean()


def fusion_smoothness_loss(stitched: torch.Tensor) -> torch.Tensor:
    dx = (stitched[:, :, :, 1:] - stitched[:, :, :, :-1]).abs().mean()
    dy = (stitched[:, :, 1:, :] - stitched[:, :, :-1, :]).abs().mean()
    return dx + dy
