from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNet50MultiScale


def _build_dlt_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src, dst: (B, 4, 2) in normalized coords [-1, 1]
    returns H: (B, 3, 3), mapping src -> dst
    """
    b = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    a1 = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    a2 = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    a = torch.stack([a1, a2], dim=2).reshape(b, 8, 8)
    rhs = torch.stack([u, v], dim=2).reshape(b, 8, 1)

    h8 = torch.linalg.solve(a, rhs).squeeze(-1)
    h9 = torch.cat([h8, torch.ones(b, 1, device=src.device, dtype=src.dtype)], dim=1)
    return h9.view(b, 3, 3)


def _warp_grid_by_h(grid: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """grid: (B,H,W,2) output coords; apply inverse homography to sample source coords."""
    b, hh, ww, _ = grid.shape
    xy1 = torch.cat([grid, torch.ones(b, hh, ww, 1, device=grid.device, dtype=grid.dtype)], dim=-1)
    h_inv = torch.inverse(h)
    xy1 = xy1.view(b, -1, 3).transpose(1, 2)
    mapped = torch.bmm(h_inv, xy1).transpose(1, 2)
    mapped_xy = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-6)
    return mapped_xy.view(b, hh, ww, 2)


class FCA(nn.Module):
    """Feature Correlation Aggregation on 1/16-scale features."""

    def __init__(self, c: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2 + 1, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_l: torch.Tensor, f_r: torch.Tensor) -> torch.Tensor:
        fl = F.normalize(f_l, dim=1)
        fr = F.normalize(f_r, dim=1)
        corr = (fl * fr).sum(1, keepdim=True)
        return self.proj(torch.cat([f_l, f_r, corr], dim=1))


class RR(nn.Module):
    """Regression block for 4-corner vertex offsets."""

    def __init__(self, c: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class LocalGridHead(nn.Module):
    def __init__(self, c: int = 512, gh: int = 9, gw: int = 9):
        super().__init__()
        self.gh, self.gw = gh, gw
        self.net = nn.Sequential(
            nn.Conv2d(c * 2, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2 * gh * gw, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, f_l_8: torch.Tensor, f_r_8_w: torch.Tensor) -> torch.Tensor:
        b = f_l_8.shape[0]
        x = self.net(torch.cat([f_l_8, f_r_8_w], dim=1)).view(b, 2, self.gh, self.gw)
        return x


class WarpStage(nn.Module):
    def __init__(
        self,
        encoder_pretrain_source: str = "imagenet",
        encoder_ckpt: str | None = None,
        encoder_radimagenet_url: str | None = None,
        encoder_strict_load: bool = False,
        grid_h: int = 9,
        grid_w: int = 9,
    ):
        super().__init__()
        self.encoder = ResNet50MultiScale(
            pretrain_source=encoder_pretrain_source,
            checkpoint_path=encoder_ckpt,
            radimagenet_url=encoder_radimagenet_url,
            strict_load=encoder_strict_load,
        )
        self.fca = FCA(c=1024)
        self.rr = RR(c=256)
        self.local_head = LocalGridHead(c=512, gh=grid_h, gw=grid_w)
        self.grid_h, self.grid_w = grid_h, grid_w

    @staticmethod
    def _base_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    def _rbf_interpolate(self, ctrl_disp: torch.Tensor, h: int, w: int, sigma: float = 0.25) -> torch.Tensor:
        """control disp (B,2,Gh,Gw) -> dense field (B,2,H,W) via Gaussian RBF."""
        b, _, gh, gw = ctrl_disp.shape
        device, dtype = ctrl_disp.device, ctrl_disp.dtype

        cy, cx = torch.meshgrid(
            torch.linspace(-1, 1, gh, device=device, dtype=dtype),
            torch.linspace(-1, 1, gw, device=device, dtype=dtype),
            indexing="ij",
        )
        ctrl_xy = torch.stack([cx, cy], dim=-1).view(1, gh * gw, 2)
        disp = ctrl_disp.permute(0, 2, 3, 1).reshape(b, gh * gw, 2)

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        pts = torch.stack([x, y], dim=-1).view(1, h * w, 1, 2)
        ctrl = ctrl_xy.view(1, 1, gh * gw, 2)
        d2 = ((pts - ctrl) ** 2).sum(-1)
        wgt = torch.exp(-0.5 * d2 / (sigma**2))
        wgt = wgt / (wgt.sum(-1, keepdim=True) + 1e-6)

        dense = torch.einsum("bpk,bkc->bpc", wgt.repeat(b, 1, 1), disp)
        return dense.view(b, h, w, 2).permute(0, 3, 1, 2)

    @staticmethod
    def _place_on_canvas(img: torch.Tensor, x_pos: torch.Tensor, out_w: int) -> torch.Tensor:
        """
        Re-center image so nipple x is at canvas center.
        img: (B,C,H,W), x_pos: (B,1) in pixel coordinate of input img.
        """
        b, _, h, w = img.shape
        device, dtype = img.device, img.dtype
        center = (out_w - 1) / 2.0
        shift = center - x_pos.view(-1, 1, 1)  # output_x = input_x + shift

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(out_w, device=device, dtype=dtype),
            indexing="ij",
        )
        xx = xx.unsqueeze(0).repeat(b, 1, 1)
        yy = yy.unsqueeze(0).repeat(b, 1, 1)

        # inverse map output->input
        xin = xx - shift
        yin = yy

        xnorm = 2.0 * xin / max(w - 1, 1) - 1.0
        ynorm = 2.0 * yin / max(h - 1, 1) - 1.0
        grid = torch.stack([xnorm, ynorm], dim=-1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_x: torch.Tensor | None = None,
        right_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        f_l = self.encoder(left)
        f_r = self.encoder(right)

        # Stage-1: global homography (1/16 features, FCA + RR + DLT)
        corr_feat = self.fca(f_l[2], f_r[2])
        corner_off = self.rr(corr_feat).view(-1, 4, 2).tanh() * 0.2

        b = left.shape[0]
        base_corners = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            device=left.device,
            dtype=left.dtype,
        ).unsqueeze(0).repeat(b, 1, 1)
        dst_corners = base_corners + corner_off
        h_c = _build_dlt_homography(base_corners, dst_corners)

        # apply coarse H to 1/8 target feature (high-res alignment stage)
        b2, _, h8, w8 = f_r[1].shape
        grid8 = self._base_grid(b2, h8, w8, right.device, right.dtype)
        grid8_h = _warp_grid_by_h(grid8, h_c)
        f_r_8_w = F.grid_sample(f_r[1], grid8_h, mode="bilinear", padding_mode="zeros", align_corners=False)

        # Stage-2: local grid deformation + Gaussian RBF
        ctrl_disp = self.local_head(f_l[1], f_r_8_w).tanh() * 0.12

        _, _, h, w = left.shape
        dense_delta = self._rbf_interpolate(ctrl_disp, h=h, w=w)

        base_grid = self._base_grid(b, h, w, left.device, left.dtype)
        grid_h = _warp_grid_by_h(base_grid, h_c)
        final_grid = grid_h + dense_delta.permute(0, 2, 3, 1)

        left_w = F.grid_sample(left, final_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        right_w = right

        # Fusion FOV: expanded canvas with nipple x around output center.
        if left_x is not None and right_x is not None:
            _, _, _, w0 = left_w.shape
            lx = left_x.view(-1, 1).to(left_w.dtype)
            rx = right_x.view(-1, 1).to(right_w.dtype)
            # preserve non-overlap on both sides for each view
            half = torch.maximum(torch.maximum(lx, w0 - lx), torch.maximum(rx, w0 - rx))
            out_w = int(torch.ceil((half.max() * 2.0) + 8.0).item())
            out_w = max(out_w, w0)
            left_w = self._place_on_canvas(left_w, lx, out_w=out_w)
            right_w = self._place_on_canvas(right_w, rx, out_w=out_w)
        overlap = ((left_w.sum(1, keepdim=True) > 0) & (right_w.sum(1, keepdim=True) > 0)).float()

        return {
            "left_warp": left_w,
            "right_warp": right_w,
            "grid": final_grid,
            "overlap": overlap,
            "homography": h_c,
            "corner_offsets": corner_off,
            "control_disp": ctrl_disp,
        }