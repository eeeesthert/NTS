from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftSeamFusionUNet(nn.Module):
    """
    Dilated-conv UNet.
    At skip levels, use |left-right| difference features and concatenate into decoder.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(6, 64, dilation=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128, dilation=2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ConvBlock(128, 256, dilation=4)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 128 + 128, 128, dilation=2)  # up + skip + diff
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 64 + 64, 64, dilation=1)

        self.seam_head = nn.Conv2d(64, 1, 1)
        self.mask_refine = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, left_warp: torch.Tensor, right_warp: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([left_warp, right_warp], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))

        up2 = self.up2(b)
        if up2.shape[-2:] != e2.shape[-2:]:
            up2 = F.interpolate(up2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        diff2 = (e2 - up2).abs()
        d2 = self.dec2(torch.cat([up2, e2, diff2], dim=1))

        up1 = self.up1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = F.interpolate(up1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        diff1 = (e1 - up1).abs()
        d1 = self.dec1(torch.cat([up1, e1, diff1], dim=1))

        seam_soft = torch.sigmoid(self.seam_head(d1))

        valid_left = (left_warp.sum(1, keepdim=True) > 0).float()
        valid_right = (right_warp.sum(1, keepdim=True) > 0).float()
        overlap = valid_left * valid_right

        # fuse seam mask with warp masks (valid supports)
        seam_in_overlap = seam_soft * overlap
        base_right = seam_in_overlap * valid_right
        refine_right = torch.sigmoid(self.mask_refine(torch.cat([base_right, valid_right], dim=1))) * valid_right

        mask_right = refine_right
        mask_left = (1.0 - refine_right) * valid_left
        norm = (mask_left + mask_right).clamp_min(1e-6)
        mask_left = mask_left / norm
        mask_right = mask_right / norm

        stitched = mask_left * left_warp + mask_right * right_warp

        return {
            "mask_left": mask_left,
            "mask_right": mask_right,
            "seam_soft": seam_soft,
            "overlap": overlap,
            "stitched": stitched,
        }
