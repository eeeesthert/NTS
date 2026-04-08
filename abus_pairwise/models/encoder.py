from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # Maybe already a state-dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]
    raise ValueError("Cannot parse checkpoint format: expected a state_dict-like object.")


def _adapt_resnet50_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        # common wrappers
        for prefix in ("module.", "encoder.", "backbone.", "resnet.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        if nk.startswith("fc."):
            continue
        out[nk] = v
    return out


class ResNet50MultiScale(nn.Module):
    """ResNet50 feature pyramid compatible with warp/fusion stages."""

    def __init__(
        self,
        pretrain_source: str = "imagenet",
        checkpoint_path: str | None = None,
        radimagenet_url: str | None = None,
        strict_load: bool = False,
    ):
        super().__init__()
        source = pretrain_source.lower()
        if source not in {"imagenet", "radimagenet", "local", "none"}:
            raise ValueError(f"Unsupported pretrain source: {pretrain_source}")

        # ImageNet source can download automatically from torchvision.
        # For radimagenet, we support optional URL-based auto-download via env var.
        if source in {"imagenet", "radimagenet"}:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        base = resnet50(weights=weights)
        if source in {"radimagenet", "local"}:
            if source == "radimagenet" and checkpoint_path is None:
                url = (radimagenet_url or "").strip() or os.getenv("RADIMAGENET_RESNET50_URL", "").strip()
                if url:
                    print(f"[encoder] downloading radimagenet checkpoint from: {url}")
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
                    sd = _adapt_resnet50_keys(_extract_state_dict(ckpt))
                    missing, unexpected = base.load_state_dict(sd, strict=strict_load)
                    if len(unexpected) > 0:
                        print(f"[encoder] unexpected keys ignored: {len(unexpected)}")
                    if len(missing) > 0:
                        print(f"[encoder] missing keys after load: {len(missing)}")
                    checkpoint_path = "__downloaded__"
                else:
                    print(
                        "[encoder] RADIMAGENET_RESNET50_URL is not set, "
                        "fallback to torchvision ImageNet weights. "
                        "Set --encoder-ckpt or env var to use real RadImageNet weights."
                    )
                    checkpoint_path = "__imagenet_fallback__"
            if checkpoint_path is None:
                raise ValueError(f"{source} source requires --encoder-ckpt path.")
            if checkpoint_path not in {"__downloaded__", "__imagenet_fallback__"}:
                ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
                sd = _adapt_resnet50_keys(_extract_state_dict(ckpt))
                missing, unexpected = base.load_state_dict(sd, strict=strict_load)
                if len(unexpected) > 0:
                    print(f"[encoder] unexpected keys ignored: {len(unexpected)}")
                if len(missing) > 0:
                    print(f"[encoder] missing keys after load: {len(missing)}")

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)   # 1/4
        f2 = self.layer2(f1)  # 1/8
        f3 = self.layer3(f2)  # 1/16
        f4 = self.layer4(f3)  # 1/32
        return [f1, f2, f3, f4]