from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List

import cv2
import torch
from torch.utils.data import Dataset


@dataclass
class PairSample:
    case_dir: Path
    left_path: Path
    right_path: Path


class ABUSPairDataset(Dataset):
    """
    Expects layout:
      dataset/case001/input1.jpg
      dataset/case001/input2.jpg
      dataset/case001/input3.jpg
      dataset/case001/nipple_x.txt   # [x1,x2,x3]
    """

    def __init__(
        self,
        root: str | Path,
        stage: str = "12",
        image_size: int | None = 512,
        augment: bool = False,
        hflip_prob: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
    ):
        self.root = Path(root)
        self.stage = stage
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.samples = self._scan_cases()

    @staticmethod
    def _slice_stem_to_id(stem: str) -> str:
        # "slice_0001" -> "0001", fallback to original stem
        if stem.startswith("slice_"):
            return stem.split("slice_", 1)[1]
        return stem

    @staticmethod
    def _collect_images(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            return []
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files: List[Path] = []
        for ext in exts:
            files.extend(sorted(path.glob(ext)))
        return files

    @staticmethod
    def _view_index(path: Path) -> int:
        if path.parent.name.startswith("input"):
            return int(path.parent.name.replace("input", "")) - 1
        if path.stem.startswith("input"):
            return int(path.stem.replace("input", "")) - 1
        raise ValueError(f"Cannot infer view index from path: {path}")

    def _scan_cases(self) -> List[PairSample]:
        # Support both:
        # 1) case001/input1.jpg, input2.jpg, input3.jpg
        # 2) case001/input1/slice_xxx.jpg, input2/slice_xxx.jpg, input3/slice_xxx.jpg
        pairs = {"12": ("input1", "input2"), "23": ("input2", "input3")}
        left_name, right_name = pairs[self.stage]
        samples: List[PairSample] = []
        for case_dir in sorted(self.root.glob("case*")):
            if not (case_dir / "nipple_x.txt").exists():
                continue

            left_candidates = self._collect_images(case_dir / f"{left_name}.jpg")
            right_candidates = self._collect_images(case_dir / f"{right_name}.jpg")

            # directory mode fallback
            if not left_candidates:
                left_candidates = self._collect_images(case_dir / left_name)
            if not right_candidates:
                right_candidates = self._collect_images(case_dir / right_name)

            if not left_candidates or not right_candidates:
                continue

            left_map = {self._slice_stem_to_id(p.stem): p for p in left_candidates}
            right_map = {self._slice_stem_to_id(p.stem): p for p in right_candidates}
            common_ids = sorted(set(left_map).intersection(right_map))

            if common_ids:
                for sid in common_ids:
                    samples.append(PairSample(case_dir, left_map[sid], right_map[sid]))
            else:
                # If naming does not match, align by sorted index.
                for lp, rp in zip(sorted(left_candidates), sorted(right_candidates)):
                    samples.append(PairSample(case_dir, lp, rp))
        return samples

    @staticmethod
    def _read_nipple_x(path: Path) -> List[float]:
        raw = path.read_text(encoding="utf-8").strip().strip("[]")
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(vals) != 3:
            raise ValueError(f"nipple_x.txt must have 3 values: {path}")
        return vals

    def _load_img(self, p: Path) -> torch.Tensor:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size is not None and self.image_size > 0:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        nipple_x = self._read_nipple_x(sample.case_dir / "nipple_x.txt")
        left = self._load_img(sample.left_path)
        right = self._load_img(sample.right_path)

        # input1/2/3 -> index 0/1/2
        left_idx = self._view_index(sample.left_path)
        right_idx = self._view_index(sample.right_path)

        left_x = torch.tensor([nipple_x[left_idx]], dtype=torch.float32)
        right_x = torch.tensor([nipple_x[right_idx]], dtype=torch.float32)

        if self.augment:
            left, right, left_x, right_x = self._apply_pair_augment(left, right, left_x, right_x)

        return {
            "left": left,
            "right": right,
            "left_x": left_x,
            "right_x": right_x,
            "case": sample.case_dir.name,
            "left_path": str(sample.left_path),
            "right_path": str(sample.right_path),
        }

    def _apply_pair_augment(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_x: torch.Tensor,
        right_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = left.shape

        # Horizontal flip (same transform on both views)
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            left = torch.flip(left, dims=[2])
            right = torch.flip(right, dims=[2])
            left_x = (w - 1) - left_x
            right_x = (w - 1) - right_x

        # Brightness / contrast jitter
        if self.brightness_jitter > 0:
            bdelta = (random.random() * 2 - 1) * self.brightness_jitter
            left = (left + bdelta).clamp(0, 1)
            right = (right + bdelta).clamp(0, 1)
        if self.contrast_jitter > 0:
            cscale = 1.0 + (random.random() * 2 - 1) * self.contrast_jitter
            left = ((left - 0.5) * cscale + 0.5).clamp(0, 1)
            right = ((right - 0.5) * cscale + 0.5).clamp(0, 1)

        return left, right, left_x, right_x