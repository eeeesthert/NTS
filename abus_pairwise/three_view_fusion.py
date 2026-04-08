from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m.astype(np.float32) / 255.0)
    return m[..., None]  


def _read_soft_or_bin_mask(fusion_dir: Path, prefix: str, side: str) -> np.ndarray:
    soft = fusion_dir / f"{prefix}_mask_{side}_soft.png"
    if soft.exists():
        return _read_mask(soft)
    binary = fusion_dir / f"{prefix}_mask_{side}_bin.png"
    return _read_mask(binary)


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    img = np.asarray(img).astype(np.float32)

    while img.ndim > 3:
        img = np.squeeze(img, axis=-1)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _gaussian_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [x]
    cur = x
    for _ in range(levels - 1):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _laplacian_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gaussian_pyramid(x, levels)
    l = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        l.append(g[i] - up)
    l.append(g[-1])
    return l


def _reconstruct_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(lap[i].shape[1], lap[i].shape[0])) + lap[i]
    return out


def gaussian_pyramid_blend(images: Iterable[np.ndarray], weights: Iterable[np.ndarray], levels: int = 5) -> np.ndarray:
    imgs = list(images)
    ws = list(weights)
    if len(imgs) != len(ws):
        raise ValueError("images and weights length mismatch")

    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    imgs = [_resize_to(i, h, w) for i in imgs]
    ws = [_resize_to(wm, h, w) for wm in ws]

    wsum = np.sum(ws, axis=0) + 1e-6
    ws = [w_i / wsum for w_i in ws]

    lap_imgs = [_laplacian_pyramid(i, levels) for i in imgs]
    gau_ws = [_gaussian_pyramid(w_i, levels) for w_i in ws]

    out_lap: list[np.ndarray] = []
    for lv in range(levels):
        wsum_lv = np.sum([gw[lv] for gw in gau_ws], axis=0) + 1e-6
        normalized_lv = [gw[lv] / wsum_lv for gw in gau_ws]
        blended = np.zeros_like(lap_imgs[0][lv], dtype=np.float32)
        for li, wi in zip(lap_imgs, normalized_lv):
            if wi.ndim == 2:
                wi = wi[..., None]
            blended += li[lv] * wi
        out_lap.append(blended)

    out = _reconstruct_laplacian(out_lap)
    return np.clip(out, 0.0, 1.0)


def _apply_clahe_rgb(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


def _normalize_case_sizes(case_out_dir: Path) -> None:
    imgs = sorted(case_out_dir.glob("threeview_*.png"))
    if len(imgs) == 0:
        return

    sizes = []
    arrays = []
    for p in imgs:
        im = _read_rgb(p)
        arrays.append((p, im))
        sizes.append(im.shape[:2])

    min_h = min(h for h, _ in sizes)
    min_w = min(w for _, w in sizes)

    for p, im in arrays:
        h, w = im.shape[:2]
        x1 = max(0, (w - min_w) // 2)
        x2 = x1 + min_w
        # only crop bottom in vertical direction
        y1, y2 = 0, min_h
        crop = im[y1:y2, x1:x2]
        cv2.imwrite(str(p), cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _apply_clahe_case(case_out_dir: Path) -> None:
    for p in sorted(case_out_dir.glob("threeview_*.png")):
        im = _read_rgb(p)
        im = _apply_clahe_rgb(im)
        cv2.imwrite(str(p), cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _masked_gray(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    return g[mask > 0]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.sqrt((a0**2).sum() * (b0**2).sum()) + 1e-8
    return float((a0 * b0).sum() / den)


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    c1 = (0.01**2)
    c2 = (0.03**2)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    num = (2 * ma * mb + c1) * (2 * cov + c2)
    den = (ma * ma + mb * mb + c1) * (va + vb + c2)
    return float(num / (den + 1e-8))


def _safe_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size < 16 or b.size < 16:
        return {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "ncc": float("nan")}
    return {
        "mse": _mse(a, b),
        "psnr": _psnr(a, b),
        "ssim": _ssim_global(a, b),
        "ncc": _ncc(a, b),
    }


def _evaluate_metrics(case12_dir: Path, out_case_dir: Path) -> None:
    f12 = case12_dir / "fusion"
    w12 = case12_dir / "warp"
    rows: list[dict[str, float | int]] = []

    for i, stitched_p in enumerate(sorted(f12.glob("*_stitched.png"))):
        prefix = stitched_p.stem.replace("_stitched", "")
        warp1 = _read_rgb(w12 / f"{prefix}_left.png")
        warp2 = _read_rgb(w12 / f"{prefix}_right.png")
        s1 = _read_mask(f12 / f"{prefix}_mask_left_soft.png")
        s2 = _read_mask(f12 / f"{prefix}_mask_right_soft.png")
        stitched = _read_rgb(stitched_p)

        h = max(warp1.shape[0], warp2.shape[0], stitched.shape[0])
        w = max(warp1.shape[1], warp2.shape[1], stitched.shape[1])
        warp1 = _resize_to(warp1, h, w)
        warp2 = _resize_to(warp2, h, w)
        stitched = _resize_to(stitched, h, w)
        s1 = _resize_to(s1, h, w)
        s2 = _resize_to(s2, h, w)

        overlap = ((s1[..., 0] > 0) & (s2[..., 0] > 0)).astype(np.uint8)
        a = _masked_gray(warp1, overlap)
        b = _masked_gray(warp2, overlap)
        reg = _safe_metrics(a, b)

        i_fuse = s1 * warp1 + s2 * warp2
        af = _masked_gray(stitched, np.ones_like(overlap))
        bf = _masked_gray(i_fuse, np.ones_like(overlap))
        fus = _safe_metrics(af, bf)

        rows.append(
            {
                "idx": i,
                "reg_mse": reg["mse"],
                "reg_psnr": reg["psnr"],
                "reg_ssim": reg["ssim"],
                "reg_ncc": reg["ncc"],
                "fus_mse": fus["mse"],
                "fus_psnr": fus["psnr"],
                "fus_ssim": fus["ssim"],
                "fus_ncc": fus["ncc"],
            }
        )

    if not rows:
        return

    out_csv = out_case_dir / "metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)


def fuse_case_from_pairwise(case12_dir: Path, case23_dir: Path, out_dir: Path, levels: int = 5, input2_boost: float = 2.0) -> int:
    f12 = case12_dir / "fusion"
    f23 = case23_dir / "fusion"
    if not f12.exists() or not f23.exists():
        return 0

    p12 = sorted(f12.glob("*_stitched.png"))
    p23 = sorted(f23.glob("*_stitched.png"))
    n = min(len(p12), len(p23))
    if n == 0:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in range(n):
        s12 = p12[i]
        s23 = p23[i]
        pre12 = s12.stem.replace("_stitched", "")
        pre23 = s23.stem.replace("_stitched", "")

        img12 = _read_rgb(s12)
        img23 = _read_rgb(s23)
        m12 = _read_soft_or_bin_mask(f12, pre12, "right")
        m23 = _read_soft_or_bin_mask(f23, pre23, "left")
        l12 = _read_soft_or_bin_mask(f12, pre12, "left")
        r12 = _read_soft_or_bin_mask(f12, pre12, "right")
        l23 = _read_soft_or_bin_mask(f23, pre23, "left")
        r23 = _read_soft_or_bin_mask(f23, pre23, "right")

        h = max(img12.shape[0], img23.shape[0])
        w = max(img12.shape[1], img23.shape[1])
        img12 = _resize_to(img12, h, w)
        img23 = _resize_to(img23, h, w)
        m12 = _resize_to(m12, h, w)
        m23 = _resize_to(m23, h, w)
        l12 = _resize_to(l12, h, w)
        r12 = _resize_to(r12, h, w)
        l23 = _resize_to(l23, h, w)
        r23 = _resize_to(r23, h, w)

        valid12 = np.clip(l12 + r12, 0.0, 1.0)
        valid23 = np.clip(l23 + r23, 0.0, 1.0)
        img2 = (m12 * img12 + m23 * img23) / (m12 + m23 + 1e-6)
        overlap2 = np.minimum(m12, m23)
        w2 = np.clip(overlap2 * input2_boost, 0.0, 1.0)
        w12 = np.clip(valid12 * (1.0 - w2), 0.0, 1.0)
        w23 = np.clip(valid23 * (1.0 - w2), 0.0, 1.0)

        fused = gaussian_pyramid_blend([img12, img23, img2], [w12, w23, w2], levels=levels)
        wt = w12 + w23 + w2
        fallback = (valid12 * img12 + valid23 * img23) / (valid12 + valid23 + 1e-6)
        missing = (wt < 1e-4) & ((valid12 + valid23) > 1e-4)
        fused = np.where(missing, fallback, fused)

        out = (fused * 255.0).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_name = out_dir / f"threeview_{i:03d}.png"
        cv2.imwrite(str(out_name), out)
        count += 1

    _normalize_case_sizes(out_dir)
    _apply_clahe_case(out_dir)
    _evaluate_metrics(case12_dir, out_dir)
    return count