from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from abus_pairwise.tri_datasets import ABUSThreeViewDataset
from abus_pairwise.pipeline import TwoStageStitcher
from abus_pairwise.three_view_pipeline import ThreeViewFixedCenterStitcher


def _masked_gray(img: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    # img: (1,3,H,W), mask: (1,1,H,W)
    x = img.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32)
    m = mask.detach().cpu().numpy()[0, 0] > 0
    g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return g[m]


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
    den = float(np.sqrt((a0**2).sum() * (b0**2).sum()) + 1e-8)
    return float((a0 * b0).sum() / den)


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(((a - ma) * (b - mb)).mean())
    num = (2 * ma * mb + c1) * (2 * cov + c2)
    den = (ma * ma + mb * mb + c1) * (va + vb + c2)
    return float(num / (den + 1e-8))


def _safe_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size < 16 or b.size < 16:
        return {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "ncc": float("nan")}
    return {"mse": _mse(a, b), "psnr": _psnr(a, b), "ssim": _ssim_global(a, b), "ncc": _ncc(a, b)}


def _compute_metrics(out: dict[str, torch.Tensor]) -> dict[str, float]:
    o12 = out["overlap_12"]
    o23 = out["overlap_32"]
    o13 = out["overlap_13"]
    valid_fixed = (out["input2_fixed"].sum(1, keepdim=True) > 0).float()

    m12 = _safe_metrics(_masked_gray(out["input1_warp"], o12), _masked_gray(out["input2_fixed"], o12))
    m23 = _safe_metrics(_masked_gray(out["input3_warp"], o23), _masked_gray(out["input2_fixed"], o23))
    m13 = _safe_metrics(_masked_gray(out["input1_warp"], o13), _masked_gray(out["input3_warp"], o13))
    mf = _safe_metrics(_masked_gray(out["stitched"], valid_fixed), _masked_gray(out["input2_fixed"], valid_fixed))

    return {
        "reg12_mse": m12["mse"],
        "reg12_psnr": m12["psnr"],
        "reg12_ssim": m12["ssim"],
        "reg12_ncc": m12["ncc"],
        "reg23_mse": m23["mse"],
        "reg23_psnr": m23["psnr"],
        "reg23_ssim": m23["ssim"],
        "reg23_ncc": m23["ncc"],
        "reg13_mse": m13["mse"],
        "reg13_psnr": m13["psnr"],
        "reg13_ssim": m13["ssim"],
        "reg13_ncc": m13["ncc"],
        "fus_fixed_mse": mf["mse"],
        "fus_fixed_psnr": mf["psnr"],
        "fus_fixed_ssim": mf["ssim"],
        "fus_fixed_ncc": mf["ncc"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset/infer")
    ap.add_argument("--checkpoint", default="./outputs_1/three_view_model.pt")
    ap.add_argument("--out-dir", default="./infer_outputs_3view")
    ap.add_argument("--metrics-csv", default="./infer_outputs_3view/metrics.csv")
    ap.add_argument("--image-size", type=int, default=0, help="<=0 keeps original size")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--encoder-pretrain-source", choices=["imagenet", "radimagenet", "local", "none"], default="local")
    ap.add_argument("--encoder-ckpt", default="./ckpt/ResNet50.pt")
    ap.add_argument("--radimagenet-url", "--net-url", dest="radimagenet_url", default=None)
    ap.add_argument("--encoder-strict-load", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    img_size = None if args.image_size <= 0 else args.image_size

    ds = ABUSThreeViewDataset(args.dataset_root, image_size=img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    pair = TwoStageStitcher(
        encoder_pretrain_source=args.encoder_pretrain_source,
        encoder_ckpt=args.encoder_ckpt,
        encoder_radimagenet_url=args.radimagenet_url,
        encoder_strict_load=args.encoder_strict_load,
    ).to(device)
    model = ThreeViewFixedCenterStitcher(pair).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    with torch.no_grad():
        for i, batch in enumerate(dl):
            i1 = batch["input1"].to(device)
            i2 = batch["input2"].to(device)
            i3 = batch["input3"].to(device)
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            x3 = batch["x3"].to(device)
            case = batch["case"][0]

            out = model(i1, i2, i3, x1, x2, x3)
            case_dir = out_dir / case
            case_dir.mkdir(parents=True, exist_ok=True)
            save_image(out["input1_warp"], case_dir / f"{i:04d}_warp1.png")
            save_image(out["input2_fixed"], case_dir / f"{i:04d}_fixed2.png")
            save_image(out["input3_warp"], case_dir / f"{i:04d}_warp3.png")
            save_image(out["stitched"], case_dir / f"{i:04d}_stitched.png")

            m = _compute_metrics(out)
            rows.append({"case": case, "idx": i, **m})

    if rows:
        metric_keys = [k for k in rows[0].keys() if k not in {"case", "idx"}]
        mean_row: dict[str, float | str] = {"case": "MEAN", "idx": -1}
        for k in metric_keys:
            vals = np.array([float(r[k]) for r in rows], dtype=np.float32)
            mean_row[k] = float(np.nanmean(vals))
        rows.append(mean_row)

        csv_path = Path(args.metrics_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[infer3] metrics saved: {csv_path}")


if __name__ == "__main__":
    main()
