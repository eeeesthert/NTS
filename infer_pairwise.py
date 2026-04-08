from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abus_pairwise.datasets import ABUSPairDataset
from abus_pairwise.pipeline import TwoStageStitcher, save_stage_results
from abus_pairwise.three_view_fusion import fuse_case_from_pairwise


def run_stage(model: TwoStageStitcher, dataset_root: str, stage: str, out_dir: str, image_size: int, device: torch.device) -> None:
    img_size = None if image_size <= 0 else image_size
    ds = ABUSPairDataset(dataset_root, stage=stage, image_size=img_size)
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            left = batch["left"].unsqueeze(0).to(device)
            right = batch["right"].unsqueeze(0).to(device)
            left_x = batch["left_x"].unsqueeze(0).to(device)
            right_x = batch["right_x"].unsqueeze(0).to(device)
            out = model(left, right, left_x=left_x, right_x=right_x)
            save_stage_results(out, Path(out_dir) / stage / batch["case"], prefix=f"{stage}_{i:03d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset/infer")
    ap.add_argument("--checkpoint", default="./outputs/shared_model.pt")
    ap.add_argument("--out-dir", default="./infer_outputs_c")
    ap.add_argument("--image-size", type=int, default=0, help="set <=0 to keep original slice size")
    ap.add_argument("--encoder-pretrain-source", choices=["imagenet", "radimagenet", "local", "none"], default="imagenet")
    ap.add_argument("--encoder-ckpt", default="./ckpt/ResNet50", help="required for radimagenet/local source")
    ap.add_argument("--radimagenet-url", "--net-url", dest="radimagenet_url", default=None, help="optional URL for auto-downloading RadImageNet weights")
    ap.add_argument("--encoder-strict-load", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--run-three-view-fusion", default=True, action="store_true", help="run three-view fusion after stage12/stage23 inference")
    ap.add_argument("--three-view-out-dir", default="./infer_outputs_c/tri_view", help="default: <out-dir>/three_view")
    ap.add_argument("--three-view-levels", type=int, default=5)
    ap.add_argument("--three-view-input2-boost", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(
        encoder_pretrain_source=args.encoder_pretrain_source,
        encoder_ckpt=args.encoder_ckpt,
        encoder_radimagenet_url=args.radimagenet_url,
        encoder_strict_load=args.encoder_strict_load,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)

    run_stage(model, args.dataset_root, stage="12", out_dir=args.out_dir, image_size=args.image_size, device=device)
    run_stage(model, args.dataset_root, stage="23", out_dir=args.out_dir, image_size=args.image_size, device=device)

    if args.run_three_view_fusion:
        pair_root = Path(args.out_dir)
        out_root = Path(args.three_view_out_dir) if args.three_view_out_dir else (pair_root / "three_view")
        total = 0
        for case12 in sorted((pair_root / "12").glob("case*")):
            case23 = pair_root / "23" / case12.name
            if not case23.exists():
                continue
            n = fuse_case_from_pairwise(
                case12,
                case23,
                out_dir=out_root / case12.name,
                levels=args.three_view_levels,
                input2_boost=args.three_view_input2_boost,
            )
            total += n
        print(f"[infer] three-view fused images saved: {total}")


if __name__ == "__main__":
    main()