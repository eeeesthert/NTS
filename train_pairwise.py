from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from abus_pairwise.datasets import ABUSPairDataset
from abus_pairwise.pipeline import LossWeights, TwoStageStitcher, compute_total_loss, save_stage_results


def build_stage_loaders(args: argparse.Namespace, stage: str) -> tuple[DataLoader, DataLoader | None]:
    img_size = None if args.image_size <= 0 else args.image_size
    ds = ABUSPairDataset(
        args.dataset_root,
        stage=stage,
        image_size=img_size,
        augment=args.augment,
        hflip_prob=args.hflip_prob,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"No training samples found for stage={stage} under {args.dataset_root}. "
            "Supported layouts: case/inputX.jpg or case/inputX/slice_xxx.jpg."
        )
    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    if val_len > 0:
        gen = torch.Generator().manual_seed(args.seed)
        ds_train, ds_val = random_split(ds, [train_len, val_len], generator=gen)
    else:
        ds_train, ds_val = ds, None

    dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2) if ds_val is not None else None
    return dl, dl_val


def evaluate_loader(
    model: TwoStageStitcher,
    dl_val: DataLoader | None,
    lw: LossWeights,
    device: torch.device,
) -> float | None:
    if dl_val is None:
        return None
    model.eval()
    s = 0.0
    n = 0
    with torch.no_grad():
        for batch in dl_val:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            left_x = batch["left_x"].to(device)
            right_x = batch["right_x"].to(device)
            out = model(left, right, left_x=left_x, right_x=right_x)
            vloss = compute_total_loss(out, left_x, right_x, lw)
            s += vloss["total"].item()
            n += 1
    return s / max(n, 1)


def train_stage(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    dl, dl_val = build_stage_loaders(args, stage)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
    lw = LossWeights()
    best_val = float("inf")
    bad_epochs = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            left_x = batch["left_x"].to(device)
            right_x = batch["right_x"].to(device)

            optim.zero_grad()
            out = model(left, right, left_x=left_x, right_x=right_x)
            losses = compute_total_loss(out, left_x, right_x, lw)
            losses["total"].backward()
            optim.step()
        scheduler.step()

        val_total = losses["total"].item()
        maybe_val = evaluate_loader(model, dl_val, lw, device)
        if maybe_val is not None:
            val_total = maybe_val

        if val_total < best_val - 1e-6:
            best_val = val_total
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        print(
            f"[{stage}] epoch={epoch+1}/{args.epochs} "
            f"total={losses['total'].item():.4f} warp_l1={losses['warp_l1'].item():.4f} "
            f"edge={losses['grid_edge'].item():.4f} angle={losses['grid_angle'].item():.4f} "
            f"seam={losses['seam_cost'].item():.4f} val_total={val_total:.4f}"
        )
        if dl_val is not None and bad_epochs >= args.early_stopping_patience:
            print(f"[{stage}] early stopping at epoch {epoch+1}, best_val={best_val:.4f}")
            break
    model.load_state_dict(best_state, strict=True)


def train_interleaved(args: argparse.Namespace, model: TwoStageStitcher, device: torch.device) -> None:
    train_loaders: dict[str, DataLoader] = {}
    val_loaders: dict[str, DataLoader | None] = {}
    for stage in ("12", "23"):
        dl, dl_val = build_stage_loaders(args, stage)
        train_loaders[stage] = dl
        val_loaders[stage] = dl_val

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
    lw = LossWeights()
    best_val = float("inf")
    bad_epochs = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        stage_last_loss: dict[str, dict[str, float]] = {}
        stage_iters = {stage: iter(dl) for stage, dl in train_loaders.items()}
        max_steps = max(len(train_loaders["12"]), len(train_loaders["23"]))

        for _ in range(max_steps):
            for stage in ("12", "23"):
                batch = next(stage_iters[stage], None)
                if batch is None:
                    continue
                left = batch["left"].to(device)
                right = batch["right"].to(device)
                left_x = batch["left_x"].to(device)
                right_x = batch["right_x"].to(device)

                optim.zero_grad()
                out = model(left, right, left_x=left_x, right_x=right_x)
                losses = compute_total_loss(out, left_x, right_x, lw)
                losses["total"].backward()
                optim.step()
                stage_last_loss[stage] = {k: v.item() for k, v in losses.items()}
        scheduler.step()

        val_parts = []
        stage_val: dict[str, float | None] = {}
        for stage in ("12", "23"):
            v = evaluate_loader(model, val_loaders[stage], lw, device)
            stage_val[stage] = v
            if v is not None:
                val_parts.append(v)
        epoch_val = sum(val_parts) / len(val_parts) if val_parts else sum(v["total"] for v in stage_last_loss.values()) / max(len(stage_last_loss), 1)

        if epoch_val < best_val - 1e-6:
            best_val = epoch_val
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        l12 = stage_last_loss.get("12")
        l23 = stage_last_loss.get("23")
        print(
            f"[12<->23] epoch={epoch+1}/{args.epochs} "
            f"12_total={l12['total']:.4f} 23_total={l23['total']:.4f} "
            f"val12={stage_val['12'] if stage_val['12'] is not None else -1:.4f} "
            f"val23={stage_val['23'] if stage_val['23'] is not None else -1:.4f} "
            f"val_joint={epoch_val:.4f}"
        )
        if val_parts and bad_epochs >= args.early_stopping_patience:
            print(f"[12<->23] early stopping at epoch {epoch+1}, best_val={best_val:.4f}")
            break
    model.load_state_dict(best_state, strict=True)


def export_samples(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    model.eval()
    img_size = None if args.image_size <= 0 else args.image_size
    ds = ABUSPairDataset(args.dataset_root, stage=stage, image_size=img_size)
    if len(ds) == 0:
        return
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            left_x = batch["left_x"].to(device)
            right_x = batch["right_x"].to(device)
            out = model(left, right, left_x=left_x, right_x=right_x)
            case = batch["case"][0]
            save_stage_results(out, Path(args.out_dir) / "results" / stage / case, prefix=f"{stage}_{i:03d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset")
    ap.add_argument("--out-dir", default="./outputs_1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=0, help="set <=0 to keep original slice size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stopping-patience", type=int, default=8)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--hflip-prob", type=float, default=0.2)
    ap.add_argument("--brightness-jitter", type=float, default=0.08)
    ap.add_argument("--contrast-jitter", type=float, default=0.08)
    ap.add_argument("--encoder-pretrain-source", choices=["imagenet", "radimagenet", "local", "none"], default="imagenet")
    ap.add_argument("--encoder-ckpt", default="./ckpt/ResNet50", help="required for radimagenet/local source")
    ap.add_argument("--radimagenet-url", "--net-url", dest="radimagenet_url", default=None, help="optional URL for auto-downloading RadImageNet weights")
    ap.add_argument("--encoder-strict-load", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shared-ckpt-name", default="shared_model.pt")
    ap.add_argument("--training-schedule", choices=["interleaved", "sequential"], default="interleaved")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(
        encoder_pretrain_source=args.encoder_pretrain_source,
        encoder_ckpt=args.encoder_ckpt,
        encoder_radimagenet_url=args.radimagenet_url,
        encoder_strict_load=args.encoder_strict_load,
    ).to(device)

    if args.training_schedule == "interleaved":
        train_interleaved(args, model=model, device=device)
    else:
        # Step-1: input1 + input2
        train_stage(args, stage="12", model=model, device=device)
        # Step-2: input2 + input3
        train_stage(args, stage="23", model=model, device=device)
    export_samples(args, stage="12", model=model, device=device)
    export_samples(args, stage="23", model=model, device=device)

    # Save single shared checkpoint for both stages.
    ckpt = Path(args.out_dir) / args.shared_ckpt_name
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] saved shared checkpoint: {ckpt}")


if __name__ == "__main__":
    main()