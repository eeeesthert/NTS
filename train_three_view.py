from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from abus_pairwise.datasets import ABUSThreeViewDataset
from abus_pairwise.pipeline import TwoStageStitcher
from abus_pairwise.three_view_pipeline import ThreeViewFixedCenterStitcher, ThreeViewLossWeights, compute_three_view_total_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset")
    ap.add_argument("--checkpoint", default="./outputs_1/shared_model.pt", help="pairwise pretrained checkpoint")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=0, help="<=0 means original size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-ckpt", default="./outputs_1/three_view_model.pt")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    img_size = None if args.image_size <= 0 else args.image_size

    ds = ABUSThreeViewDataset(args.dataset_root, image_size=img_size)
    if len(ds) == 0:
        raise RuntimeError("No three-view samples found.")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    pair = TwoStageStitcher().to(device)
    pair.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model = ThreeViewFixedCenterStitcher(pair).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    w = ThreeViewLossWeights()

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            i1 = batch["input1"].to(device)
            i2 = batch["input2"].to(device)
            i3 = batch["input3"].to(device)
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            x3 = batch["x3"].to(device)

            optim.zero_grad()
            out = model(i1, i2, i3, x1, x2, x3)
            loss_dict = compute_three_view_total_loss(out, x1, x2, x3, w=w)
            loss_dict["total"].backward()
            optim.step()

        print(
            f"[3view] epoch={epoch+1}/{args.epochs} total={loss_dict['total'].item():.4f} "
            f"warp={loss_dict['warp_l1'].item():.4f} seam={loss_dict['seam_cost'].item():.4f} "
            f"fixed={loss_dict['fixed_consistency'].item():.4f} "
            f"cross13={loss_dict['cross13_warp'].item():.4f}"
        )

    torch.save(model.state_dict(), args.out_ckpt)
    print(f"[3view] saved: {args.out_ckpt}")


if __name__ == "__main__":
    main()
