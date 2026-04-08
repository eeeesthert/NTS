from __future__ import annotations

import argparse
from pathlib import Path

from abus_pairwise.three_view_fusion import fuse_case_from_pairwise


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse stage12 + stage23 outputs into three-view result with Gaussian pyramid")
    ap.add_argument("--pairwise-root", default="./outputs/results", help="contains 12/<case>/ and 23/<case>/")
    ap.add_argument("--out-dir", default="./outputs/no_clahe/three_view")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--input2-boost", type=float, default=3.0, help="higher value gives input2 overlap larger weights")
    args = ap.parse_args()

    root = Path(args.pairwise_root)
    stage12 = root / "12"
    stage23 = root / "23"
    if not stage12.exists() or not stage23.exists():
        raise FileNotFoundError(f"Missing stage folders under {root}: expected 12/ and 23/")

    total = 0
    for case12 in sorted(stage12.glob("case*")):
        case_name = case12.name
        case23 = stage23 / case_name
        if not case23.exists():
            continue
        saved = fuse_case_from_pairwise(
            case12,
            case23,
            out_dir=Path(args.out_dir) / case_name,
            levels=args.levels,
            input2_boost=args.input2_boost,
        )
        print(f"[{case_name}] saved={saved}")
        total += saved

    print(f"Done. total saved: {total}")


if __name__ == "__main__":
    main()
