# NTS

attempt to stitch three-view of ABUS for one time

## Three-view extension (input2 as fixed view)

现在支持把现有的两两拼接网络扩展为三视图训练/融合流程：

- 使用同一个 pairwise 模型做两次配准：
  - `input1 -> input2`
  - `input3 -> input2`
- 两次结果统一到 `input2` 的坐标系后做三路融合。
- 损失函数仍复用原先定义（warp/fusion 各项不变），只是改成对两条分支做平均汇总；
  另外新增一项 `fixed_consistency`，约束两条分支得到的 fixed 视图一致。
- 由于 `input1` 与 `input3` 在乳头附近通常也有重叠，新增跨分支约束：
  `cross13_warp`（重叠区 L1）与 `cross13_nipple`（乳头热图一致性），默认权重较小（0.3/0.2）。

### 训练命令

```bash
python train_three_view.py \
  --dataset-root ./dataset \
  --checkpoint ./outputs_1/shared_model.pt \
  --out-ckpt ./outputs_1/three_view_model.pt
```

核心实现位于：

- `abus_pairwise/three_view_pipeline.py`
- `abus_pairwise/datasets.py` 中的 `ABUSThreeViewDataset`
