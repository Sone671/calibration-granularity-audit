# LightGBM 完整网格最终修复补充协议

冻结日期：2026-07-29，在 V2 预检中止后、最终重跑前写入。

在 `FROZEN_LIGHTGBM_FULL_GRID_REPAIR_ADDENDUM.md` 的分位数逐样本排序修复之外，唯一再加入的修改是复刻冻结的边界 origin 规则：London/Ausgrid 的全量校准和测试对从 `start - horizon_steps` 开始，使目标索引从 `start` 起；UCI 维持冻结基线的从 `start` 起 origin 规则。

该修复不涉及任何结果值、模型参数、用户选择、特征、训练预算、分群、覆盖率、跨度、校准、统计量或汇总规则。最终输出固定写入 `lightgbm_full_grid_v3/`。接受标准仍为三个数据集 80%/1h 的所有方法级指标与既有 `*_scoring/` 结果最大绝对差不超过 $10^{-10}$；不通过即不进入主文。
