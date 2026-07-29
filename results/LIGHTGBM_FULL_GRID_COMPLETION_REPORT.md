# LightGBM 完整网格完成报告

## 完整性

- 平衡面板：280 个 forecaster × coverage × horizon × month 环境；每种预测器均为 140 个环境。
- 唯一 dataset × calendar-month 窗口：35；不将 280 个配置行视为独立时间重复。
- LightGBM 80%/1h 与既有冻结输出的逐指标复现审计全部通过。
- 90% LightGBM 使用目标匹配的 0.05/0.95 基础分位区间；旧 0.1/0.9 共享基础区间结果仅保留为敏感性。
- TCI 两项均采用用户内有效样本数权重，再对用户宏平均，因此严格非负。

## 主结果

- lightgbm_quantile，80%、1h：GCR=31.43% (11/35)。
- lightgbm_quantile，80%、6h：GCR=14.29% (5/35)。
- lightgbm_quantile，90%、1h：GCR=31.43% (11/35)。
- lightgbm_quantile，90%、6h：GCR=28.57% (10/35)。
- persistence_quantile_interval，80%、1h：GCR=25.71% (9/35)。
- persistence_quantile_interval，80%、6h：GCR=22.86% (8/35)。
- persistence_quantile_interval，90%、1h：GCR=20.00% (7/35)。
- persistence_quantile_interval，90%、6h：GCR=17.14% (6/35)。

- 全平衡面板的严格 GCR 为 23.93%；同步月份 block=2 描述性区间为 [15.36%, 32.86%]。
- 详细 PICP、MPIW、Winkler score 和双层级 gap 位于 `BALANCED_FULL_GRID_PANEL.csv`；该报告不从网格中重新选择 ACI 或 CSGR 参数。
