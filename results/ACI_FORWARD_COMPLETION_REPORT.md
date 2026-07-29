# 严格前向 ACI 完成报告

本报告对应冻结清单 `FROZEN_ACI_RUN_MANIFEST.md`，在完整结果生成后记录；不修改其中的设计、参数或源代码哈希。

## 完整性与复现审计

- 6 个完整运行：3 个数据集 × LightGBM quantile / persistence quantile interval；
- 80% coverage、1h horizon、KM3、35 个唯一 `dataset × calendar-month` 测试窗口；
- 共 70 个 `predictor × window` ACI 比较观察、490 个 `method × predictor × window` 指标行；
- 三种静态 CQR 在六个运行中的 210 个对应指标行，均与既有冻结 benchmark 完全一致（最大绝对差 0）；
- 所有 PICP、MPIW、Winkler interval score、用户 gap 与运行分群 gap 均为有限值。

## 预定义结果（描述性）

| 基础预测器 | 静态 CQR GCR | ACI GCR |
|---|---:|---:|
| LightGBM | 11/35 (31.43%) | 2/35 (5.71%) |
| Persistence | 9/35 (25.71%) | 2/35 (5.71%) |

合并行仅便于描述相同 35 个窗口上的两种预测器：静态 20/70 (28.57%)，ACI 4/70 (5.71%)；不得视为 70 个独立时间重复。

相对同粒度静态 User-CQR，ACI-user 在六个数据集—预测器组合中均降低宏用户 gap、最大运行分群 gap 与 Winkler interval score；MPIW 的变化方向不一致。因此结果支持“当前固定 ACI 规则下的经验改进”，不支持相关时序上的分布无关有限样本保证，也不支持对 Mamba 或其他骨干的泛化断言。

## 输出

- `aci_lgbm_london`、`aci_lgbm_ausgrid`、`aci_lgbm_uci`；
- `aci_persistence_london`、`aci_persistence_ausgrid`、`aci_persistence_uci`；
- 汇总：`ACI_FORWARD_PANEL.csv`、`ACI_FORWARD_SUMMARY_BY_DATASET.csv`、`ACI_FORWARD_PAIR_DIFFERENCES.csv`、`ACI_FORWARD_CONFLICT_SUMMARY.csv`、`ACI_FORWARD_REPORT.json`。

