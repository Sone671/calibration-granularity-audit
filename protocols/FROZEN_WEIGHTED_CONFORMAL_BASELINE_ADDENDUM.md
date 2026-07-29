# 指数近时加权 CQR 基线冻结补充协议

## 目的

在不修改原始多粒度 benchmark 主终点、不重训基础预测器、也不使用目标月标签的前提下，补充一类面向时间漂移的现代加权共形基线。原冻结结果仍作为主分析；本补充实验只用于回答审稿中的基线完整性问题。

## 信息集与候选方法

- 每个自然月仍只使用开始前 56 天校准块；目标月标签不进入校准分数或权重。
- 保留 Global-CQR、Segment-CQR（Mondrian/分组条件 CQR）和 User-CQR。
- 新增 Exponentially Recency-Weighted CQR（ERW-CQR），分别在 global、segment 和 user 三种粒度计算加权分位数。
- ACI 继续作为“月内标签可及时反馈”场景的时序基线；ERW-CQR 与静态 CQR 属于“月内无即时反馈”场景，不与 ACI 作跨信息集胜负比较。

## 权重与加权分位数

校准观察距目标月开始的时间为 `age_days`，权重冻结为

`w = 2 ** (-age_days / 14)`。

主半衰期为 14 天，理由是它等于 56 天校准块的四分之一，既保留多个周周期，又使近期月份状态获得更高权重。该值在读取 ERW-CQR 结果前冻结，不根据数据集或窗口调节。

加权 conformal 分位数将一个权重为 1 的未来测试原子计入总质量；若校准累计权重不足以达到目标分位数，则返回无穷宽区间。当前样本量下预计不会触发，但实现必须保留该保护。

## 冻结验证范围

- 基础预测器：persistence quantile interval；该预测器无需重训，可完整覆盖冻结网格。
- 数据集：London、Ausgrid、UCI Electricity。
- 目标覆盖率：80%、90%。
- 预测跨度：1 h、6 h。
- 主报告：140 个 configuration × window 观察；另报告 80%/1 h 的 35 个唯一窗口。

LightGBM 不重新训练以避免将新的大规模模型随机性混入原始冻结审计。ERW-CQR 在 LightGBM 上的可迁移性作为局限性明确报告。

## 评价

报告 PICP、MPIW、Winkler interval score、宏用户绝对覆盖差、最大运行分群绝对覆盖差和 `L_0.5`。跨数据集的 interval score 仅报告相对同窗口 Global-CQR 的比例变化，不平均绝对量纲。

## 禁止事项

- 不根据结果更换半衰期或只保留有利的数据集/配置。
- 不把 ERW-CQR 描述为非平稳序列上的分布无关有限样本保证。
- 不把 Segment-CQR 的运行分群解释为受保护群体。
- 不用本补充实验覆盖或改写原始 GCR、RRR、TCI、ROG 主结果。
