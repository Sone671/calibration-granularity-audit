# 投稿前四项补强冻结协议

冻结日期：2026-07-28。本文档在查看本轮新增结果前写入，任何未通过此协议的额外试验不得进入主结论。

## 目标与范围

本轮只补强当前“诊断＋benchmark＋指标体系＋解析性质”论文的可比性、统计表述和稳健性；不将论文重构为新的预测模型或新的共形理论论文。Mamba-ACI 的近期文献将纳入相关工作，但 Mamba 骨干复现不属于本轮。

## A. 近期相关工作、理论定位与统计单位

1. 新增 2024--2026 年直接相关的概率负荷预测、时序共形与 Mamba-ACI 文献，至少包括 Suresh et al. (2025, KBS 328:114222, DOI: 10.1016/j.knosys.2025.114222)；
2. 将“轻理论贡献”改为“诊断指标的解析性质与边界”；不再声称新的时序有限样本覆盖保证；
3. 全文统一使用：
   - 35 个唯一的 dataset × calendar-month 测试窗口；
   - 175 个 configuration × window 观察；
   - 不称 35 个相邻月份或 175 个配置观察为独立样本。

## B. 区间质量与时间块推断

### B1. 区间指标

在现有 raw/global/segment/user 的每一测试窗口上，新增：

- MPIW = mean(U_i-L_i)；
- 目标覆盖率为 $\tau$、$\alpha=1-\tau$ 的 Winkler/interval score：

$$
\mathrm{IS}_{\alpha}(L,U;y)=(U-L)+\frac{2}{\alpha}(L-y)\mathbb I(y<L)+\frac{2}{\alpha}(y-U)\mathbb I(y>U).
$$

不计算 CRPS；当前输出只有有限个分位数，不能把离散 80%/90% 区间伪装成完整预测分布。WIS 也不进入本轮主结果，因为其标准定义需要多层嵌套区间。

### B2. 联合移动块 bootstrap

- 基本重采样单位是同一数据集内的一个自然月，并联合保留该月的全部配置；
- 在每个数据集的 11/12 月时序中使用 circular moving-block bootstrap；
- 主分析 block length = 2 months，敏感性 block length = 3 months；
- 每种设置 10,000 次，随机种子 20260728；
- 报告总体 GCR、按预测器 GCR、LightGBM vs persistence（80%, 1h）的匹配风险差、persistence 内 90%-80% 风险差和 6h-1h 风险差；
- 置信区间是时间相关下的描述性不确定性评估，不用于把 3 个数据集推广为总体因果结论。

## C. 运行分群与实质冲突敏感性

### C1. 训练期分群方案

主评估设置固定为目标覆盖率 80%、预测跨度 1h。预定义方案：

| ID | 分群方法 | K |
|---|---|---:|
| KM2 | deterministic K-means++ | 2 |
| KM3 | deterministic K-means++ | 3（当前主设置） |
| KM4 | deterministic K-means++ | 4 |
| KM5 | deterministic K-means++ | 5 |
| WARD3 | Ward hierarchical clustering | 3 |

所有聚类均只使用原始训练期的同一九维用户画像，标准化方式和随机种子保持固定。由于现有 LightGBM 将 cluster 作为输入特征，LightGBM 的每个方案均重新训练全部三个分位数模型；persistence 不使用 cluster 特征，但其 segment-CQR 与运行分群指标用相同新标签重算。不得只重划测试评价分区而仍把旧 cluster 特征喂给 LightGBM，并把该结果称为完整管线敏感性。

### C2. 实质冲突阈值

对每个已有 configuration-window，令 $\delta\in\{0,0.0025,0.005,0.01\}$。只有在

$$
\Delta_{\mathrm{user}}<-\delta,\qquad \Delta_{\mathrm{seg}}>\delta
$$

时才计为 material personalization conflict；反向冲突同理。报告按 $\delta$ 的总体、预测器和数据集分层比率。该分析不重新选择阈值，也不按有利阈值声明主结论。

## D. 严格前向 ACI 粒度基线

### D1. 范围

在主配置（80% coverage、1h horizon）上，对 LightGBM quantile 和 persistence quantile interval 分别实现：

- aci_global；
- aci_segment；
- aci_user。

原始、静态 rolling global/segment/user CQR 同时保留。ACI 只补充时间自适应的直接比较，不替代四项诊断的原定义。

### D2. 固定的前向协议

1. 每个自然月环境开始时，使用此前 56 天的基础区间分数初始化各粒度的有序 score set，初始 $\alpha_0=0.20$；
2. 在测试月内按目标时间戳严格前向预测；当前时刻的修正只可使用该时刻之前的标签；
3. 使用 batched ACI 更新

$$
\alpha_{t+1}=\Pi_{[0.01,0.50]}\{\alpha_t+\gamma(\alpha_0-\bar e_t)\},
$$

其中 $\bar e_t$ 是 global/segment 在该时间戳的平均未覆盖率，user 版本为每位用户自身的 0/1 未覆盖；$\gamma=0.005\times \Delta t/(30\text{ minutes})$；
4. 每个时间步从环境开始时固定的 score set 读取 $1-\alpha_t$ 分位数；score set 在一个自然月内不吸收测试期标签。该选择使 ACI 的在线 alpha 更新与冻结的 56 天 split-CQR score-set 设计可比；
5. 目标月结束后重置 alpha 和 score set。不得用该月结果选择 gamma、截断区间、起始 alpha 或更新频率；
6. 前向路由器不属于本轮，ACI 仅作为时间自适应校准基线。

### D3. 结论门槛

ACI 结果即使不改善 GCR 也必须报告。若 ACI 改善覆盖 gap 但显著恶化 MPIW/interval score，不得称为整体改进。若 ACI 全局或用户版本仍显示高 TCI/GCR，则该结果支持粒度诊断并非静态 CQR 特有。

## 禁止事项

- 不引入 Mamba、Transformer 或新的深度预测骨干以事后追逐 SOTA；
- 不根据初始结果改变 block length、K、聚类方法、material threshold 或 ACI gamma；
- 不将 175 行面板当作 175 个独立时间重复；
- 不将 ACI 的经验结果解释为时序过程上的分布无关有限样本保证。

## 区间分数运行哈希

    6C63FB022EC9CCA44E7D430C36515DBBA1A2FDB21DB4E462760EF71C77560586  run_london_benchmark.py
    D6ED3A47EB48AAA362E8C1D615D7E79702B97003F6CFEA82D7850DB7BB4084A2  run_ausgrid_benchmark.py
    7938A3C916B1222D3473D89D89DD93089A9C256FE430B01100294D82239E5C7C  run_uci_benchmark.py
    BD39BF6FA0798D8AE66F492CE557923633C0A8FAA11C4490B6CB89198B9FD42D  run_naive_robustness.py

新增字段仅为 winkler_interval_score；历史目录不得覆盖。LightGBM 输出目录固定为
lightgbm_london_scoring、lightgbm_ausgrid_scoring、lightgbm_uci_scoring；
persistence 输出目录固定为
naive_london_scoring、naive_ausgrid_scoring、naive_uci_scoring。
