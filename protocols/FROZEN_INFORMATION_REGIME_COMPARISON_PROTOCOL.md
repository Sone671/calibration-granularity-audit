# ACI--CSGR 信息场景比较、配对推断与 CSGR 消融冻结协议

冻结日期：2026-07-29。本协议在查看本轮新汇总、ACI 推断和 CSGR 消融结果前写入。其目标是统一解释 ACI 与 CSGR，而不是以事后结果在两者间选择“胜者”。历史 benchmark、ACI 和 CSGR 输出不得覆盖。

## 1. 研究问题与信息场景

两种方法使用不同的部署信息集，禁止将它们叙述为同一反馈条件下可相互替代的竞争者。

| 场景 | 部署时可用标签 | 候选机制 | 可部署选择 |
|---|---|---|---|
| 月内无即时反馈（M） | 仅目标月开始前 56 天 | Static Global/Segment/User-CQR, CSGR | CSGR 在月初选择一个静态粒度并固定整月 |
| 月内有即时反馈（O） | 当前预测完成后，后续时点可用此前标签 | Static 与 ACI Global/Segment/User | ACI 在月内更新 alpha；不由 CSGR 在本轮选择 |

主比较只在共同的 80% coverage、1h horizon、三数据集、两基础预测器上进行，即 35 个唯一 `dataset × calendar-month` 窗口对应 70 个 `predictor × window` 观察。跨场景表格可并列同一结果指标，但不得声称 CSGR 应胜过拥有即时反馈的 ACI。

## 2. 统一评价口径

在每个结果单元报告 PICP、MPIW、Winkler interval score、宏用户绝对 coverage gap、最大运行分群 coverage gap、严格 GCR 与

$$
L_{0.5}=0.5A_{\mathrm{user}}+0.5A_{\mathrm{seg}}.
$$

场景 M 的固定策略基线为 Static Global/Segment/User-CQR、事后最佳静态固定策略和静态逐月 oracle；CSGR 只与这些静态候选比较。场景 O 报告 Static 与 ACI Global/Segment/User，并在 ACI 候选族内另报事后最佳固定策略和逐月 oracle；这些 oracle 只作不可实现下界。

绝对 MPIW 和 interval score 不跨负荷单位直接池化。对它们报告数据集--预测器内的同窗口配对差与相对变化；所有数据集的 pooled 描述只使用无量纲风险、coverage gap、GCR 或相对 score 变化。

## 3. ACI 配对时间块推断

主差异为 ACI-user minus Static User-CQR 的用户 gap、最大分群 gap、Winkler interval score、MPIW、PICP 和 $L_{0.5}$；同时报告 ACI-global minus Static Global-CQR 与 ACI-segment minus Static Segment-CQR。GCR 风险差定义为

$$
\Delta\mathrm{GCR}=\mathbb I\{\text{ACI-user versus ACI-global conflict}\}
-\mathbb I\{\text{Static User versus Static Global conflict}\}.
$$

基本重采样单位为同一数据集的自然月，联合保留该月两种预测器的全部方法和配对差。使用 circular moving-block bootstrap，主 block length=2 个月、敏感性 block length=3 个月，10,000 次、seed `20260729`。报告数据集--预测器内区间；对于所有数据集的总结，报告覆盖 gap/GCR/$L_{0.5}$ 的同步月份块区间和 interval-score 的相对变化区间。它们是时间相关下的描述性不确定性评估，不是独立 70 单元或总体因果区间。

## 4. CSGR 最小消融

使用既有、冻结的 static-CQR 伪未来折和测试月指标，不重训预测器，不改变 56 天校准块、主 $\lambda=0.5$ 或候选策略。主消融在 persistence 的完整 140 个 configuration--window 观察上进行；LightGBM 的 35 个外部验证窗口只作方向复核。

1. 扩展折的标准误门槛 $k\in\{0,0.5,1,2\}$，其中局部策略仅在 $\bar d-k\operatorname{se}(d)>0$ 时合格；
2. 回退锚点：Global fallback，或在无局部策略合格时选择三个候选的最低历史折均值（history-best fallback）；
3. 损失：$\eta=0$ 与 $\eta=0.01$ 的归一化 interval-score 敏感性；
4. 时间验证布局：三折 35/42/49 天扩展折；以及仅用最后 7 天验证、直接取该折最小损失的 `recent-7d direct` 对照（它没有标准误屏幕，不称为 CSGR）。

每个变体报告相对事后最佳固定策略和 Global 的 $L_{0.5}$ 差、静态 GCR、用户/分群 gap、选择计数及 block=2/3 时间块区间。不得根据消融结果改写 CSGR 的主定义；主规则固定为三折、$k=1$、Global fallback、$\eta=0$。

## 5. 命名、论文与后续网格

论文中的 CSGR 扩写为 **Cross-Fitted Stability-Screened Granularity Router**。``Stability-screened'' 描述一倍标准误屏幕；不使用 ``safe'' 作为风险保证表述。

中文稿继续作为内部审计源。实验与写作冻结后，另建英文 KBS 模板稿和匿名复现包；中文稿不作为投稿文件。LightGBM 完整 80%/90% × 1h/6h 网格在本协议前三部分完成、结论固定后才启动，并须另有运行清单。

