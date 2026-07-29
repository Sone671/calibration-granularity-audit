# 群体—个体校准冲突诊断Benchmark冻结协议

## 研究问题

不再检验某个新收缩方法是否普遍最优，而是系统回答：在时间漂移下，global、group与user滚动CQR何时发生群体—个体Pareto冲突，策略排序多久反转，pooled指标隐藏多少时间不稳定，以及理想环境路由相对固定策略有多大潜在价值？

## 数据与环境

- London：现有冻结500户矩阵；前12个月训练，随后11个完整自然月逐月测试；
- Ausgrid：前24个月训练，随后12个完整自然月逐月测试；
- UCI Electricity：固定训练截止2013-12，2014年12个月逐月测试；
- 每个测试月只使用开始前56天标签校准；
- 第一阶段使用1小时horizon、80%coverage、LightGBM quantile基础模型；结论稳定后再加入第二模型、90%coverage与额外horizon。

## 校准策略

固定比较`raw`、`rolling_global_norm`、`rolling_group_norm`、`rolling_user_norm`。所有CQR score按训练期用户尺度标准化；group不做额外层次收缩，以直接比较校准粒度。

## 核心指标

1. `Granularity Conflict Rate (GCR)`：user相对global降低用户gap、同时增加最大group gap的窗口比例；同时报告反向冲突。
2. `Rank-Reversal Rate (RRR)`：对lambda `{0,.25,.5,.75,1}`的标量损失`lambda*user_gap+(1-lambda)*group_gap`，策略两两相邻窗口排序符号反转的比例。
3. `Temporal Cancellation Index (TCI)`：`mean_window_user_gap - pooled_user_gap`及相对比值，量化跨月过/欠覆盖抵消。
4. `Routing Oracle Gap (ROG)`：最佳固定粒度的平均损失减去逐窗口oracle粒度的平均损失，量化环境路由的最大潜在收益。

所有指标同时保存逐窗原始值，不只报告汇总平均。

## 第一阶段输出与质量门槛

- London至少10个有效测试窗、每窗至少95%目标用户可评估；
- 保存window、per-user-window、correction与诊断汇总；
- 指标函数通过合成单元测试；
- 不以冲突率高低作为GO/NO-GO，避免按希望的现象筛选；
- 若London基础流程通过，再无修改地迁移到Ausgrid与UCI。

## 禁止事项

- 不加入V1–V5收缩方法挤占主比较；
- 不把数据驱动cluster称为社会公平群体，统一称operational segments；
- 不根据诊断结果挑选月份、lambda或策略；
- 不把方法失败次数作为论文贡献。
