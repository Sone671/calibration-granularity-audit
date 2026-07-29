# LightGBM 完整网格冻结协议

冻结日期：2026-07-29。本协议在运行 LightGBM 的 80/90% × 1/6h 网格前写入。它补足学习型预测器的实验平衡性，不改变 ACI、CSGR 或主诊断的既有结论。

## 设计

- 数据：London、Ausgrid、UCI Electricity；沿用冻结的训练期、训练期用户筛选、KM3 分群、56 天校准块和自然月测试窗口。
- 预测器：每个 `dataset × horizon` 以冻结特征、随机种子、600,000 条训练样本、250 轮训练三个 LightGBM 分位数模型（0.1/0.5/0.9）。1h 和 6h 分别训练，避免把 1h 模型的输出事后当作 6h 预测。
- 校准：对每个预测窗口和覆盖率 `tau in {0.80, 0.90}`，在四种静态策略 `raw/global/segment/user` 上计算归一化 split-CQR 修正；`tau=0.90` 仅改变 CQR 目标覆盖率，不以测试期标签调参。
- 结果：逐 `dataset × coverage × horizon × month × method` 写出 PICP、MPIW、Winkler interval score、宏用户 coverage gap 与最大分群 gap，并写出用户、校正量、GCR、RRR、TCI 和 ROG 所需明细。

## 运行与判读

完整网格预计有 140 个 `configuration × month` 环境：London 44、Ausgrid 48、UCI 48。它与 persistence 使用同一网格，从而可检验预测器、覆盖率和跨度的交互；它不是 ACI 或 CSGR 的新调参数据。原有 LightGBM 80%/1h 输出保留不覆盖；运行后先进行逐指标复现审计，再允许合入论文。

所有完整性检查、运行参数、代码 SHA-256 和结果目录将在运行完成报告中记录。
