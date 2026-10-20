# MSB-GP 优化目标：红线验收标准

本文件把当前一轮 MSB-GP 迭代的四条优化目标固化为**可执行的红线验收测试**。
"红线"指：任一测试未通过，则该次变更不得作为发布候选。

## 背景

MSB-GP 是一个多核 REML 基因组预测器，通过 softmax 受限的核权重在
`A（加性） / D（显性） / AA,AD,DD（上位性） / Gaussian / Poly / RSI`
之间做极大似然竞争。历史迭代出现过两类典型失败：

- 复杂核（Gaussian/Poly/上位性）在**纯加性**数据上抢走加性信号，导致预测与
  可解释性同时退化；
- 为修复可解释性而过度收缩非线性核，导致**真实上位性**无法被识别。

因此本轮的优化目标从"细粒度方差分解准确"转向"**预测优先、结构可控**"。
默认 MSB-GP 不调用 GBLUP/RR-BLUP/GMKL；外部基线只作为独立 benchmark 对手。

当前原生候选池为 `core`、`sparse`、`epi`、`full`。其中 `epi` 表示
`additive + dominance + AA/AD/DD` 的显式上位性候选，用泄漏-free inner OOF
判断是否值得进入最终 MSB-GP 预测。

## 四条优化目标

| # | 目标 | 级别 | 验收方式 |
|---|------|------|----------|
| 1 | 纯 A 场景不被复杂 kernel 破坏 | **必须** | P0 加性模拟：MSB-GP（gate 开）OOF Pearson ≥ core-only − 0.02，且给出 gate 是否触发 |
| 2 | A→D 变化能够正确跟踪 | **必须** | P0–P4：A/D 恢复权重随真实 A/D 比例的 Spearman 相关 |
| 3 | 有 epistasis 时能识别 higher-order signal 增加 | **必须** | P0–P8：OOF `higher_order_score` 随真实上位性比例的单调性（Spearman） |
| 4 | 预测性能优先于细粒度方差分解 | **优先声明** | 所有图中以 Pearson/MSE 为主指标；方差分解仅作诊断 |

## 机制：prediction-first higher-order gate

对**每个内层 CV 折**，同时拟合：

- **full** 核库（A + D + AA/AD/DD + Gaussian + Poly + 可选 RSI）
- **core** 核库（仅 A + D，`epi_kernels=False, rsi=False`）

得到两条 leakage-free 的 out-of-fold 预测，定义

```
higher_order_score = r(full_oof, y) - r(core_oof, y)
```

当 `higher_order_score <= gate_threshold`（默认 0.0）时，最终模型**回退到 core 核库**，
并用 core 的 OOF 重新估计仿射校准系数。结果：

- 纯加性数据：高阶核不带来 OOF 增益 → gate 拒绝 → 等价于 GBLUP 的加性/显性基线，
  **不会被复杂核破坏**（目标 1）。
- 显性由 core 集内的 D 核承载，因此 A→D 连续变化不依赖 gate（目标 2）。
- 真实上位性数据：full 相对 core 有 OOF 增益 → gate 通过 → 保留高阶核；
  且 `higher_order_score` 本身随上位性比例上升（目标 3）。
- 因为 gate 的判据是 **OOF 预测**，预测性能天然优先于方差分解（目标 4）。

## 自动红线测试

`tests/test_objective_redlines.py` 覆盖以下数值红线（小规模、秒级）：

1. `test_pure_additive_gate_rejects_or_matches_core`
2. `test_dominance_tracked_by_core`
3. `test_higher_order_score_positive_on_epistasis`
4. `test_gate_disabled_is_backward_compatible`

## 大规模验证

`experiments/run_objective_validation.py` 在 P0–P8（n=800, m=1000, 5-fold）上
对比 `GBLUP / MSB-GP-Core / MSB-GP-Full / MSB-GP(gate)`，输出：

- `objective_validation_summary.csv`（各架构 × 方法的 Pearson/MSE/gate 通过率）
- `objective_validation_weights.csv`（每折核权重，用于目标 2/3）
- `objective_checks.json`（目标 1/3 的自动判定）

## 与旧行为的关系

- 旧默认 = `adaptive_gate=False`（即 `MSB-GP-Full`），可用 `MSBCal(adaptive_gate=False)` 复现。
- 新默认 = `adaptive_gate=True`，且 `n_inner_repeats=1, n_ensemble=1`，符合"收缩到可靠基线"的既有决定。
- gate 只增加内层 CV 的拟合次数，不改变最终模型的公开 API 与输出格式。

## 数值/内存优化红线

当前 1.0.0 发布候选已加入的低风险精确优化：

- `scipy.sparse` CSR/CSC 基因型输入可直接进入 `MSBGP` / `MSBCal`；
- additive、Gaussian、Poly 的训练/交叉核使用稀疏友好的中心化 Gram 公式，避免为了这些核先构造稠密中心化基因型；
- dominance 和显式 epistasis 在需要 0/1/2 编码时只对必要子矩阵稠密化；
- 训练核和交叉核按需构造，默认不常驻保存全部 `n x n` kernel；
- dense 与 sparse 路径在 smoke test 中要求在 `rtol=1e-9, atol=1e-9` 内一致；
- `save_genotype_memmap` / `load_genotype_memmap` 支持 int8 `.npy` 或 `.dat`+JSON 基因型存储，模型可直接消费 memmap，dense-vs-memmap 路径要求在 `rtol=1e-9, atol=1e-9` 内一致；
- REML 支持 `reml_init="heuristic"` / `reml_init="multi_start"` 与 `reml_multistart`，多起点选择是确定性的，并记录收敛状态、梯度范数、迭代次数、边界参数和所选起点。

仍未作为默认启用的近似优化：Nyström/RFF/partial Cholesky、Hutchinson/random trace、
AI-REML、GPU mixed precision。它们只有在专门验证预测精度不退化后，才能进入正式默认流程。


