# Stage 1.2 预案：Xenium baseline

## 目标
在进入跨平台联合训练前，先冻结 Xenium-only baseline，作为后续 Stage 2/3 的对照。

## 推荐分两层
### A. Xenium pseudo-spot baseline（主）
- 用于和未来 joint pseudo-spot unified model 对比
- 核心样本：NCBI784
- split：x-position contiguous 5-fold，固定 fold4 为 test
- 参考 Visium sample：暂定 `SPA124`（沿用 GLIP 默认值，后续阶段保持固定）
- 指标：overall Pearson, gene Pearson

### B. Xenium cell-level baseline（辅）
- 用于后续 cell-level refinement 阶段的参照
- 不作为当前 Stage 2 晋级门槛主指标

## 执行顺序
1. 先等 Visium Stage 1.1 baseline 开跑并确认配置无误
2. 准备 pseudo-spot cache / split manifest
3. 启动 Xenium pseudo-spot baseline
4. 如资源允许，再补 cell-level baseline

## 当前冻结决定
- Xenium 主 baseline 以 pseudo-spot 为主
- cell-level baseline 作为后续扩展参考，不阻塞 Stage 2 最小联合训练
