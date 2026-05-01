# BRCA Visium/Xenium 分阶段实验计划（Ralph 执行版）

## 任务范围
- Visium：仅使用 HEST BRCA `SPA119`-`SPA154`
- Xenium：核心使用 `NCBI784`
- 允许后续为泛化性补充额外 BRCA 外部数据或本地 `NCBI785`
- 代码结构参考：`/data/yujk/GLIP`、`/data/yujk/BLEEP`
- 基线以 GLIP 为主

## 已确认本地数据
- HEST Visium BRCA 本地可用：`/data/yujk/hovernet2feature/HEST/hest_data/st/SPA119.h5ad` 至 `SPA154.h5ad`
- HEST Xenium 本地可用：`/data/yujk/hovernet2feature/HEST/hest_data_Xenium/st/NCBI784.h5ad`
- 额外本地 Xenium：`NCBI785`（Breast，同 study），`TENX120`/`TENX139`/`TENX157`（非乳腺，不纳入核心训练）
- GLIP 仓库已含 Xenium 处理产物：`/data/yujk/GLIP/processed/NCBI784_*`

## 总原则
1. **从最核心模块开始，一个一个加。**
2. **任何阶段只允许引入一个核心变化。**
3. **只有在完全相同 train/test 数据、相同 split、相同评测脚本下，当前阶段全面优于上一阶段，才能进入下一阶段。**
4. 如果未超越上一阶段，只能：
   - 调整该阶段参数/训练策略，或
   - 回退该模块设计并重做
   不允许直接跳到下一阶段。
5. 每个阶段必须产出：
   - 一个阶段记录 markdown
   - 一条或多条 CSV 账本记录
6. 每次较大改动必须：
   - git commit
   - push 到对应远程仓库

## 比较口径补充（2026-04-14 用户更新）
- 由于当前 Stage 2 使用的共享基因空间与用户手动 baseline 的基因集合不同，**不再要求绝对数值必须严格超过原 baseline**。
- baseline 作为方向性参考，重点看：
  1. 各模块是否稳定跑通；
  2. full run 是否相对上一阶段/上一配置提升；
  3. 是否避免一侧明显退化。
- **smoke run 仅用于打通 pipeline，不用于正式比较或晋级判断。**
- 只有 full run（完整配置、完整目标集或当前阶段约定的正式训练规模）才进入对比表与晋级判断。
- 每个模块必须保留独立参数接口，便于用户手动控制开关。

## 晋级门槛（默认）
当前阶段晋级到下一阶段的必要条件：
- 使用与上一阶段**完全相同**的数据清单与 split manifest
- 在主指标上满足：
  - Visium 主测试指标不低于上一阶段
  - Xenium 主测试指标不低于上一阶段
  - 至少一侧主指标有明确提升
- 若加入的是“泛化模块”，还需：
  - 外部/低资源泛化指标不低于上一阶段
- 若出现一侧提升、一侧明显下降：判定为**未晋级**

## 主指标（先冻结，Stage 0 后可微调一次）
- Visium：overall Pearson, gene Pearson
- Xenium pseudo-spot：overall Pearson, gene Pearson
- Xenium cell-level（仅在 cell head 阶段后纳入）：overall Pearson, gene Pearson
- 辅助指标：MSE/MAE、平台 probe、训练稳定性

## 分阶段结构

### Stage 0：数据与评测冻结
目标：先把“比什么、怎么比、数据怎么切”定死。
输出：
- `dataset_manifest.json/csv`
- `split_manifest.json`
- `metrics_schema.json`
- 统一结果 CSV 账本
- 乳腺 BRCA donor / sample / patient 对照表
- NCBI784 训练/测试分区规则（先 region/x-position holdout，后续如有更多 Xenium BRCA donor 再升级 donor-held-out）

### Stage 1：严格复现基线
1. GLIP Visium-only baseline（SPA119-154 leave-one-out / 固定 protocol）
2. GLIP Xenium-only baseline（NCBI784 固定 split）
3. 必要时复现 BLEEP/HisToGene/His2ST 的已有结果引用表，但不作为首个执行目标
晋级条件：
- 基线结果可重复
- 评测脚本与结果账本冻结

### Stage 2：最小跨平台核心
只加一个最核心能力：
- **naive joint training / pseudo-spot unified training**
不加 shared/private，不加 OT，不加 completion。
目标：先回答“单纯联合训练是否比单平台更稳”。
晋级条件：
- 在相同 split 下，联合训练至少不弱于两个单平台基线，且至少一侧明显提升

### Stage 3：逐模块增强（一次只加一个）
推荐顺序：
1. platform-conditioned decoder / platform token
2. shared/private latent split
3. anti-collapse 机制（双 decoder + domain confusion/private classifier + decorrelation）
4. OT 对齐
每加一个模块都必须重新在同一 split 上对比上一阶段。

### Stage 4：扩展能力
仅在 Stage 3 主结果站稳后进入：
1. held-out gene completion
2. Xenium cell-level refinement head
3. 额外外部 BRCA 泛化测试（优先本地 `NCBI785`，不足时再下载外部 BRCA Xenium/Visium 数据）

### Stage 5：收尾
- 汇总所有阶段文档
- 汇总 CSV 账本
- 画最终主表/附表
- 做最终 architect/verifier 风格核查

## 当前已知关键限制
1. Visium BRCA 有 36 个样本（SPA119-154），可做严格 sample-level leave-one-out。
2. Xenium 核心只有 `NCBI784` 一个 BRCA 主样本；若只用它，**不能宣称 donor-level Xenium 泛化**，最多做 region-held-out / spatial-held-out。
3. 因此若要写强“跨平台泛化”，后续应优先：
   - 检查 `NCBI785` 是否可作为外部 breast Xenium 测试
   - 或补充下载额外 BRCA Xenium / Visium 数据

## 立即执行顺序
1. Stage 0.1：生成 BRCA Visium/Xenium 数据清单与元数据表
2. Stage 0.2：冻结 split 方案与结果账本字段
3. Stage 1.1：复现 GLIP Visium baseline
4. Stage 1.2：复现 GLIP Xenium baseline
5. 只有 Stage 1 稳定后，才开始 Stage 2
