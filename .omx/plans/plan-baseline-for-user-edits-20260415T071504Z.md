# GLIP BRCA Visium/Xenium 当前计划基线版（供修改）

## 目标
在 `/data/yujk/GLIP` 上推进 BRCA 跨平台空间转录组实验计划，当前仅整理历史共识与执行状态，供下一轮计划修改使用；本文件不触发实现。

## 已同步的记忆来源
- 上层上下文：`/data/yujk/.omx/context/brca-visium-xenium-generalization-experiments-20260413T094233Z.md`
- 原始 PRD：`/data/yujk/.omx/plans/prd-brca-visium-xenium-staged-experiment-program-20260413T094908Z.md`
- 原始 ledger：`/data/yujk/.omx/plans/brca_visium_xenium_iteration_ledger.csv`
- 本地同步副本：`/data/yujk/GLIP/.omx/plans/` 下对应文件

## 当前范围冻结
- Visium：HEST BRCA `SPA119`–`SPA154`
- Xenium：核心 `NCBI784`
- 仓库：`/data/yujk/GLIP`
- baseline：GLIP 为主
- 候选附加数据：`NCBI785` 仅作为未来可选外测候选，若使用必须显式披露

## 当前规则冻结
1. 从最核心模块开始，一个一个加。
2. 每个阶段只允许一个核心变化。
3. 仅在相同 train/test 数据、相同 split、相同评测脚本下，当前阶段优于上一阶段，才允许晋级。
4. smoke run 仅用于打通 pipeline，不用于正式比较或晋级。
5. full run 才能写入正式对比并作为晋级依据。
6. 每个阶段必须产出 markdown 记录和 CSV ledger 记录。
7. 重大改动需 git commit + push。

## 当前比较口径
- 因 Stage 2 的共享 gene space 与用户手动 baseline 的 gene set 不完全一致，当前不再强制要求绝对数值严格超过手动 baseline。
- 当前更关注：
  1. pipeline 是否稳定；
  2. full run 是否优于上一版/上一配置；
  3. 是否避免某一侧明显退化。

## 当前阶段结构
### Stage 0
冻结数据、split、metrics schema、ledger。

### Stage 1
复现单平台 baseline：
1. GLIP Visium-only
2. GLIP Xenium-only

### Stage 2
最小跨平台核心：
- naive joint training / pseudo-spot unified training
- 不引入 shared/private、OT、completion

### Stage 3
逐模块增强（候选顺序）：
1. platform-conditioned decoder / token
2. shared/private latent split
3. anti-collapse
4. OT alignment

### Stage 4
扩展能力：
1. held-out gene completion
2. Xenium cell-level refinement head
3. 外部 BRCA 泛化测试

### Stage 5
收尾汇总与最终核查。

## 当前已知限制
1. Visium BRCA 共 36 个样本，可做 sample-level leave-one-out。
2. Xenium 核心仅 `NCBI784` 一个 BRCA 主样本；若只用它，不能宣称 donor-level Xenium 泛化。
3. 若要写强泛化结论，后续需优先检查 `NCBI785` 是否可作为外部 breast Xenium 测试，或补充额外 BRCA 数据。

## 当前执行状态（基于 ledger）
- Stage 0：已完成并冻结。
- Stage 1：
  - Visium baseline 多次因 OOM / 长评测链路受阻；做过 resume patch。
  - Xenium pseudospot baseline 曾达到较好的中间结果（best overall 约 0.7785，best gene 约 0.5594），但整套 Stage 1 并未以清晰“正式完成版”收束进晋级判断。
- Stage 2：
  - smoke1 已跑通但结果明显不足；
  - smoke2 通过 target-specific bank 明显优于 smoke1；
  - 更大 pilot 出现 OOM；
  - 当前仍停留在 Stage 2 同阶段调参 / 修正，未晋级。

## 当前最值得修改的计划点
1. 是否还保留 Stage 1 严格复现作为正式前置。
2. Stage 2 的正式通过线如何定义得更明确。
3. Stage 3 的模块顺序是否调整。
4. 是否把“减少 short / 以 full 为主”写成硬规则。
5. 是否允许引入 `NCBI785`，以及只做外测还是纳入训练。

## 本轮规划目标
下一轮用户将在本基线版上提出修改；修改后再生成新 PRD / 新 test-spec / 新执行顺序。

## Architect 复核摘要（本轮 ralplan）
### 最强反论点
当前计划过于强调方法学纯度和阶段纪律，但在现实资源约束下不够决策高效；若继续把 Stage 1 严格复现作为重前置，可能拖慢真正关键科学问题（跨平台 joint 是否值得继续）的验证速度。

### 核心权衡
1. 严谨性 vs 实验吞吐
2. 基线完备性 vs 提前探索
3. 公平比较 vs 实际可比性
4. 阶段门禁 vs 研究灵活性

### 建议的修改方向
1. 把“正式晋级线”和“探索线”拆开
2. 把 Stage 1 从“严格复现”改成“冻结可接受的比较锚点”
3. 把 Stage 2 通过线改成更可执行的 full-run 稳定 + 同阶段相对提升规则
4. 把资源 / 基础设施约束显式写进计划正文

### 当前主要歧义
1. Stage 1 未完全封账但已进入 Stage 2 探索，和原计划文字有冲突
2. Visium 正式 protocol 是 sample-level LOO 还是 patient-holdout 仍需明确
3. baseline comparator 需要唯一化
4. Stage 4 目标混杂，后续最好拆开
