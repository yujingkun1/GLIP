# GLIP BRCA Visium/Xenium 计划骨架 v2（供下一轮修改）

## 0. 本文件用途
本文件用于承接历史 ralplan/ralph 记忆，并把后续需要用户修改的关键计划点收束成一个更可改、可执行、可判定的骨架。
本文件本轮不触发实现，仅作为下一轮 PRD/test-spec 重写前的 planning snapshot。

## 1. 范围冻结
- 仓库：`/data/yujk/GLIP`
- Visium 核心范围：HEST BRCA `SPA119`–`SPA154`
- Xenium 核心范围：`NCBI784`
- 候选附加数据：`NCBI785` 仅作为可选外测候选；若启用，必须显式披露用途（外测 / 训练 / 调参）
- 当前 baseline 主参照：GLIP

## 2. 计划总原则（修订版）
1. 核心方法模块仍然遵守“一次只加一个主要变化”。
2. 正式可晋级结果与探索性结果必须分轨记录，不能混写。
3. smoke run 只用于 pipeline 打通；full run 才进入正式比较。
4. 同一阶段内允许探索性小试，但不得自动视为晋级依据。
5. 重大代码与实验口径修改必须写阶段文档、更新 ledger，并 git commit + push。

## 3. 双轨结构
### 3.1 正式线（promotion-eligible）
用途：最终进入主表、决定是否进入下一 stage。
要求：
- 使用冻结的数据范围与 split manifest
- 使用冻结的正式评测脚本
- 达到本文件定义的 full-run 条件
- 结果写入正式 ledger 条目，带 promotion_decision

### 3.2 探索线（exploratory / not promotion-eligible）
用途：快速筛掉明显无效方案、验证资源可行性、寻找值得进入正式线的候选。
要求：
- 必须显式标注 exploratory
- 可以缩小数据或 epoch，但不能作为正式晋级依据
- 只能影响“是否值得进入正式 full run”，不能直接推进 stage

## 4. 正式比较锚点
### 4.1 正式 split policy
- **Visium 正式主 protocol：sample-level leave-one-out**
  - 理由：与现有 GLIP/BLEEP 类结果更可比，且已在历史执行中作为主要 protocol 使用。
- **Visium patient-holdout**：保留为更严格泛化附加评测，不作为当前主晋级 protocol。
- **Xenium 正式主 protocol：NCBI784 region/x-position holdout（固定 fold4 为 test）**
  - 理由：当前只有一个核心 BRCA Xenium donor，无法宣称 donor-level 泛化。

### 4.2 正式 baseline comparator
当前统一规定：
1. **Visium comparator**：GLIP Visium-only 在正式主 protocol 下的可接受锚点（若 full 36-fold 代价过高，可先定义 reduced-but-frozen comparator，并明确标为 interim official anchor）
2. **Xenium comparator**：GLIP Xenium pseudo-spot 在固定 split 下的正式锚点
3. 用户手动 baseline：作为方向性外部参考，不直接作为唯一晋级门槛

## 5. Stage 1 重新定义
### 旧定义
- 严格复现两个 baseline

### 新定义
- **冻结并确认可接受的正式比较锚点**

### Stage 1 通过条件
以下全部满足才算 Stage 1 完成：
1. Visium comparator 已明确为以下两者之一：
   - full official anchor，或
   - reduced-but-frozen interim official anchor
2. Xenium comparator 已明确为正式官方锚点
3. 两个 comparator 的数据、split、metrics、run_dir、脚本口径已经写入 ledger / 阶段文档
4. 后续 Stage 2+ 均明确以这些 comparator 作为正式比较锚点

## 6. Stage 2 重新定义
### 方法范围
- 仅允许：naive joint training / pseudo-spot unified training
- 明确不允许：shared/private、OT、completion、cell-level refinement

### Stage 2 full-run 最小定义
一个 Stage 2 结果只有同时满足以下条件，才可视为 full run：
1. 使用正式冻结的数据范围
2. 使用正式主 split policy
3. 使用当前阶段约定的完整训练规模（不得是 smoke/capped toy subset）
4. 至少完成当前阶段定义的完整训练轮次或完整早停规则
5. 完整输出 Visium 与 Xenium 主指标
6. 结果、配置、run_dir、commit_sha 写入正式 ledger

### Stage 2 正式通过线
Stage 2 进入下一阶段，必须满足：
1. full run 稳定完成；
2. 相比上一版 Stage 2 full run，有可复述的整体改进；
3. Visium / Xenium 任一侧不能出现灾难性退化；
4. 与 Stage 1 comparator 比较时，作为方向性参考不出现明显不可接受落差；
5. 若失败原因为基础设施问题，则记为 `infra-blocked`，不得直接判为方法失败；
6. 若 full run 成功但效果不佳，则记为 `method-blocked`，需留在 Stage 2 继续调参或重构同阶段方法。

## 7. 资源 / 基础设施约束
### 7.1 资源预算
- 需要为每个正式 full run 明确记录：GPU、batch size、eval batch size、num_workers、是否 AMP、是否 gradient accumulation
- 若 OOM，需在阶段文档中标注触发位置（train / optimizer init / eval / bank construction）

### 7.2 smoke / pilot / full 的定义
- **smoke**：仅验证 pipeline 通路；数据规模和训练轮次都可缩小；不得用于晋级
- **pilot**：较大规模试跑，用于估计 full-run 可行性；不得自动用于晋级
- **full**：满足 Stage 2 full-run 最小定义，才是正式结果

### 7.3 blocked 分类
- `infra-blocked`：OOM、磁盘、卡被占、评测链路超长但方法逻辑未被证伪
- `method-blocked`：资源可跑通，但主指标无法达到当前阶段预期

## 8. 阶段结构（重排版）
### Stage 0
冻结数据、split、metrics schema、ledger

### Stage 1
冻结正式 comparator（而非执着于所有 baseline 都完整复现）

### Stage 2
最小 joint training

### Stage 3
逐模块增强（候选顺序待用户确认）：
1. platform-conditioned decoder / token
2. shared/private latent split
3. anti-collapse
4. OT alignment

### Stage 4（建议拆子轨）
- Stage 4A：held-out gene completion
- Stage 4B：Xenium cell-level refinement head
- Stage 4C：外部 BRCA 泛化测试

### Stage 5
收尾汇总、主表/附表、最终核查

## 9. 历史执行状态摘要
- Stage 0：已完成
- Stage 1：已有大量执行证据，但“正式锚点定义”仍待补齐
- Stage 2：已有 smoke / target-bank smoke / OOM pilot；当前结论是**仍停留在同阶段修正，不具备晋级条件**

## 10. 下一轮最需要用户修改确认的 5 个问题
1. 是否接受“正式线 / 探索线”双轨结构？
2. Visium comparator 是必须 full 36-fold，还是允许先用 frozen reduced protocol 作为 interim official anchor？
3. Stage 2 的“灾难性退化”你想怎么定义？
4. Stage 3 的模块顺序是否调整？
5. `NCBI785` 是否允许进入计划；若允许，只做外测还是也可参与训练？

## 11. ADR（当前 planning snapshot）
### Decision
将原始单线 staged plan 修订为“正式线 + 探索线”的双轨计划骨架，并把 Stage 1/Stage 2 的通过线与资源约束写明，作为下一轮用户修改的基础。

### Drivers
- 现实存在 OOM / 长评测 / 吞吐限制
- 历史执行已部分突破原始阶段门禁
- 当前最需要的是可修改、可收束的计划骨架，而不是继续口头摘要

### Alternatives considered
- 维持原 PRD 不动，只做口头解释：被拒绝，因为关键歧义仍未显式收束
- 直接进入新 PRD 重写：被拒绝，因为用户尚未给出修改决定

### Why chosen
该骨架保留原计划的实验纪律，同时允许把现实探索行为纳入规则而非留在灰区。

### Consequences
下一轮用户可直接在本骨架上修改关键决策项，再生成新的正式 PRD / test-spec。

### Follow-ups
- 用户确认修改项
- 基于确认结果生成新 PRD
- 再生成新 test-spec 与执行顺序
