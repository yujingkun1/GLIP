# GLIP BRCA Visium/Xenium 计划骨架 v3（Nature Communications 实验版，待执行）

## 0. 当前状态
- 本文件只更新计划，不启动实验。
- 下一条用户命令才进入执行。
- 所有正式比较必须基于 **full run**，不得用 smoke 结果直接做论文级比较结论。

## 1. 论文目标与标准
目标是按 **Nature Communications** 标准组织实验：
- 问题定义清楚
- 比较设置公平
- 外部泛化实验独立成节
- 功能性扩展实验（gene completion / cell-level decoder）单独验证
- 生物学意义验证作为独立结果模块，而不是附带分析
- 所有主结论都基于正式 full run，而不是 smoke / 临时试跑

## 2. 核心数据范围（当前冻结）
### 核心训练集
- Visium：HEST BRCA `SPA119`–`SPA154`
- Xenium：`NCBI784`

### 候选泛化集
- Xenium 候选：`NCBI785`
- HEST 其他 Visium 候选：待核实哪些属于乳腺癌、且与核心训练批次不同

### 当前原则
- 若引入新数据，必须显式披露：
  - 数据名
  - 是否 HEST 内部数据
  - 用途（训练 / 外测 / 调参）
  - 是否参与正式主表

## 3. 正式线 / 探索线
### 正式线
用于论文主表、晋级判断、正式结论。

### 探索线
仅用于：
- 筛选方案
- 验证资源可行性
- 决定是否值得跑 full

**探索线结果不得直接写成论文正式比较结论。**

## 4. Nature Communications 版本的核心实验结构

### Experiment 1：双模态联合训练是否优于单模态训练
目标：回答“同时使用 Visium + Xenium 训练，是否比只用单一模态训练更好”。

#### 1A. Visium 测试集评估
在留出的 **Visium test** 上比较：
- Visium-only 模型
- Xenium-only 迁移/不适配对照（若可定义）
- Visium+Xenium joint 模型

主问题：
- joint 训练是否优于只用 Visium 训练
- joint 是否提高稳定性与泛化，而不是仅提升单一指标

#### 1B. Xenium 测试集评估
在留出的 **Xenium test** 上比较：
- Xenium-only 模型
- Visium-only 迁移/不适配对照（若可定义）
- Visium+Xenium joint 模型

主问题：
- joint 训练是否优于只用 Xenium 训练
- 是否证明跨模态信息对高分辨率空间表达预测有增益

#### 1C. 正式比较要求
- 必须基于 **full run**
- Visium 和 Xenium 两侧都要报告主指标
- 不允许用 smoke 直接支持“joint 更好”的论文结论

---

### Experiment 2：未见批次 / 未见数据泛化能力
目标：回答“联合训练得到的表示，是否对未见批次/未见数据有更强泛化性”。

#### 2A. 泛化设定
优先寻找：
- **HEST 内部、与训练核心集不同批次/不同来源的 BRCA 数据**
- 若严格 BRCA 不足，则在论文中清晰说明退而求其次的外测策略

#### 2B. 当前候选
- Xenium：`NCBI785`（优先作为外部/未见数据测试候选，默认不直接并入核心训练）
- Visium：从 HEST 现有非 `SPA119–154` 的候选中筛选适合的乳腺/相近分布数据

#### 2C. 比较目标
比较：
- 单模态训练模型
- joint 模型
在 **未见批次/未见数据** 上的表现差异。

#### 2D. 要点
- 该实验是论文强泛化性证据的关键模块
- 数据筛选要非常谨慎，避免把“相同病人/相同批次/信息泄露”误写成泛化
- 执行前必须先补一份独立的数据审计说明

---

### Experiment 3：随机 mask 基因的 gene completion / cross-platform completion
目标：回答“是否能借助 Xenium 训练到的信息，补全 Visium 原本没有或被 mask 的基因表达”。

#### 3A. 实验设定
- 在共同基因空间与扩展基因空间中分别设计 mask 方案
- 对 Visium 随机 mask 一部分基因
- 评估模型是否能恢复这些被 mask 的表达

#### 3B. 重点比较
- Visium-only completion
- joint training completion
- 不同 mask ratio / 不同基因子集

#### 3C. 论文价值
这是“跨平台互补信息是否真正有用”的功能性证据，不只是主任务副产物。

---

### Experiment 4：单细胞 decoder 头
目标：在已有 joint/shared 表示基础上，加一个 **cell-level decoder head**，验证模型是否能预测 Xenium 单细胞表达。

#### 4A. 任务定义
- 在 spot/pseudospot 主干基础上增加 cell-level decoder
- 主干共享，decoder 分头
- 该实验单独成节，不与 Stage 2 naive joint 混合汇报

#### 4B. 比较问题
- 加 cell-level head 后，Xenium cell-level 预测是否有效
- 是否会影响 spot-level / pseudospot-level 主任务
- 是否存在多任务收益或冲突

#### 4C. 报告要求
- 单独报告 cell-level 指标
- 同时报告主任务是否退化

---

### Experiment 5：生物学意义验证
目标：证明模型输出不仅数值更好，而且更有生物学解释性与转化意义。

#### 5A. pathway analysis
- 比较真实表达与预测表达在 pathway 活性层面的相关性/一致性

#### 5B. biomarker analysis
- 关注关键乳腺癌 biomarker 的空间表达模式是否被正确恢复

#### 5C. survival / clinical relevance
- 若数据条件允许，进一步分析预测表达构建的 signature 是否保留与生存/临床分层相关的趋势

#### 5D. 原则
- 生物学实验不能只做“看起来合理”的图
- 必须是可重复、可定量、可与真实表达对照的分析模块

## 5. 技术迭代线（保留）
技术路线总体不变，仍按模块逐步加：
1. naive joint training
2. platform-conditioned decoder / token
3. shared/private latent split
4. anti-collapse
5. OT alignment
6. gene completion branch
7. cell-level decoder head

但现在这些模块必须服务于上面的 **5 个论文核心实验问题**，而不是只做技术堆叠。

## 6. 正式比较规则（强化版）
1. **只允许 full run 进入正式比较表**
2. smoke 只能证明 pipeline 通了
3. pilot 只能证明值得跑 full
4. Nature Communications 主结论必须来自：
   - 正式冻结数据
   - 正式冻结 split
   - 正式 full run
   - 正式 ledger 记录

## 7. 下一步执行前必须补齐的计划细节
在你下一条命令开始执行前，我需要按这个新计划进一步冻结：
1. Experiment 2 的未见批次/未见数据候选清单
2. Experiment 3 的 mask 策略与评价指标
3. Experiment 4 的 cell-level decoder 设计边界
4. Experiment 5 的 pathway / biomarker / survival 具体分析方案
5. “灾难性退化”阈值定义

## 8. 当前最核心的 planning decision
后续执行将不再只是“技术 stage 推进”，而是转成 **Nature Communications 论文实验框架驱动**：
- 先围绕论文问题设计实验
- 再决定每个技术模块是否值得进入 full run

## 9. 当前已锁定的数据审计结论（Ralph fresh evidence）
### 官方依据
- HEST 官方 README（本地缓存）明确支持通过 metadata CSV 按 `organ == Breast` 与 `oncotree_code == IDC` 过滤，再下载对应子集。
- 证据文件：`/data/yujk/GLIP/.omx/plans/hest_official_evidence_20260415T071504Z.md`

### 本地候选池 fresh evidence
- 本地额外 Visium 候选数：52
- 本地额外 Xenium 候选数：4
- 候选池文件：`/data/yujk/GLIP/.omx/plans/local_hest_candidate_pool_20260415T071504Z.json`

### 当前先冻结的未见数据候选策略
1. **未见 Xenium 外测首选：`NCBI785`**
   - 原因：HEST 本地已存在；与核心训练 Xenium `NCBI784` 分离；最适合作为第一优先未见 Xenium 外测候选
2. **未见 Visium 外测：待官方 metadata 审计后，从本地额外 Visium 候选中筛出 breast/IDC 优先样本**
3. 在 metadata 审计完成前：
   - 不把任何新样本并入核心训练
   - 仅先把 `NCBI785` 冻结为优先外测候选


## 10. User override on unseen-data experiment (2026-04-15)
- Formal unseen-data generalization must use **the same cancer type but a different tissue**.
- User explicitly clarified that the current local candidate pool does **not** satisfy this requirement.
- Therefore Experiment 2 now requires an **official HEST search + download** of a qualifying dataset before formal execution.
- Local candidates may still be exploratory references, but not the formal unseen-data benchmark.

## 11. User hard gate for Experiment 2 (2026-04-15)
- Formal unseen-data generalization must satisfy **same cancer type + different tissue**.
- Current local candidates are insufficient for the formal benchmark.
- Therefore Experiment 2 cannot formally start until an official HEST search/download identifies a qualifying dataset.
- Any local candidate analysis before that is exploratory only and cannot enter the paper's formal generalization table.

## User clarification (BRCA scope)
- The study cancer type is fixed to **BRCA / breast cancer**.
- Current core samples (`SPA119-154`, `NCBI784`) are all on the BRCA line.
- Therefore Experiment 2 should be interpreted as: **same cancer type = BRCA/IDC breast cancer**, **different tissue = non-breast tissue such as lymph node metastasis**.

## BRCA vs IDC wording note
- The study is on the **BRCA / breast cancer line**.
- In official HEST metadata, the relevant core samples are labeled with the more specific oncotree subtype **IDC** rather than the broader code `BRCA`.
- Therefore formal Experiment 2 should be described as: same BRCA line, specifically **IDC**, transferred from **Breast** tissue to **Lymph node** tissue.


## 12. Experiment 2 formal dataset freeze (fresh evidence)
- Formal unseen-data benchmark is now grounded locally using downloaded official HEST samples `NCBI681-684`.
- These are Visium lymph-node IDC samples, used to evaluate BRCA/IDC-line generalization from Breast tissue to Lymph node tissue.
- Local root: `/data/yujk/hovernet2feature/HEST/hest_data_experiment2_batch_hestenv_dbg`
