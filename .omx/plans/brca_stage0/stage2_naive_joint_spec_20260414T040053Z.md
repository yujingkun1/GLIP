# Stage 2 设计：最小 naive joint Visium + Xenium pseudo-spot

## 用户新指令
- 单平台 baseline 直接采用用户手动跑出的结果，不再阻塞于复现
- 仍然遵守：一次只加一个核心变化、同样的数据/测试逻辑、记录所有结果

## 本阶段唯一核心变化
**将 Visium spot 与 Xenium pseudo-spot 放入同一个共享 gene space 中做最小 naive joint training。**

不加入：
- shared/private latent
- OT
- completion
- cell-level refinement
- 任何额外数据集

## 共享 gene space
- 使用 `NCBI784_ref_SPA124` pseudo-spot gene list 生成 313-gene shared file：
  - `configs/brca_shared_genes_ncbi784_ref_spa124_313.txt`
- Visium 通过该 gene file 对齐到相同 313 维表达空间
- Xenium pseudo-spot 原生就是这 313 维

## 新脚本
- `train_joint_brca_naive.py`

## 输出目标
- joint bank 同时包含 Visium train + Xenium pseudo-spot train
- 分别评估：
  - held-out Visium sample
  - Xenium fold4

## 说明
这一步是 Stage 2 的最小核心，不代表最终方法，只用于回答：
> 不做任何 fancy 模块，单纯把两平台放到共享 313-gene space 中一起训，是否会在任一侧优于已接受 baseline。
