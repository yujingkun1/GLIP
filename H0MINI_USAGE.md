# H0-mini Integration for GLIP

本文档说明如何在 GLIP 中使用 H0-mini 作为图像编码器。

## 概述

H0-mini 是 Bioptimus 在 4300 万病理图像上预训练的 ViT-Base/14 模型，专门针对组织学图像优化。

**关键特性**：
- 输入尺寸：224×224
- Patch 划分：16×16 grid（每个 patch 14×14 像素）
- 输出维度：768
- 两种输出模式：
  - `pooled`：CLS token（用于 spot-level 任务）
  - `patch_tokens`：256 个 patch tokens（用于 cell-level 任务）

## 快速开始

### 1. 训练 Visium spot-level 模型

使用 H0-mini 替代 ResNet 或 UNI2-h：

```bash
python train_joint_brca_naive.py \
    --model_name h0mini \
    --visium_hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --epochs 10 \
    --batch_size 32
```

### 2. 与其他模型对比

```bash
# 使用 ResNet50
python train_joint_brca_naive.py --model_name resnet50 ...

# 使用 UNI2-h
python train_joint_brca_naive.py --model_name uni ...

# 使用 H0-mini
python train_joint_brca_naive.py --model_name h0mini ...
```

### 3. 单细胞级别特征提取

使用 patch tokens 模式提取细胞级别特征：

```python
from glip.visium.modules import ImageEncoder_H0mini
from glip.visium.patch_cell_matching import PatchCellMatcher
import torch

# 1. 加载 H0-mini encoder（patch_tokens 模式）
encoder = ImageEncoder_H0mini(
    output_mode="patch_tokens",
    trainable=False,
    checkpoint_path="/data/yujk/H0-mini/pytorch_model.bin"
)
encoder.eval()

# 2. 提取 patch tokens
image = torch.randn(1, 3, 224, 224)  # 替换为真实图像
with torch.no_grad():
    patch_tokens = encoder(image)  # (1, 256, 768)

# 3. 加载细胞 mask 并匹配
matcher = PatchCellMatcher()
cell_masks = matcher.load_cell_masks_from_parquet("cells.parquet")

cell_features, indices, areas = matcher.match_patches_to_cells(
    patch_tokens[0],  # (256, 768)
    cell_masks,
    aggregation='weighted_mean'  # 按重叠面积加权
)

print(f"提取了 {cell_features.shape[0]} 个细胞的特征")
# cell_features: (num_cells, 768)
```

## 实现细节

### 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `glip/visium/config.py` | 添加 H0-mini 配置常量 |
| `glip/visium/modules.py` | 添加 `ImageEncoder_H0mini` 类 |
| `glip/visium/models.py` | 添加 `CLIPModel_H0mini` 类 |
| `glip/visium/dataset.py` | 支持 H0-mini 归一化参数 |
| `glip/visium/patch_cell_matching.py` | 新增 patch-cell 匹配模块 |
| `glip/xenium/config.py` | 添加 H0-mini 配置 |
| `glip/xenium/pseudospot.py` | 支持 H0-mini 归一化 |
| `train_joint_brca_naive.py` | 添加 H0-mini 模型分支 |

### 归一化参数

H0-mini 使用与 ImageNet 不同的归一化参数：

```python
# H0-mini
mean = [0.707223, 0.578729, 0.703617]
std = [0.211883, 0.230117, 0.177517]

# ImageNet（其他模型）
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

数据集会根据 `model_name` 参数自动选择正确的归一化。

### 参数冻结

默认情况下，H0-mini 的参数是冻结的（`trainable=False`），只训练投影头。这样可以：
- 加快训练速度
- 减少内存占用
- 保持预训练特征的质量

## Patch-Cell 匹配

`PatchCellMatcher` 类用于将 patch tokens 与细胞 mask 进行空间匹配。

### 工作原理

1. **空间映射**：每个 patch token 对应图像中 14×14 像素的区域
2. **重叠计算**：计算每个细胞 mask 与哪些 patches 重叠
3. **特征聚合**：根据重叠情况聚合 patch 特征

### 聚合方式

- `mean`：简单平均
- `max`：最大池化
- `weighted_mean`：按重叠面积加权平均（推荐）

### 示例

```python
from glip.visium.patch_cell_matching import match_patches_to_cells
from shapely.geometry import Point

# 创建测试细胞 mask
cell_masks = [
    Point(112, 112).buffer(20),  # 中心细胞
    Point(50, 50).buffer(15),    # 左上角细胞
]

# 匹配
cell_features, indices, areas = match_patches_to_cells(
    patch_tokens,
    cell_masks,
    aggregation='weighted_mean'
)

# 查看每个细胞覆盖的 patches
for i, idx_list in enumerate(indices):
    print(f"细胞 {i} 覆盖 {len(idx_list)} 个 patches: {idx_list}")
```

### 可视化

```python
matcher = PatchCellMatcher()
matcher.visualize_matching(
    patch_tokens,
    cell_masks,
    save_path='patch_cell_matching.png'
)
```

## 命令行参数

训练脚本支持以下 H0-mini 相关参数：

```bash
--model_name h0mini          # 使用 H0-mini 编码器
--model_name h0-mini         # 别名（同上）
```

其他参数与现有模型相同：
- `--batch_size`：批次大小
- `--epochs`：训练轮数
- `--lr`：学习率
- `--visium_hest_data_dir`：Visium 数据目录
- `--xenium_sample_id`：Xenium 样本 ID

## 性能对比

### 模型对比

| 模型 | 参数量 | 输出维度 | 预训练数据 |
|------|--------|---------|-----------|
| ResNet50 | 25M | 2048 | ImageNet |
| UNI2-h | 307M | 1536 | 100M 病理图像 |
| H0-mini | 86M | 768 | 43M 病理图像 |

### 优势

- **专门优化**：针对病理图像预训练
- **空间信息**：支持 patch tokens 提取细胞级别特征
- **参数效率**：比 UNI2-h 小 3.5 倍
- **灵活性**：支持 spot-level 和 cell-level 任务

## 故障排查

### 问题：模型加载失败

**错误**：`403 Client Error` 或 `Cannot access gated repo`

**解决**：代码已更新为从本地加载，无需 Hugging Face 授权。确保权重文件存在：
```bash
ls -lh /data/yujk/H0-mini/pytorch_model.bin
```

### 问题：归一化参数错误

**症状**：训练损失异常高或不收敛

**解决**：确保数据集初始化时传入了 `model_name` 参数：
```python
dataset = CLIPDataset(..., model_name='h0mini')
```

### 问题：内存不足

**解决**：
1. 减小 batch size
2. 使用 `output_mode='pooled'`（而非 `patch_tokens`）
3. 确保参数冻结（`trainable=False`）

## 测试

运行集成测试：

```bash
python test_h0mini_integration.py
```

测试内容：
- ✓ ImageEncoder_H0mini（pooled 模式）
- ✓ ImageEncoder_H0mini（patch_tokens 模式）
- ✓ CLIPModel_H0mini
- ✓ PatchCellMatcher
- ✓ 归一化参数配置

## 许可证

H0-mini 使用 CC-BY-NC-ND-4.0 许可证，仅限非商业学术用途。

## 参考

- H0-mini 模型：https://huggingface.co/bioptimus/H0-mini
- GLIP 项目：/data/yujk/GLIP
