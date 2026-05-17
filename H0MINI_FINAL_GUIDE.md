# ✅ H0-mini 使用说明（最终版）

## 🚀 一行命令启动完整实验

```bash
cd /data/yujk/GLIP

# Visium 5-Fold CV (30 epochs, 推荐)
./run_h0mini_experiment.sh visium 30

# Xenium (20 epochs, 推荐)
./run_h0mini_experiment.sh xenium 20

# Joint 5-Fold CV (30 epochs, 推荐)
./run_h0mini_experiment.sh joint 30
```

## 📁 目录结构（所有 folds 在一个文件夹）

```
/data/yujk/GLIP/experiments/
├── h0mini_visium/
│   └── h0mini_visium_5fold_30ep_20260512_143022/    ← 一次完整实验
│       ├── experiment_summary.txt                    ← 实验总体信息
│       ├── fold0/                                    ← 第 0 个 fold
│       │   ├── training.log                          ← 训练日志
│       │   ├── fold_info.txt                         ← Fold 信息
│       │   ├── checkpoints/                          ← 模型权重
│       │   └── results/                              ← 评估结果
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       └── fold4/
```

## 🔍 查看实验结果

```bash
# 1. 查看实验总体信息
cat /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/experiment_summary.txt

# 2. 查看某个 fold 的训练日志（实时）
tail -f /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/fold0/training.log

# 3. 查看所有 folds 的状态
cat /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/fold*/fold_info.txt

# 4. 查看全局实验日志
cat /data/yujk/GLIP/experiments/experiment_log.txt
```

## ⏱️ 预计时间

| 任务 | Epochs | 时间/Fold | 总时间 (5 folds) |
|------|--------|----------|-----------------|
| Visium | 30 | ~75 小时 | ~375 小时 (15.6 天) |
| Xenium | 20 | ~50 小时 | ~50 小时 (2.1 天) |
| Joint | 30 | ~75 小时 | ~375 小时 (15.6 天) |

**注意**：Folds 是顺序执行的，不是并行。

## 📊 实验命名规范

```
h0mini_{task}_5fold_{epochs}ep_{timestamp}
```

示例：
- `h0mini_visium_5fold_30ep_20260512_143022`
- `h0mini_xenium_rep1_20ep_20260512_143022`
- `h0mini_joint_5fold_30ep_20260512_143022`

## ✨ 自动记录的信息

### 实验级别（`experiment_summary.txt`）
```
Experiment: H0-mini Visium 5-Fold Cross-Validation
Date: 20260512_143022
Model: H0-mini (frozen)
Task: Visium spot-level prediction
Total Folds: 5
Epochs per Fold: 30
Batch Size: 32
Data: BRCA Visium (36 samples)
Genes: 227 shared genes
Notes: Baseline H0-mini experiment, no VAE
Command: ./run_h0mini_experiment.sh visium 30
```

### Fold 级别（`fold_info.txt`）
```
Fold: 0/5
Epochs: 30
Status: Completed at Mon May 12 14:30:22 CST 2026
```

## 🎯 快速测试（1 epoch）

```bash
# 快速验证代码（仅用于测试，不适合实际实验）
./run_h0mini_experiment.sh visium 1
```

## 🆚 与 ResNet 对比

只需修改训练脚本中的模型参数：
- H0-mini: `--model h0mini`
- ResNet50: `--model resnet50`

其他参数完全相同！

## 📝 完整文档

- **`H0MINI_INTEGRATION_COMPLETE.md`** - 完成总结
- **`H0MINI_QUICKSTART.md`** - 快速开始
- **`H0MINI_STANDARDIZED_COMMANDS.md`** - 规范化命令
- **`H0MINI_PREDICTION_COMMANDS.md`** - 详细参数说明
- **`H0MINI_USAGE.md`** - 实现细节和 API

## 🔧 故障排查

### 问题：训练中断

**解决**：查看日志找到错误原因
```bash
tail -100 /data/yujk/GLIP/experiments/h0mini_visium/*/fold*/training.log
```

### 问题：内存不足

**解决**：减小 batch size
```bash
# 修改脚本中的 --batch_size 32 为 --batch_size 16
```

### 问题：想要并行运行多个 folds

**解决**：手动启动多个 GPU
```bash
# GPU 0: fold 0
CUDA_VISIBLE_DEVICES=0 python train_visium.py --fold_index 0 ... &

# GPU 1: fold 1
CUDA_VISIBLE_DEVICES=1 python train_visium.py --fold_index 1 ... &
```

## 💡 最佳实践

1. **先测试**：用 1 epoch 快速验证
   ```bash
   ./run_h0mini_experiment.sh visium 1
   ```

2. **监控训练**：使用 tmux 或 screen
   ```bash
   tmux new -s h0mini
   ./run_h0mini_experiment.sh visium 30
   # Ctrl+B, D 分离会话
   ```

3. **定期检查**：每天查看一次进度
   ```bash
   tail -20 /data/yujk/GLIP/experiments/h0mini_visium/*/fold*/training.log
   ```

4. **备份结果**：实验完成后立即备份
   ```bash
   tar -czf h0mini_visium_backup.tar.gz experiments/h0mini_visium/
   ```

## 🎉 开始实验

```bash
cd /data/yujk/GLIP
./run_h0mini_experiment.sh visium 30
```

祝实验顺利！🚀
