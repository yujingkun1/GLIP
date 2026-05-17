# H0-mini 实验快速指南

## 🚀 最简单的使用方式（自动 5-Fold）

使用提供的脚本，一行命令启动规范化的 5-fold 交叉验证实验：

```bash
cd /data/yujk/GLIP

# Visium 5-Fold CV (30 epochs each)
./run_h0mini_experiment.sh visium 30

# Xenium (20 epochs)
./run_h0mini_experiment.sh xenium 20

# Joint 5-Fold CV (30 epochs each)
./run_h0mini_experiment.sh joint 30
```

**注意**：Visium 和 Joint 会自动运行 5 个 folds，Xenium 只运行一次。

## 📁 自动生成的目录结构

```
/data/yujk/GLIP/experiments/
├── h0mini_visium/
│   └── h0mini_visium_5fold_30ep_20260512_143022/    # 一个实验的所有 folds
│       ├── experiment_summary.txt                    # 实验总体信息
│       ├── fold0/
│       │   ├── training.log
│       │   ├── fold_info.txt
│       │   ├── checkpoints/
│       │   └── results/
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       └── fold4/
├── h0mini_xenium/
│   └── h0mini_xenium_rep1_20ep_20260512_143022/
└── h0mini_joint/
    └── h0mini_joint_5fold_30ep_20260512_143022/     # 一个实验的所有 folds
        ├── experiment_summary.txt
        ├── fold0/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        └── fold4/
```

**优势**：
- ✅ 所有 5 个 folds 在同一个大文件夹
- ✅ 便于管理和对比
- ✅ 一个时间戳对应一次完整实验

## ✅ 自动记录的信息

每个实验自动保存：
- ✅ 完整的训练日志（`training.log`）
- ✅ 实验配置信息（`experiment_info.txt`）
- ✅ 时间戳（精确到秒）
- ✅ 使用的命令
- ✅ 全局实验日志（`experiments/experiment_log.txt`）

## 📊 运行 5-Fold 交叉验证

```bash
# 依次运行 5 个 folds
for fold in 0 1 2 3 4; do
    ./run_h0mini_experiment.sh visium ${fold} 30
done
```

或者使用后台运行：
```bash
nohup ./run_h0mini_experiment.sh visium 0 30 > fold0.out 2>&1 &
nohup ./run_h0mini_experiment.sh visium 1 30 > fold1.out 2>&1 &
# ... 依此类推
```

## 🔍 查看实验结果

```bash
# 查看所有实验
ls -lt /data/yujk/GLIP/experiments/h0mini_visium/

# 查看实验总体信息
cat /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/experiment_summary.txt

# 查看某个 fold 的训练日志
tail -f /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/fold0/training.log

# 查看所有 folds 的状态
cat /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_*/fold*/fold_info.txt

# 查看全局实验日志
cat /data/yujk/GLIP/experiments/experiment_log.txt
```

## 📝 推荐的 Epoch 设置

| 任务 | 推荐 Epochs | 预计时间 (单GPU) |
|------|------------|-----------------|
| Visium | 30-50 | 75-125 小时 |
| Xenium | 20-30 | 50-75 小时 |
| Joint | 30-50 | 75-125 小时 |

## 🆚 与 ResNet 对比实验

```bash
# H0-mini
./run_h0mini_experiment.sh visium 0 30

# ResNet50 (修改脚本中的 --model 参数)
# 或者直接运行原始命令
python train_visium.py --model resnet50 --exp_name /data/yujk/GLIP/experiments/resnet50_visium/... --max_epochs 30
```

## 💡 提示

1. **实验命名自动化**：包含日期和时间戳，避免覆盖
2. **日志自动保存**：所有输出都保存到 `training.log`
3. **实验可追溯**：`experiment_info.txt` 记录完整配置
4. **全局日志**：`experiments/experiment_log.txt` 记录所有实验

## 📚 详细文档

- **完整命令**: `H0MINI_STANDARDIZED_COMMANDS.md`
- **参数说明**: `H0MINI_PREDICTION_COMMANDS.md`
- **使用教程**: `H0MINI_USAGE.md`
- **快速参考**: `H0MINI_QUICK_REFERENCE.md`

---

**就这么简单！一行命令，规范化实验，完整记录。**
