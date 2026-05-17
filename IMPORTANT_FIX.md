# ⚠️ 重要提示

## 你刚才运行的命令有问题

### ❌ 错误的命令
```bash
./run_h0mini_experiment.sh visium 0   # 0 epochs - 不会训练！
```

从日志可以看到文件名：`h0mini_visium_5fold_0ep_20260512_180820`
- `0ep` 表示 0 epochs
- 这就是为什么训练立即完成的原因

### ✅ 正确的命令

```bash
# Visium (30 epochs, 推荐)
./run_h0mini_experiment.sh visium 30

# Xenium (20 epochs, 推荐) - 已修复参数错误
./run_h0mini_experiment.sh xenium 20

# Joint (30 epochs, 推荐)
./run_h0mini_experiment.sh joint 30
```

## 🔧 已修复的问题

### 1. Xenium 参数错误
**问题**：使用了错误的参数名称
- ❌ `--model_name` → ✅ `--model`
- ❌ `--exp_name` → ✅ `--run-dir`
- ❌ `--xenium_sample_id` → ✅ `--sample-id`
- ❌ `--device_id` → ✅ `--device`

**状态**：✅ 已修复

### 2. Visium 训练太快
**问题**：使用了 0 epochs
**原因**：命令参数错误

**解决**：使用正确的 epochs 数量

## 📝 正确的使用方法

### 快速测试（1 epoch）
```bash
./run_h0mini_experiment.sh visium 1
./run_h0mini_experiment.sh xenium 1
./run_h0mini_experiment.sh joint 1
```

### 完整训练（推荐）
```bash
./run_h0mini_experiment.sh visium 30
./run_h0mini_experiment.sh xenium 20
./run_h0mini_experiment.sh joint 30
```

## 🎯 现在可以重新运行

```bash
cd /data/yujk/GLIP

# 删除之前的错误实验（可选）
rm -rf /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_0ep_*

# 运行正确的命令
./run_h0mini_experiment.sh visium 30
```

## ⏱️ 预计时间

| 命令 | Epochs | 预计时间 |
|------|--------|---------|
| `visium 1` | 1 | ~2.5 小时 (测试) |
| `visium 30` | 30 | ~75 小时/fold (完整) |
| `xenium 1` | 1 | ~2.5 小时 (测试) |
| `xenium 20` | 20 | ~50 小时 (完整) |
| `joint 1` | 1 | ~2.5 小时 (测试) |
| `joint 30` | 30 | ~75 小时/fold (完整) |

## 💡 建议

1. **先测试**：用 1 epoch 验证代码
   ```bash
   ./run_h0mini_experiment.sh visium 1
   ```

2. **确认无误后**：运行完整训练
   ```bash
   ./run_h0mini_experiment.sh visium 30
   ```

3. **使用 tmux**：避免断开连接
   ```bash
   tmux new -s h0mini
   ./run_h0mini_experiment.sh visium 30
   # Ctrl+B, D 分离会话
   ```
