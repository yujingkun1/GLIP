# H0-mini 实验结果汇总

## 📊 已完成实验的 5-Fold 汇总结果

### 1. Visium (30 epochs)
**实验路径**: `/data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_30ep_20260512_181302`

```
============================================================
5-Fold Cross-Validation Summary
============================================================
Number of folds: 5

Overall Pearson: 0.589270 ± 0.031694
Gene Pearson:    0.150260 ± 0.048993
Spot Pearson:    0.573909 ± 0.026573
============================================================
```

**Per-fold 结果**:
- Fold 0: overall=0.606224, gene=0.166518, spot=0.588802
- Fold 1: overall=0.569681, gene=0.071578, spot=0.580450
- Fold 2: overall=0.628240, gene=0.158709, spot=0.612306
- Fold 3: overall=0.604062, gene=0.222072, spot=0.543077
- Fold 4: overall=0.538144, gene=0.132425, spot=0.544908

---

### 2. Joint (30 epochs)
**实验路径**: `/data/yujk/GLIP/experiments/h0mini_joint/h0mini_joint_5fold_30ep_20260513_004807`

```
============================================================
5-Fold Cross-Validation Summary (Joint)
============================================================

Visium Results:
  Number of folds: 5
  Overall Pearson: 0.584757 ± 0.027313
  Gene Pearson:    0.151099 ± 0.033567
  Spot Pearson:    0.565319 ± 0.032247

Xenium Results:
  Number of folds: 5
  Overall Pearson: 0.742194 ± 0.034160
  Gene Pearson:    0.503800 ± 0.062761
  Spot Pearson:    0.809768 ± 0.008745
============================================================
```

**Visium Per-fold 结果**:
- Fold 0: overall=0.612529, gene=0.167463, spot=0.586654
- Fold 1: overall=0.617231, gene=0.089356, spot=0.610824
- Fold 2: overall=0.546579, gene=0.142549, spot=0.542212
- Fold 3: overall=0.563648, gene=0.181700, spot=0.519129
- Fold 4: overall=0.583798, gene=0.174429, spot=0.567777

**Xenium Per-fold 结果**:
- Fold 0: overall=0.730976, gene=0.459506, spot=0.807014
- Fold 1: overall=0.784258, gene=0.583594, spot=0.822506
- Fold 2: overall=0.720468, gene=0.472672, spot=0.806317
- Fold 3: overall=0.779022, gene=0.573681, spot=0.815969
- Fold 4: overall=0.696246, gene=0.429548, spot=0.797034

---

### 3. Xenium (20 epochs)
**实验路径**: `/data/yujk/GLIP/experiments/h0mini_xenium/h0mini_xenium_ncbi784_20ep_20260513_004707`

**注意**: Xenium 是单次运行（不是 5-fold），使用 test_fold=4/5

```
Final test metrics:
  Overall Pearson: 0.7347
  Gene Pearson:    0.5177
  Spot Pearson:    0.7455
```

---

## 📈 结果对比

### Overall Pearson 对比

| 任务 | Overall Pearson | Gene Pearson | Spot Pearson |
|------|----------------|--------------|--------------|
| **Visium** | 0.589 ± 0.032 | 0.150 ± 0.049 | 0.574 ± 0.027 |
| **Joint (Visium)** | 0.585 ± 0.027 | 0.151 ± 0.034 | 0.565 ± 0.032 |
| **Joint (Xenium)** | 0.742 ± 0.034 | 0.504 ± 0.063 | 0.810 ± 0.009 |
| **Xenium** | 0.735 | 0.518 | 0.746 |

### 关键观察

1. **Visium vs Joint (Visium)**:
   - Overall Pearson 非常接近 (0.589 vs 0.585)
   - Joint 训练没有显著提升 Visium 性能

2. **Xenium 性能**:
   - Joint 训练的 Xenium 结果 (0.742) 与单独训练 (0.735) 相当
   - Xenium 的 Spot Pearson 非常高 (0.810 vs 0.746)

3. **稳定性**:
   - Visium 的标准差较小 (0.027-0.032)
   - Xenium 在 Joint 训练中的 Spot Pearson 非常稳定 (std=0.009)

---

## 🔧 自动汇总功能

### 已添加功能

从现在开始，所有新的 5-fold 实验都会自动计算汇总统计：

```bash
# 运行实验
./run_h0mini_experiment.sh visium 30

# 实验完成后会自动生成：
# - fold_summary.json  (机器可读)
# - fold_summary.txt   (人类可读)
```

### 手动计算汇总

如果需要为已完成的实验计算汇总：

```bash
# Visium
python compute_fold_summary.py /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_30ep_YYYYMMDD_HHMMSS

# Joint
python compute_fold_summary.py /data/yujk/GLIP/experiments/h0mini_joint/h0mini_joint_5fold_30ep_YYYYMMDD_HHMMSS
```

---

## 📁 输出文件

每个实验目录现在包含：

```
h0mini_visium_5fold_30ep_YYYYMMDD_HHMMSS/
├── experiment_summary.txt      # 实验配置
├── fold_summary.json          # 5-fold 汇总 (JSON)
├── fold_summary.txt           # 5-fold 汇总 (文本)
├── fold0/
│   ├── training.log
│   └── fold_info.txt
├── fold1/
├── fold2/
├── fold3/
└── fold4/
```

---

## 🎯 使用建议

### 查看汇总结果

```bash
# 快速查看文本版本
cat /data/yujk/GLIP/experiments/h0mini_visium/*/fold_summary.txt

# 查看 JSON 版本（用于后续分析）
cat /data/yujk/GLIP/experiments/h0mini_visium/*/fold_summary.json
```

### 比较不同实验

```bash
# 查看所有 Visium 实验的汇总
for dir in /data/yujk/GLIP/experiments/h0mini_visium/*/; do
    echo "=== $dir ==="
    cat "$dir/fold_summary.txt" 2>/dev/null || echo "No summary found"
    echo ""
done
```

---

## 📝 总结

✅ **已完成**:
1. 为已完成的 Visium 和 Joint 实验计算了 5-fold 汇总
2. 创建了自动汇总脚本 `compute_fold_summary.py`
3. 更新了 `run_h0mini_experiment.sh`，自动计算汇总
4. 生成了 JSON 和文本两种格式的汇总文件

✅ **以后的实验**:
- 所有新的 5-fold 实验都会自动生成汇总统计
- 汇总文件保存在实验目录中
- 包含均值、标准差和每个 fold 的详细结果
