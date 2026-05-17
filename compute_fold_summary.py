#!/usr/bin/env python3
"""
计算 5-fold 交叉验证的汇总统计（支持 Visium 和 Joint）
"""

import re
import sys
import json
from pathlib import Path
import numpy as np


def extract_pearson_from_visium_log(log_file):
    """从 Visium 训练日志中提取最终的 Pearson 相关系数"""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # 格式: "Fold 1 Pearson: overall=0.606224, gene=0.166518, spot=0.588802"
    pattern = r'Fold \d+ Pearson: overall=([\d.]+), gene=([-\d.]+), spot=([\d.]+)'
    matches = re.findall(pattern, content)

    if not matches:
        return None

    # 取最后一个匹配（最终结果）
    overall, gene, spot = matches[-1]
    return {
        'overall': float(overall),
        'gene': float(gene),
        'spot': float(spot)
    }


def extract_pearson_from_joint_log(log_file):
    """从 Joint 训练日志中提取最终的 Pearson 相关系数"""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # 查找 Visium 和 Xenium 的测试指标
    visium_pattern = r"Visium test metrics: \{[^}]*'overall_pearson': ([\d.]+)[^}]*'mean_gene_pearson': ([-\d.]+)[^}]*'mean_spot_pearson': ([\d.]+)"
    xenium_pattern = r"Xenium test metrics: \{[^}]*'overall_pearson': ([\d.]+)[^}]*'mean_gene_pearson': ([-\d.]+)[^}]*'mean_spot_pearson': ([\d.]+)"

    visium_matches = re.findall(visium_pattern, content)
    xenium_matches = re.findall(xenium_pattern, content)

    result = {}

    if visium_matches:
        overall, gene, spot = visium_matches[-1]
        result['visium'] = {
            'overall': float(overall),
            'gene': float(gene),
            'spot': float(spot)
        }

    if xenium_matches:
        overall, gene, spot = xenium_matches[-1]
        result['xenium'] = {
            'overall': float(overall),
            'gene': float(gene),
            'spot': float(spot)
        }

    return result if result else None


def compute_visium_summary(exp_dir):
    """计算 Visium 实验的 5-fold 汇总统计"""
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None

    # 收集所有 fold 的结果
    results = []
    for i in range(5):
        fold_dir = exp_path / f"fold{i}"
        log_file = fold_dir / "training.log"

        if not log_file.exists():
            print(f"Warning: Log file not found for fold{i}")
            continue

        pearson = extract_pearson_from_visium_log(log_file)
        if pearson:
            results.append(pearson)
            print(f"Fold {i}: overall={pearson['overall']:.6f}, gene={pearson['gene']:.6f}, spot={pearson['spot']:.6f}")
        else:
            print(f"Warning: Could not extract Pearson from fold{i}")

    if not results:
        print("Error: No valid results found")
        return None

    # 计算均值和标准差
    overall_values = [r['overall'] for r in results]
    gene_values = [r['gene'] for r in results]
    spot_values = [r['spot'] for r in results]

    summary = {
        'num_folds': len(results),
        'overall': {
            'mean': float(np.mean(overall_values)),
            'std': float(np.std(overall_values)),
            'values': overall_values
        },
        'gene': {
            'mean': float(np.mean(gene_values)),
            'std': float(np.std(gene_values)),
            'values': gene_values
        },
        'spot': {
            'mean': float(np.mean(spot_values)),
            'std': float(np.std(spot_values)),
            'values': spot_values
        }
    }

    return summary


def compute_joint_summary(exp_dir):
    """计算 Joint 实验的 5-fold 汇总统计"""
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None

    # 收集所有 fold 的结果
    visium_results = []
    xenium_results = []

    for i in range(5):
        fold_dir = exp_path / f"fold{i}"
        log_file = fold_dir / "training.log"

        if not log_file.exists():
            print(f"Warning: Log file not found for fold{i}")
            continue

        pearson = extract_pearson_from_joint_log(log_file)
        if pearson:
            if 'visium' in pearson:
                visium_results.append(pearson['visium'])
                print(f"Fold {i} Visium: overall={pearson['visium']['overall']:.6f}, gene={pearson['visium']['gene']:.6f}, spot={pearson['visium']['spot']:.6f}")
            if 'xenium' in pearson:
                xenium_results.append(pearson['xenium'])
                print(f"Fold {i} Xenium: overall={pearson['xenium']['overall']:.6f}, gene={pearson['xenium']['gene']:.6f}, spot={pearson['xenium']['spot']:.6f}")
        else:
            print(f"Warning: Could not extract Pearson from fold{i}")

    if not visium_results and not xenium_results:
        print("Error: No valid results found")
        return None

    summary = {}

    # Visium 汇总
    if visium_results:
        overall_values = [r['overall'] for r in visium_results]
        gene_values = [r['gene'] for r in visium_results]
        spot_values = [r['spot'] for r in visium_results]

        summary['visium'] = {
            'num_folds': len(visium_results),
            'overall': {
                'mean': float(np.mean(overall_values)),
                'std': float(np.std(overall_values)),
                'values': overall_values
            },
            'gene': {
                'mean': float(np.mean(gene_values)),
                'std': float(np.std(gene_values)),
                'values': gene_values
            },
            'spot': {
                'mean': float(np.mean(spot_values)),
                'std': float(np.std(spot_values)),
                'values': spot_values
            }
        }

    # Xenium 汇总
    if xenium_results:
        overall_values = [r['overall'] for r in xenium_results]
        gene_values = [r['gene'] for r in xenium_results]
        spot_values = [r['spot'] for r in xenium_results]

        summary['xenium'] = {
            'num_folds': len(xenium_results),
            'overall': {
                'mean': float(np.mean(overall_values)),
                'std': float(np.std(overall_values)),
                'values': overall_values
            },
            'gene': {
                'mean': float(np.mean(gene_values)),
                'std': float(np.std(gene_values)),
                'values': gene_values
            },
            'spot': {
                'mean': float(np.mean(spot_values)),
                'std': float(np.std(spot_values)),
                'values': spot_values
            }
        }

    return summary


def print_visium_summary(summary):
    """打印 Visium 汇总结果"""
    if not summary:
        return

    print("\n" + "="*60)
    print("5-Fold Cross-Validation Summary")
    print("="*60)
    print(f"Number of folds: {summary['num_folds']}")
    print()
    print(f"Overall Pearson: {summary['overall']['mean']:.6f} ± {summary['overall']['std']:.6f}")
    print(f"Gene Pearson:    {summary['gene']['mean']:.6f} ± {summary['gene']['std']:.6f}")
    print(f"Spot Pearson:    {summary['spot']['mean']:.6f} ± {summary['spot']['std']:.6f}")
    print("="*60)


def print_joint_summary(summary):
    """打印 Joint 汇总结果"""
    if not summary:
        return

    print("\n" + "="*60)
    print("5-Fold Cross-Validation Summary (Joint)")
    print("="*60)

    if 'visium' in summary:
        print("\nVisium Results:")
        print(f"  Number of folds: {summary['visium']['num_folds']}")
        print(f"  Overall Pearson: {summary['visium']['overall']['mean']:.6f} ± {summary['visium']['overall']['std']:.6f}")
        print(f"  Gene Pearson:    {summary['visium']['gene']['mean']:.6f} ± {summary['visium']['gene']['std']:.6f}")
        print(f"  Spot Pearson:    {summary['visium']['spot']['mean']:.6f} ± {summary['visium']['spot']['std']:.6f}")

    if 'xenium' in summary:
        print("\nXenium Results:")
        print(f"  Number of folds: {summary['xenium']['num_folds']}")
        print(f"  Overall Pearson: {summary['xenium']['overall']['mean']:.6f} ± {summary['xenium']['overall']['std']:.6f}")
        print(f"  Gene Pearson:    {summary['xenium']['gene']['mean']:.6f} ± {summary['xenium']['gene']['std']:.6f}")
        print(f"  Spot Pearson:    {summary['xenium']['spot']['mean']:.6f} ± {summary['xenium']['spot']['std']:.6f}")

    print("="*60)


def save_summary(summary, output_file, is_joint=False):
    """保存汇总结果到 JSON 文件"""
    if not summary:
        return

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")

    # 保存可读的文本版本
    txt_file = output_file.with_suffix('.txt')
    with open(txt_file, 'w') as f:
        if is_joint:
            f.write("="*60 + "\n")
            f.write("5-Fold Cross-Validation Summary (Joint)\n")
            f.write("="*60 + "\n")

            if 'visium' in summary:
                f.write("\nVisium Results:\n")
                f.write(f"  Number of folds: {summary['visium']['num_folds']}\n")
                f.write(f"  Overall Pearson: {summary['visium']['overall']['mean']:.6f} ± {summary['visium']['overall']['std']:.6f}\n")
                f.write(f"  Gene Pearson:    {summary['visium']['gene']['mean']:.6f} ± {summary['visium']['gene']['std']:.6f}\n")
                f.write(f"  Spot Pearson:    {summary['visium']['spot']['mean']:.6f} ± {summary['visium']['spot']['std']:.6f}\n")
                f.write("\n  Per-fold results:\n")
                for i in range(summary['visium']['num_folds']):
                    f.write(f"    Fold {i}: overall={summary['visium']['overall']['values'][i]:.6f}, "
                           f"gene={summary['visium']['gene']['values'][i]:.6f}, "
                           f"spot={summary['visium']['spot']['values'][i]:.6f}\n")

            if 'xenium' in summary:
                f.write("\nXenium Results:\n")
                f.write(f"  Number of folds: {summary['xenium']['num_folds']}\n")
                f.write(f"  Overall Pearson: {summary['xenium']['overall']['mean']:.6f} ± {summary['xenium']['overall']['std']:.6f}\n")
                f.write(f"  Gene Pearson:    {summary['xenium']['gene']['mean']:.6f} ± {summary['xenium']['gene']['std']:.6f}\n")
                f.write(f"  Spot Pearson:    {summary['xenium']['spot']['mean']:.6f} ± {summary['xenium']['spot']['std']:.6f}\n")
                f.write("\n  Per-fold results:\n")
                for i in range(summary['xenium']['num_folds']):
                    f.write(f"    Fold {i}: overall={summary['xenium']['overall']['values'][i]:.6f}, "
                           f"gene={summary['xenium']['gene']['values'][i]:.6f}, "
                           f"spot={summary['xenium']['spot']['values'][i]:.6f}\n")
        else:
            f.write("="*60 + "\n")
            f.write("5-Fold Cross-Validation Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Number of folds: {summary['num_folds']}\n")
            f.write("\n")
            f.write(f"Overall Pearson: {summary['overall']['mean']:.6f} ± {summary['overall']['std']:.6f}\n")
            f.write(f"Gene Pearson:    {summary['gene']['mean']:.6f} ± {summary['gene']['std']:.6f}\n")
            f.write(f"Spot Pearson:    {summary['spot']['mean']:.6f} ± {summary['spot']['std']:.6f}\n")
            f.write("="*60 + "\n")
            f.write("\nPer-fold results:\n")
            for i in range(summary['num_folds']):
                f.write(f"  Fold {i}: overall={summary['overall']['values'][i]:.6f}, "
                       f"gene={summary['gene']['values'][i]:.6f}, "
                       f"spot={summary['spot']['values'][i]:.6f}\n")

    print(f"Summary also saved to: {txt_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_fold_summary.py <experiment_directory>")
        print("\nExample:")
        print("  python compute_fold_summary.py /data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_30ep_20260512_181302")
        print("  python compute_fold_summary.py /data/yujk/GLIP/experiments/h0mini_joint/h0mini_joint_5fold_30ep_20260513_004807")
        sys.exit(1)

    exp_dir = sys.argv[1]

    print(f"Computing summary for: {exp_dir}")
    print()

    # 检测实验类型
    exp_path = Path(exp_dir)
    is_joint = 'joint' in exp_path.name

    # 计算汇总
    if is_joint:
        summary = compute_joint_summary(exp_dir)
        if summary:
            print_joint_summary(summary)
            output_file = exp_path / "fold_summary.json"
            save_summary(summary, output_file, is_joint=True)
    else:
        summary = compute_visium_summary(exp_dir)
        if summary:
            print_visium_summary(summary)
            output_file = exp_path / "fold_summary.json"
            save_summary(summary, output_file, is_joint=False)


if __name__ == "__main__":
    main()
