import cv2
import numpy as np
import os
import random
from pathlib import Path

# ==================== 配置 ====================
INPUT_DIR = Path(r"D:\project1\inputPatchNLL\inputPatchNLL")
GT_DIR = Path(r"D:\project1\gtPatchNLL\gtPatchNLL")

NUM_SAMPLES = 5000
RANDOM_SEED = 42
SUFFIXES = ['.png', '.jpg', '.jpeg']

# 筛选参数
REMOVE_DARKEST_RATIO = 0.1   # 去掉最暗 10%
R2_THRESHOLD = 0.9           # 线性拟合优度

print("🚀 基于 LSD 数据集估计 α（统计筛选版）")
print("=" * 60)

# ==================== 1. 随机抽样 ====================
all_files = [f for f in os.listdir(INPUT_DIR) if Path(f).suffix.lower() in SUFFIXES]
print(f"总可用对数: {len(all_files)}")

random.seed(RANDOM_SEED)
selected_files = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
print(f"随机抽样: {len(selected_files)} 对")

# ==================== 2. 计算亮度 + α + R² ====================
brightness_list = []
alpha_list = []
r2_list = []

for fname in selected_files:
    low_path = INPUT_DIR / fname
    gt_path = GT_DIR / fname
    if not gt_path.exists():
        continue

    low = cv2.imread(str(low_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if low is None or gt is None:
        continue

    low_f = low.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0

    # 亮度
    bright = low_f.mean()
    brightness_list.append(bright)

    # α（最小二乘 = MLE）
    alpha = np.sum(low_f * gt_f) / (np.sum(gt_f * gt_f) + 1e-8)
    alpha_list.append(alpha)

    # R²
    pred = alpha * gt_f
    ss_res = np.sum((low_f - pred) ** 2)
    ss_tot = np.sum((low_f - low_f.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    r2_list.append(r2)

brightness_arr = np.array(brightness_list)
alpha_arr = np.array(alpha_list)
r2_arr = np.array(r2_list)

# ==================== Step 1：去掉最暗 10% ====================
dark_thresh = np.percentile(brightness_arr, REMOVE_DARKEST_RATIO * 100)
mask_bright = brightness_arr >= dark_thresh

print(f"\n🔹 Step 1: 去掉最暗 {REMOVE_DARKEST_RATIO*100:.0f}%")
print(f"   亮度阈值: {dark_thresh:.4f}")
print(f"   剩余样本: {np.sum(mask_bright)} / {len(brightness_arr)}")

# ==================== Step 2：α 分位数筛选 ====================
alpha_valid = alpha_arr[mask_bright]

alpha_p5 = np.percentile(alpha_valid, 5)
alpha_p95 = np.percentile(alpha_valid, 95)

mask_alpha = (alpha_arr >= alpha_p5) & (alpha_arr <= alpha_p95)

print(f"\n🔹 Step 2: α 分位数筛选")
print(f"   α ∈ [{alpha_p5:.4f}, {alpha_p95:.4f}]")

# ==================== Step 3：R² 筛选 ====================
mask_r2 = r2_arr >= R2_THRESHOLD

print(f"\n🔹 Step 3: R² ≥ {R2_THRESHOLD}")

# ==================== 最终筛选 ====================
mask_final = mask_bright & mask_alpha & mask_r2
filtered_alphas = alpha_arr[mask_final]

print(f"\n最终保留样本: {len(filtered_alphas)} 对")

# ==================== 结果 ====================
if len(filtered_alphas) == 0:
    print("❌ 无有效样本，请放宽条件")
else:
    mean_alpha = np.mean(filtered_alphas)
    std_alpha = np.std(filtered_alphas)

    print("\n" + "=" * 60)
    print("📊 最终估计结果")
    print("=" * 60)
    print(f"α = {mean_alpha:.4f} ± {std_alpha:.4f}")
    print(f"α 范围: [{filtered_alphas.min():.4f}, {filtered_alphas.max():.4f}]")

# ==================== 保存 ====================
with open("alpha_lsd_quantile.txt", "w") as f:
    f.write(f"alpha_mean = {mean_alpha:.6f}\n")
    f.write(f"alpha_std = {std_alpha:.6f}\n")
    f.write(f"num_pairs_used = {len(filtered_alphas)}\n")
    f.write(f"brightness_remove_ratio = {REMOVE_DARKEST_RATIO}\n")
    f.write(f"alpha_p5 = {alpha_p5:.6f}\n")
    f.write(f"alpha_p95 = {alpha_p95:.6f}\n")
    f.write(f"r2_threshold = {R2_THRESHOLD}\n")
    f.write(f"random_seed = {RANDOM_SEED}\n")

print("\n✅ 结果已保存到 alpha_lsd_quantile.txt")