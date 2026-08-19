"""
lsd_alpha_estimation.py
Estimate alpha from LSD dataset with filtering strategy:
1. Remove darkest 10% (brightness < 10th percentile)
2. Keep alpha within 5%-95% percentile
3. Keep R^2 >= 0.9

Fixed random seed = 42 for reproducibility
"""

import cv2
import numpy as np
from pathlib import Path
import random

# ==================== Configuration ====================
LSD_INPUT_DIR = Path(r"D:\project1\inputPatchNLL\inputPatchNLL")
LSD_GT_DIR = Path(r"D:\project1\gtPatchNLL\gtPatchNLL")

NUM_PAIRS = 5000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("LSD Dataset Alpha Estimation with Filtering")
print("=" * 60)
print(f"Low-light dir: {LSD_INPUT_DIR}")
print(f"Ground truth dir: {LSD_GT_DIR}")
print(f"Random seed: {RANDOM_SEED}")
print("=" * 60)

# ==================== Get all images ====================
all_files = list(LSD_INPUT_DIR.glob("*.*"))
print(f"\nTotal low-light images: {len(all_files)}")

if NUM_PAIRS < len(all_files):
    selected_files = random.sample(all_files, NUM_PAIRS)
    print(f"Randomly selected: {NUM_PAIRS} pairs")
else:
    selected_files = all_files
    print(f"Using all: {len(selected_files)} pairs")


# ==================== Compute brightness ====================
def compute_brightness(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# ==================== Estimate alpha (CORRECTED) ====================
def estimate_alpha(img_low, img_normal):
    """
    Estimate alpha using least squares: alpha = (I_low · I_normal) / (I_normal · I_normal)
    """
    low_float = img_low.astype(np.float32) / 255.0
    normal_float = img_normal.astype(np.float32) / 255.0

    numerator = np.sum(low_float * normal_float)  # I_low · I_normal
    denominator = np.sum(normal_float * normal_float)  # I_normal · I_normal

    if denominator < 1e-8:
        return None, None

    alpha = numerator / denominator  # CORRECT: alpha should be < 1

    # Compute R^2
    predicted = alpha * normal_float
    ss_res = np.sum((low_float - predicted) ** 2)
    ss_tot = np.sum((low_float - np.mean(low_float)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return alpha, r2


# ==================== Step 1: Compute brightness ====================
print("\n[Step 1] Computing brightness for all low-light images...")

brightness_values = []
pair_info = []

for img_path in selected_files:
    gt_path = LSD_GT_DIR / img_path.name
    if not gt_path.exists():
        continue

    b = compute_brightness(img_path)
    if b is not None:
        brightness_values.append(b)
        pair_info.append((img_path, gt_path, b))

brightness_array = np.array(brightness_values)
print(f"Valid pairs: {len(brightness_array)}")

# ==================== Step 2: Remove darkest 10% ====================
bottom_10_threshold = np.percentile(brightness_array, 10)
print(f"\n[Step 2] Bottom 10% brightness threshold: {bottom_10_threshold:.2f}")

filtered_pairs = [(img, gt, b) for img, gt, b in pair_info if b >= bottom_10_threshold]
print(f"After removing darkest 10%: {len(filtered_pairs)} pairs")

# ==================== Step 3: Estimate alpha ====================
print("\n[Step 3] Estimating alpha for remaining pairs...")

alphas = []
r2_values = []

for img_path, gt_path, brightness in filtered_pairs:
    low_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    normal_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    if low_img is None or normal_img is None:
        continue

    alpha, r2 = estimate_alpha(low_img, normal_img)
    if alpha is not None and r2 is not None:
        alphas.append(alpha)
        r2_values.append(r2)

print(f"Estimated alpha for {len(alphas)} pairs")
print(f"Alpha range before filtering: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
print(f"Alpha mean before filtering: {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")

# ==================== Step 4: Filter by R^2 >= 0.9 ====================
print("\n[Step 4] Filtering by R^2 >= 0.9...")

r2_array = np.array(r2_values)
r2_filter = r2_array >= 0.9

alphas_filtered_r2 = [a for i, a in enumerate(alphas) if r2_filter[i]]
r2_filtered = [r for i, r in enumerate(r2_values) if r2_filter[i]]

print(f"After R^2 >= 0.9: {len(alphas_filtered_r2)} pairs")
if len(r2_filtered) > 0:
    print(f"  R^2 range: [{np.min(r2_filtered):.4f}, {np.max(r2_filtered):.4f}]")

if len(alphas_filtered_r2) == 0:
    print("No pairs passed R^2 filter!")
    exit(1)

# ==================== Step 5: Filter by 5%-95% percentile ====================
print("\n[Step 5] Filtering by alpha 5%-95% percentile...")

alpha_array = np.array(alphas_filtered_r2)
alpha_5th = np.percentile(alpha_array, 5)
alpha_95th = np.percentile(alpha_array, 95)

percentile_filter = (alpha_array >= alpha_5th) & (alpha_array <= alpha_95th)
alphas_final = alpha_array[percentile_filter]

print(f"Alpha 5th percentile: {alpha_5th:.4f}")
print(f"Alpha 95th percentile: {alpha_95th:.4f}")
print(f"After 5%-95% percentile filter: {len(alphas_final)} pairs")

# ==================== Final Results ====================
print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)

if len(alphas_final) > 0:
    mean_alpha = np.mean(alphas_final)
    std_alpha = np.std(alphas_final)

    print(f"Number of valid pairs: {len(alphas_final)}")
    print(f"Alpha = {mean_alpha:.4f} ± {std_alpha:.4f}")
    print(f"Alpha range: [{np.min(alphas_final):.4f}, {np.max(alphas_final):.4f}]")
else:
    print("No valid pairs after all filters!")
    mean_alpha, std_alpha = 0, 0

# ==================== Save Results ====================
output_file = Path("alpha_estimation_lsd_filtered.txt")
with open(output_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("LSD Dataset Alpha Estimation Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
    f.write(f"Total raw pairs: {len(selected_files)}\n")
    f.write(f"After removing darkest 10%: {len(filtered_pairs)}\n")
    f.write(f"After R^2 >= 0.9: {len(alphas_filtered_r2)}\n")
    f.write(f"After 5%-95% percentile: {len(alphas_final)}\n\n")
    f.write(f"Final alpha = {mean_alpha:.6f} ± {std_alpha:.6f}\n")
    if len(alphas_final) > 0:
        f.write(f"Alpha range: [{np.min(alphas_final):.6f}, {np.max(alphas_final):.6f}]\n")

print(f"\n✅ Results saved to: {output_file}")