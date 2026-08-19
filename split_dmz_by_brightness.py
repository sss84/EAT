import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# ==================== 配置路径 ====================
DMZ_ROOT = r"D:\project1\eat\DMZ"
TRAIN_IMAGES = os.path.join(DMZ_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DMZ_ROOT, "train", "labels")
VAL_IMAGES = os.path.join(DMZ_ROOT, "val", "images")
VAL_LABELS = os.path.join(DMZ_ROOT, "val", "labels")

# 输出划分文件
OUTPUT_DIR = os.path.join(DMZ_ROOT, "splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DMZ数据集亮度划分")
print("=" * 60)
print(f"训练集图片: {TRAIN_IMAGES}")
print(f"验证集图片: {VAL_IMAGES}")
print("=" * 60)


# ==================== 统计亮度 ====================
def compute_image_brightness(image_path):
    """计算单张图像的灰度均值"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean()


def process_split(images_dir, labels_dir, split_name):
    """处理一个数据集划分（train/val）"""
    print(f"\n处理 {split_name} 集...")

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"  ❌ {images_dir} 中没有找到图像")
        return [], []

    print(f"  找到 {len(image_files)} 张图像")

    # 计算每张图像的亮度
    brightness_data = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        brightness = compute_image_brightness(img_path)

        if brightness is not None:
            brightness_data.append({
                'file': img_file,
                'brightness': brightness,
                'path': img_path,
                'label_path': os.path.join(labels_dir,
                                           img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png',
                                                                                                             '.txt'))
            })

        if (i + 1) % 500 == 0:
            print(f"    已处理 {i + 1}/{len(image_files)} 张")

    print(f"  成功计算 {len(brightness_data)} 张图像的亮度")

    # 按亮度排序
    brightness_data.sort(key=lambda x: x['brightness'])

    # 取中位数作为阈值
    threshold = brightness_data[len(brightness_data) // 2]['brightness']
    print(f"  {split_name} 集亮度中位数: {threshold:.2f}")

    # 划分亮/暗子集
    dark_images = [item for item in brightness_data if item['brightness'] < threshold]
    bright_images = [item for item in brightness_data if item['brightness'] >= threshold]

    print(f"  暗光子集: {len(dark_images)} 张 (亮度 < {threshold:.2f})")
    print(f"  亮光子集: {len(bright_images)} 张 (亮度 >= {threshold:.2f})")

    return dark_images, bright_images


# ==================== 处理训练集和验证集 ====================
train_dark, train_bright = process_split(TRAIN_IMAGES, TRAIN_LABELS, "train")
val_dark, val_bright = process_split(VAL_IMAGES, VAL_LABELS, "val")


# ==================== 保存划分结果 ====================
def save_split(images, split_name, subset):
    """保存划分结果到txt文件"""
    filename = os.path.join(OUTPUT_DIR, f"{split_name}_{subset}.txt")
    with open(filename, 'w') as f:
        for item in images:
            f.write(f"{item['file']}\n")
    print(f"  已保存: {filename}")

    # 同时保存带路径的版本（方便直接使用）
    path_filename = os.path.join(OUTPUT_DIR, f"{split_name}_{subset}_paths.txt")
    with open(path_filename, 'w') as f:
        for item in images:
            f.write(f"{item['path']}\n")
    print(f"  已保存: {path_filename}")


print("\n" + "=" * 60)
print("保存划分结果")
print("=" * 60)

# 训练集
save_split(train_dark, "train", "dark")
save_split(train_bright, "train", "bright")

# 验证集
save_split(val_dark, "val", "dark")
save_split(val_bright, "val", "bright")

# ==================== 统计报告 ====================
print("\n" + "=" * 60)
print("划分完成报告")
print("=" * 60)
print(f"训练集:")
print(f"  亮光子集: {len(train_bright)} 张")
print(f"  暗光子集: {len(train_dark)} 张")
print(f"验证集:")
print(f"  亮光子集: {len(val_bright)} 张")
print(f"  暗光子集: {len(val_dark)} 张")
print("=" * 60)
print(f"划分文件保存在: {OUTPUT_DIR}")
print("=" * 60)