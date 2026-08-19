"""
最简单直接的ExDark修复脚本
"""

import os
import zipfile
import shutil
import random

# 1. 找到ZIP文件
zip_files = [f for f in os.listdir() if f.lower().endswith('.zip')]
if not zip_files:
    print("❌ 没有找到ZIP文件")
    exit()

zip_file = zip_files[0]
print(f"找到ZIP文件: {zip_file}")

# 2. 创建输出目录
output_dir = "exdark_fixed_images"
os.makedirs(output_dir, exist_ok=True)

# 3. 简单解压（不处理乱码）
print("解压中...")
count = 0

with zipfile.ZipFile(zip_file, 'r') as zf:
    for file_info in zf.infolist():
        # 只处理图像文件
        if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # 直接提取，用数字重命名
                ext = os.path.splitext(file_info.filename)[1]
                if not ext:
                    ext = '.jpg'

                output_path = os.path.join(output_dir, f"img_{count:05d}{ext}")

                with zf.open(file_info) as source:
                    with open(output_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                count += 1
                if count % 100 == 0:
                    print(f"已提取 {count} 张...")

            except Exception as e:
                continue

print(f"\n✅ 提取完成: {count} 张图像")
print(f"保存在: {output_dir}")

# 4. 创建训练集
if count > 0:
    train_dir = "lowlight_train"
    os.makedirs(train_dir, exist_ok=True)

    # 复制最多500张
    all_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.seed(42)
    selected = random.sample(all_images, min(500, len(all_images)))

    for i, src in enumerate(selected):
        dst = os.path.join(train_dir, f"train_{i:04d}.jpg")
        shutil.copy2(src, dst)

    print(f"✅ 训练集创建完成: {len(selected)} 张图像")
    print(f"位置: {train_dir}")