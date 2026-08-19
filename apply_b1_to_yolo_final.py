# apply_b1_to_yolo_final.py
import torch
import os
import cv2
import numpy as np
from PIL import Image
import shutil
import sys
from pathlib import Path

print("🚀 最终：应用B1到你的YOLO数据集")
print("=" * 60)

# ==================== 配置 ====================
YOLO_DATASET = r"D:\苏雅文\Desktop\ultralytics-main\DMZ"  # 你的4900张YOLO数据集
OUTPUT_DATASET = r"D:\苏雅文\Desktop\ultralytics-main\DMZ_B1_FINAL"
OFFSET_STRENGTH = 0.0020  # 测试得到的最佳强度！

print(f"源数据集: {YOLO_DATASET}")
print(f"目标数据集: {OUTPUT_DATASET}")
print(f"偏移强度: {OFFSET_STRENGTH}")
print(f"目标亮度: 0.1529")

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 加载B1模型 ====================
print("\n1. 加载训练好的B1模型...")
sys.path.append('.')

try:
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet

    # 创建模型
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
    diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)
    diffusion = diffusion.to(device)

    # 加载B1权重
    weights_file = "b1_original_method.pth"
    state_dict = torch.load(weights_file, map_location=device)
    diffusion.offset_net.load_state_dict(state_dict)
    diffusion.eval()

    print(f"✅ 加载B1权重: {weights_file}")

except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit(1)


# ==================== 处理函数 ====================
def apply_b1_degradation(image_path):
    """对单张图像应用B1退化"""
    try:
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (width, height)

        # 调整大小到128x128（B1模型输入尺寸）
        img_resized = img.resize((128, 128))

        # 转换为tensor [-1, 1]
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1,3,128,128]
        img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
        img_tensor = img_tensor.to(device)

        # 设置偏移强度
        if hasattr(diffusion, 'offset_strength'):
            diffusion.offset_strength = OFFSET_STRENGTH

        # 应用B1退化
        with torch.no_grad():
            # 添加噪声并退化
            t = torch.randint(0, 100, (1,), device=device)
            degraded = diffusion.q_sample(img_tensor, t)

            # 确保值范围
            degraded = torch.clamp(degraded, -1, 1)

            # 转换回PIL图像
            degraded = degraded.squeeze(0).cpu()
            degraded = (degraded + 1) * 0.5  # [-1,1] -> [0,1]
            degraded = degraded.permute(1, 2, 0).numpy() * 255
            degraded = np.clip(degraded, 0, 255).astype(np.uint8)

        # 恢复原始尺寸
        degraded_img = Image.fromarray(degraded)
        degraded_img = degraded_img.resize(original_size)

        return degraded_img

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None


# ==================== 处理所有图像 ====================
print("\n2. 处理你的4900张YOLO图像...")

splits = ['train', 'val']
total_processed = 0
total_success = 0

for split in splits:
    print(f"\n处理 {split} 集...")

    # 源路径
    src_img_dir = os.path.join(YOLO_DATASET, split, "images")
    src_lbl_dir = os.path.join(YOLO_DATASET, split, "labels")

    # 目标路径
    dst_img_dir = os.path.join(OUTPUT_DATASET, split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_DATASET, split, "labels")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # 检查源目录
    if not os.path.exists(src_img_dir):
        print(f"❌ 找不到源目录: {src_img_dir}")
        continue

    # 获取图像文件
    img_files = [f for f in os.listdir(src_img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    total_processed += len(img_files)
    print(f"找到 {len(img_files)} 张图像")

    success_count = 0

    for i, img_file in enumerate(img_files):
        # 源路径
        src_img_path = os.path.join(src_img_dir, img_file)

        # 目标路径（保持相同文件名）
        dst_img_path = os.path.join(dst_img_dir, img_file)

        # 处理图像
        processed_img = apply_b1_degradation(src_img_path)

        if processed_img is not None:
            processed_img.save(dst_img_path, 'JPEG', quality=95)
            success_count += 1
            total_success += 1
        else:
            # 如果失败，复制原始图像
            shutil.copy2(src_img_path, dst_img_path)

        # 复制标签文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(src_lbl_dir, label_file)
        dst_label_path = os.path.join(dst_lbl_dir, label_file)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(img_files)} 张")

    print(f"✅ {split}完成: {success_count}/{len(img_files)} 张成功")

# ==================== 复制配置文件 ====================
print("\n3. 复制配置文件...")

yaml_src = os.path.join(YOLO_DATASET, "data.yaml")
yaml_dst = os.path.join(OUTPUT_DATASET, "data.yaml")

if os.path.exists(yaml_src):
    shutil.copy2(yaml_src, yaml_dst)
    print(f"✅ 配置文件复制: {yaml_dst}")
else:
    print(f"⚠️  找不到data.yaml，需要手动创建")

# ==================== 统计报告 ====================
print("\n" + "=" * 60)
print("📊 处理完成报告")
print("=" * 60)
print(f"总处理图像: {total_processed} 张")
print(f"成功处理: {total_success} 张")
print(f"成功率: {total_success / total_processed * 100:.1f}%")
print(f"输出位置: {OUTPUT_DATASET}")
print(f"使用强度: {OFFSET_STRENGTH}")
print("=" * 60)

# ==================== YOLO训练命令 ====================
print(f"\n📋 YOLO训练命令:")
print(f"cd D:\\苏雅文\\Desktop\\ultralytics-main")
print(f"python train.py \\")
print(f"  --data {OUTPUT_DATASET}/data.yaml \\")
print(f"  --weights yolov8n.pt \\")
print(f"  --epochs 50 \\")
print(f"  --imgsz 640 \\")
print(f"  --batch 16 \\")
print(f"  --name yolov8_b1_lowlight")

print("\n" + "=" * 60)
print("🎉 全部准备完成！现在可以训练YOLO了！")
print("=" * 60)