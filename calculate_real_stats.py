"""
计算ExDark数据集的真实亮度统计量
用于B1重新训练的目标值
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import json

# 配置
TRAIN_DIR = "lowlight_train"  # 你的500张ExDark图像


def calculate_statistics():
    """计算亮度统计量"""
    print("🔢 计算ExDark真实亮度统计...")
    print(f"图像目录: {TRAIN_DIR}")

    # 检查目录
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 目录不存在: {TRAIN_DIR}")
        return None, None

    # 获取图像文件
    image_files = [f for f in os.listdir(TRAIN_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("❌ 没有找到图像文件")
        return None, None

    # 计算统计量
    all_brightness = []
    brightness_details = []

    for img_file in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(TRAIN_DIR, img_file)

        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  无法读取: {img_file}")
                continue

            # 转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化到[0, 1]
            img_float = img_rgb.astype(np.float32) / 255.0

            # 计算亮度（标准灰度公式）
            R = img_float[:, :, 0]
            G = img_float[:, :, 1]
            B = img_float[:, :, 2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B

            # 统计
            mean_val = np.mean(Y)
            std_val = np.std(Y)

            all_brightness.append(mean_val)
            brightness_details.append({
                'file': img_file,
                'mean': float(mean_val),
                'std': float(std_val)
            })

        except Exception as e:
            print(f"❌ 处理 {img_file} 时出错: {e}")
            continue

    # 计算总体统计
    if len(all_brightness) == 0:
        print("❌ 没有成功处理的图像")
        return None, None

    total_mean = np.mean(all_brightness)
    total_std = np.std(all_brightness)

    # 分位数统计
    brightness_array = np.array(all_brightness)
    percentiles = {
        '10%': np.percentile(brightness_array, 10),
        '25%': np.percentile(brightness_array, 25),
        '50%': np.percentile(brightness_array, 50),  # 中位数
        '75%': np.percentile(brightness_array, 75),
        '90%': np.percentile(brightness_array, 90)
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("📊 EXDARK亮度统计结果")
    print("=" * 60)
    print(f"图像数量: {len(all_brightness)}")
    print(f"平均亮度 (TARGET_MEAN): {total_mean:.6f}")
    print(f"亮度标准差 (TARGET_STD): {total_std:.6f}")
    print(f"亮度范围: [{np.min(all_brightness):.6f}, {np.max(all_brightness):.6f}]")
    print(f"中位数: {percentiles['50%']:.6f}")
    print("\n亮度分位数:")
    for key, value in percentiles.items():
        print(f"  {key}: {value:.6f}")

    # 保存详细结果
    results = {
        'statistics': {
            'TARGET_MEAN': float(total_mean),
            'TARGET_STD': float(total_std),
            'min': float(np.min(all_brightness)),
            'max': float(np.max(all_brightness)),
            'median': float(percentiles['50%']),
            'num_images': len(all_brightness)
        },
        'percentiles': {k: float(v) for k, v in percentiles.items()},
        'per_image_stats': brightness_details[:100]  # 只保存前100张的详细数据
    }

    with open('exdark_real_statistics.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细统计保存至: exdark_real_statistics.json")

    # 生成Python代码片段
    print("\n" + "=" * 60)
    print("📝 复制以下代码到B1训练脚本:")
    print("=" * 60)
    print(f"TARGET_MEAN = {total_mean:.6f}  # ExDark平均亮度")
    print(f"TARGET_STD = {total_std:.6f}    # ExDark亮度标准差")

    # 可视化
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 1. 亮度分布直方图
        plt.subplot(1, 3, 1)
        plt.hist(all_brightness, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=total_mean, color='red', linestyle='--', label=f'均值: {total_mean:.4f}')
        plt.axvline(x=percentiles['50%'], color='green', linestyle=':', label=f'中位数: {percentiles["50%"]:.4f}')
        plt.xlabel('亮度')
        plt.ylabel('频数')
        plt.title('ExDark亮度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 箱型图
        plt.subplot(1, 3, 2)
        plt.boxplot(all_brightness, vert=False)
        plt.xlabel('亮度')
        plt.title('ExDark亮度箱型图')
        plt.grid(True, alpha=0.3)

        # 3. 亮度趋势
        plt.subplot(1, 3, 3)
        plt.plot(sorted(all_brightness), 'o', alpha=0.5, markersize=2)
        plt.xlabel('图像序号（按亮度排序）')
        plt.ylabel('亮度')
        plt.title('ExDark亮度排序')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exdark_brightness_analysis.png', dpi=150)
        print(f"📈 可视化图表保存至: exdark_brightness_analysis.png")

    except ImportError:
        print("⚠️  无法生成图表，请安装matplotlib")

    return total_mean, total_std


def main():
    print("🚀 ExDark真实亮度统计计算工具")
    print("=" * 60)

    mean, std = calculate_statistics()

    if mean is not None and std is not None:
        print("\n" + "=" * 60)
        print("🎯 B1训练目标值:")
        print("=" * 60)
        print(f"TARGET_MEAN = {mean:.6f}")
        print(f"TARGET_STD  = {std:.6f}")
        print("=" * 60)

        # 创建配置文件
        config_content = f"""# ExDark亮度统计配置
# 用于B1偏移网络训练
# 基于{len(os.listdir(TRAIN_DIR))}张低光图像计算

TARGET_MEAN = {mean:.6f}  # ExDark平均亮度
TARGET_STD = {std:.6f}    # ExDark亮度标准差

# 训练建议
RECOMMENDED_STRENGTH = 0.015  # 建议偏移强度
NUM_TRAINING_STEPS = 800      # 建议训练步数
"""

        with open('b1_target_config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"✅ 配置文件保存至: b1_target_config.py")

        # 创建可直接导入的模块
        with open('b1_targets.py', 'w', encoding='utf-8') as f:
            f.write(f"""# B1训练目标值
TARGET_MEAN = {mean:.6f}
TARGET_STD = {std:.6f}

def get_targets():
    \"\"\"返回B1训练目标值\"\"\"
    return TARGET_MEAN, TARGET_STD
""")

        print(f"✅ 可直接导入的模块: b1_targets.py")


if __name__ == "__main__":
    main()