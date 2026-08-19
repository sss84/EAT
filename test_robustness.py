"""
最终B1验证 - 确认0.0020是正确的强度
"""

import torch
import sys
import os

print("🎯 最终B1验证 - 确认最佳强度")
print("=" * 60)

# 设备
device = torch.device("cuda:0")
print(f"使用设备: {device}")

# 目标亮度
TARGET_MEAN = 0.152910

# 导入模型
sys.path.append('.')
from eat.diffusion_model import GaussianDiffusion, Unet

# 创建模型
model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)
diffusion = diffusion.to(device)

# 加载权重
weights_file = "b1_original_method.pth"
state_dict = torch.load(weights_file, map_location=device)
diffusion.offset_net.load_state_dict(state_dict)
diffusion.eval()

print(f"✅ 加载权重: {weights_file}")

# ==================== 最终验证测试 ====================
print(f"\n最终验证测试 (目标亮度: {TARGET_MEAN:.4f})")
print("=" * 60)

# 测试你原来的最佳强度范围
test_strengths = [0.0015, 0.0018, 0.0020, 0.0022, 0.0025]

results = []
diffusion.eval()

with torch.no_grad():
    for strength in test_strengths:
        if hasattr(diffusion, 'offset_strength'):
            diffusion.offset_strength = strength

        # 固定种子，确保可复现
        torch.manual_seed(42)
        images = diffusion.sample(batch_size=2)

        # 计算亮度
        images_01 = (images + 1) * 0.5
        Y = 0.299 * images_01[:, 0] + 0.587 * images_01[:, 1] + 0.114 * images_01[:, 2]
        brightness = Y.mean().item()

        error = abs(brightness - TARGET_MEAN)
        results.append((strength, brightness, error))

        print(f"  强度 {strength:.4f}: 亮度 = {brightness:.4f} (误差: {error:.4f})")

# 找到最佳
best_strength, best_brightness, best_error = min(results, key=lambda x: x[2])

print(f"\n🎯 确认的最佳强度: {best_strength:.4f}")
print(f"   对应亮度: {best_brightness:.4f}")
print(f"   目标亮度: {TARGET_MEAN:.4f}")
print(f"   误差: {best_error:.4f}")

# ==================== 鲁棒性确认 ====================
print(f"\n鲁棒性确认 (强度: {best_strength:.4f})")
print("-" * 60)

seeds = [42, 123, 456, 789, 999]
brightness_values = []

for seed in seeds:
    torch.manual_seed(seed)
    with torch.no_grad():
        if hasattr(diffusion, 'offset_strength'):
            diffusion.offset_strength = best_strength

        images = diffusion.sample(batch_size=1)
        images_01 = (images + 1) * 0.5
        Y = 0.299 * images_01[:, 0] + 0.587 * images_01[:, 1] + 0.114 * images_01[:, 2]
        brightness = Y.mean().item()
        brightness_values.append(brightness)

        print(f"  种子 {seed:4d}: 亮度 = {brightness:.4f}")

avg_brightness = sum(brightness_values) / len(brightness_values)
std_brightness = (sum((b - avg_brightness) ** 2 for b in brightness_values) / len(brightness_values)) ** 0.5

print(f"\n📊 统计:")
print(f"  平均亮度: {avg_brightness:.4f}")
print(f"  标准差: {std_brightness:.4f}")
print(f"  变异系数: {std_brightness / avg_brightness * 100:.2f}%")

print("\n" + "=" * 60)
print("✅ 最终验证完成！")
print("=" * 60)
print(f"🎯 确认的最佳强度: {best_strength:.4f}")
print(f"📊 平均亮度: {avg_brightness:.4f} ± {std_brightness:.4f}")
print(f"🎯 目标亮度: {TARGET_MEAN:.4f}")
print(f"📈 误差: {abs(avg_brightness - TARGET_MEAN):.4f}")
print("=" * 60)

# 生成应用代码
print(f"\n📋 应用到YOLO数据集的代码:")
print(f"OFFSET_STRENGTH = {best_strength:.4f}  # 确认的最佳强度")
print(f"# 对应亮度: {avg_brightness:.4f} ± {std_brightness:.4f}")
print(f"# 目标亮度: {TARGET_MEAN:.4f}")
print(f"# 误差: {abs(avg_brightness - TARGET_MEAN):.4f}")