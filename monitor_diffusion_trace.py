import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eat import Unet, GaussianDiffusion


def analyze_image_brightness(img_tensor):
    """分析图像亮度 - 修复维度问题"""
    # img_tensor 形状: [1, 3, H, W]
    # 我们需要先压缩批次维度
    if len(img_tensor.shape) == 4:
        img = img_tensor.squeeze(0)  # 变成 [3, H, W]
    else:
        img = img_tensor

    if img.shape[0] == 3:  # RGB
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        gray = img[0]
    return gray.mean().item()


def analyze_image_contrast(img_tensor):
    """分析图像对比度 - 修复维度问题"""
    if len(img_tensor.shape) == 4:
        img = img_tensor.squeeze(0)
    else:
        img = img_tensor

    if img.shape[0] == 3:
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        gray = img[0]
    return gray.std().item()


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = Unet(dim=64, channels=3)
diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=100  # 进一步减少步数以快速测试
).to(device)

if not hasattr(diffusion, 'offset_net'):
    print("❌ 请先完成 offset_net 的集成！")
    exit()

print("=" * 70)
print("扩散轨迹监控实验 - 修复版")
print("=" * 70)
print(f"模型时间步总数: {diffusion.num_timesteps}")
print(f"偏移强度默认值: {diffusion.offset_strength}")

# 存储不同强度下的轨迹数据
trajectories = {}

test_strengths = [0.0, 0.05, 0.1]

for strength in test_strengths:
    print(f"\n{'=' * 40}")
    print(f"测试偏移强度: {strength}")
    print(f"{'=' * 40}")

    diffusion.offset_strength = strength

    brightness_history = []
    contrast_history = []
    time_steps = []

    # 手动实现采样循环以便记录
    batch_size = 1
    channels = 3
    height = width = 64
    shape = (batch_size, channels, height, width)

    # 初始噪声
    img = torch.randn(shape, device=device)

    # 只监控关键时间步（更稀疏，加快速度）
    monitor_steps = [95, 80, 60, 40, 20, 10, 0]  # 从大到小

    print(f"监控时间步: {monitor_steps}")
    print(f"{'时间步':<6} {'亮度':<10} {'对比度':<10}")
    print("-" * 30)

    for t in reversed(range(0, diffusion.num_timesteps)):
        # 执行单步采样
        img, x_start = diffusion.p_sample(img, t, x_self_cond=None)

        # 只在关键步记录
        if t in monitor_steps:
            # 移动到CPU分析
            x_start_cpu = x_start.detach().cpu()

            try:
                brightness = analyze_image_brightness(x_start_cpu)
                contrast = analyze_image_contrast(x_start_cpu)

                brightness_history.append(brightness)
                contrast_history.append(contrast)
                time_steps.append(t)

                print(f"t={t:03d}  {brightness:.6f}  {contrast:.6f}")
            except Exception as e:
                print(f"t={t:03d}  分析失败: {e}")
                print(f"x_start形状: {x_start_cpu.shape}")

    trajectories[strength] = {
        'brightness': brightness_history,
        'contrast': contrast_history,
        'timesteps': time_steps
    }

print("\n" + "=" * 70)
print("数据分析报告")
print("=" * 70)

# 检查数据完整性
print("\n数据完整性检查:")
for strength in test_strengths:
    data = trajectories[strength]
    print(f"强度={strength}: {len(data['timesteps'])}个时间点")
    if len(data['timesteps']) == 0:
        print(f"  ❌ 无数据！请检查p_sample返回值")
        continue

    print(f"  时间步: {data['timesteps']}")
    print(f"  亮度值: {[f'{v:.4f}' for v in data['brightness']]}")
    print(f"  对比度: {[f'{v:.4f}' for v in data['contrast']]}")

# 如果数据完整，进行分析
if all(len(trajectories[s]['timesteps']) > 0 for s in test_strengths):
    print("\n" + "=" * 70)
    print("亮度对比表")
    print("=" * 70)
    print("时间步 | " + " | ".join([f"强度={s}" for s in test_strengths]) + " | 趋势")
    print("-" * 80)

    for i, t in enumerate(trajectories[test_strengths[0]]['timesteps']):
        values = []
        for strength in test_strengths:
            val = trajectories[strength]['brightness'][i]
            values.append(f"{val:.4f}")

        # 判断趋势
        val0 = trajectories[test_strengths[0]]['brightness'][i]
        val1 = trajectories[test_strengths[1]]['brightness'][i] if len(test_strengths) > 1 else val0
        val2 = trajectories[test_strengths[2]]['brightness'][i] if len(test_strengths) > 2 else val0

        trend = ""
        if val2 < val1 < val0:
            trend = "✅ 强度↑→亮度↓"
        elif val2 > val1 > val0:
            trend = "⚠️  强度↑→亮度↑"
        else:
            trend = "❓ 不规则"

        print(f"t={t:03d} | " + " | ".join(values) + f" | {trend}")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)

    for strength in test_strengths:
        if strength == 0.0:
            print(f"【基线对照】强度={strength}:")
        else:
            print(f"【偏移网络】强度={strength}:")

        data = trajectories[strength]

        # 计算变化
        if len(data['brightness']) >= 2:
            start_bright = data['brightness'][0]  # 最早（t最大）
            end_bright = data['brightness'][-1]  # 最晚（t最小）
            bright_change = end_bright - start_bright

            start_contrast = data['contrast'][0]
            end_contrast = data['contrast'][-1]
            contrast_change = end_contrast - start_contrast

            print(f"  亮度变化: {start_bright:.4f} → {end_bright:.4f} (Δ={bright_change:+.4f})")
            print(f"  对比度变化: {start_contrast:.4f} → {end_contrast:.4f} (Δ={contrast_change:+.4f})")

            # 评估
            if strength > 0:
                if bright_change < 0:
                    print(f"  {'✅' if bright_change < -0.01 else '⚠️ '} 产生亮度降低")
                if contrast_change < 0:
                    print(f"  {'✅' if contrast_change < -0.01 else '⚠️ '} 产生对比度降低")
        print()
else:
    print("\n❌ 数据不完整，无法进行分析")
    print("可能原因:")
    print("1. p_sample 函数返回值与预期不同")
    print("2. 时间步监控点设置不当")
    print("3. 分析函数仍有维度问题")

print("=" * 70)
print("📋 注意:")
print("  1. 亮度值范围: 通常为[-1,1]经过clamp后")
print("  2. 理想趋势: 强度越大，亮度/对比度越低")
print("  3. 变化量Δ: 应为负值表示退化")
print("=" * 70)