import torch
from torchvision.utils import save_image
from eat import Unet, GaussianDiffusion

print("=" * 60)
print("Phase 1 验证：偏移网络效果对比测试")
print("=" * 60)

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 2. 初始化模型
print("\n[1/4] 初始化模型...")
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=3
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=100,  # 减少步数加快测试
    objective='pred_v'
).to(device)

print(f"✓ 模型初始化完成")

# 3. 检查偏移网络
print("\n[2/4] 检查偏移网络集成...")
print(f"  是否有 offset_net: {hasattr(diffusion, 'offset_net')}")
print(f"  当前偏移强度: {getattr(diffusion, 'offset_strength', '未定义')}")

if not hasattr(diffusion, 'offset_net'):
    print("❌ 错误：offset_net 未定义！")
    print("请确保在 eat.py 文件中：")
    print("  1. GaussianDiffusion.__init__() 中添加了 self.offset_net 定义")
    print("  2. GaussianDiffusion.__init__() 中添加了 self.offset_strength = 0.1")
    exit()

# 测试偏移网络
test_input = torch.randn(1, 3, 64, 64).to(device)
with torch.no_grad():
    offset = diffusion.offset_net(test_input)
print(f"  ✓ 偏移网络测试通过")
print(f"    输入: {test_input.shape} -> 输出: {offset.shape}")

# 4. 对比测试不同偏移强度
print("\n[3/4] 对比测试不同偏移强度...")
diffusion.eval()

results = {}
strengths = [0.0, 0.05, 0.1]

with torch.no_grad():
    for strength in strengths:
        print(f"  测试强度 {strength}...")
        diffusion.offset_strength = strength

        # 采样
        samples = diffusion.sample(batch_size=1)

        # 保存结果
        img_normalized = (samples + 1) * 0.5  # [-1,1] -> [0,1]
        filename = f"result_strength_{strength}.png"
        save_image(img_normalized, filename)

        results[strength] = {
            'shape': samples.shape,
            'range': (samples.min().item(), samples.max().item()),
            'filename': filename
        }

        print(f"    ✓ 已保存: {filename}")

# 5. 结果报告
print("\n[4/4] 测试结果汇总")
print("-" * 40)

for strength in strengths:
    info = results[strength]
    print(f"强度 {strength}:")
    print(f"  形状: {info['shape']}")
    print(f"  像素范围: [{info['range'][0]:.3f}, {info['range'][1]:.3f}]")
    print(f"  文件: {info['filename']}")

print("\n" + "=" * 60)
print("🎉 Phase 1 验证完成！")
print("=" * 60)
print("✅ 请打开生成的图片，对比观察：")
print("   1. result_strength_0.0.png (无偏移，基线)")
print("   2. result_strength_0.05.png (弱偏移)")
print("   3. result_strength_0.1.png (强偏移)")
print("\n🔍 观察要点：")
print("   • 整体亮度是否变化？")
print("   • 对比度是否变化？")
print("   • 细节清晰度是否变化？")
print("   • 看起来更像低光照片吗？")
print("=" * 60)