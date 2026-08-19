# apply_b1_final_fixed.py
import torch
import os
import sys

print("🚀 最终B1应用到YOLO数据集")
print("=" * 60)

# 添加当前目录到路径
sys.path.append('.')

# 尝试导入
try:
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet

    print("✅ 成功导入扩散模型")
except ImportError:
    print("⚠️ 无法导入扩散模型，尝试其他方式...")

    # 尝试直接导入
    try:
        import denoising_diffusion_pytorch

        print(f"✅ 找到模块: {denoising_diffusion_pytorch.__file__}")

        # 动态获取类
        GaussianDiffusion = denoising_diffusion_pytorch.GaussianDiffusion
        Unet = denoising_diffusion_pytorch.Unet
    except:
        print("❌ 所有导入方式都失败")
        exit(1)

# 继续运行原脚本
exec(open("apply_b1_to_yolo_final.py").read())