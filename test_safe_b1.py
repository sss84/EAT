
# test_safe_b1.py
import torch
import os
import sys

print("🚀 安全B1测试（跳过编码问题）")
print("=" * 60)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 检查B1权重
weights_file = "b1_original_method.pth"
if os.path.exists(weights_file):
    size_mb = os.path.getsize(weights_file) / 1024 / 1024
    print(f"✅ B1权重文件: {weights_file} ({size_mb:.1f} MB)")
else:
    print(f"❌ 找不到B1权重文件")

print("\n🎯 直接运行最终应用脚本（指定UTF-8编码）:")

# 创建修复版应用脚本
fix_code = '''
import sys
import os

# 强制UTF-8编码
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# 执行原脚本但跳过有问题的部分
exec(open("apply_b1_to_yolo_final.py", encoding='utf-8', errors='ignore').read())
'''

fix_path = "apply_b1_fixed.py"
with open(fix_path, 'w', encoding='utf-8') as f:
    f.write(fix_code)

print(f"📄 已创建修复脚本: {fix_path}")
print(f"📋 运行命令: python {fix_path}")
