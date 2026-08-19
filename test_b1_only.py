# check_diffusion_params.py
import inspect
from eat import GaussianDiffusion

print("🔍 检查GaussianDiffusion构造函数参数")
print("=" * 60)

# 获取构造函数参数
init_signature = inspect.signature(GaussianDiffusion.__init__)
params = list(init_signature.parameters.keys())

print("构造函数参数列表:")
for i, param in enumerate(params[1:], 1):  # 跳过self
    default = init_signature.parameters[param].default
    if default == inspect.Parameter.empty:
        print(f"  {i:2d}. {param} (必填)")
    else:
        print(f"  {i:2d}. {param} = {default}")

# 查找偏移网络相关参数
print("\n📌 查找偏移网络参数:")
offset_params = [p for p in params if 'offset' in p.lower() or 'strength' in p.lower()]
for p in offset_params:
    print(f"  - {p}")

print("\n📌 查找频域参数:")
freq_params = [p for p in params if 'freq' in p.lower() or 'alpha' in p.lower()]
for p in freq_params:
    print(f"  - {p}")